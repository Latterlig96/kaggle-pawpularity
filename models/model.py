import swin_transformer
import torch
from pytorch_lightning import LightningModule

import efficientnet
import levit_transformer
from pytorch_grad_cam import GradCAMPlusPlus
from ..augmentations import mixup, Augmentation


class Model(LightningModule):

    supported_models = {
        'EfficientNetV2Large': efficientnet.__dict__['EfficientNetV2Large'],
        'EfficientNetV2Medium': efficientnet.__dict__['EfficientNetV2Medium'],
        'EfficientNetV2Small': efficientnet.__dict__['EfficientNetV2Small'],
        'Levit': levit_transformer.__dict__['Levit'],
        'SwinLarge': swin_transformer.__dict__['SwinLarge'],
        'SwinSmall': swin_transformer.__dict__['SwinSmall'],
        'SwinTiny': swin_transformer.__dict__['SwinTiny']
    }

    supported_loss = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss
    }    

    supported_optimizers = {
        'Adam': torch.optim.Adam
    }

    supported_schedulers = {
        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    }

    def __init__(self,
                cfg):
        super().__init__()
        self.cfg = cfg
        self._build_model()
        self.save_hyperparameters(self.cfg)
        self.transform = Augmentation(self.cfg)

    def _build_model(self):
        if self.cfg.model_name not in self.supported_models:
            raise ValueError(f"{self.cfg.model_name} not supported, check your configuration")
        self.model = self.supported_models[self.cfg.model_name]
    
    def _build_criterion(self):
        if self.cfg.loss not in self.supported_loss:
            raise ValueError(f"{self.cfg.loss} not supported, check your configuration")
        self.critertion = self.supported_loss[self.cfg.loss]
    
    def _build_optimizer(self):
        if self.cfg.optimizer not in self.supported_optimizers:
            raise ValueError(f"{self.cfg.optimizer} not supported, check your configuration")
        self.optimizer = self.supported_optimizers[self.cfg.optimizer]
    
    def _build_scheduler(self):
        if self.cfg.scheduler not in self.supported_schedulers:
            raise ValueError(f"{self.cfg.optimizer} not supported, check your configuration")
        self.scheduler = self.supported_schedulers[self.cfg.scheduler]

    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        loss, pred, labels = self._share_step(batch, 'train')
        return {'loss': loss, 'pred': pred, 'labels': labels}
        
    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self._share_step(batch, 'val')
        return {'loss': loss, 'pred': pred, 'labels': labels}

    def _share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        images = self.transform.get_augmentation_by_mode(mode)(images)
        
        if torch.rand(1)[0] < 0.5 and mode == 'train':
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeeze(1)
            loss = self.criterion(logits, target_a) * lam + \
                (1 - lam) * self.criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self.criterion(logits, labels)
        
        pred = logits.sigmoid().detach().cpu() * 100.
        labels = labels.detach().cpu() * 100.
        return loss, pred, labels
        
    def training_epoch_end(self, outputs):
        self._share_epoch_end(outputs, 'train')
    
    def validation_epoch_end(self, outputs):
        self._share_epoch_end(outputs, 'val')    
        
    def _share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f'{mode}_loss', metrics)
    
    def check_gradcam(self, dataloader, target_layer, target_category, reshape_transform=None):
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer, 
            use_cuda=self.cfg.trainer.gpus, 
            reshape_transform=reshape_transform)
        
        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform.get_augmentation_by_mode('val')(org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()
        
        grayscale_cam = cam(input_tensor=images, target_category=target_category, eigen_smooth=True)
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.
        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        optimizer = self._build_optimizer(
            self.parameters(), **self.cfg.optimizer.params
        )

        scheduler = self._build_scheduler(
            optimizer,
            **self.cfg.scheduler.params
        )

        return optimizer, scheduler
