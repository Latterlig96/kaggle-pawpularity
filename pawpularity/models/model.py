import torch
import torchvision.transforms as T
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_lightning import LightningModule
from pawpularity.augmentations import mixup
from . import efficientnet, levit_transformer, swin_transformers, vision_transformers, learnable_resizer


class Model(LightningModule):

    supported_models = {
        'EfficientNetV2Large': efficientnet.__dict__['EfficientNetV2Large'],
        'EfficientNetV2Medium': efficientnet.__dict__['EfficientNetV2Medium'],
        'EfficientNetV2Small': efficientnet.__dict__['EfficientNetV2Small'],
        'EfficientNetB0': efficientnet.__dict__['EfficientNetB0'],
        'EfficientNetB1': efficientnet.__dict__['EfficientNetB1'],
        'EfficientNetB2': efficientnet.__dict__['EfficientNetB2'],
        'EfficientNetB3': efficientnet.__dict__['EfficientNetB3'],
        'EfficientNetB4': efficientnet.__dict__['EfficientNetB4'],
        'EfficientNetB5': efficientnet.__dict__['EfficientNetB5'],
        'Levit': levit_transformer.__dict__['Levit'],
        'SwinLarge': swin_transformers.__dict__['SwinLarge'],
        'SwinLargev2': swin_transformers.__dict__['SwinLargev2'],
        'SwinSmall': swin_transformers.__dict__['SwinSmall'],
        'SwinTiny': swin_transformers.__dict__['SwinTiny'],
        'ViTTiny': vision_transformers.__dict__['ViTTiny'],
        'ViTTinyv2': vision_transformers.__dict__['ViTTinyv2'],
        'ViTSmall': vision_transformers.__dict__['ViTSmall'],
        'ViTSmallv2': vision_transformers.__dict__['ViTSmallv2'],
        'ViTLarge': vision_transformers.__dict__['ViTLarge'],
        'ViTLargev2': vision_transformers.__dict__['ViTLargev2'],
        'ViTHybridTiny': vision_transformers.__dict__['ViTHybridTiny'],
        'ViTHybridTinyv2': vision_transformers.__dict__['ViTHybridTinyv2'],
        'ViTHybridSmall': vision_transformers.__dict__['ViTHybridSmall'],
        'ViTHybridSmallv2': vision_transformers.__dict__['ViTHybridSmallv2'],
        'ViTHybridLarge': vision_transformers.__dict__['ViTHybridLarge'],
        'ViTHybridLargev2': vision_transformers.__dict__['ViTHybridLargev2'],
    }

    supported_loss = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss
    }

    supported_optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }

    supported_schedulers = {
        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    }

    def __init__(self,
                 cfg):
        super().__init__()
        self.cfg = cfg
        self._build_model()
        self._build_criterion()
        self.save_hyperparameters(self.cfg.asdict)

    def _build_model(self):
        if self.cfg.model_name not in self.supported_models:
            raise ValueError(
                f"{self.cfg.model_name} not supported, check your configuration")
        self.model = self.supported_models[self.cfg.model_name](self.cfg)

    def _build_criterion(self):
        if self.cfg.loss not in self.supported_loss:
            raise ValueError(
                f"{self.cfg.loss} not supported, check your configuration")
        self.criterion = self.supported_loss[self.cfg.loss]()

    def _build_optimizer(self):
        if self.cfg.optimizer['name'] not in self.supported_optimizers:
            raise ValueError(
                f"{self.cfg.optimizer} not supported, check your configuration")
        self.optimizer = self.supported_optimizers[self.cfg.optimizer['name']](
            self.parameters(), **self.cfg.optimizer['params'])

    def _build_scheduler(self):
        if self.cfg.scheduler['name'] not in self.supported_schedulers:
            raise ValueError(
                f"{self.cfg.optimizer} not supported, check your configuration")
        self.scheduler = self.supported_schedulers[self.cfg.scheduler['name']](
            self.optimizer,  **self.cfg.scheduler['params'])

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self._share_step(batch, 'train')
        return {'loss': loss, 'pred': pred, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self._share_step(batch, 'val')
        return {'loss': loss, 'pred': pred, 'labels': labels}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        logits, embeddings = self.model(images, True)
        pred = logits.squeeze(1).sigmoid().detach().cpu().numpy() * 100.
        embeddings = embeddings.detach().cpu().numpy()
        return {'pred': pred, 'embeddings': embeddings}

    def _share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0

        if torch.rand(1)[0] < 0.5 and mode == 'train':
            mix_images, target_a, target_b, lam = mixup(
                images, labels, alpha=0.5)
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

        inv_normalize = T.Normalize(mean=[-m/s for m, s in zip(self.cfg.image_mean, self.cfg.image_std)],
                                    std=[1/s for s in self.cfg.image_std])
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer,
            use_cuda=self.cfg.trainer['gpus'],
            reshape_transform=reshape_transform)

        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = org_images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()
        grayscale_cam = cam(input_tensor=images,
                            target_category=target_category, eigen_smooth=True)
        org_images = inv_normalize(images)
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1)

        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):

        self._build_optimizer()

        self._build_scheduler()

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}


class ResizerModel(LightningModule):

    supported_models = {
        'Resizer': learnable_resizer.__dict__['Resizer']
    }

    supported_loss = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        'MSE': torch.nn.MSELoss
    }

    supported_optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }

    supported_schedulers = {
        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    }

    def __init__(self,
                 cfg):
        super().__init__()
        self.cfg = cfg
        self._build_model()
        self._build_criterion()

    def _build_model(self):
        if self.cfg.model_name not in self.supported_models:
            raise ValueError(
                f"{self.cfg.model_name} not supported, check your configuration")
        self.model = self.supported_models[self.cfg.model_name](self.cfg)

    def _build_criterion(self):
        if self.cfg.loss not in self.supported_loss:
            raise ValueError(
                f"{self.cfg.loss} not supported, check your configuration")
        self.criterion = self.supported_loss[self.cfg.loss]()

    def _build_optimizer(self):
        if self.cfg.optimizer['name'] not in self.supported_optimizers:
            raise ValueError(
                f"{self.cfg.optimizer} not supported, check your configuration")
        self.optimizer = self.supported_optimizers[self.cfg.optimizer['name']](
            self.parameters(), **self.cfg.optimizer['params'])

    def _build_scheduler(self):
        if self.cfg.scheduler['name'] not in self.supported_schedulers:
            raise ValueError(
                f"{self.cfg.optimizer} not supported, check your configuration")
        self.scheduler = self.supported_schedulers[self.cfg.scheduler['name']](
            self.optimizer,  **self.cfg.scheduler['params'])

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
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        y_hat = y_hat.detach().cpu()
        y = y.detach().cpu()
        return loss, y_hat, y

    def _share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = ((labels - preds) ** 2).mean()
        self.log(f'{mode}_loss', metrics)

    def training_epoch_end(self, training_step_outputs):
        self._share_epoch_end(training_step_outputs, 'train')

    def validation_epoch_end(self, validation_step_outputs):
        self._share_epoch_end(validation_step_outputs, 'val')

    def configure_optimizers(self):

        self._build_optimizer()

        self._build_scheduler()

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
