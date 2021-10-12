import timm
import torch.nn as nn


class EfficientNetV2Large(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnetv2_l', pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetV2Medium(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnetv2_m', pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetV2Small(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnetv2_s', pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
