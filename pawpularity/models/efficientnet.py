import timm
import torch.nn as nn


class EfficientNetV2Large(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnetv2_l', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetV2Medium(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnetv2_m', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetV2Small(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnetv2_s', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetB0(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetB1(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnet_b1', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetB2(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnet_b2', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetB3(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnet_b1', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetB4(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class EfficientNetB5(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'efficientnet_b5', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x