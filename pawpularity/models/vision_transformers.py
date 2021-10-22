import timm
import torch.nn as nn

class ViTTiny(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTTinyv2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_tiny_patch16_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTSmall(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_small_patch32_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTSmallv2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_small_patch32_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTLarge(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_large_patch32_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTLargev2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_large_patch32_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTHybridTiny(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_tiny_r_s16_p8_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTHybridTinyv2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_tiny_r_s16_p8_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTHybridSmall(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_small_r26_s32_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTHybridSmallv2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_small_r26_s32_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class ViTHybridLarge(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_large_r50_s32_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ViTHybridLargev2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'vit_large_r50_s32_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
