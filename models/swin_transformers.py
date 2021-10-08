import timm
import torch.nn as nn


class SwinLarge(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class SwinSmall(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_small_patch4_window7_22', pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class SwinTiny(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
