import timm
import torch.nn as nn


class Levit(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'levit_256', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.output_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
