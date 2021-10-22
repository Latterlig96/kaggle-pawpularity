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
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
