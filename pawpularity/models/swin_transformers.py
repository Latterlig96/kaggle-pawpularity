import timm
import torch.nn as nn


class SwinLarge(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(
            self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(
            num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)

        self.apply_resizer = self.cfg.resizer['apply']
        self.apply_stn = self.cfg.stn['apply']
        self.apply_after_resizer = self.cfg.stn['apply_after_resizer']

        if self.apply_resizer:
            from .learnable_resizer import Resizer
            self.resizer = Resizer(self.cfg)

        if self.apply_stn:
            from .stn import StnLarge
            self.stn1 = StnLarge(self.cfg)
        if self.apply_after_resizer:
            from .stn import StnSmall
            self.stn2 = StnSmall(self.cfg)

    def forward(self, x, return_embeds: bool = False):
        if self.apply_stn:
            x = self.stn1(x)
        if self.apply_resizer:
            x = self.resizer(x)
        if self.apply_after_resizer:
            x = self.stn2(x)
        x = self.backbone(x)
        if return_embeds:
            embeddings = x
            logits = self.fc(x)
            return logits, embeddings
        x = self.fc(x)
        return x


class SwinLargev2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'swin_base_patch4_window12_384', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(
            self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(
            num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)

        self.apply_resizer = self.cfg.resizer['apply']
        self.apply_stn = self.cfg.stn['apply']
        self.apply_after_resizer = self.cfg.stn['apply_after_resizer']

        if self.apply_resizer:
            from .learnable_resizer import Resizer
            self.resizer = Resizer(self.cfg)

        if self.apply_stn:
            from .stn import StnLarge
            self.stn1 = StnLarge(self.cfg)
        if self.apply_after_resizer:
            from .stn import StnSmall
            self.stn2 = StnSmall(self.cfg)

    def forward(self, x, return_embeds: bool = False):
        if self.apply_stn:
            x = self.stn1(x)
        if self.apply_resizer:
            x = self.resizer(x)
        if self.apply_after_resizer:
            x = self.stn2(x)
        x = self.backbone(x)
        if return_embeds:
            embeddings = x
            logits = self.fc(x)
            return logits, embeddings
        x = self.fc(x)
        return x


class SwinSmall(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'swin_small_patch4_window7_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(
            self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(
            num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)

        self.apply_resizer = self.cfg.resizer['apply']
        self.apply_stn = self.cfg.stn['apply']
        self.apply_after_resizer = self.cfg.stn['apply_after_resizer']

        if self.apply_resizer:
            from .learnable_resizer import Resizer
            self.resizer = Resizer(self.cfg)

        if self.apply_stn:
            from .stn import StnLarge
            self.stn1 = StnLarge(self.cfg)
        if self.apply_after_resizer:
            from .stn import StnSmall
            self.stn2 = StnSmall(self.cfg)

    def forward(self, x, return_embeds: bool = False):
        if self.apply_stn:
            x = self.stn1(x)
        if self.apply_resizer:
            x = self.resizer(x)
        if self.apply_after_resizer:
            x = self.stn2(x)
        x = self.backbone(x)
        if return_embeds:
            embeddings = x
            logits = self.fc(x)
            return logits, embeddings
        x = self.fc(x)
        return x


class SwinTiny(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=self.cfg.use_pretrained, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.dropout = None if not self.cfg.use_dropout else nn.Dropout(
            self.cfg.dropout_rate)
        self.fc = nn.Sequential(self.dropout, nn.Linear(
            num_features, self.cfg.output_dim)) if self.dropout else nn.Linear(num_features, self.cfg.output_dim)

        self.apply_resizer = self.cfg.resizer['apply']
        self.apply_stn = self.cfg.stn['apply']
        self.apply_after_resizer = self.cfg.stn['apply_after_resizer']

        if self.apply_resizer:
            from .learnable_resizer import Resizer
            self.resizer = Resizer(self.cfg)

        if self.apply_stn:
            from .stn import StnLarge
            self.stn1 = StnLarge(self.cfg)
        if self.apply_after_resizer:
            from .stn import StnSmall
            self.stn2 = StnSmall(self.cfg)

    def forward(self, x, return_embeds: bool = False):
        if self.apply_stn:
            x = self.stn1(x)
        if self.apply_resizer:
            x = self.resizer(x)
        if self.apply_after_resizer:
            x = self.stn2(x)
        x = self.backbone(x)
        if return_embeds:
            embeddings = x
            logits = self.fc(x)
            return logits, embeddings
        x = self.fc(x)
        return x
