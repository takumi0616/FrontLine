import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

from swin_unet import SwinTransformerSys
from main_v3_config import CFG


class SwinUnetModel(nn.Module):
    def __init__(self, num_classes=6, in_chans=93, model_cfg=None):
        super(SwinUnetModel, self).__init__()
        cfg = model_cfg if model_cfg is not None else CFG["STAGE1"]["model"]
        self.swin_unet = SwinTransformerSys(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            depths_decoder=cfg["depths_decoder"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            norm_layer=cfg["norm_layer"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            final_upsample=cfg["final_upsample"],
        )

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits


class SwinUnetModelStage2(nn.Module):
    def __init__(self, num_classes=6, in_chans=1, model_cfg=None):
        super(SwinUnetModelStage2, self).__init__()
        cfg = model_cfg if model_cfg is not None else CFG["STAGE2"]["model"]
        self.swin_unet = SwinTransformerSys(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            depths_decoder=cfg["depths_decoder"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            qkv_bias=cfg["qkv_bias"],
            qk_scale=cfg["qk_scale"],
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
            drop_path_rate=cfg["drop_path_rate"],
            norm_layer=cfg["norm_layer"],
            ape=cfg["ape"],
            patch_norm=cfg["patch_norm"],
            use_checkpoint=cfg["use_checkpoint"],
            final_upsample=cfg["final_upsample"],
        )

    def forward(self, x):
        return self.swin_unet(x)


class DiceLoss(nn.Module):
    def __init__(self, classes: int = 6):
        super(DiceLoss, self).__init__()
        self.classes = classes

    def forward(self, inputs, targets):
        smooth = 1e-5
        total_loss = 0.0
        inputs = torch.softmax(inputs, dim=1)

        for i in range(self.classes):
            inp_flat = inputs[:, i].contiguous().view(-1)
            tgt_flat = (targets == i).float().view(-1)
            intersection = (inp_flat * tgt_flat).sum()
            dice_score = (2.0 * intersection + smooth) / (inp_flat.sum() + tgt_flat.sum() + smooth)
            dice_loss = 1 - dice_score
            total_loss += dice_loss
        return total_loss / self.classes


# Keep the same API as the original script
ce_loss = nn.CrossEntropyLoss()
dice_loss = DiceLoss(classes=CFG["STAGE1"]["num_classes"])  # both Stage1/2 have 6 classes


def combined_loss(inputs, targets):
    loss_ce = ce_loss(inputs, targets)
    loss_dc = dice_loss(inputs, targets)
    return loss_ce + loss_dc


__all__ = [
    "SwinUnetModel",
    "SwinUnetModelStage2",
    "DiceLoss",
    "combined_loss",
    "ce_loss",
    "dice_loss",
]
