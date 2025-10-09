"""
概要:
    v3 パイプラインで用いるモデル関連のクラスと損失関数を定義するモジュール。
    - Stage1: GSM 93ch 入力に対して 6クラス（none+前線5種）のロジットを出力する Swin-UNet ラッパ
    - Stage2: 1ch のクラスマップ入力を 6クラスに補正する Swin-UNet ラッパ
    - 損失: マルチクラス Dice 損失と CE+Dice の複合損失

構成:
    - SwinUnetModel: Stage1 用モデル（入力=93ch、出力=6ch）
    - SwinUnetModelStage2: Stage2 用モデル（入力=1ch、出力=6ch）
    - DiceLoss: マルチクラス Dice 損失
    - combined_loss: 交差エントロピー + Dice の複合損失（学習安定と輪郭強調の両立を狙う）

注意:
    - 具体的なモデルのハイパーパラメータは main_v3_config.CFG["STAGE1"/"STAGE2"]["model"] に従う
    - SwinTransformerSys の詳細実装は swin_unet.py を参照
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

from swin_unet import SwinTransformerSys
from main_v3_config import CFG


class SwinUnetModel(nn.Module):
    """
    概要:
        Stage1（Swin-UNet）によるマルチクラスセグメンテーションのためのラッパーモデル。
        SwinTransformerSys を内部に保持し、入力（GSM 93ch）から各クラスのロジットを出力する。

    入力:
        - num_classes (int): 出力クラス数（既定: 6 = none + 5 前線）
        - in_chans (int): 入力チャネル数（既定: 93 = 31変数×3時刻）
        - model_cfg (dict|None): モデル設定（未指定時は CFG["STAGE1"]["model"] を使用）

    出力:
        - forward(x): (B, num_classes, H, W) のロジット
    """
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
        """
        概要:
            入力テンソルから各クラスのロジットマップを計算する。

        入力:
            - x (Tensor): 形状 (B, in_chans, H, W) の入力特徴（例: GSM 93ch）

        出力:
            - logits (Tensor): 形状 (B, num_classes, H, W) のロジット
        """
        logits = self.swin_unet(x)
        return logits


class SwinUnetModelStage2(nn.Module):
    """
    概要:
        Stage2（Swin ベースの補正器）用のラッパーモデル。
        入力は 1ch のクラスマップ（擬似劣化や Stage1 の argmax 等）を想定し、
        6クラスのロジットを出力する。

    入力:
        - num_classes (int): 出力クラス数（既定: 6）
        - in_chans (int): 入力チャネル数（既定: 1）
        - model_cfg (dict|None): モデル設定（None の場合は CFG["STAGE2"]["model"] を使用）

    出力:
        - forward(x): (B, num_classes, H, W) のロジット
    """
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
        """
        概要:
            1ch 入力のクラスマップから Stage2 Swin-UNet によりロジットを算出する。

        入力:
            - x (Tensor): 形状 (B, in_chans=1, H, W)

        出力:
            - logits (Tensor): 形状 (B, num_classes=6, H, W)
        """
        return self.swin_unet(x)


class DiceLoss(nn.Module):
    """
    概要:
        マルチクラス Dice 損失。各クラスに対して Dice スコアを計算し、平均を損失として返す。
        画素単位の重なりを重視するため、クラス不均衡に比較的ロバスト。

    入力:
        - classes (int): クラス数（例: 6）

    出力:
        - forward(inputs, targets): スカラー損失（Tensor, shape=()）
    """
    def __init__(self, classes: int = 6):
        super(DiceLoss, self).__init__()
        self.classes = classes

    def forward(self, inputs, targets):
        """
        概要:
            ロジット inputs と GT ラベル targets から Dice 損失を計算する。

        入力:
            - inputs (Tensor): 形状 (B, C, H, W) のモデル出力ロジット
            - targets (Tensor): 形状 (B, H, W) の整数クラスラベル（0..C-1）

        出力:
            - loss (Tensor): スカラー（全クラス平均 Dice 損失）
        """
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
ce_loss = nn.CrossEntropyLoss()  # ピクセル単位の多クラス交差エントロピー損失
dice_loss = DiceLoss(classes=CFG["STAGE1"]["num_classes"])  # both Stage1/2 have 6 classes


def combined_loss(inputs, targets):
    """
    概要:
        交差エントロピー（CE）と Dice 損失の和による複合損失。

    入力:
        - inputs (Tensor): (B, C, H, W) のロジット
        - targets (Tensor): (B, H, W) の整数クラスラベル

    出力:
        - loss (Tensor): CE + Dice のスカラー損失
    """
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
