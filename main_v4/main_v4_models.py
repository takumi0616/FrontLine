"""
ファイル概要（main_v4_models.py）:
- 役割:
  v4 パイプラインで用いるモデル本体（Swin-UNet ラッパ）と損失関数（DiceLoss, CE+Dice 複合）を定義する。
  各ステージ毎に in_chans / num_classes を切り替えて使用できるようにし、CFG 側のモデル設定をそのまま注入可能。
- 入出力:
  - SwinUnetModel: forward(x) で (B,C,H,W) のロジットを返す。x のチャネル数は各ステージの入力仕様に一致させる。
  - DiceLoss: forward(logits, targets) でスカラー損失を返す（targets は整数クラスの 2D マップ）。
  - make_combined_loss: CE + Dice の合成損失（足し合わせ）を返すクロージャ関数。
- 実装上の注意:
  - swin_unet.SwinTransformerSys は v3 由来の実装を想定し、sys.path を調整してインポートしている。
  - DiceLoss は one-hot 化を内部で行わず、整数クラス（B,H,W）を前提とする（CE と合わせて使用）。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

# swin_unet 実装は v3 同様の場所にあることを想定
# 例: src/FrontLine/main_v3/swin_unet.py を sys.path に載せている前提
try:
    from swin_unet import SwinTransformerSys
except Exception as e:
    # ユーザー環境により import パスが異なる場合に備え、親ディレクトリも探索
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    from swin_unet import SwinTransformerSys  # type: ignore

from .main_v4_config import CFG


class SwinUnetModel(nn.Module):
    """
    クラス概要:
      Swin-UNet の薄いラッパクラス。CFG から渡されたモデル設定（パッチサイズ/深さ/ヘッド数等）を用いて
      任意の in_chans / num_classes を持つセグメンテーションネットを構築する。

    入力:
      - in_chans (int): 入力チャネル数（Stage1=93, Stage2=94, Stage3=96, Stage4=97 など）
      - num_classes (int): 出力クラス数（Stage1/3/4=2, Stage2=3）
      - model_cfg (dict): main_v4_config.CFG["STAGE*"]["model"] を想定したハイパーパラメータ辞書

    処理:
      - SwinTransformerSys のインスタンスを生成し、与えられた in_chans / num_classes で初期化。

    出力:
      - forward(x): (B,C,H,W) のロジットテンソル（C=num_classes）
    """
    def __init__(self, in_chans: int, num_classes: int, model_cfg: dict):
        super().__init__()
        self.swin_unet = SwinTransformerSys(
            img_size=model_cfg["img_size"],
            patch_size=model_cfg["patch_size"],
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=model_cfg["embed_dim"],
            depths=model_cfg["depths"],
            depths_decoder=model_cfg["depths_decoder"],
            num_heads=model_cfg["num_heads"],
            window_size=model_cfg["window_size"],
            mlp_ratio=model_cfg["mlp_ratio"],
            qkv_bias=model_cfg["qkv_bias"],
            qk_scale=model_cfg["qk_scale"],
            drop_rate=model_cfg["drop_rate"],
            attn_drop_rate=model_cfg["attn_drop_rate"],
            drop_path_rate=model_cfg["drop_path_rate"],
            norm_layer=model_cfg["norm_layer"],
            ape=model_cfg["ape"],
            patch_norm=model_cfg["patch_norm"],
            use_checkpoint=model_cfg["use_checkpoint"],
            final_upsample=model_cfg["final_upsample"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        関数概要:
          入力テンソル x を Swin-UNet 本体へ通し、クラスごとのロジットを返す。

        入力:
          - x (torch.Tensor): (B, in_chans, H, W) 形式のミニバッチ入力

        処理:
          - SwinTransformerSys にそのまま入力して前向き計算を実施。

        出力:
          - torch.Tensor: (B, num_classes, H, W) のロジット（softmax 前）
        """
        return self.swin_unet(x)


class DiceLoss(nn.Module):
    """
    クラス概要:
      マルチクラス用の Dice 損失。CE と組み合わせて使用することを想定。
      one-hot 化は内部で行わず、targets は整数クラス (B,H,W) を想定する。

    入力（コンストラクタ）:
      - classes (int): クラス数（num_classes）

    使い方:
      - forward(logits, targets) を呼び出すと、各クラスの Dice 損失を平均したスカラーを返す。
    """
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        関数概要:
          ロジットを softmax で確率化し、各クラスごとの Dice 係数に基づく損失（1 - Dice）を計算して平均する。

        入力:
          - inputs (torch.Tensor): (B, C, H, W) ロジット（C=num_classes）
          - targets (torch.Tensor): (B, H, W) の整数クラスマップ [0..C-1]

        処理:
          - probs = softmax(inputs)
          - 各クラス c について、pred_c と tgt_c の積分（intersection）と和（denominator）から Dice を計算。
          - 1 - Dice を損失としてクラス平均を返す。

        出力:
          - torch.Tensor: スカラー損失（バッチ平均済み）
        """
        smooth = 1e-5
        total_loss = 0.0
        probs = torch.softmax(inputs, dim=1)  # (B,C,H,W)
        B, C, H, W = probs.shape

        for c in range(self.classes):
            pred_c = probs[:, c, :, :].contiguous().view(B, -1)       # (B, HW)
            tgt_c = (targets == c).float().contiguous().view(B, -1)   # (B, HW)
            intersection = (pred_c * tgt_c).sum(dim=1)                # (B,)
            denom = pred_c.sum(dim=1) + tgt_c.sum(dim=1) + smooth
            dice_c = (2.0 * intersection + smooth) / denom            # (B,)
            loss_c = 1.0 - dice_c
            total_loss += loss_c.mean()

        return total_loss / float(self.classes)


def make_combined_loss(num_classes: int):
    """
    関数概要:
      交差エントロピー（CE）と DiceLoss を足し合わせた複合損失を返すファクトリ関数。

    入力:
      - num_classes (int): クラス数（DiceLoss 構築に使用）

    処理:
      - nn.CrossEntropyLoss() と DiceLoss(classes=num_classes) を内部に保持するクロージャ関数を構築。
      - 呼び出し時に CE(inputs, targets) + Dice(inputs, targets) を返す。

    出力:
      - Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        (inputs, targets) -> スカラー損失 を返す関数オブジェクト
    """
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss(classes=num_classes)

    def _loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ce(inputs, targets) + dice(inputs, targets)

    return _loss


__all__ = [
    "SwinUnetModel",
    "DiceLoss",
    "make_combined_loss",
]
