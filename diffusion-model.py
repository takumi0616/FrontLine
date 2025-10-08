"""
拡散モデル（Denoising Diffusion Probabilistic Models / DDIM系）を用いて、
Stage1（例: Swin-UNet）の出力確率マップを確率的に「補正（refine）」するためのユーティリティ群。

提供クラス:
- DiffusionCorrector
    概要: 条件なし（unconditional）の拡散モデル。Stage1の確率マップを初期状態 x0 と見なし、
          任意時刻 t_start まで前向きノイズ付加した x_t から逆拡散で再構成（refine）する。
- ConditionalDiffusionCorrector
    概要: 条件付き（conditional）の拡散モデル。cond（例: Stage1確率）を条件として結合し、
          cond に整合する x0（GT確率）を再構成するように学習・生成を行う。
- _CondUNetWrapper（内部用）
    概要: GaussianDiffusion は model(x_t, t) を想定しているため、U-Net に条件を結合したい場合に
          x_t と cond をチャネル結合して渡すための薄いラッパ。

使い所:
- 既存のセグメンテーションやクラス確率出力の「滑らかさ・一貫性」を拡散モデルで補正したい場合
- 学習では GT 確率（one-hotなど）から DDPM/ DDIM でノイズを推定するタスクとして学習
- 推論では Stage1 確率を「初期状態」として逆拡散の開始点を構成し、確率的に修正

注意:
- 本モジュールは denoising-diffusion-pytorch の Unet / GaussianDiffusion を内部で利用
- EMA / accelerate 等の学習用依存は Trainer でのみ使われるため、ここでは軽量スタブで回避
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 依存: ローカルクローンの denoising-diffusion-pytorch を直接 import
# パス: src/FrontLine/denoising-diffusion-pytorch/denoising_diffusion_pytorch
import sys
from pathlib import Path as _Path
# denoising diffusion package path candidates (underscore / hyphen / absolute)
_pkg_candidates = [
    _Path(__file__).parent / "denoising_diffusion_pytorch",
    _Path(__file__).parent / "denoising-diffusion-pytorch",
    _Path("/home/takumi/docker_miniconda/src/FrontLine/denoising_diffusion_pytorch"),
]
for p in _pkg_candidates:
    try:
        if p.exists():
            s = str(p)
            if s not in sys.path:
                sys.path.append(s)
    except Exception:
        pass

# 軽量スタブで学習用依存(ema_pytorch / accelerate)のインポート失敗を回避
# ※ denoising_diffusion_pytorch の Trainer でのみ使用されるため、Corrector用途では未使用
import types as _types
if 'ema_pytorch' not in sys.modules:
    _ema = _types.ModuleType('ema_pytorch')
    class EMA:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def state_dict(self): return {}
        def load_state_dict(self, *args, **kwargs): pass
        def update(self): pass
        @property
        def ema_model(self): return None
    _ema.EMA = EMA
    sys.modules['ema_pytorch'] = _ema

if 'accelerate' not in sys.modules:
    _acc = _types.ModuleType('accelerate')
    class Accelerator:
        def __init__(self, *args, **kwargs): pass
        @property
        def device(self):
            import torch as _t
            return _t.device('cpu')
        def prepare(self, *args, **kwargs):
            # denoising-diffusion-pytorch.Trainer 互換だが未使用のためパススルー
            return args if len(args) != 1 else args[0]
        def backward(self, loss):
            try:
                loss.backward()
            except Exception:
                pass
        @property
        def is_main_process(self): return True
        @property
        def scaler(self): return None
        def clip_grad_norm_(self, *args, **kwargs): pass
        def wait_for_everyone(self): pass
        def unwrap_model(self, m): return m
        def print(self, *a, **k): print(*a, **k)
        def autocast(self):
            from contextlib import nullcontext
            return nullcontext()
    _acc.Accelerator = Accelerator
    sys.modules['accelerate'] = _acc

from denoising_diffusion_pytorch import Unet, GaussianDiffusion


class DiffusionCorrector(nn.Module):
    """
    概要:
        Swin-Unet（Predictor）の出力確率マップを、拡散モデルで確率的に補正（refine）するためのラッパ。
        学習時は one-hot化された確率マップ（6チャネル）をそのまま学習データとして使用（unpaired でも可）。
        推論時は Stage1 の確率マップを初期状態 x0 と見なし、q(x_t | x0) で t=t_start のノイズを加え、
        その x_t から t=t_start..0 で逆拡散して補正結果を得る。

    入力（主に forward / correct_from_probs で使用）:
        - 画像サイズ image_size（H=W）
        - チャネル数 channels（前線クラスの one-hot で 6 を想定）
        - U-Net の基本幅 base_dim と解像度スケール dim_mults

    処理:
        - 内部に denoising-diffusion-pytorch の Unet と GaussianDiffusion を構築
        - forward(x) で学習時の損失（スカラー）を返す
        - correct_from_probs(prob_map, ...) で Stage1 の確率マップから補正済み確率を生成

    出力:
        - forward: 損失（torch.Tensor, scalar）
        - correct_from_probs: 補正後の確率マップ（(ensemble*B, C, H, W), 0..1, 各画素でチャネル正規化済み）
    """
    def __init__(
        self,
        image_size: int = 128,
        channels: int = 6,
        base_dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.0,
        objective: str = 'pred_v',
        beta_schedule: str = 'sigmoid',
        timesteps: int = 1000,
        sampling_timesteps: int = 20,
        auto_normalize: bool = True,
        flash_attn: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        概要:
            拡散モデル（U-Net + GaussianDiffusion）の構成要素を初期化する。

        入力:
            - image_size (int): 入出力の正方形画像サイズ H=W
            - channels (int): クラス確率マップのチャネル数（例: 6）
            - base_dim (int): U-Net のベースとなる特徴チャネル幅
            - dim_mults (Tuple[int, ...]): 各解像度段の幅倍率
            - dropout (float): U-Net 内の dropout 率
            - objective (str): 損失の目的（'pred_v' など denoising-diffusion-pytorch 準拠）
            - beta_schedule (str): βスケジュール（'sigmoid' 等）
            - timesteps (int): 学習・生成の総ステップ数 T
            - sampling_timesteps (int): 生成（DDIM）のステップ数
            - auto_normalize (bool): [0,1] と [-1,1] の相互変換を GaussianDiffusion 内部に任せるか
            - flash_attn (bool): U-Net の flash attention 有効化フラグ（本構成では full-attn 無効のため影響小）
            - device (torch.device|None): モデルを配置するデバイス。None の場合は CUDA があれば CUDA、なければ CPU

        処理:
            - denoising-diffusion-pytorch の U-Net と GaussianDiffusion を組み立てる
            - モデルを指定デバイスに移動する

        出力:
            なし（初期化のみ）
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        # 1) U-Net (Dhariwal & Nichol 構成に近いが attention はオフ)
        self.unet = Unet(
            dim=base_dim,
            init_dim=None,
            out_dim=channels,          # objectiveにより実際には v / noise を出力
            dim_mults=dim_mults,
            channels=channels,
            self_condition=False,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            dropout=dropout,
            attn_dim_head=32,
            attn_heads=4,
            full_attn=(False, False, False, False),  # すべての段で full attention を無効化
            flash_attn=flash_attn
        )

        # 2) 拡散プロセス
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,  # DDIM ステップ数
            objective=objective,                    # 'pred_v' が安定
            beta_schedule=beta_schedule,           # 'sigmoid' は >64解像度向けに良好
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.0,
            auto_normalize=auto_normalize,
            offset_noise_strength=0.0,
            min_snr_loss_weight=False,
            immiscible=False
        )

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.device = device

    # nn.Module の forward は「訓練時の loss を返す」(denoising-diffusion-pytorch の設計に準拠)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要:
            学習（教師あり）時に使用。拡散過程の目的（例: ノイズ/速度 v の予測）に対する損失を計算して返す。

        入力:
            - x (Tensor): 形状 (B, C=channels, H=image_size, W=image_size)、値域 [0,1] を想定した確率マップ

        処理:
            - GaussianDiffusion の呼び出しにより、サンプリング時刻 t をランダムに取り、対応する損失を算出

        出力:
            - loss (Tensor): スカラー損失
        """
        return self.diffusion(x)

    @property
    def num_timesteps(self) -> int:
        """
        概要:
            拡散過程の総時刻数 T（学習/サンプリングに用いる離散ステップ数）を返す。

        入力:
            なし

        処理:
            - 内部の GaussianDiffusion から num_timesteps を参照

        出力:
            - T (int): 総ステップ数
        """
        return self.diffusion.num_timesteps

    def save(self, path: str):
        """
        概要:
            モデルの state_dict をファイルに保存する。

        入力:
            - path (str): 保存先パス（.pt / .pth など）

        処理:
            - torch.save(self.state_dict(), path)

        出力:
            なし（ファイル出力）
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """
        概要:
            ファイルから state_dict を読み込み、現在のインスタンスに適用する。

        入力:
            - path (str): 読み込み元パス
            - strict (bool): strict=True の場合、キー一致が厳密である必要がある

        処理:
            - torch.load(..., map_location='cpu') で state_dict を取得
            - load_state_dict で適用
            - 既定のデバイス self.device に移動

        出力:
            なし（モデルパラメータが更新される）
        """
        sd = torch.load(path, map_location='cpu')
        self.load_state_dict(sd, strict=strict)
        self.to(self.device)

    @torch.inference_mode()
    def _denoise_from(
        self,
        x_t: torch.Tensor,
        t_start: int,
        steps: Optional[int] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        概要:
            任意の時刻 t_start にある x_t から 0 まで DDIM ライクな更新式で逆拡散し、x_0 を再構成する。

        入力:
            - x_t (Tensor): 形状 (B, C, H, W)、正規化済み [-1,1] を想定（GaussianDiffusion.normalize 済）
            - t_start (int): 逆拡散を開始する時刻（0..T-1）
            - steps (int|None): サンプリングステップ数（指定なしは self.diffusion.sampling_timesteps）
            - eta (float): DDIM の stochasticity パラメータ（0 で deterministic）

        処理:
            - 時刻シーケンス [t_start, ..., 0, -1] を構成し、(t, t_next) のペアごとにモデル予測を用いて更新
            - model_predictions から (pred_noise, x_start_pred) を取得
            - DDIM 更新式:
                alpha = a_t, alpha_next = a_{t_next}
                sigma = eta * sqrt((1 - a_t / a_{t_next}) * (1 - a_{t_next}) / (1 - a_t))
                c = sqrt(1 - a_{t_next} - sigma^2)
                x_{t_next} = sqrt(a_{t_next}) * x_start + c * pred_noise + sigma * N(0, I)
            - t_next < 0 のときは x_start を最終出力とする

        出力:
            - x_0 (Tensor): 形状 (B, C, H, W)、値域 [0,1]（内部で unnormalize 済）
        """
        assert x_t.ndim == 4 and x_t.shape[1] == self.channels
        img = x_t
        B = img.shape[0]
        device = img.device

        total_T = self.num_timesteps
        steps = steps or self.diffusion.sampling_timesteps
        steps = max(1, int(steps))

        # 時刻列: [t_start, ..., 0, -1]
        times = torch.linspace(-1, t_start, steps + 1, dtype=torch.int64, device=device).flip(0)
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))  # [(t, t_next), ...]

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((B,), int(time), device=device, dtype=torch.long)

            # モデル予測
            pred_noise, x_start_pred, *_ = self.diffusion.model_predictions(
                img, time_cond, None, clip_x_start=True, rederive_pred_noise=True
            )
            x_start = x_start_pred

            if time_next < 0:
                img = x_start
                continue

            alpha = self.diffusion.alphas_cumprod[time]
            alpha_next = self.diffusion.alphas_cumprod[time_next]

            # DDIM 更新式
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return self.diffusion.unnormalize(img)

    @torch.inference_mode()
    def correct_from_probs(
        self,
        prob_map: torch.Tensor,
        steps: Optional[int] = None,
        t_start: Optional[int] = None,
        sigma_override: Optional[float] = None,
        ensemble: int = 1
    ) -> torch.Tensor:
        """
        概要:
            Stage1 の確率マップを初期状態 x0 とみなし、t_start まで前向きノイズ付加で x_t を作ってから
            逆拡散で補正済み確率を生成する。

        入力:
            - prob_map (Tensor): 形状 (B, C, H, W) または (B, H, W, C)、値域 [0,1] の確率マップ
            - steps (int|None): 逆拡散ステップ数（DDIMのサンプルステップ数）。未指定時は学習時設定
            - t_start (int|None): 逆拡散の開始時刻（0..T-1）。未指定なら T-1（最も強いノイズ）
            - sigma_override (float|None): 予備パラメータ（現状未使用）
            - ensemble (int): サンプルの枚数（多サンプルでアンサンブル）

        処理:
            - 入力を (B, C, H, W) に揃え、[0,1] -> [-1,1] へ正規化
            - 任意回数（ensemble）だけ、t_start のノイズを付加してから _denoise_from で逆拡散
            - 出力を [0,1] にクリップ後、チャネル方向で確率正規化（Softmax 的だが明示正規化）

        出力:
            - rec (Tensor): (ensemble*B, C, H, W)、値域 [0,1]、各画素でチャネル正規化済み
        """
        if prob_map.ndim == 4 and prob_map.shape[1] == self.channels:
            # (B, C, H, W)
            x0 = prob_map.to(self.device, dtype=torch.float32)
        elif prob_map.ndim == 4 and prob_map.shape[-1] == self.channels:
            # (B, H, W, C) -> (B, C, H, W)
            x0 = prob_map.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        else:
            raise ValueError(f"prob_map shape must be (B,C,H,W) or (B,H,W,C) with C={self.channels}, got {tuple(prob_map.shape)}")

        B, C, H, W = x0.shape
        assert C == self.channels, f"channels mismatch: expected {self.channels}, got {C}"
        assert H == self.image_size and W == self.image_size, f"image_size mismatch: expected {self.image_size}, got {(H, W)}"

        # x0 を [-1,1] 域に正規化
        x0_norm = self.diffusion.normalize(x0)

        t_start = self.num_timesteps - 1 if t_start is None else int(t_start)
        t_start = max(0, min(self.num_timesteps - 1, t_start))

        outs: List[torch.Tensor] = []
        for _ in range(ensemble):
            t_vec = torch.full((B,), t_start, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0_norm)
            # 前向きノイズ付加で x_t を作る
            x_t = self.diffusion.q_sample(x_start=x0_norm, t=t_vec, noise=noise)
            # x_t から逆拡散
            rec = self._denoise_from(x_t, t_start=t_start, steps=steps, eta=0.0)  # [0,1]
            # チャネル方向の確率正規化
            rec = torch.clamp(rec, 0.0, 1.0)
            rec = rec / (rec.sum(dim=1, keepdim=True) + 1e-8)
            outs.append(rec)

        return torch.cat(outs, dim=0)

    @staticmethod
    def one_hot_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        概要:
            クラスラベル画像（整数）を one-hot 確率マップ（0/1）に変換するヘルパ関数。

        入力:
            - labels (Tensor): 形状 (B, H, W)、int64、値域 [0..num_classes-1]
            - num_classes (int): クラス数 C

        処理:
            - F.one_hot により (B, H, W, C) を得て、(B, C, H, W) に次元並べ替え

        出力:
            - oh (Tensor): 形状 (B, C, H, W)、float32、値は {0,1}
        """
        assert labels.ndim == 3
        B, H, W = labels.shape
        oh = F.one_hot(labels.long(), num_classes=num_classes)  # (B, H, W, C)
        oh = oh.permute(0, 3, 1, 2).contiguous().float()
        return oh


class _CondUNetWrapper(nn.Module):
    """
    概要:
        条件付き拡散を既存の GaussianDiffusion に改造無しで適用するための薄いラッパ。
        GaussianDiffusion は model(x_t, t) のシグネチャを仮定しているため、条件 cond を
        事前に内部へセットし、forward で x_t と cond をチャネル結合してから U-Net に渡す。

    入力:
        - inner_unet (Unet): 実体となる U-Net モデル
        - data_channels (int): データ側のチャネル（例: 6）
        - cond_channels (int): 条件側のチャネル（例: 6）

    処理:
        - set_cond(cond) で条件 Tensor を記憶
        - forward(x_t, t) のたびに cond をバッチ整合し、torch.cat([x_t, cond], dim=1)

    出力:
        - inner_unet の出力（pred_v / noise 等、GaussianDiffusion に準拠）
    """
    def __init__(self, inner_unet: Unet, data_channels: int, cond_channels: int):
        super().__init__()
        self.inner = inner_unet
        self._cond: Optional[torch.Tensor] = None
        self.cond_channels = cond_channels
        # GaussianDiffusion 側の属性チェックに対応するため、必要属性を明示保持（入力はデータ側C）
        # - model.channels: データ側のチャネル数（x_t の C、例: 6）
        # - model.out_dim: 出力チャネル数（データ側 C と等しい必要あり）
        # - model.self_condition: 自己条件付けフラグ（未使用なら False）
        self.channels = int(data_channels)
        self.out_dim = int(getattr(inner_unet, "out_dim", data_channels))
        self.self_condition = getattr(inner_unet, "self_condition", False)

    def set_cond(self, cond: torch.Tensor):
        """
        概要:
            条件 Tensor を内部にセットする。

        入力:
            - cond (Tensor): 形状 (B, cond_channels, H, W)

        処理:
            - 形状検証のみ（正規化は呼び出し側で実施）
            - 後続の forward() 呼び出し時に使用

        出力:
            なし（内部状態更新）
        """
        # cond: (B, cond_channels, H, W)
        if cond.ndim != 4 or cond.shape[1] != self.cond_channels:
            raise ValueError(f"cond shape must be (B,{self.cond_channels},H,W), got {tuple(cond.shape)}")
        self._cond = cond

    def forward(self, x: torch.Tensor, time: torch.Tensor, *args, **kwargs):
        """
        概要:
            設定済みの cond を x とチャネル結合し、U-Net に渡す。

        入力:
            - x (Tensor): 形状 (B, C_data, H, W) の拡散中テンソル x_t
            - time (Tensor): 形状 (B,) の時刻インデックス t
            - *args, **kwargs: U-Net への追加引数（denoising-diffusion-pytorch 準拠）

        処理:
            - set_cond(cond) 済みであることを確認
            - バッチ数が異なる場合は cond を繰り返し/切り詰めで合わせる
            - u = concat([x, cond], dim=1) を U-Net に入力

        出力:
            - U-Net の出力（pred_v / noise 等）
        """
        if self._cond is None:
            raise RuntimeError("Condition tensor is not set. Call set_cond(cond) before forward().")
        # バッチ次元の整合
        if self._cond.shape[0] != x.shape[0]:
            c = self._cond
            if c.shape[0] < x.shape[0]:
                rep = (x.shape[0] + c.shape[0] - 1) // c.shape[0]
                c = c.repeat(rep, 1, 1, 1)[:x.shape[0]]
            else:
                c = c[:x.shape[0]]
        else:
            c = self._cond
        u = torch.cat([x, c], dim=1)
        return self.inner(u, time, *args, **kwargs)


class ConditionalDiffusionCorrector(nn.Module):
    """
    概要:
        条件 cond（例: Stage1確率）と正解 x0（GT 確率）のペアで学習する条件付き拡散モデル。
        学習では cond は常にモデルに結合され、拡散は x0 側にのみ適用される。
        推論では cond に近い x0 を t_start から逆拡散で再構成し、補正済み確率を得る。

    入力（主に forward / correct_from_probs_cond で使用）:
        - image_size, channels, cond_channels（例: C_data=C_cond=6）
        - U-Net の基本幅 base_dim と解像度スケール dim_mults

    処理:
        - _CondUNetWrapper を通して、x_t と cond をチャネル結合して U-Net に入力
        - 学習時は forward(x0, cond) を呼び、損失を返す
        - 推論時は correct_from_probs_cond(cond, ...) で refined 確率を生成

    出力:
        - forward: 損失（torch.Tensor, scalar）
        - correct_from_probs_cond: 補正後の確率マップ（(ensemble*B, C, H, W)）
    """
    def __init__(
        self,
        image_size: int = 128,
        channels: int = 6,
        cond_channels: int = 6,
        base_dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.0,
        objective: str = 'pred_v',
        beta_schedule: str = 'sigmoid',
        timesteps: int = 1000,
        sampling_timesteps: int = 20,
        auto_normalize: bool = True,
        flash_attn: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        概要:
            条件付き拡散モデル（U-Net + GaussianDiffusion + 条件連結ラッパ）を初期化する。

        入力:
            - image_size (int): 入出力の正方形画像サイズ H=W
            - channels (int): データ側のチャネル数（例: 6）
            - cond_channels (int): 条件側のチャネル数（例: 6）
            - base_dim (int), dim_mults (Tuple[int,...]): U-Net のスケール設定
            - dropout (float): U-Net 内の dropout 率
            - objective, beta_schedule, timesteps, sampling_timesteps, auto_normalize, flash_attn: 拡散の各種設定
            - device (torch.device|None): モデル配置先

        処理:
            - 入力は x_t(6) + cond(6) = 12 チャネル、出力はデータ側 6 チャネルの U-Net を構築
            - _CondUNetWrapper で cond を内部保持し、forward 時に結合
            - GaussianDiffusion を組み立てる

        出力:
            なし（初期化）
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.cond_channels = cond_channels

        # 入力は x_t(6) + cond(6) = 12 チャネル、出力はデータ側 6 チャネル
        unet_in_ch = channels + cond_channels
        base_unet = Unet(
            dim=base_dim,
            init_dim=None,
            out_dim=channels,             # データ側のチャネル数
            dim_mults=dim_mults,
            channels=unet_in_ch,          # 入力チャネル数（x_t + cond）
            self_condition=False,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            dropout=dropout,
            attn_dim_head=32,
            attn_heads=4,
            full_attn=(False, False, False, False),
            flash_attn=flash_attn
        )
        self._wrapper = _CondUNetWrapper(base_unet, data_channels=channels, cond_channels=cond_channels)

        self.diffusion = GaussianDiffusion(
            model=self._wrapper,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.0,
            auto_normalize=auto_normalize,
            offset_noise_strength=0.0,
            min_snr_loss_weight=False,
            immiscible=False
        )

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.device = device

    def _to_bchw(self, arr: torch.Tensor) -> torch.Tensor:
        """
        概要:
            (B, C, H, W) あるいは (B, H, W, C) を (B, C, H, W) 形式に整える。

        入力:
            - arr (Tensor): 4次元テンソル。データ側 C または 条件側 C と一致している必要あり

        処理:
            - 既に (B, C, H, W) で、C が data/cond どちらかと一致していればそのまま返す
            - (B, H, W, C) の場合は permute で並べ替える
            - 上記以外は例外

        出力:
            - bchw (Tensor): 形状 (B, C, H, W)
        """
        # (B,C,H,W) or (B,H,W,C) を (B,C,H,W) にする
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {arr.ndim}D")
        if arr.shape[1] == self.channels or arr.shape[1] == self.cond_channels:
            return arr
        if arr.shape[-1] in (self.channels, self.cond_channels):
            return arr.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unexpected tensor shape {tuple(arr.shape)}")

    def set_condition(self, cond: torch.Tensor):
        """
        概要:
            条件 cond を [-1,1] に正規化して内部ラッパにセットする。

        入力:
            - cond (Tensor): 形状 (B, C=cond_channels, H=image_size, W=image_size) または (B, H, W, C)
                              値域 [0,1] を想定

        処理:
            - _to_bchw で (B,C,H,W) へ整形
            - 空間サイズが image_size と一致しているか検証
            - GaussianDiffusion.normalize で [-1,1] に正規化
            - _wrapper.set_cond(cond_norm) を実行

        出力:
            なし（内部状態更新）
        """
        cond = self._to_bchw(cond).to(self.device, dtype=torch.float32)
        if cond.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"Condition spatial size mismatch: expected {(self.image_size, self.image_size)}, got {tuple(cond.shape[2:])}")
        cond_norm = self.diffusion.normalize(cond)
        self._wrapper.set_cond(cond_norm)

    def forward(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        概要:
            学習用の損失を返す。必要に応じて cond を先にセットしてから呼ぶ。

        入力:
            - x0 (Tensor): 形状 (B, C=channels, H, W)、[0,1] の GT 確率（one-hot など）
            - cond (Tensor|None): 形状 (B, C=cond_channels, H, W) または (B, H, W, C)、[0,1]
                                  指定された場合は内部に set してから学習損失を計算

        処理:
            - cond が与えられた場合は set_condition(cond)
            - GaussianDiffusion(x0) を呼び、損失を得る

        出力:
            - loss (Tensor): スカラー損失
        """
        if cond is not None:
            self.set_condition(cond)
        return self.diffusion(x0)

    @property
    def num_timesteps(self) -> int:
        """
        概要:
            拡散過程の総時刻数 T を返す（GaussianDiffusion.num_timesteps）。

        入力:
            なし

        処理:
            - 内部から参照

        出力:
            - T (int)
        """
        return self.diffusion.num_timesteps

    @torch.inference_mode()
    def _denoise_from_cond(
        self,
        x_t: torch.Tensor,
        t_start: int,
        steps: Optional[int] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        概要:
            既に set_condition(cond_norm) 済みである前提で、条件付きで x_t から 0 まで逆拡散する。

        入力:
            - x_t (Tensor): 形状 (B, C, H, W)、[-1,1] の正規化域
            - t_start (int): 開始時刻（0..T-1）
            - steps (int|None): サンプリングステップ数
            - eta (float): DDIM の stochasticity

        処理:
            - _wrapper に保持された cond を用い、model_predictions により DDIM 更新式で x_{t-1} を生成
            - t_next < 0 のときは x_start を返す

        出力:
            - x_0 (Tensor): (B, C, H, W)、[0,1]
        """
        assert x_t.ndim == 4 and x_t.shape[1] == self.channels
        img = x_t
        B = img.shape[0]
        device = img.device

        total_T = self.num_timesteps
        steps = steps or self.diffusion.sampling_timesteps
        steps = max(1, int(steps))

        times = torch.linspace(-1, t_start, steps + 1, dtype=torch.int64, device=device).flip(0)
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((B,), int(time), device=device, dtype=torch.long)
            pred_noise, x_start_pred, *_ = self.diffusion.model_predictions(
                img, time_cond, None, clip_x_start=True, rederive_pred_noise=True
            )
            x_start = x_start_pred

            if time_next < 0:
                img = x_start
                continue

            alpha = self.diffusion.alphas_cumprod[time]
            alpha_next = self.diffusion.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return self.diffusion.unnormalize(img)

    @torch.inference_mode()
    def correct_from_probs_cond(
        self,
        cond: torch.Tensor,
        steps: Optional[int] = None,
        t_start: Optional[int] = None,
        ensemble: int = 1
    ) -> torch.Tensor:
        """
        概要:
            条件 cond（例: Stage1確率）を用いて refined 確率をサンプルする。

        入力:
            - cond (Tensor): 形状 (B, C=channels, H, W) または (B, H, W, C)、[0,1]
            - steps (int|None): 逆拡散ステップ数（未指定は学習時設定）
            - t_start (int|None): 開始時刻（未指定は T-1）
            - ensemble (int): 生成サンプル数

        処理:
            - cond を (B,C,H,W) に整え、set_condition(cond) で内部に [-1,1] 正規化してセット
            - 初期 x0_norm を cond と同一分布からとし、q(x_t|x0_norm, t_start) で前向きにノイズ付加
            - _denoise_from_cond により逆拡散し、[0,1] にクリップ後チャネル正規化

        出力:
            - rec (Tensor): (ensemble*B, C, H, W)、[0,1]
        """
        c_bchw = self._to_bchw(cond).to(self.device, dtype=torch.float32)
        if c_bchw.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"Condition spatial size mismatch: expected {(self.image_size, self.image_size)}, got {tuple(c_bchw.shape[2:])}")

        # cond を条件としてセット（内部で [-1,1] 域へ正規化）
        self.set_condition(c_bchw)

        # 初期 x0 を cond と同一分布から開始し、t_start まで前向きに進めてから逆拡散
        x0_norm = self.diffusion.normalize(c_bchw)

        t_start = self.num_timesteps - 1 if t_start is None else int(t_start)
        t_start = max(0, min(self.num_timesteps - 1, t_start))

        outs: List[torch.Tensor] = []
        for _ in range(max(1, ensemble)):
            t_vec = torch.full((c_bchw.shape[0],), t_start, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0_norm)
            x_t = self.diffusion.q_sample(x_start=x0_norm, t=t_vec, noise=noise)
            rec = self._denoise_from_cond(x_t, t_start=t_start, steps=steps, eta=0.0)  # [0,1]
            rec = torch.clamp(rec, 0.0, 1.0)
            rec = rec / (rec.sum(dim=1, keepdim=True) + 1e-8)
            outs.append(rec)

        return torch.cat(outs, dim=0)
