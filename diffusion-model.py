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
    Swin-Unet (Predictor) の出力を確率的に補正するための拡散モデルラッパ
    - 学習: 6チャネル(one-hot前線クラス, 0-5)の画像をそのまま学習データとして使用 (unpaired)
    - 推論: Stage1の確率マップ(0..1, shape HxWx6) を初期状態 x0 とみなし、
            q(x_t | x0) で t=t_start のノイズを加えてから、t=t_start..0 で逆拡散
    - 出力: 6チャネルの確率マップ (各画素でチャンネル正規化して確率へ)
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
        x: (B, C=channels, H=image_size, W=image_size), 値域は [0,1] を想定
        return: scalar loss
        """
        return self.diffusion(x)

    @property
    def num_timesteps(self) -> int:
        return self.diffusion.num_timesteps

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True):
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
        任意の時刻 t_start の x_t から 0 まで逆拡散 (DDIM ライク)
        x_t: (B, C, H, W)  正規化済み([-1,1]域)を想定
        return: (B, C, H, W) in [0,1] （自動で unnormalize）
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
        Stage1の確率マップから補正結果を生成
        prob_map: (B, H, W, C) or (B, C, H, W) かつ 0..1
        steps: 逆拡散ステップ数（DDIMのサンプルステップ数）。省略時は学習時設定を使用
        t_start: 逆拡散の開始時刻（0..T-1）。未指定なら T-1
        sigma_override: 使わない（将来的にPSDベースの σ → t_start マッピングに対応予定）
        ensemble: 生成サンプル数
        return: (ensemble*B, C, H, W) 0..1 の確率マップ（各画素でチャネル正規化済み）
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
        labels: (B, H, W) int64 in [0..num_classes-1]
        return: (B, C, H, W) float32 in {0,1}
        """
        assert labels.ndim == 3
        B, H, W = labels.shape
        oh = F.one_hot(labels.long(), num_classes=num_classes)  # (B, H, W, C)
        oh = oh.permute(0, 3, 1, 2).contiguous().float()
        return oh
