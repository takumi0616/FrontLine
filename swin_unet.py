import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import math


class MoEFFNGating(nn.Module):
    """
    概要:
        Mixture-of-Experts 風の前段全結合(FFN)ブロック。
        ゲーティングネットワークで各エキスパートへの重みを計算し、エキスパートの出力を重み付き合成する。

    入力:
        - dim (int): 入力/出力特徴量の次元数
        - hidden_dim (int): 各エキスパート内部の中間層の次元数
        - num_experts (int): エキスパート（FFN）の本数

    処理:
        - gating_network: 入力 x から各次元ごとの重み（softmax後）を算出
        - experts: num_experts 本の MLP を並列適用（同一構造）
        - 重み付き合成: (weights.unsqueeze(0) * outputs).sum(dim=0)

    出力:
        - (Tensor): 形状は入力と同じ (..., dim)
    """
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])

    def forward(self, x):
        """
        概要:
            入力に対してエキスパートの重みを計算し、各エキスパート出力を重み付きで合成する。

        入力:
            - x (Tensor): 形状 (..., dim)

        処理:
            - weights = softmax(gating_network(x), dim=-1)
            - outputs = [expert(x) for expert in experts] を stack
            - 要素ごとに weights を掛けて合計

        出力:
            - (Tensor): 形状 (..., dim) の合成出力
        """
        weights = self.gating_network(x)
        weights = torch.nn.functional.softmax(weights, dim=-1)
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs


class Mlp(nn.Module):
    """
    概要:
        Swinブロック内で用いられる2層MLP（全結合→活性化→Dropout→全結合→Dropout）。

    入力:
        - in_features (int): 入力次元
        - hidden_features (int|None): 中間層の次元（省略時は in_features）
        - out_features (int|None): 出力次元（省略時は in_features）
        - act_layer (nn.Module): 活性化関数（既定: GELU）
        - drop (float): Dropout率

    出力:
        - forward(x): (Tensor) 入力と同形か out_features の次元に変換されたテンソル
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        概要:
            2層全結合による非線形変換（間に活性化とDropout）を行う。

        入力:
            - x (Tensor): (..., in_features)

        出力:
            - y (Tensor): (..., out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    概要:
        入力特徴 (B, H, W, C) を window_size ごとのパッチに分割する。

    入力:
        - x (Tensor): (B, H, W, C)
        - window_size (int): ウィンドウの辺長

    処理:
        - 画像を (H//ws, W//ws) のグリッドで分割し、それぞれ (ws, ws, C) の小パッチに整形

    出力:
        - windows (Tensor): (B * (H//ws) * (W//ws), ws, ws, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    概要:
        window_partition の逆操作。分割パッチを (B, H, W, C) に戻す。

    入力:
        - windows (Tensor): (B * nW, ws, ws, C)
        - window_size (int): ws
        - H, W (int): 復元後の高さ・幅

    出力:
        - x (Tensor): (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    概要:
        Swin Transformer のウィンドウ注意 (Window-based Multi-Head Self-Attention) 層。
        相対位置バイアスを用いて、局所窓内で自己注意を計算する。

    入力:
        - dim (int): 入出力の埋め込み次元
        - window_size (Tuple[int,int]): ウィンドウサイズ (Wh, Ww)
        - num_heads (int): ヘッド数
        - qkv_bias, qk_scale, attn_drop, proj_drop: 注意計算の補助パラメータ

    出力:
        - forward(x, mask): (Tensor) 形状 (B*nW, N, dim)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        概要:
            ウィンドウ内のトークンに対して自己注意を計算し、線形射影で出力次元に戻す。

        入力:
            - x (Tensor): (B_* , N, C) ここで B_* = バッチ×ウィンドウ数, N = ウィンドウ内トークン数
            - mask (Tensor|None): シフトウィンドウ時のマスク (nW, N, N)

        処理:
            - qkv = Linear(x) をヘッドに分割し、スケーリング付きスコア qk^T を計算
            - 相対位置バイアスを加算し、softmax→dropout
            - 値 v との重み付き和を取り、最後に線形射影・dropout

        出力:
            - y (Tensor): (B_*, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        """
        概要:
            インスタンスの簡易要約文字列を返す（デバッグ用途）。
        """
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """
        概要:
            FLOPs（概算）を返す。

        入力:
            - N (int): ウィンドウ内トークン数 (ws*ws)

        出力:
            - flops (int): 演算量の概算
        """
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """
    概要:
        Swin Transformer の基本ブロック。Shifted Window Attention と MLP からなる。
        入力は (B, H*W, C) のトークン列で、ウィンドウ分割→注意→逆変換を行う。

    入力:
        - dim (int): 埋め込み次元 C
        - input_resolution (Tuple[int,int]): トークンを画像に戻したときの (H, W)
        - num_heads (int): 注意ヘッド数
        - window_size (int): ウィンドウ辺長
        - shift_size (int): シフト量（0 なら非シフト）
        - mlp_ratio (float): MLP の拡張比
        - その他 drop 系、正規化層など

    出力:
        - forward(x): (B, H*W, C)
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        概要:
            正規化→（オプションでシフト）→ウィンドウ分割→注意→逆変換→逆シフト→残差 + MLP の流れ。

        入力:
            - x (Tensor): (B, H*W, C)

        出力:
            - y (Tensor): (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self):
        """
        概要:
            インスタンスの簡易要約文字列。
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        """
        概要:
            当該ブロックのおおよその FLOPs を返す。
        """
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """
    概要:
        2x2 のパッチを結合してダウンサンプリングする層（パッチマージ）。
        入力 (B, H*W, C) -> 出力 (B, (H/2)*(W/2), 2C)

    入力:
        - input_resolution (Tuple[int,int]): (H, W)
        - dim (int): 入力チャネル C
        - norm_layer: 正規化層（既定: LayerNorm）

    出力:
        - forward(x): (B, (H/2)*(W/2), 2C)
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        概要:
            2x2 ブロック (x0, x1, x2, x3) をチャネル方向に結合し、正規化→線形で次段の次元に圧縮。

        入力:
            - x (Tensor): (B, H*W, C)

        出力:
            - y (Tensor): (B, (H/2)*(W/2), 2C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """
        概要:
            FLOPs の概算。
        """
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    """
    概要:
        パッチレベルでのアップサンプリング（x2）を行う層。
        線形拡張→rearrange で空間次元を2倍にする。

    入力:
        - input_resolution (Tuple[int,int]): (H, W) 入力の空間解像度
        - dim (int): 入力チャネル
        - dim_scale (int): スケール（既定 2）
        - norm_layer: 正規化層（既定: LayerNorm）

    出力:
        - forward(x): (B, (2H)*(2W), C/2)
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        概要:
            線形層でチャネルを 2 倍にし、rearrange で空間次元に展開→正規化。

        入力:
            - x (Tensor): (B, H*W, C)

        出力:
            - y (Tensor): (B, (2H)*(2W), C/2)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',                    p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    概要:
        最終段のアップサンプリング（x4）を一括で行う層。
        ConvTranspose ではなく、線形展開＋rearrange で4倍に拡大。

    入力:
        - input_resolution (Tuple[int,int]): (H, W)
        - dim (int): 入力チャネル
        - dim_scale (int): 4 を想定
        - norm_layer: 正規化層

    出力:
        - forward(x): (B, 4H*4W, dim)
    """
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        入力:
            - x (Tensor): (B, H*W, dim)
        出力:
            - y (Tensor): (B, 4H*4W, dim)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',                    p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class FinalPatchExpand(nn.Module):
    """
    概要:
        最終アップサンプリング用の一般化クラス（dim_scale 倍）。

    入力:
        - input_resolution (Tuple[int,int])
        - dim (int)
        - dim_scale (int)
        - norm_layer

    出力:
        - forward(x): (B, (dim_scale*H)*(dim_scale*W), dim)
    """
    def __init__(self, input_resolution, dim, dim_scale, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim_scale * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        入力:
            - x (Tensor): (B, H*W, dim)
        出力:
            - y (Tensor): (B, (dim_scale*H)*(dim_scale*W), dim)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(        x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',                    p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """
    概要:
        Swin ブロックを depth 個積層したエンコーダ層。必要に応じて PatchMerging による下流ダウンサンプルを持つ。

    入力:
        - dim (int), input_resolution (H,W), depth (int), num_heads, window_size, ほか
        - downsample: PatchMerging などのクラス（None ならダウンサンプルなし）

    出力:
        - forward(x): 出力トークン (B, H'*W', C') （downsample 有無で空間/チャネルが変化）
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """
        概要:
            depth 回の SwinTransformerBlock を適用し、必要なら downsample で空間を半分にする。

        入力:
            - x (Tensor): (B, H*W, C)

        出力:
            - y (Tensor): (B, H'*W', C')  (downsample ありなら H' = H/2, C' = 2C など)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """
        概要:
            本レイヤー全体の FLOPs 概算。
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """
    概要:
        デコーダ側のアップサンプリング層。複数の Swin ブロック + PatchExpand で空間解像を上げる。

    入力:
        - dim, input_resolution, depth, num_heads, window_size, ほか
        - upsample: PatchExpand などのクラス（None ならアップサンプルなし）

    出力:
        - forward(x): (B, (2H)*(2W), C/2) など、解像度とチャネルが変わる
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        """
        概要:
            depth 回の Swin ブロックを適用し、必要ならアップサンプリングで空間を2倍にする。

        入力:
            - x (Tensor): (B, H*W, C)

        出力:
            - y (Tensor): (B, (2H)*(2W), C/2) など
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    """
    概要:
        画像 (B, C, H, W) をパッチに分割し、各パッチを埋め込み (embed_dim) に射影して
        (B, N, embed_dim) のトークン列に変換する。

    入力:
        - img_size (int|Tuple[int,int]): 入力画像サイズ
        - patch_size (int|Tuple[int,int]): パッチサイズ
        - in_chans (int): 入力チャネル数
        - embed_dim (int): 埋め込み次元
        - norm_layer: 埋め込み後の正規化層（patch_norm=True のとき）

    出力:
        - forward(x): (B, N, embed_dim)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        入力:
            - x (Tensor): (B, C, H, W) かつ H, W は img_size と一致
        出力:
            - y (Tensor): (B, N, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """
        概要:
            埋め込み層の FLOPs 概算。
        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerSys(nn.Module):
    """
    概要:
        U-Net 風のエンコーダ・デコーダ構成で出力マップを生成する Swin Transformer システム。
        エンコーダで解像度を段階的に下げつつ特徴抽出し、デコーダで skip 結合しながら元解像度に戻す。

    入力:
        - img_size, patch_size, in_chans, num_classes
        - embed_dim, depths, depths_decoder, num_heads, window_size, mlp_ratio
        - qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate
        - norm_layer, ape(絶対位置埋め込みの有無), patch_norm, use_checkpoint
        - final_upsample ("expand_first"): 最終アップサンプル方式

    出力:
        - forward(x): (B, num_classes, H, W) のロジット
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=patch_size,
                dim=embed_dim
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        概要:
            重み初期化。Linearはtruncated normal、LayerNormは標準初期化。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        概要:
            weight decay を適用しないパラメータ名の集合を返す。
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """
        概要:
            weight decay を適用しないキーワードを返す。
        """
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        """
        概要:
            パッチ埋め込み→（絶対位置を加算）→ドロップアウト→エンコーダ層を通して特徴を得る。

        入力:
            - x (Tensor): (B, in_chans, H, W)

        出力:
            - x (Tensor): 最終層出力 (B, H_L*W_L, C_L)
            - x_downsample (List[Tensor]): 各段の特徴（skip 結合用）
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        """
        概要:
            デコーダ側のアップサンプルを行い、skip 結合で特徴を連結・整形する。

        入力:
            - x (Tensor): エンコーダ最終段の出力
            - x_downsample (List[Tensor]): 各解像度での特徴（高解像から順に使用）

        出力:
            - y (Tensor): (B, H*W, embed_dim)
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def up_x(self, x):
        """
        概要:
            トークン列 (B, H*W, C) を最終的な空間テンソル (B, C, H, W) に戻し、1x1 Conv で出力する。

        入力:
            - x (Tensor): (B, H*W, embed_dim)

        出力:
            - y (Tensor): (B, num_classes, H, W)
        """
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, self.up.dim_scale * H, self.up.dim_scale * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output(x)
        return x

    def forward(self, x):
        """
        概要:
            エンコーダ→デコーダ→空間復元→出力（ロジット）までの全処理。

        入力:
            - x (Tensor): (B, in_chans, H, W)

        出力:
            - y (Tensor): (B, num_classes, H, W)
        """
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x(x)
        return x

    def flops(self):
        """
        概要:
            ネットワーク全体のおおよその FLOPs を算出。
        """
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
