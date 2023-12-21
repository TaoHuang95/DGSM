# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.parameter import Parameter
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Conv_up(nn.Module):
    def __init__(self, c_in, up_factor):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        ## x3 00
        ## x2 11
        if up_factor == 2:
            modules_tail = [
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=1, output_padding=1),
                conv(64, c_in, 3)]
        elif up_factor == 3:
            modules_tail = [
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=0, output_padding=0),
                conv(64, c_in, 3)]
        elif up_factor == 4:
            modules_tail = [
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                conv(64, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):
        out = self.body(input)
        out = self.tail(out)
        return out


class Conv_down(nn.Module):
    def __init__(self, c_in, up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        if up_factor == 4:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                conv(64, c_in, 3)]
        elif up_factor == 3:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                conv(64, c_in, 3)]
        elif up_factor == 2:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                conv(64, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):
        out = self.body(input)
        out = self.tail(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
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

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
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
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, isDownsample=True):
        super().__init__()
        self.dim = dim
        self.isDownsample = isDownsample
        if self.isDownsample:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)
        else:
            self.conv = nn.Conv2d(dim, 4 * dim, 3, 1, 1)
            self.pixelshuffle = nn.PixelShuffle(2)
            self.reduction = nn.Linear(dim, int(0.5 * dim), bias=False)
            self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape

        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        if self.isDownsample:
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

            x = self.norm(x)
            x = self.reduction(x)
        else:
            x = self.pixelshuffle(self.conv(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
            x = x.view(B, -1, C)

            x = self.norm(x)
            x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
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

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, isDownsample=True)
        else:
            self.downsample = None

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer, isDownsample=False)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        H, W = x_size
        B, _, C = x.size()
        x = self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        short_con = x
        if self.downsample is not None:
            x = x.view(B, H, W, C)
            x = self.downsample(x)
        if self.upsample is not None:
            x = x.view(B, H, W, C)
            x = self.upsample(x)
        return x, short_con

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
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

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class DGSMP(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(DGSMP, self).__init__()
        self.Ch = 28
        self.s = 1

        self.img_range = img_range

        self.upscale = upscale

        self.window_size = window_size

        self.Isdownsample = [True, True, False, False, False]
        self.Isupsample = [False, False, True, True, False]

        # # The modules for learning the measurement matrix A and A^T
        # self.AT = Conv_up(c_in=self.Ch, up_factor=self.factor)
        # self.A = Conv_down(c_in=self.Ch, up_factor=self.factor)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        ## Dense connection
        self.conv = nn.Conv2d(self.Ch, 64, kernel_size=3, stride=1, padding=1)
        self.Den_con1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con2 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con3 = nn.Conv2d(64 * 3, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con4 = nn.Conv2d(64 * 4, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con5 = nn.Conv2d(64 * 5, 64, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(64 * 6, 64, kernel_size=1, stride=1, padding=0)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        size_change = [1, 2, 4, 2, 1]

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0] // size_change[i_layer],
                                           patches_resolution[1] // size_change[i_layer]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         downsample=PatchMerging if self.Isdownsample[i_layer] else None,
                         upsample=PatchMerging if self.Isupsample[i_layer] else None,
                         resi_connection=resi_connection
                         )
            if self.Isdownsample[i_layer]:
                embed_dim = embed_dim * 2
            else:
                embed_dim = int(embed_dim * 0.5)
            self.layers.append(layer)
        self.short_cat_1 = nn.Linear(128 * 2, 128, bias=False)
        self.short_cat_2 = nn.Linear(64 * 2, 64, bias=False)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_U = nn.Conv2d(64, 28, 3, 1, 1)
        self.conv_W = nn.Sequential(nn.Conv2d(64, 28, 3, 1, 1))

        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size * 4 - h % (self.window_size * 4)) % (self.window_size * 4)
        mod_pad_w = (self.window_size * 4 - w % (self.window_size * 4)) % (self.window_size * 4)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w = x.size()
        return x

    def y2x(self, y):
        ##  Spilt operator
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, 28, sz[2], sz[2]]).cuda()
        for t in range(28):
            temp = y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def x2y(self, x):
        ##  Shift and Sum operator
        sz = x.size()
        if len(sz) == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = x.size()
        y = torch.zeros([bs, 1, sz[2], sz[2] + 2 * 27]).cuda()
        for t in range(28):
            y[:, :, :, 0 + 2 * t: sz[2] + 2 * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
        return y

    def forward_features(self, fufeat):
        Feat_list = []
        fufeat_size = (fufeat.shape[2], fufeat.shape[3])
        x_feat = self.patch_embed(fufeat)
        if self.ape:
            x_feat = x_feat + self.absolute_pos_embed
        x_feat = self.pos_drop(x_feat)

        x_size = fufeat_size
        l = 0
        for layer in self.layers:
            x_feat, short_Feat = layer(x_feat, x_size)
            Feat_list.append(short_Feat)
            if len(Feat_list) == 3:
                x_feat = self.short_cat_1(torch.cat((Feat_list[1], x_feat), dim=2))
            elif len(Feat_list) == 4:
                x_feat = self.short_cat_2(torch.cat((Feat_list[0], x_feat), dim=2))
            if self.Isdownsample[l]:
                x_size = (x_size[0] // 2, x_size[1] // 2)
            else:
                x_size = (x_size[0] * 2, x_size[1] * 2)
            l = l + 1

        x_feat = self.norm(x_feat)  # B L C
        x_feat = self.patch_unembed(x_feat, fufeat_size)
        return x_feat

    def recon(self, res1, res2, Xt, i):
        if i == 0:
            delta = self.delta_0
        # elif i == 1:
        #     delta = self.delta_1
        # elif i == 2:
        #     delta = self.delta_2
        # elif i == 3:
        #     delta = self.delta_3
        # elif i == 4:
        #     delta = self.delta_4
        # elif i == 5:
        #     delta = self.delta_5

        Xt = Xt - 2 * delta * (res1 + res2)
        return Xt

    def forward(self, y, mask):
        ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
        Xt = self.y2x(y)
        sz = Xt.size()

        feature_list = []
        for i in range(0, self.s):
            AXt = self.x2y(Xt * mask.repeat(1, 28, 1, 1))

            AXt_y = self.y2x(AXt - y)

            Res1 = AXt_y * mask.repeat(1, 28, 1, 1)

            X = self.check_image_size(Xt)
            feat = self.conv(X)
            shortcut = feat

            if i == 0:
                feature_list.append(feat)
                fufeat = self.Den_con1(feat)  # 256 x 256 x 64
            # elif i == 1:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con2(torch.cat(feature_list, 1))
            # elif i == 2:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con3(torch.cat(feature_list, 1))
            # elif i == 3:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con4(torch.cat(feature_list, 1))
            # elif i == 4:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con5(torch.cat(feature_list, 1))
            # elif i == 5:
            #     feature_list.append(feat)
            #     fufeat = self.Den_con6(torch.cat(feature_list, 1))

            # shortcut = fufeat
            ## generate U and w
            x_feat = self.forward_features(fufeat)
            x_feat = x_feat + shortcut

            U = self.conv_U(x_feat)
            W = torch.exp(self.conv_W(x_feat))
            # W = F.relu(self.conv_W(x_feat))

            U = U[:, :, :sz[2], :sz[3]]
            W = W[:, :, :sz[2], :sz[3]]

            ## w * (x − u)
            Res2 = (Xt - U).mul(W)

            ## Reconstructing HSIs
            Xt = self.recon(Res1, Res2, Xt, i)

        return Xt

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 10
    height = 200
    width = 200
    model = DGSMP(upscale=1, img_size=(height, width),
                  window_size=window_size, img_range=1., depths=[2, 2, 2, 2, 2],
                  embed_dim=64, num_heads=[4, 4, 4, 4, 4], mlp_ratio=4)
    model = model.cuda()
    print(model)

    x = torch.randn((1, 1, height, width+54)).to(device)
    m = torch.randn((1, 1, height, width)).to(device)
    x = model(x, m)
    print(x.shape)
