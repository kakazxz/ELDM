# -*- encoding:utf-8 -*-
"""
@Author: Kaka
@IDE: PyCharm
@Project: diff
@File: img2img
@TIME: 2022/11/18 10:34
@Description:
"""
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
import imageio
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
import numpy as np
# constants
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from loss import FocalLoss,perceptual_loss

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# 损失函数
mse_loss = nn.MSELoss(reduction='none')
focal_loss = FocalLoss(alpha=0.8, gamma=2.0)

import torchvision.models as models
# 加载预训练的 VGG-19 用于感知损失
vgg19 = models.vgg19(pretrained=True).features
vgg_layers = [0, 5, 10, 19, 28]  # 选择部分卷积层用于感知损失
selected_layers = [vgg19[i] for i in vgg_layers]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = nn.Sequential(*selected_layers).eval().to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:

            yield data

def val_cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def img_to_groups(num, divisor,channel,imgsize,device):
    groups = num // divisor
    remainder = num % divisor
    # arr = [divisor] * groups
    arr=[]
    for i in range(0,groups):
        arr.append(torch.randn([divisor,channel,imgsize,imgsize],device=device))
    if remainder > 0:
        arr.append(torch.randn([remainder,channel,imgsize,imgsize],device=device))
    return arr



def sample_img_to_groups(dl,num, divisor,device,results_folder,q_sample,num_timesteps):
    # num_timesteps=1
    groups = num // divisor
    remainder = num % divisor
    # arr = [divisor] * groups
    arr=[]

    datas=[]
    conds=[]
    conds_n = []
    for i in range(0,groups):
        data, cond = next(dl)

        data=data.to(device).float()
        cond_n = normalize_to_neg_one_to_one(cond).float()

        # t = torch.randint(0, num_timesteps, (divisor,), device=device).long()
        t = torch.full((divisor,),num_timesteps-1,  device=device).long()

        # sample=q_sample(data_n,t)
        sample = torch.randn_like(data,device=device)
        datas.append(data)

        conds.append(cond)
        conds_n.append(cond_n)
        arr.append(sample)

    if remainder > 0:
        data, cond = next(dl)
        data=data[0:remainder]
        cond = cond[0:remainder]
        # data = data.to(device)
        data = data.to(device).float()
        cond_n = normalize_to_neg_one_to_one(cond).float()
        # t = torch.randint(0, num_timesteps, (remainder,), device=device).long()
        t = torch.full((remainder,), num_timesteps-1, device=device).long()
        sample = q_sample(data, t)

        datas.append(data)

        conds.append(cond)
        conds_n.append(cond_n)
        arr.append(sample)
        # arr.append(torch.randn([remainder,channel,imgsize,imgsize],device=device))
    # utils.save_image(torch.cat(datas, dim=0), str(results_folder / 'target.png'),nrow=int(math.sqrt(num)))
    # utils.save_image(torch.cat(conds, dim=0), str(results_folder / 'input.png'), nrow=int(math.sqrt(num)))
    # utils.save_image(torch.cat(arr, dim=0), str(results_folder / 'noise.png'), nrow=int(math.sqrt(num)))
    return arr,conds_n

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules


def tensor2np(tensor):
    """把 0-1 的 float32 张量转换成 0-255 的 uint8 numpy HWC 格式"""
    return (tensor.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype(np.uint8)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Residual_Cross(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x,content, *args, **kwargs):
        return self.fn(x,content, *args, **kwargs) + x


# def Upsample(dim, dim_out = None):
#     return nn.Sequential(
#         nn.Upsample(scale_factor = 2, mode = 'nearest'),
#         nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
#     )

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.ConvTranspose2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class PreNorm_Cross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x,content):
        x = self.norm(x)
        return self.fn(x,content)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        # print(x.shape)
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)


# class LinearAttention_Cross(nn.Module):
#     def __init__(self, dim, cond_dim,heads = 4, dim_head = 32):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         # self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
#         self.to_q = nn.Conv2d(dim, hidden_dim , 1, bias=False)
#         self.to_k = nn.Conv2d(cond_dim, hidden_dim , 1, bias=False)
#         self.to_v = nn.Conv2d(cond_dim, hidden_dim , 1, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Conv2d(hidden_dim, dim, 1),
#             LayerNorm(dim)
#         )
#
#     def forward(self, x,content):
#         b, c, h, w = x.shape
#         # qkv = self.to_qkv(x).chunk(3, dim = 1)
#         # q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
#         # print(x.shape,content.shape)
#         q = self.to_q(x)
#         k = self.to_k(content)
#         v = self.to_v(content)
#         q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads)
#         k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
#         v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)
#         q = q.softmax(dim = -2)
#         k = k.softmax(dim = -1)
#
#         q = q * self.scale
#         v = v / (h * w)
#
#         context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
#
#         out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
#         out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
#         return self.to_out(out)

class LinearAttention_Cross(nn.Module):
    def __init__(self, dim, cond_dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(cond_dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(cond_dim, hidden_dim, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, content):
        b, c, h, w = x.shape
        _, _, h_c, w_c = content.shape

        q = self.to_q(x)  # (b, hidden_dim, h, w)
        k = self.to_k(content)  # (b, hidden_dim, h_c, w_c)
        v = self.to_v(content)  # (b, hidden_dim, h_c, w_c)

        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads)  # (b, h, c, n)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)  # (b, h, c, m)
        v = rearrange(v, 'b (h d) x y -> b h d (x y)', h=self.heads)  # (b, h, d, m)

        q = q.softmax(dim=-1)
        # k = k.softmax(dim=-1)

        # context: k^T @ v -> (b, h, c, d)
        context = torch.einsum('b h c m, b h d m -> b h c d', k, v)
        context = context / (h_c * w_c)
        # out: context @ q -> (b, h, d, n)
        out = torch.einsum('b h c d, b h c n -> b h d n', context, q)

        out = rearrange(out, 'b h d (x y) -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class Attention_Cross(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, hidden_dim , 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim , 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim , 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x,content):
        b, c, h, w = x.shape
        q = self.to_q(x)
        k = self.to_k(content)
        v = self.to_v(content)
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class ConditionalLinearAttention(nn.Module):
    def __init__(self, dim, cond_dim=None, heads=4, dim_head=32):
        """
        Linear Attention with condition injected into K and V via FiLM-like modulation.

        Args:
            dim (int): Input feature dimension.
            cond_dim (int, optional): Condition embedding dimension. If None, no condition.
            heads (int): Number of attention heads.
            dim_head (int): Dimension per head.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # Optional projection for condition
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.to_cond = nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(cond_dim, hidden_dim * 4, 1)  # 2x for scale & shift (k and v)
            )
            # Initialize to identity (no change at start)
            nn.init.zeros_(self.to_cond[-1].weight)
            nn.init.zeros_(self.to_cond[-1].bias)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, cond=None):
        """
        Args:
            x (tensor): Input feature map, shape (b, c, h, w)
            cond (tensor): Condition embedding, shape (b, c_cond, h, w) or (b, c_cond, 1, 1)

        Returns:
            tensor: Output feature map with condition applied
        """
        b, c, h, w = x.shape

        # Compute QKV
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # -----------------------------
        # Inject condition into K and V
        # -----------------------------
        if self.cond_dim is not None and cond is not None:
            # Project condition to (scale_k, shift_k, scale_v, shift_v)
            cond_out = self.to_cond(cond)
            cond_scale_k, cond_shift_k, cond_scale_v, cond_shift_v = cond_out.chunk(4, dim=1)

            # Apply modulation: k = (1 + scale) * k + shift
            k = (1 + cond_scale_k) * k + cond_shift_k
            v = (1 + cond_scale_v) * v + cond_shift_v

        # -----------------------------
        # Linear Attention
        # -----------------------------
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)  # column-wise
        k = k.softmax(dim=-1)  # row-wise

        q = q * self.scale
        v = v / (h * w)  # normalize to prevent large values

        # Context: k^T @ v -> (b h c d)
        context = torch.einsum('b h c n, b h e n -> b h c e', k, v)

        # Output: context @ q
        out = torch.einsum('b h c e, b h c n -> b h e n', context, q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

# model
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        condition_channel=None,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.condition_channel= condition_channel if condition_channel is not None else channels
        # input_channels = channels * (2 if self_condition else 1)
        # input_channels = channels + self.condition_channel if self_condition else channels
        input_channels=channels*2
        init_dim = default(init_dim, dim)
        # self.init_att=PreNorm_Cross(input_channels, ConditionalLinearAttention(input_channels,input_channels))
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                # Residual_Cross(PreNorm_Cross(dim_in, LinearAttention_Cross(dim_in))),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            # if ind==0:
            #     self.downs.append(nn.ModuleList([
            #         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            #         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            #         Residual_Cross(PreNorm_Cross(dim_in, LinearAttention_Cross(dim_in,16))),
            #         # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            #         Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            #     ]))
            # else:
            #     self.downs.append(nn.ModuleList([
            #         block_klass(dim_in, dim_in, time_emb_dim = time_dim),
            #         block_klass(dim_in, dim_in, time_emb_dim = time_dim),
            #         # Residual_Cross(PreNorm_Cross(dim_in, LinearAttention_Cross(dim_in))),
            #         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            #         Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            #     ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            # if is_last:
            #
            #     self.ups.append(nn.ModuleList([
            #         block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
            #         block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
            #         Residual_Cross(PreNorm_Cross(dim_in, LinearAttention_Cross(dim_in,16))),
            #         Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            #     ]))
            # else:
            #     self.ups.append(nn.ModuleList([
            #         block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
            #         block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
            #         Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            #         Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            #     ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time,c, x_self_cond = None):
        x = torch.cat((c, x), dim = 1)
        # x = self.init_att(x,c)
        # print(x.shape)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for idn,(block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            # if idn==0:
            #     x = attn(x,c)
            # else:
            #     x = attn(x)
            # x = attn(x,c.pop(0))
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for idn,(block1, block2, attn, upsample) in enumerate(self.ups):
            is_last = idn == (len(self.ups) - 1)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            # if is_last:
            #     x=attn(x,c)
            # else:
            #     x = attn(x)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        # print(x.max(),x.min())
        return x

# gaussian diffusion trainer class


# # 编码器
# class Encoder(nn.Module):
#     def __init__(self, in_channels=1, latent_dim=16,dim=4):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(dim*2),
#             nn.ReLU(True),
#             nn.Conv2d(dim*2, latent_dim, kernel_size=4, stride=2, padding=1),
#         )
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# # 图像重建解码器
# class Decoder(nn.Module):
#     def __init__(self, latent_dim=16, out_channels=1,dim=4):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, dim*2, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim*2, dim, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, out_channels, kernel_size=4, stride=2, padding=1),
#         )
#
#     def forward(self, z):
#         return self.decoder(z)
#
# # 整体自动编码器模型
# class Autoencoder(nn.Module):
#     def __init__(self, in_channels=32,out_channels=16,latent_dim=128,dim=32):
#         super(Autoencoder, self).__init__()
#         self.channels = in_channels
#         self.encoder = Encoder(in_channels, latent_dim,dim)
#         self.decoder = Decoder(latent_dim, out_channels,dim)
#     def forward(self, x, time,c, x_self_cond = None):
#         x = torch.cat((c, x), dim = 1)
#         z = self.encoder(x)
#         y_recon = self.decoder(z)
#         return y_recon




def sinusoidal_embedding(x, dim):
    device = x.device
    half_dim = dim// 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


# 时间层嵌入映射
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, t):
        temb = sinusoidal_embedding(t, self.dim)
        temb = self.linear(temb.to(device=t.device))
        return temb  # shape: [B, dim*4]


# 带时间条件的残差块（基础构建块）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.convs(x)

        # 融合时间信息
        if self.time_mlp is not None and t_emb is not None:
            # t_emb: [B, time_dim]
            t_emb = self.time_mlp(t_emb)  # -> [B, C_out]
            t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # -> [B, C_out, 1, 1]
            h = h + t_emb

        h = h + self.shortcut(x)
        return h


# 扩散模型去噪网络（基于你原有结构升级）
class DenoiserNet(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, out_channels=1, dim=32, latent_dim=64, num_steps=1000):
        super().__init__()
        self.channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels

        # 输入：x_t (noisy image) + condition c
        ch = in_channels + cond_channels

        # 初始卷积
        self.attn1 = LinearAttention_Cross(ch,cond_channels)#Residual_Cross(PreNorm_Cross(ch, LinearAttention_Cross(ch,cond_channels)))
        self.init_conv = nn.Conv2d(ch, dim, 3, padding=1)

        # 时间嵌入
        self.time_embed = TimeEmbedding(latent_dim)

        # 编码器路径（下采样）
        self.enc1 = ResidualBlock(dim, dim, latent_dim * 4)
        self.pool1 = nn.Conv2d(dim, dim, 4, stride=2, padding=1)

        self.enc2 = ResidualBlock(dim, dim * 2, latent_dim * 4)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, 4, stride=2, padding=1)

        self.enc3 = ResidualBlock(dim * 2, latent_dim, latent_dim * 4)

        # 解码器路径（上采样）
        self.dec1 = ResidualBlock(latent_dim, dim * 2, latent_dim * 4)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim * 2, 4, stride=2, padding=1)

        self.dec2 = ResidualBlock(dim * 2, dim, latent_dim * 4)
        self.up2 = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)

        self.dec3 = ResidualBlock(dim, dim, latent_dim * 4)
        self.attn2 =  LinearAttention_Cross(dim,cond_channels)#Residual_Cross(PreNorm_Cross(dim, LinearAttention_Cross(dim,cond_channels)))
        # 输出层
        self.final_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x, t, c):
        """
        x: 当前加噪图像 [B, C_in, H, W]
        t: 时间步 [B] (int tensor)
        c: 条件图像 [B, C_cond, H, W]
        """
        # 拼接输入与条件
        net = torch.cat([x, c], dim=1)  # [B, C_in + C_cond, H, W]
        net=self.attn1(net,c)
        # 初始卷积
        h = self.init_conv(net)

        # 时间嵌入
        t_emb = self.time_embed(t)  # [B, latent_dim*4]

        # 编码器
        h1 = self.enc1(h, t_emb)
        h = self.pool1(h1)

        h2 = self.enc2(h, t_emb)
        h = self.pool2(h2)

        h = self.enc3(h, t_emb)

        # 解码器（带跳跃连接）
        h = self.dec1(h, t_emb)
        h = self.up1(h)
        h = h + h2  # skip connection

        h = self.dec2(h, t_emb)
        h = self.up2(h)
        h = h + h1  # skip connection

        h = self.dec3(h, t_emb)
        h = self.attn2(h, c)
        # 输出预测（例如：预测噪声）
        out = self.final_conv(h)
        return out


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusionImg(nn.Module):
    def __init__(
        self,
        model,
        c_model,
        decoder,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()


        self.model = model
        self.c_model = c_model
        self.decoder = decoder
        self.channels = self.model.channels
        # self.self_condition = self.model.self_condition
        self.self_condition = False

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # print(self.num_timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        c=self.c_model(x_self_cond,train=False)
        model_output = self.model(x, t, c)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            # x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            # x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean# + (0.5 * model_log_variance).exp() * noise
        # print((0.5 * model_log_variance).exp())
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, img,cimg,result_temp=None):
        batch, device = img.shape[0], self.betas.device

        # img = torch.randn(shape, device=device)

        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        #     self_cond = x_start if self.self_condition else None
        #     img, x_start = self.p_sample(img, t, self_cond)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # self_cond = cimg if self.self_condition else None
            img, x_start = self.p_sample(img, t, cimg)

            if result_temp != None:
                saveimg = unnormalize_to_zero_to_one(img)
                utils.save_image(saveimg, str(result_temp/f't-{self.num_timesteps-t}.png'),
                                 nrow=int(math.sqrt(batch)))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample_loop_tmp(self, img,cimg,result_temp=None,num_timesteps=100):
        batch, device = img.shape[0], self.betas.device

        # img = torch.randn(shape, device=device)

        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        #     self_cond = x_start if self.self_condition else None
        #     img, x_start = self.p_sample(img, t, self_cond)

        for t in tqdm(reversed(range(0, num_timesteps-1)), desc='sampling loop time step', total=num_timesteps):
            # self_cond = cimg if self.self_condition else None
            img, x_start = self.p_sample(img, t, cimg,clip_denoised = False)

            if result_temp != None:
                saveimg = unnormalize_to_zero_to_one(img)
                utils.save_image(saveimg, str(result_temp/f't-{num_timesteps-t}.png'),
                                 nrow=int(math.sqrt(batch)))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample_loop(self, img,cimg,result_temp=None):
        batch, device = img.shape[0], self.betas.device

        # img = torch.randn(shape, device=device)

        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        #     self_cond = x_start if self.self_condition else None
        #     img, x_start = self.p_sample(img, t, self_cond)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # self_cond = cimg if self.self_condition else None
            img, x_start = self.p_sample(img, t, cimg)

            if result_temp != None:
                saveimg = unnormalize_to_zero_to_one(img)
                utils.save_image(saveimg, str(result_temp/f't-{self.num_timesteps-t}.png'),
                                 nrow=int(math.sqrt(batch)))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, img, cimg,clip_denoised = True,result_temp=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = img.shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)

        x_start = None

        # x_self_cond=None
        # t = torch.full((batch,), self.sampling_timesteps, device=device).long()
        # img = self.model(img, t, x_self_cond)

        for time, time_next in tqdm(time_pairs, desc = 'ddim_sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # self_cond = x_start if self.self_condition else None
            # self_cond = cimg if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cimg, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if result_temp!=None:
                saveimg = unnormalize_to_zero_to_one(img)
                utils.save_image(saveimg, str(result_temp/f't-{self.num_timesteps-time}.png'),
                                 nrow=int(math.sqrt(batch)))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, img,cimg,result_temp=None):#batch_size
        # print(result_temp)
        # image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # return sample_fn((batch_size, channels, image_size, image_size)) #######
        return sample_fn(img,cimg,result_temp=result_temp)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start,cond_img, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        c=self.c_model(cond_img)
        model_out = self.model(x, t, c)

        # maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity


        if self.objective == 'pred_noise':
            target = noise
            pred_noise = model_out
            x_start_pred = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == 'pred_x0':
            target = x_start
            x_start_pred = model_out

        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
            x_start_pred = self.predict_start_from_v(x, t, model_out)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        x_start_pred=unnormalize_to_zero_to_one(x_start_pred)

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        # print(loss.shape)
        # loss =loss*(t.float()/self.num_timesteps).unsqueeze(1)
        return loss.mean(),x_start_pred,t

    def forward(self, img,cond_img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # t = torch.randint(0, 1, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        # cond_img = normalize_to_neg_one_to_one(cond_img)
        return self.p_losses(img,cond_img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        keys=['fd','qd']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        # print(self.paths)
        self.keys=keys
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x_path = self.paths[index]
        cond_imgs=[]
        for idx in range(1,len(self.keys)):
            cond_path = str(x_path).replace(self.keys[0],self.keys[idx])
            cond_img = Image.open(cond_path)
            cond_imgs.append(self.transform(cond_img))

        cond_img_t=torch.cat(cond_imgs,dim=0)
        # print(cond_img_t.shape)
        x_img = Image.open(x_path)

        return self.transform(x_img),cond_img_t

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        val_folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        num_workers = None,
        keys=["fd","qd"],
        load_path=None,
        result_temp=None,
        train_set=None,
        val_set=None
    ):
        super().__init__()

        if result_temp!=None:
            self.result_temp = Path(result_temp)
            self.result_temp.mkdir(exist_ok=True)
        else:
            self.result_temp=None


        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model


        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.channels = diffusion_model.channels
        # dataset and dataloader
        self.keys=keys
        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to,keys=keys)
        # self.val_ds = Dataset(val_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to,keys=keys)
        self.ds=train_set
        self.val_ds = val_set
        if num_workers is None:num_workers=cpu_count()
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)
        val_dl = DataLoader(self.val_ds, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=4)

        dl = self.accelerator.prepare(dl)
        val_dl = self.accelerator.prepare(val_dl)
        self.dl = cycle(dl)
        # self.val_dl = val_cycle(val_dl)
        self.val_dl=val_dl

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        # print("ssssssssssssss",self.results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if load_path != None:
            self.load(load_path)


    def save(self, milestone,psnr,ssim):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}-{psnr}-{ssim}.pt'))

    def load(self, load_path):
        accelerator = self.accelerator
        device = accelerator.device

        # data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        data = torch.load(load_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'],strict=False)

        self.step = data['step']
        try:
            self.opt.load_state_dict(data['opt'])
        except:
            pass
        # print(self.ema)
        self.ema.load_state_dict(data['ema'],strict=False)

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def test_form_data(self,x_path):
        import numpy as np
        results_folder_target = os.path.join(self.results_folder, "target")
        results_folder_ouput=os.path.join(self.results_folder,"ouput")

        os.makedirs(results_folder_target,exist_ok=True)
        os.makedirs(results_folder_ouput, exist_ok=True)

        file_name=os.path.split(x_path)[-1]
        device = self.accelerator.device
        # print(device)
        self.ema.ema_model.eval()
        cond_imgs=[]
        for idx in range(1,len(self.keys)):
            print(x_path)
            print(str(x_path).replace(self.keys[0],self.keys[idx]))
            cond_path = str(x_path).replace(self.keys[0],self.keys[idx])
            cond_img = np.load(cond_path)
            cond_imgs.append(np.expand_dims(cond_img,0))
        cond_imgs=np.concatenate(cond_imgs, axis=0)
        cond_img_t=torch.from_numpy(cond_imgs).unsqueeze(0).to(device).float()
        x_img=np.expand_dims(np.load(x_path), 0)
        # print(x_img.max())
        x_img = torch.from_numpy(x_img).unsqueeze(0).to(device).float()
        x_img_n=normalize_to_neg_one_to_one(x_img)
        cond_n = normalize_to_neg_one_to_one(cond_img_t)
        # print(x_img_n.shape,cond_n.shape)
        t = torch.full((1,), self.model.num_timesteps - 1, device=device).long()
        sample = self.model.q_sample(x_img_n, t)

        reuslt_img=self.ema.ema_model.sample(sample,cond_n,self.result_temp)
        reuslt_img=reuslt_img.cpu().squeeze().numpy()
        x_img=x_img.cpu().squeeze().numpy()
        # print(os.path.join(results_folder_ouput,file_name.replace("npy","png")),x_img.shape,x_img.max())
        imageio.imsave(os.path.join(results_folder_ouput,file_name.replace("npy","jpg")),np.array(reuslt_img*255,dtype=np.uint8))
        imageio.imsave(os.path.join(results_folder_target, file_name.replace("npy","jpg")), np.array(x_img*255,dtype=np.uint8))
        # print(reuslt_img.shape)
        if self.result_temp!=None:
            frames_t=[]
            extra_paths = os.listdir(self.result_temp)
            extra_paths.sort(key=lambda x: int(x[2:-4]))
            for extra_path in extra_paths:
                frames_t.append(imageio.imread(os.path.join(self.result_temp,extra_path)))
            imageio.mimsave( os.path.join(results_folder_ouput,file_name.replace("npy","gif")), frames_t)
        return x_img,reuslt_img





    def train(self):
        import time

        accelerator = self.accelerator
        device = accelerator.device
        # smaple_imgs = img_to_groups(self.num_samples, self.batch_size, self.channels, self.image_size, device)
        print("sssss",self.results_folder)
        # smaple_imgs,conda_imgs = sample_img_to_groups(self.val_dl,self.num_samples, self.batch_size,device,
        #                                    self.results_folder,self.model.q_sample,self.model.num_timesteps)

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                a=time.time()

                total_loss = 0.
                total_loss_x=0.
                for _ in range(self.gradient_accumulate_every):
                    input, target, input_latent,  target_latent= next(self.dl)#.to(device)
                    data, cond = target_latent.to(device).float(), input.to(device).float()
                    with self.accelerator.autocast():
                        loss,x_start_pred,t = self.model(data,cond)
                        target=target.to(device).float()
                        recon_image=self.model.decoder(x_start_pred)
                        weights=1-(t.float()/self.model.num_timesteps)
                        batch_size=target.shape[0]
                        mse_per_sample = mse_loss(recon_image, target)
                        mse_per_sample = mse_per_sample.view(batch_size, -1).mean(dim=1)
                        loss_mse=(mse_per_sample * weights).mean()
                        # loss_perceptual = perceptual_loss(recon_image, target, vgg_model,weights)
                        # loss_perceptual = mse_loss(data, x_start_pred)
                        # loss_perceptual = loss_perceptual.view(batch_size, -1).mean(dim=1)
                        # loss_perceptual=(loss_perceptual * weights).mean()

                        loss_x = loss_mse# + loss_perceptual
                        # print(loss_x.shape,loss.shape,t.shape)
                        loss_x= loss_x / self.gradient_accumulate_every
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        total_loss_x+=loss_x.item()

                    self.accelerator.backward(loss+loss_x)#

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f} lossx: {total_loss_x:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.ema.ema_model.eval()
                        psnr_list=[]
                        ssim_list=[]

                        with torch.no_grad():
                            for input, target, input_latent,  target_latent in self.val_dl:
                                input   = input.float().to(device)
                                input_latent = input_latent.float().to(device)

                                x_img_n = normalize_to_neg_one_to_one(input_latent)

                                cond_n=input

                                pred_latents =self.ema.ema_model.p_sample_loop_tmp(x_img_n,cond_n,result_temp=None,num_timesteps=100)
                                all_images = self.ema.ema_model.decoder(pred_latents)
                                # 转成 numpy 再算指标
                                gt_np = tensor2np(target)
                                pred_np = tensor2np(all_images)

                                for i in range(gt_np.shape[0]):  # 逐张图计算
                                    psnr = compare_psnr(gt_np[i], pred_np[i], data_range=255)
                                    ssim = compare_ssim(gt_np[i], pred_np[i], data_range=255, channel_axis=-1)
                                    psnr_list.append(psnr)
                                    ssim_list.append(ssim)

                            avg_psnr = np.mean(psnr_list)
                            avg_ssim = np.mean(ssim_list)
                            print(f"[Val] Step {self.step}  PSNR: {avg_psnr:.4f}  SSIM: {avg_ssim:.4f}")



                pbar.update(1)
                b = time.time()
                # print("use:",b-a)

        accelerator.print('training complete')


class ConditionalEncoder(nn.Module):
    def __init__(self, in_channels=1, dim=4):
        super(ConditionalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x,train=True):

        x=self.encoder(x)


        return x
