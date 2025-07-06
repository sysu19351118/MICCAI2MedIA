# efficient_mamba_encoder.py

import math
import torch
import torch.nn as nn
from timm.layers import SqueezeExcite, DropPath

# =================================================================================
# 基础工具模块 (你提供的代码)
# =================================================================================

class LayerNorm2D(nn.Module):
    """2D张量（B C H W）的通道LayerNorm"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized

class LayerNorm1D(nn.Module):
    """1D张量（B C L）的通道LayerNorm"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized

class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x

class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)
        
    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x

class Stem(nn.Module):
    def __init__(self,  in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None))

    def forward(self, x):
        return self.conv(x)

class PatchMerging(nn.Module):
    def __init__(self,  in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )
        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x

class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim
        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0) 
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)
        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L= x.shape
        H = int(math.sqrt(L))
        BCdt = self.dw(self.BCdt_proj(x).view(batch,-1, H, H)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1) 
        A = (dt + self.A.view(1,-1,1)).softmax(-1) 
        AB = (A * B) 
        h = x @ AB.transpose(-2,-1) 
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) 
        h = self.out_proj(h * self.act(z)+ h * self.D)
        y = h @ C
        y = y.view(batch,-1,H,H).contiguous()
        return y, h

class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(dim)
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        self.alpha = nn.Parameter(1e-4 * torch.ones(4,dim), requires_grad=True)
        
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4,-1,1,1)
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)
        x_prev = x
        y, h = self.mixer(self.norm(x.flatten(2))) 
        x = (1-alpha[1]) * x_prev + alpha[1] * y
        x = (1-alpha[2]) * x + alpha[2] * self.dwconv2(x)
        x = (1-alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, h

# =================================================================================
# 完整的编码器实现 (新增)
# =================================================================================

class EfficientMambaEncoder(nn.Module):
    """
    使用 EfficientViMBlock 构建的完整图像编码器。
    这个类将 Stem、多个阶段的 EfficientViMBlock 和 PatchMerging 组合在一起。
    """
    def __init__(self, in_chans=3, depths=[2, 2, 8, 2], dims=[64, 128, 256, 512],
                 mlp_ratios=4.0, ssd_expands=1, state_dims=16, output_dim=512):
        super().__init__()
        self.dims = dims
        
        # 1. Stem: 将输入图像 (B, 3, H, W) 转换成特征图 (B, C, H/16, W/16)
        self.stem = Stem(in_dim=in_chans, dim=dims[0])
        
        # 2. 构建四个阶段 (Stage)
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        num_stages = len(depths)
        
        for i in range(num_stages):
            # 添加每个阶段的 blocks
            stage_blocks = nn.Sequential(*[
                EfficientViMBlock(
                    dim=dims[i], 
                    mlp_ratio=mlp_ratios, 
                    ssd_expand=ssd_expands, 
                    state_dim=state_dims
                ) for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            
            # 在阶段之间添加下采样层 (除了最后一个阶段)
            if i < num_stages - 1:
                downsampler = PatchMerging(in_dim=dims[i], out_dim=dims[i+1])
                self.downsamplers.append(downsampler)

        # 3. 输出头 (Head)
        self.norm = LayerNorm2D(dims[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], output_dim) if output_dim > 0 else nn.Identity()

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stages
        for i, stage in enumerate(self.stages):
            # 在一个 stage 内部，每个 block 的输入是上一个 block 的输出
            for block in stage:
                x, _ = block(x) # EfficientViMBlock 返回 (x, h)，我们只关心主干特征 x
            
            # 下采样
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        
        # Head
        x = self.norm(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
        
        return x