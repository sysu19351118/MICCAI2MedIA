# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from einops import repeat
from timm.models.layers import DropPath

# 为了代码独立性，这里重新定义了DropPath的__repr__
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =========================================================================================
# 核心依赖：自定义CUDA算子接口、显著性扫描相关的辅助函数
# 这部分代码是 SaliencyMB_SS2D 能够运行的基础，特别是 SelectiveScan 类。
# =========================================================================================

# 尝试导入自定义的CUDA核心库
try:
    import selective_scan_cuda as selective_scan_cuda
except ImportError:
    print("警告: 无法导入 'selective_scan_cuda_core'。SaliencyMB_SS2D 将无法运行。")
    print("请确保已编译并安装了 Vmamba 的自定义 CUDA 算子。")
    selective_scan_cuda = None

# 仅当CUDA库成功导入时才定义相关函数
if selective_scan_cuda:
    
    class SelectiveScan(torch.autograd.Function):
        """
        这个类是PyTorch与自定义CUDA算子`selective_scan`之间的桥梁。
        它定义了前向和反向传播的逻辑。
        """
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}"
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # 确保张量是连续的，以满足CUDA算子的要求
            if u.stride(-1) != 1: u = u.contiguous()
            if delta.stride(-1) != 1: delta = delta.contiguous()
            if D is not None: D = D.contiguous()
            if B.stride(-1) != 1: B = B.contiguous()
            if C.stride(-1) != 1: C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            
            # 调用CUDA算子的反向传播函数
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
            
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    # ----------------- 显著性图处理相关的辅助函数 -----------------

    def extract_non_zero_values(tensor):
        """从张量中提取所有非零值及其位置掩码"""
        B, C, H, W = tensor.shape
        flat_tensor = tensor.view(B, C, -1)
        mask = flat_tensor != 0
        non_zero_values = flat_tensor[mask].view(B, C, -1)
        non_zero_positions = flat_tensor.nonzero(as_tuple=True)[2].view(B, C, -1)
        return non_zero_values, non_zero_positions

    def restore_tensor(original_shape, output_tensor, mask):
        """根据掩码将提取的值恢复到原始形状的零张量中"""
        B, C, H, W = original_shape
        flat_restored_tensor = torch.zeros((B, C, H * W), device=output_tensor.device)
        flat_restored_tensor.scatter_(dim=-1, index=mask, src=output_tensor)
        restored_tensor = flat_restored_tensor.view(B, C, H, W)
        return restored_tensor

    def traverse_saliency_tensor(tensor):
        """
        按照特定的“之”字形或贪心路径遍历显著性图，提取非零元素的索引序列。
        这是实现显著性扫描路径的关键。
        """
        tensor = tensor.squeeze()
        result_index = []
        current_row = 0
        direction = 'left_right'
        while current_row < tensor.size(0):
            non_zero_indices = torch.nonzero(tensor[current_row], as_tuple=False).squeeze()
            if non_zero_indices.ndim == 0:
                non_zero_indices = non_zero_indices.unsqueeze(0)
            
            if direction == 'right_left':
                non_zero_indices = torch.flip(non_zero_indices, dims=[-1])
                
            if len(non_zero_indices) > 0:
                non_zero_index = non_zero_indices + current_row * tensor.size(1) # H*W
                result_index.extend(non_zero_index.tolist())
                
                if current_row < tensor.size(0) - 1:
                    last_index = non_zero_indices[-1].item()
                    next_non_zero_indices = torch.nonzero(tensor[current_row + 1], as_tuple=False).squeeze()
                    if next_non_zero_indices.ndim == 0:
                        next_non_zero_indices = next_non_zero_indices.unsqueeze(0)

                    if len(next_non_zero_indices) > 0:
                        left_index = next_non_zero_indices[0].item()
                        right_index = next_non_zero_indices[-1].item()
                        left_dist = abs(last_index - left_index)
                        right_dist = abs(last_index - right_index)
                        direction = 'left_right' if left_dist <= right_dist else 'right_left'
            current_row += 1
        return torch.tensor(result_index, device=tensor.device, dtype=torch.long)

    def restore_saliency_tensor(original_shape, non_zero_values, non_zero_indexs):
        """根据遍历得到的索引序列，将处理后的值恢复到原始形状的零张量中"""
        B, C, H, W = original_shape
        flat_restored_tensor = torch.zeros((B, C, H * W), device=non_zero_values.device)
        flat_restored_tensor.scatter_(dim=-1, index=non_zero_indexs, src=non_zero_values)
        restored_tensor = flat_restored_tensor.view(B, C, H, W)
        return restored_tensor

    def CrossScan_saliency(x_rgb, gt):
        """
        显著性交叉扫描：根据显著性图(gt)将输入特征(x_rgb)分割为显著和非显著部分，
        并按特定顺序拼接成四个扫描序列。
        """
        B, C, H, W = x_rgb.shape
        xs = x_rgb.new_empty((B, 4, C, H * W))
        index_s_list = []
        mask_ns_list = []
        for b in range(B):
            # 遍历显著图得到显著区域的扫描索引
            non_zero_indexs = traverse_saliency_tensor(gt[b:b+1, :])
            non_zero_indexs_expanded = non_zero_indexs.unsqueeze(0).expand(C, -1).unsqueeze(0)
            
            # 提取显著区域(s)和非显著区域(ns)的特征
            s = torch.gather(x_rgb[b:b+1, :].view(1, C, H*W), 2, non_zero_indexs_expanded)
            ns, mask_ns = extract_non_zero_values(x_rgb[b:b+1, :] * (1-gt[b:b+1, :]))
            
            # 构建四个扫描序列
            scan_1 = torch.cat((s, ns), -1)
            scan_1_flip = torch.cat((s.flip(dims=[-1]), ns.flip(dims=[-1])), -1)
            scan_2 = torch.cat((ns, s), -1)
            scan_2_flip = torch.cat((ns.flip(dims=[-1]), s.flip(dims=[-1])), -1)
            
            xs[b:b+1, 0] = scan_1
            xs[b:b+1, 1] = scan_1_flip
            xs[b:b+1, 2] = scan_2
            xs[b:b+1, 3] = scan_2_flip
            
            index_s_list.append(non_zero_indexs_expanded)
            mask_ns_list.append(mask_ns)
        return xs, index_s_list, mask_ns_list

    def CrossMerge_saliency(ys, index_s, mask_ns, gt):
        """
        显著性交叉合并：将在四个扫描序列上计算得到的结果(ys)合并，
        恢复到原始的二维图像特征形式。
        """
        B, K, C, H, W = ys.shape
        L = H * W
        ys = ys.view(B, K, C, -1)
        out_ys = ys.new_empty(B, C, H, W)
        
        for b in range(B):
            len_s = int(gt.view(B, -1, L)[b:b+1, :].sum(dim=-1).item())
            
            # 恢复每个扫描序列的结果
            y1_s = restore_saliency_tensor([1, C, H, W], ys[b:b+1, 0, :, :len_s], index_s[b])
            y1_ns = restore_tensor([1, C, H, W], ys[b:b+1, 0, :, len_s:], mask_ns[b])
            y1 = y1_s + y1_ns

            y2_s = restore_saliency_tensor([1, C, H, W], ys[b:b+1, 1, :, :len_s].flip(dims=[-1]), index_s[b])
            y2_ns = restore_tensor([1, C, H, W], ys[b:b+1, 1, :, len_s:].flip(dims=[-1]), mask_ns[b])
            y2 = y2_s + y2_ns

            y3_ns = restore_tensor([1, C, H, W], ys[b:b+1, 2, :, :len_s], mask_ns[b])
            y3_s = restore_saliency_tensor([1, C, H, W], ys[b:b+1, 2, :, len_s:], index_s[b])
            y3 = y3_s + y3_ns

            y4_ns = restore_tensor([1, C, H, W], ys[b:b+1, 3, :, :len_s].flip(dims=[-1]), mask_ns[b])
            y4_s = restore_saliency_tensor([1, C, H, W], ys[b:b+1, 3, :, len_s:].flip(dims=[-1]), index_s[b])
            y4 = y4_s + y4_ns
            
            # 将四个结果相加
            out_ys[b] = y1 + y2 + y3 + y4
            
        return out_ys.view(B, C, L)

    def cross_selective_scan_saliency_k2(
        x: torch.Tensor=None, 
        gt: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None, # 未使用
        softmax_version=False,
        nrows=-1,
        delta_softplus=True,
        flip=False # 未使用
    ):
        """
        封装了完整的显著性扫描流程。
        """
        B, D, H, W = x.shape
        D_state, N = A_logs.shape # d_state, d_inner
        K, D_inner, R = dt_projs_weight.shape # K, d_inner, dt_rank
        L = H * W

        if nrows < 1:
            nrows = 1 if D % 4 != 0 else (2 if D % 2 != 0 else 4)

        # 1. 准备扫描序列
        x_fuse, index_s_list, mask_ns_list = CrossScan_saliency(x, gt)
        
        # 2. 投影计算 dt, B, C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        # 3. 准备选择性扫描的输入参数
        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
        
        # 4. 执行选择性扫描
        ys: torch.Tensor = SelectiveScan.apply(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        # 5. 合并扫描结果
        y = CrossMerge_saliency(ys, index_s_list, mask_ns_list, gt)
        
        # 6. 后处理
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm1(y).to(x.dtype)
        
        return y


# =====================================================
# 最终提取的目标类：Saliency Mamba 模块
# =====================================================

class SaliencyMB_SS2D(nn.Module):
    """
    Saliency Mamba Selective Scan 2D (SaliencyMB_SS2D)
    该模块利用显著性先验信息来指导状态空间模型(SSM)的扫描过程。
    它首先根据给定的显著性图(gt)对输入特征进行重新排序，
    形成特殊的扫描序列，然后通过选择性扫描机制进行处理，
    最后将结果恢复并输出。
    """
    def __init__(
        self,
        # 模型基本维度
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # 深度可分离卷积相关参数
        d_conv=3,
        conv_bias=True,
        # 其他参数
        dropout=0.,
        bias=False,
        # dt（时间步）初始化参数
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False, # 未在此模块的特定流程中使用
        flip=False, # 未在此模块的特定流程中使用
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # 深度可分离卷积层
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # SSM 参数初始化 (x_proj, dt_proj, A, D)
        self.K = 4 # 对应四个扫描方向
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)

        # 输出处理
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # 通道注意力机制 (SE-like)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """初始化dt投影层"""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """初始化A参数的对数"""
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """初始化D参数（跳跃连接）"""
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
     
    def forward_corev2(self, x: torch.Tensor, gt: torch.Tensor, nrows=-1):
        """核心计算函数，调用封装好的显著性扫描流程"""
        # 检查依赖是否满足
        if not selective_scan_cuda:
            raise RuntimeError("SaliencyMB_SS2D 无法执行，因为 'selective_scan_cuda_core' 未找到。")

        return cross_selective_scan_saliency_k2(
            x, gt, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm, None, self.softmax_version, 
            nrows=nrows,
        )

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, H, W, C)。
            gt (torch.Tensor): 显著性先验图，形状为 (B, 1, H, W)，值在 [0, 1] 之间。
        Returns:
            torch.Tensor: 输出特征图，形状与输入x相同。
        """
        # 输入投影
        x = self.in_proj(x)
        
        # 深度可分离卷积和激活
        if self.d_conv > 1:
            x_trans = x.permute(0, 3, 1, 2).contiguous()
            x_conv = self.act(self.conv2d(x_trans))
        else:
            x_conv = x.permute(0, 3, 1, 2).contiguous()
        
        # 调用核心的显著性扫描
        y = self.forward_corev2(x_conv, gt)
        
        # 应用通道注意力
        b, d, h, w = x_conv.shape
        x_squeeze = self.avg_pool(x_conv).view(b, d)
        x_exitation = self.fc(x_squeeze).view(b, 1, 1, d) # 恢复空间维度以便广播
        y = y * x_exitation
        
        # 输出投影和dropout
        out = self.dropout(self.out_proj(y))
        
        return out

# ====================== 示例用法 ======================
if __name__ == '__main__':
    # 检查CUDA是否可用以及自定义算子是否已加载
    if torch.cuda.is_available() and selective_scan_cuda:
        print("CUDA 和自定义算子 'selective_scan_cuda' 可用，开始测试。")

        # 定义模型和输入
        device = "cuda"
        B, H, W, C = 2, 64, 64, 96
        
        # SaliencyMB_SS2D 模块
        # d_state可以根据需要调整，例如 d_model 的 1/16 或 1/6
        model = SaliencyMB_SS2D(d_model=C, d_state=16, ssm_ratio=2.0).to(device)

        # 随机生成输入特征和显著性图
        input_feature = torch.randn(B, H, W, C).to(device)
        
        # 生成一个模拟的显著性图 (gt)，比如中间一个方块是显著的
        saliency_map = torch.zeros(B, 1, H, W).to(device)
        saliency_map[:, :, H//4:H*3//4, W//4:W*3//4] = 1
        
        print(f"输入特征形状: {input_feature.shape}")
        print(f"显著性图形状: {saliency_map.shape}")

        # 前向传播
        output = model(input_feature, saliency_map)

        print(f"输出特征形状: {output.shape}")
        
        # 检查输出形状是否正确
        assert output.shape == input_feature.shape, "输出形状与输入形状不匹配！"
        
        print("\nSaliencyMB_SS2D 模块测试成功！")

    else:
        print("无法进行测试，因为 CUDA 或 'selective_scan_cuda' 不可用。")