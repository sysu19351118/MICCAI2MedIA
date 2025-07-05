import torch
from mamba_ssm import Mamba
import torch.nn as nn
from typing import Dict, Any, List
import torch.nn.functional as F
import pdb

class MLP( nn.Module ):
    """
    用于特征交叉与降维
    """
    def __init__( self,
        in_features: List[int], out_features: List[int],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.mlp = nn.Sequential()
        for i, o in zip( in_features[:-1], out_features[:-1] ):
            self.mlp.add_module( "linear_%d_%d" % ( i, o ), nn.Linear( i, o ) )
            self.mlp.add_module( "layerNorm_%d_%d" % ( i, o ), nn.LayerNorm( o ) )
            self.mlp.add_module( "relu_%d_%d" % ( i, o ), nn.ReLU() )
            self.mlp.add_module( "dropout_%d_%d" % ( i, o ), nn.Dropout( p = dropout ) )
        
        # 最后一层：不使用激活函数与Dropout
        i, o = in_features[-1], out_features[-1]
        self.mlp.add_module( "linear_%d_%d" % ( i, o ), nn.Linear( i, o ) )


    def forward( self, input ):
        """
        input: torch.Tensor, shape [batch_size, input_dim]
        """
        output = self.mlp( input )
        return output 


class MambaBlock(nn.Module):
    def __init__(
        self, 
        indims,
        outdims,
        indims_fusion,
        outdims_fusion,
        dim_model,
        d_state = 128,
        d_conv = 4,
        expand = 2,
    ):
        super().__init__()

        self.mlp = MLP(indims, outdims)
        self.mlp_res = MLP(indims, outdims)
        dim_model = outdims[-1]
        self.forward_linear = nn.Linear(dim_model, dim_model)
        self.forward_ssm = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )

        self.backward_linear = nn.Linear(dim_model, dim_model)
        self.backward_ssm = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.activation =  nn.LeakyReLU(negative_slope=0.01) 
        self.out_mlp = MLP(indims_fusion, outdims_fusion)

    
    def forward(self, x):
        x_norm = F.normalize(x)
        x_up = self.mlp(x_norm)
        
        # 正向聚合
        x_forward= self.forward_linear(x_up)
        x_forward = self.forward_ssm(x_forward)
        # 逆向聚合
        x_backward = self.backward_linear(x_up)
        x_backward_re = torch.flip(x_backward, dims=[1])
        x_backward_re = self.backward_ssm(x_backward_re)
        x_backward = torch.flip(x_backward_re, dims=[1])
        # res部分
        z = self.activation(self.mlp_res(x_norm))
        x_after_ssm_up = torch.mul(x_forward, z) 
        x_after_ssm_down = torch.mul(x_backward, z) 
        x_fusion = x_after_ssm_up + x_after_ssm_down
        x_fusion = self.out_mlp(x_fusion)
        
        output = x_fusion + x

        return output


class MambaFusionTransformer(nn.Module):
    def __init__(self, 
            num_layer,
            indims,
            outdims,
            indims_fusion,
            outdims_fusion,
            dim_model,
            mmindims = [152064, 2048, 1024],
            mmoutdims = [2048, 1024, 512],
            d_state = 128,
            d_conv = 4,
            expand = 2,
        ):
        super().__init__()

        self.layers = nn.ModuleList([
                MambaBlock(
                indims=indims,
                outdims=outdims,
                indims_fusion=indims_fusion,
                outdims_fusion=outdims_fusion,
                dim_model = dim_model,
            ) for _ in range(num_layer)
        ])

        # 模态融合
        self.mlp_mmfusion = MLP(mmindims, mmoutdims)



    def forward(self, x, x_text):
        batch_size = x.shape[0]
        for layer in self.layers:
            x = layer(x)

        x = torch.cat([x_text.unsqueeze(dim=1), x], dim=1)
        x =self.mlp_mmfusion(x.view(batch_size,-1))
        return x

        

        