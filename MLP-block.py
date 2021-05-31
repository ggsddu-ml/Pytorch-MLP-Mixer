import einops
import numpy as np
import torch.nn as nn
import torch
device = torch.cuda.device(['cuda' if torch.cuda.is_available() else 'cpu' ])
'''
BN，LN，IN，GN从学术化上解释差异：
BatchNorm：batch方向做归一化，算NHW的均值，对小batchsize效果不好；BN主要缺点是对batchsize的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布
LayerNorm：channel方向做归一化，算CHW的均值，主要对RNN作用明显；
InstanceNorm：一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值；这样与batchsize无关，不受其约束。
SwitchableNorm是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

'''


class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, dropout = 0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x
'''

论文的思想是吧per-location(channel-mixing) 以及cross-location(token-mixing)
清晰的分开来
'''
class Mixer_struc(nn.Module):
    def __init__(self, patches: int , token_dim: int, dim: int,channel_dim: int,dropout = 0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(patches,token_dim,self.dropout)
        self.MLP_block_chan = MLPBlock(patches,channel_dim,self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self,x):
        out = self.LayerNorm(x)
        out = einops.rearrange(out, 'b n d -> b d n')
        out = self.MLP_block_token(out)
        out = einops.rearrange(out, 'b d n -> b n d')
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2+=out
        return out2

class MLP_Mixer(nn.Module):
    def __init__(self, image_size, patch_size, token_dim, channel_dim, num_classes, dim, num_blocks):
        super(MLP_Mixer, self).__init__()
        n_patches =(image_size//patch_size) **2
        self.patch_size_embbeder = nn.Conv2d(kernel_size=n_patches, stride=n_patches, in_channels=3, out_channels= dim)
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=n_patches, token_dim=token_dim,channel_dim=channel_dim,dim=dim) for i in range(num_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
    def forward(self,x):
        out = self.patch_size_embbeder(x)
        out = einops.rearrange(out,"n c h w -> n (h w) c")
        for block in self.blocks:
            out = block(out)
        out = self.Layernorm1(out)
        out = out.mean(dim = 1)
        result = self.classifier(out)
        return result

model = MLP_Mixer(image_size= 256,patch_size= 16 , dim = 512 ,num_classes= 100,num_blocks=8,token_dim=256 , channel_dim=2048)

