import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        
        # 保证卷积后特征图尺寸不变
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)                            # [B,C,1,1]

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B,1,C]
        y = y.transpose(-1, -2).unsqueeze(-1)           # [B,C,1,1]

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
