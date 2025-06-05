import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        # 创建1x1卷积用于特征映射
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        # 用于处理最低层特征的额外卷积
        self.extra_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 假设x是一个特征列表，从最高层到最低层
        last_inner = self.inner_blocks[0](x[0])
        results = [self.layer_blocks[0](last_inner)]
        
        # 构建自顶向下路径
        for i in range(1, len(x)):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))
        
        # 添加额外的金字塔层
        last_results = results[-1]
        extra_result = self.extra_block(last_results)
        results.append(extra_result)
        
        return results