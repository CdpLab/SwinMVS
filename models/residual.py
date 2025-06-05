import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DepthOptimizationModule(nn.Module):
    def __init__(self, in_channels, num_blocks=3):
        super(DepthOptimizationModule, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块
        residual_blocks = []
        for i in range(num_blocks):
            residual_blocks.append(ResidualBlock(64, 64))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # 输出卷积层
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 初始处理
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 残差处理
        x = self.residual_blocks(x)
        
        # 输出深度图
        depth = self.conv2(x)
        
        return depth