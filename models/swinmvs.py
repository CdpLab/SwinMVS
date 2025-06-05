import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .fpn import FPN
from .unet import UNetWithCBAM
from .residual import DepthOptimizationModule

class FeatureExtraction(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(FeatureExtraction, self).__init__()
        # 使用SwinTransformer作为主干网络
        self.swin_transformer = SwinTransformer(
            img_size=224, patch_size=4, in_chans=in_channels,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]
        )

        # FPN网络，融合多尺度特征
        self.fpn = FPN(
            in_channels_list=[96, 192, 384, 768],
            out_channels=out_channels
        )

        # 最终特征压缩
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 提取多尺度特征
        features = self.swin_transformer.forward_features(x)

        # 转换特征格式以适应FPN
        features = [feat.transpose(1, 2).view(feat.size(0), -1, int(np.sqrt(feat.size(1))), int(np.sqrt(feat.size(1))))
                    for feat in features]

        # FPN融合
        fpn_features = self.fpn(features)

        # 使用最高分辨率的特征
        final_feature = self.final_conv(fpn_features[0])

        return final_feature

class SwinMVS(nn.Module):
    def __init__(self, num_depth=192, interval_scale=1.06):
        super(SwinMVS, self).__init__()
        self.num_depth = num_depth
        self.interval_scale = interval_scale

        # 特征提取网络
        self.feature_extraction = FeatureExtraction(in_channels=3, out_channels=32)

        # 代价体正则化网络
        self.cost_volume_regularization = UNetWithCBAM(n_channels=32 * 3, n_classes=1)

        # 深度图优化模块
        self.depth_optimization = DepthOptimizationModule(in_channels=4)  # 输入: [深度图, 参考图像特征, 相机参数]

    def forward(self, imgs, proj_matrices, depth_values):
        # 1. 特征提取
        features = []
        for img in imgs:
            feature = self.feature_extraction(img)
            features.append(feature)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # 2. 构建代价体
        cost_volume = self.build_cost_volume(ref_feature, src_features, ref_proj, src_projs, depth_values)

        # 3. 代价体正则化
        cost_reg = self.cost_volume_regularization(cost_volume)
        cost_reg = cost_reg.squeeze(1)

        # 4. 估计初始深度图
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        # 5. 深度图优化
        optimized_depth = self.depth_optimization(torch.cat([depth, ref_feature], dim=1))

        return {
            'depth': depth,
            'optimized_depth': optimized_depth,
            'probability': prob_volume
        }

    def build_cost_volume(self, ref_feature, src_features, ref_proj, src_projs, depth_values):
        batch, channels, height, width = ref_feature.shape
        num_depth = depth_values.shape[1]
        num_views = len(src_features) + 1

        # 初始化代价体
        cost_volume = torch.zeros(batch, channels * 3, num_depth, height, width).cuda()

        # 参考特征
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)

        # 对每个源视图
        for src_idx, (src_feature, src_proj) in enumerate(zip(src_features, src_projs)):
            # 计算变换矩阵
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))

            # 对每个深度平面
            warped_volume = torch.zeros_like(ref_volume)
            for d in range(num_depth):
                depth = depth_values[:, d].view(batch, 1, 1, 1)
                # 单应性变换
                warped_feature = homo_warping(src_feature, proj, depth)
                warped_volume[:, :, d] = warped_feature

            # 计算差异
            diff = torch.abs(ref_volume - warped_volume)
            # 连接特征和差异
            concat_volume = torch.cat([ref_volume, warped_volume, diff], dim=1)

            # 添加到代价体
            cost_volume += concat_volume

        # 平均
        cost_volume = cost_volume / num_views

        return cost_volume

# 辅助函数
def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

def homo_warping(src_fea, proj_mat, depth):
    batch, channels, height, width = src_fea.shape
    num_depth = depth.shape[1]

    # 创建网格
    y, x = torch.meshgrid(torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                          torch.arange(0, width, dtype=torch.float32, device=src_fea.device))
    x = x.reshape(1, 1, height, width).expand(batch, 1, -1, -1)
    y = y.reshape(1, 1, height, width).expand(batch, 1, -1, -1)
    grid = torch.cat((x, y), 1)  # [B,2,H,W]

    # 投影
    D = depth.view(batch, 1, 1, 1, num_depth)
    grid = grid.view(batch, 2, height, width, 1).expand(-1, -1, -1, -1, num_depth)
    xyz = torch.cat((grid, torch.ones_like(grid[:, :1])), 1)  # [B,3,H,W,D]

    # 变换
    rot = proj_mat[:, :3, :3].view(batch, 3, 3, 1, 1)
    trans = proj_mat[:, :3, 3:4].view(batch, 3, 1, 1, 1)
    xyz = torch.matmul(rot, xyz) + trans * D

    # 归一化
    xy = xyz[:, :2] / (xyz[:, 2:3] + 1e-10)  # [B,2,H,W,D]

    # 采样
    xy = xy.permute(0, 3, 2, 4, 1).contiguous().view(batch, height * width * num_depth, 1, 2)
    xy[..., 0] = xy[..., 0] / ((width - 1) / 2) - 1
    xy[..., 1] = xy[..., 1] / ((height - 1) / 2) - 1

    src_fea_expand = src_fea.unsqueeze(4).expand(-1, -1, -1, -1, num_depth)
    src_fea_expand = src_fea_expand.permute(0, 3, 2, 4, 1).contiguous().view(batch, height * width * num_depth, channels, 1)

    warped_fea = F.grid_sample(src_fea_expand, xy, mode='bilinear', padding_mode='zeros', align_corners=True)
    warped_fea = warped_fea.view(batch, height, width, num_depth, channels).permute(0, 4, 1, 2, 3).contiguous()

    return warped_fea