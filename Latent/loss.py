import torch
import torch.nn as nn
import torch.nn.functional as F



# 感知损失
def perceptual_loss(input, target, vgg_model, weights=None):
    """
    计算加权感知损失（支持样本级权重）

    Args:
        input (Tensor): 模型输出图像，shape [B, C, H, W]
        target (Tensor): 真实图像，shape [B, C, H, W]
        vgg_model (nn.Sequential): 预训练的 VGG 特征提取层（如前5层）
        weights (Tensor or None): 每个样本的权重，shape [B]，可为 None 表示等权重

    Returns:
        loss (Tensor): 标量，加权后的感知损失
    """
    if input.size(1) == 1:
        input = input.repeat(1, 3, 1, 1)
    if target.size(1) == 1:
        target = target.repeat(1, 3, 1, 1)

    device = input.device
    if weights is not None:
        weights = weights.to(device)

    total_loss = 0.0
    batch_size = input.shape[0]

    for layer in vgg_model:
        input = layer(input)
        target = layer(target)

        # 计算逐样本的 MSE 损失: [B, ...] -> [B]
        # 使用 reduction='none'，然后在空间和通道维度上取均值
        mse_per_sample = F.mse_loss(input, target, reduction='none')  # [B, C, H, W]
        mse_per_sample = mse_per_sample.view(batch_size, -1).mean(dim=1)  # [B]，每个样本的平均MSE

        if weights is not None:
            # 加权平均: weights 是 [B]
            layer_loss = (mse_per_sample * weights).mean()
        else:
            # 等权重（相当于 reduction='mean'）
            layer_loss = mse_per_sample.mean()

        total_loss += layer_loss

    return total_loss


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()