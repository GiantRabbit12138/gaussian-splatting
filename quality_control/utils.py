from typing import Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F

def depth_patch_loss(
    depth_render: torch.Tensor,
    depth_gt: torch.Tensor,
    quality_mask: torch.Tensor,
    patch_size: int = 32,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    基于质量掩码的深度图分块损失计算

    参数:
        depth_render (Tensor): 渲染深度图 (B, 1, H, W)
        depth_gt (Tensor): 真实深度图 (B, 1, H, W)
        quality_mask (Tensor): 质量掩码 (B, H, W)，1表示低质量区域
        patch_size (int): 分块尺寸，默认32
        eps (float): 数值稳定项，默认1e-8

    返回:
        loss (Tensor): 标量损失值
    """
    # 维度校验
    assert depth_render.dim() == 4 and depth_gt.dim() == 4, "输入应为4D张量 (B,1,H,W)"
    assert quality_mask.shape == depth_render.shape[:2] + depth_render.shape[2:], "掩码形状不匹配"

    # Step 1: 深度图分块
    render_patches = patchify_depth(depth_render, patch_size)  # (B*N,1,p,p)
    gt_patches = patchify_depth(depth_gt, patch_size)

    # Step 2: 生成块级掩码
    block_mask = patchify_mask(quality_mask, patch_size)  # (B*N,)

    # Step 3: 块归一化（支持外部统计量）
    render_norm = normalize_patch(render_patches, eps=eps)
    gt_norm = normalize_patch(gt_patches, eps=eps)

    # Step 4: 计算纯掩码驱动的损失
    return masked_l2(render_norm, gt_norm, block_mask)

def normalize_patch(
    patches: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    归一化

    参数:
        patches (Tensor): 输入块 (N,1,p,p)
        mean (Tensor, optional): 外部提供的均值 (N,1,1,1)
        std (Tensor, optional): 外部提供的标准差 (N,1,1,1)
        eps (float): 数值稳定项

    返回:
        normalized (Tensor): 归一化后的块 (N,1,p,p)
    """
    # 计算或使用外部统计量
    if mean is None:
        mean = patches.mean(dim=(1,2,3), keepdim=True)  # (N,1,1,1)
    if std is None:
        std = patches.std(dim=(1,2,3), keepdim=True) + eps

    return (patches - mean) / std

def masked_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    纯掩码驱动的L2损失计算

    参数:
        pred (Tensor): 预测值 (N,1,p,p)
        target (Tensor): 真实值 (N,1,p,p)
        mask (Tensor): 掩码 (N,)，True表示需要计算损失的区域

    返回:
        loss (Tensor): 标量损失值
    """
    # 计算基础差异
    diff = pred - target
    l2 = (diff ** 2).mean(dim=(1,2,3))  # (N,)

    # 应用掩码筛选
    if torch.any(mask):
        return l2[mask].mean()
    return torch.tensor(0.0, device=pred.device)

# 以下辅助函数保持不变
def patchify_depth(depth: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = depth.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    depth_pad = F.pad(depth, (0, pad_w, 0, pad_h), mode='reflect')
    return depth_pad.unfold(2, patch_size, patch_size)\
                   .unfold(3, patch_size, patch_size)\
                   .contiguous()\
                   .view(b, c, -1, patch_size, patch_size)\
                   .permute(0, 2, 1, 3, 4)\
                   .reshape(-1, c, patch_size, patch_size)

def patchify_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    # mask_exp = mask.unsqueeze(1)
    mask_patches = patchify_depth(mask.float(), patch_size)
    return mask_patches.sum(dim=(1,2,3)) > 0
