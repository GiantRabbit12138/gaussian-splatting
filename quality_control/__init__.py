"""
增强版图像质量控制模块

更新内容：
1. 支持单张图像输入 (C, H, W)
2. 添加分块可视化调试功能
3. 增强mask可视化输出
"""

from typing import Callable, Dict, Optional, Tuple, Union
import os
import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

from quality_control.fish import fish

class QualityController:
    def __init__(
        self,
        patch_size: int = 64,
        device: torch.device = torch.device('cpu'),
        debug_dir: Optional[str] = None
    ):
        self.patch_size = patch_size
        self.device = device
        self.debug_dir = debug_dir
        self.iqa_methods: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {}

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def register_iqa(self, name: str, func: Callable[[Tensor, Tensor], Tensor]) -> None:
        self.iqa_methods[name] = func

    def process(
        self,
        gt_rgb: Tensor,
        render_rgb: Tensor,
        method: str,
        threshold: float = 0.1,
        adaptive: bool = False,
        save_debug: bool = False
    ) -> Tensor:
        gt_rgb = gt_rgb.to(self.device)
        render_rgb = render_rgb.to(self.device)

        # 输入维度处理 (C, H, W) -> (1, C, H, W)
        gt_rgb = gt_rgb.unsqueeze(0) if gt_rgb.ndim == 3 else gt_rgb
        render_rgb = render_rgb.unsqueeze(0) if render_rgb.ndim == 3 else render_rgb

        self._validate_inputs(gt_rgb, render_rgb, method)

        # 分块处理
        gt_patches = self._patchify(gt_rgb)
        render_patches = self._patchify(render_rgb)

        # print(f"gt图像分块形状: {gt_patches.shape}")
        # print(f"render图像分块形状: {render_patches.shape}")

        # print(f"gt图像分块数量: {gt_patches.shape[0]}")
        # print(f"render图像分块数量: {render_patches.shape[0]}")

        # # 调试分块可视化
        # if save_debug and self.debug_dir:
        #     self._save_patch_debug(gt_patches, render_patches)

        # 质量评估
        score_func = self.iqa_methods[method]
        scores = score_func(gt_patches, render_patches)
        # print(f"{method}得分: {scores}")

        # 阈值处理
        if adaptive:
            threshold = self._calculate_adaptive_threshold(scores)
        patch_mask = ((scores > 0) & (scores < threshold)).float()

        # 重组掩码
        full_mask = self._unpatchify(patch_mask, gt_rgb.shape)

        # 可视化输出
        if save_debug and self.debug_dir:
            self._visualize_results(gt_rgb, render_rgb, full_mask)

        return full_mask.squeeze(0)  # 返回 (1, H, W) -> (H, W)

    def _patchify(self, img: Tensor) -> Tensor:
        """
        功能：将输入图像分割为不重叠的块
        """

        # 获取输入图像形状
        b, c, h, w = img.shape
        # 计算需要填充的像素数
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        # 执行边缘填充
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

        # 展开图像为块
        patches = img.unfold(2, self.patch_size, self.patch_size)\
                    .unfold(3, self.patch_size, self.patch_size)\
                    .contiguous()\
                    .view(b, c, -1, self.patch_size, self.patch_size)\
                    .permute(0, 2, 1, 3, 4)\
                    .reshape(-1, c, self.patch_size, self.patch_size)
        return patches

    def _unpatchify(self, patch_mask: Tensor, original_shape: Tuple[int]) -> Tensor:
        # 获取原始图像形状
        b, c, h, w = original_shape
        # 计算分块数量
        num_patches_h = (h + self.patch_size - 1) // self.patch_size
        num_patches_w = (w + self.patch_size - 1) // self.patch_size

        # 调整掩码形状(将一维mask变为二维)
        mask = patch_mask.view(b, -1, num_patches_h, num_patches_w)
        # 将掩码使用最近邻插值恢复到原始图像大小
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        return mask

    def _save_patch_debug(self, gt_patches: Tensor, render_patches: Tensor):
        """保存分块调试图像"""
        # 保存前16个块对比
        for i in range(min(16, gt_patches.size(0))):
            patch_dir = os.path.join(self.debug_dir, f"patch_{i:04d}")
            os.makedirs(patch_dir, exist_ok=True)

            save_image(gt_patches[i], os.path.join(patch_dir, "gt_patch.png"))
            save_image(render_patches[i], os.path.join(patch_dir, "render_patch.png"))

    def _visualize_results(self, gt_tensor: Tensor, render_tensor: Tensor, mask_tensor: Tensor):
        """生成可视化对比图"""
        plt.figure(figsize=(18, 6))

        # 反标准化用于显示
        if self.device == torch.device('cpu'):
            gt_vis = gt_tensor.squeeze(0).permute(1, 2, 0).clamp(0,1).numpy()
            render_vis = render_tensor.squeeze(0).permute(1, 2, 0).clamp(0,1).numpy()
        else:
            gt_vis = gt_tensor.squeeze(0).permute(1, 2, 0).clamp(0,1).cpu().numpy()
            render_vis = render_tensor.squeeze(0).permute(1, 2, 0).clamp(0,1).cpu().numpy()

        # 创建带透明度的掩码覆盖层
        if self.device == torch.device('cpu'):
            mask = mask_tensor.numpy().squeeze()
        else:
            mask = mask_tensor.cpu().numpy().squeeze()
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = [1, 0, 0, 0.3]  # 红色半透明覆盖

        # 子图1：真实图像
        plt.subplot(1,3,1)
        plt.imshow(gt_vis)
        plt.title("Ground Truth")
        plt.axis('off')

        # 子图2：渲染图像
        plt.subplot(1,3,2)
        plt.imshow(render_vis)
        plt.title("Rendered Image")
        plt.axis('off')

        # 子图3：质量分析结果
        plt.subplot(1,3,3)
        plt.imshow(render_vis)
        plt.imshow(overlay)
        plt.title("Quality Analysis\n(Red = Low Quality Regions)")
        plt.axis('off')

        plt.savefig(os.path.join(self.debug_dir, "result_comparison.png"), bbox_inches='tight')
        plt.close()

    def _validate_inputs(self, gt_rgb: Tensor, render_rgb: Tensor, method: str):
        if gt_rgb.shape != render_rgb.shape:
            raise ValueError(f"形状不匹配: gt {gt_rgb.shape} vs render {render_rgb.shape}")
        if method not in self.iqa_methods:
            raise KeyError(f"未注册的方法: {method}")

    def _calculate_adaptive_threshold(self, scores: Tensor) -> float:
        # 改进的自适应阈值算法
        q1 = torch.quantile(scores, 0.25)
        q3 = torch.quantile(scores, 0.75)
        iqr = q3 - q1
        return float(q3 + 1.5 * iqr)

# 预定义评估方法（示例）
def register_default_iqa(controller: QualityController):
    # PSNR实现示例
    def psnr(gt, render):
        mse = torch.mean((gt - render) ** 2, dim=[1,2,3])
        return 10 * torch.log10(1.0 / (mse + 1e-8))
    controller.register_iqa('psnr', lambda x,y: psnr(x,y))

    # SSIM占位实现
    def ssim(gt, render):
        return torch.ones(gt.size(0), device=gt.device)  # 实际需实现
    controller.register_iqa('ssim', ssim)

    # 注册 fish 方法
    def fish_wrapper(gt_patches, render_patches):
        """
        包装 fish 函数以适配 Tensor 输入
        输入: (B, C, H, W) 的 Tensor
        输出: (B,) 的 Tensor 分数
        """
        # 将 Tensor 转换为 numpy 数组
        batch_size = gt_patches.size(0)
        scores = []

        # 移动到 CPU 并转换为 numpy
        gt_np = gt_patches.permute(0, 2, 3, 1).cpu().numpy()
        render_np = render_patches.permute(0, 2, 3, 1).cpu().numpy()

        # 遍历每个 patch，调用 fish 函数
        for i in range(batch_size):
            # 注意：fish 函数处理的是 RGB 图像，输入应为 (H, W, 3)
            score = fish(gt_np[i])
            scores.append(score)

        return torch.tensor(scores, device=gt_patches.device)

    controller.register_iqa('fish', fish_wrapper)
