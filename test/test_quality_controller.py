import os
from typing import Optional, Callable, Dict
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 从之前代码中导入必要的类
from quality_control import QualityController, register_default_iqa

def load_image(image_path: str) -> torch.Tensor:
    """加载并预处理图像

    Args:
        image_path: 图像文件路径

    Returns:
        tensor: 标准化后的图像张量 (3, H, W)
    """
    # 加载图像并转换为RGB
    img = Image.open(image_path).convert('RGB')

    # 定义预处理流程
    transform = transforms.Compose([
        transforms.ToTensor(),          # 转换为Tensor并归一化到[0,1]
        # transforms.Normalize(          # 使用ImageNet统计量标准化
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    return transform(img)

def inverse_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """反标准化处理用于可视化"""
    inv_transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
    ])
    return inv_transform(tensor)

if __name__ == '__main__':
    # 初始化质量控制模块
    debug_dir = "./quality_analysis_results"
    controller = QualityController(
        patch_size=11,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        debug_dir=debug_dir
    )
    register_default_iqa(controller)

    # 加载真实图像（示例路径，请替换为实际路径）
    gt_path = "output/book_mask_rgb_mono_depth_cxcy/test/ours_30000/gt/00001.png"
    render_path = "output/book_mask_rgb_mono_depth_cxcy/test/ours_30000/renders/00001.png"

    # try:
    # 加载并预处理图像
    gt_tensor = load_image(gt_path)
    render_tensor = load_image(render_path)

    # 检查尺寸一致性
    assert gt_tensor.shape == render_tensor.shape, "图像尺寸不匹配"

    print(f"输入图像尺寸: {gt_tensor.shape}")

    # 执行质量分析
    with torch.no_grad():
        quality_mask = controller.process(
            gt_tensor,
            render_tensor,
            method='psnr',
            threshold=30.0,  # PSNR阈值设置
            save_debug=True
        )
        # 扩展维度以匹配模型输出 (B, C, H, W)
        quality_mask = quality_mask.unsqueeze(0)  # (1,1,H,W)
        print(f"质量掩码尺寸: {quality_mask.shape}")

    # except FileNotFoundError as e:
    #     print(f"文件加载错误: {str(e)}")
    # except AssertionError as e:
    #     print(f"输入验证失败: {str(e)}")
    # except Exception as e:
    #     print(f"发生未知错误: {str(e)}")
