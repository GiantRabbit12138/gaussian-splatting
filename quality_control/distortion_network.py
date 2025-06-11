import io
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def generate_random_kernel(kernel_size: int = 5, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """生成归一化随机模糊核。"""
    kernel = torch.rand(1, 1, kernel_size, kernel_size, device=device)
    kernel /= kernel.sum()
    return kernel


def add_gaussian_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    """向图像添加高斯噪声。"""
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0.0, 1.0)


def adjust_contrast_brightness(img: torch.Tensor, contrast: float, brightness: float) -> torch.Tensor:
    """调整对比度和亮度。"""
    return torch.clamp(img * contrast + brightness, 0.0, 1.0)


def adjust_saturation(img: torch.Tensor, saturation: float) -> torch.Tensor:
    """调整饱和度。"""
    gray = img.mean(dim=1, keepdim=True)
    return torch.clamp(img * saturation + gray * (1 - saturation), 0.0, 1.0)


def jpeg_compress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """模拟 JPEG 压缩伪影。"""
    device = img.device
    img_cpu = (img * 255).byte().cpu()
    compressed = []

    for i in img_cpu:
        pil = TF.to_pil_image(i)
        buffer = io.BytesIO()
        pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        jpeg = Image.open(buffer)
        compressed.append(TF.to_tensor(jpeg))

    return torch.stack(compressed).to(device=device)


def ramp_fn(step: int, max_step: int) -> float:
    """扰动强度线性递增。"""
    return min(step / max_step, 1.0)

def interpolate(min_val, max_val, ramp_weight):
    return min_val * (1.0 - ramp_weight) + max_val * ramp_weight


class DistortionArgs:
    """
    图像扰动参数配置，提供默认值。

    - kernel_size: 随机模糊核的大小，越大越模糊。决定随机模糊核的大小，越大模糊越明显。奇数
    - noise_range: 高斯噪声的标准差范围。控制噪声强度，标准差越大，图像越“脏”。
        - (0.0, 0.02)：轻微噪声；
        - (0.0, 0.05)：中等噪声；
        - (0.0, 0.1)：强噪声。
    - contrast_range: 对比度调整的范围。图像对比度调整范围（乘法因子）。图像变得更暗/更亮，对比更低/更高。
        - (0.9, 1.1)：轻微变化；
        - (0.8, 1.2)：中等变化；
        - (0.6, 1.4)：强变化。
    - brightness_range: 图像亮度偏移范围（加法因子）。使图像整体更亮或更暗。
        - (-0.05, 0.05)：轻微；
        - (-0.1, 0.1)：中等；
        - (-0.2, 0.2)：较大。
    - saturation_range: 图像饱和度调整系数。控制色彩丰富程度（颜色与灰度之间混合）。
        - saturation = 0.0 → 全灰；
        - saturation = 1.0 → 原图；
        - saturation > 1.0 → 更鲜艳。
        - (0.7, 1.3)：比较自然；
        - (0.5, 1.5)：更强变化。
    - jpeg_quality_range: JPEG图像压缩质量范围。模拟压缩伪影（block effect）。通常质量低于 70 就可能有明显失真。
        - (30, 95)：常用；
        - (50, 90)：更中等失真；
        - (10, 80)：用于强压缩退化。
    - no_jpeg: 是否禁用 JPEG 压缩。\n
    '''
    | 参数名                  | 类型               | 控制内容         | 推荐范围             | 备注          |
    | -------------------- | ---------------- | ------------ | ---------------- | ----------- |
    | `max_step`           | `int`            | ramp变化的最大步数  | `10000`          | 与训练迭代上限一致   |
    | `kernel_size`        | `int`            | 模糊卷积核大小      | `3, 5, 7`        | 必须为奇数       |
    | `noise_range`        | `(float, float)` | 高斯噪声标准差范围    | `(0.0, 0.05)`    | 像素值级别       |
    | `contrast_range`     | `(float, float)` | 对比度乘因子范围     | `(0.8, 1.2)`     | 小于1暗淡，大于1增强 |
    | `brightness_range`   | `(float, float)` | 亮度加偏移值范围     | `(-0.1, 0.1)`    | 会偏移整个图像亮度   |
    | `saturation_range`   | `(float, float)` | 饱和度调整因子      | `(0.7, 1.3)`     | 接近1为自然色     |
    | `jpeg_quality_range` | `(int, int)`     | JPEG压缩质量范围   | `(30, 95)`       | 越低越有压缩伪影    |
    | `no_jpeg`            | `bool`           | 是否关闭 JPEG 压缩 | `False` 或 `True` | 用于禁用该扰动项    |
    '''
    """
    def __init__(self):
        self.max_step = 30000
        self.kernel_size = 5
        self.noise_range = (0.0, 0.1)
        self.contrast_range = (0.6, 1.4)
        self.brightness_range = (-0.2, 0.2)
        self.saturation_range = (0.5, 1.5)
        self.jpeg_quality_range = (30, 95)
        self.no_jpeg = False

class DistortionNet(nn.Module):
    """图像扰动模块，适用于数据增强或退化建模。"""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, img: torch.Tensor, step: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        对输入图像施加一系列扰动操作。

        Args:
            img: 输入图像 tensor，形状为 (B, C, H, W)，值范围 [0, 1]。
            step: 当前训练步数，用于调节扰动强度。

        Returns:
            处理后的图像张量，以及参数字典。
        """
        device = img.device
        ramp = ramp_fn(step, self.args.max_step)
        params = {}

        # 保存原始尺寸
        B, C, H, W = img.shape

        # 模糊
        kernel = generate_random_kernel(self.args.kernel_size, device)
        pad_size = self.args.kernel_size // 2
        # 处理边缘黑边，使用reflect代替zero填充
        img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        img = F.conv2d(img_padded, kernel.expand(img.size(1), -1, -1, -1),
                       padding=self.args.kernel_size // 2, groups=img.size(1))
        img = img[:, :, pad_size:pad_size+H, pad_size:pad_size+W]
        params['blur_kernel'] = self.args.kernel_size

        # 高斯噪声
        noise_min, noise_max = self.args.noise_range
        noise_std = interpolate(noise_min, random.uniform(noise_min, noise_max), ramp)
        img = add_gaussian_noise(img, noise_std)
        params['noise_std'] = noise_std

        # 对比度
        contrast_min, contrast_max = self.args.contrast_range
        contrast = interpolate(1.0, random.uniform(contrast_min, contrast_max), ramp)

        # 亮度
        brightness_min, brightness_max = self.args.brightness_range
        brightness = interpolate(0.0, random.uniform(brightness_min, brightness_max), ramp)

        img = adjust_contrast_brightness(img, contrast, brightness)
        params['contrast'] = contrast
        params['brightness'] = brightness

        # 饱和度
        sat_min, sat_max = self.args.saturation_range
        saturation = interpolate(1.0, random.uniform(sat_min, sat_max), ramp)
        img = adjust_saturation(img, saturation)
        params['saturation'] = saturation

        # JPEG 压缩
        if not getattr(self.args, 'no_jpeg', False):
            q_min, q_max = self.args.jpeg_quality_range
            jpeg_quality = int(interpolate(95, random.uniform(q_min, q_max), ramp))
            img = jpeg_compress(img, jpeg_quality)
            params['jpeg_quality'] = jpeg_quality
        else:
            params['jpeg_quality'] = None

        return img, params
