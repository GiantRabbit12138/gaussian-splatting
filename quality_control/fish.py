import numpy as np
import pywt
import cv2
from typing import Tuple

def dwt_cdf97(image: np.ndarray, level: int) -> list:
    """
    使用Cohen-Daubechies-Feauveau 9/7小波进行多级小波分解。

    Args:
        image: 输入图像，二维数组
        level: 分解级别

    Returns:
        list: 小波分解系数，结构为[cA_n, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
    """
    return pywt.wavedec2(image, wavelet='bior4.4', mode='periodization', level=level)

def ssq(subbands: dict) -> float:
    """计算子带锐度评估值，添加空数组检查"""
    alpha = 0.8
    # 检查子带是否存在且非空
    for key in ['LH', 'HL', 'HH']:
        if subbands[key].size == 0:
            return 0.0
    try:
        e_lh = np.log10(1 + np.mean(subbands['LH']**2))
        e_hl = np.log10(1 + np.mean(subbands['HL']**2))
        e_hh = np.log10(1 + np.mean(subbands['HH']**2))
    except ValueError:  # 处理无效输入
        return 0.0
    return alpha * e_hh + (1 - alpha) * (e_lh + e_hl) / 2

def fish(image: np.ndarray) -> float:
    """
    图像级FISH算法实现。

    Args:
        image: 输入图像，支持RGB或灰度

    Returns:
        float: 全局锐度评估值
    """
    # 图像预处理
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float64)

    # 小波分解
    coeffs = dwt_cdf97(image, 3)

    # 各层权重
    alphas = [4, 2, 1]
    scores = []

    # 遍历三个细节层（从低频到高频）
    for i, alpha in enumerate(alphas, start=1):
        cH, cV, cD = coeffs[i]  # 当前层的水平、垂直、对角细节
        subbands = {'LH': cV, 'HL': cH, 'HH': cD}
        scores.append(ssq(subbands))

    return np.dot(scores, alphas)

def fish_bb(image: np.ndarray) -> Tuple[float, np.ndarray]:
    """块级FISH算法，修复索引越界问题"""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float64)
    height, width = image.shape

    blk_size = 4
    levels = 3
    coeffs = dwt_cdf97(image, levels)

    # 计算分块数量，确保最小为1
    rows = max(1, (height // (2 * blk_size)) - 1)
    cols = max(1, (width // (2 * blk_size)) - 1)
    dst_map = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            subband_scores = []
            for level in range(1, levels + 1):
                cH, cV, cD = coeffs[level]
                scale = 2 ** (level - 1)
                step = max(1, blk_size // scale)  # 步长至少为1

                # 计算分块坐标并限制边界
                y_start = max(0, i * step)
                y_end = min(cH.shape[0], (i + 2) * step)
                x_start = max(0, j * step)
                x_end = min(cH.shape[1], (j + 2) * step)

                # 跳过无效区域
                if y_end <= y_start or x_end <= x_start:
                    continue

                # 提取子带并检查空数组
                hl = cH[y_start:y_end, x_start:x_end]
                lh = cV[y_start:y_end, x_start:x_end]
                hh = cD[y_start:y_end, x_start:x_end]
                if hl.size == 0 or lh.size == 0 or hh.size == 0:
                    continue

                subbands = {'LH': lh, 'HL': hl, 'HH': hh}
                score = ssq(subbands)
                subband_scores.append(score)

            # 仅当三层都有得分时计算加权和
            if len(subband_scores) == levels:
                weights = [4, 2, 1]
                dst_map[i, j] = np.dot(subband_scores, weights)
            else:
                dst_map[i, j] = 0  # 无效块得分为0

    # 处理全零情况
    valid_scores = dst_map[dst_map != 0]
    if len(valid_scores) == 0:
        return 0.0, dst_map

    # 取前1%非零得分计算
    top_percent = max(1, len(valid_scores) // 100)
    sorted_scores = np.sort(valid_scores)[::-1]
    top_scores = sorted_scores[:top_percent]

    return np.sqrt(np.mean(top_scores**2)), dst_map

# if __name__ == "__main__":
#     # 示例用法
#     img_1 = cv2.imread('00000-original.png', cv2.IMREAD_COLOR)
#     # img_2 = cv2.imread('00000-cup_cuboid-noise_qc.png', cv2.IMREAD_COLOR)

#     # 图像级评估
#     global_score = fish(img_1)
#     print(f"Global FISH score: {global_score:.4f}")
#     # global_score = fish(img_2)
#     # print(f"Global FISH score: {global_score:.4f}")

#     # # 块级评估
#     # bb_score, bb_map = fish_bb(img)
#     # print(f"Block-based FISH score: {bb_score:.4f}")
#     # cv2.imshow('Sharpness Map', bb_map / bb_map.max())
#     # cv2.waitKey(0)
#     # cv2.imwrite('sharpness_map.png', (bb_map / bb_map.max() * 255).astype(np.uint8))

#     # # 测试模糊图像 vs 清晰图像
#     # # 清晰图像
#     # sharp_img = cv2.imread("00000.png", cv2.IMREAD_COLOR)
#     # sharp_score = fish(sharp_img)  # 假设输出 6.8

#     # # 模糊图像（高斯模糊）
#     # blurred_img = cv2.GaussianBlur(sharp_img, (15,15), 5)
#     # cv2.imwrite("blurred.png", blurred_img)
#     # blurred_score = fish(blurred_img)  # 假设输出 1.2

#     # print(f"清晰图像得分: {sharp_score:.2f}, 模糊图像得分: {blurred_score:.2f}")
#     # # 输出示例: 清晰图像得分: 6.80, 模糊图像得分: 1.20
