import numpy as np
from scipy.spatial import KDTree
import pycolmap
from pathlib import Path
from quality_control.fish import fish
import cv2

def quaternion_to_rotation_matrix(q):
    """
    将四元数 (w, x, y, z) 转换为旋转矩阵

    参数:
        q: 四元数 (w, x, y, z)

    返回:
        3x3 旋转矩阵
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])

def remove_duplicate_cameras(input_path, output_path, threshold=-1.0):
    """
    移除重复相机及关联的3D点

    参数:
        input_path:  输入sparse文件夹路径
        output_path: 输出sparse文件夹路径
        threshold:   相机位置判重阈值(单位:米, 默认为-1.0, 表示自动计算阈值)
    """
    # 确保输出目录存在
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 加载COLMAP模型
    print(f"加载模型: {input_path}")
    reconstruction = pycolmap.Reconstruction(input_path)
    all_points_before_clean = len(list(reconstruction.points3D.keys()))

    # 步骤1: 提取相机位置并检测重复项
    camera_positions = []
    camera_ids = []
    # image_names = []

    print("计算相机位置...")
    for image_id, image in reconstruction.images.items():
        # cam_from_world = reconstruction.image(image_id).cam_from_world
        # matrix = cam_from_world.matrix()
        # R = matrix[:3, :3]
        # t = matrix[:3, 3]
        camera_center = reconstruction.image(image_id).projection_center()

        # 计算相机中心在世界坐标系中的位置: c = -R^T * t
        # camera_center = -R.T @ t
        camera_positions.append(camera_center)
        camera_ids.append(image_id)
        # image_names.append(image.name)

    # 计算相机位置的标准差
    positions = np.array(camera_positions)
    std_dev = np.std(positions, axis=0)
    mean_std = np.mean(std_dev)
    recommended_threshold = mean_std * 0.1  # 使用标准差的10%
    print(f"推荐阈值: {recommended_threshold:.4f} 米")

    threshold = recommended_threshold if threshold < 0 else threshold

    # 使用KDTree快速查找邻近相机
    print("检测重复相机...")
    kdtree = KDTree(camera_positions)
    duplicate_groups = kdtree.query_ball_point(camera_positions, r=threshold)

    # 步骤2: 识别重复相机组
    duplicate_sets = []
    visited = set()
    for i, group in enumerate(duplicate_groups):
        if i in visited:
            continue

        # 创建新的重复组
        current_group = set(group)
        visited.update(group)

        # 查找所有相连的组
        queue = list(group)
        while queue:
            idx = queue.pop()
            for neighbor in duplicate_groups[idx]:
                if neighbor not in visited:
                    current_group.add(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)

        duplicate_sets.append(current_group)

    # 步骤3: 标记保留和移除的相机
    keep_images = set()
    duplicate_image_ids = set()

    for group in duplicate_sets:
        # 按图像ID排序以确保一致性
        sorted_ids = sorted([camera_ids[i] for i in group])

        # # 保留组内第一个相机
        # keep_images.add(sorted_ids[0])

        # 图像路径构造（假设图像存放在 images/ 目录下）
        image_paths = [
            Path(input_path).parent.parent / "images" / f"rgb_{image_id:05d}.png"
            for image_id in sorted_ids
        ]

        # 评估清晰度
        scores = []
        for image_id, img_path in zip(sorted_ids, image_paths):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"无法加载图像: {img_path}")
                score = fish(img)
                scores.append((score, image_id))
            except Exception as e:
                print(f"图像 {image_id} 清晰度评估失败: {e}")
                scores.append((float('-inf'), image_id))  # 标记为最低优先级

        # 按清晰度排序，保留最高得分
        scores.sort(reverse=True)
        best_image_id = scores[0][1]
        # 保留组内图像清晰度最高的相机
        keep_images.add(best_image_id)
        # 标记重复项
        for image_id in sorted_ids:
            if image_id != best_image_id:
                duplicate_image_ids.add(image_id)

        # # 标记重复项
        # for image_id in sorted_ids[1:]:
        #     duplicate_image_ids.add(image_id)

    # 步骤2: 移除重复相机
    print("移除重复相机...")
    removed_image_names = set()
    for image_id in duplicate_image_ids:
        reconstruction.deregister_image(image_id)
        # 保存移除的图像的名字
        removed_image_names.add(reconstruction.images[image_id].name)

    # 步骤3: 移除与重复相机关联的3D点
    print("清理3D点云...")
    all_points_after_clean = len(list(reconstruction.points3D.keys()))
    points_removed = all_points_before_clean - all_points_after_clean

    # 步骤4: 保存更新后的模型
    print("保存结果...")
    reconstruction.write(output_path)
    reconstruction.export_PLY(Path(output_path)  / "cleaned_model.ply")

    stats = [
        "\n清理结果统计:",
        f"原始相机数: {len(camera_ids)}",
        f"移除相机数: {len(duplicate_image_ids)}",
        f"保留相机数: {len(keep_images)}",
        f"移除3D点数: {points_removed}",
        f"清理完成! 结果已保存至: {output_path}"
    ]

    # 打印统计信息
    print("\n".join(stats))

    # 保存统计信息到txt文件
    stats_file = Path(output_path) / "cleaning_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stats))

    print(f"\n统计信息已保存至: {stats_file}")

    return duplicate_image_ids, removed_image_names

# 使用示例
if __name__ == "__main__":
    # sparse/0文件夹路径
    input_sparse = "../data/white_box/sparse/0"
    output_sparse = "../data/white_box/sparse_cleaned/0"
    duplicate_image_ids, duplicate_image_names = remove_duplicate_cameras(input_sparse,
                                                                          output_sparse)
    print(f"重复相机ID: {duplicate_image_ids}")
    print(f"重复相机names: {duplicate_image_names}")
