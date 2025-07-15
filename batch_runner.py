import os
import subprocess
from typing import List

# ====== 用户配置区域 ======
# 配置数据集名称和对应的输出文件夹名称（需一一对应）
dataset_names: List[str] = [
    "book",
    "cube_knife",
    "cuboid_colmap",
    "cup_cuboid_colmap",
    "cylinder_cup_colmap",
    "pencil_case_and_tape",
    "power_bank",
    "soap_packaging_box",
    "thermos_cup",
    "transparent_cup",
    "white_box",
]

postfix_str = "_dense_fish_remove_duplicate_cams"
output_names: List[str] = [
    "book"+postfix_str,
    "cube_knife"+postfix_str,
    "cuboid"+postfix_str,
    "cup_cuboid"+postfix_str,
    "cylinder_cup"+postfix_str,
    "pencil_case_and_tape"+postfix_str,
    "power_bank"+postfix_str,
    "soap_packaging_box"+postfix_str,
    "thermos_cup"+postfix_str,
    "transparent_cup"+postfix_str,
    "white_box"+postfix_str,
]

# 检查配置合法性
assert len(dataset_names) == len(output_names), "数据集和输出文件夹数量必须一致"

# ====== 路径配置 ======
# 基础路径配置（根据实际情况修改）
BASE_DATA_DIR = "./data"         # 数据集根目录
BASE_OUTPUT_DIR = "./output"     # 输出根目录

# ====== 训练参数配置 ======
# train.py的固定参数（根据实际情况修改）
TRAIN_IMAGE_DIR = "images_add_masks"     # 对应 -i 参数
TRAIN_DEPTH_DIR = "mono_depth_maps"      # 对应 -d 参数
TRAIN_FLAGS = ["--eval",  # 开启评估
            #    "--enable_distortion_net",  # 图像失真模块
            #    "--enable_quality_control",  # 开启质量控制模块
                "--remove_duplicate_cameras",
                "--densify_pointcloud",
               ]

# 均匀采样时的相机采样比例
TRAIN_CAM_PERCENTAGE = {dataset_names[0] : "89",
                        dataset_names[1] : "52",
                        dataset_names[2] : "72",
                        dataset_names[3] : "55",
                        dataset_names[4] : "88",
                        dataset_names[5] : "62",
                        dataset_names[6] : "54",
                        dataset_names[7] : "46",
                        dataset_names[8] : "63",
                        dataset_names[9] : "70",
                        dataset_names[10] : "58"}

# ====== 执行函数 ======
def run_command(command: List[str], step_name: str, cwd: str = None) -> bool:
    """执行命令行并检查结果"""
    print(f"\n▶▶ 开始执行 {step_name}:")
    print(" ".join(command))

    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ {step_name} 执行失败，退出码 {result.returncode}")
        return False
    print(f"✅ {step_name} 执行成功")
    return True

def process_dataset(data_name: str, output_name: str) -> bool:
    """处理单个数据集的全流程"""
    # 构建路径
    source_path = os.path.join(BASE_DATA_DIR, data_name)
    model_path = os.path.join(BASE_OUTPUT_DIR, output_name)

    # 计算深度图到SFM的缩放和偏移量
    depth_scale_cmd = [
        "python", "./utils/make_depth_scale.py",
        "--base_dir", source_path,
        "--depths_dir", os.path.join(source_path, TRAIN_DEPTH_DIR),
    ]

    if not run_command(depth_scale_cmd, " 计算深度图到SFM的缩放和偏移量"):
        return False

    # 训练阶段
    train_cmd = [
        "python", "train.py",
        "-s", source_path,
        "-i", TRAIN_IMAGE_DIR,
        "-d", TRAIN_DEPTH_DIR,
        "-m", model_path,
        # "--train_cam_percentage", TRAIN_CAM_PERCENTAGE[data_name],
    ] + TRAIN_FLAGS

    if not run_command(train_cmd, "训练"):
        return False

    # 渲染阶段
    render_cmd = [
        "python", "render.py",
        "-m", model_path
    ]
    if not run_command(render_cmd, "渲染"):
        return False

    # 评估阶段
    metric_cmd = [
        "python", "metrics.py",
        "-m", model_path
    ]
    if not run_command(metric_cmd, "评估"):
        return False

    return True

# ====== 主流程 ======
if __name__ == "__main__":
    print(f"🚀 开始处理 {len(dataset_names)} 个数据集")

    for idx, (data_name, out_name) in enumerate(zip(dataset_names, output_names)):
        print(f"\n{'='*40}")
        print(f"处理数据集 {idx+1}/{len(dataset_names)}")
        print(f"数据集名称: {data_name}")
        print(f"输出目录: {os.path.join(BASE_OUTPUT_DIR, out_name)}")
        print(f"{'='*40}")

        success = process_dataset(data_name, out_name)

        if not success:
            print(f"⛔ 数据集 {data_name} 处理流程中断")
            # 是否继续处理后续数据集？如需严格中断，可在此处添加 break
            # break

    print("\n所有数据集处理完成")

    # ====== 新增的压缩流程 ======
    print(f"\n{'='*40}")
    print("开始打包所有输出文件夹")

    # 构造压缩包名称
    tar_name = f"all{postfix_str}.tar.gz"

    # 构造tar命令
    tar_cmd = [
        "tar",
        "-czvf",
        tar_name,
    ] + output_names  # 添加所有要压缩的文件夹名称

    # 执行压缩命令
    success = run_command(tar_cmd, "压缩输出文件夹", cwd=BASE_OUTPUT_DIR)

    if success:
        print(f"✅ 成功创建压缩包: {os.path.abspath(os.path.join(BASE_OUTPUT_DIR, tar_name))}")
    else:
        print("❌ 压缩包创建失败")

    print("="*40)
