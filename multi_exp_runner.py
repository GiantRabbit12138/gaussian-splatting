import os
import subprocess
from typing import List, Tuple

# ====== 用户配置区域 ======
# 配置数据集名称（输出文件夹名称会根据不同实验自动生成）
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

# ====== 实验配置 ======
# 定义不同实验的后缀和对应的训练参数
# 格式: (后缀字符串, 训练标志列表)
experiment_configs: List[Tuple[str, List[str]]] = [
    (
        "_mask_rgb_mono_depth_0.5_view",
        ["--eval"]
    ),
    (
        "_mask_rgb_mono_depth_0.5_view_add_noise",
        ["--eval", "--add_noise"]
    ),
    (
        "_mask_rgb_mono_depth_0.5_view_add_noise_qc",
        ["--eval", "--add_noise", "--enable_quality_control"]
    ),
]

# ====== 路径配置 ======
# 基础路径配置（根据实际情况修改）
BASE_DATA_DIR = "./data"         # 数据集根目录
BASE_OUTPUT_DIR = "./output"     # 输出根目录

# ====== 训练参数配置 ======
# 这些参数在实验中保持不变
TRAIN_IMAGE_DIR = "images_add_masks"     # 对应 -i 参数
TRAIN_DEPTH_DIR = "mono_depth_maps"      # 对应 -d 参数

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

def process_dataset(data_name: str, output_name: str, train_flags: List[str]) -> bool:
    """处理单个数据集的全流程（使用指定的训练参数）"""
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

    # 训练阶段（使用传入的训练参数）
    train_cmd = [
        "python", "train.py",
        "-s", source_path,
        "-i", TRAIN_IMAGE_DIR,
        "-d", TRAIN_DEPTH_DIR,
        "-m", model_path
    ] + train_flags

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

def process_experiment(exp_config: Tuple[str, List[str]]) -> List[str]:
    """处理单个实验配置（所有数据集）"""
    exp_postfix, exp_flags = exp_config
    print(f"\n{'='*60}")
    print(f"🚀 开始处理实验: {exp_postfix}")
    print(f"训练参数: {' '.join(exp_flags)}")
    print(f"{'='*60}")

    # 为当前实验生成所有输出文件夹名称
    exp_output_names = [d + exp_postfix for d in dataset_names]

    # 处理当前实验的所有数据集
    for idx, (data_name, out_name) in enumerate(zip(dataset_names, exp_output_names)):
        print(f"\n{'='*40}")
        print(f"处理数据集 {idx+1}/{len(dataset_names)}")
        print(f"数据集名称: {data_name}")
        print(f"输出目录: {os.path.join(BASE_OUTPUT_DIR, out_name)}")
        print(f"{'='*40}")

        success = process_dataset(data_name, out_name, exp_flags)

        if not success:
            print(f"⛔ 数据集 {data_name} 处理流程中断")
            # 是否继续处理后续数据集？如需严格中断，可在此处添加 break
            # break

    return exp_output_names

# ====== 主流程 ======
if __name__ == "__main__":
    print(f"🚀 开始处理 {len(experiment_configs)} 个实验配置")
    all_exp_outputs = []  # 保存所有实验的输出文件夹名称

    # 处理每个实验配置
    for exp_config in experiment_configs:
        exp_outputs = process_experiment(exp_config)
        all_exp_outputs.extend(exp_outputs)

        # 为当前实验的所有输出创建压缩包
        exp_postfix = exp_config[0]
        tar_name = f"all{exp_postfix}.tar.gz"
        print(f"\n{'='*60}")
        print(f"开始打包实验 {exp_postfix} 的输出文件夹")

        tar_cmd = [
            "tar",
            "-czvf",
            tar_name,
        ] + exp_outputs  # 添加当前实验的所有输出文件夹名称

        success = run_command(tar_cmd, "压缩实验输出", cwd=BASE_OUTPUT_DIR)
        if success:
            print(f"✅ 成功创建实验压缩包: {os.path.abspath(os.path.join(BASE_OUTPUT_DIR, tar_name))}")
        else:
            print("❌ 实验压缩包创建失败")
        print(f"{'='*60}")

    # 创建包含所有实验输出的总压缩包
    print(f"\n{'='*60}")
    print("开始打包所有实验输出")

    master_tar_name = "all_experiments.tar.gz"
    tar_cmd = [
        "tar",
        "-czvf",
        master_tar_name,
    ] + [f"all{exp[0]}.tar.gz" for exp in experiment_configs]  # 添加所有实验的压缩包名称

    success = run_command(tar_cmd, "压缩所有实验输出", cwd=BASE_OUTPUT_DIR)
    if success:
        print(f"✅ 成功创建总压缩包: {os.path.abspath(os.path.join(BASE_OUTPUT_DIR, master_tar_name))}")
    else:
        print("❌ 总压缩包创建失败")
    print(f"{'='*60}")

    print("\n所有实验处理完成")