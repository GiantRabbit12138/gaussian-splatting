import os
import subprocess
from typing import List

# ====== DTU数据集批量处理配置 ======
# DTU数据集根目录
DTU_DATA_DIR = "./data/DTU-PROCESSED"
# 输出根目录
BASE_OUTPUT_DIR = "./output"
# 后缀名，可自定义
postfix_str = "_dtu_3dgs_nodepth_iftc"

# ====== 训练参数配置 ======
TRAIN_IMAGE_DIR = "images"
TRAIN_DEPTH_DIR = "mono_depth_maps"
TRAIN_FLAGS = [
    "--eval",
    # "--remove_duplicate_cameras",
    # "--densify_pointcloud",
    # "--enable_distortion_net",  # 图像失真模块
    # "--enable_quality_control",  # 开启质量控制模块
]

# ====== 执行函数 ======
def run_command(command: List[str], step_name: str, cwd: str = None, log_file_path: str = None) -> bool:
    print(f"\n▶▶ 开始执行 {step_name}:")
    print(" ".join(command))
    if log_file_path:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n===== {step_name} =====\n")
            log_file.write(" ".join(command) + "\n")
            result = subprocess.run(command, cwd=cwd, stdout=log_file, stderr=log_file)
    else:
        result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ {step_name} 执行失败，退出码 {result.returncode}")
        if log_file_path:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"❌ {step_name} 执行失败，退出码 {result.returncode}\n")
        return False
    print(f"✅ {step_name} 执行成功")
    if log_file_path:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"✅ {step_name} 执行成功\n")
    return True

def process_dtu_scan(scan_name: str, output_name: str, use_depth_dir: bool = True) -> bool:
    source_path = os.path.join(DTU_DATA_DIR, scan_name)
    model_path = os.path.join(BASE_OUTPUT_DIR, output_name)
    os.makedirs(model_path, exist_ok=True)
    log_file_path = os.path.join(model_path, "log.txt")

    # # 计算深度图到SFM的缩放和偏移量
    # depth_scale_cmd = [
    #     "python", "./utils/make_depth_scale.py",
    #     "--base_dir", source_path,
    #     "--depths_dir", os.path.join(source_path, TRAIN_DEPTH_DIR),
    # ]
    # if not run_command(depth_scale_cmd, "计算深度图到SFM的缩放和偏移量", log_file_path=log_file_path):
    #     return False

    # 训练阶段
    train_cmd = [
        "python", "train.py",
        "-s", source_path,
        "-i", TRAIN_IMAGE_DIR,
        "-m", model_path,
    ]
    if use_depth_dir:
        train_cmd += ["-d", TRAIN_DEPTH_DIR]
    train_cmd += TRAIN_FLAGS
    if not run_command(train_cmd, "训练", log_file_path=log_file_path):
        return False

    # 渲染阶段
    render_cmd = [
        "python", "render.py",
        "-m", model_path
    ]
    if not run_command(render_cmd, "渲染", log_file_path=log_file_path):
        return False

    # 评估阶段
    metric_cmd = [
        "python", "metrics.py",
        "-m", model_path
    ]
    if not run_command(metric_cmd, "评估", log_file_path=log_file_path):
        return False

    return True

# ====== 主流程 ======
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量处理DTU数据集")
    parser.add_argument('--use_depth_dir', type=str, required=True, choices=['true', 'false'],
                        help='是否传递-d参数给train.py，true为传递，false为不传递（必填）')
    args = parser.parse_args()
    use_depth_dir = args.use_depth_dir.lower() == 'true'

    # 获取所有scanXXX文件夹
    scan_names = [d for d in os.listdir(DTU_DATA_DIR) if os.path.isdir(os.path.join(DTU_DATA_DIR, d)) and d.startswith("scan")]
    output_names = [scan+postfix_str for scan in scan_names]

    print(f"🚀 开始处理 {len(scan_names)} 个DTU数据集")
    for idx, (scan_name, out_name) in enumerate(zip(scan_names, output_names)):
        print(f"\n{'='*40}")
        print(f"处理DTU数据集 {idx+1}/{len(scan_names)}")
        print(f"数据集名称: {scan_name}")
        print(f"输出目录: {os.path.join(BASE_OUTPUT_DIR, out_name)}")
        print(f"{'='*40}")
        success = process_dtu_scan(scan_name, out_name, use_depth_dir=use_depth_dir)
        if not success:
            print(f"⛔ 数据集 {scan_name} 处理流程中断")
            # break  # 如需严格中断可取消注释
    print("\n所有DTU数据集处理完成")

    # ====== 新增的压缩流程 ======
    print(f"\n{'='*40}")
    print("开始打包所有输出文件夹")
    tar_name = f"all_dtu{postfix_str}.tar.gz"
    tar_cmd = [
        "tar",
        "-czvf",
        tar_name,
    ] + output_names
    success = run_command(tar_cmd, "压缩输出文件夹", cwd=BASE_OUTPUT_DIR)
    if success:
        print(f"✅ 成功创建压缩包: {os.path.abspath(os.path.join(BASE_OUTPUT_DIR, tar_name))}")
    else:
        print("❌ 压缩包创建失败")
    print("="*40)
