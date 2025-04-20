import os
import re
import argparse
from typing import List, Tuple

def validate_directory(path: str) -> None:
    """验证目录是否存在且可访问"""
    if not os.path.isdir(path):
        raise ValueError(f"目录不存在: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError(f"目录访问权限不足: {path}")

def parse_filename(filename: str) -> Tuple[str, int]:
    """
    解析符合格式的文件名
    Args:
        filename: 原始文件名（需符合{prefix}_{number}.png格式）
    Returns:
        (前缀字符串, 数字部分)
    Raises:
        ValueError: 当文件名不符合格式时抛出
    """
    pattern = re.compile(r"^(.+?)_(\d+)\.png$", re.IGNORECASE)
    match = pattern.match(filename)
    if not match:
        raise ValueError(f"文件名格式无效: {filename}")
    
    prefix = match.group(1)
    number = int(match.group(2))
    return prefix, number

def generate_new_filename(new_prefix: str, number: int, digits: int) -> str:
    """生成新文件名"""
    return f"{new_prefix}_{number:0{digits}d}.png"

def batch_rename_files(
    dir_path: str,
    new_prefix: str,
    num_digits: int,
    dry_run: bool = False
) -> List[Tuple[str, str]]:
    """
    批量重命名文件主函数
    Args:
        dir_path: 目标目录路径
        new_prefix: 新前缀
        num_digits: 数字部分位数
        dry_run: 试运行模式（不实际执行重命名）
    Returns:
        包含（原文件名，新文件名）的元组列表
    """
    validate_directory(dir_path)
    renamed_files = []
    
    for filename in os.listdir(dir_path):
        if not filename.lower().endswith(".png"):
            continue
        
        try:
            _, number = parse_filename(filename)
        except ValueError as e:
            print(f"跳过文件 {filename}: {str(e)}")
            continue
        
        new_filename = generate_new_filename(new_prefix, number, num_digits)
        old_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, new_filename)
        
        renamed_files.append((filename, new_filename))
        
        if not dry_run:
            try:
                os.rename(old_path, new_path)
            except OSError as e:
                print(f"重命名失败 {filename} -> {new_filename}: {str(e)}")
    
    return renamed_files

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="批量重命名图片文件工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "directory",
        type=str,
        help="包含PNG文件的目录路径"
    )
    parser.add_argument(
        "new_prefix",
        type=str,
        help="新的文件名前缀"
    )
    parser.add_argument(
        "num_digits",
        type=int,
        help="数字部分的位数",
        choices=range(1, 10),
        metavar="[1-9]"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式（不实际执行操作）"
    )
    
    args = parser.parse_args()
    
    try:
        results = batch_rename_files(
            args.directory,
            args.new_prefix,
            args.num_digits,
            args.dry_run
        )
        
        print(f"处理完成，共 {len(results)} 个文件：")
        for old, new in results:
            print(f"{old} -> {new}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()