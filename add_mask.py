"""Image processing utility for adding mask to alpha channel."""
from pathlib import Path
from typing import Union, Tuple
from PIL import Image
from tqdm import tqdm
import argparse

def validate_image_sizes(rgb_image: Image.Image, mask_image: Image.Image) -> None:
    """Validate that RGB and mask images have the same dimensions.
    
    Args:
        rgb_image: PIL Image object of RGB image
        mask_image: PIL Image object of mask image
    
    Raises:
        ValueError: If image dimensions don't match
    """
    if rgb_image.size != mask_image.size:
        raise ValueError(
            f"Image size mismatch: RGB {rgb_image.size} vs Mask {mask_image.size}"
        )

def process_single_image(
    rgb_path: Union[str, Path],
    mask_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """Process a single image by adding mask to alpha channel.
    
    Args:
        rgb_path: Path to input RGB image
        mask_dir: Directory containing mask images
        output_dir: Directory to save processed images
    """
    rgb_path = Path(rgb_path)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get corresponding mask path
    mask_path = mask_dir / rgb_path.name
    
    if not mask_path.exists():
        print(f"Mask not found for {rgb_path.name}, skipping...")
        return

    try:
        # Open images with context managers
        with Image.open(rgb_path) as rgb_img, Image.open(mask_path) as mask_img:
            # Convert RGB image to RGBA mode
            rgba_img = rgb_img.convert("RGBA")
            
            # Validate image dimensions
            validate_image_sizes(rgba_img, mask_img)
            
            # Process mask (convert to single channel and binarize)
            mask = mask_img.convert("L")  # Convert to grayscale
            alpha_channel = mask.point(lambda x: 255 if x > 128 else 0)  # Binarize
            
            # Create new image with alpha channel
            processed_img = Image.merge(
                "RGBA", 
                (*rgba_img.split()[:3], alpha_channel)
            )
            
            # Save processed image
            output_path = output_dir / rgb_path.name
            processed_img.save(output_path, "PNG")
            # print(f"Processed: {output_path}")
            
    except (IOError, ValueError) as e:
        print(f"Error processing {rgb_path.name}: {str(e)}")

def batch_process_images(
    rgb_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    output_dir: Union[str, Path],
    extensions: Tuple[str] = (".png", ".jpg", ".jpeg")
) -> None:
    """Batch process all images in a directory with progress tracking.
    
    Args:
        rgb_dir: Directory containing RGB images
        mask_dir: Directory containing mask images
        output_dir: Directory to save processed images
        extensions: Tuple of valid image extensions
    """
    rgb_dir = Path(rgb_dir)
    
    # 预先生成文件列表用于进度条统计
    image_paths = [
        p for p in rgb_dir.iterdir() 
        if p.suffix.lower() in extensions
    ]
    
    # 初始化进度条
    with tqdm(
        total=len(image_paths),
        desc="Processing Images",
        unit="img",
        dynamic_ncols=True,  # 自动适应终端宽度
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ) as pbar:
        for img_path in image_paths:
            process_single_image(img_path, mask_dir, output_dir)
            pbar.update(1)  # 手动更新进度

if __name__ == "__main__":
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="将RGB图像(png格式)和alpha通道的mask合并到一起",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--rgb_dir",
        type=str,
        help="包含PNG文件的目录路径"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="包含alpha通道的mask的目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="保存处理后的图像的目录路径"
    )
    
    args = parser.parse_args()
    
    # Example usage
    RGB_DIR = Path(args.rgb_dir)
    MASK_DIR = Path(args.mask_dir)
    OUTPUT_DIR = Path(args.output_dir)
    
    batch_process_images(RGB_DIR, MASK_DIR, OUTPUT_DIR)