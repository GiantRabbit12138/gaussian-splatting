import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

from quality_control.distortion_network import DistortionNet, DistortionArgs


def show_image(img_tensor, title=''):
    """显示图像 tensor（C, H, W）"""
    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')


def main():
    # 加载并预处理图像
    image = Image.open('data/cuboid_colmap/images/rgb_00156.png').convert('RGB')
    transform = T.Compose([
        # T.Resize((256, 256)),
        T.ToTensor()
    ])
    img = transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模块和参数
    args = DistortionArgs()
    distortion_net = DistortionNet(args)

    # 应用扰动
    step = [0, 1000, 5000, 10000, 20000, 30000]

    for i in range(len(step)):
        out_img, param_dict = distortion_net(img, step=step[i])
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        show_image(img[0], 'Original')
        plt.subplot(1, 2, 2)
        show_image(out_img[0], f'Distorted at step {step[i]}')
        plt.tight_layout()
        plt.savefig(f'./distorted_{step[i]}.png')
        print(param_dict)


if __name__ == '__main__':
    """
    Python 的相对导入（如 ..quality_control...）只在模块作为包运行时才有效，不能直接运行该 .py 文件。也就是说：
    你是直接运行 test/test_distortion_network.py 这个文件；
    但此时 Python 并不知道这是哪个包中的模块，于是相对导入失败。

    在项目根目录（包含 quality_control/ 和 test/ 的目录）下运行：
    python -m test.test_distortion_network
    """
    main()
