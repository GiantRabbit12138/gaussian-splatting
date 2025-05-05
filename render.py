#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_opa import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import numpy as np
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, near_prune=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    if near_prune:
        all_distances = []
        for idx, view in enumerate(tqdm(views, desc="Collecting distances", ascii=True, dynamic_ncols=True)):
            # 计算每个点到当前相机中心的距离
            distance = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1)
            all_distances.append(distance.min().item())  # 取当前视角下最近的点作为参考

        # 使用 numpy 来计算分位数
        all_distances = np.array(all_distances)
        percentile_threshold = 100
        near_auto = np.percentile(all_distances, percentile_threshold)
        print(f"Auto near distance: {near_auto}")
        
        mask_near = None
        for idx, view in enumerate(tqdm(views, desc="Prune near points", ascii=True, dynamic_ncols=True)):
            distance = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True)
            mask_temp = distance < near_auto
            mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
        gaussians.prune_points_inference(mask_near)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
        results = render(view, gaussians, pipeline, background, inference=True)
        rendering = results["render"]
        rendering *= view.alpha_mask  # 增加mask
        gt = view.original_image[0:3, :, :]
        depth = results["depth"] * view.alpha_mask
        depth[(depth < 0)] = 0
        # 生成深度分布直方图
        plt.figure(figsize=(10,4))
        plt.hist(results["depth"].detach().cpu().numpy().flatten(), bins=200, log=True)
        plt.title(f"View {idx} Depth Distribution")
        plt.xlabel("Depth Value")
        plt.ylabel("Log Frequency")
        plt.savefig(os.path.join(depth_path, f'{idx:05d}_hist.png'))
        plt.close()
        depth = (depth / (depth.max() + 1e-5)).detach().cpu().numpy().squeeze()
        depth = (depth * 255).astype(np.uint8)

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # 带colorbar的版本
        plt.figure(figsize=(10,8))
        img = plt.imshow(depth, cmap='turbo')
        plt.colorbar(img, label='8 bit Depth [0-255]', shrink=0.8)
        plt.axis('off')  # 可选：隐藏坐标轴
        plt.savefig(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), bbox_inches='tight', dpi=150)
        plt.close()
        # plt.imsave(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth, cmap='jet')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, near_prune: bool):
    with torch.no_grad():
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            gaussians = GaussianModel(dataset.sh_degree)
            # 加载模型参数
            (model_params, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
            gaussians.restore(model_params)
            gaussians.neural_renderer.keep_sigma=True
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, near_prune)

        if not skip_test:
            gaussians = GaussianModel(dataset.sh_degree)
            # 加载模型参数
            (model_params, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
            gaussians.restore(model_params)
            gaussians.neural_renderer.keep_sigma=True
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, near_prune)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--near_prune", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.near_prune)