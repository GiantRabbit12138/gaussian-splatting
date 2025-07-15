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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from arguments import ModelParams
from view_selection.colmap_clean_cameras import remove_duplicate_cameras
import cv2
import random
from utils.camera_utils import cameraList_from_camInfos
from scene.cameras import Camera
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Open3D is not installed. Please install it to use this feature.")
    OPEN3D_AVAILABLE = False
from torchvision.utils import save_image
from utils.graphics_utils import getWorld2View

class CameraInfo(NamedTuple):
    uid: int  # 唯一标识符
    R: np.array  # 世界坐标系到相机坐标系的旋转矩阵
    T: np.array  # 世界坐标系到相机坐标系的平移向量
    FovY: np.array  # 垂直视场角（弧度）
    FovX: np.array  # 水平视场角（弧度）
    cx: np.array
    cy: np.array
    depth_params: dict # 深度图相关参数
    image_path: str  # RGB文件路径
    image_name: str  # RGB文件名
    depth_path: str  # 深度图文件路径
    width: int  # RGB图像宽度（像素）
    height: int  # RGB图像高度（像素）
    is_test: bool  # 是否为测试视角

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            cx = intr.params[1]
            cy = intr.params[2]
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            cx = intr.params[2]
            cy = intr.params[3]
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # 计算投影矩阵中的偏移量
        cx = (cx - width / 2) / width * 2
        cy = (cy - height / 2) / height * 2

        n_remove = len(extr.name.split('.')[-1]) + 1
        original_name = extr.name[:-n_remove]
        # 临时修改，兼容名称为depth_开头的深度图文件名称
        # depth_filename = original_name if "mono" in depths_folder else original_name.replace("rgb_", "depth_")
        depth_filename = original_name
        depth_params = None
        if depths_params is not None:
            try:
                # depth_params = depths_params[extr.name[:-n_remove]]
                depth_params = depths_params[depth_filename]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{depth_filename}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)  # 使用全零法线作为占位符
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def generate_tsdf_pointcloud_from_cam_infos(cameras: list[Camera],
                                            cam_infos: list[CameraInfo],
                                            depth_scale=1000.0,
                                            depth_trunc=2.0,
                                            num_points: int = 1_000_000,
                                            pcd_path: str = None):
    """
    从相机信息生成TSDF点云
    :param cam_infos: CameraInfo列表
    :param depth_scale: 深度图缩放因子（毫米转米）
    :param depth_trunc: 最大有效深度（米）
    :return: BasicPointCloud对象
    """
    print("通过TSDF生成点云...")
    os.makedirs(pcd_path, exist_ok=True)
    print(f"生成点云路径: {pcd_path}")

    # 初始化TSDF体
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,  # 体素大小
        sdf_trunc=0.4,      # SDF截断值
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    points_list = []
    colors_list = []
    normals_list = []
    train_cams_num = len(cameras)
    samples_per_frame = (num_points + train_cams_num) // train_cams_num
    transforms = {}

    # 处理每个相机
    for idx, cam_info in enumerate(cameras):
        # 跳过没有深度图的相机
        if cam_info.original_invdepthmap is None:
            print(f"Warning: Skipping camera {cam_info.image_name} - depth map not found")
            continue

        # 还原原始内参cx, cy (归一化前的值)
        cx_original = (cam_info.cx * cam_info.image_width / 2) + (cam_info.image_width / 2)
        cy_original = (cam_info.cy * cam_info.image_height / 2) + (cam_info.image_height / 2)

        # 计算焦距fx, fy (通过FOV)
        fx = cam_info.image_width / (2 * np.tan(cam_info.FoVx / 2))
        fy = cam_info.image_height / (2 * np.tan(cam_info.FoVy / 2))

        # 创建Open3D内参对象
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=cam_info.image_width,
            height=cam_info.image_height,
            fx=fx, fy=fy,
            cx=cx_original, cy=cy_original
        )

        try:
            # 读取彩色图像tensor
            color_tensor = cam_info.original_image
            if color_tensor is None:
                print(f"Warning: Skipping camera {cam_info.image_name} - image not found")
                continue
            # color_tensor = color_tensor * cam_info.alpha_mask
            # save_image(color_tensor.detach().cpu(), os.path.join(pcd_path, f"{cam_info.image_name}_color_tensor.png"))

            # 读取深度图
            depth_16bit = cv2.imread(
                os.path.join(cam_infos[idx].depth_path), cv2.IMREAD_ANYDEPTH
            )

            depth_tensor = cam_info.original_invdepthmap
            if depth_tensor is None:
                print(f"Warning: Skipping camera {cam_info.image_name} - depth image not found")
                continue
            depth_tensor = depth_tensor * cam_info.depth_mask
            # save_image(depth_tensor.detach().cpu(), os.path.join(pcd_path, f"{cam_info.image_name}_depth_tensor.png"))

            # 调整张量维度顺序 (C,H,W) -> (H,W,C) 或 (H,W)
            def prepare_for_o3d(tensor, is_color=True):
                """准备张量用于Open3D图像"""
                # 确保张量在CPU上且连续
                tensor = tensor.detach().contiguous().cpu()

                # 转换为NumPy数组
                np_array = tensor.numpy()

                # 调整维度顺序
                if np_array.ndim == 3:
                    # 颜色图像: (C,H,W) -> (H,W,C)
                    if np_array.shape[0] == 3 or np_array.shape[0] == 1:
                        np_array = np_array.transpose(1, 2, 0)
                    # 深度图像: (1,H,W) -> (H,W)
                    elif np_array.shape[2] == 1:
                        np_array = np_array.squeeze(2)

                # 确保数组是C连续的
                if not np_array.flags.c_contiguous:
                    np_array = np.ascontiguousarray(np_array)

                return np_array

            # 准备颜色图像 (确保是uint8)
            color_np = prepare_for_o3d(color_tensor, is_color=True)
            color_np = (color_np * 255).astype(np.uint8)
            o3d_color = o3d.geometry.Image(color_np)

            # 准备深度图像 (确保是uint16)
            depth_np = prepare_for_o3d(depth_tensor, is_color=False)

            # 将深度值从米转换为毫米 (uint16范围0-65535对应0-65.535米)
            # depth_np = depth_np * (2 ** 16)
            # depth_np = depth_np.clip(0, 65535).astype(np.uint16)
            # o3d_depth = o3d.geometry.Image(depth_np.astype(np.uint16))
            o3d_depth = o3d.geometry.Image(depth_16bit)

            # 保存颜色图像
            # color_path = os.path.join(pcd_path, f"{cam_info.image_name}_color.png")
            # o3d.io.write_image(color_path, o3d_color)

            # 保存深度图像
            # depth_path = os.path.join(pcd_path, f"{cam_info.image_name}_depth.png")
            # o3d.io.write_image(depth_path, o3d_depth)

            # 创建RGBD图像
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False
            )

            # 构建世界到相机变换矩阵 [R|t]
            # OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c = np.eye(4)
            w2c[:3, :3] = cam_info.R.transpose()
            w2c[:3, 3] = cam_info.T
            # w2c = np.matmul(np.array(w2c), OPENGL_TO_OPENCV)

            # 计算相机到世界变换 (TSDF需要c2w)
            c2w = np.linalg.inv(w2c)
            # # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            # c2w[0:3, 1:3] *= -1
            # keep_original_world_coordinate = False
            # if not keep_original_world_coordinate:
            #     c2w = c2w[np.array([0, 2, 1, 3]), :]
            #     c2w[2, :] *= -1

            # c2w = np.matmul(np.array(c2w), OPENGL_TO_OPENCV)

            transform_data = {
                "w2c": w2c.tolist(),
                "c2w": c2w.tolist()
            }
            transforms[cam_info.image_name] = transform_data

            # 积分到TSDF体
            volume.integrate(rgbd, intrinsic, c2w)
            print(f"Integrated camera: {cam_info.image_name}")

            # 提取点云
            pcd = volume.extract_point_cloud()

            # randomly select samples_per_frame points from points
            samples_per_frame = min(samples_per_frame, len(pcd.points))
            mask = random.sample(range(len(pcd.points)), samples_per_frame)
            mask = np.asarray(mask)
            color = np.asarray(pcd.colors)[mask]
            point = np.asarray(pcd.points)[mask]
            normal = np.zeros_like(point)  # 法向量置零

            points_list.append(np.asarray(point))
            colors_list.append(np.asarray(color))
            normals_list.append(np.asarray(normal))
            # colors = np.asarray(pcd.colors)
            # points = np.asarray(pcd.points)

        except Exception as e:
            print(f"Error processing camera {cam_info.image_name}: {str(e)}")

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)
    normals = np.vstack(normals_list)

    # ensure final num points is exact
    if points.shape[0] > num_points:
        indices = np.random.choice(points.shape[0], size=num_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        normals = normals[indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if pcd_path is not None:
        if os.path.exists(os.path.join(pcd_path, "fused_tsdf.ply")):
            os.remove(os.path.join(pcd_path, "fused_tsdf.ply"))
        o3d.io.write_point_cloud(os.path.join(pcd_path, "fused_tsdf.ply"), pcd)

    print(f"Generated TSDF point cloud with {len(points)} points")

    with open('transforms.json', 'w', encoding='utf-8') as f:
        json.dump(transforms, f, indent=2)

    return BasicPointCloud(points, colors, normals)


def generate_pointcloud_from_depth(cameras: list[Camera],
                                            cam_infos: list[CameraInfo],
                                            depth_scale=1000.0,
                                            depth_trunc=2.0,
                                            num_points: int = 1_000_000,
                                            pcd_path: str = None):
    """
    从相机信息生成TSDF点云
    :param cam_infos: CameraInfo列表
    :param depth_scale: 深度图缩放因子（毫米转米）
    :param depth_trunc: 最大有效深度（米）
    :return: BasicPointCloud对象
    """
    print("通过真实深度图和相机内参生成点云...")
    os.makedirs(pcd_path, exist_ok=True)
    print(f"生成点云路径: {pcd_path}")

    points_list = []
    colors_list = []
    normals_list = []
    train_cams_num = len(cameras)
    samples_per_frame = (num_points + train_cams_num) // train_cams_num
    transforms = {}

    # 处理每个相机
    for idx, cam_info in enumerate(cameras):
        # 还原原始内参cx, cy (归一化前的值)
        cx_original = (cam_info.cx * cam_info.image_width / 2) + (cam_info.image_width / 2)
        cy_original = (cam_info.cy * cam_info.image_height / 2) + (cam_info.image_height / 2)

        # 计算焦距fx, fy (通过FOV)
        fx = cam_info.image_width / (2 * np.tan(cam_info.FoVx / 2))
        fy = cam_info.image_height / (2 * np.tan(cam_info.FoVy / 2))

        # 创建Open3D内参对象
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=cam_info.image_width,
            height=cam_info.image_height,
            fx=fx, fy=fy,
            cx=cx_original, cy=cy_original
        )

        try:
            # 读取彩色图像tensor
            color_tensor = cam_info.original_image
            if color_tensor is None:
                print(f"Warning: Skipping camera {cam_info.image_name} - image not found")
                continue
            # color_tensor = color_tensor * cam_info.alpha_mask
            # save_image(color_tensor.detach().cpu(), os.path.join(pcd_path, f"{cam_info.image_name}_color_tensor.png"))

            # 读取深度图
            depth_16bit = cv2.imread(
                os.path.join(cam_infos[idx].depth_path), cv2.IMREAD_ANYDEPTH
            )

            # 深度图
            o3d_depth = o3d.geometry.Image(depth_16bit)

            # 保存颜色图像
            # color_path = os.path.join(pcd_path, f"{cam_info.image_name}_color.png")
            # o3d.io.write_image(color_path, o3d_color)

            # 保存深度图像
            # depth_path = os.path.join(pcd_path, f"{cam_info.image_name}_depth.png")
            # o3d.io.write_image(depth_path, o3d_depth)

            # 构建世界到相机变换矩阵 [R|t]
            # OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c = np.eye(4)
            w2c[:3, :3] = cam_info.R.transpose()
            w2c[:3, 3] = cam_info.T
            # w2c = np.matmul(np.array(w2c), OPENGL_TO_OPENCV)

            # 计算相机到世界变换 (TSDF需要c2w)
            c2w = np.linalg.inv(w2c)

            # 计算第idx个相机到第一个相机的变换矩阵
            w2c0 = np.eye(4)
            w2c0[:3, :3] = cameras[0].R.transpose()
            w2c0[:3, 3] = cameras[0].T
            ci2c0 = np.matmul(w2c0, c2w)

            transform_data = {
                "w2c": w2c.tolist(),
                "c2w": c2w.tolist()
            }
            transforms[cam_info.image_name] = transform_data

            # 3. 将深度图转换为点云（深度单位缩放：毫米→米）
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth=o3d_depth,
                intrinsic=intrinsic,
                # extrinsic=ci2c0,
                depth_scale=1000.0,  # 深度图单位为毫米时用1000; 若为米则用1.0
                depth_trunc=3.0,  # 忽略超过3m的点（可选）
                stride=10  # 步长（可下采样）
            )

            pcd.transform(ci2c0)

            if pcd_path is not None:
                o3d.io.write_point_cloud(
                    os.path.join(pcd_path, str(cam_info.image_name).split('.')[0] + "_pc.ply"),
                    pcd
                )

            # point = np.asarray(pcd.points)
            # points_list.append(np.asarray(point))

        except Exception as e:
            print(f"Error processing camera {cam_info.image_name}: {str(e)}")

    # points = np.vstack(points_list)
    # colors = np.zeros_like(points)
    # normals = np.zeros_like(points)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    # if pcd_path is not None:
    #     if os.path.exists(os.path.join(pcd_path, "depth2pc.ply")):
    #         os.remove(os.path.join(pcd_path, "depth2pc.ply"))
    #     o3d.io.write_point_cloud(os.path.join(pcd_path, "depth2pc.ply"), pcd)

    # print(f"Generated point cloud with {len(points)} points")

    with open('transforms.json', 'w', encoding='utf-8') as f:
        json.dump(transforms, f, indent=2)

    return BasicPointCloud()

def readColmapSceneInfoRemoveDuplicateCameras(args : ModelParams, llffhold=8):
    path = args.source_path
    images = args.images
    depths = args.depths
    eval = args.eval
    train_test_exp = args.train_test_exp
    print(f"path: {path}")
    print(f"images: {images}")
    print(f"depths: {depths}")
    print(f"eval: {eval}")
    print(f"train_test_exp: {train_test_exp}")

    # 从colmap生成的文件中读取相机的内参和外参
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # 确定test_cam_names_list
    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    # 读取所有的相机信息(class CameraInfo)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 如果需要去除重复的相机视角
    duplicate_image_ids = None
    duplicate_image_names = None
    if args.remove_duplicate_cameras:
        input_sparse = os.path.join(path, "sparse", "0")
        output_sparse = os.path.join(path, "sparse_cleaned", "0")
        threshold = -1.0  # 使用推荐阈值，或自定义数值
        # 执行清理，返回被移除的相机ID集合
        duplicate_image_ids, duplicate_image_names = remove_duplicate_cameras(
            input_sparse, output_sparse, threshold)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    print(f"训练相机数量: {len(train_cam_infos)}")
    test_cam_infos = [c for c in cam_infos if c.is_test]

    if args.remove_duplicate_cameras:
        train_cam_infos_tmp = [c for c in train_cam_infos if c.image_name not in duplicate_image_names]
        train_cam_infos = train_cam_infos_tmp
        print(f"训练相机数量(去重): {len(train_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    cleaned_model_path = os.path.join(path, "sparse_cleaned/0/cleaned_model.ply")
    dense_ply_path = os.path.join(path, "dense/fused_clean.ply")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        # 若要移除重复相机
        if args.remove_duplicate_cameras:
            pcd = fetchPly(cleaned_model_path)
        else:
            pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 视图去重后的信息测试
    debug_points_num = True
    if debug_points_num and args.remove_duplicate_cameras and not args.densify_pointcloud:
        pcd_debug = fetchPly(ply_path)
        print("点云数量：", len(pcd_debug.points))
        pcd_debug = fetchPly(cleaned_model_path)
        print("点云数量(视图去重后)：", len(pcd_debug.points))

    # ==============使用dense点云==============
    if args.densify_pointcloud:
        try:
            pcd = fetchPly(dense_ply_path)
            print(f"使用{dense_ply_path}, 使用点云数量: {len(pcd.points)}")
        except Exception as e:
            pcd = fetchPly(ply_path)
            print(f"未找到{dense_ply_path}, 使用稀疏点云: {e}")

    # # ==============TSDF==============
    # # 创建相机列表(Camera类)
    # train_cameras_list = cameraList_from_camInfos(cam_infos=train_cam_infos,
    #                                               resolution_scale=1.0,
    #                                               args=args,
    #                                               is_nerf_synthetic=False,
    #                                               is_test_dataset=False)
    # # 是否创建TSDF点云
    # if args.densify_pointcloud:
    #     # 验证深度图路径存在
    #     if args.depths == "":
    #         raise ValueError("使用TSDF时必须指定深度图路径")

    #     # 生成TSDF点云
    #     try:
    #         # pcd = generate_tsdf_pointcloud_from_cam_infos(
    #         #     cameras=train_cameras_list,
    #         #     cam_infos=train_cam_infos,
    #         #     depth_scale=1000.0,  # 深度图单位为毫米
    #         #     depth_trunc=3.0,      # 最大有效深度4米
    #         #     pcd_path=os.path.join(path, "dense", "0")
    #         # )
    #         pcd = generate_pointcloud_from_depth(
    #             cameras=train_cameras_list,
    #             cam_infos=train_cam_infos,
    #             depth_scale=1000.0,  # 深度图单位为毫米
    #             depth_trunc=3.0,      # 最大有效深度4米
    #             pcd_path=os.path.join(path, "dense", "0")
    #         )
    #     except Exception as e:
    #         print(f"TSDF生成点云失败: {e}")
    #         # 回退到原始点云
    #         pcd = fetchPly(ply_path)
    #         print("回退到原始点云")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Colmap remove_duplicate_cameras" : readColmapSceneInfoRemoveDuplicateCameras
}
