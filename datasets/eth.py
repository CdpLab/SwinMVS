import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import json

class ETH3DDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train=True):
        super(ETH3DDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.robust_train = robust_train
        self.resize_h = 512
        self.resize_w = 640

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scenes = f.readlines()
            scenes = [line.rstrip() for line in scenes]

        for scene in scenes:
            # 读取相机参数和图像列表
            cam_file = os.path.join(self.datapath, scene, "cameras.json")
            img_dir = os.path.join(self.datapath, scene, "images")
            depth_dir = os.path.join(self.datapath, scene, "depth_maps")
            
            if not os.path.exists(cam_file) or not os.path.exists(img_dir):
                continue
            
            with open(cam_file, 'r') as f:
                cam_data = json.load(f)
            
            # 获取所有图像ID
            img_ids = sorted([int(f.split('.')[0]) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
            
            # 构建视图对
            for ref_idx, ref_id in enumerate(img_ids):
                # 选择相邻的视图作为源视图
                src_indices = []
                for i in range(1, self.nviews):
                    src_idx = (ref_idx + i) % len(img_ids)
                    src_indices.append(src_idx)
                
                src_ids = [img_ids[idx] for idx in src_indices]
                metas.append((scene, ref_id, src_ids))
        
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, scene, img_id):
        cam_file = os.path.join(self.datapath, scene, "cameras.json")
        
        with open(cam_file, 'r') as f:
            cam_data = json.load(f)
        
        # 查找对应ID的相机参数
        cam_info = None
        for cam in cam_data['cameras']:
            if cam['image_id'] == img_id:
                cam_info = cam
                break
        
        if cam_info is None:
            raise ValueError(f"Camera info not found for image ID {img_id}")
        
        # 内参矩阵
        K = np.array(cam_info['K']).reshape(3, 3)
        # 外参矩阵
        R = np.array(cam_info['R']).reshape(3, 3)
        T = np.array(cam_info['T']).reshape(3, 1)
        # 深度范围
        depth_min = cam_info.get('depth_min', 0.1)
        depth_max = cam_info.get('depth_max', 100.0)
        
        # 构建投影矩阵
        RT = np.hstack([R, T])
        P = K @ RT
        
        return {"K": K, "R": R, "T": T, "P": P, "depth_min": depth_min, "depth_max": depth_max}

    def read_img(self, scene, img_id):
        img_dir = os.path.join(self.datapath, scene, "images")
        # 尝试不同的文件扩展名
        for ext in ['.jpg', '.png', '.JPG', '.PNG']:
            img_file = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(img_file):
                break
        
        if not os.path.exists(img_file):
            raise ValueError(f"Image not found for ID {img_id} in {scene}")
        
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 调整图像大小
        img = cv2.resize(img, (self.resize_w, self.resize_h))
        # 归一化
        img = img.astype(np.float32) / 255.0
        return img

    def read_depth(self, scene, img_id):
        depth_dir = os.path.join(self.datapath, scene, "depth_maps")
        depth_file = os.path.join(depth_dir, f"{img_id}.pfm")
        
        if not os.path.exists(depth_file):
            return None
        
        # 读取深度图
        with open(depth_file, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise Exception('Not a PFM file.')
            
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception('Malformed PFM header.')
            
            scale = float(f.readline().decode('utf-8').rstrip())
            if scale < 0:  # little-endian
                endian = '<'
                scale = -scale
            else:
                endian = '>'  # big-endian
            
            data = np.fromfile(f, endian + 'f')
            shape = (height, width, 3) if color else (height, width)
            
            data = np.reshape(data, shape)
            data = np.flipud(data)
        
        # 调整大小
        depth = cv2.resize(data, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        return depth

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scene, ref_id, src_ids = meta
        view_ids = [ref_id] + src_ids

        imgs = []
        proj_matrices = []
        depth = None
        depth_values = None

        for i, vid in enumerate(view_ids):
            # 读取图像
            img = self.read_img(scene, vid)
            imgs.append(img)

            # 读取相机参数
            cam = self.read_cam_file(scene, vid)
            
            # 构建投影矩阵
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :3, :3] = cam["K"]
            proj_mat[0, :3, 3] = 0
            proj_mat[1, :3, :4] = np.matmul(cam["K"], np.hstack([cam["R"], cam["T"]]))
            
            proj_matrices.append(proj_mat)

            # 只对参考视图读取深度图
            if i == 0:
                depth = self.read_depth(scene, ref_id)
                if depth is not None:
                    # 构建深度值范围
                    depth_values = np.linspace(cam["depth_min"], cam["depth_max"], 192, dtype=np.float32)

        # 转换为numpy数组
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # 数据增强
        if self.mode == "train" and self.robust_train:
            # 随机颜色增强
            for i in range(1, self.nviews):
                # 亮度
                brightness = np.random.uniform(0.8, 1.2)
                imgs[i] = np.clip(imgs[i] * brightness, 0, 1)
                # 对比度
                contrast = np.random.uniform(0.8, 1.2)
                imgs[i] = np.clip((imgs[i] - 0.5) * contrast + 0.5, 0, 1)
                # 饱和度
                gray = 0.299 * imgs[i, 0] + 0.587 * imgs[i, 1] + 0.114 * imgs[i, 2]
                saturation = np.random.uniform(0.8, 1.2)
                imgs[i, 0] = np.clip(gray + saturation * (imgs[i, 0] - gray), 0, 1)
                imgs[i, 1] = np.clip(gray + saturation * (imgs[i, 1] - gray), 0, 1)
                imgs[i, 2] = np.clip(gray + saturation * (imgs[i, 2] - gray), 0, 1)

        # 转换为torch张量
        imgs = torch.from_numpy(imgs).float()
        proj_matrices = torch.from_numpy(proj_matrices).float()
        if depth is not None:
            depth = torch.from_numpy(depth).float()
        if depth_values is not None:
            depth_values = torch.from_numpy(depth_values).float()

        return {
            "imgs": imgs,
            "proj_matrices": proj_matrices,
            "depth": depth,
            "depth_values": depth_values,
            "filename": f"{scene}/view_{ref_id}"
        }