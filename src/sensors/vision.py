"""
双目视觉感知模块
================

支持 Intel RealSense D435i 双目深度相机
- 双目校正
- 深度估计
- 深度-彩色对齐
- 物体检测特征
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class CameraModel(Enum):
    """相机模型类型"""
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    KANNALA_BRANDT = "kannala_brandt"


@dataclass
class CameraIntrinsics:
    """相机内参"""
    width: int
    height: int
    fx: float  # 焦距 x
    fy: float  # 焦距 y
    cx: float  # 主点 x
    cy: float  # 主点 y
    model: CameraModel = CameraModel.PINHOLE
    coeffs: Optional[np.ndarray] = None  # 畸变系数


@dataclass
class StereoExtrinsics:
    """双目外参 - 右相机相对于左相机的变换"""
    rotation: np.ndarray  # 3x3 旋转矩阵
    translation: np.ndarray  # 3x1 平移向量


@dataclass
class StereoFrame:
    """双目图像帧"""
    left_image: np.ndarray  # H x W x 3, BGR
    right_image: np.ndarray  # H x W x 3, BGR
    depth: Optional[np.ndarray] = None  # H x W, 米
    timestamp: float = 0.0
    frame_id: int = 0


class BinocularCamera:
    """
    双目相机接口
    
    支持 RealSense D435i 和通用双目相机
    """
    
    def __init__(
        self,
        left_serial: Optional[str] = None,
        right_serial: Optional[str] = None,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        self.left_serial = left_serial
        self.right_serial = right_serial
        self.resolution = resolution
        self.fps = fps
        
        # 默认内参 (RealSense D435i)
        self.left_intrinsics = CameraIntrinsics(
            width=640, height=480,
            fx=385.5, fy=385.5,
            cx=319.5, cy=239.5
        )
        self.right_intrinsics = CameraIntrinsics(
            width=640, height=480,
            fx=385.5, fy=385.5,
            cx=319.5, cy=239.5
        )
        
        # 默认外参 (D435i 基线 50mm)
        self.extrinsics = StereoExtrinsics(
            rotation=np.eye(3),
            translation=np.array([-0.05, 0.0, 0.0])  # 50mm 基线
        )
        
        self._is_opened = False
        
    def open(self) -> bool:
        """打开相机"""
        # TODO: 实现 RealSense SDK 接口
        # try:
        #     import pyrealsense2 as rs
        #     self._pipeline = rs.pipeline()
        #     ...
        # except ImportError:
        #     pass
        self._is_opened = True
        print(f"[BinocularCamera] Opened: {self.left_serial} <-> {self.right_serial}")
        return True
    
    def close(self):
        """关闭相机"""
        if self._is_opened:
            self._is_opened = False
            print("[BinocularCamera] Closed")
    
    def capture(self) -> StereoFrame:
        """捕获一帧双目图像"""
        if not self._is_opened:
            raise RuntimeError("Camera not opened")
        
        # TODO: 实现实际采集
        # 这里返回模拟数据用于测试
        h, w = self.resolution[1], self.resolution[0]
        left = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        return StereoFrame(
            left_image=left,
            right_image=right,
            depth=None,
            timestamp=0.0,
            frame_id=0
        )
    
    def set_extrinsics(self, extrinsics: StereoExtrinsics):
        """设置双目外参"""
        self.extrinsics = extrinsics
    
    def get_extrinsics(self) -> StereoExtrinsics:
        """获取双目外参"""
        return self.extrinsics
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DepthProcessor:
    """
    深度图像处理器
    
    功能:
    - 双目匹配
    - 深度滤波
    - 深度-彩色对齐
    - 3D点云生成
    """
    
    def __init__(
        self,
        left_intrinsics: CameraIntrinsics,
        right_intrinsics: CameraIntrinsics,
        extrinsics: StereoExtrinsics
    ):
        self.left_intrinsics = left_intrinsics
        self.right_intrinsics = right_intrinsics
        self.extrinsics = extrinsics
        
        # 校正映射表
        self._left_map_x = None
        self._left_map_y = None
        self._right_map_x = None
        self._right_map_y = None
        
    def rectify(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校正双目图像"""
        # TODO: 实现双目标定校正
        # 使用 cv2.stereoRectify 和 initUndistortRectifyMap
        return left_img, right_img
    
    def compute_depth(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        计算深度图
        
        使用立体匹配算法 (SGBM/BM/DeepLearning)
        """
        # TODO: 实现立体匹配
        h, w = left_img.shape[:2]
        return np.zeros((h, w), dtype=np.float32)
    
    def filter_depth(
        self,
        depth: np.ndarray,
        min_dist: float = 0.1,
        max_dist: float = 10.0
    ) -> np.ndarray:
        """
        深度滤波
        
        - 去除无效值
        - 平滑处理
        - 空洞填补
        """
        filtered = depth.copy()
        filtered[filtered < min_dist] = 0
        filtered[filtered > max_dist] = 0
        return filtered
    
    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        深度图转点云
        
        Returns:
            points: N x 3 (X, Y, Z) 米
            colors: N x 3 (R, G, B) [0-1], 可选
        """
        h, w = depth.shape
        fx = self.left_intrinsics.fx
        fy = self.left_intrinsics.fy
        cx = self.left_intrinsics.cx
        cy = self.left_intrinsics.cy
        
        # 生成像素坐标
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 过滤无效深度
        valid = depth > 0
        
        # 反投影
        x = (u[valid] - cx) * depth[valid] / fx
        y = (v[valid] - cy) * depth[valid] / fy
        z = depth[valid]
        
        points = np.stack([x, y, z], axis=-1)
        
        colors = None
        if rgb is not None:
            colors = rgb[valid].astype(np.float32) / 255.0
            
        return points, colors
    
    def project_to_3d(
        self,
        u: float,
        v: float,
        depth: float
    ) -> np.ndarray:
        """单点反投影到3D"""
        fx = self.left_intrinsics.fx
        fy = self.left_intrinsics.fy
        cx = self.left_intrinsics.cx
        cy = self.left_intrinsics.cy
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])


# AGV五级双目感知规格
AGV_STEREO_GRADES = {
    'S': {'baseline_mm': 50, 'fov': 85, 'range_m': (0.2, 3.0)},
    'M': {'baseline_mm': 50, 'fov': 87, 'range_m': (0.2, 5.0)},
    'L': {'baseline_mm': 75, 'fov': 91, 'range_m': (0.3, 8.0)},
    'XL': {'baseline_mm': 100, 'fov': 95, 'range_m': (0.3, 10.0)},
    'XXL': {'baseline_mm': 120, 'fov': 100, 'range_m': (0.5, 15.0)}
}


def get_stereo_spec(grade: str) -> dict:
    """获取AGV指定等级的立体视觉规格"""
    return AGV_STEREO_GRADES.get(grade, AGV_STEREO_GRADES['M'])
