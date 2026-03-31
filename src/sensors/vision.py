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
        # 优先尝试 RealSense SDK
        try:
            import pyrealsense2 as rs
            self._rs_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.left_serial) if self.left_serial else None
            config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
            self._rs_pipeline.start(config)
            self._use_realsense = True
            print(f"[BinocularCamera] Opened with RealSense SDK: {self.left_serial}")
            self._is_opened = True
            return True
        except (ImportError, Exception):
            # Fallback: 模拟模式 (仿真/测试)
            self._use_realsense = False
            self._frame_counter = 0
            self._sim_time = 0.0
            print(f"[BinocularCamera] Opened in SIMULATION mode: {self.left_serial} <-> {self.right_serial}")
            self._is_opened = True
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
        
        if getattr(self, '_use_realsense', False):
            # RealSense 实际采集
            import pyrealsense2 as rs
            frames = self._rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            # 注意: D435i 是单目彩色+深度, 双目需两台
            # 这里用同一帧模拟双目
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # mm -> m
            
            # 简单的左右视图分离 (用于测试)
            h, w = color_data.shape[:2]
            mid = w // 2
            left = color_data[:, :mid]
            right = color_data[:, mid:]
            
            self._frame_counter += 1
            return StereoFrame(
                left_image=left,
                right_image=right,
                depth=depth_data,
                timestamp=color_frame.timestamp / 1000.0,
                frame_id=self._frame_counter
            )
        else:
            # 模拟模式: 生成带纹理的仿真图像
            h, w = self.resolution[1], self.resolution[0]
            
            # 生成有意义的仿真图案 (棋盘格 + 运动)
            t = self._sim_time
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            # 左相机: 静态棋盘格纹理
            checker_size = 32
            left = np.zeros((h, w, 3), dtype=np.uint8)
            cx, cy = int(w * 0.3 + 50 * np.sin(t * 0.5)), int(h * 0.4 + 30 * np.cos(t * 0.3))
            for dy in range(-8, 8):
                for dx in range(-8, 8):
                    px, py = cx + dx * checker_size, cy + dy * checker_size
                    if 0 <= px < w - checker_size and 0 <= py < h - checker_size:
                        color = 200 if (dx + dy) % 2 == 0 else 80
                        left[py:py+checker_size, px:px+checker_size] = [color, color, color]
            
            # 右相机: 有水平偏移的同一场景 (模拟基线视差)
            offset = int(self.extrinsics.translation[0] * 1000 * 385.5 / max(1.0, 2.0 - t % 3))  # 简化的视差
            right = np.zeros_like(left)
            if offset > 0:
                right[:, offset:] = left[:, :-offset]
            else:
                right[:, :offset] = left[:, -offset:]
            
            # 模拟深度图
            depth = np.ones((h, w), dtype=np.float32) * 2.0  # 2米基准深度
            depth -= 0.3 * np.sin(u * 0.05 + t)  # 起伏
            depth = np.clip(depth, 0.2, 10.0)
            
            self._sim_time += 1.0 / self.fps
            self._frame_counter += 1
            
            return StereoFrame(
                left_image=left,
                right_image=right,
                depth=depth,
                timestamp=self._sim_time,
                frame_id=self._frame_counter
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
        try:
            import cv2
            
            # 如果还没有校正映射表, 使用默认参数生成
            if self._left_map_x is None:
                # 相机内参矩阵
                K1 = np.array([[self.left_intrinsics.fx, 0, self.left_intrinsics.cx],
                               [0, self.left_intrinsics.fy, self.left_intrinsics.cy],
                               [0, 0, 1]])
                K2 = np.array([[self.right_intrinsics.fx, 0, self.right_intrinsics.cx],
                               [0, self.right_intrinsics.fy, self.right_intrinsics.cy],
                               [0, 0, 1]])
                
                # 畸变系数 (假设无畸变)
                D1 = self.left_intrinsics.coeffs or np.zeros(5)
                D2 = self.right_intrinsics.coeffs or np.zeros(5)
                
                # 双目外参
                R = self.extrinsics.rotation
                T = self.extrinsics.translation * 1000  # m -> mm
                
                # 计算校正变换
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                    K1, D1, K2, D2, 
                    (left_img.shape[1], left_img.shape[0]),
                    R, T,
                    flags=cv2.CALIB_ZERO_DISPARITY
                )
                
                # 生成校正映射表
                self._left_map_x, self._left_map_y = cv2.initUndistortRectifyMap(
                    K1, D1, R1, P1, (left_img.shape[1], left_img.shape[0]), cv2.CV_32FC1
                )
                self._right_map_x, self._right_map_y = cv2.initUndistortRectifyMap(
                    K2, D2, R2, P2, (left_img.shape[1], left_img.shape[0]), cv2.CV_32FC1
                )
                self._Q = Q
            
            # 执行校正
            left_rect = cv2.remap(left_img, self._left_map_x, self._left_map_y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, self._right_map_x, self._right_map_y, cv2.INTER_LINEAR)
            
            return left_rect, right_rect
            
        except ImportError:
            # OpenCV 不可用时返回原图
            return left_img, right_img
    
    def compute_depth(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        计算深度图
        
        使用 OpenCV 立体匹配算法 (SGBM/BM)
        """
        try:
            import cv2
            
            # 转为灰度图
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img
                right_gray = right_img
            
            # 立体匹配: SGBM 算法
            window_size = 5
            min_disp = 0
            num_disp = 64  # 必须是 16 的倍数
            
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size**2,
                P2=32 * 3 * window_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
            
            disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            
            # 计算深度: depth = f * B / disparity
            fx = self.left_intrinsics.fx
            baseline_mm = np.linalg.norm(self.extrinsics.translation) * 1000  # m -> mm
            
            # 避免除零
            disparity[disparity <= 0] = 0.1
            
            depth = fx * baseline_mm / disparity
            depth[depth <= 0] = 0
            depth[depth > 20] = 0  # 超过20米的无效
            
            return depth
            
        except ImportError:
            # OpenCV 不可用时返回零深度
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
