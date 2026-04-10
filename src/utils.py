"""
SuperModel 工具模块
==================

通用工具函数库
- 数据验证
- 坐标变换
- 信号处理
- 数值计算
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass


# =============================================================================
# 数据验证工具
# =============================================================================

def validate_vector(vec: np.ndarray, expected_dim: int, name: str = "vector") -> np.ndarray:
    """
    验证向量维度
    
    Args:
        vec: 输入向量
        expected_dim: 期望维度
        name: 向量名称 (用于错误信息)
        
    Returns:
        验证后的向量 (确保是numpy数组)
        
    Raises:
        ValueError: 维度不匹配
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape != (expected_dim,):
        raise ValueError(f"{name} 维度错误: 期望 ({expected_dim},), 实际 {vec.shape}")
    return vec


def validate_matrix(mat: np.ndarray, expected_shape: Tuple[int, int], name: str = "matrix") -> np.ndarray:
    """
    验证矩阵维度
    
    Args:
        mat: 输入矩阵
        expected_shape: 期望形状 (rows, cols)
        name: 矩阵名称 (用于错误信息)
        
    Returns:
        验证后的矩阵
        
    Raises:
        ValueError: 维度不匹配
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != expected_shape:
        raise ValueError(f"{name} 形状错误: 期望 {expected_shape}, 实际 {mat.shape}")
    return mat


def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制值在指定范围内"""
    return max(min_val, min(max_val, value))


def clamp_vector(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """限制向量范数"""
    norm = np.linalg.norm(vec)
    if norm > max_norm:
        return vec * (max_norm / norm)
    return vec


# =============================================================================
# 坐标变换工具
# =============================================================================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    欧拉角转旋转矩阵 (ZYX顺序)
    
    Args:
        roll: 翻滚角 (rad)
        pitch: 俯仰角 (rad)
        yaw: 偏航角 (rad)
        
    Returns:
        3x3 旋转矩阵
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ], dtype=np.float32)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转欧拉角 (ZYX顺序)
    
    Args:
        R: 3x3 旋转矩阵
        
    Returns:
        [roll, pitch, yaw] 弧度
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵
    
    Args:
        q: [qw, qx, qy, qz] 四元数
        
    Returns:
        3x3 旋转矩阵
    """
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ], dtype=np.float32)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转四元数
    
    Args:
        R: 3x3 旋转矩阵
        
    Returns:
        [qw, qx, qy, qz] 四元数
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    return q / np.linalg.norm(q)


def pose_to_transform_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    位姿转4x4变换矩阵
    
    Args:
        position: [x, y, z] 位置
        orientation: [qw, qx, qy, qz] 四元数
        
    Returns:
        4x4 变换矩阵
    """
    R = quaternion_to_rotation_matrix(orientation)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def transform_pointcloud(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    变换点云
    
    Args:
        points: Nx3 点云
        transform: 4x4 变换矩阵
        
    Returns:
        变换后的 Nx3 点云
    """
    points_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    points_t = (transform @ points_h.T).T
    return points_t[:, :3]


# =============================================================================
# 信号处理工具
# =============================================================================

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    移动平均滤波
    
    Args:
        data: 输入数据
        window: 窗口大小
        
    Returns:
        滤波后的数据
    """
    if window <= 1:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    指数移动平均
    
    Args:
        data: 输入数据
        alpha: 平滑系数 (0-1), 越大越敏感
        
    Returns:
        滤波后的数据
    """
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def lowpass_filter(data: np.ndarray, cutoff_freq: float, sample_rate: float) -> np.ndarray:
    """
    简单一阶低通滤波器
    
    Args:
        data: 输入数据
        cutoff_freq: 截止频率 (Hz)
        sample_rate: 采样率 (Hz)
        
    Returns:
        滤波后的数据
    """
    dt = 1.0 / sample_rate
    RC = 1.0 / (2 * np.pi * cutoff_freq)
    alpha = dt / (RC + dt)
    
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def bandpass_filter(
    data: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: float
) -> np.ndarray:
    """
    带通滤波 (简化版 - 级联低通和高通)
    
    Args:
        data: 输入数据
        low_cutoff: 低频截止频率 (Hz)
        high_cutoff: 高频截止频率 (Hz)
        sample_rate: 采样率 (Hz)
        
    Returns:
        滤波后的数据
    """
    # 先高通 (去除直流偏置)
    if low_cutoff > 0:
        data = data - lowpass_filter(data, low_cutoff, sample_rate)
    # 再低通 (去除高频噪声)
    if high_cutoff < sample_rate / 2:
        data = lowpass_filter(data, high_cutoff, sample_rate)
    return data


def derivative(data: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    计算数值微分
    
    Args:
        data: 输入数据
        dt: 时间步长
        
    Returns:
        导数近似
    """
    return np.gradient(data, dt)


def integral(data: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    计算数值积分 (梯形法)
    
    Args:
        data: 输入数据
        dt: 时间步长
        
    Returns:
        积分结果
    """
    return np.cumsum(data) * dt


def normalize(data: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    归一化到指定范围
    
    Args:
        data: 输入数据
        min_val: 输出最小值
        max_val: 输出最大值
        
    Returns:
        归一化后的数据
    """
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min < 1e-10:
        return np.full_like(data, (min_val + max_val) / 2)
    normalized = (data - data_min) / (data_max - data_min)
    return normalized * (max_val - min_val) + min_val


# =============================================================================
# 数值计算工具
# =============================================================================

def wrap_angle(angle: float) -> float:
    """将角度_wrap到 [-π, π] 范围"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def wrap_angle_deg(angle: float) -> float:
    """将角度_wrap到 [-180, 180] 范围"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def interpolate_linear(p0: float, p1: float, t: float) -> float:
    """线性插值"""
    return p0 + (p1 - p0) * t


def interpolate_cubic(p0: float, p1: float, v0: float, v1: float, t: float) -> float:
    """
    三次埃尔米特插值
    
    Args:
        p0, p1: 端点位置
        v0, v1: 端点速度
        t: 参数 [0, 1]
    """
    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return h00*p0 + h10*v0 + h01*p1 + h11*v1


def smooth_step(t: float) -> float:
    """平滑步进函数 (S曲线)"""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def trajectory_generator_linear(
    start: np.ndarray,
    end: np.ndarray,
    duration: float,
    dt: float
) -> List[np.ndarray]:
    """
    线性轨迹生成
    
    Args:
        start: 起始位置
        end: 目标位置
        duration: 持续时间 (秒)
        dt: 时间步长
        
    Returns:
        轨迹点列表
    """
    n_steps = int(duration / dt)
    trajectory = []
    for i in range(n_steps + 1):
        t = i / n_steps
        t_smooth = smooth_step(t)
        point = start + (end - start) * t_smooth
        trajectory.append(point)
    return trajectory


# =============================================================================
# 几何计算工具
# =============================================================================

def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    点到线段的距离
    
    Args:
        point: 查询点
        line_start: 线段起点
        line_end: 线段终点
        
    Returns:
        最短距离
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return np.linalg.norm(point - line_start)
    line_unit = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit)
    proj_length = np.clip(proj_length, 0, line_len)
    projection = line_start + line_unit * proj_length
    return np.linalg.norm(point - projection)


def circle_intersection(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float
) -> List[np.ndarray]:
    """
    两圆交点
    
    Args:
        c1, c2: 圆心 (2D)
        r1, r2: 半径
        
    Returns:
        交点列表
    """
    d = np.linalg.norm(c2 - c1)
    if d > r1 + r2 or d < abs(r1 - r2):
        return []  # 无交点
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)
    
    cx = c1[0] + a * (c2[0] - c1[0]) / d
    cy = c1[1] + a * (c2[1] - c1[1]) / d
    
    rx = -h * (c2[1] - c1[1]) / d
    ry = h * (c2[0] - c1[0]) / d
    
    return [
        np.array([cx + rx, cy + ry], dtype=np.float32),
        np.array([cx - rx, cy - ry], dtype=np.float32)
    ]


def closest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> Tuple[int, float]:
    """
    找到轨迹上最近点
    
    Args:
        point: 查询点
        trajectory: NxD 轨迹点
        
    Returns:
        (最近点索引, 到该点的距离)
    """
    distances = np.linalg.norm(trajectory - point, axis=1)
    idx = np.argmin(distances)
    return idx, distances[idx]


# =============================================================================
# 单位转换工具
# =============================================================================

def deg_to_rad(deg: float) -> float:
    """度转弧度"""
    return deg * np.pi / 180.0


def rad_to_deg(rad: float) -> float:
    """弧度转度"""
    return rad * 180.0 / np.pi


def mps_to_rpm(mps: float, radius: float) -> float:
    """线速度(m/s)转电机转速(rpm)"""
    return mps / (2 * np.pi * radius) * 60


def rpm_to_mps(rpm: float, radius: float) -> float:
    """电机转速(rpm)转线速度(m/s)"""
    return rpm / 60 * 2 * np.pi * radius


def rpm_to_radps(rpm: float) -> float:
    """rpm转rad/s"""
    return rpm * 2 * np.pi / 60.0


def radps_to_rpm(radps: float) -> float:
    """rad/s转rpm"""
    return radps * 60.0 / (2 * np.pi)


# =============================================================================
# 日志和重试工具
# =============================================================================

import logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """速率限制器"""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def allow(self) -> bool:
        """检查是否允许调用"""
        import time
        now = time.time()
        # 清除过期调用
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> callable:
    """重试装饰器"""
    import time
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}, "
                        f"retrying in {current_delay}s"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator
