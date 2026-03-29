"""
力觉感知模块
============

六维力矩传感器接口
- 六轴力/力矩测量 (Fx, Fy, Fz, Tx, Ty, Tz)
- 负载估计
- 接触检测
- 协作安全监控

支持传感器:
- ATI Force/Torque 传感器
- 关节力矩传感器
- 灵巧手指尖力传感器
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class ForceSensorType(Enum):
    """力传感器类型"""
    SIX_AXIS = "six_axis"        # 六维力矩传感器
    THREE_AXIS = "three_axis"   # 三维力传感器
    JOINT_TORQUE = "joint_torque"  # 关节力矩
    FINGER_TIP = "finger_tip"   # 手指尖力


@dataclass
class Wrench:
    """
    力旋量 (力与力矩的组合)
    
    表示作用于物体的完整力状态
    """
    force: np.ndarray           # 3, 力向量 (Fx, Fy, Fz), N
    torque: np.ndarray           # 3, 力矩向量 (Tx, Ty, Tz), N·m
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "default"
    
    def __post_init__(self):
        if isinstance(self.force, list):
            self.force = np.array(self.force, dtype=np.float32)
        if isinstance(self.torque, list):
            self.torque = np.array(self.torque, dtype=np.float32)
        if self.force.shape != (3,):
            raise ValueError(f"Force must be shape (3,), got {self.force.shape}")
        if self.torque.shape != (3,):
            raise ValueError(f"Torque must be shape (3,), got {self.torque.shape}")
    
    @property
    def magnitude(self) -> float:
        """力向量大小"""
        return np.linalg.norm(self.force)
    
    @property
    def torque_magnitude(self) -> float:
        """力矩大小"""
        return np.linalg.norm(self.torque)
    
    def to_vector(self) -> np.ndarray:
        """转换为6维向量 [Fx, Fy, Fz, Tx, Ty, Tz]"""
        return np.concatenate([self.force, self.torque])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, **kwargs) -> 'Wrench':
        """从6维向量创建"""
        if vec.shape != (6,):
            raise ValueError(f"Vector must be shape (6,), got {vec.shape}")
        return cls(force=vec[:3], torque=vec[3:], **kwargs)
    
    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> 'Wrench':
        """
        坐标变换
        
        将力旋量从传感器坐标系变换到世界坐标系
        
        Args:
            rotation: 3x3 旋转矩阵
            translation: 3 平移向量
        """
        # 力向量直接旋转
        new_force = rotation @ self.force
        # 力矩 = 旋转后的力矩 + 平移叉乘力
        new_torque = rotation @ self.torque + np.cross(translation, new_force)
        return Wrench(force=new_force, torque=new_torque, sensor_id=self.sensor_id)


@dataclass
class ForceCalibration:
    """力传感器标定参数"""
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_range: Tuple[float, float] = (-1000, 1000)   # N
    torque_range: Tuple[float, float] = (-100, 100)     # N·m
    # 温漂补偿
    temp_coefficient: Optional[np.ndarray] = None


@dataclass 
class ContactState:
    """接触状态"""
    is_contact: bool
    contact_force: float = 0.0
    contact_point: Optional[np.ndarray] = None  # 3D接触点
    normal_vector: Optional[np.ndarray] = None   # 法向方向
    slip_probability: float = 0.0


class ForceTorqueSensor:
    """
    六维力矩传感器接口
    
    支持:
    - ATI Force/Torque 系列
    - 关节力矩传感器
    - 多传感器融合
    """
    
    def __init__(
        self,
        sensor_type: ForceSensorType = ForceSensorType.SIX_AXIS,
        sensor_id: str = "ft_0",
        calibration: Optional[ForceCalibration] = None,
        ip_address: Optional[str] = None,
        ethernet_type: str = "UDP"
    ):
        """
        Args:
            sensor_type: 传感器类型
            sensor_id: 传感器标识
            calibration: 标定参数
            ip_address: 网口传感器IP
            ethernet_type: TCP/UDP
        """
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.ip_address = ip_address
        self.ethernet_type = ethernet_type
        self.calibration = calibration or ForceCalibration()
        
        # 采样配置
        self.sampling_rate = 100  # Hz
        self._is_streaming = False
        self._socket = None
        
        # 状态
        self._last_wrench: Optional[Wrench] = None
        self._wrench_history: List[Wrench] = []
        self._contact_threshold = 2.0  # N, 接触检测阈值
        
        # 工具坐标系 (TCP)
        self.tool_center: np.ndarray = np.zeros(3)  # 工具中心点
        
    def open(self) -> bool:
        """打开传感器连接"""
        # TODO: 实现硬件接口
        # - Net F/T (ATI): UDP/TCP socket
        # - USB: hidapi
        # - CAN: canable
        if self.ip_address:
            print(f"[ForceTorqueSensor] Connecting to {self.ip_address} ({self.ethernet_type})")
        self._is_streaming = True
        print(f"[ForceTorqueSensor] Opened: {self.sensor_id}, Type={self.sensor_type.value}")
        return True
    
    def close(self):
        """关闭传感器连接"""
        if self._is_streaming:
            self._is_streaming = False
            if self._socket:
                self._socket.close()
            print(f"[ForceTorqueSensor] {self.sensor_id} Closed")
    
    def capture(self) -> Wrench:
        """采集一帧力数据"""
        if not self._is_streaming:
            raise RuntimeError("Force sensor not opened")
        
        # TODO: 实现实际数据采集
        # 这里返回模拟数据
        if self.sensor_type == ForceSensorType.SIX_AXIS:
            # 模拟六维力矩
            force = np.array([
                np.random.randn() * 2.0,
                np.random.randn() * 2.0,
                -9.8 + np.random.randn() * 0.5  # 补偿重力
            ])
            torque = np.random.randn(3) * 0.5
        else:
            force = np.random.randn(3) * 5.0
            torque = np.zeros(3)
        
        # 应用标定
        raw = np.concatenate([force, torque])
        calibrated = raw * self.calibration.scale + self.calibration.bias
        
        wrench = Wrench.from_vector(calibrated, sensor_id=self.sensor_id)
        self._last_wrench = wrench
        self._wrench_history.append(wrench)
        
        # 限制历史长度
        if len(self._wrench_history) > 1000:
            self._wrench_history = self._wrench_history[-500:]
        
        return wrench
    
    def get_wrench(self) -> Optional[Wrench]:
        """获取最新力数据"""
        return self._last_wrench
    
    def detect_contact(
        self,
        wrench: Optional[Wrench] = None,
        threshold: Optional[float] = None
    ) -> ContactState:
        """
        接触检测
        
        基于力信号突变检测接触事件
        """
        if wrench is None:
            wrench = self._last_wrench
        if wrench is None:
            return ContactState(is_contact=False)
        
        if threshold is None:
            threshold = self._contact_threshold
        
        contact_force = wrench.magnitude
        
        # 接触判断 (力大于阈值)
        is_contact = contact_force > threshold
        
        # 简化滑移估计
        slip_prob = 0.0
        if is_contact and len(self._wrench_history) > 10:
            # 力波动估计滑移
            recent = [w.magnitude for w in self._wrench_history[-10:]]
            std = np.std(recent)
            slip_prob = min(std / 5.0, 1.0)
        
        return ContactState(
            is_contact=is_contact,
            contact_force=contact_force,
            slip_probability=slip_prob
        )
    
    def estimate_payload(self, wrench: Optional[Wrench] = None) -> float:
        """
        估计负载重量
        
        基于静止状态下的重力分量估计
        """
        if wrench is None:
            wrench = self._last_wrench
        if wrench is None:
            return 0.0
        
        # 简化估计: Fz 重力分量
        gravity = 9.81
        payload = abs(wrench.force[2]) / gravity
        return max(0, payload)
    
    def set_tool_center(self, tool_mass: float, tool_com: np.ndarray):
        """
        设置工具中心参数
        
        用于重力补偿
        
        Args:
            tool_mass: 工具质量 (kg)
            tool_com: 工具质心在传感器坐标系中的位置 (m)
        """
        self.tool_center = tool_com
        
        # 计算重力补偿
        gravity_compensation = np.array([0, 0, tool_mass * 9.81])
        torque_compensation = np.cross(tool_com, gravity_compensation)
        
        # 更新偏置
        self.calibration.bias[3:6] = -torque_compensation
        print(f"[ForceTorqueSensor] Tool center set: mass={tool_mass}kg, COM={tool_com}")
    
    def calibrate_bias(self, num_samples: int = 100):
        """
        偏置校准
        
        在无负载状态下采集零点
        """
        print(f"[ForceTorqueSensor] Calibrating bias with {num_samples} samples...")
        
        biases = []
        for _ in range(num_samples):
            w = self.capture()
            biases.append(w.to_vector())
        
        self.calibration.bias = -np.mean(biases, axis=0)
        print(f"[ForceTorqueSensor] Bias calibrated: {self.calibration.bias}")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class WrenchProcessor:
    """
    力旋量信号处理器
    
    功能:
    - 噪声滤波
    - 异常值去除
    - 物理一致性检验
    - 协方差估计
    """
    
    def __init__(
        self,
        filter_alpha: float = 0.3,
        outlier_threshold: float = 3.0
    ):
        """
        Args:
            filter_alpha: 指数移动平均系数
            outlier_threshold: 异常值倍数 (标准差)
        """
        self.filter_alpha = filter_alpha
        self.outlier_threshold = outlier_threshold
        self._filtered_wrench: Optional[np.ndarray] = None
        self._running_mean = np.zeros(6)
        self._running_var = np.ones(6)
        
    def filter(self, wrench: np.ndarray, return_wrench: bool = False):
        """
        滤波处理
        
        Args:
            wrench: 6维力旋量
            return_wrench: 是否返回Wrench对象
            
        Returns:
            filtered: 6维滤波后力旋量
        """
        if self._filtered_wrench is None:
            self._filtered_wrench = wrench.copy()
            return Wrench.from_vector(wrench) if return_wrench else wrench
        
        # 指数移动平均
        self._filtered_wrench = (
            self.filter_alpha * wrench +
            (1 - self.filter_alpha) * self._filtered_wrench
        )
        
        return Wrench.from_vector(self._filtered_wrench) if return_wrench else self._filtered_wrench
    
    def remove_outliers(
        self,
        wrench: np.ndarray,
        history: List[np.ndarray]
    ) -> np.ndarray:
        """
        去除异常值
        
        基于历史数据检测并修正异常测量
        """
        if len(history) < 5:
            return wrench
        
        # 统计
        history_arr = np.array(history)
        mean = np.mean(history_arr, axis=0)
        std = np.std(history_arr, axis=0) + 1e-6
        
        # 检测
        z_scores = np.abs(wrench - mean) / std
        
        # 替换异常值
        outlier_mask = z_scores > self.outlier_threshold
        if np.any(outlier_mask):
            wrench = np.where(outlier_mask, mean, wrench)
        
        return wrench
    
    def estimate_covariance(self, history: List[np.ndarray]) -> np.ndarray:
        """
        估计测量协方差矩阵
        
        用于卡尔曼滤波等状态估计
        """
        if len(history) < 2:
            return np.eye(6)
        
        history_arr = np.array(history)
        return np.cov(history_arr, rowvar=False) + np.eye(6) * 1e-6


# AGV五级力觉规格
AGV_FORCE_GRADES = {
    'S':  {'axes': 3, 'force_range': 100,  'torque_range': 10,   'resolution': 0.1,  'sampling_hz': 100},
    'M':  {'axes': 6, 'force_range': 200,  'torque_range': 20,   'resolution': 0.05, 'sampling_hz': 500},
    'L':  {'axes': 6, 'force_range': 500,  'torque_range': 50,   'resolution': 0.02, 'sampling_hz': 1000},
    'XL': {'axes': 6, 'force_range': 1000, 'torque_range': 100,  'resolution': 0.01, 'sampling_hz': 2000},
    'XXL': {'axes': 6, 'force_range': 5000, 'torque_range': 500, 'resolution': 0.005, 'sampling_hz': 5000},
}


def get_force_spec(grade: str) -> dict:
    """获取AGV指定等级的力觉规格"""
    return AGV_FORCE_GRADES.get(grade, AGV_FORCE_GRADES['M'])


class VirtualForceSensor:
    """
    虚拟力觉传感器 (仿真环境使用)
    
    模拟六维力矩传感器，用于:
    - 仿真环境中的力反馈
    - 接触力预测和验证
    - 阻抗控制算法测试
    """
    
    def __init__(
        self,
        sensor_id: str = "virtual_force",
        noise_level: float = 0.02,
        bias_range: float = 0.1
    ):
        self.sensor_id = sensor_id
        self.noise_level = noise_level
        self.bias_range = bias_range
        self._is_opened = False
        self._frame_id = 0
        self._current_wrench = Wrench(
            force=np.zeros(3),
            torque=np.zeros(3),
            timestamp=0.0,
            frame_id=0,
            sensor_id=sensor_id
        )
        self._bias = np.zeros(6)
    
    def open(self) -> bool:
        self._is_opened = True
        # 随机初始化偏置
        self._bias = np.random.randn(6) * self.bias_range
        return True
    
    def close(self):
        self._is_opened = False
    
    def simulate_contact(
        self,
        force: Tuple[float, float, float],
        torque: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        add_noise: bool = True
    ) -> Wrench:
        """
        模拟接触力
        
        Args:
            force: 力向量 (Fx, Fy, Fz) in N
            torque: 力矩向量 (Tx, Ty, Tz) in N·m
            add_noise: 是否添加噪声
            
        Returns:
            Wrench with simulated force/torque
        """
        f = np.array(force, dtype=np.float32)
        t = np.array(torque, dtype=np.float32)
        
        if add_noise:
            noise_f = np.random.randn(3) * self.noise_level * 10
            noise_t = np.random.randn(3) * self.noise_level * 0.5
            f = f + noise_f
            t = t + noise_t
        
        wrench = Wrench(
            force=f + self._bias[:3],
            torque=t + self._bias[3:],
            timestamp=0.0,
            frame_id=self._frame_id,
            sensor_id=self.sensor_id
        )
        self._current_wrench = wrench
        self._frame_id += 1
        return wrench
    
    def simulate_payload(
        self,
        mass: float = 1.0,
        com_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity: float = 9.81
    ) -> Wrench:
        """
        模拟负载重力
        
        Args:
            mass: 负载质量 (kg)
            com_offset: 重心偏移 (x, y, z) in m
            gravity: 重力加速度 (m/s^2)
            
        Returns:
            Wrench with payload gravity
        """
        fz = -mass * gravity  # 向下为负
        tx = mass * gravity * com_offset[1]
        ty = -mass * gravity * com_offset[0]
        tz = 0.0
        
        return self.simulate_contact(
            force=(0.0, 0.0, fz),
            torque=(tx, ty, tz),
            add_noise=False
        )
    
    def simulate_collision(
        self,
        direction: Tuple[float, float, float],
        peak_force: float = 50.0,
        duration_ms: float = 100.0,
        decay: str = "exponential"
    ) -> List[Wrench]:
        """
        模拟碰撞事件
        
        Args:
            direction: 碰撞方向 (归一化)
            peak_force: 峰值力 (N)
            duration_ms: 持续时间 (毫秒)
            decay: 衰减模式 ('exponential' or 'linear')
            
        Returns:
            List of Wrench measurements during collision
        """
        frames = []
        n_frames = int(duration_ms / 10)  # 假设10ms一帧
        direction = np.array(direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        for i in range(n_frames):
            t = i / n_frames
            
            if decay == "exponential":
                amplitude = peak_force * np.exp(-5 * t)
            else:
                amplitude = peak_force * (1 - t)
            
            force = direction * amplitude + np.random.randn(3) * self.noise_level * 10
            torque = np.random.randn(3) * self.noise_level * 0.5
            
            wrench = Wrench(
                force=force.astype(np.float32),
                torque=torque.astype(np.float32),
                timestamp=i * 0.01,
                frame_id=self._frame_id,
                sensor_id=self.sensor_id
            )
            frames.append(wrench)
            self._frame_id += 1
        
        return frames
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
