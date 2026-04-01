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
        
        # 噪声等级 (仿真用)
        self.noise_level: float = 0.5  # N or N·m
        
    def open(self) -> bool:
        """打开传感器连接
        
        仿真模式: 初始化模拟传感器
        硬件模式: 建立网络/USB/CAN 连接
        
        Returns:
            bool: 连接是否成功
        """
        import time
        self._is_streaming = True
        self._last_wrench = None
        self._wrench_history = []
        self._frame_id = 0
        
        if self.ip_address:
            # ATI Net F/T 风格网络接口
            if self.ethernet_type == "udp":
                print(f"[ForceTorqueSensor] UDP connecting to {self.ip_address}:5000")
            elif self.ethernet_type == "tcp":
                print(f"[ForceTorqueSensor] TCP connecting to {self.ip_address}:5000")
            else:
                print(f"[ForceTorqueSensor] Connecting to {self.ip_address}")
        else:
            # 本地接口
            interface_map = {
                ForceSensorType.SIX_AXIS: "USB HID / CAN",
                ForceSensorType.THREE_AXIS: "USB HID",
                ForceSensorType.JOINT_TORQUE: "CAN / EtherCAT",
                ForceSensorType.FINGER_TIP: "SPI / USB"
            }
            interface = interface_map.get(self.sensor_type, "USB")
            print(f"[ForceTorqueSensor] Opened ({interface}): {self.sensor_id}")
        
        # 仿真: 读取校准数据
        print(f"[ForceTorqueSensor] Calibration loaded: scale={self.calibration.scale}, "
              f"bias={self.calibration.bias}")
        print(f"[ForceTorqueSensor] Tool center: {self.tool_center}")
        
        return True
    
    def close(self):
        """关闭传感器连接"""
        if self._is_streaming:
            self._is_streaming = False
            if self._socket:
                self._socket.close()
            print(f"[ForceTorqueSensor] {self.sensor_id} Closed")
    
    def capture(self) -> Wrench:
        """采集一帧力数据
        
        仿真模式: 生成基于物理模型的模拟力数据
        硬件模式: 从 Net F/T (UDP/TCP) / USB HID / CAN 总线读取
        
        Returns:
            Wrench: 六维力旋量 (Fx, Fy, Fz, Tx, Ty, Tz)
        """
        if not self._is_streaming:
            raise RuntimeError("Force sensor not opened")
        
        import time
        t = time.time()
        
        # --- 仿真模式: 基于物理模型的力数据生成 ---
        if self.sensor_type == ForceSensorType.SIX_AXIS:
            # ATI 风格六维力矩传感器仿真
            
            # 1. 重力补偿 (传感器固定,负载在工具中心上方)
            gravity = np.array([0.0, 0.0, -9.81])  # Z轴负方向为重力
            mass_estimate = 0.55  # kg, 估计负载质量(约5.4N重力)
            gravity_force = gravity * mass_estimate  # F = mg
            
            # 2. 工具中心偏移带来的力矩
            tcp_offset = self.tool_center  # 工具中心偏移
            gravity_torque = np.cross(tcp_offset, gravity_force)
            
            # 3. 环境扰动 (随机接触力)
            contact_noise = np.array([
                np.random.randn() * 0.5,   # Fx
                np.random.randn() * 0.5,   # Fy
                np.random.randn() * 0.2    # Fz
            ])
            torque_noise = np.random.randn(3) * 0.1
            
            # 4. 传感器噪声 (带宽相关,高采样率时噪声更大)
            force_noise = np.random.randn(3) * self.noise_level
            torque_noise_total = np.random.randn(3) * self.noise_level * 0.1
            
            # 5. 温漂 (长时间运行后偏置漂移)
            drift_seconds = t % 3600  # 每小时漂移周期
            drift_bias = 0.001 * np.sin(drift_seconds / 100)
            
            # 组合
            force = gravity_force + contact_noise + force_noise
            torque = gravity_torque + torque_noise + torque_noise_total
            torque += drift_bias
            
        elif self.sensor_type == ForceSensorType.THREE_AXIS:
            # 三维力传感器 (简化)
            force = np.random.randn(3) * 5.0
            force[2] = -5.0 + np.random.randn() * 0.5  # 主要承受垂直力
            torque = np.zeros(3)
            
        elif self.sensor_type == ForceSensorType.JOINT_TORQUE:
            # 关节力矩传感器
            force = np.zeros(3)
            torque = np.random.randn(3) * 2.0  # 关节力矩 Nm
            torque[0] = -10.0 + np.random.randn() * 1.0  # 假设关节1承受较大力矩
            
        else:  # FINGER_TIP
            # 灵巧手指尖力
            force = np.array([
                np.random.randn() * 1.0,
                np.random.randn() * 1.0,
                -2.0 + np.random.randn() * 0.3  # 抓取力
            ])
            torque = np.zeros(3)
        
        # 应用标定 (scale + bias)
        raw = np.concatenate([force, torque])
        calibrated = raw * self.calibration.scale + self.calibration.bias
        
        wrench = Wrench.from_vector(
            calibrated,
            timestamp=t,
            frame_id=self._frame_id,
            sensor_id=self.sensor_id
        )
        self._frame_id += 1
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
    
    def compute_force_direction(self, wrench: np.ndarray) -> np.ndarray:
        """
        计算力向量方向 (归一化)
        
        Args:
            wrench: 6维力旋量
            
        Returns:
            direction: 3维单位力向量方向
        """
        force = wrench[:3]
        norm = np.linalg.norm(force)
        if norm < 1e-6:
            return np.zeros(3)
        return force / norm
    
    def compute_equivalent_wrench_at(
        self,
        wrench: np.ndarray,
        translation: np.ndarray
    ) -> np.ndarray:
        """
        计算等效到指定点的力旋量
        
        用于力矩在不同参考点的变换
        
        Args:
            wrench: 原始6维力旋量
            translation: 从原始点到目标点的平移向量
            
        Returns:
            equivalent_wrench: 等效6维力旋量
        """
        force = wrench[:3]
        torque = wrench[3:6]
        
        # 新的力矩 = 原始力矩 + 平移叉乘力
        new_torque = torque + np.cross(translation, force)
        
        return np.concatenate([force, new_torque])


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

    def simulate_surface_contact(
        self,
        surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        contact_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        penetration_depth: float = 0.001,
        stiffness: float = 1000.0,
        damping: float = 50.0
    ) -> Wrench:
        """
        模拟表面接触力 (弹簧阻尼模型)
        
        Args:
            surface_normal: 表面法向量 (归一化)
            contact_point: 接触点位置 (m)
            penetration_depth: 穿透深度 (m)
            stiffness: 接触刚度 (N/m)
            damping: 接触阻尼 (N·s/m)
            
        Returns:
            Wrench with surface contact force
        """
        normal = np.array(surface_normal, dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        
        # 弹簧阻尼力
        spring_force = stiffness * penetration_depth
        damping_force = damping * np.random.randn() * 0.1  # 简化阻尼
        
        total_force = spring_force + damping_force
        
        # 力作用方向为法向量反方向
        force = -normal * total_force
        
        # 力矩 = r x F (接触点相对于传感器中心的力矩)
        contact_pt = np.array(contact_point, dtype=np.float32)
        torque = np.cross(contact_pt, force)
        
        return self.simulate_contact(
            force=tuple(force),
            torque=tuple(torque),
            add_noise=True
        )
    
    def simulate_friction_contact(
        self,
        normal_force: float = 10.0,
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        friction_coeff: float = 0.3,
        object_mass: float = 1.0
    ) -> Wrench:
        """
        模拟摩擦力
        
        Args:
            normal_force: 法向接触力 (N)
            velocity: 滑移速度 (m/s)
            friction_coeff: 摩擦系数
            object_mass: 物体质量 (kg)
            
        Returns:
            Wrench with friction force
        """
        import math
        v = np.array(velocity, dtype=np.float32)
        v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        
        if v_mag < 1e-6:
            return self.simulate_contact((0, 0, 0), (0, 0, 0), add_noise=False)
        
        # 摩擦力方向与速度方向相反
        friction_direction = -v / v_mag
        
        # 库仑摩擦模型
        max_friction = friction_coeff * normal_force
        
        # 静摩擦 vs 动摩擦
        if v_mag < 0.01:
            friction_magnitude = min(max_friction, object_mass * 9.81 * friction_coeff)
        else:
            friction_magnitude = max_friction * 0.8  # 动摩擦略小
        
        friction_force = friction_direction * friction_magnitude
        
        return self.simulate_contact(
            force=tuple(friction_force),
            torque=(0.0, 0.0, 0.0),
            add_noise=True
        )
