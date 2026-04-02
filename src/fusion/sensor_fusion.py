"""
传感器融合模块 (Sensor Fusion)
支持扩展卡尔曼滤波(EKF)、互补滤波、多传感器数据融合
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


class SensorFusion(ABC):
    """传感器融合基类"""

    @abstractmethod
    def update(self, measurements: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """更新融合状态"""
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """获取当前状态"""
        pass


class ComplementaryFilter(SensorFusion):
    """互补滤波器 - 适用于IMU数据融合"""

    def __init__(self, alpha: float = 0.98):
        """
        alpha: 高通滤波权重 (陀螺仪)
        1-alpha: 低通滤波权重 (加速度计)
        """
        self.alpha = alpha
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = 0.0
        self._initialized = False

    def update(self, measurements: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """
        measurements包含:
        - 'accel': 加速度 [ax, ay, az]
        - 'gyro': 角速度 [wx, wy, wz]
        """
        accel = measurements.get('accel')
        gyro = measurements.get('gyro')

        if accel is not None and len(accel) >= 3:
            # 从加速度计算姿态
            pitch_acc = np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2))
            roll_acc = np.arctan2(-accel[0], accel[2])
        else:
            pitch_acc = 0.0
            roll_acc = 0.0

        if gyro is not None and len(gyro) >= 3:
            # 陀螺仪积分
            self._pitch += gyro[1] * dt  # pitch rate
            self._roll += gyro[0] * dt   # roll rate
            self._yaw += gyro[2] * dt    # yaw rate
        else:
            return np.array([self._roll, self._pitch, self._yaw])

        # 互补滤波融合
        if not self._initialized:
            self._pitch = pitch_acc
            self._roll = roll_acc
            self._initialized = True
        else:
            self._pitch = self.alpha * self._pitch + (1 - self.alpha) * pitch_acc
            self._roll = self.alpha * self._roll + (1 - self.alpha) * roll_acc

        return np.array([self._roll, self._pitch, self._yaw])

    def get_state(self) -> np.ndarray:
        return np.array([self._roll, self._pitch, self._yaw])

    def reset(self):
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = 0.0
        self._initialized = False


class ExtendedKalmanFilter(SensorFusion):
    """扩展卡尔曼滤波器 (EKF) - 用于传感器融合"""

    def __init__(self, state_dim: int, measurement_dim: int, 
                 process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # 状态 [x, y, theta, v, omega] 或类似
        self._state = np.zeros(state_dim)
        # 协方差矩阵
        self._P = np.eye(state_dim)
        # 过程噪声
        self.Q = np.eye(state_dim) * process_noise
        # 测量噪声
        self.R = np.eye(measurement_dim) * measurement_noise
        # 状态转移矩阵 (需要根据模型设置)
        self.F = np.eye(state_dim)
        # 观测矩阵 (需要根据模型设置)
        self.H = np.zeros((measurement_dim, state_dim))

    def initialize(self, state: np.ndarray, P: Optional[np.ndarray] = None):
        """初始化状态"""
        self._state = state.copy()
        if P is not None:
            self._P = P.copy()

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """预测步骤"""
        # 更新状态转移矩阵 (简单匀速模型)
        self._state = self.F @ self._state
        self._P = self.F @ self._P @ self.F.T + self.Q

    def correct(self, measurement: np.ndarray):
        """校正步骤"""
        # 计算卡尔曼增益
        S = self.H @ self._P @ self.H.T + self.R
        K = self._P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        innovation = measurement - self.H @ self._state
        self._state = self._state + K @ innovation

        # 更新协方差
        I = np.eye(self.state_dim)
        self._P = (I - K @ self.H) @ self._P

    def update(self, measurements: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """完整EKF更新"""
        # 预测
        self.predict(dt)

        # 简单观测: 取第一个可用的测量
        if measurements:
            z = np.array(list(measurements.values())[0])
            if len(z) == self.measurement_dim:
                self.correct(z)

        return self._state.copy()

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        return self._P.copy()

    def set_matrices(self, F: np.ndarray, H: np.ndarray):
        """设置状态转移和观测矩阵"""
        self.F = F
        self.H = H


class MultiSensorFusion:
    """多传感器数据融合中心"""

    def __init__(self):
        self.fusion_methods: Dict[str, SensorFusion] = {}
        self._weights: Dict[str, float] = {}
        self._last_update_time: Dict[str, float] = {}

    def add_fusion_method(self, name: str, method: SensorFusion, weight: float = 1.0):
        """添加融合方法"""
        self.fusion_methods[name] = method
        self._weights[name] = weight

    def update(self, sensor_data: Dict[str, Dict[str, np.ndarray]], dt: float) -> Dict[str, np.ndarray]:
        """更新所有融合方法"""
        results = {}
        total_weight = sum(self._weights.values())

        for name, data in sensor_data.items():
            if name in self.fusion_methods:
                state = self.fusion_methods[name].update(data, dt)
                results[name] = state

        return results

    def get_fused_state(self) -> np.ndarray:
        """获取融合后的最终状态"""
        if not self.fusion_methods:
            return np.array([])

        states = []
        weights = []

        for name, method in self.fusion_methods.items():
            state = method.get_state()
            if len(state) > 0:
                states.append(state)
                weights.append(self._weights.get(name, 1.0))

        if not states:
            return np.array([])

        weights = np.array(weights) / sum(weights)
        return sum(w * s for w, s in zip(weights, states))

    def fuse_tactile_force(self, tactile_data, force_data) -> np.ndarray:
        """融合触觉和力觉数据"""
        tactile_vec = tactile_data.to_vector() if hasattr(tactile_data, 'to_vector') else np.array([0])
        force_vec = force_data.to_vector() if hasattr(force_data, 'to_vector') else np.array([0])

        # 简单加权融合
        return 0.3 * tactile_vec + 0.7 * force_vec

    def fuse_imu_odom(self, imu_data, odom_data) -> Dict:
        """融合IMU和里程计数据"""
        # EKF融合
        fused = {
            'pose': np.zeros(3),  # x, y, theta
            'twist': np.zeros(3), # vx, vy, omega
            'covariance': np.eye(3)
        }

        # 互补滤波
        if hasattr(imu_data, 'acceleration') and hasattr(imu_data, 'angular_velocity'):
            imu_fusion = ComplementaryFilter(alpha=0.96)
            imu_fusion.update({
                'accel': imu_data.acceleration,
                'gyro': imu_data.angular_velocity
            }, dt=0.01)
            fused['pose'][2] = imu_fusion.get_state()[2]  # yaw

        return fused
