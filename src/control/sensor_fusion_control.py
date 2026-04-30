# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
传感器融合控制模块
================

统一的传感器→融合→控制闭环模块

功能:
- 传感器原始数据采集
- 互补滤波/EKF姿态融合
- 触觉/力觉/IMU多模态融合
- 统一控制指令输出
- AGV五级规格适配

支持等级: S / M / L / XL / XXL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum


class FusionControlGrade(Enum):
    """融合控制等级"""
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


@dataclass
class SensorFusionControlState:
    """融合控制状态"""
    # 传感器原始数据
    imu_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_mag: Optional[np.ndarray] = None
    force: np.ndarray = field(default_factory=lambda: np.zeros(6))  # Fx,Fy,Fz,Tx,Ty,Tz
    tactile_pressure: Optional[np.ndarray] = None

    # 融合状态
    fused_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    fused_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fused_position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 接触状态
    contact_detected: bool = False
    contact_force: float = 0.0
    slip_probability: float = 0.0

    # 控制输出
    velocity_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 元数据
    timestamp: float = 0.0
    frame_id: int = 0


@dataclass
class FusionControlConfig:
    """融合控制配置"""
    grade: FusionControlGrade = FusionControlGrade.M
    fusion_algorithm: str = "complementary"  # "complementary" / "ekf"
    control_frequency: int = 100  # Hz
    imu_enabled: bool = True
    force_enabled: bool = True
    tactile_enabled: bool = True
    # 互补滤波参数
    complementary_alpha: float = 0.96
    # EKF参数
    ekf_process_noise: float = 0.01
    ekf_measurement_noise: float = 0.1


class SensorFusionController:
    """
    传感器融合控制统一接口

    将 IMU/力觉/触觉 传感器数据融合为统一感知状态，
    并输出控制指令到执行器。

    支持 AGV 五级规格 (S/M/L/XL/XXL)
    """

    GRADE_CONFIGS = {
        FusionControlGrade.S: {
            "fusion_frequency": 50,
            "fusion_algorithm": "complementary",
            "control_frequency": 50,
            "imu_sample_rate": 100,
            "force_sample_rate": 100,
            "tactile_sample_rate": 50,
        },
        FusionControlGrade.M: {
            "fusion_frequency": 100,
            "fusion_algorithm": "complementary",
            "control_frequency": 100,
            "imu_sample_rate": 200,
            "force_sample_rate": 500,
            "tactile_sample_rate": 100,
        },
        FusionControlGrade.L: {
            "fusion_frequency": 200,
            "fusion_algorithm": "ekf",
            "control_frequency": 200,
            "imu_sample_rate": 500,
            "force_sample_rate": 1000,
            "tactile_sample_rate": 200,
        },
        FusionControlGrade.XL: {
            "fusion_frequency": 500,
            "fusion_algorithm": "ekf",
            "control_frequency": 500,
            "imu_sample_rate": 1000,
            "force_sample_rate": 2000,
            "tactile_sample_rate": 500,
        },
        FusionControlGrade.XXL: {
            "fusion_frequency": 1000,
            "fusion_algorithm": "ekf",
            "control_frequency": 1000,
            "imu_sample_rate": 2000,
            "force_sample_rate": 5000,
            "tactile_sample_rate": 1000,
        },
    }

    def __init__(
        self,
        grade: FusionControlGrade = FusionControlGrade.M,
        config: Optional[FusionControlConfig] = None,
        sensor_id: str = "fusion_ctrl_0"
    ):
        self.grade = grade
        self.sensor_id = sensor_id

        # 获取等级配置
        grade_cfg = self.GRADE_CONFIGS[grade]
        self.fusion_frequency = grade_cfg["fusion_frequency"]
        self.control_frequency = grade_cfg["control_frequency"]
        # config 优先于 grade 默认值
        self.config = config
        if self.config is None:
            self.config = FusionControlConfig(grade=grade)
            self.fusion_algorithm = grade_cfg["fusion_algorithm"]
        else:
            self.fusion_algorithm = self.config.fusion_algorithm

        # 初始化滤波器
        self._pose_filter = _ComplementaryFilter(alpha=self.config.complementary_alpha)
        self._ekf: Optional[_SimpleEKF] = None
        if self.fusion_algorithm == "ekf":
            self._ekf = _SimpleEKF(
                process_noise=self.config.ekf_process_noise,
                measurement_noise=self.config.ekf_measurement_noise
            )

        # 状态
        self._state = SensorFusionControlState()
        self._frame_id = 0
        self._is_running = False

        # 速度/位置积分
        self._velocity = np.zeros(3)
        self._position = np.zeros(3)
        self._initialized = False

    def start(self):
        """启动融合控制器"""
        self._is_running = True
        self._frame_id = 0
        self._initialized = False
        print(f"[SensorFusionController] Started (grade={self.grade.value}, "
              f"algorithm={self.fusion_algorithm}, "
              f"fusion_freq={self.fusion_frequency}Hz, "
              f"ctrl_freq={self.control_frequency}Hz)")

    def stop(self):
        """停止融合控制器"""
        self._is_running = False
        print(f"[SensorFusionController] {self.sensor_id} Stopped")

    def update(
        self,
        imu_accel: Optional[np.ndarray] = None,
        imu_gyro: Optional[np.ndarray] = None,
        imu_mag: Optional[np.ndarray] = None,
        force_wrench: Optional[np.ndarray] = None,
        tactile_pressure: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> SensorFusionControlState:
        """
        更新融合控制状态

        Args:
            imu_accel: 3D 加速度 (m/s^2)
            imu_gyro: 3D 角速度 (rad/s)
            imu_mag: 3D 磁力计 (optional)
            force_wrench: 6D 力旋量 [Fx,Fy,Fz,Tx,Ty,Tz]
            tactile_pressure: 触觉压力图 (H x W)
            dt: 时间步长 (秒)

        Returns:
            SensorFusionControlState: 当前融合控制状态
        """
        if dt is None:
            dt = 1.0 / self.fusion_frequency

        self._state.frame_id = self._frame_id

        # 1. 存储传感器原始数据
        if imu_accel is not None:
            self._state.imu_accel = np.array(imu_accel, dtype=np.float32)
        if imu_gyro is not None:
            self._state.imu_gyro = np.array(imu_gyro, dtype=np.float32)
        if imu_mag is not None:
            self._state.imu_mag = np.array(imu_mag, dtype=np.float32)
        if force_wrench is not None:
            self._state.force = np.array(force_wrench, dtype=np.float32)
        if tactile_pressure is not None:
            self._state.tactile_pressure = np.array(tactile_pressure, dtype=np.float32)

        # 2. 姿态融合
        if imu_accel is not None and imu_gyro is not None:
            if self.fusion_algorithm == "complementary":
                rpy = self._pose_filter.update(imu_accel, imu_gyro, dt)
            else:  # ekf
                rpy = self._ekf.update(imu_accel, imu_gyro, dt) if self._ekf else np.zeros(3)

            self._state.fused_pose = rpy

            # 3. 速度/位置积分 (去除重力)
            gravity = np.array([0.0, 0.0, 9.81])
            accel_body = imu_accel - gravity
            self._velocity = self._velocity + accel_body * dt
            self._velocity = self._velocity * 0.98  # 阻尼
            self._position = self._position + self._velocity * dt
            self._state.fused_velocity = self._velocity.copy()
            self._state.fused_position = self._position.copy()

        # 4. 接触检测
        if force_wrench is not None:
            force_mag = np.linalg.norm(force_wrench[:3])
            torque_mag = np.linalg.norm(force_wrench[3:6])
            self._state.contact_force = force_mag
            self._state.contact_detected = force_mag > 2.0  # 2N threshold

        # 5. 滑移检测
        if tactile_pressure is not None and force_wrench is not None:
            pressure_var = np.var(tactile_pressure)
            force_change = abs(np.linalg.norm(force_wrench[:3]) - self._state.contact_force)
            # 压力分布变化 + 力突变 → 滑移
            slip_score = min(pressure_var * 10 + force_change * 0.1, 1.0)
            self._state.slip_probability = slip_score

        # 6. 生成控制指令 (基于感知状态)
        self._compute_control_commands(dt)

        self._state.timestamp += dt
        self._frame_id += 1
        self._initialized = True

        return self._state

    def _compute_control_commands(self, dt: float):
        """基于感知状态计算控制指令"""
        # 姿态稳定控制
        roll, pitch, yaw = self._state.fused_pose
        # 目标姿态 = 水平 → 误差 = 当前姿态
        Kp_roll = 5.0
        Kp_pitch = 5.0
        Kp_yaw = 2.0

        torque_cmd = np.array([
            -Kp_roll * roll,   # roll 修正力矩
            -Kp_pitch * pitch,  # pitch 修正力矩
            -Kp_yaw * yaw,     # yaw 修正力矩
        ])

        # 接触时添加力控
        if self._state.contact_detected:
            # 导纳控制: 根据接触力调整速度
            force_dir = self._state.force[:3] / (np.linalg.norm(self._state.force[:3]) + 1e-6)
            admittance_gain = 0.05
            velocity_adjust = force_dir * self._state.contact_force * admittance_gain
            self._state.velocity_cmd = self._velocity + velocity_adjust
        else:
            self._state.velocity_cmd = self._velocity.copy()

        self._state.torque_cmd = torque_cmd

    def get_state(self) -> SensorFusionControlState:
        """获取当前状态"""
        return self._state

    def get_control_command(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取控制指令 (velocity, torque)"""
        return self._state.velocity_cmd.copy(), self._state.torque_cmd.copy()

    def reset(self):
        """重置状态"""
        self._state = SensorFusionControlState()
        self._velocity = np.zeros(3)
        self._position = np.zeros(3)
        self._frame_id = 0
        self._initialized = False
        self._pose_filter.reset()
        if self._ekf:
            self._ekf.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class _ComplementaryFilter:
    """简化互补滤波器"""
    def __init__(self, alpha: float = 0.96):
        self.alpha = alpha
        self._euler = np.zeros(3)

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        # 加速度计估算
        accel_n = accel / (np.linalg.norm(accel) + 1e-6)
        accel_roll = np.arctan2(accel_n[1], accel_n[2])
        accel_pitch = np.arctan2(-accel_n[0], np.sqrt(accel_n[1]**2 + accel_n[2]**2))

        # 陀螺仪积分
        self._euler[0] += gyro[0] * dt  # roll
        self._euler[1] += gyro[1] * dt  # pitch
        self._euler[2] += gyro[2] * dt  # yaw

        # 互补融合
        self._euler[0] = self.alpha * self._euler[0] + (1 - self.alpha) * accel_roll
        self._euler[1] = self.alpha * self._euler[1] + (1 - self.alpha) * accel_pitch

        return self._euler.copy()

    def reset(self):
        self._euler = np.zeros(3)


class _SimpleEKF:
    """简化扩展卡尔曼滤波器"""
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._state = np.zeros(3)
        self._P = np.eye(3) * 0.1  # 状态协方差

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        # 预测
        self._state = self._state + gyro * dt
        F = np.eye(3)
        Q = np.eye(3) * self.process_noise
        self._P = F @ self._P @ F.T + Q

        # 观测 (加速度计)
        accel_n = accel / (np.linalg.norm(accel) + 1e-6)
        z = np.array([
            np.arctan2(accel_n[1], accel_n[2]),
            np.arctan2(-accel_n[0], np.sqrt(accel_n[1]**2 + accel_n[2]**2)),
            self._state[2]
        ])

        # 更新
        H = np.eye(3)
        R = np.eye(3) * self.measurement_noise
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._state = self._state + K @ (z - H @ self._state)
        self._P = (np.eye(3) - K @ H) @ self._P

        return self._state.copy()

    def reset(self):
        self._state = np.zeros(3)
        self._P = np.eye(3) * 0.1


# AGV五级融合控制规格
AGV_FUSION_CONTROL_GRADES = {
    'S':  {'freq': 50,  'algorithm': 'complementary', 'imu_rate': 100,  'force_rate': 100,  'tactile_rate': 50,  'latency_ms': 20},
    'M':  {'freq': 100, 'algorithm': 'complementary', 'imu_rate': 200,  'force_rate': 500,  'tactile_rate': 100, 'latency_ms': 10},
    'L':  {'freq': 200, 'algorithm': 'ekf',          'imu_rate': 500,  'force_rate': 1000, 'tactile_rate': 200, 'latency_ms': 5},
    'XL': {'freq': 500, 'algorithm': 'ekf',          'imu_rate': 1000, 'force_rate': 2000, 'tactile_rate': 500, 'latency_ms': 2},
    'XXL': {'freq': 1000, 'algorithm': 'ekf',        'imu_rate': 2000, 'force_rate': 5000, 'tactile_rate': 1000, 'latency_ms': 1},
}


def get_fusion_control_spec(grade: str) -> dict:
    """获取AGV指定等级的融合控制规格"""
    return AGV_FUSION_CONTROL_GRADES.get(grade, AGV_FUSION_CONTROL_GRADES['M'])
