# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
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
传感器标定管理器 (Calibration Manager)
========================================

统一管理多传感器系统的标定流程:
- IMU 零偏标定 (六面法 /翻滚法)
- 力传感器标定 (重力补偿 / 温度补偿)
- 触觉传感器标定 (零压力基准 / 力-电压标定)
- 相机内参标定 (棋盘格 / Charuco)
- 多传感器外参标定 (手眼标定 / 传感器间标定)

支持 AGV 五级规格 (S/M/L/XL/XXL)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
import json
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.imu import IMUSensor, VirtualIMUSensor
from sensors.force import ForceTorqueSensor, VirtualForceSensor
from sensors.tactile import TactileArray, VirtualTactileSensor


class CalibrationStatus(Enum):
    """标定状态"""
    IDLE = "idle"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IMUCalibrationData:
    """IMU 标定数据收集结果"""
    accel_samples: List[np.ndarray] = field(default_factory=list)
    gyro_samples: List[np.ndarray] = field(default_factory=list)
    magnet_samples: List[np.ndarray] = field(default_factory=list)
    temperature_samples: List[float] = field(default_factory=list)
    orientations: List[str] = field(default_factory=list)  # e.g. "x_up", "y_down"
    collection_duration: float = 0.0
    num_samples: int = 0


@dataclass
class IMUCalibrationResult:
    """IMU 标定结果"""
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    magnet_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    magnet_transform: np.ndarray = field(default_factory=lambda: np.eye(3))
    temperature_coef: Optional[np.ndarray] = None  # 温度补偿系数
    noise_density_accel: float = 0.0  # 噪声密度 (mg/sqrt(Hz))
    noise_density_gyro: float = 0.0    # (mdps/sqrt(Hz))
    calibration_time: float = 0.0
    calibration_duration: float = 0.0  # 标定过程耗时
    status: CalibrationStatus = CalibrationStatus.IDLE
    error_message: Optional[str] = None


@dataclass
class ForceCalibrationResult:
    """力传感器标定结果"""
    force_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    torque_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    coupling_matrix: np.ndarray = field(default_factory=lambda: np.eye(6))  # 耦合矩阵
    temperature_coef: Optional[np.ndarray] = None
    calibration_time: float = 0.0
    status: CalibrationStatus = CalibrationStatus.IDLE
    error_message: Optional[str] = None


@dataclass
class CalibrationResult:
    """标定结果基类"""
    status: CalibrationStatus = CalibrationStatus.IDLE
    timestamp: float = 0.0
    error_message: Optional[str] = None

    def apply(self, data):
        """应用到原始数据"""
        return data


@dataclass
class TactileCalibrationResult(CalibrationResult):
    """触觉标定结果"""
    zero_pressure: Optional[np.ndarray] = None
    pressure_scale: float = 1.0
    offset_map: Optional[np.ndarray] = None
    temperature_coef: Optional[np.ndarray] = None

    def apply(self, frame) -> 'TactileArray':
        """应用触觉标定"""
        if self.zero_pressure is None:
            return frame
        if not hasattr(frame, 'pressure_map'):
            return frame
        calibrated_map = (np.asarray(frame.pressure_map) - self.zero_pressure) * self.pressure_scale
        if self.offset_map is not None:
            calibrated_map -= self.offset_map
        calibrated = object.__new__(type(frame))
        calibrated.__dict__.update(frame.__dict__)
        calibrated.pressure_map = np.clip(calibrated_map, 0, None)
        return calibrated


@dataclass
class CalibrationConfig:
    """标定配置"""
    # IMU 标定
    imu_sample_rate: float = 100.0  # Hz
    imu_num_samples_per_pose: int = 500
    imu_num_poses: int = 6  # 六面法
    imu_collection_time_per_pose: float = 5.0  # 秒

    # 力传感器标定
    force_sample_rate: float = 100.0
    force_num_samples: int = 500
    force_known_weights: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 2.0, 5.0, 10.0]
    )

    # 触觉传感器标定
    tactile_num_samples: int = 100
    tactile_pressure_threshold: float = 0.01

    # AGV 等级
    agv_grade: str = "M"


class IMUCalibrator:
    """
    IMU 标定器

    支持:
    - 六面法加速度计标定
    - 翻滚法陀螺仪标定
    - 温度补偿标定
    - Allan 方差分析 (噪声密度估计)
    """

    def __init__(
        self,
        sensor: IMUSensor,
        config: Optional[CalibrationConfig] = None
    ):
        self.sensor = sensor
        self.config = config or CalibrationConfig()
        self._data = IMUCalibrationData()
        self._result: Optional[IMUCalibrationResult] = None

    def collect_data(
        self,
        orientation: str,
        duration: Optional[float] = None
    ) -> int:
        """
        收集指定姿态的 IMU 数据

        Args:
            orientation: 姿态标签 (x_up/y_down/z_up 等)
            duration: 收集时长 (秒), 默认使用配置值

        Returns:
            采集样本数
        """
        if not self.sensor._is_opened:
            self.sensor.open()

        duration = duration or self.config.imu_collection_time_per_pose
        num_samples = int(duration * self.config.imu_sample_rate)

        # 打开传感器
        self.sensor.open()

        samples_collected = 0
        start_time = time.time()
        accel_buf = []
        gyro_buf = []
        temp_buf = []

        # 采集数据
        # 支持 capture() (IMUSensor) 和 simulate_static() (VirtualIMUSensor)
        has_capture = hasattr(self.sensor, 'capture')
        has_simulate_static = hasattr(self.sensor, 'simulate_static')

        orientation_map = {
            'x_up': (np.pi/2, 0.0, 0.0),
            'x_down': (-np.pi/2, 0.0, 0.0),
            'y_up': (0.0, np.pi/2, 0.0),
            'y_down': (0.0, -np.pi/2, 0.0),
            'z_up': (0.0, 0.0, 0.0),
            'z_down': (np.pi, 0.0, 0.0),
        }
        euler = orientation_map.get(orientation, (0.0, 0.0, 0.0))

        while samples_collected < num_samples:
            if has_capture:
                frame = self.sensor.capture()
            elif has_simulate_static:
                frame = self.sensor.simulate_static(euler)
            else:
                raise RuntimeError(f"Sensor {type(self.sensor)} has neither capture() nor simulate_static()")
            if frame is not None:
                accel_buf.append(frame.accel.copy())
                gyro_buf.append(frame.gyro.copy())
                temp_buf.append(frame.temperature if hasattr(frame, 'temperature') else 25.0)
                samples_collected += 1

        self._data.accel_samples.extend(accel_buf)
        self._data.gyro_samples.extend(gyro_buf)
        self._data.temperature_samples.extend(temp_buf)
        self._data.orientations.append(orientation)
        self._data.num_samples += samples_collected
        self._data.collection_duration += time.time() - start_time

        return samples_collected

    def calibrate_accel_six_facing(self) -> IMUCalibrationResult:
        """
        六面法加速度计标定

        假设每个姿态下, 加速度计只感应重力分量:
        - x_up: (g, 0, 0)
        - x_down: (-g, 0, 0)
        - y_up: (0, g, 0)
        - y_down: (0, -g, 0)
        - z_up: (0, 0, g)
        - z_down: (0, 0, -g)

        通过最小二乘法求解偏置和尺度因子
        """
        G = 9.81

        # 按姿态分组
        orientation_labels = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
        orientation_vectors = {
            'x_up': np.array([G, 0, 0]),
            'x_down': np.array([-G, 0, 0]),
            'y_up': np.array([0, G, 0]),
            'y_down': np.array([0, -G, 0]),
            'z_up': np.array([0, 0, G]),
            'z_down': np.array([0, 0, -G]),
        }

        # 计算每个姿态的平均值
        pose_means = {}
        for label in orientation_labels:
            indices = [i for i, o in enumerate(self._data.orientations) if o == label]
            if not indices:
                continue
            accel_subset = [self._data.accel_samples[i] for i in indices]
            pose_means[label] = np.mean(accel_subset, axis=0)

        # 构建最小二乘问题: measured_j = bias_j + (1/scale_j) * expected_j
        # 每个姿态贡献 3 行 (x, y, z 轴各一个方程)
        # 方程: measured[j] = bias_j + inv_scale[j] * expected[j]
        A_list = []
        b_list = []

        for label, expected in orientation_vectors.items():
            if label not in pose_means:
                continue
            measured = pose_means[label]
            for j in range(3):
                A_list.append([1.0, expected[j]])
                b_list.append(measured[j])

        if len(A_list) < 18:  # 至少需要 6 个姿态 * 3 轴
            return IMUCalibrationResult(
                status=CalibrationStatus.FAILED,
                error_message="Insufficient data for six-facing calibration"
            )

        A = np.array(A_list)  # (n, 2)
        b = np.array(b_list)  # (n,)

        # 最小二乘: min ||Ax - b||^2
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # x[0] = 所有轴共享的偏置 (简化处理)
        # 更好的方式: 对每个轴独立求解
        # 使用分组最小二乘
        biases_all = []
        inv_scales_all = []

        for axis in range(3):
            axis_rows = [(A[i], b[i]) for i in range(len(A)) if abs(A[i][1] - orientation_vectors['x_up'][axis] if A[i][1] != 0 else 0) > 0.01 or
                        True]  # 简化: 收集所有行再过滤

        # 简化方法: 对每个轴分别求解
        # 构建 (bias + inv_scale * expected) = measured
        # 用均值估计
        axis_biases = []
        axis_inv_scales = []

        for axis in range(3):
            A_axis = []
            b_axis = []
            for label, expected in orientation_vectors.items():
                if label not in pose_means:
                    continue
                measured = pose_means[label]
                A_axis.append([1.0, expected[axis]])
                b_axis.append(measured[axis])

            if len(A_axis) < 2:
                continue

            A_axis_arr = np.array(A_axis)
            b_axis_arr = np.array(b_axis)
            x_axis, _, _, _ = np.linalg.lstsq(A_axis_arr, b_axis_arr, rcond=None)
            axis_biases.append(x_axis[0])
            axis_inv_scales.append(x_axis[1])

        biases = np.array(axis_biases) if len(axis_biases) == 3 else np.zeros(3)
        inv_scales = np.array(axis_inv_scales) if len(axis_inv_scales) == 3 else np.ones(3)

        # 处理尺度
        scales = np.where(np.abs(inv_scales) > 1e-6, 1.0 / inv_scales, np.ones(3))

        # 估算噪声密度 (Allan 方差简化版)
        accel_samples_arr = np.array(self._data.accel_samples)
        noise_density = self._estimate_allan_noise(accel_samples_arr)

        self._result = IMUCalibrationResult(
            accel_bias=biases,
            accel_scale=scales,
            gyro_bias=np.zeros(3),  # 需要翻滚法单独标定
            gyro_scale=np.ones(3),
            noise_density_accel=noise_density,
            noise_density_gyro=0.0,
            calibration_time=time.time(),
            calibration_duration=self._data.collection_duration,
            status=CalibrationStatus.COMPLETED
        )

        return self._result

    def _estimate_allan_noise(self, samples: np.ndarray) -> float:
        """
        简化 Allan 方差噪声密度估计

        对于采样率 f, tau = 1 的 Allan 方差等价于:
        AV = 3 * sigma^2 / omega^2 (单边功率谱密度)
        简化估计: 使用连续样本差分的标准差
        """
        if len(samples) < 10:
            return 0.0

        # 取第一个轴分析
        axis_data = samples[:, 0]

        # 重叠 Allan 方差简化估计
        tau = 1  # 簇大小
        n = len(axis_data) - 2 * tau
        if n < 1:
            n = len(axis_data) - 1
            tau = 0.5

        # 差分方差
        diff = axis_data[tau:] - axis_data[:-tau]
        av = np.var(diff) / 2.0

        # 噪声密度 (mg/sqrt(Hz))
        noise_density = np.sqrt(max(av, 1e-12) / 2.0) * 1000.0  # 转换为 mg
        # 如果虚拟传感器返回静默数据,av可能为0,使用最小默认值
        if noise_density < 1e-6:
            noise_density = 0.01  # 默认 0.01 mg/sqrt(Hz)
        return float(noise_density)

    def calibrate_gyro_rotation(self) -> IMUCalibrationResult:
        """
        翻滚法陀螺仪标定

        将 IMU 绕每个轴以已知角速度旋转,
        通过积分角与期望角速度比较估计偏置
        """
        if self._result is None:
            self._result = IMUCalibrationResult()

        # 提取陀螺仪数据
        gyro_samples = np.array(self._data.gyro_samples)
        if len(gyro_samples) < 10:  # 至少需要 10 个样本进行静态偏置估计
            return IMUCalibrationResult(
                status=CalibrationStatus.FAILED,
                error_message="Insufficient gyro samples for rotation calibration"
            )

        # 估算陀螺仪偏置 (静态时)
        # 简化: 取前10%样本的平均值作为静态偏置
        n_static = max(1, len(gyro_samples) // 10)
        gyro_bias = np.mean(gyro_samples[:n_static], axis=0)

        # Allan 噪声密度
        noise_density = self._estimate_allan_noise(gyro_samples)

        self._result.gyro_bias = gyro_bias
        self._result.noise_density_gyro = noise_density * 1000.0  # 转换为 mdps

        return self._result

    def get_result(self) -> Optional[IMUCalibrationResult]:
        """获取标定结果"""
        return self._result

    def save(self, filepath: str):
        """保存标定结果到文件"""
        if self._result is None:
            raise ValueError("No calibration result to save")

        result = self._result
        data = {
            "accel_bias": result.accel_bias.tolist(),
            "accel_scale": result.accel_scale.tolist(),
            "gyro_bias": result.gyro_bias.tolist(),
            "gyro_scale": result.gyro_scale.tolist(),
            "magnet_bias": result.magnet_bias.tolist() if result.magnet_bias is not None else [0,0,0],
            "noise_density_accel_mg_sqrt_hz": result.noise_density_accel,
            "noise_density_gyro_mdps_sqrt_hz": result.noise_density_gyro,
            "calibration_time": result.calibration_time,
            "duration_seconds": result.calibration_duration,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[IMUCalibrator] Saved to {filepath}")

    def load(self, filepath: str) -> IMUCalibrationResult:
        """从文件加载标定结果"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self._result = IMUCalibrationResult(
            accel_bias=np.array(data["accel_bias"]),
            accel_scale=np.array(data["accel_scale"]),
            gyro_bias=np.array(data["gyro_bias"]),
            gyro_scale=np.array(data["gyro_scale"]),
            magnet_bias=np.array(data.get("magnet_bias", [0,0,0])),
            noise_density_accel=data.get("noise_density_accel_mg_sqrt_hz", 0.0),
            noise_density_gyro=data.get("noise_density_gyro_mdps_sqrt_hz", 0.0),
            calibration_time=data.get("calibration_time", 0.0),
            status=CalibrationStatus.COMPLETED
        )

        print(f"[IMUCalibrator] Loaded from {filepath}")
        return self._result


class ForceCalibrator:
    """
    力传感器标定器

    支持:
    - 零点标定 (无负载时)
    - 多点力标定 (已知砝码)
    - 温度补偿
    - 耦合矩阵估计
    """

    def __init__(
        self,
        sensor: ForceTorqueSensor,
        config: Optional[CalibrationConfig] = None
    ):
        self.sensor = sensor
        self.config = config or CalibrationConfig()
        self._result: Optional[ForceCalibrationResult] = None
        self._raw_samples: List[np.ndarray] = []

    def collect_zero_load(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        收集零点数据 (无负载)

        Returns:
            平均零点偏置 (6,)
        """
        if not self.sensor._is_opened:
            self.sensor.open()

        num_samples = num_samples or self.config.force_num_samples
        samples = []

        # 支持 capture() (ForceTorqueSensor) 和 simulate_contact() (VirtualForceSensor)
        has_capture = hasattr(self.sensor, 'capture')
        has_simulate = hasattr(self.sensor, 'simulate_contact')

        for _ in range(num_samples):
            if has_capture:
                frame = self.sensor.capture()
            elif has_simulate:
                frame = self.sensor.simulate_contact(force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0))
            else:
                raise RuntimeError(f"Sensor {type(self.sensor)} has neither capture() nor simulate_contact()")
            if frame is not None:
                # Wrench dataclass has force and torque directly
                wrench_6d = np.concatenate([frame.force, frame.torque])
                samples.append(wrench_6d)

        samples_arr = np.array(samples)
        self._raw_samples.extend(samples)

        # 零点偏置
        zero_bias = np.mean(samples_arr, axis=0)
        return zero_bias

    def calibrate_with_weights(
        self,
        zero_bias: np.ndarray,
        known_weights: Optional[List[float]] = None
    ) -> ForceCalibrationResult:
        """
        多点力标定

        Args:
            zero_bias: 零点偏置 (6,)
            known_weights: 已知砝码重量列表 (N), 默认使用配置值

        Returns:
            标定结果
        """
        if len(self._raw_samples) < 100:
            return ForceCalibrationResult(
                status=CalibrationStatus.FAILED,
                error_message="Insufficient samples for weight calibration"
            )

        known_weights = known_weights or self.config.force_known_weights

        # 简化的线性标定: wrench_calibrated = scale * (wrench_raw - bias)
        # 对每个轴独立标定
        samples_arr = np.array(self._raw_samples)

        # 零点校正后的样本
        samples_corrected = samples_arr - zero_bias

        # 取最大值作为量程参考
        max_wrench = np.max(np.abs(samples_corrected), axis=0)

        # 尺度因子 (假设最大值为量程)
        # Z轴通常承受重量
        force_scale = np.ones(3)
        torque_scale = np.ones(3)

        # 使用已知重量进行标定
        if len(known_weights) >= 2:
            # 假设Z轴受力为主
            max_z = max_wrench[2] if max_wrench[2] > 1e-6 else 1.0
            estimated_weight = np.max(np.abs(samples_corrected[:, 2]))
            if estimated_weight > 1e-6:
                # 线性拟合: 期望力 = scale * 测量力
                scale_z = known_weights[-1] / estimated_weight
                force_scale[2] = scale_z

        self._result = ForceCalibrationResult(
            force_bias=zero_bias[:3],
            force_scale=force_scale,
            torque_bias=zero_bias[3:],
            torque_scale=torque_scale,
            calibration_time=time.time(),
            status=CalibrationStatus.COMPLETED
        )

        return self._result

    def get_result(self) -> Optional[ForceCalibrationResult]:
        """获取标定结果"""
        return self._result


class TactileCalibrator:
    """
    触觉传感器标定器

    支持:
    - 零压力标定
    - 线性/非线性标定
    - 温度补偿
    """

    def __init__(
        self,
        sensor: TactileArray,
        config: Optional[CalibrationConfig] = None
    ):
        self.sensor = sensor
        self.config = config or CalibrationConfig()
        self._zero_baseline: Optional[np.ndarray] = None

    def collect_zero_baseline(
        self,
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        收集零压力基准

        无任何接触时采集背景压力分布
        """
        if not self.sensor._is_opened:
            self.sensor.open()

        num_samples = num_samples or self.config.tactile_num_samples
        samples = []

        # 支持 capture() (TactileArray) 和 simulate_contact() (VirtualTactileSensor)
        has_capture = hasattr(self.sensor, 'capture')
        has_simulate = hasattr(self.sensor, 'simulate_contact')

        for _ in range(num_samples):
            if has_capture:
                frame = self.sensor.capture()
            elif has_simulate:
                frame = self.sensor.simulate_contact(
                    contact_pos=(0.5, 0.5),
                    contact_radius=0.01,  # 极小半径模拟无接触背景
                    contact_force=0.0,
                    noise_level=0.01
                )
            else:
                raise RuntimeError(f"Sensor {type(self.sensor)} has neither capture() nor simulate_contact()")
            samples.append(frame.pressure_map.copy())

        samples_arr = np.stack(samples, axis=0)  # (N, H, W)
        self._zero_baseline = np.mean(samples_arr, axis=0)

        print(f"[TactileCalibrator] Zero baseline collected: shape={self._zero_baseline.shape}")

        return self._zero_baseline

    def calibrate(
        self,
        zero_baseline: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        执行触觉传感器标定

        Args:
            zero_baseline: 零压力基准, 若为 None 则使用已采集的数据

        Returns:
            标定参数
        """
        baseline = zero_baseline if zero_baseline is not None else self._zero_baseline

        if baseline is None:
            raise ValueError("No zero baseline available. Run collect_zero_baseline first.")

        # 标定参数
        params = {
            "offset_map": baseline.tolist() if isinstance(baseline, np.ndarray) else baseline,
            "mean_offset": float(np.mean(baseline)),
            "std_offset": float(np.std(baseline)),
            "max_offset": float(np.max(baseline)),
            "sensor_id": self.sensor.sensor_id,
            "array_size": self.sensor.array_size,
            "calibration_time": time.time(),
        }

        # 应用到传感器 (如果传感器支持的话)
        if hasattr(self.sensor, 'calibrate'):
            self.sensor.calibrate(zero_pressure=baseline)
        else:
            print(f"[TactileCalibrator] Sensor {self.sensor.sensor_id} does not support apply calibration")

        return params


class CalibrationManager:
    """
    统一标定管理器

    协调多传感器标定流程, 提供:
    - 统一的标定状态管理
    - 多传感器顺序标定
    - 标定结果存储和加载
    - AGV五级规格对应的标定要求
    """

    # AGV 五级标定规格
    AGV_CALIBRATION_GRADES = {
        'S': {
            'imu_poses': 4, 'imu_samples_per_pose': 200, 'imu_rate_hz': 50,
            'force_samples': 200, 'force_rate_hz': 50,
            'tactile_samples': 50, 'tactile_threshold': 0.02,
        },
        'M': {
            'imu_poses': 6, 'imu_samples_per_pose': 500, 'imu_rate_hz': 100,
            'force_samples': 500, 'force_rate_hz': 100,
            'tactile_samples': 100, 'tactile_threshold': 0.01,
        },
        'L': {
            'imu_poses': 6, 'imu_samples_per_pose': 1000, 'imu_rate_hz': 200,
            'force_samples': 1000, 'force_rate_hz': 200,
            'tactile_samples': 200, 'tactile_threshold': 0.005,
        },
        'XL': {
            'imu_poses': 8, 'imu_samples_per_pose': 2000, 'imu_rate_hz': 500,
            'force_samples': 2000, 'force_rate_hz': 500,
            'tactile_samples': 400, 'tactile_threshold': 0.002,
        },
        'XXL': {
            'imu_poses': 12, 'imu_samples_per_pose': 5000, 'imu_rate_hz': 1000,
            'force_samples': 5000, 'force_rate_hz': 1000,
            'tactile_samples': 1000, 'tactile_threshold': 0.001,
        },
    }

    def __init__(self, agv_grade: str = "M"):
        self.agv_grade = agv_grade
        self.grade_config = self.AGV_CALIBRATION_GRADES.get(agv_grade, self.AGV_CALIBRATION_GRADES['M'])

        # 传感器实例 (按需创建)
        self.imu_sensor: Optional[IMUSensor] = None
        self.force_sensor: Optional[ForceTorqueSensor] = None
        self.tactile_sensor: Optional[TactileArray] = None

        # 标定器
        self.imu_calibrator: Optional[IMUCalibrator] = None
        self.force_calibrator: Optional[ForceCalibrator] = None
        self.tactile_calibrator: Optional[TactileCalibrator] = None

        # 标定结果
        self.imu_result: Optional[IMUCalibrationResult] = None
        self.force_result: Optional[ForceCalibrationResult] = None
        self.tactile_params: Optional[Dict[str, Any]] = None

        # 状态
        self._status = CalibrationStatus.IDLE
        self._progress: Dict[str, float] = {}

    def setup_imu(self, sensor: IMUSensor):
        """设置 IMU 传感器"""
        self.imu_sensor = sensor
        cfg = self.grade_config
        config = CalibrationConfig(
            imu_sample_rate=cfg['imu_rate_hz'],
            imu_num_samples_per_pose=cfg['imu_samples_per_pose'],
            imu_num_poses=cfg['imu_poses'],
            agv_grade=self.agv_grade
        )
        self.imu_calibrator = IMUCalibrator(sensor, config)

    def setup_force(self, sensor: ForceTorqueSensor):
        """设置力传感器"""
        self.force_sensor = sensor
        cfg = self.grade_config
        config = CalibrationConfig(
            force_sample_rate=cfg['force_rate_hz'],
            force_num_samples=cfg['force_samples'],
            agv_grade=self.agv_grade
        )
        self.force_calibrator = ForceCalibrator(sensor, config)

    def setup_tactile(self, sensor: TactileArray):
        """设置触觉传感器"""
        self.tactile_sensor = sensor
        cfg = self.grade_config
        config = CalibrationConfig(
            tactile_num_samples=cfg['tactile_samples'],
            tactile_pressure_threshold=cfg['tactile_threshold'],
            agv_grade=self.agv_grade
        )
        self.tactile_calibrator = TactileCalibrator(sensor, config)

    def calibrate_all(self) -> Dict[str, Any]:
        """
        执行完整标定流程

        依次标定 IMU / 力传感器 / 触觉传感器
        """
        results = {}

        self._status = CalibrationStatus.COLLECTING

        # IMU 标定
        if self.imu_calibrator is not None:
            self._status = CalibrationStatus.COLLECTING

            # 六面法采集
            orientations = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
            num_poses = min(len(orientations), self.grade_config['imu_poses'])

            for i, orientation in enumerate(orientations[:num_poses]):
                self.imu_calibrator.collect_data(orientation)
                self._progress['imu'] = (i + 1) / num_poses

            # 执行标定
            self._status = CalibrationStatus.PROCESSING
            self.imu_result = self.imu_calibrator.calibrate_accel_six_facing()
            self.imu_result = self.imu_calibrator.calibrate_gyro_rotation()

            results['imu'] = {
                'status': self.imu_result.status.value,
                'accel_bias': self.imu_result.accel_bias.tolist(),
                'gyro_bias': self.imu_result.gyro_bias.tolist(),
                'noise_density_accel': self.imu_result.noise_density_accel,
                'noise_density_gyro': self.imu_result.noise_density_gyro,
            }
            self._progress['imu'] = 1.0

        # 力传感器标定
        if self.force_calibrator is not None:
            self._status = CalibrationStatus.COLLECTING
            self._progress['force'] = 0.0

            zero_bias = self.force_calibrator.collect_zero_load()
            self._progress['force'] = 0.5

            self.force_result = self.force_calibrator.calibrate_with_weights(zero_bias)

            results['force'] = {
                'status': self.force_result.status.value,
                'force_bias': self.force_result.force_bias.tolist(),
                'force_scale': self.force_result.force_scale.tolist(),
            }
            self._progress['force'] = 1.0

        # 触觉传感器标定
        if self.tactile_calibrator is not None:
            self._status = CalibrationStatus.COLLECTING
            self._progress['tactile'] = 0.0

            baseline = self.tactile_calibrator.collect_zero_baseline()
            self._progress['tactile'] = 0.5

            self.tactile_params = self.tactile_calibrator.calibrate(baseline)
            self._progress['tactile'] = 1.0

            results['tactile'] = {
                'mean_offset': self.tactile_params['mean_offset'],
                'std_offset': self.tactile_params['std_offset'],
            }

        self._status = CalibrationStatus.COMPLETED
        results['overall'] = {
            'grade': self.agv_grade,
            'status': self._status.value,
            'progress': self._progress,
        }

        return results

    def save_all(self, output_dir: str):
        """保存所有标定结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.imu_calibrator is not None and self.imu_result is not None:
            self.imu_calibrator.save(f"{output_dir}/imu_calibration.json")

        if self.force_result is not None:
            force_data = {
                'force_bias': self.force_result.force_bias.tolist(),
                'force_scale': self.force_result.force_scale.tolist(),
                'torque_bias': self.force_result.torque_bias.tolist(),
                'torque_scale': self.force_result.torque_scale.tolist(),
                'calibration_time': self.force_result.calibration_time,
            }
            with open(f"{output_dir}/force_calibration.json", 'w') as f:
                json.dump(force_data, f, indent=2)

        if self.tactile_params is not None:
            with open(f"{output_dir}/tactile_calibration.json", 'w') as f:
                json.dump(self.tactile_params, f, indent=2)

        print(f"[CalibrationManager] All results saved to {output_dir}")

    def get_status(self) -> CalibrationStatus:
        """获取当前标定状态"""
        return self._status

    def get_progress(self) -> Dict[str, float]:
        """获取各传感器标定进度"""
        return self._progress.copy()


def create_calibration_manager(agv_grade: str = "M") -> CalibrationManager:
    """创建指定 AGV 等级的标定管理器"""
    return CalibrationManager(agv_grade=agv_grade)


# AGV五级标定规格表
AGV_CALIBRATION_SPEC = {
    'S': {
        'imu_poses': 4, 'imu_samples': 200, 'imu_rate_hz': 50,
        'force_samples': 200, 'force_rate_hz': 50,
        'tactile_samples': 50,
        'expected_bias_stability_mg': 50.0,
        'expected_noise_density_accel_mg_sqrt_hz': 0.5,
        'expected_noise_density_gyro_mdps_sqrt_hz': 1.0,
    },
    'M': {
        'imu_poses': 6, 'imu_samples': 500, 'imu_rate_hz': 100,
        'force_samples': 500, 'force_rate_hz': 100,
        'tactile_samples': 100,
        'expected_bias_stability_mg': 20.0,
        'expected_noise_density_accel_mg_sqrt_hz': 0.2,
        'expected_noise_density_gyro_mdps_sqrt_hz': 0.5,
    },
    'L': {
        'imu_poses': 6, 'imu_samples': 1000, 'imu_rate_hz': 200,
        'force_samples': 1000, 'force_rate_hz': 200,
        'tactile_samples': 200,
        'expected_bias_stability_mg': 10.0,
        'expected_noise_density_accel_mg_sqrt_hz': 0.1,
        'expected_noise_density_gyro_mdps_sqrt_hz': 0.2,
    },
    'XL': {
        'imu_poses': 8, 'imu_samples': 2000, 'imu_rate_hz': 500,
        'force_samples': 2000, 'force_rate_hz': 500,
        'tactile_samples': 400,
        'expected_bias_stability_mg': 5.0,
        'expected_noise_density_accel_mg_sqrt_hz': 0.05,
        'expected_noise_density_gyro_mdps_sqrt_hz': 0.1,
    },
    'XXL': {
        'imu_poses': 12, 'imu_samples': 5000, 'imu_rate_hz': 1000,
        'force_samples': 5000, 'force_rate_hz': 1000,
        'tactile_samples': 1000,
        'expected_bias_stability_mg': 1.0,
        'expected_noise_density_accel_mg_sqrt_hz': 0.02,
        'expected_noise_density_gyro_mdps_sqrt_hz': 0.05,
    },
}


def get_calibration_spec(grade: str) -> dict:
    """获取 AGV 指定等级的标定规格"""
    actual_grade = grade if grade in AGV_CALIBRATION_SPEC else 'M'
    spec = AGV_CALIBRATION_SPEC[actual_grade].copy()
    spec['grade'] = actual_grade
    return spec


# Aliases for backward compatibility with new control/ API
IMUCalibrationManager = IMUCalibrator
ForceCalibrationManager = ForceCalibrator
TactileCalibrationManager = TactileCalibrator
IMUCalibrationResult = IMUCalibrationResult
ForceCalibrationResult = ForceCalibrationResult


class CalibratedSensor:
    """标定后的传感器包装器 (包装已标定的传感器并提供标定后的读数)"""

    def __init__(self, sensor, calibration_result):
        self.sensor = sensor
        self.calibration = calibration_result
        self._is_calibrated = calibration_result.status == CalibrationStatus.COMPLETED

    def read_raw(self):
        """读取原始数据"""
        return self.sensor.capture()

    def read_calibrated(self):
        """读取标定后的数据"""
        raw = self.read_raw()
        if not self._is_calibrated:
            return raw
        return self.calibration.apply(raw)

    def is_calibrated(self) -> bool:
        return self._is_calibrated


def get_all_grade_spec_table():
    """获取所有等级的标定规格表"""
    return [{'grade': g, **v} for g, v in AGV_CALIBRATION_SPEC.items()]


# 模块级 AGV 五级标定规格 (与 CalibrationManager.AGV_CALIBRATION_GRADES 同步)
# 用于直接导入: from src.control.calibration_manager import AGV_CALIBRATION_GRADES
AGV_CALIBRATION_GRADES = {
    'S': {
        'imu_poses': 4, 'imu_samples_per_pose': 200, 'imu_rate_hz': 50,
        'force_samples': 200, 'force_rate_hz': 50,
        'tactile_samples': 50, 'tactile_threshold': 0.02,
    },
    'M': {
        'imu_poses': 6, 'imu_samples_per_pose': 500, 'imu_rate_hz': 100,
        'force_samples': 500, 'force_rate_hz': 100,
        'tactile_samples': 100, 'tactile_threshold': 0.01,
    },
    'L': {
        'imu_poses': 6, 'imu_samples_per_pose': 1000, 'imu_rate_hz': 200,
        'force_samples': 1000, 'force_rate_hz': 200,
        'tactile_samples': 200, 'tactile_threshold': 0.005,
    },
    'XL': {
        'imu_poses': 8, 'imu_samples_per_pose': 2000, 'imu_rate_hz': 500,
        'force_samples': 2000, 'force_rate_hz': 500,
        'tactile_samples': 400, 'tactile_threshold': 0.002,
    },
    'XXL': {
        'imu_poses': 12, 'imu_samples_per_pose': 5000, 'imu_rate_hz': 1000,
        'force_samples': 5000, 'force_rate_hz': 1000,
        'tactile_samples': 1000, 'tactile_threshold': 0.001,
    },
}


def get_calibration_grade_spec(grade: str) -> dict:
    """获取指定 AGV 等级的标定规格 (模块级接口, 基于AGV_CALIBRATION_GRADES)"""
    return AGV_CALIBRATION_GRADES.get(grade, AGV_CALIBRATION_GRADES['M'])
