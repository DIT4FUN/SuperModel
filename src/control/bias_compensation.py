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
传感器偏置补偿控制模块
====================

实时传感器偏置估计与补偿
- IMU加速度计/陀螺仪偏置在线估计
- 力传感器偏置与漂移补偿
- 触觉传感器零点漂移校正
- 多传感器联合偏置估计

AGV五级支持: S/M/L/XL/XXL

依赖模块:
- src/sensors.imu.IMUFrame
- src/sensors.force.Wrench  
- src/sensors.tactile.TactileFrame
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import time


class BiasType(Enum):
    """偏置类型"""
    CONSTANT = "constant"        # 常值偏置
    LINEAR_DRIFT = "linear_drift"  # 线性漂移
    TEMPERATURE_DEPENDENT = "temp_dependent"  # 温度相关
    STOCHASTIC = "stochastic"    # 随机游走


@dataclass
class IMUBiasState:
    """IMU偏置状态"""
    accel_bias: np.ndarray       # 3, 加速度计偏置 (m/s^2)
    gyro_bias: np.ndarray         # 3, 陀螺仪偏置 (rad/s)
    accel_bias_std: np.ndarray   # 3, 偏置标准差
    gyro_bias_std: np.ndarray     # 3, 偏置标准差
    temperature: float = 25.0     # 当前温度
    timestamp: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.accel_bias, list):
            self.accel_bias = np.array(self.accel_bias, dtype=np.float32)
        if isinstance(self.gyro_bias, list):
            self.gyro_bias = np.array(self.gyro_bias, dtype=np.float32)
        if isinstance(self.accel_bias_std, list):
            self.accel_bias_std = np.array(self.accel_bias_std, dtype=np.float32)
        if isinstance(self.gyro_bias_std, list):
            self.gyro_bias_std = np.array(self.gyro_bias_std, dtype=np.float32)


@dataclass
class ForceBiasState:
    """力传感器偏置状态"""
    force_bias: np.ndarray       # 3, 力偏置 (N)
    torque_bias: np.ndarray       # 3, 力矩偏置 (N·m)
    force_bias_std: np.ndarray   # 3, 偏置标准差
    torque_bias_std: np.ndarray   # 3, 偏置标准差
    drift_rate: np.ndarray        # 6, 漂移率 (N/s, N·m/s)
    timestamp: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.force_bias, list):
            self.force_bias = np.array(self.force_bias, dtype=np.float32)
        if isinstance(self.torque_bias, list):
            self.torque_bias = np.array(self.torque_bias, dtype=np.float32)


@dataclass
class TactileBiasState:
    """触觉传感器偏置状态"""
    pressure_offset: np.ndarray   # HxW, 压力偏置
    temperature_offset: Optional[np.ndarray] = None  # HxW, 温度偏置
    timestamp: float = 0.0


@dataclass
class BiasCompensationConfig:
    """偏置补偿配置"""
    # IMU
    accel_bias_limit: float = 0.5        # m/s^2
    gyro_bias_limit: float = 0.1          # rad/s
    accel_noise_std: float = 0.01       # m/s^2
    gyro_noise_std: float = 0.001       # rad/s
    # Force
    force_bias_limit: float = 10.0      # N
    torque_bias_limit: float = 1.0      # N·m
    drift_limit: float = 0.001          # N/s
    # Tactile
    tactile_offset_limit: float = 0.1   # normalized
    # Adaptive estimation
    adaptation_rate: float = 0.01       # 偏置估计学习率
    stationary_window_s: float = 2.0     # 静止检测窗口 (秒)
    stationary_accel_thresh: float = 0.05  # 静止加速度阈值 (m/s^2)
    stationary_gyro_thresh: float = 0.02  # 静止角速度阈值 (rad/s)
    # Control
    enable_imu: bool = True
    enable_force: bool = True
    enable_tactile: bool = True
    grade: str = "M"                     # AGV等级


# === AGV五级规格 ===
AGV_BIAS_COMPENSATION_GRADES = {
    "S": BiasCompensationConfig(
        adaptation_rate=0.005,
        stationary_window_s=3.0,
        grade="S"
    ),
    "M": BiasCompensationConfig(
        adaptation_rate=0.01,
        stationary_window_s=2.0,
        grade="M"
    ),
    "L": BiasCompensationConfig(
        adaptation_rate=0.02,
        stationary_window_s=1.5,
        grade="L"
    ),
    "XL": BiasCompensationConfig(
        adaptation_rate=0.05,
        stationary_window_s=1.0,
        grade="XL"
    ),
    "XXL": BiasCompensationConfig(
        adaptation_rate=0.1,
        stationary_window_s=0.5,
        grade="XXL"
    ),
}


def get_bias_compensation_spec(grade: str) -> BiasCompensationConfig:
    """获取AGV五级偏置补偿规格"""
    return AGV_BIAS_COMPENSATION_GRADES.get(grade, AGV_BIAS_COMPENSATION_GRADES["M"])


class IMUBiasEstimator:
    """
    IMU偏置在线估计器
    
    使用静止检测 + 自适应滤波估计加速度计和陀螺仪偏置
    当检测到静止时,累积观测进行偏置估计
    """
    
    def __init__(self, config: Optional[BiasCompensationConfig] = None):
        self.config = config or get_bias_compensation_spec("M")
        self.reset()
    
    def reset(self):
        """重置估计器状态"""
        self.accel_bias = np.zeros(3, dtype=np.float32)
        self.gyro_bias = np.zeros(3, dtype=np.float32)
        self.accel_bias_var = np.ones(3, dtype=np.float32) * 0.1
        self.gyro_bias_var = np.ones(3, dtype=np.float32) * 0.01
        self.accel_history: List[np.ndarray] = []
        self.gyro_history: List[np.ndarray] = []
        self.last_update_time = time.time()
        self.stationary_count = 0
        self._is_stationary = False
    
    def is_stationary(self, accel: np.ndarray, gyro: np.ndarray) -> bool:
        """静止检测"""
        accel_mag = np.linalg.norm(accel - self.accel_bias)
        gyro_mag = np.linalg.norm(gyro - self.gyro_bias)
        g = 9.81
        accel_deviation = abs(accel_mag - g)
        return (accel_deviation < self.config.stationary_accel_thresh and
                gyro_mag < self.config.stationary_gyro_thresh)
    
    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float,
        temperature: float = 25.0
    ) -> IMUBiasState:
        """
        更新偏置估计
        
        Args:
            accel: 加速度计读数 (3,)
            gyro: 陀螺仪读数 (3,)
            dt: 时间步长 (秒)
            temperature: 当前温度
            
        Returns:
            偏置状态
        """
        accel = np.asarray(accel, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        
        stationary = self.is_stationary(accel, gyro)
        self._is_stationary = stationary
        
        if stationary:
            self.stationary_count += 1
            self.accel_history.append(accel.copy())
            self.gyro_history.append(gyro.copy())
            
            window_size = int(self.config.stationary_window_s / dt)
            if len(self.accel_history) > window_size:
                self.accel_history.pop(0)
                self.gyro_history.pop(0)
            
            if len(self.accel_history) >= 10:
                accel_mean = np.mean(self.accel_history, axis=0)
                gyro_mean = np.mean(self.gyro_history, axis=0)
                g = np.array([0.0, 0.0, 9.81], dtype=np.float32)
                self.accel_bias = accel_mean - g
                self.gyro_bias = gyro_mean
                
                accel_var = np.var(self.accel_history, axis=0)
                gyro_var = np.var(self.gyro_history, axis=0)
                rate = self.config.adaptation_rate
                self.accel_bias_var = (1 - rate) * self.accel_bias_var + rate * accel_var
                self.gyro_bias_var = (1 - rate) * self.gyro_bias_var + rate * gyro_var
        
        self.accel_bias = np.clip(
            self.accel_bias,
            -self.config.accel_bias_limit,
            self.config.accel_bias_limit
        )
        self.gyro_bias = np.clip(
            self.gyro_bias,
            -self.config.gyro_bias_limit,
            self.config.gyro_bias_limit
        )
        
        self.last_update_time = time.time()
        
        return IMUBiasState(
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias_std=np.sqrt(self.accel_bias_var),
            gyro_bias_std=np.sqrt(self.gyro_bias_var),
            temperature=temperature,
            timestamp=self.last_update_time
        )
    
    def compensate(self, accel: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        补偿IMU读数
        
        Args:
            accel: 原始加速度 (3,)
            gyro: 原始角速度 (3,)
            
        Returns:
            (补偿后加速度, 补偿后角速度)
        """
        accel = np.asarray(accel, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        return accel - self.accel_bias, gyro - self.gyro_bias
    
    def get_state(self) -> IMUBiasState:
        """获取当前偏置状态"""
        return IMUBiasState(
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias_std=np.sqrt(self.accel_bias_var),
            gyro_bias_std=np.sqrt(self.gyro_bias_var),
            timestamp=self.last_update_time
        )


class ForceBiasEstimator:
    """
    力传感器偏置估计器
    
    支持: 静态偏置估计 + 漂移补偿 + 温度补偿
    """
    
    def __init__(self, config: Optional[BiasCompensationConfig] = None):
        self.config = config or get_bias_compensation_spec("M")
        self.reset()
    
    def reset(self):
        """重置估计器"""
        self.force_bias = np.zeros(3, dtype=np.float32)
        self.torque_bias = np.zeros(3, dtype=np.float32)
        self.force_bias_std = np.ones(3, dtype=np.float32) * 1.0
        self.torque_bias_std = np.ones(3, dtype=np.float32) * 0.1
        self.force_drift = np.zeros(3, dtype=np.float32)
        self.torque_drift = np.zeros(3, dtype=np.float32)
        self.last_update_time = time.time()
        self.last_force = np.zeros(3, dtype=np.float32)
        self.last_torque = np.zeros(3, dtype=np.float32)
        self.history: List[Tuple[float, np.ndarray, np.ndarray]] = []
    
    def calibrate(self, wrench_history: List[Tuple[float, np.ndarray, np.ndarray]]):
        """
        离线校准: 基于历史数据计算偏置
        
        Args:
            wrench_history: List of (timestamp, force, torque)
        """
        forces = np.array([w[1] for w in wrench_history])
        torques = np.array([w[2] for w in wrench_history])
        
        self.force_bias = -np.mean(forces, axis=0)
        self.torque_bias = -np.mean(torques, axis=0)
        
        self.force_bias_std = np.std(forces, axis=0)
        self.torque_bias_std = np.std(torques, axis=0)
        
        if len(wrench_history) > 1:
            dt = wrench_history[-1][0] - wrench_history[0][0]
            if dt > 0:
                self.force_drift = (forces[-1] - forces[0]) / dt
                self.torque_drift = (torques[-1] - torques[0]) / dt
    
    def update(
        self,
        force: np.ndarray,
        torque: np.ndarray,
        dt: float
    ) -> ForceBiasState:
        """
        更新偏置估计
        
        Args:
            force: 力读数 (3,)
            torque: 力矩读数 (3,)
            dt: 时间步长 (秒)
            
        Returns:
            偏置状态
        """
        force = np.asarray(force, dtype=np.float32)
        torque = np.asarray(torque, dtype=np.float32)
        
        force_diff = force - self.last_force
        torque_diff = torque - self.last_torque
        
        self.force_drift += 0.001 * (force_diff / dt - self.force_drift)
        self.torque_drift += 0.001 * (torque_diff / dt - self.torque_drift)
        
        self.force_drift = np.clip(
            self.force_drift, -self.config.drift_limit, self.config.drift_limit
        )
        self.torque_drift = np.clip(
            self.torque_drift, -self.config.drift_limit, self.config.drift_limit
        )
        
        rate = self.config.adaptation_rate
        self.force_bias += rate * (force + self.force_bias) * dt
        self.torque_bias += rate * (torque + self.torque_bias) * dt
        
        self.force_bias = np.clip(
            self.force_bias,
            -self.config.force_bias_limit,
            self.config.force_bias_limit
        )
        self.torque_bias = np.clip(
            self.torque_bias,
            -self.config.torque_bias_limit,
            self.config.torque_bias_limit
        )
        
        self.last_force = force.copy()
        self.last_torque = torque.copy()
        current_time = time.time()
        
        return ForceBiasState(
            force_bias=self.force_bias.copy(),
            torque_bias=self.torque_bias.copy(),
            force_bias_std=self.force_bias_std.copy(),
            torque_bias_std=self.torque_bias_std.copy(),
            drift_rate=np.concatenate([self.force_drift, self.torque_drift]),
            timestamp=current_time
        )
    
    def compensate(self, force: np.ndarray, torque: np.ndarray, dt: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        补偿力传感器读数
        
        Args:
            force: 原始力 (3,)
            torque: 原始力矩 (3,)
            dt: 时间步长 (秒)
            
        Returns:
            (补偿后力, 补偿后力矩)
        """
        force = np.asarray(force, dtype=np.float32)
        torque = np.asarray(torque, dtype=np.float32)
        
        compensated_force = force - self.force_bias - self.force_drift * dt
        compensated_torque = torque - self.torque_bias - self.torque_drift * dt
        return compensated_force, compensated_torque
    
    def get_state(self) -> ForceBiasState:
        """获取当前偏置状态"""
        return ForceBiasState(
            force_bias=self.force_bias.copy(),
            torque_bias=self.torque_bias.copy(),
            force_bias_std=self.force_bias_std.copy(),
            torque_bias_std=self.torque_bias_std.copy(),
            drift_rate=np.concatenate([self.force_drift, self.torque_drift]),
            timestamp=time.time()
        )


class TactileBiasEstimator:
    """
    触觉传感器偏置估计器
    
    用于补偿传感器零点漂移和温度漂移
    """
    
    def __init__(self, array_size: Tuple[int, int] = (16, 16),
                 config: Optional[BiasCompensationConfig] = None):
        self.array_size = array_size
        self.config = config or get_bias_compensation_spec("M")
        self.reset()
    
    def reset(self):
        """重置估计器"""
        self.pressure_offset = np.zeros(self.array_size, dtype=np.float32)
        self.temperature_offset = np.zeros(self.array_size, dtype=np.float32) if self.config.enable_tactile else None
        self.var = np.ones(self.array_size, dtype=np.float32) * 0.01
        self.last_update_time = time.time()
        self._calibrated = False
    
    def calibrate(self, pressure_map: np.ndarray, temperature_map: Optional[np.ndarray] = None):
        """
        离线校准: 基于无接触时的传感器读数计算偏置
        
        Args:
            pressure_map: 无接触时的压力分布 (HxW)
            temperature_map: 无接触时的温度分布 (HxW)
        """
        self.pressure_offset = pressure_map.copy()
        if temperature_map is not None:
            self.temperature_offset = temperature_map.copy()
        self._calibrated = True
    
    def update(
        self,
        pressure_map: np.ndarray,
        temperature_map: Optional[np.ndarray] = None
    ) -> TactileBiasState:
        """
        在线更新偏置估计
        
        Args:
            pressure_map: 压力分布 (HxW)
            temperature_map: 温度分布 (HxW)
            
        Returns:
            偏置状态
        """
        rate = self.config.adaptation_rate
        
        if not self._calibrated:
            self.pressure_offset = np.zeros_like(pressure_map)
            self._calibrated = True
        
        self.pressure_offset += rate * (pressure_map - self.pressure_offset)
        self.pressure_offset = np.clip(
            self.pressure_offset,
            -self.config.tactile_offset_limit,
            self.config.tactile_offset_limit
        )
        
        if temperature_map is not None and self.temperature_offset is not None:
            self.temperature_offset += rate * (temperature_map - self.temperature_offset)
        
        self.last_update_time = time.time()
        
        return TactileBiasState(
            pressure_offset=self.pressure_offset.copy(),
            temperature_offset=self.temperature_offset.copy() if self.temperature_offset is not None else None,
            timestamp=self.last_update_time
        )
    
    def compensate(self, pressure_map: np.ndarray, temperature_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        补偿触觉传感器读数
        
        Args:
            pressure_map: 原始压力分布 (HxW)
            temperature_map: 原始温度分布 (HxW)
            
        Returns:
            补偿后压力分布
        """
        return pressure_map - self.pressure_offset
    
    def get_state(self) -> TactileBiasState:
        """获取当前偏置状态"""
        return TactileBiasState(
            pressure_offset=self.pressure_offset.copy(),
            temperature_offset=self.temperature_offset.copy() if self.temperature_offset is not None else None,
            timestamp=self.last_update_time
        )


class MultiSensorBiasCompensator:
    """
    多传感器联合偏置补偿器
    
    整合IMU/力/触觉偏置估计,统一接口
    支持AGV五级配置
    """
    
    def __init__(self, grade: str = "M"):
        self.grade = grade
        self.config = get_bias_compensation_spec(grade)
        
        self.imu_estimator = IMUBiasEstimator(self.config) if self.config.enable_imu else None
        self.force_estimator = ForceBiasEstimator(self.config) if self.config.enable_force else None
        self.tactile_estimator: Optional[TactileBiasEstimator] = None
        
        self._initialized = False
        self.compensation_count = 0
        self.total_bias_magnitude = 0.0
    
    def initialize_tactile(self, array_size: Tuple[int, int]):
        """初始化触觉偏置估计器"""
        self.tactile_estimator = TactileBiasEstimator(array_size, self.config)
    
    def reset(self):
        """重置所有估计器"""
        if self.imu_estimator:
            self.imu_estimator.reset()
        if self.force_estimator:
            self.force_estimator.reset()
        if self.tactile_estimator:
            self.tactile_estimator.reset()
        self._initialized = False
        self.compensation_count = 0
        self.total_bias_magnitude = 0.0
    
    def update_imu(self, accel: np.ndarray, gyro: np.ndarray, dt: float,
                   temperature: float = 25.0) -> IMUBiasState:
        """更新IMU偏置"""
        if self.imu_estimator is None:
            raise RuntimeError("IMU estimator not initialized")
        return self.imu_estimator.update(accel, gyro, dt, temperature)
    
    def update_force(self, force: np.ndarray, torque: np.ndarray,
                    dt: float) -> ForceBiasState:
        """更新力传感器偏置"""
        if self.force_estimator is None:
            raise RuntimeError("Force estimator not initialized")
        return self.force_estimator.update(force, torque, dt)
    
    def update_tactile(self, pressure_map: np.ndarray,
                       temperature_map: Optional[np.ndarray] = None) -> TactileBiasState:
        """更新触觉传感器偏置"""
        if self.tactile_estimator is None:
            raise RuntimeError("Tactile estimator not initialized")
        return self.tactile_estimator.update(pressure_map, temperature_map)
    
    def compensate_imu(self, accel: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """补偿IMU读数"""
        if self.imu_estimator is None:
            return accel, gyro
        return self.imu_estimator.compensate(accel, gyro)
    
    def compensate_force(self, force: np.ndarray, torque: np.ndarray,
                        dt: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """补偿力传感器读数"""
        if self.force_estimator is None:
            return force, torque
        return self.force_estimator.compensate(force, torque, dt)
    
    def compensate_tactile(self, pressure_map: np.ndarray,
                           temperature_map: Optional[np.ndarray] = None) -> np.ndarray:
        """补偿触觉传感器读数"""
        if self.tactile_estimator is None:
            return pressure_map
        return self.tactile_estimator.compensate(pressure_map, temperature_map)
    
    def step(self, dt: float = 0.01) -> Dict[str, float]:
        """
        执行一步偏置补偿统计
        
        Returns:
            统计信息字典
        """
        imu_bias_mag = 0.0
        force_bias_mag = 0.0
        tactile_bias_mag = 0.0
        
        if self.imu_estimator:
            s = self.imu_estimator.get_state()
            imu_bias_mag = float(np.linalg.norm(s.accel_bias) + np.linalg.norm(s.gyro_bias))
        
        if self.force_estimator:
            s = self.force_estimator.get_state()
            force_bias_mag = float(np.linalg.norm(s.force_bias) + np.linalg.norm(s.torque_bias))
        
        if self.tactile_estimator:
            s = self.tactile_estimator.get_state()
            tactile_bias_mag = float(np.linalg.norm(s.pressure_offset))
        
        self.total_bias_magnitude += imu_bias_mag + force_bias_mag + tactile_bias_mag
        self.compensation_count += 1
        
        return {
            "imu_bias_mag": imu_bias_mag,
            "force_bias_mag": force_bias_mag,
            "tactile_bias_mag": tactile_bias_mag,
            "total_bias_mag": imu_bias_mag + force_bias_mag + tactile_bias_mag,
            "avg_bias_mag": self.total_bias_magnitude / self.compensation_count if self.compensation_count > 0 else 0.0,
            "grade": self.grade,
        }
    
    def get_summary(self) -> Dict[str, any]:
        """获取补偿器状态摘要"""
        summary: Dict[str, any] = {"grade": self.grade, "compensations": self.compensation_count}
        
        if self.imu_estimator:
            s = self.imu_estimator.get_state()
            summary["imu"] = {
                "accel_bias": s.accel_bias.tolist(),
                "gyro_bias": s.gyro_bias.tolist(),
                "accel_bias_std": s.accel_bias_std.tolist(),
                "gyro_bias_std": s.gyro_bias_std.tolist(),
            }
        
        if self.force_estimator:
            s = self.force_estimator.get_state()
            summary["force"] = {
                "force_bias": s.force_bias.tolist(),
                "torque_bias": s.torque_bias.tolist(),
                "drift_rate": s.drift_rate.tolist(),
            }
        
        if self.tactile_estimator:
            s = self.tactile_estimator.get_state()
            summary["tactile"] = {
                "pressure_offset_shape": s.pressure_offset.shape,
                "offset_mean": float(np.mean(s.pressure_offset)),
            }
        
        return summary


# === AGV五级偏置补偿规格表 ===
def get_agv_bias_spec_table() -> Dict:
    """
    获取AGV五级偏置补偿规格表
    
    Returns:
        五级规格对照表
    """
    return {
        "S": {
            "accel_bias_limit": 0.5, "gyro_bias_limit": 0.1,
            "force_bias_limit": 10.0, "torque_bias_limit": 1.0,
            "adaptation_rate": 0.005, "stationary_window_s": 3.0,
            "control_freq_hz": 50,
        },
        "M": {
            "accel_bias_limit": 0.3, "gyro_bias_limit": 0.05,
            "force_bias_limit": 5.0, "torque_bias_limit": 0.5,
            "adaptation_rate": 0.01, "stationary_window_s": 2.0,
            "control_freq_hz": 100,
        },
        "L": {
            "accel_bias_limit": 0.2, "gyro_bias_limit": 0.02,
            "force_bias_limit": 2.0, "torque_bias_limit": 0.2,
            "adaptation_rate": 0.02, "stationary_window_s": 1.5,
            "control_freq_hz": 200,
        },
        "XL": {
            "accel_bias_limit": 0.1, "gyro_bias_limit": 0.01,
            "force_bias_limit": 1.0, "torque_bias_limit": 0.1,
            "adaptation_rate": 0.05, "stationary_window_s": 1.0,
            "control_freq_hz": 500,
        },
        "XXL": {
            "accel_bias_limit": 0.05, "gyro_bias_limit": 0.005,
            "force_bias_limit": 0.5, "torque_bias_limit": 0.05,
            "adaptation_rate": 0.1, "stationary_window_s": 0.5,
            "control_freq_hz": 1000,
        },
    }


# === 导出符号 ===
__all__ = [
    'BiasType', 'IMUBiasState', 'ForceBiasState', 'TactileBiasState',
    'BiasCompensationConfig', 'AGV_BIAS_COMPENSATION_GRADES',
    'get_bias_compensation_spec', 'IMUBiasEstimator', 'ForceBiasEstimator',
    'TactileBiasEstimator', 'MultiSensorBiasCompensator',
    'get_agv_bias_spec_table',
]
