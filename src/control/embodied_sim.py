"""
具身智能仿真环境
================

完整的闭环具身智能仿真系统

功能:
- 物理环境仿真 (位置/速度/姿态/接触力)
- 传感器仿真 (触觉/力觉/IMU/编码器)
- 传感器噪声与延迟建模
- 具身智能控制器闭环仿真
- Gymnasium 标准接口
- AGV五级规格适配

使用示例:
    from src.control.embodied_sim import EmbodiedSimEnv

    env = EmbodiedSimEnv(grade='M', backend='gym')
    obs = env.reset()
    for _ in range(1000):
        action = agent.predict(obs)  # 模型输出控制指令
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    # 或使用 EmbodiedSimulator 直接控制
    sim = EmbodiedSimulator(grade='M')
    sim.reset()
    state = sim.get_state()
    # ... 仿真循环
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List
from enum import Enum
import time


class SimBackend(Enum):
    """仿真后端"""
    GYM = "gym"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    NONE = "none"


class EmbodiedSimGrade(Enum):
    """具身仿真等级"""
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


# ============================================================================
# AGV五级仿真环境参数
# ============================================================================

AGV_SIM_GRADES: Dict[str, dict] = {
    'S': {
        'dt': 0.02,
        'control_rate': 50,
        'sensor_rate': 50,
        'max_linear_speed': 0.5,
        'max_angular_speed': 1.5,
        'max_linear_accel': 0.5,
        'max_angular_accel': 2.0,
        'tactile_array_size': (8, 8),
        'force_noise_std': 0.5,
        'imu_noise_std': 0.01,
        'encoder_resolution': 1024,
        'payload_kg': 30.0,
        'vehicle_mass_kg': 15.0,
        'wheel_radius_m': 0.05,
        'wheelbase_m': 0.3,
    },
    'M': {
        'dt': 0.01,
        'control_rate': 100,
        'sensor_rate': 100,
        'max_linear_speed': 1.5,
        'max_angular_speed': 2.0,
        'max_linear_accel': 1.0,
        'max_angular_accel': 3.0,
        'tactile_array_size': (16, 16),
        'force_noise_std': 0.2,
        'imu_noise_std': 0.005,
        'encoder_resolution': 2048,
        'payload_kg': 100.0,
        'vehicle_mass_kg': 35.0,
        'wheel_radius_m': 0.07,
        'wheelbase_m': 0.4,
    },
    'L': {
        'dt': 0.005,
        'control_rate': 200,
        'sensor_rate': 200,
        'max_linear_speed': 2.0,
        'max_angular_speed': 1.5,
        'max_linear_accel': 1.5,
        'max_angular_accel': 2.0,
        'tactile_array_size': (24, 24),
        'force_noise_std': 0.1,
        'imu_noise_std': 0.002,
        'encoder_resolution': 4096,
        'payload_kg': 300.0,
        'vehicle_mass_kg': 80.0,
        'wheel_radius_m': 0.07,
        'wheelbase_m': 0.6,
    },
    'XL': {
        'dt': 0.002,
        'control_rate': 500,
        'sensor_rate': 500,
        'max_linear_speed': 2.5,
        'max_angular_speed': 1.2,
        'max_linear_accel': 2.0,
        'max_angular_accel': 1.5,
        'tactile_array_size': (32, 32),
        'force_noise_std': 0.05,
        'imu_noise_std': 0.001,
        'encoder_resolution': 8192,
        'payload_kg': 600.0,
        'vehicle_mass_kg': 150.0,
        'wheel_radius_m': 0.0825,
        'wheelbase_m': 0.7,
    },
    'XXL': {
        'dt': 0.001,
        'control_rate': 1000,
        'sensor_rate': 1000,
        'max_linear_speed': 3.0,
        'max_angular_speed': 1.0,
        'max_linear_accel': 2.5,
        'max_angular_accel': 1.0,
        'tactile_array_size': (48, 48),
        'force_noise_std': 0.02,
        'imu_noise_std': 0.0005,
        'encoder_resolution': 16384,
        'payload_kg': 1200.0,
        'vehicle_mass_kg': 300.0,
        'wheel_radius_m': 0.1,
        'wheelbase_m': 0.9,
    },
}


def get_sim_grade_spec(grade: str) -> dict:
    """获取指定AGV等级的仿真规格"""
    return AGV_SIM_GRADES.get(grade, AGV_SIM_GRADES['M'])


# ============================================================================
# 仿真环境状态
# ============================================================================

@dataclass
class SimEnvironmentState:
    """仿真环境状态"""
    timestamp: float = 0.0
    dt: float = 0.01

    # 位置与姿态
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) world frame
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # (4,) quaternion
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) roll, pitch, yaw

    # 速度
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) m/s
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) rad/s

    # 控制输入
    applied_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))  # (6,) Fx,Fy,Fz,Tx,Ty,Tz

    # 传感器原始读数
    imu_accel_raw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro_raw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_accel_noisy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro_noisy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    force_reading: np.ndarray = field(default_factory=lambda: np.zeros(6))  # (6,) Fx,Fy,Fz,Tx,Ty,Tz
    tactile_pressure_map: Optional[np.ndarray] = None

    # 编码器
    wheel_positions: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (2,) 左/右轮
    wheel_velocities: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # 接触状态
    contact_forces: Dict[str, np.ndarray] = field(default_factory=dict)
    contact_points: List[Tuple[float, float, float]] = field(default_factory=list)

    # 环境交互
    payload_mass_kg: float = 0.0
    external_disturbance: np.ndarray = field(default_factory=lambda: np.zeros(6))
    terrain_type: str = "flat"  # flat, rough, slope

    # ============================================================
    # 电池状态 (新增)
    # ============================================================
    battery_soc: float = 1.0  # 0-1
    battery_voltage: float = 48.0  # V
    battery_current: float = 0.0  # A
    battery_remaining_wh: float = 500.0  # Wh

    # ============================================================
    # 车轮滑移状态 (新增)
    # ============================================================
    wheel_slip_l: float = 0.0  # 左轮滑移率 0-1
    wheel_slip_r: float = 0.0  # 右轮滑移率 0-1
    terrain_friction: float = 0.8  # 地形摩擦系数

    # ============================================================
    # 电机温度状态 (新增)
    # ============================================================
    motor_temp_l: float = 25.0  # 左电机温度 °C
    motor_temp_r: float = 25.0  # 右电机温度 °C
    motor_overheating: bool = False  # 是否过热

    def to_array(self) -> np.ndarray:
        """展平为观测向量 (扩展至27维)"""
        obs = [
            self.position,  # 3
            self.euler,  # 3
            self.linear_velocity,  # 3
            self.angular_velocity,  # 3
            self.imu_accel_noisy,  # 3
            self.imu_gyro_noisy,  # 3
            self.wheel_positions,  # 2
            self.wheel_velocities,  # 2
            # 新增: 电池/SOC (3)
            np.array([self.battery_soc, self.battery_voltage, self.battery_current]),
            # 新增: 车轮滑移 (3)
            np.array([self.wheel_slip_l, self.wheel_slip_r, self.terrain_friction]),
            # 新增: 电机温度 (3)
            np.array([self.motor_temp_l, self.motor_temp_r, 1.0 if self.motor_overheating else 0.0]),
        ]
        return np.concatenate(obs)


# ============================================================================
# 传感器噪声模型
# ============================================================================

class SensorNoiseModel:
    """传感器噪声模型"""

    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.spec = get_sim_grade_spec(grade)

        # 噪声标准差
        self.force_noise_std = self.spec['force_noise_std']
        self.imu_noise_std = self.spec['imu_noise_std']

        # 偏置
        self._imu_accel_bias = np.zeros(3)
        self._imu_gyro_bias = np.zeros(3)
        self._force_bias = np.zeros(6)

        # 随机游走偏置 (模拟漂移)
        self._accel_bias_drift = np.zeros(3)
        self._gyro_bias_drift = np.zeros(3)

    def add_imu_noise(self, accel_true: np.ndarray, gyro_true: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """添加IMU噪声"""
        # 随机游走偏置
        self._accel_bias_drift += np.random.randn(3) * self.imu_noise_std * np.sqrt(dt)
        self._gyro_bias_drift += np.random.randn(3) * self.imu_noise_std * np.sqrt(dt)

        accel_noisy = accel_true + self._imu_accel_bias + self._accel_bias_drift
        accel_noisy += np.random.randn(3) * self.imu_noise_std

        gyro_noisy = gyro_true + self._imu_gyro_bias + self._gyro_bias_drift
        gyro_noisy += np.random.randn(3) * self.imu_noise_std

        return accel_noisy, gyro_noisy

    def add_force_noise(self, wrench_true: np.ndarray) -> np.ndarray:
        """添加力觉传感器噪声"""
        noise = np.random.randn(6) * self.force_noise_std
        return wrench_true + self._force_bias + noise

    def calibrate_accel_bias(self, accel_samples: np.ndarray):
        """标定加速度计偏置"""
        self._imu_accel_bias = -np.mean(accel_samples, axis=0)
        self._accel_bias_drift = np.zeros(3)

    def calibrate_gyro_bias(self, gyro_samples: np.ndarray):
        """标定陀螺仪偏置"""
        self._imu_gyro_bias = -np.mean(gyro_samples, axis=0)
        self._gyro_bias_drift = np.zeros(3)

    def reset(self):
        """重置噪声模型"""
        self._imu_accel_bias = np.zeros(3)
        self._imu_gyro_bias = np.zeros(3)
        self._accel_bias_drift = np.zeros(3)
        self._gyro_bias_drift = np.zeros(3)
        self._force_bias = np.zeros(6)


# ============================================================================
# 物理仿真器
# ============================================================================

class PhysicsSimulator:
    """
    简化的AGV物理仿真器

    实现:
    - 差分驱动运动学
    - 简化的接触力学
    - 有效载荷动力学
    - 电池/SOC仿真
    - 车轮滑移建模
    - 电机温度仿真
    """

    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.spec = get_sim_grade_spec(grade)

        self.mass = self.spec['vehicle_mass_kg']
        self.wheel_radius = self.spec['wheel_radius_m']
        self.wheelbase = self.spec['wheelbase_m']

        # 状态
        self.position = np.zeros(3)  # world frame
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)

        # 车轮
        self.wheel_l_pos = 0.0  # rad
        self.wheel_r_pos = 0.0
        self.wheel_l_vel = 0.0
        self.wheel_r_vel = 0.0

        # 负载
        self.payload_mass = 0.0

        # 地形
        self.terrain = "flat"

        # 接触
        self.contact_z = 0.0  # 接触面高度

        # ============================================================
        # 电池/SOC仿真 (新增)
        # ============================================================
        self.battery_capacity_wh = 500.0  # 电池容量 Wh
        self.battery_soc = 1.0  # 当前SOC 0-1
        self.battery_voltage = 48.0  # 标称电压 V
        self.battery_current_draw = 0.0  # 当前放电电流 A

        # ============================================================
        # 车轮滑移建模 (新增)
        # ============================================================
        self.wheel_slip_l = 0.0  # 左轮滑移率
        self.wheel_slip_r = 0.0  # 右轮滑移率
        self.terrain_friction = 0.8  # 地形摩擦系数

        # ============================================================
        # 电机温度仿真 (新增)
        # ============================================================
        self.motor_temp_l = 25.0  # 左电机温度 °C
        self.motor_temp_r = 25.0  # 右电机温度 °C
        self.motor_ambient_temp = 25.0  # 环境温度 °C
        self.motor_thermal_resistance = 0.5  # 热阻 °C/W
        self.motor_time_constant = 100.0  # 热时间常数 s
        self.motor_max_temp = 80.0  # 电机最高允许温度 °C

    def reset(self, position: np.ndarray = None, orientation: np.ndarray = None):
        """重置仿真状态"""
        self.position = position if position is not None else np.zeros(3)
        self.orientation = orientation if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)
        self.wheel_l_pos = 0.0
        self.wheel_r_pos = 0.0
        self.wheel_l_vel = 0.0
        self.wheel_r_vel = 0.0
        # 重置电池/SOC
        self.battery_soc = 1.0
        self.battery_current_draw = 0.0
        # 重置滑移
        self.wheel_slip_l = 0.0
        self.wheel_slip_r = 0.0
        self.terrain_friction = 0.8
        # 重置电机温度
        self.motor_temp_l = 25.0
        self.motor_temp_r = 25.0

    def set_payload(self, mass_kg: float):
        """设置有效载荷"""
        self.payload_mass = mass_kg

    def set_terrain(self, terrain: str):
        """设置地形"""
        self.terrain = terrain

    def step(self, cmd_vx: float, cmd_wz: float, dt: float, external_wrench: np.ndarray = None):
        """
        推进物理仿真一步

        Args:
            cmd_vx: 期望线速度 m/s
            cmd_wz: 期望角速度 rad/s
            dt: 仿真步长
            external_wrench: 外部力/力矩 (6,)
        """
        if external_wrench is None:
            external_wrench = np.zeros(6)

        # 限幅
        max_v = self.spec['max_linear_speed']
        max_w = self.spec['max_angular_speed']
        cmd_vx = np.clip(cmd_vx, -max_v, max_v)
        cmd_wz = np.clip(cmd_wz, -max_w, max_w)

        # ============================================================
        # 车轮滑移建模 (新增)
        # ============================================================
        r = self.wheel_radius
        b = self.wheelbase
        
        # 理论车轮速度
        wheel_vel_l_theoretical = (cmd_vx - cmd_wz * b / 2) / r
        wheel_vel_r_theoretical = (cmd_vx + cmd_wz * b / 2) / r
        
        # 地形摩擦系数 (根据地形类型)
        if self.terrain == "rough":
            self.terrain_friction = 0.5  # 粗糙地面摩擦低
        elif self.terrain == "slope":
            self.terrain_friction = 0.6  # 坡道摩擦中等
        elif self.terrain == "wet":
            self.terrain_friction = 0.4  # 湿滑地面
        else:  # flat
            self.terrain_friction = 0.8  # 平地摩擦高
        
        # 有效摩擦力限制
        effective_vx = cmd_vx * self.terrain_friction
        effective_wz = cmd_wz * self.terrain_friction
        
        # 计算滑移率 (车轮实际速度与期望速度的差异)
        self.wheel_slip_l = min(1.0, abs(wheel_vel_l_theoretical - cmd_vx / r) * 0.1)
        self.wheel_slip_r = min(1.0, abs(wheel_vel_r_theoretical - cmd_vx / r) * 0.1)
        
        # 在粗糙地面上增加额外的随机滑移
        if self.terrain == "rough":
            self.wheel_slip_l += abs(np.random.randn() * 0.05)
            self.wheel_slip_r += abs(np.random.randn() * 0.05)
        
        self.wheel_slip_l = min(1.0, self.wheel_slip_l)
        self.wheel_slip_r = min(1.0, self.wheel_slip_r)

        # 速度平滑 (考虑滑移影响)
        alpha = 0.8 * self.terrain_friction
        vx = alpha * effective_vx + (1 - alpha) * self.linear_vel[0]
        wz = alpha * effective_wz + (1 - alpha) * self.angular_vel[2]

        self.linear_vel[0] = vx
        self.angular_vel[2] = wz

        # 位置更新
        self.position[0] += vx * dt * np.cos(self._get_yaw())
        self.position[1] += vx * dt * np.sin(self._get_yaw())
        self.position[2] += 0.0  # 保持平地

        # 航向角更新
        yaw = self._get_yaw() + wz * dt
        self.orientation = self._euler_to_quat(np.array([0.0, 0.0, yaw]))

        # 车轮编码器 (考虑滑移)
        self.wheel_l_vel = wheel_vel_l_theoretical * (1 - self.wheel_slip_l)
        self.wheel_r_vel = wheel_vel_r_theoretical * (1 - self.wheel_slip_r)
        self.wheel_l_pos += self.wheel_l_vel * dt
        self.wheel_r_pos += self.wheel_r_vel * dt

        # ============================================================
        # 电池/SOC仿真 (新增)
        # ============================================================
        # 估算电流消耗 (基于速度和加速度)
        total_mass = self.mass + self.payload_mass
        power_draw = (abs(vx) * total_mass * 0.5 + 
                     abs(wz) * total_mass * 0.2 +
                     10.0)  # 基础功耗 (控制器等)
        
        # 电流 = 功率 / 电压
        self.battery_current_draw = power_draw / self.battery_voltage
        
        # 更新SOC (Wh -> Ah -> SOC变化)
        energy_consumed_wh = power_draw * dt  # Wh
        soc_delta = energy_consumed_wh / self.battery_capacity_wh
        self.battery_soc = max(0.0, min(1.0, self.battery_soc - soc_delta))

        # 地形影响
        if self.terrain == "slope":
            gravity_component = 9.81 * 0.1  # 10%坡度
            self.linear_vel[0] -= gravity_component * dt * 0.1
        elif self.terrain == "rough":
            disturbance = np.random.randn(3) * 0.01
            self.linear_vel += disturbance * dt

        # ============================================================
        # 电机温度仿真 (新增)
        # ============================================================
        # 计算电机功率消耗
        motor_power_l = abs(self.wheel_l_vel) * 10.0 + 5.0  # W
        motor_power_r = abs(self.wheel_r_vel) * 10.0 + 5.0  # W
        
        # 温升 = 功率 * 热阻 * (1 - 衰减系数)
        temp_rise_l = motor_power_l * self.motor_thermal_resistance * (1 - self.motor_temp_l / self.motor_max_temp)
        temp_rise_r = motor_power_r * self.motor_thermal_resistance * (1 - self.motor_temp_r / self.motor_max_temp)
        
        # 一阶惯性环节: T(t+dt) = T(t) + (T_ambient - T(t) + temp_rise) * dt / tau
        self.motor_temp_l += (self.motor_ambient_temp - self.motor_temp_l + temp_rise_l) * dt / self.motor_time_constant
        self.motor_temp_r += (self.motor_ambient_temp - self.motor_temp_r + temp_rise_r) * dt / self.motor_time_constant

    def get_battery_state(self) -> Dict[str, float]:
        """
        获取电池状态 (新增)
        
        Returns:
            包含soc、电压、电流等信息的字典
        """
        return {
            'soc': self.battery_soc,
            'voltage': self.battery_voltage,
            'current': self.battery_current_draw,
            'capacity_wh': self.battery_capacity_wh,
            'remaining_wh': self.battery_soc * self.battery_capacity_wh,
        }

    def get_wheel_slip(self) -> Tuple[float, float]:
        """
        获取车轮滑移率 (新增)
        
        Returns:
            (左轮滑移率, 右轮滑移率)
        """
        return self.wheel_slip_l, self.wheel_slip_r

    def get_motor_temperatures(self) -> Tuple[float, float]:
        """
        获取电机温度 (新增)
        
        Returns:
            (左电机温度, 右电机温度) °C
        """
        return self.motor_temp_l, self.motor_temp_r

    def _get_yaw(self) -> float:
        """从四元数获取偏航角"""
        q = self.orientation
        return np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                          1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))

    def _euler_to_quat(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数 (ZYX顺序)"""
        cy = np.cos(euler[2] * 0.5)
        sy = np.sin(euler[2] * 0.5)
        cp = np.cos(euler[1] * 0.5)
        sp = np.sin(euler[1] * 0.5)
        cr = np.cos(euler[0] * 0.5)
        sr = np.sin(euler[0] * 0.5)
        return np.array([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ])

    def compute_imu_reading(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算IMU读数 (去除重力后的加速度 + 角速度)

        Returns:
            accel: (3,) 去除重力后的加速度 m/s²
            gyro: (3,) 角速度 rad/s
        """
        # 重力向量 (world frame)
        gravity = np.array([0.0, 0.0, -9.81])

        # 车身加速度 (world frame)
        body_accel = np.array([
            (self.linear_vel[0]) / 0.1 if False else 0.0,
            0.0,
            0.0
        ])  # 简化: 忽略哥氏力

        # 旋转到body frame
        R = self._quat_to_rot(self.orientation)
        accel_body = R.T @ (body_accel - gravity)
        gyro_body = R.T @ self.angular_vel

        return accel_body, gyro_body

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
        ])


# ============================================================================
# 触觉仿真器
# ============================================================================

class TactileSimulator:
    """触觉阵列仿真器"""

    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.spec = get_sim_grade_spec(grade)
        self.array_size = self.spec['tactile_array_size']
        self._pressure_map = np.zeros(self.array_size)
        self._contact_active = False
        self._contact_center = (self.array_size[0] // 2, self.array_size[1] // 2)
        self._contact_radius = 2

    def reset(self):
        """重置"""
        self._pressure_map = np.zeros(self.array_size)
        self._contact_active = False

    def apply_contact(self, force: float, center: Tuple[int, int] = None, radius: int = None):
        """
        在触觉阵列上施加接触压力

        Args:
            force: 接触力 N
            center: 接触中心 (row, col)
            radius: 接触半径 (像素)
        """
        self._contact_active = True
        self._contact_center = center if center is not None else (
            self.array_size[0] // 2, self.array_size[1] // 2)
        self._contact_radius = radius if radius is not None else 2

        r, c = self._contact_center
        rad = self._contact_radius
        h, w = self.array_size

        # 生成高斯压力分布
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - r) ** 2 + (x - c) ** 2)
        pressure = force * np.exp(-(dist ** 2) / (2 * rad ** 2))
        pressure[dist > rad * 2] = 0

        self._pressure_map = np.clip(pressure, 0, 1)

    def release_contact(self):
        """释放接触"""
        self._contact_active = False
        self._pressure_map = np.zeros(self.array_size) * 0.95  # 衰减残留

    def get_pressure_map(self) -> np.ndarray:
        """获取当前压力图"""
        if not self._contact_active:
            # 指数衰减到零
            self._pressure_map *= 0.9
        return self._pressure_map.copy()


# ============================================================================
# 完整具身仿真器
# ============================================================================

class EmbodiedSimulator:
    """
    完整的具身智能仿真器

    提供:
    - 物理仿真 (位置/速度/姿态)
    - 传感器仿真 (IMU/力觉/触觉/编码器)
    - 传感器噪声建模
    - AGV五级规格适配
    """

    def __init__(self, grade: str = 'M', seed: int = None):
        """
        Args:
            grade: AGV五级等级 (S/M/L/XL/XXL)
            seed: 随机种子
        """
        self.grade = grade
        self.spec = get_sim_grade_spec(grade)
        self.dt = self.spec['dt']

        if seed is not None:
            np.random.seed(seed)

        # 子仿真器
        self.physics = PhysicsSimulator(grade=grade)
        self.sensor_noise = SensorNoiseModel(grade=grade)
        self.tactile_sim = TactileSimulator(grade=grade)

        # 状态
        self.state = SimEnvironmentState()
        self._step_count = 0
        self._sim_time = 0.0
        self._is_running = False

    def reset(self, initial_position: np.ndarray = None) -> SimEnvironmentState:
        """
        重置仿真环境

        Args:
            initial_position: 初始位置 (3,)

        Returns:
            初始环境状态
        """
        self.physics.reset(position=initial_position)
        self.sensor_noise.reset()
        self.tactile_sim.reset()
        self._step_count = 0
        self._sim_time = 0.0
        self._is_running = True

        # 初始化状态
        self.state = SimEnvironmentState(dt=self.dt)
        self.state.position = self.physics.position.copy()
        self.state.orientation = self.physics.orientation.copy()
        self.state.linear_velocity = self.physics.linear_vel.copy()
        self.state.angular_velocity = self.physics.angular_vel.copy()
        self.state.wheel_positions = np.array([self.physics.wheel_l_pos, self.physics.wheel_r_pos])
        self.state.wheel_velocities = np.array([self.physics.wheel_l_vel, self.physics.wheel_r_vel])

        return self.state

    def step(
        self,
        cmd_vx: float,
        cmd_wz: float,
        contact_force: float = 0.0,
        contact_center: Tuple[int, int] = None,
    ) -> SimEnvironmentState:
        """
        仿真一步

        Args:
            cmd_vx: 期望线速度 m/s
            cmd_wz: 期望角速度 rad/s
            contact_force: 触觉接触力 N
            contact_center: 触觉接触中心

        Returns:
            当前环境状态
        """
        if not self._is_running:
            raise RuntimeError("Simulator not running. Call reset() first.")

        # 物理仿真一步
        self.physics.step(cmd_vx, cmd_wz, self.dt)

        # IMU 传感器仿真
        accel_true, gyro_true = self.physics.compute_imu_reading()
        accel_noisy, gyro_noisy = self.sensor_noise.add_imu_noise(accel_true, gyro_true, self.dt)

        # 触觉仿真
        if contact_force > 0.01:
            self.tactile_sim.apply_contact(contact_force, contact_center)
        else:
            self.tactile_sim.release_contact()
        tactile_map = self.tactile_sim.get_pressure_map()

        # 力觉仿真 (基于运动状态)
        force_reading = self._compute_force_reading(cmd_vx, cmd_wz)

        # 更新状态
        self.state.timestamp = self._sim_time
        self.state.dt = self.dt
        self.state.position = self.physics.position.copy()
        self.state.orientation = self.physics.orientation.copy()
        self.state.linear_velocity = self.physics.linear_vel.copy()
        self.state.angular_velocity = self.physics.angular_vel.copy()
        self.state.wheel_positions = np.array([self.physics.wheel_l_pos, self.physics.wheel_r_pos])
        self.state.wheel_velocities = np.array([self.physics.wheel_l_vel, self.physics.wheel_r_vel])

        self.state.imu_accel_raw = accel_true.copy()
        self.state.imu_gyro_raw = gyro_true.copy()
        self.state.imu_accel_noisy = accel_noisy.copy()
        self.state.imu_gyro_noisy = gyro_noisy.copy()

        self.state.force_reading = force_reading
        self.state.tactile_pressure_map = tactile_map

        # ============================================================
        # 更新电池/SOC状态 (新增)
        # ============================================================
        battery_state = self.physics.get_battery_state()
        self.state.battery_soc = battery_state['soc']
        self.state.battery_voltage = battery_state['voltage']
        self.state.battery_current = battery_state['current']
        self.state.battery_remaining_wh = battery_state['remaining_wh']

        # ============================================================
        # 更新车轮滑移状态 (新增)
        # ============================================================
        slip_l, slip_r = self.physics.get_wheel_slip()
        self.state.wheel_slip_l = slip_l
        self.state.wheel_slip_r = slip_r
        self.state.terrain_friction = self.physics.terrain_friction

        # ============================================================
        # 更新电机温度状态 (新增)
        # ============================================================
        temp_l, temp_r = self.physics.get_motor_temperatures()
        self.state.motor_temp_l = temp_l
        self.state.motor_temp_r = temp_r
        self.state.motor_overheating = (temp_l >= self.physics.motor_max_temp or 
                                        temp_r >= self.physics.motor_max_temp)

        self._step_count += 1
        self._sim_time += self.dt

        return self.state

    def _compute_force_reading(self, cmd_vx: float, cmd_wz: float) -> np.ndarray:
        """基于运动状态计算力觉读数"""
        # 简化的力觉模型: 根据加速度估计接触力
        mass = self.physics.mass + self.physics.payload_mass
        accel_estimate = (cmd_vx - self.physics.linear_vel[0]) / self.dt if self.dt > 0 else 0.0
        estimated_force = mass * accel_estimate

        # 力觉噪声
        force_noisy = self.sensor_noise.add_force_noise(
            np.array([estimated_force, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        return force_noisy

    def get_state(self) -> SimEnvironmentState:
        """获取当前仿真状态"""
        return self.state

    def set_payload(self, mass_kg: float):
        """设置有效载荷"""
        self.physics.set_payload(mass_kg)

    def set_terrain(self, terrain: str):
        """设置地形"""
        self.physics.set_terrain(terrain)

    def stop(self):
        """停止仿真"""
        self._is_running = False

    def get_observation(self) -> np.ndarray:
        """获取当前观测向量 (用于强化学习)"""
        return self.state.to_array()

    def get_sensor_dict(self) -> Dict[str, Any]:
        """获取传感器数据字典 (扩展包含电池/滑移/温度)"""
        return {
            'imu': {
                'accel': self.state.imu_accel_noisy.copy(),
                'gyro': self.state.imu_gyro_noisy.copy(),
                'accel_raw': self.state.imu_accel_raw.copy(),
                'gyro_raw': self.state.imu_gyro_raw.copy(),
            },
            'force': self.state.force_reading.copy(),
            'tactile': self.state.tactile_pressure_map.copy() if self.state.tactile_pressure_map is not None else None,
            'encoders': {
                'wheel_l_pos': self.state.wheel_positions[0],
                'wheel_r_pos': self.state.wheel_positions[1],
                'wheel_l_vel': self.state.wheel_velocities[0],
                'wheel_r_vel': self.state.wheel_velocities[1],
            },
            'pose': {
                'position': self.state.position.copy(),
                'orientation': self.state.orientation.copy(),
                'linear_vel': self.state.linear_velocity.copy(),
                'angular_vel': self.state.angular_velocity.copy(),
            },
            # 新增: 电池状态
            'battery': {
                'soc': self.state.battery_soc,
                'voltage': self.state.battery_voltage,
                'current': self.state.battery_current,
                'remaining_wh': self.state.battery_remaining_wh,
            },
            # 新增: 车轮滑移状态
            'wheel_slip': {
                'slip_l': self.state.wheel_slip_l,
                'slip_r': self.state.wheel_slip_r,
                'friction': self.state.terrain_friction,
            },
            # 新增: 电机温度状态
            'motor_temp': {
                'motor_l': self.state.motor_temp_l,
                'motor_r': self.state.motor_temp_r,
                'overheating': self.state.motor_overheating,
            },
        }


# ============================================================================
# Gymnasium 环境接口
# ============================================================================

GYM_AVAILABLE = False
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    pass


class EmbodiedSimEnv:
    """
    Gymnasium 兼容的具身智能仿真环境

    观测空间: 位置(3) + 姿态(3) + 线速度(3) + 角速度(3) + IMU加计(3) + IMU陀螺(3) + 轮位置(2) + 轮速度(2) + 电池(3) + 滑移(3) + 温度(3) = 31维
    动作空间: 线速度(-1,1) + 角速度(-1,1) = 2维
    """

    if GYM_AVAILABLE:
        def __init__(
            self,
            grade: str = 'M',
            max_episode_steps: int = 1000,
            reward_type: str = 'tracking',
        ):
            self.grade = grade
            self.spec = get_sim_grade_spec(grade)
            self.max_episode_steps = max_episode_steps
            self.reward_type = reward_type

            self.sim = EmbodiedSimulator(grade=grade)

            # 观测空间: 31维 (扩展了电池/滑移/温度)
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(31,), dtype=np.float32
            )

            # 动作空间: [vx_cmd, wz_cmd]，归一化到 [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

            self._step_count = 0
            self._target_vx = 0.0
            self._target_wz = 0.0

        def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
            """重置环境"""
            if seed is not None:
                np.random.seed(seed)

            self._step_count = 0
            self._target_vx = np.random.uniform(-0.5, 0.5) * self.spec['max_linear_speed']
            self._target_wz = np.random.uniform(-0.5, 0.5) * self.spec['max_angular_speed']

            self.sim.reset()
            obs = self.sim.get_observation()
            return obs.astype(np.float32), {}

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
            """
            执行一步

            Args:
                action: [vx_cmd, wz_cmd]，范围 [-1, 1]

            Returns:
                obs, reward, terminated, truncated, info
            """
            # 解码动作
            max_v = self.spec['max_linear_speed']
            max_w = self.spec['max_angular_speed']
            cmd_vx = float(action[0]) * max_v
            cmd_wz = float(action[1]) * max_w

            # 仿真一步
            state = self.sim.step(cmd_vx, cmd_wz)

            # 计算奖励
            reward = self._compute_reward(state)

            # 检查终止
            self._step_count += 1
            terminated = False
            truncated = self._step_count >= self.max_episode_steps

            # 检测碰撞/翻车
            pos = state.position
            if np.abs(pos[0]) > 5.0 or np.abs(pos[1]) > 5.0:
                terminated = True
                reward -= 10.0

            info = {
                'position': state.position.copy(),
                'velocity': state.linear_velocity.copy(),
                'target_vx': self._target_vx,
                'target_wz': self._target_wz,
            }

            obs = state.to_array()
            return obs.astype(np.float32), reward, terminated, truncated, info

        def _compute_reward(self, state: SimEnvironmentState) -> float:
            """计算奖励"""
            if self.reward_type == 'tracking':
                # 速度跟踪奖励
                v_error = (state.linear_velocity[0] - self._target_vx) ** 2
                w_error = (state.angular_velocity[2] - self._target_wz) ** 2
                reward = -v_error - w_error
                # 每步小奖励
                reward += 0.01
                return reward
            elif self.reward_type == 'energy_efficient':
                # 能量效率奖励
                v_error = (state.linear_velocity[0] - self._target_vx) ** 2
                w_error = (state.angular_velocity[2] - self._target_wz) ** 2
                reward = -v_error - w_error - 0.001 * np.sum(state.force_reading ** 2)
                return reward
            else:
                return 0.0

        def render(self, mode: str = 'human'):
            """渲染 (当前仅打印状态)"""
            if mode == 'human':
                state = self.sim.get_state()
                print(f"pos=({state.position[0]:.3f}, {state.position[1]:.3f}) "
                      f"vel=({state.linear_velocity[0]:.3f}) "
                      f"yaw=({self._get_yaw(state.orientation):.3f})")

        def close(self):
            """关闭环境"""
            self.sim.stop()

        @staticmethod
        def _get_yaw(q: np.ndarray) -> float:
            return np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                              1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


def create_sim_env(grade: str = 'M', backend: str = 'gym') -> EmbodiedSimEnv:
    """工厂函数: 创建仿真环境"""
    return EmbodiedSimEnv(grade=grade)


# ============================================================================
# 五级规格快速查询
# ============================================================================

def get_grade_summary() -> str:
    """获取五级规格摘要"""
    lines = ["AGV五级仿真规格摘要:", "-" * 50]
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        spec = get_sim_grade_spec(grade)
        lines.append(
            f"  {grade}: dt={spec['dt']*1000:.0f}ms, "
            f"rate={spec['control_rate']}Hz, "
            f"tactile={spec['tactile_array_size']}, "
            f"max_speed={spec['max_linear_speed']}m/s, "
            f"payload={spec['payload_kg']}kg"
        )
    return "\n".join(lines)
