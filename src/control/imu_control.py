"""
IMU控制模块
============

基于惯性测量单元的控制:
- 姿态稳定控制
- 运动估计
- 倾角补偿
- AHRS (姿态航向参考系统)

集成:
- IMUSensor → AttitudeStabilizer
- PoseEstimator → MotionEstimator
- IMUFrame → IMUBasedController
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.imu import IMUSensor, IMUFrame, PoseEstimator, Pose, get_imu_spec


@dataclass
class IMUControlParams:
    """IMU控制参数"""
    # 姿态控制增益
    Kp_attitude: float = 5.0
    Ki_attitude: float = 0.1
    Kd_attitude: float = 1.0
    
    # 角速度限制
    max_angular_velocity: float = 2.0  # rad/s
    
    # 倾角报警阈值
    tilt_warning: float = 0.261  # 15 deg
    tilt_critical: float = 0.524  # 30 deg
    
    # 静止检测阈值
    motion_threshold: float = 0.5  # m/s^2
    
    # 控制频率
    control_rate: float = 100.0  # Hz
    
    # AGV等级
    grade: str = 'M'
    
    @classmethod
    def from_grade(cls, grade: str) -> 'IMUControlParams':
        configs = {
            'S':  cls(Kp_attitude=2.0, Kd_attitude=0.5, max_angular_velocity=1.0, control_rate=50, grade='S'),
            'M':  cls(Kp_attitude=5.0, Kd_attitude=1.0, max_angular_velocity=2.0, control_rate=100, grade='M'),
            'L':  cls(Kp_attitude=8.0, Kd_attitude=2.0, max_angular_velocity=3.0, control_rate=200, grade='L'),
            'XL': cls(Kp_attitude=10.0, Kd_attitude=3.0, max_angular_velocity=5.0, control_rate=500, grade='XL'),
            'XXL': cls(Kp_attitude=15.0, Kd_attitude=5.0, max_angular_velocity=10.0, control_rate=1000, grade='XXL'),
        }
        return configs.get(grade, cls())


class AttitudeStabilizer:
    """
    姿态稳定控制器
    
    功能:
    - 保持目标姿态
    - 倾角检测与补偿
    - 抗干扰控制
    """
    
    def __init__(
        self,
        imu_sensor: IMUSensor,
        params: Optional[IMUControlParams] = None
    ):
        self.imu = imu_sensor
        self.params = params or IMUControlParams()
        self.pose_estimator = PoseEstimator(sample_rate=self.params.control_rate)
        
        # 目标姿态
        self._target_euler = np.zeros(3)  # roll, pitch, yaw
        self._target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 积分状态
        self._euler_error_integral = np.zeros(3)
        self._last_euler_error = np.zeros(3)
        
        # 倾角状态
        self._tilt_warning = False
        self._tilt_critical = False
        
    def set_target_attitude(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        """设置目标姿态"""
        self._target_euler = np.array([roll, pitch, yaw])
        self._target_quaternion = self._euler_to_quaternion(self._target_euler)
    
    def update(self, current_frame: Optional[IMUFrame] = None, dt: float = 0.01) -> np.ndarray:
        """
        更新姿态控制
        
        Args:
            current_frame: 当前IMU帧
            dt: 时间步长
            
        Returns:
            control_torque: 控制力矩 (3,)
        """
        if current_frame is None:
            current_frame = self.imu.capture()
        
        # 更新姿态估计
        pose = self.pose_estimator.update(
            current_frame.accel,
            current_frame.gyro,
            current_frame.mag,
            dt
        )
        
        current_euler = pose.to_euler()
        
        # 姿态误差 (考虑角度回绕)
        euler_error = self._angle_diff(current_euler, self._target_euler)
        
        # 积分项
        self._euler_error_integral += euler_error * dt
        self._euler_error_integral = np.clip(
            self._euler_error_integral, -0.5, 0.5
        )
        
        # 微分项
        d_error = (euler_error - self._last_euler_error) / (dt + 1e-6)
        self._last_euler_error = euler_error.copy()
        
        # PID 控制
        control_torque = (
            self.params.Kp_attitude * euler_error +
            self.params.Ki_attitude * self._euler_error_integral +
            self.params.Kd_attitude * d_error
        )
        
        # 限制角速度
        max_omega = self.params.max_angular_velocity
        control_torque = np.clip(control_torque, -max_omega, max_omega)
        
        # 倾角检测
        roll_abs = abs(current_euler[0])
        pitch_abs = abs(current_euler[1])
        tilt_magnitude = np.sqrt(roll_abs**2 + pitch_abs**2)
        
        self._tilt_warning = tilt_magnitude > self.params.tilt_warning
        self._tilt_critical = tilt_magnitude > self.params.tilt_critical
        
        return control_torque.astype(np.float32)
    
    def get_tilt_status(self) -> Dict[str, any]:
        """获取倾角状态"""
        pose = self.pose_estimator.get_pose()
        euler = pose.to_euler()
        
        roll_abs = abs(euler[0])
        pitch_abs = abs(euler[1])
        tilt_mag = np.sqrt(roll_abs**2 + pitch_abs**1)
        
        return {
            'roll': float(euler[0]),
            'pitch': float(euler[1]),
            'yaw': float(euler[2]),
            'tilt_magnitude': float(tilt_mag),
            'tilt_warning': self._tilt_warning,
            'tilt_critical': self._tilt_critical,
            'is_stable': tilt_mag < self.params.tilt_warning
        }
    
    def is_moving(self, current_frame: Optional[IMUFrame] = None) -> bool:
        """检测是否在运动"""
        if current_frame is None:
            current_frame = self.imu.capture()
        
        accel_mag = np.linalg.norm(current_frame.accel)
        gyro_mag = np.linalg.norm(current_frame.gyro)
        
        # 静止: 加速度接近1g, 角速度接近0
        accel_deviation = abs(accel_mag - 9.81)
        is_still = accel_deviation < self.params.motion_threshold and gyro_mag < 0.1
        
        return not is_still
    
    @staticmethod
    def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """计算角度差 (处理回绕)"""
        diff = a - b
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return diff
    
    @staticmethod
    def _euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数"""
        roll, pitch, yaw = euler
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])


class MotionEstimator:
    """
    运动估计器
    
    基于IMU积分估计:
    - 速度
    - 位置
    - 轨迹
    """
    
    def __init__(
        self,
        imu_sensor: IMUSensor,
        remove_gravity: bool = True
    ):
        self.imu = imu_sensor
        self.remove_gravity = remove_gravity
        
        # 积分状态
        self._velocity = np.zeros(3)
        self._position = np.zeros(3)
        self._trajectory: List[Tuple[float, np.ndarray, np.ndarray]] = []  # (t, pos, vel)
        
        self._initialized = False
        
    def reset(self):
        """重置积分状态"""
        self._velocity = np.zeros(3)
        self._position = np.zeros(3)
        self._trajectory = []
        self._initialized = False
    
    def update(
        self,
        current_frame: Optional[IMUFrame] = None,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新运动估计
        
        Args:
            current_frame: 当前IMU帧
            dt: 时间步长
            
        Returns:
            velocity: 估计速度 (3,)
            position: 估计位置 (3,)
        """
        if current_frame is None:
            current_frame = self.imu.capture()
        
        accel = current_frame.accel.copy()
        
        # 去除重力
        if self.remove_gravity:
            gravity = np.array([0.0, 0.0, 9.81])
            accel = accel - gravity
        
        # 积分
        self._velocity = self._velocity + accel * dt
        self._position = self._position + self._velocity * dt
        
        # 记录轨迹
        t = current_frame.timestamp
        self._trajectory.append((t, self._position.copy(), self._velocity.copy()))
        
        # 限制轨迹长度
        if len(self._trajectory) > 10000:
            self._trajectory = self._trajectory[-5000:]
        
        self._initialized = True
        
        return self._velocity.copy(), self._position.copy()
    
    def get_trajectory(self, max_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取轨迹数据
        
        Returns:
            times: 时间戳数组
            positions: 位置数组 (N, 3)
        """
        if not self._trajectory:
            return np.array([]), np.zeros((0, 3))
        
        # 降采样
        step = max(1, len(self._trajectory) // max_points)
        sampled = self._trajectory[::step]
        
        times = np.array([t for t, _, _ in sampled])
        positions = np.array([pos for _, pos, _ in sampled])
        
        return times, positions
    
    def estimate_displacement(
        self,
        duration: float,
        sample_rate: float = 100.0
    ) -> float:
        """
        估计给定时间内的位移
        
        Args:
            duration: 持续时间 (秒)
            sample_rate: 采样率
            
        Returns:
            total_displacement: 总位移 (米)
        """
        n_samples = int(duration * sample_rate)
        
        if len(self._trajectory) < n_samples:
            # 积分不足，使用最近的速度估计
            avg_velocity = np.linalg.norm(self._velocity)
            return avg_velocity * duration
        
        start_idx = max(0, len(self._trajectory) - n_samples)
        positions = np.array([pos for _, pos, _ in self._trajectory[start_idx:]])
        
        # 计算路径长度
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        return float(np.sum(distances))


# AGV五级IMU控制规格
AGV_IMU_CONTROL_GRADES = {
    'S':  IMUControlParams.from_grade('S'),
    'M':  IMUControlParams.from_grade('M'),
    'L':  IMUControlParams.from_grade('L'),
    'XL': IMUControlParams.from_grade('XL'),
    'XXL': IMUControlParams.from_grade('XXL'),
}


def get_imu_control_spec(grade: str) -> IMUControlParams:
    """获取AGV指定等级的IMU控制参数"""
    return AGV_IMU_CONTROL_GRADES.get(grade, AGV_IMU_CONTROL_GRADES['M'])
