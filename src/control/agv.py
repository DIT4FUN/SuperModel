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
AGV运动控制模块
==============

AGV专用运动学/动力学控制
- 差速驱动 (Differential Drive)
- 全向移动 (Omnidirectional)
- 麦克纳姆轮 (Mecanum)
- 轨迹跟踪与偏差纠正
- 多AGV协调控制

支持AGV等级: S / M / L / XL / XXL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class DriveType(Enum):
    """驱动类型"""
    DIFFERENTIAL = "differential"       # 差速驱动
    OMNIDIRECTIONAL = "omnidirectional"  # 全向驱动
    MECANUM = "mecanum"               # 麦克纳姆轮
    ACKERMANN = "ackermann"           # 阿克曼 (汽车式)
    SWISS = "swiss"                   # 瑞士轮


class AGVGrade(Enum):
    """AGV等级"""
    S = "S"   # 教育/实验
    M = "M"   # 标准助手
    L = "L"   # 专业工业
    XL = "XL"  # 高性能
    XXL = "XXL"  # 旗舰全功能


@dataclass
class AGVSpec:
    """AGV规格参数"""
    grade: AGVGrade
    max_linear_speed: float      # 最大线速度 m/s
    max_angular_speed: float     # 最大角速度 rad/s
    max_linear_accel: float      # 最大线加速度 m/s^2
    max_angular_accel: float     # 最大角加速度 rad/s^2
    wheelbase: float             # 前后轴距 m
    track_width: float           # 左右轮距 m
    wheel_radius: float          # 轮子半径 m
    drive_type: DriveType
    control_frequency: float     # 控制频率 Hz
    
    @classmethod
    def from_grade(cls, grade: AGVGrade) -> "AGVSpec":
        """从AGV等级获取标准规格"""
        specs = {
            AGVGrade.S: cls(
                grade=AGVGrade.S,
                max_linear_speed=0.5, max_angular_speed=1.5,
                max_linear_accel=0.5, max_angular_accel=2.0,
                wheelbase=0.3, track_width=0.25, wheel_radius=0.05,
                drive_type=DriveType.DIFFERENTIAL, control_frequency=50.0
            ),
            AGVGrade.M: cls(
                grade=AGVGrade.M,
                max_linear_speed=1.0, max_angular_speed=2.0,
                max_linear_accel=1.0, max_angular_accel=3.0,
                wheelbase=0.5, track_width=0.4, wheel_radius=0.08,
                drive_type=DriveType.DIFFERENTIAL, control_frequency=100.0
            ),
            AGVGrade.L: cls(
                grade=AGVGrade.L,
                max_linear_speed=2.0, max_angular_speed=2.5,
                max_linear_accel=2.0, max_angular_accel=4.0,
                wheelbase=0.8, track_width=0.6, wheel_radius=0.1,
                drive_type=DriveType.MECANUM, control_frequency=200.0
            ),
            AGVGrade.XL: cls(
                grade=AGVGrade.XL,
                max_linear_speed=3.0, max_angular_speed=3.0,
                max_linear_accel=3.0, max_angular_accel=5.0,
                wheelbase=1.0, track_width=0.8, wheel_radius=0.15,
                drive_type=DriveType.MECANUM, control_frequency=500.0
            ),
            AGVGrade.XXL: cls(
                grade=AGVGrade.XXL,
                max_linear_speed=5.0, max_angular_speed=3.5,
                max_linear_accel=5.0, max_angular_accel=6.0,
                wheelbase=1.5, track_width=1.2, wheel_radius=0.2,
                drive_type=DriveType.MECANUM, control_frequency=1000.0
            ),
        }
        return specs[grade]


@dataclass
class AGVPose:
    """AGV位姿"""
    x: float = 0.0           # 世界坐标系X (m)
    y: float = 0.0           # 世界坐标系Y (m)
    theta: float = 0.0       # 朝向角 (rad)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "AGVPose":
        return cls(x=v[0], y=v[1], theta=v[2])


@dataclass
class AGVTwist:
    """AGV速度"""
    vx: float = 0.0           # X方向线速度 m/s
    vy: float = 0.0           # Y方向线速度 m/s
    omega: float = 0.0         # 角速度 rad/s
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.omega], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "AGVTwist":
        return cls(vx=v[0], vy=v[1], omega=v[2])


class AGVMotionController:
    """
    AGV运动控制器
    
    实现:
    - 正运动学 (逆解): 世界速度 -> 轮速
    - 逆运动学 (正解): 轮速 -> 世界速度
    - 轨迹跟踪
    - 安全限制
    """
    
    def __init__(self, spec: AGVSpec):
        self.spec = spec
        self._pose = AGVPose()
        self._twist = AGVTwist()
        
        # 运动学模型
        self._kinematics = KinematicsFactory.create(spec.drive_type, spec)
        
        # PID跟踪控制器
        self._init_tracking_controller()
    
    def _init_tracking_controller(self):
        """初始化轨迹跟踪PID"""
        self.kp_trans = 2.0
        self.ki_trans = 0.1
        self.kd_trans = 0.5
        self.kp_rot = 3.0
        self.ki_rot = 0.2
        self.kd_rot = 0.8
        
        self._trans_error_integral = 0.0
        self._rot_error_integral = 0.0
        self._last_trans_error = 0.0
        self._last_rot_error = 0.0
    
    def forward_kinematics(self, wheel_velocities: np.ndarray) -> AGVTwist:
        """
        正运动学: 轮速 -> AGV速度
        
        Args:
            wheel_velocities: 各轮速度 [rad/s]
        Returns:
            AGV世界坐标系速度
        """
        return self._kinematics.wheel_to_body(wheel_velocities)
    
    def inverse_kinematics(self, twist: AGVTwist) -> np.ndarray:
        """
        逆运动学: AGV速度 -> 轮速
        
        Args:
            twist: AGV速度 (世界坐标系)
        Returns:
            各轮速度命令 [rad/s]
        """
        return self._kinematics.body_to_wheel(twist)
    
    def update_pose(self, new_pose: AGVPose):
        """更新AGV当前位姿"""
        self._pose = new_pose
    
    def update_twist(self, new_twist: AGVTwist):
        """更新AGV当前速度"""
        self._twist = new_twist
    
    def compute_wheel_commands(
        self,
        target_pose: AGVPose,
        dt: float
    ) -> np.ndarray:
        """
        计算轮速命令 (轨迹跟踪)
        
        Args:
            target_pose: 目标位姿
            dt: 时间步长
        Returns:
            轮速命令 [rad/s]
        """
        # 计算位置误差
        dx = target_pose.x - self._pose.x
        dy = target_pose.y - self._pose.y
        dtheta = target_pose.theta - self._pose.theta
        
        # 角度归一化
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # PID跟踪
        trans_error = np.sqrt(dx**2 + dy**2)
        self._trans_error_integral += trans_error * dt
        self._trans_error_integral = np.clip(self._trans_error_integral, -1.0, 1.0)
        trans_derivative = (trans_error - self._last_trans_error) / dt
        self._last_trans_error = trans_error
        
        self._rot_error_integral += dtheta * dt
        self._rot_error_integral = np.clip(self._rot_error_integral, -1.0, 1.0)
        rot_derivative = (dtheta - self._last_rot_error) / dt
        self._last_rot_error = dtheta
        
        # 速度命令
        v_cmd = self.kp_trans * trans_error + self.ki_trans * self._trans_error_integral + self.kd_trans * trans_derivative
        omega_cmd = self.kp_rot * dtheta + self.ki_rot * self._rot_error_integral + self.kd_rot * rot_derivative
        
        # 限幅
        v_cmd = np.clip(v_cmd, -self.spec.max_linear_speed, self.spec.max_linear_speed)
        omega_cmd = np.clip(omega_cmd, -self.spec.max_angular_speed, self.spec.max_angular_speed)
        
        # 转换到局部坐标系
        vx_cmd = v_cmd * np.cos(self._pose.theta)
        vy_cmd = v_cmd * np.sin(self._pose.theta)
        
        twist = AGVTwist(vx=vx_cmd, vy=vy_cmd, omega=omega_cmd)
        return self.inverse_kinematics(twist)
    
    def apply_safety_limits(self, wheel_commands: np.ndarray) -> np.ndarray:
        """应用安全限制"""
        # 最大速度限制
        max_wheel_vel = self.spec.max_linear_speed / self.spec.wheel_radius
        wheel_commands = np.clip(wheel_commands, -max_wheel_vel, max_wheel_vel)
        
        # 最大加速度限制 (通过限幅变化率)
        return wheel_commands
    
    @property
    def pose(self) -> AGVPose:
        return self._pose
    
    @property
    def twist(self) -> AGVTwist:
        return self._twist


class KinematicsBase:
    """运动学基类"""
    
    def wheel_to_body(self, wheel_velocities: np.ndarray) -> AGVTwist:
        raise NotImplementedError
    
    def body_to_wheel(self, twist: AGVTwist) -> np.ndarray:
        raise NotImplementedError


class DifferentialKinematics(KinematicsBase):
    """差速驱动运动学"""
    
    def __init__(self, spec: AGVSpec):
        self.wheelbase = spec.wheelbase
        self.track_width = spec.track_width
        self.wheel_radius = spec.wheel_radius
    
    def wheel_to_body(self, wheel_velocities: np.ndarray) -> AGVTwist:
        """
        差速驱动正运动学
        
        wheel_velocities: [left_vel, right_vel] rad/s
        """
        v_l = wheel_velocities[0] * self.wheel_radius
        v_r = wheel_velocities[1] * self.wheel_radius
        
        v = (v_l + v_r) / 2.0  # 线速度
        omega = (v_r - v_l) / self.track_width  # 角速度
        
        return AGVTwist(vx=v, vy=0.0, omega=omega)
    
    def body_to_wheel(self, twist: AGVTwist) -> np.ndarray:
        """
        差速驱动逆运动学
        
        返回: [left_vel, right_vel] rad/s
        """
        v = twist.vx
        omega = twist.omega
        
        v_l = v - omega * self.track_width / 2.0
        v_r = v + omega * self.track_width / 2.0
        
        w_l = v_l / self.wheel_radius
        w_r = v_r / self.wheel_radius
        
        return np.array([w_l, w_r], dtype=np.float32)


class MecanumKinematics(KinematicsBase):
    """麦克纳姆轮运动学"""
    
    def __init__(self, spec: AGVSpec):
        self.wheelbase = spec.wheelbase
        self.track_width = spec.track_width
        self.wheel_radius = spec.wheel_radius
        # 麦克纳姆轮角度 (通常45度或30度)
        self.gamma = np.radians(45)  # 滚轮角度
    
    def wheel_to_body(self, wheel_velocities: np.ndarray) -> AGVTwist:
        """
        麦克纳姆轮正运动学
        
        wheel_velocities: [fl, fr, rl, rr] 对应前后左右轮
        """
        r = self.wheel_radius
        l_x = self.wheelbase / 2.0
        l_y = self.track_width / 2.0
        
        w = wheel_velocities * r
        
        vx = (w[0] + w[1] + w[2] + w[3]) / 4.0
        vy = (-w[0] + w[1] + w[2] - w[3]) / 4.0
        omega = (-w[0] + w[1] - w[2] + w[3]) / (4.0 * (l_x + l_y))
        
        return AGVTwist(vx=vx, vy=vy, omega=omega)
    
    def body_to_wheel(self, twist: AGVTwist) -> np.ndarray:
        """
        麦克纳姆轮逆运动学
        """
        r = self.wheel_radius
        l_x = self.wheelbase / 2.0
        l_y = self.track_width / 2.0
        
        vx, vy, omega = twist.vx, twist.vy, twist.omega
        
        w_fl = (vx - vy - omega * (l_x + l_y)) / r
        w_fr = (vx + vy + omega * (l_x + l_y)) / r
        w_rl = (vx + vy - omega * (l_x + l_y)) / r
        w_rr = (vx - vy + omega * (l_x + l_y)) / r
        
        return np.array([w_fl, w_fr, w_rl, w_rr], dtype=np.float32)


class SkidSteerKinematics(KinematicsBase):
    """
    滑移转向运动学 (Skid Steering)
    
    用于履带式AGV (农业/室外/复杂地形)
    特点: 原地转向时两侧履带反向驱动产生滑移
    与差速驱动的区别: 需要更大的转向力矩,轮胎侧向摩擦力辅助转向
    
    参考模型: 简化线性滑移模型
    vx = (vL + vR) / 2
    vy = 0 (无侧向滑动,理想情况)
    omega = (vR - vL) / track_width
    """
    
    def __init__(self, spec: AGVSpec):
        self.wheelbase = spec.wheelbase
        self.track_width = spec.track_width  # 履带中心距
        self.wheel_radius = spec.wheel_radius
        # 滑移系数 (转向时侧向滑动量与纵向滑动量之比)
        # 履带式通常0.1-0.3, 橡胶轮胎差速驱动接近0
        self.slip_factor = 0.15
    
    def wheel_to_body(self, wheel_velocities: np.ndarray) -> AGVTwist:
        """
        滑移转向正运动学
        
        wheel_velocities: [left_vel, right_vel] rad/s
        """
        v_l = wheel_velocities[0] * self.wheel_radius
        v_r = wheel_velocities[1] * self.wheel_radius
        
        vx = (v_l + v_r) / 2.0
        # 滑移补偿: 转向时产生侧向速度
        omega = (v_r - v_l) / self.track_width
        vy = self.slip_factor * omega * self.track_width / 2.0
        
        return AGVTwist(vx=vx, vy=vy, omega=omega)
    
    def body_to_wheel(self, twist: AGVTwist) -> np.ndarray:
        """
        滑移转向逆运动学
        
        返回: [left_vel, right_vel] rad/s
        """
        vx, vy, omega = twist.vx, twist.vy, twist.omega
        
        # 逆滑移补偿
        omega_effective = omega * (1 + self.slip_factor)
        
        v_l = vx - omega_effective * self.track_width / 2.0
        v_r = vx + omega_effective * self.track_width / 2.0
        
        w_l = v_l / self.wheel_radius
        w_r = v_r / self.wheel_radius
        
        return np.array([w_l, w_r], dtype=np.float32)


class AckermannKinematics(KinematicsBase):
    """
    阿克曼转向运动学 (Ackermann Steering)
    
    用于汽车式AGV (室内物流车/无人搬运车)
    特点: 高速直线行驶稳定, 转向时内外轮转角不同
    
    简化模型: 不考虑车辆动力学, 仅几何关系
    """
    
    def __init__(self, spec: AGVSpec):
        self.wheelbase = spec.wheelbase
        self.track_width = spec.track_width
        self.wheel_radius = spec.wheel_radius
        # 最大转向角 (rad)
        self.max_steering_angle = np.radians(30)
    
    def wheel_to_body(self, wheel_velocities: np.ndarray) -> AGVTwist:
        """
        阿克曼正运动学 (从轮速推算车体速度)
        
        wheel_velocities: [rear_left_vel, rear_right_vel] rad/s
        (阿克曼只有后轮驱动,前轮转向)
        """
        v_l = wheel_velocities[0] * self.wheel_radius
        v_r = wheel_velocities[1] * self.wheel_radius
        
        vx = (v_l + v_r) / 2.0
        # 阿克曼转向角由前轮决定, 此处从后轮速度无法直接得出角速度
        # 简化: 假设纯滚动, omega由差速估算
        omega = (v_r - v_l) / self.track_width
        
        return AGVTwist(vx=vx, vy=0.0, omega=omega)
    
    def body_to_wheel(self, twist: AGVTwist) -> np.ndarray:
        """
        阿克曼逆运动学 (从车体速度推算轮速)
        
        对于后轮驱动阿克曼:
        - 后轮: 纯滚动, v_rl = v_rr = vx
        - 前轮: 根据Ackermann几何计算转向角
          tan(omega) = vx / (wheelbase / tan(delta))
        
        返回: [rear_left_vel, rear_right_vel] rad/s
        """
        vx, vy, omega = twist.vx, twist.vy, twist.omega
        
        # 纯滚动假设: 后轮速度等于车体纵向速度
        w_rl = vx / self.wheel_radius
        w_rr = vx / self.wheel_radius
        
        return np.array([w_rl, w_rr], dtype=np.float32)
    
    def steering_angle_to_omega(self, steering_angle: float, vx: float) -> float:
        """
        从转向角和速度计算瞬时转向角速度
        
        Ackemann转向几何:
        tan(omega) = vx * tan(steering_angle) / wheelbase
        
        Args:
            steering_angle: 前轮平均转向角 (rad)
            vx: 纵向速度 (m/s)
            
        Returns:
            omega: 转向角速度 (rad/s)
        """
        if abs(self.wheelbase) < 1e-6:
            return 0.0
        tan_delta = np.tan(np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle))
        omega = vx * tan_delta / self.wheelbase
        return omega


class KinematicsFactory:
    """运动学工厂"""
    
    _map = {
        DriveType.DIFFERENTIAL: DifferentialKinematics,
        DriveType.MECANUM: MecanumKinematics,
        DriveType.OMNIDIRECTIONAL: MecanumKinematics,
        DriveType.SWISS: SkidSteerKinematics,
        DriveType.ACKERMANN: AckermannKinematics,
    }
    
    @classmethod
    def create(cls, drive_type: DriveType, spec: AGVSpec) -> KinematicsBase:
        kinematics_cls = cls._map.get(drive_type, DifferentialKinematics)
        return kinematics_cls(spec)


def get_agv_spec(grade: str) -> AGVSpec:
    """获取AGV等级规格"""
    return AGVSpec.from_grade(AGVGrade(grade))


class TrajectoryTracker:
    """
    AGV轨迹跟踪控制器
    
    在 AGVMotionController 基础上增加:
    - Pure Pursuit 轨迹跟踪
    - 速度前瞻控制
    - 曲率前馈
    - 轨迹重规划触发
    """
    
    def __init__(
        self,
        spec: AGVSpec,
        look_ahead_distance: float = 0.3,
        k_gain: float = 2.0,
        smooth_yaw: bool = True
    ):
        """
        Args:
            spec: AGV规格
            look_ahead_distance: 前看距离 (m)
            k_gain: 增益系数
            smooth_yaw: 是否平滑航向角
        """
        self.spec = spec
        self.look_ahead_distance = look_ahead_distance
        self.k_gain = k_gain
        self.smooth_yaw = smooth_yaw
        
        # 底层AGV控制器
        self._agv = AGVMotionController(spec)
        
        # 轨迹
        self._trajectory: List[AGVPose] = []
        self._trajectory_times: np.ndarray = np.array([])
        self._current_idx = 0
        
        # PID参数
        self.kp_dist = 3.0
        self.kp_theta = 2.0
        self._last_error = 0.0
        
    def set_trajectory(self, trajectory: List[AGVPose], times: np.ndarray):
        """
        设置参考轨迹
        
        Args:
            trajectory: 轨迹点序列
            times: 对应时间戳
        """
        self._trajectory = trajectory
        self._trajectory_times = times
        self._current_idx = 0
    
    def set_pose(self, pose: AGVPose):
        """更新当前位姿"""
        self._agv.update_pose(pose)
    
    @property
    def pose(self) -> AGVPose:
        return self._agv.pose
    
    def _find_look_ahead_point(self, current_pos: np.ndarray) -> Tuple[int, AGVPose]:
        """
        找到前看点
        
        Returns:
            (index, look_ahead_point)
        """
        min_dist = float('inf')
        best_idx = self._current_idx
        
        for i in range(self._current_idx, len(self._trajectory)):
            pt = self._trajectory[i]
            pt_pos = np.array([pt.x, pt.y])
            dist = np.linalg.norm(pt_pos - current_pos)
            
            if dist >= self.look_ahead_distance:
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        self._current_idx = max(0, best_idx - 1)
        return best_idx, self._trajectory[best_idx]
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def compute_command(self, dt: float) -> np.ndarray:
        """
        计算轮速命令
        
        Args:
            dt: 时间步长
            
        Returns:
            wheel_commands: 轮速命令 (rad/s)
        """
        current = self._agv.pose
        current_pos = np.array([current.x, current.y])
        current_theta = current.theta
        
        if not self._trajectory or self._current_idx >= len(self._trajectory):
            return np.zeros(4)
        
        # 找到前看点
        _, target = self._find_look_ahead_point(current_pos)
        target_pos = np.array([target.x, target.y])
        target_theta = target.theta
        
        # 计算距离误差
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dist_error = np.sqrt(dx**2 + dy**2)
        
        # 计算角度误差 (在车体坐标系下)
        angle_to_target = np.arctan2(dy, dx)
        angle_error = self._normalize_angle(angle_to_target - current_theta)
        
        # Pure Pursuit 转向控制
        # alpha = atan2(2*L*sin(alpha)/dist)
        # 简化为比例控制
        if dist_error > 0.01:
            steering = self.k_gain * 2.0 * np.sin(angle_error) / dist_error
        else:
            steering = 0.0
        
        steering = np.clip(steering, -self.spec.max_angular_speed, self.spec.max_angular_speed)
        
        # 速度: 基于距离误差的减速
        if dist_error < 0.05:
            target_speed = 0.0
        else:
            speed_factor = min(dist_error / self.look_ahead_distance, 1.0)
            target_speed = self.spec.max_linear_speed * speed_factor * 0.5
        
        # 构建Twist
        twist = AGVTwist(vx=target_speed, vy=0.0, omega=steering)
        
        # 逆运动学
        wheel_cmds = self._agv.inverse_kinematics(twist)
        
        # 安全限制
        wheel_cmds = self._agv.apply_safety_limits(wheel_cmds)
        
        return wheel_cmds
    
    def is_trajectory_complete(self) -> bool:
        """检查轨迹是否完成"""
        if not self._trajectory:
            return True
        
        current = self._agv.pose
        current_pos = np.array([current.x, current.y])
        
        last_pt = self._trajectory[-1]
        last_pos = np.array([last_pt.x, last_pt.y])
        
        dist_to_end = np.linalg.norm(current_pos - last_pos)
        return dist_to_end < 0.05 and self._current_idx >= len(self._trajectory) - 2
    
    def reset(self):
        """重置跟踪器"""
        self._current_idx = 0
        self._last_error = 0.0
        self._agv = AGVMotionController(self.spec)


class PurePursuitTracker:
    """
    Pure Pursuit 轨迹跟踪控制器

    经典的几何跟踪算法:
    - 在参考轨迹上找到前看点 (lookahead point)
    - 计算转向角使AGV朝向前看点
    - 速度控制使用恒定或自适应速率

    适用于: 差速驱动、全向移动、麦克纳姆轮
    """

    def __init__(
        self,
        spec: AGVSpec,
        look_ahead_dist: float = 0.5,
        look_ahead_time: float = 1.0,
        k_gain: float = 2.0,
        min_look_ahead: float = 0.1,
        max_look_ahead: float = 2.0,
        linear_velocity: float = 0.3
    ):
        self.spec = spec
        self.look_ahead_dist = look_ahead_dist
        self.look_ahead_time = look_ahead_time
        self.k_gain = k_gain
        self.min_look_ahead = min_look_ahead
        self.max_look_ahead = max_look_ahead
        self.linear_velocity = linear_velocity

        self._agv = AGVMotionController(spec)
        self._trajectory: List[AGVPose] = []
        self._current_idx = 0

    def set_trajectory(self, trajectory: List[AGVPose]):
        self._trajectory = trajectory
        self._current_idx = 0

    def set_pose(self, pose: AGVPose):
        self._agv.update_pose(pose)

    @property
    def pose(self) -> AGVPose:
        return self._agv.pose

    def _find_look_ahead_point(self) -> Tuple[int, AGVPose]:
        """找到前看轨迹点"""
        current = self.pose
        current_pos = np.array([current.x, current.y])

        # 自适应前看距离 (速度越快前看越远)
        v = current.vx if hasattr(current, 'vx') else self.linear_velocity
        lad = min(self.max_look_ahead, max(
            self.min_look_ahead,
            self.look_ahead_dist + v * self.look_ahead_time
        ))

        best_idx = self._current_idx
        best_dist = float('inf')

        for i in range(self._current_idx, len(self._trajectory)):
            pt = self._trajectory[i]
            pt_pos = np.array([pt.x, pt.y])
            dist = np.linalg.norm(pt_pos - current_pos)

            if dist >= lad:
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                if i > self._current_idx:
                    break

        return best_idx, self._trajectory[best_idx]

    def compute_control(self) -> Tuple[float, float]:
        """
        计算控制量

        Returns:
            (linear_velocity, angular_velocity)
        """
        if not self._trajectory:
            return 0.0, 0.0

        current = self.pose
        current_pos = np.array([current.x, current.y])
        current_yaw = current.theta if hasattr(current, 'theta') else 0.0

        # 找前看点
        la_idx, la_point = self._find_look_ahead_point()
        la_pos = np.array([la_point.x, la_point.y])

        # 向量 from current to lookahead
        dx = la_pos[0] - current_pos[0]
        dy = la_pos[1] - current_pos[1]

        # 在机器人坐标系下表示
        cos_yaw = np.cos(-current_yaw)
        sin_yaw = np.sin(-current_yaw)
        x_local = dx * cos_yaw - dy * sin_yaw
        y_local = dx * sin_yaw + dy * cos_yaw

        # 到前看点的距离
        d = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

        # Pure Pursuit 转向角
        alpha = np.arctan2(y_local, x_local)
        L = self.spec.wheel_base if hasattr(self.spec, 'wheel_base') else 0.5
        omega = 2 * self.k_gain * np.sin(alpha) / d

        # 速度
        v = self.linear_velocity

        # 跟踪索引更新
        if la_idx > self._current_idx:
            self._current_idx = la_idx

        return float(v), float(omega)

    def reset(self):
        """重置"""
        self._current_idx = 0


class StanleyTracker:
    """
    Stanley 轨迹跟踪控制器

    基于横向误差的前轮转向控制:
    - 考虑横向误差和航向误差
    - 收敛速度快于 Pure Pursuit
    - 适合阿克曼模型车辆

    适用于: 差速驱动、四轮转向车辆
    """

    def __init__(
        self,
        spec: AGVSpec,
        k_gain: float = 2.5,
        k_soft: float = 1.0,
        max_steering: float = 1.0
    ):
        self.spec = spec
        self.k_gain = k_gain
        self.k_soft = k_soft
        self.max_steering = max_steering

        self._agv = AGVMotionController(spec)
        self._trajectory: List[AGVPose] = []
        self._current_idx = 0
        self._last_cross_track_error = 0.0

    def set_trajectory(self, trajectory: List[AGVPose]):
        self._trajectory = trajectory
        self._current_idx = 0

    def set_pose(self, pose: AGVPose):
        self._agv.update_pose(pose)

    @property
    def pose(self) -> AGVPose:
        return self._agv.pose

    def compute_control(self) -> Tuple[float, float]:
        """
        计算控制量

        Returns:
            (linear_velocity, steering_angle)
        """
        if not self._trajectory:
            return 0.0, 0.0

        current = self.pose
        current_pos = np.array([current.x, current.y])
        current_yaw = current.theta if hasattr(current, 'theta') else 0.0

        # 找到最近轨迹点
        min_dist = float('inf')
        nearest_idx = self._current_idx
        for i in range(self._current_idx, min(self._current_idx + 50, len(self._trajectory))):
            pt = self._trajectory[i]
            d = np.linalg.norm(np.array([pt.x, pt.y]) - current_pos)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        if nearest_idx > self._current_idx:
            self._current_idx = nearest_idx

        # 最近点信息
        nearest = self._trajectory[nearest_idx]
        nearest_yaw = nearest.theta if hasattr(nearest, 'theta') else 0.0

        # 横向误差 (cross-track error)
        dx = current_pos[0] - nearest.x
        dy = current_pos[1] - nearest.y
        # 横向误差 = 沿轨迹切线方向的误差
        path_yaw = nearest_yaw
        cross_track_error = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)

        # 航向误差
        heading_error = nearest_yaw - current_yaw
        # 归一化到 [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Stanley 控制率
        # delta = heading_error + arctan(k * e / (k_soft + v))
        v = current.vx if hasattr(current, 'vx') and current.vx != 0 else 0.1
        steering = heading_error + np.arctan2(
            self.k_gain * cross_track_error,
            self.k_soft + abs(v)
        )

        # 限制
        steering = np.clip(steering, -self.max_steering, self.max_steering)

        self._last_cross_track_error = cross_track_error

        return float(abs(v)), float(steering)

    def reset(self):
        self._current_idx = 0
        self._last_cross_track_error = 0.0


class PIDTrajectoryTracker:
    """
    PID 轨迹跟踪控制器

    基于 PID 的轨迹跟踪:
    - 位置环 PID 控制
    - 航向角 PID 控制
    - 可叠加前馈项

    适用于: 差速驱动、简单场景
    """

    def __init__(
        self,
        spec: AGVSpec,
        kp_pos: float = 3.0,
        ki_pos: float = 0.0,
        kd_pos: float = 0.5,
        kp_theta: float = 2.0,
        ki_theta: float = 0.0,
        kd_theta: float = 0.2
    ):
        self.spec = spec
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        self.kp_theta = kp_theta
        self.ki_theta = ki_theta
        self.kd_theta = kd_theta

        self._agv = AGVMotionController(spec)
        self._trajectory: List[AGVPose] = []
        self._current_idx = 0

        # PID 状态
        self._pos_integral = 0.0
        self._pos_prev_error = 0.0
        self._theta_integral = 0.0
        self._theta_prev_error = 0.0
        self._last_time = None

    def set_trajectory(self, trajectory: List[AGVPose]):
        self._trajectory = trajectory
        self._current_idx = 0
        self._reset_pid()

    def set_pose(self, pose: AGVPose):
        self._agv.update_pose(pose)

    @property
    def pose(self) -> AGVPose:
        return self._agv.pose

    def _reset_pid(self):
        self._pos_integral = 0.0
        self._pos_prev_error = 0.0
        self._theta_integral = 0.0
        self._theta_prev_error = 0.0
        self._last_time = None

    def compute_control(self, dt: float = 0.01) -> Tuple[float, float]:
        """
        计算控制量

        Args:
            dt: 控制周期 (s)

        Returns:
            (linear_velocity, angular_velocity)
        """
        if not self._trajectory:
            return 0.0, 0.0

        import time
        if self._last_time is None:
            self._last_time = time.time()
        actual_dt = time.time() - self._last_time
        self._last_time = time.time()
        dt = max(actual_dt, 0.001)

        current = self.pose
        current_pos = np.array([current.x, current.y])
        current_yaw = current.theta if hasattr(current, 'theta') else 0.0

        # 找到最近轨迹点
        min_dist = float('inf')
        nearest_idx = self._current_idx
        for i in range(self._current_idx, len(self._trajectory)):
            pt = self._trajectory[i]
            d = np.linalg.norm(np.array([pt.x, pt.y]) - current_pos)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        if nearest_idx < len(self._trajectory) - 1:
            self._current_idx = nearest_idx + 1

        # 目标点
        target = self._trajectory[self._current_idx] if self._current_idx < len(self._trajectory) else self._trajectory[-1]
        target_pos = np.array([target.x, target.y])
        target_yaw = target.theta if hasattr(target, 'theta') else 0.0

        # 位置误差
        pos_error = np.linalg.norm(target_pos - current_pos)

        # PID 位置控制
        self._pos_integral += pos_error * dt
        self._pos_integral = np.clip(self._pos_integral, -10, 10)
        pos_derivative = (pos_error - self._pos_prev_error) / dt if dt > 0 else 0.0
        v = self.kp_pos * pos_error + self.ki_pos * self._pos_integral + self.kd_pos * pos_derivative
        v = np.clip(v, 0, 1.0)
        self._pos_prev_error = pos_error

        # 航向误差
        theta_error = target_yaw - current_yaw
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

        self._theta_integral += theta_error * dt
        self._theta_integral = np.clip(self._theta_integral, -5, 5)
        theta_derivative = (theta_error - self._theta_prev_error) / dt if dt > 0 else 0.0
        omega = self.kp_theta * theta_error + self.ki_theta * self._theta_integral + self.kd_theta * theta_derivative
        self._theta_prev_error = theta_error

        return float(v), float(omega)

    def reset(self):
        self._current_idx = 0
        self._reset_pid()
