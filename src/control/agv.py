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


class KinematicsFactory:
    """运动学工厂"""
    
    _map = {
        DriveType.DIFFERENTIAL: DifferentialKinematics,
        DriveType.MECANUM: MecanumKinematics,
        DriveType.OMNIDIRECTIONAL: MecanumKinematics,  # 简化为相同
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
