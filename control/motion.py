"""
运动控制模块 (Motion Control)
支持AGV运动学模型、轨迹规划、导航控制
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class WheelType(Enum):
    """轮子类型"""
    DIFFENTIAL = "differential"    # 差速驱动
    MECANUM = "mecanum"           # 全向轮(Mecanum)
    OMNI = "omni"                 # 全向轮(Omni)
    STEERING = "steering"         # 转向驱动
    SWERVE = "swerve"            # Swerve驱动


@dataclass
class Pose2D:
    """2D位姿"""
    x: float = 0.0          # X坐标 (m)
    y: float = 0.0          # Y坐标 (m)
    theta: float = 0.0      # 朝向角 (rad)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Pose2D':
        return cls(x=arr[0], y=arr[1], theta=arr[2])

    def transform(self, dx: float, dy: float, dtheta: float) -> 'Pose2D':
        """相对变换"""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        new_x = self.x + cos_t * dx - sin_t * dy
        new_y = self.y + sin_t * dx + cos_t * dy
        new_theta = self.theta + dtheta
        return Pose2D(x=new_x, y=new_y, theta=new_theta)


@dataclass
class Twist2D:
    """2D速度"""
    vx: float = 0.0   # X方向线速度 (m/s)
    vy: float = 0.0   # Y方向线速度 (m/s)
    omega: float = 0.0  # 角速度 (rad/s)

    def to_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.omega])


class KinematicsModel:
    """运动学模型基类"""

    def __init__(self, wheel_type: WheelType):
        self.wheel_type = wheel_type

    def forward(self, wheel_velocities: np.ndarray) -> Twist2D:
        """正运动学: 轮速 -> 机器人速度"""
        raise NotImplementedError

    def inverse(self, twist: Twist2D) -> np.ndarray:
        """逆运动学: 机器人速度 -> 轮速"""
        raise NotImplementedError

    def integrate(self, pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D:
        """积分更新位姿"""
        dx = twist.vx * dt
        dy = twist.vy * dt
        dtheta = twist.omega * dt
        return pose.transform(dx, dy, dtheta)


class DifferentialDrive(KinematicsModel):
    """差速驱动模型"""

    def __init__(self, wheel_separation: float = 0.5, wheel_radius: float = 0.1):
        super().__init__(WheelType.DIFFERENTIAL)
        self.L = wheel_separation  # 左右轮间距
        self.r = wheel_radius      # 轮子半径

    def forward(self, wheel_velocities: np.ndarray) -> Twist2D:
        """轮速 -> 机器人速度"""
        v_l, v_r = wheel_velocities[0], wheel_velocities[1]
        v = self.r * (v_r + v_l) / 2
        omega = self.r * (v_r - v_l) / self.L
        return Twist2D(vx=v, vy=0, omega=omega)

    def inverse(self, twist: Twist2D) -> np.ndarray:
        """机器人速度 -> 轮速"""
        v, omega = twist.vx, twist.omega
        v_r = (v + omega * self.L / 2) / self.r
        v_l = (v - omega * self.L / 2) / self.r
        return np.array([v_l, v_r])


class MecanumDrive(KinematicsModel):
    """Mecanum全向驱动模型"""

    def __init__(self, wheel_radius: float = 0.1, 
                 lx: float = 0.3, ly: float = 0.3):
        super().__init__(WheelType.MECANUM)
        self.r = wheel_radius
        self.lx = lx  # X方向半轴距
        self.ly = ly  # Y方向半轴距

    def forward(self, wheel_velocities: np.ndarray) -> Twist2D:
        """轮速(4个) -> 机器人速度"""
        r = self.r
        lx, ly = self.lx, self.ly
        w1, w2, w3, w4 = wheel_velocities

        vx = r * (w1 + w2 + w3 + w4) / 4
        vy = r * (-w1 + w2 + w3 - w4) / 4
        omega = r * (-w1 + w2 - w3 + w4) / (4 * (lx + ly))
        return Twist2D(vx=vx, vy=vy, omega=omega)

    def inverse(self, twist: Twist2D) -> np.ndarray:
        """机器人速度 -> 4个轮速"""
        r = self.r
        lx, ly = self.lx, self.ly
        vx, vy, omega = twist.vx, twist.vy, twist.omega

        w1 = (vx - vy - omega * (lx + ly)) / r
        w2 = (vx + vy + omega * (lx + ly)) / r
        w3 = (vx + vy - omega * (lx + ly)) / r
        w4 = (vx - vy + omega * (lx + ly)) / r
        return np.array([w1, w2, w3, w4])


class TrajectoryPlanner:
    """轨迹规划器"""

    def __init__(self, max_velocity: float = 1.0, max_acceleration: float = 1.0):
        self.max_v = max_velocity
        self.max_a = max_acceleration

    def trapezoidal_profile(self, start: float, end: float, duration: float, 
                            dt: float) -> np.ndarray:
        """梯形速度曲线"""
        n = int(duration / dt)
        positions = np.linspace(start, end, n)
        
        # 计算速度轮廓
        v_max = (end - start) / duration
        v_max = min(v_max, self.max_v)
        
        # 简化: 匀加速后匀减速
        t_accel = v_max / self.max_a
        positions = np.zeros(n)
        
        for i in range(n):
            t = i * dt
            if t < t_accel:
                s = 0.5 * self.max_a * t * t
            elif t < duration - t_accel:
                s = v_max * t - 0.5 * v_max * t_accel
            else:
                t_rem = duration - t
                s = (end - start) - 0.5 * self.max_a * t_rem * t_rem
            positions[i] = start + (end - start) * s / (end - start) if end != start else start
        
        return np.clip(positions, min(start, end), max(start, end))

    def circular_interpolation(self, center: Tuple[float, float], 
                              radius: float, start_angle: float,
                              end_angle: float, dt: float) -> np.ndarray:
        """圆弧插值"""
        angles = np.linspace(start_angle, end_angle, int(abs(end_angle - start_angle) / (dt * 0.5)))
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.stack([x, y], axis=1)

    def line_segment(self, start: Tuple[float, float], 
                    end: Tuple[float, float], dt: float) -> np.ndarray:
        """直线段"""
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        n = max(int(dist / (dt * self.max_v)), 2)
        points = np.linspace(start, end, n)
        return points


class MotionController:
    """AGV运动控制器"""

    def __init__(self, kinematics: KinematicsModel):
        self.kinematics = kinematics
        self.current_pose = Pose2D()
        self.target_pose = Pose2D()
        self.current_twist = Twist2D()
        self.planner = TrajectoryPlanner()
        self._trajectory: Optional[np.ndarray] = None
        self._traj_idx = 0

    def set_target_pose(self, pose: Pose2D):
        """设置目标位姿"""
        self.target_pose = pose

    def set_target_twist(self, twist: Twist2D):
        """设置目标速度"""
        self.target_twist = twist
        self.current_twist = twist

    def compute_error(self) -> Tuple[float, float, float]:
        """计算位姿误差"""
        dx = self.target_pose.x - self.current_pose.x
        dy = self.target_pose.y - self.current_pose.y
        dtheta = self.target_pose.theta - self.current_pose.theta
        # 角度归一化
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        return dx, dy, dtheta

    def pose_feedback_control(self, kp: float = 1.0) -> Twist2D:
        """位姿反馈控制 (简单P控制)"""
        dx, dy, dtheta = self.compute_error()
        
        # 在机器人坐标系下表示误差
        cos_t = np.cos(self.current_pose.theta)
        sin_t = np.sin(self.current_pose.theta)
        err_x = cos_t * dx + sin_t * dy
        err_y = -sin_t * dx + cos_t * dy
        
        vx = kp * err_x
        omega = kp * dtheta
        
        return Twist2D(vx=vx, vy=0, omega=omega)

    def step(self, dt: float) -> np.ndarray:
        """执行一步控制, 返回轮速"""
        # 更新位姿
        twist = self.pose_feedback_control()
        self.current_pose = self.kinematics.integrate(self.current_pose, twist, dt)
        self.current_twist = twist
        
        # 逆运动学得到轮速
        wheel_vels = self.kinematics.inverse(twist)
        return wheel_vels

    def follow_trajectory(self, trajectory: np.ndarray, dt: float) -> np.ndarray:
        """沿轨迹运动"""
        if self._traj_idx >= len(trajectory):
            return np.zeros(4)  # 停止
        
        self.target_pose.x = trajectory[self._traj_idx, 0]
        self.target_pose.y = trajectory[self._traj_idx, 1]
        self._traj_idx += 1
        
        return self.step(dt)

    def move_to(self, x: float, y: float, theta: float, dt: float) -> np.ndarray:
        """移动到目标点"""
        self.target_pose = Pose2D(x=x, y=y, theta=theta)
        return self.step(dt)

    def get_state(self) -> Dict:
        """获取状态"""
        return {
            "pose": self.current_pose.to_array(),
            "twist": self.current_twist.to_array(),
            "target": self.target_pose.to_array(),
            "error": self.compute_error()
        }


class AGVController(MotionController):
    """AGV专用控制器"""

    def __init__(self, agv_model: str = "差速AGV", **kwargs):
        wheel_separation = kwargs.get("wheel_separation", 0.5)
        wheel_radius = kwargs.get("wheel_radius", 0.1)
        kinematics = DifferentialDrive(wheel_separation, wheel_radius)
        super().__init__(kinematics)

        self.agv_model = agv_model
        self.max_linear_speed = kwargs.get("max_linear_speed", 1.5)  # m/s
        self.max_angular_speed = kwargs.get("max_angular_speed", 2.0)  # rad/s
        self.safety_radius = kwargs.get("safety_radius", 0.5)  # m

    def limit_twist(self, twist: Twist2D) -> Twist2D:
        """限幅"""
        v_mag = np.sqrt(twist.vx**2 + twist.vy**2)
        if v_mag > self.max_linear_speed:
            scale = self.max_linear_speed / v_mag
            twist.vx *= scale
            twist.vy *= scale
        twist.omega = np.clip(twist.omega, -self.max_angular_speed, self.max_angular_speed)
        return twist

    def step(self, dt: float) -> np.ndarray:
        """执行控制步"""
        twist = self.pose_feedback_control()
        twist = self.limit_twist(twist)
        self.current_twist = twist
        self.current_pose = self.kinematics.integrate(self.current_pose, twist, dt)
        return self.kinematics.inverse(twist)

    def stop(self) -> np.ndarray:
        """停止"""
        return np.zeros(2)  # 差速驱动2个轮子
