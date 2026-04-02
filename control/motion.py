"""
运动控制模块 (Motion Control)
运动学模型、轨迹规划、AGV控制器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Pose2D:
    """
    二维位姿

    Attributes:
        x: X坐标 (m)
        y: Y坐标 (m)
        theta: 航向角 (rad)
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def to_array(self) -> np.ndarray:
        """转换为数组"""
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Pose2D':
        """从数组创建"""
        return cls(x=arr[0], y=arr[1], theta=arr[2])

    def distance_to(self, other: 'Pose2D') -> float:
        """到另一点的几何距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle_to(self, other: 'Pose2D') -> float:
        """到另一点的航向角"""
        return np.arctan2(other.y - self.y, other.x - self.x)

    def __add__(self, other: 'Pose2D') -> 'Pose2D':
        return Pose2D(x=self.x + other.x, y=self.y + other.y, theta=self.theta + other.theta)

    def __sub__(self, other: 'Pose2D') -> 'Pose2D':
        return Pose2D(x=self.x - other.x, y=self.y - other.y, theta=self.theta - other.theta)

    def __repr__(self) -> str:
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, theta={np.degrees(self.theta):.1f}°)"


@dataclass
class Twist2D:
    """
    二维速度

    Attributes:
        vx: X方向线速度 (m/s)
        vy: Y方向线速度 (m/s)
        omega: 角速度 (rad/s)
    """
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0

    def to_array(self) -> np.ndarray:
        """转换为数组"""
        return np.array([self.vx, self.vy, self.omega])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Twist2D':
        """从数组创建"""
        return cls(vx=arr[0], vy=arr[1], omega=arr[2])

    def speed(self) -> float:
        """获取线速度大小"""
        return np.sqrt(self.vx**2 + self.vy**2)

    def __repr__(self) -> str:
        return f"Twist2D(vx={self.vx:.3f}, vy={self.vy:.3f}, omega={np.degrees(self.omega):.1f}°/s)"


class KinematicsModel(ABC):
    """运动学模型基类"""

    def __init__(self, name: str = "KinematicsModel"):
        self.name = name

    @abstractmethod
    def forward(self, wheel_velocities: np.ndarray) -> Twist2D:
        """
        正运动学: 轮速 -> 机器人速度

        Args:
            wheel_velocities: 轮速数组 [rad/s]

        Returns:
            Twist2D: 机器人速度
        """
        pass

    @abstractmethod
    def inverse(self, twist: Twist2D) -> np.ndarray:
        """
        逆运动学: 机器人速度 -> 轮速

        Args:
            twist: 机器人速度

        Returns:
            轮速数组 [rad/s]
        """
        pass

    @abstractmethod
    def integrate(self, pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D:
        """
        积分更新位姿

        Args:
            pose: 当前位姿
            twist: 速度
            dt: 时间步长

        Returns:
            更新后的位姿
        """
        pass


class DifferentialDrive(KinematicsModel):
    """
    差速驱动运动学模型
    两轮差速: 左右轮速决定机器人运动
    """

    def __init__(
        self,
        name: str = "DifferentialDrive",
        wheel_separation: float = 0.5,  # 轮间距 (m)
        wheel_radius: float = 0.1  # 轮半径 (m)
    ):
        super().__init__(name)
        self.wheel_separation = wheel_separation
        self.wheel_radius = wheel_radius

    def forward(self, wheel_velocities: np.ndarray) -> Twist2D:
        """
        正运动学

        Args:
            wheel_velocities: [omega_left, omega_right] (rad/s)

        Returns:
            Twist2D: [vx, vy, omega]
        """
        omega_l, omega_r = wheel_velocities[0], wheel_velocities[1]

        # 线速度
        v_l = omega_l * self.wheel_radius
        v_r = omega_r * self.wheel_radius
        vx = (v_l + v_r) / 2.0
        vy = 0.0  # 差速机器人不能侧向移动

        # 角速度
        omega = (v_r - v_l) / self.wheel_separation

        return Twist2D(vx=vx, vy=vy, omega=omega)

    def inverse(self, twist: Twist2D) -> np.ndarray:
        """
        逆运动学

        Args:
            twist: [vx, vy, omega]

        Returns:
            [omega_left, omega_right] (rad/s)
        """
        vx, omega = twist.vx, twist.omega

        # 差速机器人侧向速度为0
        v_l = vx - omega * self.wheel_separation / 2.0
        v_r = vx + omega * self.wheel_separation / 2.0

        # 转换为角速度
        omega_l = v_l / self.wheel_radius
        omega_r = v_r / self.wheel_radius

        return np.array([omega_l, omega_r])

    def integrate(self, pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D:
        """
        积分更新位姿

        公式:
        x' = x + vx*cos(theta)*dt
        y' = y + vx*sin(theta)*dt
        theta' = theta + omega*dt
        """
        x = pose.x + twist.vx * np.cos(pose.theta) * dt
        y = pose.y + twist.vx * np.sin(pose.theta) * dt
        theta = pose.theta + twist.omega * dt

        # 角度归一化到 [-pi, pi]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        return Pose2D(x=x, y=y, theta=theta)


class MecanumDrive(KinematicsModel):
    """
    Mecanum轮运动学模型 (全向移动)
    四个Mecanum轮呈45°安装
    """

    def __init__(
        self,
        name: str = "MecanumDrive",
        wheelbase_x: float = 0.5,  # X方向轮间距 (m)
        wheelbase_y: float = 0.5,  # Y方向轮间距 (m)
        wheel_radius: float = 0.1,  # 轮半径 (m)
        gear_ratio: float = 1.0  # 齿轮比
    ):
        super().__init__(name)
        self.wheelbase_x = wheelbase_x
        self.wheelbase_y = wheelbase_y
        self.wheel_radius = wheel_radius
        self.gear_ratio = gear_ratio

        # 运动学矩阵 (简化)
        self._matrix = np.array([
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [-1/(wheelbase_x/2 + wheelbase_y/2),
             1/(wheelbase_x/2 + wheelbase_y/2),
             -1/(wheelbase_x/2 + wheelbase_y/2),
             1/(wheelbase_x/2 + wheelbase_y/2)]
        ]) / self.wheel_radius

    def forward(self, wheel_velocities: np.ndarray) -> Twist2D:
        """
        正运动学

        Args:
            wheel_velocities: [omega_fl, omega_fr, omega_rl, omega_rr] (rad/s)

        Returns:
            Twist2D: [vx, vy, omega]
        """
        # 简化的正运动学
        v_fl = wheel_velocities[0] * self.wheel_radius
        v_fr = wheel_velocities[1] * self.wheel_radius
        v_rl = wheel_velocities[2] * self.wheel_radius
        v_rr = wheel_velocities[3] * self.wheel_radius

        vx = (v_fl + v_fr + v_rl + v_rr) / 4.0
        vy = (-v_fl + v_fr + v_rl - v_rr) / 4.0

        L = self.wheelbase_x / 2 + self.wheelbase_y / 2
        omega = (-v_fl + v_fr - v_rl + v_rr) / (4 * L)

        return Twist2D(vx=vx, vy=vy, omega=omega)

    def inverse(self, twist: Twist2D) -> np.ndarray:
        """
        逆运动学

        Args:
            twist: [vx, vy, omega]

        Returns:
            [omega_fl, omega_fr, omega_rl, omega_rr] (rad/s)
        """
        vx, vy, omega = twist.vx, twist.vy, twist.omega
        r = self.wheel_radius
        L = self.wheelbase_x / 2 + self.wheelbase_y / 2

        omega_fl = (vx + vy + omega * L) / r
        omega_fr = (vx - vy - omega * L) / r
        omega_rl = (vx - vy + omega * L) / r
        omega_rr = (vx + vy - omega * L) / r

        return np.array([omega_fl, omega_fr, omega_rl, omega_rr])

    def integrate(self, pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D:
        """
        积分更新位姿 (全向移动)
        """
        cos_t, sin_t = np.cos(pose.theta), np.sin(pose.theta)

        # 在机器人坐标系的速度
        v_local = np.array([twist.vx * cos_t + twist.vy * sin_t,
                           -twist.vx * sin_t + twist.vy * cos_t])

        x = pose.x + twist.vx * cos_t * dt - twist.vy * sin_t * dt
        y = pose.y + twist.vx * sin_t * dt + twist.vy * cos_t * dt
        theta = pose.theta + twist.omega * dt

        theta = np.arctan2(np.sin(theta), np.cos(theta))

        return Pose2D(x=x, y=y, theta=theta)


class TrajectoryPlanner:
    """
    轨迹规划器
    支持直线、圆弧、贝塞尔曲线
    """

    def __init__(self, name: str = "TrajectoryPlanner"):
        self.name = name

    def plan_line(
        self,
        start: Pose2D,
        end: Pose2D,
        max_velocity: float = 1.0,
        max_acceleration: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        规划直线轨迹

        Args:
            start: 起始位姿
            end: 终止位姿
            max_velocity: 最大速度
            max_acceleration: 最大加速度

        Returns:
            轨迹字典包含时间、位置、速度、加速度数组
        """
        # 计算距离
        distance = start.distance_to(end)
        direction = start.angle_to(end)

        if distance < 1e-6:
            return {'t': np.array([0]), 'positions': np.array([start.to_array()]),
                    'velocities': np.array([[0, 0, 0]]), 'accelerations': np.array([[0, 0, 0]])}

        # 梯形速度规划
        # t_accel = max_velocity / max_acceleration
        # d_accel = 0.5 * max_acceleration * t_accel**2
        # if 2 * d_accel > distance:
        #     # 三角形速度曲线
        #     t_accel = np.sqrt(distance / max_acceleration)
        #     max_velocity = max_acceleration * t_accel

        # 时间
        t_accel = max_velocity / max_acceleration
        d_accel = 0.5 * max_acceleration * t_accel**2

        if distance <= 2 * d_accel:
            # 三角形
            t_accel = np.sqrt(distance / max_acceleration)
            t_total = 2 * t_accel
        else:
            # 梯形
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / max_velocity
            t_total = 2 * t_accel + t_cruise

        # 生成轨迹点
        dt = 0.01
        t = np.arange(0, t_total + dt, dt)

        positions = []
        velocities = []
        accelerations = []

        for tt in t:
            # 归一化进度
            s = tt / t_total

            # 位置 (x, y, theta)
            x = start.x + (end.x - start.x) * s
            y = start.y + (end.y - start.y) * s
            theta = direction  # 航向保持

            positions.append([x, y, theta])

            # 速度
            if tt < t_accel:
                v = max_acceleration * tt
                a = max_acceleration
            elif tt < t_total - t_accel:
                v = max_velocity
                a = 0
            else:
                v = max_velocity - max_acceleration * (tt - (t_total - t_accel))
                a = -max_acceleration

            velocities.append([v * np.cos(direction), v * np.sin(direction), 0])
            accelerations.append([a * np.cos(direction), a * np.sin(direction), 0])

        return {
            't': t,
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations)
        }

    def plan_arc(
        self,
        start: Pose2D,
        center: Tuple[float, float],
        angle: float,  # 弧度
        max_velocity: float = 1.0,
        max_acceleration: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        规划圆弧轨迹

        Args:
            start: 起始位姿
            center: 圆心 (x, y)
            angle: 弧度 (正=逆时针)
            max_velocity: 最大速度
            max_acceleration: 最大加速度

        Returns:
            轨迹字典
        """
        # 计算半径
        radius = np.sqrt((start.x - center[0])**2 + (start.y - center[1])**2)

        if radius < 1e-6:
            return self.plan_line(start, Pose2D(x=start.x + 0.1, y=start.y), max_velocity, max_acceleration)

        # 弧长
        arc_length = abs(angle) * radius

        # 时间
        t_total = arc_length / max_velocity

        # 生成轨迹点
        dt = 0.01
        t = np.arange(0, t_total + dt, dt)

        positions = []
        velocities = []
        accelerations = []

        for tt in t:
            s = tt / t_total
            current_angle = angle * s

            # 角度计算
            start_angle = np.arctan2(start.y - center[1], start.x - center[0])
            current_pos_angle = start_angle + current_angle

            x = center[0] + radius * np.cos(current_pos_angle)
            y = center[1] + radius * np.sin(current_pos_angle)
            theta = current_pos_angle + np.pi / 2 * np.sign(angle)  # 切向

            positions.append([x, y, theta])

            # 速度
            v = max_velocity * np.ones(3)
            v[2] = 0
            velocities.append(v)
            accelerations.append([0, 0, 0])

        return {
            't': t,
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations)
        }


class MotionController:
    """
    运动控制器
    基于轨迹跟踪的控制器
    """

    def __init__(
        self,
        kinematics: KinematicsModel,
        name: str = "MotionController"
    ):
        self.kinematics = kinematics
        self.name = name

        # PID参数
        self._pos_kp = 1.0
        self._pos_ki = 0.0
        self._pos_kd = 0.1

        # 轨迹
        self._trajectory: Optional[Dict] = None
        self._trajectory_index = 0

    def set_trajectory(self, trajectory: Dict[str, np.ndarray]):
        """设置轨迹"""
        self._trajectory = trajectory
        self._trajectory_index = 0

    def compute_twist(self, current_pose: Pose2D, dt: float) -> Twist2D:
        """
        计算速度指令

        Args:
            current_pose: 当前位姿
            dt: 时间步长

        Returns:
            Twist2D: 目标速度
        """
        if self._trajectory is None:
            return Twist2D()

        t = self._trajectory['t']
        positions = self._trajectory['positions']

        if self._trajectory_index >= len(positions) - 1:
            return Twist2D()  # 轨迹结束

        # 获取当前目标点
        target_pos = positions[self._trajectory_index]

        # 位置误差
        error_x = target_pos[0] - current_pose.x
        error_y = target_pos[1] - current_pose.y

        # 简单的P控制
        vx = self._pos_kp * error_x
        vy = self._pos_kp * error_y

        # 角度误差
        target_theta = target_pos[2]
        error_theta = target_theta - current_pose.theta
        error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))
        omega = self._pos_kp * error_theta

        return Twist2D(vx=vx, vy=vy, omega=omega)

    def advance(self) -> bool:
        """
        前进到轨迹下一时刻

        Returns:
            是否还有轨迹点
        """
        if self._trajectory is None:
            return False

        self._trajectory_index += 1
        return self._trajectory_index < len(self._trajectory['t'])

    def is_trajectory_complete(self) -> bool:
        """检查轨迹是否完成"""
        if self._trajectory is None:
            return True
        return self._trajectory_index >= len(self._trajectory['t'])


class AGVController:
    """
    AGV专用控制器
    集成运动学模型、轨迹规划、轮速控制
    """

    def __init__(
        self,
        name: str = "AGVController",
        kinematics: Optional[KinematicsModel] = None,
        wheel_separation: float = 0.5,
        wheel_radius: float = 0.1
    ):
        self.name = name

        # 运动学模型
        if kinematics is None:
            self.kinematics = DifferentialDrive(
                wheel_separation=wheel_separation,
                wheel_radius=wheel_radius
            )
        else:
            self.kinematics = kinematics

        # 当前状态
        self._pose = Pose2D()
        self._twist = Twist2D()

        # 轨迹规划器
        self._planner = TrajectoryPlanner()

        # 轨迹
        self._trajectory: Optional[Dict] = None
        self._trajectory_time = 0.0

        # 限制参数
        self.max_velocity = 1.0  # m/s
        self.max_acceleration = 1.0  # m/s²
        self.max_omega = 2.0  # rad/s

    def set_target_pose(self, pose: Pose2D):
        """设置目标位姿"""
        # 规划直线轨迹
        self._trajectory = self._planner.plan_line(
            self._pose, pose,
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration
        )
        self._trajectory_time = 0.0

    def set_target_twist(self, twist: Twist2D):
        """设置目标速度"""
        self._twist = Twist2D(
            vx=np.clip(twist.vx, -self.max_velocity, self.max_velocity),
            vy=np.clip(twist.vy, -self.max_velocity, self.max_velocity),
            omega=np.clip(twist.omega, -self.max_omega, self.max_omega)
        )

    def step(self, dt: float) -> np.ndarray:
        """
        步进控制

        Args:
            dt: 时间步长 (s)

        Returns:
            轮速数组 [rad/s]
        """
        if self._trajectory is not None and self._trajectory_time < self._trajectory['t'][-1]:
            # 轨迹跟踪模式
            # 找到当前时间的轨迹点
            t_array = self._trajectory['t']
            idx = np.searchsorted(t_array, self._trajectory_time, side='right') - 1
            idx = max(0, min(idx, len(t_array) - 2))

            # 获取目标速度和位置
            if idx < len(self._trajectory['velocities']):
                vel = self._trajectory['velocities'][idx]
                self._twist = Twist2D(vx=vel[0], vy=vel[1], omega=vel[2])

            self._trajectory_time += dt
        else:
            # 直接速度模式
            pass

        # 积分更新位姿
        self._pose = self.kinematics.integrate(self._pose, self._twist, dt)

        # 逆运动学计算轮速
        wheel_velocities = self.kinematics.inverse(self._twist)

        return wheel_velocities

    def move_to(self, x: float, y: float, theta: float = None, dt: float = 0.01) -> np.ndarray:
        """
        移动到目标位置

        Args:
            x: 目标X (m)
            y: 目标Y (m)
            theta: 目标航向角 (rad), 如果为None则不控制朝向
            dt: 时间步长

        Returns:
            各时刻的轮速数组
        """
        # 目标位姿
        if theta is None:
            theta = self._pose.theta  # 保持当前朝向

        target = Pose2D(x=x, y=y, theta=theta)

        # 规划轨迹
        self.set_target_pose(target)

        # 执行轨迹
        wheel_velocities_history = []
        while not (self._trajectory_time >= self._trajectory['t'][-1] if self._trajectory else True):
            wheel_vel = self.step(dt)
            wheel_velocities_history.append(wheel_vel)

            # 检查是否接近目标
            if self._pose.distance_to(target) < 0.01:
                break

        return np.array(wheel_velocities_history)

    def stop(self) -> np.ndarray:
        """
        停止 (减速到0)

        Returns:
            停止前的轮速
        """
        self._twist = Twist2D()
        return self.kinematics.inverse(self._twist)

    def get_state(self) -> Dict[str, any]:
        """
        获取AGV状态

        Returns:
            状态字典
        """
        return {
            'pose': self._pose,
            'twist': self._twist,
            'speed': self._twist.speed(),
            'trajectory_progress': self._trajectory_time / self._trajectory['t'][-1] if self._trajectory else 0
        }

    def set_pose(self, x: float, y: float, theta: float):
        """设置位姿 (用于定位)"""
        self._pose = Pose2D(x=x, y=y, theta=theta)

    def reset(self):
        """重置控制器"""
        self._pose = Pose2D()
        self._twist = Twist2D()
        self._trajectory = None
        self._trajectory_time = 0.0
