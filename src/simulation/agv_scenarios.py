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
AGV 仿真场景模块
===============

AGV 特化仿真场景:
- 物料运输 (Transport)
- 路径导航 (Navigation)
- 多机协同 (Multi-AGV)
- 自主充电 (Docking)
- 避障导航 (Obstacle Avoidance)

支持 Gymnasium 接口，可与 RL 训练框架集成
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Callable
import time


# =============================================================================
# AGV 物理参数
# =============================================================================

@dataclass
class AGVPhysicsConfig:
    """AGV 物理配置"""
    # 运动学
    wheel_base: float = 0.5          # 前后轮轴距 (m)
    wheel_radius: float = 0.1         # 轮子半径 (m)
    max_linear_speed: float = 2.0     # 最大线速度 (m/s)
    max_angular_speed: float = 2.0    # 最大角速度 (rad/s)
    max_accel: float = 1.0            # 最大加速度 (m/s^2)
    max_angular_accel: float = 3.0    # 最大角加速度 (rad/s^2)

    # 质量/惯性
    mass: float = 50.0                # AGV 质量 (kg)
    inertia_z: float = 20.0           # Z轴转动惯量 (kg·m^2)

    # 摩擦/阻力
    rolling_friction: float = 0.02    # 滚动阻力
    rotational_friction: float = 0.5   # 旋转阻力

    # 里程计噪声
    odom_noise_linear: float = 0.01   # 线速度噪声 (m/s)
    odom_noise_angular: float = 0.02   # 角速度噪声 (rad/s)

    # 控制延迟
    control_delay: float = 0.05        # 控制延迟 (s)

    # AGV 等级
    grade: str = 'M'

    @classmethod
    def from_grade(cls, grade: str) -> 'AGVPhysicsConfig':
        configs = {
            'S': cls(wheel_base=0.3, wheel_radius=0.05, max_linear_speed=0.5,
                     mass=20.0, inertia_z=5.0, grade='S'),
            'M': cls(wheel_base=0.5, wheel_radius=0.1, max_linear_speed=2.0,
                     mass=50.0, inertia_z=20.0, grade='M'),
            'L': cls(wheel_base=0.8, wheel_radius=0.15, max_linear_speed=3.0,
                     mass=100.0, inertia_z=50.0, grade='L'),
            'XL': cls(wheel_base=1.0, wheel_radius=0.2, max_linear_speed=5.0,
                      mass=200.0, inertia_z=100.0, grade='XL'),
            'XXL': cls(wheel_base=1.5, wheel_radius=0.3, max_linear_speed=8.0,
                       mass=500.0, inertia_z=300.0, grade='XXL'),
        }
        return configs.get(grade, cls())


@dataclass
class AGVState:
    """AGV 状态"""
    # 位置 (x, y, theta) - 全局坐标系
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    # 速度 (线速度, 角速度)
    v: float = 0.0          # 线速度 (m/s)
    omega: float = 0.0      # 角速度 (rad/s)

    # 加速度
    a: float = 0.0          # 线加速度 (m/s^2)
    alpha: float = 0.0      # 角加速度 (rad/s^2)

    # 里程计 (带噪声的估计)
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_theta: float = 0.0
    odom_v: float = 0.0
    odom_omega: float = 0.0

    # 传感器数据
    imu_accel: np.ndarray = None   # 3: 加速度
    imu_gyro: np.ndarray = None     # 3: 角速度
    imu_mag: Optional[np.ndarray] = None  # 3: 磁力计

    # 电池
    battery_level: float = 100.0    # 0-100 %
    battery_voltage: float = 48.0   # V

    # 安全状态
    emergency_stop: bool = False
    obstacle_detected: bool = False

    timestamp: float = 0.0

    def __post_init__(self):
        if self.imu_accel is None:
            self.imu_accel = np.zeros(3, dtype=np.float32)
        if self.imu_gyro is None:
            self.imu_gyro = np.zeros(3, dtype=np.float32)
        if self.imu_mag is None:
            self.imu_mag = np.zeros(3, dtype=np.float32)


class AGVSimulator:
    """
    AGV 仿真器

    实现差速驱动 (Differential Drive) 运动学模型:
    - v_left, v_right -> v, omega
    - v, omega -> x, y, theta

    支持:
    - 物料运输场景
    - 路径跟踪
    - 障碍物检测与响应
    - 里程计/IMU 传感器模拟
    - 电池管理
    """

    def __init__(
        self,
        config: Optional[AGVPhysicsConfig] = None,
        initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        obstacles: Optional[List[Tuple[float, float, float]]] = None,  # (x, y, radius)
        waypoints: Optional[List[Tuple[float, float]]] = None,
    ):
        self.config = config or AGVPhysicsConfig()
        self.physics = self.config

        # 状态
        self.state = AGVState(
            x=initial_pose[0],
            y=initial_pose[1],
            theta=initial_pose[2],
        )

        # 场景
        self.obstacles = obstacles or []
        self.waypoints = waypoints or []
        self.current_waypoint_idx = 0

        # 传感器噪声
        self._rng = np.random.default_rng()

        # 时间
        self._time = 0.0
        self._step_count = 0
        self._dt = 0.01

        # 回调
        self._callbacks: List[Callable] = []

        # 里程计累积误差
        self._odom_drift = np.zeros(3)  # x, y, theta 漂移

    # -------------------------------------------------------------------------
    # 运动学
    # -------------------------------------------------------------------------

    def velocity_to_wheel_speeds(self, v: float, omega: float) -> Tuple[float, float]:
        """线速度 + 角速度 -> 左右轮速度"""
        wb = self.physics.wheel_base
        r = self.physics.wheel_radius

        # 差速驱动逆运动学
        v_left = r * (v - omega * wb / 2)
        v_right = r * (v + omega * wb / 2)

        return v_left, v_right

    def wheel_speeds_to_velocity(self, v_left: float, v_right: float) -> Tuple[float, float]:
        """左右轮速度 -> 线速度 + 角速度"""
        wb = self.physics.wheel_base
        r = self.physics.wheel_radius

        # v_l = r*(v - omega*wb/2)  =>  v = (v_l + v_r) / (2r)
        v = (v_left + v_right) / (2 * r)
        # v_r = r*(v + omega*wb/2)  =>  omega = (v_r - v_l) / (r*wb)
        omega = (v_right - v_left) / (r * wb)

        return v, omega

    def set_pose(self, x: float, y: float, theta: float):
        """设置位姿"""
        self.state.x = x
        self.state.y = y
        self.state.theta = theta

    def set_velocity(self, v: float, omega: float):
        """设置速度命令"""
        # 限幅
        v = np.clip(v, -self.physics.max_linear_speed, self.physics.max_linear_speed)
        omega = np.clip(omega, -self.physics.max_angular_speed, self.physics.max_angular_speed)

        self.state.v = v
        self.state.omega = omega

    # -------------------------------------------------------------------------
    # 仿真步进
    # -------------------------------------------------------------------------

    def step(self, action: np.ndarray, dt: Optional[float] = None) -> AGVState:
        """
        仿真一步

        Args:
            action: [v_cmd, omega_cmd] 或 [v_left, v_right, ...] 差分驱动
            dt: 时间步长

        Returns:
            AGVState: 当前状态
        """
        if dt is None:
            dt = self._dt

        # 解析动作
        if len(action) == 2:
            v_cmd, omega_cmd = action[0], action[1]
        elif len(action) == 4:
            # 左右轮速度
            v_cmd, omega_cmd = self.wheel_speeds_to_velocity(action[0], action[1])
        else:
            v_cmd, omega_cmd = 0.0, 0.0

        # 限幅
        v_cmd = np.clip(v_cmd, -self.physics.max_linear_speed, self.physics.max_linear_speed)
        omega_cmd = np.clip(omega_cmd, -self.physics.max_angular_speed, self.physics.max_angular_speed)

        # 一阶滤波 (平滑)
        alpha = 0.8
        v = alpha * self.state.v + (1 - alpha) * v_cmd
        omega = alpha * self.state.omega + (1 - alpha) * omega_cmd

        # 加速度限制
        dv = v - self.state.v
        max_dv = self.physics.max_accel * dt
        if abs(dv) > max_dv:
            v = self.state.v + np.sign(dv) * max_dv

        domega = omega - self.state.omega
        max_domega = self.physics.max_angular_accel * dt
        if abs(domega) > max_domega:
            omega = self.state.omega + np.sign(domega) * max_domega

        # 运动学积分 (自行车模型的简化)
        theta = self.state.theta
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt

        # 更新位姿
        self.state.x += dx
        self.state.y += dy
        self.state.theta += dtheta

        # 角度归一化
        self.state.theta = np.arctan2(np.sin(self.state.theta), np.cos(self.state.theta))

        # 更新速度/加速度
        self.state.a = (v - self.state.v) / dt if dt > 0 else 0.0
        self.state.alpha = (omega - self.state.omega) / dt if dt > 0 else 0.0
        self.state.v = v
        self.state.omega = omega

        # 更新里程计 (带漂移)
        self._update_odometry(dt)

        # 更新 IMU
        self._update_imu()

        # 更新电池
        self._update_battery(v, dt)

        # 检查障碍物
        self._check_obstacles()

        # 时间
        self._time += dt
        self._step_count += 1

        # 回调
        for cb in self._callbacks:
            cb(self.state)

        return self.state

    def _update_odometry(self, dt: float):
        """更新里程计 (带传感器噪声和漂移)"""
        # 里程计噪声
        noise_lin = self.physics.odom_noise_linear * self._rng.standard_normal() * dt
        noise_ang = self.physics.odom_noise_angular * self._rng.standard_normal() * dt

        # 累积漂移
        self._odom_drift[0] += 0.001 * self._rng.standard_normal() * dt
        self._odom_drift[1] += 0.001 * self._rng.standard_normal() * dt
        self._odom_drift[2] += 0.0005 * self._rng.standard_normal() * dt

        # 里程计积分
        dx = self.state.v * np.cos(self.state.theta) * dt
        dy = self.state.v * np.sin(self.state.theta) * dt
        dtheta = self.state.omega * dt

        self.state.odom_x += dx + noise_lin
        self.state.odom_y += dy + noise_lin
        self.state.odom_theta += dtheta + noise_ang
        self.state.odom_v = self.state.v + noise_lin
        self.state.odom_omega = self.state.omega + noise_ang

        # 应用漂移
        self.state.odom_x += self._odom_drift[0]
        self.state.odom_y += self._odom_drift[1]
        self.state.odom_theta += self._odom_drift[2]
        self.state.odom_theta = np.arctan2(np.sin(self.state.odom_theta), np.cos(self.state.odom_theta))

    def _update_imu(self):
        """模拟 IMU 数据"""
        # 陀螺仪 = 角速度 + 偏置 + 噪声
        gyro_bias = np.array([0.0, 0.0, 0.01 * self._rng.standard_normal()])
        self.state.imu_gyro = self.state.omega + gyro_bias

        # 加速度 = 重力 + 运动加速度
        gravity = np.array([0.0, 0.0, 9.81])
        # 切向加速度分量
        tangent_accel = self.state.a * np.array([np.cos(self.state.theta), np.sin(self.state.theta), 0.0])
        # 离心加速度
        centripetal = self.state.v * self.state.omega * np.array([-np.sin(self.state.theta), np.cos(self.state.theta), 0.0])

        accel_noise = 0.05 * self._rng.standard_normal(3)
        self.state.imu_accel = gravity + tangent_accel + centripetal + accel_noise

        # 磁力计 (指向北方 + 干扰)
        mag_noise = 0.02 * self._rng.standard_normal(3)
        heading = self.state.theta
        mag_north = np.array([np.cos(heading), np.sin(heading), 0.0])
        self.state.imu_mag = mag_north + mag_noise

    def _update_battery(self, v: float, dt: float):
        """更新电池状态"""
        # 消耗: 静止时 0.5A, 运动时按速度
        i_base = 0.5  # A
        i_motion = abs(v) * 2.0  # A per m/s
        i_total = i_base + i_motion

        # 电压 ~ 48V 标称, 下降至 42V 截止
        v_battery = self.state.battery_voltage
        voltage_drop = (100.0 - self.state.battery_level) * 0.06  # 每 1% 用电约降 0.06V
        self.state.battery_voltage = max(42.0, 48.0 - voltage_drop)

        # 电量消耗
        Ah_consumed = (i_total / 3600.0) * dt  # Ah
        self.state.battery_level = max(0.0, self.state.battery_level - Ah_consumed / 20.0 * 100)

    def _check_obstacles(self):
        """检查障碍物碰撞"""
        self.state.obstacle_detected = False
        for ox, oy, oradius in self.obstacles:
            dist = np.sqrt((self.state.x - ox)**2 + (self.state.y - oy)**2)
            safety_dist = oradius + 0.3  # 安全距离
            if dist < safety_dist:
                self.state.obstacle_detected = True
                # 触发紧急停止
                self.state.emergency_stop = True
                self.state.v = 0.0
                break

    # -------------------------------------------------------------------------
    # 场景管理
    # -------------------------------------------------------------------------

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """设置障碍物列表 (x, y, radius)"""
        self.obstacles = obstacles

    def add_obstacle(self, x: float, y: float, radius: float = 0.3):
        """添加障碍物"""
        self.obstacles.append((x, y, radius))

    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """设置路径点"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """获取当前目标点"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None

    def advance_waypoint(self):
        """前进到下一路径点"""
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1

    def distance_to_waypoint(self) -> float:
        """到当前路径点的距离"""
        wp = self.get_current_waypoint()
        if wp is None:
            return 0.0
        return np.sqrt((self.state.x - wp[0])**2 + (self.state.y - wp[1])**2)

    def angle_to_waypoint(self) -> float:
        """到当前路径点的角度"""
        wp = self.get_current_waypoint()
        if wp is None:
            return 0.0
        dx = wp[0] - self.state.x
        dy = wp[1] - self.state.y
        target_angle = np.arctan2(dy, dx)
        return target_angle - self.state.theta

    # -------------------------------------------------------------------------
    # 状态查询
    # -------------------------------------------------------------------------

    def get_pose(self) -> Tuple[float, float, float]:
        """获取位姿 (x, y, theta)"""
        return self.state.x, self.state.y, self.state.theta

    def get_velocity(self) -> Tuple[float, float]:
        """获取速度 (v, omega)"""
        return self.state.v, self.state.omega

    def get_odometry(self) -> Tuple[float, float, float, float, float]:
        """获取里程计 (odom_x, odom_y, odom_theta, odom_v, odom_omega)"""
        return (self.state.odom_x, self.state.odom_y, self.state.odom_theta,
                self.state.odom_v, self.state.odom_omega)

    def get_imu(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """获取 IMU 数据 (accel, gyro, mag)"""
        return self.state.imu_accel, self.state.imu_gyro, self.state.imu_mag

    def get_state_dict(self) -> Dict[str, Any]:
        """获取完整状态字典"""
        return {
            'pose': (self.state.x, self.state.y, self.state.theta),
            'velocity': (self.state.v, self.state.omega),
            'acceleration': (self.state.a, self.state.alpha),
            'odometry': (self.state.odom_x, self.state.odom_y, self.state.odom_theta,
                         self.state.odom_v, self.state.odom_omega),
            'imu': {
                'accel': self.state.imu_accel.copy(),
                'gyro': self.state.imu_gyro.copy(),
                'mag': self.state.imu_mag.copy() if self.state.imu_mag is not None else None,
            },
            'battery': {
                'level': self.state.battery_level,
                'voltage': self.state.battery_voltage,
            },
            'safety': {
                'emergency_stop': self.state.emergency_stop,
                'obstacle_detected': self.state.obstacle_detected,
            },
            'waypoint_idx': self.current_waypoint_idx,
            'time': self._time,
            'step': self._step_count,
        }

    def reset(
        self,
        pose: Optional[Tuple[float, float, float]] = None
    ):
        """重置仿真"""
        if pose is not None:
            self.state.x, self.state.y, self.state.theta = pose
        else:
            self.state.x = 0.0
            self.state.y = 0.0
            self.state.theta = 0.0

        self.state.v = 0.0
        self.state.omega = 0.0
        self.state.a = 0.0
        self.state.alpha = 0.0
        self.state.battery_level = 100.0
        self.state.battery_voltage = 48.0
        self.state.emergency_stop = False
        self.state.obstacle_detected = False

        self._time = 0.0
        self._step_count = 0
        self._odom_drift = np.zeros(3)
        self.current_waypoint_idx = 0

    def register_callback(self, cb: Callable):
        """注册状态回调"""
        self._callbacks.append(cb)


# =============================================================================
# 路径跟踪控制器
# =============================================================================

class AGVPurePursuitController:
    """
    Pure Pursuit 路径跟踪控制器

    实现:
    - 前视距离 (lookahead) 自适应调整
    - 速度规划 (高速时增大前视距离)
    - 曲率限幅
    """

    def __init__(
        self,
        lookahead_min: float = 0.1,
        lookahead_max: float = 1.0,
        lookahead_gain: float = 0.5,  # v * gain = lookahead
        k_gain: float = 1.0,          # 控制器增益
    ):
        self.lookahead_min = lookahead_min
        self.lookahead_max = lookahead_max
        self.lookahead_gain = lookahead_gain
        self.k_gain = k_gain

    def compute_command(
        self,
        agv: AGVSimulator,
        lookahead_override: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算控制命令

        Returns:
            (v_cmd, omega_cmd)
        """
        # 获取当前状态
        x, y, theta = agv.get_pose()
        v, _ = agv.get_velocity()

        # 前视距离
        if lookahead_override is not None:
            ld = lookahead_override
        else:
            ld = min(self.lookahead_max, max(self.lookahead_min, self.lookahead_gain * abs(v)))

        # 在路径上寻找前视点
        target = self._find_lookahead_point(x, y, theta, ld, agv.waypoints)
        if target is None:
            return 0.0, 0.0

        # 计算弧长
        alpha = np.arctan2(target[1] - y, target[0] - x) - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # 曲率
        k = (2 * np.sin(alpha)) / ld
        k = np.clip(k, -3.0, 3.0)

        # 线速度命令 (保持)
        v_cmd = v

        # 角速度命令
        omega_cmd = self.k_gain * k * v_cmd

        return v_cmd, omega_cmd

    def _find_lookahead_point(
        self,
        x: float, y: float, theta: float,
        ld: float,
        waypoints: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """在路径上寻找距离当前位置约 ld 的前视点"""
        if not waypoints:
            return None

        # 从当前位置向前的路径点
        for i, (wx, wy) in enumerate(waypoints):
            dist = np.sqrt((wx - x)**2 + (wy - y)**2)
            if dist >= ld:
                return (wx, wy)

        # 如果没有找到, 返回最后一个点
        return waypoints[-1] if waypoints else None


# =============================================================================
# AGV 状态机
# =============================================================================

class AGVStateMachine:
    """
    AGV 状态机

    状态:
    - IDLE: 空闲等待
    - MOVING: 运动中
    - NAVIGATING: 导航中 (沿路径)
    - DOCKING: 对接中
    - CHARGING: 充电中
    - ERROR: 错误
    - ESTOP: 紧急停止
    """

    IDLE = "idle"
    MOVING = "moving"
    NAVIGATING = "navigating"
    DOCKING = "docking"
    CHARGING = "charging"
    ERROR = "error"
    ESTOP = "estop"

    def __init__(self):
        self._state = self.IDLE
        self._prev_state = self.IDLE
        self._error_code: Optional[str] = None

    @property
    def state(self) -> str:
        return self._state

    def transition(self, new_state: str):
        """状态切换"""
        if new_state != self._state:
            self._prev_state = self._state
            self._state = new_state

    def is_allowed(self, target_state: str) -> bool:
        """检查是否允许切换"""
        # 允许从任何状态进入 ESTOP
        if target_state == self.ESTOP:
            return True
        # ESTOP 只允许转到 ERROR 或 IDLE
        if self._state == self.ESTOP:
            return target_state in (self.ERROR, self.IDLE)
        # ERROR 只允许转到 IDLE
        if self._state == self.ERROR:
            return target_state == self.IDLE
        return True

    def update(self, agv_state: AGVState) -> str:
        """根据 AGV 状态更新状态机"""
        if agv_state.emergency_stop:
            self.transition(self.ESTOP)
        elif agv_state.battery_level < 10.0:
            self.transition(self.DOCKING)
        elif abs(agv_state.v) > 0.01:
            self.transition(self.MOVING)
        else:
            self.transition(self.IDLE)

        return self._state


def get_agv_physics_spec(grade: str) -> 'AGVPhysicsConfig':
    """获取 AGV 五级物理规格"""
    return AGVPhysicsConfig.from_grade(grade)
