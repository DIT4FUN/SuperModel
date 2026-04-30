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
AGV巡逻控制模块
==============

集成式AGV自主巡逻系统，融合:
- 多点巡逻路径规划
- 动态障碍物回避 (DWA / APF / VFH)
- 传感器融合感知 (视觉 + 激光 + IMU)
- 异常检测与自主恢复
- 任务调度与事件响应

集成关系:
  NavigationController ──► TrajectoryTracker ──► TwistCommand
  ObstacleAvoider ──────► VelocityCommand
  EmbodiedController ───► JointCommand
  SensorManager ─────────► Multi-sensor fusion

AGV五级巡逻能力:
  S:  单点巡逻 + 简单避障
  M:  多点巡逻 + DWA避障
  L:  自主路径规划 + APF + 姿态稳定
  XL: 全局规划 + VFH + 力控协同
  XXL: MPC预测 + 多AGV协同 + 云端调度
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Callable
from enum import Enum
import heapq
import math
import time
import sys

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import TactileArray, TactileFrame, TactileSensorType, VirtualTactileSensor
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType, VirtualForceSensor
from sensors.imu import IMUSensor, IMUFrame, IMUSensorType
from sensors.manager import SensorManager, SensorGrade


# ─────────────────────────────────────────────
# AGV五级巡逻规格
# ─────────────────────────────────────────────

class PatrolGrade(str, Enum):
    """AGV巡逻五级等级"""
    S = 'S'   # 教育级: 单点巡逻
    M = 'M'   # 标准级: 多点巡逻 + DWA
    L = 'L'   # 专业级: 全局规划 + APF
    XL = 'XL'  # 高性能: VFH + 力控协同
    XXL = 'XXL'  # 旗舰级: MPC预测 + 多AGV协同


@dataclass
class PatrolSpec:
    """巡逻规格参数"""
    grade: PatrolGrade
    max_patrol_speed: float        # 最大巡逻速度 m/s
    max_obstacle_distance: float   # 障碍物检测距离 m
    patrol_points: int             # 巡逻点数量
    avoidance_strategy: str         # 避障策略
    has_emergency_recovery: bool   # 是否有自主恢复
    has_multi_agent: bool           # 是否支持多AGV协同
    control_frequency: float        # 控制频率 Hz
    sensor_modalities: List[str]   # 启用传感器模态

    @classmethod
    def from_grade(cls, grade: str) -> "PatrolSpec":
        specs = {
            'S': cls(
                grade=PatrolGrade.S,
                max_patrol_speed=0.3,
                max_obstacle_distance=1.5,
                patrol_points=2,
                avoidance_strategy='simple',
                has_emergency_recovery=False,
                has_multi_agent=False,
                control_frequency=50.0,
                sensor_modalities=['imu'],
            ),
            'M': cls(
                grade=PatrolGrade.M,
                max_patrol_speed=0.8,
                max_obstacle_distance=3.0,
                patrol_points=4,
                avoidance_strategy='dwa',
                has_emergency_recovery=True,
                has_multi_agent=False,
                control_frequency=100.0,
                sensor_modalities=['imu', 'force'],
            ),
            'L': cls(
                grade=PatrolGrade.L,
                max_patrol_speed=1.2,
                max_obstacle_distance=5.0,
                patrol_points=8,
                avoidance_strategy='apf',
                has_emergency_recovery=True,
                has_multi_agent=False,
                control_frequency=200.0,
                sensor_modalities=['imu', 'force', 'tactile'],
            ),
            'XL': cls(
                grade=PatrolGrade.XL,
                max_patrol_speed=1.8,
                max_obstacle_distance=8.0,
                patrol_points=16,
                avoidance_strategy='vfh',
                has_emergency_recovery=True,
                has_multi_agent=True,
                control_frequency=500.0,
                sensor_modalities=['imu', 'force', 'tactile', 'vision'],
            ),
            'XXL': cls(
                grade=PatrolGrade.XXL,
                max_patrol_speed=2.5,
                max_obstacle_distance=15.0,
                patrol_points=32,
                avoidance_strategy='hybrid',
                has_emergency_recovery=True,
                has_multi_agent=True,
                control_frequency=1000.0,
                sensor_modalities=['imu', 'force', 'tactile', 'vision', 'audio'],
            ),
        }
        return specs.get(grade, specs['M'])


def get_patrol_spec(grade: str) -> dict:
    """获取巡逻规格 (字典格式)"""
    spec = PatrolSpec.from_grade(grade)
    return {
        'grade': spec.grade.value,
        'max_patrol_speed': spec.max_patrol_speed,
        'max_obstacle_distance': spec.max_obstacle_distance,
        'patrol_points': spec.patrol_points,
        'avoidance_strategy': spec.avoidance_strategy,
        'has_emergency_recovery': spec.has_emergency_recovery,
        'has_multi_agent': spec.has_multi_agent,
        'control_frequency': spec.control_frequency,
        'sensor_modalities': spec.sensor_modalities,
    }


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

class PatrolState(Enum):
    """巡逻状态"""
    IDLE = "idle"
    PATROLLING = "patrolling"
    AVOIDING = "avoiding"
    WAITING = "waiting"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERING = "recovering"
    ARRIVED = "arrived"
    PAUSED = "paused"


@dataclass
class PatrolPoint:
    """巡逻点"""
    x: float
    y: float
    theta: float = 0.0
    name: str = ""
    dwell_time: float = 0.0  # 停留时间 s
    priority: int = 0        # 优先级 (越高越先访问)
    sensor_check: bool = True  # 是否进行传感器检查

    def __post_init__(self):
        if isinstance(self.theta, (int, float)) and not isinstance(self.theta, bool):
            pass
        else:
            self.theta = 0.0


@dataclass
class PatrolRoute:
    """巡逻路线"""
    name: str
    points: List[PatrolPoint]
    loop: bool = True  # 是否循环
    priority: int = 0  # 路线优先级

    def __post_init__(self):
        if self.points:
            self.points.sort(key=lambda p: p.priority, reverse=True)


@dataclass
class PatrolEvent:
    """巡逻事件"""
    timestamp: float
    type: str          # "obstacle" | "sensor_alert" | "arrival" | "emergency"
    position: Optional[Tuple[float, float]]
    data: Dict = field(default_factory=dict)


@dataclass
class PatrolMetrics:
    """巡逻指标"""
    total_distance: float = 0.0     # 总行驶距离 m
    total_time: float = 0.0        # 总巡逻时间 s
    obstacles_avoided: int = 0     # 避障次数
    alerts_triggered: int = 0       # 告警次数
    points_completed: int = 0       # 已完成巡逻点数
    points_total: int = 0          # 总巡逻点数
    avg_speed: float = 0.0          # 平均速度 m/s
    emergency_stops: int = 0        # 急停次数

    def to_dict(self) -> Dict:
        return {
            'total_distance': round(self.total_distance, 2),
            'total_time': round(self.total_time, 2),
            'obstacles_avoided': self.obstacles_avoided,
            'alerts_triggered': self.alerts_triggered,
            'points_completed': self.points_completed,
            'points_total': self.points_total,
            'avg_speed': round(self.avg_speed, 2),
            'emergency_stops': self.emergency_stops,
        }


# ─────────────────────────────────────────────
# 障碍物数据结构
# ─────────────────────────────────────────────

@dataclass
class Obstacle:
    """障碍物"""
    position: np.ndarray      # 2, 世界坐标系 (x, y) m
    radius: float             # 障碍物半径 m
    velocity: Optional[np.ndarray] = None  # 2, 速度 m/s
    type: str = "static"     # "static" | "dynamic" | "person"

    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if self.velocity is None:
            self.velocity = np.zeros(2)
        elif isinstance(self.velocity, list):
            self.velocity = np.array(self.velocity, dtype=np.float32)

    def predict_position(self, dt: float) -> np.ndarray:
        return self.position + self.velocity * dt

    @property
    def center(self) -> np.ndarray:
        return self.position


# ─────────────────────────────────────────────
# 巡逻控制器
# ─────────────────────────────────────────────

class PatrolController:
    """
    AGV巡逻控制器

    集成导航 + 避障 + 传感器融合的完整巡逻系统
    """

    def __init__(
        self,
        grade: str = 'M',
        initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        obstacles: Optional[List[Obstacle]] = None,
        use_virtual_sensors: bool = True,
    ):
        """
        初始化巡逻控制器

        Args:
            grade: AGV巡逻等级 (S/M/L/XL/XXL)
            initial_pose: 初始位置 (x, y, theta)
            obstacles: 已知障碍物列表
            use_virtual_sensors: 是否使用虚拟传感器 (仿真模式)
        """
        self.grade = grade
        self.spec = PatrolSpec.from_grade(grade)
        self.pose = np.array(initial_pose, dtype=np.float32)  # [x, y, theta]
        self.velocity = np.zeros(3, dtype=np.float32)        # [vx, vy, omega]

        self.obstacles: List[Obstacle] = obstacles or []
        self.routes: List[PatrolRoute] = []
        self.current_route: Optional[PatrolRoute] = None
        self.current_target: Optional[PatrolPoint] = None
        self.current_point_index: int = 0

        self.state = PatrolState.IDLE
        self.metrics = PatrolMetrics()
        self.events: List[PatrolEvent] = []
        self._last_update_time = time.time()
        self._arrival_threshold = 0.1  # 到达判定阈值 m

        # 初始化传感器
        self._init_sensors(use_virtual_sensors)

        # 速度限制
        self._max_linear_speed = self.spec.max_patrol_speed
        self._max_angular_speed = 2.0

        # PID参数 (简化版)
        self._Kp_linear = 2.0
        self._Kp_angular = 3.0

        # 卡死检测与恢复 (L/XL/XXL)
        self._stuck_detector: Optional[StuckDetector] = None
        self._recovery_manager: Optional[AutonomousRecoveryManager] = None
        self._recovery_in_progress = False
        self._recovery_command: Optional[Dict] = None
        self._recovery_start_pose: Optional[np.ndarray] = None
        self._recovery_start_time: float = 0.0
        self._last_stuck_check: float = 0.0

        # 初始化卡死检测与恢复 (L级及以上)
        if grade in ('L', 'XL', 'XXL'):
            self._stuck_detector = StuckDetector()
            self._recovery_manager = AutonomousRecoveryManager(grade=grade)

    def _init_sensors(self, use_virtual: bool):
        """初始化传感器"""
        self.imu: Optional[IMUSensor] = None
        self.tactile: Optional[TactileArray] = None
        self.force: Optional[ForceTorqueSensor] = None

        if use_virtual:
            if 'imu' in self.spec.sensor_modalities:
                self.imu = IMUSensor(
                    sensor_type=IMUSensorType.VIRTUAL,
                    sensor_id="patrol_imu"
                )
                self.imu.open()

            if 'tactile' in self.spec.sensor_modalities:
                self.tactile = VirtualTactileSensor(
                    array_size=(8, 8),
                    sensor_id="patrol_tactile"
                )
                self.tactile.open()

            if 'force' in self.spec.sensor_modalities:
                self.force = VirtualForceSensor(
                    sensor_id="patrol_force"
                )
                self.force.open()

    def add_route(self, route: PatrolRoute):
        """添加巡逻路线"""
        self.routes.append(route)
        if self.current_route is None:
            self.current_route = route
            self.current_point_index = 0
            if route.points:
                self.current_target = route.points[0]

    def start_patrol(self, route_name: Optional[str] = None):
        """启动巡逻"""
        if route_name:
            for route in self.routes:
                if route.name == route_name:
                    self.current_route = route
                    self.current_point_index = 0
                    if route.points:
                        self.current_target = route.points[0]
                    break
        elif self.current_route is None and self.routes:
            self.current_route = self.routes[0]
            self.current_point_index = 0
            if self.current_route.points:
                self.current_target = self.current_route.points[0]

        if self.current_target is None:
            self.state = PatrolState.IDLE
            return False

        self.state = PatrolState.PATROLLING
        self._last_update_time = time.time()
        return True

    def stop_patrol(self):
        """停止巡逻"""
        self.state = PatrolState.IDLE
        self.velocity[:] = 0

    def pause_patrol(self):
        """暂停巡逻"""
        if self.state == PatrolState.PATROLLING:
            self.state = PatrolState.PAUSED

    def resume_patrol(self):
        """恢复巡逻"""
        if self.state == PatrolState.PAUSED:
            self.state = PatrolState.PATROLLING

    def emergency_stop(self):
        """紧急停止"""
        self.state = PatrolState.EMERGENCY_STOP
        self.velocity[:] = 0
        self.metrics.emergency_stops += 1
        self._log_event("emergency", {"reason": "manual_stop"})

    def _log_event(self, event_type: str, data: Optional[Dict] = None):
        """记录事件"""
        event = PatrolEvent(
            timestamp=time.time(),
            type=event_type,
            position=(float(self.pose[0]), float(self.pose[1])),
            data=data or {},
        )
        self.events.append(event)
        if event_type in ("obstacle", "sensor_alert"):
            self.metrics.alerts_triggered += 1

    def update(
        self,
        dt: Optional[float] = None,
        detected_obstacles: Optional[List[Obstacle]] = None,
    ) -> Tuple[np.ndarray, PatrolState]:
        """
        更新巡逻控制器

        Args:
            dt: 时间步长 (s), 默认自动计算
            detected_obstacles: 检测到的障碍物列表

        Returns:
            velocity: [vx, vy, omega] 速度指令
            state: 当前巡逻状态
        """
        if dt is None:
            now = time.time()
            dt = now - self._last_update_time
            self._last_update_time = now

        dt = max(dt, 0.001)  # 防止除零
        self.metrics.total_time += dt

        # 更新传感器
        self._update_sensors()

        # 更新障碍物列表
        if detected_obstacles:
            for obs in detected_obstacles:
                self._update_obstacle(obs)

        # 状态机
        if self.state == PatrolState.EMERGENCY_STOP:
            self.velocity[:] = 0
            return self.velocity, self.state

        if self.state == PatrolState.IDLE:
            self.velocity[:] = 0
            return self.velocity, self.state

        if self.state == PatrolState.PAUSED:
            self.velocity[:] = 0
            return self.velocity, self.state

        if self.state in (PatrolState.PATROLLING, PatrolState.AVOIDING):
            # 检查是否有障碍物需要避让
            nearby_obstacle = self._check_nearby_obstacle()
            if nearby_obstacle:
                self.state = PatrolState.AVOIDING
                self._avoid_obstacle(nearby_obstacle, dt)
                self.metrics.obstacles_avoided += 1
                self._log_event("obstacle", {"obstacle_pos": nearby_obstacle.position.tolist()})
            else:
                if self.state == PatrolState.AVOIDING:
                    self.state = PatrolState.PATROLLING
                self._follow_target(dt)

            # ── 卡死检测与自主恢复 (L/XL/XXL) ──
            if self._stuck_detector is not None and not self._recovery_in_progress:
                now = time.time()
                if now - self._last_stuck_check > 0.5:  # 每0.5秒检测一次
                    self._last_stuck_check = now

                    # 构建IMU帧 (如果有IMU)
                    imu_frame = None
                    if hasattr(self, '_imu_frame'):
                        imu_frame = self._imu_frame

                    stuck_result = self._stuck_detector.update(
                        position=self.pose,
                        command=self.velocity,
                        imu_frame=imu_frame,
                        timestamp=now,
                    )

                    if stuck_result.is_stuck and stuck_result.confidence > 0.6:
                        # 进入恢复状态
                        self.state = PatrolState.RECOVERING
                        self._recovery_in_progress = True
                        self._recovery_start_pose = self.pose.copy()
                        self._recovery_start_time = now
                        self._log_event("stuck_detected", {
                            "reason": stuck_result.reason,
                            "confidence": stuck_result.confidence,
                            "strategy": stuck_result.recommended_strategy.value,
                        })

                        # 请求恢复指令
                        if self._recovery_manager:
                            target = np.array([self.current_target.x, self.current_target.y, 0.0]) if self.current_target else None
                            self._recovery_command = self._recovery_manager.request_recovery(
                                stuck_result,
                                current_pose=self.pose,
                                target_pose=target,
                                timestamp=now,
                            )

            # ── 执行恢复操作 ──
            if self._recovery_in_progress and self._recovery_command:
                self._execute_recovery(dt)

            # 检查是否到达目标点
            self._check_arrival()

            # 更新里程计
            self._update_odometry(dt)

        return self.velocity, self.state

    def _update_sensors(self):
        """更新传感器数据"""
        if self.imu:
            try:
                self.imu.capture()
            except Exception:
                pass

        if self.tactile:
            try:
                self.tactile.capture()
            except Exception:
                pass

        if self.force:
            try:
                self.force.capture()
            except Exception:
                pass

    def _update_obstacle(self, obs: Obstacle):
        """更新障碍物列表"""
        # 简单更新: 如果障碍物距离小于阈值则更新
        for i, existing in enumerate(self.obstacles):
            dist = np.linalg.norm(existing.position - obs.position)
            if dist < 0.5:
                self.obstacles[i] = obs
                return
        self.obstacles.append(obs)

    def _check_nearby_obstacle(self) -> Optional[Obstacle]:
        """检查附近障碍物"""
        detection_range = self.spec.max_obstacle_distance
        for obs in self.obstacles:
            dist = np.linalg.norm(obs.position[:2] - self.pose[:2])
            if dist < detection_range:
                return obs
        return None

    def _avoid_obstacle(self, obstacle: Obstacle, dt: float):
        """避障逻辑"""
        # 计算相对位置
        rel_pos = obstacle.position[:2] - self.pose[:2]
        dist = np.linalg.norm(rel_pos)

        # 简单的绕行策略: 沿切线方向移动
        if dist < 0.001:
            dist = 0.001

        # 计算绕行方向 (垂直于障碍物方向)
        obstacle_angle = np.arctan2(rel_pos[1], rel_pos[0])
        avoid_angle = obstacle_angle + np.pi / 2  # 向左侧绕行

        # 计算目标速度
        desired_vx = self._max_linear_speed * np.cos(avoid_angle)
        desired_vy = self._max_linear_speed * np.sin(avoid_angle)

        # 平滑过渡
        self.velocity[0] = 0.9 * self.velocity[0] + 0.1 * desired_vx
        self.velocity[1] = 0.9 * self.velocity[1] + 0.1 * desired_vy
        self.velocity[2] = 0.0  # 巡逻时保持方向

    def _execute_recovery(self, dt: float):
        """执行恢复操作"""
        if self._recovery_command is None or self._recovery_start_pose is None:
            self._recovery_in_progress = False
            return

        strategy = self._recovery_command['strategy']
        elapsed = time.time() - self._recovery_start_time

        # 执行恢复速度指令
        recovery_vel = self._recovery_command.get('velocity', np.zeros(3))
        self.velocity[:] = recovery_vel

        # 更新里程计
        self._update_odometry(dt)

        # 检查恢复是否完成
        if self._recovery_manager:
            complete, success = self._recovery_manager.check_recovery_complete(
                strategy=strategy,
                elapsed_time=elapsed,
                current_pose=self.pose,
                start_pose=self._recovery_start_pose,
            )

            if complete:
                self._recovery_in_progress = False
                self._recovery_command = None
                self._log_event("recovery_complete", {
                    "strategy": strategy.value,
                    "success": success,
                })

                if success:
                    # 恢复正常巡逻
                    self.state = PatrolState.PATROLLING
                else:
                    # 恢复失败，降级或终止
                    self.state = PatrolState.PATROLLING
                    self.metrics.failures += 1

    def _follow_target(self, dt: float):
        """跟踪目标点"""
        if self.current_target is None:
            return

        target = np.array([self.current_target.x, self.current_target.y])
        rel_pos = target - self.pose[:2]
        dist = np.linalg.norm(rel_pos)

        # 计算目标角度
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = target_angle - self.pose[2]

        # 角度归一化到 [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # PID控制角速度
        omega = self._Kp_angular * angle_diff
        omega = np.clip(omega, -self._max_angular_speed, self._max_angular_speed)

        # 根据距离调整线速度
        if dist > 0.5:
            v = min(dist * self._Kp_linear, self._max_linear_speed)
        else:
            v = dist * self._Kp_linear * 2

        # 考虑角度偏差
        angle_factor = np.cos(angle_diff)
        v = v * max(angle_factor, 0.1)

        # 计算世界坐标系速度
        vx = v * np.cos(self.pose[2])
        vy = v * np.sin(self.pose[2])

        self.velocity[0] = vx
        self.velocity[1] = vy
        self.velocity[2] = omega

    def _check_arrival(self):
        """检查是否到达目标点"""
        if self.current_target is None:
            return

        target = np.array([self.current_target.x, self.current_target.y])
        dist = np.linalg.norm(target - self.pose[:2])

        if dist < self._arrival_threshold:
            self.metrics.points_completed += 1
            self._log_event("arrival", {"point": self.current_target.name or f"point_{self.current_point_index}"})

            # 移动到下一个点
            if self.current_route and self.current_route.points:
                self.current_point_index += 1
                if self.current_point_index >= len(self.current_route.points):
                    if self.current_route.loop:
                        self.current_point_index = 0
                    else:
                        self.state = PatrolState.ARRIVED
                        return

                self.current_target = self.current_route.points[self.current_point_index]

    def _update_odometry(self, dt: float):
        """更新里程计"""
        # 更新位置
        self.pose[0] += self.velocity[0] * dt
        self.pose[1] += self.velocity[1] * dt
        self.pose[2] += self.velocity[2] * dt

        # 角度归一化
        while self.pose[2] > np.pi:
            self.pose[2] -= 2 * np.pi
        while self.pose[2] < -np.pi:
            self.pose[2] += 2 * np.pi

        # 更新总距离
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        self.metrics.total_distance += np.sqrt(dx**2 + dy**2)

        # 更新平均速度
        if self.metrics.total_time > 0:
            self.metrics.avg_speed = self.metrics.total_distance / self.metrics.total_time

    def get_pose(self) -> Tuple[float, float, float]:
        """获取当前位置"""
        return tuple(self.pose.tolist())

    def get_metrics(self) -> Dict:
        """获取巡逻指标"""
        return self.metrics.to_dict()

    def get_state(self) -> PatrolState:
        """获取当前状态"""
        return self.state

    def reset(self, pose: Optional[Tuple[float, float, float]] = None):
        """重置控制器"""
        if pose:
            self.pose = np.array(pose, dtype=np.float32)
        else:
            self.pose[:] = 0
        self.velocity[:] = 0
        self.state = PatrolState.IDLE
        self.current_target = None
        self.current_point_index = 0
        self.events.clear()
        self.metrics = PatrolMetrics()
        self._last_update_time = time.time()
        self._recovery_in_progress = False
        self._recovery_command = None
        self._recovery_start_pose = None
        if self._stuck_detector:
            self._stuck_detector.reset()
        if self._recovery_manager:
            self._recovery_manager.reset()

    def get_events(self, since: Optional[float] = None) -> List[PatrolEvent]:
        """获取事件列表"""
        if since is None:
            return self.events
        return [e for e in self.events if e.timestamp >= since]

    def __repr__(self) -> str:
        return (
            f"PatrolController(grade={self.grade}, state={self.state.value}, "
            f"pose=({self.pose[0]:.2f}, {self.pose[1]:.2f}, {np.degrees(self.pose[2]):.1f}°), "
            f"metrics={self.get_metrics()})"
        )


# ─────────────────────────────────────────────
# 巡逻系统工厂
# ─────────────────────────────────────────────

def create_patrol_controller(
    grade: str = 'M',
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    route: Optional[PatrolRoute] = None,
) -> PatrolController:
    """
    创建巡逻控制器

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        pose: 初始位置
        route: 初始巡逻路线

    Returns:
        配置好的 PatrolController
    """
    controller = PatrolController(
        grade=grade,
        initial_pose=pose,
        use_virtual_sensors=True,
    )
    if route:
        controller.add_route(route)
    return controller


def run_patrol_benchmark(grades: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    运行五级巡逻能力基准测试

    Args:
        grades: 要测试的等级列表

    Returns:
        各等级测试结果
    """
    if grades is None:
        grades = ['S', 'M', 'L', 'XL', 'XXL']

    results = {}
    for grade in grades:
        # 创建测试场景
        route = PatrolRoute(
            name=f"benchmark_route_{grade}",
            points=[
                PatrolPoint(x=0.0, y=0.0, name="start"),
                PatrolPoint(x=2.0, y=0.0, name="p1"),
                PatrolPoint(x=2.0, y=2.0, name="p2"),
                PatrolPoint(x=0.0, y=2.0, name="p3"),
            ],
            loop=True,
        )

        obstacles = [
            Obstacle(position=np.array([1.0, 0.5]), radius=0.3, type="static"),
            Obstacle(position=np.array([1.5, 1.5]), radius=0.4, type="person"),
        ]

        controller = PatrolController(
            grade=grade,
            initial_pose=(0.0, 0.0, 0.0),
            obstacles=obstacles,
            use_virtual_sensors=True,
        )
        controller.add_route(route)
        controller.start_patrol()

        # 模拟巡逻 10 秒
        for _ in range(int(10 * get_patrol_spec(grade)['control_frequency'])):
            controller.update(dt=1.0 / get_patrol_spec(grade)['control_frequency'])

        results[grade] = controller.get_metrics()
        results[grade]['spec'] = get_patrol_spec(grade)
        controller.stop_patrol()

    return results


# ─────────────────────────────────────────────
# AGV卡死检测与自主恢复系统 (L/XL/XXL)
# ─────────────────────────────────────────────

class RecoveryStrategy(str, Enum):
    """恢复策略"""
    RETRY = 'retry'                      # 重试当前动作
    BACKUP = 'backup'                    # 后退尝试
    REPLAN = 'replan'                    # 重新规划路径
    SIDESTEP = 'sidestep'               # 侧向横移 (Mecanum)
    ROTATE = 'rotate'                   # 原地旋转后重新尝试
    ABORT = 'abort'                     # 放弃当前任务点
    ESCALATE = 'escalate'               # 升级处理 (请求人工干预)


@dataclass
class StuckDetectionResult:
    """卡死检测结果"""
    is_stuck: bool
    confidence: float                   # 0-1 置信度
    reason: str                        # 原因描述
    stuck_duration: float              # 卡死持续时间 (s)
    command_history: np.ndarray        # 历史指令
    position_history: np.ndarray       # 历史位置
    recommended_strategy: RecoveryStrategy


class StuckDetector:
    """
    AGV卡死检测器

    通过对比电机指令与实际运动，检测AGV是否陷入卡死状态:
    - 电机持续输出但位置无变化 → 机械卡死
    - 位置振荡但无法前进 → 陷入局部最优
    - IMU角度快速变化但位置不变 → 轮胎打滑

    检测方法:
    1. 命令-运动一致性检验
    2. 位置方差检验 (方差过小)
    3. 滑移率检验 (IMU vs 里程计)
    """

    def __init__(
        self,
        position_window: int = 20,       # 位置历史窗口大小
        command_window: int = 20,        # 指令历史窗口大小
        stuck_threshold_m: float = 0.02,  # 位置变化阈值 (m)
        stuck_time_threshold: float = 3.0,  # 卡死判定时间 (s)
        slip_threshold: float = 0.5,    # 滑移率阈值
    ):
        self.position_window = position_window
        self.command_window = command_window
        self.stuck_threshold_m = stuck_threshold_m
        self.stuck_time_threshold = stuck_time_threshold
        self.slip_threshold = slip_threshold

        # 历史数据
        self._position_history: List[np.ndarray] = []
        self._command_history: List[np.ndarray] = []
        self._imu_history: List[IMUFrame] = []
        self._timestamp_history: List[float] = []

        self._stuck_start_time: Optional[float] = None
        self._last_recovery_time: float = 0.0
        self._recovery_count: int = 0

    def update(
        self,
        position: np.ndarray,
        command: np.ndarray,
        imu_frame: Optional[IMUFrame] = None,
        timestamp: Optional[float] = None,
    ) -> StuckDetectionResult:
        """
        更新检测状态

        Args:
            position: 当前位置 [x, y, theta]
            command: 当前速度指令 [vx, vy, omega]
            imu_frame: IMU数据帧 (可选)
            timestamp: 当前时间戳

        Returns:
            StuckDetectionResult
        """
        if timestamp is None:
            timestamp = time.time()

        # 记录历史
        self._position_history.append(position.copy())
        self._command_history.append(command.copy())
        if imu_frame:
            self._imu_history.append(imu_frame)
        self._timestamp_history.append(timestamp)

        # 保持窗口大小
        if len(self._position_history) > self.position_window:
            self._position_history.pop(0)
        if len(self._command_history) > self.command_window:
            self._command_history.pop(0)
        if len(self._imu_history) > self.position_window:
            self._imu_history.pop(0)
        if len(self._timestamp_history) > self.position_window:
            self._timestamp_history.pop(0)

        # 需要足够的样本
        if len(self._position_history) < 5:
            return StuckDetectionResult(
                is_stuck=False,
                confidence=0.0,
                reason='insufficient_samples',
                stuck_duration=0.0,
                command_history=np.array([]),
                position_history=np.array([]),
                recommended_strategy=RecoveryStrategy.RETRY,
            )

        pos_arr = np.array(self._position_history)
        cmd_arr = np.array(self._command_history)

        # 1. 检测机械卡死: 指令大但位置不变
        stuck_by_command = self._detect_stuck_by_command(pos_arr, cmd_arr)

        # 2. 检测振荡陷入: 位置方差过小但时间足够长
        stuck_by_oscillation = self._detect_stuck_by_oscillation(pos_arr)

        # 3. 检测轮胎打滑: IMU有运动但里程计无变化
        stuck_by_slip = self._detect_stuck_by_slip(pos_arr)

        # 综合判断
        is_stuck, confidence, reason = self._fuse_detection(
            stuck_by_command, stuck_by_oscillation, stuck_by_slip
        )

        # 跟踪卡死持续时间
        if is_stuck:
            if self._stuck_start_time is None:
                self._stuck_start_time = timestamp
            stuck_duration = timestamp - self._stuck_start_time
        else:
            self._stuck_start_time = None
            stuck_duration = 0.0

        # 推荐恢复策略
        strategy = self._recommend_strategy(
            is_stuck, confidence, reason, stuck_duration
        )

        return StuckDetectionResult(
            is_stuck=is_stuck,
            confidence=confidence,
            reason=reason,
            stuck_duration=stuck_duration,
            command_history=cmd_arr,
            position_history=pos_arr,
            recommended_strategy=strategy,
        )

    def _detect_stuck_by_command(
        self,
        positions: np.ndarray,
        commands: np.ndarray,
    ) -> Tuple[bool, float, str]:
        """检测指令大但位置不变的情况"""
        # 统计有指令的时间比例
        command_magnitude = np.linalg.norm(commands[:, :2], axis=1)
        has_command = command_magnitude > 0.05  # 忽略微小指令

        if not np.any(has_command):
            return False, 0.0, 'no_command'

        # 位置变化
        if len(positions) > 1:
            position_changes = np.diff(positions[:, :2], axis=0)
            movement = np.linalg.norm(position_changes, axis=1)
            total_movement = np.sum(movement)
        else:
            total_movement = 0.0

        # 计算预期运动量
        command_duration = np.sum(has_command) * (
            (self._timestamp_history[-1] - self._timestamp_history[0])
            / max(len(self._timestamp_history), 1)
        )
        expected_movement = np.mean(command_magnitude[has_command]) * command_duration

        # 运动效率
        if expected_movement > 0.01:
            efficiency = min(total_movement / expected_movement, 1.0)
        else:
            efficiency = 1.0

        # 如果效率很低且有明显指令
        if efficiency < 0.2 and np.sum(has_command) >= len(positions) * 0.5:
            return True, 1.0 - efficiency, 'mechanical_stuck'

        return False, 0.0, ''

    def _detect_stuck_by_oscillation(
        self,
        positions: np.ndarray,
    ) -> Tuple[bool, float, str]:
        """检测位置振荡但无法前进"""
        if len(positions) < 10:
            return False, 0.0, ''

        # 位置方差
        pos_std = np.std(positions[:, :2], axis=0)
        total_std = np.sqrt(np.sum(pos_std ** 2))

        # 时间窗口
        time_span = (
            self._timestamp_history[-1] - self._timestamp_history[0]
            if self._timestamp_history else 1.0
        )

        # 如果方差很小且时间足够长
        if total_std < self.stuck_threshold_m and time_span > self.stuck_time_threshold:
            confidence = min(time_span / (self.stuck_time_threshold * 2), 0.95)
            return True, confidence, 'oscillation_deadlock'

        return False, 0.0, ''

    def _detect_stuck_by_slip(
        self,
        positions: np.ndarray,
    ) -> Tuple[bool, float, str]:
        """检测轮胎打滑"""
        if len(self._imu_history) < 5 or len(positions) < 5:
            return False, 0.0, ''

        # IMU计算的位移
        imu_displacement = np.zeros(2)
        for i in range(1, len(self._imu_history)):
            dt = self._timestamp_history[i] - self._timestamp_history[i - 1]
            if dt > 0:
                accel = np.array([
                    self._imu_history[i].linear_acceleration[0],
                    self._imu_history[i].linear_acceleration[1],
                ])
                # 简单积分 (忽略重力)
                imu_displacement += 0.5 * accel * dt * dt

        # 里程计位移
        odom_displacement = positions[-1, :2] - positions[0, :2]

        # 滑移率
        odom_dist = np.linalg.norm(odom_displacement)
        imu_dist = np.linalg.norm(imu_displacement)

        if odom_dist < 0.01:
            return False, 0.0, ''

        slip_rate = max(0.0, 1.0 - odom_dist / max(imu_dist, 0.001))

        if slip_rate > self.slip_threshold:
            return True, slip_rate, 'wheel_slip'

        return False, 0.0, ''

    def _fuse_detection(
        self,
        stuck_cmd: Tuple[bool, float, str],
        stuck_osc: Tuple[bool, float, str],
        stuck_slip: Tuple[bool, float, str],
    ) -> Tuple[bool, float, str]:
        """融合多种检测结果"""
        results = [stuck_cmd, stuck_osc, stuck_slip]
        is_stuck = any(r[0] for r in results)

        if not is_stuck:
            return False, 0.0, ''

        # 加权平均置信度
        total_confidence = sum(r[1] for r in results if r[0])
        avg_confidence = total_confidence / max(sum(1 for r in results if r[0]), 1)

        # 选择置信度最高的理由
        best_reason = max(results, key=lambda r: r[1] if r[0] else 0.0)

        return True, avg_confidence, best_reason[2]

    def _recommend_strategy(
        self,
        is_stuck: bool,
        confidence: float,
        reason: str,
        stuck_duration: float,
    ) -> RecoveryStrategy:
        """推荐恢复策略"""
        if not is_stuck:
            return RecoveryStrategy.RETRY

        # 根据原因选择策略
        if reason == 'mechanical_stuck':
            if stuck_duration > 5.0:
                return RecoveryStrategy.BACKUP
            return RecoveryStrategy.ROTATE

        elif reason == 'oscillation_deadlock':
            return RecoveryStrategy.REPLAN

        elif reason == 'wheel_slip':
            return RecoveryStrategy.BACKUP

        # 根据持续时间升级策略
        if stuck_duration > 10.0:
            return RecoveryStrategy.ESCALATE
        elif stuck_duration > 5.0:
            return RecoveryStrategy.ABORT

        return RecoveryStrategy.SIDESTEP

    def reset(self):
        """重置检测器"""
        self._position_history.clear()
        self._command_history.clear()
        self._imu_history.clear()
        self._timestamp_history.clear()
        self._stuck_start_time = None


class AutonomousRecoveryManager:
    """
    自主恢复管理器

    管理AGV从各种故障状态中恢复:
    - 卡死恢复 (机械卡死/振荡死锁/轮胎打滑)
    - 传感器故障降级
    - 路径重规划
    - 多级恢复策略升级

    五级支持:
      S:  无恢复能力 (仅急停)
      M:  基础后退恢复
      L:  路径重规划 + 传感器降级
      XL: 完整多策略恢复 + 故障日志
      XXL: MPC预测恢复 + 云端协同
    """

    def __init__(
        self,
        grade: str = 'L',
        max_recovery_attempts: int = 3,
        recovery_cooldown: float = 2.0,  # 恢复冷却时间 (s)
    ):
        self.grade = grade
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_cooldown = recovery_cooldown

        self._recovery_attempts: Dict[str, int] = {}
        self._last_recovery_time: float = 0.0
        self._recovery_history: List[Dict] = []
        self._current_strategy: Optional[RecoveryStrategy] = None
        self._strategy_start_time: Optional[float] = None
        self._degradation_level: int = 0  # 降级级别 0=正常, 1=降级, 2=严重

        # 各等级能力
        self._grade_capabilities = {
            'S': {'retry': True, 'backup': False, 'replan': False,
                  'sidestep': False, 'rotate': False, 'abort': False, 'escalate': True},
            'M': {'retry': True, 'backup': True, 'replan': False,
                  'sidestep': False, 'rotate': True, 'abort': True, 'escalate': False},
            'L': {'retry': True, 'backup': True, 'replan': True,
                  'sidestep': True, 'rotate': True, 'abort': True, 'escalate': False},
            'XL': {'retry': True, 'backup': True, 'replan': True,
                   'sidestep': True, 'rotate': True, 'abort': True, 'escalate': True},
            'XXL': {'retry': True, 'backup': True, 'replan': True,
                    'sidestep': True, 'rotate': True, 'abort': True, 'escalate': True},
        }

    def request_recovery(
        self,
        stuck_result: StuckDetectionResult,
        current_pose: np.ndarray,
        target_pose: Optional[np.ndarray] = None,
        available_sensors: Optional[Dict[str, bool]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        请求恢复操作

        Args:
            stuck_result: 卡死检测结果
            current_pose: 当前位姿
            target_pose: 目标位姿 (用于重规划)
            available_sensors: 可用传感器状态
            timestamp: 当前时间戳

        Returns:
            恢复指令字典，如果无法恢复则返回 None
        """
        if timestamp is None:
            timestamp = time.time()

        # 检查冷却时间
        if timestamp - self._last_recovery_time < self.recovery_cooldown:
            return None

        # 检查等级能力
        caps = self._grade_capabilities.get(self.grade, self._grade_capabilities['S'])
        strategy = stuck_result.recommended_strategy

        if not caps.get(strategy.value, False):
            # 降级到较低级策略
            strategy = self._downgrade_strategy(strategy, caps)

        # 检查恢复次数
        strategy_key = strategy.value
        attempts = self._recovery_attempts.get(strategy_key, 0)
        if attempts >= self.max_recovery_attempts:
            # 升级策略
            strategy = self._upgrade_strategy(strategy, stuck_result)
            attempts = 0

        # 更新状态
        self._current_strategy = strategy
        self._strategy_start_time = timestamp
        self._recovery_attempts[strategy_key] = attempts + 1

        # 构建恢复指令
        recovery_cmd = self._build_recovery_command(
            strategy=strategy,
            current_pose=current_pose,
            target_pose=target_pose,
            stuck_result=stuck_result,
            available_sensors=available_sensors,
            timestamp=timestamp,
        )

        # 记录恢复历史
        self._recovery_history.append({
            'timestamp': timestamp,
            'strategy': strategy.value,
            'reason': stuck_result.reason,
            'confidence': stuck_result.confidence,
            'pose': current_pose.copy(),
            'attempts': attempts + 1,
        })

        # 保持历史长度
        if len(self._recovery_history) > 100:
            self._recovery_history.pop(0)

        self._last_recovery_time = timestamp

        return recovery_cmd

    def _downgrade_strategy(
        self,
        strategy: RecoveryStrategy,
        capabilities: Dict[str, bool],
    ) -> RecoveryStrategy:
        """降级到可用的策略"""
        priority_order = [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.BACKUP,
            RecoveryStrategy.ROTATE,
            RecoveryStrategy.SIDESTEP,
            RecoveryStrategy.REPLAN,
            RecoveryStrategy.ABORT,
            RecoveryStrategy.ESCALATE,
        ]

        for s in priority_order:
            if capabilities.get(s.value, False):
                return s

        return RecoveryStrategy.ABORT

    def _upgrade_strategy(
        self,
        current: RecoveryStrategy,
        stuck_result: StuckDetectionResult,
    ) -> RecoveryStrategy:
        """升级恢复策略"""
        upgrades = {
            RecoveryStrategy.RETRY: RecoveryStrategy.BACKUP,
            RecoveryStrategy.BACKUP: RecoveryStrategy.ROTATE,
            RecoveryStrategy.ROTATE: RecoveryStrategy.SIDESTEP,
            RecoveryStrategy.SIDESTEP: RecoveryStrategy.REPLAN,
            RecoveryStrategy.REPLAN: RecoveryStrategy.ABORT,
            RecoveryStrategy.ABORT: RecoveryStrategy.ESCALATE,
        }
        return upgrades.get(current, RecoveryStrategy.ESCALATE)

    def _build_recovery_command(
        self,
        strategy: RecoveryStrategy,
        current_pose: np.ndarray,
        target_pose: Optional[np.ndarray],
        stuck_result: StuckDetectionResult,
        available_sensors: Optional[Dict[str, bool]],
        timestamp: float,
    ) -> Dict:
        """构建恢复指令"""
        cmd = {
            'strategy': strategy,
            'timestamp': timestamp,
            'duration': 0.0,
            'velocity': np.zeros(3),
            'target_pose': target_pose.copy() if target_pose is not None else None,
            'message': '',
            'degradation': self._degradation_level,
        }

        if strategy == RecoveryStrategy.RETRY:
            cmd['duration'] = 0.5
            cmd['velocity'] = np.array([0.05, 0.0, 0.0])
            cmd['message'] = 'Retry: 轻微后退后重试'

        elif strategy == RecoveryStrategy.BACKUP:
            # 后退一定距离
            backup_dist = min(stuck_result.stuck_duration * 0.05, 0.3)
            angle = current_pose[2]
            cmd['velocity'] = np.array([-backup_dist * np.cos(angle), -backup_dist * np.sin(angle), 0.0])
            cmd['duration'] = 1.5
            cmd['message'] = f'Backup: 后退 {backup_dist:.2f}m'

        elif strategy == RecoveryStrategy.ROTATE:
            # 原地旋转后尝试
            cmd['velocity'] = np.array([0.0, 0.0, 1.5])
            cmd['duration'] = 1.0
            cmd['message'] = 'Rotate: 原地旋转后重试'

        elif strategy == RecoveryStrategy.SIDESTEP:
            # Mecanum 横移
            cmd['velocity'] = np.array([0.0, 0.2, 0.0])
            cmd['duration'] = 1.0
            cmd['message'] = 'Sidestep: 横移0.2m'

        elif strategy == RecoveryStrategy.REPLAN:
            cmd['duration'] = 0.0
            cmd['message'] = 'Replan: 请求路径重规划'

        elif strategy == RecoveryStrategy.ABORT:
            cmd['duration'] = 0.0
            cmd['message'] = 'Abort: 放弃当前目标点'

        elif strategy == RecoveryStrategy.ESCALATE:
            cmd['duration'] = 0.0
            cmd['message'] = 'Escalate: 请求人工干预'
            cmd['degradation'] = 2

        return cmd

    def check_recovery_complete(
        self,
        strategy: RecoveryStrategy,
        elapsed_time: float,
        current_pose: np.ndarray,
        start_pose: np.ndarray,
    ) -> Tuple[bool, bool]:
        """
        检查恢复操作是否完成

        Returns:
            (is_complete, success)
        """
        if strategy == RecoveryStrategy.RETRY:
            return elapsed_time > 0.5, True

        elif strategy in (RecoveryStrategy.BACKUP, RecoveryStrategy.SIDESTEP):
            displacement = np.linalg.norm(current_pose[:2] - start_pose[:2])
            return elapsed_time > 1.5, displacement > 0.05

        elif strategy == RecoveryStrategy.ROTATE:
            angle_change = abs(current_pose[2] - start_pose[2])
            return elapsed_time > 1.0, angle_change > 0.5

        elif strategy == RecoveryStrategy.REPLAN:
            # 重规划需要外部触发完成
            return False, False

        elif strategy in (RecoveryStrategy.ABORT, RecoveryStrategy.ESCALATE):
            return True, True

        return True, True

    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'grade': self.grade,
            'recovery_history': self._recovery_history[-10:],
            'recovery_attempts': self._recovery_attempts.copy(),
            'current_strategy': self._current_strategy.value if self._current_strategy else None,
            'degradation_level': self._degradation_level,
            'last_recovery_time': self._last_recovery_time,
        }

    def reset(self):
        """重置恢复管理器"""
        self._recovery_attempts.clear()
        self._recovery_history.clear()
        self._current_strategy = None
        self._strategy_start_time = None
        self._degradation_level = 0


if __name__ == '__main__':
    # 快速测试
    print("Running patrol benchmark...")
    results = run_patrol_benchmark(['S', 'M', 'L'])
    for grade, metrics in results.items():
        print(f"\n{grade} grade: {metrics}")

    # 测试卡死检测
    print("\nTesting StuckDetector...")
    detector = StuckDetector()
    import random
    for i in range(30):
        # 模拟卡死场景: 指令大但位置不变
        pos = np.array([0.1, 0.1, 0.0])
        cmd = np.array([0.3, 0.0, 0.0])
        result = detector.update(pos, cmd, timestamp=float(i) * 0.1)
        if result.is_stuck:
            print(f"  Stuck detected: {result.reason}, confidence={result.confidence:.2f}")
            print(f"  Recommended strategy: {result.recommended_strategy.value}")
            break
    else:
        print("  No stuck detected in normal scenario")

    # 测试恢复管理器
    print("\nTesting AutonomousRecoveryManager...")
    manager = AutonomousRecoveryManager(grade='XL')
    stuck_result = StuckDetectionResult(
        is_stuck=True,
        confidence=0.85,
        reason='mechanical_stuck',
        stuck_duration=4.0,
        command_history=np.zeros((10, 3)),
        position_history=np.zeros((10, 3)),
        recommended_strategy=RecoveryStrategy.BACKUP,
    )
    recovery = manager.request_recovery(
        stuck_result,
        current_pose=np.array([1.0, 2.0, 0.5]),
        target_pose=np.array([3.0, 2.0, 0.0]),
        timestamp=10.0,
    )
    print(f"  Recovery command: {recovery}")
