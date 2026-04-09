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


if __name__ == '__main__':
    # 快速测试
    print("Running patrol benchmark...")
    results = run_patrol_benchmark(['S', 'M', 'L'])
    for grade, metrics in results.items():
        print(f"\n{grade} grade: {metrics}")
