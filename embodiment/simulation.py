"""
Enhanced Embodiment Simulator - 增强型具身仿真环境
支持多AGV仿真、全传感器模拟、物理引擎、场景交互
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
from enum import Enum
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

# from sensors.tactile import TactileArray as TactileSensor
# from sensors.force import SixAxisFTSensor as ForceTorqueSensor
# from sensors.imu import IMUSensor


class SimulationScene(Enum):
    """仿真场景类型"""
    FACTORY_WAREHOUSE = "warehouse"
    LOGISTICS_CENTER = "logistics"
    OUTDOOR_CAMPUS = "outdoor"
    FACTORY_FLOOR = "factory"


@dataclass
class SimAGVConfig:
    """仿真AGV配置"""
    urdf_path: str = "r2d2.urdf"
    start_position: Tuple[float, float, float] = (0.0, 0.0, 0.1)
    start_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    max_velocity: float = 1.5
    max_omega: float = 2.0
    wheel_radius: float = 0.076
    wheel_distance: float = 0.32
    has_tactile_sensor: bool = True
    has_force_sensor: bool = True
    has_imu_sensor: bool = True


@dataclass
class SimSceneConfig:
    """仿真场景配置"""
    scene_type: str = "warehouse"  # warehouse, factory, outdoor
    obstacles: List[Tuple[float, float, float]] = None  # (x, y, radius)
    walls: List[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2, height)
    charging_stations: List[Tuple[float, float]] = None
    cargo_locations: List[Tuple[float, float]] = None


class EmbodimentSimulator:
    """
    具身仿真环境主类
    支持多AGV并行仿真、全传感器模拟、物理碰撞、场景交互
    """

    def __init__(
        self,
        scene_config: SimSceneConfig = None,
        scene: Optional[SimulationScene] = None,
        gui: bool = False,
        dt: float = 0.01
    ):
        # 兼容测试模式：通过scene参数初始化
        if scene is not None:
            scene_config = SimSceneConfig(scene_type=scene.value)
            gui = False  # 测试默认关闭GUI

        self.scene_config = scene_config or SimSceneConfig()
        self.gui = gui
        self.dt = dt
        self.scene_config = scene_config or SimSceneConfig()
        self.gui = gui
        self.dt = dt
        self.client_id = -1
        self.agvs: Dict[int, Dict] = {}  # agv_id -> {config, body_id, sensors, state}
        self.next_agv_id = 0
        self.current_time = 0.0
        self.gravity = -9.81

        # 初始化物理引擎
        self._init_physics()
        # 加载场景
        self._load_scene()

    def _init_physics(self):
        """初始化PyBullet物理引擎"""
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)

    def _load_scene(self):
        """加载仿真场景"""
        # 加载地面
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # 加载仓库场景
        if self.scene_config.scene_type == "warehouse":
            # 货架：用cube代替避免URDF找不到
            for i in range(5):
                for j in range(3):
                    x = 2.0 + i * 1.5
                    y = -3.0 + j * 2.0
                    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.6, 1.0], physicsClientId=self.client_id)
                    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=[x, y, 1.0], physicsClientId=self.client_id)
            # 充电区：用cube代替避免URDF找不到
            if self.scene_config.charging_stations:
                for (x, y) in self.scene_config.charging_stations:
                    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.05], physicsClientId=self.client_id)
                    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=[x, y, 0.05], physicsClientId=self.client_id)
            # 货物：用cube代替避免URDF找不到
            if self.scene_config.cargo_locations:
                for (x, y) in self.scene_config.cargo_locations:
                    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], physicsClientId=self.client_id)
                    p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_shape, basePosition=[x, y, 0.2], physicsClientId=self.client_id)

        # 加载障碍物
        if self.scene_config.obstacles:
            for (x, y, r) in self.scene_config.obstacles:
                col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=0.5, physicsClientId=self.client_id)
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=[x, y, 0.25], physicsClientId=self.client_id)

        # 加载墙壁
        if self.scene_config.walls:
            for (x1, y1, x2, y2, h) in self.scene_config.walls:
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                angle = math.atan2(dy, dx)
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, 0.1, h/2], physicsClientId=self.client_id)
                pos = [(x1+x2)/2, (y1+y2)/2, h/2]
                orn = p.getQuaternionFromEuler([0, 0, angle], physicsClientId=self.client_id)
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=pos, baseOrientation=orn, physicsClientId=self.client_id)

    def add_agv(self, *args, config: SimAGVConfig = None, agv_type: Optional[str] = None, initial_pos: Optional[Tuple[float, float, float]] = None) -> int:
        """
        添加AGV到仿真环境，返回AGV ID
        支持调用方式：
        1. 原生：add_agv(config: SimAGVConfig)
        2. 测试：add_agv(agv_type: str, initial_pos: Tuple[float, float, float])
        3. 测试：add_agv(agv_type="LEVEL_X", initial_pos=(x,y,z))
        """
        # 处理位置参数调用
        if len(args) >= 1 and isinstance(args[0], str) and "LEVEL" in args[0]:
            agv_type = args[0]
            if len(args) >= 2:
                initial_pos = args[1]
        
        if config is None:
            config = SimAGVConfig()
        
        # 兼容测试参数
        if initial_pos is not None:
            config.start_position = initial_pos
        if agv_type is not None:
            level = int(agv_type.split("_")[-1]) if "_" in agv_type else 1
            config.max_velocity = 1.0 + 0.1 * level
        agv_id = self.next_agv_id
        self.next_agv_id += 1

        # 加载AGV URDF
        body_id = p.loadURDF(
            config.urdf_path,
            config.start_position,
            config.start_orientation,
            physicsClientId=self.client_id
        )

        # 初始化传感器（模拟）
        sensors = {}
        if config.has_tactile_sensor:
            sensors["tactile"] = []
        if config.has_force_sensor:
            sensors["force_torque"] = []
        if config.has_imu_sensor:
            sensors["imu"] = {}

        # 初始化AGV状态
        self.agvs[agv_id] = {
            "config": config,
            "body_id": body_id,
            "sensors": sensors,
            "state": {
                "x": config.start_position[0],
                "y": config.start_position[1],
                "theta": 0.0,
                "v": 0.0,
                "omega": 0.0,
                "battery_level": 1.0,
                "obstacles": [],
                "gripper_state": "open"
            }
        }

        # 启用关节力矩传感器
        for joint in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
            p.enableJointForceTorqueSensor(body_id, joint, enableSensor=True, physicsClientId=self.client_id)

        return agv_id

    def set_agv_command(self, agv_id: int, v: float, omega: float):
        """设置AGV的运动指令：线速度v (m/s)，角速度omega (rad/s)"""
        if agv_id not in self.agvs:
            return
        agv = self.agvs[agv_id]
        config = agv["config"]

        # 差速运动学转换：v, omega -> 左右轮速度
        v_left = v - omega * config.wheel_distance / 2
        v_right = v + omega * config.wheel_distance / 2

        # 转换为角速度 (rad/s)
        w_left = v_left / config.wheel_radius
        w_right = v_right / config.wheel_radius

        # 设置电机速度
        p.setJointMotorControl2(
            agv["body_id"], 0, p.VELOCITY_CONTROL,
            targetVelocity=w_left, force=100,
            physicsClientId=self.client_id
        )
        p.setJointMotorControl2(
            agv["body_id"], 1, p.VELOCITY_CONTROL,
            targetVelocity=w_right, force=100,
            physicsClientId=self.client_id
        )

    def set_gripper_command(self, agv_id: int, command: str):
        """设置夹爪指令：open/close/hold"""
        if agv_id not in self.agvs:
            return
        agv = self.agvs[agv_id]
        agv["state"]["gripper_state"] = command

        if command == "open":
            target = 0.05
        elif command == "close":
            target = 0.0
        else:
            return

        # 设置夹爪关节位置
        p.setJointMotorControl2(
            agv["body_id"], 2, p.POSITION_CONTROL,
            targetPosition=target, force=50,
            physicsClientId=self.client_id
        )

    def step(self, duration: Optional[float] = None) -> Dict:
        """执行一步或多步仿真，返回当前状态（支持测试兼容参数）"""
        if duration is None:
            duration = self.dt
        
        # 执行多步仿真
        steps = int(duration / self.dt)
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            self.current_time += self.dt

        # 更新每个AGV的状态
        all_states = {}
        for agv_id, agv in self.agvs.items():
            body_id = agv["body_id"]

            # 获取位置和朝向
            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
            euler = p.getEulerFromQuaternion(orn, physicsClientId=self.client_id)
            x, y, _ = pos
            theta = euler[2]

            # 获取速度
            lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
            v = math.hypot(lin_vel[0], lin_vel[1])
            omega = ang_vel[2]

            # 更新状态
            agv["state"]["x"] = x
            agv["state"]["y"] = y
            agv["state"]["theta"] = theta
            agv["state"]["v"] = v
            agv["state"]["omega"] = omega
            # 模拟电池消耗
            agv["state"]["battery_level"] = max(0.0, agv["state"]["battery_level"] - 0.0001 * v)

            # 障碍物检测 (激光雷达模拟)
            obstacles = []
            for angle in np.linspace(-math.pi/2, math.pi/2, 18):
                ray_from = [x, y, 0.2]
                ray_to = [
                    x + math.cos(theta + angle) * 3.0,
                    y + math.sin(theta + angle) * 3.0,
                    0.2
                ]
                result = p.rayTest(ray_from, ray_to, physicsClientId=self.client_id)
                hit_fraction = result[0][2]
                if hit_fraction < 1.0:
                    dist = hit_fraction * 3.0
                    ox = x + math.cos(theta + angle) * dist
                    oy = y + math.sin(theta + angle) * dist
                    obstacles.append((ox, oy, 0.2))
            agv["state"]["obstacles"] = obstacles

            # 更新传感器数据（模拟）
            sensor_data = {}
            if "imu" in agv["sensors"]:
                # 模拟IMU数据
                sensor_data["imu"] = {
                    "accelerometer": [lin_vel[0]/self.dt if self.dt !=0 else 0, lin_vel[1]/self.dt if self.dt !=0 else 0, self.gravity],
                    "gyroscope": ang_vel,
                    "timestamp": self.current_time
                }
            if "force_torque" in agv["sensors"]:
                # 模拟力传感器数据
                sensor_data["force_torque"] = [0.0]*6
            if "tactile" in agv["sensors"]:
                # 模拟触觉传感器数据
                sensor_data["tactile"] = [0.0]*16

            all_states[agv_id] = {
                "state": agv["state"].copy(),
                "sensors": sensor_data,
                "current_time": self.current_time
            }

        if self.gui:
            time.sleep(self.dt)

        # 兼容测试返回格式
        return {
            "time": self.current_time,
            "agvs": all_states,
            "obstacles": []
        }

    def reset(self):
        """重置仿真环境"""
        for agv_id, agv in self.agvs.items():
            config = agv["config"]
            p.resetBasePositionAndOrientation(
                agv["body_id"],
                config.start_position,
                config.start_orientation,
                physicsClientId=self.client_id
            )
            p.resetBaseVelocity(
                agv["body_id"],
                [0, 0, 0],
                [0, 0, 0],
                physicsClientId=self.client_id
            )
            agv["state"]["battery_level"] = 1.0
        self.current_time = 0.0

    def close(self):
        """关闭仿真环境"""
        if self.client_id >= 0:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = -1

    def add_obstacle(self, type: str, position: Tuple[float, float, float], size: Tuple[float, ...]):
        """添加障碍物（测试兼容接口）"""
        if type == "box":
            half_extents = [s/2 for s in size]
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=position, physicsClientId=self.client_id)
        elif type == "cylinder":
            radius, height = size
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.client_id)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=position, physicsClientId=self.client_id)

    def check_collision(self, agv_id: int) -> bool:
        """检查AGV是否发生碰撞（测试兼容接口）"""
        if agv_id not in self.agvs:
            return False
        body_id = self.agvs[agv_id]["body_id"]
        contact_points = p.getContactPoints(body_id, physicsClientId=self.client_id)
        # 排除和地面的碰撞（地面的body id是0）
        for cp in contact_points:
            if cp[2] != 0:
                return True
        return False

    def get_current_state(self) -> Dict:
        """获取当前仿真状态（测试兼容接口）"""
        return {
            "time": self.current_time,
            "agvs": self.agvs.copy(),
            "obstacles": []
        }


# 测试兼容别名
EmbodiedSimulation = EmbodimentSimulator
