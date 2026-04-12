"""
Enhanced Embodiment Simulator - 增强型具身仿真环境
支持多AGV仿真、全传感器模拟、物理引擎、场景交互
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

from sensors.tactile import TactileArray as TactileSensor
from sensors.force import ForceTorqueSensor
from sensors.imu import IMUSensor


@dataclass
class SimAGVConfig:
    """仿真AGV配置"""
    urdf_path: str = "urdf/agv_v2.urdf"
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
        gui: bool = True,
        dt: float = 0.01
    ):
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
            # 货架
            for i in range(5):
                for j in range(3):
                    x = 2.0 + i * 1.5
                    y = -3.0 + j * 2.0
                    p.loadURDF("urdf/shelf.urdf", [x, y, 0.0], physicsClientId=self.client_id)
            # 充电区
            if self.scene_config.charging_stations:
                for (x, y) in self.scene_config.charging_stations:
                    p.loadURDF("urdf/charging_station.urdf", [x, y, 0.01], physicsClientId=self.client_id)
            # 货物
            if self.scene_config.cargo_locations:
                for (x, y) in self.scene_config.cargo_locations:
                    p.loadURDF("urdf/cargo_box.urdf", [x, y, 0.2], physicsClientId=self.client_id)

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

    def add_agv(self, config: SimAGVConfig = None) -> int:
        """添加AGV到仿真环境，返回AGV ID"""
        config = config or SimAGVConfig()
        agv_id = self.next_agv_id
        self.next_agv_id += 1

        # 加载AGV URDF
        body_id = p.loadURDF(
            config.urdf_path,
            config.start_position,
            config.start_orientation,
            physicsClientId=self.client_id
        )

        # 初始化传感器
        sensors = {}
        if config.has_tactile_sensor:
            sensors["tactile"] = TactileSensor()
        if config.has_force_sensor:
            sensors["force_torque"] = ForceTorqueSensor()
        if config.has_imu_sensor:
            sensors["imu"] = IMUSensor()

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

    def step(self) -> Dict[int, Dict]:
        """执行一步仿真，返回所有AGV的最新状态和传感器数据"""
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

            # 更新传感器数据
            sensor_data = {}
            if "imu" in agv["sensors"]:
                # 模拟IMU数据
                accel = [lin_vel[0]/self.dt, lin_vel[1]/self.dt, self.gravity]
                gyro = ang_vel
                sensor_data["imu"] = agv["sensors"]["imu"].read(accel, gyro, self.current_time)
            if "force_torque" in agv["sensors"]:
                # 读取力传感器数据
                ft = p.getJointState(body_id, 2, physicsClientId=self.client_id)[2]
                sensor_data["force_torque"] = agv["sensors"]["force_torque"].read(ft[:3], ft[3:], self.current_time)
            if "tactile" in agv["sensors"]:
                # 模拟触觉传感器数据
                contact_points = p.getContactPoints(body_id, physicsClientId=self.client_id)
                pressure = len(contact_points) * 0.1
                sensor_data["tactile"] = agv["sensors"]["tactile"].read([pressure]*16, self.current_time)

            all_states[agv_id] = {
                "state": agv["state"].copy(),
                "sensors": sensor_data,
                "current_time": self.current_time
            }

        if self.gui:
            time.sleep(self.dt)

        return all_states

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
