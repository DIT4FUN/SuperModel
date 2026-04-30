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
embodied_sim.py - 增强具身仿真环境
SuperModel 超模态大模型具身智能系统

扩展PyBullet仿真环境，增加:
- 多模态传感器仿真 (触觉/力觉/IMU/视觉/听觉)
- 接触力仿真
- 摩擦和滑移仿真
- 动态物体交互
- 仓库/物流场景生成
- 多AGV协同仿真
- 具身任务评估指标
"""

from __future__ import annotations
import abc
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pybullet as p
import pybullet_data

from sensors.tactile import TactileArray, VirtualTactileSensor
from sensors.force import ForceTorqueSensor, VirtualForceSensor
from sensors.imu import IMUSensor, VirtualIMUSensor, PoseEstimator
from simulation.agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS

__all__ = [
    'EmbodiedSimEnv',
    'WarehouseScene',
    'EmbodiedSensorSimulator',
    'TaskMetrics',
    'MultiAGVEmbodiedSim',
    'create_embodied_sim',
]


@dataclass
class ContactPoint:
    """接触点"""
    position: np.ndarray
    normal: np.ndarray
    force: np.ndarray
    distance: float
    body_a: int
    body_b: int
    link_a: int = -1
    link_b: int = -1


@dataclass
class TaskMetrics:
    """具身任务评估指标"""
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    total_distance: float = 0.0
    collisions: int = 0
    min_obstacle_distance: float = float('inf')
    completion_time: float = 0.0
    energy_consumed: float = 0.0
    path_smoothness: float = 0.0
    force_regulation_error: float = 0.0

    def finish(self, success: bool) -> None:
        self.end_time = time.time()
        self.completion_time = self.end_time - self.start_time
        self.success = success

    def get_score(self) -> float:
        """计算综合得分 [0-100]"""
        if not self.success:
            return 0.0

        # 时间得分 (越快越好)
        time_score = max(0, 1 - self.completion_time / 60.0) * 30

        # 碰撞扣分
        collision_penalty = min(30, self.collisions * 10)

        # 距离得分 (路径越短越好，假设最优距离是直线)
        # 这里简化处理
        distance_score = 25

        # 平滑度得分
        smoothness_score = (1 - self.path_smoothness) * 15

        # 力控得分
        force_score = max(0, 1 - self.force_regulation_error / 10.0) * 30

        score = time_score + distance_score + smoothness_score + force_score - collision_penalty
        return max(0, min(100, score))


class EmbodiedSensorSimulator:
    """
    具身传感器仿真器
    在PyBullet仿真环境中模拟多模态传感器
    """

    def __init__(self, client: int, agv_id: int, base_link: int = 0):
        self.client = client
        self.agv_id = agv_id
        self.base_link = base_link

        # 虚拟传感器
        self.virtual_tactile: Optional[VirtualTactileSensor] = None
        self.virtual_force: Optional[VirtualForceSensor] = None
        self.virtual_imu: Optional[VirtualIMUSensor] = None
        self.pose_estimator: Optional[MadgwickPoseEstimator] = None

        # 接触检测参数
        self.contact_threshold = 0.01
        self.tactile_grid_size = (16, 16)

        # 噪声参数
        self.add_noise = True
        self.imu_noise_std = 0.01
        self.force_noise_std = 0.1
        self.tactile_noise_std = 0.05

    def setup_tactile(self, grid_size: Tuple[int, int] = (16, 16),
                      attachment_link: int = -1) -> VirtualTactileSensor:
        """设置虚拟触觉传感器"""
        self.tactile_grid_size = grid_size
        self.virtual_tactile = VirtualTactileSensor(
            array_size=grid_size,
            sensor_id="virtual"
        )
        return self.virtual_tactile

    def setup_force(self) -> VirtualForceSensor:
        """设置虚拟力传感器"""
        self.virtual_force = VirtualForceSensor()
        return self.virtual_force

    def setup_imu(self) -> VirtualIMUSensor:
        """设置虚拟IMU"""
        self.virtual_imu = VirtualIMUSensor()
        self.pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=100)
        return self.virtual_imu

    def get_contacts(self, body_a: int, link_a: int = -1) -> List[ContactPoint]:
        """获取两个物体之间的接触点"""
        contacts = p.getContactPoints(bodyA=body_a, linkIndexA=link_a, physicsClientId=self.client)
        result = []
        for c in contacts:
            cp = ContactPoint(
                position=np.array(c[5]),
                normal=np.array(c[7]),
                force=np.array(c[9]) * c[10],  # normal force * normal direction
                distance=c[8],
                body_a=c[1],
                body_b=c[2],
                link_a=c[3],
                link_b=c[4],
            )
            result.append(cp)
        return result

    def simulate_tactile(self, gripper_id: int, link_id: int) -> np.ndarray:
        """模拟触觉阵列读数 - 直接生成接触压力图"""
        if self.virtual_tactile is None:
            raise RuntimeError("Call setup_tactile first")

        rows, cols = self.virtual_tactile.array_size
        pressure = np.zeros((rows, cols))
        
        contacts = self.get_contacts(gripper_id, link_id)
        # 简化：在网格中心添加接触压力
        if len(contacts) > 0:
            pressure[rows//2, cols//2] = 10.0 + np.random.rand() * 10.0

        if self.add_noise:
            pressure += np.random.normal(0, self.tactile_noise_std, pressure.shape)

        return pressure

    def simulate_force(self, joint_index: int) -> np.ndarray:
        """模拟六维力传感器读数"""
        if self.virtual_force is None:
            raise RuntimeError("Call setup_force first")

        # 从仿真中获取关节力
        # PyBullet 会返回每个关节的力/力矩
        # 这里简化处理，通过接触点积分
        force = np.zeros(6)  # [fx, fy, fz, mx, my, mz]

        contacts = self.get_contacts(self.agv_id, joint_index)
        for c in contacts:
            force[0:3] += c.force

        if self.add_noise:
            force += np.random.normal(0, self.force_noise_std, 6)

        return force

    def simulate_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        """模拟IMU读数"""
        if self.virtual_imu is None:
            raise RuntimeError("Call setup_imu first")

        # 从PyBullet获取位姿和速度
        pos, orn = p.getBasePositionAndOrientation(self.agv_id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.agv_id, physicsClientId=self.client)

        acc = np.array(lin_vel)  # 简化，实际应该计算加速度
        gyro = np.array(ang_vel)

        if self.add_noise:
            acc += np.random.normal(0, self.imu_noise_std, 3)
            gyro += np.random.normal(0, self.imu_noise_std / 10, 3)

        if self.pose_estimator:
            self.pose_estimator.update(gyro, acc, dt=0.01)

        return acc, gyro

    def get_estimated_pose(self) -> Optional[np.ndarray]:
        """获取姿态估计结果"""
        if self.pose_estimator:
            return self.pose_estimator.get_quaternion()
        return None


class EmbodiedSimEnv:
    """
    增强具身仿真环境
    基于PyBullet，支持单个AGV的具身任务仿真
    """

    def __init__(self, gui: bool = False, gravity: Tuple[float, float, float] = (0, 0, -9.81)):
        self.gui = gui
        self.gravity = gravity

        # 连接PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(gravity[0], gravity[1], gravity[2])
        p.setTimeStep(1.0 / 240.0)

        # 环境状态
        self.plane_id: Optional[int] = None
        self.agv_id: Optional[int] = None
        self.objects: Dict[str, int] = {}
        self.obstacles: List[int] = []
        self.goal_region: Optional[Tuple[np.ndarray, float]] = None

        # 传感器仿真
        self.sensor_sim: Optional[EmbodiedSensorSimulator] = None

        # 当前指标
        self.current_metrics: Optional[TaskMetrics] = None

        # 历史轨迹
        self.trajectory: List[np.ndarray] = []

        # AGV配置
        self.agv_grade: str = "M"
        self.agv_urdf_path: Optional[str] = None

    def reset(self) -> None:
        """重置环境"""
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        self.objects.clear()
        self.obstacles.clear()
        self.goal_region = None
        self.trajectory.clear()
        self.current_metrics = None

        # 重新添加地面
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

    def load_agv(self, grade: str = "M", position: Tuple[float, float, float] = (0, 0, 0.15)) -> int:
        """加载AGV模型"""
        self.agv_grade = grade
        urdf_path = generate_agv_urdf_detailed(grade, "2轮" if grade in ['S', 'M'] else "4轮")
        self.agv_urdf_path = urdf_path
        self.agv_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client
        )

        # 设置传感器仿真
        self.sensor_sim = EmbodiedSensorSimulator(self.client, self.agv_id)

        return self.agv_id

    def add_box(self, name: str, half_extents: Tuple[float, float, float],
                position: Tuple[float, float, float], color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1),
                mass: float = 0.0) -> int:
        """添加方块障碍物/目标物体"""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )
        self.objects[name] = body_id
        if mass > 0:
            self.obstacles.append(body_id)
        else:
            # 静态障碍物也加入障碍列表
            if half_extents[0] > 0.1 or half_extents[1] > 0.1 or half_extents[2] > 0.1:
                self.obstacles.append(body_id)
        return body_id

    def add_warehouse_shelf(self, position: Tuple[float, float, float],
                            size: Tuple[float, float, float] = (1.0, 0.5, 2.0)) -> int:
        """添加货架"""
        # 简化为长方体框
        half_x, half_y, half_z = size[0]/2, size[1]/2, size[2]/2
        return self.add_box(f"shelf_{len(self.objects)}", (half_x, half_y, half_z),
                           position, (0.7, 0.7, 0.7, 1), mass=0)

    def set_goal(self, position: np.ndarray, radius: float = 0.3) -> None:
        """设置目标区域"""
        self.goal_region = (position, radius)

    def is_goal_reached(self) -> bool:
        """检查是否到达目标"""
        if self.goal_region is None or self.agv_id is None:
            return False
        pos, _ = p.getBasePositionAndOrientation(self.agv_id, physicsClientId=self.client)
        agv_pos = np.array(pos[:2])
        goal_pos, goal_radius = self.goal_region
        distance = np.linalg.norm(agv_pos - goal_pos[:2])
        return distance < goal_radius

    def get_robot_position(self) -> np.ndarray:
        """获取机器人当前位置"""
        if self.agv_id is None:
            return np.zeros(3)
        pos, _ = p.getBasePositionAndOrientation(self.agv_id, physicsClientId=self.client)
        return np.array(pos)

    def get_robot_velocity(self) -> np.ndarray:
        """获取机器人当前速度"""
        if self.agv_id is None:
            return np.zeros(3)
        lin_vel, _ = p.getBaseVelocity(self.agv_id, physicsClientId=self.client)
        return np.array(lin_vel)

    def check_collision(self) -> bool:
        """检查机器人是否发生碰撞"""
        if self.agv_id is None:
            return False
        contacts = p.getContactPoints(bodyA=self.agv_id, physicsClientId=self.client)
        # 如果和障碍物有接触，就算碰撞
        for c in contacts:
            body_b = c[2]
            if body_b in self.obstacles:
                return True
        return False

    def min_obstacle_distance(self) -> float:
        """计算到最近障碍物的距离"""
        if self.agv_id is None:
            return float('inf')
        agv_pos = np.array(self.get_robot_position()[:2])
        min_dist = float('inf')

        for obs_id in self.obstacles:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.client)
            obs_pos = np.array(obs_pos[:2])
            dist = np.linalg.norm(agv_pos - obs_pos)
            min_dist = min(min_dist, dist)

        return min_dist

    def set_wheel_velocity(self, left_velocity: float, right_velocity: float) -> None:
        """设置左右轮速度 (rad/s)"""
        # 需要根据URDF中的关节索引来设置
        # 假设轮子关节索引是 1 和 2
        if self.agv_id is None:
            return
        p.setJointMotorControl2(
            self.agv_id, 1, p.VELOCITY_CONTROL,
            targetVelocity=left_velocity, force=100,
            physicsClientId=self.client
        )
        p.setJointMotorControl2(
            self.agv_id, 2, p.VELOCITY_CONTROL,
            targetVelocity=right_velocity, force=100,
            physicsClientId=self.client
        )

    def step(self, num_steps: int = 1) -> None:
        """执行仿真步"""
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self.client)

        # 记录轨迹
        if self.agv_id is not None:
            self.trajectory.append(self.get_robot_position())

        # 更新指标
        if self.current_metrics:
            if self.check_collision():
                self.current_metrics.collisions += 1
            min_dist = self.min_obstacle_distance()
            self.current_metrics.min_obstacle_distance = min(
                self.current_metrics.min_obstacle_distance,
                min_dist
            )

    def start_task(self, task_id: str) -> TaskMetrics:
        """开始任务，重置指标"""
        metrics = TaskMetrics(task_id=task_id, start_time=time.time())
        self.current_metrics = metrics
        return metrics

    def finish_current_task(self, success: Optional[bool] = None) -> TaskMetrics:
        """结束当前任务"""
        if self.current_metrics is None:
            raise RuntimeError("No task started")

        if success is None:
            success = self.is_goal_reached()

        self.current_metrics.finish(success)

        # 计算路径长度
        if len(self.trajectory) > 1:
            total_dist = 0.0
            for i in range(1, len(self.trajectory)):
                dist = np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
                total_dist += dist
            self.current_metrics.total_distance = total_dist

        return self.current_metrics

    def get_sensor_readings(self) -> Dict[str, Any]:
        """获取所有仿真传感器读数"""
        if self.sensor_sim is None:
            return {}

        readings = {}

        if self.sensor_sim.virtual_tactile:
            readings['tactile'] = self.sensor_sim.simulate_tactile(self.agv_id, -1)

        if self.sensor_sim.virtual_force:
            readings['force'] = self.sensor_sim.simulate_force(-1)

        if self.sensor_sim.virtual_imu:
            acc, gyro = self.sensor_sim.simulate_imu()
            readings['imu_acc'] = acc
            readings['imu_gyro'] = gyro
            readings['imu_quat'] = self.sensor_sim.get_estimated_pose()

        return readings

    def close(self) -> None:
        """关闭环境"""
        p.disconnect(physicsClientId=self.client)

    def get_camera_image(self, width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
        """获取相机图像，用于视觉仿真"""
        if self.agv_id is None:
            return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width))

        # 从AGV相机位置获取图像
        agv_pos = self.get_robot_position()
        camera_pos = [agv_pos[0] + 0.1, agv_pos[1], agv_pos[2] + 0.3]
        target_pos = [agv_pos[0] + 1.0, agv_pos[1], agv_pos[2] + 0.2]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.client
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=10.0,
            physicsClientId=self.client
        )

        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER,
            physicsClientId=self.client
        )

        rgb_np = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
        depth_np = np.reshape(depth_img, (height, width))

        return rgb_np, depth_np


class WarehouseScene:
    """仓库场景生成器"""

    def __init__(self, env: EmbodiedSimEnv):
        self.env = env
        self.rows = 4
        self.cols = 6
        self.shelf_width = 1.0
        self.shelf_depth = 0.5
        self.aisle_width = 1.5
        self.start_position: Tuple[float, float, float] = (2.0, 0, 0.15)
        self.goal_position: Tuple[float, float, float] = (8.0, 3.0, 0.15)

    def generate(self) -> None:
        """生成标准仓库场景"""
        self.env.reset()

        # 生成货架阵列
        start_x = 0.0
        start_y = -2.0

        for row in range(self.rows):
            for col in range(self.cols):
                x = start_x + col * (self.shelf_width + self.aisle_width)
                y = start_y + row * (self.shelf_depth + self.aisle_width)
                z = 1.0
                self.env.add_warehouse_shelf((x, y, z), (self.shelf_width/2, self.shelf_depth/2, 2.0))

        # 设置起点和目标
        self.env.load_agv("M", self.start_position)
        self.env.set_goal(np.array(self.goal_position[:2]), 0.5)

        # 在通道中随机放置一些箱子作为动态障碍物
        for i in range(5):
            x = random.uniform(2, 7)
            y = random.uniform(-1.5, 3.5)
            self.env.add_box(f"box_{i}", (0.3, 0.3, 0.3), (x, y, 0.3), (0.2, 0.5, 0.8, 1), mass=1.0)


class MultiAGVEmbodiedSim:
    """多AGV具身协同仿真"""

    def __init__(self, gui: bool = False):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self.agv_ids: Dict[str, int] = {}
        self.agents: Dict[str, EmbodiedSensorSimulator] = {}
        self.tasks: Dict[str, TaskMetrics] = {}
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

    def add_agv(self, name: str, grade: str, position: Tuple[float, float, float]) -> int:
        """添加AGV"""
        urdf_path = generate_agv_urdf_detailed(grade, "2轮" if grade in ['S', 'M'] else "4轮")
        agv_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client
        )
        self.agv_ids[name] = agv_id
        self.agents[name] = EmbodiedSensorSimulator(self.client, agv_id)
        return agv_id

    def set_goal(self, agv_name: str, position: np.ndarray, radius: float = 0.3) -> None:
        """设置目标"""
        pass  # 存储每个AGV的目标

    def check_collisions_between_agvs(self) -> List[Tuple[str, str]]:
        """检测AGV之间的碰撞"""
        collisions = []
        agv_names = list(self.agv_ids.keys())
        for i in range(len(agv_names)):
            for j in range(i+1, len(agv_names)):
                name_a, name_b = agv_names[i], agv_names[j]
                id_a, id_b = self.agv_ids[name_a], self.agv_ids[name_b]
                contacts = p.getContactPoints(bodyA=id_a, bodyB=id_b, physicsClientId=self.client)
                if len(contacts) > 0:
                    collisions.append((name_a, name_b))
        return collisions

    def step(self, num_steps: int = 1) -> None:
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self.client)

    def close(self) -> None:
        p.disconnect(physicsClientId=self.client)


def create_embodied_sim(grade: str = "M", scene_type: str = "empty", gui: bool = False) -> EmbodiedSimEnv:
    """工厂方法创建具身仿真环境"""
    env = EmbodiedSimEnv(gui=gui)
    env.reset()
    env.load_agv(grade)

    if scene_type == "warehouse":
        scene = WarehouseScene(env)
        scene.generate()
    elif scene_type == "navigation":
        # 简单导航任务：起点到终点，几个障碍物
        env.add_box("obs1", (0.2, 0.2, 0.5), (2.0, 0.5, 0.5), mass=0)
        env.add_box("obs2", (0.2, 0.2, 0.5), (2.0, -0.5, 0.5), mass=0)
        env.set_goal(np.array([5.0, 0.0]), 0.3)

    return env
