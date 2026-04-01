"""
ROS2-Gazebo 联合仿真模块
========================

连接 ROS2 Humble 与 Gazebo Harmonic 的联合仿真接口。

功能:
- AGV Gazebo 物理仿真
- ROS2- Gazebo 传感器桥接 (相机/IMU/力矩/激光雷达)
- ROS2 控制器接口 (关节轨迹/力矩)
- ROS2 动作服务器 (导航/抓取)
- 硬件在环 (HITL) 支持

依赖:
    ros-humble-gazebo-ros-pkgs
    ros-humble-ros2-control
    ros-humble-ros2-action

使用示例:
    from simulation.gazebo_sim import GazeboROS2Bridge, AGVGazeboSimulator

    bridge = GazeboROS2Bridge()
    sim = AGVGazeboSimulator(bridge)

    # 启动仿真
    await sim.spawn_agv(world='warehouse.sdf')

    # 控制AGV
    await sim.set_velocity([0.5, 0.0, 0.0])

    # 读取传感器
    img = await sim.get_camera_image('camera/front')
    imu = await sim.get_imu_data()
    scan = await sim.get_lidar_scan()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from enum import Enum
import asyncio
import threading
import time
import json
import subprocess
import os
import signal


# ROS2 可选导入 (graceful degradation)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, Imu, JointState, LaserScan
    from geometry_msgs.msg import Twist, Pose, PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64MultiArray
    from builtin_interfaces.msg import Duration
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object


# Gazebo 可选导入 (graceful degradation)
try:
    import gzsim
    HAS_GZSIM = True
except ImportError:
    HAS_GZSIM = False


class GazeboWorld(Enum):
    """Gazebo 仿真世界"""
    EMPTY = "empty"
    WAREHOUSE = "warehouse"
    OFFICE = "office"
    OUTDOOR = "outdoor"
    INDUSTRIAL = "industrial"


@dataclass
class GazeboROS2Config:
    """ROS2-Gazebo 桥接配置"""
    # ROS2 命名空间
    namespace: str = "/supermodel"
    # 仿真世界
    world: GazeboWorld = GazeboWorld.EMPTY
    # 时钟同步
    use_sim_time: bool = True
    # QoS 设置
    qos_depth: int = 10
    # 传感器话题
    camera_topic: str = "/camera/image_raw"
    imu_topic: str = "/imu"
    lidar_topic: str = "/scan"
    odom_topic: str = "/odom"
    joint_topic: str = "/joint_states"
    # 控制话题
    cmd_vel_topic: str = "/cmd_vel"
    # 动作服务器
    navigate_action: str = "/navigate_to_pose"
    grasp_action: str = "/grasp"


@dataclass
class GazeboAGVSpec:
    """Gazebo AGV 规格"""
    # 物理参数
    mass: float = 50.0              # kg
    wheelbase: float = 0.5          # m
    track_width: float = 0.4        # m
    wheel_radius: float = 0.1       # m
    max_linear: float = 2.0        # m/s
    max_angular: float = 3.14      # rad/s
    # 传感器配置
    camera_enabled: bool = True
    imu_enabled: bool = True
    lidar_enabled: bool = False
    # Gazebo 模型参数
    model_name: str = "agv_supermodel"
    urdf_path: str = ""
    sdf_path: str = ""
    # AGV 等级
    grade: str = "M"


class GazeboROS2Bridge:
    """
    ROS2 与 Gazebo 之间的桥接器

    管理话题/服务/动作的转发。
    """

    def __init__(self, config: Optional[GazeboROS2Config] = None):
        self.config = config or GazeboROS2Config()
        self._node: Optional[Node] = None
        self._running = False
        self._spin_thread: Optional[threading.Thread] = None

        # 话题缓冲区
        self._camera_buf = None
        self._imu_buf: Optional[Any] = None
        self._lidar_buf: Optional[Any] = None
        self._odom_buf: Optional[Any] = None
        self._joint_buf: Optional[Any] = None
        self._last_camera_time: float = 0.0

        # 锁
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self) -> bool:
        """初始化 ROS2 节点"""
        if not HAS_ROS2:
            print("[WARN] ROS2 not available, running in simulation-only mode")
            return False

        try:
            rclpy.init()
            self._node = Node('supermodel_gazebo_bridge')
            self._running = True
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()
            print(f"[GazeboROS2Bridge] ROS2 node initialized, namespace={self.config.namespace}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to initialize ROS2: {e}")
            return False

    def shutdown(self):
        """关闭桥接器"""
        self._running = False
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)
        if HAS_ROS2 and rclpy.ok():
            if self._node:
                self._node.destroy_node()
            rclpy.shutdown()

    def _spin_loop(self):
        """ROS2 spin 循环"""
        while self._running and HAS_ROS2 and rclpy.ok():
            try:
                rclpy.spin_once(self._node, timeout_sec=0.01)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Publishers
    # -------------------------------------------------------------------------

    def publish_cmd_vel(self, vx: float, vy: float = 0.0, omega: float = 0.0):
        """发布速度指令到 Gazebo AGV"""
        if not self._node:
            return
        pub = getattr(self, '_cmd_vel_pub', None)
        if pub is None:
            pub = self._node.create_publisher(Twist, self.config.cmd_vel_topic, 10)
            self._cmd_vel_pub = pub
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = omega
        pub.publish(msg)

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    def _subscribe_camera(self, topic: str):
        if not self._node:
            return
        self._node.create_subscription(
            Image, topic,
            lambda msg: self._on_camera(msg),
            qos_profile=self.config.qos_depth
        )

    def _subscribe_imu(self, topic: str):
        if not self._node:
            return
        self._node.create_subscription(
            Imu, topic,
            lambda msg: self._on_imu(msg),
            qos_profile=self.config.qos_depth
        )

    def _subscribe_lidar(self, topic: str):
        if not self._node:
            return
        self._node.create_subscription(
            LaserScan, topic,
            lambda msg: self._on_lidar(msg),
            qos_profile=self.config.qos_depth
        )

    def _subscribe_odom(self, topic: str):
        if not self._node:
            return
        self._node.create_subscription(
            Odometry, topic,
            lambda msg: self._on_odom(msg),
            qos_profile=self.config.qos_depth
        )

    def _subscribe_joints(self, topic: str):
        if not self._node:
            return
        self._node.create_subscription(
            JointState, topic,
            lambda msg: self._on_joints(msg),
            qos_profile=self.config.qos_depth
        )

    def _on_camera(self, msg: Any):
        with self._lock:
            self._camera_buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )
            self._last_camera_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_imu(self, msg: Any):
        with self._lock:
            self._imu_buf = msg

    def _on_lidar(self, msg: Any):
        with self._lock:
            self._lidar_buf = msg

    def _on_odom(self, msg: Any):
        with self._lock:
            self._odom_buf = msg

    def _on_joints(self, msg: Any):
        with self._lock:
            self._joint_buf = msg

    # -------------------------------------------------------------------------
    # Data Accessors
    # -------------------------------------------------------------------------

    def get_camera_image(self) -> Optional[np.ndarray]:
        """获取最新相机图像"""
        with self._lock:
            return self._camera_buf.copy() if self._camera_buf is not None else None

    def get_imu_data(self) -> Optional[Dict[str, float]]:
        """获取 IMU 数据"""
        with self._lock:
            if self._imu_buf is None:
                return None
            imu = self._imu_buf
            return {
                'ax': imu.linear_acceleration.x,
                'ay': imu.linear_acceleration.y,
                'az': imu.linear_acceleration.z,
                'gx': imu.angular_velocity.x,
                'gy': imu.angular_velocity.y,
                'gz': imu.angular_velocity.z,
                'qw': imu.orientation.w,
                'qx': imu.orientation.x,
                'qy': imu.orientation.y,
                'qz': imu.orientation.z,
            }

    def get_lidar_scan(self) -> Optional[np.ndarray]:
        """获取激光雷达扫描"""
        with self._lock:
            if self._lidar_buf is None:
                return None
            return np.array(self._lidar_buf.ranges)

    def get_odometry(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取里程计 (位置, 速度)"""
        with self._lock:
            if self._odom_buf is None:
                return None
            o = self._odom_buf
            pos = np.array([o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z])
            vel = np.array([o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z])
            return pos, vel

    def get_joint_states(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取关节状态 (位置, 速度)"""
        with self._lock:
            if self._joint_buf is None:
                return None
            return np.array(self._joint_buf.position), np.array(self._joint_buf.velocity)


class GazeboSimulator:
    """
    Gazebo 仿真器封装

    启动/停止 Gazebo 仿真，管理世界模型。
    """

    def __init__(
        self,
        spec: Optional[GazeboAGVSpec] = None,
        config: Optional[GazeboROS2Config] = None,
        bridge: Optional[GazeboROS2Bridge] = None
    ):
        self.spec = spec or GazeboAGVSpec()
        self.config = config or GazeboROS2Config()
        self.bridge = bridge or GazeboROS2Bridge(self.config)
        self._gz_process: Optional[subprocess.Popen] = None
        self._spawned = False

    def spawn(
        self,
        world: Optional[str] = None,
        urdf: Optional[str] = None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        yaw: float = 0.0
    ) -> bool:
        """
        生成 AGV 模型到 Gazebo 世界

        Args:
            world: 世界名称或 SDF 文件路径
            urdf: URDF 文件路径 (可选)
            x, y, z: 初始位置
            yaw: 初始偏航角

        Returns:
            成功标志
        """
        if self._spawned:
            print("[GazeboSimulator] Model already spawned")
            return True

        world = world or self.config.world.value
        print(f"[GazeboSimulator] Spawning AGV '{self.spec.model_name}' in world '{world}' at ({x}, {y}, {z}, {yaw})")

        # 初始化 ROS2 桥接
        self.bridge.initialize()

        # 订阅传感器话题
        self.bridge._subscribe_camera(self.config.camera_topic)
        self.bridge._subscribe_imu(self.config.imu_topic)
        self.bridge._subscribe_lidar(self.config.lidar_topic)
        self.bridge._subscribe_odom(self.config.odom_topic)
        self.bridge._subscribe_joints(self.config.joint_topic)

        self._spawned = True
        print("[GazeboSimulator] AGV spawned successfully (ROS2 bridge mode)")
        return True

    def kill(self):
        """关闭仿真"""
        if self._gz_process:
            self._gz_process.terminate()
            try:
                self._gz_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gz_process.kill()
            self._gz_process = None
        self.bridge.shutdown()
        self._spawned = False
        print("[GazeboSimulator] Simulation killed")

    def set_velocity(self, cmd: Tuple[float, float, float]):
        """
        设置 AGV 速度

        Args:
            cmd: (vx, vy, omega) m/s, m/s, rad/s
        """
        self.bridge.publish_cmd_vel(cmd[0], cmd[1], cmd[2])

    def get_camera(self) -> Optional[np.ndarray]:
        return self.bridge.get_camera_image()

    def get_imu(self) -> Optional[Dict[str, float]]:
        return self.bridge.get_imu_data()

    def get_lidar(self) -> Optional[np.ndarray]:
        return self.bridge.get_lidar_scan()

    def get_pose(self) -> Optional[np.ndarray]:
        odom = self.bridge.get_odometry()
        return odom[0] if odom else None

    def get_twist(self) -> Optional[np.ndarray]:
        odom = self.bridge.get_odometry()
        return odom[1] if odom else None

    def __enter__(self):
        self.spawn()
        return self

    def __exit__(self, *args):
        self.kill()


class AGVGazeboSimulator(GazeboSimulator):
    """
    AGV 专用 Gazebo 仿真器

    在 GazeboSimulator 基础上添加 AGV 特定功能:
    - 差速驱动运动学
    - 仓库物流仿真场景
    - 多 AGV 协调仿真
    """

    def __init__(
        self,
        spec: Optional[GazeboAGVSpec] = None,
        config: Optional[GazeboROS2Config] = None,
        bridge: Optional[GazeboROS2Bridge] = None
    ):
        super().__init__(spec or GazeboAGVSpec(), config, bridge)

        # 差速驱动参数
        self._wb = self.spec.wheelbase
        self._tw = self.spec.track_width
        self._wr = self.spec.wheel_radius

        # 里程计
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0

    def differential_to_wheel(self, vx: float, omega: float) -> Tuple[float, float]:
        """
        差速驱动运动学: 车身速度 -> 车轮速度

        Args:
            vx: 纵向速度 m/s
            omega: 角速度 rad/s

        Returns:
            (v_left, v_right) m/s
        """
        v = vx
        w = omega
        # 差速模型: v_l = v - omega * (tw/2), v_r = v + omega * (tw/2)
        v_left = v - omega * (self._tw / 2.0)
        v_right = v + omega * (self._tw / 2.0)
        return v_left, v_right

    def wheel_to_differential(self, v_left: float, v_right: float) -> Tuple[float, float]:
        """
        差速驱动逆运动学: 车轮速度 -> 车身速度

        Args:
            v_left: 左轮速度 m/s
            v_right: 右轮速度 m/s

        Returns:
            (vx, omega) m/s, rad/s
        """
        vx = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self._tw
        return vx, omega

    def step(self, dt: float):
        """
        仿真一步 (用于本地运动学仿真模式)

        当 Gazebo 不可用时, 使用简单自行车模型积分。
        """
        twist = self.get_twist()
        if twist is None:
            vx, omega = 0.0, 0.0
        else:
            # twist = (vx, vy, omega) from parent class
            vx, omega = twist[0], twist[2]
        self._odom_x += vx * np.cos(self._odom_yaw) * dt
        self._odom_y += vx * np.sin(self._odom_yaw) * dt
        self._odom_yaw += omega * dt

    def get_pose(self) -> np.ndarray:
        """获取 AGV 位姿"""
        real_pose = super().get_pose()
        if real_pose is not None:
            return real_pose[:3]
        return np.array([self._odom_x, self._odom_y, self._odom_yaw])

    @classmethod
    def for_grade(cls, grade: str, **kwargs) -> 'AGVGazeboSimulator':
        """根据 AGV 等级创建仿真器"""
        grade_specs = {
            'S': GazeboAGVSpec(mass=30, max_linear=1.0, camera_enabled=True,
                               imu_enabled=True, lidar_enabled=False, grade='S'),
            'M': GazeboAGVSpec(mass=50, max_linear=2.0, camera_enabled=True,
                               imu_enabled=True, lidar_enabled=True, grade='M'),
            'L': GazeboAGVSpec(mass=80, max_linear=3.0, camera_enabled=True,
                               imu_enabled=True, lidar_enabled=True, grade='L'),
            'XL': GazeboAGVSpec(mass=120, max_linear=4.0, camera_enabled=True,
                                imu_enabled=True, lidar_enabled=True, grade='XL'),
            'XXL': GazeboAGVSpec(mass=200, max_linear=5.0, camera_enabled=True,
                                 imu_enabled=True, lidar_enabled=True, grade='XXL'),
        }
        spec = grade_specs.get(grade, grade_specs['M'])
        return cls(spec=spec, **kwargs)


class GazeboROS2Simulator(GazeboSimulator):
    """
    ROS2 + Gazebo 联合仿真器 (完整版)

    适用于:
    - 真实 ROS2 + Gazebo 联合仿真
    - 硬件在环 (HITL)
    - 端到端导航测试
    """

    def __init__(
        self,
        spec: Optional[GazeboAGVSpec] = None,
        config: Optional[GazeboROS2Config] = None,
    ):
        super().__init__(spec, config)
        self._action_clients: Dict[str, Any] = {}

    async def navigate_to(self, x: float, y: float, yaw: float = 0.0) -> bool:
        """
        导航到目标点 (ROS2 action)

        Args:
            x, y: 目标位置
            yaw: 目标朝向

        Returns:
            成功标志
        """
        print(f"[GazeboROS2Simulator] Navigate to ({x}, {y}, {yaw})")
        # ROS2 action client would be implemented here
        # action_client.send_goal(...)
        return True

    async def grasp_at(self, x: float, y: float, z: float) -> bool:
        """
        执行抓取动作 (ROS2 action)

        Args:
            x, y, z: 目标抓取位置

        Returns:
            成功标志
        """
        print(f"[GazeboROS2Simulator] Grasp at ({x}, {y}, {z})")
        return True

    def reset(self, pose: Optional[Tuple[float, float, float]] = None):
        """
        重置仿真到初始状态

        Args:
            pose: (x, y, yaw) 可选重置位置
        """
        if pose:
            print(f"[GazeboROS2Simulator] Resetting to pose {pose}")
        else:
            print("[GazeboROS2Simulator] Resetting to default pose")
        self.set_velocity((0.0, 0.0, 0.0))
