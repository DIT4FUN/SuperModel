"""
ROS2 控制接口模块
================

ROS2 Humble 集成接口
- JointTrajectory 控制器接口
- 话题订阅/发布
- 服务端/客户端
- 参数服务器
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import threading
import time


class ControlInterfaceMode(Enum):
    """控制接口模式"""
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"


@dataclass
class JointCommand:
    """关节命令"""
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None
    time_from_start: float = 0.0


@dataclass
class JointState:
    """关节状态反馈"""
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
    timestamp: float = 0.0


class ROS2JointTrajectoryInterface:
    """
    ROS2 JointTrajectory 控制器接口
    
    模拟 ROS2 rclpy 接口，用于轨迹执行
    支持:
    - 关节位置/速度/力矩轨迹
    - 轨迹跟踪监控
    - 超时/失败检测
    """
    
    def __init__(
        self,
        joint_names: List[str],
        interface_mode: ControlInterfaceMode = ControlInterfaceMode.POSITION
    ):
        """
        Args:
            joint_names: 关节名称列表
            interface_mode: 控制接口模式
        """
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.mode = interface_mode
        
        # 状态
        self._is_active = False
        self._current_traj_idx = 0
        self._trajectory: Optional[JointCommand] = None
        self._start_time = 0.0
        
        # 回调
        self._command_callback: Optional[Callable] = None
        self._goal_callback: Optional[Callable] = None
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 统计
        self._sent_commands = 0
        self._failed_commands = 0
    
    def activate(self):
        """激活接口"""
        self._is_active = True
        self._current_traj_idx = 0
        print(f"[ROS2JointTrajectory] Activated with {self.num_joints} joints, mode={self.mode.value}")
    
    def deactivate(self):
        """停用接口"""
        self._is_active = False
        print("[ROS2JointTrajectory] Deactivated")
    
    def set_command_callback(self, callback: Callable[[JointCommand], None]):
        """设置命令回调"""
        self._command_callback = callback
    
    def set_goal_callback(self, callback: Callable[[bool, str], None]):
        """设置目标完成回调 (success, message)"""
        self._goal_callback = callback
    
    def send_trajectory(self, trajectory: List[JointCommand]) -> bool:
        """
        发送关节轨迹
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            发送是否成功
        """
        if not self._is_active:
            print("[ROS2JointTrajectory] Cannot send trajectory: not active")
            return False
        
        with self._lock:
            self._trajectory = trajectory
            self._current_traj_idx = 0
            self._start_time = time.time()
        
        print(f"[ROS2JointTrajectory] Trajectory sent: {len(trajectory)} points")
        return True
    
    def send_point(self, point: JointCommand) -> bool:
        """
        发送单个轨迹点
        
        Args:
            point: 关节命令
            
        Returns:
            发送是否成功
        """
        if not self._is_active:
            return False
        
        if self._command_callback:
            try:
                self._command_callback(point)
                self._sent_commands += 1
                return True
            except Exception as e:
                self._failed_commands += 1
                print(f"[ROS2JointTrajectory] Command failed: {e}")
                return False
        
        return True
    
    def update(self, current_state: JointState) -> Optional[JointCommand]:
        """
        更新轨迹执行
        
        Args:
            current_state: 当前关节状态
            
        Returns:
            下一个命令 (如果有)
        """
        if not self._is_active or self._trajectory is None:
            return None
        
        with self._lock:
            if self._current_traj_idx >= len(self._trajectory):
                # 轨迹完成
                if self._goal_callback:
                    self._goal_callback(True, "Trajectory completed successfully")
                self._trajectory = None
                return None
            
            # 获取当前点
            cmd = self._trajectory[self._current_traj_idx]
            
            # 检查是否到达
            if self._check_point_reached(current_state, cmd):
                self._current_traj_idx += 1
                if self._current_traj_idx < len(self._trajectory):
                    return self._trajectory[self._current_traj_idx]
                else:
                    if self._goal_callback:
                        self._goal_callback(True, "All points reached")
                    return None
            
            return cmd
    
    def _check_point_reached(self, state: JointState, cmd: JointCommand) -> bool:
        """检查目标点是否到达"""
        tolerance = 0.01  # 1cm
        
        if cmd.positions is not None:
            diff = np.abs(state.positions - cmd.positions)
            return np.all(diff < tolerance)
        
        return True
    
    def cancel(self) -> bool:
        """
        取消当前轨迹
        
        Returns:
            取消是否成功
        """
        with self._lock:
            if self._trajectory is not None:
                self._trajectory = None
                self._current_traj_idx = 0
                if self._goal_callback:
                    self._goal_callback(False, "Trajectory cancelled")
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "sent_commands": self._sent_commands,
            "failed_commands": self._failed_commands,
            "success_rate": (
                self._sent_commands / (self._sent_commands + self._failed_commands)
                if self._sent_commands + self._failed_commands > 0 else 1.0
            ),
            "current_trajectory_length": (
                len(self._trajectory) if self._trajectory else 0
            ),
            "current_index": self._current_traj_idx if self._trajectory else 0
        }


class ROS2TopicInterface:
    """
    ROS2 话题接口
    
    提供话题订阅/发布功能
    """
    
    def __init__(self, node_name: str = "supermodel_control"):
        self.node_name = node_name
        self._subscribers: Dict[str, Callable] = {}
        self._publishers: Dict[str, Any] = {}
        self._topic_data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        print(f"[ROS2Topic] Node '{node_name}' created (mock)")
    
    def create_subscription(
        self,
        topic: str,
        msg_type: str,
        callback: Callable
    ):
        """创建订阅者"""
        self._subscribers[topic] = callback
        print(f"[ROS2Topic] Subscriber created: {topic} [{msg_type}]")
    
    def create_publisher(self, topic: str, msg_type: str):
        """创建发布者"""
        self._publishers[topic] = MockPublisher(topic, msg_type)
        print(f"[ROS2Topic] Publisher created: {topic} [{msg_type}]")
    
    def publish(self, topic: str, data: Any):
        """发布消息"""
        if topic in self._publishers:
            self._publishers[topic].publish(data)
            with self._lock:
                self._topic_data[topic] = data
    
    def subscribe(self, topic: str, data: Any):
        """接收订阅消息"""
        if topic in self._subscribers:
            self._subscribers[topic](data)
    
    def get_topic_data(self, topic: str) -> Optional[Any]:
        """获取话题最新数据"""
        with self._lock:
            return self._topic_data.get(topic)


class MockPublisher:
    """模拟 ROS2 Publisher"""
    
    def __init__(self, topic: str, msg_type: str):
        self.topic = topic
        self.msg_type = msg_type
        self._published_count = 0
    
    def publish(self, data: Any):
        self._published_count += 1
    
    def get_count(self) -> int:
        return self._published_count


class ROS2ServiceInterface:
    """
    ROS2 服务接口
    
    提供服务服务端/客户端功能
    """
    
    def __init__(self, node_name: str = "supermodel_service"):
        self.node_name = node_name
        self._services: Dict[str, Callable] = {}
        self._clients: Dict[str, Any] = {}
        self._call_results: Dict[str, Any] = {}
        
        print(f"[ROS2Service] Node '{node_name}' created (mock)")
    
    def create_service(
        self,
        service_name: str,
        srv_type: str,
        callback: Callable
    ):
        """创建服务端"""
        self._services[service_name] = callback
        print(f"[ROS2Service] Service created: {service_name} [{srv_type}]")
    
    def call_service(self, service_name: str, request: Any) -> Optional[Any]:
        """调用服务"""
        if service_name in self._services:
            try:
                result = self._services[service_name](request)
                self._call_results[service_name] = result
                return result
            except Exception as e:
                print(f"[ROS2Service] Service call failed: {e}")
                return None
        return None
    
    def create_client(self, service_name: str, srv_type: str):
        """创建客户端"""
        self._clients[service_name] = MockClient(service_name, srv_type)
        print(f"[ROS2Service] Client created: {service_name} [{srv_type}]")


class MockClient:
    """模拟 ROS2 服务客户端"""
    
    def __init__(self, service_name: str, srv_type: str):
        self.service_name = service_name
        self.srv_type = srv_type
    
    def call(self, request: Any) -> Any:
        """同步调用"""
        return {"success": True, "message": "mock response"}


# ROS2 标准话题名定义
class ROSTopics:
    """ROS2 标准话题名"""
    
    # 传感器话题
    CAMERA_LEFT = "/supermodel/camera/left"
    CAMERA_RIGHT = "/supermodel/camera/right"
    AUDIO = "/supermodel/audio"
    TACTILE = "/supermodel/tactile"
    IMU = "/supermodel/imu"
    FORCE = "/supermodel/force"
    
    # 控制话题
    JOINT_TRAJECTORY_CMD = "/supermodel/joint_trajectory/command"
    JOINT_TRAJECTORY_FB = "/supermodel/joint_trajectory/feedback"
    JOINT_STATES = "/supermodel/joint_states"
    GRIPPER_CMD = "/supermodel/gripper/command"
    TWIST_CMD = "/supermodel/twist/command"
    
    # 规划话题
    PLANNER_REQUEST = "/supermodel/planner/request"
    PLANNER_RESULT = "/supermodel/planner/result"
    
    # 感知话题
    PERCEPTION_RESULT = "/supermodel/perception/result"
    FUSION_OUTPUT = "/supermodel/fusion/output"


# ROS2 标准服务名
class ROSServices:
    """ROS2 标准服务名"""
    
    PERCEPTION = "/supermodel/perception"
    PLANNING = "/supermodel/planning"
    EXECUTE_SKILL = "/supermodel/execute_skill"
    GET_STATE = "/supermodel/get_state"
    SET_MODE = "/supermodel/set_mode"
    RESET = "/supermodel/reset"


# ROS2 参数名
class ROSParams:
    """ROS2 参数名"""
    
    # 控制参数
    CONTROL_RATE = "control.rate"
    MAX_VELOCITY = "control.max_velocity"
    MAX_ACCELERATION = "control.max_acceleration"
    
    # 感知参数
    CAMERA_EXPOSURE = "camera.exposure"
    IMU_SAMPLE_RATE = "imu.sample_rate"
    
    # 融合参数
    FUSION_STRATEGY = "fusion.strategy"
    HIDDEN_DIM = "fusion.hidden_dim"


# AGV五级ROS2规格
ROS2_AGV_GRADES = {
    'S': {
        'topics': 5, 'services': 3, 'max_freq_hz': 50,
        'realtime': False, 'qos_depth': 10
    },
    'M': {
        'topics': 10, 'services': 5, 'max_freq_hz': 100,
        'realtime': False, 'qos_depth': 10
    },
    'L': {
        'topics': 20, 'services': 10, 'max_freq_hz': 200,
        'realtime': True, 'qos_depth': 5
    },
    'XL': {
        'topics': 30, 'services': 15, 'max_freq_hz': 500,
        'realtime': True, 'qos_depth': 3
    },
    'XXL': {
        'topics': 50, 'services': 25, 'max_freq_hz': 1000,
        'realtime': True, 'qos_depth': 1
    },
}


def get_ros2_spec(grade: str) -> dict:
    """获取AGV指定等级的ROS2规格"""
    return ROS2_AGV_GRADES.get(grade, ROS2_AGV_GRADES['M'])
