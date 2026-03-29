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


class ActionGoalStatus(Enum):
    """Action 目标状态"""
    UNKNOWN = 0
    ACCEPTED = 1
    EXECUTING = 2
    CANCELLED = 3
    SUCCEEDED = 4
    ABORTED = 5


@dataclass
class ActionFeedback:
    """Action 反馈"""
    sequence: int
    percent_complete: float
    current_joint_positions: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    message: str = ""


@dataclass
class ActionResult:
    """Action 结果"""
    success: bool
    message: str
    final_positions: Optional[np.ndarray] = None
    execution_time: float = 0.0
    trajectory_length: int = 0


class ROS2ActionInterface:
    """
    ROS2 Action 接口
    
    支持 long-running 任务:
    - JointTrajectory Action (FollowJointTrajectory)
    - 任务执行 Action
    - 自定义 Action
    
    提供:
    - Goal 管理 (发送/取消/状态查询)
    - Feedback 回调
    - Result 处理
    """

    GOAL_TIMEOUT_SEC = 300.0  # 默认目标超时

    def __init__(self, action_name: str = "joint_trajectory_action"):
        """
        Args:
            action_name: Action 名称
        """
        self.action_name = action_name
        self._is_server_active = False
        self._active_goal_handle: Optional[str] = None
        self._goal_handle_counter = 0

        # Goal 存储
        self._goals: Dict[str, Dict] = {}
        self._current_trajectory: Optional[List[JointCommand]] = None
        self._current_goal_idx = 0
        self._goal_start_time = 0.0

        # 回调
        self._feedback_callback: Optional[Callable[[ActionFeedback], None]] = None
        self._result_callback: Optional[Callable[[ActionResult], None]] = None
        self._goal_callback: Optional[Callable[[List[JointCommand]], Optional[ActionResult]]] = None

        # 线程安全
        self._lock = threading.Lock()

        # 统计
        self._total_goals = 0
        self._succeeded_goals = 0
        self._cancelled_goals = 0
        self._aborted_goals = 0

        print(f"[ROS2Action] Interface created: {action_name}")

    def set_feedback_callback(self, callback: Callable[[ActionFeedback], None]):
        """设置反馈回调"""
        self._feedback_callback = callback

    def set_result_callback(self, callback: Callable[[ActionResult], None]):
        """设置结果回调"""
        self._result_callback = callback

    def set_goal_callback(
        self, callback: Callable[[List[JointCommand]], Optional[ActionResult]]
    ):
        """设置目标处理回调"""
        self._goal_callback = callback

    # --- Server Side ---

    def start_server(self):
        """启动 Action Server"""
        self._is_server_active = True
        print(f"[ROS2Action] Server started: {self.action_name}")

    def stop_server(self):
        """停止 Action Server"""
        self._is_server_active = False
        # 取消所有活跃目标
        with self._lock:
            for goal_id, goal in self._goals.items():
                if goal["status"] in (ActionGoalStatus.ACCEPTED, ActionGoalStatus.EXECUTING):
                    self._cancel_goal_internal(goal_id, "Server stopped")
        print(f"[ROS2Action] Server stopped: {self.action_name}")

    def send_goal(
        self,
        trajectory: List[JointCommand],
        goal_id: Optional[str] = None
    ) -> str:
        """
        接收目标 (Server 端)

        Args:
            trajectory: 要执行的轨迹
            goal_id: 目标 ID (可选，自动生成)

        Returns:
            goal_id: 分配的目标 ID
        """
        if not self._is_server_active:
            raise RuntimeError("Action server not active")

        if goal_id is None:
            self._goal_handle_counter += 1
            goal_id = f"goal_{self._goal_handle_counter}"

        with self._lock:
            self._goals[goal_id] = {
                "status": ActionGoalStatus.ACCEPTED,
                "trajectory": trajectory,
                "current_idx": 0,
                "start_time": time.time(),
                "result": None,
            }
            self._total_goals += 1
            self._active_goal_handle = goal_id

        print(f"[ROS2Action] Goal accepted: {goal_id}, {len(trajectory)} points")
        return goal_id

    def update_server(self, current_state: JointState) -> bool:
        """
        更新 Action Server 状态

        Args:
            current_state: 当前关节状态

        Returns:
            是否还有活跃目标
        """
        if not self._is_server_active or self._active_goal_handle is None:
            return False

        with self._lock:
            goal = self._goals.get(self._active_goal_handle)
            if goal is None:
                return False

            trajectory = goal["trajectory"]
            idx = goal["current_idx"]

            if idx >= len(trajectory):
                # 轨迹完成
                result = ActionResult(
                    success=True,
                    message="Trajectory completed successfully",
                    final_positions=current_state.positions.copy(),
                    execution_time=time.time() - goal["start_time"],
                    trajectory_length=len(trajectory),
                )
                goal["status"] = ActionGoalStatus.SUCCEEDED
                goal["result"] = result
                self._succeeded_goals += 1
                self._active_goal_handle = None

                if self._result_callback:
                    self._result_callback(result)
                return False

            # 检查当前点是否到达
            cmd = trajectory[idx]
            tolerance = 0.01
            if cmd.positions is not None:
                diff = np.abs(current_state.positions - cmd.positions)
                if np.all(diff < tolerance):
                    goal["current_idx"] = idx + 1

                    # 发送反馈
                    if self._feedback_callback:
                        feedback = ActionFeedback(
                            sequence=idx,
                            percent_complete=goal["current_idx"] / len(trajectory),
                            current_joint_positions=current_state.positions.copy(),
                            error=diff if idx > 0 else None,
                            message=f"Point {idx+1}/{len(trajectory)} reached",
                        )
                        self._feedback_callback(feedback)

            goal["status"] = ActionGoalStatus.EXECUTING
            return True

    def cancel_goal(self, goal_id: str) -> bool:
        """取消目标"""
        with self._lock:
            return self._cancel_goal_internal(goal_id, "User requested cancellation")

    def _cancel_goal_internal(self, goal_id: str, reason: str) -> bool:
        """内部取消目标"""
        goal = self._goals.get(goal_id)
        if goal is None:
            return False

        if goal["status"] in (ActionGoalStatus.SUCCEEDED, ActionGoalStatus.CANCELLED, ActionGoalStatus.ABORTED):
            return False

        goal["status"] = ActionGoalStatus.CANCELLED
        self._cancelled_goals += 1

        if self._active_goal_handle == goal_id:
            self._active_goal_handle = None

        print(f"[ROS2Action] Goal cancelled: {goal_id}, reason={reason}")
        return True

    def get_goal_status(self, goal_id: str) -> Optional[ActionGoalStatus]:
        """获取目标状态"""
        with self._lock:
            goal = self._goals.get(goal_id)
            return goal["status"] if goal else None

    # --- Client Side ---

    def send_goal_async(
        self,
        trajectory: List[JointCommand],
        timeout_sec: float = GOAL_TIMEOUT_SEC
    ) -> str:
        """
        异步发送目标 (Client 端)

        Args:
            trajectory: 要执行的轨迹
            timeout_sec: 超时时间

        Returns:
            goal_id: 发送的目标 ID
        """
        goal_id = f"client_goal_{int(time.time()*1000)}"

        with self._lock:
            self._goals[goal_id] = {
                "status": ActionGoalStatus.ACCEPTED,
                "trajectory": trajectory,
                "current_idx": 0,
                "start_time": time.time(),
                "timeout": timeout_sec,
                "result": None,
            }
            self._total_goals += 1
            self._active_goal_handle = goal_id

        print(f"[ROS2Action] Async goal sent: {goal_id}")
        return goal_id

    def wait_for_result(self, goal_id: str, timeout_sec: Optional[float] = None) -> Optional[ActionResult]:
        """
        等待目标结果 (轮询模拟)

        Args:
            goal_id: 目标 ID
            timeout_sec: 超时时间

        Returns:
            ActionResult if completed, None if timeout
        """
        start = time.time()
        timeout = timeout_sec or self.GOAL_TIMEOUT_SEC

        while True:
            with self._lock:
                goal = self._goals.get(goal_id)
                if goal is None:
                    return None

                status = goal["status"]
                if status == ActionGoalStatus.SUCCEEDED:
                    return goal["result"]
                elif status in (ActionGoalStatus.CANCELLED, ActionGoalStatus.ABORTED):
                    return ActionResult(
                        success=False,
                        message=f"Goal {status.name.lower()}",
                    )

            if time.time() - start > timeout:
                print(f"[ROS2Action] Goal wait timeout: {goal_id}")
                return None

            time.sleep(0.01)

    def cancel_all_goals(self) -> int:
        """取消所有活跃目标"""
        count = 0
        with self._lock:
            for goal_id, goal in self._goals.items():
                if goal["status"] in (ActionGoalStatus.ACCEPTED, ActionGoalStatus.EXECUTING):
                    goal["status"] = ActionGoalStatus.CANCELLED
                    self._cancelled_goals += 1
                    count += 1
                    if self._active_goal_handle == goal_id:
                        self._active_goal_handle = None
        print(f"[ROS2Action] Cancelled {count} goals")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            active = sum(
                1 for g in self._goals.values()
                if g["status"] in (ActionGoalStatus.ACCEPTED, ActionGoalStatus.EXECUTING)
            )
            return {
                "total_goals": self._total_goals,
                "succeeded": self._succeeded_goals,
                "cancelled": self._cancelled_goals,
                "aborted": self._aborted_goals,
                "active": active,
                "success_rate": (
                    self._succeeded_goals / self._total_goals
                    if self._total_goals > 0 else 0.0
                ),
            }


class ROS2ParameterInterface:
    """
    ROS2 参数服务器接口

    提供参数 get/set/列表/订阅功能
    模拟 rclpy 参数服务
    """

    def __init__(self, node_name: str = "supermodel_params"):
        self.node_name = node_name
        self._parameters: Dict[str, Any] = {}
        self._param_subscriptions: Dict[str, Callable] = {}
        self._param_types: Dict[str, type] = {}

        # 预定义默认参数
        self._set_default_parameters()

        print(f"[ROS2Param] Parameter interface created for '{node_name}'")

    def _set_default_parameters(self):
        """设置默认参数"""
        defaults = {
            # 控制参数
            ROSParams.CONTROL_RATE: 100.0,
            ROSParams.MAX_VELOCITY: 1.0,
            ROSParams.MAX_ACCELERATION: 5.0,
            # 感知参数
            ROSParams.CAMERA_EXPOSURE: 0.033,
            ROSParams.IMU_SAMPLE_RATE: 200.0,
            # 融合参数
            ROSParams.FUSION_STRATEGY: "late",
            ROSParams.HIDDEN_DIM: 256,
            # AGV 参数
            "agv.max_linear_velocity": 1.5,
            "agv.max_angular_velocity": 1.0,
            "agv.wheel_base": 0.5,
            "agv.track_width": 0.4,
            "agv.wheel_radius": 0.1,
        }
        for name, value in defaults.items():
            self.set_parameter(name, value)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """获取参数"""
        return self._parameters.get(name, default)

    def set_parameter(self, name: str, value: Any) -> bool:
        """
        设置参数

        Args:
            name: 参数名
            value: 参数值

        Returns:
            设置是否成功
        """
        old_value = self._parameters.get(name)
        self._parameters[name] = value
        self._param_types[name] = type(value)

        # 触发订阅回调
        if name in self._param_subscriptions and old_value != value:
            try:
                self._param_subscriptions[name](value)
            except Exception as e:
                print(f"[ROS2Param] Subscription callback error for {name}: {e}")

        return True

    def get_parameters(self, names: List[str]) -> Dict[str, Any]:
        """批量获取参数"""
        return {name: self.get_parameter(name) for name in names}

    def list_parameters(self, prefix: str = "") -> List[str]:
        """列出参数名"""
        if prefix:
            return [k for k in self._parameters.keys() if k.startswith(prefix)]
        return list(self._parameters.keys())

    def declare_parameter(self, name: str, value: Any, descriptor: Optional[Dict] = None):
        """
        声明参数 (带元数据)

        Args:
            name: 参数名
            value: 默认值
            descriptor: 参数描述符 (read_only, default_value, etc.)
        """
        self._parameters[name] = value
        self._param_types[name] = type(value)

        if descriptor:
            print(f"[ROS2Param] Declared '{name}' = {value} ({descriptor.get('description', '')})")
        else:
            print(f"[ROS2Param] Declared '{name}' = {value}")

    def subscribe_parameter_change(self, name: str, callback: Callable[[Any], None]):
        """订阅参数变化"""
        self._param_subscriptions[name] = callback

    def load_from_dict(self, params: Dict[str, Any]):
        """从字典加载参数"""
        for name, value in params.items():
            self.set_parameter(name, value)
        print(f"[ROS2Param] Loaded {len(params)} parameters")

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return self._parameters.copy()


class ROS2ComponentInterface:
    """
    ROS2 组件接口

    封装常用 ROS2 功能组件:
    - Lifecycle 管理
    - Component 加载
    - Node 组合
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self._state = "unconfigured"  # unconfigured -> inactive -> active -> shutdown
        self._callbacks: Dict[str, Callable] = {}

        print(f"[ROS2Component] '{component_name}' created")

    def on_configure(self, callback: Callable[[], bool]):
        self._callbacks["configure"] = callback

    def on_activate(self, callback: Callable[[], bool]):
        self._callbacks["activate"] = callback

    def on_deactivate(self, callback: Callable[[], bool]):
        self._callbacks["deactivate"] = callback

    def on_cleanup(self, callback: Callable[[], bool]):
        self._callbacks["cleanup"] = callback

    def on_shutdown(self, callback: Callable[[], bool]):
        self._callbacks["shutdown"] = callback

    def configure(self) -> bool:
        if "configure" in self._callbacks:
            success = self._callbacks["configure"]()
            if success:
                self._state = "inactive"
            return success
        self._state = "inactive"
        return True

    def activate(self) -> bool:
        if self._state != "inactive":
            print(f"[ROS2Component] Cannot activate from state: {self._state}")
            return False
        if "activate" in self._callbacks:
            success = self._callbacks["activate"]()
            if success:
                self._state = "active"
            return success
        self._state = "active"
        return True

    def deactivate(self) -> bool:
        if self._state != "active":
            return False
        if "deactivate" in self._callbacks:
            success = self._callbacks["deactivate"]()
            if success:
                self._state = "inactive"
            return success
        self._state = "inactive"
        return True

    def cleanup(self) -> bool:
        if "cleanup" in self._callbacks:
            success = self._callbacks["cleanup"]()
            if success:
                self._state = "unconfigured"
            return success
        self._state = "unconfigured"
        return True

    def shutdown(self) -> bool:
        if "shutdown" in self._callbacks:
            success = self._callbacks["shutdown"]()
            self._state = "shutdown"
            return success
        self._state = "shutdown"
        return True

    def get_state(self) -> str:
        return self._state

    def __enter__(self):
        self.configure()
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()
        self.cleanup()
