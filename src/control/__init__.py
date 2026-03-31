"""
SuperModel 控制模块
==================

执行层模块:
- motion: 运动控制 (速度/位置/力矩控制)
- trajectory: 轨迹规划 (RRT, 多项式, S曲线)
- skill: 技能库调度
- planner: 任务规划器
- impedance: 阻抗控制
- mpc: 模型预测控制 (MPC)
- ros2_interface: ROS2 Humble 集成接口
- safety_controller: 安全监控与故障容忍
- agv: AGV运动学/动力学控制
- multi_agent: 多智能体协调控制 (L/XL/XXL)
- teleop: 遥操作控制 (主从同步/共享控制/力反馈)
"""

from .motion import MotionController, JointTrajectory, TwistCommand, AdaptivePIDController
from .trajectory import (
    TrajectoryGenerator, RRTPlanner, ScurveGenerator,
    JointWaypoint, CartesianWaypoint, TrajectoryConfig,
    PlanningAlgorithm, get_trajectory_spec
)
from .skill import SkillLibrary, Skill, SkillRegistry
from .planner import TaskPlanner, Task, TaskStatus
from .impedance import ImpedanceController
from .safety_controller import (
    SafetyController, SafetyConfig, SafetyLevel, SafetyEvent,
    SafetyResponse, SafetyEventRecord, JointStateSnapshot,
    SafetyCheckResult, get_safety_spec
)
from .ros2_interface import (
    ROS2JointTrajectoryInterface, ROS2TopicInterface, ROS2ServiceInterface,
    ROS2ActionInterface, ROS2ParameterInterface, ROS2ComponentInterface,
    JointCommand, JointState, ControlInterfaceMode,
    ActionGoalStatus, ActionFeedback, ActionResult,
    ROSTopics, ROSServices, ROSParams, get_ros2_spec
)
from .mpc import (
    MPCConfig, JointStateMP, DynamicsModel,
    JointSpaceMPC, CartesianMPC, get_mpc_spec
)
from .agv import (
    AGVMotionController, TrajectoryTracker, AGVSpec, AGVPose, AGVTwist,
    DriveType, AGVGrade,
    DifferentialKinematics, MecanumKinematics, get_agv_spec
)
from .multi_agent import (
    MultiAgentCoordinator, FormationType, CoordinationState,
    AgentState, FormationSlot, CoordinationTask, CollisionRisk,
    get_coordination_spec
)
from .teleop import (
    TeleoperationController, TeleopMode, TeleopState, AuthorityLevel,
    MasterState, SlaveState, TeleopCommand, TeleopConfig, TeleopMetrics,
    SafetyMonitor, LatencyCompensator, SharedControlBlender,
    AGV_TELEOP_GRADES, get_teleop_spec
)
from .tactile_control import (
    TactileServoController, TactileServoParams, GraspQualityController,
    AGV_TACTILE_CONTROL_GRADES, get_tactile_control_spec
)
from .force_control import (
    ForceController, ForceControlParams, HybridForcePositionController,
    AGV_FORCE_CONTROL_GRADES, get_force_control_spec
)
from .imu_control import (
    AttitudeStabilizer, IMUControlParams, MotionEstimator,
    AGV_IMU_CONTROL_GRADES, get_imu_control_spec
)

__all__ = [
    'MotionController', 'JointTrajectory', 'TwistCommand', 'AdaptivePIDController',
    'TrajectoryGenerator', 'RRTPlanner', 'ScurveGenerator',
    'JointWaypoint', 'CartesianWaypoint', 'TrajectoryConfig',
    'PlanningAlgorithm', 'get_trajectory_spec',
    'SkillLibrary', 'Skill', 'SkillRegistry',
    'TaskPlanner', 'Task', 'TaskStatus',
    'ImpedanceController',
    'MPCConfig', 'JointStateMP', 'DynamicsModel',
    'JointSpaceMPC', 'CartesianMPC', 'get_mpc_spec',
    'SafetyController', 'SafetyConfig', 'SafetyLevel', 'SafetyEvent',
    'SafetyResponse', 'SafetyEventRecord', 'JointStateSnapshot',
    'SafetyCheckResult', 'get_safety_spec',
    'ROS2JointTrajectoryInterface', 'ROS2TopicInterface', 'ROS2ServiceInterface',
    'ROS2ActionInterface', 'ROS2ParameterInterface', 'ROS2ComponentInterface',
    'JointCommand', 'JointState', 'ControlInterfaceMode',
    'ActionGoalStatus', 'ActionFeedback', 'ActionResult',
    'ROSTopics', 'ROSServices', 'ROSParams', 'get_ros2_spec',
    'AGVMotionController', 'TrajectoryTracker', 'AGVSpec', 'AGVPose', 'AGVTwist',
    'DriveType', 'AGVGrade',
    'DifferentialKinematics', 'MecanumKinematics', 'get_agv_spec',
    'MultiAgentCoordinator', 'FormationType', 'CoordinationState',
    'AgentState', 'FormationSlot', 'CoordinationTask', 'CollisionRisk',
    'get_coordination_spec',
    'TeleoperationController', 'TeleopMode', 'TeleopState', 'AuthorityLevel',
    'MasterState', 'SlaveState', 'TeleopCommand', 'TeleopConfig', 'TeleopMetrics',
    'SafetyMonitor', 'LatencyCompensator', 'SharedControlBlender',
    'AGV_TELEOP_GRADES', 'get_teleop_spec',
    'TactileServoController', 'TactileServoParams', 'GraspQualityController',
    'AGV_TACTILE_CONTROL_GRADES', 'get_tactile_control_spec',
    'ForceController', 'ForceControlParams', 'HybridForcePositionController',
    'AGV_FORCE_CONTROL_GRADES', 'get_force_control_spec',
    'AttitudeStabilizer', 'IMUControlParams', 'MotionEstimator',
    'AGV_IMU_CONTROL_GRADES', 'get_imu_control_spec',
]
