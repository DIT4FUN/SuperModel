"""
SuperModel 控制模块
==================

执行层模块:
- motor: 电机控制 (DC/BLDC/伺服/步进 + PID)
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
- supervisor: 控制子系统监管 (生命周期/模式切换/故障恢复)
- sensor_fusion_control: 传感器融合控制 (统一感知→控制闭环)
- skill_dispatcher: 跨模态技能协调执行器 (多技能并发调度/资源仲裁)
"""

from .motion import MotionController, JointTrajectory, TwistCommand, AdaptivePIDController
from .trajectory import (
    TrajectoryGenerator, RRTPlanner, ScurveGenerator,
    JointWaypoint, CartesianWaypoint, TrajectoryConfig,
    PlanningAlgorithm, get_trajectory_spec,
    VelocityProfiler, VelocityProfile,
)
from .trajectory_planning import (
    TrajectoryPlanner, TrajectoryPoint, Trajectory,
    Waypoint, PurePursuitTracker, StanleyTracker,
    PIDTrajectoryTracker, RRTStarPlanner, MinimumSnapTrajectory,
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
    DifferentialKinematics, MecanumKinematics, get_agv_spec,
    PurePursuitTracker, StanleyTracker, PIDTrajectoryTracker
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
from .supervisor import (
    ControlSupervisor, ControllerInterface, SupervisorConfig,
    ControlState, ControllerMetrics, ControlMode, HealthStatus,
    MockJointController, MockCartesianController, MockImpedanceController,
    GradeAwareSupervisor, SupervisorGrade, SupervisorGradeSpec,
    get_supervisor_spec, get_supervisor_config
)
from .autotune import (
    AutoTuner, TunerConfig, TunerResult, TuningMethod,
    SimulatedPlant, autotune_pid
)
from .motor import (
    Motor, MotorState, MotorControlMode,
    DCMotor, BLDCmotor, ServoMotor, StepperMotor,
    PIDController as MotorPIDController, MotorController
)
from .obstacle_avoidance import (
    ObstacleAvoider, DynamicWindowApproach, ArtificialPotentialField,
    VectorFieldHistogram, AvoidanceConfig, DWAConfig, APFConfig, VFHConfig,
    Obstacle, VelocityCommand, TrajectorySample,
    AvoidanceStrategy, get_obstacle_avoidance_spec
)
from .sensorimotor import (
    SensorimotorIntegration, SensorimotorConfig, SensorimotorState,
    SensorimotorSimulator,
    AGV_SENSORIMOTOR_GRADES, get_sensorimotor_spec
)
from .trajectory_planning import (
    TrajectoryPlanner, TrajectoryPoint, Trajectory,
    Waypoint, PurePursuitTracker, StanleyTracker,
    PIDTrajectoryTracker, RRTStarPlanner, MinimumSnapTrajectory,
)
from .embodied_control import (
    EmbodiedController, EmbodiedState, EmbodiedCommand,
    EmbodiedControlParams, EmbodiedTaskExecutor,
    EmbodiedGrade, AGV_EMBODIED_GRADES, get_embodied_spec,
    SensorHealthMonitor,
)
from .navigation import (
    NavigationController, OccupancyGrid, Path, Waypoint,
    PlannerType, NavigationState,
    DijkstraPlanner, AStarPlanner,
    create_navigation_grid,
)
from .patrol_control import (
    PatrolController, PatrolRoute, PatrolPoint, PatrolState,
    Obstacle, PatrolGrade, PatrolSpec,
    create_patrol_controller, run_patrol_benchmark, get_patrol_spec,
    PatrolMetrics, PatrolEvent,
)
from .adaptive_gain import (
    AdaptiveGainScheduler, GainSchedule, AdaptationState,
    GainBlendController, ModelReferenceAdaptiveController,
    AdaptationStrategy, get_adaptive_gain_spec, AGV_ADAPTIVE_GAIN_GRADES
)
from .sensor_fusion_control import (
    SensorFusionController, SensorFusionControlState, FusionControlConfig,
    FusionControlGrade, get_fusion_control_spec,
    AGV_FUSION_CONTROL_GRADES,
)
from .skill_dispatcher import (
    SkillDispatcher, SkillRequest, SkillResult, SkillStatus,
    SkillPriority, SkillDefinition, ResourceType,
    AGV_SKILL_DISPATCHER_GRADES, get_skill_dispatcher_spec,
    create_skill_dispatcher,
    create_grasp_skill, create_navigate_skill, create_place_skill,
)
from .behavior_tree import (
    BehaviorTree, BTNode, BTContext, NodeState,
    Selector, Sequence, Parallel,
    Condition, Action, SubTree,
    Inverter, RepeatUntil, RetryUntil, Timeout, RateLimiter,
    BTGrade, AGV_BT_GRADES,
    create_for_grade, create_safe_selector, create_action_sequence,
)
from .bias_compensation import (
    IMUBiasEstimator, ForceBiasEstimator, TactileBiasEstimator,
    MultiSensorBiasCompensator, IMUBiasState, ForceBiasState, TactileBiasState,
    BiasCompensationConfig, AGV_BIAS_COMPENSATION_GRADES,
    get_bias_compensation_spec, get_agv_bias_spec_table,
)
from .simulation import (
    SimulationInterface, SimulationBackend, SimulationGrade,
    SimulationConfig, SimState, AGVSimParams,
    AGV_SIM_PARAMS, AGV_SIMULATION_GRADES,
    get_agv_sim_params, get_simulation_spec,
)

from .embodied_sim import (
    EmbodiedSimulator, EmbodiedSimEnv, SimEnvironmentState,
    SensorNoiseModel, PhysicsSimulator, TactileSimulator,
    EmbodiedSimGrade, SimBackend,
    AGV_SIM_GRADES, get_sim_grade_spec, create_sim_env, get_grade_summary,
)

__all__ = [
    'MotionController', 'JointTrajectory', 'TwistCommand', 'AdaptivePIDController',
    'TrajectoryGenerator', 'RRTPlanner', 'ScurveGenerator',
    'JointWaypoint', 'CartesianWaypoint', 'TrajectoryConfig',
    'PlanningAlgorithm', 'get_trajectory_spec',
    'VelocityProfiler', 'VelocityProfile',
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
    'PurePursuitTracker', 'StanleyTracker', 'PIDTrajectoryTracker',
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
    'ControlSupervisor', 'ControllerInterface', 'SupervisorConfig',
    'ControlState', 'ControllerMetrics', 'ControlMode', 'HealthStatus',
    'MockJointController', 'MockCartesianController', 'MockImpedanceController',
    'SupervisorGrade', 'SupervisorGradeSpec', 'get_supervisor_spec',
    'get_supervisor_config', 'GradeAwareSupervisor',
    # autotune
    'AutoTuner', 'TunerConfig', 'TunerResult', 'TuningMethod',
    'SimulatedPlant', 'autotune_pid',
    # motor
    'Motor', 'MotorState', 'MotorControlMode',
    'DCMotor', 'BLDCmotor', 'ServoMotor', 'StepperMotor',
    'MotorPIDController', 'MotorController',
    # sensorimotor
    'SensorimotorIntegration', 'SensorimotorConfig', 'SensorimotorState',
    'SensorimotorSimulator',
    'AGV_SENSORIMOTOR_GRADES', 'get_sensorimotor_spec',
    # obstacle_avoidance
    'ObstacleAvoider', 'DynamicWindowApproach', 'ArtificialPotentialField',
    'VectorFieldHistogram', 'AvoidanceConfig', 'DWAConfig', 'APFConfig', 'VFHConfig',
    'Obstacle', 'VelocityCommand', 'TrajectorySample',
    'AvoidanceStrategy', 'get_obstacle_avoidance_spec',
    # embodied_control
    'EmbodiedController', 'EmbodiedState', 'EmbodiedCommand',
    'EmbodiedControlParams', 'EmbodiedTaskExecutor',
    'EmbodiedGrade', 'AGV_EMBODIED_GRADES', 'get_embodied_spec',
    'SensorHealthMonitor',
    # navigation
    'NavigationController', 'OccupancyGrid', 'Path', 'Waypoint',
    'PlannerType', 'NavigationState',
    'DijkstraPlanner', 'AStarPlanner',
    'create_navigation_grid',
    # patrol_control
    'PatrolController', 'PatrolRoute', 'PatrolPoint', 'PatrolState',
    'Obstacle', 'PatrolGrade', 'PatrolSpec',
    'create_patrol_controller', 'run_patrol_benchmark', 'get_patrol_spec',
    'PatrolMetrics', 'PatrolEvent',
    # adaptive_gain
    'AdaptiveGainScheduler', 'GainSchedule', 'AdaptationState',
    'GainBlendController', 'ModelReferenceAdaptiveController',
    'AdaptationStrategy', 'get_adaptive_gain_spec', 'AGV_ADAPTIVE_GAIN_GRADES',
    # sensor_fusion_control
    'SensorFusionController', 'SensorFusionControlState', 'FusionControlConfig',
    'FusionControlGrade', 'get_fusion_control_spec', 'AGV_FUSION_CONTROL_GRADES',
    # skill_dispatcher
    'SkillDispatcher', 'SkillRequest', 'SkillResult', 'SkillStatus',
    'SkillPriority', 'SkillDefinition', 'ResourceType',
    'AGV_SKILL_DISPATCHER_GRADES', 'get_skill_dispatcher_spec',
    'create_skill_dispatcher',
    'create_grasp_skill', 'create_navigate_skill', 'create_place_skill',
    # behavior_tree
    'BehaviorTree', 'BTNode', 'BTContext', 'NodeState',
    'Selector', 'Sequence', 'Parallel',
    'Condition', 'Action', 'SubTree',
    'Inverter', 'RepeatUntil', 'RetryUntil', 'Timeout', 'RateLimiter',
    'BTGrade', 'AGV_BT_GRADES',
    'create_for_grade', 'create_safe_selector', 'create_action_sequence',
    # bias_compensation
    'IMUBiasEstimator', 'ForceBiasEstimator', 'TactileBiasEstimator',
    'MultiSensorBiasCompensator', 'IMUBiasState', 'ForceBiasState', 'TactileBiasState',
    'BiasCompensationConfig', 'AGV_BIAS_COMPENSATION_GRADES',
    'get_bias_compensation_spec', 'get_agv_bias_spec_table',
    # simulation
    'SimulationInterface', 'SimulationBackend', 'SimulationGrade',
    'SimulationConfig', 'SimState', 'AGVSimParams',
    'AGV_SIM_PARAMS', 'AGV_SIMULATION_GRADES',
    'get_agv_sim_params', 'get_simulation_spec',
    # embodied_sim
    'EmbodiedSimulator', 'EmbodiedSimEnv', 'SimEnvironmentState',
    'SensorNoiseModel', 'PhysicsSimulator', 'TactileSimulator',
    'EmbodiedSimGrade', 'SimBackend',
    'AGV_SIM_GRADES', 'get_sim_grade_spec', 'create_sim_env', 'get_grade_summary',
]
