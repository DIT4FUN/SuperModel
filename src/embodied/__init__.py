"""
embodied - 具身智能模块
============================

行为树具身任务规划 + 层级任务分解 + AGV五级规格适配
仿真环境增强 + 真实AGV硬件接口适配
"""

from .behavior_tree import (
    NodeStatus,
    BTNode,
    SequenceNode,
    SelectorNode,
    ParallelNode,
    RepeaterNode,
    UntilFailNode,
    UntilSuccessNode,
    InverterNode,
    ConditionNode,
    ActionNode,
    BehaviorTree,
    EmbodiedTaskPlanner,
    AGVTaskPlanner,
    TaskStatus,
    Blackboard,
    EmbodiedTask,
    AGVCheckBatteryCondition,
    AGVCheckSafeCondition,
    AGVCheckPositionReached,
    AGVMoveToAction,
    AGVGraspAction,
    AGVReleaseAction,
    # 多AGV蜂群协同
    AGVNegotiateRoleAction,
    AGVMoveToFormationAction,
    AGVCheckFormationReachedCondition,
    AGVParallelGraspAction,
    AGVCoordinatedMoveToAction,
    AGVParallelReleaseAction,
    MultiAGVBehaviorTreePlanner,
    # 别名
    AGVNegotiateRole,
    AGVMoveToFormation,
    AGVCheckFormationReached,
    AGVParallelGrasp,
    AGVCoordinatedMoveTo,
    AGVParallelRelease,
)

from .simulation_enhancement import (
    PhysicsParameters,
    SensorNoiseModel,
    DelaySimulator,
    CollisionEnhancer,
    EnvironmentGenerator,
    WarehouseSceneGenerator,
    EmbodiedSimulationEnhancer,
    DynamicObstacleGenerator,
    MultiAGVSimulationEnhancer,
    WeatherType,
    WeatherEffect,
    Obstacle,
    WEATHER_EFFECTS,
)

from .real_agv_interface import (
    AGVHardwareConfig,
    HardwareInterface,
    CANBusDriver,
    ZLAC8015DController,
    LidarInterface,
    IMUInterface,
    TactileInterface,
    ForceSensorInterface,
    RealAGVController,
    ThreadedSensorReader,
)

from .scene_intelligence import (
    SceneType,
    SceneContext,
    SceneRule,
    SafetyRule,
    NavigationRule,
    InteractionRule,
    SceneFeatures,
    SceneIntelligence,
    SceneConfig,
    get_scene_intelligence,
)

from .scene_coordination import (
    AGVSceneRole,
    SceneCoordinationConfig,
    AGVSceneState,
    SceneCoordinator,
    MultiSceneSwarmController,
)

from .deployment import (
    DeploymentState,
    HealthStatus,
    DeploymentConfig,
    HealthCheckResult,
    DeploymentValidator,
    HealthMonitor,
    EmergencyProcedure,
    DeploymentManager,
    create_deployment_manager,
)

from .scene_task_planner import (
    SceneTaskConfig,
    SceneTaskTemplate,
    SceneTaskLibrary,
    SceneTaskPlanner,
    WarehouseTaskPlanner,
    HospitalTaskPlanner,
    FactoryTaskPlanner,
    RestaurantTaskPlanner,
    OutdoorTaskPlanner,
    SceneAdaptationEngine,
    get_scene_task_planner,
)

from .agv_swarm_coordinator import (
    TaskPriority,
    TaskStatus,
    SwarmTask,
    AGVSwarmMember,
    SwarmConflict,
    AGVSwarmCoordinator,
)

from .task_executor import (
    ExecutionPhase,
    ExecutionResult,
    TaskExecutionRecord,
    MemoryEnhancedExecutor,
    ScenarioTaskExecutor,
    create_task_executor,
    create_executor_from_config,
)

from .memory_integration import (
    EmbodiedMemoryEntry,
    EmbodiedSkill,
    EmbodiedMemoryManager,
    create_embodied_memory_manager,
    connect_to_long_term_memory,
)

from .collaborative_slam import (
    MapQuality,
    FeaturePoint,
    MapFragment,
    PoseConstraint,
    CollaborativeSlamAgent,
    MapFusionEngine,
    CollaborativeSlamCoordinator,
    get_collaborative_slam_coordinator,
)

from .hil_testing import (
    HILTestStage,
    HILTestCase,
    HILTestResult,
    HILTestReport,
    SensorReplay,
    CANBusHILSimulator,
    ControlCommandValidator,
    SensorActuatorHILLoop,
    HILTestRunner,
    run_hil_validation,
)

from .embodied_pipeline import (
    PipelineMode,
    PipelineState,
    PipelineConfig,
    TaskRequest,
    TaskResult,
    EmbodiedPipeline,
    create_embodied_pipeline,
    create_pipeline_from_config,
)

from .embodied_skill import (
    SkillStatus,
    SkillCategory,
    SkillMetrics,
    SkillVersion,
    EmbodiedSkillDefinition,
    EmbodiedSkill,
    EmbodiedSkillRegistry,
    get_global_skill_registry,
    create_skill_registry,
)

__all__ = [
    # 枚举
    'NodeStatus',
    'TaskStatus',
    'TaskPriority',
    # 节点类
    'BTNode',
    'SequenceNode',
    'SelectorNode',
    'ParallelNode',
    'RepeaterNode',
    'UntilFailNode',
    'UntilSuccessNode',
    'InverterNode',
    'ConditionNode',
    'ActionNode',
    # 行为树和规划器
    'BehaviorTree',
    'Blackboard',
    'EmbodiedTask',
    'EmbodiedTaskPlanner',
    'AGVTaskPlanner',
    # AGV专用节点
    'AGVCheckBatteryCondition',
    'AGVCheckSafeCondition',
    'AGVCheckPositionReached',
    'AGVMoveToAction',
    'AGVGraspAction',
    'AGVReleaseAction',
    # 多AGV蜂群协同节点
    'AGVNegotiateRoleAction',
    'AGVMoveToFormationAction',
    'AGVCheckFormationReachedCondition',
    'AGVParallelGraspAction',
    'AGVCoordinatedMoveToAction',
    'AGVParallelReleaseAction',
    'MultiAGVBehaviorTreePlanner',
    # 蜂群节点别名
    'AGVNegotiateRole',
    'AGVMoveToFormation',
    'AGVCheckFormationReached',
    'AGVParallelGrasp',
    'AGVCoordinatedMoveTo',
    'AGVParallelRelease',
    # 仿真增强
    'PhysicsParameters',
    'SensorNoiseModel',
    'DelaySimulator',
    'CollisionEnhancer',
    'EnvironmentGenerator',
    'WarehouseSceneGenerator',
    'EmbodiedSimulationEnhancer',
    # 真实硬件接口
    'AGVHardwareConfig',
    'HardwareInterface',
    'CANBusDriver',
    'ZLAC8015DController',
    'LidarInterface',
    'IMUInterface',
    'TactileInterface',
    'ForceSensorInterface',
    'RealAGVController',
    'ThreadedSensorReader',
    # 场景智能
    'SceneType',
    'SceneContext',
    'SceneRule',
    'SafetyRule',
    'NavigationRule',
    'InteractionRule',
    'SceneFeatures',
    'SceneIntelligence',
    'SceneConfig',
    'get_scene_intelligence',
    # 场景协同
    'AGVSceneRole',
    'SceneCoordinationConfig',
    'AGVSceneState',
    'SceneCoordinator',
    'MultiSceneSwarmController',
    # 部署管理
    'DeploymentState',
    'HealthStatus',
    'DeploymentConfig',
    'HealthCheckResult',
    'DeploymentValidator',
    'HealthMonitor',
    'EmergencyProcedure',
    'DeploymentManager',
    'create_deployment_manager',
    # 场景任务规划
    'SceneTaskConfig',
    'SceneTaskTemplate',
    'SceneTaskLibrary',
    'SceneTaskPlanner',
    'WarehouseTaskPlanner',
    'HospitalTaskPlanner',
    'FactoryTaskPlanner',
    'RestaurantTaskPlanner',
    'OutdoorTaskPlanner',
    'SceneAdaptationEngine',
    'get_scene_task_planner',
    # 多AGV蜂群协调器
    'SwarmTask',
    'AGVSwarmMember',
    'SwarmConflict',
    'AGVSwarmCoordinator',
    # 协同SLAM
    'MapFragment',
    'FeaturePoint',
    'PoseConstraint',
    'CollaborativeSlamAgent',
    'MapFusionEngine',
    'CollaborativeSlamCoordinator',
    'get_collaborative_slam_coordinator',
    # HIL硬件在环测试
    'HILTestStage',
    'HILTestCase',
    'HILTestResult',
    'HILTestReport',
    'SensorReplay',
    'CANBusHILSimulator',
    'ControlCommandValidator',
    'SensorActuatorHILLoop',
    'HILTestRunner',
    'run_hil_validation',
    # 任务执行器
    'ExecutionPhase',
    'ExecutionResult',
    'TaskExecutionRecord',
    'MemoryEnhancedExecutor',
    'ScenarioTaskExecutor',
    'create_task_executor',
    'create_executor_from_config',
    # 具身记忆集成
    'EmbodiedMemoryEntry',
    'EmbodiedMemoryManager',
    'create_embodied_memory_manager',
    'connect_to_long_term_memory',
    # 具身技能管理
    'SkillStatus',
    'SkillCategory',
    'SkillMetrics',
    'SkillVersion',
    'EmbodiedSkillDefinition',
    'EmbodiedSkill',
    'EmbodiedSkillRegistry',
    'get_global_skill_registry',
    'create_skill_registry',
    # 具身智能统一Pipeline
    'PipelineMode',
    'PipelineState',
    'PipelineConfig',
    'TaskRequest',
    'TaskResult',
    'EmbodiedPipeline',
    'create_embodied_pipeline',
    'create_pipeline_from_config',
]


# 医疗场景化具身智能
from .healthcare_scene import (
    HealthcareZone,
    HealthcareRiskLevel,
    PatientCallPriority,
    MedicationType,
    SpecimenCategory,
    HealthcareTask,
    HealthcareTaskLibrary,
    HealthcareSceneController,
    InfectionControlMonitor,
    PatientCallHandler,
    MedicationDeliveryPlanner,
    SpecimenTransportManager,
    get_healthcare_scene_controller,
)

# 工业制造场景化具身智能
from .industrial_scene import (
    ProductionLineType,
    WorkstationType,
    MaterialType,
    QualityGrade,
    ToolType,
    ProductionTask,
    ProductionLineController,
    QualityInspectionStation,
    PredictiveMaintenanceMonitor,
    ToolManagementSystem,
    SafetyMonitoringSystem,
    MaterialFlowCoordinator,
    get_industrial_scene_controller,
)

# 联邦学习多AGV协同
from .federated_learning import (
    FLClientState,
    FLRoundResult,
    LocalTrainingResult,
    FederatedClient,
    FederatedServer,
    DifferentialPrivacy,
    ByzantineFilter,
    AdaptiveAggregator,
    FederatedLearningCoordinator,
    create_federated_learning_system,
)

__version__ = "3.9.1"
