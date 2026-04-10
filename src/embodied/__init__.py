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
)

from .simulation_enhancement import (
    PhysicsParameters,
    SensorNoiseModel,
    DelaySimulator,
    CollisionEnhancer,
    EnvironmentGenerator,
    WarehouseSceneGenerator,
    EmbodiedSimulationEnhancer,
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

__all__ = [
    # 枚举
    'NodeStatus',
    'TaskStatus',
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
]

__version__ = "2.0.0"
