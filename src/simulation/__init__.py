"""
SuperModel 仿真环境模块
======================

提供机器人仿真环境:
- RobotSimulator: 基础机器人仿真
- SensorSimulator: 传感器数据仿真
- PhysicsEngine: 物理引擎封装
- SceneManager: 场景管理
- TrajectoryRecorder: 轨迹记录与回放
- SuperModelGymEnv: Gymnasium RL 环境

支持:
- Mujoco 仿真引擎
- PyBullet 仿真引擎
- 自定义仿真 (无外部依赖)
- Gymnasium RL 训练接口
"""

from .environment import (
    RobotSimulator, SensorSimulator, SimConfig,
    PhysicsEngine, SceneManager, TrajectoryRecorder,
    create_scene, PRESET_SCENES
)
from .gym_env import (
    SuperModelGymEnv, GymEnvConfig,
    make_env, collect_rollout, get_gym_spec,
    register_gym_envs
)
from .agv_scenarios import (
    AGVSimulator, AGVPhysicsConfig, AGVState, AGVStateMachine,
    AGVPurePursuitController,
    get_agv_physics_spec
)
from .mujoco_sim import (
    MuJoCoSimulator, MuJoCoConfig, ControlMode,
    create_mujoco_simulator, HAS_MUJOCO
)
from .pybullet_sim import (
    PyBulletSimulator, PyBulletConfig, PyBulletGUI,
    create_pybullet_simulator, generate_agv_urdf,
    get_pybullet_spec, HAS_PYBULLET
)

__all__ = [
    # 基础仿真
    'RobotSimulator', 'SensorSimulator', 'SimConfig',
    'PhysicsEngine', 'SceneManager', 'TrajectoryRecorder',
    'create_scene', 'PRESET_SCENES',
    # Gymnasium 环境
    'SuperModelGymEnv', 'GymEnvConfig',
    'make_env', 'collect_rollout', 'get_gym_spec',
    'register_gym_envs',
    # AGV 仿真
    'AGVSimulator', 'AGVPhysicsConfig', 'AGVState', 'AGVStateMachine',
    'AGVPurePursuitController', 'get_agv_physics_spec',
    # MuJoCo 仿真
    'MuJoCoSimulator', 'MuJoCoConfig', 'ControlMode',
    'create_mujoco_simulator', 'HAS_MUJOCO',
    # PyBullet 仿真
    'PyBulletSimulator', 'PyBulletConfig', 'PyBulletGUI',
    'create_pybullet_simulator', 'generate_agv_urdf',
    'get_pybullet_spec', 'HAS_PYBULLET',
]


def get_agv_physics_spec(grade: str) -> 'AGVPhysicsConfig':
    """获取 AGV 五级物理规格"""
    return AGVPhysicsConfig.from_grade(grade)
