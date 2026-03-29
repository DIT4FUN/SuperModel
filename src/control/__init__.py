"""
SuperModel 控制模块
==================

执行层模块:
- motion: 运动控制 (速度/位置/力矩控制)
- trajectory: 轨迹规划 (RRT, 多项式, S曲线)
- skill: 技能库调度
- planner: 任务规划器
- impedance: 阻抗控制
- ros2_interface: ROS2 Humble 集成接口
"""

from .motion import MotionController, JointTrajectory, TwistCommand
from .trajectory import (
    TrajectoryGenerator, RRTPlanner, ScurveGenerator,
    JointWaypoint, CartesianWaypoint, TrajectoryConfig,
    PlanningAlgorithm, get_trajectory_spec
)
from .skill import SkillLibrary, Skill, SkillRegistry
from .planner import TaskPlanner, Task, TaskStatus
from .impedance import ImpedanceController
from .ros2_interface import (
    ROS2JointTrajectoryInterface, ROS2TopicInterface, ROS2ServiceInterface,
    JointCommand, JointState, ControlInterfaceMode,
    ROSTopics, ROSServices, ROSParams, get_ros2_spec
)

__all__ = [
    'MotionController', 'JointTrajectory', 'TwistCommand',
    'TrajectoryGenerator', 'RRTPlanner', 'ScurveGenerator',
    'JointWaypoint', 'CartesianWaypoint', 'TrajectoryConfig',
    'PlanningAlgorithm', 'get_trajectory_spec',
    'SkillLibrary', 'Skill', 'SkillRegistry',
    'TaskPlanner', 'Task', 'TaskStatus',
    'ImpedanceController',
    'ROS2JointTrajectoryInterface', 'ROS2TopicInterface', 'ROS2ServiceInterface',
    'JointCommand', 'JointState', 'ControlInterfaceMode',
    'ROSTopics', 'ROSServices', 'ROSParams', 'get_ros2_spec'
]
