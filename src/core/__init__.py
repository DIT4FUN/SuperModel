"""
SuperModel Core Goals System
============================

超模态大模型核心目标系统 - 底层记忆与持续执行框架

核心目标层级 (Priority 0 = 最高优先):
  P0 保护人类安全  - 绝对优先级, 不可被覆盖
  P1 遵循人类指令  - 在P0前提下执行
  P2 善良品质      - 决策时考虑善意和伦理
  P3 热爱世界      - 积极维护环境生态
  P4 自我生存安全  - 保护自身硬件和软件完整性
  P5 自我进化      - 持续学习和能力提升

子系统:
  core_goals.py       - 核心目标定义与优先级管理
  safety_shield.py    - 安全护盾 (P0执行器)
  value_judgment.py   - 价值判断 (P2/P3执行器)
  self_preservation.py - 自我保存 (P4执行器)
  self_evolution.py   - 自我进化 (P5执行器)
  context_understanding.py - 上下文理解 (实时场景重建)
  decision_making.py  - 决策引擎 (整合所有维度)
  interaction.py      - 环境交互接口
  goal_dispatcher.py   - 目标调度器 (持续执行引擎)
  core_brain.py       - 核心大脑 (整体集成)

持续执行流程:
  sensors → context_understanding → decision_making
                                        ↓
                              goal_dispatcher (always running)
                                        ↓
                              ┌────────┴────────┐
                            safety_shield   interaction
"""

from .core_goals import (
    CoreGoal,
    GoalPriority,
    GoalState,
    GoalContext,
    CoreGoalsSystem,
)
from .safety_shield import SafetyShield, SafetyLevel
from .value_judgment import ValueJudgment, EthicalPrinciple
from .self_preservation import SelfPreservation
from .self_evolution import SelfEvolution
from .context_understanding import ContextUnderstanding
from .decision_making import DecisionMaking
from .interaction import InteractionManager
from .goal_dispatcher import GoalDispatcher
from .core_brain import CoreBrain

__all__ = [
    # Core goals
    "CoreGoal",
    "GoalPriority",
    "GoalState",
    "GoalContext",
    "CoreGoalsSystem",
    # Sub-systems
    "SafetyShield",
    "SafetyLevel",
    "ValueJudgment",
    "EthicalPrinciple",
    "SelfPreservation",
    "SelfEvolution",
    "ContextUnderstanding",
    "DecisionMaking",
    "InteractionManager",
    "GoalDispatcher",
    "CoreBrain",
]
