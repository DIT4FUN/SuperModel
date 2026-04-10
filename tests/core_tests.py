"""
Core Goals System Tests
=======================

测试 SuperModel 核心目标系统的所有模块:

测试模块:
  - core_goals.py     - 核心目标定义与优先级
  - safety_shield.py  - P0 安全护盾
  - value_judgment.py - P2/P3 价值判断
  - self_preservation.py - P4 自我保存
  - self_evolution.py - P5 自我进化
  - context_understanding.py - 上下文理解
  - decision_making.py - 决策引擎
  - interaction.py - 交互接口
  - goal_dispatcher.py - 目标调度器
  - core_brain.py - 核心大脑集成
"""

import unittest
import numpy as np
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.core_goals import (
    CoreGoal, GoalPriority, GoalState, GoalContext, CoreGoalsSystem,
)
from src.core.safety_shield import SafetyShield, SafetyConfig, SafetyLevel, DangerType, SafetyResponse
from src.core.value_judgment import ValueJudgment, EthicalPrinciple, EthicalAssessment
from src.core.self_preservation import SelfPreservation, SelfPreservationState
from src.core.self_evolution import SelfEvolution, EvolutionDimension, SkillProficiency
from src.core.context_understanding import (
    ContextUnderstanding, ContextMode, ContextRepresentation,
    SpatialContext, TemporalContext, SocialContext,
)
from src.core.decision_making import DecisionMaking, DecisionType, DecisionResult, ActionCandidate
from src.core.interaction import InteractionManager, InteractionState, ExecutionResult
from src.core.goal_dispatcher import GoalDispatcher, DispatcherMode, DispatcherConfig
from src.core.core_brain import CoreBrain


class TestCoreGoalsSystem(unittest.TestCase):
    """测试核心目标系统"""

    def test_goals_initialization(self):
        """测试目标初始化"""
        system = CoreGoalsSystem()
        goals = system.get_all_goals()

        self.assertEqual(len(goals), 6)  # 6个核心目标

        # 检查优先级
        priorities = [g.priority.value for g in goals]
        self.assertEqual(priorities, [0, 1, 2, 3, 4, 5])  # 0-5升序

    def test_p0_is_human_safety(self):
        """测试P0是保护人类安全"""
        system = CoreGoalsSystem()
        p0 = system.get_goal("p0_human_safety")

        self.assertIsNotNone(p0)
        self.assertEqual(p0.priority, GoalPriority.P0_HUMAN_SAFETY)
        self.assertTrue(p0.is_critical)
        self.assertTrue(p0.always_active)

    def test_always_active_goals(self):
        """测试所有目标都是持续激活"""
        system = CoreGoalsSystem()
        active = system.get_active_goals()

        # 所有6个目标都应该是always_active
        self.assertEqual(len(active), 6)

    def test_evaluate_all_goals(self):
        """测试目标评估"""
        system = CoreGoalsSystem()

        # 创建测试上下文
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([0.0, 0.0, 0.0]),
            human_positions=[np.array([2.0, 0.0, 0.0])],
            robot_battery_level=0.8,
            robot_temperature=30.0,
            environment_hazardous=False,
        )

        scores = system.evaluate_all_goals(ctx)

        self.assertEqual(len(scores), 6)  # 6个目标都有评分
        for goal_id, score in scores.items():
            self.assertIn(goal_id, [g.goal_id for g in system.get_all_goals()])
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_get_decision_weights(self):
        """测试决策权重计算"""
        system = CoreGoalsSystem()
        ctx = GoalContext()

        weights = system.get_decision_weights(ctx)

        self.assertIn("p0_human_safety", weights)
        self.assertGreater(
            weights["p0_human_safety"],
            weights["p5_self_evolution"]
        )  # P0权重应大于P5

    def test_conflict_resolution(self):
        """测试冲突解决"""
        system = CoreGoalsSystem()

        goal_a = system.get_goal("p0_human_safety")
        goal_b = system.get_goal("p5_self_evolution")

        winner, loser = system.resolve_conflict(goal_a, goal_b, GoalContext())

        # P0应始终优先
        self.assertEqual(winner.goal_id, "p0_human_safety")

    def test_status_summary(self):
        """测试状态摘要"""
        system = CoreGoalsSystem()
        summary = system.get_status_summary()

        self.assertEqual(summary["total_goals"], 6)
        self.assertEqual(summary["active_goals"], 6)
        self.assertIn("goal_states", summary)


class TestSafetyShield(unittest.TestCase):
    """测试安全护盾 (P0执行器)"""

    def setUp(self):
        self.shield = SafetyShield(grade="M")

    def test_safe_context(self):
        """测试安全上下文"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[np.array([5.0, 0.0, 0.0])],  # 5米外
            nearby_obstacles=[],
            robot_battery_level=0.8,
            robot_temperature=30.0,
        )

        is_safe, response, reason = self.shield.check_context(ctx)

        self.assertTrue(is_safe)
        self.assertEqual(response, SafetyResponse.NONE)
        self.assertIsNone(reason)

    def test_human_too_close(self):
        """测试人员距离过近"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[np.array([0.2, 0.0, 0.0])],  # 20cm
            nearby_obstacles=[],
        )

        is_safe, response, reason = self.shield.check_context(ctx)

        self.assertFalse(is_safe)
        self.assertEqual(response, SafetyResponse.EMERGENCY_STOP)
        self.assertIsNotNone(reason)

    def test_obstacle_too_close(self):
        """测试障碍物过近"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[],
            nearby_obstacles=[type('obj', (), {'position': np.array([0.1, 0.0, 0.0])})()],
        )

        is_safe, response, reason = self.shield.check_context(ctx)

        self.assertFalse(is_safe)
        self.assertIn(response, [SafetyResponse.STOP, SafetyResponse.EMERGENCY_STOP])

    def test_emergency_stop(self):
        """测试紧急停止触发"""
        self.shield.trigger_emergency_stop("test_reason")

        self.assertTrue(self.shield._emergency_stop_active)
        self.assertEqual(self.shield._emergency_stop_reason, "test_reason")

        # 释放
        self.shield.release_emergency_stop()
        self.assertFalse(self.shield._emergency_stop_active)

    def test_get_override_action(self):
        """测试获取安全覆盖动作"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([1.0, 0.0, 0.0]),
            human_positions=[np.array([0.5, 0.0, 0.0])],
        )

        # 有危险时应返回零动作
        override = self.shield.get_override_action(ctx, np.array([1.0, 0, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(override, 0.0))

    def test_grade_configs(self):
        """测试AGV等级配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            shield = SafetyShield(grade=grade)
            self.assertEqual(shield.config.grade.value, grade)


class TestValueJudgment(unittest.TestCase):
    """测试价值判断 (P2/P3执行器)"""

    def setUp(self):
        self.judge = ValueJudgment()

    def test_judge_safe_action(self):
        """测试安全动作评估"""
        action = np.zeros(6)
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            human_positions=[np.array([3.0, 0.0, 0.0])],
        )

        assessments = self.judge.judge_action(action, ctx)

        self.assertGreater(len(assessments), 0)

        # 应该有NON_HARM评估
        non_harm = next(
            (a for a in assessments if a.principle == EthicalPrinciple.NON_HARM),
            None
        )
        self.assertIsNotNone(non_harm)
        self.assertGreaterEqual(non_harm.score, 0.0)

    def test_ethical_compliance_pass(self):
        """测试伦理合规通过"""
        # 使用最高分数确保综合评分 >= 0.3
        # weighted_score = (1.0*10 + 1.0*6) / 45 = 16/45 = 0.356
        assessments = [
            EthicalAssessment(
                principle=EthicalPrinciple.NON_HARM,
                score=1.0,
                reasoning="安全",
            ),
            EthicalAssessment(
                principle=EthicalPrinciple.HONESTY,
                score=1.0,
                reasoning="诚实",
            ),
        ]

        is_compliant, reason = self.judge.check_ethical_compliance(assessments)

        self.assertTrue(is_compliant)

    def test_ethical_compliance_fail(self):
        """测试伦理合规失败"""
        assessments = [
            EthicalAssessment(
                principle=EthicalPrinciple.NON_HARM,
                score=-0.7,  # 违反不伤害原则
                reasoning="危险动作",
            ),
        ]

        is_compliant, reason = self.judge.check_ethical_compliance(assessments)

        self.assertFalse(is_compliant)

    def test_record_interaction(self):
        """测试交互记录"""
        self.judge.record_interaction(
            was_helpful=True,
            was_collaborative=True,
        )

        self.assertEqual(self.judge._metrics.total_interactions, 1)
        self.assertEqual(self.judge._metrics.helpful_actions, 1)
        self.assertEqual(self.judge._metrics.collaborative_actions, 1)


class TestSelfPreservation(unittest.TestCase):
    """测试自我保存 (P4执行器)"""

    def setUp(self):
        self.sp = SelfPreservation()

    def test_initial_health(self):
        """测试初始健康状态"""
        self.assertGreaterEqual(self.sp.get_health_score(), 0.0)
        self.assertLessEqual(self.sp.get_health_score(), 1.0)

    def test_update_state(self):
        """测试状态更新"""
        ctx = GoalContext(
            robot_battery_level=0.8,
            robot_temperature=35.0,
            self_health_score=0.9,
        )

        self.sp.update_state(ctx)

        self.assertAlmostEqual(self.sp._state.battery_level, 0.8)

    def test_battery_safety(self):
        """测试电池安全检查"""
        # 低电量
        self.sp._state.battery_level = 0.1
        safe, reason = self.sp.check_battery_safety()
        self.assertFalse(safe)

        # 正常电量
        self.sp._state.battery_level = 0.5
        safe, reason = self.sp.check_battery_safety()
        self.assertTrue(safe)

    def test_protective_action_low_health(self):
        """测试低健康时的保护动作"""
        self.sp._state.overall_health = 0.2

        action, reason = self.sp.get_protective_action(GoalContext())

        self.assertTrue(np.allclose(action, 0.0))  # 应该停止


class TestSelfEvolution(unittest.TestCase):
    """测试自我进化 (P5执行器)"""

    def setUp(self):
        self.evo = SelfEvolution()

    def test_record_experience(self):
        """测试记录经验"""
        exp_id = self.evo.record_experience(
            context_type="navigation",
            action=np.array([0.5, 0, 0, 0, 0, 0]),
            outcome=0.8,
            tags=["exploration", "forward"],
        )

        self.assertIsNotNone(exp_id)
        self.assertGreater(len(self.evo._experiences), 0)

    def test_update_skill(self):
        """测试技能更新"""
        self.evo.update_skill("navigation", success=True, quality_score=0.8)

        skill = self.evo.get_skill_proficiency("navigation")
        self.assertIsNotNone(skill)
        self.assertGreater(skill.success_rate, 0.0)

    def test_learning_progress(self):
        """测试学习进度"""
        # 记录一些经验
        for i in range(10):
            self.evo.record_experience(
                context_type="test",
                action=np.zeros(6),
                outcome=0.5 + 0.1 * (i % 3),
            )

        progress = self.evo.get_learning_progress()
        self.assertGreaterEqual(progress, 0.0)
        self.assertLessEqual(progress, 1.0)

    def test_exploration_vs_exploitation(self):
        """测试探索vs利用"""
        ctx = GoalContext()

        is_exp_1, _ = self.evo.get_exploration_action(ctx, epsilon=1.0)  # 100%探索
        is_exp_2, _ = self.evo.get_exploration_action(ctx, epsilon=0.0)  # 0%探索

        self.assertTrue(is_exp_1)
        self.assertFalse(is_exp_2)


class TestContextUnderstanding(unittest.TestCase):
    """测试上下文理解"""

    def setUp(self):
        self.ctx_und = ContextUnderstanding()

    def test_update_basic(self):
        """测试基本更新"""
        result = self.ctx_und.update(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([0.5, 0.0, 0.0]),
            laser_ranges=np.array([5.0, 5.0, 5.0]),
            human_positions=[np.array([3.0, 0.0, 0.0])],
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ContextRepresentation)
        self.assertEqual(result.mode, ContextMode.NORMAL)

    def test_hazard_mode(self):
        """测试危险模式"""
        result = self.ctx_und.update(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([1.0, 0.0, 0.0]),
            laser_ranges=np.array([0.2, 0.2, 0.2]),  # 极近障碍
            human_positions=[],
        )

        self.assertIn(result.mode, [ContextMode.CAUTIOUS, ContextMode.EMERGENCY])
        self.assertGreaterEqual(result.hazard_level, 0.5)  # 0.2m距离→hazard=0.5

    def test_cautious_mode_human_near(self):
        """测试人员接近时的谨慎模式"""
        result = self.ctx_und.update(
            robot_position=np.array([0.0, 0.0, 0.0]),
            human_positions=[np.array([0.8, 0.0, 0.0])],  # 80cm
        )

        self.assertIn(result.mode, [ContextMode.CAUTIOUS, ContextMode.EMERGENCY])

    def test_confidence(self):
        """测试置信度计算"""
        result = self.ctx_und.update(
            vision=np.random.randn(512),
            laser_ranges=np.random.randn(360),
            human_positions=[np.array([2.0, 0.0, 0.0])],
        )

        self.assertGreater(result.confidence, 0.5)


class TestDecisionMaking(unittest.TestCase):
    """测试决策引擎"""

    def setUp(self):
        self.dm = DecisionMaking()

    def test_basic_decision(self):
        """测试基本决策"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[np.array([5.0, 0.0, 0.0])],
            robot_battery_level=0.8,
            robot_temperature=30.0,
        )

        result = self.dm.decide(ctx)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, DecisionResult)
        self.assertEqual(result.action.shape, (6,))

    def test_decision_types(self):
        """测试决策类型"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
        )

        result = self.dm.decide(ctx)

        # 应该有有效的决策类型
        self.assertIsInstance(result.decision_type, DecisionType)

    def test_safety_override(self):
        """测试安全覆盖"""
        shield = SafetyShield(grade="M")
        dm = DecisionMaking(safety_shield=shield)

        # 危险上下文
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([2.0, 0.0, 0.0]),
            human_positions=[np.array([0.3, 0.0, 0.0])],  # 极近
        )

        result = dm.decide(ctx)

        # 应该触发安全覆盖
        self.assertFalse(result.safety_passed)

    def test_instruction_following(self):
        """测试指令执行"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_instructions=["前进"],
        )

        result = self.dm.decide(ctx, instruction="前进")

        self.assertIsNotNone(result)


class TestInteractionManager(unittest.TestCase):
    """测试交互管理器"""

    def setUp(self):
        self.interaction = InteractionManager()

    def test_execute_basic(self):
        """测试基本执行"""
        action = np.array([0.5, 0, 0, 0, 0, 0])
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[np.array([5.0, 0.0, 0.0])],
        )

        result = self.interaction.execute(action, ctx, blocking=False)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)

    def test_emergency_stop(self):
        """测试紧急停止"""
        self.interaction.emergency_stop("test")

        self.assertEqual(self.interaction._state, InteractionState.EMERGENCY_STOP)

    def test_action_preprocessing(self):
        """测试动作预处理"""
        # 过大动作应该被限幅
        action = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        ctx = GoalContext()

        # 手动测试预处理
        processed = self.interaction._preprocess_action(action, ctx)

        self.assertTrue(np.all(processed[:3] <= 2.0))  # 线性速度限幅
        self.assertTrue(np.all(processed[3:] <= 1.5))  # 角速度限幅


class TestGoalDispatcher(unittest.TestCase):
    """测试目标调度器"""

    def test_dispatcher_config(self):
        """测试调度器配置"""
        config = DispatcherConfig(
            target_cycle_period_ms=20.0,
        )

        self.assertEqual(config.target_cycle_period_ms, 20.0)

    def test_dispatcher_modes(self):
        """测试调度器模式"""
        config = DispatcherConfig()
        dispatcher = GoalDispatcher(config=config)

        self.assertEqual(dispatcher._mode, DispatcherMode.REAL_TIME)

        dispatcher.set_mode(DispatcherMode.PAUSED)
        self.assertEqual(dispatcher._mode, DispatcherMode.PAUSED)

    def test_emergency_stop(self):
        """测试调度器紧急停止"""
        dispatcher = GoalDispatcher()

        dispatcher.trigger_emergency_stop("danger_test")

        self.assertTrue(dispatcher._emergency_stop_active)
        self.assertEqual(dispatcher._emergency_reason, "danger_test")

        dispatcher.release_emergency_stop()
        self.assertFalse(dispatcher._emergency_stop_active)


class TestCoreBrain(unittest.TestCase):
    """测试核心大脑"""

    def setUp(self):
        self.brain = CoreBrain(grade="M")

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.brain._grade, "M")
        self.assertIsNotNone(self.brain._goals)
        self.assertIsNotNone(self.brain._safety)
        self.assertIsNotNone(self.brain._value_judge)
        self.assertIsNotNone(self.brain._self_preservation)
        self.assertIsNotNone(self.brain._evolution)
        self.assertIsNotNone(self.brain._context_understanding)
        self.assertIsNotNone(self.brain._decision_making)
        self.assertIsNotNone(self.brain._interaction)

    def test_update_context(self):
        """测试上下文更新"""
        ctx = self.brain.update_context(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([0.5, 0.0, 0.0]),
            laser_ranges=np.array([5.0] * 360),
            human_positions=[np.array([3.0, 0.0, 0.0])],
            robot_battery_level=0.9,
            robot_temperature=35.0,
        )

        self.assertIsNotNone(ctx)
        self.assertIsInstance(ctx, GoalContext)
        self.assertEqual(self.brain._context, ctx)

    def test_decide_without_context(self):
        """测试无上下文时的决策"""
        # 默认上下文
        decision = self.brain.decide()
        self.assertIsNotNone(decision)

    def test_decide_with_context(self):
        """测试有上下文时的决策"""
        self.brain.update_context(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[np.array([3.0, 0.0, 0.0])],
        )

        decision = self.brain.decide()
        self.assertIsNotNone(decision)

    def test_step(self):
        """测试单步"""
        self.brain.update_context(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.zeros(3),
            human_positions=[np.array([3.0, 0.0, 0.0])],
        )

        decision, execution = self.brain.step()

        self.assertIsNotNone(decision)
        self.assertIsNotNone(execution)

    def test_emergency_stop(self):
        """测试紧急停止"""
        self.brain.trigger_emergency_stop("danger")
        self.assertTrue(self.brain._dispatcher._emergency_stop_active)

    def test_goals_status(self):
        """测试目标状态"""
        status = self.brain.get_goals_status()
        self.assertIn("total_goals", status)
        self.assertEqual(status["total_goals"], 6)

    def test_all_scores(self):
        """测试所有评分"""
        self.brain.update_context(
            robot_position=np.array([0.0, 0.0, 0.0]),
        )

        scores = self.brain.get_all_scores()
        self.assertEqual(len(scores), 6)

    def test_get_status(self):
        """测试完整状态"""
        status = self.brain.get_status()

        self.assertIn("goals", status)
        self.assertIn("safety_shield", status)
        self.assertIn("value_judgment", status)
        self.assertIn("self_preservation", status)
        self.assertIn("self_evolution", status)
        self.assertIn("decision_making", status)
        self.assertIn("interaction", status)

    def test_different_grades(self):
        """测试不同AGV等级"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            brain = CoreBrain(grade=grade)
            self.assertEqual(brain._grade, grade)


class TestGoalContext(unittest.TestCase):
    """测试GoalContext"""

    def test_human_distance(self):
        """测试人员距离计算"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            human_positions=[
                np.array([3.0, 0.0, 0.0]),
                np.array([5.0, 0.0, 0.0]),
            ],
        )

        dist = ctx.get_human_distance()
        self.assertAlmostEqual(dist, 3.0)

    def test_no_human_distance(self):
        """测试无人员时距离"""
        ctx = GoalContext(robot_position=np.array([0.0, 0.0, 0.0]))

        dist = ctx.get_human_distance()
        self.assertIsNone(dist)

    def test_is_safe_for_movement(self):
        """测试移动安全性"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            environment_hazardous=False,
            nearby_obstacles=[],
        )

        self.assertTrue(ctx.is_safe_for_movement())

    def test_is_unsafe_when_hazardous(self):
        """测试危险环境时不安全"""
        ctx = GoalContext(
            robot_position=np.array([0.0, 0.0, 0.0]),
            environment_hazardous=True,
        )

        self.assertFalse(ctx.is_safe_for_movement())


if __name__ == '__main__':
    unittest.main()
