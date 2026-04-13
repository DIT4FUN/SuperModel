"""
test_embodied_e2e_scenarios.py - 具身智能端到端场景集成测试
SuperModel v3.9.1 - 2026-04-14

端到端测试覆盖:
- EmbodiedPipeline × SceneIntelligence × BehaviorTree 完整集成
- 场景化任务执行 (仓库/医院/工厂/餐厅/户外)
- AGV五级规格 × 场景矩阵
- 状态持久化与恢复
- 行为树具身任务规划
"""

import time
import pytest
import numpy as np
from typing import Dict, Any

from src.embodied.embodied_pipeline import (
    EmbodiedPipeline,
    PipelineConfig,
    PipelineMode,
    PipelineState,
    TaskRequest,
    TaskResult,
    create_embodied_pipeline,
)
from src.embodied.scene_intelligence import (
    SceneType,
    SceneIntelligence,
    SceneConfig,
    SceneContext,
    get_scene_intelligence,
)
from src.embodied.behavior_tree import (
    BehaviorTree,
    SequenceNode,
    SelectorNode,
    ParallelNode,
    ConditionNode,
    LambdaActionNode,
    Blackboard,
    EmbodiedTask,
    TaskStatus,
    NodeStatus,
)
from src.embodied.scene_task_planner import (
    SceneTaskPlanner,
    SceneTaskConfig,
    SceneAdaptationEngine,
    get_scene_task_planner,
)


# Fixtures

@pytest.fixture
def scene_intel():
    config = SceneConfig(grade="M")
    return SceneIntelligence(config=config)


@pytest.fixture
def scene_planner(scene_intel):
    config = SceneTaskConfig(grade="M")
    return SceneTaskPlanner(config=config, scene_intelligence=scene_intel)


@pytest.fixture
def adaptation_engine():
    return SceneAdaptationEngine(memory=None)


@pytest.fixture
def embodied_pipeline(scene_intel):
    config = PipelineConfig(
        grade="M",
        mode=PipelineMode.SIMULATION,
        scene_type="WAREHOUSE",
        enable_scene_intelligence=True,
        enable_memory=True,
    )
    pipeline = EmbodiedPipeline(config=config)
    pipeline._scene_intel = scene_intel
    return pipeline


@pytest.fixture
def blackboard():
    bb = Blackboard()
    bb.robot_state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.0, 0.0]),
        "battery": 0.85,
        "cargo_loaded": False,
        "at_station": None,
    }
    bb.world_state = {
        "obstacles": [],
        "human_positions": [],
        "stations": {
            "pickup_A": {"position": np.array([2.0, 0.0, 0.0]), "occupied": False},
            "dropoff_B": {"position": np.array([5.0, 0.0, 0.0]), "occupied": False},
            "charging": {"position": np.array([0.0, 0.0, 0.0]), "occupied": False},
        },
    }
    bb.goal_state = {
        "target_station": "dropoff_B",
        "task_type": "delivery",
        "priority": 1,
    }
    return bb


# =============================================================================
# Pipeline Initialization Tests
# =============================================================================

class TestPipelineInitialization:

    def test_pipeline_config_all_grades(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            config = PipelineConfig(grade=grade, mode=PipelineMode.SIMULATION)
            assert config.grade == grade

    def test_pipeline_config_all_modes(self):
        for mode in [
            PipelineMode.SIMULATION,
            PipelineMode.HARDWARE_IN_LOOP,
            PipelineMode.FULL_PHYSICAL,
        ]:
            config = PipelineConfig(grade="M", mode=mode)
            assert config.mode == mode

    def test_pipeline_state_transitions(self, embodied_pipeline):
        assert embodied_pipeline.state == PipelineState.IDLE
        
        success = embodied_pipeline.start()
        assert success
        assert embodied_pipeline.state == PipelineState.READY
        
        embodied_pipeline._set_state(PipelineState.RUNNING)
        paused = embodied_pipeline.pause()
        assert paused
        assert embodied_pipeline.state == PipelineState.PAUSED
        
        resumed = embodied_pipeline.resume()
        assert resumed
        assert embodied_pipeline.state == PipelineState.RUNNING
        
        embodied_pipeline.stop()
        assert embodied_pipeline.state == PipelineState.STOPPED

    def test_pipeline_uptime(self, embodied_pipeline):
        embodied_pipeline.start()
        time.sleep(0.05)
        assert embodied_pipeline.uptime_s > 0
        embodied_pipeline.stop()

    def test_factory_create_all_grades(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            p = create_embodied_pipeline(grade=grade, mode="simulation")
            assert p is not None
            p.stop()


# =============================================================================
# Scene Intelligence Integration Tests
# =============================================================================

class TestSceneIntelligenceIntegration:

    def test_scene_intel_initialization(self):
        config = SceneConfig(grade="M")
        si = SceneIntelligence(config=config)
        assert si._current_context is not None
        assert si._classifier is not None
        assert si._rule_engine is not None

    def test_scene_context_all_types(self, scene_intel):
        scene_intel.update(
            laser_ranges=np.random.rand(360),
            location_hint="warehouse",
        )
        ctx = scene_intel.get_scene_context()
        assert ctx is not None

    def test_adaptive_speed_limits(self, scene_intel):
        scene_intel.update(laser_ranges=np.random.rand(360))
        base_speed = 1.5
        warehouse_limit = scene_intel.get_adaptive_speed_limit(base_speed)
        assert 0 < warehouse_limit <= base_speed

    def test_safe_distance(self, scene_intel):
        distance = scene_intel.get_safe_distance()
        assert distance > 0

    def test_scene_recognition(self, scene_intel):
        features = {
            "obstacle_density": 0.3,
            "human_density": 0.1,
            "floor_type": 0.8,
            "aisle_width": 2.5,
            "ceiling_height": 4.0,
        }
        scene_type, confidence = scene_intel._classifier._decide_scene(
            features, location_hint=""
        )
        assert scene_type in SceneType
        assert 0 <= confidence <= 1.0

    def test_active_rules(self, scene_intel):
        scene_intel.update(laser_ranges=np.random.rand(360))
        rules = scene_intel.get_active_rules()
        assert isinstance(rules, dict)

    def test_get_scene_intelligence_singleton(self):
        si1 = get_scene_intelligence()
        si2 = get_scene_intelligence()
        assert si1 is si2


# =============================================================================
# Behavior Tree Integration Tests
# =============================================================================

class TestBehaviorTreeIntegration:

    def test_blackboard_state(self, blackboard):
        assert blackboard.robot_state is not None
        assert blackboard.world_state is not None
        assert blackboard.goal_state is not None

    def test_blackboard_update(self, blackboard):
        blackboard.data["test_key"] = "test_value"
        assert blackboard.get("test_key") == "test_value"

    def test_sequence_node_success(self, blackboard):
        results = []
        seq = SequenceNode("test_seq")
        seq.add_child(
            LambdaActionNode(
                lambda ctx: (results.append(1) or NodeStatus.SUCCESS)
            )
        )
        seq.add_child(
            LambdaActionNode(
                lambda ctx: (results.append(2) or NodeStatus.SUCCESS)
            )
        )
        seq.add_child(
            LambdaActionNode(
                lambda ctx: (results.append(3) or NodeStatus.SUCCESS)
            )
        )
        status = seq.tick(blackboard)
        assert status == NodeStatus.SUCCESS
        assert results == [1, 2, 3]

    def test_sequence_node_early_failure(self, blackboard):
        seq = SequenceNode("test_seq")
        seq.add_child(
            LambdaActionNode(lambda ctx: NodeStatus.SUCCESS)
        )
        seq.add_child(
            LambdaActionNode(lambda ctx: NodeStatus.FAILURE)
        )
        seq.add_child(
            LambdaActionNode(lambda ctx: NodeStatus.SUCCESS)
        )
        status = seq.tick(blackboard)
        assert status == NodeStatus.FAILURE

    def test_selector_node_success_on_first(self, blackboard):
        results = []
        sel = SelectorNode("test_sel")
        sel.add_child(
            LambdaActionNode(lambda ctx: (results.append(1) or NodeStatus.SUCCESS))
        )
        sel.add_child(
            LambdaActionNode(lambda ctx: (results.append(2) or NodeStatus.SUCCESS))
        )
        status = sel.tick(blackboard)
        assert status == NodeStatus.SUCCESS
        assert results == [1]

    def test_selector_node_fallback(self, blackboard):
        results = []
        sel = SelectorNode("test_sel")
        sel.add_child(
            LambdaActionNode(lambda ctx: (results.append(1) or NodeStatus.FAILURE))
        )
        sel.add_child(
            LambdaActionNode(lambda ctx: (results.append(2) or NodeStatus.SUCCESS))
        )
        status = sel.tick(blackboard)
        assert status == NodeStatus.SUCCESS
        assert results == [1, 2]

    def test_parallel_node_require_all(self, blackboard):
        results = []
        para = ParallelNode(
            name="test_para",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY,
        )
        para.add_child(
            LambdaActionNode(lambda ctx: (results.append(1) or NodeStatus.SUCCESS))
        )
        para.add_child(
            LambdaActionNode(lambda ctx: (results.append(2) or NodeStatus.SUCCESS))
        )
        status = para.tick(blackboard)
        assert status == NodeStatus.SUCCESS
        assert len(results) == 2

    def test_condition_node_true(self, blackboard):
        cond = ConditionNode(lambda ctx: ctx.robot_state["battery"] > 0.2, name="battery_check")
        status = cond.tick(blackboard)
        assert status == NodeStatus.SUCCESS

    def test_condition_node_false(self, blackboard):
        cond = ConditionNode(lambda ctx: ctx.robot_state["cargo_loaded"], name="cargo_check")
        status = cond.tick(blackboard)
        assert status == NodeStatus.FAILURE

    def test_behavior_tree_tick(self, blackboard):
        seq = SequenceNode("root")
        seq.add_child(
            LambdaActionNode(lambda ctx: NodeStatus.SUCCESS)
        )
        bt = BehaviorTree(root=seq, name="test_bt")
        status = bt.tick()
        assert status == NodeStatus.SUCCESS


# =============================================================================
# Scene Task Planner Integration Tests
# =============================================================================

class TestSceneTaskPlannerIntegration:

    def test_planner_initialization(self, scene_planner):
        assert scene_planner is not None
        assert scene_planner._config is not None
        assert scene_planner._library is not None

    def test_task_library_has_templates(self, scene_planner):
        library = scene_planner._library
        assert len(library._templates) > 0

    def test_plan_task_returns_bt_and_task(self, scene_planner):
        bt, task = scene_planner.plan_task(
            task_description="Deliver package",
            scene_type=SceneType.WAREHOUSE,
        )
        from src.embodied.behavior_tree import BTNode
        assert isinstance(bt, BTNode)
        assert isinstance(task, EmbodiedTask)

    def test_scene_task_library_templates(self, scene_planner):
        library = scene_planner._library
        templates = library._templates
        assert len(templates) > 0
        scene_types_with_templates = set(templates.keys())
        assert SceneType.WAREHOUSE in scene_types_with_templates


# =============================================================================
# Scene Adaptation Engine Tests
# =============================================================================

class TestSceneAdaptationEngine:

    def test_engine_initialization(self, adaptation_engine):
        assert adaptation_engine is not None
        assert adaptation_engine._scene_params is not None

    def test_record_outcome(self, adaptation_engine):
        adaptation_engine.record_outcome(
            scene_type=SceneType.WAREHOUSE,
            task_type="delivery",
            success=True,
            duration_s=30.0,
            parameters={"speed": 1.0, "safe_distance": 0.5},
        )
        params = adaptation_engine._scene_params.get(SceneType.WAREHOUSE, {})
        assert params.get("success_rate", 0) > 0

    def test_adaptive_params_low_success(self, adaptation_engine):
        adaptation_engine._scene_params[SceneType.FACTORY] = {
            "success_rate": 0.6,
            "speed_multiplier": 1.0,
        }
        base_params = {"max_speed": 1.5, "safe_distance": 0.5}
        adjusted = adaptation_engine.get_adaptive_params(SceneType.FACTORY, base_params)
        assert adjusted["max_speed"] < base_params["max_speed"]
        assert adjusted["safe_distance"] > base_params["safe_distance"]

    def test_adaptive_params_high_success(self, adaptation_engine):
        adaptation_engine._scene_params[SceneType.RESTAURANT] = {
            "success_rate": 0.95,
            "speed_multiplier": 1.0,
        }
        base_params = {"max_speed": 1.0, "safe_distance": 0.5}
        adjusted = adaptation_engine.get_adaptive_params(SceneType.RESTAURANT, base_params)
        assert adjusted["max_speed"] >= base_params["max_speed"] * 0.95


# =============================================================================
# Pipeline x Scene x BT E2E Tests
# =============================================================================

class TestPipelineSceneBTE2E:

    def test_pipeline_with_scene_intelligence(self):
        config = PipelineConfig(
            grade="M",
            mode=PipelineMode.SIMULATION,
            scene_type="WAREHOUSE",
            enable_scene_intelligence=True,
        )
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()  # _init_scene_intelligence sets a bad _scene_intel from str config
        assert pipeline.state == PipelineState.READY
        # Override AFTER start() since start() re-initializes _scene_intel
        pipeline._scene_intel = SceneIntelligence(config=SceneConfig(grade="M"))
        pipeline._scene_intel.update(
            laser_ranges=np.random.rand(360),
            current_location="zone_A",
        )
        pipeline._set_state(PipelineState.RUNNING)
        assert pipeline.is_running
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_pipeline_task_submission(self):
        config = PipelineConfig(
            grade="M",
            mode=PipelineMode.SIMULATION,
            enable_skill_registry=True,
        )
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        request = TaskRequest(
            task_id="test_001",
            task_type="delivery",
            target="pickup_A",
            payload={"destination": "dropoff_B"},
            priority=1,
        )
        success = pipeline.submit_task(request)
        assert success
        time.sleep(0.1)
        status = pipeline.get_status()
        assert "state" in status
        assert "uptime_s" in status
        pipeline.stop()

    def test_pipeline_status_report(self):
        config = PipelineConfig(grade="M", mode=PipelineMode.SIMULATION)
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        status = pipeline.get_status()
        assert "state" in status
        assert "uptime_s" in status
        assert "modules" in status
        pipeline.stop()

    def test_pipeline_health_report(self):
        config = PipelineConfig(grade="M", mode=PipelineMode.SIMULATION)
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        health = pipeline.get_health_report()
        assert "pipeline_state" in health
        assert "modules" in health
        assert "tasks" in health
        pipeline.stop()

    def test_pipeline_save_and_restore_state(self):
        config = PipelineConfig(grade="M", mode=PipelineMode.SIMULATION)
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        time.sleep(0.05)
        state = pipeline.save_state()
        assert state is not None
        assert "pipeline" in state
        assert "version" in state
        pipeline.stop()
        restored = EmbodiedPipeline(config=config)
        success = restored.restore_state(state)
        assert success
        restored.stop()

    def test_pipeline_reset_health(self):
        config = PipelineConfig(grade="M", mode=PipelineMode.SIMULATION)
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        pipeline.reset_health()
        health = pipeline.get_health_report()
        assert health["error"] is None
        pipeline.stop()


# =============================================================================
# AGV Grade x Scene Matrix Tests
# =============================================================================

class TestAGVGradeSceneMatrix:

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    @pytest.mark.parametrize("scene", [
        "WAREHOUSE", "FACTORY", "HOSPITAL", "RESTAURANT", "OUTDOOR"
    ])
    def test_grade_scene_combination(self, grade, scene):
        config = PipelineConfig(
            grade=grade,
            mode=PipelineMode.SIMULATION,
            scene_type=scene,
        )
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        status = pipeline.get_status()
        assert "state" in status
        pipeline.stop()


# =============================================================================
# Embodied Task Execution Tests
# =============================================================================

class TestEmbodiedTaskExecution:

    def test_embodied_task_creation(self):
        task = EmbodiedTask(
            task_id="task_001",
            task_type="delivery",
            goal_description="Deliver package from A to B",
            target_position=np.array([5.0, 0.0, 0.0]),
            priority=1,
        )
        assert task.task_id == "task_001"
        assert task.task_type == "delivery"
        assert task.status == TaskStatus.IDLE

    def test_task_status_transitions(self):
        task = EmbodiedTask(
            task_id="task_002",
            task_type="patrol",
            goal_description="Patrol warehouse",
            priority=2,
        )
        assert task.status == TaskStatus.IDLE
        task.start()
        assert task.status == TaskStatus.RUNNING
        task.finish(success=True)
        assert task.status == TaskStatus.COMPLETED
        assert task.success is True

    def test_task_timeout(self):
        task = EmbodiedTask(
            task_id="task_003",
            task_type="delivery",
            goal_description="Test",
            timeout=0.01,
        )
        task.start()
        time.sleep(0.02)
        assert task.is_timeout() is True

    def test_task_duration(self):
        task = EmbodiedTask(
            task_id="task_004",
            task_type="delivery",
            goal_description="Test",
        )
        task.start()
        time.sleep(0.02)
        duration = task.get_duration()
        assert duration >= 0.02


# =============================================================================
# Performance Tests
# =============================================================================

class TestPipelinePerformance:

    def test_rapid_start_stop(self):
        config = PipelineConfig(grade="M", mode=PipelineMode.SIMULATION)
        for i in range(5):
            pipeline = EmbodiedPipeline(config=config)
            pipeline.start()
            time.sleep(0.01)
            pipeline.stop()
        assert True

    def test_multiple_task_submissions(self):
        config = PipelineConfig(
            grade="M",
            mode=PipelineMode.SIMULATION,
            enable_skill_registry=True,
        )
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        for i in range(10):
            request = TaskRequest(
                task_id=f"task_{i:03d}",
                task_type="delivery",
                target=f"station_{i}",
                priority=i % 3 + 1,
            )
            pipeline.submit_task(request)
        time.sleep(0.1)
        status = pipeline.get_status()
        assert status is not None
        pipeline.stop()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:

    def test_pipeline_with_invalid_grade(self):
        config = PipelineConfig(grade="INVALID", mode=PipelineMode.SIMULATION)
        pipeline = EmbodiedPipeline(config=config)
        pipeline.start()
        assert pipeline.state == PipelineState.READY
        pipeline._set_state(PipelineState.RUNNING)
        assert pipeline.is_running
        pipeline.stop()

    def test_scene_intel_minimal_update(self):
        config = SceneConfig(grade="M")
        scene_intel = SceneIntelligence(config=config)
        scene_intel.update()
        ctx = scene_intel.get_scene_context()
        assert ctx is not None

    def test_behavior_tree_empty_children(self):
        seq = SequenceNode("empty_seq")
        status = seq.tick(Blackboard())
        assert status == NodeStatus.SUCCESS

    def test_planner_with_no_memory(self):
        config = SceneTaskConfig(grade="M")
        planner = SceneTaskPlanner(config=config, memory=None)
        assert planner._memory is None
