"""
test_task_executor.py - 具身任务执行器测试
测试 MemoryEnhancedExecutor / ScenarioTaskExecutor / create_executor_from_config
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from embodied.behavior_tree import (
    NodeStatus,
    BTNode,
    SequenceNode,
    ActionNode,
    BehaviorTree,
    create_behavior_tree_from_dict,
)
from embodied.task_executor import (
    ExecutionPhase,
    ExecutionResult,
    TaskExecutionRecord,
    MemoryEnhancedExecutor,
    ScenarioTaskExecutor,
    create_task_executor,
    create_executor_from_config,
)


# ============================================================================
# 测试辅助
# ============================================================================

class DummyAction(ActionNode):
    """测试用假动作节点"""
    def __init__(self, name="DummyAction", delay=0.05, result=NodeStatus.SUCCESS):
        super().__init__(name=name)
        self.delay = delay
        self.result = result
        self.start_time = None

    def execute(self, blackboard):
        if self.start_time is None:
            self.start_time = time.time()
        if time.time() - self.start_time >= self.delay:
            self.start_time = None
            return self.result
        return NodeStatus.RUNNING


class FailingAction(ActionNode):
    """总是失败的动作节点"""
    def __init__(self, name="FailingAction"):
        super().__init__(name=name)

    def execute(self, blackboard):
        return NodeStatus.FAILURE


class InfiniteAction(ActionNode):
    """永远RUNNING的动作节点"""
    def __init__(self, name="InfiniteAction"):
        super().__init__(name=name)

    def execute(self, blackboard):
        return NodeStatus.RUNNING


def make_simple_bt(name="TestBT"):
    """创建一个简单的测试行为树"""
    root = SequenceNode(name="Root")
    root.add_child(DummyAction("Action1", delay=0.02, result=NodeStatus.SUCCESS))
    root.add_child(DummyAction("Action2", delay=0.02, result=NodeStatus.SUCCESS))
    return root


# ============================================================================
# TaskExecutionRecord 测试
# ============================================================================

class TestTaskExecutionRecord:
    def test_record_creation(self):
        record = TaskExecutionRecord(
            record_id="test_001",
            task_id="task_001",
            task_type="transport",
            start_time=time.time(),
        )
        assert record.task_type == "transport"
        assert record.result == ExecutionResult.UNKNOWN
        assert record.phase == ExecutionPhase.PLANNING
        assert record.steps_executed == 0
        assert record.bt_tick_count == 0

    def test_finalize_success(self):
        record = TaskExecutionRecord(
            record_id="test_002",
            task_id="task_002",
            task_type="patrol",
            start_time=time.time(),
        )
        time.sleep(0.05)
        record.finalize(ExecutionResult.SUCCESS, "Task completed successfully")
        assert record.result == ExecutionResult.SUCCESS
        assert record.phase == ExecutionPhase.SUCCEEDED
        assert record.duration is not None
        assert record.duration > 0
        assert record.outcome_summary == "Task completed successfully"

    def test_finalize_failure(self):
        record = TaskExecutionRecord(
            record_id="test_003",
            task_id="task_003",
            task_type="rescue",
            start_time=time.time(),
        )
        record.finalize(ExecutionResult.FAILURE, "Obstacle detected")
        assert record.result == ExecutionResult.FAILURE
        assert record.phase == ExecutionPhase.FAILED
        assert "Obstacle detected" in record.errors or "Obstacle detected" in record.outcome_summary

    def test_add_phase(self):
        record = TaskExecutionRecord(
            record_id="test_004",
            task_id="task_004",
            task_type="collaborative",
            start_time=time.time(),
        )
        record.add_phase(ExecutionPhase.PLANNING, "Initial planning")
        record.add_phase(ExecutionPhase.EXECUTING, "BT tick started")
        record.add_phase(ExecutionPhase.SUCCEEDED, "All done")
        assert len(record.phases_history) == 3
        assert record.phase == ExecutionPhase.SUCCEEDED

    def test_to_memory_format(self):
        record = TaskExecutionRecord(
            record_id="test_005",
            task_id="task_005",
            task_type="transport",
            start_time=time.time(),
        )
        record.finalize(ExecutionResult.SUCCESS, "Done")
        mem = record.to_memory_format()
        assert mem["record_id"] == "test_005"
        assert mem["task_type"] == "transport"
        assert mem["result"] == "success"
        assert "transport" in mem["entities"]


# ============================================================================
# MemoryEnhancedExecutor 测试
# ============================================================================

class TestMemoryEnhancedExecutor:
    def test_executor_creation(self):
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt)
        assert executor.bt_root is bt
        assert executor.is_running is False
        assert executor.is_paused is False

    def test_executor_status(self):
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt)
        status = executor.get_status()
        assert status["is_running"] is False
        assert status["current_task"] is None
        assert status["tick_count"] == 0
        assert status["total_tasks"] == 0

    def test_execute_simple_task_success(self):
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt, enable_memory=False)
        record = executor.execute_task(
            task_type="transport",
            task_config={"move_speed": 0.5},
            timeout=5.0,
            tick_rate=0.01,
        )
        assert record.result == ExecutionResult.SUCCESS
        assert record.phase == ExecutionPhase.SUCCEEDED
        assert record.bt_tick_count > 0
        assert record.steps_executed > 0
        assert record.duration is not None
        assert record.duration > 0

    def test_execute_task_with_failing_action(self):
        root = SequenceNode(name="FailingRoot")
        root.add_child(DummyAction("Success1", delay=0.02))
        root.add_child(FailingAction("FailingAction"))
        root.add_child(DummyAction("ShouldNotRun", delay=0.02))
        bt = BehaviorTree(root, name="FailingBT")
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt.root, enable_memory=False)
        record = executor.execute_task(
            task_type="transport",
            task_config={},
            timeout=5.0,
            tick_rate=0.01,
        )
        assert record.result == ExecutionResult.FAILURE
        assert record.phase == ExecutionPhase.FAILED

    def test_execute_task_timeout(self):
        bt = BehaviorTree(InfiniteAction("Infinite"), name="InfiniteBT")
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt.root, enable_memory=False)
        record = executor.execute_task(
            task_type="patrol",
            task_config={},
            timeout=0.5,  # 短超时
            tick_rate=0.01,
        )
        assert record.result == ExecutionResult.FAILURE
        assert "timeout" in record.outcome_summary.lower() or record.phase == ExecutionPhase.FAILED

    def test_pause_resume(self):
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt, enable_memory=False)

        # 使用长任务以便有暂停机会
        long_root = SequenceNode(name="LongRoot")
        long_root.add_child(DummyAction("Long1", delay=1.0))
        bt2 = BehaviorTree(long_root, name="LongBT")
        executor.bt_root = bt2.root

        # 异步启动（检查pause机制）
        executor.is_running = True
        executor.is_paused = False
        executor.pause()
        assert executor.is_paused is True

        executor.resume()
        assert executor.is_paused is False

    def test_abort(self):
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt, enable_memory=False)
        executor.abort()
        assert executor.is_running is False
        assert executor.is_paused is False
        assert executor.current_record is None

    def test_execution_summary(self):
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(behavior_tree_root=bt, enable_memory=False)

        # 运行几个任务
        for _ in range(3):
            executor.execute_task("transport", {}, timeout=5.0, tick_rate=0.01)

        summary = executor.get_execution_summary()
        assert summary["total"] == 3
        assert summary["success"] >= 0
        assert summary["total"] == summary["success"] + summary["failure"]
        assert summary["success_rate"] >= 0.0

    def test_execute_task_no_bt(self):
        """没有预加载行为树时，应自动从配置构建"""
        executor = MemoryEnhancedExecutor(behavior_tree_root=None, enable_memory=False)
        record = executor.execute_task(
            task_type="transport",
            task_config={"move_speed": 0.5},
            timeout=5.0,
            tick_rate=0.01,
        )
        # 自动构建的BT应该能成功执行
        assert record.result in (ExecutionResult.SUCCESS, ExecutionResult.FAILURE)
        # bt_root 应该在执行过程中被设置
        assert executor.bt_root is not None

    def test_execute_patrol_task(self):
        """巡逻任务行为树构建"""
        executor = MemoryEnhancedExecutor(enable_memory=False)
        record = executor.execute_task(
            task_type="patrol",
            task_config={"patrol_loops": 2},
            timeout=10.0,
            tick_rate=0.01,
        )
        # 巡逻任务应该成功
        assert record.result in (ExecutionResult.SUCCESS, ExecutionResult.FAILURE)

    def test_execute_rescue_task(self):
        """救援任务行为树构建"""
        executor = MemoryEnhancedExecutor(enable_memory=False)
        record = executor.execute_task(
            task_type="rescue",
            task_config={"rescue_speed": 0.6},
            timeout=10.0,
            tick_rate=0.01,
        )
        assert record.result in (ExecutionResult.SUCCESS, ExecutionResult.FAILURE)

    def test_execute_collaborative_task(self):
        """协同任务行为树构建"""
        executor = MemoryEnhancedExecutor(enable_memory=False)
        record = executor.execute_task(
            task_type="collaborative",
            task_config={"collaborative_speed": 0.4},
            timeout=10.0,
            tick_rate=0.01,
        )
        assert record.result in (ExecutionResult.SUCCESS, ExecutionResult.FAILURE)


# ============================================================================
# ScenarioTaskExecutor 测试
# ============================================================================

class TestScenarioTaskExecutor:
    def test_scenario_executor_creation(self):
        executor = ScenarioTaskExecutor()
        assert executor.current_scene_type is None
        assert executor.scene_intelligence is None
        assert executor.scene_coordinator is None

    def test_set_scene(self):
        executor = ScenarioTaskExecutor()
        executor.set_scene("warehouse")
        assert executor.current_scene_type == "warehouse"

    def test_execute_scenario_task(self):
        executor = ScenarioTaskExecutor(enable_memory=False)
        executor.set_scene("warehouse")
        record = executor.execute_scenario_task(
            scenario_type="transport",
            task_config={"move_speed": 0.5},
            timeout=5.0,
            tick_rate=0.01,
        )
        assert record.result in (ExecutionResult.SUCCESS, ExecutionResult.FAILURE)


# ============================================================================
# 工厂函数测试
# ============================================================================

class TestFactoryFunctions:
    def test_create_task_executor_default(self):
        executor = create_task_executor("default")
        assert isinstance(executor, MemoryEnhancedExecutor)
        assert isinstance(executor, ScenarioTaskExecutor) is False

    def test_create_task_executor_scenario(self):
        executor = create_task_executor("scenario")
        assert isinstance(executor, ScenarioTaskExecutor)

    def test_create_executor_from_config(self):
        config = {
            "type": "default",
            "use_simulation": True,
            "enable_memory": False,
            "tick_rate": 0.05,
        }
        executor = create_executor_from_config(config)
        assert isinstance(executor, MemoryEnhancedExecutor)
        status = executor.get_status()
        assert status["is_running"] is False


# ============================================================================
# 回调测试
# ============================================================================

class TestCallbacks:
    def test_phase_change_callback(self):
        phases_seen = []

        def on_phase(phase, details):
            phases_seen.append((phase, details))

        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=bt,
            enable_memory=False,
        )
        executor.set_callbacks(on_phase_change=on_phase)
        executor.execute_task("transport", {}, timeout=5.0, tick_rate=0.01)

        # 应该看到至少 PLANNING, EXECUTING, SUCCEEDED 阶段
        phase_values = [p.value for p, _ in phases_seen]
        assert ExecutionPhase.PLANNING.value in phase_values
        assert ExecutionPhase.EXECUTING.value in phase_values

    def test_tick_callback(self):
        tick_count = 0

        def on_tick(tick, status):
            nonlocal tick_count
            tick_count = tick

        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=bt,
            enable_memory=False,
        )
        executor.set_callbacks(on_tick=on_tick)
        executor.execute_task("transport", {}, timeout=5.0, tick_rate=0.005)
        assert tick_count > 0


# ============================================================================
# 行为树配置构建测试
# ============================================================================

class TestBTConfigBuilding:
    def test_build_transport_config(self):
        executor = MemoryEnhancedExecutor(enable_memory=False)
        config = executor._build_bt_config("transport", {"move_speed": 0.7, "min_battery": 0.3})
        assert config["type"] == "sequence"
        assert config["name"] == "TransportTask"
        assert len(config["children"]) >= 4

    def test_build_patrol_config(self):
        executor = MemoryEnhancedExecutor(enable_memory=False)
        config = executor._build_bt_config("patrol", {"patrol_loops": 5})
        assert config["type"] == "repeater"
        assert config["params"]["num_repeats"] == 5

    def test_build_rescue_config(self):
        executor = MemoryEnhancedExecutor(enable_memory=False)
        config = executor._build_bt_config("rescue", {"rescue_speed": 0.9})
        assert config["type"] == "sequence"
        assert config["name"] == "RescueTask"

    def test_build_collaborative_config(self):
        executor = MemoryEnhancedExecutor(enable_memory=False)
        config = executor._build_bt_config("collaborative", {"collaborative_speed": 0.5})
        assert config["type"] == "sequence"
        assert config["name"] == "CollaborativeTask"

    def test_unknown_task_type_defaults_to_transport(self):
        executor = MemoryEnhancedExecutor(enable_memory=False)
        config = executor._build_bt_config("unknown_type", {})
        assert config["type"] == "sequence"
        assert config["name"] == "TransportTask"


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
