"""
test_task_executor.py - 任务执行器及性能分析器测试
===================================================

测试 TaskPerformanceProfiler, MemoryEnhancedExecutor,
ScenarioTaskExecutor 的功能。
"""

import pytest
import time


class TestTaskPerformanceProfiler:
    """TaskPerformanceProfiler 性能分析器测试"""

    def test_profiler_initialization(self):
        """测试性能分析器初始化"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler(enabled=True)
        assert profiler.enabled is True
        assert profiler._current_task_id is None

    def test_start_end_task(self):
        """测试任务开始和结束追踪"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_001", "navigation")
        assert profiler._current_task_id == "task_001"
        assert profiler._task_start_time is not None
        time.sleep(0.05)
        report = profiler.end_task()
        assert report['task_id'] == "task_001"
        assert report['total_time_s'] >= 0.04

    def test_phase_timing(self):
        """测试执行阶段计时"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_002", "grasp")
        profiler.start_phase('planning')
        time.sleep(0.02)
        profiler.end_phase('planning')
        profiler.start_phase('execution')
        time.sleep(0.02)
        profiler.end_phase('execution')
        report = profiler.end_task()
        assert 'planning' in report['phases']
        assert 'execution' in report['phases']
        assert report['phases']['planning'] >= 0.015
        assert report['phases']['execution'] >= 0.015

    def test_record_node_time(self):
        """测试节点执行时间记录"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_003", "navigation")
        profiler.record_node_time("MoveTo", 0.005)
        profiler.record_node_time("MoveTo", 0.007)
        profiler.record_node_time("CheckSafe", 0.001)
        report = profiler.end_task()
        assert "MoveTo" in report['node_times']
        assert report['node_times']['MoveTo']['count'] == 2
        assert report['node_times']['MoveTo']['avg_ms'] > 0

    def test_record_tick_time(self):
        """测试 tick 时间记录"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_004", "transport")
        for _ in range(10):
            profiler.record_tick_time(0.01)
        report = profiler.end_task()
        assert report['tick_stats']['count'] == 10
        assert report['tick_stats']['avg_ms'] > 0

    def test_record_sensor_time(self):
        """测试传感器处理时间记录"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_005", "inspection")
        profiler.record_sensor_time(0.002)
        profiler.record_sensor_time(0.003)
        profiler.record_sensor_time(0.001)
        report = profiler.end_task()
        assert report['sensor_stats']['count'] == 3
        assert report['sensor_stats']['avg_ms'] > 0

    def test_record_memory_time(self):
        """测试记忆检索时间记录"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_006", "replan")
        profiler.record_memory_time(0.015)
        profiler.record_memory_time(0.020)
        report = profiler.end_task()
        assert report['memory_stats']['count'] == 2
        assert report['memory_stats']['avg_ms'] >= 15

    def test_record_action_time(self):
        """测试动作执行时间记录"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_007", "grasp")
        profiler.record_action_time(0.050)
        report = profiler.end_task()
        assert report['action_stats']['count'] == 1
        assert report['action_stats']['avg_ms'] >= 50

    def test_percentile_stats(self):
        """测试 P50/P95/P99 统计"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_008", "inspect")
        for i in range(20):
            profiler.record_tick_time(0.005 + i * 0.001)  # 递增时间
        report = profiler.end_task()
        assert report['tick_stats']['p50_ms'] > 0
        assert report['tick_stats']['p95_ms'] > report['tick_stats']['p50_ms']
        assert report['tick_stats']['p99_ms'] >= report['tick_stats']['p95_ms']

    def test_get_realtime_report(self):
        """测试实时性能报告"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler()
        profiler.start_task("task_009", "move")
        profiler.start_phase('execution')
        # 记录一些 tick 数据以测试 percentile 输出
        for _ in range(5):
            profiler.record_tick_time(0.01)
        time.sleep(0.03)
        rtreport = profiler.get_realtime_report()
        assert rtreport['current_task'] == "task_009"
        assert rtreport['elapsed_s'] >= 0.02
        assert 'tick_count' in rtreport
        assert rtreport['tick_count'] == 5
        assert 'tick_p50_ms' in rtreport
        profiler.end_task()

    def test_disabled_profiler(self):
        """测试禁用状态的分析器"""
        from src.embodied.task_executor import TaskPerformanceProfiler
        profiler = TaskPerformanceProfiler(enabled=False)
        profiler.start_task("task_010", "test")
        profiler.record_tick_time(0.1)
        profiler.record_node_time("TestNode", 0.2)
        report = profiler.end_task()
        assert report.get('tick_stats', {}).get('count', 0) == 0
        assert "TestNode" not in report.get('node_times', {})


class TestExecutionPhaseResult:
    """ExecutionPhase 和 ExecutionResult 枚举测试"""

    def test_execution_phases(self):
        """测试执行阶段枚举"""
        from src.embodied.task_executor import ExecutionPhase
        assert ExecutionPhase.PLANNING.value == "planning"
        assert ExecutionPhase.EXECUTING.value == "executing"
        assert ExecutionPhase.MONITORING.value == "monitoring"
        assert ExecutionPhase.SUCCEEDED.value == "succeeded"
        assert ExecutionPhase.FAILED.value == "failed"

    def test_execution_results(self):
        """测试执行结果枚举"""
        from src.embodied.task_executor import ExecutionResult
        assert ExecutionResult.SUCCESS.value == "success"
        assert ExecutionResult.FAILURE.value == "failure"
        assert ExecutionResult.RUNNING.value == "running"
        assert ExecutionResult.ABORTED.value == "aborted"


class TestTaskExecutionRecord:
    """TaskExecutionRecord 记录测试"""

    def test_record_finalize(self):
        """测试记录完成"""
        from src.embodied.task_executor import TaskExecutionRecord, ExecutionResult
        record = TaskExecutionRecord(
            record_id="rec_001",
            task_id="task_001",
            task_type="navigation",
            start_time=time.time() - 10.0,
        )
        record.finalize(ExecutionResult.SUCCESS, "Task completed")
        assert record.result == ExecutionResult.SUCCESS
        assert record.duration is not None
        assert record.duration >= 10.0
        assert record.outcome_summary == "Task completed"

    def test_record_add_phase(self):
        """测试添加执行阶段"""
        from src.embodied.task_executor import TaskExecutionRecord, ExecutionPhase
        record = TaskExecutionRecord(
            record_id="rec_002",
            task_id="task_002",
            task_type="grasp",
            start_time=time.time(),
        )
        record.add_phase(ExecutionPhase.PLANNING, "creating plan")
        record.add_phase(ExecutionPhase.EXECUTING, "executing")
        assert len(record.phases_history) == 2
        assert record.phase == ExecutionPhase.EXECUTING

    def test_record_add_error(self):
        """测试记录错误"""
        from src.embodied.task_executor import TaskExecutionRecord
        record = TaskExecutionRecord(
            record_id="rec_003",
            task_id="task_003",
            task_type="move",
            start_time=time.time(),
        )
        record.add_error("Collision detected")
        assert len(record.errors) == 1
        assert "Collision" in record.errors[0]

    def test_record_to_memory_format(self):
        """测试转换为记忆格式"""
        from src.embodied.task_executor import TaskExecutionRecord, ExecutionResult
        record = TaskExecutionRecord(
            record_id="rec_004",
            task_id="task_004",
            task_type="transport",
            start_time=time.time(),
        )
        record.finalize(ExecutionResult.SUCCESS, "OK")
        mem = record.to_memory_format()
        assert mem['record_id'] == "rec_004"
        assert mem['task_type'] == "transport"
        assert mem['result'] == "success"


# ============================================================
# apply_learned_adjustments 测试
# ============================================================

class TestApplyLearnedAdjustments:
    """测试基于历史经验的参数学习调整"""

    def _make_executor_with_mock_memory(self, experiences):
        """创建带有模拟记忆系统的执行器"""
        from src.embodied.task_executor import MemoryEnhancedExecutor
        
        class MockMemory:
            def retrieve(self, query, limit=5):
                return experiences
            def search(self, query, top_k=5):
                return experiences
        
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=None,
            memory_system=MockMemory(),
            enable_memory=True,
        )
        return executor

    def test_no_experiences_returns_unchanged(self):
        """测试无经验时返回原始配置"""
        executor = self._make_executor_with_mock_memory([])
        
        config = {'move_speed': 0.5, 'safety_margin': 1.0}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert adjusted == config

    def test_success_experience_speed_boost(self):
        """测试成功经验提升速度"""
        experiences = [
            {
                'result': 'success',
                'performance_metrics': {'optimal_speed': 0.8},
            },
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert 'move_speed' in adjusted
        assert adjusted['move_speed'] >= 0.5  # 不低于原值

    def test_failure_experience_decreases_speed(self):
        """测试失败经验降低速度"""
        experiences = [
            {
                'result': 'failure',
                'outcome_summary': 'Collision during transport',
            },
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5, 'safety_margin': 1.0}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert adjusted['move_speed'] < 0.5  # 速度降低
        assert adjusted['safety_margin'] > 1.0  # 安全裕量增加

    def test_mixed_experiences_balanced(self):
        """测试混合经验时平衡调整"""
        experiences = [
            {'result': 'success', 'performance_metrics': {}},
            {'result': 'success', 'performance_metrics': {}},
            {'result': 'failure', 'outcome_summary': 'Failed'},
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5, 'position_threshold': 0.1}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert '_learned_from' in adjusted
        assert adjusted['_learned_from']['success_count'] == 2
        assert adjusted['_learned_from']['failure_count'] == 1

    def test_all_failures_conservative(self):
        """测试全部失败经验时采用保守策略"""
        experiences = [
            {'result': 'failure', 'outcome_summary': 'Failed 1'},
            {'result': 'failure', 'outcome_summary': 'Failed 2'},
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        # 0.5 * 0.9^2 (两次失败) * 0.7 (全失败保守策略)
        assert adjusted['move_speed'] == 0.5 * 0.9 * 0.9 * 0.7  # 大幅降速
        # 安全裕量: 1.0 * 1.1^2 = 1.21 (两次失败累积, setdefault不会覆盖已有值)
        assert adjusted['safety_margin'] >= 1.2  # 安全裕量已有值
        assert 'timeout_factor' in adjusted  # 双倍超时

    def test_known_failure_patterns_recorded(self):
        """测试失败模式被记录"""
        experiences = [
            {'result': 'failure', 'outcome_summary': 'Obstacle collision'},
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert 'known_failure_patterns' in adjusted
        assert 'Obstacle collision' in adjusted['known_failure_patterns']

    def test_learned_metadata_attached(self):
        """测试学习元数据被附加到配置"""
        experiences = [
            {'result': 'success', 'performance_metrics': {}},
            {'result': 'success', 'performance_metrics': {}},
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert '_learned_from' in adjusted
        assert adjusted['_learned_from']['total_experiences'] == 2

    def test_safety_margin_cap(self):
        """测试安全裕量有上限"""
        experiences = [
            {'result': 'failure', 'outcome_summary': 'Failed'},
            {'result': 'failure', 'outcome_summary': 'Failed'},
            {'result': 'failure', 'outcome_summary': 'Failed'},
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        config = {'move_speed': 0.5, 'safety_margin': 1.0}
        adjusted = executor.apply_learned_adjustments(config, 'transport')
        
        assert adjusted['safety_margin'] <= 2.0  # 不超过 2.0


class TestRetrieveRelevantExperience:
    """测试从记忆中检索相关经验"""

    def _make_executor_with_mock_memory(self, experiences):
        from src.embodied.task_executor import MemoryEnhancedExecutor
        
        class MockMemory:
            def retrieve(self, query, limit=5):
                return experiences
            def search(self, query, top_k=5):
                return experiences
        
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=None,
            memory_system=MockMemory(),
            enable_memory=True,
        )
        return executor

    def test_retrieve_experience_success(self):
        """测试成功检索经验"""
        experiences = [
            {'task_type': 'transport', 'result': 'success'},
        ]
        executor = self._make_executor_with_mock_memory(experiences)
        
        result = executor.retrieve_relevant_experience('transport', limit=5)
        
        assert len(result) == 1
        assert result[0]['result'] == 'success'

    def test_retrieve_experience_empty(self):
        """测试无经验时返回空列表"""
        executor = self._make_executor_with_mock_memory([])
        
        result = executor.retrieve_relevant_experience('unknown_task', limit=5)
        
        assert result == []

    def test_memory_disabled_returns_empty(self):
        """测试禁用记忆时返回空列表"""
        from src.embodied.task_executor import MemoryEnhancedExecutor
        
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=None,
            memory_system=None,
            enable_memory=False,
        )
        
        result = executor.retrieve_relevant_experience('transport', limit=5)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
