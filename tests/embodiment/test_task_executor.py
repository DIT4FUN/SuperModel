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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
