"""
test_pipeline_diagnostics.py - Pipeline 诊断与错误恢复测试
=====================================================

测试 SuperModel 具身智能 Pipeline 的诊断和错误恢复功能:
- ErrorRecoveryPolicy 枚举
- DiagnosticCollector 诊断收集器
- EmbodiedPipeline.get_diagnostics()
- EmbodiedPipeline.get_error_recovery_suggestions()
- EmbodiedPipeline.attempt_auto_recovery()
"""

import time
import pytest

from src.embodied.embodied_pipeline import (
    PipelineMode,
    PipelineState,
    PipelineConfig,
    TaskRequest,
    TaskResult,
    EmbodiedPipeline,
    ErrorRecoveryPolicy,
    DiagnosticCollector,
)


# ============================================================
# DiagnosticCollector 测试
# ============================================================

class TestDiagnosticCollector:
    """DiagnosticCollector 单元测试"""

    def test_collector_creation(self):
        """测试诊断收集器创建"""
        collector = DiagnosticCollector()
        assert collector._max_history == 10000
        assert len(collector._snapshots) == 0
        assert len(collector._error_log) == 0

    def test_record_tick(self):
        """测试快照记录"""
        collector = DiagnosticCollector(max_history=100)
        collector.record_tick('RUNNING', 2, 5, {'cpu': 45.0, 'memory': 120.0})
        collector.record_tick('RUNNING', 3, 4, {'cpu': 50.0, 'memory': 125.0})
        assert len(collector._snapshots) == 2
        assert collector._snapshots[0]['pipeline_state'] == 'RUNNING'
        assert collector._snapshots[0]['metrics']['cpu'] == 45.0

    def test_record_error(self):
        """测试错误记录"""
        collector = DiagnosticCollector()
        collector.record_error('EMBODIED_BT_001', '行为树执行超时', {'task_id': 'test123'})
        assert len(collector._error_log) == 1
        assert collector._error_log[0]['error_code'] == 'EMBODIED_BT_001'
        assert collector._error_log[0]['context']['task_id'] == 'test123'

    def test_generate_report(self):
        """测试报告生成"""
        collector = DiagnosticCollector()
        collector.record_tick('RUNNING', 1, 0, {'cpu': 40.0})
        collector.record_tick('RUNNING', 2, 0, {'cpu': 60.0})
        collector.record_error('TEST_ERROR', 'Test error message')
        report = collector.generate_report()
        assert 'report_time' in report
        assert report['snapshots_collected'] == 2
        assert report['errors_logged'] == 1
        assert 'cpu' in report['metric_summaries']
        assert report['state_distribution']['RUNNING'] == 2
        assert len(report['recent_errors']) == 1

    def test_metric_trend(self):
        """测试指标趋势分析"""
        collector = DiagnosticCollector()
        for i in range(10):
            collector.record_tick('RUNNING', i, 0, {'queue_size': float(i * 2)})
        trend = collector.get_metric_trend('queue_size', window_size=10)
        assert trend is not None
        assert trend['first'] == 0.0
        assert trend['last'] == 18.0
        assert trend['delta'] == 18.0
        assert trend['slope'] == 2.0

    def test_metric_trend_insufficient_data(self):
        """测试数据不足时返回None"""
        collector = DiagnosticCollector()
        collector.record_tick('RUNNING', 1, 0, {'cpu': 50.0})
        trend = collector.get_metric_trend('cpu')
        assert trend is None

    def test_export(self, tmp_path):
        """测试JSON导出"""
        import json
        collector = DiagnosticCollector()
        collector.record_tick('RUNNING', 1, 0, {'cpu': 50.0})
        collector.record_error('TEST', 'Test')
        path = tmp_path / 'diagnostics.json'
        ok = collector.export(str(path))
        assert ok is True
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data['snapshots_collected'] == 1


# ============================================================
# Pipeline 错误恢复建议测试
# ============================================================

class TestErrorRecoverySuggestions:
    """Pipeline 错误恢复建议测试"""

    def test_suggestions_in_normal_state(self):
        """正常状态下返回空或低优先级建议"""
        p = EmbodiedPipeline()
        p.start()
        suggestions = p.get_error_recovery_suggestions()
        # 正常状态不应有高优先级建议
        high_priority = [s for s in suggestions if s['priority'] <= 2]
        assert len(high_priority) == 0
        p.stop()

    def test_suggestions_in_error_state(self):
        """ERROR状态产生高优先级建议"""
        p = EmbodiedPipeline()
        p.start()
        # 手动触发ERROR状态
        p._state = PipelineState.ERROR
        p._error_message = "Simulated failure"
        suggestions = p.get_error_recovery_suggestions()
        assert len(suggestions) > 0
        assert suggestions[0]['priority'] == 1
        assert 'reset_health' in suggestions[0]['action']
        p.stop()

    def test_suggestions_with_queue_overflow(self):
        """任务队列积压产生建议"""
        p = EmbodiedPipeline()
        p.start()
        # 模拟队列积压
        for i in range(15):
            req = TaskRequest(
                task_id=f'overflow_{i}',
                task_type='transport',
                target='test',
                priority=2,
            )
            p._task_queue.append(req)
        suggestions = p.get_error_recovery_suggestions()
        queue_suggestions = [s for s in suggestions if 'queue' in s['action'] or 'worker' in s['action']]
        assert len(queue_suggestions) > 0
        p.stop()


# ============================================================
# Pipeline 自动恢复测试
# ============================================================

class TestAutoRecovery:
    """Pipeline 自动恢复测试"""

    def test_auto_recovery_reset_health(self):
        """测试ERROR状态自动恢复"""
        p = EmbodiedPipeline()
        p.start()
        p._state = PipelineState.ERROR
        p._error_message = "Test error"
        result = p.attempt_auto_recovery()
        assert 'recovered' in result
        assert 'attempts' in result
        assert 'final_state' in result
        # reset_health 应该成功
        reset_attempts = [a for a in result['attempts'] if a['strategy'] == 'reset_health']
        assert len(reset_attempts) == 1
        p.stop()

    def test_auto_recovery_clear_queue(self):
        """测试清空队列恢复"""
        p = EmbodiedPipeline()
        p.start()
        # 模拟队列积压
        for i in range(10):
            req = TaskRequest(
                task_id=f'clear_{i}',
                task_type='transport',
                target='test',
                priority=2,
            )
            p._task_queue.append(req)
        result = p.attempt_auto_recovery()
        clear_attempts = [a for a in result['attempts'] if a['strategy'] == 'clear_queue']
        assert len(clear_attempts) == 1
        assert clear_attempts[0]['success'] is True
        assert '已清空' in clear_attempts[0]['message']
        p.stop()


# ============================================================
# Pipeline 完整诊断报告测试
# ============================================================

class TestDiagnostics:
    """Pipeline 完整诊断报告测试"""

    def test_get_diagnostics_structure(self):
        """测试诊断报告结构完整性"""
        p = EmbodiedPipeline()
        p.start()
        report = p.get_diagnostics()
        # 验证顶层键
        assert 'generated_at' in report
        assert 'pipeline' in report
        assert 'health' in report
        assert 'performance' in report
        assert 'modules' in report
        assert 'tasks' in report
        assert 'recovery_suggestions' in report
        # 验证pipeline子节
        assert 'version' in report['pipeline']
        assert 'state' in report['pipeline']
        assert 'grade' in report['pipeline']
        assert 'mode' in report['pipeline']
        assert 'scene_type' in report['pipeline']
        p.stop()

    def test_get_diagnostics_modules(self):
        """测试模块状态诊断"""
        p = EmbodiedPipeline()
        p.start()
        report = p.get_diagnostics()
        modules = report['modules']
        # 基础模块
        assert 'behavior_tree' in modules
        assert 'scene_intelligence' in modules
        assert 'simulation' in modules
        # 验证模块状态格式
        for name, status in modules.items():
            assert 'status' in status
            assert 'available' in status
        p.stop()

    def test_get_diagnostics_with_tasks(self):
        """测试带任务历史的诊断"""
        p = EmbodiedPipeline()
        p.start()
        # 执行几个任务
        for i in range(5):
            result = p.execute_task('transport', target=f'station_{i}')
            assert result is not None
        report = p.get_diagnostics()
        assert report['tasks']['completed'] == 5
        assert 'total_completed' in report['performance']
        p.stop()

    def test_get_diagnostics_success_rate(self):
        """测试成功率计算"""
        p = EmbodiedPipeline()
        p.start()
        # 模拟完成一些任务
        for i in range(10):
            req = TaskRequest(
                task_id=f'diag_{i}',
                task_type='transport',
                target='test',
                priority=2,
            )
            p._completed_tasks.append(
                TaskResult(
                    task_id=f'diag_{i}',
                    success=(i % 2 == 0),
                    phase='completed',
                    duration_ms=100.0 * (i + 1),
                )
            )
        report = p.get_diagnostics()
        assert report['health']['success_rate_100'] == 0.5
        p.stop()

    def test_get_diagnostics_percentiles(self):
        """测试性能百分位数"""
        p = EmbodiedPipeline()
        p.start()
        # 填充一些任务
        for i in range(200):
            req = TaskRequest(
                task_id=f'perc_{i}',
                task_type='transport',
                target='test',
                priority=2,
            )
            p._completed_tasks.append(
                TaskResult(
                    task_id=f'perc_{i}',
                    success=True,
                    phase='completed',
                    duration_ms=50.0 + i * 2.0,
                )
            )
        report = p.get_diagnostics()
        perf = report['performance']
        assert perf['p50_duration_ms'] > 0
        assert perf['p95_duration_ms'] > perf['p50_duration_ms']
        assert perf['p99_duration_ms'] > perf['p95_duration_ms']
        p.stop()


# ============================================================
# ErrorRecoveryPolicy 测试
# ============================================================

class TestErrorRecoveryPolicy:
    """错误恢复策略枚举测试"""

    def test_policy_values(self):
        """测试策略枚举值"""
        assert ErrorRecoveryPolicy.MANUAL.value == "manual"
        assert ErrorRecoveryPolicy.RETRY.value == "retry"
        assert ErrorRecoveryPolicy.FALLBACK.value == "fallback"
        assert ErrorRecoveryPolicy.RESTART_MODULE.value == "restart_module"
        assert ErrorRecoveryPolicy.FULL_RESET.value == "full_reset"

    def test_policy_count(self):
        """测试策略数量"""
        assert len(ErrorRecoveryPolicy) == 5


# ============================================================
# 集成测试
# ============================================================

class TestDiagnosticsIntegration:
    """诊断功能集成测试"""

    def test_diagnostics_after_normal_operation(self):
        """正常运行后的诊断报告"""
        p = EmbodiedPipeline(grade='L', scene_type='FACTORY')
        p.start()
        # 执行任务
        for i in range(3):
            p.execute_task('transport', target=f'ws_{i}')
        # 获取诊断
        diag = p.get_diagnostics()
        assert diag['pipeline']['grade'] == 'L'
        assert diag['pipeline']['scene_type'] == 'FACTORY'
        assert diag['pipeline']['state'] in ('ready', 'running', 'READY', 'RUNNING')
        p.stop()

    def test_recovery_suggestions_after_failed_task(self):
        """任务失败后的恢复建议"""
        p = EmbodiedPipeline()
        p.start()
        # 模拟任务失败
        req = TaskRequest(
            task_id='failed_001',
            task_type='transport',
            target='test',
            priority=2,
        )
        p._completed_tasks.append(
            TaskResult(
                task_id='failed_001',
                success=False,
                phase='execution',
                duration_ms=0.0,
                error='Simulated failure',
            )
        )
        # 添加足够多的失败
        for i in range(50):
            req2 = TaskRequest(
                task_id=f'fail_{i}',
                task_type='transport',
                target='test',
                priority=2,
            )
            p._completed_tasks.append(
                TaskResult(
                    task_id=f'fail_{i}',
                    success=(i < 15),
                    phase='completed',
                    duration_ms=100.0,
                )
            )
        suggestions = p.get_error_recovery_suggestions()
        # 应该有关于高失败率的建议
        analysis = [s for s in suggestions if 'analyze' in s['action'] or 'failure' in s['action'].lower()]
        assert len(analysis) > 0
        p.stop()
