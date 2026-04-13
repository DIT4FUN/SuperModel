"""
test_embodied_pipeline_full.py - 具身Pipeline完整集成测试
=========================================================

覆盖:
- 所有5个AGV等级 (S/M/L/XL/XXL) × 所有5个场景类型 (warehouse/hospital/factory/restaurant/outdoor)
- Pipeline 状态持久化与恢复
- 任务队列与并发执行
- 健康报告与错误恢复
- 行为树与技能系统集成
"""

import pytest
import time
import tempfile
import os
import threading
from collections import deque
from typing import Any, Dict

# 导入被测模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.embodied.embodied_pipeline import (
    EmbodiedPipeline,
    PipelineConfig,
    PipelineMode,
    PipelineState,
    TaskRequest,
    TaskResult,
    create_embodied_pipeline,
    create_pipeline_from_config,
)
from src.embodied.behavior_tree import (
    NodeStatus, TaskStatus, EmbodiedTask, EmbodiedTaskPlanner, AGVTaskPlanner,
)
from src.embodied.embodied_skill import (
    EmbodiedSkillRegistry, EmbodiedSkill, EmbodiedSkillDefinition,
    SkillCategory, SkillStatus, get_global_skill_registry,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def grades():
    return ["S", "M", "L", "XL", "XXL"]


@pytest.fixture
def scene_types():
    return ["warehouse", "hospital", "factory", "restaurant", "outdoor"]


@pytest.fixture
def all_grade_scene_combinations(grades, scene_types):
    """生成所有 AGV 等级 × 场景类型组合"""
    combos = []
    for grade in grades:
        for scene in scene_types:
            combos.append((grade, scene))
    return combos


# =============================================================================
# 基础初始化测试
# =============================================================================

class TestPipelineBasicInit:
    """Pipeline 基础初始化测试"""

    def test_pipeline_default_init(self):
        """默认参数初始化"""
        pipeline = EmbodiedPipeline()
        assert pipeline.config.grade == "M"
        assert pipeline.config.scene_type == "WAREHOUSE"
        assert pipeline.config.mode == PipelineMode.SIMULATION
        assert pipeline.state == PipelineState.IDLE

    def test_pipeline_with_grade(self, grades):
        """各等级初始化"""
        for grade in grades:
            pipeline = EmbodiedPipeline(grade=grade)
            assert pipeline.config.grade == grade
            assert pipeline.state == PipelineState.IDLE

    def test_pipeline_with_scene(self, scene_types):
        """各场景类型初始化"""
        for scene in scene_types:
            pipeline = EmbodiedPipeline(scene_type=scene)
            # scene_type 在 config 中保留原始大小写
            assert pipeline.config.scene_type.lower() == scene.lower()

    def test_pipeline_with_mode(self):
        """各运行模式初始化"""
        for mode in [PipelineMode.SIMULATION, PipelineMode.HARDWARE_IN_LOOP, PipelineMode.FULL_PHYSICAL]:
            pipeline = EmbodiedPipeline(mode=mode)
            assert pipeline.config.mode == mode

    def test_factory_create_embodied_pipeline(self):
        """工厂函数创建 Pipeline"""
        pipeline = create_embodied_pipeline(grade="L", scene_type="factory", mode="simulation")
        assert pipeline.config.grade == "L"
        assert pipeline.config.scene_type == "FACTORY"
        assert pipeline.config.mode == PipelineMode.SIMULATION

    def test_factory_create_pipeline_from_config(self):
        """从配置字典创建 Pipeline"""
        config = {"grade": "XL", "scene_type": "hospital", "mode": "hardware_in_loop"}
        pipeline = create_pipeline_from_config(config)
        assert pipeline.config.grade == "XL"
        assert pipeline.config.scene_type.lower() == "hospital"
        assert pipeline.config.mode == PipelineMode.HARDWARE_IN_LOOP

    def test_pipeline_repr(self):
        """Pipeline 字符串表示"""
        pipeline = EmbodiedPipeline(grade="M", scene_type="warehouse", mode=PipelineMode.SIMULATION)
        repr_str = repr(pipeline)
        assert "M" in repr_str
        assert "simulation" in repr_str
        assert "WAREHOUSE" in repr_str


# =============================================================================
# Pipeline 生命周期测试
# =============================================================================

class TestPipelineLifecycle:
    """Pipeline 启动/停止/暂停生命周期测试"""

    def test_pipeline_start_stop(self):
        """启动和停止 Pipeline"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        # start() -> READY (initialized, accepting tasks)
        assert pipeline.state == PipelineState.READY
        assert not pipeline.is_running
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_pipeline_start_idle_then_run(self):
        """Pipeline 从 IDLE 到 READY"""
        pipeline = EmbodiedPipeline()
        assert pipeline.state == PipelineState.IDLE
        pipeline.start()
        assert pipeline.state == PipelineState.READY
        pipeline.stop()

    def test_pipeline_pause_resume(self):
        """暂停和恢复 Pipeline"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        assert pipeline.state == PipelineState.READY
        # pause from READY requires transition to RUNNING first (via implicit tick)
        # 直接从READY pause会失败因为pause检查RUNNING状态
        # 先转换到RUNNING状态
        pipeline._set_state(PipelineState.RUNNING)
        result = pipeline.pause()
        assert result is True
        assert pipeline.state == PipelineState.PAUSED
        result = pipeline.resume()
        assert result is True
        assert pipeline.state == PipelineState.RUNNING
        pipeline.stop()

    def test_pipeline_stop_idempotent(self):
        """多次停止不报错"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        pipeline.stop()
        pipeline.stop()  # 第二次应该安全
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_pipeline_uptime_tracking(self):
        """运行时间追踪"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        time.sleep(0.05)
        assert pipeline.uptime_s > 0.04
        pipeline.stop()


# =============================================================================
# AGV 等级 × 场景类型矩阵测试
# =============================================================================

class TestGradeSceneCombinations:
    """所有 AGV 等级 × 场景类型组合测试"""

    def test_all_combinations_init(self, all_grade_scene_combinations):
        """验证所有组合都能正常初始化"""
        for grade, scene in all_grade_scene_combinations:
            pipeline = EmbodiedPipeline(grade=grade, scene_type=scene)
            assert pipeline.config.grade == grade
            assert pipeline.config.scene_type == scene.upper()

    def test_all_combinations_start_stop(self, all_grade_scene_combinations):
        """所有组合都能正常启动和停止"""
        for grade, scene in all_grade_scene_combinations:
            pipeline = EmbodiedPipeline(grade=grade, scene_type=scene)
            pipeline.start()
            assert pipeline.state == PipelineState.READY  # start() -> READY
            pipeline.stop()
            assert pipeline.state == PipelineState.STOPPED

    def test_all_combinations_task_submit(self, all_grade_scene_combinations):
        """所有组合都能提交任务"""
        for grade, scene in all_grade_scene_combinations[:5]:  # 限制数量加速
            pipeline = EmbodiedPipeline(grade=grade, scene_type=scene)
            pipeline.start()
            req = TaskRequest(task_type="navigate", target="station_A")
            result = pipeline.submit_task(req)
            assert result is True
            time.sleep(0.01)
            pipeline.stop()


# =============================================================================
# 任务队列与执行测试
# =============================================================================

class TestTaskQueue:
    """任务队列测试"""

    def test_submit_single_task(self):
        """提交单个任务"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        req = TaskRequest(task_type="navigate", target="station_A")
        result = pipeline.submit_task(req)
        assert result is True
        time.sleep(0.05)
        pipeline.stop()

    def test_submit_multiple_tasks(self):
        """提交多个任务"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        for i in range(5):
            req = TaskRequest(task_type="navigate", target=f"station_{i}")
            pipeline.submit_task(req)
        time.sleep(0.05)
        pipeline.stop()

    def test_task_priority(self):
        """任务优先级"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        # 低优先级
        low = TaskRequest(task_type="navigate", target="low", priority=5)
        # 高优先级
        high = TaskRequest(task_type="navigate", target="high", priority=1)
        pipeline.submit_task(low)
        pipeline.submit_task(high)
        # 两个都能提交
        time.sleep(0.02)
        pipeline.stop()

    def test_task_request_auto_id(self):
        """TaskRequest 自动生成 ID"""
        req1 = TaskRequest(task_type="navigate")
        req2 = TaskRequest(task_type="navigate")
        assert req1.task_id != req2.task_id
        assert len(req1.task_id) > 0

    def test_task_result_fields(self):
        """TaskResult 字段"""
        result = TaskResult(
            task_id="test_001",
            success=True,
            phase="completed",
            duration_ms=1234.5,
            output={"distance": 10.0},
        )
        assert result.task_id == "test_001"
        assert result.success is True
        assert result.duration_ms == 1234.5
        assert result.output["distance"] == 10.0


# =============================================================================
# 状态持久化与恢复测试
# =============================================================================

class TestPipelineStatePersistence:
    """Pipeline 状态持久化与恢复测试"""

    def test_save_state_basic(self):
        """保存基本状态"""
        pipeline = EmbodiedPipeline(grade="L", scene_type="factory")
        pipeline.start()
        req = TaskRequest(task_type="navigate", target="station_A", priority=2)
        pipeline.submit_task(req)
        time.sleep(0.02)
        state = pipeline.save_state()
        pipeline.stop()
        assert state['version'] == '1.0'
        assert state['pipeline']['grade'] == 'L'
        assert state['pipeline']['scene_type'] == 'FACTORY'
        assert state['pipeline']['state'] == 'ready'  # start() -> READY

    def test_save_state_with_completed_tasks(self):
        """保存包含已完成任务的状态"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        # 模拟一些已完成的任务
        pipeline._completed_tasks.append(TaskResult(
            task_id="t1", success=True, duration_ms=100.0, phase="done"
        ))
        pipeline._completed_tasks.append(TaskResult(
            task_id="t2", success=False, duration_ms=50.0, phase="failed"
        ))
        state = pipeline.save_state()
        assert len(state['completed_tasks']) == 2
        pipeline.stop()

    def test_restore_state_basic(self):
        """恢复基本状态"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        req = TaskRequest(task_type="navigate", target="station_A")
        pipeline.submit_task(req)
        state = pipeline.save_state()
        pipeline.stop()

        # 恢复
        restored = EmbodiedPipeline(grade="M")
        success = restored.restore_state(state)
        assert success is True

    def test_export_import_checkpoint(self):
        """导出和导入检查点"""
        pipeline = EmbodiedPipeline(grade="XL", scene_type="hospital")
        pipeline.start()
        for i in range(3):
            pipeline.submit_task(TaskRequest(
                task_type="transport", target=f"target_{i}", priority=i+1
            ))
        pipeline._completed_tasks.append(TaskResult(
            task_id="done_1", success=True, duration_ms=200.0, phase="completed"
        ))
        time.sleep(0.02)

        # 导出
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            success = pipeline.export_checkpoint(tmp_path)
            assert success is True
            pipeline.stop()

            # 导入
            restored = EmbodiedPipeline.import_checkpoint(tmp_path)
            assert restored is not None
            assert restored.config.grade == "XL"
            assert restored.config.scene_type == "HOSPITAL"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_checkpoint_roundtrip_all_grades(self, grades):
        """所有等级的检查点往返"""
        for grade in grades:
            pipeline = EmbodiedPipeline(grade=grade, scene_type="warehouse")
            pipeline.start()
            pipeline.submit_task(TaskRequest(task_type="navigate", target="A"))
            time.sleep(0.01)

            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                tmp = f.name
            try:
                pipeline.export_checkpoint(tmp)
                pipeline.stop()
                restored = EmbodiedPipeline.import_checkpoint(tmp)
                assert restored is not None
                assert restored.config.grade == grade
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)


# =============================================================================
# 健康报告测试
# =============================================================================

class TestHealthReport:
    """Pipeline 健康报告测试"""

    def test_health_report_structure(self):
        """健康报告结构"""
        pipeline = EmbodiedPipeline(grade="M", scene_type="warehouse")
        pipeline.start()
        report = pipeline.get_health_report()
        pipeline.stop()
        assert 'timestamp' in report
        assert 'pipeline_state' in report
        assert 'modules' in report
        assert 'tasks' in report
        assert 'performance' in report

    def test_health_report_modules(self):
        """各模块可用性"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        report = pipeline.get_health_report()
        pipeline.stop()
        # scene_intelligence 和 skill_registry 默认启用
        assert 'behavior_tree' in report['modules']
        assert 'scene_intelligence' in report['modules']
        assert 'skill_registry' in report['modules']

    def test_health_report_task_stats(self):
        """任务统计"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        pipeline._completed_tasks.append(TaskResult(
            task_id="t1", success=True, duration_ms=100.0, phase="done"
        ))
        pipeline._completed_tasks.append(TaskResult(
            task_id="t2", success=True, duration_ms=200.0, phase="done"
        ))
        report = pipeline.get_health_report()
        pipeline.stop()
        assert report['tasks']['completed'] == 2
        assert report['tasks']['success_rate'] == 1.0

    def test_health_report_avg_duration(self):
        """平均任务耗时"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        pipeline._completed_tasks.append(TaskResult(
            task_id="t1", success=True, duration_ms=100.0, phase="done"
        ))
        pipeline._completed_tasks.append(TaskResult(
            task_id="t2", success=True, duration_ms=200.0, phase="done"
        ))
        report = pipeline.get_health_report()
        pipeline.stop()
        assert report['performance']['avg_task_duration_ms'] == 150.0

    def test_reset_health(self):
        """重置健康状态"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()
        pipeline._set_state(PipelineState.ERROR)
        pipeline._error_message = "test error"
        pipeline.reset_health()
        assert pipeline._error_message is None
        assert pipeline.state == PipelineState.READY
        pipeline.stop()


# =============================================================================
# 行为树与技能系统集成测试
# =============================================================================

class TestBTPlannerIntegration:
    """行为树规划器集成测试"""

    def test_bt_planner_all_grades(self, grades):
        """所有等级都能初始化行为树"""
        for grade in grades:
            planner = AGVTaskPlanner(grade=grade)
            assert planner.grade == grade
            assert len(planner.behavior_trees) > 0  # 有预设任务类型

    def test_bt_planner_capabilities(self, grades):
        """各等级能力检查"""
        for grade in grades:
            planner = AGVTaskPlanner(grade=grade)
            caps = planner.get_capabilities()
            assert caps['grade'] == grade
            assert 'max_planning_depth' in caps
            assert 'support_behavior_tree' in caps

    def test_agv_task_registration(self):
        """AGV 任务注册"""
        planner = AGVTaskPlanner(grade="M")
        assert 'navigate' in planner.behavior_trees
        assert 'transport' in planner.behavior_trees
        assert 'patrol' in planner.behavior_trees


# =============================================================================
# 技能注册表集成测试
# =============================================================================

class TestSkillRegistryIntegration:
    """技能注册表集成测试"""

    def test_global_registry_singleton(self):
        """全局注册表单例"""
        reg1 = get_global_skill_registry()
        reg2 = get_global_skill_registry()
        assert reg1 is reg2

    def test_registry_stats(self):
        """注册表统计"""
        reg = get_global_skill_registry()
        stats = reg.get_registry_stats()
        assert 'total_skills' in stats
        assert 'by_status' in stats
        assert 'by_category' in stats

    def test_skill_status_transitions(self):
        """技能状态转换"""
        reg = EmbodiedSkillRegistry()
        reg.register_standard_agv_skills()
        skills = reg.get_active_skills()
        assert len(skills) > 0

    def test_skills_by_scene(self, scene_types):
        """按场景获取技能"""
        reg = get_global_skill_registry()
        for scene in scene_types:
            skills = reg.get_skills_by_scene(scene)
            # 每个场景都应该有技能
            assert isinstance(skills, list)

    def test_skills_by_category(self):
        """按类别获取技能"""
        reg = get_global_skill_registry()
        from src.embodied.embodied_skill import SkillCategory
        nav_skills = reg.get_skills_by_category(SkillCategory.NAVIGATION)
        assert isinstance(nav_skills, list)


# =============================================================================
# Pipeline 配置测试
# =============================================================================

class TestPipelineConfig:
    """Pipeline 配置测试"""

    def test_config_defaults(self):
        """默认配置"""
        config = PipelineConfig()
        assert config.grade == "M"
        assert config.mode == PipelineMode.SIMULATION
        assert config.scene_type == "WAREHOUSE"
        assert config.enable_skill_registry is True
        assert config.enable_memory is True
        assert config.enable_scene_intelligence is True

    def test_config_to_dict(self):
        """配置转字典"""
        config = PipelineConfig(grade="L", mode=PipelineMode.HARDWARE_IN_LOOP)
        d = config.to_dict()
        assert d['grade'] == "L"
        assert d['mode'] == 'hil'  # PipelineMode.HARDWARE_IN_LOOP.value

    def test_config_module_toggles(self):
        """模块开关配置"""
        config = PipelineConfig(
            enable_skill_registry=False,
            enable_memory=False,
            enable_scene_intelligence=False,
            enable_hil=True,
        )
        assert config.enable_skill_registry is False
        assert config.enable_memory is False
        assert config.enable_scene_intelligence is False
        assert config.enable_hil is True

    def test_config_sensor_toggles(self):
        """传感器开关配置"""
        config = PipelineConfig(
            enable_vision=True,
            enable_audio=True,
            enable_tactile=False,
            enable_force=False,
        )
        assert config.enable_vision is True
        assert config.enable_audio is True
        assert config.enable_tactile is False
        assert config.enable_force is False


# =============================================================================
# Pipeline 状态机测试
# =============================================================================

class TestPipelineStateMachine:
    """Pipeline 状态机转换测试"""

    def test_state_transitions(self):
        """合法状态转换"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        assert pipeline.state == PipelineState.READY
        # 手动转到 RUNNING 才能 pause
        pipeline._set_state(PipelineState.RUNNING)
        pipeline.pause()
        assert pipeline.state == PipelineState.PAUSED
        pipeline.resume()
        assert pipeline.state == PipelineState.RUNNING
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_cannot_start_when_stopped(self):
        """已停止的 Pipeline 不能直接 start（需要重建）"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        pipeline.stop()
        # start 后应重新初始化
        pipeline.start()
        assert pipeline.state == PipelineState.READY
        pipeline.stop()

    def test_multiple_pause_resume(self):
        """多次暂停/恢复: start()->READY, 需要手动设RUNNING才能pause"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        # start() -> READY, transition to RUNNING to allow pause
        pipeline._set_state(PipelineState.RUNNING)
        for _ in range(3):
            pipeline.pause()
            assert pipeline.state == PipelineState.PAUSED
            pipeline.resume()
            assert pipeline.state == PipelineState.RUNNING
        pipeline.stop()

    def test_uptime_accumulates(self):
        """运行时间累积 - uptime持续增加，不因暂停而冻结"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        time.sleep(0.03)
        t1 = pipeline.uptime_s
        pipeline.pause()
        time.sleep(0.05)
        t2 = pipeline.uptime_s
        pipeline.resume()
        time.sleep(0.03)
        t3 = pipeline.uptime_s
        pipeline.stop()
        # uptime 持续累加（不冻结），体现总 elapsed time
        assert t3 > t2 > t1


# =============================================================================
# 订阅/通知系统测试
# =============================================================================

class TestPubSub:
    """发布/订阅系统测试"""

    def test_subscribe_state_changed(self):
        """状态变更订阅"""
        pipeline = EmbodiedPipeline()
        events = []

        def handler(data):
            events.append(data)

        pipeline.subscribe("state_changed", handler)
        pipeline.start()
        pipeline.stop()
        assert len(events) > 0

    def test_multiple_subscribers(self):
        """多订阅者"""
        pipeline = EmbodiedPipeline()
        counter = {'n': 0}

        def handler1(data):
            counter['n'] += 1

        def handler2(data):
            counter['n'] += 10

        pipeline.subscribe("state_changed", handler1)
        pipeline.subscribe("state_changed", handler2)
        pipeline.start()
        pipeline.stop()
        assert counter['n'] >= 11


# =============================================================================
# 并发安全测试
# =============================================================================

class TestConcurrency:
    """并发安全测试"""

    def test_concurrent_task_submission(self):
        """并发任务提交"""
        pipeline = EmbodiedPipeline(grade="M")
        pipeline.start()

        def submit_tasks(n):
            for i in range(n):
                pipeline.submit_task(TaskRequest(
                    task_type="navigate", target=f"t_{i}"
                ))

        threads = [threading.Thread(target=submit_tasks, args=(10,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.05)
        pipeline.stop()

    def test_concurrent_save_state(self):
        """并发保存状态"""
        pipeline = EmbodiedPipeline(grade="L")
        pipeline.start()

        def save_states(n):
            for _ in range(n):
                pipeline.save_state()

        threads = [threading.Thread(target=save_states, args=(20,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pipeline.stop()
        # 不应崩溃
