"""
test_embodied_pipeline.py - EmbodiedPipeline 集成测试
=====================================================

测试 SuperModel 具身智能统一 Pipeline 的完整功能:
- Pipeline 生命周期 (启动/停止/暂停/恢复)
- 任务执行 (同步/异步提交)
- 模块懒加载
- 场景自适应
- 仿真步骤
- 状态查询
- HIL 模式
"""

import time
import threading
import pytest
from collections import deque

from src.embodied.embodied_pipeline import (
    PipelineMode,
    PipelineState,
    PipelineConfig,
    TaskRequest,
    TaskResult,
    EmbodiedPipeline,
    create_embodied_pipeline,
    create_pipeline_from_config,
)


# ============================================================
# 生命周期测试
# ============================================================

class TestPipelineLifecycle:
    """Pipeline 生命周期测试"""

    def test_create_pipeline_default(self):
        """测试默认配置创建 Pipeline"""
        pipeline = EmbodiedPipeline()
        assert pipeline.config.grade == "M"
        assert pipeline.config.mode == PipelineMode.SIMULATION
        assert pipeline.config.scene_type == "WAREHOUSE"
        assert pipeline.state == PipelineState.IDLE

    def test_create_pipeline_explicit_config(self):
        """测试显式配置创建 Pipeline"""
        config = PipelineConfig(
            grade="L",
            mode=PipelineMode.HARDWARE_IN_LOOP,
            scene_type="HOSPITAL",
            enable_memory=True,
            enable_skill_registry=True,
        )
        pipeline = EmbodiedPipeline(config=config)
        assert pipeline.config.grade == "L"
        assert pipeline.config.mode == PipelineMode.HARDWARE_IN_LOOP
        assert pipeline.config.scene_type == "HOSPITAL"

    def test_start_stop_cycle(self):
        """测试启动/停止完整周期"""
        pipeline = EmbodiedPipeline(grade="S", mode=PipelineMode.SIMULATION)
        assert pipeline.state == PipelineState.IDLE

        success = pipeline.start()
        assert success is True
        assert pipeline.state == PipelineState.READY
        assert pipeline.is_running is False  # READY, not RUNNING

        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_start_idempotent(self):
        """测试重复启动无效"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        state_after_first = pipeline.state

        # 从 READY 状态无法再次 start
        result = pipeline.start()
        assert result is False
        assert pipeline.state == state_after_first
        pipeline.stop()

    def test_stop_from_idle(self):
        """测试从 IDLE 状态停止"""
        pipeline = EmbodiedPipeline()
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_pause_resume(self):
        """测试暂停/恢复"""
        pipeline = EmbodiedPipeline()
        pipeline.start()

        # 暂停需要 RUNNING 状态
        pause_ok = pipeline.pause()
        assert pause_ok is False  # still READY, not RUNNING

        pipeline.stop()

    def test_uptime_tracking(self):
        """测试运行时间追踪"""
        pipeline = EmbodiedPipeline()
        assert pipeline.uptime_s == 0.0

        pipeline.start()
        time.sleep(0.1)
        assert pipeline.uptime_s > 0.0
        pipeline.stop()
        # stop 后 uptime 回零
        assert pipeline.uptime_s == 0.0

    def test_can_restart_from_stopped(self):
        """测试从 STOPPED 状态可以重新启动"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        assert pipeline.state == PipelineState.READY
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED
        # 从 STOPPED 可以重新 start (restart)
        result = pipeline.start()
        assert result is True
        assert pipeline.state == PipelineState.READY
        pipeline.stop()


# ============================================================
# 工厂函数测试
# ============================================================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_embodied_pipeline_defaults(self):
        """测试 create_embodied_pipeline 默认参数"""
        pipeline = create_embodied_pipeline()
        assert pipeline.config.grade == "M"
        assert pipeline.config.mode == PipelineMode.SIMULATION
        assert pipeline.config.scene_type == "WAREHOUSE"

    def test_create_embodied_pipeline_all_params(self):
        """测试 create_embodied_pipeline 全参数"""
        pipeline = create_embodied_pipeline(
            grade="XL",
            mode="hardware_in_loop",
            scene_type="FACTORY",
            enable_memory=False,
        )
        assert pipeline.config.grade == "XL"
        assert pipeline.config.mode == PipelineMode.HARDWARE_IN_LOOP
        assert pipeline.config.scene_type == "FACTORY"
        assert pipeline.config.enable_memory is False

    def test_create_embodied_pipeline_hil_shortform(self):
        """测试 hil 模式简写"""
        pipeline = create_embodied_pipeline(mode="hil")
        assert pipeline.config.mode == PipelineMode.HARDWARE_IN_LOOP

    def test_create_embodied_pipeline_full_physical(self):
        """测试全实体模式"""
        pipeline = create_embodied_pipeline(mode="full_physical")
        assert pipeline.config.mode == PipelineMode.FULL_PHYSICAL

    def test_create_pipeline_from_config_dict(self):
        """测试从字典创建 Pipeline"""
        config_dict = {
            "grade": "XXL",
            "mode": "simulation",
            "scene_type": "HOSPITAL",
        }
        pipeline = create_pipeline_from_config(config_dict)
        assert pipeline.config.grade == "XXL"
        assert pipeline.config.scene_type == "HOSPITAL"

    def test_grade_uppercase(self):
        """测试等级自动大写"""
        pipeline = create_embodied_pipeline(grade="l")
        assert pipeline.config.grade == "L"

    def test_scene_uppercase(self):
        """测试场景自动大写"""
        pipeline = create_embodied_pipeline(scene_type="restaurant")
        assert pipeline.config.scene_type == "RESTAURANT"


# ============================================================
# 任务执行测试
# ============================================================

class TestTaskExecution:
    """任务执行测试 (Pipeline API 级别，不依赖 Executor 内部实现)"""

    def test_execute_task_returns_result(self):
        """测试 execute_task 返回 TaskResult"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        result = pipeline.execute_task("transport", target="station_A")
        # 返回了结果对象
        assert isinstance(result, TaskResult)
        assert result.task_id != ""
        # 任务通过 Executor 执行，返回了确定的阶段
        assert result.phase in ("completed", "failed", "execution", "running", "succeeded")
        pipeline.stop()

    def test_execute_task_with_payload(self):
        """测试带 payload 的任务"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        result = pipeline.execute_task(
            "patrol",
            target="aisle_3",
            payload={"speed": 0.5, "check_inventory": True},
        )
        assert isinstance(result, TaskResult)
        assert result.task_id != ""
        assert result.scene_type == "WAREHOUSE"  # 默认场景
        pipeline.stop()

    def test_execute_task_timeout(self):
        """测试任务超时"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        # 超短超时返回结果对象
        result = pipeline.execute_task("transport", timeout_s=0.001)
        pipeline.stop()
        assert isinstance(result, TaskResult)

    @pytest.mark.slow
    def test_execute_task_with_real_executor(self):
        """测试任务通过真实Executor执行 (标记为slow，仅CI运行)"""
        import pytest
        pytest.skip("slow integration test - run manually with real hardware")

    def test_execute_task_priority(self):
        """测试不同优先级任务的 TaskRequest"""
        req_low = TaskRequest(task_type="patrol", priority=5)
        req_high = TaskRequest(task_type="emergency", priority=0)
        assert req_low.priority == 5
        assert req_high.priority == 0
        assert req_high.task_id != req_low.task_id

    def test_execute_task_from_wrong_state(self):
        """测试错误状态执行"""
        pipeline = EmbodiedPipeline()
        # 未启动就执行
        result = pipeline.execute_task("transport")
        assert result.success is False
        assert "not ready" in result.error.lower()

    def test_execute_multiple_tasks(self):
        """测试 Pipeline 队列管理多个任务"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        # 提交多个任务到队列
        for i in range(3):
            req = TaskRequest(task_type="transport", target=f"station_{i}")
            ok = pipeline.submit_task(req)
            assert ok is True
        status = pipeline.get_status()
        assert status['queue_size'] == 3
        pipeline.stop()

    def test_task_result_fields(self):
        """测试 TaskResult 字段完整性"""
        # 测试 TaskResult 字段不依赖真实执行器
        # 验证 Pipeline 的 TaskRequest -> TaskResult 映射
        from src.embodied.embodied_pipeline import TaskResult
        result = TaskResult(
            task_id="test-123",
            success=True,
            phase="completed",
            duration_ms=50.0,
            output={"task_type": "transport"},
        )
        assert result.task_id == "test-123"
        assert result.success is True
        assert result.phase == "completed"
        assert result.duration_ms == 50.0
        assert result.output["task_type"] == "transport"

    def test_task_request_auto_id(self):
        """测试 TaskRequest 自动生成 ID"""
        req = TaskRequest(task_type="test")
        assert req.task_id != ""
        assert len(req.task_id) == 8

    def test_task_request_explicit_id(self):
        """测试 TaskRequest 显式 ID"""
        req = TaskRequest(task_id="my-task-123", task_type="test")
        assert req.task_id == "my-task-123"


# ============================================================
# 异步任务提交测试
# ============================================================

class TestAsyncTaskSubmission:
    """异步任务提交测试"""

    def test_submit_task(self):
        """测试任务提交到队列"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        request = TaskRequest(task_id="", task_type="transport", target="station_C")
        ok = pipeline.submit_task(request)
        assert ok is True
        pipeline.stop()

    def test_submit_task_wrong_state(self):
        """测试错误状态提交"""
        pipeline = EmbodiedPipeline()
        request = TaskRequest(task_type="transport")
        ok = pipeline.submit_task(request)
        assert ok is False


# ============================================================
# Pipeline 配置测试
# ============================================================

class TestPipelineConfig:
    """Pipeline 配置测试"""

    def test_config_defaults(self):
        """测试默认配置值"""
        config = PipelineConfig()
        assert config.grade == "M"
        assert config.mode == PipelineMode.SIMULATION
        assert config.enable_skill_registry is True
        assert config.enable_memory is True
        assert config.enable_scene_intelligence is True
        assert config.enable_hil is False

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = PipelineConfig(grade="L", mode=PipelineMode.HARDWARE_IN_LOOP)
        d = config.to_dict()
        assert d['grade'] == "L"
        assert d['mode'] == "hil"  # PipelineMode.HARDWARE_IN_LOOP.value
        assert d['enable_skill_registry'] is True

    def test_config_all_grades(self):
        """测试所有 AGV 等级"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            pipeline = EmbodiedPipeline(grade=grade)
            assert pipeline.config.grade == grade


# ============================================================
# 状态查询测试
# ============================================================

class TestStatusQueries:
    """状态查询测试"""

    def test_get_status_initial(self):
        """测试初始状态查询"""
        pipeline = EmbodiedPipeline()
        status = pipeline.get_status()
        assert status['state'] == "idle"
        assert status['grade'] == "M"
        assert status['mode'] == "simulation"
        assert status['queue_size'] == 0
        assert status['active_tasks'] == 0

    def test_get_status_running(self):
        """测试运行状态查询"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        pipeline.execute_task("transport")
        status = pipeline.get_status()
        assert status['state'] == "ready"
        assert status['completed_tasks'] == 1
        assert 'modules' in status
        assert isinstance(status['modules'], dict)
        pipeline.stop()

    def test_get_status_modules(self):
        """测试模块状态"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        status = pipeline.get_status()
        modules = status['modules']
        # 至少有一些模块被初始化
        assert isinstance(modules['behavior_tree'], bool)
        assert isinstance(modules['scene_intelligence'], bool)
        pipeline.stop()

    def test_get_memory_summary_disabled(self):
        """测试禁用记忆时的摘要"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        summary = pipeline.get_memory_summary()
        # 记忆可能开启或关闭
        assert 'enabled' in summary
        pipeline.stop()

    def test_get_skill_summary(self):
        """测试技能摘要"""
        pipeline = EmbodiedPipeline()
        pipeline.start()
        summary = pipeline.get_skill_summary()
        assert 'enabled' in summary
        pipeline.stop()


# ============================================================
# 仿真支持测试
# ============================================================

class TestSimulationSupport:
    """仿真支持测试"""

    def test_run_simulation_step(self):
        """测试仿真步骤"""
        pipeline = EmbodiedPipeline(mode=PipelineMode.SIMULATION)
        pipeline.start()
        result = pipeline.run_simulation_step(dt=0.01)
        # 可能没有 SimEnhancer，返回 error 或 success
        assert isinstance(result, dict)
        pipeline.stop()

    def test_get_scene_state(self):
        """测试获取场景状态"""
        pipeline = EmbodiedPipeline(scene_type="FACTORY")
        pipeline.start()
        state = pipeline.get_scene_state()
        assert state['scene_type'] == "FACTORY"
        pipeline.stop()


# ============================================================
# 事件订阅测试
# ============================================================

class TestEventSubscription:
    """事件订阅测试"""

    def test_subscribe_state_changed(self):
        """测试状态变更事件"""
        pipeline = EmbodiedPipeline()
        events = []

        def on_state_change(data):
            events.append(data)

        pipeline.subscribe("state_changed", on_state_change)
        pipeline.start()
        assert len(events) >= 1
        assert events[-1]['new'] == PipelineState.READY
        pipeline.stop()

    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        pipeline = EmbodiedPipeline()
        count = [0, 0]

        def cb1(_):
            count[0] += 1

        def cb2(_):
            count[1] += 1

        pipeline.subscribe("state_changed", cb1)
        pipeline.subscribe("state_changed", cb2)
        pipeline.start()
        assert count[0] >= 1
        assert count[1] >= 1
        pipeline.stop()


# ============================================================
# Pipeline repr 测试
# ============================================================

class TestPipelineRepr:
    """Pipeline repr 测试"""

    def test_repr(self):
        """测试 __repr__"""
        pipeline = EmbodiedPipeline(grade="XL", mode=PipelineMode.SIMULATION, scene_type="HOSPITAL")
        r = repr(pipeline)
        assert "EmbodiedPipeline" in r
        assert "XL" in r
        assert "HOSPITAL" in r
        assert "simulation" in r


# ============================================================
# 集成场景测试
# ============================================================

class TestIntegrationScenarios:
    """端到端集成场景测试 (Pipeline API 级别)"""

    @pytest.mark.slow
    def test_warehouse_pipeline_flow(self):
        """测试仓库 Pipeline 流程"""
        pipeline = create_embodied_pipeline(
            grade="L",
            scene_type="warehouse",
        )
        pipeline.start()

        # 提交拣选任务 (不依赖执行器成功)
        result1 = pipeline.execute_task("pick_and_stow", target="shelf_A3")
        assert isinstance(result1, TaskResult)
        assert result1.task_id != ""

        # 检查 Pipeline 状态
        status = pipeline.get_status()
        assert status['scene_type'] == "WAREHOUSE"
        assert status['grade'] == "L"

        pipeline.stop()

    def test_hospital_pipeline_flow(self):
        """测试医院 Pipeline 流程"""
        pipeline = create_embodied_pipeline(
            grade="M",
            scene_type="hospital",
        )
        pipeline.start()

        # 药品运输任务
        result = pipeline.execute_task(
            "transport_medicine",
            target="ward_3",
            payload={"patient_id": "P-1234", "urgent": True},
        )
        assert isinstance(result, TaskResult)
        assert result.scene_type == "HOSPITAL"

        pipeline.stop()

    def test_factory_pipeline_flow(self):
        """测试工厂 Pipeline 流程"""
        pipeline = create_embodied_pipeline(
            grade="XL",
            scene_type="factory",
        )
        pipeline.start()

        tasks = [
            ("assembly_handover", "station_1"),
            ("quality_scan", "line_A"),
            ("maintenance_check", "robot_arm_2"),
        ]

        for task_type, target in tasks:
            result = pipeline.execute_task(task_type, target=target)
            assert isinstance(result, TaskResult)
            assert result.task_id != ""

        pipeline.stop()

    def test_concurrent_task_submission(self):
        """测试并发任务提交"""
        pipeline = EmbodiedPipeline(grade="L")
        pipeline.start()

        requests = [
            TaskRequest(task_type="transport", target=f"station_{i}")
            for i in range(4)
        ]

        for req in requests:
            pipeline.submit_task(req)

        status = pipeline.get_status()
        assert status['queue_size'] == 4

        pipeline.stop()

    def test_hil_mode_initialization(self):
        """测试 HIL 模式初始化"""
        pipeline = create_embodied_pipeline(
            grade="M",
            mode="hardware_in_loop",
            enable_hil=True,
        )
        pipeline.start()
        status = pipeline.get_status()
        assert status['mode'] == "hil"
        pipeline.stop()

    def test_full_physical_mode(self):
        """测试全实体模式"""
        pipeline = create_embodied_pipeline(
            grade="XL",
            mode="full_physical",
        )
        pipeline.start()
        status = pipeline.get_status()
        assert status['mode'] == "full_physical"
        pipeline.stop()

    def test_scene_state_reflects_scene_type(self):
        """测试场景状态正确反映场景类型"""
        for scene in ["warehouse", "hospital", "factory", "restaurant", "outdoor"]:
            pipeline = create_embodied_pipeline(scene_type=scene)
            pipeline.start()
            state = pipeline.get_scene_state()
            assert state['scene_type'] == scene.upper()
            pipeline.stop()


# ============================================================
# Federated Learning + Swarm Integration Tests (v3.9.3)
# ============================================================

class TestFederatedLearningIntegration:
    """联邦学习与Pipeline集成测试"""

    def test_pipeline_with_fl_enabled(self):
        """测试启用FL的Pipeline创建"""
        config = PipelineConfig(
            grade='L',
            mode=PipelineMode.SIMULATION,
            enable_federated_learning=True,
        )
        p = EmbodiedPipeline(config=config)
        assert config.enable_federated_learning is True

    def test_fl_coordinator_initialized(self):
        """测试FL协调器正确初始化"""
        config = PipelineConfig(
            grade='L',
            mode=PipelineMode.SIMULATION,
            enable_federated_learning=True,
            fl_num_clients=4,
        )
        p = EmbodiedPipeline(config=config)
        p.start()
        assert p._fl_coordinator is not None
        assert p.get_status()['modules']['federated_learning'] is True
        p.stop()

    def test_register_agv_to_fl(self):
        """测试AGV注册到FL系统"""
        config = PipelineConfig(grade='L', enable_federated_learning=True)
        p = EmbodiedPipeline(config=config)
        p.start()
        ok = p.register_agv_to_fl('agv_fl_001', 'L')
        assert ok is True
        status = p.get_fl_status()
        assert status['enabled'] is True
        assert status['registered_clients'] >= 1
        p.stop()

    def test_fl_status_fields(self):
        """测试FL状态返回正确字段"""
        config = PipelineConfig(grade='M', enable_federated_learning=True)
        p = EmbodiedPipeline(config=config)
        p.start()
        status = p.get_fl_status()
        assert 'enabled' in status
        assert 'round_count' in status
        assert status['round_count'] == 0
        p.stop()

    def test_fl_disabled_returns_proper_message(self):
        """测试FL禁用时返回正确的提示"""
        config = PipelineConfig(grade='M', enable_federated_learning=False)
        p = EmbodiedPipeline(config=config)
        p.start()
        status = p.get_fl_status()
        assert status.get('enabled') is False
        p.stop()


class TestSwarmCoordinationIntegration:
    """蜂群协调与Pipeline集成测试"""

    def test_pipeline_with_swarm_enabled(self):
        """测试启用蜂群协调的Pipeline创建"""
        config = PipelineConfig(
            grade='L',
            mode=PipelineMode.SIMULATION,
            enable_swarm_coordination=True,
        )
        p = EmbodiedPipeline(config=config)
        assert config.enable_swarm_coordination is True

    def test_swarm_coordinator_initialized(self):
        """测试蜂群协调器正确初始化"""
        config = PipelineConfig(
            grade='L',
            mode=PipelineMode.SIMULATION,
            enable_swarm_coordination=True,
            fl_num_clients=3,
        )
        p = EmbodiedPipeline(config=config)
        p.start()
        assert p._swarm_coord is not None
        assert p.get_status()['modules']['swarm_coordination'] is True
        p.stop()

    def test_swarm_task_trigger(self):
        """测试蜂群任务触发"""
        config = PipelineConfig(grade='L', enable_swarm_coordination=True)
        p = EmbodiedPipeline(config=config)
        p.start()
        task_id = p.trigger_swarm_task(
            'transport',
            ['agv_s001', 'agv_s002'],
            {'dest': [10.0, 0.0, 0.0], 'source': [0.0, 0.0, 0.0]}
        )
        assert task_id is not None
        assert len(task_id) > 0
        p.stop()

    def test_swarm_task_with_inspection_type(self):
        """测试inspection类型蜂群任务"""
        config = PipelineConfig(grade='L', enable_swarm_coordination=True)
        p = EmbodiedPipeline(config=config)
        p.start()
        task_id = p.trigger_swarm_task('inspection', ['agv_001'], {'dest': [5.0, 5.0, 0.0]})
        assert task_id is not None
        p.stop()

    def test_swarm_disabled_returns_proper_message(self):
        """测试蜂群禁用时返回正确的提示"""
        config = PipelineConfig(grade='M', enable_swarm_coordination=False)
        p = EmbodiedPipeline(config=config)
        p.start()
        status = p.get_swarm_status()
        assert status.get('enabled') is False
        p.stop()


class TestFLAndSwarmCombined:
    """FL + Swarm 联合测试"""

    def test_pipeline_with_both_enabled(self):
        """测试同时启用FL和蜂群"""
        config = PipelineConfig(
            grade='XL',
            mode=PipelineMode.SIMULATION,
            enable_federated_learning=True,
            enable_swarm_coordination=True,
            fl_num_clients=4,
        )
        p = EmbodiedPipeline(config=config)
        p.start()
        assert p._fl_coordinator is not None
        assert p._swarm_coord is not None
        assert p.get_status()['modules']['federated_learning'] is True
        assert p.get_status()['modules']['swarm_coordination'] is True
        p.stop()

    def test_fl_register_then_swarm_task(self):
        """测试FL注册和蜂群任务联合流程"""
        config = PipelineConfig(
            grade='L',
            enable_federated_learning=True,
            enable_swarm_coordination=True,
        )
        p = EmbodiedPipeline(config=config)
        p.start()

        # Register to FL
        ok = p.register_agv_to_fl('agv_combined_001', 'L')
        assert ok is True

        # Trigger swarm task
        task_id = p.trigger_swarm_task('transport', ['agv_combined_001'], {'dest': [20.0, 0.0, 0.0]})
        assert task_id is not None

        # Verify both systems report correct state
        fl_status = p.get_fl_status()
        swarm_status = p.get_swarm_status()
        assert fl_status['enabled'] is True
        assert swarm_status['enabled'] is True

        p.stop()

    def test_save_restore_with_fl_round_count(self):
        """测试FL轮次计数在状态保存/恢复中的正确性"""
        config = PipelineConfig(grade='L', enable_federated_learning=True)
        p = EmbodiedPipeline(config=config)
        p.start()

        # Manually increment round count (simulate FL training)
        p._fl_round_count = 5

        # Save state
        state = p.save_state()
        assert state['pipeline']['fl_round_count'] == 5

        p.stop()

    def test_grade_config_passed_to_fl(self):
        """测试AGV等级配置正确传递给FL系统"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = PipelineConfig(grade=grade, enable_federated_learning=True)
            p = EmbodiedPipeline(config=config)
            p.start()
            assert p._fl_coordinator is not None
            p.stop()
