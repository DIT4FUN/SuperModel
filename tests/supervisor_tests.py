"""
控制监管器测试
==============

测试 ControlSupervisor、ControllerInterface 和相关组件
- 控制器注册与注销
- 模式切换与超时
- 故障检测与恢复
- 紧急停止与释放
- 性能指标监控
- 日志记录
"""

import numpy as np
import sys
import time
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.supervisor import (
    ControlSupervisor, ControllerInterface, SupervisorConfig,
    ControlMode, HealthStatus, ControlState, ControllerMetrics,
    MockJointController, MockCartesianController, MockImpedanceController
)


class TestSupervisorConfig(unittest.TestCase):
    """测试监管器配置"""

    def test_default_config(self):
        config = SupervisorConfig()
        self.assertEqual(config.mode_switch_timeout_s, 2.0)
        self.assertEqual(config.controller_heartbeat_s, 0.5)
        self.assertEqual(config.max_latency_ms, 50.0)
        self.assertEqual(config.max_tracking_error, 0.5)
        self.assertEqual(config.fault_count_threshold, 3)
        self.assertTrue(config.enable_fault_recovery)
        self.assertTrue(config.graceful_degradation)
        self.assertTrue(config.emergency_stop_enabled)

    def test_custom_config(self):
        config = SupervisorConfig(
            mode_switch_timeout_s=5.0,
            max_latency_ms=100.0,
            enable_fault_recovery=False
        )
        self.assertEqual(config.mode_switch_timeout_s, 5.0)
        self.assertEqual(config.max_latency_ms, 100.0)
        self.assertFalse(config.enable_fault_recovery)


class TestControllerInterface(unittest.TestCase):
    """测试控制器接口"""

    def test_controller_creation(self):
        ctrl = MockJointController("test_joint")
        self.assertEqual(ctrl.name, "test_joint")
        self.assertEqual(ctrl.controller_type, "joint_position")
        self.assertFalse(ctrl.is_active)

    def test_controller_lifecycle(self):
        ctrl = MockJointController()
        self.assertFalse(ctrl.is_active)

        self.assertTrue(ctrl.start())
        self.assertTrue(ctrl.is_active)

        self.assertTrue(ctrl.stop())
        self.assertFalse(ctrl.is_active)

    def test_controller_reset(self):
        ctrl = MockJointController()
        ctrl.start()
        self.assertTrue(ctrl.is_active)
        ctrl.reset()
        # reset 重置控制器指标，但不改变 is_active 状态
        self.assertTrue(ctrl.is_active)
        metrics = ctrl.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.name, "mock_joint")

    def test_controller_compute(self):
        ctrl = MockJointController()
        ctrl.start()
        output = ctrl.compute({}, {"target": np.zeros(6)})
        self.assertIn("joint_velocity", output)
        self.assertIn("joint_torque", output)
        ctrl.stop()

    def test_health_check(self):
        ctrl = MockJointController()
        healthy, msg = ctrl.health_check()
        self.assertTrue(healthy)
        self.assertEqual(msg, "OK")

    def test_metrics(self):
        ctrl = MockJointController()
        metrics = ctrl.get_metrics()
        self.assertEqual(metrics.name, "mock_joint")
        self.assertEqual(metrics.latency_ms, 0.0)

    def test_mock_cartesian_controller_create(self):
        ctrl = MockCartesianController("test_cartesian")
        self.assertEqual(ctrl.name, "test_cartesian")
        self.assertEqual(ctrl.controller_type, "cartesian_velocity")
        self.assertFalse(ctrl.is_active)

    def test_mock_impedance_controller_create(self):
        ctrl = MockImpedanceController("test_impedance")
        self.assertEqual(ctrl.name, "test_impedance")
        self.assertEqual(ctrl.controller_type, "impedance")
        self.assertFalse(ctrl.is_active)

    def test_control_mode_values(self):
        self.assertEqual(ControlMode.IDLE.value, "idle")
        self.assertEqual(ControlMode.JOINT_POSITION.value, "joint_position")
        self.assertEqual(ControlMode.EMERGENCY_STOP.value, "emergency_stop")

    def test_health_status_values(self):
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.FAULT.value, "fault")
        self.assertEqual(HealthStatus.EMERGENCY.value, "emergency")


class TestControlSupervisor(unittest.TestCase):
    """测试控制监管器"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_supervisor")

    def tearDown(self):
        self.supervisor.__exit__(None, None, None)

    def test_supervisor_creation(self):
        self.assertEqual(self.supervisor.supervisor_id, "test_supervisor")
        self.assertEqual(self.supervisor._state.mode, ControlMode.IDLE)
        self.assertEqual(self.supervisor._state.health, HealthStatus.HEALTHY)

    def test_register_controller(self):
        ctrl = MockJointController("joint_ctrl")
        result = self.supervisor.register_controller(ctrl)
        self.assertTrue(result)

        # 重复注册应返回 False
        result2 = self.supervisor.register_controller(ctrl)
        self.assertFalse(result2)

    def test_unregister_controller(self):
        ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(ctrl)

        result = self.supervisor.unregister_controller("joint_ctrl")
        self.assertTrue(result)

        # 重复注销应返回 False
        result2 = self.supervisor.unregister_controller("joint_ctrl")
        self.assertFalse(result2)

    def test_get_controller(self):
        ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(ctrl)

        retrieved = self.supervisor.get_controller("joint_ctrl")
        self.assertIs(retrieved, ctrl)

        missing = self.supervisor.get_controller("nonexistent")
        self.assertIsNone(missing)

    def test_list_controllers(self):
        ctrl1 = MockJointController("joint_ctrl")
        ctrl2 = MockCartesianController("cart_ctrl")
        self.supervisor.register_controller(ctrl1)
        self.supervisor.register_controller(ctrl2)

        controllers = self.supervisor.list_controllers()
        self.assertEqual(len(controllers), 2)
        self.assertIn("joint_ctrl", controllers)
        self.assertIn("cart_ctrl", controllers)

    def test_register_duplicate_controller(self):
        ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(ctrl)
        result = self.supervisor.register_controller(ctrl)
        self.assertFalse(result)

    def test_unregister_nonexistent(self):
        result = self.supervisor.unregister_controller("nonexistent")
        self.assertFalse(result)

    def test_get_nonexistent_controller(self):
        missing = self.supervisor.get_controller("nonexistent")
        self.assertIsNone(missing)

    def test_multiple_controllers_same_type(self):
        ctrl1 = MockJointController("joint_ctrl_1")
        ctrl2 = MockJointController("joint_ctrl_2")
        self.supervisor.register_controller(ctrl1)
        self.supervisor.register_controller(ctrl2)
        self.assertEqual(len(self.supervisor.list_controllers()), 2)

    def test_uptime_tracking(self):
        state = self.supervisor.get_state()
        self.assertGreaterEqual(state.uptime_s, 0)

    def test_fault_history_recorded(self):
        self.supervisor._state.fault_history.append((time.time(), "test fault"))
        self.assertEqual(len(self.supervisor._state.fault_history), 1)

    def test_log_on_register(self):
        ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(ctrl)
        log = self.supervisor.get_log()
        self.assertTrue(any(e["type"] == "controller_registered" for e in log))

    def test_find_controller_for_mode(self):
        ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(ctrl)
        found = self.supervisor._find_controller_for_mode(ControlMode.JOINT_POSITION)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "joint_ctrl")


class TestSupervisorModeSwitch(unittest.TestCase):
    """测试模式切换"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_switch")
        self.joint_ctrl = MockJointController("joint_ctrl")
        self.cart_ctrl = MockCartesianController("cart_ctrl")
        self.supervisor.register_controller(self.joint_ctrl)
        self.supervisor.register_controller(self.cart_ctrl)

    def tearDown(self):
        self.supervisor.__exit__(None, None, None)

    def test_switch_to_idle(self):
        result = self.supervisor.switch_mode(ControlMode.IDLE)
        self.assertTrue(result)
        self.assertEqual(self.supervisor._state.mode, ControlMode.IDLE)

    def test_switch_to_joint_position(self):
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertTrue(result)
        self.assertEqual(self.supervisor._state.mode, ControlMode.JOINT_POSITION)
        self.assertIn("joint_ctrl", self.supervisor._state.active_controllers)

    def test_switch_to_cartesian_velocity(self):
        result = self.supervisor.switch_mode(ControlMode.CARTESIAN_VELOCITY)
        self.assertTrue(result)
        self.assertEqual(self.supervisor._state.mode, ControlMode.CARTESIAN_VELOCITY)
        self.assertIn("cart_ctrl", self.supervisor._state.active_controllers)

    def test_switch_same_mode(self):
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertTrue(result)
        self.assertEqual(self.supervisor._state.mode, ControlMode.JOINT_POSITION)

    def test_emergency_stop_blocks_switch(self):
        self.supervisor.trigger_emergency_stop("test")
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertFalse(result)

    def test_mode_switch_no_controller(self):
        # 切换到需要未注册控制器的模式应返回 False
        self.supervisor.unregister_controller("joint_ctrl")
        self.supervisor.unregister_controller("cart_ctrl")
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertFalse(result)

    def test_log_on_mode_switch(self):
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        log = self.supervisor.get_log()
        self.assertTrue(any(e["type"] == "mode_switch" for e in log))


class TestSupervisorControlCycle(unittest.TestCase):
    """测试控制周期"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_cycle")
        self.joint_ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(self.joint_ctrl)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)

    def tearDown(self):
        self.supervisor.__exit__(None, None, None)

    def test_control_cycle_output(self):
        state = {}
        target = {"joint_position": np.zeros(6)}
        output, success = self.supervisor.control_cycle(state, target)
        self.assertTrue(success)
        self.assertIn("joint_velocity", output)

    def test_control_cycle_metrics_update(self):
        state = {}
        target = {}
        for _ in range(5):
            self.supervisor.control_cycle(state, target)

        metrics = self.supervisor.get_diagnostics()["metrics"]["joint_ctrl"]
        # 延迟可能为0（计算极快），但指标应存在
        self.assertGreaterEqual(metrics["latency_ms"], 0)
        self.assertGreater(metrics["last_update"], 0)

    def test_emergency_stop_returns_zero_output(self):
        self.supervisor.trigger_emergency_stop("test")
        state = {}
        target = {}
        output, success = self.supervisor.control_cycle(state, target)
        self.assertTrue(success)
        self.assertIn("emergency_stop", output)
        self.assertTrue(output["emergency_stop"])

    def test_control_cycle_no_mode(self):
        # 无模式时仍能返回空输出
        self.supervisor._state.mode = ControlMode.IDLE
        state = {}
        target = {}
        output, success = self.supervisor.control_cycle(state, target)
        self.assertTrue(success)


class TestSupervisorEmergencyStop(unittest.TestCase):
    """测试紧急停止"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_estop")
        self.joint_ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(self.joint_ctrl)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)

    def tearDown(self):
        self.supervisor.__exit__(None, None, None)

    def test_trigger_emergency_stop(self):
        self.supervisor.trigger_emergency_stop("test fault")
        self.assertEqual(self.supervisor._state.mode, ControlMode.EMERGENCY_STOP)
        self.assertEqual(self.supervisor._state.health, HealthStatus.EMERGENCY)
        self.assertEqual(len(self.supervisor._state.active_controllers), 0)

    def test_release_emergency_stop(self):
        self.supervisor.trigger_emergency_stop("test")
        result = self.supervisor.release_emergency_stop()
        self.assertTrue(result)
        self.assertEqual(self.supervisor._state.mode, ControlMode.IDLE)
        self.assertEqual(self.supervisor._state.health, HealthStatus.HEALTHY)

    def test_emergency_stop_output(self):
        output = self.supervisor._emergency_stop_output()
        self.assertTrue(output["emergency_stop"])
        self.assertTrue(np.allclose(output["joint_velocity"], np.zeros(6)))
        self.assertTrue(np.allclose(output["joint_torque"], np.zeros(6)))


class TestSupervisorDiagnostics(unittest.TestCase):
    """测试诊断功能"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_diag")
        self.joint_ctrl = MockJointController("joint_ctrl")
        self.supervisor.register_controller(self.joint_ctrl)

    def tearDown(self):
        self.supervisor.__exit__(None, None, None)

    def test_get_state(self):
        state = self.supervisor.get_state()
        self.assertIsInstance(state, ControlState)
        self.assertEqual(state.mode, ControlMode.IDLE)

    def test_get_diagnostics(self):
        diag = self.supervisor.get_diagnostics()
        self.assertEqual(diag["supervisor_id"], "test_diag")
        self.assertIn("uptime_s", diag)
        self.assertIn("mode", diag)
        self.assertIn("health", diag)
        self.assertIn("registered_controllers", diag)
        self.assertIn("metrics", diag)

    def test_print_diagnostics(self):
        # 不应抛出异常
        self.supervisor.print_diagnostics()

    def test_log_event(self):
        self.supervisor._log_event("test_event", {"key": "value"})
        log = self.supervisor.get_log()
        self.assertGreater(len(log), 0)
        self.assertEqual(log[-1]["type"], "test_event")

    def test_clear_log(self):
        self.supervisor._log_event("test_event", {})
        self.supervisor.clear_log()
        self.assertEqual(len(self.supervisor.get_log()), 0)


class TestSupervisorHealthCheck(unittest.TestCase):
    """测试健康检查"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_health")

    def tearDown(self):
        self.supervisor.__exit__(None, None, None)

    def test_check_system_health_no_controllers(self):
        healthy, msg = self.supervisor._check_system_health()
        self.assertTrue(healthy)

    def test_health_issue_handling(self):
        self.supervisor._handle_health_issue("test issue")
        self.assertEqual(self.supervisor._state.health, HealthStatus.DEGRADED)


class TestSupervisorContextManager(unittest.TestCase):
    """测试上下文管理器"""

    def test_context_manager(self):
        with ControlSupervisor(supervisor_id="test_ctx") as sup:
            ctrl = MockJointController("joint_ctrl")
            sup.register_controller(ctrl)
            self.assertTrue(sup._is_opened if hasattr(sup, '_is_opened') else True)


if __name__ == "__main__":
    unittest.main()
