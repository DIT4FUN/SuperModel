"""
Control Supervisor 测试
========================

测试控制子系统监管模块
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.control.supervisor import (
    ControlSupervisor, ControllerInterface, SupervisorConfig,
    ControlState, ControllerMetrics, ControlMode, HealthStatus,
    MockJointController, MockCartesianController, MockImpedanceController
)


class TestControllerInterface(unittest.TestCase):
    """测试控制器标准接口"""

    def test_mock_joint_controller_create(self):
        ctrl = MockJointController("test_joint")
        self.assertEqual(ctrl.name, "test_joint")
        self.assertEqual(ctrl.controller_type, "joint_position")
        self.assertFalse(ctrl.is_active)

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

    def test_controller_start_stop(self):
        ctrl = MockJointController("test_ctrl")
        self.assertTrue(ctrl.start())
        self.assertTrue(ctrl.is_active)
        self.assertTrue(ctrl.stop())
        self.assertFalse(ctrl.is_active)

    def test_controller_compute(self):
        ctrl = MockJointController("test_ctrl")
        ctrl.start()
        state = {}
        target = {}
        output = ctrl.compute(state, target)
        self.assertIn("joint_velocity", output)
        self.assertIn("joint_torque", output)
        self.assertEqual(len(output["joint_velocity"]), 6)
        ctrl.stop()

    def test_controller_metrics(self):
        ctrl = MockJointController("test_ctrl")
        metrics = ctrl.get_metrics()
        self.assertIsInstance(metrics, ControllerMetrics)
        self.assertEqual(metrics.name, "test_ctrl")

    def test_controller_health_check(self):
        ctrl = MockJointController("test_ctrl")
        healthy, msg = ctrl.health_check()
        self.assertTrue(healthy)
        self.assertEqual(msg, "OK")


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
            fault_count_threshold=5
        )
        self.assertEqual(config.mode_switch_timeout_s, 5.0)
        self.assertEqual(config.max_latency_ms, 100.0)
        self.assertEqual(config.fault_count_threshold, 5)


class TestControlSupervisor(unittest.TestCase):
    """测试控制监管器"""

    def setUp(self):
        self.supervisor = ControlSupervisor(supervisor_id="test_supervisor")
        self.mock_joint = MockJointController("joint_ctrl")
        self.mock_cartesian = MockCartesianController("cartesian_ctrl")
        self.mock_impedance = MockImpedanceController("impedance_ctrl")

    def tearDown(self):
        # Clean up
        for name in list(self.supervisor.list_controllers()):
            self.supervisor.unregister_controller(name)

    def test_supervisor_create(self):
        self.assertEqual(self.supervisor.supervisor_id, "test_supervisor")
        state = self.supervisor.get_state()
        self.assertIsInstance(state, ControlState)
        self.assertEqual(state.mode, ControlMode.IDLE)
        self.assertEqual(state.health, HealthStatus.HEALTHY)
        self.assertEqual(len(state.active_controllers), 0)

    def test_register_controller(self):
        result = self.supervisor.register_controller(self.mock_joint)
        self.assertTrue(result)
        controllers = self.supervisor.list_controllers()
        self.assertIn("joint_ctrl", controllers)

    def test_register_duplicate_controller(self):
        self.supervisor.register_controller(self.mock_joint)
        result = self.supervisor.register_controller(self.mock_joint)
        self.assertFalse(result)

    def test_unregister_controller(self):
        self.supervisor.register_controller(self.mock_joint)
        result = self.supervisor.unregister_controller("joint_ctrl")
        self.assertTrue(result)
        self.assertNotIn("joint_ctrl", self.supervisor.list_controllers())

    def test_unregister_nonexistent(self):
        result = self.supervisor.unregister_controller("nonexistent")
        self.assertFalse(result)

    def test_get_controller(self):
        self.supervisor.register_controller(self.mock_joint)
        ctrl = self.supervisor.get_controller("joint_ctrl")
        self.assertIsNotNone(ctrl)
        self.assertEqual(ctrl.name, "joint_ctrl")

    def test_get_nonexistent_controller(self):
        ctrl = self.supervisor.get_controller("nonexistent")
        self.assertIsNone(ctrl)

    def test_list_controllers(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.register_controller(self.mock_cartesian)
        controllers = self.supervisor.list_controllers()
        self.assertEqual(len(controllers), 2)
        self.assertIn("joint_ctrl", controllers)
        self.assertIn("cartesian_ctrl", controllers)

    def test_mode_switch_to_joint_position(self):
        self.supervisor.register_controller(self.mock_joint)
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertTrue(result)
        state = self.supervisor.get_state()
        self.assertEqual(state.mode, ControlMode.JOINT_POSITION)
        self.assertIn("joint_ctrl", state.active_controllers)

    def test_mode_switch_to_cartesian_velocity(self):
        self.supervisor.register_controller(self.mock_cartesian)
        result = self.supervisor.switch_mode(ControlMode.CARTESIAN_VELOCITY)
        self.assertTrue(result)
        state = self.supervisor.get_state()
        self.assertEqual(state.mode, ControlMode.CARTESIAN_VELOCITY)
        self.assertIn("cartesian_ctrl", state.active_controllers)

    def test_mode_switch_same_mode(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertTrue(result)

    def test_mode_switch_no_controller(self):
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertFalse(result)

    def test_mode_switch_from_emergency_stop(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.trigger_emergency_stop("test fault")
        result = self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.assertFalse(result)
        state = self.supervisor.get_state()
        self.assertEqual(state.mode, ControlMode.EMERGENCY_STOP)

    def test_control_cycle(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)

        state = {}
        target = {}
        output, success = self.supervisor.control_cycle(state, target)

        self.assertTrue(success)
        self.assertIn("joint_velocity", output)

    def test_control_cycle_emergency_stop(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.trigger_emergency_stop("test fault")

        state = {}
        target = {}
        output, success = self.supervisor.control_cycle(state, target)

        self.assertTrue(success)
        self.assertIn("emergency_stop", output)
        self.assertTrue(output["emergency_stop"])

    def test_control_cycle_no_mode(self):
        state = {}
        target = {}
        output, success = self.supervisor.control_cycle(state, target)
        self.assertTrue(success)

    def test_emergency_stop(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)

        self.supervisor.trigger_emergency_stop("critical fault")

        state = self.supervisor.get_state()
        self.assertEqual(state.mode, ControlMode.EMERGENCY_STOP)
        self.assertEqual(state.health, HealthStatus.EMERGENCY)
        self.assertEqual(len(state.active_controllers), 0)

    def test_emergency_stop_output(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        self.supervisor.trigger_emergency_stop("test")

        state = {}
        target = {}
        output, _ = self.supervisor.control_cycle(state, target)

        self.assertTrue(output["emergency_stop"])
        self.assertTrue(np.allclose(output["joint_velocity"], np.zeros(6)))
        self.assertTrue(np.allclose(output["joint_torque"], np.zeros(6)))

    def test_release_emergency_stop(self):
        self.supervisor.trigger_emergency_stop("test")
        result = self.supervisor.release_emergency_stop()
        self.assertTrue(result)
        state = self.supervisor.get_state()
        self.assertEqual(state.mode, ControlMode.IDLE)
        self.assertEqual(state.health, HealthStatus.HEALTHY)

    def test_get_diagnostics(self):
        self.supervisor.register_controller(self.mock_joint)
        diag = self.supervisor.get_diagnostics()

        self.assertEqual(diag["supervisor_id"], "test_supervisor")
        self.assertEqual(diag["mode"], "idle")
        self.assertEqual(diag["health"], "healthy")
        self.assertIn("joint_ctrl", diag["registered_controllers"])
        self.assertIn("metrics", diag)

    def test_get_diagnostics_metrics(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)

        # Run a few cycles to update metrics
        for _ in range(3):
            self.supervisor.control_cycle({}, {})

        diag = self.supervisor.get_diagnostics()
        metrics = diag["metrics"]["joint_ctrl"]
        self.assertIn("latency_ms", metrics)
        self.assertIn("tracking_error", metrics)
        self.assertIn("success_rate", metrics)

    def test_print_diagnostics(self):
        self.supervisor.register_controller(self.mock_joint)
        # Should not raise
        self.supervisor.print_diagnostics()

    def test_log_event(self):
        self.supervisor._log_event("test_event", {"key": "value"})
        log = self.supervisor.get_log(max_entries=1)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["type"], "test_event")

    def test_clear_log(self):
        self.supervisor._log_event("event1", {})
        self.supervisor._log_event("event2", {})
        self.assertGreater(len(self.supervisor.get_log()), 0)
        self.supervisor.clear_log()
        self.assertEqual(len(self.supervisor.get_log()), 0)

    def test_context_manager(self):
        with ControlSupervisor(supervisor_id="test_ctx") as sup:
            sup.register_controller(MockJointController("ctrl1"))
            state = sup.get_state()
            self.assertIsNotNone(state)

    def test_multiple_controllers_same_type(self):
        """测试同类型多控制器注册"""
        ctrl1 = MockJointController("joint_1")
        ctrl2 = MockJointController("joint_2")
        self.supervisor.register_controller(ctrl1)
        self.supervisor.register_controller(ctrl2)
        self.assertEqual(len(self.supervisor.list_controllers()), 2)

    def test_find_controller_for_mode(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.register_controller(self.mock_cartesian)
        self.supervisor.register_controller(self.mock_impedance)

        # Test different modes
        result = self.supervisor._find_controller_for_mode(ControlMode.JOINT_POSITION)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "joint_ctrl")

        result = self.supervisor._find_controller_for_mode(ControlMode.CARTESIAN_VELOCITY)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "cartesian_ctrl")

        result = self.supervisor._find_controller_for_mode(ControlMode.IMPEDANCE)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "impedance_ctrl")

        result = self.supervisor._find_controller_for_mode(ControlMode.TELEOP)
        self.assertIsNone(result)

    def test_uptime_tracking(self):
        self.supervisor.register_controller(self.mock_joint)
        self.supervisor.control_cycle({}, {})
        state = self.supervisor.get_state()
        self.assertGreaterEqual(state.uptime_s, 0)


class TestControlSupervisorFaultHandling(unittest.TestCase):
    """测试监管器故障处理"""

    def setUp(self):
        self.config = SupervisorConfig(
            enable_fault_recovery=True,
            graceful_degradation=True,
            fault_count_threshold=2
        )
        self.supervisor = ControlSupervisor(
            config=self.config,
            supervisor_id="fault_test"
        )
        self.mock_joint = MockJointController("faulty_joint")
        self.supervisor.register_controller(self.mock_joint)

    def tearDown(self):
        for name in list(self.supervisor.list_controllers()):
            self.supervisor.unregister_controller(name)

    def test_fault_history_recorded(self):
        self.supervisor.trigger_emergency_stop("critical fault")
        state = self.supervisor.get_state()
        self.assertGreater(len(state.fault_history), 0)
        ts, msg = state.fault_history[-1]
        self.assertIn("critical fault", msg)

    def test_log_on_register(self):
        log = self.supervisor.get_log(max_entries=10)
        register_events = [e for e in log if e["type"] == "controller_registered"]
        self.assertGreater(len(register_events), 0)

    def test_log_on_mode_switch(self):
        self.supervisor.switch_mode(ControlMode.JOINT_POSITION)
        log = self.supervisor.get_log(max_entries=10)
        switch_events = [e for e in log if e["type"] == "mode_switch"]
        self.assertGreater(len(switch_events), 0)


class TestControlMode(unittest.TestCase):
    """测试控制模式枚举"""

    def test_control_mode_values(self):
        expected_modes = [
            "idle", "joint_position", "joint_velocity", "joint_torque",
            "cartesian_velocity", "cartesian_position", "impedance",
            "force", "admittance", "teleop", "autonomous", "emergency_stop"
        ]
        actual = [m.value for m in ControlMode]
        for em in expected_modes:
            self.assertIn(em, actual)


class TestHealthStatus(unittest.TestCase):
    """测试健康状态枚举"""

    def test_health_status_values(self):
        expected = ["healthy", "degraded", "fault", "emergency"]
        actual = [s.value for s in HealthStatus]
        for e in expected:
            self.assertIn(e, actual)


if __name__ == '__main__':
    unittest.main()
