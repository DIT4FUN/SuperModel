"""
GradeAwareSupervisor 测试
=========================

测试 AGV五级感知控制监管器
- SupervisorGradeSpec 五级规格
- GradeAwareSupervisor 各等级行为
- 看门狗功能 (XL/XXL级)
- 故障容忍 (XXL级)
- 冗余控制器注册
"""

import numpy as np
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.supervisor import (
    SupervisorGrade, SupervisorGradeSpec, get_supervisor_spec, get_supervisor_config,
    GradeAwareSupervisor, ControlSupervisor, ControllerInterface,
    ControlMode, SupervisorConfig, MockJointController, MockCartesianController
)


class TestSupervisorGradeSpec(unittest.TestCase):
    """测试 AGV五级监管器规格"""

    def test_all_grades_have_specs(self):
        for grade in SupervisorGrade:
            spec = get_supervisor_spec(grade)
            self.assertIsInstance(spec, SupervisorGradeSpec)
            self.assertEqual(spec.grade, grade)

    def test_s_grade_basic(self):
        spec = get_supervisor_spec(SupervisorGrade.S)
        self.assertEqual(spec.target_rate_hz, 50.0)
        self.assertEqual(spec.target_latency_ms, 20.0)
        self.assertFalse(spec.fault_prediction)
        self.assertFalse(spec.fault_recovery)
        self.assertEqual(spec.controller_redundancy, 0)
        self.assertFalse(spec.watchdog_enabled)
        self.assertEqual(spec.emergency_stop_level, 1)

    def test_m_grade_standard(self):
        spec = get_supervisor_spec(SupervisorGrade.M)
        self.assertEqual(spec.target_rate_hz, 100.0)
        self.assertEqual(spec.target_latency_ms, 10.0)
        self.assertTrue(spec.fault_isolation)
        self.assertTrue(spec.fault_recovery)
        self.assertTrue(spec.graceful_degradation)
        self.assertFalse(spec.watchdog_enabled)

    def test_l_grade_enhanced(self):
        spec = get_supervisor_spec(SupervisorGrade.L)
        self.assertEqual(spec.target_rate_hz, 200.0)
        self.assertEqual(spec.target_latency_ms, 5.0)
        self.assertTrue(spec.fault_prediction)
        self.assertEqual(spec.controller_redundancy, 1)

    def test_xl_grade_high_performance(self):
        spec = get_supervisor_spec(SupervisorGrade.XL)
        self.assertEqual(spec.target_rate_hz, 500.0)
        self.assertEqual(spec.target_latency_ms, 2.0)
        self.assertTrue(spec.watchdog_enabled)
        self.assertEqual(spec.watchdog_timeout_ms, 5.0)
        self.assertEqual(spec.controller_redundancy, 2)
        self.assertTrue(spec.hot_swap)
        self.assertEqual(spec.emergency_stop_level, 4)

    def test_xxl_grade_critical(self):
        spec = get_supervisor_spec(SupervisorGrade.XXL)
        self.assertEqual(spec.target_rate_hz, 1000.0)
        self.assertEqual(spec.target_latency_ms, 1.0)
        self.assertEqual(spec.max_latency_ms, 5.0)
        self.assertTrue(spec.watchdog_enabled)
        self.assertEqual(spec.watchdog_timeout_ms, 2.0)
        self.assertEqual(spec.controller_redundancy, 3)
        self.assertEqual(spec.safety_level, "critical")
        self.assertEqual(spec.emergency_stop_level, 5)

    def test_rate_increases_with_grade(self):
        prev_rate = 0
        for grade in [SupervisorGrade.S, SupervisorGrade.M, SupervisorGrade.L,
                      SupervisorGrade.XL, SupervisorGrade.XXL]:
            spec = get_supervisor_spec(grade)
            self.assertGreater(spec.target_rate_hz, prev_rate)
            prev_rate = spec.target_rate_hz

    def test_latency_decreases_with_grade(self):
        prev_latency = float('inf')
        for grade in [SupervisorGrade.S, SupervisorGrade.M, SupervisorGrade.L,
                      SupervisorGrade.XL, SupervisorGrade.XXL]:
            spec = get_supervisor_spec(grade)
            self.assertLess(spec.target_latency_ms, prev_latency)
            prev_latency = spec.target_latency_ms

    def test_redundancy_increases_with_grade(self):
        prev_redundancy = -1
        for grade in [SupervisorGrade.S, SupervisorGrade.M, SupervisorGrade.L,
                      SupervisorGrade.XL, SupervisorGrade.XXL]:
            spec = get_supervisor_spec(grade)
            self.assertGreaterEqual(spec.controller_redundancy, prev_redundancy)
            prev_redundancy = spec.controller_redundancy


class TestGetSupervisorConfig(unittest.TestCase):
    """测试从等级获取配置"""

    def test_config_matches_spec(self):
        for grade in SupervisorGrade:
            spec = get_supervisor_spec(grade)
            config = get_supervisor_config(grade)
            self.assertEqual(config.target_rate_hz, spec.target_rate_hz)
            self.assertEqual(config.target_latency_ms, spec.target_latency_ms)
            self.assertEqual(config.max_latency_ms, spec.max_latency_ms)

    def test_s_grade_loose_thresholds(self):
        config = get_supervisor_config(SupervisorGrade.S)
        self.assertGreater(config.max_tracking_error, 0.3)
        self.assertGreater(config.mode_switch_timeout_s, 3.0)

    def test_xxl_grade_strict_thresholds(self):
        config = get_supervisor_config(SupervisorGrade.XXL)
        self.assertLess(config.max_tracking_error, 0.2)
        self.assertLess(config.mode_switch_timeout_s, 0.5)


class TestGradeAwareSupervisorInit(unittest.TestCase):
    """测试 GradeAwareSupervisor 初始化"""

    def test_init_default_grade(self):
        sup = GradeAwareSupervisor()
        self.assertEqual(sup.grade, SupervisorGrade.M)
        self.assertIsNotNone(sup.grade_spec)
        self.assertIsNotNone(sup.config)

    def test_init_all_grades(self):
        for grade in SupervisorGrade:
            sup = GradeAwareSupervisor(grade=grade, supervisor_id=f"test_{grade.value}")
            self.assertEqual(sup.grade, grade)

    def test_watchdog_enabled_xl_xxl_only(self):
        for grade in SupervisorGrade:
            sup = GradeAwareSupervisor(grade=grade)
            if grade in [SupervisorGrade.XL, SupervisorGrade.XXL]:
                self.assertTrue(sup._watchdog_enabled)
            else:
                self.assertFalse(sup._watchdog_enabled)

    def test_hot_standby_xl_xxl_only(self):
        for grade in SupervisorGrade:
            sup = GradeAwareSupervisor(grade=grade)
            if grade in [SupervisorGrade.XL, SupervisorGrade.XXL]:
                self.assertTrue(sup._hot_standby)
            else:
                self.assertFalse(sup._hot_standby)

    def test_fault_tolerance_xxl_only(self):
        for grade in SupervisorGrade:
            sup = GradeAwareSupervisor(grade=grade)
            if grade == SupervisorGrade.XXL:
                self.assertTrue(sup._fault_tolerance_enabled)
            else:
                self.assertFalse(sup._fault_tolerance_enabled)


class TestGradeAwareSupervisorRegistration(unittest.TestCase):
    """测试冗余控制器注册"""

    def setUp(self):
        self.sup = GradeAwareSupervisor(grade=SupervisorGrade.XL)

    def test_register_with_redundancy_primary(self):
        ctrl = MockJointController("primary_joint")
        modes = [ControlMode.JOINT_POSITION, ControlMode.JOINT_VELOCITY]
        result = self.sup.register_with_redundancy(ctrl, modes, is_primary=True)
        self.assertTrue(result)
        self.assertEqual(self.sup._primary_controllers["joint_position"], "primary_joint")

    def test_register_with_redundancy_backup(self):
        primary = MockJointController("primary_joint")
        backup = MockJointController("backup_joint")
        modes = [ControlMode.JOINT_POSITION]
        self.sup.register_with_redundancy(primary, modes, is_primary=True)
        result = self.sup.register_with_redundancy(backup, modes, is_primary=False)
        self.assertTrue(result)
        self.assertIn("backup_joint", self.sup._backup_controllers["joint_position"])

    def test_register_with_redundancy_multiple_modes(self):
        ctrl = MockCartesianController("cartesian_ctrl")
        modes = [ControlMode.CARTESIAN_VELOCITY, ControlMode.CARTESIAN_POSITION]
        self.sup.register_with_redundancy(ctrl, modes, is_primary=True)
        self.assertEqual(self.sup._primary_controllers["cartesian_velocity"], "cartesian_ctrl")
        self.assertEqual(self.sup._primary_controllers["cartesian_position"], "cartesian_ctrl")


class TestGradeAwareWatchdog(unittest.TestCase):
    """测试看门狗功能 (XL/XXL级)"""

    def setUp(self):
        self.sup = GradeAwareSupervisor(grade=SupervisorGrade.XL)

    def test_watchdog_not_enabled_for_lower_grades(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.M)
        self.assertFalse(sup._watchdog_enabled)

    def test_kick_watchdog_updates_timer(self):
        import time
        self.sup.kick_watchdog("test_ctrl")
        self.assertIn("test_ctrl", self.sup._watchdog_timers)

    def test_watchdog_check_no_timeout(self):
        healthy, msg = self.sup._check_watchdog()
        self.assertTrue(healthy)

    def test_watchdog_step_returns_true_when_enabled(self):
        result = self.sup.step_watchdog()
        self.assertTrue(result)

    def test_watchdog_step_xxl(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.XXL)
        result = sup.step_watchdog()
        self.assertTrue(result)


class TestGradeAwareFaultTolerance(unittest.TestCase):
    """测试故障容忍 (XXL级)"""

    def test_no_fault_tolerance_for_lower_grades(self):
        for grade in [SupervisorGrade.S, SupervisorGrade.M, SupervisorGrade.L, SupervisorGrade.XL]:
            sup = GradeAwareSupervisor(grade=grade)
            self.assertFalse(sup._fault_tolerance_enabled)

    def test_fault_tolerance_enabled_xxl(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.XXL)
        self.assertTrue(sup._fault_tolerance_enabled)

    def test_step_fault_tolerance_no_fault(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.XXL)
        result = sup.step_fault_tolerance(fault_detected=False)
        self.assertTrue(result)
        self.assertEqual(sup._consecutive_faults, 0)

    def test_step_fault_tolerance_with_fault(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.XXL)
        for _ in range(5):
            sup.step_fault_tolerance(fault_detected=True)
        self.assertGreater(sup._consecutive_faults, 0)

    def test_step_fault_tolerance_recovery(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.XXL)
        sup.step_fault_tolerance(fault_detected=True)
        sup.step_fault_tolerance(fault_detected=False)
        self.assertEqual(sup._consecutive_faults, 0)


class TestGradeCapabilities(unittest.TestCase):
    """测试等级能力查询"""

    def test_capabilities_structure(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.L)
        caps = sup.get_grade_capabilities()
        self.assertIn("grade", caps)
        self.assertIn("performance", caps)
        self.assertIn("fault_handling", caps)
        self.assertIn("safety", caps)
        self.assertIn("redundancy", caps)
        self.assertIn("watchdog", caps)
        self.assertIn("diagnostics", caps)

    def test_xxl_capabilities_comprehensive(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.XXL)
        caps = sup.get_grade_capabilities()
        self.assertEqual(caps["grade"], "XXL")
        self.assertTrue(caps["fault_handling"]["prediction"])
        self.assertTrue(caps["redundancy"]["hot_swap"])
        self.assertEqual(caps["redundancy"]["controller_count"], 3)

    def test_s_capabilities_basic(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.S)
        caps = sup.get_grade_capabilities()
        self.assertEqual(caps["grade"], "S")
        self.assertFalse(caps["fault_handling"]["prediction"])
        self.assertEqual(caps["redundancy"]["controller_count"], 0)


class TestGradeAwareSupervisorModeSwitch(unittest.TestCase):
    """测试模式切换 (继承父类)"""

    def setUp(self):
        self.sup = GradeAwareSupervisor(grade=SupervisorGrade.M)

    def test_mode_switch_idle(self):
        result = self.sup.switch_mode(ControlMode.IDLE)
        self.assertTrue(result)

    def test_mode_switch_to_joint_position(self):
        ctrl = MockJointController("joint_ctrl")
        self.sup.register_controller(ctrl)
        result = self.sup.switch_mode(ControlMode.JOINT_POSITION)
        self.assertTrue(result)
        self.assertEqual(self.sup._state.mode, ControlMode.JOINT_POSITION)


class TestGradeAwareEmergencyStop(unittest.TestCase):
    """测试紧急停止"""

    def test_emergency_stop_all_grades(self):
        for grade in SupervisorGrade:
            sup = GradeAwareSupervisor(grade=grade)
            sup.trigger_emergency_stop("test")
            self.assertEqual(sup._state.health.value, "emergency")
            self.assertEqual(sup._state.mode, ControlMode.EMERGENCY_STOP)

    def test_emergency_stop_recovery(self):
        sup = GradeAwareSupervisor(grade=SupervisorGrade.M)
        sup.trigger_emergency_stop("test")
        result = sup.release_emergency_stop()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
