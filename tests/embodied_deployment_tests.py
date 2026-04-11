"""
具身智能部署管理模块测试
测试 DeploymentValidator, HealthMonitor, EmergencyProcedure, DeploymentManager
"""

import unittest
import numpy as np
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embodied.deployment import (
    DeploymentState,
    HealthStatus,
    DeploymentConfig,
    HealthCheckResult,
    DeploymentValidator,
    HealthMonitor,
    EmergencyProcedure,
    DeploymentManager,
    create_deployment_manager,
)


class TestDeploymentConfig(unittest.TestCase):
    """部署配置测试"""

    def test_default_config(self):
        cfg = DeploymentConfig()
        self.assertEqual(cfg.grade, "M")
        self.assertTrue(cfg.emergency_stop_enabled)
        self.assertEqual(cfg.health_check_interval_s, 5.0)

    def test_grade_config(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            cfg = DeploymentConfig(grade=grade)
            self.assertEqual(cfg.grade, grade)

    def test_custom_config(self):
        cfg = DeploymentConfig(
            grade="L",
            enable_vision=True,
            enable_tactile=False,
            enable_force=False,
            health_check_interval_s=10.0,
        )
        self.assertEqual(cfg.grade, "L")
        self.assertFalse(cfg.enable_tactile)


class TestDeploymentValidator(unittest.TestCase):
    """部署验证器测试"""

    def setUp(self):
        self.config = DeploymentConfig(grade="M")
        self.validator = DeploymentValidator(self.config)

    def test_validate_config_valid(self):
        results = self.validator.validate_config()
        self.assertGreater(len(results), 0)
        # 至少config组件应该healthy
        config_result = next(r for r in results if r.component == "config")
        self.assertEqual(config_result.status, HealthStatus.HEALTHY)

    def test_validate_config_invalid_grade(self):
        self.config.grade = "INVALID"
        validator = DeploymentValidator(self.config)
        results = validator.validate_config()
        config_result = next(r for r in results if r.component == "config")
        self.assertEqual(config_result.status, HealthStatus.FAILED)

    def test_validate_config_no_sensors(self):
        self.config.enable_tactile = False
        self.config.enable_force = False
        validator = DeploymentValidator(self.config)
        results = validator.validate_config()
        warning_result = next((r for r in results if 'Neither tactile' in r.message), None)
        self.assertIsNotNone(warning_result)
        self.assertEqual(warning_result.status, HealthStatus.WARNING)

    def test_validate_config_emergency_disabled(self):
        self.config.emergency_stop_enabled = False
        validator = DeploymentValidator(self.config)
        results = validator.validate_config()
        warning_result = next((r for r in results if 'Emergency stop' in r.message), None)
        self.assertIsNotNone(warning_result)
        self.assertEqual(warning_result.status, HealthStatus.WARNING)

    def test_validate_grade_specs(self):
        results = self.validator.validate_grade_specs()
        self.assertGreater(len(results), 0)
        grade_result = results[0]
        self.assertEqual(grade_result.component, "grade_spec")
        self.assertEqual(grade_result.status, HealthStatus.HEALTHY)
        self.assertIn("control_freq", grade_result.details)
        self.assertIn("max_speed", grade_result.details)


class TestHealthMonitor(unittest.TestCase):
    """健康监控器测试"""

    def setUp(self):
        self.config = DeploymentConfig(health_history_size=50)
        self.monitor = HealthMonitor(self.config)

    def test_initial_state(self):
        self.assertEqual(self.monitor.get_state(), DeploymentState.IDLE)
        self.assertEqual(self.monitor.get_overall_status(), HealthStatus.UNKNOWN)

    def test_report_health_healthy(self):
        result = HealthCheckResult(
            component="test_sensor",
            status=HealthStatus.HEALTHY,
            message="All good",
        )
        self.monitor.report_health(result)
        self.assertEqual(self.monitor.get_overall_status(), HealthStatus.HEALTHY)

    def test_report_health_warning(self):
        result = HealthCheckResult(
            component="test_sensor",
            status=HealthStatus.WARNING,
            message="Minor issue",
        )
        self.monitor.report_health(result)
        self.assertEqual(self.monitor.get_overall_status(), HealthStatus.WARNING)

    def test_report_health_critical(self):
        for _ in range(10):
            self.monitor.report_health(HealthCheckResult(
                component="test_sensor",
                status=HealthStatus.HEALTHY,
            ))
        for _ in range(5):
            self.monitor.report_health(HealthCheckResult(
                component="test_sensor",
                status=HealthStatus.CRITICAL,
            ))
        self.assertEqual(self.monitor.get_overall_status(), HealthStatus.CRITICAL)

    def test_set_state(self):
        self.monitor.set_state(DeploymentState.RUNNING)
        self.assertEqual(self.monitor.get_state(), DeploymentState.RUNNING)

    def test_callback_triggered(self):
        callback_results = []
        def cb(r):
            callback_results.append(r)
        self.monitor.add_callback(cb)
        self.monitor.report_health(HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
        ))
        self.assertEqual(len(callback_results), 1)

    def test_health_summary(self):
        self.monitor.report_health(HealthCheckResult(
            component="imu",
            status=HealthStatus.HEALTHY,
        ))
        summary = self.monitor.get_health_summary()
        self.assertIn("state", summary)
        self.assertIn("overall_status", summary)
        self.assertIn("components", summary)


class TestEmergencyProcedure(unittest.TestCase):
    """紧急停车程序测试"""

    def setUp(self):
        self.config = DeploymentConfig()
        self.emergency = EmergencyProcedure(self.config)

    def test_initial_state(self):
        info = self.emergency.get_stop_info()
        self.assertIsNone(info["reason"])
        self.assertIsNone(info["time"])

    def test_trigger(self):
        self.emergency.trigger("Collision detected")
        info = self.emergency.get_stop_info()
        self.assertEqual(info["reason"], "Collision detected")
        self.assertIsNotNone(info["time"])

    def test_stop_callback(self):
        called = []
        def cb():
            called.append(True)
        self.emergency.register_stop_callback(cb)
        self.emergency.trigger("Test stop")
        self.assertEqual(len(called), 1)

    def test_check_safety_conditions_safe(self):
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 5.0, "pitch_deg": 3.0},
            force_data={"total_magnitude_N": 10.0},
            tactile_data={"max_pressure_pct": 50.0},
        )
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

    def test_check_safety_conditions_tilt(self):
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 50.0, "pitch_deg": 10.0},
        )
        self.assertFalse(is_safe)
        self.assertIn("tilted", reason)

    def test_check_safety_conditions_collision(self):
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            force_data={"total_magnitude_N": 100.0},
        )
        self.assertFalse(is_safe)
        self.assertIn("Collision", reason)

    def test_check_safety_conditions_pressure(self):
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            tactile_data={"max_pressure_pct": 98.0},
        )
        self.assertFalse(is_safe)
        self.assertIn("pressure", reason)


class TestDeploymentManager(unittest.TestCase):
    """部署管理器测试"""

    def setUp(self):
        self.manager = create_deployment_manager(grade="M")

    def test_pre_deployment_check(self):
        all_passed, results = self.manager.pre_deployment_check()
        self.assertTrue(all_passed or len(results) > 0)
        config_result = next((r for r in results if r.component == "config"), None)
        self.assertIsNotNone(config_result)

    def test_deploy_success(self):
        success = self.manager.deploy()
        self.assertTrue(success)
        self.assertEqual(self.manager.health_monitor.get_state(), DeploymentState.RUNNING)

    def test_shutdown(self):
        self.manager.deploy()
        self.manager.shutdown()
        self.assertEqual(self.manager.health_monitor.get_state(), DeploymentState.SHUTDOWN)

    def test_get_status(self):
        status = self.manager.get_status()
        self.assertIn("state", status)
        self.assertIn("health_summary", status)
        self.assertIn("emergency_info", status)

    def test_manager_with_different_grades(self):
        for grade in ["S", "L", "XL"]:
            manager = create_deployment_manager(grade=grade)
            all_passed, _ = manager.pre_deployment_check()
            self.assertTrue(all_passed)


class TestHealthCheckResult(unittest.TestCase):
    """健康检查结果测试"""

    def test_is_healthy(self):
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
        )
        self.assertTrue(result.is_healthy())
        self.assertFalse(result.is_critical())

    def test_is_critical(self):
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.CRITICAL,
        )
        self.assertFalse(result.is_healthy())
        self.assertTrue(result.is_critical())

    def test_warning_not_critical(self):
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.WARNING,
        )
        self.assertTrue(result.is_healthy())
        self.assertFalse(result.is_critical())


if __name__ == "__main__":
    unittest.main()
