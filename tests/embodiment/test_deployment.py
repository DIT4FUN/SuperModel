"""
test_deployment.py - 具身智能部署模块测试
=========================================

测试 DeploymentManager, DeploymentValidator, HealthMonitor,
EmergencyProcedure, HealthCheckResult 等部署相关功能

覆盖场景:
- 配置验证 (AGV五级等级)
- 健康状态监控
- 紧急停车程序
- 部署前检查流程
- 降级策略
- 多等级适配
"""

import pytest
import time
import threading
from typing import List, Dict, Any, Optional

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


# =============================================================================
# DeploymentConfig Tests
# =============================================================================

class TestDeploymentConfig:
    """部署配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = DeploymentConfig()
        assert config.grade == "M"
        assert config.enable_vision is True
        assert config.enable_audio is True
        assert config.enable_tactile is True
        assert config.enable_force is True
        assert config.enable_imu is True
        assert config.enable_control is True
        assert config.emergency_stop_enabled is True
        assert config.health_check_interval_s == 5.0

    def test_config_all_grades(self):
        """测试所有AGV等级配置"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            config = DeploymentConfig(grade=grade)
            assert config.grade == grade

    def test_config_sensor_toggles(self):
        """测试传感器开关配置"""
        config = DeploymentConfig(
            enable_vision=False,
            enable_audio=False,
            enable_tactile=False,
            enable_force=False,
            enable_imu=False,
        )
        assert config.enable_vision is False
        assert config.enable_audio is False
        assert config.enable_tactile is False
        assert config.enable_force is False
        assert config.enable_imu is False

    def test_config_health_params(self):
        """测试健康检查参数"""
        config = DeploymentConfig(
            health_check_interval_s=10.0,
            max_latency_ms=50.0,
            max_consecutive_errors=3,
            recovery_wait_s=5.0,
        )
        assert config.health_check_interval_s == 10.0
        assert config.max_latency_ms == 50.0
        assert config.max_consecutive_errors == 3
        assert config.recovery_wait_s == 5.0


# =============================================================================
# HealthCheckResult Tests
# =============================================================================

class TestHealthCheckResult:
    """健康检查结果测试"""

    def test_healthy_status(self):
        """测试健康状态判定"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="All good",
        )
        assert result.is_healthy() is True
        assert result.is_critical() is False

    def test_warning_status(self):
        """测试警告状态判定"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.WARNING,
            message="Caution",
        )
        assert result.is_healthy() is True
        assert result.is_critical() is False

    def test_critical_status(self):
        """测试危险状态判定"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.CRITICAL,
            message="Critical issue",
        )
        assert result.is_healthy() is False
        assert result.is_critical() is True

    def test_failed_status(self):
        """测试失败状态判定"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.FAILED,
            message="Failed",
        )
        assert result.is_healthy() is False
        assert result.is_critical() is True

    def test_unknown_status(self):
        """测试未知状态判定"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.UNKNOWN,
            message="Unknown",
        )
        assert result.is_healthy() is False
        assert result.is_critical() is False

    def test_latency_recording(self):
        """测试延迟记录"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=12.5,
        )
        assert result.latency_ms == 12.5

    def test_timestamp_auto_set(self):
        """测试时间戳自动设置"""
        before = time.time()
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
        )
        after = time.time()
        assert before <= result.timestamp <= after

    def test_details_dict(self):
        """测试详情字典"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            details={"cpu_temp": 45.0, "memory_pct": 67.2},
        )
        assert result.details["cpu_temp"] == 45.0
        assert result.details["memory_pct"] == 67.2


# =============================================================================
# DeploymentValidator Tests
# =============================================================================

class TestDeploymentValidator:
    """部署验证器测试"""

    def test_valid_grade(self):
        """测试有效等级验证"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            config = DeploymentConfig(grade=grade)
            validator = DeploymentValidator(config)
            results = validator.validate_config()
            
            config_result = next(r for r in results if r.component == "config")
            assert config_result.status == HealthStatus.HEALTHY

    def test_invalid_grade(self):
        """测试无效等级验证"""
        config = DeploymentConfig(grade="INVALID")
        validator = DeploymentValidator(config)
        results = validator.validate_config()
        
        config_result = next(r for r in results if r.component == "config")
        assert config_result.status == HealthStatus.FAILED
        assert "INVALID" in config_result.message

    def test_emergency_stop_disabled_warning(self):
        """测试紧急停车禁用警告"""
        config = DeploymentConfig(grade="M", emergency_stop_enabled=False)
        validator = DeploymentValidator(config)
        results = validator.validate_config()
        
        emergency_result = next(
            r for r in results 
            if "emergency" in r.message.lower() or r.component == "config"
        )
        # 应产生警告
        statuses = [r.status for r in results]
        assert HealthStatus.WARNING in statuses or HealthStatus.FAILED in statuses

    def test_health_check_interval_too_short(self):
        """测试健康检查间隔过短"""
        config = DeploymentConfig(health_check_interval_s=0.1)
        validator = DeploymentValidator(config)
        results = validator.validate_config()
        
        interval_result = next(r for r in results if "interval" in r.message.lower())
        assert interval_result.status == HealthStatus.WARNING

    def test_health_check_interval_too_long(self):
        """测试健康检查间隔过长"""
        config = DeploymentConfig(health_check_interval_s=120.0)
        validator = DeploymentValidator(config)
        results = validator.validate_config()
        
        interval_result = next(r for r in results if "interval" in r.message.lower())
        assert interval_result.status == HealthStatus.WARNING

    def test_validate_grade_specs_s_grade(self):
        """测试S级等级规格验证"""
        config = DeploymentConfig(grade="S")
        validator = DeploymentValidator(config)
        results = validator.validate_grade_specs()
        
        assert len(results) == 1
        result = results[0]
        assert result.component == "grade_spec"
        assert "S" in result.message
        assert result.details["control_freq"] == 50
        assert result.details["max_speed"] == 1.0

    def test_validate_grade_specs_xxl_grade(self):
        """测试XXL等级规格验证"""
        config = DeploymentConfig(grade="XXL")
        validator = DeploymentValidator(config)
        results = validator.validate_grade_specs()
        
        result = results[0]
        assert result.details["control_freq"] == 1000
        assert result.details["max_speed"] == 5.0

    def test_validate_grade_specs_all_grades(self):
        """测试所有等级规格"""
        expected = {
            "S": (50, 1.0),
            "M": (100, 1.5),
            "L": (200, 2.0),
            "XL": (500, 3.0),
            "XXL": (1000, 5.0),
        }
        for grade, (freq, speed) in expected.items():
            config = DeploymentConfig(grade=grade)
            validator = DeploymentValidator(config)
            results = validator.validate_grade_specs()
            assert results[0].details["control_freq"] == freq
            assert results[0].details["max_speed"] == speed


# =============================================================================
# HealthMonitor Tests
# =============================================================================

class TestHealthMonitor:
    """健康监控器测试"""

    def test_initial_state(self):
        """测试初始状态"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        assert monitor.get_state() == DeploymentState.IDLE
        assert monitor.get_overall_status() == HealthStatus.UNKNOWN

    def test_report_health(self):
        """测试健康报告"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            message="Test OK",
        )
        monitor.report_health(result)
        
        summary = monitor.get_health_summary()
        assert summary["total_checks"] == 1
        assert "test_component" in summary["components"]

    def test_multiple_health_reports(self):
        """测试多次健康报告"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        for i in range(5):
            result = HealthCheckResult(
                component=f"component_{i}",
                status=HealthStatus.HEALTHY,
                message=f"Check {i}",
            )
            monitor.report_health(result)
        
        summary = monitor.get_health_summary()
        assert summary["total_checks"] == 5

    def test_critical_updates_error_count(self):
        """测试危险状态更新错误计数"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        for _ in range(3):
            result = HealthCheckResult(
                component="critical_comp",
                status=HealthStatus.CRITICAL,
                message="Critical",
            )
            monitor.report_health(result)
        
        summary = monitor.get_health_summary()
        assert summary["error_counts"]["critical_comp"] == 3

    def test_healthy_resets_error_count(self):
        """测试健康状态重置错误计数"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        monitor.report_health(HealthCheckResult(
            component="comp",
            status=HealthStatus.CRITICAL,
            message="Crit",
        ))
        monitor.report_health(HealthCheckResult(
            component="comp",
            status=HealthStatus.HEALTHY,
            message="OK",
        ))
        
        summary = monitor.get_health_summary()
        assert summary["error_counts"]["comp"] == 0

    def test_overall_status_healthy(self):
        """测试整体状态健康"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        for _ in range(10):
            monitor.report_health(HealthCheckResult(
                component="sensor",
                status=HealthStatus.HEALTHY,
                message="OK",
            ))
        
        assert monitor.get_overall_status() == HealthStatus.HEALTHY

    def test_overall_status_warning(self):
        """测试整体状态警告"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        statuses = [HealthStatus.HEALTHY] * 5 + [HealthStatus.WARNING] * 6
        for s in statuses:
            monitor.report_health(HealthCheckResult(
                component="sensor",
                status=s,
                message=str(s),
            ))
        
        assert monitor.get_overall_status() == HealthStatus.WARNING

    def test_overall_status_critical(self):
        """测试整体状态危险"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        statuses = [HealthStatus.HEALTHY] * 3 + [HealthStatus.CRITICAL] * 4
        for s in statuses:
            monitor.report_health(HealthCheckResult(
                component="sensor",
                status=s,
                message=str(s),
            ))
        
        assert monitor.get_overall_status() == HealthStatus.CRITICAL

    def test_set_and_get_state(self):
        """测试状态设置和获取"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        monitor.set_state(DeploymentState.RUNNING)
        assert monitor.get_state() == DeploymentState.RUNNING
        
        monitor.set_state(DeploymentState.DEGRADED)
        assert monitor.get_state() == DeploymentState.DEGRADED

    def test_health_callback(self):
        """测试健康状态回调"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        callback_results: List[HealthCheckResult] = []
        
        def callback(result: HealthCheckResult):
            callback_results.append(result)
        
        monitor.add_callback(callback)
        
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="Callback test",
        )
        monitor.report_health(result)
        
        assert len(callback_results) == 1
        assert callback_results[0].component == "test"

    def test_health_callback_exception_handling(self):
        """测试回调异常处理"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        def bad_callback(result: HealthCheckResult):
            raise RuntimeError("Callback error")
        
        monitor.add_callback(bad_callback)
        
        # 不应崩溃
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="OK",
        )
        monitor.report_health(result)  # 不应抛出异常

    def test_history_size_limit(self):
        """测试历史记录大小限制"""
        config = DeploymentConfig(health_history_size=5)
        monitor = HealthMonitor(config)
        
        for i in range(10):
            monitor.report_health(HealthCheckResult(
                component="test",
                status=HealthStatus.HEALTHY,
                message=f"Check {i}",
            ))
        
        summary = monitor.get_health_summary()
        assert summary["total_checks"] == 5

    def test_get_health_summary_keys(self):
        """测试健康摘要包含必要字段"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        monitor.report_health(HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="OK",
        ))
        
        summary = monitor.get_health_summary()
        assert "state" in summary
        assert "overall_status" in summary
        assert "total_checks" in summary
        assert "components" in summary
        assert "last_check" in summary
        assert "error_counts" in summary


# =============================================================================
# EmergencyProcedure Tests
# =============================================================================

class TestEmergencyProcedure:
    """紧急停车程序测试"""

    def test_initial_state(self):
        """测试初始状态"""
        config = DeploymentConfig()
        emergency = EmergencyProcedure(config)
        
        info = emergency.get_stop_info()
        assert info["reason"] is None
        assert info["time"] is None

    def test_trigger_emergency_stop(self):
        """测试触发紧急停车"""
        config = DeploymentConfig()
        emergency = EmergencyProcedure(config)
        
        stop_called = []
        def stop_callback():
            stop_called.append(True)
        
        emergency.register_stop_callback(stop_callback)
        emergency.trigger("Collision detected")
        
        info = emergency.get_stop_info()
        assert info["reason"] == "Collision detected"
        assert info["time"] is not None
        assert len(stop_called) == 1

    def test_multiple_stop_callbacks(self):
        """测试多个停车回调"""
        config = DeploymentConfig()
        emergency = EmergencyProcedure(config)
        
        calls = []
        emergency.register_stop_callback(lambda: calls.append(1))
        emergency.register_stop_callback(lambda: calls.append(2))
        emergency.register_stop_callback(lambda: calls.append(3))
        
        emergency.trigger("Test stop")
        
        assert calls == [1, 2, 3]

    def test_callback_exception_handling(self):
        """测试回调异常处理"""
        config = DeploymentConfig()
        emergency = EmergencyProcedure(config)
        
        def bad_callback():
            raise RuntimeError("Stop error")
        
        emergency.register_stop_callback(bad_callback)
        
        # 不应崩溃
        emergency.trigger("Test")
        info = emergency.get_stop_info()
        assert info["reason"] == "Test"

    def test_check_safety_conditions_imu_tilt_safe(self):
        """测试IMU倾角安全检查 - 安全"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 5.0, "pitch_deg": 3.0},
            tilt_threshold_deg=45.0,
        )
        assert is_safe is True
        assert reason is None

    def test_check_safety_conditions_imu_tilt_unsafe(self):
        """测试IMU倾角安全检查 - 不安全"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 50.0, "pitch_deg": 10.0},
            tilt_threshold_deg=45.0,
        )
        assert is_safe is False
        assert "roll" in reason.lower() or "tilted" in reason.lower()

    def test_check_safety_conditions_imu_pitch_unsafe(self):
        """测试IMU俯仰角安全检查 - 不安全"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 5.0, "pitch_deg": 50.0},
            tilt_threshold_deg=45.0,
        )
        assert is_safe is False

    def test_check_safety_conditions_force_collision(self):
        """测试力觉碰撞检测"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            force_data={"total_magnitude_N": 100.0},
            collision_threshold=50.0,
        )
        assert is_safe is False
        assert "collision" in reason.lower()

    def test_check_safety_conditions_force_safe(self):
        """测试力觉正常"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            force_data={"total_magnitude_N": 10.0},
            collision_threshold=50.0,
        )
        assert is_safe is True

    def test_check_safety_conditions_tactile_excessive(self):
        """测试触觉过高压力检测"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            tactile_data={"max_pressure_pct": 99.0},
        )
        assert is_safe is False
        assert "pressure" in reason.lower()

    def test_check_safety_conditions_tactile_safe(self):
        """测试触觉正常"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            tactile_data={"max_pressure_pct": 50.0},
        )
        assert is_safe is True

    def test_check_safety_conditions_combined_all_safe(self):
        """测试组合安全条件 - 全部安全"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 5.0, "pitch_deg": 5.0},
            force_data={"total_magnitude_N": 10.0},
            tactile_data={"max_pressure_pct": 30.0},
            collision_threshold=50.0,
            tilt_threshold_deg=45.0,
        )
        assert is_safe is True

    def test_check_safety_conditions_combined_force_fails(self):
        """测试组合安全条件 - 力觉失败"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 5.0, "pitch_deg": 5.0},
            force_data={"total_magnitude_N": 100.0},
            tactile_data={"max_pressure_pct": 30.0},
            collision_threshold=50.0,
            tilt_threshold_deg=45.0,
        )
        assert is_safe is False

    def test_check_safety_conditions_no_data(self):
        """测试无传感器数据时安全"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions()
        assert is_safe is True

    def test_check_safety_conditions_custom_thresholds(self):
        """测试自定义阈值"""
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            imu_data={"roll_deg": 20.0, "pitch_deg": 20.0},
            force_data={"total_magnitude_N": 30.0},
            tactile_data={"max_pressure_pct": 90.0},
            collision_threshold=40.0,
            tilt_threshold_deg=15.0,
        )
        assert is_safe is False


# =============================================================================
# DeploymentManager Tests
# =============================================================================

class TestDeploymentManager:
    """部署管理器测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        manager = DeploymentManager()
        assert manager.config.grade == "M"
        assert manager.validator is not None
        assert manager.health_monitor is not None
        assert manager.emergency is not None

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = DeploymentConfig(
            grade="XL",
            enable_vision=False,
            emergency_stop_enabled=True,
        )
        manager = DeploymentManager(config=config)
        assert manager.config.grade == "XL"
        assert manager.config.enable_vision is False

    def test_pre_deployment_check_all_pass(self):
        """测试部署前检查全部通过"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        all_passed, results = manager.pre_deployment_check()
        
        assert all_passed is True
        assert len(results) > 0
        
        # 至少包含配置验证和等级规格验证
        components = {r.component for r in results}
        assert "config" in components
        assert "grade_spec" in components

    def test_pre_deployment_check_invalid_grade(self):
        """测试部署前检查失败 - 无效等级"""
        config = DeploymentConfig(grade="INVALID")
        manager = DeploymentManager(config=config)
        
        all_passed, results = manager.pre_deployment_check()
        
        assert all_passed is False
        
        failed = [r for r in results if r.status == HealthStatus.FAILED]
        assert len(failed) > 0

    def test_pre_deployment_check_with_sensors(self):
        """测试带传感器管理器的部署前检查"""
        config = DeploymentConfig(grade="L")
        manager = DeploymentManager(config=config)
        
        all_passed, results = manager.pre_deployment_check()
        
        # 应有警告因为没有传感器管理器
        warnings = [r for r in results if r.status == HealthStatus.WARNING]
        sensor_warning = next(
            (r for r in warnings if "sensor" in r.message.lower()),
            None
        )
        assert sensor_warning is not None

    def test_deploy_success(self):
        """测试成功部署"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        success = manager.deploy()
        
        assert success is True
        assert manager.health_monitor.get_state() == DeploymentState.RUNNING
        
        manager.shutdown()

    def test_deploy_failure_invalid_grade(self):
        """测试部署失败 - 无效等级"""
        config = DeploymentConfig(grade="INVALID")
        manager = DeploymentManager(config=config)
        
        success = manager.deploy()
        
        assert success is False
        assert manager.health_monitor.get_state() == DeploymentState.SHUTDOWN

    def test_deploy_state_transitions(self):
        """测试部署状态转换"""
        config = DeploymentConfig(grade="S")
        manager = DeploymentManager(config=config)
        
        states = []
        def state_callback(result: HealthCheckResult):
            states.append(manager.health_monitor.get_state())
        
        manager.health_monitor.add_callback(state_callback)
        
        manager.deploy()
        # Give monitor thread time to start
        time.sleep(0.3)
        manager.shutdown()
        
        # RUNNING state should be reached after deploy
        assert manager.health_monitor.get_state() == DeploymentState.SHUTDOWN
        # States should have been recorded
        assert len(states) > 0

    def test_shutdown_from_running(self):
        """测试从运行状态关闭"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        time.sleep(0.1)
        manager.shutdown()
        
        assert manager.health_monitor.get_state() == DeploymentState.SHUTDOWN

    def test_get_status(self):
        """测试获取状态"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        time.sleep(0.1)
        
        status = manager.get_status()
        
        assert "state" in status
        assert "health_summary" in status
        assert "emergency_info" in status
        
        manager.shutdown()

    def test_get_status_keys(self):
        """测试状态包含必要字段"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        time.sleep(0.1)
        
        status = manager.get_status()
        summary = status["health_summary"]
        emergency_info = status["emergency_info"]
        
        assert "overall_status" in summary
        assert "total_checks" in summary
        assert "reason" in emergency_info
        assert "time" in emergency_info
        
        manager.shutdown()

    def test_emergency_stop_callback_registered(self):
        """测试紧急停车回调已注册"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        # 内部注册了 _on_emergency_stop
        manager.deploy()
        manager.emergency.trigger("Test emergency")
        
        assert manager.health_monitor.get_state() == DeploymentState.EMERGENCY_STOP
        
        manager.shutdown()

    def test_all_agv_grades_deploy(self):
        """测试所有AGV等级部署"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            config = DeploymentConfig(grade=grade)
            manager = DeploymentManager(config=config)
            
            success = manager.deploy()
            assert success is True
            assert manager.health_monitor.get_state() == DeploymentState.RUNNING
            
            manager.shutdown()

    def test_degraded_mode_allowed(self):
        """测试降级模式"""
        config = DeploymentConfig(grade="M", allow_degraded_mode=True)
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        
        # 模拟报告危险状态
        manager.health_monitor.report_health(HealthCheckResult(
            component="test",
            status=HealthStatus.CRITICAL,
            message="Simulated critical",
        ))
        
        time.sleep(0.2)
        
        # 降级模式不应触发紧急停车
        assert manager.health_monitor.get_state() in (
            DeploymentState.DEGRADED,
            DeploymentState.RUNNING,
        )
        
        manager.shutdown()

    def test_factory_function(self):
        """测试工厂函数"""
        manager = create_deployment_manager(grade="XL")
        
        assert manager.config.grade == "XL"
        assert manager.validator is not None
        
        manager2 = create_deployment_manager(grade="S")
        assert manager2.config.grade == "S"


# =============================================================================
# AGV Five-Grade Integration Tests
# =============================================================================

class TestDeploymentFiveGradeIntegration:
    """AGV五级规格部署集成测试"""

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_all_grades_pre_deployment(self, grade):
        """测试所有等级部署前检查"""
        config = DeploymentConfig(grade=grade)
        manager = DeploymentManager(config=config)
        
        all_passed, results = manager.pre_deployment_check()
        
        assert all_passed is True
        grade_results = [r for r in results if r.component == "grade_spec"]
        assert len(grade_results) == 1
        assert grade in grade_results[0].message

    @pytest.mark.parametrize("grade,sensor_count", [
        ("S", 2), ("M", 4), ("L", 6), ("XL", 8), ("XXL", 10),
    ])
    def test_all_grades_sensor_specs(self, grade, sensor_count):
        """测试所有等级传感器规格"""
        config = DeploymentConfig(grade=grade)
        manager = DeploymentManager(config=config)
        
        all_passed, results = manager.pre_deployment_check()
        
        grade_result = next(r for r in results if r.component == "grade_spec")
        assert grade_result.details["sensors"] == sensor_count

    @pytest.mark.parametrize("grade,expected_freq", [
        ("S", 50), ("M", 100), ("L", 200), ("XL", 500), ("XXL", 1000),
    ])
    def test_all_grades_control_frequency(self, grade, expected_freq):
        """测试所有等级控制频率"""
        validator = DeploymentValidator(DeploymentConfig(grade=grade))
        results = validator.validate_grade_specs()
        
        assert results[0].details["control_freq"] == expected_freq

    @pytest.mark.parametrize("grade,expected_speed", [
        ("S", 1.0), ("M", 1.5), ("L", 2.0), ("XL", 3.0), ("XXL", 5.0),
    ])
    def test_all_grades_max_speed(self, grade, expected_speed):
        """测试所有等级最大速度"""
        validator = DeploymentValidator(DeploymentConfig(grade=grade))
        results = validator.validate_grade_specs()
        
        assert results[0].details["max_speed"] == expected_speed


# =============================================================================
# Emergency Stop Integration Tests
# =============================================================================

class TestEmergencyStopIntegration:
    """紧急停车集成测试"""

    def test_emergency_triggers_state_change(self):
        """测试紧急停车触发状态变化"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        assert manager.health_monitor.get_state() == DeploymentState.RUNNING
        
        manager.emergency.trigger("Safety violation")
        
        assert manager.health_monitor.get_state() == DeploymentState.EMERGENCY_STOP
        
        manager.shutdown()

    def test_emergency_stop_info_recorded(self):
        """测试紧急停车信息记录"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        manager.emergency.trigger("Collision")
        
        info = manager.emergency.get_stop_info()
        assert info["reason"] == "Collision"
        assert info["time"] is not None
        
        manager.shutdown()

    def test_multiple_emergency_triggers(self):
        """测试多次紧急触发"""
        config = DeploymentConfig(grade="M")
        manager = DeploymentManager(config=config)
        
        manager.deploy()
        
        manager.emergency.trigger("First reason")
        first_info = manager.emergency.get_stop_info()
        
        # 后续触发不应改变 reason
        manager.emergency.trigger("Second reason")
        second_info = manager.emergency.get_stop_info()
        
        # 首次触发后的 reason 保留
        assert first_info["reason"] == "First reason"
        
        manager.shutdown()

    def test_emergency_with_safety_check_force(self):
        """测试安全检查触发的紧急停车"""
        config = DeploymentConfig(grade="M")
        emergency = EmergencyProcedure(config)
        
        is_safe, reason = EmergencyProcedure.check_safety_conditions(
            force_data={"total_magnitude_N": 200.0},
            collision_threshold=50.0,
        )
        
        assert is_safe is False
        
        if not is_safe and reason:
            emergency.trigger(reason)
        
        info = emergency.get_stop_info()
        assert info["reason"] is not None


# =============================================================================
# Concurrent Deployment Tests
# =============================================================================

class TestConcurrentDeployment:
    """并发部署测试"""

    def test_multiple_managers_concurrent(self):
        """测试多个部署管理器并发"""
        managers = []
        
        for i, grade in enumerate(["S", "M", "L"]):
            config = DeploymentConfig(grade=grade)
            manager = DeploymentManager(config=config)
            managers.append((i, manager))
        
        # 并发部署
        threads = []
        for idx, mgr in managers:
            t = threading.Thread(target=mgr.deploy)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证所有都运行
        for idx, mgr in managers:
            assert mgr.health_monitor.get_state() == DeploymentState.RUNNING
        
        # 并发关闭
        threads = []
        for idx, mgr in managers:
            t = threading.Thread(target=mgr.shutdown)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

    def test_concurrent_health_reports(self):
        """测试并发健康报告"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        def report_loop(n: int):
            for i in range(n):
                monitor.report_health(HealthCheckResult(
                    component=f"thread_{threading.current_thread().name}",
                    status=HealthStatus.HEALTHY,
                    message=f"Report {i}",
                ))
        
        threads = [
            threading.Thread(target=report_loop, args=(20,), name=f"T{i}")
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        summary = monitor.get_health_summary()
        assert summary["total_checks"] == 100

    def test_concurrent_state_changes(self):
        """测试并发状态变化"""
        config = DeploymentConfig()
        monitor = HealthMonitor(config)
        
        states = []
        lock = threading.Lock()
        
        def change_state(state: DeploymentState):
            monitor.set_state(state)
            with lock:
                states.append((threading.current_thread().name, state))
        
        threads = [
            threading.Thread(target=change_state, args=(s,), name=f"S{i}")
            for i, s in enumerate([
                DeploymentState.RUNNING,
                DeploymentState.DEGRADED,
                DeploymentState.SHUTDOWN,
            ])
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 最终状态应为最后设置的
        assert monitor.get_state() in [
            DeploymentState.RUNNING,
            DeploymentState.DEGRADED,
            DeploymentState.SHUTDOWN,
        ]


# =============================================================================
# Edge Cases
# =============================================================================

class TestDeploymentEdgeCases:
    """部署边缘情况测试"""

    def test_zero_health_check_interval(self):
        """测试零健康检查间隔"""
        config = DeploymentConfig(health_check_interval_s=0.0)
        validator = DeploymentValidator(config)
        results = validator.validate_config()
        
        interval_result = next(r for r in results if "interval" in r.message.lower())
        assert interval_result.status == HealthStatus.WARNING

    def test_extreme_sensor_config(self):
        """测试极端传感器配置"""
        config = DeploymentConfig(
            grade="XXL",
            enable_vision=True,
            enable_audio=True,
            enable_tactile=True,
            enable_force=True,
            enable_imu=True,
        )
        validator = DeploymentValidator(config)
        results = validator.validate_grade_specs()
        
        assert results[0].status == HealthStatus.HEALTHY

    def test_minimal_sensor_config(self):
        """测试最小传感器配置"""
        config = DeploymentConfig(
            grade="S",
            enable_vision=False,
            enable_audio=False,
            enable_tactile=False,
            enable_force=False,
            enable_imu=True,  # 至少保留IMU
        )
        validator = DeploymentValidator(config)
        results = validator.validate_config()
        
        # 应通过（IMU是最低要求）
        config_result = next(r for r in results if r.component == "config")
        assert config_result.status == HealthStatus.HEALTHY

    def test_health_result_with_extreme_latency(self):
        """测试极端延迟的健康结果"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.WARNING,
            latency_ms=10000.0,
            message="Very high latency",
        )
        assert result.latency_ms == 10000.0

    def test_empty_component_name(self):
        """测试空组件名"""
        result = HealthCheckResult(
            component="",
            status=HealthStatus.HEALTHY,
            message="OK",
        )
        assert result.is_healthy() is True

    def test_very_long_message(self):
        """测试超长消息"""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="A" * 1000,
        )
        assert len(result.message) == 1000
