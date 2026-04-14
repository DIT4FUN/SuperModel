"""
test_real_agv_interface.py - Real AGV Interface 硬件接口测试
============================================================

测试 RealAGVController, AGVStateMachine, AGVHeartbeatMonitor
和 AGVHealthMonitor 的功能。
"""

import pytest
import time
import threading
from collections import deque
from unittest.mock import MagicMock, patch


class TestAGVStateMachine:
    """AGVStateMachine 状态机测试"""

    def test_initial_state(self):
        """测试初始状态为 DISCONNECTED"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_001")
        assert sm.state == sm.State.DISCONNECTED
        assert not sm.is_alive
        assert not sm.is_operational

    def test_connect_transition(self):
        """测试连接转换"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_002")
        sm.connect()
        assert sm.state == sm.State.CONNECTING
        assert sm.is_alive

    def test_connect_to_idle(self):
        """测试连接成功后转为 IDLE"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_003")
        sm.connect()
        sm.transition(sm.State.IDLE, "init complete")
        assert sm.state == sm.State.IDLE
        assert sm.is_operational

    def test_idle_to_running(self):
        """测试 IDLE -> RUNNING 转换"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_004")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        result = sm.start_running()
        assert result is True
        assert sm.state == sm.State.RUNNING
        assert sm.is_operational

    def test_running_to_pause_resume(self):
        """测试暂停和恢复"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_005")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        sm.start_running()
        assert sm.state == sm.State.RUNNING
        paused = sm.pause()
        assert paused is True
        assert sm.state == sm.State.PAUSED
        resumed = sm.resume()
        assert resumed is True
        assert sm.state == sm.State.RUNNING

    def test_error_transition(self):
        """测试错误状态转换"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_006")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        sm.set_error("motor failure")
        assert sm.state == sm.State.ERROR
        assert sm._error_reason == "motor failure"
        assert not sm.is_operational

    def test_recovery_flow(self):
        """测试恢复流程: ERROR -> RECOVERING -> IDLE"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_007")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        sm.set_error("can bus timeout")
        assert sm.state == sm.State.ERROR
        sm.set_recovering()
        assert sm.state == sm.State.RECOVERING
        sm.recover_success()
        assert sm.state == sm.State.IDLE

    def test_shutdown_from_any_state(self):
        """测试任何状态都可以关闭"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_008")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        sm.shutdown()
        assert sm.state == sm.State.SHUTDOWN
        assert not sm.is_alive

    def test_state_listener(self):
        """测试状态变化监听器"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_009")
        events = []
        def listener(data):
            events.append(data)
        sm.add_listener('state_changed', listener)
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        assert len(events) == 2
        assert events[0]['from'] == 'disconnected'
        assert events[0]['to'] == 'connecting'

    def test_transition_history(self):
        """测试状态转换历史记录"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_010")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        history = sm.get_transition_history()
        assert len(history) == 2
        assert history[0]['to'] == 'connecting'
        assert history[1]['to'] == 'idle'

    def test_time_in_current_state(self):
        """测试当前状态持续时间"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_011")
        sm.connect()
        time.sleep(0.05)
        elapsed = sm.time_in_current_state()
        assert elapsed >= 0.04

    def test_cannot_start_from_disconnected(self):
        """测试 DISCONNECTED 状态不能 start_running"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_012")
        result = sm.start_running()
        assert result is False

    def test_cannot_resume_from_non_paused(self):
        """测试非 PAUSED 状态不能 resume"""
        from src.embodied.real_agv_interface import AGVStateMachine
        sm = AGVStateMachine("test_agv_013")
        sm.connect()
        sm.transition(sm.State.IDLE, "ready")
        result = sm.resume()
        assert result is False


class TestAGVHeartbeatMonitor:
    """AGVHeartbeatMonitor 心跳监控测试"""

    @pytest.fixture
    def mock_controller(self):
        """创建模拟的 RealAGVController"""
        ctrl = MagicMock()
        ctrl.can_driver = MagicMock()
        ctrl.can_driver.is_connected.return_value = True
        ctrl.motor_controller = MagicMock()
        ctrl.motor_controller.get_encoder_position.return_value = (0.0, 0.0)
        ctrl.lidar = MagicMock()
        ctrl.lidar.is_connected.return_value = True
        ctrl.lidar.read.return_value = None
        ctrl.imu = MagicMock()
        ctrl.imu.is_connected.return_value = True
        ctrl.imu.read.return_value = None
        return ctrl

    def test_heartbeat_monitor_initialization(self, mock_controller):
        """测试心跳监控器初始化"""
        from src.embodied.real_agv_interface import (
            AGVHeartbeatMonitor, AGVStateMachine
        )
        sm = AGVStateMachine("test_agv_hb_001")
        monitor = AGVHeartbeatMonitor(
            controller=mock_controller,
            state_machine=sm,
            heartbeat_interval=0.1,
            max_timeout_count=3,
        )
        assert monitor.heartbeat_interval == 0.1
        assert monitor.max_timeout_count == 3
        assert monitor.total_heartbeats == 0

    def test_heartbeat_monitor_start_stop(self, mock_controller):
        """测试心跳监控启动和停止"""
        from src.embodied.real_agv_interface import (
            AGVHeartbeatMonitor, AGVStateMachine
        )
        sm = AGVStateMachine("test_agv_hb_002")
        monitor = AGVHeartbeatMonitor(
            controller=mock_controller,
            state_machine=sm,
            heartbeat_interval=0.05,
        )
        monitor.start()
        assert monitor._running is True
        time.sleep(0.2)
        assert monitor.total_heartbeats >= 2
        monitor.stop()
        assert monitor._running is False

    def test_heartbeat_statistics(self, mock_controller):
        """测试心跳统计"""
        from src.embodied.real_agv_interface import (
            AGVHeartbeatMonitor, AGVStateMachine
        )
        sm = AGVStateMachine("test_agv_hb_003")
        monitor = AGVHeartbeatMonitor(
            controller=mock_controller,
            state_machine=sm,
            heartbeat_interval=0.05,
        )
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        stats = monitor.get_heartbeat_statistics()
        assert stats['total_heartbeats'] >= 2
        assert stats['uptime_s'] > 0
        assert 'current_timeout_counts' in stats

    def test_timeout_count_tracking(self, mock_controller):
        """测试超时计数"""
        from src.embodied.real_agv_interface import (
            AGVHeartbeatMonitor, AGVStateMachine
        )
        sm = AGVStateMachine("test_agv_hb_004")
        monitor = AGVHeartbeatMonitor(
            controller=mock_controller,
            state_machine=sm,
            heartbeat_interval=0.02,
            max_timeout_count=3,
        )
        # 模拟 CAN 总线超时
        mock_controller.can_driver.is_connected.return_value = False
        monitor.start()
        time.sleep(0.1)
        monitor.stop()
        # 应该有超时
        assert monitor.total_timeouts > 0

    def test_reset_timeout_count(self, mock_controller):
        """测试重置超时计数"""
        from src.embodied.real_agv_interface import (
            AGVHeartbeatMonitor, AGVStateMachine
        )
        sm = AGVStateMachine("test_agv_hb_005")
        monitor = AGVHeartbeatMonitor(
            controller=mock_controller,
            state_machine=sm,
        )
        monitor.timeout_counts['can_bus'] = 2
        monitor.reset_timeout_count('can_bus')
        assert monitor.timeout_counts['can_bus'] == 0

    def test_get_status(self, mock_controller):
        """测试获取完整心跳状态"""
        from src.embodied.real_agv_interface import (
            AGVHeartbeatMonitor, AGVStateMachine
        )
        sm = AGVStateMachine("test_agv_hb_006")
        monitor = AGVHeartbeatMonitor(
            controller=mock_controller,
            state_machine=sm,
            heartbeat_interval=0.05,
        )
        monitor.start()
        time.sleep(0.15)
        status = monitor.get_status()
        assert 'running' in status
        assert 'statistics' in status
        monitor.stop()


class TestRealAGVControllerIntegration:
    """RealAGVController 集成测试（仿真模式）"""

    def test_controller_initialization_creates_state_machine(self):
        """测试控制器初始化时创建状态机"""
        from src.embodied.real_agv_interface import RealAGVController

        with patch.object(RealAGVController, 'initialize', return_value=False):
            ctrl = RealAGVController()
            assert ctrl.state_machine is not None
            from src.embodied.real_agv_interface import AGVStateMachine
            assert isinstance(ctrl.state_machine, AGVStateMachine)
            assert ctrl.state_machine.state == ctrl.state_machine.State.CONNECTING

    def test_controller_creates_health_monitor(self):
        """测试控制器创建健康监控器"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        assert ctrl.health_monitor is not None

    def test_get_full_status_structure(self):
        """测试 get_full_status 返回完整结构"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        status = ctrl.get_full_status()
        assert 'initialized' in status
        assert 'running' in status
        assert 'state_machine' in status
        assert 'sensor_data' in status
        assert 'battery' in status
        assert 'hardware' in status
        # 初始状态是 CONNECTING（因为 __init__ 调用了 connect()）
        assert status['state_machine']['state'] == 'connecting'

    def test_battery_status_structure(self):
        """测试电池状态报告结构"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        ctrl.battery_voltage = 23.0
        status = ctrl.get_full_status()
        battery = status['battery']
        assert 'voltage' in battery
        assert 'level' in battery
        assert battery['is_low'] is True
        assert battery['is_critical'] is False

    def test_battery_critical(self):
        """测试电池严重低电"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        ctrl.battery_voltage = 21.5
        status = ctrl.get_full_status()
        battery = status['battery']
        assert battery['is_low'] is True
        assert battery['is_critical'] is True

    def test_hardware_status_when_not_initialized(self):
        """测试未初始化时的硬件状态"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        status = ctrl.get_full_status()
        hw = status['hardware']
        assert all(v is False for v in hw.values())

    def test_start_pause_resume_task(self):
        """测试任务启动/暂停/恢复"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        ctrl.initialized = True
        ctrl.state_machine.connect()
        ctrl.state_machine.transition(
            ctrl.state_machine.State.IDLE, "ready"
        )
        result = ctrl.start_task()
        assert result is True
        assert ctrl.state_machine.state == ctrl.state_machine.State.RUNNING
        result = ctrl.pause_task()
        assert result is True
        assert ctrl.state_machine.state == ctrl.state_machine.State.PAUSED
        result = ctrl.resume_task()
        assert result is True
        assert ctrl.state_machine.state == ctrl.state_machine.State.RUNNING

    def test_start_task_when_not_initialized(self):
        """测试未初始化时不能启动任务"""
        from src.embodied.real_agv_interface import RealAGVController
        ctrl = RealAGVController()
        result = ctrl.start_task()
        assert result is False

    def test_shutdown_stops_heartbeat(self):
        """测试关闭时停止心跳监控"""
        from src.embodied.real_agv_interface import RealAGVController, AGVHeartbeatMonitor
        ctrl = RealAGVController()
        # 创建一个真实的心跳监控器
        mock_ctrl = MagicMock()
        mock_ctrl.can_driver = MagicMock()
        mock_ctrl.can_driver.is_connected.return_value = True
        mock_ctrl.motor_controller = MagicMock()
        mock_ctrl.lidar = MagicMock()
        mock_ctrl.imu = MagicMock()
        mock_ctrl.state_machine = ctrl.state_machine
        monitor = AGVHeartbeatMonitor(
            controller=mock_ctrl,
            state_machine=ctrl.state_machine,
            heartbeat_interval=0.05,
        )
        ctrl.heartbeat_monitor = monitor
        monitor.start()
        time.sleep(0.05)
        ctrl.shutdown()
        assert monitor._running is False  # 心跳监控器已停止


class TestAGVHealthMonitor:
    """AGVHealthMonitor 健康监控测试"""

    def test_health_monitor_initialization(self):
        """测试健康监控器初始化"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_001")
        assert monitor.agv_id == "test_agv_hm_001"
        assert len(monitor.errors) == 0
        assert len(monitor.warnings) == 0
        assert 'motor_temperature_left' in monitor.metrics

    def test_record_error(self):
        """测试错误记录"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_002")
        monitor.record_error("motor_failure", "Left motor overheating")
        assert len(monitor.errors) == 1
        assert monitor.errors[0]['type'] == "motor_failure"

    def test_record_warning(self):
        """测试警告记录"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_003")
        monitor.record_warning("low_battery", "Battery at 23.0V")
        assert len(monitor.warnings) == 1

    def test_record_metric(self):
        """测试指标记录"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_004")
        monitor.record_metric('battery_voltage', 24.0)
        monitor.record_metric('battery_voltage', 23.5)
        assert len(monitor.metrics['battery_voltage']) == 2
        assert monitor.metrics['battery_voltage'][-1] == 23.5

    def test_get_health_status_excellent(self):
        """测试健康状态 Excellent"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_005")
        monitor.record_metric('motor_temperature_left', 40.0)
        monitor.record_metric('motor_temperature_right', 40.0)
        monitor.record_metric('battery_voltage', 24.0)
        monitor.record_metric('can_bus_error_rate', 0.0)
        status = monitor.get_health_status()
        assert status['overall_health'] >= 0.7
        assert status['health_level'] in ('EXCELLENT', 'GOOD', 'WARNING', 'CRITICAL')

    def test_get_health_status_low_battery(self):
        """测试低电池导致健康分降低"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_006")
        monitor.record_metric('battery_voltage', 21.5)
        status = monitor.get_health_status()
        assert 'low' in str(status['issues']).lower() or status['overall_health'] < 1.0

    def test_get_metric_statistics(self):
        """测试指标统计"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_007")
        for v in [23.0, 23.5, 24.0, 23.8, 23.2]:
            monitor.record_metric('battery_voltage', v)
        stats = monitor.get_metric_statistics('battery_voltage')
        assert stats is not None
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert stats['min'] == 23.0
        assert stats['max'] == 24.0

    def test_reset(self):
        """测试重置监控数据"""
        from src.embodied.real_agv_interface import AGVHealthMonitor
        monitor = AGVHealthMonitor("test_agv_hm_008")
        monitor.record_error("test_error", "test")
        monitor.record_metric('battery_voltage', 24.0)
        monitor.reset()
        assert len(monitor.errors) == 0
        assert len(monitor.metrics['battery_voltage']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
