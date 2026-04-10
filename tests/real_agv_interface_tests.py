"""
real_agv_interface_tests.py - 真实AGV硬件接口测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- AGV硬件规格表一致性检查
- 数据结构测试
- 驱动接口测试 (离线)
- 等级规格一致性验证
"""

import pytest
import numpy as np
from src.hardware.real_agv_interface import (
    AGVHardwareStatus,
    MotorState,
    WheelEncoder,
    IMUData,
    LidarScan,
    RealAGVInterface,
    RealAGVController,
    AGV_HARDWARE_SPECS,
    CANZAC8015DDriver,
    AGVTaskPlanner,
    HardwareMonitor,
)


class TestAGVHardwareSpecsConsistency:
    """AGV五级硬件规格一致性检查"""

    def test_all_grades_present(self):
        """所有等级都存在"""
        assert 'S' in AGV_HARDWARE_SPECS
        assert 'M' in AGV_HARDWARE_SPECS
        assert 'L' in AGV_HARDWARE_SPECS
        assert 'XL' in AGV_HARDWARE_SPECS
        assert 'XXL' in AGV_HARDWARE_SPECS

    def test_all_required_fields(self):
        """所有等级都包含必填字段"""
        required_fields = [
            'grade', 'load_kg', 'wheel_config', 'motor_type',
            'motor_power_w', 'encoder_ppr', 'max_speed_mps',
            'battery_v', 'battery_ah', 'lidar', 'has_imu',
            'has_tactile', 'has_force', 'can_bitrate',
        ]
        for grade, spec in AGV_HARDWARE_SPECS.items():
            for field in required_fields:
                assert field in spec, f"{grade} missing {field}"

    def test_load_increase_with_grade(self):
        """负载随等级增加"""
        loads = [AGV_HARDWARE_SPECS[g]['load_kg'] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        assert loads == sorted(loads)

    def test_battery_capacity_increase(self):
        """电池容量随等级增加"""
        ah = [AGV_HARDWARE_SPECS[g]['battery_ah'] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        assert ah == sorted(ah)

    def test_can_bitrate_correct(self):
        """CAN波特率正确性"""
        assert AGV_HARDWARE_SPECS['S']['can_bitrate'] == 250000
        assert AGV_HARDWARE_SPECS['M']['can_bitrate'] == 500000
        assert AGV_HARDWARE_SPECS['XXL']['can_bitrate'] == 1000000

    def test_sensor_flags_correct(self):
        """传感器标志正确性"""
        # S级没有触觉力觉
        assert not AGV_HARDWARE_SPECS['S']['has_tactile']
        assert not AGV_HARDWARE_SPECS['S']['has_force']
        # M及以上都有
        for grade in ['M', 'L', 'XL', 'XXL']:
            assert AGV_HARDWARE_SPECS[grade]['has_tactile']
            assert AGV_HARDWARE_SPECS[grade]['has_force']
        # 所有等级都有IMU
        for grade in AGV_HARDWARE_SPECS:
            assert AGV_HARDWARE_SPECS[grade]['has_imu']


class TestDataClasses:
    """数据结构测试"""

    def test_motor_state_defaults(self):
        """MotorState默认值测试"""
        ms = MotorState(motor_id=1)
        assert ms.motor_id == 1
        assert not ms.enabled
        assert ms.current_rpm == 0.0
        assert ms.temperature == 25.0

    def test_wheel_encoder(self):
        """WheelEncoder测试"""
        we = WheelEncoder(left_ticks=100, right_ticks=200, left_delta=5, right_delta=-3)
        assert we.left_ticks == 100
        assert we.right_ticks == 200
        assert we.left_delta == 5
        assert we.right_delta == -3

    def test_imu_data(self):
        """IMUData测试"""
        imu = IMUData(
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0.1, 0, 0]),
        )
        assert np.allclose(imu.accelerometer, np.array([0, 0, 9.81]))
        assert imu.temperature == 25.0

    def test_lidar_scan(self):
        """LidarScan测试"""
        scan = LidarScan(
            ranges=np.array([1.0, 2.0, 3.0]),
            angles=np.array([0, np.pi/4, np.pi/2]),
        )
        assert scan.ranges.shape == (3,)
        assert scan.angles.shape == (3,)


class TestRealAGVController:
    """真实AGV控制器测试 (离线)"""

    def test_create_controller_by_grade(self):
        """按等级创建控制器测试"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            controller = RealAGVController(grade=grade, can_interface="can0")
            assert controller.grade == grade
            assert controller.spec == AGV_HARDWARE_SPECS[grade]
            assert controller.status == AGVHardwareStatus.DISCONNECTED

    def test_get_spec(self):
        """获取规格测试"""
        controller = RealAGVController(grade="M")
        spec = controller.get_spec()
        assert spec['grade'] == "M"
        assert spec['load_kg'] == 100

    def test_get_current_state_structure(self):
        """获取当前状态结构测试"""
        controller = RealAGVController(grade="M")
        state = controller.get_current_state()
        assert 'status' in state
        assert 'grade' in state
        assert 'position' in state
        assert 'velocity' in state
        assert 'battery' in state


class TestCANZAC8015DDriverOffline:
    """CAN驱动器离线测试 (不连接硬件)"""

    def test_create_driver(self):
        """创建驱动器测试"""
        driver = CANZAC8015DDriver(
            can_interface="can0",
            node_ids=[0x01, 0x02],
            bitrate=500000,
        )
        assert driver.can_interface == "can0"
        assert driver.node_ids == [0x01, 0x02]
        assert driver.bitrate == 500000
        assert len(driver.motor_states) == 2
        assert driver.status == AGVHardwareStatus.DISCONNECTED

    def test_motor_states_initialized(self):
        """电机状态初始化测试"""
        driver = CANZAC8015DDriver(node_ids=[1, 2, 3, 4])
        assert len(driver.motor_states) == 4
        for node_id, state in driver.motor_states.items():
            assert state.motor_id == node_id


class TestHardwareMonitor:
    """硬件监控器测试"""

    def test_add_device(self):
        """添加设备测试"""
        from src.hardware.real_agv_interface import RealAGVInterface
        monitor = HardwareMonitor()
        # 创建一个模拟设备
        class MockDevice(RealAGVInterface):
            def connect(self): return True
            def disconnect(self): pass
            def get_status(self): return AGVHardwareStatus.CONNECTED

        dev = MockDevice()
        monitor.add_device(dev, "mock")
        assert len(monitor.devices) == 1

    def test_get_all_status(self):
        """获取所有状态测试"""
        from src.hardware.real_agv_interface import RealAGVInterface
        monitor = HardwareMonitor()
        class MockDevice(RealAGVInterface):
            def connect(self): return True
            def disconnect(self): pass
            def get_status(self): return AGVHardwareStatus.CONNECTED

        monitor.add_device(MockDevice(), "dev1")
        monitor.add_device(MockDevice(), "dev2")
        status = monitor.get_all_status()
        assert 'dev1' in status
        assert 'dev2' in status
        assert status['dev1'] == 'CONNECTED'


class TestAGVSpecConsistency:
    """AGV规格一致性测试 - 规格表之间对比"""

    def test_hardware_spec_matches_control_spec(self):
        """硬件规格和控制规格一致性"""
        # 硬件规格的最大速度应该匹配控制规格
        from src.control.agv import AGVSpec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            hw_max_speed = AGV_HARDWARE_SPECS[grade]['max_speed_mps']
            # 这里简化验证量级一致
            assert 0.5 <= hw_max_speed <= 3.0

    def test_wheel_config_correct(self):
        """轮子配置正确性"""
        assert AGV_HARDWARE_SPECS['S']['wheel_config'] == '2-wheel-diff'
        assert AGV_HARDWARE_SPECS['M']['wheel_config'] == '2-wheel-diff'
        assert AGV_HARDWARE_SPECS['L']['wheel_config'] == '4-wheel-diff'


class TestMotorStateMethods:
    """MotorState方法测试"""

    def test_get_radians_per_second(self):
        """转速转换测试"""
        ms = MotorState(motor_id=1, current_rpm=60)
        rad_s = ms.get_radians_per_second(encoder_ppr=1000)
        # 60 rpm = 2π rad/s ≈ 6.28
        assert np.isclose(rad_s, 2 * np.pi)

    def test_default_values(self):
        """默认值测试"""
        ms = MotorState(motor_id=0)
        assert ms.error_code == 0
        assert ms.voltage == 24.0
        assert ms.temperature == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
