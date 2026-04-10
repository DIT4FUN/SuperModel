"""
硬件模块综合测试
================

测试:
- CAN Bus 接口 (CANopen协议/虚拟总线)
- 传感器硬件桥接器
- AGV五级硬件规格

Author: SuperModel Team
Version: v2.57.0
"""

import unittest
import numpy as np
import time
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware.canbus import (
    CANFrame, CANBusState, CANopenNodeState, CANopenPDO, SensorCANConfig,
    VirtualCANBus, create_can_bus,
    IMUCANopenNode, ForceTorqueCANopenNode, TactileCANopenNode,
    AGV_CAN_GRADES, get_can_spec,
)
from src.hardware.sensor_bridge import (
    SensorDataType, SensorProtocol, SensorHealth,
    SensorData, SensorHardwareConfig,
    SimulatedSensorInterface, SensorHardwareBridge,
    AGV_SENSOR_BRIDGE_GRADES, get_bridge_spec,
)


class TestVirtualCANBus(unittest.TestCase):
    """虚拟CAN总线测试"""

    def test_creation(self):
        """测试创建"""
        bus = VirtualCANBus(channel_id=0)
        self.assertEqual(bus.channel_id, 0)
        self.assertEqual(bus.get_bus_state(), CANBusState.CLOSED)

    def test_open_close(self):
        """测试打开/关闭"""
        bus = VirtualCANBus()
        self.assertTrue(bus.open())
        self.assertEqual(bus.get_bus_state(), CANBusState.OPEN)
        bus.close()
        self.assertEqual(bus.get_bus_state(), CANBusState.CLOSED)

    def test_context_manager(self):
        """测试上下文管理器"""
        with VirtualCANBus() as bus:
            self.assertEqual(bus.get_bus_state(), CANBusState.OPEN)
        self.assertEqual(bus.get_bus_state(), CANBusState.CLOSED)

    def test_send_receive(self):
        """测试帧收发"""
        with VirtualCANBus() as bus:
            frame = CANFrame(can_id=0x123, data=bytes([1, 2, 3, 4]))
            self.assertTrue(bus.send(frame))
            received = bus.receive(timeout=0.1)
            self.assertIsNotNone(received)
            self.assertEqual(received.can_id, 0x123)
            self.assertEqual(received.data, bytes([1, 2, 3, 4]))

    def test_listener(self):
        """测试监听器"""
        with VirtualCANBus() as bus:
            received_frames = []
            bus.add_listener(lambda f: received_frames.append(f))
            bus.send(CANFrame(can_id=0x100, data=bytes([0xAA])))
            time.sleep(0.05)
            self.assertEqual(len(received_frames), 1)
            self.assertEqual(received_frames[0].can_id, 0x100)

    def test_error_injection(self):
        """测试错误注入"""
        bus = VirtualCANBus()
        bus.inject_error("bus_off")
        self.assertEqual(bus.get_bus_state(), CANBusState.BUS_OFF)
        bus.reset()
        self.assertEqual(bus.get_bus_state(), CANBusState.CLOSED)

    def test_multiple_frames(self):
        """测试多帧发送"""
        with VirtualCANBus() as bus:
            for i in range(10):
                bus.send(CANFrame(can_id=0x200 + i, data=bytes([i])))
            time.sleep(0.05)
            count = 0
            while bus.receive(timeout=0.01):
                count += 1
            self.assertEqual(count, 10)


class TestCANopenNode(unittest.TestCase):
    """CANopen节点测试"""

    def setUp(self):
        self.bus = VirtualCANBus()
        self.bus.open()

    def tearDown(self):
        self.bus.close()

    def test_imu_node_creation(self):
        """测试IMU CANopen节点创建"""
        node = IMUCANopenNode(node_id=5, bus=self.bus)
        self.assertEqual(node.node_id, 5)
        self.assertEqual(node.state, CANopenNodeState.INITIALISING)

    def test_imu_node_start_stop(self):
        """测试IMU节点启动/停止"""
        node = IMUCANopenNode(node_id=5, bus=self.bus)
        node.start()
        self.assertEqual(node.state, CANopenNodeState.OPERATIONAL)
        node.stop()
        self.assertEqual(node.state, CANopenNodeState.STOPPED)

    def test_imu_node_config(self):
        """测试IMU节点配置"""
        node = IMUCANopenNode(node_id=5, bus=self.bus)
        node.set_output_config(
            enable_accel=True,
            enable_gyro=True,
            enable_euler=True,
            sample_rate=200
        )
        node.start()

        # 模拟接收PDO数据 (TPDO2 = 0x280 + node_id, 欧拉角+温度)
        euler_data = struct.pack("<3f", 0.1, -0.2, 0.05)
        temp_data = struct.pack("<f", 28.5)
        pdo2_frame = CANFrame(can_id=0x285, data=euler_data + temp_data)
        self.bus.send(pdo2_frame)
        time.sleep(0.05)

        euler = node.get_euler()
        self.assertIsNotNone(euler)
        self.assertAlmostEqual(euler[0], 0.1, places=1)

        node.stop()

    def test_force_node_creation(self):
        """测试力觉CANopen节点创建"""
        node = ForceTorqueCANopenNode(node_id=10, bus=self.bus)
        self.assertEqual(node.node_id, 10)
        self.assertFalse(node.is_saturated())

    def test_force_node_operations(self):
        """测试力觉节点操作"""
        node = ForceTorqueCANopenNode(node_id=10, bus=self.bus)
        node.start()
        wrench = node.get_wrench()
        self.assertIsNone(wrench)  # 初始无数据

        # 模拟TPDO2数据 (欧拉角+温度格式, 但用于验证节点响应)
        euler_data = struct.pack("<3f", 0.0, 0.0, 0.0)
        temp_data = struct.pack("<f", 25.0)
        pdo2_frame = CANFrame(can_id=0x28A, data=euler_data + temp_data)
        self.bus.send(pdo2_frame)
        time.sleep(0.05)

        # 验证饱和标志方法可用
        self.assertFalse(node.is_saturated())

        node.stop()

    def test_tactile_node_creation(self):
        """测试触觉CANopen节点创建"""
        node = TactileCANopenNode(
            node_id=15,
            bus=self.bus,
            array_size=(8, 8)
        )
        self.assertEqual(node.node_id, 15)
        self.assertFalse(node.is_contact())


class TestSensorBridge(unittest.TestCase):
    """传感器硬件桥接器测试"""

    def test_creation(self):
        """测试创建"""
        bridge = SensorHardwareBridge(name="test_bridge")
        self.assertEqual(bridge.name, "test_bridge")
        self.assertEqual(bridge.list_sensors(), [])

    def test_register_sensor(self):
        """测试注册传感器"""
        bridge = SensorHardwareBridge()
        config = SensorHardwareConfig(
            sensor_id="test_imu",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
        )
        sensor = SimulatedSensorInterface(config)
        bridge.register("test_imu", sensor)
        self.assertIn("test_imu", bridge.list_sensors())

    def test_open_close_all(self):
        """测试批量打开/关闭"""
        bridge = SensorHardwareBridge()
        for i in range(3):
            config = SensorHardwareConfig(
                sensor_id=f"sensor_{i}",
                sensor_type=SensorDataType.IMU,
                protocol=SensorProtocol.SIMULATED,
                sample_rate=50,
            )
            sensor = SimulatedSensorInterface(config)
            bridge.register(f"sensor_{i}", sensor)

        results = bridge.open_all()
        self.assertTrue(all(results.values()))

        bridge.close_all()
        for sensor in bridge._interfaces.values():
            self.assertFalse(sensor.is_opened)

    def test_create_sensor_factory(self):
        """测试工厂方法创建传感器"""
        bridge = SensorHardwareBridge()

        # 仿真IMU
        imu = bridge.create_sensor(
            "imu1", SensorDataType.IMU,
            SensorProtocol.SIMULATED,
            use_simulated=True,
            sample_rate=100
        )
        self.assertIsInstance(imu, SimulatedSensorInterface)
        bridge.register("imu1", imu)

        # 仿真力觉
        force = bridge.create_sensor(
            "force1", SensorDataType.FORCE_TORQUE,
            SensorProtocol.SIMULATED,
            use_simulated=True
        )
        self.assertIsInstance(force, SimulatedSensorInterface)
        bridge.register("force1", force)

        # 仿真触觉
        tactile = bridge.create_sensor(
            "tactile1", SensorDataType.TACTILE,
            SensorProtocol.SIMULATED,
            use_simulated=True,
            frame_size=(8, 8)
        )
        self.assertIsInstance(tactile, SimulatedSensorInterface)
        bridge.register("tactile1", tactile)

        self.assertEqual(len(bridge.list_sensors()), 3)

    def test_get_latest_data(self):
        """测试获取最新数据"""
        bridge = SensorHardwareBridge()
        config = SensorHardwareConfig(
            sensor_id="imu_latest",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=100,
        )
        sensor = SimulatedSensorInterface(config)
        bridge.register("imu_latest", sensor)
        bridge.open_all()
        bridge.start_all()

        time.sleep(0.3)  # 等待数据流

        latest = bridge.get_latest("imu_latest")
        self.assertIsNotNone(latest)
        self.assertIsNotNone(latest.accel)
        self.assertEqual(len(latest.accel), 3)

        bridge.stop_all()
        bridge.close_all()

    def test_get_all_data(self):
        """测试获取历史数据"""
        bridge = SensorHardwareBridge()
        config = SensorHardwareConfig(
            sensor_id="imu_history",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=50,
        )
        sensor = SimulatedSensorInterface(config)
        bridge.register("imu_history", sensor)
        bridge.open_all()
        bridge.start_all()

        time.sleep(0.5)

        all_data = bridge.get_all("imu_history", max_count=10)
        self.assertGreaterEqual(len(all_data), 1)

        bridge.stop_all()
        bridge.close_all()

    def test_health_status(self):
        """测试健康状态监控"""
        bridge = SensorHardwareBridge()
        config = SensorHardwareConfig(
            sensor_id="imu_health",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=100,
        )
        sensor = SimulatedSensorInterface(config)
        bridge.register("imu_health", sensor)
        bridge.open_all()
        bridge.start_all()

        time.sleep(0.2)
        health = bridge.get_health("imu_health")
        self.assertEqual(health, SensorHealth.OK)

        bridge.stop_all()
        bridge.close_all()

    def test_unregister(self):
        """测试注销传感器"""
        bridge = SensorHardwareBridge()
        config = SensorHardwareConfig(
            sensor_id="imu_unreg",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
        )
        sensor = SimulatedSensorInterface(config)
        bridge.register("imu_unreg", sensor)
        bridge.unregister("imu_unreg")
        self.assertNotIn("imu_unreg", bridge.list_sensors())


class TestSimulatedSensor(unittest.TestCase):
    """仿真传感器接口测试"""

    def test_imu_simulated(self):
        """测试IMU仿真"""
        config = SensorHardwareConfig(
            sensor_id="sim_imu",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=100,
        )
        with SimulatedSensorInterface(config) as sensor:
            data = sensor.read()
            self.assertIsNotNone(data)
            self.assertIsNotNone(data.accel)
            self.assertEqual(len(data.accel), 3)
            self.assertAlmostEqual(data.accel[2], -9.81, places=1)
            self.assertIsNotNone(data.gyro)
            self.assertEqual(len(data.gyro), 3)

    def test_force_simulated(self):
        """测试力觉仿真"""
        config = SensorHardwareConfig(
            sensor_id="sim_force",
            sensor_type=SensorDataType.FORCE_TORQUE,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=100,
        )
        with SimulatedSensorInterface(config) as sensor:
            data = sensor.read()
            self.assertIsNotNone(data)
            self.assertIsNotNone(data.wrench)
            self.assertEqual(len(data.wrench), 6)

    def test_tactile_simulated(self):
        """测试触觉仿真"""
        config = SensorHardwareConfig(
            sensor_id="sim_tactile",
            sensor_type=SensorDataType.TACTILE,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=50,
            frame_size=(8, 8),
        )
        with SimulatedSensorInterface(config) as sensor:
            data = sensor.read()
            self.assertIsNotNone(data)
            self.assertIsNotNone(data.tactile_array)
            self.assertEqual(data.tactile_array.shape, (8, 8))

    def test_noise_models(self):
        """测试不同噪声模型"""
        config = SensorHardwareConfig(
            sensor_id="noise_test",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=100,
        )

        # 高斯噪声
        sensor_g = SimulatedSensorInterface(config, noise_model="gaussian")
        sensor_g.open()
        for _ in range(5):
            sensor_g.read()
        sensor_g.close()

        # 无噪声
        sensor_n = SimulatedSensorInterface(config, noise_model="none")
        sensor_n.open()
        for _ in range(5):
            sensor_n.read()
        sensor_n.close()

    def test_wave_models(self):
        """测试波形模型"""
        config = SensorHardwareConfig(
            sensor_id="wave_test",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.SIMULATED,
            sample_rate=100,
        )

        for wave in ["sine", "none"]:
            sensor = SimulatedSensorInterface(config, wave_model=wave)
            sensor.open()
            for _ in range(10):
                sensor.read()
            sensor.close()


class TestAGVGrades(unittest.TestCase):
    """AGV五级硬件规格测试"""

    def test_can_grades(self):
        """测试CAN总线五级规格"""
        grades = ["S", "M", "L", "XL", "XXL"]
        for g in grades:
            spec = get_can_spec(g)
            self.assertIn("bitrate", spec)
            self.assertIn("max_nodes", spec)
            self.assertIn("can_channels", spec)
            self.assertIn("supported_sensors", spec)
            self.assertGreater(spec["max_nodes"], 0)
            self.assertGreater(spec["can_channels"], 0)

    def test_can_grades_keys(self):
        """测试CAN五级规格键一致性"""
        self.assertEqual(set(AGV_CAN_GRADES.keys()), {"S", "M", "L", "XL", "XXL"})

    def test_bridge_grades(self):
        """测试硬件桥接器五级规格"""
        grades = ["S", "M", "L", "XL", "XXL"]
        for g in grades:
            spec = get_bridge_spec(g)
            self.assertIn("max_sensors", spec)
            self.assertIn("supported_types", spec)
            self.assertIn("supported_protocols", spec)
            self.assertIn("total_bandwidth_mbps", spec)
            self.assertIn("max_sample_rate_hz", spec)
            self.assertGreater(spec["max_sensors"], 0)
            # 带宽应随等级增加
            grade_order = {"S": 0, "M": 1, "L": 2, "XL": 3, "XXL": 4}
            self.assertGreater(
                spec["total_bandwidth_mbps"],
                AGV_SENSOR_BRIDGE_GRADES[grades[max(0, grade_order[g] - 1)]]["total_bandwidth_mbps"]
                if grade_order[g] > 0 else 0
            )

    def test_bridge_grades_keys(self):
        """测试硬件桥接器五级规格键一致性"""
        self.assertEqual(set(AGV_SENSOR_BRIDGE_GRADES.keys()), {"S", "M", "L", "XL", "XXL"})

    def test_grade_progression(self):
        """测试等级递进关系"""
        prev_sensors = 0
        prev_bandwidth = 0.0
        prev_rate = 0
        for grade in ["S", "M", "L", "XL", "XXL"]:
            can_spec = get_can_spec(grade)
            bridge_spec = get_bridge_spec(grade)
            # 规格应随等级严格递增
            self.assertGreaterEqual(can_spec["max_nodes"], prev_sensors if grade != "S" else 1)
            self.assertGreater(bridge_spec["total_bandwidth_mbps"], prev_bandwidth)
            self.assertGreater(bridge_spec["max_sample_rate_hz"], prev_rate)
            prev_sensors = can_spec["max_nodes"]
            prev_bandwidth = bridge_spec["total_bandwidth_mbps"]
            prev_rate = bridge_spec["max_sample_rate_hz"]


class TestSensorData(unittest.TestCase):
    """标准传感器数据格式测试"""

    def test_imu_data(self):
        """测试IMU数据填充"""
        data = SensorData(
            sensor_id="imu_test",
            sensor_type=SensorDataType.IMU,
            protocol=SensorProtocol.CAN,
            accel=np.array([0.1, 0.2, -9.81]),
            gyro=np.array([0.01, -0.01, 0.005]),
            euler=np.array([0.0, 0.0, 0.0]),
            temperature=25.0,
        )
        self.assertEqual(data.sensor_id, "imu_test")
        self.assertEqual(data.sensor_type, SensorDataType.IMU)
        self.assertEqual(data.protocol, SensorProtocol.CAN)
        np.testing.assert_array_almost_equal(data.accel, [0.1, 0.2, -9.81])

    def test_force_data(self):
        """测试力觉数据填充"""
        data = SensorData(
            sensor_id="force_test",
            sensor_type=SensorDataType.FORCE_TORQUE,
            protocol=SensorProtocol.CAN,
            wrench=np.array([10.0, -5.0, 0.0, 0.5, -0.3, 0.1]),
            is_saturated=False,
        )
        self.assertEqual(data.sensor_type, SensorDataType.FORCE_TORQUE)
        np.testing.assert_array_almost_equal(data.wrench[:3], [10.0, -5.0, 0.0])

    def test_tactile_data(self):
        """测试触觉数据填充"""
        tactile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        data = SensorData(
            sensor_id="tactile_test",
            sensor_type=SensorDataType.TACTILE,
            protocol=SensorProtocol.SIMULATED,
            tactile_array=tactile,
            total_pressure=float(np.sum(tactile)),
            contact_center=(0.5, 0.5),
            is_contact=True,
        )
        self.assertEqual(data.sensor_type, SensorDataType.TACTILE)
        self.assertEqual(data.tactile_array.shape, (8, 8))


import struct


if __name__ == "__main__":
    unittest.main()
