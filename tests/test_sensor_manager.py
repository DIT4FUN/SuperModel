"""
传感器管理器测试
================

测试统一传感器管理器 (SensorManager)
- 生命周期管理
- 同步/异步采集
- 健康监控
- 回调机制
"""

import numpy as np
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.manager import (
    SensorManager, SensorManagerConfig, SensorDataFrame, SensorGrade
)


class TestSensorManagerConfig(unittest.TestCase):
    """传感器管理器配置测试"""
    
    def test_default_config(self):
        config = SensorManagerConfig()
        self.assertEqual(config.grade, SensorGrade.M)
        self.assertTrue(config.vision_enabled)
        self.assertTrue(config.audio_enabled)
    
    def test_config_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = SensorManagerConfig(grade=grade)
            self.assertEqual(config.grade.value, grade)
    
    def test_rate_per_grade(self):
        rates = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = SensorManagerConfig(grade=grade)
            rates[grade] = config.capture_rate_hz
        
        self.assertLess(rates['S'], rates['M'])
        self.assertLess(rates['M'], rates['L'])
        self.assertLess(rates['L'], rates['XL'])
        self.assertLess(rates['XL'], rates['XXL'])


class TestSensorDataFrame(unittest.TestCase):
    """统一数据帧测试"""
    
    def test_frame_creation(self):
        frame = SensorDataFrame(timestamp=1.0, frame_id=1)
        self.assertEqual(frame.timestamp, 1.0)
        self.assertEqual(frame.frame_id, 1)
        self.assertIsNone(frame.vision)
        self.assertIsNone(frame.audio)
    
    def test_get_modalities_empty(self):
        frame = SensorDataFrame(timestamp=0.0, frame_id=0)
        self.assertEqual(frame.get_modalities(), [])
    
    def test_get_modalities_partial(self):
        mock_vision = MagicMock()
        mock_audio = MagicMock()
        frame = SensorDataFrame(
            timestamp=1.0, frame_id=1,
            vision=mock_vision, audio=mock_audio
        )
        mods = frame.get_modalities()
        self.assertIn("vision", mods)
        self.assertIn("audio", mods)
        self.assertEqual(len(mods), 2)
    
    def test_get_modalities_all(self):
        frame = SensorDataFrame(
            timestamp=1.0, frame_id=1,
            vision=MagicMock(),
            audio=MagicMock(),
            tactile=MagicMock(),
            force=MagicMock(),
            imu=MagicMock()
        )
        mods = frame.get_modalities()
        self.assertEqual(set(mods), {"vision", "audio", "tactile", "force", "imu"})
    
    def test_is_healthy_default(self):
        frame = SensorDataFrame(timestamp=0.0, frame_id=0)
        self.assertTrue(frame.is_healthy())
    
    def test_is_healthy_all_true(self):
        frame = SensorDataFrame(
            timestamp=0.0, frame_id=0,
            healthy={"vision": True, "audio": True, "tactile": True}
        )
        self.assertTrue(frame.is_healthy())
    
    def test_is_healthy_one_false(self):
        frame = SensorDataFrame(
            timestamp=0.0, frame_id=0,
            healthy={"vision": True, "audio": False}
        )
        self.assertFalse(frame.is_healthy())
    
    def test_latencies_ms(self):
        frame = SensorDataFrame(
            timestamp=1.0, frame_id=1,
            latencies_ms={"vision": 5.2, "audio": 2.1}
        )
        self.assertEqual(frame.latencies_ms["vision"], 5.2)
        self.assertEqual(frame.latencies_ms["audio"], 2.1)


class TestSensorManager(unittest.TestCase):
    """传感器管理器测试"""
    
    def setUp(self):
        """每个测试前创建干净的管理器"""
        self.config = SensorManagerConfig(grade="M")
        self.manager = SensorManager(self.config)
    
    def tearDown(self):
        self.manager.close_all()
    
    def test_manager_init(self):
        self.assertIsNotNone(self.manager.config)
        self.assertEqual(self.manager.config.grade, SensorGrade.M)
        self.assertFalse(self.manager._is_open)
        self.assertFalse(self.manager._is_async_running)
    
    def test_open_all_without_sensors(self):
        """在无传感器情况下打开 (模拟环境中)"""
        # 强制禁用所有传感器
        self.config.vision_enabled = False
        self.config.audio_enabled = False
        self.config.tactile_enabled = False
        self.config.force_enabled = False
        self.config.imu_enabled = False
        
        manager = SensorManager(self.config)
        result = manager.open_all()
        self.assertTrue(result)
        self.assertTrue(manager._is_open)
        manager.close_all()
    
    @patch('sensors.manager.SensorManager._import_and_create_sensors')
    def test_open_all_with_mock_sensors(self, mock_import):
        """测试打开所有传感器 (模拟)"""
        # 创建 mock 传感器
        mock_vision = MagicMock()
        mock_vision.open = MagicMock(return_value=True)
        mock_audio = MagicMock()
        mock_audio.open = MagicMock(return_value=True)
        
        self.manager._vision_sensor = mock_vision
        self.manager._audio_sensor = mock_audio
        self.manager._sensor_health["vision"] = True
        self.manager._sensor_health["audio"] = True
        
        result = self.manager.open_all()
        self.assertTrue(result)
    
    def test_capture_all_no_sensors(self):
        """无传感器时采集应返回空帧"""
        # 配置无传感器
        self.config.vision_enabled = False
        config_no_sensors = SensorManagerConfig()
        config_no_sensors.vision_enabled = config_no_sensors.audio_enabled = False
        config_no_sensors.tactile_enabled = config_no_sensors.force_enabled = config_no_sensors.imu_enabled = False
        
        manager = SensorManager(config_no_sensors)
        manager.open_all()
        
        frame = manager.capture_all()
        self.assertIsInstance(frame, SensorDataFrame)
        self.assertEqual(frame.get_modalities(), [])
        self.assertTrue(frame.is_healthy())
        
        manager.close_all()
    
    @patch('sensors.manager.SensorManager._import_and_create_sensors')
    def test_capture_single_modality(self, mock_import):
        """测试单模态采集"""
        mock_sensor = MagicMock()
        mock_sensor.capture = MagicMock(return_value=MagicMock())
        
        self.manager._imu_sensor = mock_sensor
        self.manager._sensor_health["imu"] = True
        self.manager._is_open = True
        
        result = self.manager.capture_single("imu")
        mock_sensor.capture.assert_called_once()
        self.assertIsNotNone(result)
    
    def test_capture_single_unknown_modality(self):
        """未知模态返回 None"""
        result = self.manager.capture_single("unknown_modality")
        self.assertIsNone(result)
    
    def test_get_health_status(self):
        """测试健康状态获取"""
        status = self.manager.get_health_status()
        
        self.assertIn("vision", status)
        self.assertIn("audio", status)
        self.assertIn("tactile", status)
        self.assertIn("force", status)
        self.assertIn("imu", status)
        self.assertIn("errors", status)
        self.assertIn("async_running", status)
        self.assertFalse(status["async_running"])
    
    def test_register_callback(self):
        """测试回调注册"""
        callback = MagicMock()
        self.manager.register_callback("imu", callback)
        
        self.assertIn("imu", self.manager._callbacks)
        self.assertEqual(len(self.manager._callbacks["imu"]), 1)
    
    def test_check_sensor_alive_not_open(self):
        """传感器未打开时检查存活"""
        result = self.manager.check_sensor_alive("imu")
        self.assertFalse(result)
    
    @patch('sensors.manager.SensorManager._import_and_create_sensors')
    def test_check_sensor_alive_recent_data(self, mock_import):
        """传感器有最近数据时存活"""
        self.manager._sensor_health["imu"] = True
        self.manager._sensor_last_ts["imu"] = time.time()
        
        result = self.manager.check_sensor_alive("imu")
        self.assertTrue(result)
    
    @patch('sensors.manager.SensorManager._import_and_create_sensors')
    def test_check_sensor_alive_timeout(self, mock_import):
        """传感器数据超时则不存活"""
        self.manager._sensor_health["imu"] = True
        self.manager._sensor_last_ts["imu"] = time.time() - 10.0  # 10秒前
        
        result = self.manager.check_sensor_alive("imu", timeout=5.0)
        self.assertFalse(result)
    
    @patch('sensors.manager.SensorManager._import_and_create_sensors')
    def test_context_manager(self, mock_import):
        """测试上下文管理器"""
        m = SensorManager()
        m._is_open = True  # simulate already-open state
        m.close_all()
        self.assertFalse(m._is_open)
    
    def test_async_capture_lifecycle(self):
        """测试异步采集生命周期"""
        # 配置无传感器，避免实际硬件依赖
        self.config.vision_enabled = self.config.audio_enabled = False
        self.config.tactile_enabled = self.config.force_enabled = self.config.imu_enabled = False
        
        manager = SensorManager(self.config)
        
        self.assertFalse(manager._is_async_running)
        
        # 启动异步采集
        manager.start_async_capture()
        self.assertTrue(manager._is_async_running)
        
        # 停止异步采集
        manager.stop_async_capture()
        self.assertFalse(manager._is_async_running)
    
    def test_get_latest_frame_no_queue(self):
        """空队列时返回 None"""
        result = self.manager.get_latest_frame()
        self.assertIsNone(result)
    
    def test_frame_generator_empty(self):
        """无数据时的生成器"""
        self.manager._is_async_running = False
        gen = self.manager.frame_generator()
        # 生成器在 async 不运行时立即停止
        frames = list(gen)
        self.assertEqual(len(frames), 0)


class TestSensorGrades(unittest.TestCase):
    """AGV五级传感器规格测试"""
    
    def test_all_grades_have_rates(self):
        """每个等级都有采集率"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = SensorManagerConfig(grade=grade)
            self.assertGreater(config.capture_rate_hz, 0)
            self.assertLessEqual(config.capture_rate_hz, 500)


if __name__ == "__main__":
    unittest.main()
