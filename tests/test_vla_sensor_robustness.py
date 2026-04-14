"""
test_vla_sensor_robustness.py - VLA传感器鲁棒性测试
====================================================

测试新增的 SensorDropoutHandler 功能和 VLA 推理管道传感器鲁棒性:
- SensorDropoutHandler 传感器掉线追踪
- 降级策略: ZERO / HOLD_LAST / EXTRAPOLATE
- 关键传感器失效紧急模式
- 传感器健康报告
- 多传感器联合掉线处理
"""

import time
import pytest
import numpy as np
import threading

from src.embodied.vla_inference import (
    VLAInferencePipeline,
    VLAPipelineConfig,
    SensorDropoutHandler,
    create_vla_inference_pipeline,
    InferencePolicy,
)
from src.embodied.vla_model import (
    VLAAction, VLAPerceptionFrame, VLAInput, create_vla_model
)


# ============================================================
# SensorDropoutHandler 测试
# ============================================================

class TestSensorDropoutHandler:
    """SensorDropoutHandler 传感器掉线处理器测试"""

    def test_handler_creation(self):
        """测试创建处理器"""
        handler = SensorDropoutHandler()
        assert handler._emergency_mode is False
        assert handler._sensor_states == {}

    def test_record_healthy_reading(self):
        """测试记录正常传感器读数"""
        handler = SensorDropoutHandler()
        handler.record_sensor_reading('camera', np.zeros((10, 10, 3)), timestamp=time.time())
        
        assert handler._sensor_states['camera'] == SensorDropoutHandler.HEALTHY
        assert handler._sensor_consecutive_failures['camera'] == 0

    def test_record_dropout(self):
        """测试记录传感器掉线 (连续失败)"""
        handler = SensorDropoutHandler()
        
        # 连续失败 3 次 (达到掉线阈值)
        for _ in range(3):
            handler.record_sensor_reading('lidar', None, timestamp=time.time())
        
        assert handler._sensor_states['lidar'] == SensorDropoutHandler.DROPOUT
        assert handler._sensor_dropout_counts['lidar'] == 1

    def test_record_failure(self):
        """测试记录传感器彻底失败 (连续失败超过阈值)"""
        handler = SensorDropoutHandler()
        
        # 连续失败 10 次 (超过失败阈值)
        for _ in range(10):
            handler.record_sensor_reading('imu', None, timestamp=time.time())
        
        assert handler._sensor_states['imu'] == SensorDropoutHandler.FAILED

    def test_recovery_from_dropout(self):
        """测试从掉线状态恢复"""
        handler = SensorDropoutHandler()
        
        # 制造掉线
        for _ in range(3):
            handler.record_sensor_reading('camera', None, timestamp=time.time())
        assert handler._sensor_states['camera'] == SensorDropoutHandler.DROPOUT
        
        # 恢复正常
        handler.record_sensor_reading('camera', np.ones((10, 10, 3)), timestamp=time.time())
        assert handler._sensor_states['camera'] == SensorDropoutHandler.HEALTHY

    def test_fallback_hold_last(self):
        """测试 HOLD_LAST 降级策略"""
        handler = SensorDropoutHandler()
        
        # 记录正常值
        data = np.array([1.0, 2.0, 3.0])
        handler.record_sensor_reading('lidar', data, timestamp=time.time())
        
        # 模拟掉线
        handler.record_sensor_reading('lidar', None, timestamp=time.time())
        handler._sensor_states['lidar'] = SensorDropoutHandler.DROPOUT
        
        # 获取降级值 (应返回最后已知值)
        fallback = handler.get_fallback_value('lidar')
        np.testing.assert_array_equal(fallback, data)

    def test_fallback_zero(self):
        """测试 ZERO 降级策略"""
        handler = SensorDropoutHandler()
        
        # 制造掉线
        handler.record_sensor_reading('force', None, timestamp=time.time())
        handler._sensor_states['force'] = SensorDropoutHandler.DROPOUT
        
        # 获取降级值 (应返回零值)
        fallback = handler.get_fallback_value('force')
        assert fallback is not None
        if isinstance(fallback, np.ndarray):
            assert np.allclose(fallback, 0)

    def test_extrapolate_lidar(self):
        """测试 EXTRAPOLATE 降级策略 (激光雷达)"""
        handler = SensorDropoutHandler()
        
        # 记录历史轨迹 (物体向外移动)
        for i in range(5):
            data = np.array([1.0 + i * 0.1] * 10)
            handler.record_sensor_reading('lidar', data, timestamp=time.time() + i)
        
        # 模拟掉线
        handler.record_sensor_reading('lidar', None, timestamp=time.time() + 5)
        handler._sensor_states['lidar'] = SensorDropoutHandler.DROPOUT
        
        # 获取降级值 (应外推)
        fallback = handler.get_fallback_value('lidar')
        assert fallback is not None
        assert isinstance(fallback, np.ndarray)

    def test_emergency_mode(self):
        """测试紧急模式"""
        handler = SensorDropoutHandler()
        
        # 创建安全动作
        safe_action = VLAAction()
        safe_action.vx = 0.0
        safe_action.vy = 0.0
        
        handler.set_emergency_mode(enabled=True, last_safe_action=safe_action)
        
        assert handler._emergency_mode is True
        assert handler.get_emergency_action() is safe_action

    def test_health_report(self):
        """测试健康报告"""
        handler = SensorDropoutHandler()
        
        handler.record_sensor_reading('camera', np.zeros((10, 10, 3)), timestamp=time.time())
        handler.record_sensor_reading('lidar', None, timestamp=time.time())
        
        for _ in range(3):
            handler.record_sensor_reading('lidar', None, timestamp=time.time())
        
        report = handler.get_health_report()
        
        assert 'camera' in report['sensors']
        assert 'lidar' in report['sensors']
        assert report['sensors']['camera'] == SensorDropoutHandler.HEALTHY
        assert report['sensors']['lidar'] == SensorDropoutHandler.DROPOUT
        assert report['emergency_mode'] is False

    def test_is_sensor_available(self):
        """测试传感器可用性检查"""
        handler = SensorDropoutHandler()
        
        handler.record_sensor_reading('camera', np.zeros((10, 10, 3)), timestamp=time.time())
        handler.record_sensor_reading('lidar', None, timestamp=time.time())
        
        for _ in range(10):
            handler.record_sensor_reading('lidar', None, timestamp=time.time())
        
        assert handler.is_sensor_available('camera') is True
        assert handler.is_sensor_available('lidar') is False

    def test_critical_sensors_all_failed(self):
        """测试所有关键传感器都失败"""
        handler = SensorDropoutHandler()
        
        # 两个关键传感器都失败
        for _ in range(10):
            handler.record_sensor_reading('lidar', None, timestamp=time.time())
            handler.record_sensor_reading('camera', None, timestamp=time.time())
        
        result = handler.is_critical_sensors_available(['lidar', 'camera'])
        assert result is False

    def test_critical_sensors_one_available(self):
        """测试至少一个关键传感器可用"""
        handler = SensorDropoutHandler()
        
        # lidar 失败，camera 正常
        for _ in range(10):
            handler.record_sensor_reading('lidar', None, timestamp=time.time())
        
        handler.record_sensor_reading('camera', np.zeros((10, 10, 3)), timestamp=time.time())
        
        result = handler.is_critical_sensors_available(['lidar', 'camera'])
        assert result is True

    def test_set_strategy(self):
        """测试设置降级策略"""
        handler = SensorDropoutHandler()
        
        handler.set_strategy('camera', SensorDropoutHandler.DropoutStrategy.ZERO)
        assert handler._strategies['camera'] == SensorDropoutHandler.DropoutStrategy.ZERO

    def test_dropout_counter_increment(self):
        """测试掉线计数增加"""
        handler = SensorDropoutHandler()
        
        # 第一次掉线
        for _ in range(3):
            handler.record_sensor_reading('camera', None, timestamp=time.time())
        
        # 恢复后再次掉线
        handler.record_sensor_reading('camera', np.zeros((10, 10, 3)), timestamp=time.time())
        for _ in range(3):
            handler.record_sensor_reading('camera', None, timestamp=time.time())
        
        assert handler._sensor_dropout_counts['camera'] == 2

    def test_base_pose_extrapolation(self):
        """测试 base_pose 外推"""
        handler = SensorDropoutHandler()
        
        # 记录历史轨迹 (AGV 向右移动)
        for i in range(5):
            data = np.array([float(i) * 0.1, 0.0, 0.0])
            handler.record_sensor_reading('base_pose', data, timestamp=time.time() + i)
        
        # 模拟掉线
        handler.record_sensor_reading('base_pose', None, timestamp=time.time() + 5)
        handler._sensor_states['base_pose'] = SensorDropoutHandler.DROPOUT
        
        # 获取降级值
        fallback = handler.get_fallback_value('base_pose')
        assert fallback is not None


# ============================================================
# VLAInferencePipeline 鲁棒性测试
# ============================================================

class TestVLAInferencePipelineRobustness:
    """VLA推理管道鲁棒性测试"""

    def test_pipeline_with_sensor_handler(self):
        """测试管道创建时包含传感器处理器"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config=config)
        
        assert hasattr(pipeline, 'sensor_handler')
        assert isinstance(pipeline.sensor_handler, SensorDropoutHandler)

    def test_get_sensor_health(self):
        """测试获取传感器健康状态"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config=config)
        
        health = pipeline.get_sensor_health()
        assert 'sensors' in health
        assert 'emergency_mode' in health
        assert health['emergency_mode'] is False

    def test_sensor_callback_with_dropout_tracking(self):
        """测试传感器回调与掉线追踪集成"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config=config)
        
        call_count = [0]
        def failing_camera():
            call_count[0] += 1
            if call_count[0] < 3:
                return np.zeros((224, 224, 3), dtype=np.float32)
            return None
        
        pipeline.register_sensor_callback('camera', failing_camera)
        
        # 触发几次回调
        for _ in range(5):
            pipeline._build_perception_frame()
        
        health = pipeline.get_sensor_health()
        assert 'camera' in health['sensors']

    def test_pipeline_emergency_mode_on_critical_sensor_failure(self):
        """测试关键传感器失败时进入紧急模式"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config=config)
        
        # 强制 lidar 为 critical sensor 并让它失败
        pipeline._critical_sensors = ['lidar']
        
        # 注册一直失败的 lidar
        def failing_lidar():
            return None
        
        pipeline.register_sensor_callback('lidar', failing_lidar)
        
        # 触发多次掉线
        for _ in range(12):
            pipeline._build_perception_frame()
        
        health = pipeline.get_sensor_health()
        assert health['emergency_mode'] is True

    def test_pipeline_dropout_stats(self):
        """测试管道掉线统计"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config=config)
        
        def failing_camera():
            return None
        
        pipeline.register_sensor_callback('camera', failing_camera)
        
        # 触发掉线
        for _ in range(5):
            pipeline._build_perception_frame()
        
        stats = pipeline.get_stats()
        assert 'total_sensor_dropouts' in stats

    def test_factory_creates_pipeline_with_sensor_handler(self):
        """测试工厂函数创建包含传感器处理器的管道"""
        pipeline = create_vla_inference_pipeline(grade="S")
        
        assert hasattr(pipeline, 'sensor_handler')
        assert isinstance(pipeline.sensor_handler, SensorDropoutHandler)

    def test_pipeline_reset_clears_sensor_handler(self):
        """测试管道重置后传感器处理器状态"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config=config)
        
        pipeline.register_sensor_callback('camera', lambda: np.zeros((224, 224, 3)))
        pipeline._build_perception_frame()
        
        # 重置
        pipeline.reset()
        
        # 处理器应该仍可工作
        health = pipeline.get_sensor_health()
        assert health is not None


class TestVLAInferencePipelineSensorIntegration:
    """VLA推理管道传感器集成测试"""

    def test_multi_sensor_fusion(self):
        """测试多传感器融合"""
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        model = create_vla_model(grade="S")
        pipeline = VLAInferencePipeline(config=config, vla_model=model)
        
        pipeline.start()
        
        # 注入多传感器数据
        pipeline.push_sensor_data('camera', np.zeros((224, 224, 3), dtype=np.float32))
        pipeline.push_sensor_data('lidar', np.ones(360))
        pipeline.push_sensor_data('battery_level', 0.8)
        pipeline.set_instruction("Move forward")
        
        # 触发推理
        frame = pipeline._build_perception_frame()
        assert frame is not None
        
        pipeline.stop()

    def test_pipeline_with_continuous_policy(self):
        """测试连续推理策略"""
        config = VLAPipelineConfig(
            grade="S",
            safety_enabled=False,
            inference_policy=InferencePolicy.CONTINUOUS,
        )
        pipeline = VLAInferencePipeline(config=config)
        
        pipeline.start()
        time.sleep(0.3)  # 让推理循环运行一会儿
        pipeline.stop()
        
        stats = pipeline.get_stats()
        assert stats['is_running'] is False

    def test_pipeline_with_single_shot_policy(self):
        """测试单步推理策略"""
        config = VLAPipelineConfig(
            grade="S",
            safety_enabled=False,
            inference_policy=InferencePolicy.SINGLE_SHOT,
        )
        model = create_vla_model(grade="S")
        pipeline = VLAInferencePipeline(config=config, vla_model=model)
        
        pipeline.start()
        
        # 单步推理
        pipeline.push_sensor_data('camera', np.zeros((224, 224, 3), dtype=np.float32))
        pipeline.set_instruction("Test")
        
        frame = pipeline._build_perception_frame()
        if frame:
            output = pipeline.trigger_inference(frame)
            assert output is not None
            assert isinstance(output.action, VLAAction)
        
        pipeline.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
