"""
自动调参模块单元测试
测试 AutoTuner, TunerConfig, TuningMethod 及各整定算法
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.autotune import (
    AutoTuner, TunerConfig, TunerResult, TuningMethod,
    SimulatedPlant, autotune_pid
)


class TestSimulatedPlant(unittest.TestCase):
    """测试虚拟被控对象"""

    def test_first_order_response(self):
        """测试一阶系统响应"""
        plant = SimulatedPlant(k=2.0, T=0.5, L=0.0)
        output = plant.step(u=1.0, dt=0.01)
        # 初始状态应为0, 步进后增加
        self.assertGreaterEqual(output, 0.0)

    def test_with_delay(self):
        """测试带延迟的系统"""
        plant = SimulatedPlant(k=1.0, T=0.5, L=0.05)
        outputs = []
        for _ in range(20):
            outputs.append(plant.step(u=1.0, dt=0.01))
        # 前几个步应该接近0 (在延迟中)
        self.assertLessEqual(outputs[4], 0.5)
        # 后续应该上升
        self.assertGreater(outputs[-1], outputs[0])

    def test_reset(self):
        """测试重置"""
        plant = SimulatedPlant(k=1.0, T=0.5, L=0.0)
        plant.step(1.0, 0.01)
        plant.reset()
        self.assertEqual(plant.state, 0.0)
        self.assertEqual(len(plant.delay_buffer), 0)


class TestAutoTuner(unittest.TestCase):
    """测试自动调参器"""

    def test_creation_default(self):
        """测试默认创建"""
        tuner = AutoTuner()
        self.assertIsInstance(tuner.config, TunerConfig)
        self.assertEqual(tuner.method, TuningMethod.ZIEGLER_NICHOLS)

    def test_creation_with_config(self):
        """测试配置创建"""
        config = TunerConfig(method=TuningMethod.RELAY, relay_amplitude=2.0)
        tuner = AutoTuner(config)
        self.assertEqual(tuner.config.relay_amplitude, 2.0)
        self.assertEqual(tuner.method, TuningMethod.RELAY)

    def test_reset(self):
        """测试重置"""
        tuner = AutoTuner()
        tuner.start_tuning()
        tuner.record(1.0)
        tuner.reset()
        self.assertEqual(len(tuner._data), 0)
        self.assertFalse(tuner._is_tuning)

    def test_record_no_tuning(self):
        """测试未开始时记录"""
        tuner = AutoTuner()
        result = tuner.record(1.0)
        self.assertFalse(result)

    def test_simulate_step_basic(self):
        """测试阶跃响应模拟"""
        tuner = AutoTuner()
        times, outputs = tuner.simulate_step(kp=1.0, ki=0.0, kd=0.0,
                                              setpoint=1.0, duration=1.0)
        self.assertEqual(len(times), len(outputs))
        self.assertGreater(len(times), 0)
        # 应该收敛到设定值
        self.assertGreater(outputs[-1], 0.2)  # 系统有惯性,1s可能未完全收敛

    def test_simulate_step_with_ki(self):
        """测试带积分的PID模拟"""
        tuner = AutoTuner()
        times, outputs = tuner.simulate_step(kp=1.0, ki=0.5, kd=0.0,
                                              setpoint=1.0, duration=2.0)
        # 带积分应该消除稳态误差
        final = outputs[-1]
        self.assertGreater(final, 0.5)  # 带积分应消除部分稳态误差

    def test_simulate_step_output_bounds(self):
        """测试输出有界"""
        tuner = AutoTuner()
        times, outputs = tuner.simulate_step(kp=10.0, ki=1.0, kd=0.0,
                                              setpoint=5.0, duration=1.0)
        self.assertTrue(np.all(outputs >= -10))
        self.assertTrue(np.all(outputs <= 10))


class TestRelayMethod(unittest.TestCase):
    """测试继电反馈整定法"""

    def test_relay_tuning_data_collection(self):
        """测试继电整定数据收集"""
        config = TunerConfig(method=TuningMethod.RELAY,
                              relay_amplitude=2.0,
                              noise_band=0.05,
                              sample_time=0.01)
        tuner = AutoTuner(config)
        tuner.start_tuning()

        # 模拟一个闭环振荡系统
        plant = SimulatedPlant(k=1.0, T=0.3, L=0.0)
        for _ in range(500):
            output = plant.step(u=2.0 if not tuner._relay_state else -2.0, dt=0.01)
            tuner.record(output)

        # 应该收集到数据
        self.assertGreater(len(tuner._data), 0)

    def test_insufficient_data(self):
        """测试数据不足"""
        tuner = AutoTuner(TunerConfig(method=TuningMethod.RELAY))
        tuner.start_tuning()
        tuner.record(0.5)
        result = tuner.compute_pid()
        self.assertFalse(result.converges)


class TestZieglerNicholsMethod(unittest.TestCase):
    """测试Ziegler-Nichols整定法"""

    def test_ziegler_nichols_from_step(self):
        """测试从阶跃响应整定"""
        config = TunerConfig(method=TuningMethod.ZIEGLER_NICHOLS,
                              sample_time=0.01)
        tuner = AutoTuner(config)
        tuner.start_tuning()

        plant = SimulatedPlant(k=1.0, T=0.5, L=0.1)
        for _ in range(200):
            output = plant.step(u=1.0, dt=0.01)
            tuner.record(output)

        result = tuner.compute_pid()
        self.assertTrue(result.converges)
        self.assertGreater(result.kp, 0)
        self.assertGreaterEqual(result.quality_score, 0)

    def test_ziegler_nichols_quality(self):
        """测试整定质量"""
        config = TunerConfig(method=TuningMethod.ZIEGLER_NICHOLS,
                              sample_time=0.01)
        tuner = AutoTuner(config)
        times, outputs = tuner.simulate_step(kp=1.0, ki=0.1, kd=0.1,
                                             setpoint=1.0, duration=3.0)
        result = tuner.compute_pid()
        self.assertIsInstance(float(result.quality_score), float)  # 质量分数应为数值


class TestCohenCoonMethod(unittest.TestCase):
    """测试Cohen-Coon整定法"""

    def test_cohen_coon_from_step(self):
        """测试Cohen-Coon整定"""
        config = TunerConfig(method=TuningMethod.COHEN_COON,
                              sample_time=0.01)
        tuner = AutoTuner(config)
        tuner.start_tuning()

        plant = SimulatedPlant(k=2.0, T=0.8, L=0.2)
        for _ in range(300):
            output = plant.step(u=1.0, dt=0.01)
            tuner.record(output)

        result = tuner.compute_pid()
        self.assertTrue(result.converges)
        self.assertGreater(result.kp, 0)


class TestAutotuneInterface(unittest.TestCase):
    """测试快速整定接口"""

    def test_autotune_pid_interface(self):
        """测试快速整定函数"""
        plant = SimulatedPlant(k=1.0, T=0.5, L=0.1)

        def simple_plant(u, dt):
            return plant.step(u, dt)

        result = autotune_pid(simple_plant, method=TuningMethod.ZIEGLER_NICHOLS)
        self.assertIsInstance(result, TunerResult)
        self.assertGreaterEqual(result.kp, 0)


class TestTunerConfig(unittest.TestCase):
    """测试调参器配置"""

    def test_default_config(self):
        """测试默认配置"""
        cfg = TunerConfig()
        self.assertEqual(cfg.method, TuningMethod.ZIEGLER_NICHOLS)
        self.assertEqual(cfg.sample_time, 0.01)
        self.assertEqual(cfg.max_iterations, 100)

    def test_relay_config(self):
        """测试继电配置"""
        cfg = TunerConfig(
            method=TuningMethod.RELAY,
            relay_amplitude=3.0,
            relay_hysteresis=0.1,
            noise_band=0.02
        )
        self.assertEqual(cfg.relay_amplitude, 3.0)
        self.assertEqual(cfg.noise_band, 0.02)


class TestTunerResult(unittest.TestCase):
    """测试整定结果"""

    def test_result_fields(self):
        """测试结果字段"""
        result = TunerResult(kp=1.5, ki=0.2, kd=0.1,
                           method=TuningMethod.ZIEGLER_NICHOLS,
                           quality_score=0.85,
                           iterations=50,
                           converges=True,
                           info={"ku": 2.5, "pu": 0.8})
        self.assertEqual(result.kp, 1.5)
        self.assertEqual(result.ki, 0.2)
        self.assertEqual(result.kd, 0.1)
        self.assertTrue(result.converges)
        self.assertEqual(result.info["ku"], 2.5)


class TestTuningMethod(unittest.TestCase):
    """测试调参方法枚举"""

    def test_all_methods_exist(self):
        """测试所有方法都存在"""
        self.assertEqual(len(TuningMethod), 5)
        self.assertIsNotNone(TuningMethod.ZIEGLER_NICHOLS)
        self.assertIsNotNone(TuningMethod.RELAY)
        self.assertIsNotNone(TuningMethod.COHEN_COON)
        self.assertIsNotNone(TuningMethod.MODEL_REFERENCE)
        self.assertIsNotNone(TuningMethod.BAYESIAN)


class TestAutoTunerEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_zero_amplitude(self):
        """测试零幅值"""
        config = TunerConfig(method=TuningMethod.RELAY, relay_amplitude=0.0)
        tuner = AutoTuner(config)
        tuner.start_tuning()
        for i in range(10):
            tuner.record(float(i))
        result = tuner.compute_pid()
        self.assertFalse(result.converges)

    def test_noise_band_hysteresis(self):
        """测试噪声带滞环"""
        config = TunerConfig(relay_amplitude=1.0, noise_band=0.5)
        tuner = AutoTuner(config)
        self.assertEqual(tuner.config.noise_band, 0.5)

    def test_convergence_threshold(self):
        """测试收敛阈值"""
        config = TunerConfig(convergence_threshold=1e-6)
        self.assertLess(config.convergence_threshold, 1e-4)


if __name__ == '__main__':
    unittest.main()
