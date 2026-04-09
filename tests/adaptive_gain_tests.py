"""
自适应增益调度模块测试
=====================

测试 AdaptiveGainScheduler、BlendController、ModelReferenceAdaptiveController
及其 AGV 五级规格
覆盖: 误差自适应、负载自适应、温度补偿、多模态融合、纳秒精度 blend
"""

import unittest
import numpy as np
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.adaptive_gain import (
    AdaptiveGainScheduler, GainBlendController, ModelReferenceAdaptiveController,
    AdaptationStrategy, AdaptationState, GainSchedule,
    get_adaptive_gain_spec, AGV_ADAPTIVE_GAIN_GRADES
)


class TestAdaptiveGainScheduler(unittest.TestCase):
    """测试自适应增益调度器"""

    def setUp(self):
        self.scheduler = AdaptiveGainScheduler(
            strategy=AdaptationStrategy.MULTI_MODAL,
            scheduler_id="test_scheduler"
        )

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.scheduler.scheduler_id, "test_scheduler")
        self.assertEqual(self.scheduler.strategy, AdaptationStrategy.MULTI_MODAL)

    def test_error_based_adaptation(self):
        """测试基于误差的自适应增益"""
        # 大误差 → 增益放大
        kp, ki, kd, kf = self.scheduler.update(error=1.0, dt=0.01)
        self.assertGreater(kp, self.scheduler.schedule.kp_base)

        # 小误差 → 增益接近基线
        self.scheduler.reset()
        gains = self.scheduler.get_gains()
        kp2, ki2, kd2, kf2 = self.scheduler.update(error=0.01, dt=0.01, temperature=25.0)
        self.assertAlmostEqual(kp2, self.scheduler.schedule.kp_base, delta=1.0)

    def test_load_based_adaptation(self):
        """测试基于负载的自适应增益"""
        scheduler = AdaptiveGainScheduler(strategy=AdaptationStrategy.LOAD_BASED)
        kp, ki, kd, kf = scheduler.update(error=0.0, dt=0.01, load_estimate=3.0)
        self.assertGreater(kp, scheduler.schedule.kp_base)

    def test_temperature_compensation(self):
        """测试温度补偿"""
        scheduler = AdaptiveGainScheduler(strategy=AdaptationStrategy.TEMP_BASED)
        kp, ki, kd, kf = scheduler.update(error=0.0, dt=0.01, temperature=60.0)
        self.assertLess(kp, scheduler.schedule.kp_base)

    def test_velocity_based_adaptation(self):
        """测试基于速度的前馈增益"""
        scheduler = AdaptiveGainScheduler(strategy=AdaptationStrategy.VELOCITY_BASED)
        kp, ki, kd, kf = scheduler.update(error=0.0, dt=0.01, velocity=2.0, acceleration=1.0)
        gains = scheduler.get_gains()
        self.assertGreater(gains['kf'], 0.0)

    def test_multi_modal_fusion(self):
        """测试多模态增益融合"""
        scheduler = AdaptiveGainScheduler(strategy=AdaptationStrategy.MULTI_MODAL)
        kp, ki, kd, kf = scheduler.update(
            error=0.5, dt=0.01,
            load_estimate=2.0, temperature=45.0,
            velocity=0.5, acceleration=0.2
        )
        self.assertGreater(kp, 0.0)

    def test_get_gains(self):
        """测试获取增益"""
        gains = self.scheduler.get_gains()
        self.assertIn('kp', gains)
        self.assertIn('ki', gains)
        self.assertIn('kd', gains)
        self.assertIn('kf', gains)
        self.assertIn('ratio', gains)

    def test_reset(self):
        """测试状态重置"""
        self.scheduler.update(error=1.0, dt=0.01)
        self.scheduler.reset()
        gains = self.scheduler.get_gains()
        self.assertEqual(gains['kp'], self.scheduler.schedule.kp_base)

    def test_enable_disable(self):
        """测试启用/禁用"""
        self.scheduler.disable()
        gains_before = self.scheduler.get_gains()
        self.scheduler.update(error=10.0, dt=0.01)
        gains_after = self.scheduler.get_gains()
        self.assertEqual(gains_before['kp'], gains_after['kp'])
        self.scheduler.enable()
        self.scheduler.update(error=10.0, dt=0.01)
        gains_enabled = self.scheduler.get_gains()
        self.assertGreater(gains_enabled['kp'], self.scheduler.schedule.kp_base)

    def test_history(self):
        """测试历史记录"""
        for i in range(15):
            self.scheduler.update(error=float(i) * 0.1, dt=0.01)
        history = self.scheduler.get_history(n=5)
        self.assertEqual(len(history), 5)

    def test_callback(self):
        """测试增益变化回调"""
        called = {'count': 0}

        def callback(kp, ki, kd, kf):
            called['count'] += 1

        self.scheduler.on_gain_change(callback)
        self.scheduler.update(error=1.0, dt=0.01)
        self.assertEqual(called['count'], 1)


class TestGainBlendController(unittest.TestCase):
    """测试纳秒精度增益混合控制器"""

    def test_creation(self):
        """测试创建"""
        blend = GainBlendController(controller_id="blend_test", blend_time=0.5)
        self.assertEqual(blend.controller_id, "blend_test")
        self.assertEqual(blend.blend_time, 0.5)

    def test_register_and_switch_config(self):
        """测试配置注册和切换"""
        blend = GainBlendController(blend_time=0.1)
        cfg1 = GainSchedule(kp_base=10.0, ki_base=1.0, kd_base=2.0)
        cfg2 = GainSchedule(kp_base=20.0, ki_base=2.0, kd_base=4.0)
        blend.register_config("low", cfg1)
        blend.register_config("high", cfg2)
        self.assertEqual(blend.get_current_config_name(), "low")
        blend.switch_config("high", blend=True)
        self.assertTrue(blend.is_blending())

    def test_instant_switch(self):
        """测试即时切换 (无 blend)"""
        blend = GainBlendController(blend_time=0.5)
        cfg1 = GainSchedule(kp_base=10.0)
        cfg2 = GainSchedule(kp_base=20.0)
        blend.register_config("low", cfg1)
        blend.register_config("high", cfg2)
        blend.switch_config("high", blend=False)
        self.assertFalse(blend.is_blending())
        gains = blend.update(dt=0.01)
        self.assertEqual(gains.kp_base, 20.0)

    def test_nano_second_precision(self):
        """测试纳秒精度 (perf_counter_ns)"""
        blend = GainBlendController(blend_time=0.05)
        cfg1 = GainSchedule(kp_base=10.0)
        cfg2 = GainSchedule(kp_base=20.0)
        blend.register_config("low", cfg1)
        blend.register_config("high", cfg2)
        blend.switch_config("high", blend=True)
        # 首次调用 - 记录起始时间，不推进
        blend.update(dt=0.01)
        # 第二次调用 - 开始 blend
        blend.update(dt=0.01)
        # blend 完成后应该切换到目标配置
        time.sleep(0.06)
        blend.update(dt=0.01)
        self.assertFalse(blend.is_blending())


class TestAdaptiveGainGrades(unittest.TestCase):
    """测试 AGV 五级自适应增益规格"""

    def test_all_grades_have_spec(self):
        """测试所有等级都有规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_adaptive_gain_spec(grade)
            self.assertIn('enabled', spec)
            self.assertIn('max_rate', spec)

    def test_grade_enabled_progression(self):
        """测试等级越高功能越强"""
        spec_s = get_adaptive_gain_spec('S')
        self.assertFalse(spec_s['enabled'])
        for grade in ['M', 'L', 'XL', 'XXL']:
            spec = get_adaptive_gain_spec(grade)
            self.assertTrue(spec['enabled'])

    def test_max_rate_increases_with_grade(self):
        """测试等级越高自适应速率越大"""
        rates = [get_adaptive_gain_spec(g)['max_rate'] for g in ['M', 'L', 'XL', 'XXL']]
        self.assertEqual(rates, sorted(rates))


class TestGainSchedule(unittest.TestCase):
    """测试增益调度配置"""

    def test_default_schedule(self):
        """测试默认配置"""
        schedule = GainSchedule()
        self.assertEqual(schedule.kp_base, 10.0)
        self.assertEqual(schedule.schedule_type, 'linear')
        self.assertEqual(schedule.bounds, (0.1, 10.0))

    def test_custom_schedule(self):
        """测试自定义配置"""
        schedule = GainSchedule(kp_base=20.0, ki_base=5.0, schedule_type='exponential')
        self.assertEqual(schedule.kp_base, 20.0)
        self.assertEqual(schedule.schedule_type, 'exponential')


class TestAdaptationState(unittest.TestCase):
    """测试自适应状态"""

    def test_state_initialization(self):
        """测试状态初始化"""
        state = AdaptationState()
        self.assertEqual(state.current_kp, 10.0)
        self.assertEqual(state.adaptation_ratio, 1.0)
        self.assertEqual(state.confidence, 1.0)


class TestModelReferenceAdaptiveController(unittest.TestCase):
    """测试模型参考自适应控制 (MRAC)"""

    def test_creation(self):
        """测试 MRAC 创建"""
        def ref_model(t, dt):
            return np.sin(t)

        mrac = ModelReferenceAdaptiveController(
            reference_model=ref_model,
            controller_id="mrac_test",
            adaptation_gain=10.0
        )
        self.assertEqual(mrac.controller_id, "mrac_test")
        self.assertEqual(mrac.adaptation_gain, 10.0)
        self.assertEqual(len(mrac._theta), 3)

    def test_compute_control(self):
        """测试 MRAC 控制计算"""
        def ref_model(t, dt):
            return 1.0

        mrac = ModelReferenceAdaptiveController(
            reference_model=ref_model,
            adaptation_gain=5.0
        )
        u = mrac.compute_control(system_state=0.5, reference_input=1.0, dt=0.01)
        self.assertIsInstance(u, float)

    def test_reset(self):
        """测试参数重置"""
        def ref_model(t, dt):
            return t

        mrac = ModelReferenceAdaptiveController(ref_model, adaptation_gain=5.0)
        mrac.compute_control(system_state=0.5, reference_input=1.0, dt=0.01)
        mrac.reset()
        np.testing.assert_array_equal(mrac.get_parameters(), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
