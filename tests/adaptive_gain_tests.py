"""
自适应增益调度模块测试
测试 AdaptiveGainScheduler, GainBlendController, ModelReferenceAdaptiveController
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.adaptive_gain import (
    AdaptiveGainScheduler, GainSchedule, AdaptationState,
    GainBlendController, ModelReferenceAdaptiveController,
    AdaptationStrategy, get_adaptive_gain_spec, AGV_ADAPTIVE_GAIN_GRADES
)


class TestAdaptiveGainScheduler(unittest.TestCase):
    """测试自适应增益调度器"""

    def setUp(self):
        self.scheduler = AdaptiveGainScheduler(
            strategy=AdaptationStrategy.MULTI_MODAL,
            schedule=GainSchedule(kp_base=10.0, ki_base=1.0, kd_base=2.0, kf_base=0.5),
            scheduler_id="test_scheduler"
        )

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.scheduler.scheduler_id, "test_scheduler")
        self.assertEqual(self.scheduler.strategy, AdaptationStrategy.MULTI_MODAL)
        self.assertIsNotNone(self.scheduler.schedule)
        self.assertTrue(self.scheduler._is_enabled)

    def test_update_basic(self):
        """测试基本更新"""
        kp, ki, kd, kf = self.scheduler.update(
            error=0.1, dt=0.01, load_estimate=1.0,
            temperature=25.0, velocity=0.0, acceleration=0.0
        )
        self.assertIsInstance(kp, float)
        self.assertIsInstance(ki, float)
        self.assertIsInstance(kd, float)
        self.assertIsInstance(kf, float)

    def test_update_error_based(self):
        """测试基于误差的自适应"""
        sched = AdaptiveGainScheduler(strategy=AdaptationStrategy.ERROR_BASED)
        gains_small = sched.update(error=0.01, dt=0.01)
        gains_large = sched.update(error=1.0, dt=0.01)
        # 误差大时应增大增益
        self.assertGreater(gains_large[0], gains_small[0])

    def test_update_load_based(self):
        """测试基于负载的自适应"""
        sched = AdaptiveGainScheduler(strategy=AdaptationStrategy.LOAD_BASED)
        gains_light = sched.update(error=0.1, dt=0.01, load_estimate=0.5)
        gains_heavy = sched.update(error=0.1, dt=0.01, load_estimate=2.0)
        # 重负载时应增大增益
        self.assertGreater(gains_heavy[0], gains_light[0])

    def test_update_temperature_based(self):
        """测试基于温度的自适应"""
        sched = AdaptiveGainScheduler(strategy=AdaptationStrategy.TEMP_BASED)
        gains_nominal = sched.update(error=0.1, dt=0.01, temperature=25.0)
        gains_hot = sched.update(error=0.1, dt=0.01, temperature=60.0)
        # 高温时应补偿增益
        self.assertIsInstance(gains_nominal[0], float)
        self.assertIsInstance(gains_hot[0], float)

    def test_update_velocity_based(self):
        """测试基于速度的自适应"""
        sched = AdaptiveGainScheduler(strategy=AdaptationStrategy.VELOCITY_BASED)
        gains_low = sched.update(error=0.1, dt=0.01, velocity=0.0, acceleration=0.0)
        gains_high = sched.update(error=0.1, dt=0.01, velocity=2.0, acceleration=1.0)
        # 高速时应增大增益
        self.assertGreater(gains_high[0], gains_low[0])

    def test_multi_modal_strategy(self):
        """测试多模态融合策略"""
        gains = self.scheduler.update(
            error=0.5, dt=0.01,
            load_estimate=1.5, temperature=30.0,
            velocity=1.0, acceleration=0.5
        )
        self.assertEqual(len(gains), 4)
        # 验证各增益在合理范围
        self.assertGreater(gains[0], 0)  # kp > 0
        self.assertGreater(gains[1], 0)  # ki > 0

    def test_disabled_returns_base(self):
        """测试禁用时返回基础增益"""
        self.scheduler.disable()
        gains = self.scheduler.update(error=1.0, dt=0.01)
        self.assertEqual(gains[0], self.scheduler.schedule.kp_base)
        self.scheduler.enable()

    def test_gain_bounds(self):
        """测试增益边界约束"""
        # 极大误差不应导致增益爆炸
        for _ in range(100):
            gains = self.scheduler.update(error=10.0, dt=0.01)
        kp = gains[0]
        # 应该有上界约束
        self.assertLess(kp, self.scheduler.schedule.kp_base * self.scheduler.schedule.bounds[1] * 5)

    def test_get_gains(self):
        """测试获取当前增益"""
        self.scheduler.update(error=0.1, dt=0.01)
        gains = self.scheduler.get_gains()
        self.assertIn('kp', gains)
        self.assertIn('ki', gains)
        self.assertIn('kd', gains)
        self.assertIn('kf', gains)
        self.assertIn('ratio', gains)
        self.assertIn('confidence', gains)

    def test_reset(self):
        """测试重置"""
        self.scheduler.update(error=0.5, dt=0.01)
        self.scheduler.reset()
        self.assertEqual(len(self.scheduler._history), 0)

    def test_history(self):
        """测试历史记录"""
        for i in range(15):
            self.scheduler.update(error=float(i) * 0.1, dt=0.01)
        history = self.scheduler.get_history(n=5)
        self.assertEqual(len(history), 5)

    def test_set_weights(self):
        """测试设置权重"""
        self.scheduler.set_weights(error=0.5, load=0.3, temperature=0.1, velocity=0.1)
        total = (self.scheduler._weights['error'] +
                 self.scheduler._weights['load'] +
                 self.scheduler._weights['temperature'] +
                 self.scheduler._weights['velocity'])
        self.assertAlmostEqual(total, 1.0, places=5)


class TestGainBlendController(unittest.TestCase):
    """测试增益混合控制器"""

    def setUp(self):
        self.blender = GainBlendController(blend_time=0.1)
        # 注册多种配置
        self.blender.register_config("stiff", GainSchedule(
            kp_base=20.0, ki_base=2.0, kd_base=5.0, kf_base=1.0))
        self.blender.register_config("compliant", GainSchedule(
            kp_base=5.0, ki_base=0.5, kd_base=1.0, kf_base=0.2))
        self.blender.register_config("medium", GainSchedule(
            kp_base=10.0, ki_base=1.0, kd_base=2.0, kf_base=0.5))

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.blender.controller_id, "blend_0")
        self.assertEqual(self.blender.blend_time, 0.1)

    def test_register_config(self):
        """测试注册配置"""
        self.assertIsNotNone(self.blender._configs.get("stiff"))
        self.assertIsNotNone(self.blender._configs.get("compliant"))

    def test_switch_immediate(self):
        """测试即时切换"""
        self.blender.switch_config("compliant", blend=False)
        self.assertEqual(self.blender.get_current_config_name(), "compliant")
        self.assertFalse(self.blender.is_blending())

    def test_switch_with_blend(self):
        """测试带混合的切换: blending完成后验证配置切换"""
        # 初始配置为 stiff (注册的第一个配置)，切换到 compliant
        self.blender.switch_config("compliant", blend=True)
        # blending 期间配置名称保持为原配置 stiff
        self.assertEqual(self.blender.get_current_config_name(), "stiff")
        self.assertTrue(self.blender.is_blending())
        # 混合切换完成 (blend_time=0.1s, 多次update直到完成)
        import time
        deadline = time.time() + 1.0
        while self.blender.is_blending() and time.time() < deadline:
            self.blender.update(dt=0.01)
        # 完成后应切换到 compliant 配置
        self.assertEqual(self.blender.get_current_config_name(), "compliant")
        self.assertFalse(self.blender.is_blending())

    def test_update_no_blend(self):
        """测试无混合更新"""
        gains = self.blender.update(dt=0.01)
        self.assertEqual(gains.kp_base, self.blender._current_gains.kp_base)

    def test_update_with_blend(self):
        """测试混合更新"""
        self.blender.switch_config("stiff", blend=True)
        # 多次更新直到混合完成
        for _ in range(20):
            self.blender.update(dt=0.01)
        # 最终应切换到stiff配置
        self.assertEqual(self.blender.get_current_config_name(), "stiff")
        self.assertFalse(self.blender.is_blending())

    def test_lerp(self):
        """测试线性插值"""
        result = self.blender._lerp(0.0, 10.0, 0.5)
        self.assertEqual(result, 5.0)

    def test_ease_in_out(self):
        """测试缓动函数"""
        result = self.blender._ease_in_out(0.5)
        self.assertEqual(result, 0.5)  # 中点应恰好为0.5


class TestModelReferenceAdaptiveController(unittest.TestCase):
    """测试模型参考自适应控制器"""

    def setUp(self):
        # 简单一阶参考模型
        def ref_model(r, dt):
            return r * 0.9
        self.mrac = ModelReferenceAdaptiveController(
            reference_model=ref_model,
            adaptation_gain=5.0
        )

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.mrac.controller_id, "mrac_0")
        self.assertEqual(self.mrac.adaptation_gain, 5.0)

    def test_compute_control(self):
        """测试控制量计算"""
        control = self.mrac.compute_control(
            system_state=0.5,
            reference_input=1.0,
            dt=0.01
        )
        self.assertIsInstance(control, float)

    def test_convergence(self):
        """测试参数收敛"""
        # 运行多次更新
        for _ in range(100):
            self.mrac.compute_control(
                system_state=0.5,
                reference_input=1.0,
                dt=0.01
            )
        params = self.mrac.get_parameters()
        self.assertEqual(len(params), 3)
        # 参数应有限
        self.assertTrue(np.all(np.isfinite(params)))

    def test_reset(self):
        """测试重置"""
        self.mrac.compute_control(1.0, 1.0, 0.01)
        self.mrac.reset()
        params = self.mrac.get_parameters()
        self.assertTrue(np.all(params == 0.0))


class TestAdaptiveGainGrades(unittest.TestCase):
    """测试AGV五级自适应增益规格"""

    def test_all_grades_have_spec(self):
        """测试所有等级都有规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_adaptive_gain_spec(grade)
            self.assertIn('enabled', spec)
            self.assertIn('strategy', spec)
            self.assertIn('max_rate', spec)

    def test_grade_s_inactive(self):
        """测试S级无自适应"""
        spec = get_adaptive_gain_spec('S')
        self.assertFalse(spec['enabled'])

    def test_grade_m_to_xxl_active(self):
        """测试M级以上启用自适应"""
        for grade in ['M', 'L', 'XL', 'XXL']:
            spec = get_adaptive_gain_spec(grade)
            self.assertTrue(spec['enabled'])

    def test_rate_increases_with_grade(self):
        """测试等级越高最大自适应速率越大"""
        prev_rate = 0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_adaptive_gain_spec(grade)
            self.assertGreaterEqual(spec['max_rate'], prev_rate)
            prev_rate = spec['max_rate']


class TestAdaptiveGainEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_zero_dt(self):
        """测试零时间步长"""
        sched = AdaptiveGainScheduler()
        gains = sched.update(error=0.1, dt=0.0)
        self.assertIsInstance(gains[0], float)

    def test_large_error(self):
        """测试大误差"""
        sched = AdaptiveGainScheduler()
        gains = sched.update(error=100.0, dt=0.01)
        self.assertTrue(np.isfinite(gains[0]))

    def test_negative_load(self):
        """测试负负载"""
        sched = AdaptiveGainScheduler(strategy=AdaptationStrategy.LOAD_BASED)
        gains = sched.update(error=0.1, dt=0.01, load_estimate=-0.5)
        self.assertTrue(np.isfinite(gains[0]))

    def test_extreme_temperature(self):
        """测试极端温度"""
        sched = AdaptiveGainScheduler(strategy=AdaptationStrategy.TEMP_BASED)
        gains = sched.update(error=0.1, dt=0.01, temperature=100.0)
        self.assertTrue(np.isfinite(gains[0]))

    def test_callback(self):
        """测试增益变化回调"""
        sched = AdaptiveGainScheduler()
        callback_called = {'count': 0}

        def callback(kp, ki, kd, kf):
            callback_called['count'] += 1

        sched.on_gain_change(callback)
        for _ in range(5):
            sched.update(error=0.1, dt=0.01)
        self.assertEqual(callback_called['count'], 5)

    def test_gain_history_limit(self):
        """测试历史记录上限"""
        sched = AdaptiveGainScheduler()
        for i in range(200):
            sched.update(error=float(i) * 0.01, dt=0.01)
        # 不应超过最大值
        self.assertLessEqual(len(sched._history), sched._max_history)


if __name__ == '__main__':
    unittest.main()
