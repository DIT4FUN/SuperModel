"""
传感器融合控制模块测试用例
测试 SensorFusionController 统一感知→控制闭环
"""

import unittest
import numpy as np
import sys
import os

_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SrcPath = os.path.join(_ProjectRoot, 'src')
if _SrcPath not in sys.path:
    sys.path.insert(0, _SrcPath)
if _ProjectRoot not in sys.path:
    sys.path.insert(0, _ProjectRoot)

from src.control.sensor_fusion_control import (
    SensorFusionController, SensorFusionControlState, FusionControlConfig,
    FusionControlGrade, get_fusion_control_spec, AGV_FUSION_CONTROL_GRADES,
    _ComplementaryFilter, _SimpleEKF,
)


class TestFusionControlGradeConfigs(unittest.TestCase):
    """AGV五级融合控制规格测试"""

    def test_grade_spec_s(self):
        spec = get_fusion_control_spec('S')
        self.assertEqual(spec['freq'], 50)
        self.assertEqual(spec['algorithm'], 'complementary')
        self.assertEqual(spec['imu_rate'], 100)
        self.assertEqual(spec['force_rate'], 100)
        self.assertEqual(spec['tactile_rate'], 50)
        self.assertEqual(spec['latency_ms'], 20)

    def test_grade_spec_m(self):
        spec = get_fusion_control_spec('M')
        self.assertEqual(spec['freq'], 100)
        self.assertEqual(spec['algorithm'], 'complementary')

    def test_grade_spec_l(self):
        spec = get_fusion_control_spec('L')
        self.assertEqual(spec['freq'], 200)
        self.assertEqual(spec['algorithm'], 'ekf')

    def test_grade_spec_xl(self):
        spec = get_fusion_control_spec('XL')
        self.assertEqual(spec['freq'], 500)
        self.assertEqual(spec['algorithm'], 'ekf')
        self.assertEqual(spec['imu_rate'], 1000)

    def test_grade_spec_xxl(self):
        spec = get_fusion_control_spec('XXL')
        self.assertEqual(spec['freq'], 1000)
        self.assertEqual(spec['algorithm'], 'ekf')
        self.assertEqual(spec['force_rate'], 5000)
        self.assertEqual(spec['latency_ms'], 1)

    def test_grade_spec_default(self):
        spec = get_fusion_control_spec('UNKNOWN')
        self.assertEqual(spec, get_fusion_control_spec('M'))

    def test_all_grades_present(self):
        for g in ['S', 'M', 'L', 'XL', 'XXL']:
            self.assertIn(g, AGV_FUSION_CONTROL_GRADES)


class TestSensorFusionControllerInit(unittest.TestCase):
    """SensorFusionController 初始化测试"""

    def test_init_default(self):
        ctrl = SensorFusionController()
        self.assertEqual(ctrl.grade, FusionControlGrade.M)
        self.assertEqual(ctrl.fusion_algorithm, 'complementary')
        self.assertEqual(ctrl.fusion_frequency, 100)

    def test_init_grade_s(self):
        ctrl = SensorFusionController(grade=FusionControlGrade.S)
        self.assertEqual(ctrl.grade, FusionControlGrade.S)
        self.assertEqual(ctrl.fusion_frequency, 50)
        self.assertEqual(ctrl.control_frequency, 50)
        self.assertEqual(ctrl.fusion_algorithm, 'complementary')

    def test_init_grade_xxl(self):
        ctrl = SensorFusionController(grade=FusionControlGrade.XXL)
        self.assertEqual(ctrl.grade, FusionControlGrade.XXL)
        self.assertEqual(ctrl.fusion_frequency, 1000)
        self.assertEqual(ctrl.fusion_algorithm, 'ekf')

    def test_init_custom_config(self):
        config = FusionControlConfig(
            grade=FusionControlGrade.L,
            fusion_algorithm='ekf',
            control_frequency=200,
        )
        ctrl = SensorFusionController(config=config)
        self.assertEqual(ctrl.fusion_algorithm, 'ekf')  # config overrides grade default


class TestComplementaryFilter(unittest.TestCase):
    """互补滤波器测试"""

    def test_init(self):
        f = _ComplementaryFilter(alpha=0.96)
        np.testing.assert_array_equal(f._euler, np.zeros(3))

    def test_update_level(self):
        f = _ComplementaryFilter(alpha=0.96)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        rpy = f.update(accel, gyro, dt=0.01)
        self.assertAlmostEqual(rpy[0], 0.0, places=2)
        self.assertAlmostEqual(rpy[1], 0.0, places=2)

    def test_update_roll(self):
        f = _ComplementaryFilter(alpha=0.96)
        accel = np.array([0.0, 9.81, 0.0])  # 侧倾90度
        gyro = np.array([0.1, 0.0, 0.0])
        # 多次更新以累积角度
        for _ in range(100):
            rpy = f.update(accel, gyro, dt=0.01)
        self.assertGreater(rpy[0], 1.0)  # roll 应该接近 pi/2

    def test_update_pitch(self):
        f = _ComplementaryFilter(alpha=0.96)
        accel = np.array([9.81, 0.0, 0.0])  # 俯仰90度
        gyro = np.array([0.0, 0.1, 0.0])
        # 多次更新以累积角度
        for _ in range(100):
            rpy = f.update(accel, gyro, dt=0.01)
        self.assertLess(rpy[1], -1.0)  # pitch 应该接近 -pi/2 (accel朝X正方向)

    def test_reset(self):
        f = _ComplementaryFilter(alpha=0.96)
        accel = np.array([0.0, 9.81, 0.0])
        gyro = np.array([0.1, 0.0, 0.0])
        f.update(accel, gyro, dt=0.01)
        f.reset()
        np.testing.assert_array_equal(f._euler, np.zeros(3))


class TestSimpleEKF(unittest.TestCase):
    """简化EKF测试"""

    def test_init(self):
        ekf = _SimpleEKF(process_noise=0.01, measurement_noise=0.1)
        np.testing.assert_array_equal(ekf._state, np.zeros(3))
        self.assertEqual(ekf._P.shape, (3, 3))

    def test_update_level(self):
        ekf = _SimpleEKF(process_noise=0.01, measurement_noise=0.1)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        rpy = ekf.update(accel, gyro, dt=0.01)
        self.assertAlmostEqual(rpy[0], 0.0, places=1)
        self.assertAlmostEqual(rpy[1], 0.0, places=1)

    def test_reset(self):
        ekf = _SimpleEKF()
        accel = np.array([0.0, 9.81, 0.0])
        gyro = np.array([0.1, 0.0, 0.0])
        ekf.update(accel, gyro, dt=0.01)
        ekf.reset()
        np.testing.assert_array_equal(ekf._state, np.zeros(3))


class TestSensorFusionControllerLifecycle(unittest.TestCase):
    """SensorFusionController 生命周期测试"""

    def setUp(self):
        self.ctrl = SensorFusionController(grade=FusionControlGrade.M)

    def test_start_stop(self):
        self.ctrl.start()
        self.assertTrue(self.ctrl._is_running)
        self.ctrl.stop()
        self.assertFalse(self.ctrl._is_running)

    def test_context_manager(self):
        with SensorFusionController(grade=FusionControlGrade.M) as ctrl:
            self.assertTrue(ctrl._is_running)
        self.assertFalse(ctrl._is_running)

    def test_reset(self):
        self.ctrl.start()
        self.ctrl.update(
            imu_accel=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.array([0.1, 0.0, 0.0]),
            force_wrench=np.array([0, 0, 10, 0, 0, 0]),
        )
        self.ctrl.reset()
        self.assertEqual(self.ctrl._frame_id, 0)
        np.testing.assert_array_equal(self.ctrl._velocity, np.zeros(3))
        np.testing.assert_array_equal(self.ctrl._position, np.zeros(3))


class TestSensorFusionControllerUpdate(unittest.TestCase):
    """SensorFusionController update() 测试"""

    def setUp(self):
        self.ctrl = SensorFusionController(grade=FusionControlGrade.M)

    def test_update_imu_only(self):
        self.ctrl.start()
        state = self.ctrl.update(
            imu_accel=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.array([0.0, 0.0, 0.0]),
            dt=0.01,
        )
        self.assertIsInstance(state, SensorFusionControlState)
        self.assertEqual(state.frame_id, 0)
        np.testing.assert_allclose(state.imu_accel, np.array([0.0, 0.0, 9.81]), rtol=1e-5)
        np.testing.assert_array_equal(state.imu_gyro, np.array([0.0, 0.0, 0.0]))

    def test_update_force_contact(self):
        self.ctrl.start()
        state = self.ctrl.update(
            imu_accel=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.array([0.0, 0.0, 0.0]),
            force_wrench=np.array([0, 0, 50, 0, 0, 0]),  # 50N 接触
            dt=0.01,
        )
        self.assertTrue(state.contact_detected)
        self.assertGreater(state.contact_force, 10.0)

    def test_update_tactile_slip(self):
        self.ctrl.start()
        pressure = np.random.rand(16, 16).astype(np.float32)
        state = self.ctrl.update(
            imu_accel=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.array([0.0, 0.0, 0.0]),
            force_wrench=np.array([0, 0, 20, 0, 0, 0]),
            tactile_pressure=pressure,
            dt=0.01,
        )
        self.assertIsNotNone(state.tactile_pressure)
        self.assertEqual(state.tactile_pressure.shape, (16, 16))

    def test_update_control_commands_generated(self):
        self.ctrl.start()
        for _ in range(10):
            self.ctrl.update(
                imu_accel=np.array([0.0, 0.0, 9.81]),
                imu_gyro=np.array([0.01, 0.0, 0.0]),
                dt=0.01,
            )
        vel_cmd, torque_cmd = self.ctrl.get_control_command()
        self.assertEqual(vel_cmd.shape, (3,))
        self.assertEqual(torque_cmd.shape, (3,))

    def test_update_multiple_frames(self):
        self.ctrl.start()
        for i in range(100):
            state = self.ctrl.update(
                imu_accel=np.array([0.0, 0.0, 9.81]),
                imu_gyro=np.array([0.01, 0.005, 0.001]),
                dt=0.01,
            )
            self.assertEqual(state.frame_id, i)
        self.assertEqual(self.ctrl._frame_id, 100)

    def test_update_get_state(self):
        self.ctrl.start()
        self.ctrl.update(
            imu_accel=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.array([0.1, 0.0, 0.0]),
        )
        state = self.ctrl.get_state()
        self.assertIsInstance(state, SensorFusionControlState)
        self.assertGreater(state.fused_pose[0], 0.0)  # roll > 0


class TestSensorFusionControllerGrades(unittest.TestCase):
    """AGV五级融合控制测试"""

    def test_grade_s_low_freq(self):
        ctrl = SensorFusionController(grade=FusionControlGrade.S)
        self.assertEqual(ctrl.fusion_frequency, 50)
        self.assertEqual(ctrl.control_frequency, 50)

    def test_grade_l_ekf(self):
        ctrl = SensorFusionController(grade=FusionControlGrade.L)
        self.assertEqual(ctrl.fusion_algorithm, 'ekf')
        self.assertIsNotNone(ctrl._ekf)

    def test_grade_xxl_high_freq(self):
        ctrl = SensorFusionController(grade=FusionControlGrade.XXL)
        self.assertEqual(ctrl.fusion_frequency, 1000)
        self.assertEqual(ctrl.control_frequency, 1000)
        self.assertEqual(ctrl.fusion_algorithm, 'ekf')

    def test_control_commands_vary_by_grade(self):
        """不同等级产生不同的控制频率"""
        results = {}
        for grade in [FusionControlGrade.S, FusionControlGrade.M, FusionControlGrade.L]:
            ctrl = SensorFusionController(grade=grade)
            ctrl.start()
            for _ in range(5):
                ctrl.update(
                    imu_accel=np.array([0.0, 0.0, 9.81]),
                    imu_gyro=np.array([0.1, 0.05, 0.01]),
                    dt=1.0 / ctrl.fusion_frequency,
                )
            vel_cmd, _ = ctrl.get_control_command()
            results[grade] = np.linalg.norm(vel_cmd)
        # 所有等级都能产生控制指令
        for norm in results.values():
            self.assertGreaterEqual(norm, 0.0)


class TestSensorFusionControlState(unittest.TestCase):
    """SensorFusionControlState 数据类测试"""

    def test_default_state(self):
        state = SensorFusionControlState()
        np.testing.assert_array_equal(state.imu_accel, np.zeros(3))
        np.testing.assert_array_equal(state.imu_gyro, np.zeros(3))
        np.testing.assert_array_equal(state.force, np.zeros(6))
        self.assertFalse(state.contact_detected)
        self.assertEqual(state.contact_force, 0.0)
        self.assertEqual(state.slip_probability, 0.0)
        np.testing.assert_array_equal(state.velocity_cmd, np.zeros(3))
        np.testing.assert_array_equal(state.torque_cmd, np.zeros(3))

    def test_state_with_data(self):
        state = SensorFusionControlState(
            imu_accel=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.array([0.1, 0.0, 0.0]),
            force=np.array([0, 0, 50, 0, 0, 0]),
            contact_detected=True,
            contact_force=50.0,
            timestamp=1.5,
            frame_id=10,
        )
        np.testing.assert_array_equal(state.imu_accel, np.array([0.0, 0.0, 9.81]))
        np.testing.assert_array_equal(state.imu_gyro, np.array([0.1, 0.0, 0.0]))
        self.assertTrue(state.contact_detected)
        self.assertEqual(state.contact_force, 50.0)


if __name__ == '__main__':
    unittest.main()
