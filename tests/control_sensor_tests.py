"""
触觉/力觉/IMU控制模块单元测试
测试 TactileServoController, ForceController, AttitudeStabilizer, CalibrationManager
以及 AGV 五级规格对应关系
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from sensors.tactile import TactileArray, TactileSensorType
from sensors.force import ForceTorqueSensor, ForceSensorType
from sensors.imu import IMUSensor, IMUSensorType

from control.tactile_control import (
    TactileServoController, TactileServoParams, GraspQualityController,
    get_tactile_control_spec, AGV_TACTILE_CONTROL_GRADES
)
from control.force_control import (
    ForceController, ForceControlParams, HybridForcePositionController,
    get_force_control_spec, AGV_FORCE_CONTROL_GRADES
)
from control.imu_control import (
    AttitudeStabilizer, IMUControlParams, MotionEstimator,
    get_imu_control_spec, AGV_IMU_CONTROL_GRADES
)
from control.calibration_manager import (
    CalibrationManager,
    CalibrationStatus,
    get_calibration_spec, get_all_grade_spec_table
)


class TestTactileServoParams(unittest.TestCase):
    """测试触觉伺服控制参数"""

    def test_from_grade_all_levels(self):
        """测试所有AGV等级的参数"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = TactileServoParams.from_grade(grade)
            self.assertEqual(params.grade, grade)
            self.assertGreater(params.Kp_position, 0)
            self.assertGreater(params.control_rate, 0)

    def test_grade_scaling(self):
        """测试等级越高控制率越高"""
        rates = [TactileServoParams.from_grade(g).control_rate for g in ['S', 'M', 'L', 'XL', 'XXL']]
        self.assertTrue(all(rates[i] <= rates[i+1] for i in range(len(rates)-1)))


class TestTactileServoController(unittest.TestCase):
    """测试触觉伺服控制器"""

    def setUp(self):
        self.tactile = TactileArray((4, 4), TactileSensorType.RESISTIVE, "test_tactile_ctrl")
        self.params = TactileServoParams(grade='M')
        self.controller = TactileServoController(self.tactile, self.params)

    def test_creation(self):
        """测试创建"""
        self.assertIsNotNone(self.controller.tactile)
        self.assertIsNotNone(self.controller.params)
        self.assertFalse(self.controller._is_grasping)

    def test_compute_control_no_contact(self):
        """测试无接触时控制信号为零"""
        self.tactile.open()
        signal = self.controller.compute_control_signal(target_force=5.0)
        np.testing.assert_array_equal(signal, np.zeros(3))
        self.tactile.close()

    def test_detect_react_slip_no_slip(self):
        """测试无滑移时反应为零"""
        self.tactile.open()
        frame = self.tactile.capture()
        reaction = self.controller.detect_and_react_slip(frame)
        self.assertEqual(reaction.shape, (3,))
        np.testing.assert_array_equal(reaction, np.zeros(3))
        self.tactile.close()

    def test_monitor_grasp_quality(self):
        """测试抓取质量监控"""
        quality = self.controller.monitor_grasp_quality()
        self.assertIn('current', quality)
        self.assertIn('average', quality)
        self.assertIn('stable', quality)

    def test_reset(self):
        """测试重置"""
        self.controller._is_grasping = True
        self.controller._target_position = np.array([1, 2, 3])
        self.controller.reset()
        self.assertFalse(self.controller._is_grasping)
        np.testing.assert_array_equal(self.controller._target_position, np.zeros(3))

    def test_get_tactile_control_spec(self):
        """测试规格获取函数"""
        for g in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_control_spec(g)
            self.assertEqual(spec.grade, g)


class TestForceControlParams(unittest.TestCase):
    """测试力控参数"""

    def test_from_grade_all_levels(self):
        """测试所有AGV等级"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = ForceControlParams.from_grade(grade)
            self.assertEqual(params.grade, grade)
            self.assertGreater(params.K_stiffness, 0)
            self.assertGreater(params.control_rate, 0)

    def test_grade_scaling(self):
        """测试等级越高参数越大"""
        for param in ['M_mass', 'D_damping', 'K_stiffness', 'Kp_force']:
            vals = [ForceControlParams.from_grade(g).__dict__[param] for g in ['S', 'M', 'L', 'XL', 'XXL']]
            self.assertTrue(all(vals[i] <= vals[i+1] for i in range(len(vals)-1)))


class TestForceController(unittest.TestCase):
    """测试力觉控制器"""

    def setUp(self):
        self.force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "test_force_ctrl")
        self.params = ForceControlParams(grade='M')
        self.controller = ForceController(self.force, self.params)

    def test_creation(self):
        """测试创建"""
        self.assertIsNotNone(self.controller.force_sensor)
        self.assertFalse(self.controller._in_collision)
        self.assertEqual(self.controller._collision_count, 0)

    def test_admittance_no_collision(self):
        """测试无碰撞时导纳控制"""
        self.force.open()
        wrench = self.force.capture()
        desired = np.array([0.0, 0.0, 5.0])
        adj = self.controller.compute_admittance(desired, wrench, dt=0.01)
        self.assertEqual(adj.shape, (3,))
        self.assertFalse(np.any(np.isnan(adj)))
        self.force.close()

    def test_detect_collision_false(self):
        """测试无碰撞检测"""
        self.force.open()
        wrench = self.force.capture()
        is_collision, magnitude = self.controller.detect_collision(wrench, threshold=1000.0)
        self.assertFalse(is_collision)
        self.force.close()

    def test_collision_response(self):
        """测试碰撞响应"""
        direction = np.array([1.0, 0.0, 0.0])
        response = self.controller.compute_collision_response(direction)
        self.assertEqual(response.shape, (3,))

    def test_reset(self):
        """测试重置"""
        self.controller._force_error_integral = np.array([1, 2, 3])
        self.controller._in_collision = True
        self.controller.reset()
        np.testing.assert_array_equal(self.controller._force_error_integral, np.zeros(3))
        self.assertFalse(self.controller._in_collision)

    def test_get_force_control_spec(self):
        """测试规格获取"""
        for g in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_control_spec(g)
            self.assertEqual(spec.grade, g)


class TestHybridForcePositionController(unittest.TestCase):
    """测试力位混合控制器"""

    def setUp(self):
        self.force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "test_hybrid")
        self.params = ForceControlParams(grade='M')
        self.hybrid = HybridForcePositionController(self.force, self.params)

    def test_creation(self):
        """测试创建"""
        np.testing.assert_array_equal(self.hybrid.force_control_axes, np.array([True, True, False]))

    def test_set_force_axes(self):
        """测试设置力控轴"""
        new_axes = np.array([False, True, True])
        self.hybrid.set_force_axes(new_axes)
        np.testing.assert_array_equal(self.hybrid.force_control_axes, new_axes)

    def test_compute_control(self):
        """测试混合控制计算"""
        self.force.open()
        target_force = np.array([5.0, 5.0, 0.0])
        target_pos = np.array([0.1, 0.1, 0.05])
        wrench = self.force.capture()
        pos_out, force_out = self.hybrid.compute_control(target_force, target_pos, wrench, dt=0.01)
        self.assertEqual(pos_out.shape, (3,))
        self.assertEqual(force_out.shape, (3,))
        self.force.close()

    def test_reset(self):
        """测试重置"""
        self.hybrid._position = np.array([1, 2, 3])
        self.hybrid.reset()
        np.testing.assert_array_equal(self.hybrid._position, np.zeros(3))


class TestIMUControlParams(unittest.TestCase):
    """测试IMU控制参数"""

    def test_from_grade_all_levels(self):
        """测试所有AGV等级"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = IMUControlParams.from_grade(grade)
            self.assertEqual(params.grade, grade)
            self.assertGreater(params.Kp_attitude, 0)

    def test_grade_scaling(self):
        """测试等级越高控制率越高"""
        rates = [IMUControlParams.from_grade(g).control_rate for g in ['S', 'M', 'L', 'XL', 'XXL']]
        self.assertTrue(all(rates[i] <= rates[i+1] for i in range(len(rates)-1)))


class TestAttitudeStabilizer(unittest.TestCase):
    """测试姿态稳定控制器"""

    def setUp(self):
        self.imu = IMUSensor(IMUSensorType.VIRTUAL)
        self.params = IMUControlParams(grade='M')
        self.stabilizer = AttitudeStabilizer(self.imu, self.params)

    def test_creation(self):
        """测试创建"""
        self.assertIsNotNone(self.stabilizer.imu)
        np.testing.assert_array_equal(self.stabilizer._target_euler, np.zeros(3))

    def test_set_target_attitude(self):
        """测试设置目标姿态"""
        self.stabilizer.set_target_attitude(roll=0.1, pitch=0.2, yaw=0.3)
        np.testing.assert_array_almost_equal(self.stabilizer._target_euler, [0.1, 0.2, 0.3])

    def test_update(self):
        """测试更新"""
        self.imu.open()
        frame = self.imu.capture()
        torque = self.stabilizer.update(frame, dt=0.01)
        self.assertEqual(torque.shape, (3,))
        self.imu.close()

    def test_tilt_status(self):
        """测试倾角状态"""
        status = self.stabilizer.get_tilt_status()
        self.assertIn('roll', status)
        self.assertIn('pitch', status)
        self.assertIn('tilt_magnitude', status)
        self.assertIn('is_stable', status)

    def test_is_moving(self):
        """测试运动检测"""
        self.imu.open()
        frame = self.imu.capture()
        is_moving = self.stabilizer.is_moving(frame)
        self.assertIsInstance(is_moving, bool)
        self.imu.close()


class TestMotionEstimator(unittest.TestCase):
    """测试运动估计器"""

    def setUp(self):
        self.imu = IMUSensor(IMUSensorType.VIRTUAL)
        self.estimator = MotionEstimator(self.imu)

    def test_creation(self):
        """测试创建"""
        np.testing.assert_array_equal(self.estimator._velocity, np.zeros(3))
        np.testing.assert_array_equal(self.estimator._position, np.zeros(3))

    def test_update(self):
        """测试更新"""
        self.imu.open()
        frame = self.imu.capture()
        vel, pos = self.estimator.update(frame, dt=0.01)
        self.assertEqual(vel.shape, (3,))
        self.assertEqual(pos.shape, (3,))
        self.imu.close()

    def test_displacement_estimation(self):
        """测试位移估计"""
        displacement = self.estimator.estimate_displacement(duration=1.0)
        self.assertGreaterEqual(displacement, 0.0)


class TestCalibrationManager(unittest.TestCase):
    """测试标定管理器"""

    def test_calibration_status_enum(self):
        """测试标定状态枚举"""
        self.assertEqual(CalibrationStatus.IDLE.value, "idle")
        self.assertEqual(CalibrationStatus.COLLECTING.value, "collecting")
        self.assertEqual(CalibrationStatus.PROCESSING.value, "processing")
        self.assertEqual(CalibrationStatus.COMPLETED.value, "completed")
        self.assertEqual(CalibrationStatus.FAILED.value, "failed")

    def test_calibration_spec_all_grades(self):
        """测试所有等级的标定规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_calibration_spec(grade)
            self.assertEqual(spec['grade'], grade)
            self.assertIn('imu_poses', spec)
            self.assertIn('force_samples', spec)
            self.assertIn('tactile_samples', spec)

    def test_all_grade_spec_table(self):
        """测试等级规格表"""
        table = get_all_grade_spec_table()
        self.assertEqual(len(table), 5)
        grades = [s['grade'] for s in table]
        self.assertEqual(grades, ['S', 'M', 'L', 'XL', 'XXL'])

    def test_calibration_manager_setup(self):
        """测试标定管理器注册传感器"""
        mgr = CalibrationManager()
        imu = IMUSensor(IMUSensorType.VIRTUAL)
        force = ForceTorqueSensor(ForceSensorType.SIX_AXIS)
        tactile = TactileArray((4, 4), TactileSensorType.RESISTIVE)

        mgr.setup_imu(imu)
        mgr.setup_force(force)
        mgr.setup_tactile(tactile)

        status = mgr.get_status()
        # Status should be IDLE before calibration
        self.assertEqual(status.value, 'idle')

    def test_calibration_manager_get_progress(self):
        """测试进度查询"""
        mgr = CalibrationManager()
        imu = IMUSensor(IMUSensorType.VIRTUAL)
        mgr.setup_imu(imu)
        # Before calibration: progress dict exists and is valid
        progress = mgr.get_progress()
        self.assertIsInstance(progress, dict)
        # After IMU calibration: imu should report 100% progress
        mgr.calibrate_all()
        progress = mgr.get_progress()
        self.assertEqual(progress['imu'], 1.0)

    def test_calibration_grade_spec_differences(self):
        """测试不同等级标定规格差异"""
        spec_s = get_calibration_spec('S')
        spec_xxl = get_calibration_spec('XXL')
        # XXL级应有更多样本
        self.assertGreater(spec_xxl['imu_samples'], spec_s['imu_samples'])
        self.assertGreater(spec_xxl['force_samples'], spec_s['force_samples'])


class TestGraspQualityController(unittest.TestCase):
    """测试抓取质量控制器"""

    def setUp(self):
        self.tactile = TactileArray((4, 4), TactileSensorType.RESISTIVE)
        self.grasp_ctrl = GraspQualityController(self.tactile, target_quality=0.7)

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.grasp_ctrl.target_quality, 0.7)
        self.assertEqual(len(self.grasp_ctrl._quality_history), 0)


if __name__ == '__main__':
    unittest.main()
