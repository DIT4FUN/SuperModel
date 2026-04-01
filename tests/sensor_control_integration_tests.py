"""
传感器-控制集成测试
====================

测试传感器模块与控制模块的集成:
- TactileServoController
- ForceController
- AttitudeStabilizer / MotionEstimator
"""

import numpy as np
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact,
    TactileSensorType, VirtualTactileSensor
)
from sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType, VirtualForceSensor
)
from sensors.imu import (
    IMUSensor, IMUFrame, PoseEstimator, IMUSensorType, VirtualIMUSensor
)
from control.tactile_control import (
    TactileServoController, TactileServoParams, GraspQualityController,
    get_tactile_control_spec
)
from control.force_control import (
    ForceController, ForceControlParams, HybridForcePositionController,
    get_force_control_spec
)
from control.imu_control import (
    AttitudeStabilizer, IMUControlParams, MotionEstimator,
    get_imu_control_spec
)


class TestTactileServoController(unittest.TestCase):
    """触觉伺服控制器测试"""
    
    def test_tactile_servo_init(self):
        """测试初始化"""
        tactile = TactileArray(array_size=(8, 8), sensor_id="test_tactile")
        params = TactileServoParams.from_grade('M')
        controller = TactileServoController(tactile, params)
        
        self.assertEqual(controller.params.grade, 'M')
        self.assertIsNotNone(controller.params.Kp_position)
        self.assertEqual(controller.params.control_rate, 50)
    
    def test_tactile_servo_no_contact(self):
        """无接触时返回零控制"""
        tactile = TactileArray(array_size=(8, 8), sensor_id="test_tactile")
        tactile.open()
        controller = TactileServoController(tactile)
        
        signal = controller.compute_control_signal(target_force=5.0)
        np.testing.assert_array_almost_equal(signal, np.zeros(3))
    
    def test_tactile_servo_contact_detection(self):
        """接触检测"""
        tactile = TactileArray(array_size=(8, 8), sensor_id="test_tactile")
        tactile.open()
        controller = TactileServoController(tactile)
        
        # 虚拟接触
        with patch.object(tactile, 'detect_contacts') as mock_detect:
            mock_detect.return_value = [
                TactileContact(
                    center=(4, 4), area=9, peak_pressure=0.8,
                    mean_pressure=0.5, centroid=(4.0, 4.0),
                    contact_force=5.0, slip_probability=0.1
                )
            ]
            
            is_contact = controller.is_contact()
            self.assertTrue(is_contact)
    
    def test_tactile_servo_slip_reaction(self):
        """滑移反应"""
        tactile = TactileArray(array_size=(8, 8), sensor_id="test_tactile")
        tactile.open()
        controller = TactileServoController(tactile)
        controller.params.slip_threshold = 0.1
        
        # 模拟高滑移信号
        high_slip_frame = TactileFrame(
            pressure_map=np.random.rand(8, 8).astype(np.float32) * 0.3,
            timestamp=time.time(), frame_id=0, sensor_id="test"
        )
        
        with patch.object(tactile, 'get_slip_signal', return_value=np.ones((8, 8)) * 0.5):
            with patch.object(tactile, 'detect_contacts', return_value=[
                TactileContact(center=(4, 4), area=9, peak_pressure=0.8,
                               mean_pressure=0.5, centroid=(4.0, 4.0),
                               contact_force=5.0, slip_probability=0.5)
            ]):
                reaction = controller.detect_and_react_slip(high_slip_frame)
                self.assertGreater(np.linalg.norm(reaction), 0)
    
    def test_grasp_quality_monitor(self):
        """抓取质量监控"""
        tactile = TactileArray(array_size=(8, 8), sensor_id="test_tactile")
        tactile.open()
        controller = TactileServoController(tactile)
        
        # 初始状态
        status = controller.monitor_grasp_quality()
        self.assertEqual(status['current'], 0.0)
    
    def test_tactile_control_grade_spec(self):
        """AGV五级触觉控制规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = get_tactile_control_spec(grade)
            self.assertEqual(params.grade, grade)
            self.assertGreater(params.Kp_position, 0)
            self.assertGreater(params.control_rate, 0)


class TestForceController(unittest.TestCase):
    """力觉控制器测试"""
    
    def test_force_controller_init(self):
        """测试初始化"""
        force = ForceTorqueSensor(sensor_id="test_force")
        params = ForceControlParams.from_grade('M')
        controller = ForceController(force, params)
        
        self.assertEqual(controller.params.grade, 'M')
        self.assertEqual(controller._collision_count, 0)
    
    def test_admittance_control(self):
        """导纳控制"""
        force = ForceTorqueSensor(sensor_id="test_force")
        force.open()
        controller = ForceController(force)
        
        desired_force = np.array([0.0, 0.0, 10.0])
        adj = controller.compute_admittance(desired_force, dt=0.01)
        
        self.assertEqual(adj.shape, (3,))
        self.assertFalse(np.any(np.isnan(adj)))
    
    def test_collision_detection_no_collision(self):
        """无碰撞检测"""
        force = ForceTorqueSensor(sensor_id="test_force")
        force.open()
        controller = ForceController(force)
        controller.params.collision_threshold = 50.0
        
        # 正常力
        is_collision, mag = controller.detect_collision()
        self.assertFalse(is_collision)
    
    def test_collision_detection_with_collision(self):
        """碰撞检测"""
        force = ForceTorqueSensor(sensor_id="test_force")
        force.open()
        controller = ForceController(force)
        controller.params.collision_threshold = 5.0
        
        # 模拟连续碰撞力 (需要多次触发才能满足检测逻辑)
        for _ in range(5):
            wrench = Wrench(force=np.array([10.0, 0.0, 0.0]), torque=np.zeros(3),
                            timestamp=time.time(), frame_id=0, sensor_id="test")
            is_collision, mag = controller.detect_collision(wrench)
        
        self.assertTrue(is_collision)
        self.assertGreater(mag, 0)
    
    def test_hybrid_force_position_control(self):
        """力位混合控制"""
        force = ForceTorqueSensor(sensor_id="test_force")
        force.open()
        controller = HybridForcePositionController(force)
        
        target_force = np.array([5.0, 5.0, 0.0])
        target_pos = np.array([0.1, 0.2, 0.0])
        
        pos_out, force_out = controller.compute_control(target_force, target_pos, dt=0.01)
        
        self.assertEqual(pos_out.shape, (3,))
        self.assertEqual(force_out.shape, (3,))
    
    def test_force_control_grade_spec(self):
        """AGV五级力控规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = get_force_control_spec(grade)
            self.assertEqual(params.grade, grade)
            self.assertGreater(params.Kp_force, 0)


class TestAttitudeStabilizer(unittest.TestCase):
    """姿态稳定控制器测试"""
    
    def test_attitude_stabilizer_init(self):
        """测试初始化"""
        imu = IMUSensor(sensor_id="test_imu")
        params = IMUControlParams.from_grade('M')
        stabilizer = AttitudeStabilizer(imu, params)
        
        self.assertEqual(stabilizer.params.grade, 'M')
        np.testing.assert_array_almost_equal(stabilizer._target_euler, np.zeros(3))
    
    def test_set_target_attitude(self):
        """设置目标姿态"""
        imu = IMUSensor(sensor_id="test_imu")
        stabilizer = AttitudeStabilizer(imu)
        
        stabilizer.set_target_attitude(roll=0.1, pitch=0.2, yaw=0.3)
        
        np.testing.assert_array_almost_equal(
            stabilizer._target_euler,
            np.array([0.1, 0.2, 0.3])
        )
    
    def test_attitude_update(self):
        """姿态更新"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        stabilizer = AttitudeStabilizer(imu)
        
        torque = stabilizer.update(dt=0.01)
        
        self.assertEqual(torque.shape, (3,))
        self.assertFalse(np.any(np.isnan(torque)))
    
    def test_tilt_status(self):
        """倾角状态"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        stabilizer = AttitudeStabilizer(imu)
        
        status = stabilizer.get_tilt_status()
        
        self.assertIn('roll', status)
        self.assertIn('pitch', status)
        self.assertIn('yaw', status)
        self.assertIn('is_stable', status)
    
    def test_is_moving(self):
        """运动检测"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        stabilizer = AttitudeStabilizer(imu)
        
        # 静止
        is_moving = stabilizer.is_moving()
        self.assertFalse(is_moving)
    
    def test_imu_control_grade_spec(self):
        """AGV五级IMU控制规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = get_imu_control_spec(grade)
            self.assertEqual(params.grade, grade)
            self.assertGreater(params.Kp_attitude, 0)


class TestMotionEstimator(unittest.TestCase):
    """运动估计器测试"""
    
    def test_motion_estimator_init(self):
        """测试初始化"""
        imu = IMUSensor(sensor_id="test_imu")
        estimator = MotionEstimator(imu)
        
        np.testing.assert_array_almost_equal(estimator._velocity, np.zeros(3))
        np.testing.assert_array_almost_equal(estimator._position, np.zeros(3))
    
    def test_motion_estimator_reset(self):
        """重置"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        estimator = MotionEstimator(imu)
        
        estimator.update(dt=0.01)
        estimator.reset()
        
        np.testing.assert_array_almost_equal(estimator._velocity, np.zeros(3))
        np.testing.assert_array_almost_equal(estimator._position, np.zeros(3))
    
    def test_motion_estimator_update(self):
        """更新"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        estimator = MotionEstimator(imu)
        
        vel, pos = estimator.update(dt=0.01)
        
        self.assertEqual(vel.shape, (3,))
        self.assertEqual(pos.shape, (3,))
    
    def test_trajectory_recording(self):
        """轨迹记录"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        estimator = MotionEstimator(imu)
        
        for _ in range(10):
            estimator.update(dt=0.01)
        
        times, positions = estimator.get_trajectory()
        self.assertEqual(len(times), 10)
        self.assertEqual(positions.shape[0], 10)
        self.assertEqual(positions.shape[1], 3)
    
    def test_displacement_estimation(self):
        """位移估计"""
        imu = IMUSensor(sensor_id="test_imu")
        imu.open()
        estimator = MotionEstimator(imu)
        
        displacement = estimator.estimate_displacement(duration=0.1)
        
        self.assertGreaterEqual(displacement, 0)


if __name__ == '__main__':
    unittest.main()


class TestAGVSensorFusionControlIntegration(unittest.TestCase):
    """AGV传感器-融合-控制联合集成测试"""

    def test_agv_grade_s_motor_specs(self):
        """S级AGV电机规格验证"""
        from simulation.agv_scenarios import AGVPhysicsConfig
        cfg = AGVPhysicsConfig.from_grade('S')
        self.assertEqual(cfg.grade, 'S')
        self.assertLess(cfg.mass, 50.0)
        self.assertLess(cfg.max_linear_speed, 2.0)

    def test_agv_grade_xxl_motor_specs(self):
        """XXL级AGV电机规格验证"""
        from simulation.agv_scenarios import AGVPhysicsConfig
        cfg = AGVPhysicsConfig.from_grade('XXL')
        self.assertEqual(cfg.grade, 'XXL')
        self.assertGreater(cfg.mass, 400.0)
        self.assertGreater(cfg.max_linear_speed, 5.0)

    def test_tactile_imu_coordinated_control(self):
        """触觉-IMU协调控制测试"""
        from control.tactile_control import TactileServoController, TactileServoParams
        from control.imu_control import AttitudeStabilizer, IMUControlParams
        
        tactile = TactileArray(array_size=(8, 8), sensor_id="coord_tactile")
        imu = IMUSensor(sensor_id="coord_imu")
        
        tactile.open()
        imu.open()
        
        tctrl = TactileServoController(tactile, TactileServoParams.from_grade('M'))
        actrl = AttitudeStabilizer(imu, IMUControlParams.from_grade('M'))
        
        # IMU保持姿态
        actrl.set_target_attitude(0.0, 0.0, 0.0)
        for _ in range(5):
            imu_frame = imu.capture()
            ctrl_out = actrl.update(imu_frame, dt=0.02)
        
        self.assertIsNotNone(ctrl_out)
        self.assertEqual(ctrl_out.shape, (3,))
        
        tactile.close()
        imu.close()

    def test_force_imu_coordinated_control(self):
        """力觉-IMU协调控制测试"""
        from control.force_control import ForceController, ForceControlParams
        from control.imu_control import AttitudeStabilizer, IMUControlParams
        
        force = ForceTorqueSensor(sensor_id="coord_force")
        imu = IMUSensor(sensor_id="coord_imu2")
        
        force.open()
        imu.open()
        
        fctrl = ForceController(force, ForceControlParams.from_grade('M'))
        actrl = AttitudeStabilizer(imu, IMUControlParams.from_grade('M'))
        
        wrench = Wrench(force=[0.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0])
        # ForceController uses compute_admittance
        ctrl = fctrl.compute_admittance(desired_force=np.zeros(3), current_wrench=wrench, dt=0.01)
        
        self.assertIsNotNone(ctrl)
        
        force.close()
        imu.close()

    def test_all_grade_force_control_specs(self):
        """所有等级力控规格验证"""
        from control.force_control import ForceControlParams, get_force_control_spec
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = ForceControlParams.from_grade(grade)
            self.assertEqual(params.grade, grade)
            
            spec = get_force_control_spec(grade)
            self.assertEqual(spec.grade, grade)
            self.assertIsNotNone(spec.Kp_force)
            self.assertIsNotNone(spec.control_rate)

    def test_all_grade_imu_control_specs(self):
        """所有等级IMU控制规格验证"""
        from control.imu_control import IMUControlParams, get_imu_control_spec
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = IMUControlParams.from_grade(grade)
            self.assertEqual(params.grade, grade)
            
            spec = get_imu_control_spec(grade)
            self.assertEqual(spec.grade, grade)
            self.assertIsNotNone(spec.Kp_attitude)
            self.assertIsNotNone(spec.control_rate)

    def test_all_grade_tactile_control_specs(self):
        """所有等级触觉控制规格验证"""
        from control.tactile_control import TactileServoParams, get_tactile_control_spec
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = TactileServoParams.from_grade(grade)
            self.assertEqual(params.grade, grade)
            
            spec = get_tactile_control_spec(grade)
            self.assertEqual(spec.grade, grade)
            self.assertIsNotNone(spec.Kp_position)
            self.assertIsNotNone(spec.control_rate)
