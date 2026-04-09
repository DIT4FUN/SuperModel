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
    TactileSensorType, PressureProcessor,
    VirtualTactileSensor, get_tactile_spec, AGV_TACTILE_GRADES
)
from sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor, WrenchProcessor,
    get_force_spec, AGV_FORCE_GRADES
)
from sensors.imu import (
    IMUSensor, IMUFrame, PoseEstimator, IMUSensorType,
    VirtualIMUSensor, get_imu_spec, AGV_IMU_GRADES
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


class TestTactileForceIMUFusion(unittest.TestCase):
    """触觉+力觉+IMU三传感器融合集成测试"""

    def setUp(self):
        self.tactile = TactileArray((8, 8), sensor_id="tactile_fusion_test")
        self.force = ForceTorqueSensor(sensor_id="force_fusion_test")
        self.imu = IMUSensor(sensor_id="imu_fusion_test")
        self.pose_est = PoseEstimator(algorithm="madgwick", sample_rate=100)
        
    def test_triple_sensor_capture(self):
        """测试三传感器同步采集"""
        self.tactile.open()
        self.force.open()
        self.imu.open()
        
        for i in range(10):
            tf = self.tactile.capture()
            fw = self.force.capture()
            im = self.imu.capture()
            
            self.assertEqual(tf.pressure_map.shape, (8, 8))
            self.assertEqual(fw.force.shape, (3,))
            self.assertEqual(im.accel.shape, (3,))
            self.assertGreater(tf.pressure_map.max(), 0)
            self.assertGreater(fw.force[2], -100)  # gravity effect
            self.assertGreater(im.accel_magnitude, 0)
        
        self.tactile.close()
        self.force.close()
        self.imu.close()

    def test_triple_sensor_with_pose_estimator(self):
        """测试三传感器+姿态估计融合"""
        self.tactile.open()
        self.force.open()
        self.imu.open()
        
        for i in range(20):
            im = self.imu.capture()
            pose = self.pose_est.update(im.accel, im.gyro, im.mag, dt=0.01)
            
            self.assertEqual(len(pose.orientation), 4)
            euler = pose.to_euler()
            self.assertEqual(len(euler), 3)
        
        self.tactile.close()
        self.force.close()
        self.imu.close()

    def test_contact_with_imu_awareness(self):
        """测试接触检测时考虑IMU姿态"""
        self.tactile.open()
        self.force.open()
        self.imu.open()
        
        # 模拟不同姿态下的接触
        self.pose_est.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # level
        im = self.imu.capture()
        self.pose_est.update(im.accel, im.gyro, dt=0.01)
        
        tf = self.tactile.capture()
        fw = self.force.capture()
        contacts = self.tactile.detect_contacts(tf)
        contact_state = self.force.detect_contact(fw)
        
        self.assertIsInstance(contacts, list)
        self.assertTrue(contact_state.is_contact in [True, False])
        self.assertIsInstance(float(contact_state.contact_force), float)
        
        self.tactile.close()
        self.force.close()
        self.imu.close()

    def test_agv_grade_specs(self):
        """测试AGV五级规格在所有传感器上的应用"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        
        for grade in grades:
            ts = get_tactile_spec(grade)
            fs = get_force_spec(grade)
            is_ = get_imu_spec(grade)
            
            self.assertIn('array', ts)
            self.assertIn('axes', fs)
            self.assertIn('accel_range', is_)
            self.assertIn('type', is_)
        
        # 确保高级规格 >= 低级规格
        tactile_res_s = get_tactile_spec('S')['res']
        tactile_res_xxl = get_tactile_spec('XXL')['res']
        self.assertLessEqual(tactile_res_s, tactile_res_xxl)

    def test_virtual_sensor_triple(self):
        """测试虚拟三传感器融合"""
        vt = VirtualTactileSensor((8, 8), sensor_id="vt_fusion")
        vf = VirtualForceSensor(sensor_id="vf_fusion")
        vi = VirtualIMUSensor(sensor_id="vi_fusion")
        
        vt.open()
        vf.open()
        vi.open()
        
        # 模拟接触
        tf = vt.simulate_contact((0.5, 0.5), contact_radius=0.3, contact_force=15.0)
        wf = vf.simulate_contact((5.0, 0.0, -10.0))
        imf = vi.simulate_static((0.0, 0.0, 0.0))
        
        self.assertEqual(tf.pressure_map.shape, (8, 8))
        self.assertEqual(wf.force.shape, (3,))
        self.assertEqual(imf.accel.shape, (3,))
        
        # 抓取质量评估 (用TactileArray而非VirtualTactileSensor)
        ta = TactileArray((8, 8), sensor_id="tactile_grip_test")
        ta.open()
        ta_frame = ta.capture()
        gq = ta.estimate_grip_quality(ta_frame)
        self.assertIn('overall', gq)
        
        vt.close()
        vf.close()
        vi.close()
        ta.close()

    def test_force_payload_estimation(self):
        """测试力传感器负载估计"""
        self.force.open()
        
        for _ in range(50):
            self.force.capture()
        
        payload = self.force.estimate_payload()
        self.assertGreaterEqual(payload, 0)
        self.assertLess(payload, 100)  # reasonable range
        
        self.force.close()

    def test_imu_self_test(self):
        """测试IMU自检"""
        self.imu.open()
        result = self.imu.self_test()
        self.assertIsInstance(result, bool)
        self.imu.close()

    def test_imu_gyro_bias_calibration(self):
        """测试IMU陀螺仪偏置校准"""
        self.imu.open()
        self.imu.calibrate_gyro_bias(num_samples=50, duration_sec=1.0)
        bias = self.imu.calibration.gyro_bias
        self.assertEqual(bias.shape, (3,))
        self.imu.close()

    def test_tactile_slip_detection(self):
        """测试触觉滑移检测"""
        vt = VirtualTactileSensor((8, 8))
        vt.open()
        
        # 模拟滑移
        frames = vt.simulate_sliding((0.1, 0.05), speed=0.02, duration_frames=20)
        self.assertEqual(len(frames), 20)
        
        # 滑移检测
        slip_result = vt.simulate_slip_detection(
            normal_force=10.0, 
            friction_coeff=0.3,
            velocity=(0.02, 0.01)
        )
        self.assertIn('slip_state', slip_result)
        self.assertIn('slip_probability', slip_result)
        
        vt.close()

    def test_force_collision_simulation(self):
        """测试力碰撞仿真"""
        vf = VirtualForceSensor()
        vf.open()
        
        frames = vf.simulate_collision(
            direction=(1.0, 0.0, 0.0),
            peak_force=50.0,
            duration_ms=100.0,
            decay='exponential'
        )
        
        self.assertGreater(len(frames), 0)
        self.assertLess(len(frames), 20)  # 100ms / 10ms = 10 frames
        
        # 峰值力应该接近预期
        max_force = max(f.magnitude for f in frames)
        self.assertGreater(max_force, 30.0)
        
        vf.close()

    def test_imu_trajectory_simulation(self):
        """测试IMU轨迹仿真"""
        vi = VirtualIMUSensor()
        vi.open()
        
        frames = vi.simulate_trajectory('circle', duration_s=1.0, dt=0.01)
        self.assertGreater(len(frames), 50)
        
        # 角速度应该不为零
        last_frame = frames[-1]
        self.assertIsNotNone(last_frame.gyro)
        
        vi.close()

    def test_pose_estimator_all_algorithms(self):
        """测试所有姿态估计算法"""
        for algo in ['madgwick', 'complementary']:
            est = PoseEstimator(algorithm=algo, sample_rate=100)
            accel = np.array([0.0, 0.0, 9.81])
            gyro = np.array([0.01, 0.02, 0.1])
            
            for _ in range(50):
                pose = est.update(accel, gyro, dt=0.01)
            
            euler = pose.to_euler()
            self.assertEqual(len(euler), 3)
            self.assertEqual(len(pose.orientation), 4)

    def test_wrench_processor(self):
        """测试力旋量处理器"""
        proc = WrenchProcessor(filter_alpha=0.3)
        
        for i in range(10):
            wrench_vec = np.array([5.0, 0.0, -10.0, 0.1, 0.2, 0.0])
            filtered = proc.filter(wrench_vec)
            self.assertEqual(filtered.shape, (6,))

    def test_pressure_processor(self):
        """测试压力处理器"""
        proc = PressureProcessor(filter_window=3)
        
        pressure = np.random.rand(8, 8).astype(np.float32)
        filtered = proc.filter(pressure)
        self.assertEqual(filtered.shape, (8, 8))
        
        centroid = proc.compute_centroid(pressure)
        self.assertEqual(len(centroid), 2)
        
        compensated = proc.compensate_baseline(pressure)
        self.assertEqual(compensated.shape, (8, 8))


if __name__ == '__main__':
    unittest.main()
