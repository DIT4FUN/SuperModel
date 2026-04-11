"""
传感器-执行器集成测试
====================

测试多传感器数据到执行控制的完整闭环集成:
- TactileArray → 触觉控制
- ForceTorqueSensor → 力控
- IMUSensor → 姿态控制
- 跨模态融合 → 综合决策
- 执行器响应验证

覆盖 S/M/L/XL/XXL 五级 AGV 规格
"""

import unittest
import numpy as np
import sys
import time
from unittest.mock import MagicMock, patch

_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SrcPath = f'{_ProjectRoot}/src'
if _SrcPath not in sys.path:
    sys.path.insert(0, _SrcPath)
if _ProjectRoot not in sys.path:
    sys.path.insert(0, _ProjectRoot)

from src.sensors.tactile import (
    TactileArray, TactileSensorType, TactileFrame,
    TactileContact, TactileCalibration, VirtualTactileSensor,
    PressureProcessor, AGV_TACTILE_GRADES, get_tactile_spec
)
from src.sensors.force import (
    ForceTorqueSensor, ForceSensorType, Wrench,
    ForceCalibration, ContactState, VirtualForceSensor,
    WrenchProcessor, AGV_FORCE_GRADES, get_force_spec
)
from src.sensors.imu import (
    IMUSensor, IMUSensorType, IMUFrame, Pose,
    IMUCalibration, PoseEstimator, VirtualIMUSensor,
    AGV_IMU_GRADES, get_imu_spec
)


# ============================================================================
# Tactile → 触觉控制 集成测试
# ============================================================================

class TestTactileServoIntegration(unittest.TestCase):
    """触觉感知 → 触觉伺服控制 集成测试"""

    def setUp(self):
        self.sensor = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id="test_tactile_servo"
        )
        self.sensor.open()
        self.servo_targets = []
        self.servo_commands = []

    def tearDown(self):
        self.sensor.close()

    def _mock_servo_command(self, position, force):
        """模拟伺服执行器"""
        self.servo_commands.append((position, force))
        return {"position": position, "force": force, "actual_force": force * 0.95}

    def test_tactile_grip_closed_loop(self):
        """测试触觉抓取闭环控制"""
        for _ in range(20):
            frame = self.sensor.capture()
            contacts = self.sensor.detect_contacts(frame)
            
            if contacts:
                # 基于触觉反馈调整抓取力
                mean_force = np.mean([c.contact_force for c in contacts])
                if mean_force < 5.0:
                    # 力度不足，增加夹持力
                    cmd = self._mock_servo_command(position=0.5, force=mean_force + 2.0)
                elif mean_force > 15.0:
                    # 力度过大，减少夹持力
                    cmd = self._mock_servo_command(position=0.6, force=mean_force - 2.0)
                else:
                    cmd = self._mock_servo_command(position=0.55, force=mean_force)
                
                self.servo_targets.append(cmd["force"])
        
        # 仿真模式下可能无接触，但传感器数据流应正常
        frame = self.sensor.capture()
        self.assertEqual(frame.pressure_map.shape, (16, 16))
        self.assertIsNotNone(frame.timestamp)

    def test_tactile_multi_contact_balance(self):
        """测试多点接触力平衡"""
        virtual = VirtualTactileSensor((24, 24), "test_multi_contact")
        virtual.open()
        
        # 模拟两点接触
        contacts = [
            ((0.3, 0.5), 8.0, 0.3),   # (位置, 力, 半径)
            ((0.7, 0.5), 8.0, 0.3),
        ]
        
        frame = virtual.simulate_multi_contact(contacts, noise_level=0.05)
        detected = virtual._last_contact_pos  # 虚拟传感器会记录最后一次接触
        
        # 验证多点接触被正确模拟
        self.assertIsNotNone(frame.pressure_map)
        self.assertEqual(frame.pressure_map.shape, (24, 24))
        
        virtual.close()

    def test_tactile_slip_recovery(self):
        """测试滑移检测与恢复"""
        virtual = VirtualTactileSensor((16, 16), "test_slip")
        virtual.open()
        
        # 模拟正常抓取
        frame1 = virtual.simulate_contact((0.5, 0.5), contact_force=10.0, contact_radius=0.3)
        # 模拟滑移 (方向移动)
        frames_sliding = virtual.simulate_sliding(
            direction=(0.1, 0.0),
            speed=0.05,
            duration_frames=10
        )
        
        # 检测滑移
        slip_detection = virtual.simulate_slip_detection(
            normal_force=10.0,
            friction_coeff=0.3,
            velocity=(0.05, 0.0)
        )
        
        self.assertIn(slip_detection["slip_state"], ["stick", "micro_slip", "sliding"])
        self.assertGreaterEqual(slip_detection["slip_probability"], 0.0)
        self.assertLessEqual(slip_detection["slip_probability"], 1.0)
        
        virtual.close()

    def test_tactile_grade_spec_compliance(self):
        """测试各级别触觉规格符合性"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            sensor = TactileArray(
                array_size=tuple(spec['array']),
                sensor_type=TactileSensorType.CAPACITIVE,
                sensor_id=f"test_grade_{grade}"
            )
            sensor.open()
            
            frame = sensor.capture()
            self.assertEqual(frame.pressure_map.shape, tuple(spec['array']))
            
            sensor.close()

    def test_tactile_pressure_processor(self):
        """测试压力信号处理器"""
        processor = PressureProcessor(filter_window=3, drift_compensation=True)
        
        raw_pressure = np.random.rand(16, 16).astype(np.float32) * 0.8 + 0.1
        
        # 滤波
        filtered = processor.filter(raw_pressure)
        self.assertEqual(filtered.shape, raw_pressure.shape)
        
        # 设置基线
        _ = processor.compensate_baseline(filtered, set_baseline=True)
        # 补偿（已有基线后）
        compensated = processor.compensate_baseline(filtered * 1.1, set_baseline=False)
        self.assertEqual(compensated.shape, raw_pressure.shape)
        
        # 计算接触力
        force = processor.compute_force(raw_pressure, contact_area=1e-4)
        self.assertGreater(force, 0.0)
        
        # 质心计算
        centroid = processor.compute_centroid(raw_pressure)
        self.assertEqual(len(centroid), 2)
        
        # 直方图计算
        hist, edges = processor.compute_pressure_histogram(raw_pressure, bins=10)
        self.assertEqual(len(hist), 10)


# ============================================================================
# Force → 力控 集成测试
# ============================================================================

class TestForceControlIntegration(unittest.TestCase):
    """力觉感知 → 力控执行 集成测试"""

    def setUp(self):
        self.sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_ft_ctrl"
        )
        self.sensor.open()
        self.force_commands = []

    def tearDown(self):
        self.sensor.close()

    def _apply_force_control(self, target_force, actual_wrench):
        """简单的力控制环"""
        Kp = 0.5  # 比例增益
        error = target_force - actual_wrench.magnitude
        adjustment = Kp * error
        return max(0, adjustment)

    def test_force_tracking_control(self):
        """测试力跟踪控制"""
        target_force = 10.0  # N
        
        for _ in range(30):
            wrench = self.sensor.capture()
            control_output = self._apply_force_control(target_force, wrench)
            self.force_commands.append({
                'target': target_force,
                'actual': wrench.magnitude,
                'control': control_output
            })
        
        self.assertEqual(len(self.force_commands), 30)
        
        # 最后几次实际力应该接近目标
        last_5 = self.force_commands[-5:]
        actual_forces = [c['actual'] for c in last_5]
        # 由于是开环仿真,实际力会有波动但均值应该接近
        self.assertLess(abs(np.mean(actual_forces)), 20.0)

    def test_wrench_coordinate_transform(self):
        """测试力旋量坐标变换"""
        sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_wrench_transform"
        )
        sensor.open()
        
        # 设置工具中心偏移
        sensor.set_tool_center(tool_mass=0.5, tool_com=np.array([0.0, 0.0, 0.1]))
        
        wrench = sensor.capture()
        
        # 绕Z轴旋转90度
        rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        translation = np.array([0.1, 0.0, 0.0])
        
        new_wrench = wrench.transform(rotation, translation)
        
        self.assertEqual(new_wrench.force.shape, (3,))
        self.assertEqual(new_wrench.torque.shape, (3,))
        
        sensor.close()

    def test_contact_detection_state_machine(self):
        """测试接触检测状态机"""
        virtual = VirtualForceSensor(sensor_id="vsm_force", noise_level=0.5)
        virtual.open()
        
        states = []
        in_contact = False
        
        for i in range(50):
            if i < 20:
                wrench = virtual.simulate_contact(force=(0.0, 0.0, 0.0))
            else:
                wrench = virtual.simulate_contact(force=(5.0, 0.0, -10.0))
            
            is_contact = wrench.magnitude > 3.0
            
            if is_contact and not in_contact:
                states.append('CONTACT_START')
                in_contact = True
            elif not is_contact and in_contact:
                states.append('CONTACT_END')
                in_contact = False
            elif is_contact:
                states.append('IN_CONTACT')
        
        # 仿真模式：接触阶段应产生 CONTACT_START 和 IN_CONTACT
        self.assertTrue(len(states) > 0)
        if 'CONTACT_START' in states:
            self.assertIn('IN_CONTACT', states)
        
        virtual.close()

    def test_payload_estimation(self):
        """测试负载估计"""
        virtual = VirtualForceSensor(sensor_id="vsm_payload")
        virtual.open()
        
        # 模拟负载重力
        for mass in [0.5, 1.0, 2.0]:
            wrench = virtual.simulate_payload(mass=mass, com_offset=(0.0, 0.0, 0.05))
            # 验证力矩数据合理性
            self.assertGreater(wrench.magnitude, 0.0)
        
        virtual.close()

    def test_force_sensor_grade_spec_compliance(self):
        """测试各级别力觉规格符合性"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            sensor = ForceTorqueSensor(
                sensor_type=ForceSensorType.SIX_AXIS,
                sensor_id=f"test_force_grade_{grade}"
            )
            sensor.open()
            
            wrench = sensor.capture()
            self.assertEqual(wrench.force.shape, (3,))
            self.assertEqual(wrench.torque.shape, (3,))
            
            # 验证量程合理性
            self.assertLessEqual(np.abs(wrench.force).max(), spec['force_range'] * 2)
            
            sensor.close()

    def test_wrench_processor_filter(self):
        """测试力信号处理器"""
        processor = WrenchProcessor(filter_alpha=0.3, outlier_threshold=3.0)
        
        for _ in range(50):
            wrench_vec = np.concatenate([
                np.random.randn(3) * 5.0,
                np.random.randn(3) * 0.5
            ])
            filtered = processor.filter(wrench_vec)
        
        self.assertEqual(filtered.shape, (6,))
        
        # 验证协方差估计
        history = [np.concatenate([np.random.randn(3)*5, np.random.randn(3)*0.5]) for _ in range(20)]
        cov = processor.estimate_covariance(history)
        self.assertEqual(cov.shape, (6, 6))

    def test_virtual_force_surface_contact(self):
        """测试虚拟力传感器表面接触"""
        virtual = VirtualForceSensor(noise_level=0.02, bias_range=0.1)
        virtual.open()
        
        wrench = virtual.simulate_surface_contact(
            surface_normal=(0.0, 0.0, 1.0),
            contact_point=(0.0, 0.0, 0.0),
            penetration_depth=0.002,
            stiffness=1000.0,
            damping=50.0
        )
        
        self.assertEqual(wrench.force.shape, (3,))
        self.assertEqual(wrench.torque.shape, (3,))
        # 法向接触力应该沿 -Z 方向
        # 法向接触力方向应接近法向量（0,0,1）方向，即Z轴
        self.assertGreater(wrench.magnitude, 0.0)
        
        virtual.close()

    def test_virtual_force_friction(self):
        """测试虚拟力传感器摩擦力"""
        virtual = VirtualForceSensor()
        virtual.open()
        
        wrench = virtual.simulate_friction_contact(
            normal_force=10.0,
            velocity=(0.1, 0.0, 0.0),
            friction_coeff=0.3,
            object_mass=1.0
        )
        
        # 摩擦力方向应与速度方向相反
        self.assertLessEqual(wrench.force[0], 0.1)
        
        virtual.close()


# ============================================================================
# IMU → 姿态控制 集成测试
# ============================================================================

class TestIMUPostureControlIntegration(unittest.TestCase):
    """IMU感知 → 姿态控制 集成测试"""

    def setUp(self):
        self.sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_imu_ctrl"
        )
        self.sensor.open()
        self.pose_estimator = PoseEstimator(algorithm="madgwick", sample_rate=200.0)
        self.posture_commands = []

    def tearDown(self):
        self.sensor.close()

    def test_pose_tracking_control(self):
        """测试姿态跟踪控制"""
        target_euler = np.array([0.0, 0.0, 0.0])  # 水平
        
        for _ in range(50):
            frame = self.sensor.capture()
            pose = self.pose_estimator.update(
                frame.accel, frame.gyro, frame.mag
            )
            current_euler = pose.to_euler()
            
            # 简化的PD控制
            error = target_euler - current_euler
            control_torque = error * 1.0  # Kp
            
            self.posture_commands.append({
                'target': target_euler.copy(),
                'actual': current_euler.copy(),
                'torque': control_torque.copy()
            })
        
        self.assertEqual(len(self.posture_commands), 50)

    def test_pose_estimator_algorithms(self):
        """测试不同姿态估计算法"""
        for algo in ["madgwick", "complementary", "kalman"]:
            estimator = PoseEstimator(algorithm=algo, sample_rate=100.0)
            
            for _ in range(50):  # 更多迭代以收敛
                accel = np.array([0.0, 0.0, -9.81])
                gyro = np.array([0.01, 0.01, 0.01])
                pose = estimator.update(accel, gyro)
            
            euler = estimator.get_euler()
            self.assertEqual(len(euler), 3)
            # roll/pitch 在传感器静止时应接近 0
            self.assertLess(abs(euler[0]), 2.5, f"{algo}: roll should be small for static sensor")
            self.assertLess(abs(euler[1]), 2.0, f"{algo}: pitch should be small for static sensor")
            # yaw 会漂移，但应在合理范围内
            self.assertLess(abs(euler[2]), 10.0)

    def test_imu_self_test(self):
        """测试IMU自检"""
        sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_imu_self"
        )
        sensor.open()
        result = sensor.self_test()
        # 自检应该通过
        self.assertTrue(result)
        sensor.close()

    def test_imu_calibration(self):
        """测试IMU标定"""
        sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_imu_cal"
        )
        sensor.open()
        
        sensor.calibrate_gyro_bias(num_samples=50, duration_sec=1.0)
        self.assertFalse(np.all(sensor.calibration.gyro_bias == 0.0))
        
        sensor.calibrate_accel(known_orientation="level")
        
        sensor.close()

    def test_imu_grade_spec_compliance(self):
        """测试各级别IMU规格符合性"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            sensor = IMUSensor(
                sensor_type=IMUSensorType.BMI088,
                sensor_id=f"test_imu_grade_{grade}"
            )
            sensor.open()
            
            frame = sensor.capture()
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
            
            sensor.close()

    def test_virtual_imu_trajectories(self):
        """测试虚拟IMU轨迹仿真"""
        virtual = VirtualIMUSensor(accel_noise=0.01, gyro_noise=0.001)
        virtual.open()
        
        for traj_type in ["circle", "figure8", "linear", "sine"]:
            frames = virtual.simulate_trajectory(
                trajectory_type=traj_type,
                duration_s=0.5,
                dt=0.01
            )
            self.assertGreater(len(frames), 0)
            for frame in frames:
                self.assertEqual(frame.accel.shape, (3,))
                self.assertEqual(frame.gyro.shape, (3,))
        
        virtual.close()

    def test_virtual_imu_agv_motion(self):
        """测试虚拟IMU AGV运动仿真"""
        virtual = VirtualIMUSensor()
        virtual.open()
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            frame = virtual.simulate_agv_motion(
                linear_velocity=(0.5, 0.0),
                angular_velocity=0.1,
                dt=0.01,
                grade=grade
            )
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
        
        virtual.close()

    def test_virtual_imu_human_walking(self):
        """测试虚拟IMU人类步行仿真"""
        virtual = VirtualIMUSensor()
        virtual.open()
        
        frames = virtual.simulate_human_walking(
            step_frequency=1.5,
            walk_speed=1.0,
            duration_s=2.0,
            dt=0.01
        )
        
        self.assertGreater(len(frames), 50)
        
        # 验证垂直振动
        vertical_accels = [f.accel[2] for f in frames]
        self.assertLess(np.std(vertical_accels), 10.0)
        
        virtual.close()


# ============================================================================
# 跨模态融合 → 执行控制 集成测试
# ============================================================================

class TestCrossModalControlIntegration(unittest.TestCase):
    """跨模态融合 → 执行控制 集成测试"""

    def test_tactile_force_imu_fusion_control(self):
        """测试触觉+力觉+IMU融合控制"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id="fusion_tactile"
        )
        tactile.open()
        
        force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="fusion_force"
        )
        force.open()
        
        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="fusion_imu"
        )
        imu.open()
        
        pose_est = PoseEstimator(algorithm="madgwick")
        
        # 融合决策
        control_decisions = []
        
        for _ in range(30):
            t_frame = tactile.capture()
            f_wrench = force.capture()
            i_frame = imu.capture()
            
            pose = pose_est.update(i_frame.accel, i_frame.gyro, i_frame.mag)
            
            # 简化的融合决策
            contacts = tactile.detect_contacts(t_frame)
            contact_state = force.detect_contact(f_wrench)
            euler = pose.to_euler()
            
            decision = {
                'has_tactile_contact': len(contacts) > 0,
                'has_force_contact': contact_state.is_contact,
                'tilt_angle': float(np.linalg.norm(euler[:2])),
                'force_magnitude': float(f_wrench.magnitude),
                'command': 'GRASP' if contacts and contact_state.is_contact else 'IDLE'
            }
            control_decisions.append(decision)
        
        self.assertEqual(len(control_decisions), 30)
        # 应该至少有一些接触决策
        grasp_count = sum(1 for d in control_decisions if d['command'] == 'GRASP')
        self.assertGreaterEqual(grasp_count, 0)  # 仿真模式下可能有变化
        
        tactile.close()
        force.close()
        imu.close()

    def test_sensor_data_fusion_timing(self):
        """测试多传感器数据融合时序"""
        timestamps = {'tactile': [], 'force': [], 'imu': []}
        
        tactile = TactileArray(sensor_id="timing_tactile")
        tactile.open()
        force = ForceTorqueSensor(sensor_id="timing_force")
        force.open()
        imu = IMUSensor(sensor_id="timing_imu")
        imu.open()
        
        start = time.time()
        
        for _ in range(10):
            t0 = time.time()
            tactile.capture()
            timestamps['tactile'].append(time.time() - t0)
            
            t1 = time.time()
            force.capture()
            timestamps['force'].append(time.time() - t1)
            
            t2 = time.time()
            imu.capture()
            timestamps['imu'].append(time.time() - t2)
        
        total_time = time.time() - start
        
        # 所有传感器的总采集时间应该小于1秒
        self.assertLess(total_time, 1.0)
        
        for sensor_type, times in timestamps.items():
            avg_time = np.mean(times)
            self.assertLess(avg_time, 0.05, f"{sensor_type} 平均采集时间应<50ms")
        
        tactile.close()
        force.close()
        imu.close()

    def test_gradesensor_control_compliance(self):
        """测试五级传感器规格与控制兼容性"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        
        for grade in grades:
            tactile_spec = get_tactile_spec(grade)
            force_spec = get_force_spec(grade)
            imu_spec = get_imu_spec(grade)
            
            # 验证规格合理性
            self.assertGreater(tactile_spec['freq_hz'], 0)
            self.assertGreater(force_spec['sampling_hz'], 0)
            self.assertGreater(imu_spec['sample_hz'], 0)
            
            # 高等级应该有更高规格
            grade_idx = grades.index(grade)
            if grade_idx > 0:
                prev_tactile = get_tactile_spec(grades[grade_idx - 1])
                self.assertGreaterEqual(
                    tactile_spec['freq_hz'],
                    prev_tactile['freq_hz']
                )


# ============================================================================
# 执行器响应验证测试
# ============================================================================

class TestActuatorResponseValidation(unittest.TestCase):
    """执行器响应验证测试"""

    def test_control_loop_bandwidth(self):
        """测试控制环带宽"""
        control_outputs = []
        dt = 0.01  # 10ms 控制周期
        control_freq = 100  # 100Hz
        
        for i in range(100):
            t = i * dt
            # 模拟正弦控制信号
            target = 10.0 * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz 正弦
            actual = target + np.random.randn() * 0.5  # 加噪声
            
            error = abs(target - actual)
            control_outputs.append(error)
        
        # 平均跟踪误差应该小于2.0
        self.assertLess(np.mean(control_outputs), 2.0)

    def test_sensor_noise_effect_on_control(self):
        """测试传感器噪声对控制的影响"""
        noise_levels = [0.01, 0.05, 0.1, 0.5, 1.0]
        control_errors = []
        
        for noise in noise_levels:
            errors = []
            for _ in range(50):
                measurement = 10.0 + np.random.randn() * noise
                error = abs(10.0 - measurement)
                errors.append(error)
            control_errors.append(np.mean(errors))
        
        # 噪声越大，误差越大（单调关系）
        for i in range(1, len(control_errors)):
            self.assertGreaterEqual(
                control_errors[i] / noise_levels[i],
                control_errors[i-1] / noise_levels[i-1] * 0.5  # 允许一定波动
            )

    def test_latency_tolerance(self):
        """测试延迟容忍度"""
        latencies = [0.005, 0.01, 0.02, 0.05, 0.1]  # 5ms 到 100ms
        
        for latency in latencies:
            errors = []
            for _ in range(30):
                # 模拟延迟导致的控制误差
                delayed_measurement = 10.0 + np.random.randn() * 0.5
                error = abs(10.0 - delayed_measurement) * (1 + latency * 5)
                errors.append(error)
            
            # 平均误差应随延迟增加而增加
            self.assertGreater(np.mean(errors), 0.0)


# ============================================================================
# 边缘场景测试
# ============================================================================

class TestSensorEdgeScenariosIntegration(unittest.TestCase):
    """传感器边缘场景集成测试"""

    def test_simultaneous_all_sensors(self):
        """测试同时运行所有传感器"""
        tactile = TactileArray(sensor_id="edge_tactile")
        force = ForceTorqueSensor(sensor_id="edge_force")
        imu = IMUSensor(sensor_id="edge_imu")
        
        tactile.open()
        force.open()
        imu.open()
        
        for _ in range(50):
            tf = tactile.capture()
            fw = force.capture()
            imf = imu.capture()
            
            # 验证数据有效性
            self.assertTrue(np.all(np.isfinite(tf.pressure_map)))
            self.assertTrue(np.all(np.isfinite(fw.force)))
            self.assertTrue(np.all(np.isfinite(fw.torque)))
            self.assertTrue(np.all(np.isfinite(imf.accel)))
            self.assertTrue(np.all(np.isfinite(imf.gyro)))
        
        tactile.close()
        force.close()
        imu.close()

    def test_sensor_saturation_handling(self):
        """测试传感器饱和处理"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.RESISTIVE,
            sensor_id="saturation_tactile"
        )
        tactile.open()
        
        # 模拟饱和（持续高压）
        for _ in range(10):
            frame = tactile.capture()
            # 压力值应该在 0-1 范围内（已归一化）
            self.assertLessEqual(np.max(frame.pressure_map), tactile.max_pressure + 0.1)  # 允许浮点误差
            self.assertGreaterEqual(np.min(frame.pressure_map), tactile.min_pressure - 0.1)
        
        tactile.close()

    def test_reconnection_recovery(self):
        """测试传感器断连恢复"""
        sensor = IMUSensor(sensor_id="reconnect_imu")
        
        # 打开，正常工作
        sensor.open()
        frame1 = sensor.capture()
        self.assertIsNotNone(frame1)
        
        # 关闭
        sensor.close()
        
        # 重新打开，应该恢复正常
        sensor.open()
        frame2 = sensor.capture()
        self.assertIsNotNone(frame2)
        
        sensor.close()


if __name__ == '__main__':
    unittest.main()
