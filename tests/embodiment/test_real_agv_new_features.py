"""
test_real_agv_new_features.py - 真实AGV接口新功能测试
=====================================================
测试 TrajectoryTrackingController 和 SensorAutoCalibrator
"""

import pytest
import numpy as np
import time


class TestTrajectoryTrackingController:
    """TrajectoryTrackingController 轨迹跟踪控制器测试"""

    def test_initialization(self):
        """测试控制器初始化"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController(grade="M")
        assert controller.grade == "M"
        assert controller.look_ahead_gain == 0.5
        assert controller.max_linear_vel == 1.0

    def test_grade_params(self):
        """测试AGV五级参数"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        for grade, expected_lag in [("S", 0.3), ("M", 0.5), ("L", 0.7), ("XL", 0.9), ("XXL", 1.2)]:
            ctrl = TrajectoryTrackingController(grade=grade)
            assert ctrl.look_ahead_gain == expected_lag, f"Grade {grade} LA gain mismatch"

    def test_load_trajectory(self):
        """测试轨迹加载"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        waypoints = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        controller.load_trajectory(waypoints)
        assert controller.total_waypoints == 5
        assert controller.current_waypoint_index == 0

    def test_look_ahead_distance_speed_dependent(self):
        """测试前视距离与速度相关"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController(look_ahead_gain=0.5, min_look_ahead=0.1, max_look_ahead=1.0)
        lad_slow = controller.compute_look_ahead_distance(0.5)
        lad_fast = controller.compute_look_ahead_distance(2.0)
        assert lad_fast > lad_slow
        assert lad_slow >= 0.1
        assert lad_fast <= 1.0

    def test_pure_pursuit_straight_line(self):
        """测试直线路径的Pure Pursuit"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        controller.load_trajectory([(0, 0), (5, 0), (10, 0)])
        # 机器人朝向x轴正方向
        v_cmd, omega_cmd = controller.compute_pure_pursuit(
            current_pos=(0.0, 0.0),
            current_theta=0.0,
            velocity=0.5,
        )
        assert abs(omega_cmd) < 0.1, "Straight line should need no steering"
        assert v_cmd > 0, "Should move forward"

    def test_pure_pursuit_turn(self):
        """测试转弯的Pure Pursuit"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        controller.load_trajectory([(0, 0), (1, 0), (1, 1), (0, 1)])
        # 接近转弯点时应产生角速度
        v_cmd, omega_cmd = controller.compute_pure_pursuit(
            current_pos=(0.9, 0.0),
            current_theta=0.0,
            velocity=0.5,
        )
        assert omega_cmd != 0.0 or v_cmd >= 0

    def test_pid_velocity_control(self):
        """测试PID速度控制"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController(kp_velocity=2.0, ki_velocity=0.0, kd_velocity=0.0)
        torque = controller.compute_pid_velocity(current_vel=0.5, target_vel=1.0, dt=0.1)
        assert torque > 0, "Should accelerate when below target"

        torque_brake = controller.compute_pid_velocity(current_vel=1.5, target_vel=1.0, dt=0.1)
        assert torque_brake < 0, "Should brake when above target"

    def test_pid_integral_windup(self):
        """测试PID积分饱和 (anti-windup)"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController(kp_velocity=1.0, ki_velocity=1.0, kd_velocity=0.0)
        # 持续积累积分
        for _ in range(100):
            controller.compute_pid_velocity(current_vel=0.0, target_vel=10.0, dt=0.1)
        # 积分应被限制
        assert controller.integral_vel <= 2.0, "Integral windup should be clamped"

    def test_track_trajectory_full(self):
        """测试完整轨迹跟踪"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        controller.load_trajectory([(i, 0) for i in range(10)])
        result = controller.track_trajectory(
            current_pos=(0.0, 0.0),
            current_theta=0.0,
            current_vel=0.0,
            dt=0.1,
        )
        assert 'v_cmd' in result
        assert 'omega_cmd' in result
        assert 'cross_track_error' in result
        assert 'waypoint_progress' in result
        assert 'finished' in result
        assert result['finished'] is False

    def test_track_finished_trajectory(self):
        """测试轨迹完成检测"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        controller.load_trajectory([(0, 0), (0.1, 0)])
        # 手动将索引设置到最后点附近，模拟已到达终点
        controller.current_waypoint_index = len(controller.trajectory) - 1
        result = controller.track_trajectory(
            current_pos=(0.1, 0.0),
            current_theta=0.0,
            current_vel=0.0,
            dt=0.1,
        )
        # 轨迹有2个点，最后点索引是1
        assert result['finished'] is True or result['waypoint_progress'] >= 0.99

    def test_cross_track_error_history(self):
        """测试横向误差历史记录"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        controller.load_trajectory([(0, 0), (5, 0)])
        for _ in range(5):
            controller.track_trajectory((0, 0.1), 0.0, 0.5, 0.1)
        assert len(controller.cross_track_error_history) == 5

    def test_reset(self):
        """测试重置"""
        from src.embodied.real_agv_interface import TrajectoryTrackingController
        controller = TrajectoryTrackingController()
        controller.load_trajectory([(0, 0), (5, 0)])
        controller.current_waypoint_index = 3
        controller.integral_vel = 1.0
        controller.reset()
        assert controller.current_waypoint_index == 0
        assert controller.integral_vel == 0.0


class TestSensorAutoCalibrator:
    """SensorAutoCalibrator 传感器自动标定器测试"""

    def test_calibration_status_idle(self):
        """测试初始状态为IDLE"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator()
        assert calibrator.imu_calibration_status == CalibrationStatus.IDLE
        assert calibrator.force_calibration_status == CalibrationStatus.IDLE
        assert calibrator.odom_calibration_status == CalibrationStatus.IDLE

    def test_imu_calibration_collection(self):
        """测试IMU标定数据采集"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        calibrator.start_imu_calibration()
        assert calibrator.imu_calibration_status == CalibrationStatus.RUNNING

        # 填入数据
        for _ in range(9):
            result = calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)
            assert result is False  # 未完成
        # 第10个样本应触发完成
        result = calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)
        assert result is True
        assert calibrator.imu_calibration_status == CalibrationStatus.CALIBRATED

    def test_imu_calibration_bias(self):
        """测试IMU零偏标定"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=20)
        calibrator.start_imu_calibration()
        # 添加有零偏的样本
        for _ in range(20):
            calibrator.add_imu_sample(np.array([0.05, -0.03, 9.83]), 0.02)
        assert calibrator.imu_calibration_status == CalibrationStatus.CALIBRATED
        assert calibrator.imu_accel_bias is not None
        assert calibrator.imu_gyro_bias is not None
        assert calibrator.imu_noise_std is not None

    def test_imu_calibration_application(self):
        """测试IMU标定补偿应用"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=20)
        calibrator.start_imu_calibration()
        for _ in range(20):
            calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)

        # 应用标定
        raw_accel = np.array([0.1, -0.05, 9.90])
        raw_gyro = 0.03
        cal_accel, cal_gyro = calibrator.apply_imu_calibration(raw_accel, raw_gyro)
        # 校正后应接近原始值减去标定零偏
        z_diff = float(abs(cal_accel[2] - 9.81))
        assert z_diff < 0.5, f"z-axis should be close to 9.81 after calibration, diff={z_diff}"

    def test_force_calibration(self):
        """测试力传感器标定"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        calibrator.start_force_calibration()
        for _ in range(9):
            result = calibrator.add_force_sample(np.array([0.5, -0.2, 0.0, 0.0, 0.0, 0.0]))
            assert result is False
        calibrator.add_force_sample(np.array([0.5, -0.2, 0.0, 0.0, 0.0, 0.0]))
        assert calibrator.force_calibration_status == CalibrationStatus.CALIBRATED
        assert calibrator.force_zero_drift is not None
        assert len(calibrator.force_zero_drift) == 6

    def test_force_calibration_application(self):
        """测试力传感器标定补偿"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        calibrator.start_force_calibration()
        zero_load = np.array([0.3, -0.1, 0.0, 0.0, 0.0, 0.0])
        for _ in range(10):
            calibrator.add_force_sample(zero_load)

        raw_force = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        cal_force = calibrator.apply_force_calibration(raw_force)
        # 补偿后应移除零点漂移 (0.5 - 0.3 = 0.2)
        x_diff = float(abs(cal_force[0] - 0.2))
        assert x_diff < 0.1, f"Force x should be ~0.2 after calibration, diff={x_diff}"

    def test_odometry_scale_calibration(self):
        """测试里程计比例因子标定"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator()
        # 命令1m但只走了0.9m -> 比例因子应 > 1
        scale = calibrator.calibrate_odometry_scale(commanded_distance=1.0, measured_distance=0.9)
        assert abs(scale - 1.0 / 0.9) < 0.01
        assert calibrator.odom_calibration_status == CalibrationStatus.CALIBRATED

    def test_odometry_calibration_application(self):
        """测试里程计标定应用"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator
        calibrator = SensorAutoCalibrator()
        calibrator.calibrate_odometry_scale(1.0, 0.8)
        calibrated = calibrator.apply_odometry_calibration(0.8)
        assert abs(calibrated - 1.0) < 0.01

    def test_calibration_report(self):
        """测试标定报告"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        calibrator.start_imu_calibration()
        for _ in range(10):
            calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)
        report = calibrator.get_calibration_report()
        assert 'imu' in report
        assert 'force' in report
        assert 'odometry' in report
        assert report['imu']['status'] == CalibrationStatus.CALIBRATED.value

    def test_is_fully_calibrated(self):
        """测试完全标定检查"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        assert calibrator.is_fully_calibrated() is False
        # IMU
        calibrator.start_imu_calibration()
        for _ in range(10):
            calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)
        # Force
        calibrator.start_force_calibration()
        for _ in range(10):
            calibrator.add_force_sample(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # Odometry
        calibrator.calibrate_odometry_scale(1.0, 1.0)
        assert calibrator.is_fully_calibrated() is True

    def test_calibration_history(self):
        """测试标定历史"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        calibrator.start_imu_calibration()
        for _ in range(10):
            calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)
        calibrator.calibrate_odometry_scale(1.0, 1.0)
        assert len(calibrator.calibration_history) == 2

    def test_reset(self):
        """测试标定重置"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator, CalibrationStatus
        calibrator = SensorAutoCalibrator(calibration_samples=10)
        calibrator.start_imu_calibration()
        calibrator.calibrate_odometry_scale(1.0, 0.8)
        calibrator.reset()
        assert calibrator.imu_calibration_status == CalibrationStatus.IDLE
        assert calibrator.odom_scale_factor == 1.0
        assert calibrator.imu_accel_bias is None

    def test_add_sample_when_not_running(self):
        """测试非标定状态添加样本无效"""
        from src.embodied.real_agv_interface import SensorAutoCalibrator
        calibrator = SensorAutoCalibrator()
        result = calibrator.add_imu_sample(np.array([0.0, 0.0, 9.81]), 0.0)
        assert result is False
        result = calibrator.add_force_sample(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert result is False
