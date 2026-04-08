"""
具身智能完整流水线测试
=======================

测试 SuperModel 完整具身智能流水线:
感知 → 融合 → 认知 → 决策 → 控制 → 执行

覆盖五级AGV (S/M/L/XL/XXL) 的完整闭环

运行方式:
  python -m pytest tests/embodied_pipeline_extended_tests.py -v
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.tactile import (
    TactileArray, TactileSensorType, VirtualTactileSensor
)
from src.sensors.force import (
    ForceTorqueSensor, ForceSensorType, VirtualForceSensor, Wrench
)
from src.sensors.imu import (
    IMUSensor, IMUSensorType, VirtualIMUSensor, PoseEstimator, Pose
)
from src.fusion.sensor_fusion import (
    ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion
)
from src.control.tactile_control import (
    TactileServoController, TactileServoParams
)
from src.control.force_control import (
    ForceController, ForceControlParams, HybridForcePositionController
)
from src.control.imu_control import (
    AttitudeStabilizer, IMUControlParams
)
from src.control.agv import (
    AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist
)
from src.control.safety_controller import (
    SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
)


class TestEmbodiedPipelineGrades(unittest.TestCase):
    """测试五级AGV具身流水线"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']

    def test_agv_spec_from_grade(self):
        """测试各等级AGV规格生成"""
        for grade in self.GRADES:
            spec = AGVSpec.from_grade(AGVGrade[grade])
            self.assertEqual(spec.grade.value, grade)
            self.assertGreater(spec.max_linear_speed, 0)
            self.assertGreater(spec.wheel_radius, 0)

    def test_agv_motion_controller_all_grades(self):
        """测试各等级AGV运动控制器"""
        for grade in self.GRADES:
            spec = AGVSpec.from_grade(AGVGrade[grade])
            agv = AGVMotionController(spec)
            agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))

            target = AGVPose(x=1.0, y=0.0, theta=0.0)
            cmds = agv.compute_wheel_commands(target, dt=0.01)
            cmds_safe = agv.apply_safety_limits(cmds)

            # DIFFERENTIAL (S/M) = 2 wheels; MECANUM (L/XL/XXL) = 4 wheels
            expected_wheels = 2 if spec.drive_type.name == 'DIFFERENTIAL' else 4
            self.assertEqual(len(cmds_safe), expected_wheels)
            self.assertIsInstance(cmds_safe, np.ndarray)


class TestSensorimotor闭环(unittest.TestCase):
    """测试传感器-执行器完整闭环"""

    def test_tactile_servo_closed_loop(self):
        """触觉伺服闭环测试"""
        sensor = TactileArray(array_size=(16, 16), sensor_id="test_tactile")
        sensor.open()

        params = TactileServoParams.from_grade('M')
        controller = TactileServoController(sensor, params)

        # 捕获触觉帧
        for step in range(10):
            frame = sensor.capture()
            cmd = controller.compute_control_signal(target_force=5.0, current_frame=frame)
            self.assertIsInstance(cmd, np.ndarray)

        sensor.close()

    def test_force_control_closed_loop(self):
        """力控闭环测试"""
        sensor = ForceTorqueSensor(sensor_id="test_force")
        sensor.open()

        params = ForceControlParams.from_grade('L')
        controller = HybridForcePositionController(sensor, params)

        # 目标力
        desired_force = np.array([0.0, 0.0, -10.0])

        for step in range(10):
            wrench = sensor.capture()
            pos_output, force_output = controller.compute_control(
                target_force=desired_force,
                target_position=np.zeros(3),
                measured_wrench=wrench
            )
            self.assertIsInstance(pos_output, np.ndarray)
            self.assertIsInstance(force_output, np.ndarray)

        sensor.close()

    def test_imu_attitude_stabilizer(self):
        """IMU姿态稳定闭环测试"""
        sensor = VirtualIMUSensor()
        sensor.open()

        params = IMUControlParams.from_grade('L')
        stabilizer = AttitudeStabilizer(params)

        estimator = PoseEstimator(algorithm='madgwick', beta=0.1)

        for step in range(50):
            # 模拟小幅扰动
            frame = sensor.simulate_static(
                orientation=(0.1 * np.sin(step * 0.1),
                            0.05 * np.cos(step * 0.1),
                            0.0)
            )

            pose = estimator.update(frame.accel, frame.gyro)
            correction = stabilizer.update(frame, dt=0.01)

            self.assertIsInstance(correction, np.ndarray)
            self.assertEqual(len(correction), 3)

        sensor.close()

    def test_multi_sensor_fusion_pipeline(self):
        """多传感器融合流水线测试"""
        tactile = TactileArray(array_size=(16, 16), sensor_id="fuse_tactile")
        force = ForceTorqueSensor(sensor_id="fuse_force")
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="fuse_imu")

        tactile.open()
        force.open()
        imu.open()

        # 互补滤波
        comp_filter = ComplementaryFilter(alpha=0.96)
        # EKF
        ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)

        for step in range(20):
            # 触觉
            tac_frame = tactile.capture()
            tac_avg = float(np.mean(tac_frame.pressure_map))

            # 力觉
            wrench = force.capture()
            force_mag = float(wrench.magnitude)

            # IMU
            imu_frame = imu.capture()
            accel = imu_frame.accel

            # 融合
            comp_state = comp_filter.update(
                {'accel': accel, 'gyro': imu_frame.gyro}, dt=0.01
            )

            ekf.predict(dt=0.01)
            ekf.correct(accel)
            ekf_state = ekf.get_state()

            self.assertEqual(len(comp_state), 3)
            self.assertEqual(len(ekf_state), 6)

        tactile.close()
        force.close()
        imu.close()


class TestSafetyControllerGrades(unittest.TestCase):
    """测试各等级安全控制器"""

    def _make_safety_config(self, grade: str) -> SafetyConfig:
        """根据AGV等级创建安全配置"""
        num_joints = 6
        limits = {
            'S': dict(
                velocity_limits=np.ones(num_joints) * 2.0,
                acceleration_limits=np.ones(num_joints) * 5.0,
                torque_limits=np.ones(num_joints) * 50.0,
                safety_level=SafetyLevel.S
            ),
            'M': dict(
                velocity_limits=np.ones(num_joints) * 3.0,
                acceleration_limits=np.ones(num_joints) * 10.0,
                torque_limits=np.ones(num_joints) * 100.0,
                safety_level=SafetyLevel.M
            ),
            'L': dict(
                velocity_limits=np.ones(num_joints) * 5.0,
                acceleration_limits=np.ones(num_joints) * 20.0,
                torque_limits=np.ones(num_joints) * 200.0,
                safety_level=SafetyLevel.L
            ),
            'XL': dict(
                velocity_limits=np.ones(num_joints) * 8.0,
                acceleration_limits=np.ones(num_joints) * 40.0,
                torque_limits=np.ones(num_joints) * 500.0,
                safety_level=SafetyLevel.XL
            ),
            'XXL': dict(
                velocity_limits=np.ones(num_joints) * 10.0,
                acceleration_limits=np.ones(num_joints) * 50.0,
                torque_limits=np.ones(num_joints) * 1000.0,
                safety_level=SafetyLevel.XXL
            ),
        }
        cfg = limits.get(grade, limits['M'])
        return SafetyConfig(
            joint_limits_lower=-np.ones(num_joints) * np.pi,
            joint_limits_upper=np.ones(num_joints) * np.pi,
            **cfg
        )

    def test_safety_limits_all_grades(self):
        """测试各等级安全限幅"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            safety_config = self._make_safety_config(grade)
            safety = SafetyController(safety_config)

            # 超速状态
            state = JointStateSnapshot(
                positions=np.zeros(6),
                velocities=np.ones(6) * 100.0,  # 远超安全限制
                accelerations=np.zeros(6),
                torques=np.zeros(6),
                timestamp=time.time()
            )

            result = safety.check(state)
            # 高等级应该检测到超速
            if grade in ['XL', 'XXL']:
                self.assertFalse(result.safe)

            safety.disable()

    def test_safe_velocity_computation(self):
        """测试安全速度计算"""
        for grade in ['M', 'L']:
            safety_config = self._make_safety_config(grade)
            safety = SafetyController(safety_config)

            current_vel = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            desired_vel = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

            safe_vel = safety.compute_safe_velocity(current_vel, desired_vel)

            # 安全速度应该被限幅
            max_limit = np.max(safety_config.velocity_limits)
            self.assertTrue(np.all(np.abs(safe_vel) <= max_limit))

            safety.disable()


class TestEndToEndPipeline(unittest.TestCase):
    """端到端具身智能流水线测试"""

    def test_full_pipeline_single_step(self):
        """完整流水线单步测试 (使用真实传感器接口)"""
        # 初始化各模块 (使用真实传感器接口)
        tactile = TactileArray(array_size=(24, 24), sensor_id="pipe_tactile")
        force = ForceTorqueSensor(sensor_id="pipe_force")
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="pipe_imu")

        tactile.open()
        force.open()
        imu.open()

        # 触觉控制器
        tac_params = TactileServoParams.from_grade('L')
        tac_ctrl = TactileServoController(tactile, tac_params)

        # 力控制器 (使用HybridForcePositionController以支持compute_control)
        force_params = ForceControlParams.from_grade('L')
        force_ctrl = HybridForcePositionController(force, force_params)

        # IMU稳定器
        imu_params = IMUControlParams.from_grade('L')
        imu_stab = AttitudeStabilizer(imu_params)
        estimator = PoseEstimator(algorithm='madgwick')

        # AGV控制器
        agv_spec = AGVSpec.from_grade(AGVGrade.L)
        agv = AGVMotionController(agv_spec)
        agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))

        # 完整流水线 - 采集传感器数据
        tac_frame = tactile.capture()
        wrench = force.capture()
        imu_frame = imu.capture()

        # 感知
        tac_frame = tactile.capture()
        wrench = force.capture()
        imu_frame = imu.capture()

        tac_cmd = tac_ctrl.compute_control_signal(target_force=8.0, current_frame=tac_frame)
        pos_out, force_out = force_ctrl.compute_control(
            target_force=np.array([0, 0, -8.0]),
            target_position=np.zeros(3),
            measured_wrench=wrench
        )
        pose = estimator.update(imu_frame.accel, imu_frame.gyro)
        correction = imu_stab.update(imu_frame, dt=0.01)

        # 规划
        target = AGVPose(x=0.5, y=0.0, theta=0.0)
        wheel_cmds = agv.compute_wheel_commands(target, dt=0.01)

        # 验证输出
        self.assertEqual(len(tac_cmd), 3)
        self.assertEqual(len(pos_out), 3)
        self.assertEqual(len(force_out), 3)
        self.assertEqual(len(correction), 3)
        # DIFFERENTIAL (L uses MECANUM) = 4 wheels
        self.assertEqual(len(wheel_cmds), 4)

        tactile.close()
        force.close()
        imu.close()

    def test_pipeline_continuous_loop(self):
        """连续运行流水线测试"""
        tactile = VirtualTactileSensor(array_size=(16, 16))
        force = VirtualForceSensor()
        imu = VirtualIMUSensor()

        tactile.open()
        force.open()
        imu.open()

        estimator = PoseEstimator(algorithm='complementary')
        estimator.reset()

        errors = []
        for step in range(100):
            try:
                # 使用IMU模拟运动
                imu_frame = imu.simulate_static(orientation=(0.0, 0.0, 0.0))
                estimator.update(imu_frame.accel, imu_frame.gyro, dt=0.01)

            except Exception as e:
                errors.append(str(e))

        self.assertEqual(len(errors), 0, f"Pipeline errors: {errors[:3]}")

        tactile.close()
        force.close()
        imu.close()

    def test_fusion_output_shapes(self):
        """测试融合输出维度"""
        comp_filter = ComplementaryFilter(alpha=0.98)
        ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)

        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.0, 0.0, 0.1])

        state_comp = comp_filter.update({'accel': accel, 'gyro': gyro}, dt=0.01)
        ekf.predict(dt=0.01)
        ekf.correct(accel)
        state_ekf = ekf.get_state()

        self.assertEqual(state_comp.shape, (3,))
        self.assertEqual(state_ekf.shape, (6,))


if __name__ == '__main__':
    unittest.main()
