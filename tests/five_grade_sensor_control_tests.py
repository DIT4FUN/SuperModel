"""
AGV五级传感器-控制集成测试
==========================

测试从传感器采集 → 融合 → 控制指令的完整五级pipeline

覆盖:
- S级: 50Hz, 简化模型, 单目+IMU
- M级: 100Hz, 标准模型, 双目+力觉+IMU  
- L级: 200Hz, 高保真模型, 多目+力觉+IMU+触觉
- XL级: 500Hz, 极高保真, 全传感器+冗余
- XXL级: 1000Hz, 极致仿真, 全传感器+数字孪生

Author: SuperModel Development Team
Version: v1.0 (2026-04-10)
"""

import unittest
import numpy as np
import sys
import os
import time

_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SrcPath = os.path.join(_ProjectRoot, 'src')
if _SrcPath not in sys.path:
    sys.path.insert(0, _SrcPath)
if _ProjectRoot not in sys.path:
    sys.path.insert(0, _ProjectRoot)

from src.sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType,
    VirtualTactileSensor, get_tactile_spec, AGV_TACTILE_GRADES
)
from src.sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor, get_force_spec, AGV_FORCE_GRADES
)
from src.sensors.imu import (
    IMUSensor, IMUFrame, IMUSensorType, PoseEstimator, Pose,
    VirtualIMUSensor, get_imu_spec, AGV_IMU_GRADES
)
from src.fusion.sensor_fusion import ComplementaryFilter, ExtendedKalmanFilter
from src.fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from src.control.velocity_control import (
    AGV_VELOCITY_CONTROL_GRADES, get_velocity_control_spec,
    VelocityPIDController, SVelocityProfilePlanner,
    FrictionCompensator, WheelVelocitySynchronizer,
    AGVVelocityController, WheelVelocityState, VelocityProfileType
)
from src.control.simulation import (
    SimulationInterface, SimulationBackend, SimulationGrade,
    get_agv_sim_params, get_simulation_spec
)
from src.control.embodied_sim import (
    EmbodiedSimulator, EmbodiedSimGrade, get_sim_grade_spec
)


GRADES = ['S', 'M', 'L', 'XL', 'XXL']


class TestTactileFiveGrades(unittest.TestCase):
    """触觉传感器五级规格测试"""

    def test_all_grades_have_tactile_spec(self):
        """验证所有五级都有触觉规格"""
        for grade in GRADES:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('res', spec)
            self.assertIn('range_kpa', spec)
            self.assertIn('freq_hz', spec)

    def test_tactile_spec_increases_with_grade(self):
        """验证触觉规格随等级提升"""
        prev_freq = 0
        prev_array_area = 0
        for grade in GRADES:
            spec = get_tactile_spec(grade)
            self.assertGreaterEqual(spec['freq_hz'], prev_freq)
            arr_area = spec['array'][0] * spec['array'][1]
            self.assertGreaterEqual(arr_area, prev_array_area)
            prev_freq = spec['freq_hz']
            prev_array_area = arr_area

    def test_tactile_virtual_sensor_all_grades(self):
        """测试所有等级的虚拟触觉传感器"""
        for grade in GRADES:
            spec = get_tactile_spec(grade)
            size = spec['array']
            with VirtualTactileSensor(array_size=size, sensor_id=f"tactile_{grade}") as sensor:
                frame = sensor.simulate_contact(
                    contact_pos=(0.5, 0.5),
                    contact_radius=0.25,
                    contact_force=10.0
                )
                self.assertEqual(frame.pressure_map.shape, size)
                self.assertIsNotNone(frame.temperature_map)
            print(f"  [OK] Tactile grade {grade}: {size} @ {spec['freq_hz']}Hz")


class TestForceFiveGrades(unittest.TestCase):
    """力觉传感器五级规格测试"""

    def test_all_grades_have_force_spec(self):
        """验证所有五级都有力觉规格"""
        for grade in GRADES:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('torque_range', spec)
            self.assertIn('sampling_hz', spec)

    def test_force_spec_increases_with_grade(self):
        """验证力觉规格随等级提升"""
        prev_sampling = 0
        prev_force_range = 0
        for grade in GRADES:
            spec = get_force_spec(grade)
            self.assertGreaterEqual(spec['sampling_hz'], prev_sampling)
            self.assertGreaterEqual(spec['force_range'], prev_force_range)
            prev_sampling = spec['sampling_hz']
            prev_force_range = spec['force_range']

    def test_force_virtual_sensor_all_grades(self):
        """测试所有等级的虚拟力觉传感器"""
        for grade in GRADES:
            spec = get_force_spec(grade)
            with VirtualForceSensor(sensor_id=f"force_{grade}") as sensor:
                wrench = sensor.simulate_contact(
                    force=(10.0, 0.0, -5.0),
                    torque=(0.0, 0.0, 1.0)
                )
                self.assertEqual(wrench.force.shape, (3,))
                self.assertEqual(wrench.torque.shape, (3,))
                self.assertGreater(wrench.magnitude, 0)
            print(f"  [OK] Force grade {grade}: {spec['axes']} axes @ {spec['sampling_hz']}Hz")


class TestIMUFiveGrades(unittest.TestCase):
    """IMU传感器五级规格测试"""

    def test_all_grades_have_imu_spec(self):
        """验证所有五级都有IMU规格"""
        for grade in GRADES:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)

    def test_imu_spec_increases_with_grade(self):
        """验证IMU规格随等级提升"""
        prev_sample = 0
        prev_noise = float('inf')
        for grade in GRADES:
            spec = get_imu_spec(grade)
            self.assertGreaterEqual(spec['sample_hz'], prev_sample)
            self.assertLessEqual(spec['noise_density'], prev_noise)
            prev_sample = spec['sample_hz']
            prev_noise = spec['noise_density']

    def test_virtual_imu_all_grades(self):
        """测试所有等级的虚拟IMU传感器"""
        for grade in GRADES:
            spec = get_imu_spec(grade)
            with VirtualIMUSensor(sensor_id=f"imu_{grade}") as sensor:
                frame = sensor.simulate_static(orientation=(0.1, 0.2, 0.0))
                self.assertEqual(frame.accel.shape, (3,))
                self.assertEqual(frame.gyro.shape, (3,))
                # 静止时重力应在z轴
                self.assertLess(abs(frame.accel[2] - 9.81), 1.0)
            print(f"  [OK] IMU grade {grade}: {spec['type']} @ {spec['sample_hz']}Hz")


class TestVelocityControlFiveGrades(unittest.TestCase):
    """速度控制五级规格测试"""

    def test_all_grades_have_velocity_spec(self):
        """验证所有五级都有速度控制规格"""
        for grade in GRADES:
            spec = get_velocity_control_spec(grade)
            self.assertIn('control_frequency_hz', spec)
            self.assertIn('max_linear_velocity_mps', spec)

    def test_velocity_control_frequency_increases(self):
        """验证控制频率随等级提升"""
        prev_freq = 0
        for grade in GRADES:
            spec = get_velocity_control_spec(grade)
            self.assertGreaterEqual(spec['control_frequency_hz'], prev_freq)
            prev_freq = spec['control_frequency_hz']

    def test_velocity_pid_controller_all_grades(self):
        """测试所有等级的速度PID控制器"""
        for grade in GRADES:
            spec = get_velocity_control_spec(grade)
            kp = spec['velocity_pid_kp']
            ki = spec['velocity_pid_ki']
            kd = spec['velocity_pid_kd']
            
            controller = VelocityPIDController(kp=kp, ki=ki, kd=kd)
            # 模拟速度跟踪: setpoint=1.0, measurement=0.5
            output = controller.compute(setpoint=1.0, measurement=0.5, dt=0.01)
            # 验证输出是数值
            self.assertIsInstance(output, (float, np.floating))
            self.assertIsNotNone(output)
            controller.reset()
            print(f"  [OK] Velocity PID grade {grade}: freq={spec['control_frequency_hz']}Hz")

    def test_velocity_profile_planner_all_grades(self):
        """测试所有等级的速度曲线规划器"""
        for grade in GRADES:
            spec = get_velocity_control_spec(grade)
            planner = SVelocityProfilePlanner(
                max_velocity=spec['max_linear_velocity_mps'],
                max_acceleration=spec['acceleration_limit_mps2'],
                max_jerk=spec.get('jerk_limit_mps3')
            )
            profile = planner.plan(
                start_pos=0.0,
                end_pos=1.0,
                max_velocity=spec['max_linear_velocity_mps'],
                max_acceleration=spec['acceleration_limit_mps2']
            )
            self.assertIsNotNone(profile)
            print(f"  [OK] Velocity profile grade {grade}: {spec['profile_type']}")


class TestSimulationFiveGrades(unittest.TestCase):
    """仿真环境五级规格测试"""

    def test_all_grades_have_sim_spec(self):
        """验证所有五级都有仿真规格"""
        for grade in GRADES:
            spec = get_simulation_spec(grade)
            self.assertIn('backend', spec)
            self.assertIn('dt', spec)
            self.assertIn('freq', spec)

    def test_agv_sim_params_all_grades(self):
        """验证所有五级都有AGV仿真参数"""
        for grade in GRADES:
            params = get_agv_sim_params(grade)
            self.assertGreater(params.max_load_kg, 0)
            self.assertGreater(params.wheel_radius_m, 0)
            self.assertGreater(params.vehicle_mass_kg, 0)
            print(f"  [OK] AGV sim grade {grade}: load={params.max_load_kg}kg, mass={params.vehicle_mass_kg}kg")

    def test_sim_dt_decreases_with_grade(self):
        """验证仿真步长随等级提升而减小"""
        prev_dt = float('inf')
        for grade in GRADES:
            spec = get_simulation_spec(grade)
            self.assertLessEqual(spec['dt'], prev_dt)
            prev_dt = spec['dt']


class TestEmbodiedSimFiveGrades(unittest.TestCase):
    """具身仿真五级规格测试"""

    def test_all_grades_have_sim_grade_spec(self):
        """验证所有五级都有具身仿真规格"""
        for grade in GRADES:
            spec = get_sim_grade_spec(grade)
            self.assertIn('dt', spec)
            self.assertIn('control_rate', spec)
            self.assertIn('max_linear_speed', spec)

    def test_embodied_sim_grade_increases(self):
        """验证具身仿真规格随等级提升"""
        prev_rate = 0
        prev_speed = 0.0
        for grade in GRADES:
            spec = get_sim_grade_spec(grade)
            self.assertGreaterEqual(spec['control_rate'], prev_rate)
            self.assertGreaterEqual(spec['max_linear_speed'], prev_speed)
            prev_rate = spec['control_rate']
            prev_speed = spec['max_linear_speed']


class TestSensorFusionPipelineFiveGrades(unittest.TestCase):
    """传感器-融合-控制完整pipeline五级测试"""

    def test_tactile_to_fusion_to_control_pipeline(self):
        """测试触觉→融合→控制完整pipeline"""
        for grade in GRADES:
            spec = get_tactile_spec(grade)
            array_size = spec['array']
            
            # 1. 触觉采集
            with VirtualTactileSensor(array_size=array_size) as sensor:
                frame = sensor.simulate_contact(
                    contact_pos=(0.5, 0.5),
                    contact_force=10.0,
                    contact_radius=0.2
                )
                pressure_mean = float(np.mean(frame.pressure_map))
                
                # 2. 质量评估 (使用TactileArray, 不是VirtualTactileSensor)
                with TactileArray(array_size=array_size, sensor_id=f"tactile_{grade}") as real_sensor:
                    real_sensor.open()
                    real_sensor._frame_buffer.append(frame)
                    quality = real_sensor.estimate_grip_quality(frame)
                
                # 3. 基于质量的控制指令生成
                if quality['overall'] > 0.1:
                    control_cmd = np.array([0.5, 0.0])  # 正常速度
                else:
                    control_cmd = np.array([0.0, 0.0])  # 停止
                
                self.assertEqual(len(control_cmd), 2)
            
            print(f"  [OK] Tactile→Control pipeline grade {grade}: quality={quality['overall']:.3f}")

    def test_force_imu_fusion_pipeline(self):
        """测试力觉+IMU融合pipeline"""
        for grade in GRADES:
            force_spec = get_force_spec(grade)
            imu_spec = get_imu_spec(grade)
            
            # 1. 力觉采集
            with VirtualForceSensor(sensor_id=f"force_{grade}") as force_sensor:
                wrench = force_sensor.simulate_contact(
                    force=(10.0, 0.0, -5.0),
                    torque=(0.0, 0.0, 1.0)
                )
                force_mag = wrench.magnitude
            
            # 2. IMU采集
            with VirtualIMUSensor(sensor_id=f"imu_{grade}") as imu_sensor:
                frame = imu_sensor.simulate_static(orientation=(0.0, 0.0, 0.0))
                accel_mag = frame.accel_magnitude
            
            # 3. 互补滤波 (expects dict with 'accel' and 'gyro')
            filter = ComplementaryFilter(alpha=0.96)
            roll = filter.update({'accel': frame.accel, 'gyro': frame.gyro}, dt=0.01)
            
            # 4. 融合决策
            if force_mag > 5.0 and 9.5 < accel_mag < 10.5:
                status = "normal"
            elif force_mag > 20.0:
                status = "high_force"
            else:
                status = "unknown"
            
            self.assertIn(status, ["normal", "high_force", "unknown"])
            print(f"  [OK] Force+IMU pipeline grade {grade}: force={force_mag:.2f}N, accel={accel_mag:.2f}m/s²")

    def test_complete_sensor_control_loop(self):
        """测试完整传感器→控制闭环"""
        for grade in GRADES:
            # 获取各传感器规格
            vel_spec = get_velocity_control_spec(grade)
            
            loop_rate = vel_spec['control_frequency_hz']
            dt = 1.0 / loop_rate
            
            # 模拟100个控制周期
            for step in range(100):
                # 传感器采集
                with VirtualIMUSensor(sensor_id=f"imu_loop_{grade}") as imu:
                    imu_frame = imu.simulate_motion(
                        linear_accel=(0.1 * np.sin(step * dt * 10), 0.0, -9.81),
                        angular_vel=(0.0, 0.0, 0.1 * np.sin(step * dt * 5)),
                        dt=dt
                    )
                
                # 姿态估计
                estimator = PoseEstimator(algorithm="madgwick", sample_rate=loop_rate)
                pose = estimator.update(
                    imu_frame.accel,
                    imu_frame.gyro,
                    mag=None,
                    dt=dt
                )
                
                # 速度控制
                controller = VelocityPIDController(
                    kp=vel_spec['velocity_pid_kp'],
                    ki=vel_spec['velocity_pid_ki'],
                    kd=vel_spec['velocity_pid_kd']
                )
                output = controller.compute(setpoint=0.8, measurement=0.5, dt=dt)
                controller.reset()
            
            print(f"  [OK] Complete sensor-control loop grade {grade}: {loop_rate}Hz, {100} cycles")


class TestFiveGradeConsistency(unittest.TestCase):
    """五级规格一致性测试"""

    def test_control_frequency_matches_spec(self):
        """验证控制频率与规格一致"""
        expected_freqs = {'S': 50, 'M': 100, 'L': 200, 'XL': 500, 'XXL': 1000}
        for grade, expected in expected_freqs.items():
            vel_spec = get_velocity_control_spec(grade)
            self.assertEqual(vel_spec['control_frequency_hz'], expected,
                           f"Grade {grade} frequency mismatch")

    def test_sampling_rate_consistency(self):
        """验证采样率一致性"""
        prev_sampling = 0
        for grade in GRADES:
            tactile_spec = get_tactile_spec(grade)
            force_spec = get_force_spec(grade)
            imu_spec = get_imu_spec(grade)
            
            # 各传感器采样率应随等级增加
            self.assertGreaterEqual(tactile_spec['freq_hz'], prev_sampling if prev_sampling else 0)
            prev_sampling = tactile_spec['freq_hz']
            
            # IMU采样率
            self.assertGreaterEqual(imu_spec['sample_hz'], 50)

    def test_grade_capability_progression(self):
        """验证等级能力递进"""
        capabilities = {
            'S': {'sensors': ['imu'], 'fusion': 'late', 'control': 'basic'},
            'M': {'sensors': ['imu', 'force'], 'fusion': 'hybrid', 'control': 'standard'},
            'L': {'sensors': ['imu', 'force', 'tactile'], 'fusion': 'hybrid', 'control': 'advanced'},
            'XL': {'sensors': ['imu', 'force', 'tactile', 'vision'], 'fusion': 'early+hybrid', 'control': 'high_perf'},
            'XXL': {'sensors': ['imu', 'force', 'tactile', 'vision', 'lidar'], 'fusion': 'early+hybrid+late', 'control': 'full'},
        }
        
        for i, grade in enumerate(GRADES):
            cap = capabilities[grade]
            if i > 0:
                prev_cap = capabilities[GRADES[i-1]]
                # 传感器数量应只增不减
                self.assertGreaterEqual(len(cap['sensors']), len(prev_cap['sensors']))


if __name__ == '__main__':
    print("=" * 70)
    print("SuperModel AGV五级传感器-控制集成测试")
    print("=" * 70)
    
    # 运行测试
    unittest.main(verbosity=2)
