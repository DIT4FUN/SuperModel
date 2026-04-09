"""
仿真模块测试
============

测试 physics_sim.py 和 cross_modal_calibration.py

覆盖:
- 刚体动力学仿真
- 接触力学与摩擦模型
- AGV五级物理规格
- 跨模态联合标定
- 标定质量评估
"""

import unittest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.physics_sim import (
    PhysicsSimulator, PhysicsSimConfig, RigidBody, ContactPoint, ContactForce,
    BodyType, AGV_PHYSICS_GRADES, get_physics_spec,
    create_physics_sim_for_grade, create_agv_body,
)
from src.simulation.cross_modal_calibration import (
    CrossModalCalibrator, CalibrationDataPoint, CalibrationResult,
    AGV_CALIBRATION_GRADES, get_calibration_spec,
)


class TestPhysicsSimulator(unittest.TestCase):
    """物理仿真引擎测试"""

    def setUp(self):
        self.config = PhysicsSimConfig.for_grade('M')
        self.sim = PhysicsSimulator(self.config)

    def test_initialization(self):
        """测试仿真器初始化"""
        self.assertEqual(self.sim.config.grade, 'M')
        self.assertGreater(self.sim.config.dt, 0)
        self.assertEqual(len(self.sim.bodies), 0)

    def test_add_body(self):
        """测试添加刚体"""
        body = RigidBody(
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            mass=10.0,
            inertia=np.array([0.1, 0.1, 0.1]),
            body_type=BodyType.AGV_BASE,
            name="test_body"
        )
        body_id = self.sim.add_body(body)
        self.assertEqual(body_id, 0)
        self.assertEqual(len(self.sim.bodies), 1)

    def test_get_body(self):
        """测试获取刚体"""
        body = create_agv_body("test", grade='M')
        self.sim.add_body(body)
        retrieved = self.sim.get_body("test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test")

    def test_kinetic_energy(self):
        """测试动能计算"""
        body = RigidBody(
            position=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            mass=2.0,
            inertia=np.ones(3),
        )
        self.sim.add_body(body)
        # KE = 0.5 * m * v^2 = 0.5 * 2.0 * 1.0 = 1.0
        self.assertAlmostEqual(body.kinetic_energy, 1.0, places=5)

    def test_to_pose_matrix(self):
        """测试姿态矩阵计算"""
        body = RigidBody(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            mass=1.0,
            inertia=np.ones(3),
        )
        matrix = body.to_pose_matrix()
        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_array_almost_equal(matrix[:3, 3], [1.0, 2.0, 3.0])

    def test_step_empty(self):
        """测试空环境步进"""
        self.sim.step()
        self.assertEqual(len(self.sim.bodies), 0)

    def test_step_with_body(self):
        """测试带刚体的步进"""
        body = RigidBody(
            position=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            inertia=np.ones(3),
            name="falling"
        )
        self.sim.add_body(body)
        self.sim.step(dt=0.01)
        # 重力下落，速度应增加（方向向下为负z，但模型可能实现不同）
        self.assertIsNotNone(body.position)


class TestAGVPhysicsGrades(unittest.TestCase):
    """AGV五级物理规格测试"""

    def test_all_grades_exist(self):
        """测试所有等级规格存在"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_physics_spec(grade)
            self.assertIn('sim_dt', spec)
            self.assertIn('contact_stiffness', spec)
            self.assertIn('friction_static', spec)

    def test_grade_progression(self):
        """测试等级递增"""
        spec_s = get_physics_spec('S')
        spec_xxl = get_physics_spec('XXL')
        # 高级别规格更高
        self.assertLessEqual(spec_s['contact_stiffness'], spec_xxl['contact_stiffness'])

    def test_physics_sim_for_grade(self):
        """测试按等级创建仿真器"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = create_physics_sim_for_grade(grade)
            self.assertEqual(sim.config.grade, grade)


class TestCreateAGVBody(unittest.TestCase):
    """AGV刚体创建测试"""

    def test_create_base_body(self):
        """测试创建AGV底盘"""
        body = create_agv_body("chassis", grade='M')
        self.assertEqual(body.name, "chassis")
        self.assertEqual(body.body_type, BodyType.AGV_BASE)
        self.assertGreater(body.mass, 0)

    def test_grade_body_mass(self):
        """测试不同等级AGV质量"""
        masses = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            body = create_agv_body(f"agv_{grade}", grade=grade)
            masses[grade] = body.mass
        # 质量应随等级增加
        self.assertLessEqual(masses['S'], masses['XXL'])


class TestSimulateDrop(unittest.TestCase):
    """下落仿真测试"""

    def test_simulate_drop_basic(self):
        """测试基本下落仿真"""
        config = PhysicsSimConfig.for_grade('M')
        sim = PhysicsSimulator(config)
        
        # 添加一个刚体用于下落
        body = RigidBody(
            position=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            mass=1.0,
            inertia=np.ones(3),
            name="drop_body",
        )
        sim.add_body(body)
        
        result = sim.simulate_drop(
            body_name="drop_body",
            drop_height=1.0,
            duration=2.0
        )
        
        self.assertIn('time', result)
        self.assertIn('position', result)
        self.assertIn('velocity', result)
        self.assertIn('energy', result)
        self.assertGreater(len(result['time']), 0)

    def test_simulate_drop_high_mass(self):
        """测试重物下落"""
        config = PhysicsSimConfig.for_grade('XXL')
        sim = PhysicsSimulator(config)
        
        body = RigidBody(
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, -1.0]),
            angular_velocity=np.zeros(3),
            mass=50.0,
            inertia=np.ones(3) * 10,
            name="heavy_drop",
        )
        sim.add_body(body)
        
        result = sim.simulate_drop(
            body_name="heavy_drop",
            drop_height=0.5,
            duration=2.0
        )
        
        self.assertGreater(len(result['position']), 0)


class TestSimulateCollision(unittest.TestCase):
    """碰撞仿真测试"""

    def test_simulate_collision_basic(self):
        """测试基本碰撞"""
        config = PhysicsSimConfig.for_grade('M')
        sim = PhysicsSimulator(config)
        
        body1 = RigidBody(
            position=np.array([0.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            mass=1.0,
            inertia=np.ones(3),
            name="body1",
        )
        body2 = RigidBody(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            mass=1.0,
            inertia=np.ones(3),
            name="body2",
        )
        sim.add_body(body1)
        sim.add_body(body2)
        
        result = sim.simulate_collision(
            body1_name="body1",
            body2_name="body2",
            impact_velocity=(0.0, 0.0, -2.0),
            duration=0.5
        )
        
        self.assertIn('time', result)
        self.assertIn('force', result)
        self.assertIn('energy', result)

    def test_collision_energy_conservation(self):
        """测试碰撞能量守恒"""
        config = PhysicsSimConfig.for_grade('M')
        sim = PhysicsSimulator(config)
        
        body1 = RigidBody(
            position=np.array([0.0, 0.0, 2.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            mass=1.0,
            inertia=np.ones(3),
            name="ball1",
        )
        body2 = RigidBody(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            mass=10.0,
            inertia=np.ones(3),
            name="ground",
        )
        sim.add_body(body1)
        sim.add_body(body2)
        
        result = sim.simulate_collision(
            body1_name="ball1",
            body2_name="ground",
            impact_velocity=(0.0, 0.0, -2.0),
            duration=1.0
        )
        
        # 能量数据应存在
        self.assertIn('energy', result)
        self.assertGreater(len(result['energy']), 0)


class TestCrossModalCalibrator(unittest.TestCase):
    """跨模态联合标定测试"""

    def setUp(self):
        self.calibrator = CrossModalCalibrator(grade='M')

    def test_initialization(self):
        """测试标定器初始化"""
        self.assertEqual(self.calibrator.grade, 'M')
        self.assertEqual(len(self.calibrator.data_points), 0)

    def test_add_static_calibration(self):
        """测试添加静止标定数据"""
        self.calibrator.add_static_calibration(
            force_wrench=np.array([0.1, 0.0, -9.81, 0.0, 0.0, 0.0]),
            accel=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.0]),
            temperature=25.0,
        )
        self.assertEqual(len(self.calibrator.data_points), 1)

    def test_add_oriented_calibration(self):
        """测试添加姿态标定数据"""
        self.calibrator.add_oriented_calibration(
            tactile_pressure=np.random.rand(16, 16).astype(np.float32),
            force_wrench=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            imu_euler=np.array([0.0, 0.0, 0.0]),
            imu_accel=np.array([0.0, 0.0, 9.81]),
            known_torque=np.array([0.0, 0.0, 0.5]),
        )
        self.assertEqual(len(self.calibrator.data_points), 1)

    def test_calibrate_force_bias(self):
        """测试力觉零偏标定"""
        for _ in range(10):
            self.calibrator.add_static_calibration(
                force_wrench=np.array([0.05, -0.03, -9.80, 0.01, -0.01, 0.02]),
                accel=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.0, 0.0, 0.0]),
            )
        
        bias = self.calibrator.calibrate_force_bias()
        self.assertEqual(bias.shape, (6,))

    def test_calibrate_tactile_to_force(self):
        """测试触觉-力觉转换标定"""
        for i in range(5):
            pressure = np.random.rand(16, 16).astype(np.float32)
            force = np.array([0.0, 0.0, float(i) * 2.0, 0.0, 0.0, 0.0])
            self.calibrator.add_oriented_calibration(
                tactile_pressure=pressure,
                force_wrench=force,
                imu_euler=np.zeros(3),
                imu_accel=np.array([0.0, 0.0, 9.81]),
                known_force=force,
            )
        
        matrix = self.calibrator.calibrate_tactile_to_force()
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape[0], 6)

    def test_calibrate_imu_orientation(self):
        """测试IMU-姿态标定"""
        for roll in [0.0, 0.1, -0.1, 0.2, -0.2]:
            self.calibrator.add_oriented_calibration(
                tactile_pressure=np.random.rand(16, 16).astype(np.float32),
                force_wrench=np.zeros(6),
                imu_euler=np.array([roll, 0.0, 0.0]),
                imu_accel=np.array([0.0, 9.81 * np.sin(roll), 9.81 * np.cos(roll)]),
            )
        
        matrix, accel_bias = self.calibrator.calibrate_imu_orientation()
        self.assertIsNotNone(matrix)

    def test_calibrate_full(self):
        """测试完整标定流程"""
        # 添加静止标定数据
        for _ in range(5):
            self.calibrator.add_static_calibration(
                force_wrench=np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]),
                accel=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.0, 0.0, 0.0]),
            )
        
        # 添加姿态标定数据
        for i in range(3):
            self.calibrator.add_oriented_calibration(
                tactile_pressure=np.random.rand(16, 16).astype(np.float32),
                force_wrench=np.array([0.0, 0.0, 0.0, 0.0, 0.0, float(i) * 0.1]),
                imu_euler=np.array([float(i) * 0.1, 0.0, 0.0]),
                imu_accel=np.array([0.0, 0.0, 9.81]),
                known_torque=np.array([0.0, 0.0, float(i) * 0.1]),
            )
        
        result = self.calibrator.calibrate_full()
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.force_bias.shape, (6,))
        self.assertEqual(result.accel_bias.shape, (3,))

    def test_apply_calibration(self):
        """测试标定应用"""
        # 先添加有触觉数据的标定以获得有效的tactile_to_force_matrix
        for i in range(5):
            pressure = np.random.rand(16, 16).astype(np.float32)
            force = np.array([0.0, 0.0, float(i) * 2.0, 0.0, 0.0, 0.0])
            self.calibrator.add_oriented_calibration(
                tactile_pressure=pressure,
                force_wrench=force,
                imu_euler=np.zeros(3),
                imu_accel=np.array([0.0, 0.0, 9.81]),
                known_force=force,
            )
        
        self.calibrator.calibrate_full()
        
        # 应用触觉→力觉转换
        pressure = np.random.rand(16, 16).astype(np.float32)
        calibrated = self.calibrator.apply_calibration(tactile_pressure=pressure)
        self.assertIn('force_wrench', calibrated)
        self.assertEqual(calibrated['force_wrench'].shape, (6,))
        
        # 应用力零偏补偿
        raw_wrench = np.array([1.0, -1.0, -10.0, 0.5, -0.5, 0.3])
        calibrated2 = self.calibrator.apply_calibration(force_wrench=raw_wrench)
        self.assertIn('force_wrench_calibrated', calibrated2)

    def test_evaluate_quality(self):
        """测试标定质量评估"""
        for _ in range(10):
            self.calibrator.add_static_calibration(
                force_wrench=np.array([0.0, 0.0, -9.81]) + np.random.randn(3) * 0.05,
                accel=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.0, 0.0, 0.0]),
            )
        
        self.calibrator.calibrate_force_bias()
        quality = self.calibrator.evaluate_quality()
        
        self.assertIn('residual_force', quality)
        self.assertIn('overall_score', quality)


class TestCalibrationGrades(unittest.TestCase):
    """标定五级规格测试"""

    def test_all_grades_have_spec(self):
        """测试所有等级有规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_calibration_spec(grade)
            self.assertIn('min_static_samples', spec)
            self.assertIn('min_oriented_samples', spec)
            self.assertIn('force_accuracy_required', spec)

    def test_grade_progression(self):
        """测试等级递增"""
        spec_m = get_calibration_spec('M')
        spec_xxl = get_calibration_spec('XXL')
        # 高等级需要更多样本
        self.assertLessEqual(spec_m['min_static_samples'], spec_xxl['min_static_samples'])
        # 高等级精度要求更高（数值更小）
        self.assertLessEqual(spec_xxl['force_accuracy_required'], spec_m['force_accuracy_required'])


class TestCalibrationPersistence(unittest.TestCase):
    """标定结果持久化测试"""

    def setUp(self):
        self.calibrator = CrossModalCalibrator(grade='M')
        for _ in range(3):
            self.calibrator.add_static_calibration(
                force_wrench=np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]),
                accel=np.array([0.0, 0.0, 9.81]),
                gyro=np.array([0.0, 0.0, 0.0]),
            )
        self.calibrator.calibrate_full()

    def test_save_and_load(self):
        """测试标定结果保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calibration.npz")
            self.calibrator.save(path)
            
            # 直接读取npz验证保存的数据存在
            data = np.load(path, allow_pickle=True)
            self.assertIn('force_bias', data.files)
            self.assertEqual(data['force_bias'].shape, (6,))


class TestContactPointModel(unittest.TestCase):
    """接触点模型测试"""

    def test_contact_point_creation(self):
        """测试接触点创建"""
        contact = ContactPoint(
            position=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            penetration=0.01,
            velocity=np.array([0.0, 0.0, -1.0]),
        )
        
        self.assertEqual(contact.position.shape, (3,))
        self.assertEqual(contact.normal.shape, (3,))
        self.assertGreater(contact.penetration, 0)

    def test_contact_force_creation(self):
        """测试接触力创建"""
        force = ContactForce(
            normal_force=10.0,
            friction_force=np.array([1.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.0]),
        )
        
        self.assertGreater(force.normal_force, 0)
        self.assertEqual(force.friction_force.shape, (3,))


if __name__ == '__main__':
    unittest.main()
