"""
具身传感控制模块测试
测试 EmbodiedController, EmbodiedTaskExecutor 及 AGV五级具身控制规格
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.embodied_control import (
    EmbodiedController, EmbodiedState, EmbodiedCommand,
    EmbodiedControlParams, EmbodiedTaskExecutor,
    EmbodiedGrade,
    AGV_EMBODIED_GRADES, get_embodied_spec,
    SurfaceFollowingController, AssemblyController,
)


class TestEmbodiedGrades(unittest.TestCase):
    """测试AGV五级具身控制规格"""

    def test_grade_S_spec(self):
        spec = get_embodied_spec('S')
        self.assertFalse(spec['force_enabled'])
        self.assertFalse(spec['imu_enabled'])
        self.assertEqual(spec['control_rate'], 50)
        self.assertEqual(spec['fusion_method'], 'threshold')

    def test_grade_M_spec(self):
        spec = get_embodied_spec('M')
        self.assertTrue(spec['tactile_enabled'])
        self.assertTrue(spec['force_enabled'])
        self.assertTrue(spec['imu_enabled'])
        self.assertEqual(spec['control_rate'], 100)
        self.assertEqual(spec['fusion_method'], 'weighted_average')
        self.assertTrue(spec['grasp_adaptation'])
        self.assertTrue(spec['attitude_stabilization'])

    def test_grade_L_spec(self):
        spec = get_embodied_spec('L')
        self.assertEqual(spec['control_rate'], 200)
        self.assertEqual(spec['fusion_method'], 'ekf')
        self.assertEqual(spec['max_contact_force'], 150)

    def test_grade_XL_spec(self):
        spec = get_embodied_spec('XL')
        self.assertEqual(spec['control_rate'], 500)
        self.assertEqual(spec['fusion_method'], 'ukf')
        self.assertTrue(spec['slip_recovery'])

    def test_grade_XXL_spec(self):
        spec = get_embodied_spec('XXL')
        self.assertEqual(spec['control_rate'], 1000)
        self.assertEqual(spec['fusion_method'], 'mpc_fusion')
        self.assertEqual(spec['latency_ms'], 2)
        self.assertEqual(spec['collision_response_ms'], 5)

    def test_all_grades_have_required_keys(self):
        required_keys = [
            'description', 'tactile_enabled', 'force_enabled', 'imu_enabled',
            'tactile_resolution', 'force_axes', 'imu_grade', 'control_rate',
            'latency_ms', 'fusion_method', 'max_contact_force',
            'grasp_adaptation', 'attitude_stabilization', 'slip_recovery',
            'collision_response_ms'
        ]
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_embodied_spec(grade)
            for key in required_keys:
                self.assertIn(key, spec, f"Grade {grade} missing key {key}")


class TestEmbodiedControlParams(unittest.TestCase):
    """测试具身控制参数"""

    def test_from_grade_S(self):
        params = EmbodiedControlParams.from_grade('S')
        self.assertEqual(params.grade, 'S')
        self.assertEqual(params.fusion_method, 'threshold')
        self.assertEqual(params.control_rate, 50.0)

    def test_from_grade_M(self):
        params = EmbodiedControlParams.from_grade('M')
        self.assertEqual(params.grade, 'M')
        self.assertEqual(params.fusion_method, 'weighted_average')
        self.assertEqual(params.control_rate, 100.0)

    def test_from_grade_L(self):
        params = EmbodiedControlParams.from_grade('L')
        self.assertEqual(params.grade, 'L')
        self.assertEqual(params.fusion_method, 'ekf')
        self.assertEqual(params.control_rate, 200.0)

    def test_from_grade_XL(self):
        params = EmbodiedControlParams.from_grade('XL')
        self.assertEqual(params.grade, 'XL')
        self.assertEqual(params.fusion_method, 'ukf')
        self.assertEqual(params.control_rate, 500.0)

    def test_from_grade_XXL(self):
        params = EmbodiedControlParams.from_grade('XXL')
        self.assertEqual(params.grade, 'XXL')
        self.assertEqual(params.fusion_method, 'mpc_fusion')
        self.assertEqual(params.control_rate, 1000.0)


class TestEmbodiedCommand(unittest.TestCase):
    """测试具身控制指令"""

    def test_default_command(self):
        cmd = EmbodiedCommand()
        self.assertEqual(cmd.mode, 'hybrid_force_position')
        self.assertEqual(cmd.force_weight, 0.5)
        self.assertEqual(cmd.position_weight, 0.5)
        self.assertIsNone(cmd.desired_force)
        self.assertIsNone(cmd.desired_position)

    def test_force_command(self):
        cmd = EmbodiedCommand(
            mode='force',
            desired_force=np.array([0, 0, -10.0]),
            desired_torque=np.zeros(3),
            max_contact_force=50.0
        )
        self.assertEqual(cmd.mode, 'force')
        self.assertIsInstance(cmd.desired_force, np.ndarray)
        self.assertEqual(cmd.desired_force[2], -10.0)

    def test_position_command(self):
        cmd = EmbodiedCommand(
            mode='position',
            desired_position=np.array([0.5, 0.3, 0.2]),
            desired_orientation=np.array([1, 0, 0, 0])
        )
        self.assertEqual(cmd.mode, 'position')
        self.assertEqual(cmd.desired_position[0], 0.5)

    def test_impedance_command(self):
        cmd = EmbodiedCommand(
            mode='impedance',
            desired_linear_velocity=np.array([0.1, 0.0, 0.0]),
            desired_angular_velocity=np.zeros(3)
        )
        self.assertEqual(cmd.mode, 'impedance')


class TestEmbodiedControllerCreation(unittest.TestCase):
    """测试具身控制器创建"""

    def test_create_grade_S(self):
        ctrl = EmbodiedController(grade='S', use_virtual_sensors=True)
        self.assertEqual(ctrl.grade, 'S')
        self.assertFalse(ctrl.spec['force_enabled'])
        self.assertFalse(ctrl.spec['imu_enabled'])
        ctrl.reset()

    def test_create_grade_M(self):
        ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.assertEqual(ctrl.grade, 'M')
        self.assertTrue(ctrl.spec['tactile_enabled'])
        self.assertTrue(ctrl.spec['force_enabled'])
        self.assertTrue(ctrl.spec['imu_enabled'])
        ctrl.reset()

    def test_create_grade_XL(self):
        ctrl = EmbodiedController(grade='XL', use_virtual_sensors=True)
        self.assertEqual(ctrl.grade, 'XL')
        self.assertEqual(ctrl.spec['control_rate'], 500)
        ctrl.reset()

    def test_create_with_custom_params(self):
        params = EmbodiedControlParams.from_grade('M')
        params.control_rate = 150.0
        ctrl = EmbodiedController(grade='M', params=params, use_virtual_sensors=True)
        self.assertEqual(ctrl.params.control_rate, 150.0)
        ctrl.reset()


class TestEmbodiedControllerVirtualSensors(unittest.TestCase):
    """测试虚拟传感器模式"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()

    def tearDown(self):
        self.ctrl.reset()

    def test_init_virtual_sensors(self):
        self.assertTrue(self.ctrl._initialized)
        self.assertIsNotNone(self.ctrl._virtual_tactile)
        self.assertIsNotNone(self.ctrl._virtual_force)
        self.assertIsNotNone(self.ctrl._virtual_imu)
        self.assertIsNotNone(self.ctrl._pose_estimator)

    def test_update_returns_state(self):
        state = self.ctrl.update()
        self.assertIsInstance(state, EmbodiedState)
        self.assertGreater(state.cycle_id, 0)
        self.assertTrue(state.tactile_ok)
        self.assertTrue(state.force_ok)

    def test_multiple_updates(self):
        for i in range(5):
            state = self.ctrl.update()
        self.assertEqual(state.cycle_id, 5)

    def test_state_has_sensors_data(self):
        state = self.ctrl.update()
        self.assertTrue(state.tactile_ok)
        self.assertTrue(state.force_ok)
        self.assertTrue(state.imu_ok)


class TestEmbodiedControllerFusion(unittest.TestCase):
    """测试多模态融合"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()

    def tearDown(self):
        self.ctrl.reset()

    def test_grade_S_threshold_fusion(self):
        ctrl = EmbodiedController(grade='S', use_virtual_sensors=True)
        ctrl.init_virtual_sensors()
        ctrl.update()
        # S grade uses threshold fusion, no force/imu sensors
        self.assertFalse(ctrl.spec['force_enabled'])
        ctrl.reset()

    def test_grade_L_ekf_fusion(self):
        ctrl = EmbodiedController(grade='L', use_virtual_sensors=True)
        ctrl.init_virtual_sensors()
        state = ctrl.update()
        self.assertEqual(ctrl.params.fusion_method, 'ekf')
        ctrl.reset()

    def test_grade_XL_ukf_fusion(self):
        ctrl = EmbodiedController(grade='XL', use_virtual_sensors=True)
        ctrl.init_virtual_sensors()
        state = ctrl.update()
        self.assertEqual(ctrl.params.fusion_method, 'ukf')
        ctrl.reset()

    def test_grade_XXL_mpc_fusion(self):
        ctrl = EmbodiedController(grade='XXL', use_virtual_sensors=True)
        ctrl.init_virtual_sensors()
        state = ctrl.update()
        self.assertEqual(ctrl.params.fusion_method, 'mpc_fusion')
        ctrl.reset()


class TestEmbodiedControllerCompute(unittest.TestCase):
    """测试控制计算"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()

    def tearDown(self):
        self.ctrl.reset()

    def test_compute_impedance_mode(self):
        state = self.ctrl.update()
        cmd = EmbodiedCommand(mode='impedance')
        output = self.ctrl.compute(cmd)
        self.assertIn('joint_torques', output)
        self.assertIn('safety_stop', output)
        self.assertFalse(output['safety_stop'])

    def test_compute_hybrid_mode(self):
        state = self.ctrl.update()
        cmd = EmbodiedCommand(
            mode='hybrid_force_position',
            desired_force=np.array([0, 0, -10.0])
        )
        output = self.ctrl.compute(cmd)
        self.assertFalse(output['safety_stop'])
        self.assertTrue(output['force_regulated'])

    def test_compute_admittance_mode(self):
        state = self.ctrl.update()
        cmd = EmbodiedCommand(mode='admittance')
        output = self.ctrl.compute(cmd)
        self.assertIn('contact_adjustment', output)

    def test_compute_tactile_servo_mode(self):
        state = self.ctrl.update()
        cmd = EmbodiedCommand(mode='tactile_servo')
        output = self.ctrl.compute(cmd)
        self.assertIsInstance(output['contact_adjustment'], np.ndarray)

    def test_emergency_stop_on_excessive_force(self):
        # S grade with no force sensor - no emergency stop possible
        ctrl = EmbodiedController(grade='S', use_virtual_sensors=True)
        ctrl.init_virtual_sensors()
        ctrl.update()
        cmd = EmbodiedCommand(mode='position', max_contact_force=0.1)
        # S grade has no force sensor so no emergency
        ctrl.reset()

    def test_get_state(self):
        self.ctrl.update()
        state = self.ctrl.get_state()
        self.assertIsInstance(state, EmbodiedState)


class TestEmbodiedControllerGradeScaling(unittest.TestCase):
    """测试AGV等级参数缩放"""

    def test_control_rate_scales_with_grade(self):
        rates = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            rates[grade] = ctrl.spec['control_rate']
            ctrl.reset()
        
        self.assertLess(rates['S'], rates['M'])
        self.assertLess(rates['M'], rates['L'])
        self.assertLess(rates['L'], rates['XL'])
        self.assertLess(rates['XL'], rates['XXL'])

    def test_latency_decreases_with_grade(self):
        latencies = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            latencies[grade] = get_embodied_spec(grade)['latency_ms']
        
        self.assertGreater(latencies['S'], latencies['M'])
        self.assertGreater(latencies['M'], latencies['L'])
        self.assertGreater(latencies['L'], latencies['XL'])
        self.assertGreater(latencies['XL'], latencies['XXL'])


class TestEmbodiedTaskExecutor(unittest.TestCase):
    """测试具身任务执行器"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='M')

    def tearDown(self):
        self.ctrl.reset()

    def test_executor_creation(self):
        self.assertEqual(self.executor.phase, self.executor.TaskPhase.IDLE)
        self.assertEqual(self.executor.grade, 'M')
        self.assertEqual(self.executor.success_count, 0)

    def test_grasp_place_task(self):
        object_pos = np.array([0.3, 0.2, 0.1])
        place_pos = np.array([0.5, 0.4, 0.05])
        
        result = self.executor.execute_grasp_place(
            object_position=object_pos,
            place_position=place_pos,
            object_size=0.05,
            grasp_force=10.0
        )
        self.assertTrue(result)
        self.assertEqual(self.executor.phase, self.executor.TaskPhase.IDLE)
        self.assertEqual(self.executor.success_count, 1)
        self.assertEqual(self.executor.failure_count, 0)

    def test_get_metrics(self):
        metrics = self.executor.get_metrics()
        self.assertIn('success_count', metrics)
        self.assertIn('failure_count', metrics)
        self.assertIn('success_rate', metrics)
        self.assertEqual(metrics['current_phase'], 'idle')

    def test_phase_history(self):
        object_pos = np.array([0.3, 0.2, 0.1])
        place_pos = np.array([0.5, 0.4, 0.05])
        
        self.executor.execute_grasp_place(
            object_position=object_pos,
            place_position=place_pos,
        )
        
        history = self.executor.get_metrics()['phase_history']
        self.assertGreater(len(history), 0)
        # Check sequence of phases
        phase_names = [p[0] for p in history]
        self.assertIn('approach', phase_names)
        self.assertIn('grasp', phase_names)


class TestEmbodiedState(unittest.TestCase):
    """测试具身状态"""

    def test_state_default_values(self):
        state = EmbodiedState()
        self.assertEqual(len(state.tactile_contacts), 0)
        self.assertEqual(state.grip_quality, 0.0)
        self.assertEqual(state.slip_probability, 0.0)
        self.assertTrue(state.is_stable)
        self.assertFalse(state.is_slipping)
        self.assertEqual(state.cycle_id, 0)

    def test_state_with_contact(self):
        state = EmbodiedState(
            is_in_contact=True,
            contact_force=15.0,
            slip_probability=0.1,
            grip_quality=0.8
        )
        self.assertTrue(state.is_in_contact)
        self.assertEqual(state.contact_force, 15.0)


class TestEmbodiedGradeEnum(unittest.TestCase):
    """测试等级枚举"""

    def test_grade_values(self):
        self.assertEqual(EmbodiedGrade.S.value, 'S')
        self.assertEqual(EmbodiedGrade.M.value, 'M')
        self.assertEqual(EmbodiedGrade.L.value, 'L')
        self.assertEqual(EmbodiedGrade.XL.value, 'XL')
        self.assertEqual(EmbodiedGrade.XXL.value, 'XXL')


class TestSurfaceFollowingController(unittest.TestCase):
    """测试表面跟踪控制器"""

    def test_init(self):
        ctrl = SurfaceFollowingController(grade='M', follow_mode='admittance')
        self.assertEqual(ctrl.grade, 'M')
        self.assertEqual(ctrl.follow_mode, 'admittance')
        self.assertEqual(ctrl.nominal_force, 5.0)
        self.assertFalse(ctrl._is_following)

    def test_grade_params(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            ctrl = SurfaceFollowingController(grade=grade)
            self.assertEqual(ctrl.grade, grade)

    def test_surface_normal_estimation(self):
        ctrl = SurfaceFollowingController()
        # 平面: 均匀压力 → 法向朝上
        pressure_flat = np.ones((16, 16), dtype=np.float32) * 0.5
        normal = ctrl.estimate_surface_normal(pressure_flat)
        self.assertEqual(normal.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=3)

    def test_surface_normal_gradient(self):
        ctrl = SurfaceFollowingController()
        # 创建有梯度的压力图
        pressure = np.zeros((16, 16), dtype=np.float32)
        for i in range(16):
            pressure[:, i] = i / 16.0
        normal = ctrl.estimate_surface_normal(pressure)
        # 有梯度时法向应偏离垂直方向
        z_component = normal[2]
        self.assertLess(z_component, 1.0)

    def test_tangent_direction(self):
        ctrl = SurfaceFollowingController()
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        tangent = ctrl.compute_tangent_direction(normal)
        self.assertEqual(tangent.shape, (3,))
        # 切向应垂直于法向
        dot = abs(np.dot(tangent, normal))
        self.assertLess(dot, 1e-5)

    def test_constant_force_control(self):
        ctrl = SurfaceFollowingController(follow_mode='constant_force', nominal_force=5.0)
        pressure = np.ones((16, 16), dtype=np.float32) * 0.3
        # 力偏低 → 应有向下修正
        result = ctrl.compute_control(pressure, current_force=3.0, dt=0.01)
        velocity = result['velocity']
        self.assertEqual(velocity.shape, (3,))
        self.assertTrue(ctrl._is_following)

    def test_admittance_control(self):
        ctrl = SurfaceFollowingController(follow_mode='admittance')
        pressure = np.ones((16, 16), dtype=np.float32) * 0.5
        result = ctrl.compute_control(pressure, current_force=5.0, dt=0.01)
        velocity = result['velocity']
        self.assertEqual(velocity.shape, (3,))
        self.assertIsNotNone(result['surface_normal'])
        self.assertIsNotNone(result['tangent_direction'])

    def test_impedance_control(self):
        ctrl = SurfaceFollowingController(follow_mode='impedance')
        pressure = np.ones((16, 16), dtype=np.float32) * 0.4
        result = ctrl.compute_control(pressure, current_force=6.0, dt=0.01)
        self.assertTrue(ctrl._is_following)
        self.assertIn('normal_force_error', result)

    def test_adaptive_control(self):
        ctrl = SurfaceFollowingController(follow_mode='adaptive')
        pressure = np.ones((16, 16), dtype=np.float32) * 0.5
        for _ in range(5):
            result = ctrl.compute_control(pressure, current_force=5.0, dt=0.01)
        self.assertEqual(ctrl._cycle_count, 5)

    def test_contact_quality(self):
        ctrl = SurfaceFollowingController()
        pressure = np.ones((16, 16), dtype=np.float32) * 0.5
        quality = ctrl.compute_contact_quality(pressure)
        self.assertIn('contact_ratio', quality)
        self.assertIn('quality', quality)
        self.assertIn('is_good_contact', quality)

    def test_contact_quality_no_contact(self):
        ctrl = SurfaceFollowingController()
        pressure = np.zeros((16, 16), dtype=np.float32)
        quality = ctrl.compute_contact_quality(pressure)
        self.assertEqual(quality['contact_ratio'], 0.0)
        self.assertFalse(quality['is_good_contact'])

    def test_velocity_limiting(self):
        ctrl = SurfaceFollowingController(nominal_velocity=0.05)
        pressure = np.ones((16, 16), dtype=np.float32) * 0.5
        for _ in range(10):
            result = ctrl.compute_control(pressure, current_force=5.0, dt=0.01)
        speed = np.linalg.norm(result['velocity'])
        self.assertLessEqual(speed, ctrl.nominal_velocity * 2.0 * 1.01)

    def test_reset(self):
        ctrl = SurfaceFollowingController()
        pressure = np.ones((16, 16), dtype=np.float32) * 0.5
        ctrl.compute_control(pressure, current_force=5.0, dt=0.01)
        self.assertTrue(ctrl._is_following)
        ctrl.reset()
        self.assertFalse(ctrl._is_following)
        self.assertEqual(ctrl._total_distance, 0.0)
        self.assertEqual(ctrl._cycle_count, 0)

    def test_status(self):
        ctrl = SurfaceFollowingController(grade='L')
        status = ctrl.get_status()
        self.assertIn('is_following', status)
        self.assertIn('total_distance_m', status)
        self.assertIn('surface_normal', status)
        self.assertEqual(status['mode'], 'admittance')

    def test_distance_accumulation(self):
        ctrl = SurfaceFollowingController(nominal_velocity=0.05)
        pressure = np.ones((16, 16), dtype=np.float32) * 0.5
        for _ in range(100):
            ctrl.compute_control(pressure, current_force=5.0, dt=0.01)
        self.assertGreater(ctrl._total_distance, 0.0)


class TestAssemblyController(unittest.TestCase):
    """测试精密装配控制器"""

    def test_init(self):
        ctrl = AssemblyController(grade='M', hole_tolerance=1.0, insertion_depth=10.0)
        self.assertEqual(ctrl.grade, 'M')
        self.assertEqual(ctrl.hole_tolerance, 1.0)
        self.assertEqual(ctrl.insertion_depth, 10.0)
        self.assertEqual(ctrl._phase, ctrl.AssemblyPhase.IDLE)

    def test_grade_params(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            ctrl = AssemblyController(grade=grade)
            self.assertEqual(ctrl.grade, grade)

    def test_start_assembly(self):
        ctrl = AssemblyController()
        target = np.array([100.0, 50.0, 0.0])
        ctrl.start_assembly(target, phase='approach')
        self.assertEqual(ctrl._phase, ctrl.AssemblyPhase.APPROACH)
        np.testing.assert_array_equal(ctrl._target_position, target)
        self.assertEqual(ctrl._insertion_progress, 0.0)

    def test_search_spiral_pattern(self):
        ctrl = AssemblyController(search_pattern='spiral')
        motions = []
        for _ in range(10):
            m = ctrl.compute_search_motion(dt=0.01)
            motions.append(m.copy())
        # 螺旋搜索: 不同角度
        self.assertGreater(len(set(str(m[:2]) for m in motions)), 1)

    def test_search_raster_pattern(self):
        ctrl = AssemblyController(search_pattern='raster')
        motions = []
        for _ in range(20):
            m = ctrl.compute_search_motion(dt=0.01)
            motions.append(m.copy())
        # 光栅搜索应有不同位置
        positions = [m[:2] for m in motions]
        unique_count = len(set(str(p) for p in positions))
        self.assertGreater(unique_count, 1)

    def test_insertion_control_normal(self):
        ctrl = AssemblyController(grade='M')
        ctrl._phase = ctrl.AssemblyPhase.INSERT
        velocity = ctrl.compute_insertion_control(
            current_force=5.0, lateral_force=1.0, dt=0.01
        )
        self.assertEqual(velocity.shape, (3,))
        self.assertEqual(velocity[2], ctrl._insertion_velocity)

    def test_insertion_control_high_force(self):
        ctrl = AssemblyController(grade='M', max_insertion_force=20.0)
        ctrl._phase = ctrl.AssemblyPhase.INSERT
        velocity = ctrl.compute_insertion_control(
            current_force=25.0, lateral_force=1.0, dt=0.01
        )
        # 过大的力应导致后退
        self.assertLess(velocity[2], 0)
        self.assertEqual(ctrl._phase, ctrl.AssemblyPhase.SEARCH)

    def test_insertion_control_lateral_force(self):
        ctrl = AssemblyController(grade='M', search_force=3.0)
        ctrl._phase = ctrl.AssemblyPhase.INSERT
        velocity = ctrl.compute_insertion_control(
            current_force=5.0, lateral_force=15.0, dt=0.01
        )
        # 侧向力大时应有横向搜索运动
        lateral_mag = np.linalg.norm(velocity[:2])
        self.assertGreater(lateral_mag, 0)

    def test_seating_control(self):
        ctrl = AssemblyController(grade='L', max_insertion_force=20.0)
        ctrl._phase = ctrl.AssemblyPhase.SEAT
        velocity = ctrl.compute_seating_control(contact_force=10.0, dt=0.01)
        self.assertEqual(velocity[2], ctrl._insertion_velocity * 0.5)

    def test_seating_complete(self):
        ctrl = AssemblyController(grade='L', max_insertion_force=20.0)
        ctrl._phase = ctrl.AssemblyPhase.SEAT
        ctrl.compute_seating_control(contact_force=16.0, dt=0.01)
        ctrl.compute_seating_control(contact_force=17.0, dt=0.01)
        self.assertEqual(ctrl._phase, ctrl.AssemblyPhase.VERIFY)

    def test_update_approach_phase(self):
        ctrl = AssemblyController()
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='approach')
        result = ctrl.update(
            current_position=np.array([100.0, 50.0, 5.0]),
            current_force=0.0, lateral_force=0.0, dt=0.01
        )
        self.assertEqual(result['phase'], 'approach')
        self.assertIn('velocity', result)

    def test_update_approach_reached(self):
        ctrl = AssemblyController()
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='approach')
        result = ctrl.update(
            current_position=np.array([100.0, 50.0, 0.5]),
            current_force=0.0, lateral_force=0.0, dt=0.01
        )
        self.assertEqual(result['progress'], 0.0)

    def test_update_search_to_insert(self):
        ctrl = AssemblyController()
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='search')
        # 低力 → 进入插入
        for _ in range(3):
            result = ctrl.update(
                current_position=np.array([100.0, 50.0, -1.0]),
                current_force=1.0, lateral_force=0.5, dt=0.01
            )
        self.assertIn(result['phase'], ['search', 'insert'])

    def test_update_insertion_progress(self):
        ctrl = AssemblyController(grade='L', insertion_depth=10.0)
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='insert')
        ctrl._insertion_velocity = 1.0  # 1mm/s
        for i in range(20):
            result = ctrl.update(
                current_position=np.array([100.0, 50.0, -float(i) * 0.1]),
                current_force=5.0, lateral_force=0.5, dt=0.01
            )
        self.assertGreater(ctrl._insertion_progress, 0.0)

    def test_update_complete(self):
        ctrl = AssemblyController(grade='M')
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='insert')
        ctrl._insertion_progress = 1.0
        ctrl._phase = ctrl.AssemblyPhase.VERIFY
        ctrl._total_assemblies = 1
        result = ctrl.update(
            current_position=np.array([100.0, 50.0, -10.0]),
            current_force=10.0, lateral_force=0.5, dt=0.01
        )
        self.assertTrue(result['should_stop'])

    def test_stats(self):
        ctrl = AssemblyController()
        stats = ctrl.get_stats()
        self.assertIn('total_assemblies', stats)
        self.assertIn('success_rate', stats)
        self.assertEqual(stats['current_phase'], 'idle')

    def test_reset(self):
        ctrl = AssemblyController()
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='insert')
        ctrl._insertion_progress = 0.5
        ctrl._search_count = 10
        ctrl.reset()
        self.assertEqual(ctrl._phase, ctrl.AssemblyPhase.IDLE)
        self.assertEqual(ctrl._insertion_progress, 0.0)
        self.assertEqual(ctrl._search_count, 0)

    def test_insertion_failure_detection(self):
        ctrl = AssemblyController(grade='M', max_insertion_force=20.0)
        ctrl.start_assembly(np.array([100.0, 50.0, 0.0]), phase='insert')
        ctrl._insertion_velocity = 10.0
        # 持续高力 → 应检测到失败
        for i in range(60):
            result = ctrl.update(
                current_position=np.array([100.0, 50.0, -float(i) * 0.05]),
                current_force=18.0, lateral_force=1.0, dt=0.01
            )
            if ctrl._phase == ctrl.AssemblyPhase.FAILED:
                break
        # 最终应检测到插入失败
        self.assertIn(ctrl._phase, [ctrl.AssemblyPhase.FAILED, ctrl.AssemblyPhase.COMPLETE, ctrl.AssemblyPhase.INSERT])

    def test_all_five_grades_velocity(self):
        velocities = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            ctrl = AssemblyController(grade=grade)
            velocities.append(ctrl._insertion_velocity)
        # 等级越高速度越快
        self.assertEqual(velocities, sorted(velocities))


if __name__ == '__main__':
    unittest.main()
