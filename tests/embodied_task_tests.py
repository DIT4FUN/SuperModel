"""
具身任务执行器测试
测试 EmbodiedTaskExecutor 的完整任务链:
- 抓取-搬运-放置
- 推动 (Push)
- 拉动 (Pull)
- 表面轮廓追踪 (Surface Trace)
- 插入任务 (Insert)
- 表面抛光 (Polish)

覆盖 S/M/L/XL/XXL 五级 AGV 具身任务能力
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.embodied_control import (
    EmbodiedController, EmbodiedState, EmbodiedCommand,
    EmbodiedControlParams, EmbodiedTaskExecutor,
    EmbodiedGrade, SensorHealthMonitor,
    AGV_EMBODIED_GRADES, get_embodied_spec,
)


class TestEmbodiedTaskExecutorInit(unittest.TestCase):
    """测试具身任务执行器初始化"""

    def test_init_default(self):
        """默认初始化"""
        ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        executor = EmbodiedTaskExecutor(ctrl, grade='M')
        
        self.assertEqual(executor.grade, 'M')
        self.assertEqual(executor.phase, EmbodiedTaskExecutor.TaskPhase.IDLE)
        self.assertEqual(executor.success_count, 0)
        self.assertEqual(executor.failure_count, 0)

    def test_init_all_grades(self):
        """所有等级初始化"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            executor = EmbodiedTaskExecutor(ctrl, grade=grade)
            self.assertEqual(executor.grade, grade)
            self.assertEqual(executor.phase, EmbodiedTaskExecutor.TaskPhase.IDLE)

    def test_phase_enum(self):
        """任务阶段枚举"""
        phases = EmbodiedTaskExecutor.TaskPhase
        self.assertEqual(phases.IDLE.value, "idle")
        self.assertEqual(phases.APPROACH.value, "approach")
        self.assertEqual(phases.CONTACT.value, "contact")
        self.assertEqual(phases.GRASP.value, "grasp")
        self.assertEqual(phases.LIFT.value, "lift")
        self.assertEqual(phases.TRANSPORT.value, "transport")
        self.assertEqual(phases.PLACE.value, "place")
        self.assertEqual(phases.RELEASE.value, "release")
        self.assertEqual(phases.RETRACT.value, "retract")


class TestGraspPlaceTask(unittest.TestCase):
    """测试抓取-搬运-放置任务"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.ctrl.set_sensors()
        self.ctrl._initialized = True
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='M')

    def test_grasp_place_basic(self):
        """基本抓取-放置流程"""
        obj_pos = np.array([0.5, 0.3, 0.05])
        place_pos = np.array([0.8, 0.6, 0.05])
        
        result = self.executor.execute_grasp_place(obj_pos, place_pos)
        
        self.assertTrue(result)
        self.assertEqual(self.executor.success_count, 1)
        self.assertEqual(self.executor.phase, EmbodiedTaskExecutor.TaskPhase.IDLE)

    def test_grasp_place_all_grades(self):
        """所有等级的抓取-放置"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            ctrl.set_sensors()
            ctrl._initialized = True
            exec = EmbodiedTaskExecutor(ctrl, grade=grade)
            
            obj_pos = np.array([0.5, 0.3, 0.05])
            place_pos = np.array([0.8, 0.6, 0.05])
            
            result = exec.execute_grasp_place(obj_pos, place_pos)
            self.assertTrue(result, f"Grade {grade} grasp-place failed")
            self.assertEqual(exec.success_count, 1)

    def test_grasp_place_multiple(self):
        """连续多次抓取-放置"""
        for i in range(3):
            obj_pos = np.array([0.3 + i * 0.1, 0.3, 0.05])
            place_pos = np.array([0.7 + i * 0.1, 0.5, 0.05])
            result = self.executor.execute_grasp_place(obj_pos, place_pos)
            self.assertTrue(result)
        
        self.assertEqual(self.executor.success_count, 3)

    def test_grasp_place_metrics(self):
        """抓取-放置指标"""
        obj_pos = np.array([0.5, 0.3, 0.05])
        place_pos = np.array([0.8, 0.6, 0.05])
        
        self.executor.execute_grasp_place(obj_pos, place_pos)
        
        metrics = self.executor.get_metrics()
        self.assertEqual(metrics['success_count'], 1)
        self.assertEqual(metrics['failure_count'], 0)
        self.assertEqual(metrics['success_rate'], 1.0)
        self.assertIsInstance(metrics['phase_history'], list)

    def test_grasp_place_with_force(self):
        """指定抓取力"""
        obj_pos = np.array([0.5, 0.3, 0.05])
        place_pos = np.array([0.8, 0.6, 0.05])
        
        result = self.executor.execute_grasp_place(
            obj_pos, place_pos, grasp_force=15.0
        )
        self.assertTrue(result)


class TestPushTask(unittest.TestCase):
    """测试推动任务"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.ctrl.set_sensors()
        self.ctrl._initialized = True
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='M')

    def test_push_basic(self):
        """基本推动任务"""
        obj_pos = np.array([0.5, 0.3, 0.0])
        push_dir = np.array([1.0, 0.0, 0.0])
        
        result = self.executor.execute_push(
            obj_pos, push_dir, push_distance=0.2, push_force=15.0
        )
        
        self.assertTrue(result)
        self.assertEqual(self.executor.success_count, 1)
        self.assertEqual(self.executor.phase, EmbodiedTaskExecutor.TaskPhase.IDLE)

    def test_push_all_grades(self):
        """所有等级的推动任务"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            ctrl.set_sensors()
            ctrl._initialized = True
            exec = EmbodiedTaskExecutor(ctrl, grade=grade)
            
            obj_pos = np.array([0.5, 0.3, 0.0])
            push_dir = np.array([1.0, 0.0, 0.0])
            
            result = exec.execute_push(obj_pos, push_dir)
            self.assertTrue(result, f"Grade {grade} push failed")

    def test_push_different_directions(self):
        """不同方向的推动"""
        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.707, 0.707, 0.0]),  # 对角线
        ]
        
        for push_dir in directions:
            obj_pos = np.array([0.5, 0.3, 0.0])
            result = self.executor.execute_push(obj_pos, push_dir)
            self.assertTrue(result, f"Push direction {push_dir} failed")

    def test_push_zero_direction(self):
        """零方向推动 (应自动处理)"""
        obj_pos = np.array([0.5, 0.3, 0.0])
        result = self.executor.execute_push(
            obj_pos, np.array([0.0, 0.0, 0.0]), push_force=10.0
        )
        self.assertTrue(result)

    def test_push_different_distances(self):
        """不同推动距离"""
        for distance in [0.05, 0.1, 0.2, 0.3]:
            obj_pos = np.array([0.3, 0.3, 0.0])
            push_dir = np.array([1.0, 0.0, 0.0])
            result = self.executor.execute_push(
                obj_pos, push_dir, push_distance=distance
            )
            self.assertTrue(result, f"Push distance {distance}m failed")


class TestPullTask(unittest.TestCase):
    """测试拉动任务"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='M', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.ctrl.set_sensors()
        self.ctrl._initialized = True
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='M')

    def test_pull_basic(self):
        """基本拉动任务"""
        obj_pos = np.array([0.8, 0.6, 0.0])
        pull_dir = np.array([-1.0, 0.0, 0.0])
        
        result = self.executor.execute_pull(
            obj_pos, pull_dir, pull_distance=0.2, pull_force=15.0
        )
        
        self.assertTrue(result)
        self.assertEqual(self.executor.success_count, 1)

    def test_pull_all_grades(self):
        """所有等级的拉动任务"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            ctrl.set_sensors()
            ctrl._initialized = True
            exec = EmbodiedTaskExecutor(ctrl, grade=grade)
            
            obj_pos = np.array([0.8, 0.6, 0.0])
            pull_dir = np.array([-1.0, 0.0, 0.0])
            
            result = exec.execute_pull(obj_pos, pull_dir)
            self.assertTrue(result, f"Grade {grade} pull failed")


class TestSurfaceTraceTask(unittest.TestCase):
    """测试表面轮廓追踪任务"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='L', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.ctrl.set_sensors()
        self.ctrl._initialized = True
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='L')

    def test_surface_trace_single_cycle(self):
        """单圈轮廓追踪"""
        surface_center = np.array([0.5, 0.5, 0.1])
        
        result = self.executor.execute_surface_trace(
            surface_center, trace_radius=0.03, num_cycles=1
        )
        
        self.assertTrue(result)
        self.assertEqual(self.executor.success_count, 1)

    def test_surface_trace_multi_cycle(self):
        """多圈轮廓追踪"""
        surface_center = np.array([0.5, 0.5, 0.1])
        
        result = self.executor.execute_surface_trace(
            surface_center, trace_radius=0.03, num_cycles=3
        )
        
        self.assertTrue(result)
        self.assertGreater(len(self.executor.phase_history), 0)

    def test_surface_trace_all_grades(self):
        """所有等级的表面追踪"""
        grades = ['M', 'L', 'XL', 'XXL']
        for grade in grades:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            ctrl.set_sensors()
            ctrl._initialized = True
            exec = EmbodiedTaskExecutor(ctrl, grade=grade)
            
            surface_center = np.array([0.5, 0.5, 0.1])
            result = exec.execute_surface_trace(surface_center, num_cycles=1)
            self.assertTrue(result, f"Grade {grade} surface trace failed")

    def test_surface_trace_different_radii(self):
        """不同半径的轮廓追踪"""
        for radius in [0.02, 0.05, 0.08]:
            result = self.executor.execute_surface_trace(
                np.array([0.5, 0.5, 0.1]),
                trace_radius=radius, num_cycles=1
            )
            self.assertTrue(result, f"Trace radius {radius}m failed")


class TestInsertTask(unittest.TestCase):
    """测试插入任务"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='L', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.ctrl.set_sensors()
        self.ctrl._initialized = True
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='L')

    def test_insert_basic(self):
        """基本插入任务"""
        target_pos = np.array([0.5, 0.5, 0.1])
        
        result = self.executor.execute_insert(
            target_pos, insertion_depth=0.03, insertion_force=20.0
        )
        
        self.assertTrue(result)
        self.assertEqual(self.executor.success_count, 1)

    def test_insert_all_grades(self):
        """所有等级的插入任务"""
        grades = ['M', 'L', 'XL', 'XXL']
        for grade in grades:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            ctrl.set_sensors()
            ctrl._initialized = True
            exec = EmbodiedTaskExecutor(ctrl, grade=grade)
            
            target_pos = np.array([0.5, 0.5, 0.1])
            result = exec.execute_insert(target_pos, insertion_depth=0.03)
            self.assertTrue(result, f"Grade {grade} insert failed")

    def test_insert_different_depths(self):
        """不同深度的插入"""
        for depth in [0.01, 0.02, 0.05, 0.08]:
            result = self.executor.execute_insert(
                np.array([0.5, 0.5, 0.1]),
                insertion_depth=depth, insertion_force=25.0
            )
            self.assertTrue(result, f"Insert depth {depth}m failed")

    def test_insert_different_forces(self):
        """不同插入力"""
        for force in [10.0, 20.0, 30.0]:
            result = self.executor.execute_insert(
                np.array([0.5, 0.5, 0.1]),
                insertion_depth=0.03, insertion_force=force
            )
            self.assertTrue(result, f"Insert force {force}N failed")


class TestPolishTask(unittest.TestCase):
    """测试表面抛光任务"""

    def setUp(self):
        self.ctrl = EmbodiedController(grade='XL', use_virtual_sensors=True)
        self.ctrl.init_virtual_sensors()
        self.ctrl.set_sensors()
        self.ctrl._initialized = True
        self.executor = EmbodiedTaskExecutor(self.ctrl, grade='XL')

    def test_polish_basic(self):
        """基本抛光任务"""
        surface_pos = np.array([0.5, 0.5, 0.1])
        surface_normal = np.array([0.0, 0.0, 1.0])
        
        result = self.executor.execute_polish(
            surface_pos, surface_normal,
            polish_area=0.05, polish_force=5.0, duration_sec=0.5
        )
        
        self.assertTrue(result)
        self.assertEqual(self.executor.success_count, 1)

    def test_polish_all_grades(self):
        """所有等级的抛光任务"""
        grades = ['L', 'XL', 'XXL']
        for grade in grades:
            ctrl = EmbodiedController(grade=grade, use_virtual_sensors=True)
            ctrl.init_virtual_sensors()
            ctrl.set_sensors()
            ctrl._initialized = True
            exec = EmbodiedTaskExecutor(ctrl, grade=grade)
            
            surface_pos = np.array([0.5, 0.5, 0.1])
            surface_normal = np.array([0.0, 0.0, 1.0])
            result = exec.execute_polish(
                surface_pos, surface_normal, duration_sec=0.3
            )
            self.assertTrue(result, f"Grade {grade} polish failed")

    def test_polish_unnormalized_normal(self):
        """非归一化法向量的抛光 (应自动归一化)"""
        surface_pos = np.array([0.5, 0.5, 0.1])
        surface_normal = np.array([0.0, 0.0, 3.0])  # 未归一化
        
        result = self.executor.execute_polish(
            surface_pos, surface_normal, duration_sec=0.3
        )
        self.assertTrue(result)


class TestSensorHealthMonitor(unittest.TestCase):
    """测试传感器健康监控"""

    def test_init_all_grades(self):
        """所有等级的健康监控初始化"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            monitor = SensorHealthMonitor(grade=grade)
            self.assertEqual(monitor.grade, grade)
            self.assertEqual(monitor._tactile_faults, 0)
            self.assertEqual(monitor._force_faults, 0)
            self.assertEqual(monitor._imu_faults, 0)

    def test_tactile_health_no_contacts(self):
        """无接触时触觉健康"""
        monitor = SensorHealthMonitor(grade='M')
        health = monitor.check_tactile_health([])
        self.assertIsInstance(health, dict)
        self.assertIn('health_score', health)
        self.assertIn('issues', health)

    def test_tactile_health_excessive_contacts(self):
        """接触过多时的触觉健康"""
        from src.sensors.tactile import TactileContact
        monitor = SensorHealthMonitor(grade='M')
        
        # 创建大量接触 (异常情况)
        fake_contacts = [
            TactileContact(
                center=(i, i), area=1, peak_pressure=0.5,
                mean_pressure=0.3, centroid=(float(i), float(i)),
                contact_force=3.0
            ) for i in range(100)
        ]
        
        health = monitor.check_tactile_health(fake_contacts)
        self.assertLess(health['health_score'], 1.0)
        self.assertIn('excessive_contacts', health['issues'])

    def test_force_health_normal(self):
        """正常力信号的力觉健康"""
        from src.sensors.force import Wrench
        monitor = SensorHealthMonitor(grade='M')
        
        wrench = Wrench(
            force=np.array([0.0, 0.0, -5.0]),
            torque=np.array([0.0, 0.0, 0.0])
        )
        
        health = monitor.check_force_health(wrench)
        self.assertIsInstance(health, dict)
        self.assertIn('health_score', health)

    def test_force_health_anomaly(self):
        """异常力信号检测"""
        from src.sensors.force import Wrench
        monitor = SensorHealthMonitor(grade='M')
        
        # 异常大的力 (超过5000N量程 -> saturated)
        wrench = Wrench(
            force=np.array([3000.0, 3000.0, 3000.0]),
            torque=np.array([500.0, 500.0, 500.0])
        )
        
        health = monitor.check_force_health(wrench)
        self.assertLess(health['health_score'], 1.0)
        self.assertGreater(len(health['issues']), 0)

    def test_imu_health_normal(self):
        """正常IMU健康"""
        from src.sensors.imu import IMUFrame
        monitor = SensorHealthMonitor(grade='M')
        
        frame = IMUFrame(
            accel=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.0]),
            mag=None
        )
        
        health = monitor.check_imu_health(frame)
        self.assertIsInstance(health, dict)
        self.assertIn('health_score', health)

    def test_imu_health_anomaly(self):
        """异常IMU数据检测"""
        from src.sensors.imu import IMUFrame
        monitor = SensorHealthMonitor(grade='M')
        
        # 异常大的加速度 (超过200 m/s²)
        frame = IMUFrame(
            accel=np.array([0.0, 0.0, 500.0]),  # 异常值
            gyro=np.array([100.0, 100.0, 100.0]),  # 异常值
            mag=None
        )
        
        health = monitor.check_imu_health(frame)
        self.assertLess(health['health_score'], 1.0)
        self.assertGreater(len(health['issues']), 0)


class TestEmbodiedFiveGrade(unittest.TestCase):
    """测试AGV五级具身控制能力差异"""

    def test_all_grades_exist(self):
        """所有五级规格存在"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_embodied_spec(grade)
            self.assertIsInstance(spec, dict)
            self.assertIn('control_rate', spec)
            self.assertIn('fusion_method', spec)
            self.assertIn('tactile_enabled', spec)
            self.assertIn('force_enabled', spec)
            self.assertIn('imu_enabled', spec)

    def test_control_rate_increases_with_grade(self):
        """等级越高控制频率越高"""
        rates = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            rates[grade] = get_embodied_spec(grade)['control_rate']
        
        self.assertLessEqual(rates['S'], rates['M'])
        self.assertLessEqual(rates['M'], rates['L'])
        self.assertLessEqual(rates['L'], rates['XL'])
        self.assertLessEqual(rates['XL'], rates['XXL'])

    def test_fusion_methods_advance_with_grade(self):
        """等级越高融合方法越先进"""
        methods = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            methods[grade] = get_embodied_spec(grade)['fusion_method']
        
        # S级使用阈值
        self.assertEqual(methods['S'], 'threshold')
        # XXL使用MPC融合
        self.assertEqual(methods['XXL'], 'mpc_fusion')

    def test_tactile_enabled_all_grades(self):
        """所有等级启用触觉"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            self.assertTrue(get_embodied_spec(grade)['tactile_enabled'])

    def test_force_enabled_m_and_above(self):
        """M级及以上启用力觉"""
        self.assertFalse(get_embodied_spec('S')['force_enabled'])
        for grade in ['M', 'L', 'XL', 'XXL']:
            self.assertTrue(get_embodied_spec(grade)['force_enabled'])

    def test_imu_enabled_m_and_above(self):
        """M级及以上启用IMU"""
        self.assertFalse(get_embodied_spec('S')['imu_enabled'])
        for grade in ['M', 'L', 'XL', 'XXL']:
            self.assertTrue(get_embodied_spec(grade)['imu_enabled'])

    def test_max_contact_force_increases(self):
        """等级越高最大接触力越大"""
        forces = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            forces[grade] = get_embodied_spec(grade)['max_contact_force']
        
        self.assertLessEqual(forces['S'], forces['M'])
        self.assertLessEqual(forces['M'], forces['L'])
        self.assertLessEqual(forces['L'], forces['XL'])
        self.assertLessEqual(forces['XL'], forces['XXL'])

    def test_latency_decreases_with_grade(self):
        """等级越高延迟越低"""
        latencies = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            latencies[grade] = get_embodied_spec(grade)['latency_ms']
        
        self.assertGreaterEqual(latencies['S'], latencies['M'])
        self.assertGreaterEqual(latencies['M'], latencies['L'])
        self.assertGreaterEqual(latencies['L'], latencies['XL'])
        self.assertGreaterEqual(latencies['XL'], latencies['XXL'])


if __name__ == '__main__':
    unittest.main()
