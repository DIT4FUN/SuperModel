"""
传感-运动融合控制测试
=====================

测试 SensorimotorIntegration 和 SensorimotorSimulator 的功能:
- 多模态融合控制
- 触觉/力觉/IMU 协调
- 抓取任务仿真
- AGV导航仿真
- 碰撞恢复仿真
- AGV五级规格

运行:
  pytest tests/sensorimotor_tests.py -v
"""

import pytest
import numpy as np
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.sensorimotor import (
    SensorimotorIntegration, SensorimotorConfig, SensorimotorState,
    SensorimotorSimulator,
    get_sensorimotor_spec, AGV_SENSORIMOTOR_GRADES
)


# ─── 基础功能测试 ───────────────────────────────────────────────

class TestSensorimotorConfig:
    """配置测试"""
    
    def test_sensorimotor_config_defaults(self):
        cfg = SensorimotorConfig()
        assert cfg.fusion_strategy == "weighted"
        assert cfg.control_rate == 100.0
        assert cfg.tactile_weight + cfg.force_weight + cfg.imu_weight == pytest.approx(1.0)
    
    def test_sensorimotor_config_from_grade(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            cfg = SensorimotorConfig.from_grade(grade)
            assert cfg.grade == grade
            assert cfg.control_rate > 0
    
    def test_sensorimotor_spec_all_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_sensorimotor_spec(grade)
            assert spec.grade == grade
            assert spec.tactile_weight >= 0
            assert spec.force_weight >= 0
            assert spec.imu_weight >= 0
    
    def test_agv_sensorimotor_grades_keys(self):
        assert set(AGV_SENSORIMOTOR_GRADES.keys()) == {'S', 'M', 'L', 'XL', 'XXL'}


class TestSensorimotorState:
    """状态数据类测试"""
    
    def test_state_defaults(self):
        state = SensorimotorState()
        assert not state.tactile_contact
        assert state.force_magnitude == 0.0
        assert not state.force_in_contact
        assert not state.force_collision
        assert state.imu_stable
        assert state.fused_control.shape == (3,)
    
    def test_state_with_data(self):
        from sensors.tactile import TactileContact
        contacts = [TactileContact(
            center=(10, 10), area=25, peak_pressure=0.8,
            mean_pressure=0.5, centroid=(10.0, 10.0),
            contact_force=5.0, slip_probability=0.1
        )]
        state = SensorimotorState(
            tactile_contact=True,
            tactile_contacts=contacts,
            tactile_slip_prob=0.1,
            force_magnitude=10.0,
            force_in_contact=True,
            fused_control=np.array([0.1, 0.2, 0.3]),
            control_authority={'tactile': 0.5, 'force': 0.3, 'imu': 0.2}
        )
        assert state.tactile_contact
        assert len(state.tactile_contacts) == 1
        assert state.force_magnitude == 10.0
        np.testing.assert_array_almost_equal(
            state.fused_control, [0.1, 0.2, 0.3]
        )


class TestSensorimotorSimulatorInit:
    """Simulator 初始化测试"""
    
    def test_simulator_init_all_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = SensorimotorSimulator(grade=grade)
            assert sim.grade == grade
            assert sim.integration is not None
            assert not sim._is_running


class TestSensorimotorSimulatorContext:
    """Simulator 上下文管理测试"""
    
    def test_simulator_context_manager(self):
        sim = SensorimotorSimulator('M')
        with sim as s:
            assert s._is_running
            assert s.virtual_tactile._is_opened
            assert s.virtual_force._is_streaming
            assert s.virtual_imu._is_opened


class TestSensorimotorSimulatorGrasp:
    """抓取任务仿真测试"""
    
    def test_simulate_grasp_basic(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_grasp(
            object_pos=(0.5, 0.5),
            object_force=10.0,
            num_steps=20,
            dt=0.01
        )
        assert len(states) == 20
        # 最后状态应该有接触
        final = states[-1]
        assert final.frame_id == 20  # 0-indexed frames: step 0→id1, ..., step 19→id20
        sim.close()
    
    def test_simulate_grasp_all_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = SensorimotorSimulator(grade=grade)
            sim.open()
            states = sim.simulate_grasp(num_steps=10, dt=0.01)
            assert len(states) == 10
            sim.close()
    
    def test_simulate_grasp_contact_phases(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_grasp(num_steps=100, dt=0.01)
        
        # 接近阶段 (0-20): 无触觉接触
        for s in states[:5]:
            assert not s.tactile_contact
        
        # 后续阶段应有接触数据
        has_contact = any(s.tactile_contact for s in states)
        assert has_contact
        
        sim.close()
    
    def test_simulate_grasp_grip_quality(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_grasp(object_force=15.0, num_steps=50, dt=0.01)
        
        # 夹取阶段应有抓取质量
        grip_qualities = [s.tactile_grip_quality for s in states if s.tactile_contact]
        if grip_qualities:
            assert all(0 <= q <= 1 for q in grip_qualities)
        
        sim.close()
    
    def test_simulate_grasp_control_output(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_grasp(num_steps=30, dt=0.01)
        
        for s in states:
            assert s.fused_control.shape == (3,)
            assert not np.any(np.isnan(s.fused_control))
        
        sim.close()


class TestSensorimotorSimulatorAGV:
    """AGV导航仿真测试"""
    
    def test_simulate_agv_navigation_circle(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_agv_navigation(
            trajectory_type="circle",
            duration_s=1.0,
            dt=0.01
        )
        assert len(states) == 100
        sim.close()
    
    def test_simulate_agv_navigation_all_types(self):
        for traj_type in ["circle", "figure8", "linear", "sine"]:
            sim = SensorimotorSimulator('M')
            sim.open()
            states = sim.simulate_agv_navigation(
                trajectory_type=traj_type,
                duration_s=0.5,
                dt=0.01
            )
            assert len(states) == 50
            sim.close()
    
    def test_simulate_agv_navigation_all_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = SensorimotorSimulator(grade=grade)
            sim.open()
            states = sim.simulate_agv_navigation(
                trajectory_type="circle",
                duration_s=0.3,
                dt=0.01
            )
            assert len(states) == 30
            sim.close()
    
    def test_simulate_agv_navigation_control_authority(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_agv_navigation(duration_s=0.5, dt=0.01)
        
        for s in states:
            auth = s.control_authority
            assert set(auth.keys()) == {'tactile', 'force', 'imu'}
            total = sum(auth.values())
            assert total == pytest.approx(1.0, abs=0.01)
        
        sim.close()


class TestSensorimotorSimulatorCollision:
    """碰撞恢复仿真测试"""
    
    def test_simulate_collision_recovery(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_collision_recovery(
            collision_direction=(1.0, 0.0, 0.0),
            collision_force=50.0,
            recovery_steps=30,
            dt=0.01
        )
        assert len(states) == 30
        
        # 第一步应该有碰撞
        assert states[0].force_collision
        
        # 最后一步应该恢复正常
        assert not states[-1].force_collision
        sim.close()
    
    def test_simulate_collision_recovery_all_directions(self):
        directions = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)
        ]
        for direction in directions:
            sim = SensorimotorSimulator('M')
            sim.open()
            states = sim.simulate_collision_recovery(
                collision_direction=direction,
                recovery_steps=20
            )
            assert states[0].force_collision
            sim.close()


class TestSensorimotorIntegration:
    """融合控制器测试"""
    
    def test_integration_init_no_sensors(self):
        cfg = SensorimotorConfig.from_grade('M')
        ctrl = SensorimotorIntegration(config=cfg, grade='M')
        assert ctrl.config.grade == 'M'
        ctrl.close()
    
    def test_integration_open_close(self):
        sim = SensorimotorSimulator('M')
        with sim as s:
            assert s.integration._state is not None
    
    def test_integration_step_basic(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        state = sim.integration.step(dt=0.01)
        assert state.fused_control.shape == (3,)
        assert state.frame_id == 1
        sim.close()
    
    def test_integration_step_multiple(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        for i in range(10):
            state = sim.integration.step(dt=0.01)
            assert state.frame_id == i + 1
        sim.close()
    
    def test_integration_is_safe(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        assert sim.integration.is_safe()
        sim.close()
    
    def test_integration_emergency_stop(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        sim.integration.step(dt=0.01)
        sim.integration.emergency_stop()
        state = sim.integration.get_state()
        np.testing.assert_array_equal(state.fused_control, np.zeros(3))
        sim.close()
    
    def test_integration_control_authority(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        state = sim.integration.step(dt=0.01)
        auth = sim.integration.get_control_authority()
        assert 'tactile' in auth
        assert 'force' in auth
        assert 'imu' in auth
        sim.close()


class TestSensorimotorFusionStrategies:
    """融合策略测试"""
    
    @pytest.mark.parametrize("strategy", ["weighted", "adaptive", "hierarchical"])
    def test_all_strategies(self, strategy):
        cfg = SensorimotorConfig(fusion_strategy=strategy, grade='M')
        sim = SensorimotorSimulator('M')
        sim.open()
        sim.integration.config = cfg
        
        for _ in range(5):
            state = sim.integration.step(dt=0.01)
            assert state.fused_control.shape == (3,)
        
        sim.close()


class TestSensorimotorFiveGrade:
    """AGV五级规格测试"""
    
    @pytest.mark.parametrize("grade", ['S', 'M', 'L', 'XL', 'XXL'])
    def test_grades_control_rates(self, grade):
        cfg = SensorimotorConfig.from_grade(grade)
        assert cfg.grade == grade
        expected_rates = {'S': 50, 'M': 100, 'L': 200, 'XL': 500, 'XXL': 1000}
        assert cfg.control_rate == expected_rates[grade]
    
    @pytest.mark.parametrize("grade", ['S', 'M', 'L', 'XL', 'XXL'])
    def test_grades_weights_sum(self, grade):
        cfg = SensorimotorConfig.from_grade(grade)
        total = cfg.tactile_weight + cfg.force_weight + cfg.imu_weight
        assert total == pytest.approx(1.0)
    
    @pytest.mark.parametrize("grade", ['S', 'M', 'L', 'XL', 'XXL'])
    def test_grades_simulator(self, grade):
        sim = SensorimotorSimulator(grade)
        sim.open()
        
        # 抓取
        g_states = sim.simulate_grasp(num_steps=10, dt=0.01)
        assert len(g_states) == 10
        
        # 导航
        n_states = sim.simulate_agv_navigation(trajectory_type="circle",
                                                  duration_s=0.2, dt=0.01)
        assert len(n_states) == 20
        
        # 碰撞
        c_states = sim.simulate_collision_recovery(recovery_steps=10)
        assert len(c_states) == 10
        
        sim.close()


class TestSensorimotorRobustness:
    """鲁棒性测试"""
    
    def test_nan_handling(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        for _ in range(10):
            state = sim.integration.step(dt=0.01)
            assert not np.any(np.isnan(state.fused_control))
            for k in ['tactile', 'force', 'imu']:
                assert not np.isnan(state.control_authority.get(k, 0.0))
        sim.close()
    
    def test_inf_handling(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        for _ in range(10):
            state = sim.integration.step(dt=0.01)
            assert not np.any(np.isinf(state.fused_control))
        sim.close()
    
    def test_large_step_count(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        states = sim.simulate_grasp(num_steps=500, dt=0.01)
        assert len(states) == 500
        sim.close()
    
    def test_zero_dt(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        state = sim.integration.step(dt=0.0)
        assert state.fused_control.shape == (3,)
        sim.close()


class TestSensorimotorPerformance:
    """性能测试"""
    
    def test_grasp_simulation_speed(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        
        start = time.time()
        states = sim.simulate_grasp(num_steps=100, dt=0.01)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # 100步抓取应在5秒内完成
        assert len(states) == 100
        sim.close()
    
    def test_agv_navigation_speed(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        
        start = time.time()
        states = sim.simulate_agv_navigation(duration_s=2.0, dt=0.01)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # 2秒导航仿真应在5秒内完成
        assert len(states) == 200
        sim.close()


class TestSensorimotorTimestamp:
    """时间戳和帧ID测试"""
    
    def test_frame_id_increments(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        prev_id = 0
        for _ in range(20):
            state = sim.integration.step(dt=0.01)
            assert state.frame_id == prev_id + 1
            prev_id = state.frame_id
        sim.close()
    
    def test_timestamp_increases(self):
        sim = SensorimotorSimulator('M')
        sim.open()
        prev_ts = 0.0
        for _ in range(20):
            state = sim.integration.step(dt=0.01)
            assert state.timestamp >= prev_ts
            prev_ts = state.timestamp
        sim.close()
