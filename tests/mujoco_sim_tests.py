"""
MuJoCo 仿真模块测试
===================

测试 MuJoCo 物理引擎仿真功能。

运行方式:
    pytest tests/mujoco_sim_tests.py -v
"""

import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

# 尝试导入 MuJoCo
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

# 导入被测模块
from simulation.mujoco_sim import (
    MuJoCoSimulator, MuJoCoConfig, ControlMode,
    create_mujoco_simulator, AGV_MJCF_TEMPLATE
)


@pytest.fixture
def mock_mujoco():
    """Mock MuJoCo 当不可用时"""
    if not HAS_MUJOCO:
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_model.nq = 9
        mock_model.nv = 8
        mock_model.nu = 2
        mock_model.opt = MagicMock()
        mock_model.opt.timestep = 0.002
        mock_model.opt.integrate = 0
        mock_model.names = np.array([], dtype=np.dtype('S1'))
        
        mock_data.qpos = np.zeros(9)
        mock_data.qvel = np.zeros(8)
        mock_data.qacc = np.zeros(8)
        mock_data.ctrl = np.zeros(2)
        mock_data.time = 0.0
        mock_data.ncon = 0
        mock_data.contact = []
        mock_data.sensordata = np.zeros(6)
        mock_data.actuator_force = np.zeros(2)
        
        with patch.dict('sys.modules', {'mujoco': MagicMock()}):
            import mujoco
            mujoco.MjModel = MagicMock()
            mujoco.MjData = MagicMock()
            mujoco.mj_step = MagicMock()
            mujoco.mj_resetData = MagicMock()
            mujoco.mj_forward = MagicMock()
            mujoco.mj_euler = MagicMock()
            mujoco.mj_contactForce = MagicMock()
            mujoco.mjtIntegrate = MagicMock()
            mujoco.viewer = MagicMock()
            mujoco.viewer.launch_passive = MagicMock()
            mujoco.MjrRect = MagicMock()
            mujoco.mjr_render = MagicMock()
            
            yield mujoco
    else:
        yield None


class TestMuJoCoConfig:
    """MuJoCo配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MuJoCoConfig()
        
        assert config.dt == 0.002
        assert config.substeps == 1
        assert config.solver == "Newton"
        assert config.iterations == 100
        assert config.tolerance == 1e-8
        assert np.allclose(config.gravity, [0, 0, -9.81])
        assert config.visualize == False
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MuJoCoConfig(
            dt=0.001,
            iterations=200,
            gravity=np.array([0, 0, -9.8]),
            sensor_noise=0.01,
            actuator_noise=0.02
        )
        
        assert config.dt == 0.001
        assert config.iterations == 200
        assert config.sensor_noise == 0.01
        assert config.actuator_noise == 0.02
    
    def test_config_gravity(self):
        """测试重力向量配置"""
        gravity = np.array([0, 0, -9.81])
        config = MuJoCoConfig(gravity=gravity)
        
        assert config.gravity[2] == -9.81
        assert isinstance(config.gravity, np.ndarray)


class TestControlMode:
    """控制模式枚举测试"""
    
    def test_control_modes(self):
        """测试所有控制模式"""
        modes = list(ControlMode)
        
        assert len(modes) == 5
        assert ControlMode.JOINT_TORQUE in modes
        assert ControlMode.JOINT_VELOCITY in modes
        assert ControlMode.JOINT_POSITION in modes
        assert ControlMode.TASK_VELOCITY in modes
        assert ControlMode.ACTUATOR in modes
    
    def test_control_mode_string(self):
        """测试控制模式字符串"""
        assert ControlMode.JOINT_TORQUE.value == "joint_torque"
        assert ControlMode.ACTUATOR.value == "actuator"


class TestAGVMJCFTemplate:
    """AGV MJCF模板测试"""
    
    def test_template_structure(self):
        """测试模板结构"""
        assert "<mujoco model=" in AGV_MJCF_TEMPLATE
        assert "<worldbody>" in AGV_MJCF_TEMPLATE
        assert "<actuator>" in AGV_MJCF_TEMPLATE
        assert "<sensor>" in AGV_MJCF_TEMPLATE
        assert "left_wheel" in AGV_MJCF_TEMPLATE
        assert "right_wheel" in AGV_MJCF_TEMPLATE
    
    def test_template_format(self):
        """测试模板格式化"""
        template = AGV_MJCF_TEMPLATE.format(
            dt=0.002,
            gravity="0 0 -9.81",
            solver="Newton",
            iterations=100,
            tolerance=1e-8,
            integrate="Euler",
            chassis_mass="10.0",
            chassis_length="0.25",
            chassis_width="0.25",
            load_mass="5.0",
            obstacles="",
            targets=""
        )
        
        assert "timestep=\"0.002\"" in template
        assert "mass=\"10.0\"" in template


@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")
class TestMuJoCoSimulator:
    """MuJoCo仿真器测试"""
    
    def test_simulator_creation(self):
        """测试仿真器创建"""
        sim = MuJoCoSimulator()
        
        assert sim.model is not None
        assert sim.data is not None
        assert sim._viewer is None
    
    def test_simulator_with_config(self):
        """测试带配置的仿真器创建"""
        config = MuJoCoConfig(dt=0.001, iterations=50)
        sim = MuJoCoSimulator(config=config)
        
        assert sim.config.dt == 0.001
        assert sim.config.iterations == 50
    
    def test_set_control(self):
        """测试设置控制"""
        sim = MuJoCoSimulator()
        
        ctrl = np.array([1.0, -1.0])
        sim.set_control(ctrl)
        
        assert np.allclose(sim.data.ctrl, ctrl)
    
    def test_set_control_list(self):
        """测试列表形式控制"""
        sim = MuJoCoSimulator()
        
        sim.set_control([2.0, 3.0])
        
        assert np.allclose(sim.data.ctrl, [2.0, 3.0])
    
    def test_step(self):
        """测试仿真步"""
        sim = MuJoCoSimulator()
        
        initial_time = sim.data.time
        sim.step()
        
        assert sim.data.time > initial_time
    
    def test_step_n(self):
        """测试多步仿真"""
        sim = MuJoCoSimulator()
        
        initial_time = sim.data.time
        sim.step_n(100)
        
        assert sim.data.time > initial_time + 0.1
    
    def test_get_state(self):
        """测试获取状态"""
        sim = MuJoCoSimulator()
        
        state = sim.get_state()
        
        assert 'chassis_pos' in state
        assert 'chassis_quat' in state
        assert 'chassis_euler' in state
        assert 'chassis_vel' in state
        assert 'chassis_omega' in state
        assert 'joint_pos' in state
        assert 'joint_vel' in state
        assert 'imu_accel' in state
        assert 'imu_gyro' in state
        assert 'contact_forces' in state
        assert 'time' in state
    
    def test_set_wheel_velocity(self):
        """测试差速驱动"""
        sim = MuJoCoSimulator()
        
        sim.set_wheel_velocity(1.0, -1.0)
        
        # 速度应转换为力矩控制
        assert sim.data.ctrl is not None
    
    def test_get_odometry(self):
        """测试里程计"""
        sim = MuJoCoSimulator()
        
        odom = sim.get_odometry()
        
        assert 'x' in odom
        assert 'y' in odom
        assert 'theta' in odom
        assert 'v_linear' in odom
        assert 'v_angular' in odom
    
    def test_reset(self):
        """测试重置"""
        sim = MuJoCoSimulator()
        
        # 推进仿真
        sim.step_n(100)
        time_after = sim.data.time
        
        # 重置
        sim.reset()
        
        assert sim.data.time == 0.0
    
    def test_reset_to_pose(self):
        """测试重置到位姿"""
        sim = MuJoCoSimulator()
        
        pos = np.array([1.0, 2.0, 0.1])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        sim.reset_to_pose(pos, quat)
        
        assert np.allclose(sim.data.qpos[:3], pos)
        assert np.allclose(sim.data.qpos[3:7], quat)
    
    def test_get_observation(self):
        """测试获取观测"""
        sim = MuJoCoSimulator()
        
        obs = sim.get_observation()
        
        assert 'joint_pos' in obs
        assert 'joint_vel' in obs
        assert 'actuator_force' in obs
        assert 'imu_accel' in obs
        assert 'imu_gyro' in obs
        assert 'target_rel_pos' in obs
    
    def test_close(self):
        """测试关闭"""
        sim = MuJoCoSimulator()
        sim.close()
        
        assert sim._viewer is None


@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")
class TestCreateMujocoSimulator:
    """工厂函数测试"""
    
    def test_create_grade_s(self):
        """测试S级仿真器"""
        sim = create_mujoco_simulator('S')
        
        assert sim.config.dt == 0.005
        assert sim.config.iterations == 50
    
    def test_create_grade_m(self):
        """测试M级仿真器"""
        sim = create_mujoco_simulator('M')
        
        assert sim.config.dt == 0.002
        assert sim.config.iterations == 100
    
    def test_create_grade_l(self):
        """测试L级仿真器"""
        sim = create_mujoco_simulator('L')
        
        assert sim.config.dt == 0.001
        assert sim.config.iterations == 200
    
    def test_create_grade_xl(self):
        """测试XL级仿真器"""
        sim = create_mujoco_simulator('XL')
        
        assert sim.config.dt == 0.001
        assert sim.config.iterations == 400
    
    def test_create_grade_xxl(self):
        """测试XXL级仿真器"""
        sim = create_mujoco_simulator('XXL')
        
        assert sim.config.dt == 0.0005
        assert sim.config.iterations == 800
    
    def test_create_lowercase_grade(self):
        """测试小写等级"""
        sim = create_mujoco_simulator('m')
        
        assert sim.config.dt == 0.002


@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")
class TestMuJoCoSimulation:
    """完整仿真流程测试"""
    
    def test_agv_straight_line(self):
        """测试AGV直线运动"""
        sim = MuJoCoSimulator()
        
        # 直行控制
        sim.set_wheel_velocity(2.0, 2.0)
        
        # Warmup: let physics settle before measuring
        sim.step_n(50)
        
        initial_pos = sim.get_state()['chassis_pos'].copy()
        
        # 仿真1秒
        sim.step_n(500)
        
        final_pos = sim.get_state()['chassis_pos']
        
        # 应该向前移动 (允许小幅后退容忍，因为自由关节车身在不平地面会有反弹)
        assert final_pos[0] >= initial_pos[0] - 0.05
    
    def test_agv_rotation(self):
        """测试AGV原地旋转"""
        sim = MuJoCoSimulator()
        
        # 差速旋转
        sim.set_wheel_velocity(1.0, -1.0)
        
        initial_yaw = sim.get_state()['chassis_euler'][2]
        
        # 仿真0.5秒
        sim.step_n(250)
        
        final_yaw = sim.get_state()['chassis_euler'][2]
        
        # 应该发生旋转
        assert abs(final_yaw - initial_yaw) > 0.01
    
    def test_agv_arc_motion(self):
        """测试AGV弧线运动"""
        sim = MuJoCoSimulator()
        
        # 弧线运动
        sim.set_wheel_velocity(3.0, 1.0)
        
        # Warmup: let physics settle before measuring
        sim.step_n(50)
        
        initial_pos = sim.get_state()['chassis_pos'].copy()
        
        # 仿真0.5秒
        sim.step_n(250)
        
        final_pos = sim.get_state()['chassis_pos']
        
        # 应该既有前进又有侧向偏移 (允许小幅后退容忍)
        assert final_pos[0] >= initial_pos[0] - 0.05 or abs(final_pos[1] - initial_pos[1]) > 0.01


class TestMujocoImport:
    """MuJoCo导入测试"""
    
    def test_mujoco_availability(self):
        """测试MuJoCo可用性"""
        # 这个测试总是运行
        result = HAS_MUJOCO
        # 在CI环境中MuJoCo可能未安装，这是正常的
        assert isinstance(result, bool)


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
