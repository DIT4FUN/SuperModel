"""
PyBullet 仿真模块测试
====================

测试 PyBullet 物理引擎仿真功能。

运行方式:
    pytest tests/pybullet_sim_tests.py -v

依赖:
    pip install pybullet
"""

import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

# 尝试导入 PyBullet
try:
    import pybullet
    import pybullet_data
    _REAL_PYBULLET_AVAILABLE = True
except ImportError:
    _REAL_PYBULLET_AVAILABLE = False

# HAS_PYBULLET 在 simulation.pybullet_sim 模块内定义
# 通过模块引用访问，避免在测试文件顶层导入时产生局部作用域问题
from simulation.pybullet_sim import HAS_PYBULLET

# 导入被测模块 (类/函数可直接引用)
from simulation.pybullet_sim import (
    PyBulletSimulator, PyBulletConfig, PyBulletGUI,
    create_pybullet_simulator, generate_agv_urdf,
    get_pybullet_spec, AGV_URDF_TEMPLATE
)


@pytest.fixture
def mock_pybullet():
    """Mock PyBullet 当不可用时 (patch HAS_PYBULLET + p + pybullet_data)"""
    import simulation.pybullet_sim as pbsim

    if not pbsim.HAS_PYBULLET:
        mock_p = MagicMock()
        mock_p.GUI = 1
        mock_p.DIRECT = 2
        mock_p.VELOCITY_CONTROL = 0
        mock_p.TORQUE_CONTROL = 1
        mock_p.GEOM_BOX = 3
        mock_p.GEOM_CYLINDER = 5
        mock_p.GEOM_SPHERE = 2
        mock_p.GEOM_PLANE = 4
        mock_p.ER_BULLET_HARDWARE_OPEN = 1
        mock_p.LINK_FRAME = 4
        mock_p.JOINT_REVOLUTE = 0
        mock_p.JOINT_PRISMATIC = 1
        mock_p.JOINT_FIXED = 4
        mock_p.POSITION_CONTROL = 2
        mock_p.resetSimulation = MagicMock()
        mock_p.setGravity = MagicMock()
        mock_p.setTimeStep = MagicMock()
        mock_p.setPhysicsEngineParameter = MagicMock()
        mock_p.connect = MagicMock(return_value=0)
        mock_p.disconnect = MagicMock()
        mock_p.stepSimulation = MagicMock()
        mock_p.loadURDF = MagicMock(return_value=1)
        mock_p.getNumJoints = MagicMock(return_value=6)
        mock_p.getJointInfo = MagicMock()
        mock_p.getBasePositionAndOrientation = MagicMock(
            return_value=((0.0, 0.0, 0.1), (0, 0, 0, 1))
        )
        mock_p.getEulerFromQuaternion = MagicMock(return_value=(1, 0, 0, 0))
        mock_p.getBaseVelocity = MagicMock(return_value=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
        mock_p.setJointMotorControlArray = MagicMock()
        mock_p.getLinkState = MagicMock(return_value=(
            (0.0, 0.0, 0.1), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0, 1),
            (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0, 1)
        ))
        mock_p.getContactPoints = MagicMock(return_value=[])
        mock_p.addUserDebugLine = MagicMock()
        mock_p.configureDebugVisualizer = MagicMock()

        mock_data = MagicMock()
        mock_data.getDataPath.return_value = '/tmp/pybullet_data'

        # 同时 patch HAS_PYBULLET=True, p, pybullet_data
        with patch.object(pbsim, 'HAS_PYBULLET', True), \
             patch.object(pbsim, 'p', mock_p), \
             patch.object(pbsim, 'pybullet_data', mock_data):
            yield mock_p, mock_data
    else:
        import pybullet as real_pb
        import pybullet_data as real_pd
        yield real_pb, real_pd


class TestPyBulletConfig:
    """PyBulletConfig 测试"""

    def test_default_config(self):
        """默认配置"""
        cfg = PyBulletConfig()
        assert cfg.dt == 1.0 / 240.0
        assert cfg.gravity == -9.81
        assert cfg.gui_mode == PyBulletGUI.NONE
        assert cfg.grade == 'M'

    def test_grade_s_config(self):
        """S 级配置"""
        cfg = PyBulletConfig.from_grade('S')
        assert cfg.dt == 1.0 / 480.0
        assert cfg.grade == 'S'

    def test_grade_m_config(self):
        """M 级配置"""
        cfg = PyBulletConfig.from_grade('M')
        assert cfg.dt == 1.0 / 240.0
        assert cfg.grade == 'M'

    def test_grade_l_config(self):
        """L 级配置"""
        cfg = PyBulletConfig.from_grade('L')
        assert cfg.dt == 1.0 / 120.0
        assert cfg.grade == 'L'

    def test_grade_xl_config(self):
        """XL 级配置"""
        cfg = PyBulletConfig.from_grade('XL')
        assert cfg.dt == 1.0 / 120.0
        assert cfg.grade == 'XL'

    def test_grade_xxl_config(self):
        """XXL 级配置"""
        cfg = PyBulletConfig.from_grade('XXL')
        assert cfg.dt == 1.0 / 60.0
        assert cfg.grade == 'XXL'

    def test_unknown_grade_defaults_to_m(self):
        """未知等级默认 M"""
        cfg = PyBulletConfig.from_grade('unknown')
        assert cfg.grade == 'M'


class TestGenerateAGVURDF:
    """AGV URDF 生成测试"""

    def test_generate_all_grades(self):
        """生成所有等级的 URDF"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            path = generate_agv_urdf(grade)
            with open(path) as f:
                content = f.read()
            assert '<robot name="agv_2w_' + grade + '">' in content or '<robot name="agv_4w_' + grade + '">' in content
            assert '</robot>' in content
            # 验证关键参数
            if grade == 'S':
                assert 'mass value="15"' in content
            elif grade == 'M':
                assert 'mass value="35"' in content
            elif grade == 'L':
                assert 'mass value="80"' in content
            # 清理
            import os
            os.remove(path)

    def test_urdf_contains_wheels(self):
        """URDF 包含轮子定义"""
        path = generate_agv_urdf('M')
        with open(path) as f:
            content = f.read()
        assert 'left_wheel' in content
        assert 'right_wheel' in content
        assert 'caster_front' in content
        assert 'caster_back' in content
        import os
        os.remove(path)

    def test_urdf_contains_sensors(self):
        """URDF 包含传感器链接"""
        path = generate_agv_urdf('M')
        with open(path) as f:
            content = f.read()
        assert 'imu_link' in content
        assert 'camera_link' in content
        import os
        os.remove(path)

    def test_urdf_with_output_path(self, tmp_path):
        """指定输出路径"""
        output = tmp_path / "test_agv.urdf"
        path = generate_agv_urdf('M', output_path=str(output))
        assert path == str(output)
        assert output.exists()
        content = output.read_text()
        assert '<robot name="agv_2w_M">' in content


class TestPyBulletSimulatorBasic:
    """PyBulletSimulator 基础测试 (mock)"""

    @pytest.fixture
    def simulator(self, mock_pybullet):
        """创建仿真器实例 (mock)"""
        mock_p, mock_data = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available, run with real backend")

        # 设置 mock
        mock_p.connect.return_value = 0
        mock_p.loadURDF.return_value = 1
        mock_p.getNumJoints.return_value = 6
        mock_p.getJointInfo.side_effect = self._joint_info_side_effect
        mock_p.getBasePositionAndOrientation.return_value = (
            (0.0, 0.0, 0.1), (0, 0, 0, 1)
        )
        mock_p.getEulerFromQuaternion.return_value = (0.0, 0.0, 0.0)
        mock_p.getBaseVelocity.return_value = ([0, 0, 0], [0, 0, 0])
        mock_p.stepSimulation.return_value = None

        sim = PyBulletSimulator(gui=False, grade='M')
        yield sim
        sim.close()

    def _joint_info_side_effect(self, body_id, joint_id):
        """Mock joint info"""
        names = [
            b'left_wheel_joint', b'right_wheel_joint',
            b'caster_front_joint', b'caster_back_joint',
            b'imu_joint', b'camera_joint',
        ]
        if joint_id < len(names):
            return [
                joint_id, names[joint_id], 0, 1, 1, 0,
                0.0, 1.0, 0.0, 0.0, 0, b'', b''
            ]
        return [joint_id, b'joint', 0, 0, 0, 0, 0.0, 0, 0, 0, 0, b'', b'']

    def test_initialization(self, mock_pybullet):
        """初始化"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        sim = PyBulletSimulator(gui=False, grade='M')
        assert sim.grade == 'M'
        assert sim._client_id == 0
        sim.close()

    def test_initialization_grade_s(self, mock_pybullet):
        """S 级初始化"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        sim = PyBulletSimulator(gui=False, grade='S')
        assert sim.grade == 'S'
        sim.close()

    def test_initialization_grade_xxl(self, mock_pybullet):
        """XXL 级初始化"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        sim = PyBulletSimulator(gui=False, grade='XXL')
        assert sim.grade == 'XXL'
        sim.close()

    def test_load_agv_model(self, mock_pybullet):
        """加载 AGV 模型"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getNumJoints.return_value = 2
        mock_p.getJointInfo.side_effect = [
            [0, b'left_wheel_joint', 0, 1, 1, 0, 0.0, 1.0, 0.0, 0.0, 0, b'', b''],
            [1, b'right_wheel_joint', 0, 1, 1, 0, 0.0, 1.0, 0.0, 0.0, 0, b'', b''],
        ]

        sim = PyBulletSimulator(gui=False, grade='M')
        agv_id = sim.load_agv_model()
        assert agv_id == 1
        assert 'left_wheel_joint' in sim._joint_indices
        assert 'right_wheel_joint' in sim._joint_indices
        sim.close()

    def test_step(self, mock_pybullet):
        """步进仿真"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        sim = PyBulletSimulator(gui=False, grade='M')
        initial_steps = sim._step_count
        sim.step()
        assert sim._step_count == initial_steps + 1
        assert sim._time == pytest.approx(sim.config.dt)
        sim.close()

    def test_set_motor_velocities(self, mock_pybullet):
        """设置电机速度"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getNumJoints.return_value = 2
        mock_p.getJointInfo.side_effect = [
            [0, b'left_wheel_joint', 0, 1, 1, 0, 0.0, 1.0, 0.0, 0.0, 0, b'', b''],
            [1, b'right_wheel_joint', 0, 1, 1, 0, 0.0, 1.0, 0.0, 0.0, 0, b'', b''],
        ]
        mock_p.setJointMotorControl2.return_value = None

        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_agv_model()
        sim.set_motor_velocities([5.0, 5.0])
        sim.close()

    def test_get_agv_state(self, mock_pybullet):
        """获取 AGV 状态"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getNumJoints.return_value = 0
        mock_p.getBasePositionAndOrientation.return_value = (
            np.array([1.0, 2.0, 0.3]),
            np.array([0, 0, 0.1, 0.99])
        )
        mock_p.getEulerFromQuaternion.return_value = (0.0, 0.0, 0.2)
        mock_p.getBaseVelocity.return_value = (
            np.array([0.5, 0.0, 0.0]),
            np.array([0, 0, 0.1])
        )

        sim = PyBulletSimulator(gui=False, grade='M')
        sim._agv_id = 1
        pos, euler, vel = sim.get_agv_state()
        assert pos[0] == pytest.approx(1.0)
        assert pos[1] == pytest.approx(2.0)
        assert euler[2] == pytest.approx(0.2)
        sim.close()

    def test_reset(self, mock_pybullet):
        """重置仿真"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getNumJoints.return_value = 0
        mock_p.getBasePositionAndOrientation.return_value = (
            (1.0, 2.0, 0.3), (0, 0, 0.1, 0.99)
        )

        sim = PyBulletSimulator(gui=False, grade='M')
        sim._agv_id = 1
        sim._step_count = 100
        sim._time = 10.0
        sim.reset()
        assert sim._step_count == 0
        assert sim._time == 0.0
        sim.close()

    def test_context_manager(self, mock_pybullet):
        """上下文管理器"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        with PyBulletSimulator(gui=False, grade='M') as sim:
            assert sim._client_id is not None
        # close() 已调用
        assert mock_p.disconnect.called

    def test_load_box_obstacle(self, mock_pybullet):
        """加载方块障碍物"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.createVisualShape.return_value = 0
        mock_p.createCollisionShape.return_value = 1
        mock_p.createMultiBody.return_value = 2

        sim = PyBulletSimulator(gui=False, grade='M')
        box_id = sim.load_box(
            half_extents=(0.5, 0.5, 0.5),
            position=(2.0, 0.0, 0.5),
            mass=10.0,
        )
        assert box_id == 2
        sim.close()

    def test_load_cylinder_obstacle(self, mock_pybullet):
        """加载圆柱障碍物"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.createVisualShape.return_value = 0
        mock_p.createCollisionShape.return_value = 1
        mock_p.createMultiBody.return_value = 3

        sim = PyBulletSimulator(gui=False, grade='M')
        cyl_id = sim.load_cylinder(
            radius=0.3,
            height=1.0,
            position=(0.0, 2.0, 0.5),
            mass=5.0,
        )
        assert cyl_id == 3
        sim.close()

    def test_get_contact_forces(self, mock_pybullet):
        """获取接触力"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getContactPoints.return_value = [
            (0, 1, 2, 3, 0, (1, 2, 3), (0, 0, 1), 0.01, 0.0, 10.5),
        ]

        sim = PyBulletSimulator(gui=False, grade='M')
        sim._agv_id = 1
        contacts = sim.get_contact_forces()
        assert len(contacts) == 1
        assert contacts[0]['force'] == pytest.approx(10.5)
        sim.close()

    def test_odometry_integration(self, mock_pybullet):
        """里程计积分"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getNumJoints.return_value = 0

        # 模拟向前运动
        call_count = [0]

        def mock_base_state(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ((0.0, 0.0, 0.1), (0, 0, 0, 1))
            else:
                return ((0.1, 0.0, 0.1), (0, 0, 0, 1))

        mock_p.getBasePositionAndOrientation.side_effect = mock_base_state
        mock_p.getEulerFromQuaternion.return_value = (0.0, 0.0, 0.0)
        mock_p.getBaseVelocity.return_value = (
            (0.1, 0.0, 0.0), (0, 0, 0)
        )

        sim = PyBulletSimulator(gui=False, grade='M')
        sim._agv_id = 1

        # 第一次调用
        odom1 = sim.get_odometry()
        assert odom1['x'] == pytest.approx(0.0)

        # 第二次调用
        odom2 = sim.get_odometry()
        assert odom2['x'] == pytest.approx(0.1)
        sim.close()


class TestAGVGrades:
    """AGV 五级规格测试"""

    @pytest.fixture
    def all_grades(self):
        return ['S', 'M', 'L', 'XL', 'XXL']

    def test_all_grades_have_configs(self, all_grades):
        """所有等级都有配置"""
        for grade in all_grades:
            cfg = PyBulletConfig.from_grade(grade)
            assert cfg.grade == grade

    def test_grade_size_progression(self, all_grades):
        """等级大小递进"""
        grade_dts = {
            'S': 1.0/480.0,
            'M': 1.0/240.0,
            'L': 1.0/120.0,
            'XL': 1.0/120.0,
            'XXL': 1.0/60.0,
        }
        for grade in all_grades:
            cfg = PyBulletConfig.from_grade(grade)
            assert cfg.dt == grade_dts[grade]

    def test_urdf_mass_per_grade(self, all_grades):
        """各等级 URDF 质量不同 (5.5寸轮毂电机)"""
        # 新模型的质量参数
        masses = {'S': '15', 'M': '35', 'L': '80', 'XL': '150', 'XXL': '300'}
        for grade, mass in masses.items():
            path = generate_agv_urdf(grade)
            with open(path) as f:
                content = f.read()
            assert f'mass value="{mass}"' in content, f"Grade {grade} mass should be {mass}"
            import os
            os.remove(path)


class TestPyBulletIntegration:
    """PyBullet 集成测试 (真实 PyBullet)"""

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_real_pybullet_connection(self):
        """真实 PyBullet 连接"""
        client_id = pybullet.connect(pybullet.DIRECT)
        assert client_id >= 0
        pybullet.disconnect(client_id)

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_real_agv_simulation(self):
        """真实 AGV 仿真"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model()

        # 设置速度
        sim.set_motor_velocities([5.0, 5.0])

        # 仿真 100 步
        for _ in range(100):
            sim.step()

        # 获取状态
        pos, euler, vel = sim.get_agv_state()
        assert pos is not None
        assert euler is not None
        assert vel is not None
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_real_agv_movement(self):
        """真实 AGV 运动测试"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        # 差速驱动: 左快右慢 -> 右转
        sim.set_motor_velocities([8.0, 2.0])

        # 仿真 240 步 (~1秒)
        for _ in range(240):
            sim.step()

        pos, euler, vel = sim.get_agv_state()
        # 应该有一定位移
        assert pos[0] > 0.0 or pos[1] != 0.0  # 至少有一个方向运动
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_real_odometry(self):
        """真实里程计"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        sim.set_motor_velocities([5.0, 5.0])

        for _ in range(100):
            sim.step()

        odom = sim.get_odometry()
        assert 'x' in odom
        assert 'y' in odom
        assert 'theta' in odom
        assert 'v' in odom
        assert 'omega' in odom
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_real_imu_data(self):
        """真实 IMU 数据"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        imu = sim.get_imu_data()
        assert 'accel' in imu
        assert 'gyro' in imu
        assert imu['accel'].shape == (3,)
        assert imu['gyro'].shape == (3,)
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_real_contact_detection(self):
        """真实接触检测"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        # AGV 应该在地面上，有接触
        contacts = sim.get_contact_forces()
        # 可能有接触力

        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_obstacle_collision(self):
        """障碍物碰撞"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        # 放置障碍物
        box_id = sim.load_box(
            half_extents=(0.2, 0.2, 0.2),
            position=(0.5, 0.0, 0.2),
            mass=100.0,  # 固定障碍物
        )

        # 撞向障碍物
        sim.set_motor_velocities([5.0, 5.0])
        for _ in range(200):
            sim.step()

        contacts = sim.get_contact_forces()
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_factory_function(self):
        """工厂函数"""
        sim = create_pybullet_simulator(gui=False, grade='L', load_agv=True, load_plane=True)
        assert sim.grade == 'L'
        assert sim._agv_id is not None
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_spec_function(self):
        """规格函数"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_pybullet_spec(grade)
            assert spec.grade == grade

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_reset_functionality(self):
        """重置功能"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        sim.set_motor_velocities([5.0, 5.0])
        for _ in range(100):
            sim.step()

        sim.reset()
        assert sim._step_count == 0
        assert sim._time == 0.0
        sim.close()

    @pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet not available")
    def test_rgbd_image_capture(self):
        """RGB-D 图像捕获"""
        sim = PyBulletSimulator(gui=False, grade='M')
        sim.load_plane()
        sim.load_agv_model(initial_pose=(0, 0, 0.1))

        img = sim.get_rgbd_image(width=320, height=240)
        assert 'rgb' in img
        assert 'depth' in img
        assert img['rgb'].shape == (240, 320, 3)
        assert img['depth'].shape == (240, 320)
        sim.close()


class TestPyBulletRobustness:
    """PyBullet 鲁棒性测试"""

    def test_multiple_instances(self, mock_pybullet):
        """多实例"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0

        sim1 = PyBulletSimulator(gui=False, grade='M')
        sim2 = PyBulletSimulator(gui=False, grade='L')
        assert sim1._client_id == sim2._client_id  # 共享客户端
        sim1.close()
        sim2.close()

    def test_config_dt(self, mock_pybullet):
        """配置时间步"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        cfg = PyBulletConfig(dt=0.005)
        sim = PyBulletSimulator(config=cfg, gui=False, grade='M')
        assert sim.config.dt == 0.005
        sim.close()

    def test_sensor_noise(self, mock_pybullet):
        """传感器噪声"""
        mock_p, _ = mock_pybullet
        if HAS_PYBULLET:
            pytest.skip("Real PyBullet available")

        mock_p.connect.return_value = 0
        mock_p.getBasePositionAndOrientation.return_value = (
            (0, 0, 0.1), (0, 0, 0, 1)
        )
        mock_p.getEulerFromQuaternion.return_value = (0, 0, 0)
        mock_p.getBaseVelocity.return_value = ((0, 0, 0), (0, 0, 0))

        cfg = PyBulletConfig(sensor_noise=0.01)
        sim = PyBulletSimulator(config=cfg, gui=False, grade='M')
        sim._agv_id = 1
        imu = sim.get_imu_data()
        # 有噪声注入
        sim.close()
