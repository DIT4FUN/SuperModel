"""
MPC 控制器测试
==============

测试 MPC 模型预测控制器的各个功能:
- 动力学模型
- 关节空间 MPC
- 笛卡尔空间 MPC
- 各 AGV 等级配置
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from control.mpc import (
    MPCConfig, DynamicsModel, JointStateMP,
    JointSpaceMPC, CartesianMPC, get_mpc_spec
)


class TestDynamicsModel:
    """动力学模型测试"""

    def test_dynamics_init(self):
        model = DynamicsModel(num_joints=6)
        assert model.n == 6
        assert len(model.M_diag) == 6
        assert len(model.damping) == 6

    def test_forward(self):
        model = DynamicsModel(num_joints=6)
        q = np.zeros(6)
        qd = np.zeros(6)
        tau = np.array([0, 0, 10, 0, 0, 0])  # 施加力矩
        qdd = model.forward(q, qd, tau)
        assert qdd.shape == (6,)
        assert qdd[2] > 0  # z 轴有关重力矩

    def test_forward_with_velocity(self):
        model = DynamicsModel(num_joints=6)
        q = np.zeros(6)
        qd = np.array([1, 0, 0, 0, 0, 0])
        tau = np.zeros(6)
        qdd = model.forward(q, qd, tau)
        assert qdd.shape == (6,)
        assert qdd[0] < 0  # 阻尼导致减速

    def test_linearize(self):
        model = DynamicsModel(num_joints=6)
        q = np.zeros(6)
        qd = np.zeros(6)
        A, B, G = model.linearize(q, qd)
        assert A.shape == (12, 12)
        assert B.shape == (12, 6)
        assert G.shape == (12,)

    def test_discrete_matrices(self):
        model = DynamicsModel(num_joints=6)
        q = np.zeros(6)
        qd = np.zeros(6)
        Ad, Bd = model.discrete_matrices(q, qd, dt=0.01)
        assert Ad.shape == (12, 12)
        assert Bd.shape == (12, 6)


class TestMPCConfig:
    """MPC 配置测试"""

    def test_default_config(self):
        cfg = MPCConfig()
        assert cfg.horizon == 20
        assert cfg.control_horizon == 10
        assert cfg.dt == 0.01
        assert cfg.grade == 'M'

    def test_grade_S(self):
        cfg = MPCConfig.for_grade('S', num_joints=6)
        assert cfg.horizon == 10
        assert cfg.control_horizon == 5
        assert cfg.dt == 0.02

    def test_grade_XXL(self):
        cfg = MPCConfig.for_grade('XXL', num_joints=6)
        assert cfg.horizon == 50
        assert cfg.control_horizon == 25
        assert cfg.solver == 'osqp'

    def test_custom_weights(self):
        cfg = MPCConfig(
            Q_pos=np.ones(6) * 200,
            Q_vel=np.ones(6) * 50,
            R_acc=np.ones(6) * 0.5
        )
        assert cfg.Q_pos[0] == 200
        assert cfg.Q_vel[0] == 50
        assert cfg.R_acc[0] == 0.5


class TestJointSpaceMPC:
    """关节空间 MPC 测试"""

    def test_init(self):
        mpc = JointSpaceMPC(num_joints=6)
        assert mpc.n == 6
        assert mpc.state_dim == 12
        assert mpc.control_dim == 6

    def test_compute_control_simple(self):
        mpc = JointSpaceMPC(num_joints=6)
        current_pos = np.zeros(6)
        current_vel = np.zeros(6)
        target_pos = np.array([0.5, 0, 0, 0, 0, 0])

        tau = mpc.compute_control_simple(current_pos, current_vel, target_pos)
        assert tau.shape == (6,)
        assert np.all(np.isfinite(tau))

    def test_compute_control_with_nonzero_current(self):
        mpc = JointSpaceMPC(num_joints=6)
        current_pos = np.array([0.3, 0.1, 0, 0, 0, 0])
        current_vel = np.array([0.1, 0.05, 0, 0, 0, 0])
        target_pos = np.array([0.5, 0.2, 0, 0, 0, 0])

        tau = mpc.compute_control_simple(current_pos, current_vel, target_pos)
        assert tau.shape == (6,)

    def test_compute_control_with_torque_limits(self):
        cfg = MPCConfig(torque_limits=np.ones(6) * 50.0)
        mpc = JointSpaceMPC(config=cfg, num_joints=6)
        current_pos = np.zeros(6)
        current_vel = np.zeros(6)
        target_pos = np.ones(6) * np.pi

        tau = mpc.compute_control_simple(current_pos, current_vel, target_pos)
        assert np.all(np.abs(tau) <= 50.0 + 1e-6)

    def test_compute_control_trajectory(self):
        mpc = JointSpaceMPC(num_joints=6)
        current_pos = np.zeros(6)
        current_vel = np.zeros(6)

        # 创建期望轨迹
        horizon = 20
        target_traj = np.tile(np.array([0.5, 0.2, 0.1, 0, 0, 0]), (horizon, 1))
        target_traj[:, 0] = np.linspace(0, 0.5, horizon)

        tau = mpc.compute_control(current_pos, target_traj, current_vel)
        assert tau.shape == (6,)

    def test_reset(self):
        mpc = JointSpaceMPC(num_joints=6)
        mpc.predicted_states = [np.zeros(12)]
        mpc.reset()
        assert len(mpc.predicted_states) == 0


class TestCartesianMPC:
    """笛卡尔空间 MPC 测试"""

    def test_init(self):
        mpc = CartesianMPC(num_joints=6)
        assert mpc.n == 6
        assert mpc.joint_mpc is not None

    def test_forward_kinematics(self):
        mpc = CartesianMPC(num_joints=6)
        q = np.zeros(6)
        pose = mpc.forward_kinematics(q)
        assert pose.shape == (6,)
        assert pose[2] > 0  # z > 0

    def test_jacobian_approx(self):
        mpc = CartesianMPC(num_joints=6)
        q = np.zeros(6)
        J = mpc.jacobian_approx(q)
        assert J.shape == (3, 6)

    def test_compute_control(self):
        mpc = CartesianMPC(num_joints=6)
        current_joint_pos = np.zeros(6)
        current_joint_vel = np.zeros(6)
        target_pose = np.array([0.3, 0.1, 0.5, 0.1, 0.0, 0.0])

        tau = mpc.compute_control(current_joint_pos, current_joint_vel, target_pose)
        assert tau.shape == (6,)


class TestGetMPCSpec:
    """MPC 规格表测试"""

    def test_all_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_mpc_spec(grade)
            assert 'horizon' in spec
            assert 'control_horizon' in spec
            assert 'dt' in spec
            assert 'max_torque' in spec
            assert 'solver' in spec

    def test_grade_increasing_horizon(self):
        horizons = [get_mpc_spec(g)['horizon'] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        assert horizons == sorted(horizons)  # 单调递增

    def test_grade_specific_values_S(self):
        spec = get_mpc_spec('S')
        assert spec['horizon'] == 10
        assert spec['control_horizon'] == 5
        assert spec['dt'] == 0.02

    def test_grade_specific_values_XXL(self):
        spec = get_mpc_spec('XXL')
        assert spec['horizon'] == 50
        assert spec['control_horizon'] == 25
        assert spec['dt'] == 0.002
        assert spec['solver'] == 'osqp'


class TestJointStateMP:
    """关节状态测试"""

    def test_init(self):
        state = JointStateMP(
            position=np.zeros(6),
            velocity=np.zeros(6)
        )
        assert state.position.shape == (6,)
        assert state.velocity.shape == (6,)
        assert state.acceleration.shape == (6,)

    def test_with_acceleration(self):
        state = JointStateMP(
            position=np.zeros(6),
            velocity=np.zeros(6),
            acceleration=np.ones(6)
        )
        assert np.all(state.acceleration == 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
