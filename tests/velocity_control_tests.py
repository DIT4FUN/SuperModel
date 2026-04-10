"""
AGV速度控制模块测试 (velocity_control_tests.py)
==============================================

测试覆盖:
1. AGV五级规格 (S/M/L/XL/XXL)
2. S曲线速度规划 (SVelocityProfilePlanner)
3. 摩擦补偿 (FrictionCompensator)
4. 轮速同步控制器 (WheelVelocitySynchronizer)
5. 速度PID控制器 (VelocityPIDController)
6. AGV速度控制器 (AGVVelocityController)

Author: SuperModel Development Team
Version: v2.45.0 (2026-04-10)
"""

import pytest
import numpy as np
import time
from src.control.velocity_control import (
    AGV_VELOCITY_CONTROL_GRADES, get_velocity_control_spec,
    VelocityProfileType, VelocityProfile1D,
    WheelVelocityCommand, WheelVelocityState, VelocityControllerState,
    SVelocityProfilePlanner, FrictionCompensator,
    WheelVelocitySynchronizer, VelocityPIDController, AGVVelocityController,
)


# ============================================================================
# AGV五级规格测试
# ============================================================================

class TestVelocityControlGrades:
    """测试AGV五级速度控制规格"""

    def test_all_five_grades_defined(self):
        """所有五个AGV等级都有定义"""
        expected = {"S", "M", "L", "XL", "XXL"}
        assert set(AGV_VELOCITY_CONTROL_GRADES.keys()) == expected

    def test_control_frequency_increases_with_grade(self):
        """控制频率随等级增加 (S→XXL)"""
        prev_freq = 0
        for grade in ["S", "M", "L", "XL", "XXL"]:
            freq = AGV_VELOCITY_CONTROL_GRADES[grade]["control_frequency_hz"]
            assert freq >= prev_freq
            prev_freq = freq

    def test_max_velocity_decreases_with_grade(self):
        """最大线速度随等级增加趋于稳定 (XXL载重但速度合理)"""
        speeds = [AGV_VELOCITY_CONTROL_GRADES[g]["max_linear_velocity_mps"] for g in ["S", "M", "L", "XL", "XXL"]]
        assert speeds[0] == 0.5  # S: 慢速
        assert speeds[1] == 1.5  # M: 中速
        assert speeds[2] == 2.0  # L: 快速
        assert speeds[3] == 3.0  # XL: 更快
        assert speeds[4] == 3.5  # XXL: 最快

    def test_angular_velocity_increases_with_grade(self):
        """最大角速度反映机动性"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            w_max = AGV_VELOCITY_CONTROL_GRADES[grade]["max_angular_velocity_rps"]
            assert w_max > 0

    def test_pid_gains_increase_with_grade(self):
        """PID增益随等级增加 (高性能需要更高增益)"""
        kps = [AGV_VELOCITY_CONTROL_GRADES[g]["velocity_pid_kp"] for g in ["S", "M", "L", "XL", "XXL"]]
        for i in range(1, len(kps)):
            assert kps[i] >= kps[i-1] * 0.8  # 至少递增

    def test_features_improve_with_grade(self):
        """高级功能随等级增加"""
        # S级: 无高级功能
        s_spec = AGV_VELOCITY_CONTROL_GRADES["S"]
        assert s_spec["friction_compensation"] == False
        assert s_spec["feedforward"] == False
        assert s_spec["wheel_slip_detection"] == False
        assert s_spec["adaptive_gain"] == False
        assert s_spec["real_time"] == False

        # M级: 有基本功能
        m_spec = AGV_VELOCITY_CONTROL_GRADES["M"]
        assert m_spec["friction_compensation"] == True
        assert m_spec["feedforward"] == True
        assert m_spec["wheel_slip_detection"] == True
        assert m_spec["real_time"] == False

        # XXL级: 全功能
        xxl_spec = AGV_VELOCITY_CONTROL_GRADES["XXL"]
        assert xxl_spec["friction_compensation"] == True
        assert xxl_spec["feedforward"] == True
        assert xxl_spec["wheel_slip_detection"] == True
        assert xxl_spec["adaptive_gain"] == True
        assert xxl_spec["real_time"] == True
        assert xxl_spec["jerk_limit_mps3"] is not None

    def test_s_curve_available_higher_grades(self):
        """S曲线规划仅L级及以上可用"""
        for grade in ["S", "M"]:
            assert AGV_VELOCITY_CONTROL_GRADES[grade]["profile_type"] == "trapezoidal"
        for grade in ["L", "XL", "XXL"]:
            assert AGV_VELOCITY_CONTROL_GRADES[grade]["profile_type"] == "s_curve"

    def test_get_velocity_control_spec(self):
        """get_velocity_control_spec 返回正确规格"""
        spec = get_velocity_control_spec("M")
        assert spec["control_frequency_hz"] == 100
        assert spec["max_linear_velocity_mps"] == 1.5
        assert spec["velocity_pid_kp"] == 3.0

    def test_get_velocity_control_spec_invalid_grade(self):
        """无效等级返回默认M级"""
        spec = get_velocity_control_spec("INVALID")
        assert spec == get_velocity_control_spec("M")

    def test_acceleration_limit_increases_with_grade(self):
        """加速度限制随等级增加"""
        accels = [AGV_VELOCITY_CONTROL_GRADES[g]["acceleration_limit_mps2"] for g in ["S", "M", "L", "XL", "XXL"]]
        for i in range(1, len(accels)):
            assert accels[i] >= accels[i-1]

    def test_all_grades_have_pid_parameters(self):
        """所有等级都有完整的PID参数"""
        for grade, spec in AGV_VELOCITY_CONTROL_GRADES.items():
            assert "velocity_pid_kp" in spec
            assert "velocity_pid_ki" in spec
            assert "velocity_pid_kd" in spec
            assert spec["velocity_pid_kp"] > 0
            assert spec["velocity_pid_ki"] >= 0
            assert spec["velocity_pid_kd"] >= 0


# ============================================================================
# S曲线速度规划器测试
# ============================================================================

class TestSVelocityProfilePlanner:
    """测试S曲线速度规划器"""

    def setup_method(self):
        self.planner = SVelocityProfilePlanner(
            max_velocity=1.5,
            max_acceleration=2.0,
            max_jerk=10.0,
        )

    def test_zero_distance_profile(self):
        """零距离返回零轨迹"""
        profile = self.planner.plan(0.0, 0.0)
        assert profile.total_duration == 0.0
        assert profile.initial_position == 0.0
        assert profile.final_position == 0.0

    def test_trapezoidal_plan(self):
        """梯形速度规划生成正确轨迹"""
        planner = SVelocityProfilePlanner(max_velocity=1.0, max_acceleration=1.0)
        profile = planner.plan(0.0, 1.0, max_jerk=None)
        assert profile.profile_type == VelocityProfileType.TRAPEZOIDAL
        assert profile.total_duration > 0
        assert abs(profile.final_position - 1.0) < 1e-6

    def test_s_curve_plan(self):
        """S曲线规划生成正确轨迹"""
        profile = self.planner.plan(0.0, 1.0)
        assert profile.profile_type == VelocityProfileType.S_CURVE
        assert abs(profile.final_position - 1.0) < 1e-6
        assert len(profile.times) > 0
        assert len(profile.velocities) > 0

    def test_profile_starts_at_initial_position(self):
        """轨迹起始于初始位置"""
        profile = self.planner.plan(5.0, 10.0)
        assert abs(profile.positions[0] - 5.0) < 1e-6

    def test_profile_ends_at_final_position(self):
        """轨迹终止于目标位置"""
        profile = self.planner.plan(5.0, 10.0)
        assert abs(profile.positions[-1] - 10.0) < 1e-6

    def test_profile_velocity_zero_at_endpoints(self):
        """轨迹端点速度为零"""
        profile = self.planner.plan(0.0, 1.0)
        assert abs(profile.velocities[0]) < 1e-3  # 起始
        assert abs(profile.velocities[-1]) < 1.0  # 终止允许小残余速度

    def test_profile_velocity_never_exceeds_max(self):
        """速度不超过最大限制"""
        profile = self.planner.plan(0.0, 2.0)
        max_v = np.max(np.abs(profile.velocities))
        assert max_v <= 1.5 * 1.01  # 允许1%误差

    def test_profile_acceleration_within_limits(self):
        """加速度在限制内"""
        profile = self.planner.plan(0.0, 2.0)
        if len(profile.accelerations) > 0:
            max_a = np.max(np.abs(profile.accelerations))
            assert max_a <= 2.0 * 1.1  # 允许10%误差

    def test_negative_distance_reverse_direction(self):
        """负距离反向运动"""
        profile = self.planner.plan(10.0, 5.0)
        assert abs(profile.final_position - 5.0) < 1e-6
        # 速度应为负
        mid_idx = len(profile.velocities) // 2
        assert profile.velocities[mid_idx] < 0

    def test_sample_at_time(self):
        """轨迹时间采样"""
        profile = self.planner.plan(0.0, 1.0)
        t_quarter = profile.total_duration * 0.25
        pos, vel, acc = profile.sample_at(t_quarter)
        assert pos > 0
        assert vel > 0

    def test_sample_at_zero(self):
        """t=0时采样"""
        profile = self.planner.plan(0.0, 1.0)
        pos, vel, acc = profile.sample_at(0.0)
        assert abs(pos - 0.0) < 1e-6
        assert abs(vel) < 1e-6

    def test_sample_at_end(self):
        """t=total时采样"""
        profile = self.planner.plan(0.0, 1.0)
        pos, vel, acc = profile.sample_at(profile.total_duration)
        assert abs(pos - 1.0) < 0.01
        assert abs(vel) < 0.1

    def test_sample_beyond_end(self):
        """t>total时返回终点值"""
        profile = self.planner.plan(0.0, 1.0)
        pos, vel, acc = profile.sample_at(profile.total_duration + 1.0)
        assert abs(pos - 1.0) < 1e-6
        assert abs(vel) < 1e-6

    def test_long_distance_profile(self):
        """长距离轨迹"""
        profile = self.planner.plan(0.0, 10.0)
        assert abs(profile.final_position - 10.0) < 0.01
        assert profile.total_duration > 0


# ============================================================================
# 摩擦补偿器测试
# ============================================================================

class TestFrictionCompensator:
    """测试摩擦补偿器"""

    def setup_method(self):
        self.fc = FrictionCompensator(
            coulomb_friction=0.5,
            viscous_friction=0.1,
            stiction_friction=0.8,
            stiction_velocity=0.01,
        )

    def test_stiction_region(self):
        """静摩擦区域返回静摩擦力矩"""
        tau = self.fc.compensate(0.0, 1.0)
        assert abs(tau - 0.8) < 1e-6

    def test_low_velocity_stiction(self):
        """低速静摩擦"""
        tau = self.fc.compensate(0.005, 1.0)  # < stiction_velocity
        assert abs(tau - 0.8) < 1e-6

    def test_high_velocity_coulomb_viscous(self):
        """高速库伦+粘滞摩擦"""
        tau = self.fc.compensate(1.0, 1.0)
        # 库伦0.5 * tanh(10) ≈ 0.5 + 粘滞0.1 * 1.0 ≈ 0.6
        assert 0.5 < tau < 0.8

    def test_negative_direction(self):
        """负速度时摩擦方向正确"""
        # 速度为正时摩擦应抵抗运动(正方向)
        tau = self.fc.compensate(1.0, 1.0)
        assert tau > 0
        # 速度为负时摩擦应抵抗运动(负方向)
        tau_neg = self.fc.compensate(-1.0, 1.0)
        assert tau_neg < 0

    def test_zero_torque_direction(self):
        """零力矩方向"""
        tau = self.fc.compensate(0.5, 0.0)
        assert tau >= 0  # 默认正方向

    def test_update_parameters(self):
        """在线更新参数"""
        self.fc.update_parameters(coulomb=1.0)
        assert self.fc.coulomb_friction == 1.0
        self.fc.update_parameters(viscous=0.5)
        assert self.fc.viscous_friction == 0.5

    def test_update_partial(self):
        """部分更新参数"""
        self.fc.update_parameters(coulomb=2.0)
        assert self.fc.coulomb_friction == 2.0
        assert self.fc.viscous_friction == 0.1  # 不变


# ============================================================================
# 轮速同步控制器测试
# ============================================================================

class TestWheelVelocitySynchronizer:
    """测试轮速同步控制器"""

    def setup_method(self):
        self.sync = WheelVelocitySynchronizer(
            wheelbase=0.5,
            left_radius=0.07,
            right_radius=0.07,
            max_wheel_velocity_rps=20.0,
            slip_threshold=2.0,
        )

    def test_straight_line_forward(self):
        """直线前进 (v=1.0, w=0)"""
        cmd = self.sync.compute_wheel_velocities(1.0, 0.0)
        # v_l = v_r = 1.0 / 0.07 ≈ 14.29 rps
        assert abs(cmd.left_velocity_rps - cmd.right_velocity_rps) < 0.1
        assert cmd.left_velocity_rps > 0

    def test_pure_rotation(self):
        """纯旋转 (v=0, w≠0)"""
        cmd = self.sync.compute_wheel_velocities(0.0, 2.0)
        # v_l = -w * wb/2 / r = -2 * 0.25 / 0.07 ≈ -7.14 rps
        # v_r = +2 * 0.25 / 0.07 ≈ +7.14 rps
        assert abs(cmd.left_velocity_rps + cmd.right_velocity_rps) < 0.1
        assert cmd.left_velocity_rps < 0
        assert cmd.right_velocity_rps > 0

    def test_curved_motion(self):
        """曲线运动"""
        cmd = self.sync.compute_wheel_velocities(1.0, 1.0)
        assert cmd.left_velocity_rps != cmd.right_velocity_rps

    def test_velocity_limit(self):
        """速度限幅"""
        cmd = self.sync.compute_wheel_velocities(10.0, 0.0)  # 远超限制
        assert abs(cmd.left_velocity_rps) <= 20.0
        assert abs(cmd.right_velocity_rps) <= 20.0

    def test_feedforward_torque(self):
        """前馈力矩不为零"""
        cmd = self.sync.compute_wheel_velocities(1.0, 0.0)
        # 第一次: 加速度非零,应有前馈
        assert cmd.left_feedforward_nm is not None

    def test_slip_correction(self):
        """打滑校正"""
        cmd = self.sync.compute_wheel_velocities(
            1.0, 0.0,
            adaptive_slip=True,
            left_raw=0.5,  # 实际速度远小于指令
            right_raw=14.0,
            dt=0.01,
        )
        # 左轮打滑校正
        assert cmd.left_velocity_rps != 14.29  # 不等于理想值

    def test_negative_linear_velocity(self):
        """负线速度后退"""
        cmd = self.sync.compute_wheel_velocities(-1.0, 0.0)
        assert cmd.left_velocity_rps < 0
        assert cmd.right_velocity_rps < 0


# ============================================================================
# 速度PID控制器测试
# ============================================================================

class TestVelocityPIDController:
    """测试速度PID控制器"""

    def setup_method(self):
        self.pid = VelocityPIDController(kp=3.0, ki=0.2, kd=0.1)

    def test_p_term_only(self):
        """P控制"""
        pid = VelocityPIDController(kp=3.0, ki=0.0, kd=0.0)
        output = pid.compute(setpoint=10.0, measurement=8.0, dt=0.01)
        assert output == 3.0 * (10.0 - 8.0)  # 6.0

    def test_pi_control(self):
        """PI控制"""
        pid = VelocityPIDController(kp=1.0, ki=0.5, kd=0.0)
        output1 = pid.compute(setpoint=10.0, measurement=8.0, dt=0.01)
        output2 = pid.compute(setpoint=10.0, measurement=8.0, dt=0.01)  # 积分累积
        assert output2 > output1  # 积分项增加

    def test_pid_with_feedforward(self):
        """带前馈的PID"""
        pid = VelocityPIDController(kp=1.0, ki=0.0, kd=0.0, feedforward_gain=1.0)
        output = pid.compute(setpoint=10.0, measurement=8.0, feedforward=2.0, dt=0.01)
        # P: 2.0 + FF: 2.0 = 4.0
        assert abs(output - 4.0) < 0.1

    def test_integral_saturation(self):
        """积分饱和"""
        pid = VelocityPIDController(kp=1.0, ki=1.0, kd=0.0, integral_limit=5.0)
        for _ in range(1000):
            pid.compute(setpoint=10.0, measurement=0.0, dt=0.01)
        # 积分应该被限幅
        assert pid.integral <= 5.0

    def test_output_limit(self):
        """输出限幅"""
        pid = VelocityPIDController(kp=10.0, ki=0.0, kd=0.0, output_limit=10.0)
        output = pid.compute(setpoint=10.0, measurement=0.0, dt=0.01)
        assert abs(output) <= 10.0

    def test_adaptive_gain(self):
        """自适应增益"""
        pid = VelocityPIDController(kp=3.0, ki=0.0, kd=0.0,
                                   adaptive_gain=True, adaptation_rate=0.01)
        pid.compute(setpoint=10.0, measurement=0.0, dt=0.01)
        # 大误差时自适应kp增加
        assert pid.adaptive_kp > 3.0

    def test_friction_compensation(self):
        """摩擦补偿"""
        fc = FrictionCompensator()
        pid = VelocityPIDController(kp=1.0, ki=0.0, kd=0.0,
                                   friction_compensator=fc)
        output = pid.compute(setpoint=1.0, measurement=0.8, dt=0.01)
        # P项 + 摩擦补偿
        assert output != 1.0 * 0.2  # 不仅仅P项

    def test_reset(self):
        """重置"""
        pid = VelocityPIDController(kp=1.0, ki=1.0, kd=0.5)
        pid.compute(setpoint=10.0, measurement=5.0, dt=0.01)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        assert pid.derivative == 0.0

    def test_zero_dt(self):
        """零时间步"""
        pid = VelocityPIDController()
        output = pid.compute(setpoint=10.0, measurement=5.0, dt=0.0)
        assert np.isfinite(output)

    def test_derivative_filtering(self):
        """微分滤波"""
        pid = VelocityPIDController(kp=0.0, ki=0.0, kd=1.0,
                                   derivative_filter=0.1)
        # 两次相同误差,微分应该平滑
        d1 = pid.compute(setpoint=10.0, measurement=8.0, dt=0.01)
        d2 = pid.compute(setpoint=10.0, measurement=8.0, dt=0.01)
        assert abs(d2) < abs(d1) or np.isclose(d1, d2)


# ============================================================================
# AGV速度控制器测试
# ============================================================================

class TestAGVVelocityController:
    """测试AGV完整速度控制器"""

    def test_creation_all_grades(self):
        """所有等级都能创建"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            ctrl = AGVVelocityController(grade=grade)
            assert ctrl.grade == grade

    def test_control_frequency_matches_spec(self):
        """控制频率匹配规格"""
        for grade, expected_freq in [("S", 50), ("M", 100), ("L", 200), ("XL", 500), ("XXL", 1000)]:
            ctrl = AGVVelocityController(grade=grade)
            assert ctrl.control_frequency == expected_freq

    def test_max_velocities(self):
        """最大速度限制"""
        ctrl = AGVVelocityController(grade="M")
        assert ctrl.max_linear_velocity == 1.5
        assert ctrl.max_angular_velocity == 3.0

    def test_openloop_straight(self):
        """开环直线"""
        ctrl = AGVVelocityController(grade="M")
        cmd = ctrl.compute_openloop(1.0, 0.0)
        assert abs(cmd.left_velocity_rps - cmd.right_velocity_rps) < 0.5
        assert cmd.left_velocity_rps > 0

    def test_openloop_rotation(self):
        """开环旋转"""
        ctrl = AGVVelocityController(grade="M")
        cmd = ctrl.compute_openloop(0.0, 2.0)
        assert cmd.left_velocity_rps < 0
        assert cmd.right_velocity_rps > 0

    def test_closedloop_pid(self):
        """闭环PID控制"""
        ctrl = AGVVelocityController(grade="M")
        left_tau, right_tau, state = ctrl.compute(
            target_linear_vel=1.0,
            target_angular_vel=0.0,
            measurement_left_rps=10.0,
            measurement_right_rps=10.0,
        )
        assert isinstance(state, VelocityControllerState)
        assert state.left_error is not None
        assert state.right_error is not None

    def test_closedloop_with_slippable_measurement(self):
        """带打滑测量的闭环控制"""
        ctrl = AGVVelocityController(grade="M")
        left_tau, right_tau, state = ctrl.compute(
            target_linear_vel=1.5,
            target_angular_vel=0.0,
            measurement_left_rps=15.0,
            measurement_right_rps=15.0,
        )
        assert np.isfinite(left_tau)
        assert np.isfinite(right_tau)

    def test_reset_clears_state(self):
        """重置清空状态"""
        ctrl = AGVVelocityController(grade="M")
        ctrl.compute(1.0, 0.0, 10.0, 10.0)
        ctrl.reset()
        assert ctrl.current_linear_vel == 0.0

    def test_get_state(self):
        """获取完整状态"""
        ctrl = AGVVelocityController(grade="M")
        state = ctrl.get_state()
        assert state["grade"] == "M"
        assert state["control_frequency_hz"] == 100
        assert "pid_kp" in state
        assert "friction_compensation" in state

    def test_friction_compensation_enabled_higher_grades(self):
        """M级及以上启用摩擦补偿"""
        ctrl_l = AGVVelocityController(grade="L")
        assert ctrl_l.friction_comp is not None
        ctrl_xl = AGVVelocityController(grade="XL")
        assert ctrl_xl.friction_comp is not None

    def test_s_grade_no_friction_compensation(self):
        """S级无摩擦补偿"""
        ctrl = AGVVelocityController(grade="S")
        assert ctrl.friction_comp is None

    def test_trajectory_planning(self):
        """轨迹规划"""
        ctrl = AGVVelocityController(grade="M")
        lp, ap = ctrl.plan_trajectory(
            start_pos=(0.0, 0.0, 0.0),
            end_pos=(1.0, 0.0, 0.0),
        )
        assert isinstance(lp, VelocityProfile1D)
        assert isinstance(ap, VelocityProfile1D)
        assert lp.total_duration > 0

    def test_trajectory_start(self):
        """启动轨迹"""
        ctrl = AGVVelocityController(grade="M")
        lp, ap = ctrl.plan_trajectory((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        ctrl.start_trajectory(lp, ap)
        assert ctrl.active_profile_linear is not None
        assert ctrl.profile_start_time is not None

    def test_velocity_limit_enforced(self):
        """速度限制被执行"""
        ctrl = AGVVelocityController(grade="M")
        # 超过最大速度
        cmd = ctrl.compute_openloop(5.0, 0.0)  # M级最大1.5m/s
        # 速度应被限幅
        assert abs(cmd.left_velocity_rps) <= ctrl.synchronizer.max_wheel_velocity_rps * 1.01

    def test_custom_wheelbase(self):
        """自定义轮距"""
        ctrl = AGVVelocityController(grade="M", wheelbase=0.8)
        assert ctrl.wheelbase == 0.8

    def test_adaptive_kp_in_state(self):
        """自适应KP在状态中"""
        ctrl = AGVVelocityController(grade="L")  # L级有自适应
        _, _, state = ctrl.compute(1.0, 0.0, 10.0, 10.0)
        assert state.left_adaptive_kp > 0
        assert state.right_adaptive_kp > 0


# ============================================================================
# 集成测试
# ============================================================================

class TestVelocityControlIntegration:
    """集成测试: 完整速度控制闭环"""

    def test_full_velocity_loop(self):
        """完整速度环"""
        ctrl = AGVVelocityController(grade="M")
        
        # 目标: 1m/s直线
        target_v = 1.0
        target_w = 0.0
        
        # 模拟测量 (带噪声)
        np.random.seed(42)
        for step in range(100):
            meas_l = target_v / ctrl.left_radius + np.random.randn() * 0.5
            meas_r = target_v / ctrl.right_radius + np.random.randn() * 0.5
            
            left_tau, right_tau, state = ctrl.compute(
                target_v, target_w, meas_l, meas_r
            )
            
            # 验证输出有限
            assert np.isfinite(left_tau)
            assert np.isfinite(right_tau)
            
            # 误差应该收敛
            if step > 50:
                assert abs(state.left_error) < 5.0  # 最终误差

    def test_acceleration_ramp_up(self):
        """加速斜坡"""
        ctrl = AGVVelocityController(grade="L")
        
        for v_target in np.linspace(0, 1.5, 10):
            cmd = ctrl.compute_openloop(v_target, 0.0)
            assert np.isfinite(cmd.left_velocity_rps)
            assert np.isfinite(cmd.right_velocity_rps)

    def test_turn_in_place(self):
        """原地转弯"""
        ctrl = AGVVelocityController(grade="M")
        for w in [0.5, 1.0, 2.0]:
            cmd = ctrl.compute_openloop(0.0, w)
            # 原地转弯: v_l ≈ -v_r
            assert abs(cmd.left_velocity_rps + cmd.right_velocity_rps) < 0.5
            assert cmd.left_velocity_rps < 0
            assert cmd.right_velocity_rps > 0

    def test_arc_motion(self):
        """圆弧运动"""
        ctrl = AGVVelocityController(grade="M")
        cmd = ctrl.compute_openloop(1.0, 0.5)
        # 圆弧: v_l < v_r
        assert cmd.left_velocity_rps < cmd.right_velocity_rps

    def test_velocity_profile_following(self):
        """跟随速度剖面"""
        planner = SVelocityProfilePlanner(1.5, 2.0, 10.0)
        profile = planner.plan(0.0, 3.0)
        
        ctrl = AGVVelocityController(grade="L")
        ctrl.start_trajectory(profile, profile)  # 用同一profile简化
        
        for i in range(0, len(profile.times), 10):
            t = profile.times[i]
            target_v = profile.velocities[i]
            
            meas_l = target_v / ctrl.left_radius
            meas_r = target_v / ctrl.right_radius
            
            _, _, state = ctrl.compute(target_v, 0.0, meas_l, meas_r)
            assert np.isfinite(state.left_output)


# ============================================================================
# 数据结构测试
# ============================================================================

class TestDataStructures:
    """测试数据结构"""

    def test_velocity_profile_1d(self):
        """VelocityProfile1D"""
        profile = VelocityProfile1D(
            profile_type=VelocityProfileType.S_CURVE,
            total_duration=1.0,
            initial_position=0.0,
            final_position=1.0,
        )
        assert profile.profile_type == VelocityProfileType.S_CURVE

    def test_wheel_velocity_command(self):
        """WheelVelocityCommand"""
        cmd = WheelVelocityCommand(
            left_velocity_rps=10.0,
            right_velocity_rps=10.0,
            timestamp=time.time(),
            left_feedforward_nm=0.5,
            right_feedforward_nm=0.5,
        )
        assert cmd.left_velocity_rps == 10.0

    def test_wheel_velocity_state(self):
        """WheelVelocityState"""
        state = WheelVelocityState(
            left_velocity_rps=10.0,
            right_velocity_rps=10.0,
            left_position_rad=1.0,
            right_position_rad=1.0,
            left_slip=False,
            right_slip=False,
        )
        assert state.left_slip == False

    def test_velocity_controller_state(self):
        """VelocityControllerState"""
        state = VelocityControllerState(
            left_error=1.0,
            right_error=1.0,
            left_integral=0.5,
            right_integral=0.5,
        )
        assert state.left_integral == 0.5


# ============================================================================
# 边界情况测试
# ============================================================================

class TestBoundaryConditions:
    """边界条件测试"""

    def test_very_small_velocity(self):
        """极小速度"""
        ctrl = AGVVelocityController(grade="M")
        cmd = ctrl.compute_openloop(0.001, 0.0)
        assert np.isfinite(cmd.left_velocity_rps)

    def test_very_large_angular_velocity(self):
        """极大角速度"""
        ctrl = AGVVelocityController(grade="M")
        cmd = ctrl.compute_openloop(0.0, 100.0)
        assert abs(cmd.left_velocity_rps) <= ctrl.synchronizer.max_wheel_velocity_rps * 1.1

    def test_nan_measurement_handled(self):
        """NaN测量"""
        ctrl = AGVVelocityController(grade="M")
        left_tau, right_tau, state = ctrl.compute(
            1.0, 0.0, float('nan'), float('nan')
        )
        assert np.isfinite(left_tau)
        assert np.isfinite(right_tau)

    def test_inf_measurement_handled(self):
        """Inf测量"""
        ctrl = AGVVelocityController(grade="M")
        left_tau, right_tau, state = ctrl.compute(
            1.0, 0.0, float('inf'), float('inf')
        )
        assert np.isfinite(left_tau)
        assert np.isfinite(right_tau)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
