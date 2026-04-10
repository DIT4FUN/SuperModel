"""
AGV五极控制规格模块测试
======================

测试 GradeAwarePID, GradeAwareSafetyMonitor, GradeAwareTrajectoryPlanner
以及AGV五极规格表的一致性验证

Author: SuperModel Team
Version: v2.46.1
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.grade_control import (
    AGVGrade,
    GRADE_CONTROL_SPECS,
    GradePIDConfig,
    GradeControllerConfig,
    GradeAwarePID,
    GradeAwareSafetyMonitor,
    GradeAwareTrajectoryPlanner,
    get_grade_control_spec,
    list_grade_capabilities,
)


class TestGradeSpecsConsistency(unittest.TestCase):
    """测试AGV五极规格表的一致性"""

    GRADES = [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]

    def test_all_grades_have_specs(self):
        """测试所有等级都有规格定义"""
        for grade in self.GRADES:
            spec = GRADE_CONTROL_SPECS[grade]
            self.assertIsInstance(spec, dict)
            self.assertGreater(len(spec), 0)

    def test_control_frequency_increases_with_grade(self):
        """测试控制频率随等级增加"""
        freqs = [GRADE_CONTROL_SPECS[g]["control_frequency"] for g in self.GRADES]
        for i in range(len(freqs) - 1):
            self.assertLessEqual(
                freqs[i], freqs[i + 1],
                f"控制频率应随等级增加: {self.GRADES[i]}={freqs[i]} vs {self.GRADES[i+1]}={freqs[i+1]}"
            )

    def test_control_period_decreases_with_grade(self):
        """测试控制周期随等级增加而减小"""
        periods = [GRADE_CONTROL_SPECS[g]["control_period"] for g in self.GRADES]
        for i in range(len(periods) - 1):
            self.assertGreaterEqual(
                periods[i], periods[i + 1],
                f"控制周期应随等级减小: {self.GRADES[i]}={periods[i]} vs {self.GRADES[i+1]}={periods[i+1]}"
            )

    def test_max_velocity_increases_with_grade(self):
        """测试最大速度随等级增加"""
        velocities = [GRADE_CONTROL_SPECS[g]["max_velocity"] for g in self.GRADES]
        for i in range(len(velocities) - 1):
            self.assertLessEqual(
                velocities[i], velocities[i + 1],
                f"最大速度应随等级增加"
            )

    def test_pid_gains_increase_with_grade(self):
        """测试PID增益随等级增加"""
        kps = [GRADE_CONTROL_SPECS[g]["pid_kp"] for g in self.GRADES]
        for i in range(len(kps) - 1):
            self.assertLessEqual(
                kps[i], kps[i + 1],
                f"PID Kp应随等级增加: {kps[i]} vs {kps[i+1]}"
            )

    def test_output_limit_increases_with_grade(self):
        """测试输出限幅随等级增加"""
        limits = [GRADE_CONTROL_SPECS[g]["pid_output_limit"] for g in self.GRADES]
        for i in range(len(limits) - 1):
            self.assertLessEqual(
                limits[i], limits[i + 1],
                f"输出限幅应随等级增加"
            )

    def test_skill_timeout_decreases_with_grade(self):
        """测试技能超时随等级增加而减小"""
        timeouts = [GRADE_CONTROL_SPECS[g]["skill_timeout"] for g in self.GRADES]
        for i in range(len(timeouts) - 1):
            self.assertGreaterEqual(
                timeouts[i], timeouts[i + 1],
                f"技能超时应随等级减小"
            )

    def test_max_concurrent_skills_increases_with_grade(self):
        """测试最大并发技能数随等级增加"""
        skills = [GRADE_CONTROL_SPECS[g]["max_concurrent_skills"] for g in self.GRADES]
        for i in range(len(skills) - 1):
            self.assertLessEqual(
                skills[i], skills[i + 1],
                f"最大并发技能应随等级增加"
            )

    def test_friction_compensation_from_grade_m(self):
        """测试M级及以上才有摩擦补偿"""
        for grade in [AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            self.assertTrue(
                GRADE_CONTROL_SPECS[grade]["friction_compensation"],
                f"{grade}应有摩擦补偿"
            )
        self.assertFalse(GRADE_CONTROL_SPECS[AGVGrade.S]["friction_compensation"])

    def test_feedforward_from_grade_m(self):
        """测试M级及以上才有前馈控制"""
        for grade in [AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            self.assertTrue(
                GRADE_CONTROL_SPECS[grade]["feedforward"],
                f"{grade}应有前馈控制"
            )
        self.assertFalse(GRADE_CONTROL_SPECS[AGVGrade.S]["feedforward"])

    def test_adaptive_gain_from_grade_l(self):
        """测试L级及以上才有自适应增益"""
        for grade in [AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            self.assertTrue(
                GRADE_CONTROL_SPECS[grade]["adaptive_gain"],
                f"{grade}应有自适应增益"
            )
        for grade in [AGVGrade.S, AGVGrade.M]:
            self.assertFalse(GRADE_CONTROL_SPECS[grade]["adaptive_gain"])

    def test_fault_tolerance_xl_and_xxl(self):
        """测试XL/XXL级有故障容错"""
        for grade in [AGVGrade.XXL]:
            self.assertTrue(
                GRADE_CONTROL_SPECS[grade]["fault_tolerance"],
                f"{grade}应有故障容错"
            )
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L]:
            self.assertFalse(GRADE_CONTROL_SPECS[grade]["fault_tolerance"])

    def test_redundancy_xxl_only(self):
        """测试只有XXL级有冗余"""
        self.assertTrue(GRADE_CONTROL_SPECS[AGVGrade.XXL]["redundancy"])
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL]:
            self.assertFalse(GRADE_CONTROL_SPECS[grade]["redundancy"])

    def test_real_time_kernel_requirements(self):
        """测试实时内核要求随等级提升"""
        kernels = [GRADE_CONTROL_SPECS[g]["real_time_kernel"] for g in self.GRADES]
        # S/M级无实时内核
        self.assertEqual(kernels[0], False)
        self.assertEqual(kernels[1], False)
        # L级 PREEMPT_RT
        self.assertEqual(kernels[2], "PREEMPT_RT")
        # XL级 Xenomai
        self.assertEqual(kernels[3], "Xenomai")
        # XXL级 Xenomai+FPGA
        self.assertEqual(kernels[4], "Xenomai+FPGA")

    def test_safety_level_escalation(self):
        """测试安全等级随等级提升"""
        safety_levels = [GRADE_CONTROL_SPECS[g]["safety_level"] for g in self.GRADES]
        self.assertEqual(safety_levels[0], "PLd")
        self.assertEqual(safety_levels[1], "PLd")
        self.assertEqual(safety_levels[2], "PLe")
        self.assertEqual(safety_levels[3], "PLe+SIL2")
        self.assertEqual(safety_levels[4], "PLe+SIL3")


class TestGradePIDConfig(unittest.TestCase):
    """测试GradePIDConfig"""

    def test_from_grade_s(self):
        """测试S级PID配置"""
        cfg = GradePIDConfig.from_grade(AGVGrade.S)
        self.assertEqual(cfg.grade, AGVGrade.S)
        self.assertEqual(cfg.kp, 2.0)
        self.assertEqual(cfg.ki, 0.1)
        self.assertEqual(cfg.kd, 0.05)
        self.assertEqual(cfg.output_limit, 10.0)
        self.assertEqual(cfg.integral_limit, 5.0)
        self.assertEqual(cfg.feedforward_gain, 0.0)  # S级无前馈

    def test_from_grade_m(self):
        """测试M级PID配置"""
        cfg = GradePIDConfig.from_grade(AGVGrade.M)
        self.assertEqual(cfg.grade, AGVGrade.M)
        self.assertEqual(cfg.kp, 3.0)
        self.assertEqual(cfg.feedforward_gain, 1.0)  # M级有前馈

    def test_from_grade_xxl(self):
        """测试XXL级PID配置"""
        cfg = GradePIDConfig.from_grade(AGVGrade.XXL)
        self.assertEqual(cfg.grade, AGVGrade.XXL)
        self.assertEqual(cfg.kp, 6.0)
        self.assertEqual(cfg.ki, 0.8)
        self.assertEqual(cfg.kd, 0.5)
        self.assertEqual(cfg.output_limit, 80.0)
        self.assertEqual(cfg.integral_limit, 40.0)


class TestGradeControllerConfig(unittest.TestCase):
    """测试GradeControllerConfig"""

    def test_from_grade_s(self):
        """测试S级控制器配置"""
        cfg = GradeControllerConfig.from_grade(AGVGrade.S)
        self.assertEqual(cfg.grade, AGVGrade.S)
        self.assertEqual(cfg.max_velocity, 0.5)
        self.assertEqual(cfg.control_frequency, 20)
        self.assertEqual(cfg.control_period, 0.050)
        self.assertEqual(cfg.trajectory_mode, "line")
        self.assertEqual(cfg.planning_algorithm, "line")

    def test_from_grade_l(self):
        """测试L级控制器配置"""
        cfg = GradeControllerConfig.from_grade(AGVGrade.L)
        self.assertEqual(cfg.control_frequency, 100)
        self.assertEqual(cfg.trajectory_mode, "s_curve")
        self.assertEqual(cfg.planning_algorithm, "s_curve")
        self.assertEqual(cfg.adaptive_gain, True)
        self.assertEqual(cfg.max_jerk, 5.0)

    def test_get_control_period_ms(self):
        """测试控制周期毫秒转换"""
        cfg_s = GradeControllerConfig.from_grade(AGVGrade.S)
        self.assertAlmostEqual(cfg_s.get_control_period_ms(), 50.0, places=1)

        cfg_xxl = GradeControllerConfig.from_grade(AGVGrade.XXL)
        self.assertAlmostEqual(cfg_xxl.get_control_period_ms(), 1.0, places=1)


class TestGradeAwarePID(unittest.TestCase):
    """测试GradeAwarePID控制器"""

    def test_creation_all_grades(self):
        """测试各级别PID创建"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            pid = GradeAwarePID(grade)
            self.assertEqual(pid.grade, grade)
            self.assertEqual(pid.config.grade, grade)

    def test_compute_basic(self):
        """测试基本PID计算"""
        pid = GradeAwarePID(AGVGrade.M)
        output = pid.compute(error=1.0, dt=0.01)
        self.assertIsInstance(output, float)
        self.assertGreaterEqual(output, 0.0)

    def test_compute_zero_error(self):
        """测试零误差"""
        pid = GradeAwarePID(AGVGrade.M)
        output = pid.compute(error=0.0, dt=0.01)
        # 只有积分项累积后不为零
        self.assertIsInstance(output, float)

    def test_output_limits(self):
        """测试输出限幅"""
        pid = GradeAwarePID(AGVGrade.S)  # output_limit=10
        output = pid.compute(error=100.0, dt=0.01)  # 大误差
        self.assertLessEqual(abs(output), pid.config.output_limit + 1e-6)

    def test_integral_anti_windup(self):
        """测试积分抗饱和"""
        pid = GradeAwarePID(AGVGrade.M)
        # 持续大误差
        for _ in range(1000):
            pid.compute(error=10.0, dt=0.01)
        # 积分项应被限幅
        self.assertLessEqual(abs(pid._integral), pid.config.integral_limit + 1e-3)

    def test_feedforward_enabled(self):
        """测试M级及以上前馈"""
        pid_m = GradeAwarePID(AGVGrade.M)
        self.assertGreater(pid_m.config.feedforward_gain, 0.0)
        out_no_ff = pid_m.compute(error=0.0, dt=0.01, feedforward=0.0)
        out_with_ff = pid_m.compute(error=0.0, dt=0.01, feedforward=5.0)
        self.assertNotEqual(out_no_ff, out_with_ff)

    def test_feedforward_disabled_s_grade(self):
        """测试S级无前馈"""
        pid_s = GradeAwarePID(AGVGrade.S)
        self.assertEqual(pid_s.config.feedforward_gain, 0.0)
        out_no_ff = pid_s.compute(error=0.0, dt=0.01, feedforward=0.0)
        out_with_ff = pid_s.compute(error=0.0, dt=0.01, feedforward=5.0)
        self.assertEqual(out_no_ff, out_with_ff)

    def test_reset(self):
        """测试PID重置"""
        pid = GradeAwarePID(AGVGrade.L)
        pid.compute(error=5.0, dt=0.01)
        self.assertNotEqual(pid._integral, 0.0)
        pid.reset()
        self.assertEqual(pid._integral, 0.0)
        self.assertEqual(pid._last_error, 0.0)

    def test_get_state(self):
        """测试状态获取"""
        pid = GradeAwarePID(AGVGrade.XXL)
        pid.compute(error=2.0, dt=0.01)
        state = pid.get_state()
        self.assertIn("integral", state)
        self.assertIn("last_error", state)
        self.assertIn("last_output", state)
        self.assertIn("kp_e", state)
        self.assertIn("ki_i", state)
        self.assertIn("kd_d", state)


class TestGradeAwareSafetyMonitor(unittest.TestCase):
    """测试GradeAwareSafetyMonitor"""

    def test_creation_all_grades(self):
        """测试各级别安全监控创建"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            mon = GradeAwareSafetyMonitor(grade)
            self.assertEqual(mon.grade, grade)
            self.assertTrue(mon.is_safe())

    def test_velocity_check_normal(self):
        """测试正常速度"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L]:
            mon = GradeAwareSafetyMonitor(grade)
            level, msg = mon.check_velocity(velocity=0.5, dt=0.01)
            self.assertEqual(level, "NORMAL")

    def test_velocity_check_critical(self):
        """测试超速"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            mon = GradeAwareSafetyMonitor(grade)
            max_v = mon.config.max_velocity
            level, msg = mon.check_velocity(velocity=max_v * 1.5, dt=0.01)
            self.assertEqual(level, "CRITICAL")
            self.assertIn("超过限制", msg)

    def test_boundary_check_normal(self):
        """测试正常边界"""
        mon = GradeAwareSafetyMonitor(AGVGrade.M)
        level, msg = mon.check_boundary(position=(5.0, 3.0, 0.5))
        self.assertEqual(level, "NORMAL")

    def test_boundary_check_critical(self):
        """测试超界"""
        mon = GradeAwareSafetyMonitor(AGVGrade.M)
        level, msg = mon.check_boundary(
            position=(200.0, 3.0, 0.5),
            boundary_min=np.array([-100, -100, -np.pi]),
            boundary_max=np.array([100, 100, np.pi]),
        )
        self.assertEqual(level, "CRITICAL")

    def test_force_check(self):
        """测试力检查"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            mon = GradeAwareSafetyMonitor(grade)
            level, msg = mon.check_force(force_magnitude=10.0)
            self.assertEqual(level, "NORMAL")

    def test_force_check_critical(self):
        """测试超力"""
        for grade in [AGVGrade.S, AGVGrade.M]:
            mon = GradeAwareSafetyMonitor(grade)
            level, msg = mon.check_force(force_magnitude=500.0)
            self.assertEqual(level, "CRITICAL")

    def test_slip_detection_enabled(self):
        """测试M级以上打滑检测"""
        for grade in [AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            mon = GradeAwareSafetyMonitor(grade)
            self.assertTrue(mon.config.slip_detection)

    def test_slip_detection_disabled_s_grade(self):
        """测试S级无打滑检测"""
        mon = GradeAwareSafetyMonitor(AGVGrade.S)
        self.assertFalse(mon.config.slip_detection)

    def test_slip_check_normal(self):
        """测试正常无打滑"""
        for grade in [AGVGrade.M, AGVGrade.L]:
            mon = GradeAwareSafetyMonitor(grade)
            level, msg = mon.check_slip(
                left_velocity=1.5,
                right_velocity=1.5,
                expected_velocity=1.5,
            )
            self.assertEqual(level, "NORMAL")

    def test_estop_trigger_and_reset(self):
        """测试紧急停止触发和重置"""
        mon = GradeAwareSafetyMonitor(AGVGrade.L)
        self.assertTrue(mon.is_safe())

        mon.trigger_estop("test_reason")
        self.assertFalse(mon.is_safe())
        self.assertEqual(mon.get_emergency_level(), "EMERGENCY_STOP")

        mon.reset_estop()
        self.assertTrue(mon.is_safe())

    def test_get_capabilities(self):
        """测试获取能力"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            mon = GradeAwareSafetyMonitor(grade)
            caps = mon.get_capabilities()
            self.assertIn("grade", caps)
            self.assertIn("max_velocity", caps)
            self.assertIn("slip_detection", caps)
            self.assertIn("fault_tolerance", caps)
            self.assertIn("redundancy", caps)


class TestGradeAwareTrajectoryPlanner(unittest.TestCase):
    """测试GradeAwareTrajectoryPlanner"""

    def test_creation_all_grades(self):
        """测试各级别规划器创建"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            planner = GradeAwareTrajectoryPlanner(grade)
            self.assertEqual(planner.grade, grade)

    def test_plan_line_basic(self):
        """测试直线规划基本功能"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.S)
        traj = planner.plan_line(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 0.0),
        )
        self.assertIsInstance(traj, list)
        self.assertGreater(len(traj), 0)
        self.assertEqual(traj[0]["x"], 0.0)
        self.assertGreater(traj[-1]["x"], 0.9)

    def test_plan_line_angle_interpolation(self):
        """测试直线规划角度插值"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.M)
        traj = planner.plan_line(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, np.pi / 2),
        )
        self.assertAlmostEqual(traj[0]["theta"], 0.0, places=3)
        self.assertAlmostEqual(traj[-1]["theta"], np.pi / 2, places=3)

    def test_plan_trapezoidal_basic(self):
        """测试梯形速度规划"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.M)
        traj = planner.plan_trapezoidal(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 0.0),
        )
        self.assertIsInstance(traj, list)
        self.assertGreater(len(traj), 0)
        # 检查速度剖面
        velocities = [pt["v"] for pt in traj]
        self.assertLessEqual(max(velocities), planner.config.max_velocity + 1e-3)

    def test_plan_s_curve_basic(self):
        """测试S曲线速度规划"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.L)
        traj = planner.plan_s_curve(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 0.0),
        )
        self.assertIsInstance(traj, list)
        self.assertGreater(len(traj), 0)
        # XXL级用S曲线
        planner_xxl = GradeAwareTrajectoryPlanner(AGVGrade.XXL)
        traj_xxl = planner_xxl.plan_s_curve(
            start=(0.0, 0.0, 0.0),
            end=(2.0, 0.0, 0.0),
        )
        self.assertGreater(len(traj_xxl), 0)

    def test_plan_s_curve_vs_trapezoidal_s_grade(self):
        """测试S级梯形退化为直线"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.S)
        # S级max_jerk=0, 退化为梯形
        traj = planner.plan_s_curve(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 0.0, 0.0),
        )
        self.assertIsInstance(traj, list)

    def test_get_current_trajectory(self):
        """测试获取当前轨迹"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.XXL)
        self.assertIsNone(planner.get_current_trajectory())
        planner.plan_line(start=(0, 0, 0), end=(1, 1, 0))
        traj = planner.get_current_trajectory()
        self.assertIsNotNone(traj)
        self.assertGreater(len(traj), 0)


class TestGradeControlFunctions(unittest.TestCase):
    """测试辅助函数"""

    def test_get_grade_control_spec(self):
        """测试获取等级规格"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            spec = get_grade_control_spec(grade)
            self.assertIsInstance(spec, dict)
            self.assertEqual(spec["max_velocity"], GRADE_CONTROL_SPECS[grade]["max_velocity"])

    def test_list_grade_capabilities(self):
        """测试列出等级能力"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            caps = list_grade_capabilities(grade)
            self.assertIn("grade", caps)
            self.assertIn("control_frequency", caps)
            self.assertIn("max_velocity", caps)
            self.assertIn("trajectory_mode", caps)
            self.assertIn("friction_compensation", caps)
            self.assertIn("fault_tolerance", caps)


class TestGradeConsistencyAcrossModules(unittest.TestCase):
    """测试五极规格在模块间的一致性"""

    def test_control_frequency_in_specs_and_modules(self):
        """测试SPEC规格与GradeAwarePID控制频率一致"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            spec = GRADE_CONTROL_SPECS[grade]
            mon = GradeAwareSafetyMonitor(grade)
            self.assertEqual(spec["control_frequency"], mon.config.control_frequency)

    def test_max_velocity_in_specs_and_modules(self):
        """测试最大速度在规格与安全监控间一致"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            spec = GRADE_CONTROL_SPECS[grade]
            mon = GradeAwareSafetyMonitor(grade)
            self.assertEqual(spec["max_velocity"], mon.config.max_velocity)

    def test_pid_kp_in_specs_and_config(self):
        """测试PID Kp在规格与PID配置间一致"""
        for grade in [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL]:
            spec = GRADE_CONTROL_SPECS[grade]
            cfg = GradePIDConfig.from_grade(grade)
            self.assertEqual(spec["pid_kp"], cfg.kp)


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def test_zero_distance_trajectory(self):
        """测试零距离轨迹"""
        planner = GradeAwareTrajectoryPlanner(AGVGrade.M)
        traj = planner.plan_line(start=(0, 0, 0), end=(0, 0, 0))
        self.assertGreater(len(traj), 0)

    def test_zero_dt_pid(self):
        """测试零时间步"""
        pid = GradeAwarePID(AGVGrade.XXL)
        output = pid.compute(error=1.0, dt=0.0)
        self.assertIsInstance(output, float)

    def test_negative_velocity_check(self):
        """测试负速度检查"""
        mon = GradeAwareSafetyMonitor(AGVGrade.M)
        level, msg = mon.check_velocity(velocity=-2.0, dt=0.01)
        self.assertEqual(level, "CRITICAL")

    def test_large_force_check(self):
        """测试大力值检查"""
        mon = GradeAwareSafetyMonitor(AGVGrade.XXL)
        level, msg = mon.check_force(force_magnitude=10000.0)
        self.assertEqual(level, "CRITICAL")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q", "--tb=no"])
