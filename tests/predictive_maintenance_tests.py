"""
预测性维护模块测试
================

测试 PredictiveMaintenanceSystem、MotorHealthMonitor、
BatterySOHEstimator、WheelHealthMonitor 及其 AGV 五级规格
"""

import unittest
import numpy as np
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from hardware.predictive_maintenance import (
    PredictiveMaintenanceSystem, MotorHealthMonitor, BatterySOHEstimator, WheelHealthMonitor,
    HealthLevel, FaultType, AGVHealthReport,
    MotorHealthMetrics, BatteryHealthMetrics, WheelHealthMetrics,
    get_predictive_maintenance_spec, AGV_PREDICTIVE_MAINTENANCE_GRADES,
    create_predictive_maintenance_system
)


class TestHealthLevelEnum(unittest.TestCase):
    """测试健康等级枚举"""

    def test_health_levels(self):
        self.assertEqual(HealthLevel.CRITICAL, 0)
        self.assertEqual(HealthLevel.FAULT, 1)
        self.assertEqual(HealthLevel.WARNING, 2)
        self.assertEqual(HealthLevel.DEGRADED, 3)
        self.assertEqual(HealthLevel.HEALTHY, 4)

    def test_health_level_ordering(self):
        self.assertLess(HealthLevel.CRITICAL, HealthLevel.FAULT)
        self.assertLess(HealthLevel.FAULT, HealthLevel.WARNING)
        self.assertLess(HealthLevel.WARNING, HealthLevel.DEGRADED)
        self.assertLess(HealthLevel.DEGRADED, HealthLevel.HEALTHY)


class TestFaultTypeEnum(unittest.TestCase):
    """测试故障类型枚举"""

    def test_fault_types_defined(self):
        self.assertEqual(FaultType.NONE, 0)
        self.assertEqual(FaultType.MOTOR_BEARING_WEAR, 1)
        self.assertEqual(FaultType.MOTOR_OVERHEATING, 2)
        self.assertEqual(FaultType.MOTOR_STALL, 3)
        self.assertEqual(FaultType.BATTERY_SOH_LOW, 4)
        self.assertEqual(FaultType.WHEEL_SLIP, 6)
        self.assertEqual(FaultType.WHEEL_MISALIGNMENT, 7)


class TestMotorHealthMonitor(unittest.TestCase):
    """测试电机健康监测器"""

    def setUp(self):
        self.monitor = MotorHealthMonitor(
            motor_id="test_motor",
            rated_current=10.0,
            rated_power=500.0,
            thermal_time_constant=300.0,
            sample_rate=200.0,
            bearing_l10_life=20000.0,
        )

    def test_creation(self):
        self.assertEqual(self.monitor.motor_id, "test_motor")
        self.assertEqual(self.monitor.rated_current, 10.0)
        self.assertIsInstance(self.monitor.metrics, MotorHealthMetrics)

    def test_normal_operation_update(self):
        """正常工况更新"""
        for _ in range(50):
            self.monitor.update(
                current=2.0, voltage=48.0, speed=1.0, dt=0.01, ambient_temp=25.0
            )
        self.assertEqual(self.monitor.metrics.health_level, HealthLevel.HEALTHY)
        self.assertEqual(self.monitor.metrics.fault_type, FaultType.NONE)
        self.assertGreaterEqual(self.monitor.metrics.efficiency, 0.0)
        self.assertLessEqual(self.monitor.metrics.efficiency, 1.0)

    def test_overcurrent_condition(self):
        """过流工况"""
        for _ in range(100):
            self.monitor.update(
                current=20.0, voltage=48.0, speed=0.01, dt=0.01, ambient_temp=25.0
            )
        # 高电流(20A, ratio=2.0) + 极低速(0.01) → 堵转风险持续增加
        self.assertGreater(self.monitor.metrics.stall_probability, 0.0)

    def test_thermal_rise(self):
        """温度上升"""
        initial_temp = self.monitor.metrics.winding_temp
        for _ in range(200):
            self.monitor.update(
                current=8.0, voltage=48.0, speed=2.0, dt=0.01, ambient_temp=25.0
            )
        self.assertGreater(self.monitor.metrics.winding_temp, initial_temp)

    def test_bearing_wear_accumulation(self):
        """轴承磨损累积"""
        initial_wear = self.monitor.metrics.bearing_wear_index
        for _ in range(500):
            self.monitor.update(
                current=3.0, voltage=48.0, speed=1.5, dt=0.01, ambient_temp=25.0
            )
        self.assertGreaterEqual(
            self.monitor.metrics.bearing_wear_index, initial_wear
        )
        self.assertLessEqual(self.monitor.metrics.bearing_wear_index, 1.0)

    def test_metrics_bounds(self):
        """指标边界检查"""
        for _ in range(100):
            self.monitor.update(
                current=5.0, voltage=48.0, speed=1.0, dt=0.01, ambient_temp=30.0
            )
        m = self.monitor.metrics
        self.assertGreaterEqual(m.bearing_wear_index, 0.0)
        self.assertLessEqual(m.bearing_wear_index, 1.0)
        self.assertGreaterEqual(m.stall_probability, 0.0)
        self.assertLessEqual(m.stall_probability, 1.0)
        self.assertGreaterEqual(m.efficiency, 0.0)
        self.assertLessEqual(m.efficiency, 1.0)
        self.assertGreater(m.winding_temp, 0.0)


class TestBatterySOHEstimator(unittest.TestCase):
    """测试电池 SOH 估计器"""

    def setUp(self):
        self.estimator = BatterySOHEstimator(
            nominal_capacity=40.0,
            nominal_voltage=48.0,
            chemistry="Li-ion",
            initial_soh=100.0,
        )

    def test_creation(self):
        self.assertEqual(self.estimator.nominal_capacity, 40.0)
        self.assertEqual(self.estimator.nominal_voltage, 48.0)
        self.assertEqual(self.estimator.metrics.soh, 100.0)

    def test_soh_degradation(self):
        """SOH 衰减"""
        for i in range(100):
            self.estimator.update(
                voltage=52.0 - i * 0.02,
                current=-5.0,
                soc=0.8 - i * 0.002,
                temperature=30.0,
                dt=3600.0,
            )
        self.assertLess(self.estimator.metrics.soh, 100.0)
        self.assertGreater(self.estimator.metrics.soh, 50.0)

    def test_cycle_counting(self):
        """循环计数"""
        self.estimator.update(voltage=54.0, current=10.0, soc=0.2, temperature=25.0, dt=100.0)
        # 模拟充电循环
        for _ in range(20):
            self.estimator.update(voltage=52.0, current=5.0, soc=0.8, temperature=25.0, dt=100.0)
        self.assertGreaterEqual(self.estimator.metrics.cycle_count, 0)

    def test_temperature_effect(self):
        """温度影响"""
        # 高温
        self.estimator.update(voltage=52.0, current=0.0, soc=0.5, temperature=50.0, dt=1.0)
        soh_high_temp = self.estimator.metrics.soh
        # 正常温度
        self.estimator2 = BatterySOHEstimator(initial_soh=100.0)
        self.estimator2.update(voltage=52.0, current=0.0, soc=0.5, temperature=25.0, dt=1.0)
        soh_normal = self.estimator2.metrics.soh
        self.assertLessEqual(soh_high_temp, soh_normal)

    def test_internal_resistance_estimation(self):
        """内阻估计"""
        for _ in range(50):
            self.estimator.update(
                voltage=50.0, current=5.0, soc=0.8, temperature=30.0, dt=1.0
            )
        self.assertGreaterEqual(self.estimator.metrics.internal_resistance, 0.01)
        self.assertLessEqual(self.estimator.metrics.internal_resistance, 0.5)

    def test_remaining_cycles(self):
        """剩余循环预估"""
        for _ in range(50):
            self.estimator.update(
                voltage=52.0, current=5.0, soc=0.8, temperature=25.0, dt=3600.0
            )
        self.assertGreater(self.estimator.metrics.estimated_remaining_cycles, 0)

    def test_battery_health_levels(self):
        """电池健康等级"""
        # soh=95.0 → HEALTHY (not < 95)
        self.estimator.metrics.soh = 95.0
        self.estimator._evaluate_battery_health()
        self.assertEqual(self.estimator.metrics.health_level, HealthLevel.HEALTHY)

        # soh=90.0 → DEGRADED (85 <= 90 < 95)
        self.estimator.metrics.soh = 90.0
        self.estimator._evaluate_battery_health()
        self.assertEqual(self.estimator.metrics.health_level, HealthLevel.DEGRADED)

        # soh=70.0 → FAULT (60 <= 70 < 75)
        self.estimator.metrics.soh = 70.0
        self.estimator._evaluate_battery_health()
        self.assertEqual(self.estimator.metrics.health_level, HealthLevel.FAULT)

        # soh=50.0 → CRITICAL (50 < 60)
        self.estimator.metrics.soh = 50.0
        self.estimator._evaluate_battery_health()
        self.assertEqual(self.estimator.metrics.health_level, HealthLevel.CRITICAL)

        # soh=75.0 → WARNING (75 <= 75 < 85, and 75.0 is not < 75.0)
        self.estimator.metrics.soh = 75.0
        self.estimator._evaluate_battery_health()
        self.assertEqual(self.estimator.metrics.health_level, HealthLevel.WARNING)


class TestWheelHealthMonitor(unittest.TestCase):
    """测试车轮健康监测器"""

    def setUp(self):
        self.monitor = WheelHealthMonitor(
            wheel_base=0.5,
            wheel_radius=0.1,
            num_wheels=4,
            drive_type="DIFFERENTIAL",
        )

    def test_creation(self):
        self.assertEqual(self.monitor.num_wheels, 4)
        self.assertIsInstance(self.monitor.metrics, WheelHealthMetrics)

    def test_normal_rolling(self):
        """正常滚动"""
        for i in range(100):
            self.monitor.update(
                encoder_counts=[1000 + i, 1000 + i, 500, 500],
                wheel_speeds=[1.0, 1.0, 0.0, 0.0],
                position=(float(i) * 0.01, 0.0, 0.0),
                reference_position=None,
                dt=0.01,
            )
        self.assertLessEqual(self.monitor.metrics.slip_ratio, 1.0)
        self.assertGreaterEqual(self.monitor.metrics.slip_ratio, 0.0)

    def test_slip_detection(self):
        """打滑检测"""
        # 模拟打滑工况: 编码器速度 vs 轮速不匹配
        for i in range(50):
            self.monitor.update(
                encoder_counts=[100, 2000, 100, 2000],  # 右侧速度突增
                wheel_speeds=[1.0, 1.0, 0.0, 0.0],
                position=(0.0, 0.0, 0.0),
                reference_position=None,
                dt=0.01,
            )
        # 打滑率应该大于 0
        self.assertGreater(self.monitor.metrics.slip_ratio, 0.0)

    def test_misalignment_detection(self):
        """对中误差检测"""
        for i in range(100):
            self.monitor.update(
                encoder_counts=[1000 + i, 500 + i // 2, 0, 0],
                wheel_speeds=[1.5, 0.8, 0.0, 0.0],
                position=(float(i) * 0.01, 0.0, 0.0),
                reference_position=None,
                dt=0.01,
            )
        self.assertGreaterEqual(self.monitor.metrics.alignment_error, 0.0)

    def test_odometry_drift_no_reference(self):
        """无参考时的里程计漂移估计"""
        for i in range(50):
            self.monitor.update(
                encoder_counts=[100, 100, 100, 100],
                wheel_speeds=[1.0, 1.0, 0.0, 0.0],
                position=(float(i) * 0.01, float(i % 5) * 0.001, 0.0),
                reference_position=None,
                dt=0.01,
            )

    def test_metrics_bounds(self):
        """指标边界检查"""
        for _ in range(50):
            self.monitor.update(
                encoder_counts=[1000, 1000, 500, 500],
                wheel_speeds=[1.0, 1.0, 0.0, 0.0],
                position=(0.0, 0.0, 0.0),
                reference_position=None,
                dt=0.01,
            )
        m = self.monitor.metrics
        self.assertGreaterEqual(m.slip_ratio, 0.0)
        self.assertLessEqual(m.slip_ratio, 1.0)
        self.assertGreaterEqual(m.alignment_error, 0.0)


class TestPredictiveMaintenanceSystem(unittest.TestCase):
    """测试预测性维护系统"""

    def setUp(self):
        self.system = PredictiveMaintenanceSystem(grade="M")
        self.system.add_motor("drive_left", rated_current=10.0, rated_power=500.0)
        self.system.add_motor("drive_right", rated_current=10.0, rated_power=500.0)
        self.system.set_battery(nominal_capacity=40.0, nominal_voltage=48.0)
        self.system.set_wheel_monitor(num_wheels=4, drive_type="DIFFERENTIAL")

    def test_creation(self):
        self.assertEqual(self.system.grade, "M")
        self.assertEqual(len(self.system._motor_monitors), 2)

    def test_update_generates_report(self):
        """更新生成健康报告"""
        report = self.system.update(timestamp=time.time())
        self.assertIsInstance(report, AGVHealthReport)
        self.assertEqual(len(report.motor_metrics), 2)
        self.assertIsNotNone(report.battery_metrics)
        self.assertIsNotNone(report.wheel_metrics)

    def test_overall_score_calculation(self):
        """整体健康分计算"""
        # 正常运行
        for _ in range(100):
            self.system._motor_monitors["drive_left"].update(
                current=2.0, voltage=48.0, speed=1.0, dt=0.01, ambient_temp=25.0
            )
            self.system._motor_monitors["drive_right"].update(
                current=2.0, voltage=48.0, speed=1.0, dt=0.01, ambient_temp=25.0
            )
            self.system._battery_estimator.update(
                voltage=52.0, current=2.0, soc=0.8, temperature=28.0, dt=1.0
            )
            self.system._wheel_monitor.update(
                encoder_counts=[1000, 1000, 500, 500],
                wheel_speeds=[1.0, 1.0, 0.0, 0.0],
                position=(0.0, 0.0, 0.0),
                dt=0.01,
            )

        report = self.system.update()
        self.assertGreater(report.overall_score, 0.0)
        self.assertLessEqual(report.overall_score, 100.0)
        self.assertGreaterEqual(report.overall_score, 50.0)

    def test_health_level_assignment(self):
        """健康等级赋值"""
        report = self.system.update()
        self.assertIsInstance(report.health_level, HealthLevel)

    def test_active_faults_collection(self):
        """活跃故障收集"""
        # 模拟一些故障工况 (高电流 + 极低速 → 持续堵转风险)
        for _ in range(200):
            self.system._motor_monitors["drive_left"].update(
                current=20.0, voltage=48.0, speed=0.01, dt=0.01, ambient_temp=40.0
            )
        report = self.system.update()
        # 堵转概率应该增加
        left_metrics = report.motor_metrics["drive_left"]
        self.assertGreater(left_metrics.stall_probability, 0.0)

    def test_recommendations_generation(self):
        """维护建议生成"""
        report = self.system.update()
        self.assertIsInstance(report.recommendations, list)

    def test_trend_analysis(self):
        """趋势分析"""
        for _ in range(5):
            self.system.update()
        trend = self.system.get_trend("overall_score", hours=24)
        self.assertIn("trend", trend)

    def test_agv_health_report_fields(self):
        """健康报告字段完整性"""
        report = self.system.update()
        self.assertTrue(hasattr(report, "timestamp"))
        self.assertTrue(hasattr(report, "overall_score"))
        self.assertTrue(hasattr(report, "health_level"))
        self.assertTrue(hasattr(report, "motor_metrics"))
        self.assertTrue(hasattr(report, "battery_metrics"))
        self.assertTrue(hasattr(report, "wheel_metrics"))
        self.assertTrue(hasattr(report, "active_faults"))
        self.assertTrue(hasattr(report, "recommendations"))


class TestAGVPredictiveMaintenanceGrades(unittest.TestCase):
    """测试 AGV 五级预测性维护规格"""

    def test_all_grades_defined(self):
        grades = ["S", "M", "L", "XL", "XXL"]
        for g in grades:
            self.assertIn(g, AGV_PREDICTIVE_MAINTENANCE_GRADES)
            self.assertIn(g, ["S", "M", "L", "XL", "XXL"])

    def test_grade_spec_structure(self):
        for grade, spec in AGV_PREDICTIVE_MAINTENANCE_GRADES.items():
            self.assertIn("motor_current_sample_rate", spec)
            self.assertIn("battery_soh_update_interval", spec)
            self.assertIn("bearing_wear_window", spec)
            self.assertIn("temp_prediction_horizon", spec)
            self.assertIn("wheel_odometry_window", spec)
            self.assertIn("health_score_baseline", spec)

    def test_grade_progression(self):
        """等级递增: 高级 = 更高采样率"""
        s_rate = AGV_PREDICTIVE_MAINTENANCE_GRADES["S"]["motor_current_sample_rate"]
        xl_rate = AGV_PREDICTIVE_MAINTENANCE_GRADES["XL"]["motor_current_sample_rate"]
        xxl_rate = AGV_PREDICTIVE_MAINTENANCE_GRADES["XXL"]["motor_current_sample_rate"]
        self.assertLess(s_rate, xl_rate)
        self.assertLess(xl_rate, xxl_rate)

    def test_get_spec(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            spec = get_predictive_maintenance_spec(grade)
            self.assertIsInstance(spec, dict)
        # 默认 fallback 到 M
        spec_default = get_predictive_maintenance_spec("INVALID")
        self.assertEqual(spec_default, AGV_PREDICTIVE_MAINTENANCE_GRADES["M"])


class TestCreatePredictiveMaintenanceSystem(unittest.TestCase):
    """测试工厂函数"""

    def test_create_s_grade(self):
        system = create_predictive_maintenance_system("S")
        self.assertEqual(system.grade, "S")

    def test_create_m_grade(self):
        system = create_predictive_maintenance_system("M")
        self.assertEqual(system.grade, "M")

    def test_create_xxl_grade(self):
        system = create_predictive_maintenance_system("XXL")
        self.assertEqual(system.grade, "XXL")
        self.assertEqual(len(system._motor_monitors), 4)


class TestMotorHealthMetricsDataclass(unittest.TestCase):
    """测试电机健康指标数据类"""

    def test_default_values(self):
        m = MotorHealthMetrics()
        self.assertEqual(m.bearing_wear_index, 0.0)
        self.assertEqual(m.winding_temp, 25.0)
        self.assertEqual(m.health_level, HealthLevel.HEALTHY)
        self.assertEqual(m.fault_type, FaultType.NONE)

    def test_custom_values(self):
        m = MotorHealthMetrics(
            bearing_wear_index=0.5,
            winding_temp=80.0,
            stall_probability=0.3,
            efficiency=0.85,
        )
        self.assertEqual(m.bearing_wear_index, 0.5)
        self.assertEqual(m.winding_temp, 80.0)


class TestBatteryHealthMetricsDataclass(unittest.TestCase):
    """测试电池健康指标数据类"""

    def test_default_values(self):
        m = BatteryHealthMetrics()
        self.assertEqual(m.soh, 100.0)
        self.assertEqual(m.cycle_count, 0)
        self.assertEqual(m.health_level, HealthLevel.HEALTHY)

    def test_custom_values(self):
        m = BatteryHealthMetrics(soh=75.0, cycle_count=500, estimated_remaining_cycles=300)
        self.assertEqual(m.soh, 75.0)
        self.assertEqual(m.cycle_count, 500)


class TestWheelHealthMetricsDataclass(unittest.TestCase):
    """测试车轮健康指标数据类"""

    def test_default_values(self):
        m = WheelHealthMetrics()
        self.assertEqual(m.slip_ratio, 0.0)
        self.assertEqual(m.alignment_error, 0.0)
        self.assertEqual(len(m.load_distribution), 4)

    def test_custom_values(self):
        m = WheelHealthMetrics(slip_ratio=0.3, alignment_error=3.0)
        self.assertEqual(m.slip_ratio, 0.3)
        self.assertEqual(m.alignment_error, 3.0)


class TestPredictiveMaintenanceIntegration(unittest.TestCase):
    """预测性维护集成测试"""

    def setUp(self):
        self.system = create_predictive_maintenance_system("L")

    def test_full_lifecycle_simulation(self):
        """完整生命周期模拟"""
        reports = []
        for step in range(200):
            dt = 0.01
            t = step * dt

            # 电机负载变化
            load_current = 3.0 + 2.0 * np.sin(t * 0.5)
            for motor_id in self.system._motor_monitors:
                self.system._motor_monitors[motor_id].update(
                    current=load_current,
                    voltage=48.0,
                    speed=1.0 + 0.5 * np.sin(t),
                    dt=dt,
                    ambient_temp=25.0 + 5.0 * np.sin(t * 0.1),
                )

            # 电池
            self.system._battery_estimator.update(
                voltage=52.0 - 0.005 * step,
                current=-3.0,
                soc=0.8 - 0.001 * step,
                temperature=30.0 + 3.0 * np.sin(t * 0.2),
                dt=dt,
            )

            # 车轮
            self.system._wheel_monitor.update(
                encoder_counts=[1000 + step, 1000 + step, 500, 500],
                wheel_speeds=[1.0, 1.0, 0.0, 0.0],
                position=(step * 0.01, 0.0, 0.0),
                dt=dt,
            )

            if step % 20 == 0:
                report = self.system.update(timestamp=t)
                reports.append(report)

        # 验证多个报告
        self.assertGreater(len(reports), 5)
        for report in reports:
            self.assertGreater(report.overall_score, 0.0)
            self.assertLessEqual(report.overall_score, 100.0)

    def test_multi_grade_consistency(self):
        """多等级一致性检查"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            system = create_predictive_maintenance_system(grade)
            report = system.update(timestamp=time.time())
            self.assertIsInstance(report.overall_score, float)
            self.assertIsInstance(report.health_level, HealthLevel)


if __name__ == "__main__":
    unittest.main()
