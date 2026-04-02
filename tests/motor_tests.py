"""
电机控制模块测试用例
测试 DC电机、BLDC、伺服电机、步进电机的仿真与控制
覆盖: Motor, DCMotor, BLDCmotor, ServoMotor, StepperMotor, PIDController, MotorController
AGV等级: S (简单电机) → XXL (高性能伺服)
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.motor import (
    Motor, MotorState, MotorControlMode,
    DCMotor, BLDCmotor, ServoMotor, StepperMotor,
    PIDController as MotorPIDController, MotorController
)


class TestMotorState(unittest.TestCase):
    """MotorState 数据类测试"""

    def test_motor_state_creation(self):
        state = MotorState(
            motor_id="motor_1",
            timestamp=1.5,
            position=0.5,
            velocity=10.0,
            current=2.0,
            temperature=25.0,
            enabled=True
        )
        self.assertEqual(state.motor_id, "motor_1")
        self.assertAlmostEqual(state.position, 0.5)
        self.assertAlmostEqual(state.velocity, 10.0)
        self.assertTrue(state.enabled)

    def test_motor_state_to_vector(self):
        state = MotorState(motor_id="test", timestamp=0.0, position=1.0, velocity=2.0)
        vec = state.to_vector()
        self.assertEqual(len(vec), 5)
        self.assertAlmostEqual(vec[0], 1.0 / (2 * np.pi))
        self.assertAlmostEqual(vec[1], 2.0 / 100.0)

    def test_motor_state_is_valid(self):
        valid = MotorState(motor_id="m", timestamp=0.0, position=0.0, velocity=0.0)
        self.assertTrue(valid.is_valid())
        invalid = MotorState(motor_id="m", timestamp=0.0, position=0.0, velocity=0.0, temperature=85.0)
        self.assertFalse(invalid.is_valid())


class TestDCMotor(unittest.TestCase):
    """直流电机测试"""

    def test_dcmotor_creation(self):
        motor = DCMotor(
            motor_id="dc_1",
            voltage=24.0,
            reduction_ratio=30.0,
            max_velocity=100.0,
            max_torque=10.0,
            armature_resistance=5.0,
            torque_constant=0.1
        )
        self.assertEqual(motor.motor_id, "dc_1")
        self.assertFalse(motor.is_enabled())

    def test_dcmotor_enable_disable(self):
        motor = DCMotor(motor_id="test_dc")
        motor.enable()
        self.assertTrue(motor.is_enabled())
        motor.disable()
        self.assertFalse(motor.is_enabled())

    def test_dcmotor_position_control(self):
        motor = DCMotor(motor_id="test_pos", reduction_ratio=10.0, max_velocity=200.0)
        motor.enable()
        motor.set_target(target=100.0, mode=MotorControlMode.POSITION)
        for _ in range(100):
            motor.step(dt=0.01)
        state = motor.get_state()
        # 验证电机正在响应控制 (位置已改变)
        self.assertGreater(state.position, 0)

    def test_dcmotor_velocity_control(self):
        motor = DCMotor(motor_id="test_vel", max_velocity=200.0)
        motor.enable()
        motor.set_target(target=50.0, mode=MotorControlMode.VELOCITY)
        for _ in range(50):
            motor.step(dt=0.01)
        state = motor.get_state()
        self.assertGreater(state.velocity, 0)

    def test_dcmotor_torque_control(self):
        motor = DCMotor(motor_id="test_dc_torque")
        motor.enable()
        motor.set_target(target=0.5, mode=MotorControlMode.TORQUE)
        motor.step(dt=0.01)
        state = motor.get_state()
        self.assertIsNotNone(state.current)

    def test_dcmotor_pwm_control(self):
        motor = DCMotor(motor_id="test_pwm", voltage=24.0)
        motor.enable()
        motor.set_target(target=0.5, mode=MotorControlMode.PWM)
        motor.step(dt=0.01)
        state = motor.get_state()
        self.assertIsNotNone(state.pwm_duty)

    def test_dcmotor_temperature_rise(self):
        motor = DCMotor(motor_id="test_temp", voltage=24.0)
        motor.enable()
        motor.set_target(target=1.0, mode=MotorControlMode.PWM)
        for _ in range(100):
            motor.step(dt=0.1)
        self.assertGreater(motor.get_state().temperature, 25.0)

    def test_dcmotor_velocity_at_zero(self):
        """速度控制时速度应从0开始"""
        motor = DCMotor(motor_id="test_dc_zero")
        motor.enable()
        motor.set_target(target=0.0, mode=MotorControlMode.VELOCITY)
        motor.step(dt=0.01)
        state = motor.get_state()
        self.assertEqual(state.velocity, 0.0)


class TestBLDCMotor(unittest.TestCase):
    """无刷直流电机测试"""

    def test_bldc_creation(self):
        motor = BLDCmotor(
            motor_id="bldc_1",
            poles=4,
            kv=1000,
            reduction_ratio=20.0,
            max_velocity=150.0,
            max_torque=5.0,
            phase_resistance=0.1,
            phase_inductance=0.001
        )
        self.assertEqual(motor.motor_id, "bldc_1")

    def test_bldc_enable_disable(self):
        motor = BLDCmotor(motor_id="test_bldc")
        motor.enable()
        self.assertTrue(motor.is_enabled())
        motor.disable()
        self.assertFalse(motor.is_enabled())

    def test_bldc_velocity_control(self):
        motor = BLDCmotor(motor_id="test_bldc_vel", max_velocity=150.0)
        motor.enable()
        motor.set_target(target=100.0, mode=MotorControlMode.VELOCITY)
        for _ in range(100):
            motor.step(dt=0.001)
        state = motor.get_state()
        self.assertGreater(state.velocity, 0)

    def test_bldc_position_control(self):
        motor = BLDCmotor(motor_id="test_bldc_pos", reduction_ratio=10.0)
        motor.enable()
        motor.set_target(target=100.0, mode=MotorControlMode.POSITION)
        for _ in range(200):
            motor.step(dt=0.001)
        state = motor.get_state()
        self.assertGreater(state.position, 0)


class TestServoMotor(unittest.TestCase):
    """伺服电机测试"""

    def test_servo_creation(self):
        motor = ServoMotor(
            motor_id="servo_1",
            angle_range=360.0,
            reduction_ratio=100.0,
            max_velocity=50.0,
            max_torque=2.0,
            position_resolution=0.01
        )
        self.assertEqual(motor.motor_id, "servo_1")

    def test_servo_enable_disable(self):
        motor = ServoMotor(motor_id="test_servo")
        motor.enable()
        self.assertTrue(motor.is_enabled())
        motor.disable()
        self.assertFalse(motor.is_enabled())

    def test_servo_position_control(self):
        motor = ServoMotor(motor_id="test_servo_pos", reduction_ratio=100.0, angle_range=360.0)
        motor.enable()
        motor.set_target(target=180.0, mode=MotorControlMode.POSITION)
        for _ in range(200):
            motor.step(dt=0.01)
        state = motor.get_state()
        self.assertGreater(state.position, 0)


class TestStepperMotor(unittest.TestCase):
    """步进电机测试"""

    def test_stepper_creation(self):
        motor = StepperMotor(
            motor_id="stepper_1",
            steps_per_rev=200,
            microsteps=16,
            reduction_ratio=10.0,
            max_velocity=20.0,
            holding_torque=1.0
        )
        self.assertEqual(motor.motor_id, "stepper_1")
        self.assertEqual(motor.microsteps, 16)

    def test_stepper_enable_disable(self):
        motor = StepperMotor(motor_id="test_stepper")
        motor.enable()
        self.assertTrue(motor.is_enabled())
        motor.disable()
        self.assertFalse(motor.is_enabled())

    def test_stepper_position_control(self):
        motor = StepperMotor(motor_id="test_stepper_pos", steps_per_rev=200, microsteps=16, reduction_ratio=10.0)
        motor.enable()
        motor.set_target(target=720.0, mode=MotorControlMode.POSITION)
        for _ in range(500):
            motor.step(dt=0.01)
        state = motor.get_state()
        self.assertGreater(state.position, 0)


class TestPIDController(unittest.TestCase):
    """PID控制器测试"""

    def test_pid_creation(self):
        pid = MotorPIDController(kp=1.0, ki=0.1, kd=0.01)
        self.assertAlmostEqual(pid.kp, 1.0)
        self.assertAlmostEqual(pid.ki, 0.1)

    def test_pid_position_mode(self):
        pid = MotorPIDController(kp=2.0, ki=0.0, kd=0.0)
        output = pid.compute(error=10.0, dt=0.01)
        self.assertGreater(output, 0)

    def test_pid_with_limits(self):
        pid = MotorPIDController(kp=1.0, ki=0.0, kd=0.0, output_limit=5.0)
        output = pid.compute(error=100.0, dt=0.01)
        self.assertLessEqual(output, 5.0)

    def test_pid_integral_windup(self):
        pid = MotorPIDController(kp=1.0, ki=1.0, kd=0.0, integral_limit=10.0)
        for _ in range(100):
            pid.compute(error=10.0, dt=0.01)
        self.assertLessEqual(pid._integral, 10.0)

    def test_pid_reset(self):
        pid = MotorPIDController(kp=1.0, ki=0.5, kd=0.1)
        pid.compute(error=10.0, dt=0.01)
        pid.reset()
        self.assertAlmostEqual(pid._integral, 0.0)


class TestMotorController(unittest.TestCase):
    """多电机控制器测试"""

    def test_motor_controller_creation(self):
        mc = MotorController()
        self.assertEqual(len(mc._motors), 0)

    def test_add_motor(self):
        mc = MotorController()
        motor = DCMotor(motor_id="dc_1")
        mc.add_motor(motor)
        self.assertEqual(len(mc._motors), 1)

    def test_remove_motor(self):
        mc = MotorController()
        motor = DCMotor(motor_id="dc_1")
        mc.add_motor(motor)
        mc.remove_motor("dc_1")
        self.assertEqual(len(mc._motors), 0)

    def test_enable_all(self):
        mc = MotorController()
        mc.add_motor(DCMotor(motor_id="dc_1"))
        mc.add_motor(DCMotor(motor_id="dc_2"))
        mc.enable_all()
        for m in mc._motors.values():
            self.assertTrue(m.is_enabled())

    def test_disable_all(self):
        mc = MotorController()
        mc.add_motor(DCMotor(motor_id="dc_1"))
        mc.enable_all()
        mc.disable_all()
        for m in mc._motors.values():
            self.assertFalse(m.is_enabled())

    def test_step_all(self):
        mc = MotorController()
        m1 = DCMotor(motor_id="dc_1")
        m2 = BLDCmotor(motor_id="bldc_1")
        mc.add_motor(m1)
        mc.add_motor(m2)
        mc.enable_all()
        results = mc.step_all(dt=0.01)
        self.assertEqual(len(results), 2)


class TestAGVGradeMotorSpecs(unittest.TestCase):
    """AGV五级电机规格测试"""

    def test_s_grade_dcmotor(self):
        """S级: 基础DC电机 ≤500kg负载"""
        motor = DCMotor(motor_id='s_grade', voltage=12.0, reduction_ratio=10.0, max_velocity=50.0, max_torque=5.0, armature_resistance=5.0, torque_constant=0.05)
        motor.enable()
        motor.set_target(target=100.0, mode=MotorControlMode.POSITION)
        for _ in range(50):
            motor.step(dt=0.01)
        self.assertGreater(motor.get_state().position, 0)

    def test_m_grade_dcmotor(self):
        """M级: 中等DC电机 500-1500kg"""
        motor = DCMotor(motor_id='m_grade', voltage=24.0, reduction_ratio=30.0, max_velocity=100.0, max_torque=10.0, armature_resistance=5.0, torque_constant=0.1)
        motor.enable()
        motor.set_target(target=50.0, mode=MotorControlMode.VELOCITY)
        for _ in range(50):
            motor.step(dt=0.01)
        self.assertGreater(motor.get_state().velocity, 0)

    def test_l_grade_bldc(self):
        """L级: BLDC 1500-3000kg"""
        motor = BLDCmotor(motor_id='l_grade', poles=4, kv=1000, reduction_ratio=20.0, max_velocity=150.0, max_torque=5.0, phase_resistance=0.1, phase_inductance=0.001)
        motor.enable()
        motor.set_target(target=100.0, mode=MotorControlMode.VELOCITY)
        for _ in range(100):
            motor.step(dt=0.001)
        self.assertGreater(motor.get_state().velocity, 0)

    def test_xl_grade_servo(self):
        """XL级: 高性能伺服 3000-5000kg"""
        motor = ServoMotor(motor_id='xl_grade', reduction_ratio=100.0, max_velocity=50.0, max_torque=5.0, angle_range=360.0)
        motor.enable()
        motor.set_target(target=180.0, mode=MotorControlMode.POSITION)
        for _ in range(200):
            motor.step(dt=0.01)
        self.assertGreater(motor.get_state().position, 0)

    def test_xxl_grade_servo(self):
        """XXL级: 超高性能伺服 >5000kg"""
        motor = ServoMotor(motor_id='xxl_grade', reduction_ratio=200.0, max_velocity=100.0, max_torque=20.0, angle_range=360.0)
        motor.enable()
        motor.set_target(target=360.0, mode=MotorControlMode.POSITION)
        for _ in range(300):
            motor.step(dt=0.01)
        state = motor.get_state()
        self.assertGreater(state.position, 0)
        self.assertLess(state.temperature, 100.0)


class TestMotorRobustness(unittest.TestCase):
    """电机鲁棒性测试"""

    def test_nan_handling(self):
        motor = DCMotor(motor_id="test_nan")
        motor._state.error = "SENSOR_ERROR"
        state = motor.get_state()
        self.assertFalse(state.is_valid())

    def test_overvoltage_protection(self):
        motor = DCMotor(motor_id="test_overvolt", voltage=24.0)
        motor.enable()
        motor.set_target(target=100.0, mode=MotorControlMode.PWM)
        motor.step(dt=0.01)
        self.assertLessEqual(motor.get_state().pwm_duty, 1.0)

    def test_negative_velocity_clamping(self):
        motor = DCMotor(motor_id="test_neg_vel", max_velocity=100.0)
        motor.enable()
        motor.set_target(target=-200.0, mode=MotorControlMode.VELOCITY)
        motor.step(dt=0.01)
        self.assertGreaterEqual(motor.get_state().velocity, -motor.max_velocity)

    def test_multiple_motor_types_together(self):
        """多类型电机混合控制"""
        mc = MotorController()
        mc.add_motor(DCMotor(motor_id="dc_1"))
        mc.add_motor(BLDCmotor(motor_id="bldc_1"))
        mc.add_motor(ServoMotor(motor_id="servo_1"))
        mc.add_motor(StepperMotor(motor_id="stepper_1"))
        self.assertEqual(len(mc._motors), 4)
        mc.enable_all()
        mc.step_all(dt=0.01)
        mc.disable_all()


if __name__ == "__main__":
    unittest.main(verbosity=2)
