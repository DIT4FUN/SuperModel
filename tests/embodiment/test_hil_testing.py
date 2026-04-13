"""
test_hil_testing.py - HIL硬件在环测试框架测试
测试HIL测试框架的核心功能
"""

import pytest
import numpy as np
import time
import json
import tempfile
import os


class TestHILTestCase:
    def test_hil_test_case_creation(self):
        from src.embodied.hil_testing import HILTestCase, HILTestStage
        case = HILTestCase(
            case_id="hil_001",
            name="Test CAN",
            description="Test CAN bus",
            target_hardware="ZLAC8015D",
            test_stage=HILTestStage.OPEN_LOOP,
            timeout_s=30.0,
        )
        assert case.case_id == "hil_001"
        assert case.target_hardware == "ZLAC8015D"
        assert case.timeout_s == 30.0


class TestHILTestResult:
    def test_hil_test_result_pass(self):
        from src.embodied.hil_testing import HILTestResult, HILTestStage
        result = HILTestResult(
            case_id="hil_001",
            passed=True,
            duration_s=5.3,
            stage_reached=HILTestStage.COMPLETED,
        )
        assert result.passed
        assert result.duration_s == 5.3

    def test_hil_test_result_fail(self):
        from src.embodied.hil_testing import HILTestResult, HILTestStage
        result = HILTestResult(
            case_id="hil_001",
            passed=False,
            duration_s=2.1,
            stage_reached=HILTestStage.OPEN_LOOP,
            error_message="Timeout",
        )
        assert not result.passed
        assert result.error_message == "Timeout"


class TestHILTestReport:
    def test_report_creation(self):
        from src.embodied.hil_testing import HILTestReport, HILTestResult, HILTestStage
        result = HILTestResult(
            case_id="hil_001",
            passed=True,
            duration_s=5.0,
            stage_reached=HILTestStage.COMPLETED,
        )
        report = HILTestReport(
            report_id="report_001",
            timestamp=time.time(),
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            results=[result],
            duration_s=5.0,
        )
        assert report.total_cases == 1
        assert report.passed_cases == 1

    def test_report_to_dict(self):
        from src.embodied.hil_testing import HILTestReport, HILTestResult, HILTestStage
        result = HILTestResult(
            case_id="hil_001",
            passed=True,
            duration_s=5.0,
            stage_reached=HILTestStage.COMPLETED,
        )
        report = HILTestReport(
            report_id="report_001",
            timestamp=time.time(),
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            results=[result],
            duration_s=5.0,
        )
        d = report.to_dict()
        assert d["total_cases"] == 1
        assert d["pass_rate"] == "100.0%"

    def test_report_save_load(self):
        from src.embodied.hil_testing import HILTestReport, HILTestResult, HILTestStage
        result = HILTestResult(
            case_id="hil_001",
            passed=True,
            duration_s=5.0,
            stage_reached=HILTestStage.COMPLETED,
        )
        report = HILTestReport(
            report_id="report_001",
            timestamp=time.time(),
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            results=[result],
            duration_s=5.0,
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = f.name
        try:
            report.save(tmp_path)
            with open(tmp_path) as f:
                loaded = json.load(f)
            assert loaded["report_id"] == "report_001"
            assert loaded["total_cases"] == 1
        finally:
            os.unlink(tmp_path)


class TestSensorReplay:
    def test_sensor_replay_from_list(self):
        from src.embodied.hil_testing import SensorReplay
        data = [
            {"timestamp": 1.0, "lidar": [1.0, 2.0]},
            {"timestamp": 2.0, "lidar": [1.1, 2.1]},
            {"timestamp": 3.0, "lidar": [1.2, 2.2]},
        ]
        replay = SensorReplay(data_source=data, replay_speed=10.0)
        assert len(replay._data) == 3

    def test_sensor_replay_callback(self):
        from src.embodied.hil_testing import SensorReplay
        data = [
            {"timestamp": 1.0, "lidar": [1.0]},
            {"timestamp": 2.0, "lidar": [2.0]},
        ]
        replay = SensorReplay(data_source=data, replay_speed=100.0)
        received = []
        def callback(record):
            received.append(record)
        replay.register_callback(callback)
        replay.start()
        time.sleep(0.3)
        replay.stop()
        assert len(received) >= 1

    def test_sensor_replay_seek(self):
        from src.embodied.hil_testing import SensorReplay
        data = [{"timestamp": float(i), "value": i} for i in range(10)]
        replay = SensorReplay(data_source=data, replay_speed=10.0)
        replay.seek(5)
        assert replay._current_idx == 5
        assert replay.get_current_record()["value"] == 5

    def test_sensor_replay_progress(self):
        from src.embodied.hil_testing import SensorReplay
        data = [{"timestamp": float(i)} for i in range(100)]
        replay = SensorReplay(data_source=data, replay_speed=1.0)
        replay.seek(25)
        assert abs(replay.get_progress() - 0.25) < 0.01


class TestCANBusHILSimulator:
    def test_can_simulator_creation(self):
        from src.embodied.hil_testing import CANBusHILSimulator
        sim = CANBusHILSimulator(interface="vcan0", baudrate=500000)
        assert sim.interface == "vcan0"
        assert sim.baudrate == 500000

    def test_can_send_frame(self):
        from src.embodied.hil_testing import CANBusHILSimulator
        sim = CANBusHILSimulator()
        sim.start()
        success = sim.send_frame(0x100, b"\x01\x02\x03\x04")
        assert success
        stats = sim.get_stats()
        assert stats["tx_count"] == 1
        sim.stop()

    def test_can_inject_frame(self):
        from src.embodied.hil_testing import CANBusHILSimulator
        sim = CANBusHILSimulator()
        sim.start()
        sim.inject_frame(0x101, b"\x05\x06\x07\x08")
        stats = sim.get_stats()
        assert stats["rx_count"] == 1
        sim.stop()

    def test_can_callback(self):
        from src.embodied.hil_testing import CANBusHILSimulator
        sim = CANBusHILSimulator()
        sim.start()
        received = []
        def cb(frame):
            received.append(frame)
        sim.register_callback(cb)
        sim.send_frame(0x100, b"\x01")
        time.sleep(0.05)
        assert len(received) >= 1
        sim.stop()

    def test_can_history(self):
        from src.embodied.hil_testing import CANBusHILSimulator
        sim = CANBusHILSimulator()
        sim.start()
        sim.send_frame(0x100, b"\x01")
        sim.send_frame(0x101, b"\x02")
        history = sim.get_history(limit=10)
        assert len(history) == 2
        sim.stop()

    def test_can_clear_history(self):
        from src.embodied.hil_testing import CANBusHILSimulator
        sim = CANBusHILSimulator()
        sim.start()
        sim.send_frame(0x100, b"\x01")
        sim.clear_history()
        assert len(sim.get_history()) == 0
        sim.stop()


class TestControlCommandValidator:
    def test_validator_creation(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        assert len(validator.command_templates) > 0

    def test_validate_velocity_command(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmd = {"type": "velocity", "vx": 0.5, "vy": 0.0, "omega": 0.1}
        valid, err, lat = validator.validate_command(cmd)
        assert valid
        assert err is None

    def test_validate_velocity_out_of_range(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmd = {"type": "velocity", "vx": 10.0, "vy": 0.0, "omega": 0.1}  # vx超出范围
        valid, err, lat = validator.validate_command(cmd)
        assert not valid
        assert "vx" in err

    def test_validate_unknown_command_type(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmd = {"type": "unknown_type"}
        valid, err, lat = validator.validate_command(cmd)
        assert not valid
        assert "Unknown command type" in err

    def test_validate_emergency_stop(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmd = {"type": "emergency_stop"}
        valid, err, lat = validator.validate_command(cmd)
        assert valid

    def test_validate_grasp_command(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmd = {"type": "grasp", "gripper_id": 0, "force": 50.0, "position": 0.05}
        valid, err, lat = validator.validate_command(cmd)
        assert valid

    def test_check_sequence_valid(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmds = [
            {"type": "velocity", "vx": 0.5, "vy": 0.0, "omega": 0.0},
            {"type": "grasp", "gripper_id": 0, "force": 50.0, "position": 0.05},
            {"type": "position", "x": 1.0, "y": 0.0, "theta": 0.0},
        ]
        expected = ["velocity", "grasp", "position"]
        valid, mismatches = validator.check_sequence(cmds, expected)
        assert valid
        assert len(mismatches) == 0

    def test_check_sequence_invalid(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        cmds = [
            {"type": "velocity", "vx": 0.5, "vy": 0.0, "omega": 0.0},
            {"type": "position", "x": 1.0, "y": 0.0, "theta": 0.0},  # 错序
            {"type": "grasp", "gripper_id": 0, "force": 50.0, "position": 0.05},
        ]
        expected = ["velocity", "grasp", "position"]
        valid, mismatches = validator.check_sequence(cmds, expected)
        assert not valid

    def test_latency_stats(self):
        from src.embodied.hil_testing import ControlCommandValidator
        validator = ControlCommandValidator()
        for _ in range(10):
            validator.validate_command({"type": "velocity", "vx": 0.5, "vy": 0.0, "omega": 0.0})
        stats = validator.get_latency_stats()
        assert "velocity" in stats
        assert stats["velocity"]["count"] == 10


class TestSensorActuatorHILLoop:
    def test_loop_creation(self):
        from src.embodied.hil_testing import SensorReplay, ControlCommandValidator, SensorActuatorHILLoop
        replay = SensorReplay([{"timestamp": 1.0}])
        validator = ControlCommandValidator()
        loop = SensorActuatorHILLoop(
            sensor_replay=replay,
            command_validator=validator,
            closed_loop=True,
            cycle_time_ms=10.0,
        )
        assert loop.cycle_time_ms == 10.0
        assert loop.closed_loop

    def test_inject_command(self):
        from src.embodied.hil_testing import SensorReplay, ControlCommandValidator, SensorActuatorHILLoop
        replay = SensorReplay([{"timestamp": 1.0}])
        validator = ControlCommandValidator()
        loop = SensorActuatorHILLoop(replay, validator)
        loop.inject_command({"type": "velocity", "vx": 0.5, "vy": 0.0, "omega": 0.0})
        status = loop.get_loop_status()
        assert status["command_buffer_size"] == 1

    def test_loop_status(self):
        from src.embodied.hil_testing import SensorReplay, ControlCommandValidator, SensorActuatorHILLoop
        replay = SensorReplay([{"timestamp": 1.0}])
        validator = ControlCommandValidator()
        loop = SensorActuatorHILLoop(replay, validator)
        status = loop.get_loop_status()
        assert "cycle_count" in status
        assert "running" in status


class TestHILTestRunner:
    def test_runner_creation(self):
        from src.embodied.hil_testing import HILTestRunner
        runner = HILTestRunner(project_name="TestProject", output_dir="/tmp/hil_test")
        assert runner.project_name == "TestProject"

    def test_register_test_case(self):
        from src.embodied.hil_testing import HILTestRunner, HILTestCase, HILTestStage
        runner = HILTestRunner()
        case = HILTestCase(
            case_id="hil_001",
            name="Test",
            description="Test case",
            target_hardware="ZLAC8015D",
            test_stage=HILTestStage.OPEN_LOOP,
        )
        runner.register_test_case(case)
        assert "hil_001" in runner._test_cases

    def test_runner_status(self):
        from src.embodied.hil_testing import HILTestRunner
        runner = HILTestRunner()
        status = runner.get_runner_status()
        assert "registered_cases" in status
        assert status["registered_cases"] == 0


class TestRunHILValidation:
    def test_run_hil_validation(self):
        from src.embodied.hil_testing import run_hil_validation
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_hil_validation(output_dir=tmpdir)
            assert report.total_cases >= 3
            assert report.passed_cases + report.failed_cases == report.total_cases
