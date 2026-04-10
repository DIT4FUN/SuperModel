"""
控制模块 (Control)
电机控制、运动控制、轨迹规划
"""

from control.motor import (
    Motor,
    MotorState,
    MotorControlMode,
    DCMotor,
    BLDCmotor,
    ServoMotor,
    StepperMotor,
    MotorController,
    PIDController
)
from control.motion import (
    KinematicsModel,
    DifferentialDrive,
    MecanumDrive,
    TrajectoryPlanner,
    MotionController,
    AGVController,
    Pose2D,
    Twist2D
)
from control.pid import PIDController as StandalonePIDController
from control.pid import PIDController2D, PIDAutotuner
from control.safety import (
    SafetyLevel,
    StopReason,
    SafetyStatus,
    SafetyMonitor,
    EmergencyStopController
)
from control.autotune import (
    AutoTuner,
    TunerConfig,
    TunerResult,
    TuningMethod,
    SimulatedPlant,
    autotune_pid
)
from control.grade_control import (
    AGVGrade,
    GRADE_CONTROL_SPECS,
    GradePIDConfig,
    GradeControllerConfig,
    GradeAwarePID,
    GradeAwareSafetyMonitor,
    GradeAwareTrajectoryPlanner,
    get_grade_control_spec,
    list_grade_capabilities
)

__all__ = [
    # motor
    'Motor', 'MotorState', 'MotorControlMode', 'DCMotor', 'BLDCmotor',
    'ServoMotor', 'StepperMotor', 'MotorController', 'PIDController',
    # motion
    'KinematicsModel', 'DifferentialDrive', 'MecanumDrive', 'TrajectoryPlanner',
    'MotionController', 'AGVController', 'Pose2D', 'Twist2D',
    # pid
    'StandalonePIDController', 'PIDController2D', 'PIDAutotuner',
    # safety
    'SafetyLevel', 'StopReason', 'SafetyStatus',
    'SafetyMonitor', 'EmergencyStopController',
    # autotune
    'AutoTuner', 'TunerConfig', 'TunerResult', 'TuningMethod',
    'SimulatedPlant', 'autotune_pid',
    # grade control
    'AGVGrade', 'GRADE_CONTROL_SPECS', 'GradePIDConfig', 'GradeControllerConfig',
    'GradeAwarePID', 'GradeAwareSafetyMonitor', 'GradeAwareTrajectoryPlanner',
    'get_grade_control_spec', 'list_grade_capabilities',
]
