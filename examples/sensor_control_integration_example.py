"""
SuperModel 五级AGV传感器-控制集成示例
========================================
展示从S级到XXL级AGV的传感器配置、控制频率和闭环延迟

功能:
1. 按AGV等级初始化传感器阵列
2. 多传感器数据融合
3. 力位混合控制示例
4. 感知-运动闭环演示
"""

import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel')

import numpy as np
import time
from typing import Dict, List, Optional

# 导入传感器模块
from sensors.tactile import (
    TactileSensor, TactileData, PressureSensor,
    TaxelArray, PiezoelectricSensor, TactileArray as TactileArrayMgr
)
from sensors.force import (
    ForceSensor, ForceData, SixAxisFTSensor,
    SingleAxisForceSensor, ForceSensorArray
)
from sensors.imu import (
    IMUSensor, IMUData, BMI088, MPU9250,
    IMUArray, quaternion_to_euler, euler_to_quaternion
)

# 导入融合模块
from fusion.sensor_fusion import (
    ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion
)

# 导入控制模块
from control.motion import AGVController, DifferentialDrive, Pose2D, Twist2D
from control.pid import PIDController, PIDController2D
from control.safety import SafetyMonitor, SafetyLevel, SafetyStatus, StopReason


# ============================================================
# AGV五级传感器配置规范
# ============================================================

AGV_GRADE_CONFIGS = {
    'S': {
        'description': '小型教学AGV',
        'payload_kg': 30,
        'control_freq_hz': 50,
        '闭环延迟_ms': 200,
        'tactile': {'type': 'PressureSensor', 'rows': 8, 'cols': 8, 'max_pressure_kpa': 500, 'sample_hz': 50},
        'force': {'type': 'SingleAxisForceSensor', 'axes': 3, 'force_range_n': 100, 'torque_range_nm': 10, 'sample_hz': 100},
        'imu': {'type': 'MPU6050', 'accel_range': '8g', 'gyro_range': '1000dps', 'sample_hz': 100},
    },
    'M': {
        'description': '中型标准AGV',
        'payload_kg': 100,
        'control_freq_hz': 100,
        '闭环延迟_ms': 80,
        'tactile': {'type': 'TaxelArray', 'rows': 16, 'cols': 16, 'max_pressure_kpa': 1000, 'sample_hz': 100},
        'force': {'type': 'SixAxisFTSensor', 'axes': 6, 'force_range_n': 200, 'torque_range_nm': 20, 'sample_hz': 500},
        'imu': {'type': 'BMI088', 'accel_range': '16g', 'gyro_range': '2000dps', 'sample_hz': 200},
    },
    'L': {
        'description': '大型工业AGV',
        'payload_kg': 300,
        'control_freq_hz': 200,
        '闭环延迟_ms': 35,
        'tactile': {'type': 'TaxelArray', 'rows': 24, 'cols': 24, 'max_pressure_kpa': 2000, 'sample_hz': 200},
        'force': {'type': 'SixAxisFTSensor', 'axes': 6, 'force_range_n': 500, 'torque_range_nm': 50, 'sample_hz': 1000},
        'imu': {'type': 'BMI088', 'accel_range': '24g', 'gyro_range': '4000dps', 'sample_hz': 500},
    },
    'XL': {
        'description': '超大型高性能AGV',
        'payload_kg': 600,
        'control_freq_hz': 500,
        '闭环延迟_ms': 15,
        'tactile': {'type': 'TaxelArray', 'rows': 32, 'cols': 32, 'max_pressure_kpa': 5000, 'sample_hz': 500},
        'force': {'type': 'SixAxisFTSensor', 'axes': 6, 'force_range_n': 1000, 'torque_range_nm': 100, 'sample_hz': 2000},
        'imu': {'type': 'ADIS16470', 'accel_range': '40g', 'gyro_range': '4000dps', 'sample_hz': 1000},
    },
    'XXL': {
        'description': '重型AGV',
        'payload_kg': 1200,
        'control_freq_hz': 1000,
        '闭环延迟_ms': 7,
        'tactile': {'type': 'TaxelArray', 'rows': 48, 'cols': 48, 'max_pressure_kpa': 10000, 'sample_hz': 1000},
        'force': {'type': 'SixAxisFTSensor', 'axes': 6, 'force_range_n': 5000, 'torque_range_nm': 500, 'sample_hz': 5000},
        'imu': {'type': 'ADIS16470', 'accel_range': '80g', 'gyro_range': '8000dps', 'sample_hz': 2000},
    },
}


def create_sensors_by_grade(grade: str) -> Dict:
    """
    按AGV等级创建传感器阵列

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)

    Returns:
        包含所有传感器的字典
    """
    cfg = AGV_GRADE_CONFIGS[grade]

    sensors = {}

    # 触觉传感器
    tactile_cfg = cfg['tactile']
    if tactile_cfg['type'] == 'PressureSensor':
        sensors['tactile'] = PressureSensor(
            sensor_id=f"{grade}_pressure",
            name=f"{grade}级压力传感器",
            max_pressure=tactile_cfg['max_pressure_kpa'] * 1000
        )
    else:
        sensors['tactile'] = TaxelArray(
            sensor_id=f"{grade}_taxel",
            name=f"{grade}级触感阵列",
            rows=tactile_cfg['rows'],
            cols=tactile_cfg['cols'],
            max_pressure=tactile_cfg['max_pressure_kpa'] * 1000
        )

    # 力觉传感器
    force_cfg = cfg['force']
    if force_cfg['type'] == 'SixAxisFTSensor':
        sensors['force'] = SixAxisFTSensor(
            sensor_id=f"{grade}_ft",
            name=f"{grade}级六维力传感器",
            model='mini40' if grade in ('S', 'M') else 'Gamma'
        )
    else:
        sensors['force'] = SingleAxisForceSensor(
            sensor_id=f"{grade}_single_axis",
            name=f"{grade}级单轴力传感器",
            axis='z',
            force_range=force_cfg['force_range_n']
        )

    # IMU传感器 (使用具体实现类)
    imu_cfg = cfg['imu']
    accel_r = f"{imu_cfg['accel_range']}g"
    gyro_r = f"{imu_cfg['gyro_range']}dps"
    if imu_cfg['type'] == 'MPU6050':
        sensors['imu'] = MPU9250(
            sensor_id=f"{grade}_imu",
            name=f"{grade}级IMU"
        )
    else:  # BMI088 or ADIS16470
        sensors['imu'] = BMI088(
            sensor_id=f"{grade}_imu",
            name=f"{grade}级IMU",
            accel_range=accel_r,
            gyro_range=gyro_r
        )

    return sensors


def create_sensor_manager(grade: str) -> Dict:
    """
    创建完整传感器管理器 (触觉+力觉+IMU阵列)

    Args:
        grade: AGV等级

    Returns:
        传感器阵列管理器字典
    """
    cfg = AGV_GRADE_CONFIGS[grade]

    managers = {}

    # 触觉阵列管理器
    tactile_cfg = cfg['tactile']
    tactile_mgr = TactileArrayMgr(name=f"{grade}级触觉阵列")
    if tactile_cfg['type'] == 'TaxelArray':
        tactile_mgr.add_sensor(TaxelArray(
            sensor_id=f"{grade}_taxel_0",
            name=f"{grade}级触感阵列-前",
            rows=tactile_cfg['rows'],
            cols=tactile_cfg['cols'],
            max_pressure=tactile_cfg['max_pressure_kpa'] * 1000
        ))
    managers['tactile'] = tactile_mgr

    # 力觉传感器阵列
    force_mgr = ForceSensorArray(name=f"{grade}级力觉阵列")
    force_cfg = cfg['force']
    force_mgr.add_sensor(SixAxisFTSensor(
        sensor_id=f"{grade}_ft_0",
        name=f"{grade}级六维力传感器",
        model='mini40' if grade in ('S', 'M') else 'Gamma'
    ))
    managers['force'] = force_mgr

    # IMU阵列
    imu_cfg = cfg['imu']
    imu_arr = IMUArray(name=f"{grade}级IMU阵列")
    if imu_cfg['type'] == 'MPU6050':
        imu = MPU9250(
            sensor_id=f"{grade}_imu_0",
            name=f"{grade}级IMU",
            accel_range='8g',
            gyro_range='1000dps'
        )
    else:  # BMI088 or ADIS16470
        imu = BMI088(
            sensor_id=f"{grade}_imu_0",
            name=f"{grade}级IMU",
            accel_range=imu_cfg['accel_range'],
            gyro_range=imu_cfg['gyro_range']
        )
    imu_arr.add_sensor(imu)
    managers['imu'] = imu_arr

    return managers


def create_fusion_pipeline(grade: str) -> Dict:
    """
    创建融合管道

    Args:
        grade: AGV等级

    Returns:
        融合器字典
    """
    cfg = AGV_GRADE_CONFIGS[grade]
    imu_cfg = cfg['imu']
    sample_hz = imu_cfg['sample_hz']

    # 互补滤波器 (用于IMU姿态融合)
    comp_filter = ComplementaryFilter(alpha=0.98)

    # 扩展卡尔曼滤波 (用于多传感器融合)
    ekf = ExtendedKalmanFilter(
        state_dim=12,  # [pos, vel, ori, bias_gyro, bias_accel]
        measurement_dim=6   # [accel, gyro]
    )

    # 多传感器融合
    multi_fusion = MultiSensorFusion()

    return {
        'comp_filter': comp_filter,
        'ekf': ekf,
        'multi_fusion': multi_fusion
    }


def create_control_pipeline(grade: str) -> Dict:
    """
    创建控制管道

    Args:
        grade: AGV等级

    Returns:
        控制器字典
    """
    cfg = AGV_GRADE_CONFIGS[grade]
    control_freq = cfg['control_freq_hz']

    # AGV运动控制器
    from control.motion import DifferentialDrive
    kinematics = DifferentialDrive(
        wheel_separation=0.35,
        wheel_radius=0.07
    )
    agv_ctrl = AGVController(
        name=f"{grade}级AGV控制器",
        kinematics=kinematics,
        wheel_separation=0.35,
        wheel_radius=0.07
    )
    agv_ctrl.max_velocity = 1.5 if grade in ('M', 'L') else 3.0
    agv_ctrl.max_omega = 2.0

    # PID控制器
    pid_linear = PIDController(kp=2.0, ki=0.1, kd=0.5)
    pid_angular = PIDController(kp=3.0, ki=0.2, kd=0.8)

    # 安全监控器 (根据grade调整参数)
    max_vel = 1.5 if grade in ('M', 'L') else 3.0
    force_thresh = {'S': 50, 'M': 100, 'L': 250, 'XL': 500, 'XXL': 1000}[grade]
    torque_thresh = {'S': 5, 'M': 10, 'L': 25, 'XL': 50, 'XXL': 100}[grade]

    safety = SafetyMonitor(
        max_velocity=max_vel,
        max_acceleration=2.0,
        force_threshold=force_thresh,
        torque_threshold=torque_thresh,
        reaction_time=0.05
    )

    return {
        'agv_ctrl': agv_ctrl,
        'pid_linear': pid_linear,
        'pid_angular': pid_angular,
        'safety': safety,
        'control_freq_hz': control_freq
    }


def run_closed_loop_demo(grade: str, duration_s: float = 2.0):
    """
    运行闭环感知-控制演示

    Args:
        grade: AGV等级
        duration_s: 演示持续时间 (秒)
    """
    print(f"\n{'='*60}")
    print(f"SuperModel 五级AGV传感器-控制闭环演示 [{grade}级]")
    print(f"{'='*60}")
    print(f"描述: {AGV_GRADE_CONFIGS[grade]['description']}")
    print(f"负载: {AGV_GRADE_CONFIGS[grade]['payload_kg']}kg")
    print(f"控制频率: {AGV_GRADE_CONFIGS[grade]['control_freq_hz']}Hz")
    print(f"闭环延迟: {AGV_GRADE_CONFIGS[grade]['闭环延迟_ms']}ms")

    # 创建传感器
    sensors = create_sensors_by_grade(grade)
    managers = create_sensor_manager(grade)
    fusion = create_fusion_pipeline(grade)
    control = create_control_pipeline(grade)

    dt = 1.0 / control['control_freq_hz']
    steps = int(duration_s / dt)

    print(f"\n初始化传感器...")
    for name, sensor in sensors.items():
        print(f"  {name}: {sensor.name}")

    print(f"\n开始闭环控制 ({steps}步, dt={dt*1000:.2f}ms)...")
    print(f"{'Step':>5} | {'Time(ms)':>8} | {'Force(N)':>10} | {'Accel(m/s²)':>12} | {'Safety':>6}")
    print("-" * 60)

    for step in range(steps):
        t = step * dt
        timestamp = t

        # 1. 传感器数据采集
        force_data = sensors['force'].read(timestamp)
        imu_data = sensors['imu'].read(timestamp)

        # 2. 传感器融合
        measurements = {
            'accel': imu_data.acceleration,
            'gyro': imu_data.angular_velocity,
            'force': force_data.wrench[:3]
        }

        # 互补滤波更新
        comp_state = fusion['comp_filter'].update(measurements, dt)

        # 3. 安全检查
        safety_status: SafetyStatus = control['safety'].check_velocity(
            velocity=1.0,  # 假设1m/s前进
            dt=dt,
            timestamp=timestamp
        )

        # 4. 控制输出
        if safety_status.level == SafetyLevel.NORMAL:
            wheel_speeds = control['agv_ctrl'].step(dt)
        else:
            wheel_speeds = np.array([0.0, 0.0])

        # 打印进度 (每10步)
        if step % max(1, steps // 20) == 0:
            force_mag = np.linalg.norm(force_data.wrench[:3])
            accel_mag = np.linalg.norm(imu_data.acceleration)
            print(f"{step:>5} | {t*1000:>8.2f} | {force_mag:>10.3f} | {accel_mag:>12.4f} | {safety_status.level.name:>6}")

    print(f"\n{'='*60}")
    print(f"闭环演示完成")
    print(f"{'='*60}")

    return {
        'grade': grade,
        'steps': steps,
        'control_freq_hz': control['control_freq_hz'],
        'sensors': list(sensors.keys()),
        'fusion': list(fusion.keys()),
        'controllers': ['agv_ctrl', 'pid_linear', 'pid_angular', 'safety']
    }


def run_all_grades_demo():
    """运行所有五级AGV演示"""
    print("\n" + "=" * 70)
    print("SuperModel 超模态大模型机器人具身智能大脑")
    print("五级AGV传感器-控制集成演示")
    print("=" * 70)

    results = {}
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        result = run_closed_loop_demo(grade, duration_s=0.5)
        results[grade] = result

    # 打印汇总表
    print("\n" + "=" * 70)
    print("五级AGV配置汇总")
    print("=" * 70)
    print(f"{'等级':>5} | {'控制频率':>10} | {'闭环延迟':>10} | {'触觉':>15} | {'力觉':>15} | {'IMU':>12}")
    print("-" * 70)
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        cfg = AGV_GRADE_CONFIGS[grade]
        r = results[grade]
        t = cfg['tactile']
        f = cfg['force']
        i = cfg['imu']
        print(f"{grade:>5} | {cfg['control_freq_hz']:>8}Hz | {cfg['闭环延迟_ms']:>7}ms | "
              f"{t['rows']}×{t['cols']}@{t['sample_hz']}Hz | "
              f"{f['axes']}轴@{f['sample_hz']}Hz | {i['type']}@{i['sample_hz']}Hz")

    print("\n" + "=" * 70)
    print("演示完成 - SuperModel v1.53.0")
    print("=" * 70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        grade = sys.argv[1].upper()
        if grade in AGV_GRADE_CONFIGS:
            run_closed_loop_demo(grade)
        else:
            print(f"未知等级: {grade}")
            print(f"可用等级: {list(AGV_GRADE_CONFIGS.keys())}")
    else:
        run_all_grades_demo()
