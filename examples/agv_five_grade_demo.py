#!/usr/bin/env python3
"""
SuperModel AGV五级完整演示
============================

展示 S → M → L → XL → XXL 五个等级的传感器→融合→控制→仿真全链路

运行方式:
    python3 examples/agv_five_grade_demo.py [--grade S|M|L|XL|XXL|ALL] [--duration 5]

版本: v2.08.0 (2026-04-09)
"""

import argparse
import sys
import time
import math
import numpy as np

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType, get_force_spec
from sensors.imu import IMUSensor, IMUSensorType, PoseEstimator, get_imu_spec
from sensors.vision import CameraModel, StereoCamera
from sensors.audio import AudioProcessor, SoundSourceLocalization
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from fusion.sensor_fusion import ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion


GRADE_CONFIG = {
    'S': {
        'description': '教学级AGV (低成本)',
        'max_load': 30, 'max_speed': 0.5, 'control_freq': 50,
        'tactile': {'array': (8, 8), 'type': TactileSensorType.RESISTIVE, 'freq': 50},
        'force': {'axes': 3, 'range': 100, 'sampling': 100},
        'imu': {'type': IMUSensorType.MPU6050, 'sample_hz': 100},
        'vision': {'channels': 1, 'resolution': (640, 480)},
        'audio': {'channels': 1},
        'control_mode': 'position',
        'pid': {'Kp': 10.0, 'Ki': 0.5, 'Kd': 2.0},
    },
    'M': {
        'description': '标准工业AGV (物流/制造业)',
        'max_load': 100, 'max_speed': 1.5, 'control_freq': 100,
        'tactile': {'array': (16, 16), 'type': TactileSensorType.CAPACITIVE, 'freq': 100},
        'force': {'axes': 6, 'range': 200, 'sampling': 500},
        'imu': {'type': IMUSensorType.BMI088, 'sample_hz': 200},
        'vision': {'channels': 2, 'resolution': (1280, 720)},
        'audio': {'channels': 2},
        'control_mode': 'position+velocity',
        'pid': {'Kp': 15.0, 'Ki': 1.0, 'Kd': 3.0},
    },
    'L': {
        'description': '精密工业AGV (重载装配)',
        'max_load': 300, 'max_speed': 2.0, 'control_freq': 200,
        'tactile': {'array': (24, 24), 'type': TactileSensorType.PIEZOELECTRIC, 'freq': 200},
        'force': {'axes': 6, 'range': 500, 'sampling': 1000},
        'imu': {'type': IMUSensorType.BMI088, 'sample_hz': 500},
        'vision': {'channels': 2, 'resolution': (1920, 1080), 'fps': 60},
        'audio': {'channels': 4},
        'control_mode': 'position+velocity+impedance',
        'pid': {'Kp': 20.0, 'Ki': 2.0, 'Kd': 5.0},
        'impedance': {'M': 2.0, 'D': 10.0, 'K': 50.0},
    },
    'XL': {
        'description': '高性能AGV (特种/协作)',
        'max_load': 600, 'max_speed': 2.5, 'control_freq': 500,
        'tactile': {'array': (32, 32), 'type': TactileSensorType.OPTICAL, 'freq': 500},
        'force': {'axes': 6, 'range': 1000, 'sampling': 2000},
        'imu': {'type': IMUSensorType.ADIS16470, 'sample_hz': 1000},
        'vision': {'channels': 2, 'resolution': (1920, 1080), 'event_camera': True},
        'audio': {'channels': 6},
        'control_mode': 'full_modal',
        'pid': {'Kp': 25.0, 'Ki': 3.0, 'Kd': 6.0},
        'impedance': {'M': 1.5, 'D': 15.0, 'K': 100.0},
        'mpc': {'horizon': 15, 'dt': 0.002},
    },
    'XXL': {
        'description': '旗舰全功能AGV (具身智能大脑)',
        'max_load': 1200, 'max_speed': 3.0, 'control_freq': 1000,
        'tactile': {'array': (48, 48), 'type': TactileSensorType.OPTICAL, 'freq': 1000},
        'force': {'axes': 6, 'range': 5000, 'sampling': 5000},
        'imu': {'type': IMUSensorType.ADIS16470, 'sample_hz': 2000, 'redundant': 2},
        'vision': {'channels': 4, 'resolution': (3840, 2160), 'lidar': True},
        'audio': {'channels': 8},
        'control_mode': 'full_modal+MPC',
        'pid': {'Kp': 30.0, 'Ki': 5.0, 'Kd': 8.0},
        'impedance': {'M': 1.0, 'D': 20.0, 'K': 200.0},
        'mpc': {'horizon': 20, 'dt': 0.001},
        'autonomous': True,
    },
}


def banner(grade: str):
    grade_icons = {'S': '🎓', 'M': '🏭', 'L': '🔧', 'XL': '🚀', 'XXL': '🧠'}
    icon = grade_icons.get(grade, '📦')
    cfg = GRADE_CONFIG[grade]
    print(f"\n{'='*70}")
    print(f"{icon} SuperModel AGV-{grade} | {cfg['description']}")
    print(f"{'='*70}")
    print(f"  负载: {cfg['max_load']}kg | 最高速度: {cfg['max_speed']}m/s | "
          f"控制频率: {cfg['control_freq']}Hz")
    print(f"  控制模式: {cfg['control_mode']}")
    print(f"  PID: Kp={cfg['pid']['Kp']}, Ki={cfg['pid']['Ki']}, Kd={cfg['pid']['Kd']}")
    if 'impedance' in cfg:
        imp = cfg['impedance']
        print(f"  阻抗: M={imp['M']}kg, D={imp['D']}N·s/m, K={imp['K']}N/m")
    if 'mpc' in cfg:
        mpc = cfg['mpc']
        print(f"  MPC: horizon={mpc['horizon']}步, dt={mpc['dt']*1000:.1f}ms")
    print(f"{'─'*70}")


def init_sensors(grade: str):
    """按等级初始化传感器"""
    cfg = GRADE_CONFIG[grade]
    
    sensors = {}
    
    # 触觉
    t_cfg = cfg['tactile']
    sensors['tactile'] = TactileArray(
        array_size=t_cfg['array'],
        sensor_type=t_cfg['type'],
        sensor_id=f"tactile_{grade}"
    )
    
    # 力觉
    f_cfg = cfg['force']
    f_type = (ForceSensorType.THREE_AXIS if f_cfg['axes'] == 3 
              else ForceSensorType.SIX_AXIS)
    sensors['force'] = ForceTorqueSensor(
        sensor_type=f_type,
        sensor_id=f"force_{grade}"
    )
    
    # IMU
    i_cfg = cfg['imu']
    sensors['imu'] = IMUSensor(
        sensor_type=i_cfg['type'],
        sensor_id=f"imu_{grade}"
    )
    
    return sensors


def init_fusion(grade: str):
    """按等级初始化融合模块"""
    cfg = GRADE_CONFIG[grade]
    
    # 融合配置
    if cfg['control_freq'] <= 100:
        # 低频: 简单互补滤波
        fusion = MultiSensorFusion(
            filter_type='complementary',
            alpha=0.96
        )
    elif cfg['control_freq'] <= 500:
        # 中频: EKF
        fusion = MultiSensorFusion(
            filter_type='ekf',
            process_noise=0.01,
            measurement_noise=0.1
        )
    else:
        # 高频: 多传感器EKF
        fusion = MultiSensorFusion(
            filter_type='ekf',
            process_noise=0.001,
            measurement_noise=0.01,
            multi_rate=True
        )
    
    return fusion


def simulate_grade(grade: str, duration: float = 3.0):
    """
    模拟单个AGV等级的完整感知→融合→控制链路
    
    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        duration: 模拟时长 (秒)
    """
    cfg = GRADE_CONFIG[grade]
    dt = 1.0 / cfg['control_freq']
    n_steps = int(duration / dt)
    
    banner(grade)
    
    # 初始化
    sensors = init_sensors(grade)
    fusion = init_fusion(grade)
    
    # 打开传感器
    for name, sensor in sensors.items():
        sensor.open()
    
    # 姿态估计器
    pose_estimator = PoseEstimator(
        algorithm='madgwick' if cfg['control_freq'] <= 200 else 'ekf',
        sample_rate=cfg['control_freq'],
        beta=0.1
    )
    
    # 统计
    tactile_frames = 0
    force_frames = 0
    imu_frames = 0
    fusion_updates = 0
    contacts_detected = 0
    contacts_total_force = 0.0
    slip_events = 0
    euler_history = []
    
    print(f"\n  开始模拟 {duration}s @ {cfg['control_freq']}Hz ({n_steps} 步)...\n")
    print(f"  {'时间':>8} | {'触觉':>8} | {'力觉':>8} | {'IMU':>8} | "
          f"{'融合':>8} | {'接触':>6} | {'滑移':>6} | {'姿态(°)'}")
    print(f"  {'─'*90}")
    
    start = time.time()
    for step in range(n_steps):
        t = step * dt
        real_t = time.time() - start
        
        # 1. 传感器采集
        # 触觉 (按配置的采样率)
        t_step = int(cfg['control_freq'] / cfg['tactile']['freq'])
        if step % t_step == 0:
            tf = sensors['tactile'].capture()
            tactile_frames += 1
            contacts = sensors['tactile'].detect_contacts(tf)
            contacts_detected += len(contacts)
            if contacts:
                contacts_total_force += sum(c.contact_force for c in contacts)
            if tf.slip_signal is not None:
                slip_prob = float(np.max(tf.slip_signal))
                if slip_prob > 0.3:
                    slip_events += 1
        
        # 力觉 (按配置的采样率)
        f_step = int(cfg['control_freq'] / cfg['force']['sampling'])
        if step % f_step == 0:
            wrench = sensors['force'].capture()
            force_frames += 1
        
        # IMU (按配置的采样率)
        i_step = int(cfg['control_freq'] / cfg['imu']['sample_hz'])
        if step % i_step == 0:
            imu_frame = sensors['imu'].capture()
            imu_frames += 1
            
            # 姿态估计
            pose = pose_estimator.update(
                imu_frame.accel, 
                imu_frame.gyro,
                imu_frame.mag,
                dt=dt
            )
            euler = pose.to_euler()
            euler_deg = euler * 180.0 / math.pi
            euler_history.append(euler_deg)
            
            # 融合
            fusion.update({'accel': imu_frame.accel, 'gyro': imu_frame.gyro}, dt=dt)
            fusion_updates += 1
        
        # 每0.5秒打印一行
        if step % (max(1, n_steps // int(duration * 2))) == 0:
            euler_str = f"R:{euler_deg[0]:+.1f} P:{euler_deg[1]:+.1f} Y:{euler_deg[2]:+.1f}"
            print(f"  {t:>8.2f} | {tactile_frames:>8} | {force_frames:>8} | "
                  f"{imu_frames:>8} | {fusion_updates:>8} | "
                  f"{contacts_detected:>6} | {slip_events:>6} | {euler_str}")
    
    # 关闭传感器
    for sensor in sensors.values():
        sensor.close()
    
    # 统计报告
    print(f"\n  {'─'*70}")
    print(f"  ✅ {grade}级模拟完成 | 耗时: {real_t:.2f}s (目标: {duration:.2f}s)")
    print(f"  📊 统计:")
    print(f"     触觉帧: {tactile_frames} ({tactile_frames/duration:.1f} Hz)")
    print(f"     力觉帧: {force_frames} ({force_frames/duration:.1f} Hz)")
    print(f"     IMU帧:  {imu_frames} ({imu_frames/duration:.1f} Hz)")
    print(f"     融合更新: {fusion_updates} ({fusion_updates/duration:.1f} Hz)")
    print(f"     接触事件: {contacts_detected}")
    print(f"     接触总力: {contacts_total_force:.2f} N")
    print(f"     滑移事件: {slip_events}")
    
    if euler_history:
        euler_arr = np.array(euler_history)
        print(f"     姿态稳定性:")
        print(f"       Roll:  {np.mean(euler_arr[:,0]):+.3f}° ± {np.std(euler_arr[:,0]):+.3f}°")
        print(f"       Pitch: {np.mean(euler_arr[:,1]):+.3f}° ± {np.std(euler_arr[:,1]):+.3f}°")
        print(f"       Yaw:   {np.mean(euler_arr[:,2]):+.3f}° ± {np.std(euler_arr[:,2]):+.3f}°")
    
    return {
        'grade': grade,
        'tactile_frames': tactile_frames,
        'force_frames': force_frames,
        'imu_frames': imu_frames,
        'fusion_updates': fusion_updates,
        'contacts': contacts_detected,
        'slips': slip_events,
    }


def run_all_grades(duration: float = 3.0):
    """运行所有五级AGV对比"""
    print("\n" + "="*70)
    print("🚀 SuperModel AGV五级完整对比演示")
    print("    S → M → L → XL → XXL")
    print("="*70 + "\n")
    
    results = []
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        r = simulate_grade(grade, duration=duration)
        results.append(r)
        time.sleep(0.5)
    
    # 对比汇总
    print("\n" + "="*70)
    print("📊 五级AGV性能对比汇总")
    print("="*70)
    print(f"  {'等级':>6} | {'触觉帧':>8} | {'力觉帧':>8} | {'IMU帧':>8} | "
          f"{'融合':>8} | {'接触':>6} | {'滑移':>6}")
    print(f"  {'─'*60}")
    for r in results:
        g = r['grade']
        cfg = GRADE_CONFIG[g]
        print(f"  {g:>6} | {r['tactile_frames']:>8} | {r['force_frames']:>8} | "
              f"{r['imu_frames']:>8} | {r['fusion_updates']:>8} | "
              f"{r['contacts']:>6} | {r['slips']:>6}")
    
    print(f"\n  {'等级':>6} | {'控制频率':>10} | {'最大速度':>10} | {'最大负载':>10} | {'控制模式':>20}")
    print(f"  {'─'*65}")
    for g, cfg in GRADE_CONFIG.items():
        print(f"  {g:>6} | {cfg['control_freq']:>9}Hz | {cfg['max_speed']:>9}m/s | "
              f"{cfg['max_load']:>9}kg | {cfg['control_mode']:>20}")
    
    print(f"\n✅ 演示完成!")


def main():
    parser = argparse.ArgumentParser(description='SuperModel AGV五级演示')
    parser.add_argument('--grade', default='ALL', 
                       choices=['S', 'M', 'L', 'XL', 'XXL', 'ALL'],
                       help='AGV等级 (默认: ALL)')
    parser.add_argument('--duration', type=float, default=3.0,
                       help='每级模拟时长秒数 (默认: 3.0)')
    args = parser.parse_args()
    
    if args.grade == 'ALL':
        run_all_grades(duration=args.duration)
    else:
        simulate_grade(args.grade, duration=args.duration)


if __name__ == '__main__':
    main()
