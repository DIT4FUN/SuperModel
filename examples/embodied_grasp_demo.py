#!/usr/bin/env python3
"""
SuperModel 具身感知抓取演示
=============================

展示触觉 + 力觉 + IMU 三传感器协同的智能抓取场景:
1. 多模态传感器初始化 (触觉阵列 + 六维力矩 + IMU)
2. 接触检测与抓取质量评估
3. 滑移检测与自适应握力控制
4. 姿态稳定与运动估计
5. 仿真抓取任务全流程

运行: python3 examples/embodied_grasp_demo.py
"""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact,
    TactileSensorType, PressureProcessor, VirtualTactileSensor,
    get_tactile_spec, TactileCalibration
)
from sensors.force import (
    ForceTorqueSensor, Wrench, ContactState, WrenchProcessor,
    ForceSensorType, VirtualForceSensor, get_force_spec
)
from sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator,
    IMUSensorType, VirtualIMUSensor, get_imu_spec
)

from control.tactile_control import TactileServoController, TactileServoParams, GraspQualityController
from control.force_control import ForceController, ForceControlParams, HybridForcePositionController
from control.imu_control import AttitudeStabilizer, IMUControlParams


def demo_sensor_initialization():
    """演示1: 多模态传感器初始化 (S/M/L/XL/XXL五级配置)"""
    print("\n" + "=" * 60)
    print("📡 具身感知抓取演示 - 传感器初始化")
    print("=" * 60)

    grades = ['S', 'M', 'L', 'XL', 'XXL']
    
    for grade in grades:
        t_spec = get_tactile_spec(grade)
        f_spec = get_force_spec(grade)
        i_spec = get_imu_spec(grade)
        
        print(f"\n  [{grade}级配置]")
        print(f"    触觉: {t_spec['array']} @ {t_spec['freq_hz']}Hz, 压力范围 {t_spec['range_kpa'][0]}-{t_spec['range_kpa'][1]}kPa")
        print(f"    力觉: {f_spec['axes']}轴, 力范围 ±{f_spec['force_range']}N, 采样 {f_spec['sampling_hz']}Hz")
        print(f"    IMU:  {i_spec['type']}, 采样 {i_spec['sample_hz']}Hz, 噪声 {i_spec['noise_density']}μg/√Hz")
    
    # 实际初始化 (使用M级配置)
    print("\n  初始化M级具身感知系统...")
    
    tactile = TactileArray(
        array_size=(16, 16),
        sensor_type=TactileSensorType.CAPACITIVE,
        sensor_id="tactile_efskin_01"
    )
    
    force = ForceTorqueSensor(
        sensor_type=ForceSensorType.SIX_AXIS,
        sensor_id="ft_ati_omega_01"
    )
    
    imu = IMUSensor(
        sensor_type=IMUSensorType.BMI088,
        sensor_id="imu_bmi088_01",
        sample_rate=200
    )
    
    tactile.open()
    force.open()
    imu.open()
    
    print("  ✅ 触觉/力觉/IMU 传感器已打开")
    return tactile, force, imu


def demo_contact_detection(tactile, force):
    """演示2: 接触检测与抓取质量评估"""
    print("\n" + "=" * 60)
    print("🤚 接触检测与抓取质量评估")
    print("=" * 60)
    
    # 采集触觉帧序列 (模拟接触)
    print("\n  采集触觉数据...")
    for i in range(5):
        frame = tactile.capture()
        if i == 0:
            print(f"    帧{i}: 压力峰值={frame.pressure_map.max():.3f}, 均值={frame.pressure_map.mean():.3f}")
    
    # 抓取质量评估
    contacts = tactile.detect_contacts()
    grip_quality = tactile.estimate_grip_quality()
    
    print(f"\n  检测到 {len(contacts)} 个接触区域:")
    for j, c in enumerate(contacts):
        print(f"    接触{j+1}: 中心={c.center}, 面积={c.area}像素, 峰值压力={c.peak_pressure:.3f}")
    
    print(f"\n  抓取质量评分:")
    print(f"    综合评分: {grip_quality['overall']:.2f}")
    print(f"    接触面积: {grip_quality['contact_area']:.2f}")
    print(f"    均匀性:   {grip_quality['uniformity']:.2f}")
    print(f"    稳定性:   {grip_quality['stability']:.2f}")
    
    # 力传感器数据
    wrench = force.capture()
    contact_state = force.detect_contact(wrench)
    
    print(f"\n  力觉数据:")
    print(f"    接触状态: {'接触中' if contact_state.is_contact else '无接触'}")
    print(f"    接触力:   {contact_state.contact_force:.2f}N")
    print(f"    力向量:   Fx={wrench.force[0]:.2f}N, Fy={wrench.force[1]:.2f}N, Fz={wrench.force[2]:.2f}N")
    print(f"    力矩:     Tx={wrench.torque[0]:.3f}Nm, Ty={wrench.torque[1]:.3f}Nm, Tz={wrench.torque[2]:.3f}Nm")


def demo_slip_detection():
    """演示3: 滑移检测与自适应握力控制"""
    print("\n" + "=" * 60)
    print("⚠️  滑移检测与自适应握力控制")
    print("=" * 60)
    
    # 使用真实触觉传感器进行滑移检测
    tactile = TactileArray(array_size=(16, 16), sensor_id="slip_test_sensor")
    tactile.open()
    
    # 创建触觉伺服控制器
    params = TactileServoParams(
        slip_threshold=0.15,
        max_force=20.0
    )
    servo = TactileServoController(tactile, params)
    
    print(f"\n  初始握力: 5.0N")
    
    # 模拟滑移动作序列 (使用虚拟触觉)
    v_tactile = VirtualTactileSensor(array_size=(16, 16))
    v_tactile.open()
    
    directions = [
        (0.1, 0.0),    # 向右
        (0.0, 0.1),   # 向上
        (-0.1, 0.0),  # 向左
        (0.05, -0.05), # 对角
    ]
    
    for step, direction in enumerate(directions):
        # 模拟滑动
        frames = v_tactile.simulate_sliding(
            direction=direction,
            speed=0.05,
            duration_frames=10
        )
        last_frame = frames[-1]
        
        # 滑移检测 - 直接计算滑移信号
        slip_signal = tactile.get_slip_signal(last_frame)
        avg_slip = np.mean(slip_signal[slip_signal > 0.05]) if slip_signal.max() > 0.05 else 0.0
        
        # 获取触觉控制器状态
        is_contact = servo.is_contact(last_frame)
        quality_metrics = servo.monitor_grasp_quality()
        
        # 手动计算滑移响应
        if avg_slip > 0.15:
            slip_status = "⚠️ 滑移"
        else:
            slip_status = "✅ 稳定"
        
        print(f"\n  步骤{step+1}: 方向={direction}, 接触={is_contact} {slip_status}")
        print(f"    平均滑移: {avg_slip:.3f}, 最大滑移: {slip_signal.max():.3f}")
        print(f"    抓取质量: {quality_metrics.get('overall', 0.0):.3f}")
    
    v_tactile.close()
    tactile.close()


def demo_pose_stability(imu):
    """演示4: 姿态稳定与运动估计"""
    print("\n" + "=" * 60)
    print("🧭 姿态稳定与运动估计")
    print("=" * 60)
    
    # 姿态估计器 (Madgwick AHRS)
    pose_est = PoseEstimator(algorithm='madgwick', sample_rate=200, beta=0.1)
    
    # 校准
    print("\n  IMU校准中...")
    imu.calibrate_gyro_bias(num_samples=200)
    print("  ✅ 陀螺仪偏置校准完成")
    
    print("\n  采集10帧数据验证姿态估计:")
    for i in range(10):
        frame = imu.capture()
        pose = pose_est.update(frame.accel, frame.gyro)
        euler = pose.to_euler()
        
        if i % 5 == 0:
            print(f"\n  帧{i}: 加速度=[{frame.accel[0]:.2f}, {frame.accel[1]:.2f}, {frame.accel[2]:.2f}] m/s²")
            print(f"       角速度=[{frame.gyro[0]:.3f}, {frame.gyro[1]:.3f}, {frame.gyro[2]:.3f}] rad/s")
            print(f"       姿态: Roll={np.degrees(euler[0]):.1f}°, Pitch={np.degrees(euler[1]):.1f}°, Yaw={np.degrees(euler[2]):.1f}°")
    
    # 姿态稳定控制器
    imu_params = IMUControlParams()
    stabilizer = AttitudeStabilizer(imu, imu_params)
    stabilizer.set_target_attitude(roll=0.0, pitch=0.0, yaw=0.0)
    
    # 获取当前估计姿态
    current_euler = pose_est.get_euler()
    print(f"\n  姿态稳定控制:")
    print(f"    当前: Roll={np.degrees(current_euler[0]):.1f}°, Pitch={np.degrees(current_euler[1]):.1f}°")
    print(f"    目标: Roll=0.0°, Pitch=0.0°")
    
    # 更新稳定器获取控制力矩
    torque_cmd = stabilizer.update(imu.capture(), dt=0.01)
    print(f"    输出力矩: Tx={torque_cmd[0]:.2f}Nm, Ty={torque_cmd[1]:.2f}Nm, Tz={torque_cmd[2]:.2f}Nm")


def demo_grasp_pipeline():
    """演示5: 完整抓取管道 (从感知到控制)"""
    print("\n" + "=" * 60)
    print("🔄 完整抓取管道演示")
    print("=" * 60)
    
    # 初始化虚拟传感器
    tactile = VirtualTactileSensor(array_size=(24, 24), sensor_id="virtual_tactile")
    force = VirtualForceSensor(sensor_id="virtual_force", noise_level=0.05)
    imu = VirtualIMUSensor(sensor_id="virtual_imu")
    
    tactile.open()
    force.open()
    imu.open()
    
    print("\n  虚拟传感器已初始化 (24×24触觉阵列 + 六维力矩 + IMU)")
    
    # 初始化控制器
    force_ctrl = ForceController(ForceControlParams())
    pose_ctrl = AttitudeStabilizer(imu, IMUControlParams())
    
    # 创建配套的触觉阵列用于分析 (虚拟传感器仅返回TactileFrame)
    real_tactile = TactileArray(array_size=(24, 24), sensor_id="tactile_for_analysis")
    real_tactile.open()
    tactile_ctrl = TactileServoController(real_tactile, TactileServoParams())
    
    # 抓取序列
    grasp_sequence = [
        ("接近", 0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("预接触", 1.0, (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
        ("闭合", 3.0, (0.0, 0.0, -5.0), (0.0, 0.0, 0.0)),
        ("举起", 2.0, (0.0, 0.0, 5.0), (0.0, 0.0, 0.0)),
        ("移动", 2.0, (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ("放下", 2.0, (0.0, 0.0, -5.0), (0.0, 0.0, 0.0)),
        ("松开", 1.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
    ]
    
    grip_force = 0.0
    
    for phase, duration, linear_accel, angular_vel in grasp_sequence:
        print(f"\n  ── 阶段: {phase} ({duration}s) ──")
        
        # IMU采集
        imu_frame = imu.simulate_motion(linear_accel, angular_vel)
        
        # 触觉采集
        if phase in ["预接触", "闭合", "举起", "移动", "放下"]:
            tactile_frame = tactile.simulate_contact(
                contact_pos=(0.5, 0.5),
                contact_radius=0.25,
                contact_force=grip_force,
                noise_level=0.03
            )
            # 使用真实触觉传感器分析虚拟帧
            contacts = real_tactile.detect_contacts(tactile_frame)
            grip_quality = real_tactile.estimate_grip_quality(tactile_frame)
            slip_signal = real_tactile.get_slip_signal(tactile_frame)
            is_contact = tactile_ctrl.is_contact(tactile_frame)
        else:
            contacts = []
            grip_quality = {'overall': 0.0}
            slip_signal = np.zeros((24, 24))
            is_contact = False
        
        # 力采集
        wrench = force.simulate_contact(
            force=(0.0, 0.0, -grip_force),
            torque=(0.0, 0.0, 0.0)
        )
        
        # 控制计算
        if phase == "闭合":
            grip_force = min(grip_force + 0.5, 8.0)
        elif phase == "松开":
            grip_force = max(grip_force - 0.5, 0.0)
        
        print(f"    握力: {grip_force:.1f}N, 接触点: {len(contacts)}, 接触: {is_contact}")
        print(f"    力觉: Fz={wrench.force[2]:.1f}N")
        print(f"    IMU: accel_mag={imu_frame.accel_magnitude:.2f}m/s²")
    
    print("\n  ✅ 完整抓取流程完成")
    
    real_tactile.close()
    tactile.close()
    force.close()
    imu.close()
    
    print("\n  ✅ 完整抓取流程完成")
    
    tactile.close()
    force.close()
    imu.close()


def demo_all_grades():
    """演示6: AGV五级配置完整对比"""
    print("\n" + "=" * 60)
    print("📊 AGV五级具身感知规格对比")
    print("=" * 60)
    
    grades = ['S', 'M', 'L', 'XL', 'XXL']
    
    print("\n  触觉传感器规格:")
    print(f"  {'等级':<6} {'阵列':<12} {'采样Hz':<10} {'压力范围(kPa)':<18} {'特性'}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*18} {'-'*20}")
    for g in grades:
        s = get_tactile_spec(g)
        features = []
        if s['temp']: features.append("温度")
        print(f"  {g:<6} {str(s['array']):<12} {s['freq_hz']:<10} {str(s['range_kpa']):<18} {','.join(features) if features else '-'}")
    
    print("\n  力觉传感器规格:")
    print(f"  {'等级':<6} {'轴数':<8} {'力范围N':<12} {'力矩范围Nm':<14} {'采样Hz':<10}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*14} {'-'*10}")
    for g in grades:
        s = get_force_spec(g)
        print(f"  {g:<6} {s['axes']:<8} ±{s['force_range']:<10} ±{s['torque_range']:<12} {s['sampling_hz']}")
    
    print("\n  IMU传感器规格:")
    print(f"  {'等级':<6} {'型号':<14} {'加速度量程g':<16} {'采样Hz':<10} {'噪声密度'}")
    print(f"  {'-'*6} {'-'*14} {'-'*16} {'-'*10} {'-'*12}")
    for g in grades:
        s = get_imu_spec(g)
        print(f"  {g:<6} {s['type']:<14} ±{s['accel_range']:<14} {s['sample_hz']:<10} {s['noise_density']}μg/√Hz")


def main():
    print("\n" + "=" * 60)
    print("🦾 SuperModel 具身感知抓取演示 v1.0")
    print("=" * 60)
    print("触发: 触觉 + 力觉 + IMU 三传感器协同智能抓取")
    
    # 演示1: 传感器初始化
    tactile, force, imu = demo_sensor_initialization()
    
    # 演示2: 接触检测
    demo_contact_detection(tactile, force)
    
    # 演示3: 滑移检测
    demo_slip_detection()
    
    # 演示4: 姿态稳定
    demo_pose_stability(imu)
    
    # 演示5: 完整抓取管道
    demo_grasp_pipeline()
    
    # 演示6: AGV五级规格对比
    demo_all_grades()
    
    # 清理
    tactile.close()
    force.close()
    imu.close()
    
    print("\n" + "=" * 60)
    print("✅ 具身感知抓取演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
