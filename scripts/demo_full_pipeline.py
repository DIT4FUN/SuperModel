#!/usr/bin/env python3
"""
SuperModel 全系统演示脚本
=======================

展示从感知到控制的完整数据流:
1. 多模态传感器采集
2. 跨模态特征融合
3. 世界模型推理
4. 运动控制输出

运行: python3 scripts/demo_full_pipeline.py
"""

import sys
import time
import numpy as np
sys.path.insert(0, 'src')

from sensors.vision import BinocularCamera, get_stereo_spec
from sensors.audio import BinauralMic, SoundLocalizer, get_audio_spec
from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from sensors.force import ForceTorqueSensor, ForceSensorType, Wrench
from sensors.imu import IMUSensor, IMUSensorType, PoseEstimator

from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, FusionStrategy,
    MultimodalInput, UnifiedRepresentation
)

from control.motion import MotionController, JointState, ControlMode
from control.impedance import ImpedanceController, ImpedanceParams

from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_sensor_layer(grade='M'):
    """演示感知层"""
    print_section("感知层 (Perception)")

    # 双目相机
    print("\n[1] 双目相机 (BinocularCamera)")
    cam = BinocularCamera(resolution=(1280, 720), fps=30)
    cam.open()
    stereo_frame = cam.capture()
    print(f"    分辨率: {stereo_frame.left_image.shape}")
    print(f"    时间戳: {stereo_frame.timestamp:.4f}s")
    cam.close()

    # 双耳麦克风
    print("\n[2] 双耳麦克风 (BinauralMic)")
    mic = BinauralMic(sample_rate=16000)
    mic.open()
    audio_frame = mic.capture()
    print(f"    采样率: {audio_frame.sample_rate}Hz")
    print(f"    样本数: {len(audio_frame.left_channel)}")
    mic.close()

    # 触觉传感器
    print("\n[3] 触觉传感器 (TactileArray)")
    tactile = TactileArray(
        array_size=get_tactile_spec(grade)['array'],
        sensor_type=TactileSensorType.CAPACITIVE
    )
    tactile.open()
    tactile_frame = tactile.capture()
    contacts = tactile.detect_contacts(tactile_frame)
    print(f"    阵列: {tactile_frame.pressure_map.shape}")
    print(f"    接触点: {len(contacts)}")
    if contacts:
        c = contacts[0]
        print(f"    峰值压力: {c.peak_pressure:.3f}, 力: {c.contact_force:.2f}N")
    tactile.close()

    # 力矩传感器
    print("\n[4] 六维力矩传感器 (ForceTorqueSensor)")
    ft = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
    ft.open()
    wrench = ft.capture()
    print(f"    力: [{wrench.force[0]:.2f}, {wrench.force[1]:.2f}, {wrench.force[2]:.2f}] N")
    print(f"    力矩: [{wrench.torque[0]:.3f}, {wrench.torque[1]:.3f}, {wrench.torque[2]:.3f}] N·m")
    contact = ft.detect_contact(wrench)
    print(f"    接触: {contact.is_contact}, 力: {contact.contact_force:.2f}N")
    ft.close()

    # IMU
    print("\n[5] IMU传感器 (IMUSensor)")
    imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
    imu.open()
    imu_frame = imu.capture()
    print(f"    加速度: {imu_frame.accel}")
    print(f"    角速度: {imu_frame.gyro}")
    print(f"    |accel|: {imu_frame.accel_magnitude:.3f} m/s²")

    # 姿态估计
    estimator = PoseEstimator(algorithm='madgwick', sample_rate=200.0)
    pose = estimator.update(imu_frame.accel, imu_frame.gyro)
    euler = pose.to_euler() * 180 / np.pi
    print(f"    姿态 (Euler): roll={euler[0]:.2f}°, pitch={euler[1]:.2f}°, yaw={euler[2]:.2f}°")
    imu.close()

    return {
        'stereo_frame': stereo_frame,
        'audio_frame': audio_frame,
        'tactile_frame': tactile_frame,
        'wrench': wrench,
        'imu_frame': imu_frame
    }


def demo_fusion_layer():
    """演示融合层"""
    print_section("融合层 (Fusion)")

    import torch

    # 融合配置
    config = FusionConfig(
        vision_dim=512,
        audio_dim=128,
        tactile_dim=64,
        force_dim=32,
        imu_dim=64,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        strategy=FusionStrategy.HYBRID
    )
    fusion = CrossModalFusion(config)
    unified = UnifiedRepresentation(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim,
        output_dim=128
    )

    print(f"\n融合网络配置:")
    print(f"    隐层维度: {config.hidden_dim}")
    print(f"    注意力头数: {config.num_heads}")
    print(f"    融合层数: {config.num_layers}")

    # 模拟多模态输入
    multimodal = MultimodalInput(
        vision=torch.randn(2, 512),
        audio=torch.randn(2, 128),
        tactile=torch.randn(2, 64),
        force=torch.randn(2, 32),
        imu=torch.randn(2, 64)
    )

    print(f"\n输入模态: {multimodal.modalities}")

    # 前向传播
    fused = fusion(multimodal)
    print(f"    融合特征维度: {fused.shape}")

    # 统一表示
    state, action, world = unified(fused)
    print(f"\n统一表示:")
    print(f"    状态表示: {state.shape}")
    print(f"    动作表示: {action.shape}")
    print(f"    世界模型表示: {world.shape}")

    # 只用部分模态
    partial = MultimodalInput(
        vision=torch.randn(2, 512),
        tactile=torch.randn(2, 64)
    )
    fused_partial = fusion(partial)
    print(f"\n部分模态融合 (vision+tactile): {fused_partial.shape}")

    return fusion, unified


def demo_control_layer():
    """演示控制层"""
    print_section("执行层 (Execution)")

    num_joints = 6

    # 运动控制器
    print(f"\n[1] 运动控制器 (MotionController)")
    mc = MotionController(
        num_joints=num_joints,
        control_rate=100.0,
    )
    mc.set_pid_gains(
        kp=np.ones(num_joints) * 2.0,
        ki=np.zeros(num_joints),
        kd=np.ones(num_joints) * 0.5
    )

    # 更新关节状态
    current = JointState(
        position=np.zeros(num_joints),
        velocity=np.zeros(num_joints),
        torque=np.zeros(num_joints)
    )
    mc.update_joint_state(current)

    # 跟踪轨迹
    target_positions = [
        np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.0]),
        np.array([0.8, 0.5, -0.3, 0.2, 0.6, 0.1]),
        np.array([1.0, 0.7, -0.4, 0.3, 0.8, 0.2]),
    ]

    print("    轨迹跟踪:")
    for i, target in enumerate(target_positions):
        torque = mc.compute_joint_torque(target)
        print(f"    Step {i+1}: target=[{target[0]:.2f}, {target[1]:.2f}, ...], torque=[{torque[0]:.2f}, {torque[1]:.2f}, ...]")

    # 阻抗控制器
    print(f"\n[2] 阻抗控制器 (ImpedanceController)")
    imp = ImpedanceController(impedance_params=ImpedanceParams.default_6d())

    desired_pos = np.array([0.5, 0.3, 0.2])
    desired_vel = np.zeros(3)
    current_pos = np.array([0.4, 0.2, 0.1])
    current_vel = np.zeros(3)
    external_wrench = np.array([2.0, 1.0, 0.5, 0.1, 0.1, 0.1])
    jacobian = np.random.randn(6, num_joints) * 0.1

    joint_torque = imp.compute_torque(
        desired_pos, desired_vel,
        current_pos, current_vel,
        external_wrench, jacobian
    )
    print(f"    关节力矩 (前3个): {joint_torque[:3]}")

    # 导纳控制器
    print(f"\n[3] 导纳控制器 (AdmittanceController)")
    from control.impedance import AdmittanceController
    adm = AdmittanceController(M=10.0, D=50.0, K=200.0)

    forces = [0.0, 2.0, 5.0, 8.0, 10.0, 5.0, 0.0]
    positions = []
    for f in forces:
        p = adm.update(external_force=f, desired_position=0.0)
        positions.append(p)
    print(f"    力→位置映射: {[(f, f'{p:.4f}') for f, p in zip(forces, positions)]}")


def demo_simulation():
    """演示仿真环境"""
    print_section("仿真层 (Simulation)")

    # 仿真配置
    config = SimConfig(
        dt=0.01,
        num_joints=6,
        position_noise=0.001,
        velocity_noise=0.01
    )

    # 创建仿真器
    sim = RobotSimulator(config=config)
    sensor_sim = SensorSimulator(sim, config)

    print(f"\n[1] 机器人仿真器")
    print(f"    关节数: {sim.n}")
    print(f"    时间步长: {config.dt}s")

    # 施加力矩跟踪轨迹
    target = np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.0])

    print(f"\n[2] 仿真步进 (100步)")
    for step in range(100):
        state = sim.step(target)

        if step % 20 == 0:
            pos = state['joint_positions'][:3]
            vel = state['joint_velocities'][:3]
            print(f"    Step {step:3d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], vel=[{vel[0]:.3f}, ...]")

    print(f"\n[3] 传感器仿真")
    noisy_pos = sensor_sim.get_noisy_joint_positions()
    noisy_vel = sensor_sim.get_noisy_joint_velocities()
    imu_data = sensor_sim.get_imu_data()
    wrench_sim = sensor_sim.get_wrench()

    print(f"    带噪声关节位置: {noisy_pos[:3]}")
    print(f"    带噪声关节速度: {noisy_vel[:3]}")
    print(f"    IMU加速度: {imu_data['accel']}")
    print(f"    末端力矩: {wrench_sim[:3]}")


def demo_agv_grades():
    """演示AGV五级规格"""
    print_section("AGV五级规格")

    from sensors.vision import get_stereo_spec
    from sensors.audio import get_audio_spec
    from sensors.tactile import get_tactile_spec
    from sensors.force import get_force_spec
    from sensors.imu import get_imu_spec

    grades = ['S', 'M', 'L', 'XL', 'XXL']

    print("\nAGV等级对比:")
    print(f"{'等级':<6} {'视觉分辨率':<14} {'触觉阵列':<12} {'力觉轴数':<10} {'IMU采样':<10} {'算力TOPS':<10}")
    print("-" * 65)

    specs = {
        'S': {'vision': '640×480', 'tactile': '8×8', 'force': '3', 'imu': '100Hz', 'compute': '<5'},
        'M': {'vision': '1280×720', 'tactile': '16×16', 'force': '6', 'imu': '200Hz', 'compute': '5-20'},
        'L': {'vision': '1280×720', 'tactile': '24×24', 'force': '6', 'imu': '500Hz', 'compute': '20-100'},
        'XL': {'vision': '1920×1080', 'tactile': '32×32', 'force': '6', 'imu': '1000Hz', 'compute': '100-300'},
        'XXL': {'vision': '多目', 'tactile': '48×48', 'force': '6', 'imu': '2000Hz', 'compute': '>300'},
    }

    for g in grades:
        s = specs[g]
        print(f"{g:<6} {s['vision']:<14} {s['tactile']:<12} {s['force']:<10} {s['imu']:<10} {s['compute']:<10}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     SuperModel 超模态机器人具身智能大脑 - 全系统演示           ║
║                                                              ║
║     感知层 → 融合层 → 认知层 → 执行层 → 仿真层                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    start_time = time.time()

    # 各层演示
    demo_agv_grades()
    sensor_data = demo_sensor_layer(grade='M')
    fusion, unified = demo_fusion_layer()
    demo_control_layer()
    demo_simulation()

    # 统计
    elapsed = time.time() - start_time
    print_section("演示完成")
    print(f"\n总耗时: {elapsed:.2f}秒")
    print("\n✅ SuperModel 全系统演示成功!")
    print("""
模块说明:
  • 感知层: 5种传感器 (视觉/听觉/触觉/力觉/IMU)
  • 融合层: 跨模态注意力网络
  • 执行层: PID控制 + 阻抗控制
  • 仿真层: 简化物理引擎 + 传感器噪声

下一步:
  1. 运行测试: python3 -m pytest tests/ -v
  2. 查看文档: docs/design/SYSTEM_ARCHITECTURE.md
  3. 学习源码: src/sensors/, src/fusion/, src/control/
""")


if __name__ == '__main__':
    main()
