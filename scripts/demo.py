#!/usr/bin/env python3
"""
SuperModel 超模态机器人具身智能大脑
=====================================

演示脚本: 展示所有核心模块的集成使用

运行方式:
    python3 scripts/demo.py
"""

import numpy as np
import time
import sys

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, CameraIntrinsics, StereoExtrinsics
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray, PressureProcessor
from sensors.force import ForceTorqueSensor, Wrench
from sensors.imu import IMUSensor, PoseEstimator

from fusion import CrossModalFusion
from perception import MultimodalInput, FusionConfig

from control.motion import MotionController, ControlMode
from control.impedance import ImpedanceController, ImpedanceParams
from control.skill import SkillLibrary

from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


def demo_sensors():
    """演示传感器数据采集"""
    print("\n" + "="*60)
    print("传感器模块演示")
    print("="*60)
    
    # 双目视觉
    print("\n[1] 双目视觉")
    with BinocularCamera() as cam:
        frame = cam.capture()
        print(f"    左图像: {frame.left_image.shape}")
        print(f"    右图像: {frame.right_image.shape}")
    
    # 双耳听觉
    print("\n[2] 双耳听觉")
    with BinauralMic() as mic:
        frame = mic.capture()
        print(f"    左通道: {len(frame.left_channel)} 采样点")
        print(f"    右通道: {len(frame.right_channel)} 采样点")
    
    # 触觉感知
    print("\n[3] 触觉感知")
    tactile = TactileArray(array_size=(16, 16))
    tactile.open()
    frame = tactile.capture()
    contacts = tactile.detect_contacts(frame)
    print(f"    压力图: {frame.pressure_map.shape}")
    print(f"    检测到接触: {len(contacts)} 个")
    tactile.close()
    
    # 六维力矩
    print("\n[4] 六维力矩")
    force_sensor = ForceTorqueSensor()
    force_sensor.open()
    wrench = force_sensor.capture()
    print(f"    力向量: {wrench.force}")
    print(f"    力矩: {wrench.torque}")
    force_sensor.close()
    
    # IMU
    print("\n[5] IMU + 姿态估计")
    with IMUSensor() as imu:
        frame = imu.capture()
        print(f"    加速度: {frame.accel}")
        print(f"    角速度: {frame.gyro}")
    
    print("\n✅ 传感器演示完成")


def demo_fusion():
    """演示跨模态融合"""
    print("\n" + "="*60)
    print("跨模态融合演示")
    print("="*60)
    
    import torch
    
    # 配置
    config = FusionConfig(
        vision_dim=512,
        audio_dim=128,
        tactile_dim=64,
        force_dim=32,
        imu_dim=64,
        hidden_dim=256,
        num_heads=4
    )
    
    fusion = CrossModalFusion(config)
    
    # 创建多模态输入 (使用torch tensors)
    multimodal = MultimodalInput(
        vision=torch.randn(1, 512),
        audio=torch.randn(1, 128),
        tactile=torch.randn(1, 64),
        force=torch.randn(1, 32),
        imu=torch.randn(1, 64)
    )
    
    # 融合
    fused_features = fusion(multimodal)
    
    print(f"    统一特征形状: {fused_features.shape}")
    print(f"    可用模态: {multimodal.modalities}")
    
    print("\n✅ 跨模态融合演示完成")


def demo_control():
    """演示运动控制"""
    print("\n" + "="*60)
    print("运动控制演示")
    print("="*60)
    
    # 仿真器
    sim_config = SimConfig(dt=0.01, num_joints=6)
    sim = RobotSimulator(sim_config)
    sensor_sim = SensorSimulator(sim, sim_config)
    
    # 控制器
    controller = MotionController(num_joints=6, control_rate=100)
    controller.kp = np.ones(6) * 2.0
    controller.ki = np.ones(6) * 0.1
    controller.kd = np.ones(6) * 0.5
    
    # 阻抗控制器
    imp_params = ImpedanceParams.default_6d()
    imp_ctrl = ImpedanceController(imp_params)
    
    # 目标位置
    target = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    
    print(f"\n    目标位置: {target}")
    print(f"    仿真步数: 100")
    
    # 仿真控制循环
    for step in range(100):
        # PID 控制
        controller._current_joint_pos = sim.joint_positions.copy()
        torque = controller.compute_joint_torque(target)
        state = sim.step(torque)
    
    print(f"    最终位置: {state['joint_positions']}")
    print(f"    最终速度: {state['joint_velocities']}")
    
    print("\n✅ 运动控制演示完成")


def demo_impedance():
    """演示阻抗控制"""
    print("\n" + "="*60)
    print("阻抗控制演示")
    print("="*60)
    
    # 仿真器
    sim = RobotSimulator(SimConfig(num_joints=6))
    imp_params = ImpedanceParams.default_6d()
    imp_ctrl = ImpedanceController(imp_params)
    
    # 阻抗控制循环
    for step in range(50):
        desired_pos = np.array([0.5, 0.0, 0.3])
        current_pos = sim.end_effector_pose[:3, 3]
        jacobian = sim.get_jacobian()[:3, :]
        
        torque = imp_ctrl.compute_torque(
            desired_position=desired_pos,
            desired_velocity=np.zeros(3),
            current_position=current_pos,
            current_velocity=np.zeros(3),
            external_wrench=np.zeros(6),
            jacobian=jacobian
        )
        
        sim.step(torque)
    
    print(f"    末端位置: {sim.end_effector_pose[:3, 3]}")
    
    print("\n✅ 阻抗控制演示完成")


def demo_skill_library():
    """演示技能库"""
    print("\n" + "="*60)
    print("技能库演示")
    print("="*60)
    
    lib = SkillLibrary()
    skills = lib.list_skills()
    print(f"    可用技能: {skills}")
    
    # 创建移动技能
    skill = lib.create_skill("move_to", {"target": [0.5, 0.0, 0.3]})
    print(f"    创建技能: {skill.config.name if skill else 'None'}")
    
    print("\n✅ 技能库演示完成")


def demo_integration():
    """完整集成演示"""
    print("\n" + "="*60)
    print("完整集成演示")
    print("="*60)
    
    # 1. 初始化所有模块
    print("\n[1] 初始化模块...")
    
    sensors = {
        'vision': BinocularCamera(),
        'audio': BinauralMic(),
        'tactile': TactileArray(array_size=(16, 16)),
        'force': ForceTorqueSensor(),
        'imu': IMUSensor()
    }
    
    fusion_config = FusionConfig(
        vision_dim=512, audio_dim=128, tactile_dim=64,
        force_dim=32, imu_dim=64, hidden_dim=256
    )
    fusion = CrossModalFusion(fusion_config)
    
    sim = RobotSimulator(SimConfig(num_joints=6))
    controller = MotionController(num_joints=6)
    
    # 2. 打开传感器
    print("[2] 打开传感器...")
    for name, sensor in sensors.items():
        sensor.open()
        print(f"    {name}: OK")
    
    # 3. 主循环
    print("[3] 运行主循环...")
    import torch
    for step in range(10):
        # 采集传感器数据
        vision_frame = sensors['vision'].capture()
        audio_frame = sensors['audio'].capture()
        tactile_frame = sensors['tactile'].capture()
        force_wrench = sensors['force'].capture()
        imu_frame = sensors['imu'].capture()
        
        # 融合 (使用torch tensors)
        # 注意: 实际需要传感器编码器将原始数据转换为特征向量
        # 此处使用模拟特征向量演示集成
        multimodal = MultimodalInput(
            vision=torch.randn(1, 512),  # 模拟视觉特征
            audio=torch.randn(1, 128),    # 模拟音频特征
            tactile=torch.randn(1, 64),   # 模拟触觉特征
            force=torch.randn(1, 32),     # 模拟力觉特征
            imu=torch.randn(1, 64)       # 模拟IMU特征
        )
        
        fused = fusion(multimodal)
        if step % 5 == 0:
            print(f"    Step {step}: fused={fused.shape}")
        
        # 简单控制
        target = np.array([0.1 * step, 0.0, 0.0, 0.0, 0.0, 0.0])
        controller._current_joint_pos = sim.joint_positions.copy()
        torque = controller.compute_joint_torque(target)
        state = sim.step(torque)
        
        if step % 5 == 0:
            print(f"    Step {step}: pos={state['joint_positions'][:3]}")
    
    # 4. 关闭传感器
    print("[4] 关闭传感器...")
    for name, sensor in sensors.items():
        sensor.close()
    
    print("\n✅ 完整集成演示完成")


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("SuperModel 超模态机器人具身智能大脑")
    print("="*60)
    
    try:
        demo_sensors()
        demo_fusion()
        demo_control()
        demo_impedance()
        demo_skill_library()
        demo_integration()
        
        print("\n" + "="*60)
        print("🎉 所有演示完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
