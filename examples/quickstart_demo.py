#!/usr/bin/env python3
"""
SuperModel 快速入门演示
========================
传感器采集 → 多模态融合 → 决策控制 → 仿真反馈

运行方式: python3 examples/quickstart_demo.py
"""

import numpy as np
import sys
import time
import torch

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, StereoFrame
from sensors.audio import BinauralMic, AudioFrame
from sensors.tactile import TactileArray, TactileFrame
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, IMUSensorType
from sensors.manager import SensorManager, SensorManagerConfig

from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, FusionStrategy, MultimodalInput
)

from control.agv import AGVMotionController, AGVGrade, AGVSpec, AGVPose, AGVTwist, DriveType
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
from control.motion import MotionController

from simulation.gym_env import SuperModelGymEnv, GymEnvConfig


def demo_sensor_collection():
    """演示: 多模态传感器采集"""
    print("\n" + "="*60)
    print("📡 传感器采集演示")
    print("="*60)
    
    # 视觉 - 双目相机
    cam = BinocularCamera(resolution=(640, 480), fps=30)
    cam.open()
    stereo_frame = cam.capture()
    print(f"  📷 双目视觉: {stereo_frame.left_image.shape}")
    cam.close()
    
    # 听觉 - 双耳麦克风
    mic = BinauralMic(sample_rate=16000)
    mic.open()
    audio_frame = mic.capture()
    print(f"  🎤 双耳听觉: {audio_frame.left_channel.shape}")
    mic.close()
    
    # 触觉 - 压力阵列
    tactile = TactileArray(array_size=(16, 16))
    tactile.open()
    tactile_frame = tactile.capture()
    print(f"  ✋ 触觉感知: pressure_map {tactile_frame.pressure_map.shape}")
    tactile.close()
    
    # 力觉 - 六轴力矩传感器
    force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
    force.open()
    wrench = force.capture()
    print(f"  💪 力觉感知: F={wrench.force}, τ={wrench.torque}")
    force.close()
    
    # IMU - 惯性测量单元
    imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
    imu.open()
    imu_frame = imu.capture()
    print(f"  🧭 IMU数据: accel={imu_frame.accel}, gyro_norm={imu_frame.gyro_magnitude:.3f} rad/s")
    imu.close()
    
    return stereo_frame, audio_frame, tactile_frame, wrench, imu_frame


def demo_sensor_manager():
    """演示: 统一传感器管理器"""
    print("\n" + "="*60)
    print("🔧 统一传感器管理器")
    print("="*60)
    
    config = SensorManagerConfig(grade="M")
    manager = SensorManager(config)
    
    manager.open_all()
    
    # 同步采集所有传感器
    frame = manager.capture_all()
    print(f"  时间戳: {frame.timestamp:.3f}")
    print(f"  视觉: {'✓' if frame.vision else '✗'}")
    print(f"  听觉: {'✓' if frame.audio else '✗'}")
    print(f"  触觉: {'✓' if frame.tactile else '✗'}")
    print(f"  力觉: {'✓' if frame.force else '✗'}")
    print(f"  IMU: {'✓' if frame.imu else '✗'}")
    
    manager.close_all()
    
    return frame


def demo_multimodal_fusion(frame):
    """演示: 跨模态融合"""
    print("\n" + "="*60)
    print("🧠 跨模态融合网络")
    print("="*60)
    
    config = FusionConfig(
        strategy=FusionStrategy.HYBRID,
        hidden_dim=256,
        num_heads=4,
        num_layers=2
    )
    fusion = CrossModalFusion(config)
    
    print(f"  融合策略: {config.strategy.value}")
    print(f"  隐藏维度: {config.hidden_dim}")
    print(f"  注意力头数: {config.num_heads}")
    print(f"  网络层数: {config.num_layers}")
    
    # 融合需要张量输入 (B x C x H x W 等格式)
    # 实际应用中需先通过 sensors/encoders.py 将传感器数据编码为张量
    print(f"  输入要求: vision(B×3×224×224), audio(B×T×F), tactile(B×N), force(B×6), imu(B×9)")
    print(f"  查看 examples/complete_system_demo.py 了解完整融合流程")
    
    return None


def demo_agv_control():
    """演示: AGV运动控制"""
    print("\n" + "="*60)
    print("🚗 AGV运动控制器")
    print("="*60)
    
    # 创建 M 级 AGV
    spec = AGVSpec.from_grade(AGVGrade.M)
    controller = AGVMotionController(spec)
    
    # 运动学正解: 轮速 -> AGV速度
    joint_velocities = np.array([0.5, 0.5])  # 左右轮速 rad/s
    twist = controller.forward_kinematics(joint_velocities)
    print(f"  运动学正解: vx={twist.vx:.3f} m/s, vy={twist.vy:.3f} m/s, ω={twist.omega:.3f} rad/s")
    print(f"  AGV规格: 最大线速度={spec.max_linear_speed} m/s, 最大角速度={spec.max_angular_speed} rad/s")
    print(f"  驱动类型: {spec.drive_type.name}, 控制频率: {spec.control_frequency} Hz")
    
    # 运动学逆解: AGV速度 -> 轮速
    desired_twist = AGVTwist(vx=0.5, vy=0.0, omega=0.0)
    joint_cmds = controller.inverse_kinematics(desired_twist)
    print(f"  运动学逆解: 左轮={joint_cmds[0]:.3f}, 右轮={joint_cmds[1]:.3f} rad/s")
    
    return controller


def demo_safety_controller():
    """演示: 五级安全控制器"""
    print("\n" + "="*60)
    print("🛡️ 五级安全控制器")
    print("="*60)
    
    config = SafetyConfig(
        joint_limits_lower=np.array([-3.14]*6),
        joint_limits_upper=np.array([3.14]*6),
        velocity_limits=np.array([2.0]*6),
        acceleration_limits=np.array([5.0]*6),
        torque_limits=np.array([10.0]*6),
        safety_level=SafetyLevel.M
    )
    safety = SafetyController(config)
    
    print(f"  当前安全等级: {safety.safety_level.name} ({safety.safety_level.value})")
    print(f"  关节限位: ±{config.joint_limits_upper[0]:.2f} rad")
    print(f"  速度限制: {config.velocity_limits[0]:.1f} rad/s")
    print(f"  力矩限制: {config.torque_limits[0]:.1f} N·m")
    print(f"  碰撞阈值: {config.collision_threshold} N")
    
    # 执行安全检查
    snapshot = JointStateSnapshot(
        positions=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        velocities=np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]),
        torques=np.array([0.5, 0.6, 0.7, 0.5, 0.6, 0.7]),
        timestamp=time.time()
    )
    result = safety.check(snapshot)
    print(f"  安全检查: {'通过 ✓' if result.safe else '警告 ✗'}")
    
    return safety


def demo_simulation():
    """演示: Gymnasium仿真环境"""
    print("\n" + "="*60)
    print("🎮 Gymnasium仿真环境")
    print("="*60)
    
    config = GymEnvConfig(grade="M")
    env = SuperModelGymEnv(config)
    
    # 重置环境
    obs, info = env.reset()
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 随机动作
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"  执行动作完成")
    print(f"  奖励: {reward:.4f}, 终止: {done}, 截断: {truncated}")
    
    env.close()
    
    return env


def main():
    """主函数: 运行所有演示"""
    print("\n" + "#"*60)
    print("#  SuperModel 超模态机器人具身智能大脑 - 快速入门")
    print("#"*60)
    
    try:
        # 1. 传感器采集
        demo_sensor_collection()
        
        # 2. 统一传感器管理
        sensor_data = demo_sensor_manager()
        
        # 3. 跨模态融合
        if sensor_data:
            demo_multimodal_fusion(sensor_data)
        
        # 4. AGV运动控制
        demo_agv_control()
        
        # 5. 五级安全控制
        demo_safety_controller()
        
        # 6. 仿真环境
        demo_simulation()
        
        print("\n" + "#"*60)
        print("#  ✅ 演示完成! 查看 examples/complete_system_demo.py 了解更多")
        print("#"*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
