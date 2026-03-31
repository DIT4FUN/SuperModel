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

from control.agv import AGVMotionController, AGVGrade, AGVSpec, AGVPose
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel
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
    print(f"  🧭 IMU姿态: rpy=[{imu_frame.roll:.2f}, {imu_frame.pitch:.2f}, {imu_frame.yaw:.2f}]°")
    imu.close()
    
    return stereo_frame, audio_frame, tactile_frame, wrench, imu_frame


def demo_sensor_manager():
    """演示: 统一传感器管理器"""
    print("\n" + "="*60)
    print("🔧 统一传感器管理器")
    print("="*60)
    
    config = SensorManagerConfig(grade="M")
    manager = SensorManager(config)
    
    manager.start_all()
    time.sleep(0.1)  # 等待传感器初始化
    
    # 同步采集所有传感器
    data = manager.capture_all()
    print(f"  已采集模态: {list(data.keys())}")
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"    {key}: shape={value.shape}")
        else:
            print(f"    {key}: {type(value).__name__}")
    
    manager.stop_all()
    manager.close_all()
    
    return data


def demo_multimodal_fusion(data):
    """演示: 跨模态融合"""
    print("\n" + "="*60)
    print("🧠 跨模态融合网络")
    print("="*60)
    
    config = FusionConfig(
        strategy=FusionStrategy.CROSS_ATTENTION,
        hidden_dim=256,
        num_heads=4,
        num_layers=2
    )
    fusion = CrossModalFusion(config)
    
    # 构建多模态输入
    inputs = MultimodalInput(
        vision=data.get('vision'),
        audio=data.get('audio'),
        tactile=data.get('tactile'),
        force=data.get('force'),
        imu=data.get('imu')
    )
    
    # 执行融合
    fused_output = fusion(inputs)
    print(f"  融合特征维度: {fused_output.fused_features.shape}")
    print(f"  注意力权重数: {len(fused_output.attention_weights)}")
    
    return fused_output


def demo_agv_control():
    """演示: AGV运动控制"""
    print("\n" + "="*60)
    print("🚗 AGV运动控制器")
    print("="*60)
    
    # 创建 M 级 AGV
    spec = AGVSpec.for_grade(AGVGrade.M)
    controller = AGVMotionController(spec)
    
    # 设置目标位姿
    target = AGVPose(x=1.0, y=0.5, theta=0.0)
    controller.set_target_pose(target)
    
    # 运动学正解
    joint_velocities = np.array([0.5, 0.5])  # 左右轮速
    twist = controller.forward_kinematics(joint_velocities)
    print(f"  运动学正解: v={twist.linear:.3f} m/s, ω={twist.angular:.3f} rad/s")
    
    # 运动学逆解
    desired_twist = controller.TwistCommand(linear=0.5, angular=0.0)
    joint_cmds = controller.inverse_kinematics(desired_twist)
    print(f"  运动学逆解: 左轮={joint_cmds[0]:.3f}, 右轮={joint_cmds[1]:.3f} rad/s")
    
    # 更新位姿
    pose = controller.update_pose(twist, dt=0.1)
    print(f"  更新后位姿: x={pose.x:.3f}, y={pose.y:.3f}, θ={pose.theta:.3f}")
    
    return controller


def demo_safety_controller():
    """演示: 五级安全控制器"""
    print("\n" + "="*60)
    print("🛡️ 五级安全控制器")
    print("="*60)
    
    config = SafetyConfig(grade="M")
    safety = SafetyController(config)
    
    # 模拟关节状态快照
    from control.safety_controller import JointStateSnapshot
    snapshot = JointStateSnapshot(
        joint_positions=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        joint_velocities=np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]),
        joint_torques=np.array([0.5, 0.6, 0.7, 0.5, 0.6, 0.7]),
        timestamp=time.time()
    )
    
    # 检查安全等级
    level = safety.check_safety_level(snapshot)
    print(f"  当前安全等级: {level.name} ({level.value})")
    print(f"  安全阈值: position={config.position_limit:.3f} rad")
    print(f"            velocity={config.velocity_limit:.3f} rad/s")
    print(f"            torque={config.torque_limit:.3f} N·m")
    
    return safety


def demo_simulation():
    """演示: Gymnasium仿真环境"""
    print("\n" + "="*60)
    print("🎮 Gymnasium仿真环境")
    print("="*60)
    
    config = GymEnvConfig(grade="M")
    env = SuperModelGymEnv(config)
    
    # 重置环境
    obs = env.reset()
    print(f"  环境观测维度: {obs['state'].shape}")
    
    # 随机动作
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"  执行动作: {action[:2]}...")
    print(f"  奖励: {reward:.4f}, 完成: {done}")
    
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
