"""
SuperModel 完整流水线演示
=========================

演示完整的超模态机器人具身智能大脑流水线:
传感器采集 → 编码 → 跨模态融合 → 世界模型 → 控制器 → 仿真反馈

这个脚本展示:
1. 多传感器同步采集
2. 传感器编码
3. 跨模态融合
4. 世界模型想象 rollout
5. 控制指令生成
6. 仿真环境反馈

运行方式:
    python scripts/demo_complete_pipeline.py
"""

import numpy as np
import torch
import time
import sys

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray, PressureProcessor
from sensors.force import ForceTorqueSensor, WrenchProcessor, ForceSensorType
from sensors.imu import IMUSensor, PoseEstimator, IMUSensorType
from sensors.encoders import create_sensor_encoder
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from learning.world_model import WorldModel, create_world_model_agent
from control.motion import MotionController, JointState, TwistCommand
from control.impedance import ImpedanceController, ImpedanceParams
from control.planner import HierarchicalPlanner, Task, TaskSpec, WorldState, Action
from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


def demo_sensor_collection():
    """演示: 多传感器数据采集"""
    print("\n" + "="*60)
    print("📡 演示1: 多传感器同步采集")
    print("="*60)
    
    # 初始化传感器
    cameras = {}
    cameras['vision'] = BinocularCamera(resolution=(640, 480), fps=30)
    cameras['audio'] = BinauralMic(sample_rate=16000, chunk_size=512)
    cameras['tactile'] = TactileArray(array_size=(16, 16))
    cameras['force'] = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
    cameras['imu'] = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
    
    # 打开所有传感器
    for name, sensor in cameras.items():
        sensor.open()
        print(f"  ✅ {name} 传感器已打开")
    
    # 同步采集
    print("\n  📸 采集数据...")
    data = {}
    data['vision'] = cameras['vision'].capture()
    data['audio'] = cameras['audio'].capture()
    data['tactile'] = cameras['tactile'].capture()
    data['force'] = cameras['force'].capture()
    data['imu'] = cameras['imu'].capture()
    
    # 打印统计信息
    print(f"  ✓ 双目视觉: {data['vision'].left_image.shape}")
    print(f"  ✓ 音频: {len(data['audio'].left_channel)} samples @ {data['audio'].sample_rate}Hz")
    print(f"  ✓ 触觉: {data['tactile'].pressure_map.shape}, 峰值压力={data['tactile'].pressure_map.max():.3f}")
    print(f"  ✓ 力觉: F={data['force'].force}, |F|={data['force'].magnitude:.2f}N")
    print(f"  ✓ IMU: accel={data['imu'].accel}, |a|={data['imu'].accel_magnitude:.2f}m/s²")
    
    # 触觉接触检测
    contacts = cameras['tactile'].detect_contacts(data['tactile'])
    print(f"  ✓ 接触检测: {len(contacts)} 个接触区域")
    
    # 关闭传感器
    for sensor in cameras.values():
        sensor.close()
    
    return data


def demo_encoder_and_fusion():
    """演示: 编码器 + 跨模态融合"""
    print("\n" + "="*60)
    print("🧠 演示2: 编码器 + 跨模态融合")
    print("="*60)
    
    # 创建编码器 (M级配置)
    encoder = create_sensor_encoder({
        'vision': (3, 224, 224),
        'audio': (100, 64),
        'tactile': (1, 16, 16),
        'force': (10, 6),
        'imu': (10, 6),
    }, grade='M')
    print("  ✅ 编码器已创建 (M级配置)")
    
    # 创建融合网络
    fusion_config = FusionConfig(
        vision_dim=512, audio_dim=128, tactile_dim=64,
        force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
    )
    fusion = CrossModalFusion(fusion_config)
    print("  ✅ 跨模态融合网络已创建")
    
    # 模拟多模态输入 - 使用编码后的特征向量
    B = 2  # batch size
    # 融合网络期望预编码的特征向量
    mmi = MultimodalInput(
        vision=torch.randn(B, 512),   # 编码后视觉特征
        audio=torch.randn(B, 128),    # 编码后音频特征
        tactile=torch.randn(B, 64),   # 编码后触觉特征
        force=torch.randn(B, 32),     # 编码后力觉特征
        imu=torch.randn(B, 64),        # 编码后IMU特征
    )
    print(f"  ✅ 多模态输入已创建 (batch_size={B})")
    print(f"     注: 融合网络期望预编码特征向量")
    
    # 融合前向传播
    with torch.no_grad():
        fused = fusion(mmi)
    print(f"  ✅ 融合输出: shape={fused.shape}, dtype={fused.dtype}")
    
    # 测试不同模态组合
    print("\n  📊 模态组合测试:")
    test_cases = [
        ("仅视觉", MultimodalInput(vision=torch.randn(1, 512))),
        ("视觉+听觉", MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128)
        )),
        ("触觉+力觉+IMU", MultimodalInput(
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64)
        )),
    ]
    
    with torch.no_grad():
        for name, mmi_test in test_cases:
            out = fusion(mmi_test)
            print(f"    {name}: output shape = {out.shape}")
    
    return fusion, encoder


def demo_world_model():
    """演示: 世界模型 + Dreamer Agent"""
    print("\n" + "="*60)
    print("🌍 演示3: 世界模型 + Dreamer Agent")
    print("="*60)
    
    # 创建世界模型智能体 (M级)
    # 注意: obs_dims 是编码后的特征维度，不是原始传感器数据维度
    obs_dims = {
        'vision': 512,     # 编码后视觉特征维度
        'audio': 128,      # 编码后音频特征维度
        'tactile': 64,     # 编码后触觉特征维度
        'force': 32,       # 编码后力觉特征维度
        'imu': 64,         # 编码后IMU特征维度
    }
    action_dim = 6
    
    agent = create_world_model_agent('M', obs_dims, action_dim)
    print("  ✅ Dreamer Agent 已创建 (M级配置)")
    print("     注: 世界模型期望编码后的特征向量，不是原始传感器数据")
    
    # 世界模型配置信息
    from learning.world_model import get_world_model_spec
    wm_spec = get_world_model_spec('M')
    print(f"\n  📊 世界模型配置 (M级):")
    print(f"     隐状态维度: {wm_spec.latent_dim}")
    print(f"     隐藏维度: {wm_spec.hidden_dim}")
    print(f"     想象步数: {wm_spec.imagination_horizon}")
    print(f"     RNN隐藏: {wm_spec.rnn_hidden_dim}")
    
    # 模拟经验数据 - 使用编码后的特征 (batch, time_steps, features)
    print("\n  📝 模拟经验存储...")
    batch_size = 32
    time_steps = 10
    
    for i in range(5):
        # 编码后的观测特征 (batch, time_steps, features)
        obs = {
            'vision': np.random.randn(batch_size, time_steps, 512).astype(np.float32),
            'audio': np.random.randn(batch_size, time_steps, 128).astype(np.float32),
            'tactile': np.random.randn(batch_size, time_steps, 64).astype(np.float32),
            'force': np.random.randn(batch_size, time_steps, 32).astype(np.float32),
            'imu': np.random.randn(batch_size, time_steps, 64).astype(np.float32),
        }
        action = np.random.randn(batch_size, action_dim).astype(np.float32)
        reward = np.random.randn(batch_size, 1).astype(np.float32)
        done = np.zeros((batch_size, 1), dtype=np.float32)
        
        agent.store_transition(obs, action, reward, obs, done)
    
    print(f"  ✅ 已存储 {5*batch_size} 条经验 (每条 {time_steps} 步)")
    
    # 训练步骤
    print("\n  🏋️ 执行训练步骤...")
    losses = agent.train_step(batch_size=16)
    print(f"    世界模型损失: {losses.get('world_model_loss', 0):.4f}")
    print(f"    Actor 损失: {losses.get('actor_loss', 0):.4f}")
    print(f"    Critic 损失: {losses.get('critic_loss', 0):.4f}")
    
    # 想象 rollout (使用初始隐状态)
    print("\n  🔮 想象 rollout...")
    horizon = 15
    try:
        latent_seq, action_seq, reward_seq, value_seq = agent.imagine(
            agent.initial_deter, agent.initial_stoch, horizon=horizon
        )
        print(f"  ✅ 想象轨迹: {horizon} 步")
        print(f"     隐状态: {len(latent_seq)} 个时间步")
        if len(reward_seq) > 0:
            total_reward = sum(float(r[0]) if hasattr(r, '__getitem__') else float(r) for r in reward_seq)
            print(f"     累积奖励: {total_reward:.2f}")
    except Exception as e:
        print(f"  ⚠️ 想象 rollout 暂时不可用 (需要更多训练数据)")
    
    return agent


def demo_control_loop():
    """演示: 完整控制回路"""
    print("\n" + "="*60)
    print("🎯 演示4: 完整控制回路")
    print("="*60)
    
    # 创建仿真器
    sim_config = SimConfig(dt=0.01, num_joints=6)
    sim = RobotSimulator(sim_config)
    sensor_sim = SensorSimulator(sim, sim_config)
    print("  ✅ 仿真器已创建")
    
    # 创建控制器
    controller = MotionController(num_joints=6, control_rate=100.0)
    controller.kp = np.ones(6) * 2.0
    controller.ki = np.zeros(6)
    controller.kd = np.ones(6) * 0.5
    print("  ✅ PID 控制器已创建")
    
    # 创建阻抗控制器
    impedance_ctrl = ImpedanceController(ImpedanceParams.default_6d())
    print("  ✅ 阻抗控制器已创建")
    
    # 目标位置
    target_position = np.array([0.5, 0.3, -0.2, 0.0, 0.0, 0.0])
    print(f"\n  🎯 目标位置: {target_position}")
    
    # 控制循环
    print("\n  🔄 开始控制循环...")
    n_steps = 100
    
    for step in range(n_steps):
        # 获取当前状态
        current_pos = sim.joint_positions.copy()
        current_vel = sensor_sim.get_noisy_joint_velocities()
        
        # 计算控制力矩
        torque = controller.compute_joint_torque(target_position)
        
        # 执行一步仿真
        state = sim.step(torque)
        
        if step % 20 == 0:
            error = np.linalg.norm(target_position - current_pos)
            print(f"    Step {step:3d}: |error| = {error:.4f}, "
                  f"torque = [{torque[0]:.2f}, {torque[1]:.2f}, ...]")
    
    # 最终误差
    final_error = np.linalg.norm(target_position - sim.joint_positions)
    print(f"\n  ✅ 控制完成: 最终误差 = {final_error:.4f}")
    
    # 测试笛卡尔速度控制
    print("\n  🔄 测试笛卡尔速度控制...")
    twist = TwistCommand(
        linear=np.array([0.05, 0.0, 0.0]),
        angular=np.zeros(3)
    )
    jacobian = sim.get_jacobian()
    joint_vel = controller.compute_cartesian_velocity(twist, jacobian)
    print(f"  ✅ 笛卡尔速度 → 关节速度: {joint_vel[:3]}")
    
    return sim, controller


def demo_task_planning():
    """演示: 层次化任务规划"""
    print("\n" + "="*60)
    print("📋 演示5: 层次化任务规划")
    print("="*60)
    
    # 创建规划器
    planner = HierarchicalPlanner()
    print("  ✅ 层次化任务网络规划器已创建")
    
    # 注册动作
    def pickup_action(state, params):
        obj = params.get('object', 'unknown')
        state.objects[obj] = state.objects.get(obj, {})
        state.objects[obj]['grasped'] = True
        state.robot_state['gripper_closed'] = True
    
    planner.action_library['grasp'] = Action(
        name='grasp',
        precondition=lambda s: True,
        effect=pickup_action,
        cost=1.0
    )
    planner.action_library['move_to'] = Action(
        name='move_to',
        precondition=lambda s: True,
        effect=lambda s, p: s.robot_state.update({'position': p.get('target', [0,0,0])}),
        cost=1.0
    )
    
    # 创建任务
    task_spec = TaskSpec(
        name='pickup',
        goal_state={'held': True},
        max_depth=3
    )
    
    print(f"\n  📋 任务: {task_spec.name}")
    print(f"  🎯 目标状态: {task_spec.goal_state}")
    
    # 层次化规划
    tasks = planner.plan_hierarchical(task_spec)
    print(f"\n  📝 分解为 {len(tasks)} 个子任务:")
    for i, task in enumerate(tasks):
        print(f"    {i+1}. {task.name} (id={task.id})")
    
    # 世界状态更新
    world_state = WorldState()
    world_state.objects['box'] = {'position': [0.5, 0.0, 0.0]}
    
    # 贪心规划
    planner.set_world_state(world_state)
    plan = planner.plan(TaskSpec(name='test', goal_state={'grasped': True}))
    print(f"\n  📝 贪心规划序列: {plan}")
    
    return planner


def demo_agv_grade_specs():
    """演示: AGV五级规格"""
    print("\n" + "="*60)
    print("📊 演示6: AGV五级规格表")
    print("="*60)
    
    from sensors.vision import get_stereo_spec
    from sensors.audio import get_audio_spec
    from sensors.tactile import get_tactile_spec
    from sensors.force import get_force_spec
    from sensors.imu import get_imu_spec
    from fusion.cross_modal_fusion import get_fusion_spec
    from learning.world_model import get_world_model_spec
    from control.trajectory import get_trajectory_spec
    from control.ros2_interface import get_ros2_spec
    from control.safety_controller import get_safety_spec, SafetyLevel
    from control.mpc import get_mpc_spec
    
    grades = ['S', 'M', 'L', 'XL', 'XXL']
    
    print(f"\n{'等级':<6} {'视觉基线':<12} {'触觉阵列':<12} {'力觉轴数':<10} {'IMU采样':<10} {'融合隐层':<10}")
    print("-" * 60)
    
    for grade in grades:
        vision_spec = get_stereo_spec(grade)
        tactile_spec = get_tactile_spec(grade)
        force_spec = get_force_spec(grade)
        imu_spec = get_imu_spec(grade)
        fusion_spec = get_fusion_spec(grade)
        
        print(f"{grade:<6} "
              f"{vision_spec['baseline_mm']:<12} "
              f"{str(tactile_spec['array']):<12} "
              f"{force_spec['axes']:<10} "
              f"{imu_spec['sample_hz']:<10} "
              f"{fusion_spec['hidden_dim']:<10}")
    
    print("\n  各等级特性:")
    for grade in grades:
        fusion_spec = get_fusion_spec(grade)
        wm_spec = get_world_model_spec(grade)
        traj_spec = get_trajectory_spec(grade)
        ros2_spec = get_ros2_spec(grade)
        print(f"\n  [{grade}级]")
        print(f"    融合: {fusion_spec['strategy']}, 推理延迟 < {fusion_spec['latency_ms']}ms")
        print(f"    世界模型: 隐状态={wm_spec.hidden_dim}, 想象步数={wm_spec.imagination_horizon}")
        print(f"    轨迹规划: {traj_spec['algorithm']} 算法")
        print(f"    ROS2: QoS深度={ros2_spec['qos_depth']}, 实时性={ros2_spec['realtime']}")


def main():
    """主函数: 运行所有演示"""
    print("\n" + "="*60)
    print("🚀 SuperModel 超模态机器人具身智能大脑 - 完整流水线演示")
    print("="*60)
    
    total_start = time.time()
    
    # 演示1: 传感器采集
    demo_sensor_collection()
    
    # 演示2: 编码器 + 融合
    demo_encoder_and_fusion()
    
    # 演示3: 世界模型
    demo_world_model()
    
    # 演示4: 控制回路
    demo_control_loop()
    
    # 演示5: 任务规划
    demo_task_planning()
    
    # 演示6: AGV规格
    demo_agv_grade_specs()
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print(f"✅ 演示完成! 总耗时: {total_time:.2f}秒")
    print("="*60)
    print("\n  📚 更多信息:")
    print("     - 文档: docs/")
    print("     - 测试: tests/")
    print("     - 配置: configs/")
    print()


if __name__ == '__main__':
    main()
