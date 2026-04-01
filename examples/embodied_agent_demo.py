"""
SuperModel 具身智能体演示
=========================

展示具身智能体 (Embodied Agent) 的完整闭环:
  传感器 → 感知 → 融合 → 世界模型 → 决策 → 控制 → 执行

本脚本演示如何将传感器模块、跨模态融合、世界模型和控制模块
组合成一个完整的具身智能体系统。

版本: v1.0.0
"""

import numpy as np
import time
import sys

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, get_stereo_spec
from sensors.audio import BinauralMic, get_audio_spec
from sensors.tactile import TactileArray, VirtualTactileSensor, get_tactile_spec
from sensors.force import ForceTorqueSensor, VirtualForceSensor, Wrench, get_force_spec
from sensors.imu import IMUSensor, VirtualIMUSensor, PoseEstimator, get_imu_spec
from sensors.manager import SensorManager
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput, create_multimodal_input
from learning.world_model import create_world_model_agent, get_world_model_spec
from control.agv import AGVMotionController, AGVSpec, get_agv_spec
from control.safety_controller import SafetyController, SafetyConfig
from simulation.environment import RobotSimulator


# =============================================================================
# 辅助函数
# =============================================================================

def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def simulate_sensor_step(sim: RobotSimulator, dt: float):
    """执行一步传感器数据采集 (虚拟传感器)"""
    sim.step(dt)
    return {
        'vision': np.random.randn(2, 256).astype(np.float32),
        'audio': np.random.randn(100, 128).astype(np.float32),
        'tactile': np.random.randn(64).astype(np.float32),
        'force': np.random.randn(6).astype(np.float32),
        'imu': np.random.randn(9).astype(np.float32),
    }


# =============================================================================
# 第1步: 初始化具身智能体
# =============================================================================
print_section("第1步: 初始化具身智能体 (Embodied Agent)")

# 选择 AGV 等级
GRADE = 'M'
print(f"AGV 等级: {GRADE}")

# 初始化传感器管理器
sensor_manager = SensorManager()
print(f"✓ 传感器管理器初始化完成")

# 获取规格
vision_spec = get_stereo_spec(GRADE)
audio_spec = get_audio_spec(GRADE)
tactile_spec = get_tactile_spec(GRADE)
force_spec = get_force_spec(GRADE)
imu_spec = get_imu_spec(GRADE)
agv_spec = get_agv_spec(GRADE)

print(f"✓ 视觉规格: {vision_spec['resolution']} @ {vision_spec['fps']} fps")
print(f"✓ 听觉规格: {audio_spec['sample_rate']} Hz, {audio_spec['channels']} 通道")
print(f"✓ 触觉规格: {tactile_spec['array_size']} 阵列")
print(f"✓ 力觉规格: {force_spec['force_range']} N / {force_spec['torque_range']} N·m")
print(f"✓ IMU 规格: {imu_spec['sampling_hz']} Hz")

# =============================================================================
# 第2步: 初始化融合网络
# =============================================================================
print_section("第2步: 初始化跨模态融合网络")

fusion_config = FusionConfig(
    strategy='middle',
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    feature_dim=128,
    inference_latency_ms=20,
)

fusion_net = CrossModalFusion(
    modality_feature_dims={
        'vision': 256,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 32,
    },
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
)

print(f"✓ 融合网络初始化: hidden_dim={fusion_config.hidden_dim}, heads={fusion_config.num_heads}")
print(f"✓ 融合策略: {fusion_config.strategy}")


# =============================================================================
# 第3步: 初始化世界模型 (Dreamer-style RSSM)
# =============================================================================
print_section("第3步: 初始化世界模型 (Dreamer RSSM)")

obs_dims = {
    'vision': 256,
    'audio': 128,
    'tactile': 64,
    'force': 32,
    'imu': 32,
}
action_dim = 3  # x, y, theta 速度控制

agent = create_world_model_agent(GRADE, obs_dims, action_dim)
print(f"✓ 世界模型智能体创建: {get_world_model_spec(GRADE)['type']}")
print(f"✓ 隐状态维度: {agent.latent_dim}")
print(f"✓ 想象训练步长: {agent.imagination_horizon}")


# =============================================================================
# 第4步: 初始化运动控制器
# =============================================================================
print_section("第4步: 初始化运动控制器")

# 差速驱动 AGV 控制器
agv_controller = AGVMotionController(agv_spec)
safety_config = SafetyConfig(
    safety_level='M',
    max_linear_speed=2.0,
    max_angular_speed=2.0,
    max_linear_accel=1.0,
    max_force=50.0,
)
safety_controller = SafetyController(agv_spec, safety_config)

print(f"✓ AGV 运动控制器初始化")
print(f"✓ 安全控制器: {safety_config.safety_level} 级")


# =============================================================================
# 第5步: 初始化仿真环境
# =============================================================================
print_section("第5步: 初始化仿真环境")

sim = RobotSimulator(
    dt=0.01,
    gravity=9.81,
    ground_friction=0.5,
)
sim.reset()

print(f"✓ 仿真环境: dt={sim.dt}s, 重力={sim.gravity} m/s²")


# =============================================================================
# 第6步: 具身智能闭环主循环
# =============================================================================
print_section("第6步: 具身智能闭环主循环")

NUM_STEPS = 50
PRINT_INTERVAL = 10

print(f"\n运行 {NUM_STEPS} 步具身智能闭环...")

total_loop_time = 0.0

for step in range(NUM_STEPS):
    step_start = time.perf_counter()

    # --- 感知阶段 ---
    sensor_data = simulate_sensor_step(sim, dt=0.01)

    # --- 多模态融合 ---
    multimodal_input = create_multimodal_input(
        vision=sensor_data['vision'],
        audio=sensor_data['audio'],
        tactile=sensor_data['tactile'],
        force=sensor_data['force'],
        imu=sensor_data['imu'],
    )

    fused_features = fusion_net(multimodal_input)

    # --- 世界模型决策 ---
    action = agent.select_action(fused_features.features, deterministic=(step < 5))

    # --- 安全检查 ---
    safe_action = safety_controller.check_and_sanitize(action)

    # --- 运动控制 ---
    twist = agv_controller.twist_from_action(safe_action)
    wheel_cmds = agv_controller.inverse_kinematics(twist)

    # --- 执行 (仿真) ---
    sim.set_wheel_velocities(wheel_cmds)

    # --- 世界模型学习 ---
    next_obs = fused_features.features.detach() + np.random.randn(*fused_features.features.shape) * 0.01
    reward = 1.0 if step % 20 < 10 else -0.1  # 简单奖励信号

    agent.store_transition(
        obs=fused_features.features.detach().numpy()[0],
        action=action.detach().numpy()[0],
        reward=reward,
        next_obs=next_obs[0],
        done=False,
    )

    if step >= 10:
        losses = agent.train_step(batch_size=4)
        if step % 20 == 0 and losses:
            loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
            print(f"  [Step {step}] 世界模型损失: {loss_str}")

    # --- 更新状态 ---
    agent.update_state(fused_features.features)

    step_time = time.perf_counter() - step_start
    total_loop_time += step_time

    if step % PRINT_INTERVAL == 0:
        pose = agv_controller.pose
        print(f"  [Step {step:3d}] pose=({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.2f}) | "
              f"action=({action[0,0]:.2f}, {action[0,1]:.2f}, {action[0,2]:.2f}) | "
              f"loop_time={step_time*1000:.1f}ms")

avg_loop_time = (total_loop_time / NUM_STEPS) * 1000
print(f"\n✓ 具身智能闭环完成!")
print(f"  平均每步耗时: {avg_loop_time:.2f} ms")
print(f"  目标实时性: {'✓ 通过' if avg_loop_time < 50 else '✗ 未通过 (需优化)'}")


# =============================================================================
# 第7步: 结果汇总
# =============================================================================
print_section("第7步: 结果汇总")

print(f"""
SuperModel 具身智能体演示 — 结果汇总
=====================================
AGV 等级:         {GRADE}
运行步数:         {NUM_STEPS}
平均每步耗时:     {avg_loop_time:.2f} ms

模块状态:
  ✓ 传感器管理器     已初始化 ({sensor_manager.get_active_sensor_count()} 个活跃传感器)
  ✓ 跨模态融合网络   已初始化 (hidden_dim={fusion_config.hidden_dim})
  ✓ 世界模型         已初始化 (RSSM, latent_dim={agent.latent_dim})
  ✓ 运动控制器       已初始化 (AGV {GRADE} 级)
  ✓ 安全控制器       已启用 ({safety_config.safety_level} 级)
  ✓ 仿真环境         已初始化 (Gymnasium 兼容)

下一阶段:
  → 扩大训练步数，验证持续学习能力
  → 接入真实硬件 (RDK X5 Ultra)
  → 部署 ROS2 实时控制接口
""")
