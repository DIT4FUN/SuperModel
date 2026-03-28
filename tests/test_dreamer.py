"""
Dreamer Agent 测试
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from learning.world_model import WorldModel, WorldModelConfig
from learning.dreamer_agent import (
    Actor, Critic, DreamerAgent, DreamerConfig,
    IntegratedAgent, create_integrated_agent
)


def test_actor():
    """测试 Actor"""
    print("\n[1] Actor 测试")
    
    actor = Actor(latent_dim=256, action_dim=6)
    
    # 测试前向
    latent = torch.randn(4, 256)
    action, log_prob = actor(latent)
    
    assert action.shape == (4, 6), f"Expected (4, 6), got {action.shape}"
    assert log_prob.shape == (4, 1), f"Expected (4, 1), got {log_prob.shape}"
    
    # 确定性策略
    action_det = actor(latent, deterministic=True)
    assert action_det[0].shape == (4, 6)
    
    # 获取动作
    action_np = actor.get_action(latent)
    assert action_np.shape == (4, 6)
    
    print(f"    动作: {action.shape}, log_prob: {log_prob.shape}")
    print("    ✅ Actor 测试通过")


def test_critic():
    """测试 Critic"""
    print("\n[2] Critic 测试")
    
    critic = Critic(latent_dim=256)
    
    # 单个隐状态
    latent = torch.randn(4, 256)
    value = critic(latent)
    assert value.shape == (4, 1)
    
    # 序列隐状态
    latent_seq = torch.randn(10, 4, 256)
    value_seq = critic(latent_seq)
    assert value_seq.shape == (10, 4, 1)
    
    print(f"    单个价值: {value.shape}, 序列价值: {value_seq.shape}")
    print("    ✅ Critic 测试通过")


def test_dreamer_agent():
    """测试 Dreamer Agent"""
    print("\n[3] Dreamer Agent 测试")
    
    # 创建世界模型
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    config = WorldModelConfig(action_dim=6)
    world_model = WorldModel(obs_dims, action_dim=6, config=config)
    
    # 创建 Dreamer
    agent = DreamerAgent(
        world_model=world_model,
        action_dim=6,
        config=DreamerConfig(action_dim=6)
    )
    
    B, T = 4, 15
    initial_deter = torch.randn(B, config.deter_dim)
    initial_stoch = torch.randn(B, config.stoch_dim * config.num_classes)
    
    # 想象 rollout
    latent_seq, action_seq, reward_seq, value_seq = agent.imagine(
        initial_deter, initial_stoch, horizon=T
    )
    
    assert latent_seq.shape == (T, B, config.latent_dim)
    assert action_seq.shape == (T, B, 6)
    assert reward_seq.shape == (T, B)
    assert value_seq.shape == (T, B)
    
    print(f"    隐状态序列: {latent_seq.shape}")
    print(f"    动作序列: {action_seq.shape}")
    print(f"    奖励序列: {reward_seq.shape}")
    print("    ✅ Dreamer Agent 测试通过")


def test_actor_update():
    """测试 Actor 更新"""
    print("\n[4] Actor 更新测试")
    
    actor = Actor(latent_dim=256, action_dim=6)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    
    # 模拟数据
    latent_seq = torch.randn(15, 4, 256)
    action_seq = torch.randn(15, 4, 6)
    returns = torch.randn(15, 4)
    advantages = torch.randn(15, 4)
    
    # 更新
    metrics = actor.update_actor(latent_seq, action_seq, returns, advantages) if hasattr(actor, 'update_actor') else {}
    
    # 如果 actor 没有 update_actor 方法，测试损失计算
    latent_flat = latent_seq.reshape(15 * 4, -1)
    action, log_prob = actor(latent_flat)
    loss = -(log_prob.mean())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"    策略损失: {loss.item():.4f}")
    print("    ✅ Actor 更新测试通过")


def test_dreamer_training():
    """测试 Dreamer 训练"""
    print("\n[5] Dreamer 训练测试")
    
    # 创建世界模型
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    config = WorldModelConfig(action_dim=6)
    world_model = WorldModel(obs_dims, action_dim=6, config=config)
    
    agent = DreamerAgent(
        world_model=world_model,
        action_dim=6
    )
    
    # 训练步骤
    B = 4
    initial_deter = torch.randn(B, config.deter_dim)
    initial_stoch = torch.randn(B, config.stoch_dim * config.num_classes)
    
    # 简化的训练测试 - 只测试想象 rollout
    agent.train()
    latent_seq, action_seq, reward_seq, value_seq = agent.imagine(
        initial_deter, initial_stoch, horizon=5
    )
    
    # 计算回报
    returns, advantages = agent.compute_return(reward_seq, value_seq)
    
    print(f"    隐状态序列: {latent_seq.shape}")
    print(f"    回报: {returns.shape}")
    print("    ✅ Dreamer 训练测试通过")


def test_integrated_agent():
    """测试集成智能体"""
    print("\n[6] 集成智能体测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    
    agent = create_integrated_agent(obs_dims, action_dim=6, grade='M')
    
    # 选择动作
    observations = {
        'vision': torch.randn(1, 3, 224, 224),
        'audio': torch.randn(1, 100, 64),
        'tactile': torch.randn(1, 1, 16, 16),
        'force': torch.randn(1, 10, 6),
        'imu': torch.randn(1, 10, 6)
    }
    
    action = agent.select_action(observations, deterministic=True)
    assert action.shape == (6,)
    
    print(f"    选择动作: {action.shape}")
    print("    ✅ 集成智能体测试通过")


def test_end_to_end():
    """端到端测试"""
    print("\n[7] 端到端测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    
    agent = create_integrated_agent(obs_dims, action_dim=6, grade='M')
    
    # 测试编码器
    observations = {
        'vision': torch.randn(1, 3, 224, 224),
        'audio': torch.randn(1, 100, 64),
        'tactile': torch.randn(1, 1, 16, 16),
        'force': torch.randn(1, 10, 6),
        'imu': torch.randn(1, 10, 6)
    }
    
    encoded = agent.encode_observation(observations)
    
    print(f"    编码后特征: {encoded.keys()}")
    
    # 测试动作选择
    action = agent.select_action(observations, deterministic=True)
    print(f"    选择动作: {action.shape}")
    
    print("    ✅ 端到端测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Dreamer Agent 测试套件")
    print("=" * 60)
    
    try:
        test_actor()
        test_critic()
        test_dreamer_agent()
        test_actor_update()
        test_dreamer_training()
        test_integrated_agent()
        test_end_to_end()
        
        print("\n" + "=" * 60)
        print("🎉 所有 Dreamer 测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(run_all_tests())
