"""
World Model 测试
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from learning.world_model import (
    WorldModel, WorldModelConfig, WorldModelAgent,
    ObservationEncoder, TransitionModel, RewardModel, ValueModel,
    ModelState, create_world_model_agent, get_world_model_spec,
    WORLD_MODEL_GRADES, ReplayBuffer
)


def test_observation_encoder():
    """测试观测编码器"""
    print("\n[1] 观测编码器测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    
    encoder = ObservationEncoder(
        obs_dims=obs_dims,
        hidden_dim=256,
        latent_dim=256
    )
    
    # 模拟观测
    observations = {
        'vision': torch.randn(4, 512),
        'audio': torch.randn(4, 128),
        'tactile': torch.randn(4, 64),
        'force': torch.randn(4, 32),
        'imu': torch.randn(4, 64)
    }
    
    encoded = encoder(observations)
    assert encoded.shape == (4, 256), f"Expected (4, 256), got {encoded.shape}"
    
    print(f"    输入: vision={observations['vision'].shape}, audio={observations['audio'].shape}")
    print(f"    输出: {encoded.shape}")
    print("    ✅ 观测编码器测试通过")


def test_transition_model():
    """测试 RSSM 转移模型"""
    print("\n[2] RSSM 转移模型测试")
    
    transition = TransitionModel(
        action_dim=6,
        deter_dim=256,
        stoch_dim=32,
        num_classes=32,
        hidden_dim=512
    )
    
    B = 4
    deter = torch.randn(B, 256)
    prev_action = torch.randn(B, 6)
    prev_stoch = torch.randn(B, 32 * 32)
    
    # 无观测 (先验)
    deter_out, logits, stoch = transition(deter, prev_action, None, prev_stoch)
    
    assert deter_out.shape == (B, 256), f"Expected (4, 256), got {deter_out.shape}"
    assert logits.shape == (B, 32 * 32), f"Expected (4, 1024), got {logits.shape}"
    assert stoch.shape == (B, 32 * 32), f"Expected (4, 1024), got {stoch.shape}"
    
    print(f"    deter: {deter.shape} -> {deter_out.shape}")
    print(f"    stoch: {stoch.shape}")
    print("    ✅ RSSM 转移模型测试通过")


def test_reward_value_models():
    """测试奖励和价值模型"""
    print("\n[3] 奖励和价值模型测试")
    
    reward_model = RewardModel(latent_dim=512, hidden_dim=256)
    value_model = ValueModel(latent_dim=512, hidden_dim=256)
    
    B = 4
    latent = torch.randn(B, 512)
    
    reward = reward_model(latent)
    value = value_model(latent)
    
    assert reward.shape == (B, 1), f"Expected (4, 1), got {reward.shape}"
    assert value.shape == (B, 1), f"Expected (4, 1), got {value.shape}"
    
    print(f"    隐状态: {latent.shape}")
    print(f"    奖励: {reward.shape}, value: {value.shape}")
    print("    ✅ 奖励和价值模型测试通过")


def test_world_model():
    """测试完整世界模型"""
    print("\n[4] 完整世界模型测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    action_dim = 6
    
    config = WorldModelConfig(action_dim=action_dim)
    model = WorldModel(obs_dims, action_dim, config)
    
    B, T = 2, 5
    
    # 模拟观测序列
    observations = {
        'vision': torch.randn(T, B, 512),
        'audio': torch.randn(T, B, 128),
        'tactile': torch.randn(T, B, 64),
        'force': torch.randn(T, B, 32),
        'imu': torch.randn(T, B, 64)
    }
    
    actions = torch.randn(T, B, action_dim)
    rewards = torch.randn(T, B)
    dones = torch.zeros(T, B)
    
    # 计算损失
    losses = model.compute_loss(observations, actions, rewards, dones)
    
    assert 'total' in losses, "Missing 'total' loss"
    assert 'reward' in losses, "Missing 'reward' loss"
    
    print(f"    损失: total={losses['total'].item():.4f}, reward={losses['reward'].item():.4f}")
    print("    ✅ 完整世界模型测试通过")


def test_world_model_agent():
    """测试世界模型智能体"""
    print("\n[5] 世界模型智能体测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    action_dim = 6
    
    agent = create_world_model_agent('M', obs_dims, action_dim)
    
    # 模拟选择动作
    obs = {
        'vision': np.random.randn(512),
        'audio': np.random.randn(128),
        'tactile': np.random.randn(64),
        'force': np.random.randn(32),
        'imu': np.random.randn(64)
    }
    
    action = agent.select_action(obs, deterministic=True)
    
    assert action.shape == (action_dim,), f"Expected ({action_dim},), got {action.shape}"
    print(f"    选择的动作: {action.shape}")
    
    # 存储经验
    next_obs = {k: np.random.randn(*v.shape) for k, v in obs.items()}
    agent.store_transition(obs, action, 0.5, next_obs, False)
    
    # 训练步骤
    losses = agent.train_step(batch_size=8)
    if losses:
        print(f"    训练损失: {losses}")
    
    print("    ✅ 世界模型智能体测试通过")


def test_replay_buffer():
    """测试经验回放缓冲区"""
    print("\n[6] 经验回放缓冲区测试")
    
    buffer = ReplayBuffer(capacity=100)
    
    # 添加经验
    for i in range(50):
        obs = {
            'vision': np.random.randn(512),
            'audio': np.random.randn(128)
        }
        action = np.random.randn(6)
        reward = np.random.randn()
        next_obs = {
            'vision': np.random.randn(512),
            'audio': np.random.randn(128)
        }
        done = False
        
        buffer.push(obs, action, reward, next_obs, done)
    
    assert len(buffer) == 50, f"Expected 50, got {len(buffer)}"
    
    # 采样
    batch = buffer.sample(8, 'cpu')
    
    assert batch['observations']['vision'].shape == (8, 512)
    assert batch['actions'].shape == (8, 6)
    assert batch['rewards'].shape == (8,)
    
    print(f"    缓冲区大小: {len(buffer)}")
    print(f"    Batch 形状: vision={batch['observations']['vision'].shape}")
    print("    ✅ 经验回放缓冲区测试通过")


def test_agv_grades():
    """测试 AGV 五级配置"""
    print("\n[7] AGV 五级世界模型配置测试")
    
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        config = get_world_model_spec(grade)
        print(f"    {grade}: latent={config.latent_dim}, hidden={config.hidden_dim}, horizon={config.imagination_horizon}")
        
    print("    ✅ AGV 五级配置测试通过")


def test_world_model_training():
    """测试世界模型训练 (使用序列批次)"""
    print("\n[8] 世界模型训练测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    action_dim = 6
    T, B = 5, 4  # 序列长度和批量大小
    
    agent = create_world_model_agent('M', obs_dims, action_dim)
    
    # 创建序列数据
    observations = {
        'vision': torch.randn(T, B, 512),
        'audio': torch.randn(T, B, 128),
        'tactile': torch.randn(T, B, 64),
        'force': torch.randn(T, B, 32),
        'imu': torch.randn(T, B, 64)
    }
    
    actions = torch.randn(T, B, action_dim)
    rewards = torch.randn(T, B)
    dones = torch.zeros(T, B)
    
    # 直接创建批次并训练
    batch = {
        'observations': {k: v.clone() for k, v in observations.items()},
        'actions': actions.clone(),
        'rewards': rewards.clone(),
        'dones': dones.clone()
    }
    
    losses_list = []
    for _ in range(5):
        agent.world_model.train()
        losses = agent.world_model.update(batch, agent.world_model_optimizer)
        losses_list.append(losses)
    
    if losses_list:
        final_loss = losses_list[-1].get('total', 0)
        print(f"    训练步数: {agent.train_steps}")
        print(f"    最终损失: {final_loss:.4f}")
    
    print("    ✅ 世界模型训练测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("World Model 测试套件")
    print("=" * 60)
    
    try:
        test_observation_encoder()
        test_transition_model()
        test_reward_value_models()
        test_world_model()
        test_replay_buffer()
        test_agv_grades()
        test_world_model_agent()
        test_world_model_training()
        
        print("\n" + "=" * 60)
        print("🎉 所有 World Model 测试通过!")
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
