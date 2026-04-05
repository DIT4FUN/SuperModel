#!/usr/bin/env python3
"""
SuperModel Gymnasium 环境演示
============================
使用 Gymnasium API 进行 AGV 强化学习训练

Usage:
    python3 run_gym_demo.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# 检查 Gymnasium
print("=" * 60)
print("SuperModel Gymnasium 环境演示")
print("=" * 60)
print(f"✅ Gymnasium version: {gym.__version__}")

# 检查 PyBullet
print("\n📦 检查仿真后端...")
try:
    import pybullet
    print(f"   PyBullet: {pybullet.__version__}")
except:
    print("   PyBullet: 未安装")

try:
    import mujoco
    print(f"   MuJoCo: {mujoco.__version__}")
except:
    print("   MuJoCo: 未安装")


# 注册 AGV 环境
def register_agv_env():
    """注册 AGV Gymnasium 环境"""
    if 'SuperModelAGV-v0' not in gym.registry:
        register(
            id='SuperModelAGV-v0',
            entry_point='simulation.gym_env:SuperModelGymEnv',
            max_episode_steps=1000,
            reward_threshold=1000.0,
        )
        print("✅ 环境 SuperModelAGV-v0 已注册")
    else:
        print("⚠️ 环境 SuperModelAGV-v0 已存在")


def basic_usage_demo():
    """基本使用演示"""
    print("\n" + "=" * 60)
    print("📚 基本使用演示")
    print("=" * 60)
    
    # 导入环境
    from simulation.gym_env import SuperModelGymEnv
    
    # 创建环境
    print("\n📦 创建环境...")
    env = SuperModelGymEnv(render_mode='rgb_array')
    
    print(f"   观测空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print(f"   状态空间: {env.observation_space.shape}")
    
    # 重置环境
    print("\n🔄 重置环境...")
    obs, info = env.reset(seed=42)
    print(f"   初始观测形状: {obs.shape}")
    
    # 运行几步
    print("\n🚀 运行 10 步仿真...")
    for i in range(10):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i < 5:
            print(f"   Step {i+1}: reward={reward:.4f}, done={terminated or truncated}")
    
    print("\n✅ 基本演示完成!")
    env.close()


def ray_env_demo():
    """Ray/RLlib 风格演示"""
    print("\n" + "=" * 60)
    print("🎮 Ray/RLlib 风格演示")
    print("=" * 60)
    
    from simulation.gym_env import SuperModelGymEnv, GymEnvConfig
    
    # 创建配置
    config = GymEnvConfig(
        dt=0.01,
        episode_length=500,
        grade='M',
        obs_type='full',
        reward_tracking=2.0,
        reward_energy=0.001,
    )
    
    print(f"\n📦 配置:")
    print(f"   dt: {config.dt}s")
    print(f"   episode_length: {config.episode_length}")
    print(f"   grade: {config.grade}")
    
    # 创建环境
    print("\n📦 创建环境...")
    env = SuperModelGymEnv(config=config, render_mode='human')
    
    # 简单的轨迹跟踪
    print("\n🚀 轨迹跟踪演示...")
    
    # 生成目标轨迹 (圆形)
    radius = 1.0
    frequency = 0.5  # Hz
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(200):
        # 目标位置 (圆形轨迹)
        t = step * config.dt
        target_x = radius * np.cos(2 * np.pi * frequency * t)
        target_y = radius * np.sin(2 * np.pi * frequency * t)
        
        # 简单的 P 控制
        current_pos = obs[:2]  # 假设前2维是位置
        error = np.array([target_x, target_y]) - current_pos
        
        # 动作空间是6维的,填充剩余维度
        action = np.zeros(6)
        action[:2] = np.clip(error * 2.0, -1, 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"   Step {step:3d}: target=({target_x:.2f}, {target_y:.2f}), "
                  f"pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                  f"reward={reward:.3f}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            obs, info = env.reset()
    
    print(f"\n📊 总奖励: {total_reward:.2f}")
    
    env.close()
    print("\n✅ Ray/RLlib 风格演示完成!")


def vector_env_demo():
    """向量化环境演示 (用于并行训练)"""
    print("\n" + "=" * 60)
    print("🔢 向量化环境演示")
    print("=" * 60)
    
    from simulation.gym_env import SuperModelGymEnv
    
    # 创建多个环境
    print("\n📦 创建 4 个并行环境...")
    
    # 注意: SuperModelGymEnv 默认不支持多环境
    # 这里演示如何使用 DummyVecEnv 或 SubprocVecEnv
    
    try:
        from gymnasium.vector import DummyVecEnv, SyncVectorEnv
        
        def make_env():
            def _init():
                return SuperModelGymEnv(render_mode=None)
            return _init
        
        # 创建同步向量化环境
        vec_env = SyncVectorEnv([make_env() for _ in range(4)])
        
        print(f"   向量化环境数量: {vec_env.num_envs}")
        
        # 运行几步
        obs = vec_env.reset()
        print(f"   观测形状: {obs.shape}")
        
        for i in range(5):
            actions = np.array([vec_env.action_space.sample() for _ in range(4)])
            obs, rewards, dones, infos = vec_env.step(actions)
            print(f"   Step {i+1}: rewards={rewards}, dones={dones}")
        
        vec_env.close()
        print("\n✅ 向量化环境演示完成!")
        
    except ImportError as e:
        print(f"⚠️ 向量化环境需要额外依赖: {e}")


def main():
    """主函数"""
    # 注册环境
    register_agv_env()
    
    # 基本使用演示
    basic_usage_demo()
    
    # Ray/RLlib 风格演示
    ray_env_demo()
    
    # 向量化环境演示
    vector_env_demo()
    
    print("\n" + "=" * 60)
    print("🎉 所有 Gymnasium 演示完成!")
    print("=" * 60)
    
    print("""
📚 进一步学习:

1. 使用 Stable-Baselines3 训练:
   pip install stable-baselines3
   
   from stable_baselines3 import PPO
   model = PPO('MlpPolicy', 'SuperModelAGV-v0')
   model.learn(total_timesteps=10000)

2. 使用 Ray/RLlib 训练:
   pip install ray[rllib]
   
   from ray.rllib.algorithms import PPO
   config = ppo.PPOConfig().environment('SuperModelAGV-v0')
   algo = config.build()

3. 使用探索环境:
   env = gym.make('SuperModelAGV-v0', render_mode='human')
""")


if __name__ == '__main__':
    main()
