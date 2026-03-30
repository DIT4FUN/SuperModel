"""
Gymnasium 环境包装器
====================

将 SuperModel 仿真环境包装为 Gymnasium 格式
支持:
- RL 训练 (PPO, SAC, TD3 等)
- 环境注册 (gym.register)
- 多场景支持
- sensorimotor learning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import time
import gymnasium as gym
from gymnasium import spaces


@dataclass
class GymEnvConfig:
    """Gym 环境配置"""
    # 仿真参数
    dt: float = 0.01              # 控制周期 (s)
    sim_dt: float = 0.001         # 物理仿真步长 (s)
    episode_length: int = 1000     # 最大 episode 长度

    # 关节配置
    num_joints: int = 6
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None

    # 奖励权重
    reward_tracking: float = 1.0    # 位置跟踪奖励
    reward_smooth: float = 0.01     # 平滑奖励
    reward_energy: float = 0.0001   # 能量效率奖励
    reward_contact: float = 0.1     # 接触奖励
    reward_collision: float = -1.0 # 碰撞惩罚

    # 观测配置
    obs_type: str = "full"        # "full" | "partial" | "image"
    obs_noise: float = 0.0         # 观测噪声

    # AGV 等级
    grade: str = "M"

    def __post_init__(self):
        if self.joint_limits_lower is None:
            self.joint_limits_lower = -np.ones(self.num_joints) * np.pi
        if self.joint_limits_upper is None:
            self.joint_limits_upper = np.ones(self.num_joints) * np.pi


class SuperModelGymEnv(gym.Env):
    """
    SuperModel Gymnasium 环境

    符合 Gymnasium API:
    - reset() -> obs, info
    - step(action) -> obs, reward, terminated, truncated, info
    - render() -> None
    - close() -> None

    观测空间: [joint_pos(6), joint_vel(6), ee_pos(3), ee_quat(4),
               imu(6), force(6), tactile(16), target_pos(3)] = 54 dims
    动作空间: 关节力矩命令 (6,)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[GymEnvConfig] = None,
        render_mode: Optional[str] = None,
        scenario: str = "reach"
    ):
        super().__init__()

        self.config = config or GymEnvConfig()
        self.render_mode = render_mode
        self.scenario = scenario

        # 仿真状态
        self._init_joint_positions()
        self._init_target()
        self._init_sensors()

        # 时间步
        self._timestep = 0
        self._episode_return = 0.0
        self._episode_length = 0

        # 空间定义
        self._define_spaces()

        # 可选渲染器
        self._viewer = None

        # 历史记录 (用于渲染)
        self._history = {
            'joint_pos': [],
            'joint_vel': [],
            'ee_pos': [],
            'rewards': [],
        }

    def _init_joint_positions(self):
        """初始化关节位置"""
        n = self.config.num_joints
        self._joint_pos = np.zeros(n, dtype=np.float32)
        self._joint_vel = np.zeros(n, dtype=np.float32)
        self._joint_torque = np.zeros(n, dtype=np.float32)
        self._joint_acc = np.zeros(n, dtype=np.float32)

        self._mass_diag = np.ones(n) * 0.5
        self._damping = np.ones(n) * 2.0

    def _init_target(self):
        """初始化目标位置"""
        n = self.config.num_joints
        if self.scenario == "reach":
            # 目标: 关节位置
            self._target_pos = np.random.uniform(
                self.config.joint_limits_lower * 0.5,
                self.config.joint_limits_upper * 0.5
            )
        elif self.scenario == "track":
            # 目标: 正弦轨迹
            self._track_phase = 0.0
            self._track_freq = 0.5  # Hz
            self._target_pos = np.zeros(n)
        elif self.scenario == "grasp":
            # 目标: 抓取位置
            self._target_pos = np.array([0.3, 0.0, 0.1, 0.0, 0.0, 0.0])
        else:
            self._target_pos = np.zeros(n)

    def _init_sensors(self):
        """初始化传感器模拟"""
        n = self.config.num_joints
        # IMU (加速度计 + 陀螺仪)
        self._imu_accel = np.zeros(3, dtype=np.float32)
        self._imu_gyro = np.zeros(3, dtype=np.float32)

        # 力矩传感器
        self._wrench = np.zeros(6, dtype=np.float32)

        # 触觉阵列 (4x4)
        self._tactile = np.zeros((4, 4), dtype=np.float32)

        # 末端执行器位置 (简化)
        self._ee_pos = np.zeros(3, dtype=np.float32)
        self._ee_quat = np.array([1, 0, 0, 0], dtype=np.float32)

    def _define_spaces(self):
        """定义观测和动作空间"""
        n = self.config.num_joints

        # 观测空间 (完整状态)
        # joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) +
        # imu(6) + wrench(6) + tactile(16) + target(6) = 53
        obs_dim = n + n + 3 + 4 + 6 + 6 + 16 + n
        high = np.full(obs_dim, 10.0, dtype=np.float32)
        low = np.full(obs_dim, -10.0, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 动作空间: 关节力矩
        # 默认: ±100 Nm
        action_high = np.ones(n) * 100.0
        action_low = -action_high
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # 目标空间 (用于 goal-conditioned RL)
        self.goal_space = spaces.Box(
            low=self.config.joint_limits_lower,
            high=self.config.joint_limits_upper,
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 随机种子
        if seed is not None:
            np.random.seed(seed)

        # 重置关节状态
        self._timestep = 0
        self._episode_return = 0.0
        self._episode_length = 0

        # 随机初始关节位置
        self._joint_pos = np.random.uniform(
            self.config.joint_limits_lower * 0.3,
            self.config.joint_limits_upper * 0.3
        ).astype(np.float32)
        self._joint_vel = np.zeros(self.config.num_joints, dtype=np.float32)
        self._joint_torque = np.zeros(self.config.num_joints, dtype=np.float32)

        # 重新生成目标
        if options and 'target' in options:
            self._target_pos = options['target'].astype(np.float32)
        else:
            self._init_target()

        # 清空历史
        self._history = {
            'joint_pos': [self._joint_pos.copy()],
            'joint_vel': [self._joint_vel.copy()],
            'ee_pos': [self._ee_pos.copy()],
            'rewards': [],
        }

        # 传感器模拟
        self._simulate_sensors()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步

        Args:
            action: 关节力矩命令 (6,)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # 限幅动作
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 物理仿真 (多步)
        self._simulate_physics(action)

        # 传感器更新
        self._simulate_sensors()

        # 时间步更新
        self._timestep += 1
        self._episode_length += 1

        # 计算奖励
        reward, reward_info = self._compute_reward(action)

        # 检查终止条件
        terminated = self._is_terminated()
        truncated = self._timestep >= self.config.episode_length

        self._episode_return += reward

        # 记录历史
        self._history['joint_pos'].append(self._joint_pos.copy())
        self._history['joint_vel'].append(self._joint_vel.copy())
        self._history['ee_pos'].append(self._ee_pos.copy())
        self._history['rewards'].append(reward)

        obs = self._get_observation()
        info = self._get_info()
        info.update(reward_info)

        return obs, reward, terminated, truncated, info

    def _simulate_physics(self, torque: np.ndarray):
        """
        简化物理仿真

        dt 内使用半隐式欧拉积分
        """
        dt = self.config.dt
        n = self.config.num_joints

        # 简化动力学
        # qdd = (tau - damping * qd) / M
        gravity_term = np.zeros(n)
        gravity_term[2] = 9.81 * 0.3  # 重力影响

        qdd = (torque - self._damping * self._joint_vel - gravity_term) / self._mass_diag

        # 半隐式欧拉
        self._joint_vel += qdd * dt
        self._joint_pos += self._joint_vel * dt

        # 限位
        self._joint_pos = np.clip(
            self._joint_pos,
            self.config.joint_limits_lower,
            self.config.joint_limits_upper
        )

        # 停止超速关节
        max_vel = np.pi * 2  # rad/s
        self._joint_vel = np.clip(self._joint_vel, -max_vel, max_vel)

        # 更新末端执行器位置 (简化 FK)
        self._update_ee_pose()

    def _update_ee_pose(self):
        """更新末端执行器位置 (简化前向运动学)"""
        n = self.config.num_joints
        # 简化: 末端位置由前两个关节决定
        link_lengths = np.array([0.3, 0.3, 0.2, 0.1, 0.1, 0.05])
        # 只使用前 3 个连杆长度
        ll3 = link_lengths[:3]
        self._ee_pos[0] = np.sum(ll3 * np.cos(np.cumsum(self._joint_pos[:3]))) * 0.3
        self._ee_pos[1] = np.sum(ll3 * np.sin(np.cumsum(self._joint_pos[:3]))) * 0.3
        self._ee_pos[2] = 0.5 + np.sum(ll3 * np.sin(self._joint_pos[:3])) * 0.5

        # 简化四元数 (从关节角)
        roll = self._joint_pos[0] * 0.3
        pitch = self._joint_pos[1] * 0.3
        yaw = self._joint_pos[2] * 0.3
        self._ee_quat = self._euler_to_quat(roll, pitch, yaw)

    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角转四元数"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz], dtype=np.float32)

    def _simulate_sensors(self):
        """模拟传感器数据"""
        n = self.config.num_joints

        # IMU (加速度计 + 陀螺仪)
        # 陀螺仪 = 关节速度的一部分 + 噪声
        self._imu_gyro = (self._joint_vel[:3] * 0.5 + 
                          np.random.randn(3) * 0.01).astype(np.float32)
        # 加速度 = 重力 + 运动加速度
        accel_noise = np.random.randn(3) * self.config.obs_noise
        self._imu_accel = (np.array([0, 0, 9.81]) +
                            self._joint_acc[:3] * 0.1 +
                            accel_noise).astype(np.float32)

        # 力矩传感器 (关节力矩 + 噪声)
        torque_noise = np.random.randn(n) * 0.5
        self._wrench[:n] = (self._joint_torque * 0.1 + torque_noise).astype(np.float32)
        self._wrench[n:] = np.random.randn(6 - n) * 0.1

        # 触觉阵列 (基于末端执行器位置和接触)
        # 简化: 无接触时为零, 有接触时基于压力
        contact_force = np.linalg.norm(self._wrench[:3])
        if contact_force > 1.0:  # 有接触
            self._tactile = np.random.uniform(0, 0.5, (4, 4)).astype(np.float32) * contact_force
        else:
            self._tactile = np.zeros((4, 4), dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """获取观测向量"""
        # joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) +
        # imu(6) + wrench(6) + tactile(16) + target(6) = 53
        tactile_flat = self._tactile.flatten()

        obs = np.concatenate([
            self._joint_pos,           # 6
            self._joint_vel,           # 6
            self._ee_pos,              # 3
            self._ee_quat,             # 4
            self._imu_accel,           # 3
            self._imu_gyro,            # 3
            self._wrench[:6],          # 6
            tactile_flat,              # 16
            self._target_pos,           # 6
        ]).astype(np.float32)

        # 添加噪声
        if self.config.obs_noise > 0:
            obs += np.random.randn(len(obs)) * self.config.obs_noise

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'timestep': self._timestep,
            'episode_length': self._episode_length,
            'episode_return': self._episode_return,
            'joint_pos': self._joint_pos.copy(),
            'joint_vel': self._joint_vel.copy(),
            'ee_pos': self._ee_pos.copy(),
            'tracking_error': float(np.linalg.norm(self._joint_pos - self._target_pos)),
            'scenario': self.scenario,
        }

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """计算奖励"""
        cfg = self.config
        n = cfg.num_joints

        # 1. 位置跟踪奖励
        pos_error = np.linalg.norm(self._joint_pos - self._target_pos)
        r_tracking = cfg.reward_tracking * np.exp(-pos_error * 2.0)

        # 2. 平滑奖励 (小速度, 小加速度)
        r_smooth = cfg.reward_smooth * (
            -np.sum(self._joint_vel ** 2) * 0.1
            - np.sum(action ** 2) * 0.01
        )

        # 3. 能量效率奖励
        r_energy = cfg.reward_energy * (-np.sum(np.abs(action)))

        # 4. 接触奖励 (力矩传感器的变化)
        contact_magnitude = np.linalg.norm(self._wrench[:n])
        r_contact = cfg.reward_contact * min(contact_magnitude * 0.1, 1.0)

        # 5. 碰撞惩罚 (关节限位附近)
        near_limit = np.sum(
            (self._joint_pos > cfg.joint_limits_upper * 0.9).astype(float) +
            (self._joint_pos < cfg.joint_limits_lower * 0.9).astype(float)
        )
        r_collision = cfg.reward_collision * near_limit * 0.1

        total_reward = r_tracking + r_smooth + r_energy + r_contact + r_collision

        reward_info = {
            'r_tracking': r_tracking,
            'r_smooth': r_smooth,
            'r_energy': r_energy,
            'r_contact': r_contact,
            'r_collision': r_collision,
            'pos_error': float(pos_error),
        }

        return total_reward, reward_info

    def _is_terminated(self) -> bool:
        """检查是否终止"""
        cfg = self.config

        # 关节超限
        if np.any(self._joint_pos > cfg.joint_limits_upper * 1.05):
            return True
        if np.any(self._joint_pos < cfg.joint_limits_lower * 1.05):
            return True

        # 速度过大
        if np.any(np.abs(self._joint_vel) > np.pi * 4):
            return True

        return False

    def render(self):
        """渲染环境 (简化)"""
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        # human 模式: 仅打印状态
        print(f"Step {self._timestep}: pos={self._joint_pos[:3].round(3)}, "
              f"vel={self._joint_vel[:3].round(3)}, "
              f"ee={self._ee_pos.round(3)}")

    def _render_rgb(self) -> np.ndarray:
        """生成 RGB 图像 (简化)"""
        # 创建一个简单的 2D 可视化
        img = np.zeros((240, 320, 3), dtype=np.uint8)

        # 绘制背景
        img[:, :] = [30, 30, 30]

        # 绘制关节位置 (简化机械臂示意图)
        h, w = img.shape[:2]
        cx, cy = w // 2, h - 50

        # 连杆
        link_len = 30
        angles = self._joint_pos[:3]

        for i, angle in enumerate(angles):
            x2 = int(cx + link_len * np.cos(angle - np.pi / 2 + i * 0.3))
            y2 = int(cy - link_len * np.sin(angle - np.pi / 2 + i * 0.3))
            cv2_line = None  # 避免导入 cv2, 用 PIL 代替
            # 简化: 用 numpy 绘制
            img = self._draw_line(img, cx, cy, x2, y2, (100, 200, 100), 3)
            cx, cy = x2, y2

        return img

    def _draw_line(self, img, x1, y1, x2, y2, color, thickness):
        """Bresenham 画线"""
        h, w = img.shape[:2]
        points = self._bresenham(x1, y1, x2, y2)
        for px, py in points:
            if 0 <= px < w and 0 <= py < h:
                img[py, px] = color
        return img

    def _bresenham(self, x1, y1, x2, y2):
        """Bresenham 画线算法"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points

    def close(self):
        """关闭环境"""
        if self._viewer is not None:
            self._viewer = None
        self._history.clear()


# =============================================================================
# 环境注册
# =============================================================================

def register_gym_envs():
    """注册所有 SuperModel Gymnasium 环境"""
    import gymnasium as gym
    from gymnasium.envs.registration import register

    scenarios = ['reach', 'track', 'grasp']
    grades = ['S', 'M', 'L', 'XL', 'XXL']

    for scenario in scenarios:
        for grade in grades:
            env_id = f"SuperModel-{scenario}-{grade}-v0"
            entry_point = 'simulation.gym_env:SuperModelGymEnv'

            try:
                register(
                    id=env_id,
                    entry_point=entry_point,
                    kwargs={
                        'scenario': scenario,
                        'config': GymEnvConfig(grade=grade)
                    },
                    max_episode_steps=1000,
                )
            except Exception:
                pass  # 已注册则跳过


# =============================================================================
# 辅助函数
# =============================================================================

def make_env(
    scenario: str = "reach",
    grade: str = "M",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None
) -> gym.Env:
    """
    创建 SuperModel Gymnasium 环境

    Args:
        scenario: 场景类型 ("reach", "track", "grasp")
        grade: AGV 等级 ("S", "M", "L", "XL", "XXL")
        render_mode: 渲染模式
        seed: 随机种子

    Returns:
        env: Gymnasium 环境
    """
    # 注册环境 (如果尚未注册)
    register_gym_envs()

    env_id = f"SuperModel-{scenario}-{grade}-v0"
    env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)

    return env


def collect_rollout(
    env: gym.Env,
    policy: callable,
    max_steps: int = 1000,
    render: bool = False
) -> Dict[str, Any]:
    """
    收集一条 rollout

    Args:
        env: Gymnasium 环境
        policy: 策略函数, 输入观测返回动作
        max_steps: 最大步数
        render: 是否渲染

    Returns:
        rollout_data: 包含 observations, actions, rewards 等
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []

    obs, info = env.reset()
    done = False
    truncated = False

    for step in range(max_steps):
        if render:
            env.render()

        observations.append(obs)
        action = policy(obs)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        infos.append(info)

        if terminated or truncated:
            break

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'infos': infos,
        'length': len(observations),
        'total_reward': float(np.sum(rewards)),
    }


def get_gym_spec(grade: str = 'M') -> Dict[str, Any]:
    """获取 Gym 环境规格 (按 AGV 等级)"""
    specs = {
        'S': {
            'dt': 0.02, 'obs_noise': 0.01, 'episode_length': 500,
            'reward_tracking': 1.0, 'max_torque': 50,
            'description': 'S级环境, 低精度, 快速响应'
        },
        'M': {
            'dt': 0.01, 'obs_noise': 0.005, 'episode_length': 1000,
            'reward_tracking': 1.0, 'max_torque': 100,
            'description': 'M级环境, 标准精度'
        },
        'L': {
            'dt': 0.01, 'obs_noise': 0.002, 'episode_length': 1000,
            'reward_tracking': 2.0, 'max_torque': 200,
            'description': 'L级环境, 高精度'
        },
        'XL': {
            'dt': 0.005, 'obs_noise': 0.001, 'episode_length': 2000,
            'reward_tracking': 5.0, 'max_torque': 500,
            'description': 'XL级环境, 超高精度'
        },
        'XXL': {
            'dt': 0.002, 'obs_noise': 0.0005, 'episode_length': 5000,
            'reward_tracking': 10.0, 'max_torque': 1000,
            'description': 'XXL级环境, 旗舰精度'
        }
    }
    return specs.get(grade, specs['M'])
