# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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


@dataclass
class GymAGVSpec:
    """AGV五级规格结构化数据类

    提供类型安全的AGV等级规格访问，包括:
    - 物理参数 (质量/速度/负载)
    - 控制参数 (周期/频率/响应时间)
    - 感知参数 (传感器型号/采样率)
    - 奖励权重 (跟踪/平滑/能量/碰撞)
    """
    grade: str = 'M'
    grade_desc: str = '中型AGV'
    payload_kg: float = 100.0
    max_speed_mps: float = 1.5
    dt: float = 0.01
    sim_dt: float = 0.001
    control_freq_hz: float = 100.0
    obs_noise: float = 0.005
    episode_length: int = 1000
    reward_tracking: float = 1.0
    reward_smooth: float = 0.01
    reward_energy: float = 0.0005
    reward_contact: float = 0.1
    reward_collision: float = -2.0
    max_torque_nm: float = 100.0
    processor: str = 'RK3588'
    ai_tops: float = 20.0
    imu_type: str = 'BMI088'
    imu_hz: float = 200.0
    camera: str = '双目D435i 720p'
    audio_ch: int = 2
    tactile_array: str = '16x16'
    force_axis: str = '6轴±200N'
    localization_mm: float = 5.0
    collision_response_ms: float = 50.0
    posture_stable_ms: float = 200.0

    @classmethod
    def from_grade(cls, grade: str) -> 'GymAGVSpec':
        spec_map = {
            'S': dict(grade='S', grade_desc='小型AGV', payload_kg=30, max_speed_mps=0.5,
                dt=0.02, sim_dt=0.002, control_freq_hz=50, obs_noise=0.01,
                episode_length=500, reward_tracking=1.0, reward_smooth=0.005,
                reward_energy=0.001, reward_contact=0.05, reward_collision=-1.0,
                max_torque_nm=50, processor='RPi 4B', ai_tops=5,
                imu_type='MPU6050', imu_hz=100, camera='单目640x480', audio_ch=1,
                tactile_array='8x8', force_axis='3轴±100N',
                localization_mm=10, collision_response_ms=100, posture_stable_ms=500),
            'M': dict(grade='M', grade_desc='中型AGV', payload_kg=100, max_speed_mps=1.5,
                dt=0.01, sim_dt=0.001, control_freq_hz=100, obs_noise=0.005,
                episode_length=1000, reward_tracking=1.0, reward_smooth=0.01,
                reward_energy=0.0005, reward_contact=0.1, reward_collision=-2.0,
                max_torque_nm=100, processor='RK3588/Nano', ai_tops=20,
                imu_type='BMI088', imu_hz=200, camera='双目D435i 720p', audio_ch=2,
                tactile_array='16x16', force_axis='6轴±200N',
                localization_mm=5, collision_response_ms=50, posture_stable_ms=200),
            'L': dict(grade='L', grade_desc='大型AGV', payload_kg=300, max_speed_mps=2.0,
                dt=0.005, sim_dt=0.0005, control_freq_hz=200, obs_noise=0.002,
                episode_length=1000, reward_tracking=2.0, reward_smooth=0.02,
                reward_energy=0.0002, reward_contact=0.2, reward_collision=-5.0,
                max_torque_nm=200, processor='Orin NX', ai_tops=100,
                imu_type='BMI088', imu_hz=500, camera='双目D455 60fps', audio_ch=4,
                tactile_array='24x24', force_axis='6轴±500N',
                localization_mm=3, collision_response_ms=20, posture_stable_ms=100),
            'XL': dict(grade='XL', grade_desc='超大型AGV', payload_kg=600, max_speed_mps=2.5,
                dt=0.002, sim_dt=0.0002, control_freq_hz=500, obs_noise=0.001,
                episode_length=2000, reward_tracking=5.0, reward_smooth=0.05,
                reward_energy=0.0001, reward_contact=0.5, reward_collision=-10.0,
                max_torque_nm=500, processor='Orin AGX', ai_tops=300,
                imu_type='ADIS16470', imu_hz=1000, camera='双目+事件相机', audio_ch=6,
                tactile_array='32x32', force_axis='6轴±1000N',
                localization_mm=1, collision_response_ms=10, posture_stable_ms=50),
            'XXL': dict(grade='XXL', grade_desc='重型AGV', payload_kg=1200, max_speed_mps=3.0,
                dt=0.001, sim_dt=0.0001, control_freq_hz=1000, obs_noise=0.0005,
                episode_length=5000, reward_tracking=10.0, reward_smooth=0.1,
                reward_energy=0.00005, reward_contact=1.0, reward_collision=-20.0,
                max_torque_nm=1000, processor='Orin AGX x2+GPU', ai_tops=500,
                imu_type='ADIS16470', imu_hz=2000, camera='多目+3D LiDAR', audio_ch=8,
                tactile_array='48x48', force_axis='6轴±5000N',
                localization_mm=0.5, collision_response_ms=5, posture_stable_ms=20),
        }
        return cls(**(spec_map.get(grade, spec_map['M'])))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (兼容旧AGV_GYM_GRADE_SPEC格式)"""
        return {
            'grade': self.grade, 'grade_desc': self.grade_desc,
            'payload_kg': self.payload_kg, 'max_speed_mps': self.max_speed_mps,
            'dt': self.dt, 'sim_dt': self.sim_dt, 'control_freq_hz': self.control_freq_hz,
            'obs_noise': self.obs_noise, 'episode_length': self.episode_length,
            'reward_tracking': self.reward_tracking, 'reward_smooth': self.reward_smooth,
            'reward_energy': self.reward_energy, 'reward_contact': self.reward_contact,
            'reward_collision': self.reward_collision, 'max_torque_nm': self.max_torque_nm,
            'processor': self.processor, 'ai_tops': self.ai_tops,
            'imu_type': self.imu_type, 'imu_hz': self.imu_hz,
            'camera': self.camera, 'audio_ch': self.audio_ch,
            'tactile_array': self.tactile_array, 'force_axis': self.force_axis,
            'localization_mm': self.localization_mm,
            'collision_response_ms': self.collision_response_ms,
            'posture_stable_ms': self.posture_stable_ms,
        }

    def get_control_params(self) -> Dict[str, Any]:
        """获取控制参数子集"""
        return {
            'dt': self.dt, 'sim_dt': self.sim_dt,
            'control_freq_hz': self.control_freq_hz,
            'max_torque_nm': self.max_torque_nm,
            'collision_response_ms': self.collision_response_ms,
            'posture_stable_ms': self.posture_stable_ms,
        }

    def get_sensor_params(self) -> Dict[str, Any]:
        """获取传感器参数子集"""
        return {
            'imu_type': self.imu_type, 'imu_hz': self.imu_hz,
            'camera': self.camera, 'audio_ch': self.audio_ch,
            'tactile_array': self.tactile_array,
            'force_axis': self.force_axis, 'obs_noise': self.obs_noise,
        }


def get_agv_control_params(grade: str) -> Dict[str, Any]:
    """获取AGV等级对应的控制参数

    Args:
        grade: AGV等级 ('S', 'M', 'L', 'XL', 'XXL')

    Returns:
        控制参数字典
    """
    return GymAGVSpec.from_grade(grade).get_control_params()


def compute_agv_reward(
    grade: str,
    tracking_error: float,
    action: np.ndarray,
    contact: float = 0.0,
    collision: bool = False,
) -> float:
    """根据AGV等级计算奖励

    Args:
        grade: AGV等级
        tracking_error: 位置跟踪误差
        action: 当前动作 (关节力矩)
        contact: 接触量 (0-1)
        collision: 是否发生碰撞

    Returns:
        奖励值
    """
    spec = GymAGVSpec.from_grade(grade)
    reward = spec.reward_tracking * np.exp(-tracking_error * 10.0)
    energy = np.sum(np.square(action)) * spec.dt
    reward += spec.reward_smooth * np.exp(-np.linalg.norm(action) * 0.01)
    reward += spec.reward_energy * (-energy)
    reward += spec.reward_contact * contact
    if collision:
        reward += spec.reward_collision
    return float(reward)


def get_gym_agv_spec(grade: str) -> GymAGVSpec:
    """获取类型安全的AGV Gym规格对象 (推荐使用)

    Example:
        >>> spec = get_gym_agv_spec('M')
        >>> print(spec.control_freq_hz)
        100.0
    """
    return GymAGVSpec.from_grade(grade)


def list_gym_agv_specs() -> Dict[str, GymAGVSpec]:
    """列出所有AGV五级Gym规格对象"""
    return {g: GymAGVSpec.from_grade(g) for g in ['S', 'M', 'L', 'XL', 'XXL']}


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


# =============================================================================
# AGV 五级 Gymnasium 环境规格表
# =============================================================================
# AGV_GYM_GRADE_SPEC now backed by GymAGVSpec (类型安全)
# 使用 get_gym_agv_spec(grade) 获取 GymAGVSpec 对象
# 使用 get_agv_control_params(grade) 获取控制参数
# 使用 get_agv_grade_spec(grade) 获取旧格式字典 (向后兼容)

AGV_GYM_GRADE_SPEC = {g: GymAGVSpec.from_grade(g).to_dict() for g in ['S', 'M', 'L', 'XL', 'XXL']}


def get_agv_grade_spec(grade: str) -> Dict[str, Any]:
    """获取指定AGV等级的Gym规格 (旧格式字典，向后兼容)"""
    return dict(AGV_GYM_GRADE_SPEC.get(grade, AGV_GYM_GRADE_SPEC['M']))



def create_agv_env(
    scenario: str = "reach",
    grade: str = "M",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    """
    创建 SuperModel AGV Gymnasium 环境 (推荐入口)

    这是推荐使用的环境创建函数，相比 make_env 增加了:
    - AGV 五级规格自动注入
    - 配置覆盖支持
    - 完整的场景预设

    Args:
        scenario: 场景类型
            - "reach": 到达目标场景
            - "track": 轨迹跟踪场景
            - "grasp": 抓取操作场景
            - "warehouse": 仓库物流场景
            - "patrol": 巡逻巡检场景
        grade: AGV 等级 ("S", "M", "L", "XL", "XXL")
        render_mode: 渲染模式 ("rgb_array", "human", None)
        seed: 随机种子
        config_overrides: GymEnvConfig 配置覆盖字典

    Returns:
        env: 配置好的 Gymnasium 环境

    AGV 五级规格速查:
        S  : 小型AGV  30kg   50Hz  RPi  5 TOPS
        M  : 中型AGV  100kg  100Hz RK3588 20 TOPS
        L  : 大型AGV  300kg  200Hz Orin NX 100 TOPS
        XL : 超大型AGV 600kg 500Hz Orin AGX 300 TOPS
        XXL: 重型AGV  1200kg 1000Hz Orin AGX x2+GPU 500+ TOPS
    """
    register_gym_envs()

    if grade not in AGV_GYM_GRADE_SPEC:
        raise ValueError(
            f"Unknown grade '{grade}'. "
            f"Valid grades: {list(AGV_GYM_GRADE_SPEC.keys())}"
        )

    grade_spec = AGV_GYM_GRADE_SPEC[grade]

    # 从五级规格构建 GymEnvConfig
    config = GymEnvConfig(
        dt=grade_spec['dt'],
        sim_dt=grade_spec['sim_dt'],
        episode_length=grade_spec['episode_length'],
        obs_noise=grade_spec['obs_noise'],
        grade=grade,
        reward_tracking=grade_spec['reward_tracking'],
        reward_smooth=grade_spec['reward_smooth'],
        reward_energy=grade_spec['reward_energy'],
        reward_contact=grade_spec['reward_contact'],
        reward_collision=grade_spec['reward_collision'],
    )

    # 应用配置覆盖
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    env_id = f"SuperModel-{scenario}-{grade}-v0"

    try:
        env = gym.make(
            env_id,
            render_mode=render_mode,
            scenario=scenario,
            config=config,
        )
    except Exception:
        # Fallback: 直接实例化
        from src.simulation.gym_env import SuperModelGymEnv
        env = SuperModelGymEnv(
            render_mode=render_mode,
            scenario=scenario,
            config=config,
        )

    if seed is not None:
        env.reset(seed=seed)

    # 注入五级规格信息到 info
    env.unwrapped._grade_spec = grade_spec

    return env


def list_agv_grade_specs() -> Dict[str, Dict[str, Any]]:
    """列出所有 AGV 五级 Gym 规格详情 (旧格式字典)"""
    return dict(AGV_GYM_GRADE_SPEC)
