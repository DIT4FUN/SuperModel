# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
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
AGV障碍物回避模块
=================

AGV避障与动态路径重规划
- 动态窗口法 (DWA / Dynamic Window Approach)
- 人工势场法 (APF / Artificial Potential Field)
- 向量场直方图 (VFH / Vector Field Histogram)
- 混合避障策略

支持AGV等级: M / L / XL / XXL (S级不支持主动避障)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Callable
from enum import Enum
import math


class AvoidanceStrategy(Enum):
    """避障策略"""
    DWA = "dwa"                         # 动态窗口法
    APF = "apf"                         # 人工势场
    VFH = "vfh"                         # 向量场直方图
    HYBRID = "hybrid"                   # 混合策略


@dataclass
class Obstacle:
    """障碍物"""
    position: np.ndarray      # 2, 世界坐标系 (x, y) m
    radius: float             # 障碍物半径 m
    velocity: np.ndarray = None  # 2, 速度 m/s
    type: str = "static"      # "static" | "dynamic"
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if self.velocity is None:
            self.velocity = np.zeros(2)
        else:
            if isinstance(self.velocity, list):
                self.velocity = np.array(self.velocity, dtype=np.float32)
    
    @property
    def center(self) -> np.ndarray:
        return self.position
    
    def predict_position(self, dt: float) -> np.ndarray:
        """预测未来位置"""
        return self.position + self.velocity * dt


@dataclass
class VelocityCommand:
    """速度指令"""
    vx: float         # x方向线速度 m/s
    vy: float         # y方向线速度 m/s
    omega: float      # 角速度 rad/s
    score: float = 0.0  # 轨迹评分
    
    def to_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.omega])


@dataclass
class TrajectorySample:
    """轨迹样本 (用于DWA)"""
    vx: float
    vy: float
    omega: float
    heading_score: float = 0.0   # 方向评分
    velocity_score: float = 0.0  # 速度评分
    obstacle_score: float = 0.0   # 避障评分
    total_score: float = 0.0     # 总评分
    clearance: float = 0.0        # 与最近障碍物距离
    
    def compute_total(self) -> float:
        self.total_score = (
            0.4 * self.heading_score +
            0.2 * self.velocity_score +
            0.4 * self.obstacle_score
        )
        return self.total_score


@dataclass
class DWAConfig:
    """DWA配置"""
    # 速度空间
    max_linear_speed: float = 1.0      # 最大线速度 m/s
    max_angular_speed: float = 2.0     # 最大角速度 rad/s
    min_linear_speed: float = 0.0      # 最小线速度 m/s
    min_angular_speed: float = -2.0    # 最小角速度 rad/s
    # 加速度限制
    max_linear_accel: float = 2.0      # 最大线加速度 m/s^2
    max_angular_accel: float = 3.0    # 最大角加速度 rad/s^2
    # 分辨率
    vx_resolution: float = 0.05         # 线速度分辨率 m/s
    vy_resolution: float = 0.05         # 线速度分辨率 m/s
    omega_resolution: float = 0.1      # 角速度分辨率 rad/s
    # 预测时间
    prediction_horizon: float = 2.0    # 预测时间窗口 s
    # 权重
    heading_weight: float = 0.4        # 方向权重
    velocity_weight: float = 0.2        # 速度权重
    obstacle_weight: float = 0.4        # 避障权重
    # 安全
    robot_radius: float = 0.3           # 机器人半径 m
    obstacle_margin: float = 0.1        # 障碍物裕度 m


@dataclass
class APFConfig:
    """人工势场配置"""
    # 吸引场参数
    attract_gain: float = 5.0           # 吸引增益
    goal_tolerance: float = 0.1        # 目标容差 m
    # 排斥场参数
    repel_gain: float = 100.0           # 排斥增益
    repel_range: float = 2.0            # 排斥场作用范围 m
    # 安全
    robot_radius: float = 0.3           # 机器人半径 m
    obstacle_margin: float = 0.1        # 障碍物裕度 m
    # 局部最小值逃脱
    escape_gain: float = 2.0            # 逃脱增益
    escape_threshold: float = 0.05      # 逃脱判定阈值


@dataclass
class VFHConfig:
    """VFH配置"""
    # 极坐标网格
    sector_angle: float = 5.0           # 扇区角度 deg
    detection_radius: float = 3.0        # 检测半径 m
    # 阈值
    obstacle_threshold: float = 50.0     # 障碍物密度阈值
    # 速度
    max_linear_speed: float = 1.0      # 最大线速度 m/s
    max_angular_speed: float = 2.0     # 最大角速度 rad/s
    # 安全
    robot_radius: float = 0.3           # 机器人半径 m


@dataclass
class AvoidanceConfig:
    """综合避障配置"""
    strategy: AvoidanceStrategy = AvoidanceStrategy.DWA
    dwa: DWAConfig = None
    apf: APFConfig = None
    vfh: VFHConfig = None
    safety_distance: float = 0.5         # 安全距离 m
    grade: str = "M"                     # AGV等级

    def __post_init__(self):
        if self.dwa is None:
            self.dwa = DWAConfig()
        if self.apf is None:
            self.apf = APFConfig()
        if self.vfh is None:
            self.vfh = VFHConfig()


class DynamicWindowApproach:
    """
    动态窗口法 (DWA)
    
    在速度空间中搜索最优控制指令
    目标函数: score = w_heading * heading + w_vel * velocity + w_obs * obstacle
    """
    
    def __init__(self, config: Optional[DWAConfig] = None):
        self.config = config or DWAConfig()
    
    def compute_velocities(
        self,
        robot_pose: np.ndarray,           # (x, y, theta)
        robot_velocity: np.ndarray,        # (vx, vy, omega)
        goal: np.ndarray,                  # (gx, gy)
        obstacles: List[Obstacle],
        dt: float = 0.1
    ) -> VelocityCommand:
        """
        计算最优速度指令
        
        Args:
            robot_pose: 机器人位姿 (x, y, theta) m, rad
            robot_velocity: 当前速度 (vx, vy, omega) m/s, rad/s
            goal: 目标点 (gx, gy) m
            obstacles: 障碍物列表
            dt: 控制周期 s
        
        Returns:
            最优速度指令
        """
        vx_cur, vy_cur, omega_cur = robot_velocity
        
        # 采样速度空间
        samples = self._sample_velocity_space(vx_cur, vy_cur, omega_cur, dt)
        
        if not samples:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0)
        
        # 评估每个样本
        best_sample = None
        best_score = -float('inf')
        
        for vx, vy, omega in samples:
            sample = TrajectorySample(vx, vy, omega)
            
            # 方向评分
            sample.heading_score = self._heading_score(
                robot_pose, vx, vy, omega, goal, dt
            )
            
            # 速度评分
            sample.velocity_score = self._velocity_score(vx, vy)
            
            # 避障评分
            sample.obstacle_score, sample.clearance = self._obstacle_score(
                robot_pose, vx, vy, omega, obstacles, dt
            )
            
            # 计算总分
            sample.compute_total()
            
            if sample.total_score > best_score:
                best_score = sample.total_score
                best_sample = sample
        
        if best_sample is None:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0)
        
        return VelocityCommand(
            vx=best_sample.vx,
            vy=best_sample.vy,
            omega=best_sample.omega,
            score=best_sample.total_score
        )
    
    def _sample_velocity_space(
        self,
        vx_cur: float, vy_cur: float, omega_cur: float,
        dt: float
    ) -> List[Tuple[float, float, float]]:
        """采样速度空间"""
        samples = []
        
        # 根据当前速度计算动态窗口
        vx_min = max(self.config.min_linear_speed, 
                     vx_cur - self.config.max_linear_accel * dt)
        vx_max = min(self.config.max_linear_speed,
                     vx_cur + self.config.max_linear_accel * dt)
        vy_min = max(self.config.min_linear_speed,
                     vy_cur - self.config.max_linear_accel * dt)
        vy_max = min(self.config.max_linear_speed,
                     vy_cur + self.config.max_linear_accel * dt)
        omega_min = max(self.config.min_angular_speed,
                        omega_cur - self.config.max_angular_accel * dt)
        omega_max = min(self.config.max_angular_speed,
                        omega_cur + self.config.max_angular_accel * dt)
        
        vx_range = np.arange(vx_min, vx_max + 1e-9, self.config.vx_resolution)
        vy_range = np.arange(vy_min, vy_max + 1e-9, self.config.vy_resolution)
        omega_range = np.arange(omega_min, omega_max + 1e-9, self.config.omega_resolution)
        
        # 限制采样数量
        max_samples = 1000
        if len(vx_range) * len(vy_range) * len(omega_range) > max_samples:
            vx_range = np.linspace(vx_min, vx_max, min(10, len(vx_range)))
            vy_range = np.linspace(vy_min, vy_max, min(10, len(vy_range)))
            omega_range = np.linspace(omega_min, omega_max, min(10, len(omega_range)))
        
        for vx in vx_range:
            for vy in vy_range:
                for omega in omega_range:
                    samples.append((vx, vy, omega))
        
        return samples
    
    def _heading_score(
        self,
        pose: np.ndarray,
        vx: float, vy: float, omega: float,
        goal: np.ndarray,
        dt: float
    ) -> float:
        """计算方向评分"""
        # 预测终点位置
        x, y, theta = pose
        pred_time = min(self.config.prediction_horizon, 3.0)
        
        # 简化的运动学模型
        x_pred = x + vx * pred_time * math.cos(theta) - vy * pred_time * math.sin(theta)
        y_pred = y + vx * pred_time * math.sin(theta) + vy * pred_time * math.cos(theta)
        
        # 计算朝向目标的角度
        dx = goal[0] - x_pred
        dy = goal[1] - y_pred
        target_angle = math.atan2(dy, dx)
        
        # 计算角度差
        angle_diff = abs(target_angle - theta)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        
        # 转换为评分 (角度差越小评分越高)
        return math.pi - angle_diff
    
    def _velocity_score(self, vx: float, vy: float) -> float:
        """计算速度评分"""
        v_mag = math.sqrt(vx**2 + vy**2)
        return v_mag / self.config.max_linear_speed
    
    def _obstacle_score(
        self,
        pose: np.ndarray,
        vx: float, vy: float, omega: float,
        obstacles: List[Obstacle],
        dt: float
    ) -> Tuple[float, float]:
        """计算避障评分"""
        x, y, theta = pose
        
        min_clearance = float('inf')
        pred_time = min(self.config.prediction_horizon, 3.0)
        
        for t in np.linspace(0, pred_time, 10):
            # 预测位置
            x_pred = x + vx * t * math.cos(theta) - vy * t * math.sin(theta)
            y_pred = y + vx * t * math.sin(theta) + vy * t * math.cos(theta)
            
            for obs in obstacles:
                # 考虑障碍物移动
                if obs.type == "dynamic":
                    obs_pos = obs.predict_position(t)
                else:
                    obs_pos = obs.position
                
                # 计算距离
                dist = math.sqrt((x_pred - obs_pos[0])**2 + (y_pred - obs_pos[1])**2)
                clearance = dist - obs.radius - self.config.robot_radius - self.config.obstacle_margin
                
                if clearance < min_clearance:
                    min_clearance = clearance
                
                # 碰撞检测
                if clearance <= 0:
                    return 0.0, 0.0
        
        # 评分: 距离越远评分越高
        if min_clearance == float('inf'):
            return 1.0, 999.0
        
        safe_range = self.config.safety_distance if hasattr(self.config, 'safety_distance') else 0.5
        score = min(1.0, min_clearance / safe_range) if min_clearance > 0 else 0.0
        return score, min_clearance


class ArtificialPotentialField:
    """
    人工势场法 (APF)
    
    目标产生吸引力，障碍物产生排斥力
   合力决定运动方向
    """
    
    def __init__(self, config: Optional[APFConfig] = None):
        self.config = config or APFConfig()
    
    def compute_force(
        self,
        robot_pose: np.ndarray,       # (x, y)
        robot_velocity: np.ndarray,  # (vx, vy)
        goal: np.ndarray,            # (gx, gy)
        obstacles: List[Obstacle]
    ) -> np.ndarray:
        """
        计算合力
        
        Args:
            robot_pose: 机器人位置 (x, y) m
            robot_velocity: 机器人速度 (vx, vy) m/s
            goal: 目标位置 (gx, gy) m
            obstacles: 障碍物列表
        
        Returns:
            力向量 (fx, fy)
        """
        # 吸引力求
        f_att = self._attractive_force(robot_pose, goal)
        
        # 排斥力求
        f_rep = self._repulsive_force(robot_pose, obstacles)
        
        # 局部最小值逃脱
        f_escape = self._escape_force(robot_pose, robot_velocity)
        
        # 合力
        f_total = f_att + f_rep + f_escape
        return f_total
    
    def _attractive_force(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """吸引力求"""
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < self.config.goal_tolerance:
            return np.zeros(2)
        
        # 锥形势场
        return self.config.attract_gain * np.array([dx, dy])
    
    def _repulsive_force(
        self, pos: np.ndarray, obstacles: List[Obstacle]
    ) -> np.ndarray:
        """排斥力求"""
        f_rep = np.zeros(2)
        
        for obs in obstacles:
            dx = pos[0] - obs.position[0]
            dy = pos[1] - obs.position[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            effective_radius = obs.radius + self.config.robot_radius + self.config.obstacle_margin
            
            if dist < self.config.repel_range:
                if dist < 1e-6:
                    # 机器人与障碍物重叠，随机方向
                    return np.array([self.config.repel_gain * 0.1, 0.0])
                
                # 排斥力大小
                f_mag = self.config.repel_gain * (
                    1.0 / dist - 1.0 / self.config.repel_range
                ) / (dist ** 2)
                
                # 排斥力方向
                f_dir = np.array([dx, dy]) / dist
                f_rep += f_mag * f_dir
        
        return f_rep
    
    def _escape_force(
        self, pos: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """局部最小值逃脱力"""
        v_mag = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        if v_mag > self.config.escape_threshold:
            return np.zeros(2)
        
        # 速度为零或接近零时，随机逃脱
        angle = np.random.uniform(0, 2 * math.pi)
        return self.config.escape_gain * np.array([math.cos(angle), math.sin(angle)])
    
    def compute_velocity(
        self,
        robot_pose: np.ndarray,
        robot_velocity: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        max_speed: float = 1.0
    ) -> np.ndarray:
        """计算速度指令"""
        force = self.compute_force(robot_pose, robot_velocity, goal, obstacles)
        
        # 限制速度
        v_desired = force  # 在APF中，力直接作为期望速度
        v_mag = math.sqrt(v_desired[0]**2 + v_desired[1]**2)
        
        if v_mag > max_speed:
            v_desired = v_desired / v_mag * max_speed
        
        return v_desired


class VectorFieldHistogram:
    """
    向量场直方图 (VFH)
    
    使用极坐标直方图表示障碍物分布
    选择最畅通的方向移动
    """
    
    def __init__(self, config: Optional[VFHConfig] = None):
        self.config = config or VFHConfig()
        # 计算扇区数量
        self.num_sectors = int(360.0 / self.config.sector_angle)
    
    def compute_direction(
        self,
        robot_pose: np.ndarray,       # (x, y, theta)
        robot_velocity: np.ndarray,   # (vx, vy, omega)
        goal: np.ndarray,             # (gx, gy)
        obstacles: List[Obstacle],
        dt: float = 0.1
    ) -> Tuple[float, np.ndarray]:
        """
        计算运动方向和速度
        
        Returns:
            (steering_angle, velocity_cmd)
        """
        x, y, theta = robot_pose
        
        # 构建极坐标直方图
        histogram = self._build_histogram(robot_pose, obstacles)
        
        # 找到最畅通的方向
        goal_angle = math.atan2(goal[1] - y, goal[0] - x)
        angle_to_goal = self._normalize_angle(goal_angle - theta)
        
        # 阈值化
        binary_histogram = self._threshold_histogram(histogram)
        
        # 选择方向
        steering_angle = self._select_direction(binary_histogram, angle_to_goal)
        
        # 计算速度
        speed = self._compute_speed(binary_histogram, steering_angle)
        
        # 计算角速度
        angle_diff = self._normalize_angle(steering_angle)
        omega_cmd = self.config.max_angular_speed * angle_diff / (math.pi / 2)
        
        # 转换到世界坐标系
        world_angle = theta + steering_angle
        vx = speed * math.cos(world_angle)
        vy = speed * math.sin(world_angle)
        
        return steering_angle, VelocityCommand(vx, vy, omega_cmd, score=1.0)
    
    def _build_histogram(
        self, pose: np.ndarray, obstacles: List[Obstacle]
    ) -> np.ndarray:
        """构建极坐标直方图"""
        histogram = np.zeros(self.num_sectors)
        x, y, theta = pose
        
        for obs in obstacles:
            dx = obs.position[0] - x
            dy = obs.position[1] - y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > self.config.detection_radius:
                continue
            
            # 计算障碍物所在扇区
            angle = math.atan2(dy, dx)
            relative_angle = self._normalize_angle(angle - theta)
            sector = int((relative_angle + math.pi) / (2 * math.pi) * self.num_sectors)
            sector = max(0, min(self.num_sectors - 1, sector))
            
            # 障碍物密度
            density = 1.0 / (dist ** 2)
            histogram[sector] += density
        
        return histogram
    
    def _threshold_histogram(self, histogram: np.ndarray) -> np.ndarray:
        """阈值化直方图"""
        return (histogram > self.config.obstacle_threshold).astype(float)
    
    def _select_direction(
        self, histogram: np.ndarray, angle_to_goal: float
    ) -> float:
        """选择最畅通的方向"""
        goal_sector = int((angle_to_goal + math.pi) / (2 * math.pi) * self.num_sectors)
        goal_sector = max(0, min(self.num_sectors - 1, goal_sector))
        
        # 找到最接近目标且畅通的方向
        best_sector = goal_sector
        min_diff = float('inf')
        
        for i, val in enumerate(histogram):
            if val < 0.5:  # 畅通
                diff = abs(i - goal_sector)
                diff = min(diff, self.num_sectors - diff)
                if diff < min_diff:
                    min_diff = diff
                    best_sector = i
        
        angle = (best_sector / self.num_sectors) * 2 * math.pi - math.pi
        return angle
    
    def _compute_speed(self, histogram: np.ndarray, steering_angle: float) -> float:
        """根据前方的开阔程度计算速度"""
        sector = int((steering_angle + math.pi) / (2 * math.pi) * self.num_sectors)
        sector = max(0, min(self.num_sectors - 1, sector))
        
        # 前方扇区
        forward_sectors = histogram[max(0, sector-2):min(len(histogram), sector+3)]
        clearance = np.mean(1.0 - forward_sectors)
        
        return self.config.max_linear_speed * clearance
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class ObstacleAvoider:
    """
    障碍物回避主控制器
    
    支持多种避障策略
    """
    
    def __init__(self, config: Optional[AvoidanceConfig] = None):
        self.config = config or AvoidanceConfig()
        
        # 初始化各策略
        self.dwa = DynamicWindowApproach(self.config.dwa)
        self.apf = ArtificialPotentialField(self.config.apf)
        self.vfh = VectorFieldHistogram(self.config.vfh)
        
        # 当前策略
        self.current_strategy = self.config.strategy
    
    def compute_command(
        self,
        robot_pose: np.ndarray,           # (x, y, theta)
        robot_velocity: np.ndarray,        # (vx, vy, omega)
        goal: np.ndarray,                  # (gx, gy)
        obstacles: List[Obstacle],
        dt: float = 0.1
    ) -> VelocityCommand:
        """
        计算避障速度指令
        
        Args:
            robot_pose: 机器人位姿 (x, y, theta) m, rad
            robot_velocity: 当前速度 (vx, vy, omega) m/s, rad/s
            goal: 目标点 (gx, gy) m
            obstacles: 障碍物列表
            dt: 控制周期 s
        
        Returns:
            速度指令
        """
        if self.current_strategy == AvoidanceStrategy.DWA:
            return self.dwa.compute_velocities(
                robot_pose, robot_velocity, goal, obstacles, dt
            )
        elif self.current_strategy == AvoidanceStrategy.APF:
            vel = self.apf.compute_velocity(
                robot_pose[:2], robot_velocity[:2], goal, obstacles
            )
            return VelocityCommand(vx=vel[0], vy=vel[1], omega=0.0, score=1.0)
        elif self.current_strategy == AvoidanceStrategy.VFH:
            _, cmd = self.vfh.compute_direction(
                robot_pose, robot_velocity, goal, obstacles, dt
            )
            return cmd
        elif self.current_strategy == AvoidanceStrategy.HYBRID:
            # 混合策略: 近距离用DWA, 远距离用APF
            dist_to_goal = np.linalg.norm(goal - robot_pose[:2])
            if dist_to_goal < 2.0:
                return self.dwa.compute_velocities(
                    robot_pose, robot_velocity, goal, obstacles, dt
                )
            else:
                vel = self.apf.compute_velocity(
                    robot_pose[:2], robot_velocity[:2], goal, obstacles
                )
                return VelocityCommand(vx=vel[0], vy=vel[1], omega=0.0, score=1.0)
        else:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0)
    
    def set_strategy(self, strategy: AvoidanceStrategy):
        """切换避障策略"""
        self.current_strategy = strategy
    
    @staticmethod
    def create_from_grade(grade: str) -> 'ObstacleAvoider':
        """从AGV等级创建避障器"""
        configs = {
            "S": AvoidanceConfig(
                strategy=AvoidanceStrategy.DWA,
                safety_distance=0.8
            ),
            "M": AvoidanceConfig(
                strategy=AvoidanceStrategy.DWA,
                safety_distance=0.6
            ),
            "L": AvoidanceConfig(
                strategy=AvoidanceStrategy.HYBRID,
                safety_distance=0.5
            ),
            "XL": AvoidanceConfig(
                strategy=AvoidanceStrategy.HYBRID,
                safety_distance=0.4
            ),
            "XXL": AvoidanceConfig(
                strategy=AvoidanceStrategy.HYBRID,
                safety_distance=0.3
            ),
        }
        cfg = configs.get(grade, configs["M"])
        cfg.grade = grade
        return ObstacleAvoider(cfg)


def get_obstacle_avoidance_spec(grade: str) -> Dict:
    """获取AGV等级对应的避障规格"""
    specs = {
        "S": {
            "strategy": "none",
            "max_obstacles": 0,
            "reaction_time": 0.0,
            "clearance": 0.0
        },
        "M": {
            "strategy": "DWA",
            "max_obstacles": 3,
            "reaction_time": 0.2,
            "clearance": 0.6
        },
        "L": {
            "strategy": "HYBRID",
            "max_obstacles": 10,
            "reaction_time": 0.1,
            "clearance": 0.5
        },
        "XL": {
            "strategy": "HYBRID",
            "max_obstacles": 25,
            "reaction_time": 0.05,
            "clearance": 0.4
        },
        "XXL": {
            "strategy": "HYBRID",
            "max_obstacles": 50,
            "reaction_time": 0.02,
            "clearance": 0.3
        },
    }
    return specs.get(grade, specs["M"])
