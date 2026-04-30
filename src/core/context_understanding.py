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
Context Understanding - 上下文理解模块
=====================================

实时融合所有传感器维度信息,构建统一的场景理解:

功能:
  - 多传感器数据融合 (视觉/听觉/触觉/力觉/IMU/激光)
  - 场景重建 (物体检测/位置/关系)
  - 人类状态识别 (位置/意图/情绪)
  - 环境状态评估 (安全/危险/动态/静态)
  - 时序上下文 (历史状态/变化趋势)
  - 预测性理解 (意图预测/轨迹预测)

输出:
  - 统一的ContextRepresentation
  - 危险等级评估
  - 机会识别
  - 行动建议

这是决策引擎的输入模块,负责将Raw传感器数据转换为高层次的场景理解。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import threading


class ContextMode(Enum):
    """上下文模式"""
    NORMAL = "normal"           # 正常操作
    CAUTIOUS = "cautious"       # 谨慎模式
    EMERGENCY = "emergency"      # 紧急模式
    IDLE = "idle"               # 空闲
    LEARNING = "learning"        # 学习模式
    TELEOP = "teleop"           # 示教模式


@dataclass
class ObjectContext:
    """场景物体上下文"""
    object_id: str
    class_name: str
    position: np.ndarray          # 3D位置 (m)
    velocity: Optional[np.ndarray] = None  # 3D速度 (m/s)
    size: Optional[np.ndarray] = None  # 3D尺寸 (m)
    confidence: float = 1.0       # 检测置信度
    is_human: bool = False        # 是否为人类
    is_dynamic: bool = False      # 是否动态物体
    emotional_state: Optional[str] = None  # 情绪状态 (如人类)
    intention: Optional[str] = None  # 意图 (如人类)


@dataclass
class SpatialContext:
    """空间上下文"""
    robot_position: np.ndarray
    robot_orientation: np.ndarray  # quaternion (x,y,z,w)
    navigable_space: float = 1.0  # [0,1] 可导航空间比例
    nearest_obstacle_distance: float = 10.0  # m
    nearest_human_distance: float = 10.0  # m
    occupancy_grid: Optional[np.ndarray] = None  # 占据栅格


@dataclass
class TemporalContext:
    """时序上下文"""
    current_time: float = field(default_factory=time.time)
    time_since_last_obstacle: float = 0.0  # s
    time_since_last_human: float = 0.0  # s
    trend: str = "stable"  # stable/increasing/decreasing
    prediction_horizon: float = 1.0  # s, 预测时间范围


@dataclass
class SocialContext:
    """社交上下文"""
    human_count: int = 0
    human_positions: List[np.ndarray] = field(default_factory=list)
    human_intentions: List[str] = field(default_factory=list)
    human_trust_level: float = 1.0  # [0,1]
    collaboration_active: bool = False
    instructions_pending: List[str] = field(default_factory=list)


@dataclass
class ContextRepresentation:
    """
    统一上下文表征 - 决策引擎的完整输入

    由以下子上下文组成:
    - spatial: 空间信息
    - temporal: 时序信息
    - social: 社交信息
    - objects: 场景物体列表
    - mode: 当前操作模式
    - hazard_level: 危险等级 [0,1]
    - opportunity_level: 机会等级 [0,1]
    """
    spatial: SpatialContext
    temporal: TemporalContext
    social: SocialContext
    objects: List[ObjectContext] = field(default_factory=list)
    mode: ContextMode = ContextMode.NORMAL
    hazard_level: float = 0.0    # [0,1]
    opportunity_level: float = 0.0  # [0,1]
    confidence: float = 1.0      # 上下文完整度 [0,1]
    raw_summary: Dict[str, Any] = field(default_factory=dict)  # 原始数据摘要


class ContextUnderstanding:
    """
    上下文理解系统 - 实时场景重建

    职责:
    1. 接收并融合多模态传感器数据
    2. 构建统一的场景表征
    3. 评估危险和机会等级
    4. 预测未来状态
    5. 提供决策所需的上下文信息

    数据流:
      sensors → fusion → scene_reconstruction → context_representation → decision_making

    使用方式:
      ctx = ContextUnderstanding()

      # 更新上下文 (每周期调用)
      ctx.update(sensor_data)

      # 获取当前上下文表征
      representation = ctx.get_context()

      # 获取危险区域
      hazard_zones = ctx.get_hazard_zones()
    """

    def __init__(self):
        self._lock = threading.RLock()

        # 当前的上下文表征
        self._current_context: Optional[ContextRepresentation] = None

        # 历史上下文 (用于时序分析)
        self._history: List[ContextRepresentation] = []
        self._max_history = 100

        # 物体追踪
        self._tracked_objects: Dict[str, ObjectContext] = {}
        self._object_id_counter = 0

        # 模式估计
        self._mode_estimators: List[Tuple[str, Callable]] = []
        self._current_mode: ContextMode = ContextMode.NORMAL

        # 回调
        self._on_context_change: Optional[Callable] = None
        self._on_hazard_detected: Optional[Callable] = None

    def update(
        self,
        vision: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        tactile: Optional[np.ndarray] = None,
        force: Optional[np.ndarray] = None,
        imu: Optional[np.ndarray] = None,
        laser_ranges: Optional[np.ndarray] = None,
        joint_positions: Optional[np.ndarray] = None,
        joint_velocities: Optional[np.ndarray] = None,
        robot_position: Optional[np.ndarray] = None,
        robot_velocity: Optional[np.ndarray] = None,
        robot_orientation: Optional[np.ndarray] = None,
        human_positions: Optional[List[np.ndarray]] = None,
        human_intentions: Optional[List[str]] = None,
        human_emotional_states: Optional[List[str]] = None,
        robot_battery_level: float = 1.0,
        robot_temperature: float = 25.0,
        robot_faults: Optional[List[str]] = None,
        raw_context: Optional[Any] = None,
    ) -> ContextRepresentation:
        """
        更新上下文 (从传感器数据)

        这是主要的数据输入接口,每控制周期调用一次:

        Args:
            vision: 视觉特征
            audio: 听觉特征
            tactile: 触觉阵列
            force: 六维力矩
            imu: IMU数据 (姿态/加速度/角速度)
            laser_ranges: 激光雷达数据
            joint_positions: 关节位置
            joint_velocities: 关节速度
            robot_position: 机器人位置
            robot_velocity: 机器人速度
            robot_orientation: 机器人朝向 (quaternion)
            human_positions: 检测到的人类位置列表
            human_intentions: 人类意图列表
            human_emotional_states: 人类情绪状态列表
            robot_battery_level: 电量 [0,1]
            robot_temperature: 温度 (°C)
            robot_faults: 故障列表
            raw_context: 原始GoalContext (用于兼容)

        Returns:
            ContextRepresentation: 统一的上下文表征
        """
        with self._lock:
            now = time.time()

            # ── 空间上下文 ──
            spatial = SpatialContext(
                robot_position=robot_position if robot_position is not None else np.zeros(3),
                robot_orientation=robot_orientation if robot_orientation is not None else np.array([0, 0, 0, 1]),
                nearest_obstacle_distance=(
                    np.min(laser_ranges) if laser_ranges is not None else 10.0
                ),
                nearest_human_distance=(
                    min([np.linalg.norm(hp - robot_position) for hp in human_positions])
                    if human_positions and robot_position is not None
                    else 10.0
                ) if human_positions else 10.0,
            )

            # ── 时序上下文 ──
            temporal = TemporalContext(
                current_time=now,
            )

            # ── 社交上下文 ──
            social = SocialContext(
                human_count=len(human_positions) if human_positions else 0,
                human_positions=human_positions or [],
                human_intentions=human_intentions or [],
            )

            # ── 更新物体追踪 ──
            objects = self._update_object_tracking(
                vision=vision,
                human_positions=human_positions,
                human_intentions=human_intentions,
                human_emotional_states=human_emotional_states,
                laser_ranges=laser_ranges,
            )

            # ── 估计当前模式 ──
            mode = self._estimate_mode(
                spatial=spatial,
                social=social,
                faults=robot_faults or [],
            )

            # ── 评估危险等级 ──
            hazard = self._evaluate_hazard_level(spatial, social, robot_faults or [])

            # ── 评估机会等级 ──
            opportunity = self._evaluate_opportunity_level(spatial, social)

            # ── 构建表征 ──
            self._current_context = ContextRepresentation(
                spatial=spatial,
                temporal=temporal,
                social=social,
                objects=objects,
                mode=mode,
                hazard_level=hazard,
                opportunity_level=opportunity,
                confidence=self._calculate_confidence(vision, laser_ranges, human_positions),
            )

            # ── 更新历史 ──
            self._history.append(self._current_context)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # ── 触发回调 ──
            if hazard > 0.5 and self._on_hazard_detected:
                self._on_hazard_detected(hazard, spatial, social)

            return self._current_context

    def update_from_goal_context(self, context: Any):
        """
        从GoalContext更新 (兼容接口)

        Args:
            context: GoalContext对象
        """
        return self.update(
            vision=getattr(context, 'vision', None),
            audio=getattr(context, 'audio', None),
            tactile=getattr(context, 'tactile', None),
            force=getattr(context, 'force', None),
            imu=getattr(context, 'imu_pose', None),
            laser_ranges=getattr(context, 'laser_ranges', None),
            joint_positions=getattr(context, 'joint_positions', None),
            joint_velocities=getattr(context, 'joint_velocities', None),
            robot_position=getattr(context, 'robot_position', None),
            robot_velocity=getattr(context, 'robot_velocity', None),
            human_positions=getattr(context, 'human_positions', None),
            human_intentions=getattr(context, 'human_intentions', None),
            robot_battery_level=getattr(context, 'robot_battery_level', 1.0),
            robot_temperature=getattr(context, 'robot_temperature', 25.0),
            robot_faults=getattr(context, 'robot_faults', None),
        )

    def _update_object_tracking(
        self,
        vision: Optional[np.ndarray],
        human_positions: Optional[List[np.ndarray]],
        human_intentions: Optional[List[str]],
        human_emotional_states: Optional[List[str]],
        laser_ranges: Optional[np.ndarray],
    ) -> List[ObjectContext]:
        """更新物体追踪"""
        objects = []

        # 追踪人类
        if human_positions:
            for i, (pos, intention, emotion) in enumerate(zip(
                human_positions,
                human_intentions or [None] * len(human_positions),
                human_emotional_states or [None] * len(human_positions),
            )):
                obj_id = f"human_{i}"
                obj = ObjectContext(
                    object_id=obj_id,
                    class_name="human",
                    position=pos,
                    is_human=True,
                    is_dynamic=True,
                    intention=intention,
                    emotional_state=emotion,
                )
                self._tracked_objects[obj_id] = obj
                objects.append(obj)

        return objects

    def _estimate_mode(
        self,
        spatial: SpatialContext,
        social: SocialContext,
        faults: List[str],
    ) -> ContextMode:
        """估计当前操作模式"""
        # 紧急模式
        if spatial.nearest_obstacle_distance < 0.3:
            return ContextMode.EMERGENCY
        if spatial.nearest_human_distance < 0.5:
            return ContextMode.EMERGENCY

        # 谨慎模式
        if spatial.nearest_obstacle_distance < 1.0:
            return ContextMode.CAUTIOUS
        if spatial.nearest_human_distance < 1.5:
            return ContextMode.CAUTIOUS

        # 示教模式 (有待处理指令)
        if social.instructions_pending:
            return ContextMode.TELEOP

        # 故障时降级
        if faults:
            return ContextMode.CAUTIOUS

        return ContextMode.NORMAL

    def _evaluate_hazard_level(
        self,
        spatial: SpatialContext,
        social: SocialContext,
        faults: List[str],
    ) -> float:
        """评估危险等级 [0, 1]"""
        hazard = 0.0

        # 障碍物距离贡献
        if spatial.nearest_obstacle_distance < 0.5:
            hazard += 0.5
        elif spatial.nearest_obstacle_distance < 1.0:
            hazard += 0.3
        elif spatial.nearest_obstacle_distance < 2.0:
            hazard += 0.1

        # 人类距离贡献
        if spatial.nearest_human_distance < 0.5:
            hazard += 0.5
        elif spatial.nearest_human_distance < 1.5:
            hazard += 0.3
        elif spatial.nearest_human_distance < 2.5:
            hazard += 0.1

        # 故障贡献
        if faults:
            hazard += min(0.3, len(faults) * 0.1)

        return min(1.0, hazard)

    def _evaluate_opportunity_level(
        self,
        spatial: SpatialContext,
        social: SocialContext,
    ) -> float:
        """评估机会等级 [0, 1]"""
        opportunity = 0.0

        # 空间充裕是机会
        if spatial.navigable_space > 0.8:
            opportunity += 0.3

        # 无障碍物是机会
        if spatial.nearest_obstacle_distance > 3.0:
            opportunity += 0.3

        # 人员指令是机会
        if social.instructions_pending:
            opportunity += 0.4

        return min(1.0, opportunity)

    def _calculate_confidence(
        self,
        vision: Optional[np.ndarray],
        laser_ranges: Optional[np.ndarray],
        human_positions: Optional[List[np.ndarray]],
    ) -> float:
        """计算上下文完整度"""
        confidence = 0.5  # 基础置信度

        if vision is not None:
            confidence += 0.2
        if laser_ranges is not None:
            confidence += 0.2
        if human_positions:
            confidence += 0.1

        return min(1.0, confidence)

    def get_context(self) -> Optional[ContextRepresentation]:
        """获取当前上下文表征"""
        return self._current_context

    def get_context_history(self, last_n: int = 10) -> List[ContextRepresentation]:
        """获取最近的上下文历史"""
        return self._history[-last_n:]

    def get_hazard_zones(self) -> List[Tuple[np.ndarray, float, float]]:
        """
        获取危险区域

        Returns:
            List[Tuple[center, radius, severity]]: 危险区域列表
        """
        if self._current_context is None:
            return []

        zones = []
        ctx = self._current_context

        # 障碍物危险区
        if ctx.spatial.nearest_obstacle_distance < 2.0:
            # 简化为球形区域
            center = ctx.spatial.robot_position.copy()
            # 假设障碍物在机器人朝向方向
            center[:2] += 1.0  # 前方1米
            radius = 1.0
            severity = 1.0 - ctx.spatial.nearest_obstacle_distance / 2.0
            zones.append((center, radius, severity))

        # 人类周围的安全区
        for human_pos in ctx.social.human_positions:
            zones.append((human_pos, 1.5, 0.5))  # 1.5m半径,中等严重度

        return zones

    def predict_future_state(
        self,
        horizon_s: float = 1.0,
        action: Optional[np.ndarray] = None,
    ) -> Optional[ContextRepresentation]:
        """
        预测未来状态

        Args:
            horizon_s: 预测时间范围 (秒)
            action: 拟执行的动作 (6维)

        Returns:
            预测的上下文表征
        """
        if self._current_context is None:
            return None

        # 简化预测: 基于当前速度和方向外推
        pred_ctx = self._current_context
        spatial = pred_ctx.spatial

        if spatial.robot_velocity is not None:
            # 简单外推
            dt = horizon_s
            predicted_pos = spatial.robot_position + spatial.robot_velocity * dt
            # (实际应该更复杂,考虑轨迹预测)
            spatial.robot_position = predicted_pos

        return pred_ctx

    def set_callbacks(
        self,
        on_context_change: Optional[Callable] = None,
        on_hazard_detected: Optional[Callable] = None,
    ):
        """设置回调"""
        self._on_context_change = on_context_change
        self._on_hazard_detected = on_hazard_detected

    def get_status(self) -> Dict[str, Any]:
        """获取上下文理解系统状态"""
        ctx = self._current_context
        return {
            "mode": ctx.mode.value if ctx else "unknown",
            "hazard_level": ctx.hazard_level if ctx else 0.0,
            "opportunity_level": ctx.opportunity_level if ctx else 0.0,
            "confidence": ctx.confidence if ctx else 0.0,
            "human_count": ctx.social.human_count if ctx else 0,
            "object_count": len(ctx.objects) if ctx else 0,
            "history_length": len(self._history),
        }
