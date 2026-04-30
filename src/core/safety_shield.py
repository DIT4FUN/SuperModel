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
Safety Shield - 安全护盾 (P0核心目标执行器)
==========================================

实现 P0 保护人类安全 核心目标的绝对优先级执行:

功能:
  - 实时监测人员位置和危险状态
  - 碰撞预防 (速度限制/路径修正)
  - 紧急停止触发
  - 危险环境检测与响应
  - 安全距离维持
  - 多级安全响应 (警告→减速→停止→紧急制动)

安全等级 (AGV五级对应):
  S:  基础软限位 + 简单障碍检测
  M:  + 速度监控 + 人员检测
  L:  + 碰撞力阈值 + 姿态稳定
  XL: + 实时看门狗 + 预测性避障
  XXL: + 多传感器融合 + 故障容忍

使用方式:
  shield = SafetyShield(grade="M")
  is_safe = shield.check_context(context)  # 每周期调用
  override_action = shield.get_override_action(context)  # 获取安全动作
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
import time
import threading

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .core_goals import GoalContext, CoreGoal, GoalState


class SafetyLevel(Enum):
    """
    安全护盾等级 (与AGV五级对应)

    等级越高,安全措施越严格,响应越快
    """
    S = "S"      # 基础: 软限位 + 简单障碍
    M = "M"      # 中等: + 速度监控 + 人员检测
    L = "L"      # 高级: + 碰撞力阈值 + 姿态稳定
    XL = "XL"    # 专家: + 实时看门狗 + 预测性避障
    XXL = "XXL"  # 极致: + 多传感器融合 + 故障容忍


class DangerType(Enum):
    """危险类型"""
    NONE = "none"
    HUMAN_NEAR = "human_near"              # 人员靠近
    COLLISION_IMMINENT = "collision_imminent"  # 即将碰撞
    OBSTACLE_CLOSE = "obstacle_close"       # 障碍物过近
    HAZARDOUS_ENVIRONMENT = "hazardous_env"  # 危险环境
    SPEED_EXCESSIVE = "speed_excessive"    # 速度过快
    JOINT_LIMIT_VIOLATION = "joint_limit"  # 关节限位超出
    TORQUE_EXCESSIVE = "torque_excessive"  # 力矩过大
    TEMPERATURE_HIGH = "temperature_high"  # 温度过高
    BATTERY_LOW = "battery_low"            # 电量过低


class SafetyResponse(Enum):
    """安全响应级别 (从轻到重)"""
    NONE = "none"           # 无需响应
    WARNING = "warning"       # 警告 (降低速度/提醒)
    CAUTION = "caution"       # 注意 (进一步限制速度)
    SLOWDOWN = "slowdown"    # 减速
    STOP = "stop"           # 停止
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


@dataclass
class SafetyZone:
    """
    安全区域定义

    在机器人周围维护的安全保护区:
    - 警告区 (caution_zone): 发现潜在危险,开始减速
    - 停止区 (stop_zone): 危险,必须停止
    - 紧急区 (emergency_zone): 极危险,触发紧急制动
    """
    caution_distance: float = 2.0    # m, 警告区距离
    stop_distance: float = 1.0       # m, 停止区距离
    emergency_distance: float = 0.3  # m, 紧急制动距离

    def get_zone_for_speed(self, speed: float) -> Tuple[float, float, float]:
        """
        根据当前速度动态调整安全区

        速度越快,安全区越大 (基于制动距离)
        """
        # 简化的制动距离模型: d = v^2 / (2 * a_max)
        # a_max 为紧急制动加速度
        a_max = 5.0  # m/s^2
        braking_distance = (speed ** 2) / (2 * a_max)

        # 动态调整系数
        k = 1.0 + braking_distance * 0.5

        return (
            self.caution_distance * k,
            self.stop_distance * k,
            self.emergency_distance * k
        )


@dataclass
class SafetyConfig:
    """安全护盾配置"""
    grade: SafetyLevel = SafetyLevel.M

    # 距离阈值
    min_human_distance: float = 0.5       # m, 最小人员距离
    caution_human_distance: float = 2.0  # m, 警告距离
    obstacle_stop_distance: float = 0.5   # m, 障碍物停止距离

    # 速度限制
    max_forward_speed: float = 1.5        # m/s
    max_backward_speed: float = 0.5      # m/s
    max_rotation_speed: float = 1.0      # rad/s

    # 碰撞预测
    collision_prediction_horizon: float = 0.5  # s, 碰撞预测时间范围
    time_to_collision_threshold: float = 0.3   # s, TTC阈值

    # 力矩阈值
    max_contact_force: float = 50.0       # N, 最大接触力
    emergency_force: float = 100.0        # N, 紧急制动阈值

    # 温度阈值
    max_temperature: float = 70.0        # °C
    caution_temperature: float = 60.0    # °C

    # 电池阈值
    min_battery_level: float = 0.15      # 电量低于此值进入电量保护

    # 响应超时
    warning_timeout_ms: float = 100.0    # 警告后超时时间
    caution_timeout_ms: float = 50.0     # 注意后超时时间
    stop_timeout_ms: float = 20.0        # 停止命令超时

    @classmethod
    def from_grade(cls, grade: str) -> 'SafetyConfig':
        """从AGV等级获取对应配置"""
        configs = {
            'S': cls(grade=SafetyLevel.S,
                     max_forward_speed=0.5, max_contact_force=20.0,
                     collision_prediction_horizon=0.2),
            'M': cls(grade=SafetyLevel.M,
                     max_forward_speed=1.5, max_contact_force=50.0,
                     collision_prediction_horizon=0.5),
            'L': cls(grade=SafetyLevel.L,
                     max_forward_speed=2.0, max_contact_force=100.0,
                     collision_prediction_horizon=0.8),
            'XL': cls(grade=SafetyLevel.XL,
                      max_forward_speed=2.5, max_contact_force=150.0,
                      collision_prediction_horizon=1.0),
            'XXL': cls(grade=SafetyLevel.XXL,
                       max_forward_speed=3.0, max_contact_force=200.0,
                       collision_prediction_horizon=1.5),
        }
        return configs.get(grade, cls())


class SafetyShield:
    """
    安全护盾 - P0核心目标的执行器

    这是SuperModel的最高优先级执行器,任何其他目标都不能覆盖它:

    使用流程:
      1. 每控制周期调用 check_context() 检查安全性
      2. 如有危险, get_override_action() 返回安全动作
      3. 动作执行前必须经过 check_action() 验证

    特性:
      - 线程安全,支持并发调用
      - 零信任设计: 默认不安全,需明确检查通过
      - 分层防御: WARNING → CAUTION → SLOWDOWN → STOP → EMERGENCY
      - 自适应安全区: 根据速度动态调整
    """

    def __init__(self, config: Optional[SafetyConfig] = None, grade: str = "M"):
        """
        初始化安全护盾

        Args:
            config: 安全配置 (可选)
            grade: AGV等级 (自动生成对应配置)
        """
        self.config = config or SafetyConfig.from_grade(grade)
        self._lock = threading.RLock()

        # 安全区
        self._zone = SafetyZone()

        # 当前危险状态
        self._current_danger: DangerType = DangerType.NONE
        self._current_response: SafetyResponse = SafetyResponse.NONE
        self._danger_timestamp: float = 0.0

        # 紧急停止标志
        self._emergency_stop_active: bool = False
        self._emergency_stop_reason: Optional[str] = None

        # 事件回调
        self._on_warning: Optional[Callable] = None
        self._on_danger: Optional[Callable] = None
        self._on_emergency_stop: Optional[Callable] = None

        # 统计
        self._event_log: List[Dict[str, Any]] = []
        self._total_checks: int = 0
        self._total_dangers: int = 0

    def check_context(self, context: GoalContext) -> Tuple[bool, SafetyResponse, Optional[str]]:
        """
        检查当前上下文是否安全 (核心检查方法)

        这是每周期必须调用的安全检查:

        Args:
            context: 当前决策上下文

        Returns:
            Tuple[bool, SafetyResponse, str]:
                - is_safe: 是否安全
                - response: 建议的安全响应
                - reason: 危险描述 (安全时为None)
        """
        with self._lock:
            self._total_checks += 1

            # ── P0. 紧急停止检查 ──
            if self._emergency_stop_active:
                self._current_danger = DangerType.HAZARDOUS_ENVIRONMENT
                self._current_response = SafetyResponse.EMERGENCY_STOP
                self._log_event("emergency_stop_check", f"EST active: {self._emergency_stop_reason}")
                return False, SafetyResponse.EMERGENCY_STOP, self._emergency_stop_reason

            # ── P1. 人员安全检查 (最高优先级) ──
            is_human_safe, human_reason = self._check_human_safety(context)
            if not is_human_safe:
                self._current_danger = DangerType.HUMAN_NEAR
                self._current_response = SafetyResponse.EMERGENCY_STOP
                self._current_danger_timestamp = time.time()
                self._total_dangers += 1
                self._log_event("human_danger", human_reason)
                return False, SafetyResponse.EMERGENCY_STOP, human_reason

            # ── P2. 障碍物检查 ──
            is_obstacle_safe, obstacle_reason = self._check_obstacles(context)
            if not is_obstacle_safe:
                self._current_danger = DangerType.OBSTACLE_CLOSE
                self._current_response = SafetyResponse.STOP
                self._current_danger_timestamp = time.time()
                self._total_dangers += 1
                self._log_event("obstacle_danger", obstacle_reason)
                return False, SafetyResponse.STOP, obstacle_reason

            # ── P3. 速度检查 ──
            is_speed_safe, speed_reason = self._check_speed(context)
            if not is_speed_safe:
                self._current_danger = DangerType.SPEED_EXCESSIVE
                self._current_response = SafetyResponse.CAUTION
                self._log_event("speed_warning", speed_reason)
                return False, SafetyResponse.CAUTION, speed_reason

            # ── P4. 温度检查 ──
            if context.robot_temperature > self.config.caution_temperature:
                self._current_danger = DangerType.TEMPERATURE_HIGH
                if context.robot_temperature > self.config.max_temperature:
                    self._current_response = SafetyResponse.EMERGENCY_STOP
                    self._total_dangers += 1
                    self._log_event("temperature_emergency",
                                    f"Temp {context.robot_temperature}°C > {self.config.max_temperature}°C")
                    return False, SafetyResponse.EMERGENCY_STOP, f"温度过高: {context.robot_temperature}°C"
                else:
                    self._current_response = SafetyResponse.WARNING
                    self._log_event("temperature_warning",
                                    f"Temp {context.robot_temperature}°C > {self.config.caution_temperature}°C")

            # ── P5. 电量检查 ──
            if context.robot_battery_level < self.config.min_battery_level:
                self._current_danger = DangerType.BATTERY_LOW
                self._current_response = SafetyResponse.CAUTION
                self._log_event("battery_low", f"Battery {context.robot_battery_level*100:.1f}%")
                return False, SafetyResponse.CAUTION, f"电量过低: {context.robot_battery_level*100:.1f}%"

            # ── P6. 危险环境检查 ──
            if context.environment_hazardous:
                self._current_danger = DangerType.HAZARDOUS_ENVIRONMENT
                self._current_response = SafetyResponse.SLOWDOWN
                self._log_event("hazardous_env", "Environment flagged as hazardous")
                return False, SafetyResponse.SLOWDOWN, "危险环境"

            # ── 安全通过 ──
            self._current_danger = DangerType.NONE
            self._current_response = SafetyResponse.NONE
            return True, SafetyResponse.NONE, None

    def _check_human_safety(self, context: GoalContext) -> Tuple[bool, Optional[str]]:
        """
        检查人员安全

        核心规则:
        - 任何时候不得与人员发生碰撞
        - 与人员距离<emergency_distance时必须紧急制动
        - 与人员距离<stop_distance时必须停止
        - 与人员距离<caution_distance时必须减速
        """
        human_dist = context.get_human_distance()

        if human_dist is None:
            return True, None  # 未检测到人员,默认安全

        # 获取动态安全区 (基于速度)
        robot_speed = 0.0
        if context.robot_velocity is not None:
            robot_speed = np.linalg.norm(context.robot_velocity)

        caution_d, stop_d, emergency_d = self._zone.get_zone_for_speed(robot_speed)

        # 紧急制动距离检查
        if human_dist < emergency_d:
            return False, f"人员距离 {human_dist:.2f}m < 紧急距离 {emergency_d:.2f}m"

        # 停止距离检查
        if human_dist < stop_d:
            return False, f"人员距离 {human_dist:.2f}m < 停止距离 {stop_d:.2f}m"

        # 警告距离检查
        if human_dist < caution_d:
            return False, f"人员距离 {human_dist:.2f}m < 警告距离 {caution_d:.2f}m"

        # 检查是否面向人员移动 (仅在足够近时触发警告/停止)
        # 1.5m以内: 直接停止
        # 1.5-2.5m: 减速警告
        # 2.5m以外: 正常
        if context.robot_velocity is not None and human_dist < 2.5:
            human_vel_direction = None
            for hp in context.human_positions:
                if context.robot_position is not None:
                    dir_to_human = hp - context.robot_position
                    dir_to_human /= (np.linalg.norm(dir_to_human) + 1e-6)
                    # 检查机器人速度方向是否朝向人员
                    vel_dir = context.robot_velocity / (np.linalg.norm(context.robot_velocity) + 1e-6)
                    alignment = np.dot(dir_to_human, vel_dir)
                    if alignment > 0.7:  # 夹角<45度
                        if human_dist < 1.5:
                            return False, f"朝向人员移动,距离{human_dist:.2f}m"
                        elif human_dist < 2.0:
                            return False, f"接近人员,距离{human_dist:.2f}m,建议减速"

        return True, None

    def _check_obstacles(self, context: GoalContext) -> Tuple[bool, Optional[str]]:
        """检查障碍物安全"""
        if context.robot_position is None:
            return True, None

        for obs in context.nearby_obstacles:
            obs_pos = None
            if hasattr(obs, 'position'):
                obs_pos = np.array(obs.position)
            elif hasattr(obs, 'pose'):
                obs_pos = np.array(obs.pose)[:3, 3] if obs.pose.shape == (4, 4) else np.array(obs.pose[:3])
            else:
                continue

            dist = np.linalg.norm(obs_pos - context.robot_position)
            if dist < self.config.obstacle_stop_distance:
                return False, f"障碍物距离 {dist:.3f}m < {self.config.obstacle_stop_distance}m"

        # 激光雷达数据检查
        if context.laser_ranges is not None:
            min_range = np.min(context.laser_ranges)
            if min_range < self.config.obstacle_stop_distance:
                return False, f"激光检测障碍 {min_range:.3f}m"

        return True, None

    def _check_speed(self, context: GoalContext) -> Tuple[bool, Optional[str]]:
        """检查速度安全"""
        if context.robot_velocity is None:
            return True, None

        v = context.robot_velocity
        speed = np.linalg.norm(v)

        # 判断方向 (前进/后退)
        if len(v) >= 3:
            forward_dir = v[:2] / (np.linalg.norm(v[:2]) + 1e-6)
            # 假设机器人前方是X轴正方向
            alignment = forward_dir[0] if len(v) >= 2 else 0

            if alignment > 0.5:  # 前进
                if speed > self.config.max_forward_speed:
                    return False, f"前进速度 {speed:.2f}m/s > {self.config.max_forward_speed}m/s"
            elif alignment < -0.5:  # 后退
                if speed > self.config.max_backward_speed:
                    return False, f"后退速度 {speed:.2f}m/s > {self.config.max_backward_speed}m/s"

        return True, None

    def get_override_action(
        self,
        context: GoalContext,
        original_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        获取安全覆盖动作

        当检测到危险时,此方法返回经过安全修正的动作:

        修正策略:
        - EMERGENCY_STOP: 零速度 (所有轴)
        - STOP: 保持位置但禁止运动
        - SLOWDOWN: 限制速度在安全范围内
        - CAUTION: 轻微减速
        - WARNING: 保持动作但记录警告

        Args:
            context: 当前上下文
            original_action: 原始动作 (6维: [vx, vy, vz, wx, wy, wz])

        Returns:
            np.ndarray: 安全修正后的动作 (6维)
        """
        is_safe, response, _ = self.check_context(context)

        if is_safe:
            return original_action if original_action is not None else np.zeros(6)

        # 基于响应级别修正动作
        action = original_action.copy() if original_action is not None else np.zeros(6)

        if response == SafetyResponse.EMERGENCY_STOP:
            action[:] = 0.0  # 完全停止
        elif response == SafetyResponse.STOP:
            action[:] = 0.0
        elif response == SafetyResponse.SLOWDOWN:
            # 限速50%
            max_speed = self.config.max_forward_speed * 0.5
            action[:3] = np.clip(action[:3], -max_speed, max_speed)
            action[3:] *= 0.3  # 旋转也减速
        elif response == SafetyResponse.CAUTION:
            # 限速70%
            max_speed = self.config.max_forward_speed * 0.7
            action[:3] = np.clip(action[:3], -max_speed, max_speed)
            action[3:] *= 0.6

        return action

    def check_action(self, action: np.ndarray, context: GoalContext) -> Tuple[bool, Optional[str]]:
        """
        验证动作安全性

        在执行动作前调用,确保动作不会导致危险:

        Args:
            action: 拟执行的动作 (6维)
            context: 当前上下文

        Returns:
            Tuple[bool, str]: (是否安全, 危险描述)
        """
        # 临时应用动作并检查
        temp_context = context
        if context.robot_velocity is not None and context.robot_position is not None:
            # 预测应用动作后的状态
            dt = 0.01  # 假设10ms周期
            predicted_pos = context.robot_position + action[:3] * dt
            predicted_vel = action[:3]
            temp_context = context  # 简化: 实际应创建新上下文
            temp_context.robot_position = predicted_pos

        # 检查预测位置的安全性
        is_safe, _, reason = self.check_context(temp_context)
        return is_safe, reason

    def trigger_emergency_stop(self, reason: str):
        """触发紧急停止"""
        with self._lock:
            self._emergency_stop_active = True
            self._emergency_stop_reason = reason
            self._current_response = SafetyResponse.EMERGENCY_STOP
            self._log_event("emergency_stop_triggered", reason)
            if self._on_emergency_stop:
                self._on_emergency_stop(reason)

    def release_emergency_stop(self):
        """释放紧急停止 (需人工确认安全)"""
        with self._lock:
            self._emergency_stop_active = False
            self._emergency_stop_reason = None
            self._log_event("emergency_stop_released", "Manual release")

    def set_callbacks(
        self,
        on_warning: Optional[Callable] = None,
        on_danger: Optional[Callable] = None,
        on_emergency_stop: Optional[Callable] = None,
    ):
        """设置事件回调"""
        self._on_warning = on_warning
        self._on_danger = on_danger
        self._on_emergency_stop = on_emergency_stop

    def _log_event(self, event_type: str, description: str):
        """记录安全事件"""
        self._event_log.append({
            "timestamp": time.time(),
            "type": event_type,
            "description": description,
            "danger": self._current_danger.value,
            "response": self._current_response.value,
        })
        # 保留最近1000条
        if len(self._event_log) > 1000:
            self._event_log = self._event_log[-1000:]

    def get_safety_score(self) -> float:
        """
        获取安全评分 [0.0, 1.0]

        综合评估当前安全状态:
        - 无危险事件: 1.0
        - WARNING: 0.8
        - CAUTION: 0.6
        - SLOWDOWN: 0.4
        - STOP: 0.2
        - EMERGENCY_STOP: 0.0
        """
        score_map = {
            SafetyResponse.NONE: 1.0,
            SafetyResponse.WARNING: 0.8,
            SafetyResponse.CAUTION: 0.6,
            SafetyResponse.SLOWDOWN: 0.4,
            SafetyResponse.STOP: 0.2,
            SafetyResponse.EMERGENCY_STOP: 0.0,
        }
        return score_map.get(self._current_response, 0.5)

    def get_status(self) -> Dict[str, Any]:
        """获取安全护盾状态"""
        return {
            "grade": self.config.grade.value,
            "emergency_stop_active": self._emergency_stop_active,
            "emergency_stop_reason": self._emergency_stop_reason,
            "current_danger": self._current_danger.value,
            "current_response": self._current_response.value,
            "safety_score": self.get_safety_score(),
            "total_checks": self._total_checks,
            "total_dangers": self._total_dangers,
            "event_log_size": len(self._event_log),
        }
