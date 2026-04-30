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
scene_coordination.py - 场景感知多机协同模块
SuperModel 超模态大模型具身智能系统

场景感知的多AGV协同:
- 场景感知任务分配
- 场景自适应编队控制
- 场景安全协同避障
- 跨场景经验迁移
- 多AGV场景协同决策
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from .scene_intelligence import SceneIntelligence, SceneContext, SceneType
    from ..control.swarm_control import SwarmController, FormationShape
else:
    SceneType = None

logger = logging.getLogger(__name__)

__all__ = [
    'AGVSceneRole',
    'SceneCoordinationConfig',
    'AGVSceneState',
    'SceneCoordinator',
    'MultiSceneSwarmController',
]


class AGVSceneRole(Enum):
    """AGV场景角色"""
    LEADER = "leader"           # 场景主导AGV
    FOLLOWER = "follower"       # 跟随AGV
    SCOUT = "scout"            # 侦察AGV (场景探索)
    GUARD = "guard"            # 安全警戒AGV
    COORDINATOR = "coordinator"  # 任务协调AGV


@dataclass
class SceneCoordinationConfig:
    """场景协同配置"""
    # 场景感知参数
    scene_reassessment_interval: float = 5.0   # seconds
    role_update_interval: float = 10.0          # seconds

    # 编队控制
    enable_scene_formation: bool = True
    formation_adaptive: bool = True

    # 安全协同
    enable_collaborative_safety: bool = True
    safety_broadcast_interval: float = 0.5      # seconds

    # 通信
    comm_range: float = 10.0                   # m
    enable_direct_comm: bool = True

    # 记忆集成
    enable_scene_memory: bool = True

    # AGV等级
    grade: str = "M"


@dataclass
class AGVSceneState:
    """AGV场景状态"""
    agv_id: str
    role: AGVSceneRole = AGVSceneRole.FOLLOWER
    scene_type: Optional[SceneType] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    task: str = ""
    battery_level: float = 1.0
    is_healthy: bool = True
    last_scene_update: float = 0.0
    nearby_agvs: Set[str] = field(default_factory=set)
    # 场景特定
    zone_id: str = ""
    confidence: float = 0.0
    safe_speed_limit: float = 1.5

    def update_position(self, pos: np.ndarray, vel: Optional[np.ndarray] = None):
        self.position = np.array(pos, dtype=float)
        if vel is not None:
            self.velocity = np.array(vel, dtype=float)
        self.last_scene_update = time.time()


# ============================================================
# 场景协同器
# ============================================================

class SceneCoordinator:
    """
    场景感知任务协调器

    负责任务分配、角色管理和场景状态维护
    """

    def __init__(
        self,
        my_agv_id: str,
        config: Optional[SceneCoordinationConfig] = None,
        scene_intelligence: Optional["SceneIntelligence"] = None,
    ):
        self._my_id = my_agv_id
        self._config = config or SceneCoordinationConfig()
        self._scene_intelligence = scene_intelligence
        self._all_states: Dict[str, AGVSceneState] = {}
        self._my_state = AGVSceneState(agv_id=my_agv_id)
        self._last_role_update = 0.0
        self._last_task_assign = 0.0
        self._logger = logging.getLogger(f"{__name__}.{my_agv_id}")

    def update_scene_context(self, context: "SceneContext"):
        """更新场景上下文"""
        self._my_state.scene_type = context.features.scene_type
        self._my_state.confidence = context.features.confidence
        self._my_state.safe_speed_limit = (
            self._scene_intelligence.get_adaptive_speed_limit(3.0)
            if self._scene_intelligence else 1.5
        )

    def register_agv(self, agv_id: str, state: Optional[AGVSceneState] = None):
        """注册AGV"""
        if agv_id not in self._all_states:
            self._all_states[agv_id] = state or AGVSceneState(agv_id=agv_id)
            self._logger.debug(f"注册AGV: {agv_id}")

    def update_agv_state(
        self,
        agv_id: str,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        role: Optional[AGVSceneRole] = None,
        task: Optional[str] = None,
        battery_level: Optional[float] = None,
        is_healthy: Optional[bool] = None,
    ):
        """更新AGV状态"""
        if agv_id not in self._all_states:
            self.register_agv(agv_id)

        state = self._all_states[agv_id]
        state.update_position(position, velocity)
        if role is not None:
            state.role = role
        if task is not None:
            state.task = task
        if battery_level is not None:
            state.battery_level = battery_level
        if is_healthy is not None:
            state.is_healthy = is_healthy

    def update_my_state(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        task: Optional[str] = None,
        battery_level: Optional[float] = None,
        is_healthy: bool = True,
    ):
        """更新本机状态"""
        self._my_state.update_position(position, velocity)
        if task is not None:
            self._my_state.task = task
        if battery_level is not None:
            self._my_state.battery_level = battery_level
        self._my_state.is_healthy = is_healthy

        # 更新所有状态中的本机记录
        self._all_states[self._my_id] = self._my_state

    def _update_roles_if_needed(self):
        """按需更新角色"""
        now = time.time()
        if now - self._last_role_update < self._config.role_update_interval:
            return

        self._last_role_update = now
        scene = self._my_state.scene_type

        if scene is None:
            return

        # 场景自适应角色分配
        if scene.value in ("warehouse", "factory"):
            # 仓库/工厂: 分配专门的侦察和协调角色
            self._assign_industrial_roles()
        elif scene.value == "hospital":
            # 医院: 安全警戒角色优先
            self._assign_hospital_roles()
        elif scene.value in ("restaurant", "office"):
            # 餐厅/办公室: 灵活角色
            self._assign_service_roles()

    def _assign_industrial_roles(self):
        """工业场景角色分配"""
        # 按ID排序决定角色
        sorted_ids = sorted(self._all_states.keys())
        my_idx = sorted_ids.index(self._my_id) if self._my_id in sorted_ids else -1

        if my_idx == 0:
            self._my_state.role = AGVSceneRole.COORDINATOR
        elif my_idx == 1:
            self._my_state.role = AGVSceneRole.SCOUT
        else:
            self._my_state.role = AGVSceneRole.FOLLOWER

    def _assign_hospital_roles(self):
        """医院场景角色分配 - 安全优先"""
        # 按电池和健康状况排序
        sorted_states = sorted(
            self._all_states.values(),
            key=lambda s: (s.is_healthy, s.battery_level),
            reverse=True,
        )

        if self._my_id == sorted_states[0].agv_id:
            self._my_state.role = AGVSceneRole.GUARD
        elif self._my_id == sorted_states[1].agv_id:
            self._my_state.role = AGVSceneRole.LEADER
        else:
            self._my_state.role = AGVSceneRole.FOLLOWER

    def _assign_service_roles(self):
        """服务场景角色分配"""
        # 低电池优先任务
        low_battery = [s for s in self._all_states.values() if s.battery_level < 0.3]
        if self._my_id in [s.agv_id for s in low_battery]:
            self._my_state.role = AGVSceneRole.FOLLOWER
        else:
            self._my_state.role = AGVSceneRole.LEADER

    def get_role(self) -> AGVSceneRole:
        """获取当前角色"""
        self._update_roles_if_needed()
        return self._my_state.role

    def get_leader_id(self) -> Optional[str]:
        """获取当前场景主导AGV"""
        for state in self._all_states.values():
            if state.role == AGVSceneRole.LEADER or state.role == AGVSceneRole.COORDINATOR:
                return state.agv_id
        return None

    def get_nearby_healthy_agvs(self, max_distance: float = 5.0) -> List[str]:
        """获取附近健康的AGV"""
        result = []
        for agv_id, state in self._all_states.items():
            if agv_id == self._my_id:
                continue
            if not state.is_healthy:
                continue
            dist = np.linalg.norm(state.position - self._my_state.position)
            if dist <= max_distance:
                result.append(agv_id)
        return result

    def get_all_states(self) -> Dict[str, AGVSceneState]:
        """获取所有AGV状态"""
        return dict(self._all_states)

    def get_my_state(self) -> AGVSceneState:
        """获取本机状态"""
        return self._my_state

    def get_scene_adaptive_formation_params(self) -> Dict[str, Any]:
        """获取场景自适应编队参数"""
        scene = self._my_state.scene_type
        if scene is None:
            return {'max_agents': 8, 'safe_distance': 0.5, 'max_speed': 1.5}

        from .scene_intelligence import SceneType
        params = {
            SceneType.WAREHOUSE: {'max_agents': 16, 'safe_distance': 0.5, 'max_speed': 2.0},
            SceneType.FACTORY: {'max_agents': 12, 'safe_distance': 0.6, 'max_speed': 1.0},
            SceneType.HOSPITAL: {'max_agents': 8, 'safe_distance': 0.8, 'max_speed': 0.8},
            SceneType.RESTAURANT: {'max_agents': 6, 'safe_distance': 0.4, 'max_speed': 1.0},
            SceneType.OFFICE: {'max_agents': 8, 'safe_distance': 0.4, 'max_speed': 1.2},
            SceneType.OUTDOOR: {'max_agents': 20, 'safe_distance': 0.3, 'max_speed': 3.0},
            SceneType.LABORATORY: {'max_agents': 4, 'safe_distance': 0.5, 'max_speed': 0.8},
            SceneType.HOME: {'max_agents': 3, 'safe_distance': 0.3, 'max_speed': 0.5},
            SceneType.UNKNOWN: {'max_agents': 8, 'safe_distance': 0.5, 'max_speed': 1.0},
        }
        return params.get(scene, params[SceneType.UNKNOWN])


# ============================================================
# 多场景蜂群控制器
# ============================================================

class MultiSceneSwarmController:
    """
    多场景蜂群控制器

    集成场景感知 + 蜂群控制:
    - 场景自适应编队
    - 场景感知避障
    - 协同任务规划
    """

    def __init__(
        self,
        my_agv_id: str,
        config: Optional[SceneCoordinationConfig] = None,
        scene_intelligence: Optional["SceneIntelligence"] = None,
    ):
        self._config = config or SceneCoordinationConfig()
        self._my_id = my_agv_id
        self._scene_intelligence = scene_intelligence
        self._coordinator = SceneCoordinator(
            my_agv_id=my_agv_id,
            config=config,
            scene_intelligence=scene_intelligence,
        )
        self._logger = logging.getLogger(f"{__name__}.{my_agv_id}")
        self._last_comm_broadcast = 0.0

    def update(
        self,
        my_position: np.ndarray,
        my_velocity: Optional[np.ndarray] = None,
        task: str = "",
        battery_level: float = 1.0,
        is_healthy: bool = True,
        received_states: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        更新蜂群控制器

        Args:
            my_position: 本机位置
            my_velocity: 本机速度
            task: 当前任务
            battery_level: 电池电量
            is_healthy: 健康状态
            received_states: 从通信接收的其他AGV状态

        Returns:
            协同控制指令
        """
        # 1. 更新本机状态
        self._coordinator.update_my_state(
            position=my_position,
            velocity=my_velocity,
            task=task,
            battery_level=battery_level,
            is_healthy=is_healthy,
        )

        # 2. 更新场景上下文
        if self._scene_intelligence:
            ctx = self._scene_intelligence.get_scene_context()
            self._coordinator.update_scene_context(ctx)

        # 3. 处理接收到的其他AGV状态
        if received_states:
            for agv_id, state_dict in received_states.items():
                self._coordinator.register_agv(agv_id)
                self._coordinator.update_agv_state(
                    agv_id=agv_id,
                    position=state_dict.get('position', np.zeros(3)),
                    velocity=state_dict.get('velocity'),
                    role=AGVSceneRole[state_dict.get('role', 'FOLLOWER')] if 'role' in state_dict else None,
                    task=state_dict.get('task'),
                    battery_level=state_dict.get('battery_level'),
                )

        # 4. 生成协同控制指令
        return self._generate_coordination_command()

    def _generate_coordination_command(self) -> Dict[str, Any]:
        """生成协同控制指令"""
        role = self._coordinator.get_role()
        my_state = self._coordinator.get_my_state()
        formation_params = self._coordinator.get_scene_adaptive_formation_params()
        nearby = self._coordinator.get_nearby_healthy_agvs(
            max_distance=self._config.comm_range
        )

        command = {
            'my_id': self._my_id,
            'role': role.value,
            'scene_type': my_state.scene_type.value if my_state.scene_type else 'unknown',
            'safe_speed_limit': my_state.safe_speed_limit,
            'formation_params': formation_params,
            'nearby_agvs': nearby,
            'leader_id': self._coordinator.get_leader_id(),
            'should_form_formation': len(nearby) >= 2 and self._config.enable_scene_formation,
            'avoidance_priority': role in (AGVSceneRole.LEADER, AGVSceneRole.GUARD),
            'timestamp': time.time(),
        }

        # 角色特定指令
        if role == AGVSceneRole.LEADER:
            command['formation_role'] = 'leader'
            command['broadcast_map_update'] = True
        elif role == AGVSceneRole.SCOUT:
            command['formation_role'] = 'scout'
            command['explore_new_areas'] = True
        elif role == AGVSceneRole.GUARD:
            command['formation_role'] = 'guard'
            command['priority_safety_zone'] = True
        elif role == AGVSceneRole.COORDINATOR:
            command['formation_role'] = 'coordinator'
            command['coordinate_tasks'] = True
        else:
            command['formation_role'] = 'follower'
            command['follow_leader'] = True

        return command

    def get_coordination_state(self) -> Dict[str, Any]:
        """获取协同状态"""
        my_state = self._coordinator.get_my_state()
        return {
            'my_id': self._my_id,
            'role': self._coordinator.get_role().value,
            'scene_type': my_state.scene_type.value if my_state.scene_type else 'unknown',
            'nearby_count': len(self._coordinator.get_nearby_healthy_agvs()),
            'total_known_agvs': len(self._coordinator.get_all_states()),
            'safe_speed': my_state.safe_speed_limit,
            'is_healthy': my_state.is_healthy,
            'battery': my_state.battery_level,
        }

    def get_coordinator(self) -> SceneCoordinator:
        """获取协调器"""
        return self._coordinator
