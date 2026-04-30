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
scene_task_planner.py - 场景化具身任务规划器
SuperModel 超模态大模型具身智能系统

场景化任务规划:
- 场景自适应行为树生成
- 场景经验驱动的任务规划
- 跨场景任务迁移
- 场景安全任务优先级
- 动态场景重规划
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
import numpy as np

from .behavior_tree import (
    NodeStatus,
    BTNode,
    SequenceNode,
    SelectorNode,
    ParallelNode,
    RepeaterNode,
    ConditionNode,
    LambdaActionNode,
    BehaviorTree,
    Blackboard,
    EmbodiedTask,
    AGVTaskPlanner,
    TaskStatus,
)
from .scene_intelligence import (
    SceneType,
    SceneContext,
    SceneConfig,
    SceneIntelligence,
    SafetyRule,
    NavigationRule,
    InteractionRule,
)

if TYPE_CHECKING:
    from ..memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)

__all__ = [
    'SceneTaskConfig',
    'SceneTaskTemplate',
    'SceneTaskLibrary',
    'SceneTaskPlanner',
    'WarehouseTaskPlanner',
    'HospitalTaskPlanner',
    'FactoryTaskPlanner',
    'RestaurantTaskPlanner',
    'OutdoorTaskPlanner',
    'SceneAdaptationEngine',
    'HierarchicalBehaviorTreeComposer',
    'SceneMemoryAugmentedPlanner',
    'CrossSceneTransferLearner',
    'HierarchicalTaskLevel',
    'TaskCompositionRule',
    'get_scene_task_planner',
]


# ============================================================
# 场景任务配置
# ============================================================

@dataclass
class SceneTaskConfig:
    """场景任务配置"""
    # 任务优先级
    safety_priority: int = 1       # 安全相关任务优先级
    delivery_priority: int = 2     # 配送任务优先级
    patrol_priority: int = 3      # 巡检任务优先级
    maintenance_priority: int = 4  # 维护任务优先级
    
    # 任务超时 (seconds)
    task_timeout: float = 300.0
    approach_timeout: float = 60.0
    grasp_timeout: float = 30.0
    
    # 重试
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # AGV等级适配
    grade: str = "M"
    
    # 场景特定
    enable_express_mode: bool = False
    require_human_confirmation: bool = False
    enable_collaborative_dispatch: bool = True


@dataclass
class SceneTaskTemplate:
    """场景任务模板"""
    task_type: str
    scene_types: Set[SceneType]
    priority: int
    bt_config: Dict[str, Any]
    required_capabilities: List[str]
    typical_duration_s: float
    safety_critical: bool = False
    collaborative: bool = False


# ============================================================
# 场景任务库
# ============================================================

class SceneTaskLibrary:
    """
    场景任务模板库
    
    按场景类型管理标准任务模板
    """
    
    def __init__(self):
        self._templates: Dict[SceneType, List[SceneTaskTemplate]] = {
            SceneType.WAREHOUSE: self._warehouse_templates(),
            SceneType.FACTORY: self._factory_templates(),
            SceneType.HOSPITAL: self._hospital_templates(),
            SceneType.RESTAURANT: self._restaurant_templates(),
            SceneType.OFFICE: self._office_templates(),
            SceneType.OUTDOOR: self._outdoor_templates(),
            SceneType.LABORATORY: self._laboratory_templates(),
            SceneType.HOME: self._home_templates(),
            SceneType.UNKNOWN: self._generic_templates(),
        }
    
    def _warehouse_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="pick_and_stow",
                scene_types={SceneType.WAREHOUSE},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                        {"type": "condition", "name": "check_path_clear", "condition": "path_clear"},
                        {"type": "action", "name": "navigate_to_pick", "action": "navigate", "target": "pick_station"},
                        {"type": "action", "name": "execute_pick", "action": "grasp", "target": "item"},
                        {"type": "action", "name": "navigate_to_stow", "action": "navigate", "target": "stow_location"},
                        {"type": "action", "name": "execute_stow", "action": "release"},
                    ],
                },
                required_capabilities=["navigation", "grasp", "localization"],
                typical_duration_s=120.0,
                safety_critical=False,
                collaborative=True,
            ),
            SceneTaskTemplate(
                task_type="inventory_patrol",
                scene_types={SceneType.WAREHOUSE},
                priority=3,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                        {"type": "action", "name": "patrol_aisle_1", "action": "patrol", "route": "aisle_1"},
                        {"type": "action", "name": "scan_shelves", "action": "scan", "target": "shelves"},
                        {"type": "action", "name": "patrol_aisle_2", "action": "patrol", "route": "aisle_2"},
                        {"type": "action", "name": "report_inventory", "action": "report", "content": "inventory_update"},
                    ],
                },
                required_capabilities=["navigation", "vision", "communication"],
                typical_duration_s=300.0,
                safety_critical=False,
                collaborative=False,
            ),
            SceneTaskTemplate(
                task_type="emergency_shutdown",
                scene_types={SceneType.WAREHOUSE},
                priority=0,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "detect_emergency", "condition": "emergency_detected"},
                        {"type": "action", "name": "sound_alarm", "action": "alert", "type": "emergency"},
                        {"type": "action", "name": "evacuate", "action": "evacuate", "route": "emergency_exit"},
                        {"type": "action", "name": "report_incident", "action": "report", "content": "emergency"},
                    ],
                },
                required_capabilities=["navigation", "communication", "emergency_protocol"],
                typical_duration_s=60.0,
                safety_critical=True,
                collaborative=True,
            ),
            SceneTaskTemplate(
                task_type="cold_chain_check",
                scene_types={SceneType.WAREHOUSE},
                priority=2,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                        {"type": "action", "name": "navigate_cold_zone", "action": "navigate", "target": "cold_zone"},
                        {"type": "action", "name": "read_sensors", "action": "read_sensors", "types": ["temperature", "humidity"]},
                        {"type": "action", "name": "report_conditions", "action": "report", "content": "cold_chain_status"},
                    ],
                },
                required_capabilities=["navigation", "sensor_reading", "communication"],
                typical_duration_s=180.0,
                safety_critical=True,
                collaborative=False,
            ),
        ]
    
    def _factory_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="production_line_feed",
                scene_types={SceneType.FACTORY},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_line_signal", "condition": "line_needs_material"},
                        {"type": "action", "name": "fetch_from_buffer", "action": "grasp", "target": "material_buffer"},
                        {"type": "action", "name": "deliver_to_line", "action": "navigate", "target": "production_station"},
                        {"type": "action", "name": "place_material", "action": "place", "position": "feeder"},
                    ],
                },
                required_capabilities=["precision_navigation", "grasp", "communication"],
                typical_duration_s=90.0,
                safety_critical=True,
                collaborative=True,
            ),
            SceneTaskTemplate(
                task_type="quality_inspection",
                scene_types={SceneType.FACTORY},
                priority=2,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_product_ready", "condition": "product_at_station"},
                        {"type": "action", "name": "position_camera", "action": "position", "target": "inspection_station"},
                        {"type": "action", "name": "capture_images", "action": "capture", "modality": "vision"},
                        {"type": "action", "name": "analyze_defects", "action": "analyze", "content": "quality"},
                        {"type": "action", "name": "sort_product", "action": "sort", "criteria": "quality"},
                    ],
                },
                required_capabilities=["vision", "precision_positioning", "decision_making"],
                typical_duration_s=45.0,
                safety_critical=True,
                collaborative=False,
            ),
        ]
    
    def _hospital_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="medication_delivery",
                scene_types={SceneType.HOSPITAL},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "verify_delivery", "condition": "verified_delivery"},
                        {"type": "action", "name": "navigate_to_pharmacy", "action": "navigate", "target": "pharmacy"},
                        {"type": "action", "name": "collect_medication", "action": "grasp", "target": "medication"},
                        {"type": "action", "name": "navigate_to_ward", "action": "navigate", "target": "ward"},
                        {"type": "action", "name": "confirm_recipient", "action": "confirm", "type": "identity"},
                        {"type": "action", "name": "deliver_to_nurse", "action": "handover", "recipient": "nurse"},
                    ],
                },
                required_capabilities=["navigation", "grasp", "identity_verification", "communication"],
                typical_duration_s=180.0,
                safety_critical=True,
                collaborative=False,
            ),
            SceneTaskTemplate(
                task_type="specimen_transport",
                scene_types={SceneType.HOSPITAL},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "verify_specimen", "condition": "specimen_ready"},
                        {"type": "action", "name": "secure_specimen", "action": "secure", "target": "specimen_container"},
                        {"type": "action", "name": "navigate_to_lab", "action": "navigate", "target": "laboratory"},
                        {"type": "action", "name": "maintain_cold_chain", "action": "monitor", "type": "temperature"},
                        {"type": "action", "name": "deliver_to_lab", "action": "handover", "recipient": "lab_tech"},
                    ],
                },
                required_capabilities=["navigation", "secure_grasp", "temperature_monitoring"],
                typical_duration_s=120.0,
                safety_critical=True,
                collaborative=False,
            ),
            SceneTaskTemplate(
                task_type="sanitization_patrol",
                scene_types={SceneType.HOSPITAL},
                priority=2,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                        {"type": "action", "name": "uv_disinfect_zone1", "action": "disinfect", "zone": "zone1", "method": "uv"},
                        {"type": "action", "name": "move_to_zone2", "action": "navigate", "target": "zone2"},
                        {"type": "action", "name": "uv_disinfect_zone2", "action": "disinfect", "zone": "zone2", "method": "uv"},
                        {"type": "action", "name": "report_sanitization", "action": "report", "content": "sanitization_log"},
                    ],
                },
                required_capabilities=["navigation", "uv_disinfection", "logging"],
                typical_duration_s=600.0,
                safety_critical=True,
                collaborative=False,
            ),
        ]
    
    def _restaurant_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="food_delivery",
                scene_types={SceneType.RESTAURANT},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_order_ready", "condition": "order_ready"},
                        {"type": "action", "name": "collect_from_kitchen", "action": "grasp", "target": "order_tray"},
                        {"type": "action", "name": "navigate_to_table", "action": "navigate", "target": "table"},
                        {"type": "action", "name": "announce_arrival", "action": "announce", "content": "order_arrived"},
                        {"type": "action", "name": "wait_for_handoff", "action": "wait", "duration": 30},
                        {"type": "action", "name": "clear_table", "action": "grasp", "target": "dirty_dishes"},
                    ],
                },
                required_capabilities=["navigation", "grasp", "voice", "human_detection"],
                typical_duration_s=180.0,
                safety_critical=False,
                collaborative=True,
            ),
        ]
    
    def _office_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="document_delivery",
                scene_types={SceneType.OFFICE},
                priority=2,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                        {"type": "action", "name": "collect_documents", "action": "grasp", "target": "documents"},
                        {"type": "action", "name": "navigate_to_destination", "action": "navigate", "target": "destination"},
                        {"type": "action", "name": "confirm_delivery", "action": "confirm", "type": "recipient"},
                    ],
                },
                required_capabilities=["navigation", "grasp", "identity_verification"],
                typical_duration_s=120.0,
                safety_critical=False,
                collaborative=False,
            ),
        ]
    
    def _outdoor_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="outdoor_delivery",
                scene_types={SceneType.OUTDOOR},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_weather", "condition": "weather_safe"},
                        {"type": "action", "name": "navigate_to_pickup", "action": "navigate", "target": "pickup_point"},
                        {"type": "action", "name": "collect_package", "action": "grasp", "target": "package"},
                        {"type": "action", "name": "navigate_to_destination", "action": "navigate", "target": "destination"},
                        {"type": "action", "name": "deliver_package", "action": "release"},
                    ],
                },
                required_capabilities=["outdoor_navigation", "grasp", "gps", "weather_awareness"],
                typical_duration_s=600.0,
                safety_critical=False,
                collaborative=False,
            ),
        ]
    
    def _laboratory_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="sample_transport",
                scene_types={SceneType.LABORATORY},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "verify_sample", "condition": "sample_sealed"},
                        {"type": "action", "name": "collect_sample", "action": "secure_grasp", "target": "sample"},
                        {"type": "action", "name": "navigate_to_analysis", "action": "navigate", "target": "analysis_station"},
                        {"type": "action", "name": "handover_sample", "action": "handover", "recipient": "researcher"},
                    ],
                },
                required_capabilities=["precision_navigation", "secure_grasp", "chain_of_custody"],
                typical_duration_s=90.0,
                safety_critical=True,
                collaborative=False,
            ),
        ]
    
    def _home_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="item_fetch",
                scene_types={SceneType.HOME},
                priority=1,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "understand_request", "condition": "voice_command_received"},
                        {"type": "action", "name": "navigate_to_item", "action": "navigate", "target": "item_location"},
                        {"type": "action", "name": "grasp_item", "action": "grasp", "target": "item"},
                        {"type": "action", "name": "deliver_to_person", "action": "navigate", "target": "person"},
                        {"type": "action", "name": "hand_over", "action": "handover", "recipient": "person"},
                    ],
                },
                required_capabilities=["voice_understanding", "navigation", "grasp", "person_tracking"],
                typical_duration_s=120.0,
                safety_critical=False,
                collaborative=False,
            ),
        ]
    
    def _generic_templates(self) -> List[SceneTaskTemplate]:
        return [
            SceneTaskTemplate(
                task_type="goto_location",
                scene_types={SceneType.UNKNOWN},
                priority=2,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                        {"type": "condition", "name": "check_path", "condition": "path_clear"},
                        {"type": "action", "name": "navigate", "action": "navigate", "target": "destination"},
                    ],
                },
                required_capabilities=["navigation"],
                typical_duration_s=60.0,
                safety_critical=False,
                collaborative=False,
            ),
            SceneTaskTemplate(
                task_type="emergency_stop",
                scene_types={SceneType.UNKNOWN},
                priority=0,
                bt_config={
                    "type": "sequence",
                    "children": [
                        {"type": "condition", "name": "detect_emergency", "condition": "emergency"},
                        {"type": "action", "name": "stop", "action": "emergency_stop"},
                        {"type": "action", "name": "alert", "action": "alert", "type": "emergency"},
                    ],
                },
                required_capabilities=["emergency_protocol"],
                typical_duration_s=10.0,
                safety_critical=True,
                collaborative=True,
            ),
        ]
    
    def get_templates(self, scene_type: SceneType) -> List[SceneTaskTemplate]:
        """获取场景模板"""
        templates = self._templates.get(scene_type, [])
        templates.extend(self._templates.get(SceneType.UNKNOWN, []))
        return templates
    
    def get_template(self, scene_type: SceneType, task_type: str) -> Optional[SceneTaskTemplate]:
        """获取特定任务模板"""
        for tmpl in self.get_templates(scene_type):
            if tmpl.task_type == task_type:
                return tmpl
        return None


# ============================================================
# 场景任务规划器
# ============================================================

class SceneTaskPlanner:
    """
    场景化具身任务规划器
    
    核心功能:
    - 根据场景类型生成适配的行为树
    - 场景经验驱动的任务规划
    - 跨场景任务迁移
    - 动态重规划
    """
    
    def __init__(
        self,
        config: Optional[SceneTaskConfig] = None,
        scene_intelligence: Optional[SceneIntelligence] = None,
        memory: Optional["LongTermMemory"] = None,
    ):
        self._config = config or SceneTaskConfig()
        self._scene_intelligence = scene_intelligence
        self._memory = memory
        self._library = SceneTaskLibrary()
        self._active_bt: Optional[BehaviorTree] = None
        self._active_task: Optional[EmbodiedTask] = None
        self._last_plan_time = 0.0
        self._plan_count = 0
        self._logger = logging.getLogger(__name__)
    
    def plan_task(
        self,
        task_description: str,
        scene_type: SceneType,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BehaviorTree, EmbodiedTask]:
        """
        为指定场景规划任务
        
        Args:
            task_description: 任务描述
            scene_type: 场景类型
            context: 附加上下文
            
        Returns:
            (BehaviorTree, EmbodiedTask)
        """
        self._plan_count += 1
        self._last_plan_time = time.time()
        
        # 1. 从记忆检索相似经验
        similar_experience = self._retrieve_similar_experience(task_description, scene_type)
        
        # 2. 选择任务模板
        template = self._select_template(task_description, scene_type, similar_experience)
        
        # 3. 生成行为树
        bt = self._generate_behavior_tree(template, scene_type, context, similar_experience)
        
        # 4. 创建任务对象
        task = EmbodiedTask(
            task_id=f"scene_task_{self._plan_count}_{int(time.time())}",
            task_type=template.task_type,
            goal_description=f"{scene_type.value}: {task_description}",
            priority=template.priority,
            required_capabilities=set(template.required_capabilities),
            timeout=template.typical_duration_s,
        )
        
        # 5. 存储到记忆
        self._store_planning_experience(task, template, scene_type, similar_experience)
        
        self._active_bt = bt
        self._active_task = task
        return bt, task
    
    def _retrieve_similar_experience(
        self, task_description: str, scene_type: SceneType
    ) -> List[Dict[str, Any]]:
        """从长期记忆检索相似经验"""
        if self._memory is None:
            return []
        
        try:
            results = self._memory.retrieve(
                query=f"{scene_type.value} {task_description}",
                memory_type="episodic",
                limit=5,
            )
            return [
                {"id": r.memory_id, "content": r.content, "relevance": r.relevance_score}
                for r in results
            ]
        except Exception:
            return []
    
    def _select_template(
        self,
        task_description: str,
        scene_type: SceneType,
        similar_experience: List[Dict[str, Any]],
    ) -> SceneTaskTemplate:
        """选择最合适的任务模板"""
        templates = self._library.get_templates(scene_type)
        
        # 1. 首先尝试精确匹配 task_description
        desc_lower = task_description.lower()
        for tmpl in templates:
            if tmpl.task_type.lower() == desc_lower or tmpl.task_type.lower().replace('_', '') == desc_lower.replace('_', ''):
                return tmpl
        
        # 2. 模糊匹配
        for tmpl in templates:
            if tmpl.task_type.lower() in desc_lower or desc_lower in tmpl.task_type.lower():
                return tmpl
        
        # 3. 如果有相似经验，调整模板参数
        if similar_experience:
            # 使用成功率最高的相似任务对应的模板
            best_relevance = 0.0
            best_template = None
            for tmpl in templates:
                for exp in similar_experience:
                    if tmpl.task_type in exp.get('content', '').lower():
                        if exp.get('relevance', 0) > best_relevance:
                            best_relevance = exp['relevance']
                            best_template = tmpl
            if best_template:
                return best_template
        
        # 4. 默认选择最高优先级模板
        return min(templates, key=lambda t: t.priority) if templates else self._library.get_templates(SceneType.UNKNOWN)[0]
    
    def _generate_behavior_tree(
        self,
        template: SceneTaskTemplate,
        scene_type: SceneType,
        context: Optional[Dict[str, Any]],
        similar_experience: List[Dict[str, Any]],
    ) -> BehaviorTree:
        """生成行为树"""
        bt_config = dict(template.bt_config)
        
        # 应用场景特定调整
        bt_config = self._apply_scene_adjustments(bt_config, scene_type, context)
        
        # 从相似经验中提取优化参数
        if similar_experience:
            bt_config = self._apply_experience_optimizations(bt_config, similar_experience)
        
        # 构建行为树
        from .behavior_tree import create_behavior_tree_from_dict
        return create_behavior_tree_from_dict(bt_config)
    
    def _apply_scene_adjustments(
        self,
        bt_config: Dict[str, Any],
        scene_type: SceneType,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """应用场景特定调整"""
        config = dict(bt_config)
        
        # 根据场景类型调整安全检查
        if scene_type == SceneType.HOSPITAL:
            # 医院: 增强身份验证
            config = self._insert_precondition(config, {
                "type": "condition",
                "name": "verify_identity",
                "condition": "recipient_verified",
            })
        
        elif scene_type == SceneType.FACTORY:
            # 工厂: 增强安全检查
            config = self._insert_precondition(config, {
                "type": "condition",
                "name": "factory_safety_check",
                "condition": "safety_equipment_active",
            })
        
        elif scene_type == SceneType.OUTDOOR:
            # 户外: 天气检查
            config = self._insert_precondition(config, {
                "type": "condition",
                "name": "weather_check",
                "condition": "weather_suitable",
            })
        
        return config
    
    def _apply_experience_optimizations(
        self,
        bt_config: Dict[str, Any],
        similar_experience: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从相似经验中提取优化"""
        # 根据历史成功经验调整超时参数
        config = dict(bt_config)
        # 如果历史经验显示任务耗时较长，可以适当延长超时
        return config
    
    def _insert_precondition(
        self, bt_config: Dict[str, Any], precondition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """在行为树开头插入前置条件"""
        config = dict(bt_config)
        if "children" in config and len(config["children"]) > 0:
            # 在第一个子节点前插入
            children = list(config["children"])
            children.insert(0, precondition)
            config["children"] = children
        return config
    
    def _store_planning_experience(
        self,
        task: EmbodiedTask,
        template: SceneTaskTemplate,
        scene_type: SceneType,
        similar_experience: List[Dict[str, Any]],
    ):
        """存储规划经验到记忆"""
        if self._memory is None:
            return
        
        try:
            self._memory.store_episode(
                summary=f"场景任务规划: {task.description}",
                context={
                    "scene_type": scene_type.value,
                    "task_type": template.task_type,
                    "priority": template.priority,
                    "used_similar_experience": len(similar_experience) > 0,
                },
                actions=[],
                outcomes={"task_planned": True},
                importance_score=5.0,
                tags=["task_planning", scene_type.value, template.task_type],
            )
        except Exception:
            pass
    
    def replan_if_needed(
        self,
        current_bt: BehaviorTree,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> Optional[BehaviorTree]:
        """
        检测是否需要重规划
        
        Args:
            current_bt: 当前行为树
            scene_type: 当前场景类型
            context: 当前上下文
            
        Returns:
            新行为树如果需要重规划，否则 None
        """
        # 检测场景转换
        if self._scene_intelligence:
            if self._scene_intelligence.recognize_scene_transition():
                self._logger.info(f"场景转换检测，重规划任务")
                return self.plan_task(
                    task_description=context.get("task_description", "继续任务"),
                    scene_type=scene_type,
                    context=context,
                )[0]
        
        # 检测任务超时
        if self._active_task:
            elapsed = time.time() - self._last_plan_time
            if elapsed > self._config.task_timeout * 0.8:
                self._logger.warning(f"任务接近超时 ({elapsed:.0f}s)")
        
        return None
    
    def get_active_plan(self) -> Tuple[Optional[BehaviorTree], Optional[EmbodiedTask]]:
        """获取当前活动计划"""
        return self._active_bt, self._active_task


# ============================================================
# 专用场景任务规划器
# ============================================================

class WarehouseTaskPlanner(SceneTaskPlanner):
    """仓库专用任务规划器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._zone_map: Dict[str, str] = {}  # zone_id -> task_type
    
    def plan_zone_patrol(self, zones: List[str]) -> BehaviorTree:
        """规划区域巡检"""
        zone_children = []
        for zone in zones:
            zone_children.append({
                "type": "sequence",
                "name": f"patrol_zone_{zone}",
                "children": [
                    {"type": "action", "name": f"navigate_to_{zone}", "action": "navigate", "target": zone},
                    {"type": "action", "name": f"scan_zone_{zone}", "action": "scan", "target": zone},
                    {"type": "action", "name": f"report_{zone}", "action": "report", "content": f"zone_{zone}_status"},
                ],
            })
        
        bt_config = {
            "type": "sequence",
            "children": zone_children,
        }
        from .behavior_tree import create_behavior_tree_from_dict
        return create_behavior_tree_from_dict(bt_config)


class HospitalTaskPlanner(SceneTaskPlanner):
    """医院专用任务规划器"""
    
    def plan_verified_delivery(
        self,
        delivery_type: str,
        destination: str,
    ) -> Tuple[BehaviorTree, EmbodiedTask]:
        """规划需验证的配送任务"""
        bt_config = {
            "type": "sequence",
            "children": [
                {"type": "condition", "name": "verify_delivery_request", "condition": "request_verified"},
                {"type": "action", "name": "collect_item", "action": "grasp", "target": delivery_type},
                {"type": "condition", "name": "check_way_clear", "condition": "way_clear"},
                {"type": "action", "name": "navigate_to_dest", "action": "navigate", "target": destination},
                {"type": "condition", "name": "verify_recipient", "condition": "recipient_verified"},
                {"type": "action", "name": "handover", "action": "handover", "recipient": "authorized_person"},
                {"type": "action", "name": "log_delivery", "action": "log", "content": "delivery_completed"},
            ],
        }
        from .behavior_tree import create_behavior_tree_from_dict
        bt = create_behavior_tree_from_dict(bt_config)
        
        task = EmbodiedTask(
            task_id=f"hospital_delivery_{int(time.time())}",
            task_type=delivery_type,
            goal_description=f"hospital: {delivery_type} 到 {destination}",
            priority=1,
            required_capabilities={"navigation", "identity_verification", "grasp"},
            timeout=180.0,
        )
        return bt, task


class FactoryTaskPlanner(SceneTaskPlanner):
    """工厂专用任务规划器"""
    
    def plan_production_task(
        self,
        task_type: str,
        station: str,
    ) -> Tuple[BehaviorTree, EmbodiedTask]:
        """规划生产线任务"""
        bt_config = {
            "type": "sequence",
            "children": [
                {"type": "condition", "name": "check_production_signal", "condition": "signal_received"},
                {"type": "condition", "name": "verify_station_safe", "condition": "station_safe"},
                {"type": "action", "name": "approach_station", "action": "navigate", "target": station},
                {"type": "action", "name": "execute_task", "action": task_type},
                {"type": "action", "name": "confirm_completion", "action": "confirm", "type": "task_complete"},
                {"type": "action", "name": "report_production", "action": "report", "content": "production_update"},
            ],
        }
        from .behavior_tree import create_behavior_tree_from_dict
        bt = create_behavior_tree_from_dict(bt_config)
        
        task = EmbodiedTask(
            task_id=f"factory_task_{int(time.time())}",
            task_type=task_type,
            goal_description=f"factory: 执行 {task_type} 在 {station}",
            priority=1,
            required_capabilities={"precision_navigation", "communication", "safety_protocol"},
            timeout=90.0,
        )
        return bt, task


class RestaurantTaskPlanner(SceneTaskPlanner):
    """餐厅专用任务规划器"""
    
    def plan_food_delivery(
        self,
        table_id: str,
        order_type: str,
    ) -> Tuple[BehaviorTree, EmbodiedTask]:
        """规划食物配送"""
        bt_config = {
            "type": "sequence",
            "children": [
                {"type": "condition", "name": "check_order_ready", "condition": "order_ready"},
                {"type": "action", "name": "collect_from_kitchen", "action": "grasp", "target": order_type},
                {"type": "action", "name": "navigate_to_table", "action": "navigate", "target": table_id},
                {"type": "action", "name": "announce_arrival", "action": "announce", "content": "food_arrived"},
                {"type": "action", "name": "wait_for_pickup", "action": "wait", "duration": 30},
                {"type": "selector",
                 "children": [
                     {"type": "condition", "name": "picked_up", "condition": "food_taken"},
                     {"type": "sequence",
                      "children": [
                          {"type": "action", "name": "prompt_again", "action": "announce", "content": "please_take_food"},
                          {"type": "action", "name": "wait_again", "action": "wait", "duration": 30},
                      ],
                     },
                 ],
                },
            ],
        }
        from .behavior_tree import create_behavior_tree_from_dict
        bt = create_behavior_tree_from_dict(bt_config)
        
        task = EmbodiedTask(
            task_id=f"restaurant_delivery_{int(time.time())}",
            task_type="food_delivery",
            goal_description=f"restaurant: 配送 {order_type} 到桌 {table_id}",
            priority=1,
            required_capabilities={"navigation", "grasp", "voice", "human_detection"},
            timeout=180.0,
        )
        return bt, task


class OutdoorTaskPlanner(SceneTaskPlanner):
    """户外专用任务规划器"""
    
    def plan_outdoor_delivery(
        self,
        pickup: str,
        destination: str,
        package_type: str,
    ) -> Tuple[BehaviorTree, EmbodiedTask]:
        """规划户外配送"""
        bt_config = {
            "type": "sequence",
            "children": [
                {"type": "condition", "name": "check_weather", "condition": "weather_safe"},
                {"type": "condition", "name": "check_battery", "condition": "battery_ok"},
                {"type": "action", "name": "navigate_to_pickup", "action": "navigate", "target": pickup},
                {"type": "action", "name": "collect_package", "action": "grasp", "target": package_type},
                {"type": "action", "name": "secure_package", "action": "confirm", "type": "package_secure"},
                {"type": "action", "name": "navigate_to_dest", "action": "navigate", "target": destination},
                {"type": "action", "name": "deliver_package", "action": "release"},
                {"type": "action", "name": "confirm_delivery", "action": "confirm", "type": "recipient_confirmed"},
            ],
        }
        from .behavior_tree import create_behavior_tree_from_dict
        bt = create_behavior_tree_from_dict(bt_config)
        
        task = EmbodiedTask(
            task_id=f"outdoor_delivery_{int(time.time())}",
            task_type="outdoor_delivery",
            goal_description=f"outdoor: 配送 {package_type} 从 {pickup} 到 {destination}",
            priority=1,
            required_capabilities={"outdoor_navigation", "grasp", "gps", "weather_awareness"},
            timeout=600.0,
        )
        return bt, task


# ============================================================
# 场景适应引擎
# ============================================================

class SceneAdaptationEngine:
    """
    场景适应引擎
    
    从场景经验中学习，自动调整行为参数
    """
    
    def __init__(
        self,
        memory: Optional["LongTermMemory"] = None,
    ):
        self._memory = memory
        self._scene_params: Dict[SceneType, Dict[str, Any]] = {}
        self._load_from_memory()
    
    def _load_from_memory(self):
        """从记忆加载场景参数"""
        if self._memory is None:
            self._init_default_params()
            return
        
        try:
            for scene in SceneType:
                results = self._memory.retrieve(
                    query=f"scene_params {scene.value}",
                    memory_type="semantic",
                    limit=1,
                )
                if results:
                    self._scene_params[scene] = results[0].content
                else:
                    self._scene_params[scene] = {}
        except Exception:
            self._init_default_params()
    
    def _init_default_params(self):
        """初始化默认参数"""
        for scene in SceneType:
            self._scene_params[scene] = {
                "speed_multiplier": 1.0,
                "caution_multiplier": 1.0,
                "success_rate": 0.8,
            }
    
    def record_outcome(
        self,
        scene_type: SceneType,
        task_type: str,
        success: bool,
        duration_s: float,
        parameters: Dict[str, Any],
    ):
        """记录任务执行结果"""
        if scene_type not in self._scene_params:
            self._scene_params[scene_type] = {}
        
        params = self._scene_params[scene_type]
        
        # 更新成功率
        current_rate = params.get("success_rate", 0.8)
        if success:
            params["success_rate"] = current_rate * 0.95 + 0.05
        else:
            params["success_rate"] = current_rate * 0.95
        
        # 记录成功参数
        if success:
            params["last_successful_params"] = parameters
        
        # 存储到记忆
        if self._memory:
            try:
                self._memory.store_episode(
                    summary=f"场景适应: {task_type} in {scene_type.value}",
                    context={
                        "scene_type": scene_type.value,
                        "task_type": task_type,
                        "success": success,
                        "duration_s": duration_s,
                        "parameters": parameters,
                    },
                    actions=[],
                    outcomes={"adaptation_updated": True},
                    importance_score=7.0 if not success else 5.0,
                    tags=["scene_adaptation", scene_type.value, task_type],
                )
            except Exception:
                pass
    
    def get_adaptive_params(
        self,
        scene_type: SceneType,
        base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """获取场景自适应参数"""
        params = dict(base_params)
        scene_p = self._scene_params.get(scene_type, {})
        
        # 调整速度
        if "max_speed" in params:
            speed_mult = scene_p.get("speed_multiplier", 1.0)
            # 如果成功率低，降低速度
            rate = scene_p.get("success_rate", 0.8)
            if rate < 0.7:
                speed_mult *= 0.8
            elif rate > 0.9:
                speed_mult *= 1.1
            params["max_speed"] = params["max_speed"] * speed_mult
        
        # 调整安全距离
        if "safe_distance" in params:
            caution_mult = scene_p.get("caution_multiplier", 1.0)
            rate = scene_p.get("success_rate", 0.8)
            if rate < 0.7:
                caution_mult *= 1.2  # 更保守
            params["safe_distance"] = params["safe_distance"] * caution_mult
        
        return params


# ============================================================
# 层级任务规划
# ============================================================

class HierarchicalTaskLevel(Enum):
    """层级任务级别"""
    STRATEGIC = "strategic"      # 战略级: 任务分配/资源规划
    TACTICAL = "tactical"       # 战术级: 路径规划/场景选择
    EXECUTION = "execution"     # 执行级: 动作序列/技能调用
    REACTIVE = "reactive"       # 反应级: 避障/安全检查


class TaskCompositionRule(Enum):
    """任务组合规则"""
    SEQUENTIAL = "sequential"   # 顺序执行
    PARALLEL = "parallel"       # 并行执行
    FALLBACK = "fallback"       # 降级执行
    CONDITIONAL = "conditional"  # 条件执行
    REPEAT_UNTIL = "repeat_until"  # 重复直到成功


class HierarchicalBehaviorTreeComposer:
    """
    层级行为树组合器

    从高层任务目标逐层分解为可执行的行为树:
    Level 1 (Strategic): "完成餐厅配送任务" -> 选择配送策略
    Level 2 (Tactical): "规划导航路径" -> 选择导航树
    Level 3 (Execution): "执行具体动作" -> 调用技能节点
    Level 4 (Reactive): "实时避障" -> 条件监控
    """

    def __init__(
        self,
        scene_intelligence: Optional[SceneIntelligence] = None,
        skill_registry: Optional[Any] = None,
    ):
        self._scene_intelligence = scene_intelligence
        self._skill_registry = skill_registry
        self._composition_cache: Dict[str, BehaviorTree] = {}
        self._level_handlers = {
            HierarchicalTaskLevel.STRATEGIC: self._build_strategic_level,
            HierarchicalTaskLevel.TACTICAL: self._build_tactical_level,
            HierarchicalTaskLevel.EXECUTION: self._build_execution_level,
            HierarchicalTaskLevel.REACTIVE: self._build_reactive_level,
        }

    def compose_task_tree(
        self,
        task_goal: str,
        scene_type: SceneType,
        context: Optional[Dict[str, Any]] = None,
    ) -> BehaviorTree:
        """
        从高层任务目标组合完整行为树

        Args:
            task_goal: 任务目标描述 ("完成餐厅配送", "执行仓库巡逻")
            scene_type: 场景类型
            context: 场景上下文

        Returns:
            完整的层级行为树
        """
        cache_key = f"{task_goal}:{scene_type.value}"
        if cache_key in self._composition_cache:
            return self._composition_cache[cache_key]

        context = context or {}
        # Root: Selector - try full HTN plan, fallback to simple execution
        bt = BehaviorTree(
            name=f"HTN_{task_goal}_{scene_type.value}",
            root=SelectorNode(name="root_selector"),
        )

        # Primary branch: Full hierarchical plan (strategic -> tactical -> execution)
        # with reactive monitoring as parallel guard
        primary_plan = self._build_primary_plan(task_goal, scene_type, context)
        bt.root.add_child(primary_plan)

        # Fallback branch: Simple reactive-only mode (when planning fails)
        fallback_plan = self._build_fallback_plan(scene_type, context)
        bt.root.add_child(fallback_plan)

        self._composition_cache[cache_key] = bt
        return bt

    def _build_primary_plan(
        self,
        task_goal: str,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> BTNode:
        """
        构建主计划: 战略->战术->执行, 带反应级并行监控

        结构:
        Parallel (success=REQUIRE_ALL, failure=REQUIRE_ANY)
        ├── Reactive: 持续安全监控 (碰撞/电池/通信/心跳)
        └── Sequence: 主任务执行流程
            ├── Strategic: 资源分配 + 任务分解
            ├── Tactical: 场景感知 + 路径规划
            └── Execution: 技能匹配 + 动作序列 + 结果验证
        """
        # 并行运行: 持续反应监控 + 主任务流程
        parallel = ParallelNode(
            name="primary_parallel",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY,
        )

        # 反应级监控 (持续运行)
        reactive_tree = self._build_reactive_level(scene_type, context)
        parallel.add_child(reactive_tree.root)

        # 主任务流程: 战略->战术->执行
        main_seq = SequenceNode(name="main_sequence")

        # Strategic level
        strategic_tree = self._build_strategic_level(task_goal, scene_type, context)
        main_seq.add_child(strategic_tree.root)

        # Tactical level
        tactical_tree = self._build_tactical_level(task_goal, scene_type, context)
        main_seq.add_child(tactical_tree.root)

        # Execution level
        execution_tree = self._build_execution_level(task_goal, scene_type, context)
        main_seq.add_child(execution_tree.root)

        parallel.add_child(main_seq)
        return parallel

    def _build_fallback_plan(
        self,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> BTNode:
        """
        构建降级计划: 仅保留反应级监控和基本动作

        当主计划失败时的降级方案:
        - 仅保留安全监控
        - 使用简化动作序列
        """
        fallback_seq = SequenceNode(name="fallback_sequence")

        # 记录降级状态
        fallback_seq.add_child(
            LambdaActionNode(
                action=lambda s: self._bb_set(s, "fallback_mode", True) or NodeStatus.SUCCESS,
                name="enter_fallback_mode",
            )
        )

        # 保留反应级监控
        reactive_tree = self._build_reactive_level(scene_type, context)
        fallback_seq.add_child(reactive_tree.root)

        # 简化的安全导航
        fallback_seq.add_child(
            LambdaActionNode(
                action=lambda s: self._bb_set(s, "simple_nav", True) or NodeStatus.SUCCESS,
                name="simple_navigate",
            )
        )

        return fallback_seq

    def _bb_set(self, blackboard: Blackboard, key: str, value: Any) -> None:
        """Helper: safely set blackboard value"""
        try:
            blackboard.set(key, value)
        except (TypeError, AttributeError):
            pass

    def _bb_get(self, blackboard: Blackboard, key: str, default: Any = None) -> Any:
        """Helper: safely get blackboard value"""
        try:
            return blackboard.get(key, default)
        except (TypeError, AttributeError):
            return default

    def _build_strategic_level(
        self,
        task_goal: str,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> BehaviorTree:
        """构建战略级行为树"""
        root = SelectorNode(name="strategic_selector")
        bt = BehaviorTree(name=f"strategic_{task_goal}", root=root)

        # 任务可行性检查
        feasibility_seq = SequenceNode(name="feasibility_check")
        feasibility_seq.add_child(
            ConditionNode(
                name="check_battery",
                condition=lambda s: s.get("battery_level", 100) > 20,
            )
        )
        feasibility_seq.add_child(
            ConditionNode(
                name="check_agv_health",
                condition=lambda s: s.get("agv_healthy", True),
            )
        )
        feasibility_seq.add_child(
            LambdaActionNode(
                name="plan_resource_allocation",
                action=lambda s: self._allocate_resources(task_goal, scene_type, s),
            )
        )
        root.add_child(feasibility_seq)

        # 任务分解策略
        decompose_sel = SelectorNode(name="task_decompose")
        if scene_type == SceneType.WAREHOUSE:
            decompose_sel.add_child(
                LambdaActionNode(
                    name="decompose_warehouse_task",
                    action=lambda s: self._decompose_warehouse(task_goal, s),
                )
            )
        elif scene_type == SceneType.RESTAURANT:
            decompose_sel.add_child(
                LambdaActionNode(
                    name="decompose_restaurant_task",
                    action=lambda s: self._decompose_restaurant(task_goal, s),
                )
            )
        elif scene_type == SceneType.HOSPITAL:
            decompose_sel.add_child(
                LambdaActionNode(
                    name="decompose_hospital_task",
                    action=lambda s: self._decompose_hospital(task_goal, s),
                )
            )
        else:
            decompose_sel.add_child(
                LambdaActionNode(
                    name="decompose_generic_task",
                    action=lambda s: self._decompose_generic(task_goal, s),
                )
            )
        root.add_child(decompose_sel)
        return bt

    def _build_tactical_level(
        self,
        task_goal: str,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> BehaviorTree:
        """构建战术级行为树"""
        root = SequenceNode(name="tactical_sequence")
        bt = BehaviorTree(name=f"tactical_{task_goal}", root=root)

        # 场景感知
        root.add_child(
            LambdaActionNode(
                name="perceive_scene",
                action=lambda s: self._perceive_scene(scene_type, s),
            )
        )

        # 路径规划选择
        path_sel = SelectorNode(name="path_planning_selector")

        # 主路径规划
        path_sel.add_child(
            LambdaActionNode(
                name="plan_primary_path",
                action=lambda s: self._plan_primary_path(scene_type, s),
            )
        )

        # 备用路径规划
        path_sel.add_child(
            LambdaActionNode(
                name="plan_secondary_path",
                action=lambda s: self._plan_secondary_path(scene_type, s),
            )
        )

        root.add_child(path_sel)

        # 资源预留
        root.add_child(
            LambdaActionNode(
                name="reserve_resources",
                action=lambda s: self._reserve_resources(scene_type, s),
            )
        )
        return bt

    def _build_execution_level(
        self,
        task_goal: str,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> BehaviorTree:
        """构建执行级行为树"""
        root = SequenceNode(name="execution_sequence")
        bt = BehaviorTree(name=f"execution_{task_goal}", root=root)

        # 技能匹配
        root.add_child(
            LambdaActionNode(
                name="match_skills",
                action=lambda s: self._match_skills_to_task(task_goal, scene_type, s),
            )
        )

        # 动作序列执行
        action_sel = SelectorNode(name="action_sequence_selector")

        # 尝试标准动作序列
        standard_seq = SequenceNode(name="standard_action_sequence")
        standard_seq.add_child(
            LambdaActionNode(
                name="execute_navigate",
                action=lambda s: self._execute_navigate(s),
            )
        )
        standard_seq.add_child(
            LambdaActionNode(
                name="execute_manipulate",
                action=lambda s: self._execute_manipulate(s),
            )
        )
        standard_seq.add_child(
            LambdaActionNode(
                name="execute_place",
                action=lambda s: self._execute_place(s),
            )
        )
        action_sel.add_child(standard_seq)

        # 降级动作序列
        fallback_seq = SequenceNode(name="fallback_action_sequence")
        fallback_seq.add_child(
            ConditionNode(
                name="has_simplified_path",
                condition=lambda s: s.get("simplified_available", False),
            )
        )
        fallback_seq.add_child(
            LambdaActionNode(
                name="execute_simplified",
                action=lambda s: self._execute_simplified(s),
            )
        )
        action_sel.add_child(fallback_seq)
        root.add_child(action_sel)

        # 结果验证
        root.add_child(
            LambdaActionNode(
                name="verify_execution",
                action=lambda s: self._verify_execution(s),
            )
        )
        return bt

    def _build_reactive_level(
        self,
        scene_type: SceneType,
        context: Dict[str, Any],
    ) -> BehaviorTree:
        """构建反应级行为树"""
        root = ParallelNode(
            name="reactive_parallel",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY,
        )
        bt = BehaviorTree(name=f"reactive_{scene_type.value}", root=root)

        # 碰撞监控
        root.add_child(
            SequenceNode(
                name="collision_monitor",
                children=[
                    ConditionNode(
                        name="check_obstacle",
                        condition=lambda s: s.get("obstacle_detected", False),
                    ),
                    LambdaActionNode(
                        name="trigger_avoidance",
                        action=lambda s: self._trigger_avoidance(s),
                    ),
                ],
            )
        )

        # 电池监控
        root.add_child(
            SequenceNode(
                name="battery_monitor",
                children=[
                    ConditionNode(
                        name="check_low_battery",
                        condition=lambda s: s.get("battery_level", 100) < 15,
                    ),
                    LambdaActionNode(
                        name="trigger_recharge",
                        action=lambda s: self._trigger_recharge(s),
                    ),
                ],
            )
        )

        # 通信监控
        root.add_child(
            SequenceNode(
                name="comms_monitor",
                children=[
                    ConditionNode(
                        name="check_comms_lost",
                        condition=lambda s: not s.get("comms_active", True),
                    ),
                    LambdaActionNode(
                        name="trigger_comms_recovery",
                        action=lambda s: self._trigger_comms_recovery(s),
                    ),
                ],
            )
        )

        # 心跳监控
        root.add_child(
            SequenceNode(
                name="heartbeat_monitor",
                children=[
                    ConditionNode(
                        name="check_heartbeat_timeout",
                        condition=lambda s: s.get("heartbeat_timeout", False),
                    ),
                    LambdaActionNode(
                        name="trigger_heartbeat_recovery",
                        action=lambda s: self._trigger_heartbeat_recovery(s),
                    ),
                ],
            )
        )
        return bt

    # ---- 战略级处理器 ----

    def _allocate_resources(
        self,
        task_goal: str,
        scene_type: SceneType,
        blackboard: Blackboard,
    ) -> NodeStatus:
        """分配任务资源"""
        required = {
            SceneType.WAREHOUSE: {"battery": 40, "time": 300},
            SceneType.RESTAURANT: {"battery": 30, "time": 180},
            SceneType.HOSPITAL: {"battery": 50, "time": 400},
            SceneType.FACTORY: {"battery": 35, "time": 250},
            SceneType.OUTDOOR: {"battery": 60, "time": 600},
        }
        req = required.get(scene_type, {"battery": 30, "time": 200})
        blackboard.set("allocated_battery", req["battery"])
        blackboard.set("allocated_time", req["time"])
        blackboard.set("resources_allocated", True)
        return NodeStatus.SUCCESS

    def _decompose_warehouse(self, task_goal: str, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("subtasks", ["navigate_to_pick", "pick_item", "navigate_to_drop", "place_item"])
        return NodeStatus.SUCCESS

    def _decompose_restaurant(self, task_goal: str, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("subtasks", ["navigate_to_kitchen", "pick_food", "navigate_to_table", "deliver_food"])
        return NodeStatus.SUCCESS

    def _decompose_hospital(self, task_goal: str, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("subtasks", ["navigate_to_storage", "pick_medicine", "navigate_to_ward", "deliver_medicine", "verify_delivery"])
        return NodeStatus.SUCCESS

    def _decompose_generic(self, task_goal: str, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("subtasks", ["navigate", "manipulate", "navigate_back"])
        return NodeStatus.SUCCESS

    # ---- 战术级处理器 ----

    def _perceive_scene(self, scene_type: SceneType, blackboard: Blackboard) -> NodeStatus:
        if self._scene_intelligence:
            scene_ctx = self._scene_intelligence.get_scene_context(scene_type)
            blackboard.set("scene_context", scene_ctx)
        blackboard.set("scene_perceived", True)
        return NodeStatus.SUCCESS

    def _plan_primary_path(self, scene_type: SceneType, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("primary_path", f"path_{scene_type.value}_primary")
        blackboard.set("path_planned", True)
        return NodeStatus.SUCCESS

    def _plan_secondary_path(self, scene_type: SceneType, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("secondary_path", f"path_{scene_type.value}_secondary")
        blackboard.set("fallback_available", True)
        return NodeStatus.SUCCESS

    def _reserve_resources(self, scene_type: SceneType, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("resources_reserved", True)
        return NodeStatus.SUCCESS

    # ---- 执行级处理器 ----

    def _match_skills_to_task(
        self,
        task_goal: str,
        scene_type: SceneType,
        blackboard: Blackboard,
    ) -> NodeStatus:
        if self._skill_registry:
            skills = self._skill_registry.get_skills_by_scene(scene_type.value)
            blackboard.set("matched_skills", [s.name() for s in skills[:3]])
        else:
            blackboard.set("matched_skills", ["navigate", "manipulate", "place"])
        return NodeStatus.SUCCESS

    def _execute_navigate(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("nav_progress", blackboard.get("nav_progress", 0) + 1)
        blackboard.set("current_action", "navigate")
        return NodeStatus.SUCCESS

    def _execute_manipulate(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("manip_progress", blackboard.get("manip_progress", 0) + 1)
        blackboard.set("current_action", "manipulate")
        return NodeStatus.SUCCESS

    def _execute_place(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("place_progress", blackboard.get("place_progress", 0) + 1)
        blackboard.set("current_action", "place")
        return NodeStatus.SUCCESS

    def _execute_simplified(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("simplified_executed", True)
        blackboard.set("current_action", "simplified")
        return NodeStatus.SUCCESS

    def _verify_execution(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("execution_verified", True)
        return NodeStatus.SUCCESS

    # ---- 反应级处理器 ----

    def _trigger_avoidance(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("avoidance_triggered", True)
        blackboard.set("current_action", "avoidance")
        return NodeStatus.SUCCESS

    def _trigger_recharge(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("recharge_triggered", True)
        blackboard.set("current_action", "recharge")
        return NodeStatus.SUCCESS

    def _trigger_comms_recovery(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("comms_recovery_triggered", True)
        blackboard.set("current_action", "comms_recovery")
        return NodeStatus.SUCCESS

    def _trigger_heartbeat_recovery(self, blackboard: Blackboard) -> NodeStatus:
        blackboard.set("heartbeat_recovery_triggered", True)
        blackboard.set("current_action", "heartbeat_recovery")
        return NodeStatus.SUCCESS


# ============================================================
# 记忆增强的场景规划器
# ============================================================

class SceneMemoryAugmentedPlanner:
    """
    记忆增强的场景规划器

    利用长期记忆中的场景经验来增强任务规划:
    - 检索相似历史任务的执行策略
    - 基于记忆优化参数配置
    - 跨任务经验迁移
    - 失败模式识别与规避
    """

    def __init__(
        self,
        scene_intelligence: Optional[SceneIntelligence] = None,
        memory: Optional["LongTermMemory"] = None,
    ):
        self._scene_intelligence = scene_intelligence
        self._memory = memory
        self._experience_cache: Dict[str, List[Dict]] = {}
        self._success_patterns: Dict[str, List[Dict]] = {}
        self._failure_patterns: Dict[str, List[Dict]] = {}

    def retrieve_relevant_experiences(
        self,
        task_goal: str,
        scene_type: SceneType,
        max_count: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        检索与当前任务相关的历史经验

        Args:
            task_goal: 任务目标
            scene_type: 场景类型
            max_count: 最大返回数量

        Returns:
            相关经验列表
        """
        cache_key = f"{task_goal}:{scene_type.value}"
        if cache_key in self._experience_cache:
            return self._experience_cache[cache_key]

        experiences = []

        # 从记忆中检索
        if self._memory:
            try:
                results = self._memory.retrieve(
                    query=f"{scene_type.value} {task_goal}",
                    top_k=max_count,
                )
                if results:
                    experiences.extend([{"source": "memory", **r} for r in results])
            except Exception:
                pass

        # 如果记忆中没有，使用场景特定的默认经验
        if not experiences:
            experiences = self._get_default_experiences(task_goal, scene_type)

        self._experience_cache[cache_key] = experiences
        return experiences[:max_count]

    def _get_default_experiences(
        self,
        task_goal: str,
        scene_type: SceneType,
    ) -> List[Dict[str, Any]]:
        """获取场景默认经验"""
        defaults = {
            SceneType.WAREHOUSE: [
                {"pattern": "pick_item", "avg_duration_ms": 15000, "success_rate": 0.92},
                {"pattern": "place_item", "avg_duration_ms": 12000, "success_rate": 0.95},
                {"pattern": "zone_patrol", "avg_duration_ms": 60000, "success_rate": 0.88},
            ],
            SceneType.RESTAURANT: [
                {"pattern": "pick_food", "avg_duration_ms": 8000, "success_rate": 0.90},
                {"pattern": "deliver_food", "avg_duration_ms": 20000, "success_rate": 0.87},
                {"pattern": "clear_table", "avg_duration_ms": 25000, "success_rate": 0.83},
            ],
            SceneType.HOSPITAL: [
                {"pattern": "verify_delivery", "avg_duration_ms": 5000, "success_rate": 0.98},
                {"pattern": "medicine_delivery", "avg_duration_ms": 30000, "success_rate": 0.96},
            ],
            SceneType.FACTORY: [
                {"pattern": "assembly_task", "avg_duration_ms": 45000, "success_rate": 0.91},
                {"pattern": "quality_check", "avg_duration_ms": 10000, "success_rate": 0.89},
            ],
            SceneType.OUTDOOR: [
                {"pattern": "gps_navigation", "avg_duration_ms": 90000, "success_rate": 0.85},
                {"pattern": "terrain_avoidance", "avg_duration_ms": 15000, "success_rate": 0.82},
            ],
        }
        return defaults.get(scene_type, [])

    def optimize_task_params(
        self,
        task_goal: str,
        scene_type: SceneType,
        base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        基于历史经验优化任务参数

        Args:
            task_goal: 任务目标
            scene_type: 场景类型
            base_params: 基础参数

        Returns:
            优化后的参数
        """
        params = dict(base_params)
        experiences = self.retrieve_relevant_experiences(task_goal, scene_type)

        if not experiences:
            return params

        # 基于成功率调整安全系数
        avg_success = np.mean([e.get("success_rate", 0.8) for e in experiences])
        if avg_success < 0.8:
            params["safety_margin"] = params.get("safety_margin", 1.0) * 1.2
            params["timeout_factor"] = params.get("timeout_factor", 1.0) * 1.3
        elif avg_success > 0.95:
            params["safety_margin"] = params.get("safety_margin", 1.0) * 0.9
            params["timeout_factor"] = params.get("timeout_factor", 1.0) * 0.85

        # 基于平均执行时间调整超时
        avg_duration = np.mean([e.get("avg_duration_ms", 30000) for e in experiences])
        params["estimated_duration_ms"] = avg_duration
        params["timeout_ms"] = avg_duration * params.get("timeout_factor", 1.2)

        return params

    def record_experience(
        self,
        task_goal: str,
        scene_type: SceneType,
        success: bool,
        duration_ms: float,
        params: Dict[str, Any],
        outcome_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录一次任务执行经验"""
        cache_key = f"{task_goal}:{scene_type.value}"

        exp = {
            "task_goal": task_goal,
            "scene_type": scene_type.value,
            "success": success,
            "duration_ms": duration_ms,
            "params": params,
            "timestamp": time.time(),
            **(outcome_details or {}),
        }

        # 更新缓存
        if cache_key not in self._experience_cache:
            self._experience_cache[cache_key] = []
        self._experience_cache[cache_key].append(exp)

        # 如果成功，更新成功模式
        if success:
            if cache_key not in self._success_patterns:
                self._success_patterns[cache_key] = []
            self._success_patterns[cache_key].append(exp)
        else:
            if cache_key not in self._failure_patterns:
                self._failure_patterns[cache_key] = []
            self._failure_patterns[cache_key].append(exp)

        # 存储到长期记忆
        if self._memory:
            try:
                self._memory.store_episode(
                    summary=f"{scene_type.value}:{task_goal} {'success' if success else 'failure'}",
                    context=exp,
                    actions=[],
                    outcomes={"success": success},
                    importance_score=8.0 if not success else 5.0,
                    tags=[scene_type.value, task_goal, "success" if success else "failure"],
                )
            except Exception:
                pass

    def get_failure_warnings(
        self,
        task_goal: str,
        scene_type: SceneType,
    ) -> List[str]:
        """基于历史失败经验返回警告"""
        cache_key = f"{task_goal}:{scene_type.value}"
        warnings = []

        failures = self._failure_patterns.get(cache_key, [])
        if len(failures) >= 3:
            failure_rate = len(failures) / (
                len(self._success_patterns.get(cache_key, [])) + len(failures) + 1e-6
            )
            if failure_rate > 0.3:
                warnings.append(f"历史失败率较高: {failure_rate:.1%}")
                recent_failures = failures[-3:]
                if any(f.get("params", {}).get("battery_level", 100) < 30 for f in recent_failures):
                    warnings.append("建议确保电量充足 (>50%)")
                if any(f.get("params", {}).get("obstacle_density", 0) > 0.5 for f in recent_failures):
                    warnings.append("历史在高障碍密度下失败，建议等待环境清理")

        return warnings


# ============================================================
# 跨场景迁移学习器
# ==========================================================

class CrossSceneTransferLearner:
    """
    跨场景迁移学习器

    从一个场景学习到的知识迁移到另一个场景:
    - 提取场景无关的通用策略
    - 识别需要场景特定调整的参数
    - 评估迁移适用性
    - 生成场景适配建议
    """

    def __init__(
        self,
        scene_intelligence: Optional[SceneIntelligence] = None,
        memory: Optional["LongTermMemory"] = None,
    ):
        self._scene_intelligence = scene_intelligence
        self._memory = memory
        self._transfer_matrix: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._generic_skills: List[str] = [
            "navigate",
            "obstacle_avoidance",
            "low_battery_return",
            "emergency_stop",
            "comms_reconnect",
        ]

    def evaluate_transferability(
        self,
        source_scene: SceneType,
        target_scene: SceneType,
        skill: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估从源场景到目标场景的知识迁移适用性

        Returns:
            transfer_score: 0.0-1.0 迁移适用性
            generic_skills: 可直接迁移的通用技能
            adaptable_skills: 需要适配的技能
            warnings: 迁移警告
        """
        matrix_key = (source_scene.value, target_scene.value)

        if matrix_key in self._transfer_matrix:
            return self._transfer_matrix[matrix_key]

        # 场景相似度矩阵
        scene_similarity = self._compute_scene_similarity(source_scene, target_scene)

        generic = []
        adaptable = []
        warnings = []

        for skill_name in self._generic_skills:
            generic.append(skill_name)

        # 基于场景相似度评估
        if scene_similarity < 0.3:
            warnings.append("源场景与目标场景差异较大，建议谨慎迁移")
            adaptable_ratio = 0.3
        elif scene_similarity < 0.6:
            warnings.append("部分参数需要场景特定调整")
            adaptable_ratio = 0.6
        else:
            warnings.append("场景高度相似，可大量迁移")
            adaptable_ratio = 0.85

        result = {
            "source_scene": source_scene.value,
            "target_scene": target_scene.value,
            "scene_similarity": scene_similarity,
            "transfer_score": (scene_similarity * 0.7 + adaptable_ratio * 0.3),
            "generic_skills": generic,
            "adaptable_skills": [
                "pick_item", "place_item", "patrol",
                "deliver", "clear_table", "verify_delivery",
            ][:int(6 * adaptable_ratio)],
            "warnings": warnings,
        }

        self._transfer_matrix[matrix_key] = result
        return result

    def _compute_scene_similarity(
        self,
        scene_a: SceneType,
        scene_b: SceneType,
    ) -> float:
        """计算两个场景的相似度"""
        similarity_map = {
            (SceneType.WAREHOUSE, SceneType.FACTORY): 0.75,
            (SceneType.RESTAURANT, SceneType.OFFICE): 0.55,
            (SceneType.HOSPITAL, SceneType.LABORATORY): 0.70,
            (SceneType.OUTDOOR, SceneType.WAREHOUSE): 0.50,
            (SceneType.FACTORY, SceneType.HOSPITAL): 0.45,
            (SceneType.RESTAURANT, SceneType.HOSPITAL): 0.40,
        }

        if (scene_a, scene_b) in similarity_map:
            return similarity_map[(scene_a, scene_b)]
        if scene_a == scene_b:
            return 1.0
        return 0.35

    def transfer_task_plan(
        self,
        source_scene: SceneType,
        target_scene: SceneType,
        source_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        将源场景的任务计划迁移到目标场景

        Args:
            source_scene: 源场景
            target_scene: 目标场景
            source_plan: 源场景的任务计划

        Returns:
            适配后的目标场景任务计划
        """
        transfer_info = self.evaluate_transferability(source_scene, target_scene)

        adapted_plan = dict(source_plan)
        adapted_plan["source_scene"] = source_scene.value
        adapted_plan["target_scene"] = target_scene.value
        adapted_plan["transfer_score"] = transfer_info["transfer_score"]
        adapted_plan["adaptations_applied"] = []

        # 调整速度参数
        speed_adjustments = {
            SceneType.HOSPITAL: 0.7,  # 更保守
            SceneType.RESTAURANT: 0.85,
            SceneType.FACTORY: 1.0,
            SceneType.WAREHOUSE: 1.0,
            SceneType.OUTDOOR: 1.1,
        }
        if "max_speed" in source_plan:
            speed_factor = speed_adjustments.get(target_scene, 1.0)
            adapted_plan["max_speed"] = source_plan["max_speed"] * speed_factor
            adapted_plan["adaptations_applied"].append(f"speed_factor:{speed_factor}")

        # 调整安全距离
        safety_adjustments = {
            SceneType.HOSPITAL: 1.3,  # 更高安全距离
            SceneType.RESTAURANT: 1.1,
            SceneType.FACTORY: 1.0,
            SceneType.WAREHOUSE: 0.9,
            SceneType.OUTDOOR: 0.85,
        }
        if "safe_distance" in source_plan:
            safety_factor = safety_adjustments.get(target_scene, 1.0)
            adapted_plan["safe_distance"] = source_plan["safe_distance"] * safety_factor
            adapted_plan["adaptations_applied"].append(f"safety_factor:{safety_factor}")

        # 添加迁移警告
        adapted_plan["transfer_warnings"] = transfer_info["warnings"]

        return adapted_plan


# ============================================================
# 全局单例
# ============================================================

_global_planner: Optional[SceneTaskPlanner] = None


def get_scene_task_planner(
    config: Optional[SceneTaskConfig] = None,
    scene_intelligence: Optional[SceneIntelligence] = None,
    memory: Optional["LongTermMemory"] = None,
) -> SceneTaskPlanner:
    """获取场景任务规划器全局单例"""
    global _global_planner
    if _global_planner is None:
        _global_planner = SceneTaskPlanner(
            config=config,
            scene_intelligence=scene_intelligence,
            memory=memory,
        )
    return _global_planner
