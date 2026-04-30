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
scene_intelligence.py - 场景化具身智能模块
SuperModel 超模态大模型具身智能系统

场景化智能:
- 场景分类与识别 (仓库/工厂/医院/餐厅/户外)
- 场景感知任务规划
- 场景自适应行为切换
- 跨场景经验迁移
- 场景安全规则引擎
- 多AGV场景协同
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum, auto
import numpy as np

if TYPE_CHECKING:
    from ..memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)

__all__ = [
    'SceneType',
    'SceneContext',
    'SceneRule',
    'SafetyRule',
    'NavigationRule',
    'InteractionRule',
    'SceneIntelligence',
    'get_scene_intelligence',
]


class SceneType(Enum):
    """场景类型枚举"""
    WAREHOUSE = "warehouse"           # 仓储物流
    FACTORY = "factory"               # 工业制造
    HOSPITAL = "hospital"             # 医院物流
    RESTAURANT = "restaurant"         # 餐厅配送
    OFFICE = "office"                 # 办公室配送
    OUTDOOR = "outdoor"              # 户外配送
    LABORATORY = "laboratory"         # 实验室
    HOME = "home"                    # 家庭服务
    UNKNOWN = "unknown"


@dataclass
class SceneFeatures:
    """场景特征"""
    scene_type: SceneType = SceneType.UNKNOWN
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    # 环境特征
    floor_type: str = "concrete"      # concrete, tile, carpet, outdoor
    floor_friction: float = 0.8
    aisle_width: float = 2.0         # m
    ceiling_height: float = 4.0      # m
    lighting_level: float = 1.0      # 0-1 normalized
    # 动态障碍密度
    obstacle_density: float = 0.1    # 0-1
    human_density: float = 0.05      # 0-1
    # 安全参数
    max_speed_safe: float = 1.5      # m/s
    safe_distance: float = 0.5       # m
    emergency_stop_dist: float = 0.2 # m

    def is_safe_for_high_speed(self) -> bool:
        """判断当前场景是否适合高速行驶"""
        return (self.obstacle_density < 0.2 and
                self.human_density < 0.1 and
                self.floor_friction > 0.5)


@dataclass
class SceneRule:
    """场景规则基类"""
    rule_id: str
    scene_types: Set[SceneType]
    priority: int = 1                # 1=highest
    enabled: bool = True


@dataclass
class SafetyRule(SceneRule):
    """安全规则"""
    min_clearance: float = 0.3       # m - 与障碍物最小距离
    max_speed: float = 1.5           # m/s
    human_detection_dist: float = 2.0 # m - 人形检测距离
    emergency_exit_required: bool = False
    fire_safety_required: bool = False


@dataclass
class NavigationRule(SceneRule):
    """导航规则"""
    use_visual_landmarks: bool = True
    use_laser_scan: bool = True
    avoid_dynamically: bool = True
    path_replan_interval: float = 1.0  # seconds
    map_update_interval: float = 5.0  # seconds


@dataclass
class InteractionRule(SceneRule):
    """交互规则"""
    allow_human_handoff: bool = True
    voice_interaction: bool = False
    touchscreen_interaction: bool = False
    wait_for_confirmation: bool = False
    express_emotion: bool = False


@dataclass
class SceneContext:
    """场景上下文"""
    features: SceneFeatures = field(default_factory=SceneFeatures)
    current_location: str = ""
    floor_level: int = 1
    current_task: str = ""
    nearby_agvs: List[str] = field(default_factory=list)
    nearby_humans: int = 0
    timestamp: float = field(default_factory=time.time)

    def get_scene_type(self) -> SceneType:
        return self.features.scene_type

    def is_safe_for_high_speed(self) -> bool:
        return (self.features.obstacle_density < 0.2 and
                self.features.human_density < 0.1 and
                self.features.floor_friction > 0.5)


@dataclass
class SceneConfig:
    """场景智能配置"""
    # 场景检测
    detection_interval: float = 2.0    # seconds
    min_confidence_threshold: float = 0.7

    # 规则更新
    rule_update_interval: float = 5.0  # seconds

    # 记忆集成
    enable_memory: bool = True
    experience_retention_days: int = 30

    # AGV等级适配
    grade: str = "M"

    # 场景特定参数
    warehouse_max_speed: float = 2.0
    factory_max_speed: float = 1.0
    hospital_max_speed: float = 0.8
    restaurant_max_speed: float = 1.0
    outdoor_max_speed: float = 3.0


# ============================================================
# 场景规则引擎
# ============================================================

class SceneRuleEngine:
    """场景规则引擎 - 根据场景类型动态加载规则"""

    def __init__(self, config: Optional[SceneConfig] = None):
        self.config = config or SceneConfig()
        self._rules: Dict[type, List[SceneRule]] = {
            SafetyRule: [],
            NavigationRule: [],
            InteractionRule: [],
        }
        self._scene_specific_rules: Dict[SceneType, Dict[type, List[SceneRule]]] = {}
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认规则集"""
        # 仓库场景规则
        self._add_scene_rule(SceneType.WAREHOUSE, SafetyRule(
            rule_id="warehouse_safety",
            scene_types={SceneType.WAREHOUSE},
            priority=1,
            min_clearance=0.3,
            max_speed=2.0,
            human_detection_dist=3.0,
        ))
        self._add_scene_rule(SceneType.WAREHOUSE, NavigationRule(
            rule_id="warehouse_nav",
            scene_types={SceneType.WAREHOUSE},
            use_visual_landmarks=False,
            use_laser_scan=True,
            avoid_dynamically=True,
        ))
        self._add_scene_rule(SceneType.WAREHOUSE, InteractionRule(
            rule_id="warehouse_interact",
            scene_types={SceneType.WAREHOUSE},
            allow_human_handoff=True,
            wait_for_confirmation=False,
        ))

        # 医院场景规则 - 最高安全等级
        self._add_scene_rule(SceneType.HOSPITAL, SafetyRule(
            rule_id="hospital_safety",
            scene_types={SceneType.HOSPITAL},
            priority=1,
            min_clearance=0.5,
            max_speed=0.8,
            human_detection_dist=5.0,
            emergency_exit_required=True,
        ))
        self._add_scene_rule(SceneType.HOSPITAL, NavigationRule(
            rule_id="hospital_nav",
            scene_types={SceneType.HOSPITAL},
            use_visual_landmarks=True,
            use_laser_scan=True,
            avoid_dynamically=True,
            path_replan_interval=0.5,
        ))
        self._add_scene_rule(SceneType.HOSPITAL, InteractionRule(
            rule_id="hospital_interact",
            scene_types={SceneType.HOSPITAL},
            allow_human_handoff=True,
            voice_interaction=True,
            wait_for_confirmation=True,
        ))

        # 工厂场景规则
        self._add_scene_rule(SceneType.FACTORY, SafetyRule(
            rule_id="factory_safety",
            scene_types={SceneType.FACTORY},
            priority=1,
            min_clearance=0.4,
            max_speed=1.0,
            human_detection_dist=2.0,
            fire_safety_required=True,
        ))
        self._add_scene_rule(SceneType.FACTORY, NavigationRule(
            rule_id="factory_nav",
            scene_types={SceneType.FACTORY},
            use_visual_landmarks=False,
            use_laser_scan=True,
            avoid_dynamically=True,
            path_replan_interval=2.0,
        ))
        self._add_scene_rule(SceneType.FACTORY, InteractionRule(
            rule_id="factory_interact",
            scene_types={SceneType.FACTORY},
            allow_human_handoff=False,
        ))

        # 餐厅场景规则
        self._add_scene_rule(SceneType.RESTAURANT, SafetyRule(
            rule_id="restaurant_safety",
            scene_types={SceneType.RESTAURANT},
            priority=1,
            min_clearance=0.2,
            max_speed=1.0,
            human_detection_dist=2.5,
        ))
        self._add_scene_rule(SceneType.RESTAURANT, InteractionRule(
            rule_id="restaurant_interact",
            scene_types={SceneType.RESTAURANT},
            allow_human_handoff=True,
            voice_interaction=True,
            wait_for_confirmation=True,
            express_emotion=True,
        ))

        # 户外场景规则
        self._add_scene_rule(SceneType.OUTDOOR, SafetyRule(
            rule_id="outdoor_safety",
            scene_types={SceneType.OUTDOOR},
            priority=1,
            min_clearance=0.5,
            max_speed=3.0,
            human_detection_dist=4.0,
        ))
        self._add_scene_rule(SceneType.OUTDOOR, NavigationRule(
            rule_id="outdoor_nav",
            scene_types={SceneType.OUTDOOR},
            use_visual_landmarks=True,
            use_laser_scan=True,
            avoid_dynamically=True,
            path_replan_interval=3.0,
        ))

    def _add_scene_rule(self, scene: SceneType, rule: SceneRule):
        """添加场景规则"""
        rule_type = type(rule)
        if scene not in self._scene_specific_rules:
            self._scene_specific_rules[scene] = {rule_type: []}
        if rule_type not in self._scene_specific_rules[scene]:
            self._scene_specific_rules[scene][rule_type] = []
        self._scene_specific_rules[scene][rule_type].append(rule)
        self._rules[rule_type].append(rule)

    def get_active_rules(self, scene: SceneType) -> Dict[type, List[SceneRule]]:
        """获取指定场景的活跃规则"""
        return self._scene_specific_rules.get(scene, {})

    def get_rule_by_type(self, rule_type: type, scene: Optional[SceneType] = None
                         ) -> List[SceneRule]:
        """获取指定类型的规则"""
        if scene is not None:
            return self._scene_specific_rules.get(scene, {}).get(rule_type, [])
        return self._rules.get(rule_type, [])

    def get_applicable_safety_rules(self, scene: SceneType) -> List[SafetyRule]:
        """获取适用的安全规则"""
        rules = []
        for st in [scene, SceneType.UNKNOWN]:
            rules.extend(self.get_rule_by_type(SafetyRule, st))
        return sorted(rules, key=lambda r: r.priority)

    def get_effective_max_speed(self, scene: SceneType) -> float:
        """根据场景获取有效最大速度"""
        safety_rules = self.get_applicable_safety_rules(scene)
        if safety_rules:
            return min(r.max_speed for r in safety_rules)
        return 2.0


# ============================================================
# 场景分类器
# ============================================================

class SceneClassifier:
    """基于传感器输入的场景分类器"""

    def __init__(self, config: Optional[SceneConfig] = None):
        self.config = config or SceneConfig()
        self._last_classification = (SceneType.UNKNOWN, 0.0)
        self._feature_buffer: List[Dict[str, float]] = []
        self._buffer_size = 10

    def classify(
        self,
        laser_ranges: Optional[np.ndarray] = None,
        vision_features: Optional[Dict[str, float]] = None,
        imu_data: Optional[Dict[str, float]] = None,
        audio_activity: float = 0.0,
        location_hint: str = "",
    ) -> Tuple[SceneType, float, SceneFeatures]:
        """
        基于多传感器输入分类场景

        Args:
            laser_ranges: 激光雷达扫描数据
            vision_features: 视觉特征 dict
            imu_data: IMU数据 dict
            audio_activity: 音频活跃度 0-1
            location_hint: 位置提示字符串

        Returns:
            (scene_type, confidence, features)
        """
        features = self._extract_features(
            laser_ranges, vision_features, imu_data, audio_activity, location_hint
        )
        self._feature_buffer.append(features)
        if len(self._feature_buffer) > self._buffer_size:
            self._feature_buffer.pop(0)

        # 决策融合
        scene_type, confidence = self._decide_scene(features, location_hint)
        self._last_classification = (scene_type, confidence)

        return scene_type, confidence, self._build_scene_features(scene_type, confidence, features)

    def _extract_features(
        self,
        laser_ranges: Optional[np.ndarray],
        vision_features: Optional[Dict[str, float]],
        imu_data: Optional[Dict[str, float]],
        audio_activity: float,
        location_hint: str,
    ) -> Dict[str, float]:
        """提取场景特征"""
        features = {
            'audio_activity': audio_activity,
            'open_space_ratio': 0.0,
            'corridor_width': 2.0,
            'obstacle_variety': 0.05,  # 低障碍密度 = 开放仓库环境
            'floor_reflectivity': 0.3,
            'ceiling_detected': 0.0,
        }

        if laser_ranges is not None and len(laser_ranges) > 0:
            valid_ranges = laser_ranges[laser_ranges > 0.1]
            if len(valid_ranges) > 0:
                features['open_space_ratio'] = float(
                    np.mean(valid_ranges > 2.0) if len(valid_ranges) > 0 else 0.0
                )
                features['corridor_width'] = float(np.std(valid_ranges) * 2)

        if vision_features:
            features.update(vision_features)

        return features

    def _decide_scene(self, features: Dict[str, float], location_hint: str
                      ) -> Tuple[SceneType, float]:
        """决策场景类型"""
        hint_lower = location_hint.lower()

        # 基于位置提示的快速判断 - 优先精确匹配/长词匹配
        if 'hospital' in hint_lower or '医院' in hint_lower or '医疗' in hint_lower:
            return SceneType.HOSPITAL, 0.95
        if 'warehouse' in hint_lower or '仓储' in hint_lower:
            return SceneType.WAREHOUSE, 0.95
        if 'factory' in hint_lower or '工厂' in hint_lower or '制造' in hint_lower:
            return SceneType.FACTORY, 0.95
        if 'restaurant' in hint_lower or '餐厅' in hint_lower or '食堂' in hint_lower:
            return SceneType.RESTAURANT, 0.95
        if 'outdoor' in hint_lower or '户外' in hint_lower or '室外' in hint_lower:
            return SceneType.OUTDOOR, 0.95
        if 'office' in hint_lower or '办公室' in hint_lower:
            return SceneType.OFFICE, 0.95
        if 'lab' in hint_lower or '实验室' in hint_lower:
            return SceneType.LABORATORY, 0.95
        # "物流" 放最后，因为很多场景都有物流
        if '物流' in hint_lower:
            return SceneType.WAREHOUSE, 0.85

        # 基于传感器特征的推断
        audio = features.get('audio_activity', 0.0)
        open_space = features.get('open_space_ratio', 0.5)
        corridor_w = features.get('corridor_width', 2.0)

        # 医院: 中等开放空间，低音频，高安全
        if 0.3 < open_space < 0.7 and audio < 0.3 and corridor_w > 1.5:
            return SceneType.HOSPITAL, 0.65

        # 仓库: 高开放空间，低音频，宽通道
        if open_space > 0.6 and audio < 0.2 and corridor_w > 2.5:
            return SceneType.WAREHOUSE, 0.70

        # 工厂: 封闭空间，低音频
        if open_space < 0.4 and audio < 0.2:
            return SceneType.FACTORY, 0.60

        # 餐厅: 中等开放空间，高音频
        if 0.3 < open_space < 0.7 and audio > 0.5:
            return SceneType.RESTAURANT, 0.65

        return SceneType.UNKNOWN, 0.3

    def _build_scene_features(
        self, scene: SceneType, confidence: float, features: Dict[str, float]
    ) -> SceneFeatures:
        """构建场景特征对象"""
        # 根据场景类型设置默认参数
        defaults = {
            SceneType.WAREHOUSE: {'floor_friction': 0.9, 'max_speed_safe': 2.0, 'aisle_width': 2.5},
            SceneType.FACTORY: {'floor_friction': 0.85, 'max_speed_safe': 1.0, 'aisle_width': 1.8},
            SceneType.HOSPITAL: {'floor_friction': 0.7, 'max_speed_safe': 0.8, 'aisle_width': 2.0},
            SceneType.RESTAURANT: {'floor_friction': 0.6, 'max_speed_safe': 1.0, 'aisle_width': 1.5},
            SceneType.OFFICE: {'floor_friction': 0.5, 'max_speed_safe': 1.2, 'aisle_width': 1.5},
            SceneType.OUTDOOR: {'floor_friction': 0.6, 'max_speed_safe': 3.0, 'aisle_width': 3.0},
            SceneType.LABORATORY: {'floor_friction': 0.75, 'max_speed_safe': 0.8, 'aisle_width': 1.2},
            SceneType.HOME: {'floor_friction': 0.5, 'max_speed_safe': 0.5, 'aisle_width': 1.0},
            SceneType.UNKNOWN: {'floor_friction': 0.8, 'max_speed_safe': 1.0, 'aisle_width': 2.0},
        }
        d = defaults.get(scene, defaults[SceneType.UNKNOWN])
        return SceneFeatures(
            scene_type=scene,
            confidence=confidence,
            features=features,
            floor_friction=d['floor_friction'],
            aisle_width=d['aisle_width'],
            max_speed_safe=d['max_speed_safe'],
            obstacle_density=features.get('obstacle_variety', 0.1),
            human_density=features.get('audio_activity', 0.0) * 0.5,
        )


# ============================================================
# 场景智能主类
# ============================================================

class SceneIntelligence:
    """
    场景化具身智能系统

    集成:
    - 场景分类识别
    - 场景规则引擎
    - 场景自适应行为
    - 长期记忆集成
    """

    def __init__(
        self,
        config: Optional[SceneConfig] = None,
        memory: Optional["LongTermMemory"] = None,
    ):
        self.config = config or SceneConfig()
        self._memory = memory
        self._classifier = SceneClassifier(config)
        self._rule_engine = SceneRuleEngine(config)
        self._current_context = SceneContext()
        self._last_update = 0.0
        self._scene_history: List[Tuple[SceneType, float, float]] = []  # (type, conf, time)

    def update(
        self,
        laser_ranges: Optional[np.ndarray] = None,
        vision_features: Optional[Dict[str, float]] = None,
        imu_data: Optional[Dict[str, float]] = None,
        audio_activity: float = 0.0,
        location_hint: str = "",
        nearby_humans: int = 0,
        nearby_agvs: Optional[List[str]] = None,
        current_task: str = "",
        current_location: str = "",
    ) -> SceneContext:
        """
        更新场景上下文

        Args:
            laser_ranges: 激光雷达数据
            vision_features: 视觉特征
            imu_data: IMU数据
            audio_activity: 音频活跃度
            location_hint: 位置提示
            nearby_humans: 附近人数
            nearby_agvs: 附近AGV列表
            current_task: 当前任务
            current_location: 当前位置

        Returns:
            更新后的 SceneContext
        """
        now = time.time()
        if now - self._last_update < self.config.detection_interval:
            return self._current_context

        # 场景分类
        scene_type, confidence, features = self._classifier.classify(
            laser_ranges, vision_features, imu_data, audio_activity, location_hint
        )

        # 记录历史
        self._scene_history.append((scene_type, confidence, now))
        if len(self._scene_history) > 100:
            self._scene_history = self._scene_history[-100:]

        # 更新上下文
        self._current_context = SceneContext(
            features=features,
            current_location=current_location,
            current_task=current_task,
            nearby_agvs=nearby_agvs or [],
            nearby_humans=nearby_humans,
            timestamp=now,
        )

        # 存储场景经验到记忆
        if self._memory is not None and confidence > self.config.min_confidence_threshold:
            self._store_scene_experience(scene_type, confidence, features)

        self._last_update = now
        return self._current_context

    def _store_scene_experience(self, scene_type: SceneType, confidence: float,
                                features: SceneFeatures):
        """存储场景经验到长期记忆"""
        if self._memory is None:
            return
        try:
            self._memory.store_episode(
                summary=f"场景识别: {scene_type.value}",
                context={
                    'scene_type': scene_type.value,
                    'confidence': confidence,
                    'floor_friction': features.floor_friction,
                    'max_speed': features.max_speed_safe,
                    'obstacle_density': features.obstacle_density,
                },
                actions=[],
                outcomes={'scene_recognized': True, 'confidence': confidence},
                importance_score=confidence * 5.0,
                tags=['场景', scene_type.value, '具身'],
            )
        except Exception:
            pass

    def get_adaptive_speed_limit(self, base_speed: float) -> float:
        """获取场景自适应速度限制"""
        scene = self._current_context.features.scene_type
        max_safe = self._rule_engine.get_effective_max_speed(scene)

        # 动态调整
        if not self._current_context.is_safe_for_high_speed():
            max_safe *= 0.5

        # 附近有人时降速
        if self._current_context.nearby_humans > 0:
            max_safe = min(max_safe, 1.0)

        return min(base_speed, max_safe)

    def get_safe_distance(self) -> float:
        """获取场景自适应安全距离"""
        scene = self._current_context.features.scene_type
        safety_rules = self._rule_engine.get_applicable_safety_rules(scene)
        if safety_rules:
            return max(r.min_clearance for r in safety_rules)
        return 0.5

    def get_active_rules(self) -> Dict[type, List[SceneRule]]:
        """获取当前活跃规则"""
        return self._rule_engine.get_active_rules(
            self._current_context.features.scene_type
        )

    def get_scene_context(self) -> SceneContext:
        """获取当前场景上下文"""
        return self._current_context

    def get_scene_history(self, last_n: int = 10
                          ) -> List[Tuple[SceneType, float, float]]:
        """获取最近场景历史"""
        return self._scene_history[-last_n:]

    def recognize_scene_transition(self) -> bool:
        """检测场景转换"""
        if len(self._scene_history) < 2:
            return False
        recent = self._scene_history[-1]
        prev = self._scene_history[-2]
        return recent[0] != prev[0] and recent[1] > 0.7


# ============================================================
# 全局单例
# ============================================================

_global_instance: Optional[SceneIntelligence] = None


def get_scene_intelligence(
    config: Optional[SceneConfig] = None,
    memory: Optional["LongTermMemory"] = None,
) -> SceneIntelligence:
    """获取场景智能全局单例"""
    global _global_instance
    if _global_instance is None:
        _global_instance = SceneIntelligence(config=config, memory=memory)
    return _global_instance
