"""
instruction_grounding.py - 自然语言指令具身接地模块
SuperModel 超模态大模型具身智能系统

功能:
- 自然语言指令解析与理解
- 指令到机器人技能/动作的映射
- 空间参考解析 (left/right, front/back, near/far)
- 时间参考解析 (now, later, after X)
- 多步指令分解
- 指代消解 (pronoun resolution)
- 指令验证与安全检查

快速使用:
    from src.embodied.instruction_grounding import InstructionGroundingModule
    
    grounder = InstructionGroundingModule()
    result = grounder.ground("go to the charging station")
    print(result.skill_name, result.action_parameters)
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'SpatialReference',
    'TemporalReference', 
    'GroundingConfidence',
    'GroundingResult',
    'InstructionParser',
    'SpatialReasoner',
    'TemporalReasoner',
    'SkillMapper',
    'InstructionGroundingModule',
    'create_grounding_module',
]


# ============================================================
# 参考系统枚举
# ============================================================

class SpatialReference(Enum):
    """空间参考方向"""
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    CENTER = "center"
    NEAR = "near"
    FAR = "far"
    ABOVE = "above"
    BELOW = "below"


class TemporalReference(Enum):
    """时间参考"""
    NOW = "now"
    SOON = "soon"
    LATER = "later"
    AFTER = "after"
    BEFORE = "before"
    WHILE = "while"
    WHEN = "when"


class GroundingConfidence(Enum):
    """接地置信度"""
    HIGH = "high"        # >= 0.9
    MEDIUM = "medium"    # 0.7-0.9
    LOW = "low"          # 0.5-0.7
    UNCERTAIN = "uncertain"  # < 0.5


# ============================================================
# 接地结果数据结构
# ============================================================

@dataclass
class GroundingResult:
    """
    指令接地结果
    
    将自然语言指令映射为可执行的机器人技能和动作参数。
    """
    # 元数据
    instruction: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    confidence_level: GroundingConfidence = GroundingConfidence.HIGH
    
    # 技能映射
    skill_name: Optional[str] = None
    skill_category: Optional[str] = None
    action_type: Optional[str] = None
    
    # 动作参数
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 空间参数
    target_position: Optional[Tuple[float, float, float]] = None
    spatial_references: Dict[str, Any] = field(default_factory=dict)
    
    # 时间参数
    temporal_reference: Optional[TemporalReference] = None
    timing_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 分解后的子指令
    sub_instructions: List[str] = field(default_factory=list)
    is_compound: bool = False
    
    # 指代消解
    resolved_references: Dict[str, Any] = field(default_factory=dict)
    
    # 安全相关
    safety_flags: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    
    # 解析过程中的推理
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instruction': self.instruction,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'skill_name': self.skill_name,
            'skill_category': self.skill_category,
            'action_type': self.action_type,
            'action_parameters': self.action_parameters,
            'target_position': self.target_position,
            'spatial_references': self.spatial_references,
            'temporal_reference': self.temporal_reference.value if self.temporal_reference else None,
            'timing_parameters': self.timing_parameters,
            'sub_instructions': self.sub_instructions,
            'is_compound': self.is_compound,
            'resolved_references': self.resolved_references,
            'safety_flags': self.safety_flags,
            'requires_confirmation': self.requires_confirmation,
            'reasoning': self.reasoning,
        }


# ============================================================
# 技能映射规则库
# ============================================================

# 技能关键词映射
SKILL_KEYWORDS: Dict[str, Dict[str, Any]] = {
    # 导航类
    "navigate": {
        "keywords": ["go", "move", "navigate", "travel", "drive", "head", "前往", "移动", "导航"],
        "skill_name": "navigation",
        "category": "navigation",
        "action_type": "twist",
    },
    "goto": {
        "keywords": ["go to", "move to", "navigate to", "前往", "移动到", "导航到"],
        "skill_name": "goto_target",
        "category": "navigation",
        "action_type": "twist",
    },
    "patrol": {
        "keywords": ["patrol", "patrol around", "巡查", "巡逻"],
        "skill_name": "patrol",
        "category": "navigation",
        "action_type": "trajectory",
    },
    "follow": {
        "keywords": ["follow", "track", "跟随", "跟踪"],
        "skill_name": "follow_target",
        "category": "navigation",
        "action_type": "twist",
    },
    "dock": {
        "keywords": ["dock", "dock at", "对接", "停靠"],
        "skill_name": "dock",
        "category": "navigation",
        "action_type": "trajectory",
    },
    
    # 操作类
    "pick": {
        "keywords": ["pick", "pick up", "grab", "grasp", "抓取", "拿起", "抓"],
        "skill_name": "pick_object",
        "category": "manipulation",
        "action_type": "gripper",
    },
    "place": {
        "keywords": ["place", "put", "put down", "drop", "放下", "放置", "投放"],
        "skill_name": "place_object",
        "category": "manipulation",
        "action_type": "gripper",
    },
    "lift": {
        "keywords": ["lift", "raise", "提升", "举起", "抬起"],
        "skill_name": "lift_object",
        "category": "manipulation",
        "action_type": "gripper",
    },
    "release": {
        "keywords": ["release", "let go", "松开", "释放"],
        "skill_name": "release_object",
        "category": "manipulation",
        "action_type": "gripper",
    },
    "push": {
        "keywords": ["push", "推动", "推"],
        "skill_name": "push_object",
        "category": "manipulation",
        "action_type": "twist",
    },
    "pull": {
        "keywords": ["pull", "拉动", "拉"],
        "skill_name": "pull_object",
        "category": "manipulation",
        "action_type": "twist",
    },
    
    # 协同类
    "handover": {
        "keywords": ["handover", "hand over", "交接", "转交"],
        "skill_name": "handover",
        "category": "collaboration",
        "action_type": "gripper",
    },
    "collaborate": {
        "keywords": ["collaborate", "cooperate", "协同", "合作"],
        "skill_name": "collaborate",
        "category": "collaboration",
        "action_type": "multi_agent",
    },
    
    # 安全类
    "stop": {
        "keywords": ["stop", "halt", "暂停", "停止", "停下"],
        "skill_name": "emergency_stop",
        "category": "safety",
        "action_type": "twist",
    },
    "avoid": {
        "keywords": ["avoid", "evade", "躲避", "避开", "规避"],
        "skill_name": "avoid_obstacle",
        "category": "safety",
        "action_type": "twist",
    },
    "retreat": {
        "keywords": ["retreat", "back away", "撤退", "后退"],
        "skill_name": "retreat",
        "category": "safety",
        "action_type": "twist",
    },
    
    # 维护类
    "charge": {
        "keywords": ["charge", "recharge", "充电", "补充电量"],
        "skill_name": "go_charge",
        "category": "maintenance",
        "action_type": "navigation",
    },
    "diagnose": {
        "keywords": ["diagnose", "check", "检修", "诊断", "检查"],
        "skill_name": "self_diagnosis",
        "category": "maintenance",
        "action_type": "inspection",
    },
    
    # 感知类
    "scan": {
        "keywords": ["scan", "scan area", "scan environment", "扫描", "环境扫描"],
        "skill_name": "environment_scan",
        "category": "perception",
        "action_type": "sensors",
    },
    "inspect": {
        "keywords": ["inspect", "检查", "巡检"],
        "skill_name": "inspect_area",
        "category": "perception",
        "action_type": "sensors",
    },
    "localize": {
        "keywords": ["localize", "where am I", "定位", "确定位置"],
        "skill_name": "self_localization",
        "category": "perception",
        "action_type": "sensors",
    },
    
    # 规划类
    "plan": {
        "keywords": ["plan", "plan route", "规划", "路径规划"],
        "skill_name": "route_planning",
        "category": "planning",
        "action_type": "navigation",
    },
    "wait": {
        "keywords": ["wait", "stand by", "等待", "待机"],
        "skill_name": "wait",
        "category": "navigation",
        "action_type": "twist",
    },
}


# ============================================================
# 空间推理器
# ============================================================

class SpatialReasoner:
    """
    空间参考推理器
    
    解析自然语言中的空间参考，将相对位置转换为绝对坐标。
    支持: left/right, front/back, near/far, above/below
    """
    
    def __init__(self, robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Args:
            robot_pose: 机器人当前位置 (x, y, theta)
        """
        self.robot_pose = robot_pose  # (x, y, theta in radians)
    
    def update_pose(self, pose: Tuple[float, float, float]) -> None:
        """更新机器人位姿"""
        self.robot_pose = pose
    
    def resolve_spatial_reference(
        self,
        reference: str,
        landmark_position: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        解析空间参考，返回绝对坐标
        
        Args:
            reference: 空间参考词 (left, right, front, back, near, far 等)
            landmark_position: 地标位置 (若reference是相对于某物体)
            
        Returns:
            目标位置 (x, y, z)
        """
        rx, ry, rtheta = self.robot_pose
        ref_lower = reference.lower().strip()
        
        # 基于地标位置解析
        if landmark_position is not None:
            lx, ly, lz = landmark_position
            distance = self._get_distance_keyword(ref_lower)
            
            if "left" in ref_lower:
                # 地标左侧 = 地标位置 + 左向偏移
                return (lx - distance, ly, lz)
            elif "right" in ref_lower:
                return (lx + distance, ly, lz)
            elif "front" in ref_lower:
                return (lx, ly + distance, lz)
            elif "back" in ref_lower:
                return (lx, ly - distance, lz)
            elif "near" in ref_lower:
                return (lx, ly, lz)
            elif "far" in ref_lower:
                return (lx * 2, ly * 2, lz)
        
        # 基于机器人自身解析
        default_distance = 1.0  # 米
        
        # 根据关键词设置偏移
        if any(k in ref_lower for k in ["left", "左边", "左侧"]):
            # 机器人左侧 (世界坐标系中需要考虑朝向)
            dx = -math.sin(rtheta) * default_distance
            dy = math.cos(rtheta) * default_distance
        elif any(k in ref_lower for k in ["right", "右边", "右侧"]):
            dx = math.sin(rtheta) * default_distance
            dy = -math.cos(rtheta) * default_distance
        elif any(k in ref_lower for k in ["front", "前方", "前面", "前"]):
            dx = math.cos(rtheta) * default_distance
            dy = math.sin(rtheta) * default_distance
        elif any(k in ref_lower for k in ["back", "后方", "后面", "后"]):
            dx = -math.cos(rtheta) * default_distance
            dy = -math.sin(rtheta) * default_distance
        elif any(k in ref_lower for k in ["near", "nearby", "附近", "靠近"]):
            dx = math.cos(rtheta) * 0.5
            dy = math.sin(rtheta) * 0.5
        elif any(k in ref_lower for k in ["far", "远处", "远离"]):
            dx = math.cos(rtheta) * 3.0
            dy = math.sin(rtheta) * 3.0
        elif any(k in ref_lower for k in ["up", "上方", "上"]):
            return (rx, ry, 1.0)
        elif any(k in ref_lower for k in ["down", "下方", "下"]):
            return (rx, ry, 0.0)
        else:
            dx, dy = 0.0, 0.0
        
        return (rx + dx, ry + dy, 0.0)
    
    def _get_distance_keyword(self, reference: str) -> float:
        """从参考词中提取距离"""
        if any(k in reference for k in ["very close", "just", "紧贴"]):
            return 0.2
        elif any(k in reference for k in ["close", "near", "近"]):
            return 0.5
        elif any(k in reference for k in ["medium", "中"]):
            return 1.0
        elif any(k in reference for k in ["far", "远"]):
            return 2.0
        elif any(k in reference for k in ["very far", "很远"]):
            return 5.0
        return 1.0  # 默认1米
    
    def calculate_relative_position(
        self,
        target: Tuple[float, float, float],
    ) -> Dict[str, Union[str, float]]:
        """
        计算目标相对于机器人的位置描述
        
        Returns:
            相对位置描述字典
        """
        rx, ry, rtheta = self.robot_pose
        tx, ty, tz = target
        
        dx = tx - rx
        dy = ty - ry
        distance = math.sqrt(dx * dx + dy * dy)
        
        # 计算相对角度
        abs_angle = math.atan2(dy, dx)
        rel_angle = abs_angle - rtheta
        
        # 归一化到 [-pi, pi]
        while rel_angle > math.pi:
            rel_angle -= 2 * math.pi
        while rel_angle < -math.pi:
            rel_angle += 2 * math.pi
        
        result = {
            'distance': distance,
            'relative_angle': rel_angle,
            'is_ahead': abs(rel_angle) < math.pi / 4,
            'is_behind': abs(rel_angle) > 3 * math.pi / 4,
            'is_left': rel_angle < -math.pi / 4 and rel_angle > -3 * math.pi / 4,
            'is_right': rel_angle > math.pi / 4 and rel_angle < 3 * math.pi / 4,
        }
        
        # 方位描述
        if result['is_ahead']:
            result['direction'] = "front"
        elif result['is_behind']:
            result['direction'] = "back"
        elif result['is_left']:
            result['direction'] = "left"
        elif result['is_right']:
            result['direction'] = "right"
        else:
            result['direction'] = "diagonal"
        
        # 距离描述
        if distance < 0.5:
            result['distance_desc'] = "very close"
        elif distance < 1.5:
            result['distance_desc'] = "close"
        elif distance < 5.0:
            result['distance_desc'] = "medium distance"
        elif distance < 10.0:
            result['distance_desc'] = "far"
        else:
            result['distance_desc'] = "very far"
        
        return result


# ============================================================
# 时间推理器
# ============================================================

class TemporalReasoner:
    """
    时间参考推理器
    
    解析自然语言中的时间参考，将相对时间转换为绝对时间戳。
    支持: now, soon, later, after X, before Y, when Z
    """
    
    TIME_KEYWORDS = {
        "now": (0.0, TemporalReference.NOW),
        "immediately": (0.0, TemporalReference.NOW),
        "right now": (0.0, TemporalReference.NOW),
        "此刻": (0.0, TemporalReference.NOW),
        "马上": (1.0, TemporalReference.SOON),
        "soon": (5.0, TemporalReference.SOON),
        "shortly": (5.0, TemporalReference.SOON),
        "later": (30.0, TemporalReference.LATER),
        "以后": (30.0, TemporalReference.LATER),
        "after": (None, TemporalReference.AFTER),
        "before": (None, TemporalReference.BEFORE),
        "while": (None, TemporalReference.WHILE),
        "when": (None, TemporalReference.WHEN),
    }
    
    def __init__(self):
        self.current_time = time.time()
    
    def update_time(self) -> None:
        """更新当前时间"""
        self.current_time = time.time()
    
    def resolve_temporal_reference(
        self,
        reference: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], TemporalReference, Dict[str, Any]]:
        """
        解析时间参考
        
        Args:
            reference: 时间参考词
            context: 额外上下文 (包含事件时间等)
            
        Returns:
            (执行时间戳, 时间参考类型, 其他参数)
        """
        ref_lower = reference.lower().strip()
        context = context or {}
        
        # 查找匹配的时间关键词
        for keyword, (delta, temp_ref) in self.TIME_KEYWORDS.items():
            if keyword in ref_lower:
                if delta is not None:
                    return (self.current_time + delta, temp_ref, {'delta_s': delta})
                else:
                    # 需要从context中获取时间
                    if temp_ref == TemporalReference.AFTER:
                        event_time = context.get('after_event_time', self.current_time + 60)
                        return (event_time, temp_ref, {'event': context.get('after_event')})
                    elif temp_ref == TemporalReference.BEFORE:
                        event_time = context.get('before_event_time', self.current_time - 60)
                        return (event_time, temp_ref, {'event': context.get('before_event')})
                    elif temp_ref == TemporalReference.WHEN:
                        event_time = context.get('when_event_time', self.current_time + 30)
                        return (event_time, temp_ref, {'condition': context.get('when_condition')})
        
        # 默认立即执行
        return (self.current_time, TemporalReference.NOW, {})


# ============================================================
# 技能映射器
# ============================================================

class SkillMapper:
    """
    技能映射器
    
    将解析的指令关键词映射到具体的机器人技能。
    支持模糊匹配和优先级排序。
    """
    
    def __init__(self):
        self.skill_registry: Dict[str, Dict[str, Any]] = SKILL_KEYWORDS.copy()
        self._build_keyword_index()
    
    def _build_keyword_index(self) -> None:
        """构建关键词倒排索引"""
        self.keyword_to_skill: Dict[str, str] = {}
        for skill_id, skill_info in self.skill_registry.items():
            for keyword in skill_info.get("keywords", []):
                self.keyword_to_skill[keyword.lower()] = skill_id
    
    def register_skill(
        self,
        skill_id: str,
        skill_name: str,
        category: str,
        action_type: str,
        keywords: List[str],
    ) -> None:
        """注册自定义技能"""
        self.skill_registry[skill_id] = {
            "skill_name": skill_name,
            "category": category,
            "action_type": action_type,
            "keywords": keywords,
        }
        for keyword in keywords:
            self.keyword_to_skill[keyword.lower()] = skill_id
    
    def map_instruction_to_skill(
        self,
        instruction: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
        """
        将指令映射到技能
        
        Args:
            instruction: 自然语言指令
            
        Returns:
            (技能ID, 技能信息, 置信度)
        """
        instruction_lower = instruction.lower()
        
        # 精确关键词匹配 - 按关键词长度降序 (优先最长匹配)
        matches: List[Tuple[str, str, float]] = []  # (keyword, skill_id, confidence)
        for keyword, skill_id in self.keyword_to_skill.items():
            if keyword in instruction_lower:
                skill_info = self.skill_registry[skill_id]
                confidence = self._calculate_keyword_confidence(
                    keyword, instruction_lower
                )
                matches.append((keyword, skill_id, confidence))
        
        if matches:
            # 选择置信度最高的匹配
            best = max(matches, key=lambda x: (x[2], len(x[0])))
            skill_info = self.skill_registry[best[1]]
            return (best[1], skill_info, best[2])
        
        # 模糊匹配 (最长公共子串) - 仅在精确匹配失败时
        best_match = None
        best_score = 0.0
        
        for keyword in self.keyword_to_skill.keys():
            score = self._fuzzy_match_score(keyword, instruction_lower)
            if score > best_score and score > 0.65:  # 高阈值避免误匹配
                best_score = score
                best_match = keyword
        
        if best_match:
            skill_id = self.keyword_to_skill[best_match]
            skill_info = self.skill_registry[skill_id]
            return (skill_id, skill_info, best_score)
        
        return (None, None, 0.0)
    
    def _calculate_keyword_confidence(
        self,
        keyword: str,
        instruction: str,
    ) -> float:
        """计算关键词匹配置信度"""
        # 完整匹配
        if keyword == instruction:
            return 1.0
        
        # 开头匹配 (关键词越长越精确)
        if instruction.startswith(keyword):
            # 更长的开头匹配更精确
            length_factor = len(keyword) / max(len(instruction), 1)
            return 0.85 + 0.1 * length_factor
        
        # 包含匹配 (关键词在指令中间)
        if keyword in instruction:
            length_factor = len(keyword) / max(len(instruction), 1)
            return 0.6 + 0.2 * length_factor
        
        return 0.2
    
    def _fuzzy_match_score(self, keyword: str, text: str) -> float:
        """
        模糊匹配得分 (基于编辑距离/子串匹配)
        
        策略:
        1. 子串匹配: 检查关键词是否作为子串出现在文本中 (最强信号)
        2. 编辑距离: 计算关键词到文本中连续片段的最小编辑距离
        3. 只有得分 >= 0.7 才算匹配 (避免随机字符串误匹配)
        """
        if not keyword or not text:
            return 0.0
        
        # 子串匹配 - 最强信号
        if keyword in text:
            return 1.0
        
        # 短关键词 (< 5字符) 不做模糊匹配
        if len(keyword) < 5:
            return 0.0
        
        # 编辑距离匹配 - 要求关键词与文本某处足够接近
        # 简化方法: 检查关键词中连续字符序列在文本中的覆盖
        max_edit_distance = int(len(keyword) * 0.25)  # 最多允许25%的编辑距离
        
        # 滑动窗口检查关键词与text子串的编辑距离
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        best_distance = float('inf')
        window_size = len(keyword_lower)
        
        for i in range(len(text_lower) - window_size + 1):
            window = text_lower[i:i + window_size]
            # 计算编辑距离(简化版)
            distance = sum(1 for a, b in zip(window, keyword_lower) if a != b)
            best_distance = min(best_distance, distance)
        
        # 如果最小编辑距离在容忍范围内
        if best_distance <= max_edit_distance:
            # 更长的关键词得分更高
            length_factor = min(1.0, len(keyword) / 8.0)
            return 0.7 + 0.2 * length_factor * (1.0 - best_distance / max(len(keyword), 1))
        
        return 0.0


# ============================================================
# 指令解析器
# ============================================================

class InstructionParser:
    """
    自然语言指令解析器
    
    功能:
    - 指令分词与词性标注
    - 复合指令分解 (and, then, after, before)
    - 目标提取
    - 约束条件提取
    - 安全关键词检测
    """
    
    # 指令分隔符
    COMPOUND_SEPARATORS = [
        r'\s+and\s+',           # "go forward and turn left"
        r'\s+then\s+',          # "pick up then place"
        r'\s+after\s+',         # "go there after charging"
        r'\s+before\s+',        # "go home before sunset"
        r'\s+while\s+',         # "move while scanning"
        r',\s*',                # "go to A, then B"
        r';\s*',                # "go to A; then B"
    ]
    
    # 安全关键词 (高优先级)
    SAFETY_KEYWORDS = [
        "careful", "caution", "carefully", "slowly", "gentle", "gentl",
        "danger", "dangerous", "hazard", "warning",
        "stop", "emergency", "halt",
        "安全", "小心", "注意", "谨慎",
    ]
    
    # 确认需求关键词
    CONFIRM_KEYWORDS = [
        "are you sure", "confirm", "请确认", "确认",
        "proceed with", "continue with",
        "execute", "run", "start",
    ]
    
    def __init__(self):
        self.compound_pattern = '|'.join(self.COMPOUND_SEPARATORS)
    
    def parse(self, instruction: str) -> GroundingResult:
        """
        解析自然语言指令
        
        Args:
            instruction: 原始指令文本
            
        Returns:
            GroundingResult
        """
        result = GroundingResult(instruction=instruction)
        
        # 基础清理
        instruction = instruction.strip()
        if not instruction:
            result.confidence = 0.0
            result.confidence_level = GroundingConfidence.UNCERTAIN
            return result
        
        # 检测复合指令
        sub_instructions = self._split_compound(instruction)
        if len(sub_instructions) > 1:
            result.is_compound = True
            result.sub_instructions = sub_instructions
            result.reasoning = f"Split into {len(sub_instructions)} sub-instructions"
        
        # 安全检查
        safety_flags = self._check_safety_keywords(instruction)
        result.safety_flags = safety_flags
        
        # 确认需求检查
        result.requires_confirmation = self._requires_confirmation(instruction)
        
        # 目标提取
        target_info = self._extract_target(instruction)
        result.target_position = target_info.get('position')
        result.action_parameters.update(target_info.get('params', {}))
        
        # 约束提取
        constraints = self._extract_constraints(instruction)
        result.action_parameters['constraints'] = constraints
        
        return result
    
    def _split_compound(self, instruction: str) -> List[str]:
        """分解复合指令"""
        parts = re.split(self.compound_pattern, instruction)
        return [p.strip() for p in parts if p.strip()]
    
    def _check_safety_keywords(self, instruction: str) -> List[str]:
        """检测安全关键词"""
        instruction_lower = instruction.lower()
        found = []
        for keyword in self.SAFETY_KEYWORDS:
            if keyword in instruction_lower:
                found.append(keyword)
        return found
    
    def _requires_confirmation(self, instruction: str) -> bool:
        """检查是否需要确认"""
        instruction_lower = instruction.lower()
        return any(kw in instruction_lower for kw in self.CONFIRM_KEYWORDS)
    
    def _extract_target(self, instruction: str) -> Dict[str, Any]:
        """
        提取目标信息
        
        解析目标位置、物体、目的地等。
        """
        info: Dict[str, Any] = {'params': {}}
        instruction_lower = instruction.lower()
        
        instruction_lower = instruction.lower()
        
        # 位置关键词 - 按特异性降序排列
        # 先检查具体名称，再检查通用关键词
        station_match = re.search(r'station\s+([a-z])', instruction_lower)
        if station_match:
            letter = station_match.group(1)
            info['params']['target_name'] = f'station_{letter.upper()}'
            info['position'] = self._get_position_from_name(f'station_{letter.upper()}')
            return info
        
        position_keywords = [
            ("charging", "charging_station"),
            ("unloading", "unloading_bay"),
            ("loading", "loading_bay"),
            ("storage", "storage_area"),
            ("warehouse", "warehouse_A"),
            ("home", "home_base"),
            ("station", "station_A"),
        ]
        
        for keyword, position_name in position_keywords:
            if keyword in instruction_lower:
                info['params']['target_name'] = position_name
                info['position'] = self._get_position_from_name(position_name)
                break
        
        # 数字距离
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|米|meter)', instruction_lower)
        if distance_match:
            info['params']['distance'] = float(distance_match.group(1))
        
        # 速度约束
        speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m/s|km/h)', instruction_lower)
        if speed_match:
            info['params']['speed'] = float(speed_match.group(1))
        
        return info
    
    def _extract_constraints(self, instruction: str) -> Dict[str, Any]:
        """提取约束条件"""
        constraints: Dict[str, Any] = {}
        instruction_lower = instruction.lower()
        
        # 速度约束
        if "slow" in instruction_lower or "慢慢" in instruction_lower:
            constraints['max_speed'] = 0.3
        if "fast" in instruction_lower or "快" in instruction_lower:
            constraints['max_speed'] = 2.0
        
        # 精度约束
        if "precise" in instruction_lower or "精确" in instruction_lower:
            constraints['precision_required'] = 0.01
        
        # 安全距离
        clearance_match = re.search(r'clearance\s+(\d+(?:\.\d+)?)', instruction_lower)
        if clearance_match:
            constraints['min_clearance'] = float(clearance_match.group(1))
        
        return constraints
    
    def _get_position_from_name(self, name: str) -> Optional[Tuple[float, float, float]]:
        """根据名称获取预定义位置"""
        POSITIONS = {
            "station_A": (5.0, 0.0, 0.0),
            "station_B": (10.0, 0.0, 0.0),
            "charging_station": (0.0, 0.0, 0.0),
            "home_base": (0.0, 0.0, 0.0),
            "warehouse_A": (10.0, 0.0, 0.0),
            "loading_bay": (8.0, 0.0, 0.0),
            "unloading_bay": (12.0, 0.0, 0.0),
            "storage_area": (15.0, 0.0, 0.0),
        }
        return POSITIONS.get(name)


# ============================================================
# 指令接地模块 (主类)
# ============================================================

class InstructionGroundingModule:
    """
    自然语言指令接地模块 - 核心类
    
    将自然语言指令转换为机器人可执行的技能和动作参数。
    
    架构:
    ┌─────────────────────────────────────────────────────────┐
    │              InstructionGroundingModule                 │
    │  ┌─────────────────┐  ┌────────────────────────────┐  │
    │  │ InstructionParser │  │ SkillMapper                 │  │
    │  │ - parse()        │  │ - map_instruction_to_skill()│  │
    │  │ - split_compound │  │ - register_skill()          │  │
    │  └─────────────────┘  └────────────────────────────┘  │
    │  ┌─────────────────┐  ┌────────────────────────────┐  │
    │  │ SpatialReasoner  │  │ TemporalReasoner           │  │
    │  │ - resolve_ref()  │  │ - resolve_temporal_ref()   │  │
    │  │ - calc_relative()│  │                            │  │
    │  └─────────────────┘  └────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
    
    使用示例:
        grounder = InstructionGroundingModule()
        
        # 简单指令
        result = grounder.ground("go to station A")
        print(f"Skill: {result.skill_name}, Params: {result.action_parameters}")
        
        # 带空间参考
        result = grounder.ground("turn left and go forward 2 meters")
        
        # 复合指令
        result = grounder.ground("pick up the box then place it on the table")
    """
    
    def __init__(
        self,
        robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        known_landmarks: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        """
        Args:
            robot_pose: 机器人初始位姿 (x, y, theta)
            known_landmarks: 已知地标字典 {name: (x, y, z)}
        """
        self.parser = InstructionParser()
        self.skill_mapper = SkillMapper()
        self.spatial_reasoner = SpatialReasoner(robot_pose)
        self.temporal_reasoner = TemporalReasoner()
        self.known_landmarks = known_landmarks or {}
        
        # 历史记录 (用于指代消解)
        self.recent_instructions: List[GroundingResult] = []
        self.recent_targets: List[Tuple[float, float, float]] = []
    
    def ground(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> GroundingResult:
        """
        将自然语言指令接地到机器人动作
        
        Args:
            instruction: 自然语言指令
            context: 额外上下文 (机器人状态、场景信息等)
            
        Returns:
            GroundingResult - 包含技能名称、动作参数等
        """
        context = context or {}
        
        # 更新内部状态
        if 'robot_pose' in context:
            self.spatial_reasoner.update_pose(context['robot_pose'])
        
        if 'known_landmarks' in context:
            self.known_landmarks.update(context['known_landmarks'])
        
        # 1. 解析指令
        result = self.parser.parse(instruction)
        
        # 2. 技能映射
        skill_id, skill_info, confidence = self.skill_mapper.map_instruction_to_skill(
            instruction
        )
        
        if skill_id:
            result.skill_name = skill_info.get("skill_name")
            result.skill_category = skill_info.get("category")
            result.action_type = skill_info.get("action_type")
            result.confidence = confidence
            result.confidence_level = self._confidence_to_level(confidence)
        else:
            result.confidence = 0.0
            result.confidence_level = GroundingConfidence.UNCERTAIN
            result.reasoning = "No matching skill found"
            return result
        
        # 3. 空间参考解析
        spatial_refs = self._extract_spatial_references(instruction)
        if spatial_refs:
            result.spatial_references = spatial_refs
            # 更新目标位置
            if 'target' in spatial_refs:
                target_pos = self.spatial_reasoner.resolve_spatial_reference(
                    spatial_refs['target']
                )
                result.target_position = target_pos
        
        # 4. 时间参考解析
        temporal_ref = self._extract_temporal_reference(instruction)
        if temporal_ref:
            exec_time, temp_ref, timing_params = self.temporal_reasoner.resolve_temporal_reference(
                temporal_ref, context
            )
            result.temporal_reference = temp_ref
            result.timing_parameters = timing_params
            if exec_time is not None:
                result.action_parameters['execution_time'] = exec_time
        
        # 5. 指代消解
        resolved = self._resolve_references(instruction, context)
        result.resolved_references = resolved
        
        # 6. 安全检查增强
        if result.safety_flags:
            result.action_parameters['safety_mode'] = 'heightened'
            result.confidence *= 0.9  # 有安全关键词，稍降低置信度
        
        # 7. 生成推理描述
        result.reasoning = self._generate_reasoning(result, instruction)
        
        # 记录到历史
        self._add_to_history(result)
        
        return result
    
    def ground_batch(
        self,
        instructions: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[GroundingResult]:
        """批量接地多个指令"""
        return [self.ground(inst, context) for inst in instructions]
    
    def ground_compound(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[GroundingResult]:
        """
        接地复合指令，返回分解后的子指令接地结果列表
        
        Args:
            instruction: 复合指令 (包含 and, then 等)
            context: 上下文
            
        Returns:
            子指令接地结果列表
        """
        sub_instructions = self.parser._split_compound(instruction)
        return [self.ground(sub, context) for sub in sub_instructions]
    
    def update_robot_pose(self, pose: Tuple[float, float, float]) -> None:
        """更新机器人位姿"""
        self.spatial_reasoner.update_pose(pose)
    
    def add_landmark(
        self,
        name: str,
        position: Tuple[float, float, float],
    ) -> None:
        """添加已知地标"""
        self.known_landmarks[name] = position
    
    def get_recent_target(self) -> Optional[Tuple[float, float, float]]:
        """获取最近的目标位置 (用于指代消解)"""
        return self.recent_targets[-1] if self.recent_targets else None
    
    def _extract_spatial_references(self, instruction: str) -> Dict[str, Any]:
        """从指令中提取空间参考"""
        refs: Dict[str, Any] = {}
        instruction_lower = instruction.lower()
        
        # 方向关键词
        direction_keywords = [
            ("left", "left"),
            ("right", "right"),
            ("front", "front"),
            ("back", "back"),
            ("forward", "front"),
            ("backward", "back"),
        ]
        
        for keyword, direction in direction_keywords:
            if keyword in instruction_lower:
                refs['direction'] = direction
                refs['target'] = direction
                break
        
        # 距离
        distance_match = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|米)', instruction_lower
        )
        if distance_match:
            refs['distance'] = float(distance_match.group(1))
        
        # 预定义目的地
        for name in self.known_landmarks:
            if name in instruction_lower:
                refs['named_target'] = name
                refs['target_position'] = self.known_landmarks[name]
                break
        
        return refs
    
    def _extract_temporal_reference(self, instruction: str) -> Optional[str]:
        """从指令中提取时间参考"""
        instruction_lower = instruction.lower()
        for keyword in TemporalReasoner.TIME_KEYWORDS.keys():
            if keyword in instruction_lower:
                return keyword
        return None
    
    def _resolve_references(
        self,
        instruction: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        指代消解
        
        解析代词和省略的目标引用:
        - "it", "that", "them" -> 之前提到的物体
        - "there", "here" -> 之前提到的位置
        """
        resolved: Dict[str, Any] = {}
        instruction_lower = instruction.lower()
        
        # 代词消解
        pronouns = {
            "it": "recent_object",
            "that": "recent_object",
            "them": "recent_objects",
            "there": "recent_location",
            "here": "current_location",
        }
        
        for pronoun, referent in pronouns.items():
            if pronoun in instruction_lower and self.recent_instructions:
                if referent == "recent_object":
                    resolved[pronoun] = self.recent_targets[-1] if self.recent_targets else None
                elif referent == "recent_location":
                    resolved[pronoun] = self.get_recent_target()
        
        # 上下文中的显式引用
        if 'target_object' in context:
            resolved['explicit_target'] = context['target_object']
        if 'target_location' in context:
            resolved['explicit_location'] = context['target_location']
        
        return resolved
    
    def _confidence_to_level(self, confidence: float) -> GroundingConfidence:
        """将置信度数值转换为等级枚举"""
        if confidence >= 0.9:
            return GroundingConfidence.HIGH
        elif confidence >= 0.7:
            return GroundingConfidence.MEDIUM
        elif confidence >= 0.5:
            return GroundingConfidence.LOW
        else:
            return GroundingConfidence.UNCERTAIN
    
    def _generate_reasoning(self, result: GroundingResult, instruction: str) -> str:
        """生成推理描述"""
        parts = []
        
        if result.skill_name:
            parts.append(f"识别为技能 '{result.skill_name}'")
        
        if result.target_position:
            x, y, z = result.target_position
            parts.append(f"目标位置: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        if result.spatial_references:
            if 'direction' in result.spatial_references:
                parts.append(f"方向: {result.spatial_references['direction']}")
            if 'distance' in result.spatial_references:
                parts.append(f"距离: {result.spatial_references['distance']}m")
        
        if result.temporal_reference:
            parts.append(f"时间: {result.temporal_reference.value}")
        
        if result.safety_flags:
            parts.append(f"安全标记: {', '.join(result.safety_flags)}")
        
        if result.is_compound:
            parts.append(f"复合指令 ({len(result.sub_instructions)} 步)")
        
        return "; ".join(parts) if parts else "无法解析"
    
    def _add_to_history(self, result: GroundingResult) -> None:
        """添加到历史记录"""
        self.recent_instructions.append(result)
        if result.target_position:
            self.recent_targets.append(result.target_position)
        
        # 限制历史长度
        max_history = 10
        if len(self.recent_instructions) > max_history:
            self.recent_instructions = self.recent_instructions[-max_history:]
        if len(self.recent_targets) > max_history:
            self.recent_targets = self.recent_targets[-max_history:]


# ============================================================
# 工厂函数
# ============================================================

def create_grounding_module(
    grade: str = "M",
    robot_pose: Optional[Tuple[float, float, float]] = None,
) -> InstructionGroundingModule:
    """
    创建指令接地模块 (根据AGV等级配置)
    
    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        robot_pose: 初始机器人位姿
        
    Returns:
        配置好的 InstructionGroundingModule
    """
    if robot_pose is None:
        robot_pose = (0.0, 0.0, 0.0)
    
    grounder = InstructionGroundingModule(robot_pose=robot_pose)
    
    # 根据等级注册额外技能
    if grade in ("L", "XL", "XXL"):
        # 高级AGV支持更多技能
        grounder.skill_mapper.register_skill(
            "precise_place",
            "precise_place",
            "manipulation",
            "gripper",
            ["place precisely", "精确定位放置", "精准放置"],
        )
        grounder.skill_mapper.register_skill(
            "force_control",
            "force_control",
            "manipulation",
            "force",
            ["apply force", "施加力", "用力控制"],
        )
    
    return grounder
