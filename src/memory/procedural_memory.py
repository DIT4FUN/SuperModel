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
Procedural Memory - 程序记忆模块
=================================

存储和管理技能、习惯、流程。

程序记忆特点:
- 技能封装: 完整的动作序列
- 条件触发: 在特定条件下执行
- 熟练度: 技能的掌握程度
- 可组合性: 技能可以组合
- 层级结构: 原语技能 -> 复合技能 -> 任务
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import json
import uuid
import time


class SkillLevel(Enum):
    """技能等级"""
    NOVICE = 1       # 新手 - 需要专注
    BEGINNER = 2     # 初学者 - 能执行
    COMPETENT = 3    # 胜任 - 能独立完成
    PROFICIENT = 4   # 熟练 - 高效完成
    EXPERT = 5       # 专家 - 自动化执行


@dataclass
class SkillPrerequisite:
    """技能前置条件"""
    skill_id: str
    min_level: SkillLevel = SkillLevel.BEGINNER


@dataclass
class Skill:
    """
    技能/程序单元
    
    Attributes:
        id: 唯一标识符
        name: 技能名称
        description: 描述
        category: 类别 (navigation, manipulation, communication, etc.)
        procedure_type: 程序类型
        steps: 步骤列表 (如果是流程型)
        code_reference: 代码引用 (如果是代码型)
        conditions: 触发条件
        prerequisites: 前置技能
        level: 当前等级
        experience_points: 经验值
        success_count: 成功次数
        failure_count: 失败次数
        last_used: 上次使用时间
        avg_duration_s: 平均执行时间
        source_episode_id: 来源记忆
        applicable_contexts: 适用场景
        incompatible_skills: 不兼容技能
        tags: 标签
        metadata: 元数据
    """
    id: str
    name: str
    description: str = ""
    category: str = "general"
    procedure_type: str = "流程"  # 流程/代码/混合
    steps: List[Dict[str, Any]] = field(default_factory=list)
    code_reference: Optional[str] = None  # 代码文件/函数引用
    conditions: List[str] = field(default_factory=list)  # 触发条件描述
    prerequisites: List[SkillPrerequisite] = field(default_factory=list)
    level: SkillLevel = SkillLevel.NOVICE
    experience_points: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_used: float = field(default_factory=time.time)
    total_usage_time_s: float = 0.0
    avg_duration_s: float = 0.0
    source_episode_id: Optional[str] = None
    applicable_contexts: List[str] = field(default_factory=list)
    incompatible_skills: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def xp_to_next_level(self) -> float:
        """到下一级所需经验"""
        level_xp = {
            SkillLevel.NOVICE: 100,
            SkillLevel.BEGINNER: 500,
            SkillLevel.COMPETENT: 2000,
            SkillLevel.PROFICIENT: 5000,
            SkillLevel.EXPERT: 10000,
        }
        current_threshold = level_xp.get(self.level, 0)
        next_threshold = level_xp.get(SkillLevel(self.level.value + 1), float('inf'))
        return max(0, next_threshold - self.experience_points)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'procedure_type': self.procedure_type,
            'steps': self.steps,
            'code_reference': self.code_reference,
            'conditions': self.conditions,
            'prerequisites': [
                {'skill_id': p.skill_id, 'min_level': p.min_level.value}
                for p in self.prerequisites
            ],
            'level': self.level.value,
            'experience_points': self.experience_points,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'last_used': self.last_used,
            'total_usage_time_s': self.total_usage_time_s,
            'avg_duration_s': self.avg_duration_s,
            'source_episode_id': self.source_episode_id,
            'applicable_contexts': self.applicable_contexts,
            'incompatible_skills': self.incompatible_skills,
            'tags': self.tags,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Skill:
        prerequisites = [
            SkillPrerequisite(
                skill_id=p['skill_id'],
                min_level=SkillLevel(p['min_level'])
            )
            for p in data.get('prerequisites', [])
        ]
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            procedure_type=data.get('procedure_type', '流程'),
            steps=data.get('steps', []),
            code_reference=data.get('code_reference'),
            conditions=data.get('conditions', []),
            prerequisites=prerequisites,
            level=SkillLevel(data.get('level', 1)),
            experience_points=data.get('experience_points', 0.0),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0),
            last_used=data.get('last_used', time.time()),
            total_usage_time_s=data.get('total_usage_time_s', 0.0),
            avg_duration_s=data.get('avg_duration_s', 0.0),
            source_episode_id=data.get('source_episode_id'),
            applicable_contexts=data.get('applicable_contexts', []),
            incompatible_skills=data.get('incompatible_skills', []),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
        )


class ProceduralMemory:
    """
    程序记忆管理器
    
    负责:
    - 技能存储和检索
    - 技能熟练度管理
    - 技能学习和发展
    - 技能组合和分解
    - 流程执行追踪
    """
    
    def __init__(
        self,
        store_path: Optional[str] = None,
    ):
        self.store_path = store_path
        
        # 存储
        self._skills: Dict[str, Skill] = {}
        self._skills_by_category: Dict[str, List[str]] = {}
        self._skills_by_context: Dict[str, List[str]] = {}  # context -> skill_ids
        
        # 技能执行追踪
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        
        # 统计
        self._total_skills = 0
        
        if store_path:
            self._load()
    
    # ==================== 技能管理 ====================
    
    def add_skill(
        self,
        name: str,
        description: str = "",
        category: str = "general",
        procedure_type: str = "流程",
        steps: Optional[List[Dict[str, Any]]] = None,
        code_reference: Optional[str] = None,
        conditions: Optional[List[str]] = None,
        applicable_contexts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source_episode_id: Optional[str] = None,
    ) -> Skill:
        """
        添加新技能
        
        Returns:
            创建的Skill对象
        """
        skill_id = str(uuid.uuid4())
        
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            category=category,
            procedure_type=procedure_type,
            steps=steps or [],
            code_reference=code_reference,
            conditions=conditions or [],
            applicable_contexts=applicable_contexts or [],
            tags=tags or [],
            source_episode_id=source_episode_id,
        )
        
        self._add_skill(skill)
        
        return skill
    
    def _add_skill(self, skill: Skill) -> None:
        """内部: 添加技能"""
        self._skills[skill.id] = skill
        
        # 按类别索引
        if skill.category not in self._skills_by_category:
            self._skills_by_category[skill.category] = []
        self._skills_by_category[skill.category].append(skill.id)
        
        # 按上下文索引
        for ctx in skill.applicable_contexts:
            if ctx not in self._skills_by_context:
                self._skills_by_context[ctx] = []
            self._skills_by_context[ctx].append(skill.id)
        
        self._total_skills += 1
        
        if self.store_path:
            self._save()
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """获取技能"""
        return self._skills.get(skill_id)
    
    def find_skill_by_name(self, name: str) -> Optional[Skill]:
        """按名称查找技能"""
        for skill in self._skills.values():
            if skill.name == name:
                return skill
        return None
    
    def get_skills_by_category(self, category: str) -> List[Skill]:
        """获取类别的所有技能"""
        skill_ids = self._skills_by_category.get(category, [])
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]
    
    def get_skills_for_context(self, context: str) -> List[Skill]:
        """获取适用场景的技能"""
        skill_ids = self._skills_by_context.get(context, [])
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]
    
    def update_skill(
        self,
        skill_id: str,
        success: bool,
        duration_s: float,
        feedback: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        更新技能熟练度
        
        Args:
            skill_id: 技能ID
            success: 是否成功
            duration_s: 执行时间
            feedback: 反馈
            
        Returns:
            更新后的技能
        """
        skill = self._skills.get(skill_id)
        if not skill:
            return None
        
        # 更新统计
        if success:
            skill.success_count += 1
        else:
            skill.failure_count += 1
        
        # 更新时间统计
        skill.total_usage_time_s += duration_s
        skill.avg_duration_s = skill.total_usage_time_s / (skill.success_count + skill.failure_count)
        skill.last_used = time.time()
        
        # 经验值更新
        base_xp = 10 if success else 2
        # 根据执行效率调整
        if skill.avg_duration_s > 0:
            efficiency = min(2.0, skill.avg_duration_s / max(duration_s, 0.1))
            xp_multiplier = 1.0 / efficiency
        else:
            xp_multiplier = 1.0
        
        skill.experience_points += base_xp * xp_multiplier
        
        # 检查升级
        self._check_level_up(skill)
        
        # 元数据
        if feedback:
            skill.metadata['last_feedback'] = feedback
        
        skill.updated_at = time.time()
        
        if self.store_path:
            self._save()
        
        return skill
    
    def _check_level_up(self, skill: Skill) -> None:
        """检查并处理升级"""
        level_thresholds = {
            1: 0,       # NOVICE
            2: 100,     # BEGINNER
            3: 500,     # COMPETENT
            4: 2000,    # PROFICIENT
            5: 5000,    # EXPERT
        }
        
        xp = skill.experience_points
        
        if xp >= 5000 and skill.level != SkillLevel.EXPERT:
            skill.level = SkillLevel.EXPERT
        elif xp >= 2000 and skill.level.value < 4:
            skill.level = SkillLevel.PROFICIENT
        elif xp >= 500 and skill.level.value < 3:
            skill.level = SkillLevel.COMPETENT
        elif xp >= 100 and skill.level.value < 2:
            skill.level = SkillLevel.BEGINNER
    
    def add_prerequisite(
        self,
        skill_id: str,
        prerequisite_id: str,
        min_level: SkillLevel = SkillLevel.BEGINNER,
    ) -> bool:
        """添加前置技能"""
        skill = self._skills.get(skill_id)
        prereq_skill = self._skills.get(prerequisite_id)
        
        if not skill or not prereq_skill:
            return False
        
        # 检查是否已存在
        for p in skill.prerequisites:
            if p.skill_id == prerequisite_id:
                return False
        
        skill.prerequisites.append(SkillPrerequisite(
            skill_id=prerequisite_id,
            min_level=min_level,
        ))
        
        skill.updated_at = time.time()
        return True
    
    def check_prerequisites(self, skill_id: str) -> Tuple[bool, List[str]]:
        """
        检查前置技能是否满足
        
        Returns:
            (是否满足, 不满足的前置列表)
        """
        skill = self._skills.get(skill_id)
        if not skill:
            return False, []
        
        unsatisfied = []
        
        for prereq in skill.prerequisites:
            prereq_skill = self._skills.get(prereq.skill_id)
            if not prereq_skill:
                unsatisfied.append(f"{prereq.skill_id} (技能不存在)")
            elif prereq_skill.level.value < prereq.min_level.value:
                unsatisfied.append(
                    f"{prereq_skill.name} (当前: {prereq_skill.level.name}, 需要: {prereq.min_level.name})"
                )
        
        return len(unsatisfied) == 0, unsatisfied
    
    # ==================== 技能检索 ====================
    
    def search_skills(
        self,
        query: str,
        category: Optional[str] = None,
        min_level: Optional[SkillLevel] = None,
        limit: int = 20,
    ) -> List[Skill]:
        """
        搜索技能
        
        Args:
            query: 查询字符串
            category: 限定类别
            min_level: 最低等级
            limit: 返回数量
            
        Returns:
            匹配的技能列表
        """
        query_lower = query.lower()
        results = []
        
        for skill in self._skills.values():
            if category and skill.category != category:
                continue
            
            if min_level and skill.level.value < min_level.value:
                continue
            
            # 名称匹配
            if query_lower in skill.name.lower():
                results.append(skill)
                continue
            
            # 标签匹配
            if any(query_lower in tag.lower() for tag in skill.tags):
                results.append(skill)
            
            # 描述匹配
            if query_lower in skill.description.lower():
                results.append(skill)
        
        # 按成功率排序
        results.sort(key=lambda s: s.success_rate, reverse=True)
        
        return results[:limit]
    
    def get_best_skill_for_task(
        self,
        task_description: str,
        context: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        找到最适合任务的技能
        
        Args:
            task_description: 任务描述
            context: 当前上下文
            
        Returns:
            最佳匹配技能
        """
        candidates = []
        
        for skill in self._skills.values():
            # 检查前置
            satisfied, _ = self.check_prerequisites(skill.id)
            if not satisfied:
                continue
            
            # 计算匹配度
            score = 0.0
            
            # 名称匹配
            if any(word in skill.name.lower() for word in task_description.lower().split()):
                score += 0.4
            
            # 标签匹配
            for tag in skill.tags:
                if tag in task_description.lower():
                    score += 0.2
            
            # 上下文匹配
            if context and context in skill.applicable_contexts:
                score += 0.3
            
            # 成功率加权
            score += skill.success_rate * 0.1
            
            if score > 0:
                candidates.append((skill, score))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    # ==================== 技能组合 ====================
    
    def compose_skills(
        self,
        skill_ids: List[str],
        name: str,
        description: str = "",
    ) -> Optional[Skill]:
        """
        组合多个技能为复合技能
        
        Returns:
            创建的复合技能
        """
        skills = [self._skills[sid] for sid in skill_ids if sid in self._skills]
        if not skills:
            return None
        
        # 合并步骤
        all_steps = []
        for skill in skills:
            all_steps.extend(skill.steps)
        
        # 合并适用上下文
        all_contexts = []
        for skill in skills:
            for ctx in skill.applicable_contexts:
                if ctx not in all_contexts:
                    all_contexts.append(ctx)
        
        # 合并标签
        all_tags = []
        for skill in skills:
            for tag in skill.tags:
                if tag not in all_tags:
                    all_tags.append(tag)
        
        composite = self.add_skill(
            name=name,
            description=description,
            category="composite",
            procedure_type="流程",
            steps=all_steps,
            applicable_contexts=all_contexts,
            tags=["composite"] + all_tags[:5],
        )
        
        # 设置前置技能
        for skill in skills:
            composite.prerequisites.append(SkillPrerequisite(
                skill_id=skill.id,
                min_level=SkillLevel.BEGINNER,
            ))
        
        return composite
    
    def decompose_skill(self, skill_id: str) -> List[Skill]:
        """
        分解技能为子技能
        
        Returns:
            子技能列表
        """
        skill = self._skills.get(skill_id)
        if not skill:
            return []
        
        # 简单实现: 按步骤分解
        sub_skills = []
        
        # 如果技能是流程型且有很多步骤，可以按子流程分解
        if skill.procedure_type == "流程" and len(skill.steps) > 5:
            # 每3-5个步骤作为一个子技能
            step_groups = [
                skill.steps[i:i+4]
                for i in range(0, len(skill.steps), 4)
            ]
            
            for i, group in enumerate(step_groups):
                sub = self.add_skill(
                    name=f"{skill.name}_part{i+1}",
                    description=f"子技能: {skill.name} 第{i+1}部分",
                    category=skill.category,
                    procedure_type="流程",
                    steps=group,
                    tags=[f"part_{i+1}"],
                )
                sub_skills.append(sub)
        
        return sub_skills
    
    # ==================== 技能执行追踪 ====================
    
    def start_execution(
        self,
        skill_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        开始技能执行
        
        Returns:
            execution_id
        """
        skill = self._skills.get(skill_id)
        if not skill:
            return None
        
        execution_id = str(uuid.uuid4())
        
        self._active_executions[execution_id] = {
            'skill_id': skill_id,
            'start_time': time.time(),
            'context': context or {},
            'current_step': 0,
            'status': 'running',
        }
        
        return execution_id
    
    def update_execution(
        self,
        execution_id: str,
        step: int,
    ) -> bool:
        """更新执行进度"""
        exec_data = self._active_executions.get(execution_id)
        if not exec_data:
            return False
        
        exec_data['current_step'] = step
        return True
    
    def complete_execution(
        self,
        execution_id: str,
        success: bool,
        feedback: Optional[str] = None,
    ) -> bool:
        """完成技能执行"""
        exec_data = self._active_executions.get(execution_id)
        if not exec_data:
            return False
        
        duration = time.time() - exec_data['start_time']
        
        # 更新技能熟练度
        self.update_skill(
            exec_data['skill_id'],
            success=success,
            duration_s=duration,
            feedback=feedback,
        )
        
        exec_data['status'] = 'completed'
        exec_data['success'] = success
        exec_data['duration'] = duration
        
        return True
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """获取活跃的执行"""
        return [
            {
                'execution_id': eid,
                'skill_id': e['skill_id'],
                'skill_name': self._skills.get(e['skill_id']).name if self._skills.get(e['skill_id']) else 'unknown',
                'elapsed_s': time.time() - e['start_time'],
                'current_step': e['current_step'],
            }
            for eid, e in self._active_executions.items()
            if e['status'] == 'running'
        ]
    
    # ==================== 统计和持久化 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        skills = list(self._skills.values())
        
        if not skills:
            return {'total_skills': 0}
        
        level_dist = {}
        for level in SkillLevel:
            level_dist[level.name] = sum(1 for s in skills if s.level == level)
        
        return {
            'total_skills': len(skills),
            'by_category': {
                cat: len(sids)
                for cat, sids in self._skills_by_category.items()
            },
            'by_level': level_dist,
            'avg_success_rate': np.mean([s.success_rate for s in skills]),
            'total_executions': sum(s.success_count + s.failure_count for s in skills),
            'active_executions': len([
                e for e in self._active_executions.values()
                if e['status'] == 'running'
            ]),
        }
    
    def _save(self) -> None:
        """保存到磁盘"""
        try:
            data = {
                'skills': {sid: s.to_dict() for sid, s in self._skills.items()},
                'total_skills': self._total_skills,
            }
            
            path = f"{self.store_path}/procedural_memory.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save procedural memory: {e}")
    
    def _load(self) -> None:
        """从磁盘加载"""
        try:
            path = f"{self.store_path}/procedural_memory.json"
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._skills = {
                sid: Skill.from_dict(sdata)
                for sid, sdata in data.get('skills', {}).items()
            }
            
            # 重建索引
            for skill in self._skills.values():
                if skill.category not in self._skills_by_category:
                    self._skills_by_category[skill.category] = []
                self._skills_by_category[skill.category].append(skill.id)
                
                for ctx in skill.applicable_contexts:
                    if ctx not in self._skills_by_context:
                        self._skills_by_context[ctx] = []
                    self._skills_by_context[ctx].append(skill.id)
            
            self._total_skills = data.get('total_skills', len(self._skills))
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Failed to load procedural memory: {e}")
    
    def __len__(self) -> int:
        return len(self._skills)
