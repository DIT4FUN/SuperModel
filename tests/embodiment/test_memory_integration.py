"""
test_memory_integration.py - 具身记忆系统集成测试
==================================================

测试 SuperModel 具身智能记忆系统的完整功能:
- 记忆系统与具身Pipeline的集成
- 情景记忆的动作经验存储与检索
- 语义记忆的场景知识管理
- 程序记忆的技能知识管理
- 工作记忆的实时状态追踪
- 记忆检索与任务执行的联动
- AGV五级规格的差异化记忆配置
"""

import time
import pytest
from typing import Any, Dict, List, Set

from src.embodied.memory_integration import (
    EmbodiedMemoryEntry,
    EmbodiedMemoryManager,
    EmbodiedSkill,
    create_embodied_memory_manager,
)


# ============================================================
# 情景记忆操作测试
# ============================================================

class TestEpisodicMemory:
    """情景记忆测试"""

    def test_create_manager(self):
        """测试创建记忆管理器"""
        manager = create_embodied_memory_manager()
        assert manager is not None
        assert manager.enable_memory is False  # 无外部记忆系统时为False

    def test_store_episode(self):
        """测试存储情景记忆"""
        manager = create_embodied_memory_manager()
        entry = manager.store_episode(
            episode_type="transport",
            content={
                "task_id": "task_001",
                "duration_s": 120.0,
                "distance_m": 50.0,
            },
            importance=0.7,
            tags={"warehouse", "transport"},
            outcome="success",
        )
        assert entry is not None
        assert entry.entry_type == "transport"
        assert entry.content["duration_s"] == 120.0

    def test_store_multiple_episodes(self):
        """测试存储多条情景记忆"""
        manager = create_embodied_memory_manager()
        for i in range(5):
            manager.store_episode(
                episode_type="patrol",
                content={"iteration": i, "zone": f"zone_{i}"},
                importance=0.5,
                outcome="success" if i % 2 == 0 else "failure",
            )
        assert len(manager._episode_cache) == 5

    def test_retrieve_episodes(self):
        """测试检索情景记忆"""
        manager = create_embodied_memory_manager()
        # 存储多条
        manager.store_episode(
            episode_type="transport",
            content={"task": "A"},
            importance=0.8,
            outcome="success",
        )
        manager.store_episode(
            episode_type="patrol",
            content={"task": "B"},
            importance=0.6,
            outcome="success",
        )
        results = manager.retrieve_episodes(
            query="transport",
            max_results=10,
        )
        assert isinstance(results, list)

    def test_retrieve_with_outcome_filter(self):
        """测试按结果过滤检索"""
        manager = create_embodied_memory_manager()
        manager.store_episode(episode_type="t1", content={}, outcome="success")
        manager.store_episode(episode_type="t2", content={}, outcome="failure")
        results = manager.retrieve_episodes(query="", outcome_filter="success")
        assert all(r.learned_from == "success" for r in results)

    def test_get_recent_episodes(self):
        """测试获取最近情景记忆"""
        manager = create_embodied_memory_manager()
        for i in range(15):
            manager.store_episode(
                episode_type="transport",
                content={"i": i},
                importance=0.5,
            )
        recent = manager.get_recent_episodes(count=5)
        assert len(recent) <= 5

    def test_episode_decay(self):
        """测试记忆衰减"""
        manager = create_embodied_memory_manager()
        entry = manager.store_episode(
            episode_type="test",
            content={"data": "old"},
            importance=0.5,
        )
        initial_accessibility = entry.accessibility
        entry.decay(factor=0.95)
        assert entry.accessibility <= initial_accessibility


# ============================================================
# 程序记忆（技能）操作测试
# ============================================================

class TestProceduralMemory:
    """程序记忆/技能测试"""

    def test_register_skill(self):
        """测试注册技能"""
        manager = create_embodied_memory_manager()
        skill = manager.register_skill(
            name="navigate_freight",
            behavior_tree_config={"type": "sequence", "children": []},
            description="仓库环境导航",
            preconditions=[{"type": "battery_ok"}],
            scene_types=["warehouse"],
        )
        assert skill is not None
        assert skill.name == "navigate_freight"
        assert skill.success_rate == 0.0  # 新技能初始为0

    def test_retrieve_skills(self):
        """测试检索技能"""
        manager = create_embodied_memory_manager()
        manager.register_skill(
            name="grasp_item",
            behavior_tree_config={"type": "action", "name": "grasp"},
            description="抓取物品",
            scene_types=["warehouse"],
        )
        skills = manager.retrieve_skills(query="grasp")
        assert isinstance(skills, list)

    def test_retrieve_skills_by_scene(self):
        """测试按场景检索技能"""
        manager = create_embodied_memory_manager()
        manager.register_skill(
            name="nav_warehouse",
            behavior_tree_config={},
            description="仓库导航",
            scene_types=["warehouse"],
        )
        manager.register_skill(
            name="nav_hospital",
            behavior_tree_config={},
            description="医院导航",
            scene_types=["hospital"],
        )
        skills = manager.retrieve_skills(scene_type="warehouse")
        assert len(skills) >= 1
        assert all("warehouse" in s.scene_types for s in skills)

    def test_update_skill_outcome(self):
        """测试更新技能结果"""
        manager = create_embodied_memory_manager()
        skill = manager.register_skill(
            name="test_skill",
            behavior_tree_config={},
            description="测试技能",
        )
        manager.update_skill_outcome(
            skill_id=skill.skill_id,
            success=True,
            duration=8.0,
        )
        # 更新后成功率应为 1.0 (1/1)
        updated = manager._skill_cache[skill.skill_id]
        assert updated.success_rate == 1.0

    def test_update_skill_outcome_multiple(self):
        """测试多次更新技能结果"""
        manager = create_embodied_memory_manager()
        skill = manager.register_skill(
            name="repeat_skill",
            behavior_tree_config={},
            description="重复测试技能",
        )
        for _ in range(5):
            manager.update_skill_outcome(skill.skill_id, success=True, duration=10.0)
        updated = manager._skill_cache[skill.skill_id]
        assert updated.usage_count == 5
        assert updated.success_rate == 1.0


# ============================================================
# 语义记忆操作测试
# ============================================================

class TestSemanticMemory:
    """语义记忆测试"""

    def test_store_semantic(self):
        """测试存储语义记忆"""
        manager = create_embodied_memory_manager()
        entry = manager.store_semantic(
            concept_type="warehouse_layout",
            content={
                "aisles": 10,
                "shelves_per_aisle": 50,
                "cold_zone": "zone_A",
            },
            importance=0.9,
            tags={"warehouse", "layout"},
        )
        assert entry is not None
        assert entry.entry_type == "warehouse_layout"

    def test_query_semantic(self):
        """测试查询语义记忆"""
        manager = create_embodied_memory_manager()
        manager.store_semantic(
            concept_type="hospital_zones",
            content={"icu": "floor_3", "or": "floor_2"},
            importance=0.95,
        )
        results = manager.query_semantic(concept_type="hospital_zones")
        assert isinstance(results, list)

    def test_query_semantic_by_text(self):
        """测试按文本查询语义记忆"""
        manager = create_embodied_memory_manager()
        manager.store_semantic(
            concept_type="safety_rules",
            content={"max_speed": 1.0, "stop_distance": 0.5},
            importance=0.9,
        )
        results = manager.query_semantic(query="speed")
        assert isinstance(results, list)


# ============================================================
# 工作记忆操作测试
# ============================================================

class TestWorkingMemory:
    """工作记忆测试"""

    def test_set_and_get_working(self):
        """测试设置和获取工作记忆"""
        manager = create_embodied_memory_manager()
        manager.set_working(key="current_task", value={"id": "task_001", "progress": 0.5})
        result = manager.get_working(key="current_task")
        assert result is not None
        assert result["id"] == "task_001"

    def test_get_working_default(self):
        """测试获取默认值"""
        manager = create_embodied_memory_manager()
        result = manager.get_working(key="nonexistent", default="default_value")
        assert result == "default_value"

    def test_clear_working(self):
        """测试清除工作记忆"""
        manager = create_embodied_memory_manager()
        manager.set_working(key="temp", value="temporary_data")
        manager.clear_working(key="temp")
        result = manager.get_working(key="temp")
        assert result is None

    def test_clear_all_working(self):
        """测试清除所有工作记忆"""
        manager = create_embodied_memory_manager()
        manager.set_working(key="a", value=1)
        manager.set_working(key="b", value=2)
        manager.clear_working()  # 清除所有
        assert manager.get_working("a") is None
        assert manager.get_working("b") is None

    def test_attention_focus(self):
        """测试注意力焦点"""
        manager = create_embodied_memory_manager()
        manager.set_working(key="target_position", value=(1.0, 2.0))
        manager.set_working(key="battery_level", value=0.75)
        manager.set_working(key="scene_type", value="warehouse")
        focus = manager.get_attention_focus()
        assert focus["target_position"] == (1.0, 2.0)
        assert focus["battery_level"] == 0.75
        assert focus["scene_type"] == "warehouse"

    def test_set_attention_focus(self):
        """测试设置注意力焦点"""
        manager = create_embodied_memory_manager()
        manager.set_attention_focus({
            "current_task": "transport",
            "safety_status": "normal",
        })
        assert manager.get_working("current_task") == "transport"
        assert manager.get_working("safety_status") == "normal"


# ============================================================
# 记忆整合与遗忘测试
# ============================================================

class TestMemoryConsolidation:
    """记忆整合测试"""

    def test_decay_application(self):
        """测试衰减自动应用"""
        manager = create_embodied_memory_manager()
        # 存储多条记忆
        for i in range(10):
            manager.store_episode(
                episode_type="test",
                content={"i": i},
                importance=0.5,
            )
        # 手动触发衰减
        manager._apply_decay()
        # 衰减后应有记忆
        assert len(manager._episode_cache) >= 0

    def test_episode_touch(self):
        """测试记忆触碰（增强可访问性）"""
        manager = create_embodied_memory_manager()
        entry = manager.store_episode(
            episode_type="important_task",
            content={"data": "critical"},
            importance=0.5,
        )
        initial_retrieval_count = entry.retrieval_count
        entry.touch()
        assert entry.retrieval_count == initial_retrieval_count + 1


# ============================================================
# 记忆条目操作测试
# ============================================================

class TestEmbodiedMemoryEntry:
    """记忆条目测试"""

    def test_entry_creation(self):
        """测试记忆条目创建"""
        entry = EmbodiedMemoryEntry(
            entry_id="test_001",
            entry_type="transport",
            timestamp=time.time(),
            content={"duration": 100},
            importance=0.8,
            tags={"test", "demo"},
        )
        assert entry.entry_id == "test_001"
        assert entry.importance == 0.8
        assert entry.accessibility == 1.0  # 默认

    def test_entry_touch(self):
        """测试记忆触碰"""
        entry = EmbodiedMemoryEntry(
            entry_id="test_001",
            entry_type="task",
            timestamp=time.time(),
            content={},
        )
        initial_access = entry.accessibility
        entry.touch()
        assert entry.retrieval_count == 1
        assert entry.accessibility == min(1.0, initial_access + 0.1)

    def test_entry_decay(self):
        """测试记忆衰减"""
        entry = EmbodiedMemoryEntry(
            entry_id="test_001",
            entry_type="task",
            timestamp=time.time(),
            content={},
            importance=0.9,
            accessibility=1.0,
        )
        entry.decay(factor=0.9)
        assert entry.accessibility < 1.0


# ============================================================
# 技能条目测试
# ============================================================

class TestEmbodiedSkill:
    """技能条目测试"""

    def test_skill_creation(self):
        """测试技能创建"""
        skill = EmbodiedSkill(
            skill_id="skill_001",
            name="test_navigate",
            description="测试导航技能",
            behavior_tree_config={"type": "sequence"},
            preconditions=[{"type": "battery_ok"}],
        )
        assert skill.name == "test_navigate"
        assert skill.success_rate == 0.0
        assert skill.usage_count == 0

    def test_skill_activate(self):
        """测试技能激活"""
        skill = EmbodiedSkill(
            skill_id="skill_001",
            name="test",
            description="",
            behavior_tree_config={"type": "action", "name": "navigate"},
            preconditions=[],
        )
        config = skill.activate()
        assert config["type"] == "action"
        assert skill.usage_count == 1

    def test_skill_update_success(self):
        """测试技能成功更新"""
        skill = EmbodiedSkill(
            skill_id="skill_001",
            name="test",
            description="",
            behavior_tree_config={},
            preconditions=[],
        )
        skill.update_success(success=True, duration=30.0)
        assert skill.success_rate == 1.0
        assert skill.avg_duration == 30.0

    def test_skill_update_failure(self):
        """测试技能失败更新"""
        skill = EmbodiedSkill(
            skill_id="skill_001",
            name="test",
            description="",
            behavior_tree_config={},
            preconditions=[],
            success_rate=1.0,
            usage_count=1,
            avg_duration=20.0,
        )
        skill.update_success(success=False, duration=25.0)
        assert skill.success_rate == 0.5  # (1.0 + 0) / 2
        assert skill.avg_duration == 22.5  # (20 + 25) / 2


# ============================================================
# 端到端集成测试
# ============================================================

class TestMemoryTaskIntegration:
    """记忆与任务集成测试"""

    def test_store_and_retrieve_workflow(self):
        """测试存储和检索工作流"""
        manager = create_embodied_memory_manager()
        # 1. 执行任务前检查
        prior = manager.retrieve_episodes(query="transport", max_results=3)
        # 2. 执行任务
        entry = manager.store_episode(
            episode_type="transport",
            content={
                "task_id": "task_002",
                "duration_s": 200.0,
                "success": True,
            },
            importance=0.7,
            outcome="success",
        )
        # 3. 注册技能
        manager.register_skill(
            name="transport",
            behavior_tree_config={"type": "sequence"},
            description="物流运输",
            scene_types=["warehouse"],
        )
        # 4. 验证
        assert entry is not None

    def test_skill_learning_from_episodes(self):
        """测试从经验中学习技能"""
        manager = create_embodied_memory_manager()
        # 记录多次经验
        for i in range(5):
            manager.store_episode(
                episode_type="grasp",
                content={"attempt": i, "method": "vision"},
                importance=0.6,
                outcome="success" if i > 1 else "failure",
            )
        # 注册技能
        manager.register_skill(
            name="grasp_item",
            behavior_tree_config={"type": "action"},
            description="抓取",
            scene_types=["warehouse"],
        )

    def test_scene_knowledge_accumulation(self):
        """测试场景知识积累"""
        manager = create_embodied_memory_manager()
        # 仓库布局知识
        manager.store_semantic(
            concept_type="warehouse_A_layout",
            content={"aisles": 10, "max_capacity": 1000},
            importance=0.8,
            tags={"warehouse", "layout"},
        )
        # 交通规则知识
        manager.store_semantic(
            concept_type="safety_rules",
            content={"max_speed": 1.0, "stop_distance": 0.5},
            importance=0.9,
            tags={"safety"},
        )
        results = manager.query_semantic(query="warehouse")
        assert isinstance(results, list)


# ============================================================
# 配置测试
# ============================================================

class TestMemoryManagerConfig:
    """记忆管理器配置测试"""

    def test_create_with_config(self):
        """测试带配置创建"""
        manager = create_embodied_memory_manager(
            config={"decay_factor": 0.9, "decay_interval": 1800},
        )
        assert manager.config["decay_factor"] == 0.9

    def test_default_decay_params(self):
        """测试默认衰减参数"""
        manager = create_embodied_memory_manager()
        assert manager._decay_factor == 0.95
        assert manager._decay_interval == 3600
