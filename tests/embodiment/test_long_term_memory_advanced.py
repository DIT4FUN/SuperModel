"""
test_long_term_memory_advanced.py - 长期记忆系统高级场景测试
============================================================
测试长期记忆系统的场景化功能:
- 情景记忆存储与检索
- 语义知识存储与查询
- 程序技能存储与匹配
- 记忆标签检索
- 记忆重要性管理
"""

import pytest
import time
import tempfile
import shutil
from typing import List, Dict, Any

from src.memory.episodic_memory import (
    EpisodicMemory, Episode, EmotionalTag
)
from src.memory.semantic_memory import (
    SemanticMemory, Concept, KnowledgeSource
)
from src.memory.procedural_memory import (
    ProceduralMemory, Skill
)
from src.memory.long_term_memory import (
    LongTermMemory, MemoryConfig, MemoryType, MemoryEntry
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage():
    """临时存储目录"""
    path = tempfile.mkdtemp(prefix="ltm_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def episodic_mem(temp_storage):
    """情景记忆实例"""
    mem = EpisodicMemory(store_path=f"{temp_storage}/episodic")
    yield mem


@pytest.fixture
def semantic_mem(temp_storage):
    """语义记忆实例"""
    mem = SemanticMemory(store_path=f"{temp_storage}/semantic")
    yield mem


@pytest.fixture
def procedural_mem(temp_storage):
    """程序记忆实例"""
    mem = ProceduralMemory(store_path=f"{temp_storage}/procedural")
    yield mem


@pytest.fixture
def ltm_system(temp_storage):
    """完整长期记忆系统"""
    return LongTermMemory(
        storage_path=f"{temp_storage}/ltm",
        vector_dim=512,
        forget_threshold_days=30,
    )


# =============================================================================
# Episodic Memory Tests - 情景记忆测试
# =============================================================================

class TestEpisodicMemory:
    """情景记忆核心功能测试"""

    def test_store_episode_basic(self, episodic_mem):
        """测试存储基本事件"""
        ep = episodic_mem.store(
            summary="AGV完成仓库A到B的搬运任务",
            entities=["AGV_01", "warehouse_A", "warehouse_B"],
            emotional_tag=EmotionalTag.POSITIVE,
            importance_score=8.0,
        )
        assert ep.id is not None
        assert "AGV" in ep.summary

    def test_episode_with_context(self, episodic_mem):
        """测试带场景上下文的事件"""
        ep = episodic_mem.store(
            summary="AGV在货架C3区域检测到障碍物",
            entities=["AGV_02", "shelf_C3"],
            context={"zone": "C3", "obstacle_detected": True},
            emotional_tag=EmotionalTag.NEGATIVE,
            importance_score=7.0,
        )
        assert ep.context is not None

    def test_episode_temporal_ordering(self, episodic_mem):
        """测试事件时间顺序"""
        ep1 = episodic_mem.store("第一个事件: AGV启动", entities=["AGV_01"])
        time.sleep(0.01)
        ep2 = episodic_mem.store("第二个事件: AGV移动", entities=["AGV_01"])
        time.sleep(0.01)
        ep3 = episodic_mem.store("第三个事件: AGV到达", entities=["AGV_01"])
        assert ep3.timestamp > ep2.timestamp > ep1.timestamp

    def test_retrieve_recent_episodes(self, episodic_mem):
        """测试检索最近事件"""
        for i in range(5):
            episodic_mem.store(f"事件 {i}: 测试数据", entities=[f"entity_{i}"])
        recent = episodic_mem.retrieve_recent(limit=3)
        assert len(recent) == 3

    def test_retrieve_by_entities(self, episodic_mem):
        """测试按实体检索"""
        episodic_mem.store("Alice在仓库工作", entities=["Alice", "warehouse"])
        episodic_mem.store("Bob在仓库工作", entities=["Bob", "warehouse"])
        episodic_mem.store("Alice去医院", entities=["Alice", "hospital"])
        results = episodic_mem.retrieve_by_entities(["Alice"])
        assert len(results) == 2

    def test_multi_entity_episode(self, episodic_mem):
        """测试多实体事件（蜂群协作场景）"""
        ep = episodic_mem.store(
            summary="AGV_01和AGV_02协同完成大型货物搬运",
            entities=["AGV_01", "AGV_02", "large_cargo"],
            emotional_tag=EmotionalTag.VERY_POSITIVE,
            importance_score=9.0,
        )
        assert len(ep.entities) >= 2


# =============================================================================
# Semantic Memory Tests - 语义记忆测试
# =============================================================================

class TestSemanticMemory:
    """语义记忆核心功能测试"""

    def test_add_concept(self, semantic_mem):
        """测试添加概念"""
        concept = semantic_mem.add_concept(
            name="AGV",
            category="robotics",
            description="自动导引车,用于物流搬运",
            properties={"max_speed": 2.0, "payload": 1000.0},
        )
        assert concept.id is not None
        assert concept.name == "AGV"

    def test_concept_query(self, semantic_mem):
        """测试概念查询"""
        semantic_mem.add_concept(name="AGV", category="robotics", description="自动导引车")
        semantic_mem.add_concept(name="robot_arm", category="robotics", description="多关节机械手")
        semantic_mem.add_concept(name="shelf", category="logistics", description="存储商品的架子")
        # search_concepts searches concept names, not categories
        results = semantic_mem.search_concepts(query="AGV")
        assert len(results) >= 1


# =============================================================================
# Procedural Memory Tests - 程序记忆测试
# =============================================================================

class TestProceduralMemory:
    """程序记忆核心功能测试"""

    def test_add_skill(self, procedural_mem):
        """测试添加技能"""
        skill = procedural_mem.add_skill(
            name="agv_navigation",
            description="AGV基础导航技能",
            steps=[
                {"step": 1, "action": "定位当前位置"},
                {"step": 2, "action": "规划路径"},
                {"step": 3, "action": "执行移动"},
            ],
        )
        assert skill.id is not None
        assert skill.name == "agv_navigation"
        assert len(skill.steps) == 3

    def test_get_skill(self, procedural_mem):
        """测试获取技能"""
        skill = procedural_mem.add_skill(
            name="test_skill",
            description="测试技能",
            steps=[{"step": 1, "action": "test"}],
        )
        retrieved = procedural_mem.get_skill(skill.id)
        assert retrieved is not None
        assert retrieved.name == "test_skill"

    def test_find_skill_by_name(self, procedural_mem):
        """测试按名称查找技能"""
        procedural_mem.add_skill(name="navigation", description="导航", steps=[])
        found = procedural_mem.find_skill_by_name("navigation")
        assert found is not None
        assert found.name == "navigation"

    def test_skill_search(self, procedural_mem):
        """测试技能搜索"""
        procedural_mem.add_skill(name="agv_nav", description="AGV导航", steps=[], category="navigation")
        procedural_mem.add_skill(name="arm_control", description="机械臂控制", steps=[], category="manipulation")
        results = procedural_mem.search_skills("AGV")
        assert len(results) >= 1


# =============================================================================
# Long Term Memory System Tests - 长期记忆系统测试
# =============================================================================

class TestLongTermMemorySystem:
    """长期记忆系统综合测试"""

    def test_store_different_memory_types(self, ltm_system):
        """测试存储不同类型记忆"""
        ep_id = ltm_system.add_memory(
            memory_type=MemoryType.EPISODIC,
            content={"summary": "AGV完成运输任务", "agv": "AGV_01"},
            importance=0.8,
            tags=["transport", "success"],
        )
        assert ep_id is not None

        sem_id = ltm_system.add_memory(
            memory_type=MemoryType.SEMANTIC,
            content={"concept": "AGV", "description": "自动导引车"},
            importance=0.9,
            tags=["robotics", "definition"],
        )
        assert sem_id is not None

        proc_id = ltm_system.add_memory(
            memory_type=MemoryType.PROCEDURAL,
            content={"skill": "navigation", "steps": ["plan", "execute", "verify"]},
            importance=0.7,
            tags=["navigation", "skill"],
        )
        assert proc_id is not None

    def test_memory_retrieval_by_tags(self, ltm_system):
        """测试按标签检索"""
        ltm_system.add_memory(MemoryType.EPISODIC, {"data": "A"}, tags=["robot", "agv"])
        ltm_system.add_memory(MemoryType.EPISODIC, {"data": "B"}, tags=["robot", "arm"])
        ltm_system.add_memory(MemoryType.SEMANTIC, {"data": "C"}, tags=["sensor"])
        results = ltm_system.search_by_tags(["robot"])
        assert len(results) >= 2

    def test_memory_importance_scoring(self, ltm_system):
        """测试记忆重要性评分"""
        high_id = ltm_system.add_memory(
            MemoryType.EPISODIC,
            {"summary": "紧急停止事件"},
            importance=0.95,
        )
        low_id = ltm_system.add_memory(
            MemoryType.EPISODIC,
            {"summary": "例行检查"},
            importance=0.1,
        )
        high_entry = ltm_system.get_memory(high_id)
        low_entry = ltm_system.get_memory(low_id)
        assert high_entry.importance > low_entry.importance

    def test_memory_persistence(self, temp_storage):
        """测试记忆持久化"""
        ltm1 = LongTermMemory(storage_path=f"{temp_storage}/persist_ltm", vector_dim=256)
        mem_id = ltm1.add_memory(
            MemoryType.EPISODIC,
            {"summary": "持久化测试"},
            importance=0.8,
        )
        ltm1.close()
        ltm2 = LongTermMemory(storage_path=f"{temp_storage}/persist_ltm", vector_dim=256)
        recovered = ltm2.get_memory(mem_id)
        assert recovered is not None
        ltm2.close()

    def test_memory_stats(self, ltm_system):
        """测试记忆统计"""
        ltm_system.add_memory(MemoryType.EPISODIC, {"data": "X"}, tags=["test"])
        ltm_system.add_memory(MemoryType.SEMANTIC, {"data": "Y"}, tags=["test"])
        stats = ltm_system.get_memory_stats()
        assert isinstance(stats, dict)


# =============================================================================
# AGV-Specific Memory Tests - AGV专用记忆测试
# =============================================================================

class TestAGVMemory:
    """AGV专用记忆场景测试"""

    def test_collision_memory(self, episodic_mem):
        """测试碰撞事件记忆"""
        ep = episodic_mem.store(
            summary="AGV_02在拐角与静止障碍物发生轻微碰撞",
            entities=["AGV_02", "corner_A", "obstacle_01"],
            emotional_tag=EmotionalTag.NEGATIVE,
            importance_score=9.0,
        )
        assert EmotionalTag.NEGATIVE == ep.emotional_tag

    def test_battery_depletion_memory(self, episodic_mem):
        """测试电量耗尽记忆"""
        ep = episodic_mem.store(
            summary="AGV_03电量耗尽,在位置(15.3, 8.7)停止",
            entities=["AGV_03", "depletion_point"],
            emotional_tag=EmotionalTag.NEGATIVE,
            importance_score=8.5,
        )
        assert ep is not None

    def test_successful_docking_memory(self, episodic_mem):
        """测试成功对接记忆"""
        ep = episodic_mem.store(
            summary="AGV_01成功对接充电桩D,充电效率95%",
            entities=["AGV_01", "charger_D"],
            emotional_tag=EmotionalTag.POSITIVE,
            importance_score=7.0,
        )
        assert EmotionalTag.POSITIVE == ep.emotional_tag

    def test_swarm_collaboration_memory(self, episodic_mem):
        """测试蜂群协作记忆"""
        ep = episodic_mem.store(
            summary="AGV_01和AGV_02协同完成大型货物搬运",
            entities=["AGV_01", "AGV_02", "large_cargo"],
            emotional_tag=EmotionalTag.VERY_POSITIVE,
            importance_score=9.0,
        )
        assert "AGV_01" in ep.entities
        assert "AGV_02" in ep.entities

    def test_navigation_failure_memory(self, episodic_mem):
        """测试导航失败记忆"""
        ep = episodic_mem.store(
            summary="AGV_04在狭窄通道SLAM定位失败,切换到手动模式",
            entities=["AGV_04", "narrow_corridor", "SLAM_failure"],
            emotional_tag=EmotionalTag.NEGATIVE,
            importance_score=8.0,
        )
        assert "SLAM" in ep.summary
