"""
test_memory_integration.py - 具身记忆集成测试
测试 EmbodiedMemoryManager / EmbodiedSkill / create_embodied_memory_manager
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from embodied.memory_integration import (
    EmbodiedMemoryEntry,
    EmbodiedSkill,
    EmbodiedMemoryManager,
    create_embodied_memory_manager,
)


# ============================================================================
# EmbodiedMemoryEntry 测试
# ============================================================================

class TestEmbodiedMemoryEntry:
    def test_entry_creation(self):
        entry = EmbodiedMemoryEntry(
            entry_id="e001",
            entry_type="episode",
            timestamp=time.time(),
            content={"task": "transport", "duration": 10.0},
            importance=0.8,
        )
        assert entry.entry_id == "e001"
        assert entry.entry_type == "episode"
        assert entry.accessibility == 1.0
        assert entry.retrieval_count == 0

    def test_touch_boosts_accessibility(self):
        entry = EmbodiedMemoryEntry(
            entry_id="e002",
            entry_type="episode",
            timestamp=time.time(),
            content={},
            accessibility=0.5,
        )
        initial = entry.accessibility
        entry.touch()
        assert entry.retrieval_count == 1
        assert entry.accessibility > initial

    def test_touch_caps_at_one(self):
        entry = EmbodiedMemoryEntry(
            entry_id="e003",
            entry_type="episode",
            timestamp=time.time(),
            content={},
        )
        entry.accessibility = 0.99
        entry.touch()
        assert entry.accessibility == 1.0

    def test_decay(self):
        entry = EmbodiedMemoryEntry(
            entry_id="e004",
            entry_type="episode",
            timestamp=time.time(),
            content={},
        )
        entry.accessibility = 1.0
        entry.decay(0.9)
        assert entry.accessibility == 0.9
        entry.decay(0.9)
        assert abs(entry.accessibility - 0.81) < 0.001


# ============================================================================
# EmbodiedSkill 测试
# ============================================================================

class TestEmbodiedSkill:
    def test_skill_creation(self):
        skill = EmbodiedSkill(
            skill_id="s001",
            name="WarehouseTransport",
            description="Transport goods in warehouse",
            behavior_tree_config={"type": "sequence", "children": []},
            preconditions=[{"type": "battery", "min": 0.2}],
            scene_types=["warehouse"],
        )
        assert skill.name == "WarehouseTransport"
        assert skill.success_rate == 0.0
        assert skill.usage_count == 0
        assert "warehouse" in skill.scene_types

    def test_activate_returns_config(self):
        skill = EmbodiedSkill(
            skill_id="s002",
            name="TestSkill",
            description="",
            behavior_tree_config={"type": "sequence", "children": [{"type": "action"}]},
            preconditions=[],
        )
        config = skill.activate()
        assert config == skill.behavior_tree_config
        assert skill.usage_count == 1
        assert skill.last_used is not None

    def test_update_success_increases_rate(self):
        skill = EmbodiedSkill(
            skill_id="s003",
            name="TestSkill",
            description="",
            behavior_tree_config={},
            preconditions=[],
        )
        skill.update_success(True, 5.0)
        assert skill.success_rate > 0.0
        assert skill.avg_duration == 5.0

    def test_update_failure_decreases_rate(self):
        skill = EmbodiedSkill(
            skill_id="s004",
            name="TestSkill",
            description="",
            behavior_tree_config={},
            preconditions=[],
        )
        # 先成功一次
        skill.update_success(True, 5.0)
        rate_after_success = skill.success_rate
        # 再失败
        skill.update_success(False, 3.0)
        assert skill.success_rate < rate_after_success

    def test_update_rolling_average_duration(self):
        skill = EmbodiedSkill(
            skill_id="s005",
            name="TestSkill",
            description="",
            behavior_tree_config={},
            preconditions=[],
        )
        skill.update_success(True, 10.0)
        assert skill.usage_count == 1
        assert skill.avg_duration == 10.0
        skill.update_success(True, 20.0)
        assert skill.usage_count == 2
        assert abs(skill.avg_duration - 15.0) < 0.001  # (10 + 20) / 2


# ============================================================================
# EmbodiedMemoryManager 测试
# ============================================================================

class TestEmbodiedMemoryManager:
    def test_manager_creation_no_args(self):
        manager = EmbodiedMemoryManager()
        assert manager.episodic is None
        assert manager.semantic is None
        assert manager.procedural is None
        assert manager.working is None
        # 本地缓存应该可用
        assert len(manager._episode_cache) == 0
        assert len(manager._skill_cache) == 0

    def test_store_episode(self):
        manager = EmbodiedMemoryManager()
        entry = manager.store_episode(
            episode_type="transport",
            content={"task": "move_pkg", "duration": 12.0},
            importance=0.9,
            tags={"warehouse", "priority"},
            outcome="success",
        )
        assert entry.entry_id != ""
        assert entry.entry_type == "transport"
        assert entry.learned_from == "success"
        assert len(manager._episode_cache) == 1

    def test_retrieve_episodes_by_type(self):
        manager = EmbodiedMemoryManager()
        manager.store_episode("transport", {"task": "t1"})
        manager.store_episode("patrol", {"task": "p1"})
        manager.store_episode("transport", {"task": "t2"})
        manager.store_episode("rescue", {"task": "r1"})

        results = manager.retrieve_episodes("transport")
        assert len(results) == 2
        types = [r.entry_type for r in results]
        assert all(t == "transport" for t in types)

    def test_retrieve_episodes_with_time_window(self):
        manager = EmbodiedMemoryManager()
        # 存入旧记忆
        old_entry = EmbodiedMemoryEntry(
            entry_id="old001",
            entry_type="transport",
            timestamp=time.time() - 7200,  # 2小时前
            content={},
        )
        manager._episode_cache.append(old_entry)

        # 存入新记忆
        manager.store_episode("transport", {"task": "new"})

        results = manager.retrieve_episodes("transport", time_window=3600)  # 1小时内
        types = [r.entry_type for r in results]
        assert all(t == "transport" for t in types)
        # 应该只有新记忆
        assert len(results) == 1

    def test_retrieve_episodes_outcome_filter(self):
        manager = EmbodiedMemoryManager()
        manager.store_episode("transport", {"task": "t1"}, outcome="success")
        manager.store_episode("transport", {"task": "t2"}, outcome="failure")
        manager.store_episode("patrol", {"task": "p1"}, outcome="success")

        success_results = manager.retrieve_episodes("transport", outcome_filter="success")
        assert len(success_results) == 1
        assert success_results[0].content["task"] == "t1"

        failure_results = manager.retrieve_episodes("transport", outcome_filter="failure")
        assert len(failure_results) == 1
        assert failure_results[0].content["task"] == "t2"

    def test_get_recent_episodes(self):
        manager = EmbodiedMemoryManager()
        for i in range(15):
            manager.store_episode("transport", {"seq": i})
        recent = manager.get_recent_episodes(count=5)
        assert len(recent) == 5
        # 应该按时间倒序
        timestamps = [e.timestamp for e in recent]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_register_and_retrieve_skill(self):
        manager = EmbodiedMemoryManager()
        skill = manager.register_skill(
            name="FastTransport",
            behavior_tree_config={"type": "sequence"},
            description="Fast transport skill",
            scene_types=["warehouse", "factory"],
            tags={"fast", "transport"},
        )
        assert skill.skill_id in manager._skill_cache

        # 按名称检索
        found = manager.retrieve_skills(query="Fast")
        assert len(found) >= 1
        assert found[0].name == "FastTransport"

        # 按场景检索
        found = manager.retrieve_skills(scene_type="warehouse")
        assert len(found) >= 1

    def test_retrieve_skills_by_success_rate(self):
        manager = EmbodiedMemoryManager()
        s1 = manager.register_skill("LowSuccess", {}, scene_types=["warehouse"])
        s2 = manager.register_skill("HighSuccess", {}, scene_types=["warehouse"])

        # s1: 1 success, 1 failure → 50% success rate
        s1.update_success(True, 5.0)
        s1.update_success(False, 5.0)
        # s2: 5 successes → 100% success rate
        for _ in range(5):
            s2.update_success(True, 5.0)

        found = manager.retrieve_skills(scene_type="warehouse", min_success_rate=0.9)
        assert len(found) >= 1
        # HighSuccess (100%) 应该排在 LowSuccess (50%) 前面
        assert found[0].name == "HighSuccess"

    def test_update_skill_outcome(self):
        manager = EmbodiedMemoryManager()
        skill = manager.register_skill("TestSkill", {})
        skill_id = skill.skill_id

        manager.update_skill_outcome(skill_id, True, 8.0)
        manager.update_skill_outcome(skill_id, True, 12.0)
        manager.update_skill_outcome(skill_id, False, 5.0)

        updated = manager._skill_cache[skill_id]
        assert updated.usage_count == 3
        # 2/3 成功率
        assert abs(updated.success_rate - 2/3) < 0.01

    def test_store_semantic(self):
        manager = EmbodiedMemoryManager()
        entry = manager.store_semantic(
            concept_type="scene",
            content={"scene": "warehouse", "shelves": 50, "aisles": 10},
            importance=0.7,
            tags={"warehouse", "layout"},
        )
        assert entry.entry_type == "scene"
        assert entry.content["shelves"] == 50
        assert len(manager._semantic_cache) == 1

    def test_query_semantic(self):
        manager = EmbodiedMemoryManager()
        manager.store_semantic("scene", {"name": "warehouse", "type": "indoor"})
        manager.store_semantic("scene", {"name": "outdoor", "type": "outdoor"})
        manager.store_semantic("object", {"name": "pallet", "weight": 100})

        results = manager.query_semantic(concept_type="scene")
        assert len(results) == 2

        results = manager.query_semantic(query="indoor")
        assert len(results) >= 1

    def test_working_memory_set_get(self):
        manager = EmbodiedMemoryManager()
        manager.set_working("battery_level", 0.85)
        manager.set_working("position", (1.0, 2.0))

        assert manager.get_working("battery_level") == 0.85
        assert manager.get_working("position") == (1.0, 2.0)
        assert manager.get_working("nonexistent", default=42) == 42

    def test_clear_working(self):
        manager = EmbodiedMemoryManager()
        manager.set_working("a", 1)
        manager.set_working("b", 2)
        manager.clear_working("a")
        assert manager.get_working("a") is None
        assert manager.get_working("b") == 2

        manager.clear_working()  # 清除全部
        assert len(manager._working_state) == 0

    def test_attention_focus(self):
        manager = EmbodiedMemoryManager()
        manager.set_attention_focus({
            "current_task": "transport",
            "target_position": (5.0, 3.0),
            "battery_level": 0.6,
        })
        focus = manager.get_attention_focus()
        assert focus["current_task"] == "transport"
        assert focus["target_position"] == (5.0, 3.0)
        assert focus["battery_level"] == 0.6

    def test_memory_summary(self):
        manager = EmbodiedMemoryManager()
        manager.store_episode("transport", {"d": 1})
        manager.store_episode("patrol", {"d": 2})
        manager.register_skill("TestSkill1", {}, scene_types=["warehouse"])
        manager.register_skill("TestSkill2", {}, scene_types=["warehouse"])
        s = manager.get_memory_summary()
        assert s["episodes_cached"] == 2
        assert s["skills_registered"] == 2
        assert s["working_keys"] == 0

    def test_decays_old_entries(self):
        manager = EmbodiedMemoryManager()
        manager._decay_interval = 0.1  # 100ms for testing
        manager._last_decay = 0

        entry = EmbodiedMemoryEntry(
            entry_id="decay_test",
            entry_type="episode",
            timestamp=time.time() - 100,  # 很久以前
            content={},
            accessibility=0.001,  # 已经接近遗忘阈值
        )
        manager._episode_cache.append(entry)
        manager._apply_decay()

        # 几乎遗忘的条目应该被移除
        assert len(manager._episode_cache) == 0


# ============================================================================
# 工厂函数测试
# ============================================================================

class TestFactoryFunctions:
    def test_create_embodied_memory_manager(self):
        manager = create_embodied_memory_manager()
        assert isinstance(manager, EmbodiedMemoryManager)

    def test_create_with_external_memory(self):
        class FakeEpisodic:
            def store(self, data): pass
            def retrieve(self, query, limit): return []

        manager = create_embodied_memory_manager(episodic_memory=FakeEpisodic())
        assert manager.episodic is not None
        assert manager.enable_memory is True


# ============================================================================
# 集成场景测试
# ============================================================================

class TestIntegrationScenario:
    """完整的记忆-执行集成场景测试"""

    def test_episode_store_retrieve_update_cycle(self):
        """
        模拟完整的经验学习周期:
        1. 执行任务并存储经验
        2. 检索相关经验
        3. 基于经验调整配置
        4. 再次执行并更新技能统计
        """
        manager = create_embodied_memory_manager()

        # Step 1: 存储多次执行经验
        for i in range(3):
            manager.store_episode(
                episode_type="transport",
                content={"iteration": i, "duration": 10.0 + i},
                outcome="success" if i < 2 else "failure",
            )

        # Step 2: 检索成功经验
        successes = manager.retrieve_episodes("transport", outcome_filter="success")
        assert len(successes) == 2

        # Step 3: 注册一个技能并更新表现
        skill = manager.register_skill(
            name="ImprovedTransport",
            behavior_tree_config={"type": "sequence", "children": []},
            scene_types=["warehouse"],
        )

        # 模拟执行10次，8次成功
        for i in range(10):
            success = i < 8
            manager.update_skill_outcome(skill.skill_id, success, 8.0 + i * 0.5)

        # Step 4: 检索高成功率技能
        best = manager.retrieve_skills(scene_type="warehouse", min_success_rate=0.7)
        assert len(best) >= 1
        assert best[0].success_rate > 0.7

    def test_memory_summarizes_correctly_after_operations(self):
        manager = create_embodied_memory_manager()

        for i in range(5):
            manager.store_episode("transport", {"i": i})

        s1 = manager.get_memory_summary()
        assert s1["episodes_cached"] == 5

        manager.store_semantic("scene", {"type": "warehouse"})

        s2 = manager.get_memory_summary()
        assert s2["semantic_entries"] == 1
        assert s2["episodes_cached"] == 5


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
