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
Memory System Tests - 记忆系统测试
===================================
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
import sys

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入记忆模块
from memory.episodic_memory import EpisodicMemory, Episode, EmotionalTag, ImportanceLevel
from memory.semantic_memory import SemanticMemory, Concept, KnowledgeSource
from memory.procedural_memory import ProceduralMemory, Skill, SkillLevel
from memory.working_memory import WorkingMemory, WorkingMemoryConfig
from memory.memory_store import MemoryStore
from memory.memory_retrieval import MemoryRetrieval, RetrievalQuery
from memory.memory_consolidation import MemoryConsolidation, ConsolidationConfig
from memory.long_term_memory import LongTermMemory, MemoryConfig


# ==================== 测试夹具 ====================

@pytest.fixture
def temp_store_path():
    """临时存储路径"""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def episodic_memory(temp_store_path):
    return EpisodicMemory(store_path=temp_store_path, max_episodes=100)


@pytest.fixture
def semantic_memory(temp_store_path):
    return SemanticMemory(store_path=temp_store_path)


@pytest.fixture
def procedural_memory(temp_store_path):
    return ProceduralMemory(store_path=temp_store_path)


@pytest.fixture
def working_memory():
    return WorkingMemory()


@pytest.fixture
def long_term_memory(temp_store_path):
    return LongTermMemory(storage_path=temp_store_path)


# ==================== 情景记忆测试 ====================

class TestEpisodicMemory:
    """情景记忆测试"""
    
    def test_store_episode(self, episodic_memory):
        """测试存储记忆"""
        ep = episodic_memory.store(
            summary="完成抓取任务",
            context={'object': 'box', 'location': 'table'},
            actions=[{'action': 'move', 'target': 'box'}],
            outcomes={'success': True},
            importance_score=7.0,
            tags=['抓取', '成功'],
        )
        
        assert ep is not None
        assert ep.summary == "完成抓取任务"
        assert ep.importance_score == 7.0
        assert '抓取' in ep.tags
    
    def test_retrieve_recent(self, episodic_memory):
        """测试检索最近记忆"""
        # 存储多条记忆
        for i in range(5):
            episodic_memory.store(
                summary=f"记忆 {i}",
                context={'index': i},
            )
        
        recent = episodic_memory.retrieve_recent(limit=3)
        assert len(recent) == 3
    
    def test_retrieve_by_entities(self, episodic_memory):
        """测试按实体检索"""
        episodic_memory.store(
            summary="测试记忆",
            context={},
            entities=['Alice', 'Bob'],
        )
        
        results = episodic_memory.retrieve_by_entities(['Alice'])
        assert len(results) >= 1
    
    def test_access_episode(self, episodic_memory):
        """测试访问记忆"""
        ep = episodic_memory.store(summary="测试访问")
        initial_access = ep.access_count
        
        episodic_memory.access_episode(ep.id)
        updated_ep = episodic_memory.get_episode(ep.id)
        
        assert updated_ep.access_count == initial_access + 1
    
    def test_statistics(self, episodic_memory):
        """测试统计"""
        episodic_memory.store(summary="测试1", importance_score=5.0)
        episodic_memory.store(summary="测试2", importance_score=8.0)
        
        stats = episodic_memory.get_statistics()
        assert stats['total_episodes'] == 2
        assert stats['avg_importance'] > 0


# ==================== 语义记忆测试 ====================

class TestSemanticMemory:
    """语义记忆测试"""
    
    def test_add_concept(self, semantic_memory):
        """测试添加概念"""
        concept = semantic_memory.add_concept(
            name="机器人",
            category="智能体",
            description="一种智能机器",
            confidence=0.9,
        )
        
        assert concept is not None
        assert concept.name == "机器人"
        assert concept.category == "智能体"
    
    def test_find_concept_by_name(self, semantic_memory):
        """测试按名称查找"""
        semantic_memory.add_concept(name="机器人", category="智能体")
        
        found = semantic_memory.find_concept_by_name("机器人")
        assert found is not None
        assert found.name == "机器人"
    
    def test_update_concept(self, semantic_memory):
        """测试更新概念"""
        concept = semantic_memory.add_concept(name="测试", category="test")
        
        updated = semantic_memory.update_concept(
            concept.id,
            properties={'new_prop': 'value'},
            confidence=0.95,
        )
        
        assert updated is not None
        assert updated.properties.get('new_prop') == 'value'
    
    def test_add_fact(self, semantic_memory):
        """测试添加事实"""
        concept = semantic_memory.add_concept(name="测试", category="test")
        
        fact = semantic_memory.add_fact(
            subject_id=concept.id,
            predicate="是",
            object_value="某种东西",
            confidence=0.8,
        )
        
        assert fact is not None
        assert fact.predicate == "是"
    
    def test_add_rule(self, semantic_memory):
        """测试添加规则"""
        rule = semantic_memory.add_rule(
            if_conditions=["条件A", "条件B"],
            then_conclusion="结论",
            confidence=0.75,
        )
        
        assert rule is not None
        assert len(rule.if_conditions) == 2
    
    def test_search_concepts(self, semantic_memory):
        """测试搜索概念"""
        semantic_memory.add_concept(name="移动机器人", category="机器人")
        semantic_memory.add_concept(name="抓取机器人", category="机器人")
        
        results = semantic_memory.search_concepts("机器人")
        assert len(results) >= 2
    
    def test_statistics(self, semantic_memory):
        """测试统计"""
        semantic_memory.add_concept(name="概念1", category="A")
        semantic_memory.add_concept(name="概念2", category="B")
        
        stats = semantic_memory.get_statistics()
        assert stats['total_concepts'] == 2


# ==================== 程序记忆测试 ====================

class TestProceduralMemory:
    """程序记忆测试"""
    
    def test_add_skill(self, procedural_memory):
        """测试添加技能"""
        skill = procedural_memory.add_skill(
            name="抓取",
            description="抓取物体",
            category="manipulation",
            steps=[
                {'step': 1, 'action': '接近'},
                {'step': 2, 'action': '夹取'},
            ],
        )
        
        assert skill is not None
        assert skill.name == "抓取"
        assert len(skill.steps) == 2
    
    def test_find_skill_by_name(self, procedural_memory):
        """测试按名称查找技能"""
        procedural_memory.add_skill(name="导航", category="navigation")
        
        found = procedural_memory.find_skill_by_name("导航")
        assert found is not None
    
    def test_update_skill(self, procedural_memory):
        """测试更新技能熟练度"""
        skill = procedural_memory.add_skill(name="测试技能", category="test")
        
        updated = procedural_memory.update_skill(
            skill.id,
            success=True,
            duration_s=5.0,
        )
        
        assert updated is not None
        assert updated.success_count == 1
        assert updated.experience_points > 0
    
    def test_prerequisites(self, procedural_memory):
        """测试前置技能"""
        skill1 = procedural_memory.add_skill(name="基础技能", category="test")
        skill2 = procedural_memory.add_skill(name="高级技能", category="test")
        
        # 添加前置技能要求 BEGINNER 级别
        procedural_memory.add_prerequisite(skill2.id, skill1.id, SkillLevel.BEGINNER)
        
        # 新技能默认是 NOVICE，所以前置条件不满足
        satisfied, unsatisfied = procedural_memory.check_prerequisites(skill2.id)
        assert not satisfied
        assert len(unsatisfied) == 1
        
        # 升级 skill1 到 BEGINNER (需要 100 XP，每次成功获得约 10 XP)
        for _ in range(10):
            procedural_memory.update_skill(skill1.id, success=True, duration_s=5.0)
        
        # 现在应该满足前置条件
        satisfied, unsatisfied = procedural_memory.check_prerequisites(skill2.id)
        assert satisfied
    
    def test_search_skills(self, procedural_memory):
        """测试搜索技能"""
        procedural_memory.add_skill(name="移动", category="navigation", tags=['移动'])
        procedural_memory.add_skill(name="抓取", category="manipulation", tags=['操作'])
        
        results = procedural_memory.search_skills("移动")
        assert len(results) >= 1


# ==================== 工作记忆测试 ====================

class TestWorkingMemory:
    """工作记忆测试"""
    
    def test_focus(self, working_memory):
        """测试焦点存储"""
        working_memory.focus("key1", {"data": "value"}, importance=8.0)
        
        value = working_memory.get_focused("key1")
        assert value is not None
    
    def test_unfocus(self, working_memory):
        """测试移除焦点"""
        working_memory.focus("key1", "value")
        result = working_memory.unfocus("key1")
        
        assert result is True
    
    def test_decay(self, working_memory):
        """测试衰减"""
        working_memory.focus("key1", "value", importance=5.0)
        time.sleep(0.1)
        
        decayed = working_memory.apply_decay(dt=1.0)
        # 至少有一些衰减
    
    def test_bindings(self, working_memory):
        """测试变量绑定"""
        working_memory.bind("x", 10)
        working_memory.bind("y", 20)
        
        assert working_memory.get_binding("x") == 10
        
        bindings = working_memory.get_all_bindings()
        assert len(bindings) == 2
    
    def test_activation(self, working_memory):
        """测试激活模式"""
        working_memory.activate("concept_1")
        working_memory.activate("concept_2")
        
        assert working_memory.is_activated("concept_1")
        
        pattern = working_memory.get_activation_pattern()
        assert "concept_1" in pattern


# ==================== 整合系统测试 ====================

class TestMemoryConsolidation:
    """记忆整合测试"""
    
    def test_consolidation_config(self):
        """测试整合配置"""
        config = ConsolidationConfig(
            min_importance_threshold=6.0,
            consolidation_interval_s=1800.0,
        )
        
        assert config.min_importance_threshold == 6.0


# ==================== 统一接口测试 ====================

class TestLongTermMemory:
    """统一长期记忆接口测试"""
    
    def test_initialization(self, temp_store_path):
        """测试初始化"""
        ltm = LongTermMemory(storage_path=temp_store_path)
        assert ltm is not None
    
    def test_store_episode(self, long_term_memory):
        """测试存储情景记忆"""
        ep = long_term_memory.store_episode(
            summary="测试记忆",
            context={'test': True},
            importance_score=7.0,
        )
        
        assert ep is not None
        assert ep.summary == "测试记忆"
    
    def test_store_knowledge(self, long_term_memory):
        """测试存储知识"""
        concept = long_term_memory.store_knowledge(
            name="测试概念",
            category="测试",
            description="测试描述",
        )
        
        assert concept is not None
    
    def test_store_skill(self, long_term_memory):
        """测试存储技能"""
        skill = long_term_memory.store_skill(
            name="测试技能",
            description="测试技能描述",
            category="test",
        )
        
        assert skill is not None
    
    def test_unified_retrieval(self, long_term_memory):
        """测试统一检索"""
        # 先存储一些数据
        long_term_memory.store_episode(
            summary="这是一个关于抓取的测试记忆",
            context={},
            tags=['抓取'],
        )
        
        results = long_term_memory.retrieve("抓取", limit=5)
        assert isinstance(results, list)
    
    def test_learn_from_interaction(self, long_term_memory):
        """测试从交互学习"""
        ep = long_term_memory.learn_from_interaction(
            interaction_type="抓取",
            summary="成功抓取箱子",
            context={'object': 'box'},
            actions=[{'step': 1, 'action': '接近'}, {'step': 2, 'action': '夹取'}],
            outcome={'success': True, 'lessons': ['接近时要稳定']},
            success=True,
            tags=['实验'],
        )
        
        assert ep is not None
        assert ep.summary == "成功抓取箱子"
    
    def test_get_status(self, long_term_memory):
        """测试获取状态"""
        long_term_memory.store_episode(summary="测试1")
        long_term_memory.store_knowledge(name="概念1", category="test")
        
        status = long_term_memory.get_status()
        
        assert 'episodic' in status
        assert 'semantic' in status
        assert 'procedural' in status
    
    def test_memory_summary(self, long_term_memory):
        """测试记忆摘要"""
        long_term_memory.store_episode(summary="测试")
        
        summary = long_term_memory.get_memory_summary()
        assert "情景记忆" in summary
        assert "语义记忆" in summary


# ==================== 集成测试 ====================

class TestMemoryIntegration:
    """记忆系统集成测试"""
    
    def test_full_workflow(self, temp_store_path):
        """测试完整工作流"""
        ltm = LongTermMemory(storage_path=temp_store_path)
        
        # 1. 从交互中学习
        ep = ltm.learn_from_interaction(
            interaction_type="导航",
            summary="成功导航到目标位置",
            context={'start': 'A', 'end': 'B'},
            actions=[
                {'step': 1, 'action': '定位起点'},
                {'step': 2, 'action': '规划路径'},
                {'step': 3, 'action': '移动'},
            ],
            outcome={'success': True},
            success=True,
        )
        
        # 2. 获取工作记忆状态
        summary = ltm.get_working_summary()
        assert summary is not None
        
        # 3. 检索记忆
        results = ltm.retrieve("导航")
        assert isinstance(results, list)
        
        # 4. 获取完整状态
        status = ltm.get_status()
        assert status['episodic']['count'] >= 1
        
        # 5. 执行整合
        consolidation_result = ltm.consolidate()
        assert consolidation_result is not None
        
        ltm.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
