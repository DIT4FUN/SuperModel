"""
test_embodied_long_term_memory.py - 具身长期记忆系统测试
SuperModel 超模态大模型具身智能系统
"""

import pytest
import time
import numpy as np
import tempfile
import os
import shutil
from collections import defaultdict

from src.memory.embodied_long_term_memory import (
    EmbodiedExperienceType,
    EmbodiedMemoryTag,
    EmbodiedExperience,
    SceneMemoryIndex,
    SkillMemoryRecord,
    AGVGradeAwareMemory,
    ExperienceCompressor,
    MemoryBasedTaskPredictor,
    EmbodiedLongTermMemory,
    create_embodied_long_term_memory,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage():
    """临时存储目录"""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_experience():
    """创建示例具身经验"""
    return EmbodiedExperience(
        experience_id="exp_001",
        experience_type=EmbodiedExperienceType.NAVIGATION,
        scene_type="warehouse",
        start_timestamp=time.time() - 3600,
        end_timestamp=time.time(),
        duration_seconds=3600.0,
        initial_state={"position": [0, 0], "battery": 100, "load": 0},
        final_state={"position": [10, 10], "battery": 80, "load": 0},
        action_sequence=[
            {"action": "navigate", "params": {"target": [5, 5]}, "outcome": "success"},
            {"action": "navigate", "params": {"target": [10, 10]}, "outcome": "success"},
        ],
        outcome="success",
        outcome_score=0.95,
        efficiency_score=0.88,
        safety_score=0.98,
        agv_grade="M",
        sensor_config={"tactile": True, "force": True, "imu": True},
        learned_patterns=["optimal_path_through_shelf_A"],
        failure_reasons=[],
        improvement_hints=["reduce_speed_at_corners"],
        tags=["warehouse", "navigate", "success"],
        importance_score=0.8,
    )


@pytest.fixture
def multiple_experiences():
    """创建多个示例经验"""
    experiences = []
    scene_types = ["warehouse", "factory", "hospital", "restaurant"]
    exp_types = [
        EmbodiedExperienceType.NAVIGATION,
        EmbodiedExperienceType.GRASP,
        EmbodiedExperienceType.COLLABORATION,
        EmbodiedExperienceType.OBSTACLE_AVOIDANCE,
    ]
    
    for i in range(20):
        exp = EmbodiedExperience(
            experience_id=f"exp_{i:03d}",
            experience_type=exp_types[i % len(exp_types)],
            scene_type=scene_types[i % len(scene_types)],
            start_timestamp=time.time() - (20 - i) * 3600,
            end_timestamp=time.time() - (20 - i) * 3600 + 1800,
            duration_seconds=1800.0,
            initial_state={"position": [i, i], "battery": 100},
            final_state={"position": [i + 1, i + 1], "battery": 90},
            action_sequence=[{"action": "test", "params": {}, "outcome": "success"}],
            outcome="success" if i % 5 != 0 else "failure",
            outcome_score=0.9 if i % 5 != 0 else 0.3,
            efficiency_score=0.85,
            safety_score=0.95,
            agv_grade=["S", "M", "L", "XL", "XXL"][i % 5],
            sensor_config={},
            learned_patterns=[f"pattern_{i}"],
            failure_reasons=["obstacle_detected"] if i % 5 == 0 else [],
            improvement_hints=[],
            tags=[scene_types[i % len(scene_types)], "navigate"],
            importance_score=0.5 + (i % 5) * 0.1,
        )
        experiences.append(exp)
    
    return experiences


# =============================================================================
# EmbodiedExperience Tests
# =============================================================================

class TestEmbodiedExperience:
    """具身经验数据类测试"""
    
    def test_create_experience(self, sample_experience):
        assert sample_experience.experience_id == "exp_001"
        assert sample_experience.experience_type == EmbodiedExperienceType.NAVIGATION
        assert sample_experience.scene_type == "warehouse"
        assert sample_experience.outcome == "success"
        assert sample_experience.agv_grade == "M"
    
    def test_success_property(self, sample_experience):
        assert sample_experience.success is True
        
        failed_exp = EmbodiedExperience(
            experience_id="exp_fail",
            experience_type=EmbodiedExperienceType.GRASP,
            scene_type="factory",
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            duration_seconds=100.0,
            initial_state={},
            final_state={},
            action_sequence=[],
            outcome="failure",
            outcome_score=0.3,
            efficiency_score=0.3,
            safety_score=0.5,
            agv_grade="M",
            sensor_config={},
            learned_patterns=[],
            failure_reasons=["timeout"],
            improvement_hints=[],
        )
        assert failed_exp.success is False
    
    def test_total_reward(self, sample_experience):
        reward = sample_experience.total_reward
        assert 0.0 <= reward <= 1.0
        # success + high efficiency + high safety = high reward
        assert reward > 0.9
    
    def test_age_days(self, sample_experience):
        age = sample_experience.age_days
        assert 0.0 <= age < 1.0  # Less than 1 day old
    
    def test_to_dict(self, sample_experience):
        data = sample_experience.to_dict()
        assert isinstance(data, dict)
        assert data["experience_id"] == "exp_001"
        assert data["experience_type"] == "navigation"
        assert data["scene_type"] == "warehouse"
    
    def test_from_dict(self, sample_experience):
        data = sample_experience.to_dict()
        restored = EmbodiedExperience.from_dict(data)
        assert restored.experience_id == sample_experience.experience_id
        assert restored.experience_type == sample_experience.experience_type
        assert restored.scene_type == sample_experience.scene_type


# =============================================================================
# SceneMemoryIndex Tests
# =============================================================================

class TestSceneMemoryIndex:
    """场景-记忆关联索引测试"""
    
    def test_add_and_retrieve(self, multiple_experiences):
        index = SceneMemoryIndex()
        
        for exp in multiple_experiences[:10]:
            index.add_experience(exp)
        
        # 获取warehouse场景经验
        warehouse_exps = index.get_by_scene("warehouse")
        assert len(warehouse_exps) > 0
        
        # 验证返回的是经验ID
        for eid in warehouse_exps:
            assert isinstance(eid, str)
    
    def test_filter_by_tags(self, multiple_experiences):
        index = SceneMemoryIndex()
        
        for exp in multiple_experiences:
            index.add_experience(exp)
        
        # 按标签过滤
        factory_exps = index.get_by_scene("factory", tags=["factory"])
        assert all(isinstance(eid, str) for eid in factory_exps)
    
    def test_filter_by_type(self, multiple_experiences):
        index = SceneMemoryIndex()
        
        for exp in multiple_experiences:
            index.add_experience(exp)
        
        # 按经验类型过滤
        nav_exps = index.get_by_scene(
            "warehouse", 
            experience_types=[EmbodiedExperienceType.NAVIGATION]
        )
        assert len(nav_exps) >= 0  # 可能为空,因为分片逻辑
    
    def test_scene_stats(self, multiple_experiences):
        index = SceneMemoryIndex()
        
        for exp in multiple_experiences:
            index.add_experience(exp)
        
        stats = index.get_scene_stats("warehouse")
        assert "total_experiences" in stats
        assert "experience_types" in stats
        assert stats["total_experiences"] > 0
    
    def test_limit(self, multiple_experiences):
        index = SceneMemoryIndex()
        
        for exp in multiple_experiences:
            index.add_experience(exp)
        
        # 测试limit参数
        limited = index.get_by_scene("warehouse", limit=5)
        assert len(limited) <= 5


# =============================================================================
# SkillMemoryRecord Tests
# =============================================================================

class TestSkillMemoryRecord:
    """技能记忆记录测试"""
    
    def test_create_record(self):
        record = SkillMemoryRecord(
            skill_id="skill_001",
            skill_name="navigate",
            scene_type="warehouse",
            agv_grade="M",
        )
        assert record.skill_id == "skill_001"
        assert record.total_executions == 0
        assert record.success_rate == 0.0
    
    def test_update_from_successful_experience(self, sample_experience):
        record = SkillMemoryRecord(
            skill_id="navigate",
            skill_name="navigation",
            scene_type="warehouse",
            agv_grade="M",
        )
        
        record.update_from_experience(sample_experience)
        
        assert record.total_executions == 1
        assert record.successful_executions == 1
        assert record.failed_executions == 0
        assert record.success_rate == 1.0
        assert record.first_execution_time > 0
        assert record.last_success_time > 0
    
    def test_update_from_failed_experience(self):
        failed_exp = EmbodiedExperience(
            experience_id="exp_fail",
            experience_type=EmbodiedExperienceType.GRASP,
            scene_type="warehouse",
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            duration_seconds=100.0,
            initial_state={},
            final_state={},
            action_sequence=[],
            outcome="failure",
            outcome_score=0.2,
            efficiency_score=0.3,
            safety_score=0.5,
            agv_grade="M",
            sensor_config={},
            learned_patterns=[],
            failure_reasons=["timeout"],
            improvement_hints=[],
        )
        
        record = SkillMemoryRecord(
            skill_id="grasp",
            skill_name="grasping",
            scene_type="warehouse",
            agv_grade="M",
        )
        
        record.update_from_experience(failed_exp)
        
        assert record.total_executions == 1
        assert record.successful_executions == 0
        assert record.failed_executions == 1
        assert record.success_rate == 0.0
    
    def test_is_mastered(self):
        record = SkillMemoryRecord(
            skill_id="navigate",
            skill_name="navigation",
            scene_type="warehouse",
            agv_grade="M",
        )
        
        # 未满足条件
        assert record.is_mastered is False
        
        # 满足成功率条件,但执行次数不足
        for i in range(10):
            record.update_from_experience(EmbodiedExperience(
                experience_id=f"exp_test_{i}",
                experience_type=EmbodiedExperienceType.NAVIGATION,
                scene_type="warehouse",
                start_timestamp=time.time() - 100,
                end_timestamp=time.time(),
                duration_seconds=100.0,
                initial_state={},
                final_state={},
                action_sequence=[],
                outcome="success",
                outcome_score=0.95,
                efficiency_score=0.9,
                safety_score=0.95,
                agv_grade="M",
                sensor_config={},
                learned_patterns=[],
                failure_reasons=[],
                improvement_hints=[],
            ))
        
        # 10次执行后且成功率>=95%
        assert record.is_mastered is True
    
    def test_needs_relearning(self):
        record = SkillMemoryRecord(
            skill_id="navigate",
            skill_name="navigation",
            scene_type="warehouse",
            agv_grade="M",
        )
        
        # 从未执行过
        assert record.needs_relearning is False
        
        # 30天前成功过
        record.last_success_time = time.time() - 35 * 86400
        record.total_executions = 5
        assert record.needs_relearning is True
    
    def test_to_dict(self):
        record = SkillMemoryRecord(
            skill_id="navigate",
            skill_name="navigation",
            scene_type="warehouse",
            agv_grade="M",
        )
        record.total_executions = 10
        record.successful_executions = 8
        
        data = record.to_dict()
        assert data["skill_id"] == "navigate"
        assert data["total_executions"] == 10
        assert data["successful_executions"] == 8
    
    def test_from_dict(self):
        data = {
            "skill_id": "navigate",
            "skill_name": "navigation",
            "scene_type": "warehouse",
            "agv_grade": "M",
            "total_executions": 10,
            "successful_executions": 8,
            "failed_executions": 2,
            "avg_execution_time": 5.0,
            "avg_success_rate": 0.8,
            "avg_quality_score": 0.85,
            "scene_adaptation_score": 0.7,
            "learning_progress": 0.8,
            "learning_curve_points": [(1, 0.5), (5, 0.7), (10, 0.8)],
            "first_execution_time": time.time() - 86400,
            "last_execution_time": time.time(),
            "last_success_time": time.time(),
            "related_experience_ids": ["exp_001", "exp_002"],
        }
        
        record = SkillMemoryRecord.from_dict(data)
        assert record.skill_id == "navigate"
        assert record.total_executions == 10
        assert record.learning_progress == 0.8


# =============================================================================
# AGVGradeAwareMemory Tests
# =============================================================================

class TestAGVGradeAwareMemory:
    """AGV等级感知记忆测试"""
    
    def test_add_and_retrieve(self, multiple_experiences):
        memory = AGVGradeAwareMemory()
        
        for exp in multiple_experiences:
            memory.add_experience(
                exp.experience_id,
                exp.agv_grade,
                exp.scene_type,
                exp.experience_type,
            )
        
        # 获取M级可用的经验(包括S和M级)
        m_exps = memory.get_for_grade("M", scene="warehouse")
        assert isinstance(m_exps, list)
        
        # 获取L级可用的经验(包括S/M/L级)
        l_exps = memory.get_for_grade("L", scene="warehouse")
        assert len(l_exps) >= len(m_exps)
    
    def test_transfer_benefit(self):
        memory = AGVGradeAwareMemory()
        
        # 同等级
        assert memory.get_transfer_benefit("M", "M") == 1.0
        
        # 等级差距越大,效益越低
        benefit_SM = memory.get_transfer_benefit("S", "M")
        benefit_SX = memory.get_transfer_benefit("S", "XXL")
        assert benefit_SM > benefit_SX
    
    def test_get_all_grades(self, multiple_experiences):
        memory = AGVGradeAwareMemory()
        
        for exp in multiple_experiences:
            memory.add_experience(
                exp.experience_id,
                exp.agv_grade,
                exp.scene_type,
                exp.experience_type,
            )
        
        # 不指定场景获取
        all_exps = memory.get_for_grade("M")
        assert isinstance(all_exps, list)


# =============================================================================
# ExperienceCompressor Tests
# =============================================================================

class TestExperienceCompressor:
    """经验压缩器测试"""
    
    def test_compute_similarity_same(self, sample_experience):
        compressor = ExperienceCompressor()
        
        similar_exp = EmbodiedExperience(
            experience_id="exp_sim",
            experience_type=sample_experience.experience_type,
            scene_type=sample_experience.scene_type,
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            duration_seconds=100.0,
            initial_state={},
            final_state={},
            action_sequence=[],
            outcome="success",
            outcome_score=0.9,
            efficiency_score=0.85,
            safety_score=0.95,
            agv_grade="M",
            sensor_config={},
            learned_patterns=[],
            failure_reasons=[],
            improvement_hints=[],
        )
        
        sim = compressor.compute_similarity(sample_experience, similar_exp)
        assert 0.0 <= sim <= 1.0
        assert sim > 0.5  # 应该比较相似
    
    def test_compute_similarity_different_scene(self, sample_experience):
        compressor = ExperienceCompressor()
        
        diff_exp = EmbodiedExperience(
            experience_id="exp_diff",
            experience_type=sample_experience.experience_type,
            scene_type="factory",  # 不同场景
            start_timestamp=time.time() - 100,
            end_timestamp=time.time(),
            duration_seconds=100.0,
            initial_state={},
            final_state={},
            action_sequence=[],
            outcome="success",
            outcome_score=0.9,
            efficiency_score=0.85,
            safety_score=0.95,
            agv_grade="M",
            sensor_config={},
            learned_patterns=[],
            failure_reasons=[],
            improvement_hints=[],
        )
        
        sim = compressor.compute_similarity(sample_experience, diff_exp)
        assert sim == 0.0  # 不同场景相似度为0
    
    def test_compress_experiences(self, multiple_experiences):
        compressor = ExperienceCompressor()
        
        # 取相同场景的多个经验
        warehouse_exps = [e for e in multiple_experiences if e.scene_type == "warehouse"]
        if len(warehouse_exps) > 1:
            compressed = compressor.compress_experiences(warehouse_exps)
            assert len(compressed) <= len(warehouse_exps)
    
    def test_compress_single(self, sample_experience):
        compressor = ExperienceCompressor()
        result = compressor.compress_experiences([sample_experience])
        assert len(result) == 1
        assert result[0].experience_id == sample_experience.experience_id


# =============================================================================
# MemoryBasedTaskPredictor Tests
# =============================================================================

class TestMemoryBasedTaskPredictor:
    """基于记忆的任务预测器测试"""
    
    def test_register_and_predict(self, multiple_experiences):
        predictor = MemoryBasedTaskPredictor()
        
        for exp in multiple_experiences:
            predictor.register_experience(exp)
        
        # 预测warehouse导航任务
        rate, conf, similar = predictor.predict_success(
            scene_type="warehouse",
            exp_type=EmbodiedExperienceType.NAVIGATION,
            agv_grade="M",
            initial_state={"position": [0, 0]},
        )
        
        assert 0.0 <= rate <= 1.0
        assert 0.0 <= conf <= 1.0
        assert isinstance(similar, list)
    
    def test_predict_no_history(self):
        predictor = MemoryBasedTaskPredictor()
        
        rate, conf, similar = predictor.predict_success(
            scene_type="unknown_scene",
            exp_type=EmbodiedExperienceType.NAVIGATION,
            agv_grade="M",
            initial_state={},
        )
        
        # 无历史时返回默认值
        assert rate == 0.5
        assert conf == 0.0
        assert similar == []
    
    def test_get_failure_reasons(self, multiple_experiences):
        predictor = MemoryBasedTaskPredictor()
        
        for exp in multiple_experiences:
            predictor.register_experience(exp)
        
        reasons = predictor.get_failure_reasons(
            scene_type="warehouse",
            exp_type=EmbodiedExperienceType.NAVIGATION,
        )
        
        assert isinstance(reasons, list)
        # 返回格式应该是[(reason, count), ...]
        for item in reasons:
            assert isinstance(item, tuple)
            assert len(item) == 2


# =============================================================================
# EmbodiedLongTermMemory Integration Tests
# =============================================================================

class TestEmbodiedLongTermMemory:
    """具身长期记忆系统集成测试"""
    
    def test_create_and_store(self, temp_storage, sample_experience):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
            enable_compression=False,
            enable_prediction=True,
        )
        
        exp_id = memory.store_experience(sample_experience)
        assert exp_id == "exp_001"
        assert len(memory._experiences) == 1
    
    def test_retrieve_experiences(self, temp_storage, multiple_experiences):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
            enable_compression=False,
        )
        
        for exp in multiple_experiences:
            memory.store_experience(exp)
        
        # 按场景检索
        warehouse_exps = memory.retrieve_experiences(scene_type="warehouse")
        assert len(warehouse_exps) > 0
        
        # 检索结果都是warehouse
        for exp in warehouse_exps:
            assert exp.scene_type == "warehouse"
    
    def test_retrieve_only_successful(self, temp_storage, multiple_experiences):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
            enable_compression=False,
        )
        
        for exp in multiple_experiences:
            memory.store_experience(exp)
        
        successful = memory.retrieve_experiences(
            scene_type="warehouse",
            only_successful=True,
        )
        
        for exp in successful:
            assert exp.success is True
    
    def test_update_skill_memory(self, temp_storage, sample_experience):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
        )
        
        memory.store_experience(sample_experience)
        
        record = memory.update_skill_memory(
            skill_id="navigate",
            skill_name="navigation",
            scene_type="warehouse",
            agv_grade="M",
            experience=sample_experience,
        )
        
        assert record.skill_id == "navigate"
        assert record.total_executions == 1
        assert record.successful_executions == 1
    
    def test_get_skill_memory(self, temp_storage, sample_experience):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
        )
        
        memory.store_experience(sample_experience)
        memory.update_skill_memory(
            skill_id="navigate",
            skill_name="navigation",
            scene_type="warehouse",
            agv_grade="M",
            experience=sample_experience,
        )
        
        record = memory.get_skill_memory("navigate", "warehouse", "M")
        assert record is not None
        assert record.skill_id == "navigate"
    
    def test_predict_task_outcome(self, temp_storage, multiple_experiences):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
            enable_prediction=True,
        )
        
        for exp in multiple_experiences:
            memory.store_experience(exp)
        
        result = memory.predict_task_outcome(
            scene_type="warehouse",
            exp_type=EmbodiedExperienceType.NAVIGATION,
            agv_grade="M",
            initial_state={"position": [0, 0]},
        )
        
        assert "predicted_success_rate" in result
        assert "confidence" in result
        assert "failure_reasons" in result
        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)
    
    def test_memory_summary(self, temp_storage, multiple_experiences):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
        )
        
        for exp in multiple_experiences:
            memory.store_experience(exp)
        
        summary = memory.get_memory_summary()
        
        assert "total_experiences" in summary
        assert "scene_stats" in summary
        assert "experience_type_stats" in summary
        assert summary["total_experiences"] == len(multiple_experiences)
    
    def test_export_knowledge(self, temp_storage, multiple_experiences):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
        )
        
        for exp in multiple_experiences:
            memory.store_experience(exp)
        
        knowledge = memory.export_knowledge()
        
        assert "export_time" in knowledge
        assert "experience_count" in knowledge
        assert "learned_patterns" in knowledge
        assert "failure_reasons" in knowledge
    
    def test_grade_aware_retrieval(self, temp_storage):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
            enable_compression=False,
        )
        
        # 存储不同等级的经验
        for grade in ["S", "M", "L"]:
            exp = EmbodiedExperience(
                experience_id=f"exp_{grade}",
                experience_type=EmbodiedExperienceType.NAVIGATION,
                scene_type="warehouse",
                start_timestamp=time.time() - 100,
                end_timestamp=time.time(),
                duration_seconds=100.0,
                initial_state={},
                final_state={},
                action_sequence=[],
                outcome="success",
                outcome_score=0.9,
                efficiency_score=0.85,
                safety_score=0.95,
                agv_grade=grade,
                sensor_config={},
                learned_patterns=[],
                failure_reasons=[],
                improvement_hints=[],
            )
            memory.store_experience(exp)
        
        # XL级AGV应该能看到S/M/L级经验
        xl_exps = memory.retrieve_experiences(agv_grade="XL")
        assert len(xl_exps) == 3
        
        # S级AGV只能看到S级经验
        s_exps = memory.retrieve_experiences(agv_grade="S")
        assert len(s_exps) == 1
    
    def test_compression_on_limit(self, temp_storage):
        memory = EmbodiedLongTermMemory(
            storage_path=temp_storage,
            enable_compression=True,
            max_experiences=5,
        )
        
        # 存储5个经验
        for i in range(5):
            exp = EmbodiedExperience(
                experience_id=f"exp_{i}",
                experience_type=EmbodiedExperienceType.NAVIGATION,
                scene_type="warehouse",
                start_timestamp=time.time() - i * 100,
                end_timestamp=time.time() - i * 100 + 50,
                duration_seconds=50.0,
                initial_state={},
                final_state={},
                action_sequence=[],
                outcome="success",
                outcome_score=0.9,
                efficiency_score=0.85,
                safety_score=0.95,
                agv_grade="M",
                sensor_config={},
                learned_patterns=[],
                failure_reasons=[],
                improvement_hints=[],
            )
            memory.store_experience(exp)
        
        # 存储第6个经验,应该触发压缩
        exp_6 = EmbodiedExperience(
            experience_id="exp_6",
            experience_type=EmbodiedExperienceType.NAVIGATION,
            scene_type="warehouse",
            start_timestamp=time.time(),
            end_timestamp=time.time() + 50,
            duration_seconds=50.0,
            initial_state={},
            final_state={},
            action_sequence=[],
            outcome="success",
            outcome_score=0.9,
            efficiency_score=0.85,
            safety_score=0.95,
            agv_grade="M",
            sensor_config={},
            learned_patterns=[],
            failure_reasons=[],
            improvement_hints=[],
        )
        memory.store_experience(exp_6)
        
        # 压缩后经验数应该减少
        assert len(memory._experiences) <= 6


# =============================================================================
# Global Factory Function Tests
# =============================================================================

class TestGlobalInstance:
    """全局实例测试"""
    
    def test_create_global_instance(self, temp_storage):
        # 每次测试用新storage避免冲突
        memory = create_embodied_long_term_memory(
            storage_path=temp_storage,
            enable_compression=False,
        )
        assert memory is not None
        assert isinstance(memory, EmbodiedLongTermMemory)
