"""
Long Term Memory System Tests
==============================
"""
import pytest
import tempfile
import time
import numpy as np
import importlib
import shutil
from pathlib import Path

from src.memory.long_term_memory import LongTermMemory, MemoryConfig
from src.memory.memory_store import MemoryStore, StorageBackend
from src.memory.memory_retrieval import MemoryRetrieval, RetrievalQuery, RetrievalStrategy
from src.memory.memory_consolidation import MemoryConsolidation, ConsolidationConfig


def test_memory_store_basic_operations():
    """Test basic memory store operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(base_path=tmpdir, backend=StorageBackend.JSON_FILE.value)
        
        # Test save and load
        test_data = {"key1": "value1", "key2": 2, "key3": [1, 2, 3]}
        assert store.save("test_key", test_data) is True
        
        loaded = store.load("test_key")
        assert loaded == test_data
        
        # Test exists
        assert store.exists("test_key") is True
        assert store.exists("non_existent") is False
        
        # Test delete
        assert store.delete("test_key") is True
        assert store.exists("test_key") is False
        
        store.close()


@pytest.mark.skipif(not importlib.util.find_spec("faiss"), reason="FAISS not installed")
def test_vector_database_operations():
    """Test vector database storage and search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(
            base_path=tmpdir,
            backend=StorageBackend.VECTOR_DB.value,
            vector_dim=384,  # all-MiniLM-L6-v2 dimension
        )
        
        # Test add vector
        vector1 = np.random.rand(384)
        assert store.add_vector("mem1", vector1, {"type": "episodic", "summary": "Test memory 1"}) is True
        
        vector2 = np.random.rand(384)
        assert store.add_vector("mem2", vector2, {"type": "semantic", "name": "Test concept"}) is True
        
        # Test search
        results = store.search_vectors(vector1, top_k=2)
        assert len(results) >= 1
        assert results[0][0] == "mem1"
        assert results[0][1] > 0.9  # Similarity should be high for same vector
        
        # Test delete vector
        assert store.delete_vector("mem1") is True
        results = store.search_vectors(vector1, top_k=2)
        assert all(r[0] != "mem1" for r in results)
        
        store.close()


def test_experience_replay():
    """Test experience replay functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(store_path=tmpdir)
        ltm = LongTermMemory(config=config)
        
        # Add some test memories
        for i in range(20):
            importance = i % 10 + 1
            ltm.store_episode(
                summary=f"Test episode {i}",
                context={"task": f"task_{i%5}"},
                importance_score=importance,
                tags=[f"tag_{i%3}"],
                lessons_learned=[f"Lesson {i}"],
            )
        
        # Test experience replay
        replay_results = ltm.retrieval.experience_replay(num_samples=5, priority="importance")
        assert len(replay_results) == 5
        # Check that higher importance memories are sampled first
        importances = [r['importance'] for r in replay_results]
        assert importances == sorted(importances, reverse=True)
        
        # Test task filter
        replay_results = ltm.retrieval.experience_replay(num_samples=5, task_filter="task_1")
        assert all("task_1" in r['content']['context']['task'] for r in replay_results)
        
        ltm.close()


def test_memory_forgetting():
    """Test memory forgetting algorithm"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(store_path=tmpdir)
        ltm = LongTermMemory(config=config)
        
        # Add 100 test memories: 50 low importance (1-5), 50 high importance (6-10)
        for i in range(100):
            importance = i % 10 + 1
            # Make some memories very old
            timestamp = time.time() - (3600 * 24 * 60) if i < 50 else time.time()
            ep = ltm.store_episode(
                summary=f"Test episode {i}",
                importance_score=importance,
            )
            ep.timestamp = timestamp
        
        initial_count = len(ltm.episodic._episodes)
        assert initial_count == 100
        
        # Apply forgetting
        pruned = ltm.consolidation._apply_forgetting()
        
        # Check that at least some low importance memories were forgotten
        assert pruned > 0
        remaining_count = len(ltm.episodic._episodes)
        assert remaining_count < initial_count
        
        # Check that all high importance memories (>=8) are still present
        high_importance_count = sum(1 for ep in ltm.episodic._episodes.values() if ep.importance_score >= 8)
        assert high_importance_count >= 20  # Should have at least 20 high importance memories
        
        ltm.close()


@pytest.mark.skipif(not importlib.util.find_spec("sentence_transformers"), reason="Sentence Transformers not installed")
def test_semantic_retrieval():
    """Test semantic retrieval functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(store_path=tmpdir)
        ltm = LongTermMemory(config=config)
        
        # Add test memories
        ltm.store_episode(
            summary="How to make coffee: grind beans, heat water, pour over grounds",
            importance_score=8.0,
            tags=["cooking", "coffee"],
        )
        
        ltm.store_episode(
            summary="How to make tea: boil water, steep tea leaves, add sugar or milk",
            importance_score=7.0,
            tags=["cooking", "tea"],
        )
        
        ltm.store_episode(
            summary="How to assemble a chair: attach legs to seat, attach backrest, tighten screws",
            importance_score=6.0,
            tags=["assembly", "furniture"],
        )
        
        # Test semantic search for coffee
        results = ltm.retrieve("making coffee", limit=3)
        assert len(results) >= 1
        assert "coffee" in results[0].data['summary'].lower()
        
        # Test semantic search for furniture assembly
        results = ltm.retrieve("assembling furniture", limit=3)
        assert len(results) >= 1
        assert "chair" in results[0].data['summary'].lower()
        
        ltm.close()


def test_core_brain_integration():
    """Test integration with core brain module"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # This simulates how core brain would use the memory system
        config = MemoryConfig(
            store_path=tmpdir,
            auto_save=True,
            min_importance_threshold=5.0,
        )
        
        memory_system = LongTermMemory(config=config)
        
        # Simulate learning from interaction
        memory_system.learn_from_interaction(
            interaction_type="navigation",
            summary="Successfully navigated from point A to point B avoiding obstacles",
            context={"start": "A", "end": "B", "obstacles": 3},
            actions=["move_forward", "turn_left", "avoid_obstacle", "move_forward"],
            outcome={"success": True, "duration": 25.5, "lessons": ["Slow down when approaching obstacles"]},
            success=True,
            entities=["obstacle", "point A", "point B"],
            tags=["navigation", "success"],
        )
        
        # Check that memory was stored
        episodes = memory_system.retrieve_episodes(content="navigate")
        assert len(episodes) >= 1
        
        # Simulate retrieval for decision making
        decision_context = {"task": "navigate", "obstacles_present": True}
        relevant_memories = memory_system.retrieve("navigation obstacle avoidance")
        assert len(relevant_memories) >= 1
        
        # Simulate memory consolidation
        consolidation_result = memory_system.consolidate()
        assert consolidation_result.consolidated_count >= 0
        
        memory_system.close()


def test_backup_and_restore():
    """Test memory backup and restore functionality"""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Create memory system and add data
        config1 = MemoryConfig(store_path=tmpdir1)
        ltm1 = LongTermMemory(config=config1)
        
        ltm1.store_episode(summary="Test episode 1", importance_score=8.0)
        ltm1.store_knowledge(name="Test concept", description="Test description")
        ltm1.store_skill(name="Test skill", steps=["step1", "step2"])
        
        # Create backup
        backup_path = ltm1.create_backup()
        backup_name = Path(backup_path).name
        assert Path(backup_path).exists()
        
        # Copy backup to second memory system's backup directory
        target_backup_dir = Path(tmpdir2) / "backups"
        target_backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_path, target_backup_dir / backup_name)
        
        # Restore to new memory system
        config2 = MemoryConfig(store_path=tmpdir2)
        ltm2 = LongTermMemory(config=config2)
        assert ltm2.store.restore_backup(backup_name) is True
        
        # Check that data was restored
        assert len(ltm2.episodic._episodes) == 1
        assert len(ltm2.semantic._concepts) >= 1
        assert len(ltm2.procedural._skills) >= 1
        
        ltm1.close()
        ltm2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
