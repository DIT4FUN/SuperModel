"""
自主学习框架测试
================

测试自主学习模块:
- PrioritizedReplayBuffer (PER)
- SumTree
- EWC (持续学习)
- MetaLearner (元学习)
- CuriosityModule (好奇心驱动探索)
- SkillLibrary (技能库)
- AutonomousLearningAgent (综合智能体)
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from learning.autonomous_learning import (
    Experience,
    AutonomousLearningConfig,
    PrioritizedReplayBuffer,
    SumTree,
    EWC,
    MetaLearner,
    CuriosityModule,
    SkillLibrary,
    AutonomousLearningAgent
)


class TestSumTree(unittest.TestCase):
    """测试 SumTree 数据结构"""

    def test_sum_tree_insert(self):
        tree = SumTree(capacity=10)
        tree.add(0.5)
        tree.add(1.0)
        self.assertGreater(tree.total(), 0)

    def test_sum_tree_find(self):
        tree = SumTree(capacity=10)
        tree.add(0.5)
        tree.add(1.0)
        tree.add(0.3)
        idx = tree.find(0.5)
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)

    def test_sum_tree_internal_update(self):
        tree = SumTree(capacity=10)
        tree.add(0.5)
        tree.add(1.0)
        old_total = tree.total()
        tree._update(1, 2.0)
        self.assertGreater(tree.total(), old_total)

    def test_sum_tree_overflow(self):
        tree = SumTree(capacity=3)
        for i in range(5):
            tree.add(float(i + 1))
        self.assertEqual(tree.n_entries, 3)

    def test_sum_tree_total(self):
        tree = SumTree(capacity=5)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        self.assertAlmostEqual(tree.total(), 6.0, places=5)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """测试优先经验回放缓冲区"""

    def test_buffer_init(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        self.assertEqual(len(buf), 0)

    def test_buffer_push(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        exp = Experience(
            state={'obs': np.random.randn(10)},
            action=np.array([1.0]),
            reward=1.0,
            next_state={'obs': np.random.randn(10)},
            done=False,
            priority=1.0,
            task_id=0
        )
        buf.push(exp)
        self.assertEqual(len(buf), 1)

    def test_buffer_sample(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            exp = Experience(
                state={'obs': np.random.randn(10)},
                action=np.array([1.0]),
                reward=1.0,
                next_state={'obs': np.random.randn(10)},
                done=False,
                priority=1.0,
                task_id=0
            )
            buf.push(exp)

        experiences, indices, weights = buf.sample(batch_size=5, beta=0.4)
        self.assertEqual(len(experiences), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(weights), 5)

    def test_buffer_capacity_limit(self):
        buf = PrioritizedReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(Experience(
                state={'obs': np.array([float(i)])},
                action=np.array([1.0]),
                reward=1.0,
                next_state={'obs': np.array([float(i)])},
                done=False,
                priority=1.0,
                task_id=0
            ))
        self.assertEqual(len(buf), 5)


class TestEWC(unittest.TestCase):
    """测试 EWC 持续学习"""

    @unittest.skipUnless(HAS_TORCH, "PyTorch not available")
    def test_ewc_init(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        ewc = EWC(model, lr=1e-3, lambda_=5000)
        self.assertEqual(ewc.lambda_, 5000)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not available")
    def test_ewc_penalty_initial(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        ewc = EWC(model)
        penalty = ewc.penalty()
        self.assertEqual(penalty.item(), 0.0)


class TestCuriosityModule(unittest.TestCase):
    """测试好奇心模块"""

    @unittest.skipUnless(HAS_TORCH, "PyTorch not available")
    def test_curiosity_init(self):
        module = CuriosityModule(state_dim=10, action_dim=5, hidden_dim=64)
        self.assertEqual(module.state_dim, 10)
        self.assertEqual(module.action_dim, 5)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not available")
    def test_curiosity_update(self):
        module = CuriosityModule(state_dim=10, action_dim=5, hidden_dim=32)
        state = np.random.randn(10)
        action = np.random.randn(5)
        next_state = np.random.randn(10)
        loss = module.update(state, action, next_state)
        self.assertIsInstance(loss, float)


class TestSkillLibrary(unittest.TestCase):
    """测试技能库"""

    def test_skill_library_init(self):
        lib = SkillLibrary(skill_dim=32, max_skills=10)
        self.assertEqual(lib.skill_dim, 32)
        self.assertEqual(lib.max_skills, 10)

    def test_skill_library_empty(self):
        lib = SkillLibrary(skill_dim=32, max_skills=10)
        self.assertEqual(len(lib.skills), 0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestAutonomousLearningAgent(unittest.TestCase):
    """测试自主学习智能体"""

    def test_autonomous_learning_agent_init(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)

        model = DummyModel()
        agent = AutonomousLearningAgent(model=model)
        self.assertIsNotNone(agent.replay_buffer)
        self.assertIsNotNone(agent.ewc)
        self.assertIsNotNone(agent.skill_library)

    def test_autonomous_learning_store(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)

        model = DummyModel()
        agent = AutonomousLearningAgent(model=model)

        state = {'obs': np.random.randn(10)}
        action = np.random.randn(5)
        reward = 1.0
        next_state = {'obs': np.random.randn(10)}
        done = False

        agent.store(state, action, reward, next_state, done)
        self.assertGreater(len(agent.replay_buffer), 0)


class TestExperience(unittest.TestCase):
    """测试 Experience 数据类"""

    def test_experience_creation(self):
        exp = Experience(
            state={'obs': np.array([1.0, 2.0])},
            action=np.array([0.5]),
            reward=1.0,
            next_state={'obs': np.array([1.5, 2.5])},
            done=False,
            priority=1.0,
            task_id=1
        )
        self.assertEqual(exp.task_id, 1)
        self.assertEqual(exp.priority, 1.0)
        self.assertFalse(exp.done)


class TestAutonomousLearningConfig(unittest.TestCase):
    """测试配置类"""

    def test_default_config(self):
        config = AutonomousLearningConfig()
        self.assertEqual(config.buffer_capacity, 100000)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.per_alpha, 0.6)
        self.assertEqual(config.per_beta, 0.4)

    def test_custom_config(self):
        config = AutonomousLearningConfig(
            buffer_capacity=50000,
            batch_size=32,
            learning_rate=1e-3
        )
        self.assertEqual(config.buffer_capacity, 50000)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-3)


if __name__ == '__main__':
    unittest.main()
