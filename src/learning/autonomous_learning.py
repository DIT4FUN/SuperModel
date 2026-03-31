"""
自主学习框架
============

在线持续学习系统，支持:
- 经验回放缓冲区 (Prioritized Experience Replay)
- 元学习 (Meta-Learning / MAML)
- 持续学习 (Continual Learning / EWC)
- 主动探索 (Active Exploration)
- 自适应技能获取

参考:
- MERLIN: Unified Learning from Language, Perception, and Action
- Continual Learning via Progressive Neural Networks
- Gradient Episodic Memory for Continual Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import copy
import random


@dataclass
class Experience:
    """单条经验"""
    state: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_state: Dict[str, np.ndarray]
    done: bool
    priority: float = 1.0  # PER 优先级
    task_id: int = 0  # 持续学习任务 ID


@dataclass
class AutonomousLearningConfig:
    """自主学习配置"""
    # 经验回放
    buffer_capacity: int = 100000
    batch_size: int = 64
    per_alpha: float = 0.6  # PER 优先级指数
    per_beta: float = 0.4   # PER 重要性采样指数
    
    # 元学习
    meta_lr: float = 1e-3
    meta_batch_size: int = 5  # 每次 meta-update 的任务数
    inner_steps: int = 5      # 内循环步数
    
    # 持续学习 EWC
    ewc_lambda: float = 5000  # EWC 惩罚权重
    fisher_samples: int = 200  # 计算 Fisher 信息矩阵的样本数
    
    # 探索
    curiosity_weight: float = 0.1
    exploration_bonus: float = 1.0
    
    # 在线学习
    learning_rate: float = 1e-4
    target_update_rate: float = 0.001
    grad_clip: float = 100.0
    
    # 技能库
    skill_dim: int = 32
    max_skills: int = 100
    skill_threshold: float = 0.9  # 技能激活阈值


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区 (PER)
    
    基于 SumTree 实现，支持优先级采样。
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
        # SumTree 实现
        self.tree = SumTree(capacity)
        
    def push(self, experience: Experience) -> None:
        """添加经验"""
        priority = experience.priority * self.max_priority
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.tree.add(priority)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, List[float]]:
        """
        采样 batch
        
        Args:
            batch_size: 批量大小
            beta: 重要性采样指数
            
        Returns:
            (experiences, indices, importance_weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = []
        experiences = []
        weights = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            sample = random.uniform(a, b)
            idx = self.tree.find(sample)
            indices.append(idx)
            experiences.append(self.buffer[idx])
            
            # 计算重要性权重
            p = self.priorities[idx]
            prob = p / self.tree.total()
            weight = (prob * len(self.buffer)) ** (-beta)
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.max()
        
        return experiences, np.array(indices), weights.tolist()
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """更新优先级"""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + 0.01) ** self.alpha  # 避免零优先级
            self.priorities[idx] = priority
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)


class SumTree:
    """SumTree for PER"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.n_entries = 0
        
    def add(self, priority: float) -> None:
        """添加优先级"""
        idx = self.n_entries
        if idx >= self.capacity:
            # 覆盖最旧的
            idx = idx % self.capacity
        
        self._update(idx, priority)
        self.data[idx] = priority
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
    def _update(self, idx: int, priority: float) -> None:
        """更新树节点"""
        tree_idx = idx + self.capacity
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += diff
            
    def total(self) -> float:
        """总优先级"""
        return self.tree[1]
    
    def find(self, value: float) -> int:
        """查找包含 value 的叶子索引"""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class EWC:
    """
    弹性权重固定 (EWC) 持续学习
    
    防止模型遗忘之前任务的关键权重。
    """
    
    def __init__(self, model: nn.Module, lr: float = 1e-3, lambda_: float = 5000):
        self.model = model
        self.lambda_ = lambda_
        self.lr = lr
        
        # 存储每个任务的最优参数和 Fisher 信息矩阵
        self.params = {}           # task_id -> optimal params
        self.fisher = {}           # task_id -> Fisher matrix (diagonal)
        self.task_ids = set()
        
    def register_task(self, task_id: int, inputs: List[Dict], batch_size: int = 200) -> None:
        """
        注册任务，计算 Fisher 信息矩阵
        
        Args:
            task_id: 任务 ID
            inputs: 样本列表
            batch_size: 计算样本数
        """
        self.model.eval()
        
        # 保存当前参数
        self.params[task_id] = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # 计算 Fisher 信息矩阵 (对角近似)
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }
        
        # 使用样本估计 Fisher
        for i in range(min(batch_size, len(inputs))):
            self.model.zero_grad()
            obs = inputs[i % len(inputs)]
            
            try:
                # 前向传播
                out = self.model(**obs)
                
                # 使用输出的平方作为损失代理
                if isinstance(out, torch.Tensor):
                    loss = (out ** 2).mean()
                else:
                    loss = sum((o ** 2).mean() for o in out if isinstance(o, torch.Tensor))
                
                loss.backward()
                
                # 累加梯度平方
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                        
            except Exception:
                continue
        
        # 归一化
        n = min(batch_size, len(inputs))
        for name in fisher:
            fisher[name] /= n
            
        self.fisher[task_id] = fisher
        self.task_ids.add(task_id)
        self.model.train()
        
    def penalty(self) -> torch.Tensor:
        """
        计算 EWC 惩罚项
        
        Returns:
            惩罚损失
        """
        if not self.task_ids:
            return torch.tensor(0.0)
            
        penalty = 0.0
        current_params = {
            name: param
            for name, param in self.model.named_parameters()
        }
        
        for task_id in self.task_ids:
            for name, param in current_params.items():
                if name in self.fisher[task_id]:
                    diff = param - self.params[task_id][name]
                    fisher = self.fisher[task_id][name]
                    penalty += (fisher * diff ** 2).sum()
                    
        return self.lambda_ * penalty


class MetaLearner:
    """
    元学习器 (MAML 变体)
    
    支持快速适应新任务。
    """
    
    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 1e-3,
        inner_steps: int = 5,
        inner_lr: float = 1e-2
    ):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        
        # 元优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def meta_update(
        self,
        tasks: List[Tuple[List[Dict], List[Dict]]],
        val_tasks: List[Tuple[List[Dict], List[Dict]]]
    ) -> Dict[str, float]:
        """
        执行元更新
        
        Args:
            tasks: 训练任务列表 [(support_set, query_set), ...]
            val_tasks: 验证任务列表
            
        Returns:
            损失字典
        """
        self.optimizer.zero_grad()
        
        total_meta_loss = 0.0
        num_tasks = len(tasks) or 1
        
        for (support, query) in tasks[:num_tasks]:
            # 复制模型用于内循环
            inner_model = copy.deepcopy(self.model)
            inner_opt = torch.optim.Adam(inner_model.parameters(), lr=self.inner_lr)
            
            # 内循环: 在 support set 上训练
            for _ in range(self.inner_steps):
                loss = self._compute_loss(inner_model, support)
                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()
                
            # 外循环: 在 query set 上计算元损失
            meta_loss = self._compute_loss(inner_model, query)
            total_meta_loss += meta_loss
            
            # 清理
            del inner_model
            
        # 元更新
        meta_loss = total_meta_loss / num_tasks
        meta_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        
        self.optimizer.step()
        
        return {'meta_loss': meta_loss.item()}
    
    def _compute_loss(self, model: nn.Module, batch: List[Dict]) -> torch.Tensor:
        """计算损失 (简化版)"""
        if not batch:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        n = 0
        
        for data in batch:
            try:
                out = model(**data)
                if isinstance(out, torch.Tensor):
                    total_loss += out.mean()
                    n += 1
                elif isinstance(out, dict) and 'loss' in out:
                    total_loss += out['loss']
                    n += 1
            except:
                continue
                
        return total_loss / max(n, 1)


class CuriosityModule:
    """
    好奇心驱动的内在奖励模块
    
    基于自监督预测误差计算探索奖励。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 逆动力学模型: (s, s') -> a_pred
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        # 正向模型: (s, a) -> s'_pred
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(device)
        
        # 优化器
        self.opt = torch.optim.Adam(
            list(self.inverse_model.parameters()) + 
            list(self.forward_model.parameters()),
            lr=1e-3
        )
        
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        计算内在奖励 (好奇心)
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            
        Returns:
            内在奖励
        """
        s = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        a = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        ns = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
        
        # 正向预测误差作为好奇心奖励
        with torch.no_grad():
            pred_next = self.forward_model(torch.cat([s, a], dim=-1))
            error = torch.norm(pred_next - ns, dim=-1).mean()
            
        return error.item()
    
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """更新好奇心模型"""
        s = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        a = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        ns = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
        
        # 预测动作
        pred_a = self.inverse_model(torch.cat([s, ns], dim=-1))
        loss_inv = F.mse_loss(pred_a, a)
        
        # 预测下一状态
        pred_ns = self.forward_model(torch.cat([s, a], dim=-1))
        loss_fwd = F.mse_loss(pred_ns, ns)
        
        loss = loss_inv + loss_fwd
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.item()


class SkillLibrary:
    """
    自适应技能库
    
    学习、存储和检索可重用的技能/行为原语。
    """
    
    def __init__(
        self,
        skill_dim: int = 32,
        max_skills: int = 100,
        threshold: float = 0.9
    ):
        self.skill_dim = skill_dim
        self.max_skills = max_skills
        self.threshold = threshold
        
        # 技能列表
        self.skills: List[Dict[str, Any]] = []
        
        # 技能嵌入器
        self.embedder = None
        
    def add_skill(
        self,
        name: str,
        policy: nn.Module,
        description: str = "",
        success_rate: float = 0.0
    ) -> int:
        """
        添加新技能
        
        Args:
            name: 技能名称
            policy: 策略网络
            description: 技能描述
            success_rate: 成功率
            
        Returns:
            技能 ID
        """
        skill_id = len(self.skills)
        
        if skill_id >= self.max_skills:
            # 替换最不常用的技能
            skill_id = self._find_least_used_skill()
            
        self.skills.append({
            'id': skill_id,
            'name': name,
            'policy': copy.deepcopy(policy),
            'description': description,
            'success_rate': success_rate,
            'usage_count': 0,
            'embedding': None,
        })
        
        return skill_id
    
    def select_skill(
        self,
        context: np.ndarray,
        min_success_rate: float = 0.5
    ) -> Optional[Tuple[int, nn.Module]]:
        """
        根据上下文选择技能
        
        Args:
            context: 当前上下文/状态
            min_success_rate: 最低成功率要求
            
        Returns:
            (skill_id, policy) 或 None
        """
        candidates = [
            (s['id'], s['policy'], s['success_rate'])
            for s in self.skills
            if s['success_rate'] >= min_success_rate
        ]
        
        if not candidates:
            return None
            
        # 按成功率加权随机选择
        total_rate = sum(s[2] for s in candidates)
        r = random.uniform(0, total_rate)
        
        cumulative = 0
        for skill_id, policy, rate in candidates:
            cumulative += rate
            if r <= cumulative:
                self.skills[skill_id]['usage_count'] += 1
                return skill_id, policy
                
        return None
    
    def update_skill(self, skill_id: int, success_rate: float) -> None:
        """更新技能成功率"""
        if 0 <= skill_id < len(self.skills):
            s = self.skills[skill_id]
            # 指数移动平均更新
            s['success_rate'] = 0.9 * s['success_rate'] + 0.1 * success_rate
    
    def _find_least_used_skill(self) -> int:
        """找到使用次数最少的技能"""
        if not self.skills:
            return 0
        return min(range(len(self.skills)), key=lambda i: self.skills[i]['usage_count'])


class AutonomousLearningAgent:
    """
    自主学习智能体
    
    整合所有学习组件:
    - 优先经验回放 (PER)
    - EWC 持续学习
    - 元学习 (MAML)
    - 好奇心探索
    - 技能库
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[AutonomousLearningConfig] = None
    ):
        self.config = config or AutonomousLearningConfig()
        self.model = model
        
        # 经验回放
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_capacity,
            alpha=self.config.per_alpha
        )
        
        # EWC 持续学习
        self.ewc = EWC(
            model,
            lambda_=self.config.ewc_lambda
        )
        
        # 元学习器
        self.meta_learner = MetaLearner(
            model,
            meta_lr=self.config.meta_lr,
            inner_steps=self.config.inner_steps
        )
        
        # 好奇心模块
        self.curiosity = None
        
        # 技能库
        self.skill_library = SkillLibrary(
            skill_dim=self.config.skill_dim,
            max_skills=self.config.max_skills,
            threshold=self.config.skill_threshold
        )
        
        # 训练状态
        self.current_task_id = 0
        self.total_steps = 0
        self.episodes = 0
        
    def store(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool,
        td_error: Optional[float] = None
    ) -> None:
        """存储经验"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=abs(td_error) if td_error else 1.0,
            task_id=self.current_task_id
        )
        self.replay_buffer.push(experience)
        
        # 更新 EWC
        if self.total_steps > 0 and self.total_steps % 1000 == 0:
            self._update_ewc()
            
    def _update_ewc(self) -> None:
        """更新 EWC Fisher 矩阵"""
        # 采样计算 Fisher
        if len(self.replay_buffer) > 0:
            samples = random.sample(
                list(self.replay_buffer.buffer),
                min(self.config.fisher_samples, len(self.replay_buffer))
            )
            self.ewc.register_task(self.current_task_id, samples)
            
    def compute_loss(self, batch: List[Experience]) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失
        
        Returns:
            (total_loss, loss_dict)
        """
        if not batch:
            return torch.tensor(0.0), {}
            
        # 提取数据
        states = batch[0].state  # 示例
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        dones = np.array([e.done for e in batch])
        
        # TODO: 实际模型前向传播计算损失
        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        ewc_loss = self.ewc.penalty()
        
        total_loss = policy_loss + value_loss + ewc_loss
        
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def train_step(self) -> Dict[str, float]:
        """单步训练"""
        self.total_steps += 1
        
        # 采样
        batch, indices, weights = self.replay_buffer.sample(
            self.config.batch_size,
            beta=self.config.per_beta
        )
        
        if not batch:
            return {}
            
        # 计算损失
        loss, loss_dict = self.compute_loss(batch)
        
        # 反向传播
        # (实际实现需要模型训练逻辑)
        
        # 更新优先级
        # td_errors = ... (计算 TD 误差)
        # self.replay_buffer.update_priorities(indices, td_errors)
        
        return loss_dict
    
    def set_task(self, task_id: int) -> None:
        """切换任务"""
        self.current_task_id = task_id
        
    def learn_from_demonstration(
        self,
        demonstrations: List[Dict]
    ) -> Dict[str, float]:
        """
        从演示中学习 (模仿学习)
        
        Args:
            demonstrations: 演示轨迹列表
            
        Returns:
            学习统计
        """
        total_loss = 0.0
        
        for demo in demonstrations:
            try:
                out = self.model(**demo)
                if isinstance(out, dict) and 'loss' in out:
                    loss = out['loss']
                elif isinstance(out, torch.Tensor):
                    loss = out.mean()
                else:
                    continue
                    
                total_loss += loss.item()
            except:
                continue
                
        return {'imitation_loss': total_loss / max(len(demonstrations), 1)}
