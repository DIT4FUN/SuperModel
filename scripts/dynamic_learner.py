#!/usr/bin/env python3
"""
dynamic_learner.py - SuperModel 动态增量学习系统
================================================

三种训练模式：
1. 快速学习 (Quick Learning): 发现有价值数据时，短时训练更新模型
2. 空闲学习 (Idle Learning): 模型空闲时，中时训练更新模型  
3. 深度学习 (Deep Learning): 夜间休眠时，长时大量数据训练

Usage:
    python scripts/dynamic_learner.py --mode quick --data_path /path/to/data.h5
    python scripts/dynamic_learner.py --mode idle
    python scripts/dynamic_learner.py --mode deep --duration 8h
"""

import argparse
import os
import sys
import json
import time
import threading
import queue
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np

logger = logging.getLogger("dynamic_learner")


# ========================
# 配置
# ========================

TRAINING_MODES = {
    "quick": {
        "max_steps": 50,
        "learning_rate": 5e-5,
        "batch_size": 16,
        "description": "快速学习 - 发现有价值数据时短时训练"
    },
    "idle": {
        "max_steps": 200,
        "learning_rate": 3e-5,
        "batch_size": 32,
        "description": "空闲学习 - 模型空闲时中时训练"
    },
    "deep": {
        "max_steps": 5000,
        "learning_rate": 1e-5,
        "batch_size": 64,
        "description": "深度学习 - 夜间长时大量数据训练"
    }
}

CHECKPOINT_DIR = "/root/projects/SuperModel/checkpoints/dynamic"
STATE_FILE = "/root/projects/SuperModel/checkpoints/dynamic/learner_state.json"


# ========================
# 数据结构
# ========================

class LearningTask:
    """学习任务"""
    def __init__(
        self,
        task_id: str,
        mode: str,
        data_path: str,
        grade: str = "M",
        max_steps: int = 100,
        learning_rate: float = 3e-5,
        priority: int = 0,
        metadata: Optional[Dict] = None
    ):
        self.task_id = task_id
        self.mode = mode
        self.data_path = data_path
        self.grade = grade
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.status = "pending"  # pending, running, completed, failed
        self.loss_history = []
        self.checkpoint_path = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "data_path": self.data_path,
            "grade": self.grade,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "loss_history": self.loss_history,
            "checkpoint_path": self.checkpoint_path
        }


@dataclass
class LearnerState:
    """学习器状态"""
    model_version: int = 0
    total_training_steps: int = 0
    last_quick_learning: Optional[str] = None
    last_idle_learning: Optional[str] = None
    last_deep_learning: Optional[str] = None
    quick_learning_count: int = 0
    idle_learning_count: int = 0
    deep_learning_count: int = 0
    pending_tasks: int = 0
    is_learning: bool = False
    current_task_id: Optional[str] = None
    model_performance: Dict[str, float] = None

    def __post_init__(self):
        self.model_performance = self.model_performance or {}


# ========================
# 动态学习器
# ========================

class DynamicLearner:
    """
    动态增量学习器
    
    支持三种训练模式：
    - Quick Learning: 快速短时训练，用于即时知识更新
    - Idle Learning: 中时训练，模型空闲时运行
    - Deep Learning: 长时训练，夜间大量数据学习
    """

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        state_file: str = STATE_FILE,
        auto_save_interval: int = 50
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(state_file)
        self.auto_save_interval = auto_save_interval

        # 任务队列
        self.task_queue = queue.PriorityQueue()
        self.task_history: Dict[str, LearningTask] = {}

        # 状态
        self.state = self._load_state()

        # 训练线程
        self.learning_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 回调
        self.on_learning_complete = None
        self.on_learning_progress = None

        logger.info(f"DynamicLearner initialized. Model version: {self.state.model_version}")

    def _load_state(self) -> LearnerState:
        """加载学习器状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    return LearnerState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return LearnerState()

    def _save_state(self):
        """保存学习器状态"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(asdict(self.state), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def submit_task(self, task: LearningTask) -> str:
        """
        提交学习任务
        
        Args:
            task: LearningTask对象
            
        Returns:
            task_id: 任务ID
        """
        # 加入队列（负优先级，因为PriorityQueue是最小堆）
        self.task_queue.put((-task.priority, task.task_id, task))
        self.task_history[task.task_id] = task
        self.state.pending_tasks += 1
        self._save_state()

        logger.info(f"Task submitted: {task.task_id} (mode={task.mode}, priority={task.priority})")
        return task.task_id

    def submit_quick_learning(
        self,
        data_path: str,
        grade: str = "M",
        priority: int = 10,
        metadata: Optional[Dict] = None
    ) -> str:
        """提交快速学习任务"""
        task = LearningTask(
            task_id=f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode="quick",
            data_path=data_path,
            grade=grade,
            max_steps=TRAINING_MODES["quick"]["max_steps"],
            learning_rate=TRAINING_MODES["quick"]["learning_rate"],
            priority=priority,
            metadata=metadata
        )
        return self.submit_task(task)

    def submit_idle_learning(
        self,
        data_path: str,
        grade: str = "M",
        priority: int = 5,
        metadata: Optional[Dict] = None
    ) -> str:
        """提交空闲学习任务"""
        task = LearningTask(
            task_id=f"idle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode="idle",
            data_path=data_path,
            grade=grade,
            max_steps=TRAINING_MODES["idle"]["max_steps"],
            learning_rate=TRAINING_MODES["idle"]["learning_rate"],
            priority=priority,
            metadata=metadata
        )
        return self.submit_task(task)

    def submit_deep_learning(
        self,
        data_path: str,
        grade: str = "M",
        duration_hours: int = 8,
        priority: int = 1,
        metadata: Optional[Dict] = None
    ) -> str:
        """提交深度学习任务"""
        # 根据duration估算steps（假设每秒1 step）
        estimated_steps = duration_hours * 3600
        task = LearningTask(
            task_id=f"deep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode="deep",
            data_path=data_path,
            grade=grade,
            max_steps=min(estimated_steps, TRAINING_MODES["deep"]["max_steps"]),
            learning_rate=TRAINING_MODES["deep"]["learning_rate"],
            priority=priority,
            metadata=metadata
        )
        return self.submit_task(task)

    def _run_training(self, task: LearningTask) -> bool:
        """
        执行训练任务
        
        Returns:
            bool: 训练是否成功
        """
        logger.info(f"Starting training: {task.task_id}")
        task.status = "running"
        task.started_at = datetime.now()
        self.state.is_learning = True
        self.state.current_task_id = task.task_id
        self._save_state()

        # 构建训练命令
        output_dir = self.checkpoint_dir / task.task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "/root/projects/SuperModel/scripts/train_real_data.py",
            "--source_type", "hdf5_dataset",
            "--data_root", os.path.dirname(task.data_path),
            "--grade", task.grade,
            "--batch_size", str(task.max_steps // 10 if task.max_steps < 100 else 32),
            "--seq_len", "32",
            "--max_steps", str(task.max_steps),
            "--lr", str(task.learning_rate),
            "--output_dir", str(output_dir),
            "--fp16",
            "--save_interval", str(self.auto_save_interval),
            "--experiment_name", task.task_id
        ]

        try:
            # 运行训练
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # 监控输出
            for line in process.stdout:
                if "Loss:" in line:
                    # 解析loss
                    try:
                        parts = line.split("Loss:")[1].split("|")
                        loss = float(parts[0].strip())
                        task.loss_history.append(loss)

                        if self.on_learning_progress:
                            self.on_learning_progress(task.task_id, loss, len(task.loss_history))
                    except:
                        pass

                if "Training complete" in line:
                    break

            process.wait()

            if process.returncode == 0:
                task.status = "completed"
                task.completed_at = datetime.now()
                task.checkpoint_path = str(output_dir)

                # 更新状态
                self.state.total_training_steps += task.max_steps
                self.state.model_version += 1
                if task.mode == "quick":
                    self.state.last_quick_learning = task.task_id
                    self.state.quick_learning_count += 1
                elif task.mode == "idle":
                    self.state.last_idle_learning = task.task_id
                    self.state.idle_learning_count += 1
                elif task.mode == "deep":
                    self.state.last_deep_learning = task.task_id
                    self.state.deep_learning_count += 1

                logger.info(f"Training completed: {task.task_id}")
                return True
            else:
                task.status = "failed"
                logger.error(f"Training failed: {task.task_id}")
                return False

        except Exception as e:
            task.status = "failed"
            logger.error(f"Training error: {task.task_id} - {e}")
            return False

        finally:
            self.state.is_learning = False
            self.state.current_task_id = None
            self.state.pending_tasks = max(0, self.state.pending_tasks - 1)
            self._save_state()

    def start(self, background: bool = True):
        """启动学习器"""
        self.is_running = True

        if background:
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()
            logger.info("DynamicLearner started in background")
        else:
            self._learning_loop()

    def _learning_loop(self):
        """学习主循环"""
        logger.info("Learning loop started")

        while self.is_running:
            try:
                # 从队列获取任务（阻塞最多1秒）
                try:
                    neg_priority, task_id, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # 执行训练
                self._run_training(task)

                # 回调
                if task.status == "completed" and self.on_learning_complete:
                    self.on_learning_complete(task)

            except Exception as e:
                logger.error(f"Learning loop error: {e}")

        logger.info("Learning loop stopped")

    def stop(self):
        """停止学习器"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("DynamicLearner stopped")

    def get_status(self) -> Dict:
        """获取学习器状态"""
        return {
            "is_running": self.is_running,
            "is_learning": self.state.is_learning,
            "current_task": self.state.current_task_id,
            "pending_tasks": self.state.pending_tasks,
            "model_version": self.state.model_version,
            "total_training_steps": self.state.total_training_steps,
            "quick_learning_count": self.state.quick_learning_count,
            "idle_learning_count": self.state.idle_learning_count,
            "deep_learning_count": self.state.deep_learning_count,
            "last_quick_learning": self.state.last_quick_learning,
            "last_idle_learning": self.state.last_idle_learning,
            "last_deep_learning": self.state.last_deep_learning
        }

    def get_task_history(self, limit: int = 10) -> List[Dict]:
        """获取任务历史"""
        tasks = sorted(
            self.task_history.values(),
            key=lambda t: t.created_at,
            reverse=True
        )[:limit]
        return [t.to_dict() for t in tasks]


# ========================
# 调度器
# ========================

class LearningScheduler:
    """
    学习调度器
    
    管理三种学习模式的调度：
    - Quick: 随时可触发
    - Idle: 检测到空闲时触发
    - Deep: 定时在夜间执行
    """

    def __init__(self, learner: DynamicLearner):
        self.learner = learner

        # 空闲检测配置
        self.idle_check_interval = 60  # 秒
        self.idle_threshold = 0.1  # CPU/内存使用率阈值
        self.idle_duration_required = 300  # 需要空闲5分钟才触发idle learning

        self.last_activity_time = time.time()
        self.idle_start_time: Optional[float] = None

        # 夜间学习配置
        self.deep_learning_start_hour = 2  # 凌晨2点
        self.deep_learning_end_hour = 6  # 早上6点

        # 调度线程
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 默认数据路径
        self.default_data_paths = [
            "/data/h5_fanuc/fanuc_manipulation.h5",
            "/data/h5_pusht/pusht_keypoints.h5",
            "/data/h5_training/koch_pick_place.h5"
        ]

    def update_activity(self):
        """更新活动状态"""
        self.last_activity_time = time.time()

    def _check_idle(self) -> bool:
        """检查是否应该触发空闲学习"""
        current_time = time.time()
        time_since_activity = current_time - self.last_activity_time

        # 如果最近有活动，不触发
        if time_since_activity < 60:  # 1分钟内
            self.idle_start_time = None
            return False

        # 如果已经空闲足够长时间
        if time_since_activity > self.idle_duration_required:
            # 检查是否是深夜（不触发idle learning）
            current_hour = datetime.now().hour
            if self.deep_learning_start_hour <= current_hour <= self.deep_learning_end_hour:
                return False  # 这是deep learning的时间段
            return True

        return False

    def _check_deep_learning_time(self) -> bool:
        """检查是否在深度学习时间窗口"""
        current_hour = datetime.now().hour
        return self.deep_learning_start_hour <= current_hour <= self.deep_learning_end_hour

    def _get_available_data_path(self) -> Optional[str]:
        """获取可用的数据路径"""
        for path in self.default_data_paths:
            if os.path.exists(path):
                return path
        return None

    def start(self, background: bool = True):
        """启动调度器"""
        self.is_running = True

        if background:
            self.scheduler_thread = threading.Thread(target=self._schedule_loop, daemon=True)
            self.scheduler_thread.start()
            logger.info("LearningScheduler started")
        else:
            self._schedule_loop()

    def _schedule_loop(self):
        """调度主循环"""
        logger.info("Schedule loop started")

        while self.is_running:
            try:
                # 检查是否有待处理任务
                status = self.learner.get_status()

                # 如果当前不在学习
                if not status["is_learning"]:
                    # 检查是否可以提交新任务
                    available_data = self._get_available_data_path()

                    if available_data:
                        # 1. 检查是否可以进行深度学习
                        if self._check_deep_learning_time():
                            logger.info("Deep learning time window active")
                            # 提交深度学习任务
                            self.learner.submit_deep_learning(
                                data_path=available_data,
                                grade="M",
                                duration_hours=4,
                                priority=1,
                                metadata={"trigger": "scheduler_deep"}
                            )
                            logger.info("Deep learning task submitted")

                        # 2. 检查是否可以进行空闲学习
                        elif self._check_idle():
                            logger.info("Idle condition detected")
                            self.learner.submit_idle_learning(
                                data_path=available_data,
                                grade="M",
                                priority=5,
                                metadata={"trigger": "scheduler_idle"}
                            )
                            logger.info("Idle learning task submitted")

                # 每分钟检查一次
                for _ in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Schedule loop error: {e}")

        logger.info("Schedule loop stopped")

    def stop(self):
        """停止调度器"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("LearningScheduler stopped")


# ========================
# 命令行接口
# ========================

def main():
    parser = argparse.ArgumentParser(description="SuperModel 动态增量学习系统")

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # Quick learning
    quick_parser = subparsers.add_parser("quick", help="快速学习")
    quick_parser.add_argument("--data", default="/data/h5_fanuc/fanuc_manipulation.h5", help="数据路径")
    quick_parser.add_argument("--grade", default="M", help="AGV等级")

    # Idle learning
    idle_parser = subparsers.add_parser("idle", help="空闲学习")
    idle_parser.add_argument("--data", default="/data/h5_fanuc/fanuc_manipulation.h5", help="数据路径")
    idle_parser.add_argument("--grade", default="M", help="AGV等级")

    # Deep learning
    deep_parser = subparsers.add_parser("deep", help="深度学习")
    deep_parser.add_argument("--data", default="/data/h5_fanuc/fanuc_manipulation.h5", help="数据路径")
    deep_parser.add_argument("--grade", default="M", help="AGV等级")
    deep_parser.add_argument("--hours", type=int, default=4, help="训练小时数")

    # Status
    subparsers.add_parser("status", help="查看状态")

    # Submit task
    submit_parser = subparsers.add_parser("submit", help="提交学习任务")
    submit_parser.add_argument("--mode", required=True, choices=["quick", "idle", "deep"])
    submit_parser.add_argument("--data", required=True, help="数据路径")
    submit_parser.add_argument("--grade", default="M", help="AGV等级")
    submit_parser.add_argument("--steps", type=int, help="训练步数")
    submit_parser.add_argument("--priority", type=int, default=5, help="优先级")

    # Start scheduler
    subparsers.add_parser("start", help="启动学习调度器")

    # History
    subparsers.add_parser("history", help="查看任务历史")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    learner = DynamicLearner()

    if args.command == "quick":
        task_id = learner.submit_quick_learning(
            data_path=args.data,
            grade=args.grade
        )
        print(f"Quick learning task submitted: {task_id}")
        learner.start(background=False)

    elif args.command == "idle":
        task_id = learner.submit_idle_learning(
            data_path=args.data,
            grade=args.grade
        )
        print(f"Idle learning task submitted: {task_id}")
        learner.start(background=False)

    elif args.command == "deep":
        task_id = learner.submit_deep_learning(
            data_path=args.data,
            grade=args.grade,
            duration_hours=args.hours
        )
        print(f"Deep learning task submitted: {task_id}")
        learner.start(background=False)

    elif args.command == "status":
        status = learner.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "submit":
        task = LearningTask(
            task_id=f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=args.mode,
            data_path=args.data,
            grade=args.grade,
            max_steps=args.steps or TRAINING_MODES[args.mode]["max_steps"],
            priority=args.priority
        )
        learner.submit_task(task)
        print(f"Task submitted: {task.task_id}")

    elif args.command == "start":
        scheduler = LearningScheduler(learner)
        scheduler.start()
        print("Learning scheduler started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
            learner.stop()

    elif args.command == "history":
        history = learner.get_task_history(limit=20)
        print(json.dumps(history, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
