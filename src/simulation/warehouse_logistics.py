"""
仓库物流仿真场景
================

SuperModel 仓库物流具身智能仿真:
- 多货架仓储环境
- 物料取放任务 (Pick-and-Place)
- 动态障碍避让
- 多AGV协调调度
- 持续学习优化路径

支持 Gymnasium 接口，可与 RL 训练框架集成
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Callable
from enum import Enum
import time


class TaskType(Enum):
    """仓库任务类型"""
    PICKUP = "pickup"       # 取货
    DELIVERY = "delivery"   # 送货
    SHELF_RESTOCK = "restock"  # 货架补货
    INVENTORY = "inventory"  # 盘点


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ShelfPosition:
    """货架位置"""
    x: float
    y: float
    shelf_id: int
    level: int = 1  # 货架层 (1=底层, 3=顶层)


@dataclass
class WarehouseTask:
    """仓库任务"""
    task_id: str
    task_type: TaskType
    source: ShelfPosition
    destination: ShelfPosition
    priority: int = 1  # 1=低, 5=高
    estimated_time: float = 60.0  # 预估完成时间 (秒)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agv: Optional[str] = None


@dataclass
class WarehouseObstacle:
    """仓库动态障碍"""
    x: float
    y: float
    radius: float = 0.3  # 障碍半径 (m)
    vx: float = 0.0      # 移动速度 (m/s)
    vy: float = 0.0
    obstacle_type: str = "human"  # human, robot, pallet


@dataclass
class WarehouseState:
    """仓库环境状态"""
    # AGV 状态 (id -> [x, y, theta, v, task_id])
    agv_states: Dict[str, List[float]] = field(default_factory=dict)

    # 任务队列
    tasks: List[WarehouseTask] = field(default_factory=list)

    # 动态障碍
    obstacles: List[WarehouseObstacle] = field(default_factory=list)

    # 仿真时间
    sim_time: float = 0.0

    # 环境状态
    warehouse_layout: str = "single_aisle"  # single_aisle, multi_aisle, u_shape

    # 货架位置
    shelves: List[ShelfPosition] = field(default_factory=list)

    # 取放操作状态
    pick_in_progress: Dict[str, bool] = field(default_factory=dict)  # agv_id -> bool


class WarehouseLogisticsScenario:
    """
    仓库物流仿真场景

    支持:
    - 单AGV任务执行
    - 多AGV协调调度
    - 动态障碍避让
    - 取放任务管理
    """

    def __init__(
        self,
        warehouse_layout: str = "multi_aisle",
        num_agvs: int = 2,
        num_shelves: int = 12,
        aisle_width: float = 2.5,
        shelf_spacing: float = 1.5,
        dt: float = 0.01,
        grade: str = 'M',
    ):
        self.warehouse_layout = warehouse_layout
        self.num_agvs = num_agvs
        self.num_shelves = num_shelves
        self.aisle_width = aisle_width
        self.shelf_spacing = shelf_spacing
        self.dt = dt
        self.grade = grade
        self.sim_time = 0.0

        # AGV 物理参数 (根据等级)
        self.agv_params = self._get_agv_params(grade)

        # 初始化环境
        self._init_shelves()
        self._init_agvs()
        self._init_tasks()

        # 动态障碍列表
        self.obstacles: List[WarehouseObstacle] = []

        # 任务完成统计
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_distance': 0.0,
            'collision_count': 0,
        }

    def _get_agv_params(self, grade: str) -> Dict[str, float]:
        """获取 AGV 物理参数"""
        params = {
            'S': {'max_speed': 0.5, 'max_accel': 0.5, 'turn_radius': 0.3},
            'M': {'max_speed': 2.0, 'max_accel': 1.0, 'turn_radius': 0.5},
            'L': {'max_speed': 3.0, 'max_accel': 1.5, 'turn_radius': 0.8},
            'XL': {'max_speed': 5.0, 'max_accel': 2.0, 'turn_radius': 1.0},
            'XXL': {'max_speed': 8.0, 'max_accel': 3.0, 'turn_radius': 1.5},
        }
        return params.get(grade, params['M'])

    def _init_shelves(self):
        """初始化货架布局"""
        self.shelves: List[ShelfPosition] = []

        if self.warehouse_layout == "single_aisle":
            # 单通道: 两排货架相对
            for row in range(3):
                for col in range(self.num_shelves // 3):
                    shelf_id = row * (self.num_shelves // 3) + col
                    x = -5.0 + col * self.shelf_spacing
                    y = -3.0 + row * 3.0
                    for level in range(1, 4):
                        self.shelves.append(ShelfPosition(x=x, y=y, shelf_id=shelf_id, level=level))

        elif self.warehouse_layout == "multi_aisle":
            # 多通道: 多个 aisle
            num_aisles = 3
            for aisle in range(num_aisles):
                aisle_x = -8.0 + aisle * 6.0
                for row in range(2):
                    for col in range(self.num_shelves // (num_aisles * 2)):
                        shelf_id = aisle * 4 + row * 2 + col
                        x = aisle_x + col * self.shelf_spacing
                        y = -2.5 + row * 5.0
                        for level in range(1, 4):
                            self.shelves.append(ShelfPosition(x=x, y=y, shelf_id=shelf_id, level=level))

        elif self.warehouse_layout == "u_shape":
            # U形布局
            positions = [
                (-6, -4), (-3, -4), (0, -4), (3, -4), (6, -4),
                (-6, 4),  (-3, 4),  (0, 4),  (3, 4),  (6, 4),
                (-6, 0),  (6, 0),
            ]
            for i, (x, y) in enumerate(positions):
                for level in range(1, 4):
                    self.shelves.append(ShelfPosition(x=x, y=y, shelf_id=i, level=level))

    def _init_agvs(self):
        """初始化 AGV"""
        self.agvs: Dict[str, Dict[str, Any]] = {}

        start_positions = [
            (0.0, 0.0, 0.0),   # AGV-1: 仓库入口
            (0.0, -2.0, 0.0),  # AGV-2: 备用位置
        ]

        for i in range(self.num_agvs):
            agv_id = f"AGV-{i+1}"
            x, y, theta = start_positions[i % len(start_positions)]
            self.agvs[agv_id] = {
                'x': x, 'y': y, 'theta': theta,
                'v': 0.0, 'omega': 0.0,
                'task_id': None,
                'state': 'idle',  # idle, moving, picking, delivering
                'path': [],
                'path_index': 0,
                'cargo': None,  # 货物重量 (kg)
            }

    def _init_tasks(self):
        """初始化任务队列"""
        self.tasks: List[WarehouseTask] = []

        if self.shelves:
            import random
            for i in range(min(5, len(self.shelves))):
                source_idx = i % len(self.shelves)
                dest_idx = (i + 3) % len(self.shelves)

                source = self.shelves[source_idx]
                dest = self.shelves[dest_idx]

                task_type = TaskType.PICKUP if i % 2 == 0 else TaskType.DELIVERY
                task = WarehouseTask(
                    task_id=f"TASK-{i+1:03d}",
                    task_type=task_type,
                    source=source,
                    destination=dest,
                    priority=random.randint(1, 5),
                )
                self.tasks.append(task)

    def add_task(self, task: WarehouseTask) -> bool:
        """添加新任务"""
        self.tasks.append(task)
        return True

    def assign_task(self, agv_id: str, task_id: str) -> bool:
        """为 AGV 分配任务"""
        if agv_id not in self.agvs:
            return False

        for task in self.tasks:
            if task.task_id == task_id and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agv = agv_id
                self.agvs[agv_id]['task_id'] = task_id
                self.agvs[agv_id]['state'] = 'moving'
                return True
        return False

    def _plan_path(self, agv_id: str, target_x: float, target_y: float) -> List[Tuple[float, float]]:
        """简单路径规划 (直线 + 转向)"""
        agv = self.agvs[agv_id]
        start_x, start_y = agv['x'], agv['y']

        # 简化为分段直线路径
        path = []
        num_segments = 3
        for i in range(num_segments + 1):
            t = i / num_segments
            px = start_x + (target_x - start_x) * t
            py = start_y + (target_y - start_y) * t
            path.append((px, py))

        return path

    def _check_collision(self, agv_id: str, new_x: float, new_y: float) -> bool:
        """检查碰撞"""
        agv = self.agvs[agv_id]
        agv_radius = 0.25  # AGV 半径 (m)

        # 检查与货架的碰撞
        for shelf in self.shelves:
            dist = np.sqrt((new_x - shelf.x)**2 + (new_y - shelf.y)**2)
            if dist < agv_radius + 0.3:  # 货架安全距离
                return True

        # 检查与其他 AGV 的碰撞
        for other_id, other in self.agvs.items():
            if other_id == agv_id:
                continue
            dist = np.sqrt((new_x - other['x'])**2 + (new_y - other['y'])**2)
            if dist < agv_radius * 2 + 0.1:
                self.stats['collision_count'] += 1
                return True

        # 检查与动态障碍的碰撞
        for obs in self.obstacles:
            dist = np.sqrt((new_x - obs.x)**2 + (new_y - obs.y)**2)
            if dist < agv_radius + obs.radius:
                self.stats['collision_count'] += 1
                return True

        return False

    def add_dynamic_obstacle(self, x: float, y: float, obstacle_type: str = "human") -> WarehouseObstacle:
        """添加动态障碍 (如移动的人员)"""
        obstacle = WarehouseObstacle(
            x=x, y=y, radius=0.3,
            vx=np.random.uniform(-0.2, 0.2),
            vy=np.random.uniform(-0.2, 0.2),
            obstacle_type=obstacle_type,
        )
        self.obstacles.append(obstacle)
        return obstacle

    def update_obstacles(self, dt: float):
        """更新动态障碍位置"""
        for obs in self.obstacles:
            obs.x += obs.vx * dt
            obs.y += obs.vy * dt

            # 边界反弹
            bounds = {'x': (-10, 10), 'y': (-6, 6)}
            if obs.x < bounds['x'][0] or obs.x > bounds['x'][1]:
                obs.vx *= -1
                obs.x = np.clip(obs.x, *bounds['x'])
            if obs.y < bounds['y'][0] or obs.y > bounds['y'][1]:
                obs.vy *= -1
                obs.y = np.clip(obs.y, *bounds['y'])

    def step(self, dt: float = None) -> Dict[str, Any]:
        """
        执行一步仿真

        Returns:
            Dict 包含环境状态和奖励信息
        """
        dt = dt or self.dt
        self.sim_time += dt

        # 更新动态障碍
        self.update_obstacles(dt)

        # 更新各 AGV
        for agv_id, agv in self.agvs.items():
            if agv['state'] == 'idle' or agv['task_id'] is None:
                continue

            # 找到对应任务
            task = None
            for t in self.tasks:
                if t.task_id == agv['task_id']:
                    task = t
                    break

            if task is None:
                continue

            # 确定目标位置
            if task.task_type == TaskType.PICKUP:
                target_x, target_y = task.source.x, task.source.y
            else:
                target_x, target_y = task.destination.x, task.destination.y

            # 规划路径
            if not agv['path'] or agv['path_index'] >= len(agv['path']):
                agv['path'] = self._plan_path(agv_id, target_x, target_y)
                agv['path_index'] = 0

            # 沿路径移动
            if agv['path_index'] < len(agv['path']):
                target = agv['path'][agv['path_index']]
                dx = target[0] - agv['x']
                dy = target[1] - agv['y']
                dist = np.sqrt(dx**2 + dy**2)

                if dist < 0.05:
                    agv['path_index'] += 1
                else:
                    # 计算速度
                    max_speed = self.agv_params['max_speed']
                    speed = min(max_speed, dist * 2)

                    # 检查碰撞
                    new_x = agv['x'] + (dx / dist) * speed * dt
                    new_y = agv['y'] + (dy / dist) * speed * dt

                    if not self._check_collision(agv_id, new_x, new_y):
                        move_dist = speed * dt
                        self.stats['total_distance'] += move_dist
                        agv['x'] = new_x
                        agv['y'] = new_y
                        agv['theta'] = np.arctan2(dy, dx)
                        agv['v'] = speed
                    else:
                        agv['v'] = 0.0

            # 检查是否到达目标
            dist_to_target = np.sqrt(
                (agv['x'] - target_x)**2 + (agv['y'] - target_y)**2
            )

            if dist_to_target < 0.1:
                if task.task_type == TaskType.PICKUP:
                    agv['state'] = 'picking'
                    agv['cargo'] = 5.0  # 假设货物 5kg
                    task.task_type = TaskType.DELIVERY
                    agv['path'] = []  # 重新规划到目的地
                    agv['path_index'] = 0

                elif task.task_type == TaskType.DELIVERY:
                    agv['state'] = 'idle'
                    agv['cargo'] = None
                    task.status = TaskStatus.COMPLETED
                    self.stats['tasks_completed'] += 1

        # 构建返回信息
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done()

        return {
            'observation': obs,
            'reward': reward,
            'done': done,
            'info': {
                'sim_time': self.sim_time,
                'stats': self.stats.copy(),
                'agv_states': {k: v.copy() for k, v in self.agvs.items()},
                'active_tasks': sum(1 for t in self.tasks if t.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]),
            }
        }

    def _get_observation(self) -> np.ndarray:
        """获取观测向量"""
        # 4 global + 4 per agv + 2 per obstacle
        obs_dim = 4 + 4 * self.num_agvs + 2 * len(self.obstacles)
        obs = np.zeros(obs_dim, dtype=np.float32)

        # 全局状态
        obs[0] = self.sim_time / 300.0  # 归一化时间
        obs[1] = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED]) / max(1, len(self.tasks))
        obs[2] = len([t for t in self.tasks if t.status == TaskStatus.PENDING]) / max(1, len(self.tasks))
        obs[3] = self.stats['collision_count'] / 100.0

        # AGV 状态
        idx = 4
        for agv in self.agvs.values():
            obs[idx] = agv['x'] / 10.0
            obs[idx+1] = agv['y'] / 10.0
            obs[idx+2] = agv['theta'] / np.pi
            obs[idx+3] = agv['v'] / self.agv_params['max_speed']
            idx += 4

        # 障碍状态
        for obs_item in self.obstacles:
            if idx + 1 < len(obs):
                obs[idx] = obs_item.x / 10.0
                obs[idx+1] = obs_item.y / 10.0
                idx += 2

        return obs

    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0

        # 任务完成奖励
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        reward += completed * 10.0

        # 碰撞惩罚
        reward -= self.stats['collision_count'] * 5.0

        # 移动效率
        total_distance = self.stats['total_distance']
        reward += total_distance * 0.1

        # 能耗惩罚 (速度平方)
        for agv in self.agvs.values():
            reward -= (agv['v'] ** 2) * 0.01

        return reward

    def _is_done(self) -> bool:
        """检查是否结束"""
        # 所有任务完成
        all_done = all(
            t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            for t in self.tasks
        )
        if all_done:
            return True

        # 时间超限
        if self.sim_time > 300.0:
            return True

        return False

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.sim_time = 0.0
        self._init_agvs()
        self._init_tasks()
        self.obstacles = []
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_distance': 0.0,
            'collision_count': 0,
        }
        return self._get_observation()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """渲染环境 (返回 RGB 数组)"""
        img_size = (400, 400)
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 240

        scale_x = img_size[0] / 20.0
        scale_y = img_size[1] / 12.0

        # 绘制货架
        for shelf in self.shelves:
            sx = int((shelf.x + 10.0) * scale_x)
            sy = int((shelf.y + 6.0) * scale_y)
            size = 6
            color = (139, 69, 19)  # 棕色
            img[sy-size:sy+size, sx-size:sx+size] = color

        # 绘制 AGV
        for agv_id, agv in self.agvs.items():
            ax = int((agv['x'] + 10.0) * scale_x)
            ay = int((agv['y'] + 6.0) * scale_y)
            color = (0, 100, 255) if agv['cargo'] is None else (0, 200, 0)
            r = 5
            img[ay-r:ay+r, ax-r:ax+r] = color

        # 绘制障碍
        for obs in self.obstacles:
            ox = int((obs.x + 10.0) * scale_x)
            oy = int((obs.y + 6.0) * scale_y)
            r = int(obs.radius * scale_x)
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        ny, nx = oy+dy, ox+dx
                        if 0 <= ny < img_size[1] and 0 <= nx < img_size[0]:
                            img[ny, nx] = (255, 0, 0)

        return img

    def get_state_dict(self) -> Dict[str, Any]:
        """获取完整状态字典"""
        return {
            'sim_time': self.sim_time,
            'agvs': self.agvs,
            'tasks': [
                {
                    'task_id': t.task_id,
                    'type': t.task_type.value,
                    'status': t.status.value,
                    'priority': t.priority,
                    'assigned': t.assigned_agv,
                }
                for t in self.tasks
            ],
            'obstacles': [
                {'x': o.x, 'y': o.y, 'type': o.obstacle_type}
                for o in self.obstacles
            ],
            'stats': self.stats,
        }


# =============================================================================
# 演示函数
# =============================================================================

def run_demo():
    """仓库物流仿真演示"""
    print("=" * 60)
    print("  SuperModel 仓库物流仿真场景")
    print("=" * 60)

    # 创建环境
    env = WarehouseLogisticsScenario(
        warehouse_layout="multi_aisle",
        num_agvs=2,
        num_shelves=12,
        grade='M',
    )

    print(f"\n环境配置:")
    print(f"  布局: {env.warehouse_layout}")
    print(f"  AGV数量: {env.num_agvs}")
    print(f"  货架数量: {len(env.shelves)}")
    print(f"  任务数量: {len(env.tasks)}")

    # 添加动态障碍
    env.add_dynamic_obstacle(2.0, 1.0, "human")
    env.add_dynamic_obstacle(-3.0, -1.0, "pallet")
    print(f"  动态障碍: {len(env.obstacles)}")

    # 分配初始任务
    for i, agv_id in enumerate(list(env.agvs.keys())[:2]):
        if i < len(env.tasks):
            env.assign_task(agv_id, env.tasks[i].task_id)

    # 运行仿真
    print(f"\n运行 100 步仿真...")
    obs = env.reset()

    for step in range(100):
        result = env.step()
        if step % 20 == 0:
            state = env.get_state_dict()
            print(f"\n  [Step {step:3d}] sim_time={state['sim_time']:.1f}s")
            for agv_id, agv in state['agvs'].items():
                print(f"    {agv_id}: pos=({agv['x']:.2f}, {agv['y']:.2f}), "
                      f"v={agv['v']:.2f}, state={agv['state']}")

    # 最终统计
    print(f"\n最终统计:")
    print(f"  完成任务: {env.stats['tasks_completed']}")
    print(f"  失败任务: {env.stats['tasks_failed']}")
    print(f"  总行驶距离: {env.stats['total_distance']:.2f} m")
    print(f"  碰撞次数: {env.stats['collision_count']}")


if __name__ == "__main__":
    run_demo()
