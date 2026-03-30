"""
任务规划模块
============

层次化任务规划
- 任务图构建
- HTN (层次任务网络) 规划
- 动作序列生成
- 任务监控与重规划
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """任务"""
    id: str
    name: str
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    parameters: Dict[str, Any] = field(default_factory=dict)
    subtasks: List['Task'] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    
    def duration(self) -> float:
        """任务持续时间"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class TaskSpec:
    """任务规格 (用于创建任务)"""
    name: str
    goal_state: Dict[str, Any]  # 目标状态
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件
    max_depth: int = 5          # 最大分解深度
    timeout: float = 60.0       # 超时时间


@dataclass
class WorldState:
    """世界状态"""
    objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    robot_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def copy(self) -> 'WorldState':
        """深拷贝"""
        new_state = WorldState(timestamp=self.timestamp)
        new_state.objects = {k: v.copy() for k, v in self.objects.items()}
        new_state.robot_state = self.robot_state.copy()
        return new_state
    
    def apply_action(self, action: str, params: Dict[str, Any]):
        """应用动作，更新状态"""
        # 预定义动作效果库
        action_effects = {
            'move_to': lambda s, p: s.robot_state.update({
                'position': p.get('position', s.robot_state.get('position', np.zeros(3))),
                'status': 'moved'
            }),
            'grasp': lambda s, p: s.objects.get(p.get('object', ''), {}).update({'grasped': True}),
            'release': lambda s, p: s.objects.update({
                name: {**obj, 'grasped': False} 
                for name, obj in s.objects.items() 
                if obj.get('grasped', False)
            }),
            'lift': lambda s, p: s.robot_state.update({'height': p.get('height', 0.2)}),
            'place': lambda s, p: s.objects.get(p.get('object', ''), {}).update({
                'position': p.get('position', s.objects.get(p.get('object', {}).get('position', np.zeros(3))))
            }),
            'push': lambda s, p: s.objects.get(p.get('object', ''), {}).update({
                'position': s.objects.get(p.get('object'), {}).get('position', np.zeros(3)) + np.array(p.get('direction', [0.1, 0, 0]))
            }),
            'open_gripper': lambda s, p: s.robot_state.update({'gripper_open': True}),
            'close_gripper': lambda s, p: s.robot_state.update({'gripper_open': False}),
        }
        
        effect_fn = action_effects.get(action)
        if effect_fn:
            effect_fn(self, params)
        
        self.timestamp = time.time()


class Action:
    """动作"""
    
    def __init__(
        self,
        name: str,
        precondition: Callable[[WorldState], bool],
        effect: Callable[[WorldState, Dict[str, Any]], None],
        cost: float = 1.0
    ):
        self.name = name
        self.precondition = precondition
        self.effect = effect
        self.cost = cost
    
    def applicable(self, state: WorldState, params: Dict) -> bool:
        """检查动作是否可执行"""
        return self.precondition(state)
    
    def execute(self, state: WorldState, params: Dict):
        """执行动作"""
        self.effect(state, params)


class TaskPlanner:
    """
    任务规划器
    
    使用层次化分解 + 搜索规划
    """
    
    def __init__(
        self,
        action_library: Optional[Dict[str, Action]] = None,
        skill_dispatcher: Optional[Any] = None
    ):
        """
        Args:
            action_library: 可用动作库
            skill_dispatcher: 技能调度器
        """
        self.action_library = action_library or {}
        self.skill_dispatcher = skill_dispatcher
        
        # 任务队列
        self._task_queue: List[Task] = []
        self._current_task: Optional[Task] = None
        
        # 规划状态
        self._world_state: Optional[WorldState] = None
        self._plan_history: List[List[str]] = []
        
    def set_world_state(self, state: WorldState):
        """设置当前世界状态"""
        self._world_state = state
    
    def add_task(self, task: Task):
        """添加任务到队列"""
        self._task_queue.append(task)
        # 按优先级排序
        self._task_queue.sort(key=lambda t: t.priority.value, reverse=True)
    
    def get_next_task(self) -> Optional[Task]:
        """获取下一个待执行任务"""
        if not self._task_queue:
            return None
        
        task = self._task_queue.pop(0)
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        self._current_task = task
        return task
    
    def plan(
        self,
        task_spec: TaskSpec,
        initial_state: Optional[WorldState] = None
    ) -> List[str]:
        """
        规划动作序列
        
        Args:
            task_spec: 任务规格
            initial_state: 初始状态
            
        Returns:
            action_sequence: 动作序列
        """
        if initial_state is not None:
            self._world_state = initial_state.copy()

        # 使用 HTN 层次化规划
        plan = self._htn_plan(task_spec)
        
        self._plan_history.append(plan)
        return plan
    
    def _htn_plan(self, task_spec: TaskSpec) -> List[str]:
        """
        HTN (层次任务网络) 规划
        
        1. 分析任务类型，选择合适的分解方法
        2. 递归分解高层任务为原子动作
        3. 验证分解结果
        
        Args:
            task_spec: 任务规格
            
        Returns:
            动作序列
        """
        task_name = task_spec.name
        goal_state = task_spec.goal_state
        max_depth = task_spec.max_depth
        
        # 方法库：任务名 -> 分解函数列表
        method_registry = {
            'transport': self._decompose_transport,
            'pickup': self._decompose_pickup,
            'place': self._decompose_place,
            'navigate': self._decompose_navigate,
            'inspect': self._decompose_inspect,
            'open_door': self._decompose_open_door,
            'assemble': self._decompose_assemble,
            'disassemble': self._decompose_disassemble,
        }
        
        decompose_fn = method_registry.get(task_name)
        
        if decompose_fn is None:
            # 无分解方法，退化为贪心规划
            return self._greedy_plan(goal_state)
        
        # 递归分解
        state = self._world_state.copy() if self._world_state else WorldState()
        actions = self._decompose_and_resolve(decompose_fn, goal_state, state, depth=0, max_depth=max_depth)
        
        return actions
    
    def _decompose_and_resolve(
        self,
        decompose_fn: callable,
        goal_state: Dict[str, Any],
        state: WorldState,
        depth: int,
        max_depth: int,
        primitive_name: Optional[str] = None
    ) -> List[str]:
        """
        递归分解任务并解析为动作序列
        
        Args:
            decompose_fn: 分解函数 (None 表示原始动作)
            goal_state: 目标状态
            state: 当前世界状态
            depth: 当前深度
            max_depth: 最大深度
            primitive_name: 原始动作名称 (当 decompose_fn 为 None 时)
            
        Returns:
            动作序列
        """
        # 如果是原始动作（非复合任务），直接返回
        if decompose_fn is None:
            return [primitive_name] if primitive_name else []
        
        if depth >= max_depth:
            # 达到最大深度，使用贪心规划
            return self._greedy_plan(goal_state)
        
        try:
            subtasks = decompose_fn(goal_state)
        except Exception:
            return self._greedy_plan(goal_state)
        
        actions = []
        for subtask in subtasks:
            if isinstance(subtask, Task):
                # 递归分解
                method_fn = self._method_for_task(subtask.name)
                if method_fn is not None:
                    # 复合任务，继续分解
                    sub_actions = self._decompose_and_resolve(
                        method_fn,
                        subtask.parameters.get('goal_state', goal_state),
                        state,
                        depth + 1,
                        max_depth
                    )
                    actions.extend(sub_actions)
                else:
                    # 原始动作，直接添加
                    actions.append(subtask.name)
            elif isinstance(subtask, str):
                # 原子动作: 优先使用动作库，否则直接添加到计划
                action = self.action_library.get(subtask)
                if action is not None:
                    if action.applicable(state, {}):
                        action.execute(state, {})
                        actions.append(subtask)
                else:
                    # 动作不在库中，但仍然是有效动作名
                    actions.append(subtask)
        
        return actions
    
    def _method_for_task(self, task_name: str) -> callable:
        """获取任务的分解方法"""
        method_registry = {
            'transport': self._decompose_transport,
            'pickup': self._decompose_pickup,
            'place': self._decompose_place,
            'navigate': self._decompose_navigate,
            'inspect': self._decompose_inspect,
            'open_door': self._decompose_open_door,
            'assemble': self._decompose_assemble,
            'disassemble': self._decompose_disassemble,
        }
        return method_registry.get(task_name)  # None if not found means primitive action
    
    def _decompose_transport(self, goal_state: Dict) -> List:
        """搬运任务分解: pickup -> navigate -> place"""
        return [
            Task(id='t1', name='pickup', parameters={'object': goal_state.get('object')}),
            Task(id='t2', name='navigate', parameters={'target': goal_state.get('destination')}),
            Task(id='t3', name='place', parameters={'location': goal_state.get('destination')}),
        ]
    
    def _decompose_pickup(self, goal_state: Dict) -> List:
        """拾取任务分解: approach -> grasp -> lift"""
        return [
            Task(id='p1', name='approach', parameters={'target': goal_state.get('object')}),
            Task(id='p2', name='grasp', parameters={'object': goal_state.get('object')}),
            Task(id='p3', name='lift', parameters={}),
        ]
    
    def _decompose_place(self, goal_state: Dict) -> List:
        """放置任务分解: move_to -> release -> retract"""
        return [
            Task(id='pl1', name='move_to', parameters={'target': goal_state.get('location')}),
            Task(id='pl2', name='release', parameters={}),
            Task(id='pl3', name='retract', parameters={}),
        ]
    
    def _decompose_navigate(self, goal_state: Dict) -> List:
        """导航任务分解: plan -> follow -> reach"""
        return [
            Task(id='n1', name='plan_route', parameters={'target': goal_state.get('target')}),
            Task(id='n2', name='follow_trajectory', parameters={}),
            Task(id='n3', name='reach_target', parameters={}),
        ]
    
    def _decompose_inspect(self, goal_state: Dict) -> List:
        """检查任务分解: move_to -> sense -> analyze"""
        return [
            Task(id='i1', name='move_to', parameters={'target': goal_state.get('location')}),
            Task(id='i2', name='sense_environment', parameters={}),
            Task(id='i3', name='analyze_data', parameters={}),
        ]
    
    def _decompose_open_door(self, goal_state: Dict) -> List:
        """开门任务分解: approach -> grasp -> pull -> pass"""
        return [
            Task(id='d1', name='move_to', parameters={'target': goal_state.get('door_position')}),
            Task(id='d2', name='grasp', parameters={'object': 'door_handle'}),
            Task(id='d3', name='pull', parameters={}),
            Task(id='d4', name='move_to', parameters={'target': goal_state.get('target_position')}),
        ]
    
    def _decompose_assemble(self, goal_state: Dict) -> List:
        """装配任务分解"""
        return [
            Task(id='a1', name='fetch', parameters={'parts': goal_state.get('parts', [])}),
            Task(id='a2', name='position', parameters={'target': goal_state.get('target')}),
            Task(id='a3', name='fasten', parameters={}),
        ]
    
    def _decompose_disassemble(self, goal_state: Dict) -> List:
        """拆卸任务分解"""
        return [
            Task(id='dd1', name='unfasten', parameters={}),
            Task(id='dd2', name='separate', parameters={'target': goal_state.get('target')}),
            Task(id='dd3', name='remove', parameters={'parts': goal_state.get('parts', [])}),
        ]
    
    def _greedy_plan(self, goal_state: Dict[str, Any]) -> List[str]:
        """贪心规划"""
        plan = []
        state = self._world_state.copy() if self._world_state else WorldState()
        
        remaining_goals = goal_state.copy()
        max_iterations = 20
        iteration = 0
        
        while remaining_goals and iteration < max_iterations:
            iteration += 1
            
            # 找最接近目标的动作
            best_action = None
            best_score = -1
            
            for name, action in self.action_library.items():
                if action.applicable(state, {}):
                    score = self._score_action(action, remaining_goals)
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            if best_action is None:
                break
            
            # 执行动作
            best_action.execute(state, {})
            plan.append(best_action.name)
            
            # 更新目标
            for key in list(remaining_goals.keys()):
                if self._goal_satisfied(state, key, remaining_goals[key]):
                    del remaining_goals[key]
        
        return plan
    
    def _score_action(self, action: Action, goals: Dict) -> float:
        """评估动作对目标的贡献"""
        # 简化评分
        return action.cost
    
    def _goal_satisfied(self, state: WorldState, key: str, value: Any) -> bool:
        """检查目标是否满足"""
        parts = key.split('.')
        obj = state.objects.get(parts[0], {})
        
        for part in parts[1:]:
            if isinstance(obj, dict):
                obj = obj.get(part, {})
            else:
                return False
        
        return obj == value
    
    def monitor_and_replan(
        self,
        current_state: WorldState,
        failed_action: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        监控并重规划
        
        当执行失败时，重新规划
        """
        if failed_action:
            # 分析失败原因
            print(f"[TaskPlanner] Action failed: {failed_action}")
        
        # 简单策略: 从当前状态重新规划
        if self._current_task:
            goal = self._current_task.parameters.get("goal_state", {})
            return self._greedy_plan(goal)
        
        return None
    
    def cancel_current_task(self):
        """取消当前任务"""
        if self._current_task:
            self._current_task.status = TaskStatus.CANCELLED
            self._current_task.end_time = time.time()
            self._current_task = None


class HierarchicalPlanner(TaskPlanner):
    """
    层次化任务网络 (HTN) 规划器
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 方法库
        self._methods: Dict[str, List[Callable]] = {}
        
        # 注册默认方法
        self._register_default_methods()
    
    def _register_default_methods(self):
        """注册默认分解方法"""
        
        def pickup_method(task_params: Dict) -> List[Task]:
            """拾取方法分解"""
            return [
                Task(id="approach", name="move_near", parameters={"target": task_params.get("object")}),
                Task(id="grasp", name="grasp", parameters={"object": task_params.get("object")}),
                Task(id="lift", name="move_up", parameters={})
            ]
        
        def place_method(task_params: Dict) -> List[Task]:
            """放置方法分解"""
            return [
                Task(id="move_to_place", name="move_to", parameters={"target": task_params.get("location")}),
                Task(id="release", name="release", parameters={}),
                Task(id="retract", name="move_back", parameters={})
            ]
        
        def navigate_method(task_params: Dict) -> List[Task]:
            """导航方法分解"""
            return [
                Task(id="plan_path", name="plan_route", parameters={"target": task_params.get("target")}),
                Task(id="follow_path", name="follow_trajectory", parameters={}),
                Task(id="reach_goal", name="reach_target", parameters={})
            ]
        
        def inspect_method(task_params: Dict) -> List[Task]:
            """检查方法分解"""
            return [
                Task(id="move_to_inspect", name="move_to", parameters={"target": task_params.get("location")}),
                Task(id="sense", name="sense_environment", parameters={}),
                Task(id="analyze", name="analyze_data", parameters={})
            ]
        
        def transport_method(task_params: Dict) -> List[Task]:
            """搬运方法分解 (pickup + navigate + place)"""
            return [
                Task(id="pickup_obj", name="pickup", parameters={"object": task_params.get("object")}),
                Task(id="navigate_to_dest", name="navigate", parameters={"target": task_params.get("destination")}),
                Task(id="place_obj", name="place", parameters={"location": task_params.get("destination")})
            ]
        
        def open_door_method(task_params: Dict) -> List[Task]:
            """开门方法分解"""
            return [
                Task(id="approach_door", name="move_to", parameters={"target": task_params.get("door_position")}),
                Task(id="grasp_handle", name="grasp", parameters={"object": "door_handle"}),
                Task(id="pull_door", name="pull", parameters={}),
                Task(id="pass_through", name="move_to", parameters={"target": task_params.get("target_position")})
            ]
        
        self._methods["pickup"] = [pickup_method]
        self._methods["place"] = [place_method]
        self._methods["navigate"] = [navigate_method]
        self._methods["inspect"] = [inspect_method]
        self._methods["transport"] = [transport_method]
        self._methods["open_door"] = [open_door_method]
    
    def register_method(self, task_name: str, method: Callable[[Dict], List[Task]]):
        """
        注册自定义分解方法
        
        Args:
            task_name: 任务名称
            method: 分解方法函数
        """
        if task_name not in self._methods:
            self._methods[task_name] = []
        self._methods[task_name].append(method)
    
    def get_available_methods(self, task_name: str) -> int:
        """获取任务可用的方法数量"""
        return len(self._methods.get(task_name, []))
    
    def backtrack(
        self,
        task: Task,
        failed_subtasks: List[str],
        attempted_methods: Optional[List[int]] = None
    ) -> Tuple[List[Task], List[int]]:
        """
        回溯搜索：当子任务失败时尝试替代方法
        
        采用深度优先搜索策略遍历方法空间，支持多级回溯。
        
        Args:
            task: 当前任务
            failed_subtasks: 失败的子任务ID列表
            attempted_methods: 已尝试的方法索引列表 (用于避免重复尝试)
            
        Returns:
            (替代子任务列表, 更新后的已尝试方法索引)
        """
        if attempted_methods is None:
            attempted_methods = []
        
        methods = self._methods.get(task.name, [])
        
        # 尝试下一个未尝试的方法
        for idx, method in enumerate(methods):
            if idx in attempted_methods:
                continue
            
            try:
                subtasks = method(task.parameters)
                if subtasks and len(subtasks) > 0:
                    # 验证子任务可行性
                    valid_subtasks = []
                    for st in subtasks:
                        if st.id not in failed_subtasks:
                            valid_subtasks.append(st)
                    
                    if valid_subtasks:
                        return valid_subtasks, attempted_methods + [idx]
            except Exception as e:
                print(f"[HierarchicalPlanner] Method {idx} for task '{task.name}' failed: {e}")
                continue
        
        # 无法回溯，返回空
        return [], attempted_methods
    
    def plan_with_replanning(
        self,
        task_spec: TaskSpec,
        initial_state: WorldState,
        max_replan_attempts: int = 3
    ) -> Tuple[List[Task], Dict[str, Any]]:
        """
        带重规划的分层规划
        
        当计划执行失败时，自动尝试替代方法。
        
        Args:
            task_spec: 任务规格
            initial_state: 初始世界状态
            max_replan_attempts: 最大重规划次数
            
        Returns:
            (最终任务序列, 元数据含重规划历史)
        """
        replan_history = []
        current_state = initial_state.copy()
        
        for attempt in range(max_replan_attempts):
            # 生成计划
            tasks, metadata = self.plan_hierarchical(
                task_spec, initial_state=current_state, validate=False
            )
            
            # 验证
            is_valid, reason = self.validate_plan(tasks, current_state)
            
            replan_entry = {
                "attempt": attempt,
                "num_tasks": len(tasks),
                "task_names": [t.name for t in tasks],
                "is_valid": is_valid,
                "reason": reason
            }
            replan_history.append(replan_entry)
            
            if is_valid:
                metadata["replan_history"] = replan_history
                metadata["successful_attempt"] = attempt
                return tasks, metadata
            
            # 找到失败点并回溯
            failed_task_names = []
            for i, task in enumerate(tasks):
                action = self.action_library.get(task.name)
                if action and not action.applicable(current_state, task.parameters):
                    failed_task_names.append(task.name)
            
            if not failed_task_names:
                break
            
            # 尝试回溯重规划
            root_task = Task(
                id="root_replan",
                name=task_spec.name,
                parameters={"goal_state": task_spec.goal_state}
            )
            alt_tasks, _ = self.backtrack(root_task, failed_task_names)
            
            if alt_tasks:
                tasks = alt_tasks
                is_valid, reason = self.validate_plan(tasks, current_state)
                if is_valid:
                    metadata["replan_history"] = replan_history
                    metadata["successful_attempt"] = attempt
                    return tasks, metadata
        
        # 所有尝试均失败
        metadata["replan_history"] = replan_history
        metadata["successful_attempt"] = -1
        metadata["final_state"] = str(current_state)
        return [], metadata
    
    def estimate_plan_cost(self, tasks: List[Task]) -> float:
        """
        估计计划成本
        
        Args:
            tasks: 任务列表
            
        Returns:
            估计的总成本
        """
        total_cost = 0.0
        for task in tasks:
            action = self.action_library.get(task.name)
            if action:
                total_cost += action.cost
            else:
                total_cost += 1.0  # 默认成本
        return total_cost
    
    def validate_plan(self, tasks: List[Task], initial_state: WorldState) -> Tuple[bool, str]:
        """
        验证计划可行性
        
        Args:
            tasks: 任务列表
            initial_state: 初始状态
            
        Returns:
            (is_valid, reason)
        """
        if not tasks:
            return False, "Empty plan"
        
        state = initial_state.copy()
        
        for i, task in enumerate(tasks):
            action = self.action_library.get(task.name)
            if action is None:
                return False, f"Unknown action at step {i}: {task.name}"
            
            if not action.applicable(state, task.parameters):
                return False, f"Action not applicable at step {i}: {task.name}"
            
            try:
                action.execute(state, task.parameters)
            except Exception as e:
                return False, f"Action execution failed at step {i}: {task.name}, error: {e}"
        
        return True, "Plan is valid"
    
    def decompose_task(
        self,
        task: Task,
        depth: int = 0,
        max_depth: int = 5
    ) -> List[Task]:
        """
        分解任务为子任务
        
        Args:
            task: 待分解任务
            depth: 当前深度
            max_depth: 最大深度
            
        Returns:
            叶子任务列表
        """
        if depth >= max_depth:
            return [task]
        
        methods = self._methods.get(task.name, [])
        if not methods:
            return [task]
        
        # 选择第一个适用方法
        for method in methods:
            subtasks = method(task.parameters)
            if subtasks:
                flat_subtasks = []
                for st in subtasks:
                    flat_subtasks.extend(self.decompose_task(st, depth + 1, max_depth))
                return flat_subtasks
        
        return [task]
    
    def plan_hierarchical(
        self,
        task_spec: TaskSpec,
        initial_state: Optional[WorldState] = None,
        validate: bool = True
    ) -> Tuple[List[Task], Dict[str, Any]]:
        """
        层次化规划
        
        1. 创建根任务
        2. 递归分解为叶子任务
        3. (可选) 验证计划可行性
        4. 返回叶子任务序列及元数据
        
        Args:
            task_spec: 任务规格
            initial_state: 初始状态 (用于验证)
            validate: 是否验证计划
            
        Returns:
            (leaf_tasks, metadata): 叶子任务列表及元数据
        """
        root_task = Task(
            id="root",
            name=task_spec.name,
            parameters={"goal_state": task_spec.goal_state}
        )
        
        # 分解
        leaf_tasks = self.decompose_task(root_task, max_depth=task_spec.max_depth)
        
        # 分配ID
        for i, task in enumerate(leaf_tasks):
            if not task.id or task.id == "root":
                task.id = f"task_{i}"
        
        # 估计成本
        estimated_cost = self.estimate_plan_cost(leaf_tasks)
        
        # 验证计划
        metadata = {
            "num_tasks": len(leaf_tasks),
            "estimated_cost": estimated_cost,
            "task_names": [t.name for t in leaf_tasks],
            "is_valid": False,
            "validation_reason": ""
        }
        
        if validate and initial_state is not None:
            is_valid, reason = self.validate_plan(leaf_tasks, initial_state)
            metadata["is_valid"] = is_valid
            metadata["validation_reason"] = reason
        
        return leaf_tasks, metadata
