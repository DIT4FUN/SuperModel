# SuperModel 具身智能模块接口规范
# Embodied Intelligence Module Interface Specification

> **文档版本**: v1.0.0
> **更新**: 2026-04-14
> **项目**: SuperModel 超模态机器人具身智能大脑
> **目标进度**: ~97%

本文档定义 SuperModel 具身智能系统所有模块的标准化接口契约，包括行为树、仿真环境、真实AGV接口、场景智能、任务规划、Pipeline等核心模块的方法签名、数据结构、事件机制和异常规范。

---

## 目录

1. [接口设计原则](#1-接口设计原则)
2. [核心数据结构](#2-核心数据结构)
3. [行为树模块 (behavior_tree)](#3-行为树模块-behavior_tree)
4. [具身Pipeline (embodied_pipeline)](#4-具身pipeline-embodied_pipeline)
5. [仿真环境 (simulation_enhancement)](#5-仿真环境-simulation_enhancement)
6. [真实AGV接口 (real_agv_interface)](#6-真实agv接口-real_agv_interface)
7. [场景智能 (scene_intelligence)](#7-场景智能-scene_intelligence)
8. [任务规划 (scene_task_planner)](#8-任务规划-scene_task_planner)
9. [部署管理 (deployment)](#9-部署管理-deployment)
10. [蜂群协调 (agv_swarm_coordinator)](#10-蜂群协调-agv_swarm_coordinator)
11. [联邦学习 (federated_learning)](#11-联邦学习-federated_learning)
12. [HIL硬件在环 (hil_testing)](#12-hil硬件在环-hil_testing)
13. [接口兼容性矩阵](#13-接口兼容性矩阵)
14. [错误码规范](#14-错误码规范)
15. [事件总线规范](#15-事件总线规范)

---

## 1. 接口设计原则

### 1.1 通用原则

- **类型安全**: 所有公开接口必须使用类型注解
- **向后兼容**: 新增接口不得破坏已有接口契约
- **可测试性**: 所有接口可通过mock实现隔离测试
- **线程安全**: 涉及并发的模块必须声明线程安全需求
- **错误处理**: 失败返回 `None` 或抛出特定异常，不返回非法状态

### 1.2 返回值规范

| 接口类型 | 成功 | 失败 | 说明 |
|---------|------|------|------|
| `get_*` 查询接口 | `T` 或数据 | `None` | 查询失败返回None |
| `check_*` 检测接口 | `bool` | `False` | 检测失败返回False |
| `execute_*` 执行接口 | `bool`/`Result` | `None`/`Result(success=False)` | 执行失败返回失败Result |
| `start_*` 启动接口 | `bool` | `False` | 启动失败返回False |
| `stop_*` 停止接口 | `bool` | `False` | 停止失败返回False |

### 1.3 AGV五级等级参数

所有接口涉及性能参数时，必须支持按AGV等级（S/M/L/XL/XXL）自动适配：

| 等级 | 负载(kg) | 最大速度(m/s) | 轮子半径(m) | 轮距(m) | 典型场景 |
|------|---------|-------------|------------|--------|---------|
| S | ≤5 | 1.0 | 0.05 | 0.30 | 餐厅/小型配送 |
| M | ≤20 | 1.5 | 0.07 | 0.45 | 医院/办公室 |
| L | ≤50 | 2.0 | 0.10 | 0.60 | 工厂/仓库 |
| XL | ≤200 | 2.5 | 0.15 | 0.80 | 重型工业 |
| XXL | >200 | 3.0 | 0.20 | 1.00 | 超重型物流 |

---

## 2. 核心数据结构

### 2.1 NodeStatus

```python
class NodeStatus(enum.Enum):
    """行为树节点状态"""
    SUCCESS = "success"      # 节点成功完成
    FAILURE = "failure"     # 节点执行失败
    RUNNING = "running"     # 节点正在执行
    SKIPPED = "skipped"      # 节点被跳过（如装饰器）或条件不满足
```

### 2.2 TaskStatus

```python
class TaskStatus(enum.Enum):
    """任务执行状态"""
    PENDING = "pending"      # 任务等待中
    RUNNING = "running"      # 任务执行中
    SUCCESS = "success"      # 任务成功完成
    FAILURE = "failure"      # 任务执行失败
    TIMEOUT = "timeout"      # 任务超时
    CANCELLED = "cancelled"  # 任务被取消
    PAUSED = "paused"        # 任务暂停
```

### 2.3 EmbodiedTask

```python
@dataclass
class EmbodiedTask:
    """具身任务定义"""
    task_id: str                          # 全局唯一任务ID
    task_type: str                        # 任务类型: transport/patrol/grasp/release/inspect/rescue/assemble
    description: str                      # 任务描述
    priority: int = 5                     # 优先级 1(high)-10(low)
    required_grade: str = "M"            # 最低AGV等级要求
    scene_type: SceneType = SceneType.WAREHOUSE  # 场景类型
    constraints: Dict[str, Any] = field(default_factory=dict)  # 任务约束
    subtasks: List['EmbodiedTask'] = field(default_factory=list)  # 子任务列表
    estimated_duration_s: float = 0.0     # 预估执行时间
    deadline: Optional[float] = None      # 截止时间戳
    parent_id: Optional[str] = None       # 父任务ID
```

### 2.4 Blackboard

```python
class Blackboard:
    """行为树共享黑板
    
    线程安全: 否（需要调用方保证线程安全）
    生命周期: 与行为树执行周期一致
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None): ...
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取黑板值，不存在返回default"""
    
    def set(self, key: str, value: Any) -> None:
        """设置黑板值"""
    
    def update(self, data: Dict[str, Any]) -> None:
        """批量更新黑板"""
    
    def delete(self, key: str) -> bool:
        """删除黑板键"""
    
    def has(self, key: str) -> bool:
        """检查键是否存在"""
    
    def to_dict(self) -> Dict[str, Any]:
        """导出黑板内容为字典"""
```

---

## 3. 行为树模块 (behavior_tree)

### 3.1 节点基类接口

```python
class BTNode(abc.ABC):
    """行为树节点基类
    
    所有节点必须实现:
    - name: str 节点名称
    - execute(blackboard) -> NodeStatus
    """
    
    name: str
    _status: Optional[NodeStatus] = None
    
    @abc.abstractmethod
    def execute(self, blackboard: Blackboard) -> NodeStatus: ...
    
    def tick(self, blackboard: Blackboard) -> NodeStatus:
        """执行一次tick，返回节点状态（默认单步执行）"""
        self._status = self.execute(blackboard)
        return self._status
    
    def reset(self) -> None:
        """重置节点状态"""
        self._status = None
    
    def get_status(self) -> Optional[NodeStatus]:
        """获取当前状态"""
        return self._status
```

### 3.2 组合节点接口

#### SequenceNode

```python
class SequenceNode(CompositeNode):
    """序列节点: 依次执行子节点，直到失败或全部成功
    
    成功条件: 所有子节点返回SUCCESS
    失败条件: 任一子节点返回FAILURE
    运行条件: 当前子节点返回RUNNING
    """
    
    def __init__(self, name: str, children: Optional[List[BTNode]] = None): ...
    
    def add_child(self, child: BTNode) -> None: ...
    
    def execute(self, blackboard: Blackboard) -> NodeStatus: ...
```

#### SelectorNode

```python
class SelectorNode(CompositeNode):
    """选择节点: 依次尝试子节点，直到成功或全部失败
    
    成功条件: 任一子节点返回SUCCESS
    失败条件: 所有子节点返回FAILURE
    运行条件: 当前子节点返回RUNNING
    """
```

#### ParallelNode

```python
class ParallelNode(CompositeNode):
    """并行节点: 同时执行所有子节点
    
    Policy.FAILURE_ON_ALL: 全部失败才失败（默认）
    Policy.SUCCESS_ON_ALL: 全部成功才成功
    Policy.FAILURE_ON_ONE: 任一失败即失败
    Policy.SUCCESS_ON_ONE: 任一成功即成功
    
    注意: RUNNING不计入成功/失败计数
    """
    
    class Policy(enum.Enum):
        FAILURE_ON_ALL = "failure_on_all"
        SUCCESS_ON_ALL = "success_on_all"
        FAILURE_ON_ONE = "failure_on_one"
        SUCCESS_ON_ONE = "success_on_one"
    
    def __init__(
        self,
        name: str,
        policy: 'ParallelNode.Policy' = Policy.FAILURE_ON_ALL,
        children: Optional[List[BTNode]] = None
    ): ...
```

### 3.3 装饰器节点接口

```python
class RepeaterNode(DecoratorNode):
    """重复执行子节点指定次数或直到失败"""
    def __init__(self, name: str, child: BTNode, limit: int = -1, reset_on_success: bool = True): ...

class UntilFailNode(DecoratorNode):
    """重复执行直到子节点失败"""
    def __init__(self, name: str, child: BTNode): ...

class UntilSuccessNode(DecoratorNode):
    """重复执行直到子节点成功"""
    def __init__(self, name: str, child: BTNode): ...

class InverterNode(DecoratorNode):
    """反转子节点结果: SUCCESS→FAILURE, FAILURE→SUCCESS, RUNNING→RUNNING"""
    def __init__(self, name: str, child: BTNode): ...
```

### 3.4 条件节点接口

```python
class ConditionNode(BTNode):
    """条件判断节点（叶节点）
    
    check(blackboard) -> bool 由子类实现
    True → SUCCESS, False → FAILURE
    """
    
    @abc.abstractmethod
    def check(self, blackboard: Blackboard) -> bool: ...
    
    def execute(self, blackboard: Blackboard) -> NodeStatus:
        return NodeStatus.SUCCESS if self.check(blackboard) else NodeStatus.FAILURE
```

### 3.5 动作节点接口

```python
class ActionNode(BTNode):
    """动作执行节点（叶节点）
    
    初始化参数在 __init__ 中，execute 实现具体逻辑
    """
    
    def __init__(self, name: str): ...
    
    @abc.abstractmethod
    def execute(self, blackboard: Blackboard) -> NodeStatus: ...
```

### 3.6 行为树接口

```python
class BehaviorTree:
    """完整行为树
    
    线程安全: 否
    """
    
    def __init__(self, root: BTNode, name: str = "BehaviorTree"): ...
    
    def tick(self, blackboard: Blackboard) -> NodeStatus:
        """执行一次tick"""
    
    def reset(self) -> None:
        """重置整棵树"""
    
    def to_dot(self) -> str:
        """导出为Graphviz DOT格式"""
    
    def get_node_count(self) -> Dict[str, int]:
        """获取节点统计: {'action': N, 'condition': N, 'composite': N, 'decorator': N}"""
```

### 3.7 AGV专用行为树节点

#### AGV移动动作

```python
class AGVMoveToAction(ActionNode):
    """AGV移动到目标位置
    
    参数:
        target: Tuple[float, float] 目标坐标 (x, y) 单位: m
        speed: float 移动速度 m/s（默认1.0）
        tolerance: float 到达容差 m（默认0.1）
    
    读取黑板:
        'current_x', 'current_y' 或 'robot_state.position'
        'desired_velocity', 'desired_omega'
    
    成功条件: 到达目标容差范围
    失败条件: 超时（60秒默认）
    """
    
    def __init__(
        self,
        target: Tuple[float, float],
        speed: float = 1.0,
        tolerance: float = 0.1
    ): ...
    
    def execute(self, blackboard: Blackboard) -> NodeStatus: ...
```

#### AGV抓取/释放动作

```python
class AGVGraspAction(ActionNode):
    """AGV抓取物体
    
    参数:
        object_position: Tuple[float, float] 物体位置
        grasp_width: float 抓取宽度（默认0.05m）
    
    写入黑板:
        'gripper_command': 'close'
        'robot_state.carrying_object': object_position
    """
    
    def __init__(self, object_position: Tuple[float, float], grasp_width: float = 0.05): ...

class AGVReleaseAction(ActionNode):
    """AGV释放物体"""
    def __init__(self, release_position: Optional[Tuple[float, float]] = None): ...
```

#### AGV蜂群协同节点

```python
class AGVNegotiateRoleAction(ActionNode):
    """AGV角色协商（领导者/跟随者）
    
    写入黑板:
        'assigned_role': 'leader' | 'follower'
        'formation_position': Tuple[float, float] 队形位置
    """
    ...

class AGVMoveToFormationAction(ActionNode):
    """移动到队形指定位置"""
    def __init__(self, formation_type: str, position_index: int): ...

class AGVCheckFormationReachedCondition(ConditionNode):
    """检查是否到达指定队形"""
    def __init__(self, formation_type: str): ...

class AGVParallelGraspAction(ParallelNode):
    """多AGV并行抓取"""
    def __init__(self, grasp_configs: List[Tuple[str, Tuple[float, float]]]): ...
```

### 3.8 具身任务规划器

```python
class EmbodiedTaskPlanner:
    """具身任务规划器基类
    
    接口规范:
    - create_task_tree(task: EmbodiedTask) -> BehaviorTree
    - estimate_duration(task: EmbodiedTask) -> float
    - validate_task(task: EmbodiedTask) -> Tuple[bool, Optional[str]]
    """
    
    def __init__(self, grade: str = "M"): ...
    
    @abc.abstractmethod
    def create_task_tree(self, task: EmbodiedTask) -> BehaviorTree: ...
    
    def estimate_duration(self, task: EmbodiedTask) -> float: ...
    
    def validate_task(self, task: EmbodiedTask) -> Tuple[bool, Optional[str]]: ...


class AGVTaskPlanner(EmbodiedTaskPlanner):
    """AGV专用任务规划器
    
    支持场景: WAREHOUSE / FACTORY / HOSPITAL / RESTAURANT / OUTDOOR
    
    额外方法:
        create_transport_tree(...) -> BehaviorTree
        create_patrol_tree(...) -> BehaviorTree
        create_rescue_tree(...) -> BehaviorTree
        create_assembly_tree(...) -> BehaviorTree
    """
    
    def __init__(
        self,
        grade: str = "M",
        scene_type: SceneType = SceneType.WAREHOUSE,
        max_speed: float = 1.5,
        safe_distance: float = 0.5
    ): ...
    
    def create_task_tree(self, task: EmbodiedTask) -> BehaviorTree: ...
    
    def create_transport_tree(
        self,
        pickup: Tuple[float, float],
        delivery: Tuple[float, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> BehaviorTree: ...
```

---

## 4. 具身Pipeline (embodied_pipeline)

### 4.1 Pipeline配置

```python
class PipelineMode(Enum):
    """Pipeline运行模式"""
    SIMULATION = "simulation"           # 纯仿真
    HARDWARE_IN_LOOP = "hil"           # 硬件在环
    FULL_PHYSICAL = "full_physical"    # 全实体


class PipelineState(Enum):
    """Pipeline状态机"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class PipelineConfig:
    """Pipeline全局配置"""
    
    # AGV等级
    grade: str = "M"
    
    # 运行模式
    mode: PipelineMode = PipelineMode.SIMULATION
    
    # 场景
    scene_type: str = "WAREHOUSE"
    
    # 模块开关
    enable_skill_registry: bool = True
    enable_memory: bool = True
    enable_scene_intelligence: bool = True
    enable_hil: bool = False
    enable_federated_learning: bool = False
    enable_swarm_coordination: bool = True
    
    # 传感器开关
    enable_vision: bool = True
    enable_audio: bool = False
    enable_tactile: bool = True
    enable_force: bool = True
    enable_imu: bool = True
    
    # 执行参数
    max_concurrent_tasks: int = 4
    task_timeout_s: float = 600.0
    health_check_interval_s: float = 5.0
    
    # 联邦学习参数
    enable_fl: bool = False
    fl_num_clients: int = 3
    fl_local_epochs: int = 5
    fl_rounds: int = 10
    fl_aggregation: str = "fedavg"
```

### 4.2 任务请求/结果

```python
@dataclass
class TaskRequest:
    """任务请求"""
    task_id: str
    task_type: str            # transport/patrol/grasp/release/inspect/rescue/assemble
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    grade: str = "M"
    timeout_s: float = 600.0
    
    def __post_init__(self):
        assert self.task_id, "task_id不能为空"
        assert self.task_type in VALID_TASK_TYPES, f"未知task_type: {self.task_type}"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    message: str = ""
    duration_s: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    retry_count: int = 0
```

### 4.3 Pipeline主接口

```python
class EmbodiedPipeline:
    """具身智能统一Pipeline
    
    线程安全: 部分方法线程安全（见各方法说明）
    单例模式: 否（使用工厂方法创建）
    """
    
    # ── 生命周期 ─────────────────────────────────────────────
    
    def __init__(
        self,
        grade: str = "M",
        scene_type: str = "WAREHOUSE",
        mode: PipelineMode = PipelineMode.SIMULATION,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """初始化Pipeline（不启动模块，按需懒加载）"""
    
    def state(self) -> PipelineState:
        """获取当前Pipeline状态（线程安全）"""
    
    def is_running(self) -> bool:
        """检查Pipeline是否在运行"""
    
    def uptime_s(self) -> float:
        """获取运行时长（秒）"""
    
    # ── 启动/停止 ─────────────────────────────────────────────
    
    def start(self) -> bool:
        """启动Pipeline，初始化所有模块
        
        返回: True=成功, False=失败
        线程安全: 否（应在单线程调用）
        """
    
    def pause(self) -> bool:
        """暂停Pipeline（暂停所有执行中的任务）"""
    
    def resume(self) -> bool:
        """恢复Pipeline"""
    
    def stop(self) -> None:
        """停止Pipeline，释放所有资源"""
    
    # ── 任务执行 ─────────────────────────────────────────────
    
    def execute_task(
        self,
        task_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
        priority: int = 5,
    ) -> TaskResult:
        """同步执行任务（阻塞直到完成或超时）
        
        参数:
            task_type: 任务类型
            parameters: 任务参数
            timeout_s: 超时秒数（默认使用config.task_timeout_s）
            priority: 优先级 1-10
        
        返回: TaskResult
        """
    
    def submit_task(self, request: TaskRequest) -> bool:
        """提交异步任务（非阻塞）
        
        返回: True=提交成功, False=提交失败
        """
    
    # ── 仿真 ─────────────────────────────────────────────
    
    def run_simulation_step(
        self,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """执行一个仿真步
        
        返回: {
            'timestamp': float,
            'scene_state': Dict,
            'sensor_data': Dict,
            'control_commands': Dict,
            'collision_detected': bool,
            'step_duration_ms': float,
        }
        """
    
    def get_scene_state(self) -> Dict[str, Any]:
        """获取当前仿真场景状态"""
    
    # ── 联邦学习 ─────────────────────────────────────────────
    
    def register_agv_to_fl(self, agv_id: str, agv_grade: str = "M") -> bool:
        """将AGV注册为联邦学习客户端"""
    
    def start_fl_round(self) -> Optional[Dict[str, Any]]:
        """启动一轮联邦学习训练
        
        返回: FLRoundResult 或 None（失败时）
        """
    
    def get_fl_status(self) -> Dict[str, Any]:
        """获取联邦学习状态"""
        # 返回: {enabled, round_count, registered_clients, last_result}
    
    # ── 蜂群协调 ─────────────────────────────────────────────
    
    def trigger_swarm_task(
        self,
        task_type: str,
        target_agvs: List[str],
        task_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """触发蜂群任务
        
        task_type: transport | patrol | inspection | assembly
        """
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """获取蜂群协调状态"""
        # 返回: {enabled, num_agvs, active_tasks}
    
    # ── 状态查询 ─────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """获取完整状态摘要
        
        返回: {
            'state': PipelineState,
            'uptime_s': float,
            'active_tasks': int,
            'memory_usage_mb': float,
            'scene_type': str,
            'mode': PipelineMode,
        }
        """
    
    def get_memory_summary(self) -> Dict[str, Any]: ...
    def get_skill_summary(self) -> Dict[str, Any]: ...
    
    # ── 事件订阅 ─────────────────────────────────────────────
    
    def subscribe(self, event: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """订阅Pipeline事件
        
        event: 'task_complete' | 'task_failed' | 'health_warning' | 'state_change' | 'swarm_update'
        """
    
    # ── 状态持久化 ─────────────────────────────────────────────
    
    def save_state(self) -> Dict[str, Any]:
        """保存Pipeline完整状态（用于热恢复）"""
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """从保存的状态恢复"""
    
    def export_checkpoint(self, path: str) -> bool:
        """导出检查点到文件"""
    
    @classmethod
    def import_checkpoint(cls, path: str, **kwargs) -> Optional['EmbodiedPipeline']:
        """从检查点文件导入"""
```

---

## 5. 仿真环境 (simulation_enhancement)

### 5.1 物理参数

```python
@dataclass
class PhysicsParameters:
    """AGV物理参数（支持五级等级自动适配）"""
    
    # 质量参数
    mass_empty: float = 35.0      # kg（M级空载）
    mass_load: float = 135.0       # kg（M级满载）
    wheel_radius: float = 0.07     # m（M级）
    wheel_base: float = 0.45       # m（M级）
    track_width: float = 0.35      # m
    
    # 摩擦力参数
    wheel_friction: float = 0.95
    ground_friction: float = 0.8
    rolling_resistance: float = 0.02
    
    # 惯性参数
    moment_of_inertia: float = 1.2   # kg·m²
    motor_inertia: float = 0.001    # kg·m²
    
    @classmethod
    def from_grade(cls, grade: str) -> 'PhysicsParameters': ...
```

### 5.2 传感器噪声模型

```python
class SensorNoiseModel:
    """传感器噪声模型
    
    使用方法:
        noise = model.add_noise(raw_signal, sensor_type, dt)
    """
    
    def __init__(
        self,
        lidar_range_noise: float = 0.03,
        lidar_angle_noise: float = 0.02,
        camera_noise_factor: float = 0.05,
        imu_gyro_noise: float = 0.01,
        imu_accel_noise: float = 0.05,
        tactile_noise: float = 0.02,
        force_noise: float = 0.1,
    ): ...
    
    def add_noise(
        self,
        signal: np.ndarray,
        sensor_type: str,
        dt: float = 0.01,
    ) -> np.ndarray:
        """添加传感器噪声"""
    
    def set_weather(self, weather: WeatherType) -> None:
        """设置天气条件影响"""
    
    def set_age(self, operating_hours: float) -> None:
        """设置设备老化影响（operating_hours: 小时）"""
```

### 5.3 仿真增强器

```python
class EmbodiedSimulationEnhancer:
    """具身仿真环境增强器
    
    主要功能:
    - 物理参数校准
    - 传感器噪声注入
    - 延迟仿真
    - 碰撞检测增强
    - 环境条件模拟
    """
    
    def __init__(
        self,
        physics: Optional[PhysicsParameters] = None,
        noise_model: Optional[SensorNoiseModel] = None,
        grade: str = "M",
    ): ...
    
    def step(
        self,
        state: Dict[str, Any],
        control: Dict[str, Any],
        dt: float,
    ) -> Dict[str, Any]:
        """执行一个仿真步
        
        参数:
            state: 当前AGV状态 {position, velocity, yaw, ...}
            control: 控制命令 {linear_velocity, angular_velocity, ...}
            dt: 时间步长（秒）
        
        返回: 更新后的状态（含噪声、延迟、碰撞）
        """
    
    def get_lidar_scan(
        self,
        state: Dict[str, Any],
        environment: 'EnvironmentGenerator',
    ) -> np.ndarray: ...
    
    def get_imu_data(
        self,
        state: Dict[str, Any],
        dt: float,
    ) -> Dict[str, np.ndarray]: ...
    
    def check_collision(
        self,
        state: Dict[str, Any],
        obstacles: List['Obstacle'],
    ) -> Tuple[bool, Optional[str]]: ...
```

### 5.4 环境生成器

```python
@dataclass
class Obstacle:
    """障碍物定义"""
    position: Tuple[float, float]  # m
    size: Tuple[float, float]       # (width, height) m
    type: str = "static"           # static/dynamic/human
    velocity: Optional[Tuple[float, float]] = None  # m/s


class EnvironmentGenerator:
    """环境场景生成器"""
    
    def __init__(self, scene_type: SceneType = SceneType.WAREHOUSE): ...
    
    def generate(
        self,
        num_obstacles: int = 10,
        floor_type: str = "concrete",
        num_agvs: int = 1,
    ) -> Dict[str, Any]:
        """生成随机环境
        
        返回: {
            'floor': {'type': str, 'friction': float},
            'obstacles': List[Obstacle],
            'charging_stations': List[Tuple[float, float]],
            'workstations': List[Tuple[float, float]],
            ' aisles': List[Dict],  # 通道定义
            'boundaries': Tuple[float, float, float, float],  # (xmin, ymin, xmax, ymax)
        }
        """
    
    def add_dynamic_obstacle(
        self,
        obstacle: Obstacle,
        trajectory: List[Tuple[float, float, float]],
    ) -> None:  # (x, y, time)
```

---

## 6. 真实AGV接口 (real_agv_interface)

### 6.1 硬件配置

```python
@dataclass
class AGVHardwareConfig:
    """真实AGV硬件配置"""
    
    grade: str = "M"
    can_interface: str = "can0"
    can_baudrate: int = 500000
    left_motor_id: int = 1
    right_motor_id: int = 2
    
    wheel_radius: float = 0.07    # m
    wheel_base: float = 0.45      # m
    max_speed: float = 1.5       # m/s
    
    lidar_port: str = "/dev/ttyUSB0"
    lidar_baudrate: int = 921600
    imu_port: str = "/dev/ttyUSB1"
    imu_baudrate: int = 115200
    
    control_frequency: float = 50.0   # Hz
    sensor_frequency: float = 100.0   # Hz
    
    @classmethod
    def from_grade(cls, grade: str) -> 'AGVHardwareConfig': ...
```

### 6.2 真实AGV控制器接口

```python
class RealAGVController:
    """真实AGV控制器
    
    线程安全: 是（内部使用锁保护共享状态）
    """
    
    def __init__(
        self,
        config: AGVHardwareConfig,
        use_simulation: bool = False,
    ): ...
    
    # ── 生命周期 ─────────────────────────────────────────────
    
    def connect(self) -> bool:
        """连接所有硬件设备
        
        依次连接: CAN Bus → 电机驱动器 → 激光雷达 → IMU
        
        返回: True=全部成功, False=任一失败
        """
    
    def disconnect(self) -> None:
        """断开所有硬件连接"""
    
    def start(self) -> bool:
        """启动传感器读取和控制循环"""
    
    def stop(self) -> None:
        """停止控制循环"""
    
    # ── 运动控制 ─────────────────────────────────────────────
    
    def set_velocity(
        self,
        linear: float,
        angular: float,
        timeout_ms: int = 100,
    ) -> bool:
        """设置线速度和角速度
        
        参数:
            linear: 线速度 m/s
            angular: 角速度 rad/s
            timeout_ms: 命令超时
        
        返回: True=成功, False=失败
        """
    
    def set_position(
        self,
        x: float,
        y: float,
        theta: float,
        timeout_ms: int = 500,
    ) -> bool:
        """设置目标位置（需要底层支持位置模式）"""
    
    def emergency_stop(self) -> None:
        """紧急停车（立即切断电机输出）
        
        注意: 这是最高优先级操作，不返回状态
        """
    
    def get_motor_status(self) -> Dict[str, Any]:
        """获取电机状态
        
        返回: {
            'left_rpm': float, 'right_rpm': float,
            'left_current': float, 'right_current': float,  # A
            'left_temperature': float, 'right_temperature': float,  # °C
            'fault_code': int,
        }
        """
    
    # ── 传感器读取 ─────────────────────────────────────────────
    
    def get_lidar_scan(self) -> Optional[np.ndarray]:
        """获取最新激光雷达扫描数据
        
        返回: 角度间隔1°的360维数组（距离，单位m），失败返回None
        """
    
    def get_imu_data(self) -> Optional[Dict[str, Any]]:
        """获取IMU数据
        
        返回: {
            'accel': np.ndarray (3,),  # m/s²
            'gyro': np.ndarray (3,),   # rad/s
            'mag': np.ndarray (3,),    # μT
            'timestamp': float,
        } 或 None
        """
    
    def get_battery_level(self) -> float:
        """获取电池电量（0.0-1.0）"""
    
    # ── 状态估计 ─────────────────────────────────────────────
    
    def get_odometry(self) -> Tuple[float, float, float]:
        """获取里程计估计 (x, y, theta)
        
        单位: m, m, rad
        """
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取估计线速度和角速度 (v, omega)"""
    
    # ── 诊断 ─────────────────────────────────────────────
    
    def run_self_test(self) -> Dict[str, bool]:
        """运行硬件自检
        
        返回: {
            'can_bus': bool,
            'left_motor': bool,
            'right_motor': bool,
            'lidar': bool,
            'imu': bool,
            'battery': bool,
        }
        """
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取整体健康状态
        
        返回: {
            'overall_health': float (0-1),
            'health_level': 'EXCELLENT'|'GOOD'|'WARNING'|'CRITICAL',
            'issues': List[str],
            'error_count': int,
        }
        """
```

---

## 7. 场景智能 (scene_intelligence)

### 7.1 场景类型与特征

```python
class SceneType(Enum):
    WAREHOUSE = "warehouse"
    FACTORY = "factory"
    HOSPITAL = "hospital"
    RESTAURANT = "restaurant"
    OFFICE = "office"
    OUTDOOR = "outdoor"
    LABORATORY = "laboratory"
    HOME = "home"
    UNKNOWN = "unknown"


@dataclass
class SceneFeatures:
    """场景特征"""
    scene_type: SceneType
    confidence: float
    features: Dict[str, float]
    
    # 环境
    floor_type: str = "concrete"
    floor_friction: float = 0.8
    aisle_width: float = 2.0       # m
    ceiling_height: float = 4.0     # m
    lighting_level: float = 1.0    # 0-1
    
    # 安全参数
    max_speed_safe: float = 1.5    # m/s
    safe_distance: float = 0.5      # m
    emergency_stop_dist: float = 0.2  # m
    
    def is_safe_for_high_speed(self) -> bool: ...


@dataclass
class SceneContext:
    """场景上下文"""
    scene_type: SceneType
    features: SceneFeatures
    timestamp: float
    active_rules: List['SceneRule']
    nearby_obstacles: List[Dict[str, Any]]
    nearby_humans: List[Dict[str, Any]]
    floor_load_capacity: float  # kg/m²
```

### 7.2 场景智能接口

```python
class SceneIntelligence:
    """场景智能管理器
    
    线程安全: 是（内部使用读写锁）
    """
    
    def __init__(
        self,
        scene_type: SceneType = SceneType.WAREHOUSE,
        config: Optional['SceneConfig'] = None,
    ): ...
    
    def detect_scene_type(
        self,
        sensor_data: Dict[str, Any],
    ) -> Tuple[SceneType, float]:
        """从传感器数据检测场景类型
        
        返回: (scene_type, confidence)
        """
    
    def get_scene_context(
        self,
        sensor_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> SceneContext:
        """获取当前场景上下文"""
    
    def evaluate_safety(
        self,
        action: Dict[str, Any],
        context: SceneContext,
    ) -> Tuple[bool, Optional[str]]:
        """评估动作安全性
        
        返回: (is_safe, reason_if_unsafe)
        """
    
    def get_navigation_params(
        self,
        context: SceneContext,
    ) -> Dict[str, Any]:
        """获取导航参数（速度限制、安全距离等）
        
        返回: {
            'max_speed': float,
            'safe_distance': float,
            'replan_threshold': float,
        }
        """
    
    def adapt_behavior(
        self,
        base_plan: BehaviorTree,
        context: SceneContext,
    ) -> BehaviorTree:
        """根据场景上下文适配行为树"""
```

---

## 8. 任务规划 (scene_task_planner)

### 8.1 任务模板

```python
@dataclass
class SceneTaskTemplate:
    """场景任务模板"""
    task_type: str
    scene_type: SceneType
    name: str
    description: str
    steps: List[Dict[str, Any]]     # 任务步骤定义
    required_skills: List[str]     # 所需技能列表
    estimated_duration_s: float
    required_grade: str = "M"
    success_criteria: Dict[str, Any]
    failure_conditions: List[str]


class SceneTaskLibrary:
    """场景任务库
    
    管理预定义任务模板，支持按场景类型检索
    """
    
    def __init__(self): ...
    
    def register_template(self, template: SceneTaskTemplate) -> None: ...
    
    def get_templates(
        self,
        scene_type: Optional[SceneType] = None,
        task_type: Optional[str] = None,
    ) -> List[SceneTaskTemplate]: ...
    
    def create_task(
        self,
        template_id: str,
        parameters: Dict[str, Any],
    ) -> EmbodiedTask: ...


class SceneTaskPlanner:
    """场景感知任务规划器
    
    核心职责:
    - 从任务库选择合适模板
    - 实例化为具体EmbodiedTask
    - 使用场景智能适配参数
    """
    
    def __init__(
        self,
        config: Optional[SceneTaskConfig] = None,
        scene_intelligence: Optional[SceneIntelligence] = None,
        memory: Optional['LongTermMemory'] = None,
    ): ...
    
    def plan_task(
        self,
        goal_description: str,
        scene_type: SceneType,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[EmbodiedTask]:
        """从目标描述规划任务
        
        使用场景记忆和历史经验优化任务参数
        """
    
    def create_task_tree(
        self,
        task: EmbodiedTask,
    ) -> BehaviorTree:
        """为任务创建行为树"""
    
    def record_outcome(
        self,
        scene_type: SceneType,
        task_type: str,
        success: bool,
        duration_s: float,
        parameters: Dict[str, Any],
    ) -> None:
        """记录任务执行结果（用于场景自适应学习）"""
    
    def get_adaptive_params(
        self,
        scene_type: SceneType,
        base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """获取场景自适应参数（成功率反馈调节）"""
```

---

## 9. 部署管理 (deployment)

### 9.1 部署状态

```python
class DeploymentState(Enum):
    UNINITIALIZED = "uninitialized"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    RUNNING = "running"
    PAUSED = "paused"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class HealthStatus(Enum):
    HEALTHY = "healthy"           # 所有检查通过
    DEGRADED = "degraded"          # 部分功能降级
    UNHEALTHY = "unhealthy"        # 需要人工干预
    CRITICAL = "critical"          # 需要立即停机


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    status: HealthStatus
    checks: Dict[str, bool]
    issues: List[str]
    timestamp: float
    recovery_actions: List[str] = field(default_factory=list)
```

### 9.2 部署管理器接口

```python
class DeploymentManager:
    """部署管理器
    
    负责:
    - 部署验证
    - 健康监控
    - 应急程序执行
    - 热恢复
    """
    
    def __init__(self, config: DeploymentConfig): ...
    
    def validate(self) -> Tuple[bool, List[str]]:
        """部署前验证（检查硬件连接、依赖等）
        
        返回: (is_valid, list_of_issues)
        """
    
    def deploy(self) -> bool:
        """执行部署"""
    
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def shutdown(self) -> None: ...
    
    def get_health(self) -> HealthCheckResult: ...
    
    def execute_emergency(
        self,
        procedure: EmergencyProcedure,
    ) -> Dict[str, Any]:
        """执行应急程序"""
```

---

## 10. 蜂群协调 (agv_swarm_coordinator)

### 10.1 蜂群任务

```python
class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class SwarmTask:
    """蜂群任务"""
    task_id: str
    task_type: str             # transport/patrol/inspection/assembly/rescue
    priority: TaskPriority
    target_positions: List[Tuple[float, float]]  # 目标位置列表
    required_agvs: int        # 所需AGV数量
    formation: str = "line"   # line/circle/v-shape/cluster
    deadline: Optional[float] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


class AGVSwarmCoordinator:
    """AGV蜂群协调器
    
    线程安全: 是
    """
    
    def __init__(self, scene: Any): ...
    
    def register_agv(
        self,
        agv_id: str,
        position: Tuple[float, float],
        grade: str = "M",
    ) -> bool: ...
    
    def assign_task(
        self,
        task: SwarmTask,
    ) -> Dict[str, Optional[str]]:
        """分配任务给AGV
        
        返回: {agv_id: assigned_task_id 或 None}
        """
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """获取蜂群整体状态"""
        # 返回: {
        #     'num_agvs': int,
        #     'active_tasks': int,
        #     'formation': str,
        #     'members': List[Dict],
        # }
        ```
    
    def rebalance(self) -> Dict[str, Any]:
        """重新平衡蜂群任务分配"""
```

---

## 11. 联邦学习 (federated_learning)

### 11.1 联邦学习接口

```python
class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: str, grade: str): ...
    
    def local_train(
        self,
        global_model_params: Dict[str, np.ndarray],
        epochs: int,
    ) -> LocalTrainingResult:
        """本地训练
        
        返回: {params, num_samples, loss}
        """
    
    def get_model_update(
        self,
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """获取模型更新（用于差分隐私）"""


class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(
        self,
        aggregation: str = "fedavg",  # fedavg/fedprox/scaffold
        privacy: Optional[DifferentialPrivacy] = None,
        byzantine_filter: Optional[ByzantineFilter] = None,
    ): ...
    
    def register_client(self, client: FederatedClient) -> bool: ...
    
    def run_round(
        self,
        selected_clients: List[FederatedClient],
        epochs: int,
    ) -> FLRoundResult: ...
```

---

## 12. HIL硬件在环 (hil_testing)

### 12.1 HIL测试接口

```python
class HILTestRunner:
    """HIL硬件在环测试运行器"""
    
    def __init__(
        self,
        real_hardware: RealAGVController,
        simulation: EmbodiedSimulationEnhancer,
    ): ...
    
    def run_test(
        self,
        test_case: HILTestCase,
    ) -> HILTestResult:
        """运行单个HIL测试"""
    
    def run_suite(
        self,
        test_suite: str = "full",
    ) -> HILTestReport:
        """运行完整测试套件
        
        test_suite: 'sensor' | 'control' | 'integration' | 'full'
        """


@dataclass
class HILTestCase:
    """HIL测试用例"""
    name: str
    stage: HILTestStage  # sensor/actuator/closed_loop
    description: str
    duration_s: float
    sensor_commands: List[Dict[str, Any]]  # 传感器注入数据
    expected_results: Dict[str, Any]
    tolerance: Dict[str, float]  # 容差


def run_hil_validation(
    real_hw: RealAGVController,
    sim: EmbodiedSimulationEnhancer,
    test_suite: str = "full",
) -> HILTestReport:
    """HIL验证快捷函数"""
```

---

## 13. 接口兼容性矩阵

| 模块 | Python版本 | 依赖模块 | 线程安全 | 序列化支持 |
|------|-----------|---------|---------|-----------|
| behavior_tree | ≥3.8 | - | 否 | to_dict/from_dict |
| embodied_pipeline | ≥3.8 | BT/SI/Memory/Swarm | 部分 | JSON |
| simulation_enhancement | ≥3.8 | numpy | 否 | to_dict |
| real_agv_interface | ≥3.8 | canlib（可选） | 是 | to_dict |
| scene_intelligence | ≥3.8 | - | 是 | to_dict |
| scene_task_planner | ≥3.8 | BT/SI/Memory | 否 | to_dict |
| deployment | ≥3.8 | Pipeline | 否 | JSON |
| agv_swarm_coordinator | ≥3.8 | Scene | 是 | to_dict |
| federated_learning | ≥3.8 | numpy | 是 | JSON |
| hil_testing | ≥3.8 | RealAGV/Sim | 否 | JSON |

---

## 14. 错误码规范

所有具身模块使用统一的错误码格式: `EMBODIED_<模块>_<编号>`

| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| EMBODIED_BT_001 | 行为树执行超时 | 重置树状态，检查黑板数据 |
| EMBODIED_BT_002 | 节点类型不匹配 | 检查树构造逻辑 |
| EMBODIED_PIPELINE_001 | Pipeline初始化失败 | 检查配置和依赖 |
| EMBODIED_PIPELINE_002 | 任务执行超时 | 增加超时或检查AGV状态 |
| EMBODIED_PIPELINE_003 | 状态机非法转换 | 检查并发调用 |
| EMBODIED_SIM_001 | 物理参数越界 | 检查AGV等级配置 |
| EMBODIED_SIM_002 | 碰撞检测失败 | 检查环境定义 |
| EMBODIED_HW_001 | CAN总线连接失败 | 检查CAN接口和波特率 |
| EMBODIED_HW_002 | 电机驱动器故障 | 检查驱动器状态寄存器 |
| EMBODIED_HW_003 | 传感器数据无效 | 检查传感器连接和标定 |
| EMBODIED_SCENE_001 | 场景类型检测失败 | 提供更多传感器数据 |
| EMBODIED_FL_001 | 联邦学习客户端离线 | 检查网络连接 |
| EMBODIED_FL_002 | 聚合失败 | 增加容错或重启服务器 |

---

## 15. 事件总线规范

Pipeline内部使用事件总线进行模块间通信：

```python
# 可订阅事件列表
PIPELINE_EVENTS = [
    'task_submitted',       # 新任务提交 (task_id, task_type)
    'task_started',         # 任务开始执行 (task_id)
    'task_complete',        # 任务成功完成 (task_id, duration_s, result)
    'task_failed',          # 任务执行失败 (task_id, error_code, message)
    'task_timeout',         # 任务超时 (task_id)
    'state_change',         # Pipeline状态变化 (old_state, new_state)
    'health_warning',        # 健康警告 (component, issue)
    'health_critical',      # 健康危急 (component, issue)
    'scene_change',         # 场景切换 (old_scene, new_scene)
    'fl_round_start',       # FL训练轮开始 (round)
    'fl_round_complete',    # FL训练轮完成 (round, result)
    'swarm_task_assigned',  # 蜂群任务分配 (task_id, agv_ids)
    'swarm_formation_done', # 队形到达 (agv_id, formation)
]
```

订阅示例:
```python
def on_task_complete(data):
    print(f"Task {data['task_id']} completed in {data['duration_s']:.2f}s")

pipeline.subscribe('task_complete', on_task_complete)
```

---

*本文档与 SuperModel PROGRESS.md 同步更新（v3.9.3, 2026-04-14）*
