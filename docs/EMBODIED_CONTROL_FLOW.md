# SuperModel 具身智能控制流程文档

> **文档版本**: v1.0.0  
> **更新**: 2026-04-13  
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档详细描述 SuperModel 具身智能系统的完整控制流程，从任务输入到执行输出的端到端链路。

---

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [Pipeline 启动流程](#2-pipeline-启动流程)
3. [任务执行主流程](#3-任务执行主流程)
4. [行为树规划流程](#4-行为树规划流程)
5. [记忆系统交互流程](#5-记忆系统交互流程)
6. [场景自适应流程](#6-场景自适应流程)
7. [硬件在环流程](#7-硬件在环hil流程)
8. [联邦学习协同流程](#8-联邦学习协同流程)
9. [状态机与错误处理](#9-状态机与错误处理)
10. [五级AGV规格适配对照](#10-五级agv规格适配对照)

---

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EmbodiedPipeline                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  任务输入  │→│ 技能注册表  │→│ 行为树规划 │→│  任务执行器        │ │
│  │ TaskInput │  │ SkillReg.  │  │  BT Plan │  │  Executor(HIL/Sim)│ │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────────┘ │
│       │               │            │                │              │
│       └───────────────┴────────────┴────────────────┘              │
│                              ↓                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    记忆系统 MemoryManager                     │   │
│  │   情景记忆  │  语义记忆  │  程序记忆  │  工作记忆            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              场景智能 SceneIntelligence                       │   │
│  │   场景上下文  │  安全规则  │  导航规则  │  交互规则           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   硬件抽象层 Hardware                          │   │
│  │  仿真器  │  CANBus  │  传感器桥接  │  RK3588 NPU             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline 启动流程

### 2.1 初始化序列

```
用户/API调用
    ↓
EmbodiedPipeline.__init__(grade, scene_type, mode)
    ↓
┌─────────────────────────────────────────┐
│ 1. 创建 PipelineConfig                  │
│    grade: AGV等级 (S/M/L/XL/XXL)        │
│    scene_type: 场景类型                 │
│    mode: SIM/HIL/FULL_PHYSICAL         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. 状态 → INITIALIZING                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 模块懒加载 (按配置开关):               │
│    _init_behavior_tree()  [总是]       │
│    _init_scene_intelligence()           │
│    _init_skill_registry()               │
│    _init_memory()                       │
│    _init_task_executor()                │
│    _init_hil() [mode==HIL时]            │
│    _init_simulation() [mode==SIM时]     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. 发布 state_changed(INITIALIZING)     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. 状态 → READY                         │
│    发布 state_changed(READY)            │
└─────────────────────────────────────────┘
```

### 2.2 启动代码示例

```python
from src.embodied import EmbodiedPipeline, PipelineMode, SceneType

# 创建 Pipeline (不传 grade 默认为 "M")
pipeline = EmbodiedPipeline(
    grade="M",
    scene=SceneType.WAREHOUSE,
    mode=PipelineMode.SIMULATION
)

# 启动 Pipeline
pipeline.start()  # 状态: READY

# 执行任务
result = pipeline.execute_task("transport", target="station_A")

# 停止
pipeline.stop()
```

---

## 3. 任务执行主流程

### 3.1 execute_task 完整流程

```
execute_task(task_type, params, options)
    │
    ├─[1] 输入验证
    │   ├─ 检查 PipelineState == READY
    │   ├─ 检查任务类型在 SKILL_CATEGORIES 中
    │   └─ 生成 task_id (UUID)
    │
    ├─[2] 状态转换: READY → RUNNING
    │       发布 state_changed(RUNNING)
    │
    ├─[3] 技能匹配 (SkillRegistry)
    │   ├─ get_best_skill_for_task(task_type)
    │   ├─ 按 reliability_score 排序
    │   ├─ 返回 EmbodiedSkill 或 None
    │   └─ 如无合适技能 → 降级到默认技能
    │
    ├─[4] 行为树规划 (EmbodiedTaskPlanner)
    │   ├─ scene_intelligence.get_adaptive_bt(task_type)
    │   ├─ 注入场景上下文 (scene_context)
    │   ├─ 注入 AGV 规格参数 (grade)
    │   └─ 返回 BehaviorTree 对象
    │
    ├─[5] 任务执行 (MemoryEnhancedExecutor)
    │   ├─ execute(task, bt, skill, timeout)
    │   │
    │   ├─[5.1] 规划阶段 (PLANNING)
    │   │   ├─ memory.retrieve(task_description)
    │   │   └─ blackboard初始化 (goal/world/robot_state)
    │   │
    │   ├─[5.2] 执行阶段 (EXECUTING)
    │   │   ├─ bt.tick() 循环
    │   │   ├─ 每个 tick: 执行器与仿真/硬件交互
    │   │   ├─ 发布 phase_change(EXECUTING)
    │   │   └─ 监控 RUNNING → SUCCESS/FAILURE
    │   │
    │   └─[5.3] 完成阶段 (COMPLETED/FAILED)
    │       ├─ memory.store(execution_record)
    │       ├─ skill.update_success(result)
    │       └─ 发布 phase_change(FINAL)
    │
    ├─[6] 结果封装
    │   ├─ TaskResult(task_id, status, duration, ...)
    │   └─ 发布 state_changed(READY)
    │
    └─[7] 返回 TaskResult
```

### 3.2 时序图 (ASCII)

```
用户          Pipeline      SkillReg      BTPlanner     Executor      Memory
 │               │              │              │             │            │
 │─execute_task─►│              │              │             │            │
 │               │─match_skill─►│              │             │            │
 │               │◄──skill──────│              │             │            │
 │               │─plan_task────│─────────────►│             │            │
 │               │◄─────────────BT─────────────│             │            │
 │               │─execute──────│──────────────│───────────►│            │
 │               │              │              │             │─retrieve──►│
 │               │              │              │             │◄─memory───�│
 │               │              │              │◄────tick────│            │
 │               │              │              │────tick───►│            │
 │               │              │              │             │─store────►│
 │               │◄────────result─────────────│─────────────│            │
 │◄──result──────│              │              │             │            │
```

---

## 4. 行为树规划流程

### 4.1 行为树节点类型

```
                    ┌─────────────┐
                    │   Selector  │ (选择: 第一个成功即停)
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
    ┌──────────┐   ┌──────────┐    ┌──────────────┐
    │Condition │   │Condition │    │  Condition   │
    │ (电池低?) │   │(障碍物?) │    │  (任务完成?)  │
    └────┬─────┘   └────┬─────┘    └──────┬───────┘
         ↓               ↓                  ↓
    ┌──────────┐   ┌──────────┐    ┌──────────────┐
    │ Sequence │   │  Parallel │    │   Action     │
    │导航+移动 │   │感知+避障  │    │  等待/重试   │
    └──────────┘   └──────────┘    └──────────────┘
```

### 4.2 行为树规划接口

```python
from src.embodied import EmbodiedTaskPlanner, SceneType

planner = EmbodiedTaskPlanner(
    scene_type=SceneType.WAREHOUSE,
    grade="M"
)

# 生成任务行为树
bt = planner.plan_task(
    task_type="transport",
    target="station_A",
    constraints={"max_time": 300, "priority": 2}
)

# 执行
bt.tick(blackboard)
```

### 4.3 场景自适应

```
场景类型          生成的行为树特点
──────────────────────────────────────────────
WAREHOUSE    - 序列节点: 定位→取货→导航→放货
             - 条件: 低电量返回 + 障碍绕行
HOSPITAL    - 安全优先: 消毒检查→无菌导航→精准投递
             - 条件: 紧急车辆让行 + 患者避让
FACTORY     - 节拍同步: 上料→传输→下料→质量检测
             - 条件: 设备就绪 + 工序完成信号
RESTAURANT  - 时效优先: 接单→备餐→配送→通知
             - 条件: 障碍检测 + 餐桌识别
OUTDOOR     - GPS+SLAM融合: 路径规划→地形适应
             - 条件: 天气预警 + 地形可行性
```

---

## 5. 记忆系统交互流程

### 5.1 记忆模块职责

| 记忆类型 | 职责 | 容量 | 衰减 |
|---------|------|------|------|
| 工作记忆 | 当前任务状态/注意力焦点 | 单任务 | 任务结束清除 |
| 情景记忆 | 任务执行经验/成功失败模式 | 1000条 | Ebbinghaus遗忘曲线 |
| 程序记忆 | 技能注册/场景专长/学习状态 | 100条 | 低衰减 |
| 语义记忆 | 概念/场景规则/安全协议 | 500条 | 几乎不衰减 |

### 5.2 记忆交互时序

```
任务规划阶段:
  Executor ──retrieve(query)──► EmbodiedMemoryManager
                               │
                               ├─情景记忆: 按任务相似度检索
                               │            cosine_similarity(recent_tasks, current)
                               │
                               ├─程序记忆: 按场景+技能检索
                               │            scene==current_scene AND skill==task_type
                               │
                               └─返回: List[EmbodiedMemoryEntry]
                                          (context_hints + learned_parameters)

任务执行阶段:
  Executor ──update_working_memory(state)──► WorkingMemory
                                            focus: {task, position, battery, safety}

任务完成阶段:
  Executor ──store(record)──► EmbodiedMemoryManager
                             │
                             ├─存入情景记忆 (Episodes)
                             ├─更新程序记忆 (Skills)
                             └─更新语义记忆 (Concepts)
```

---

## 6. 场景自适应流程

### 6.1 SceneIntelligence 工作机制

```python
from src.embodied import SceneIntelligence, SceneType, SceneContext

si = SceneIntelligence(
    scene_type=SceneType.FACTORY,
    grade="M"
)

# 场景上下文更新
context = si.update_context(
    position=(x, y),
    nearby_objects=[obj1, obj2],
    time_of_day="morning",
    battery_level=0.85
)

# 获取场景特定行为
safety_rules = si.get_active_safety_rules()
# → [SafetyRule(name="紧急停止", priority=0, ...),
#    SafetyRule(name="区域准入", priority=1, ...)]

navigation_params = si.get_navigation_params()
# → {max_speed: 0.8, safe_distance: 0.5, ...}
```

### 6.2 场景切换流程

```
场景类型变化检测 (位置区域 + 任务类型)
    ↓
SceneIntelligence.on_scene_transition(new_scene)
    ↓
┌─────────────────────────────────────────┐
│ 1. 更新 scene_context                    │
│ 2. 重新加载场景安全规则                   │
│ 3. 通知行为树重规划                       │
│ 4. 更新传感器融合权重                     │
│ 5. 发送 scene_changed 事件               │
└─────────────────────────────────────────┘
    ↓
Pipeline 重新匹配技能 + 生成新行为树
```

---

## 7. 硬件在环(HIL)流程

### 7.1 HIL 架构

```
┌─────────────────────────────────────────────────┐
│                  RK3588 NPU                      │
│  ┌───────────────────────────────────────────┐  │
│  │           SuperModel Brain                 │  │
│  │   感知融合 → 决策 → 控制指令               │  │
│  └───────────────────────────────────────────┘  │
│              ↓ 控制指令 (CAN)                     │
│  ┌───────────────────────────────────────────┐  │
│  │         真实AGV执行器                       │  │
│  │   ZLAC8015D驱动器 → 电机 → 运动            │  │
│  └───────────────────────────────────────────┘  │
│              ↓ 传感器反馈 (CAN/USB)               │
│  ┌───────────────────────────────────────────┐  │
│  │         真实传感器                          │  │
│  │   激光雷达 + IMU + 力触觉 + 视觉           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 7.2 HIL 执行模式

```python
# HIL 模式: 仿真环境 + 真实传感器
pipeline = EmbodiedPipeline(
    grade="M",
    scene=SceneType.WAREHOUSE,
    mode=PipelineMode.HARDWARE_IN_LOOP
)
pipeline.start()

# 传感器数据来自真实硬件
# 控制指令发送到真实AGV
# 物理引擎计算AGV运动 (不开环控制)
result = pipeline.execute_task("transport", target="A")
```

---

## 8. 联邦学习协同流程

### 8.1 多AGV联邦学习架构

```
        ┌─────────────────────────────────────────┐
        │          FederatedServer (边缘节点)       │
        │   • 全局模型聚合                          │
        │   • Byzantine 过滤                       │
        │   • 差分隐私 (DP-Adam)                   │
        └────────────────┬────────────────────────┘
                         │ 聚合后的全局模型
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  AGV-1  │    │  AGV-2  │    │  AGV-3  │
    │ (Client)│    │ (Client)│    │ (Client)│
    └─────────┘    └─────────┘    └─────────┘
```

### 8.2 联邦学习轮次流程

```
1. Server: 初始化全局模型 (init_global_model)
2. Server: 分发全局模型到所有 Client (broadcast_global_model)
3. Client-AGV:
   3a. 本地训练 (local_training)
       - 收集当前任务经验
       - 计算梯度 (loss on local data)
       - 应用差分隐私 (Gaussian noise)
   3b. 上传梯度 (upload_gradients)
       - 哈希验证完整性
       - 压缩编码
4. Server:
   4a. Byzantine 过滤 (过滤恶意/异常梯度)
   4b. 自适应聚合 (FedAvg with contribution weights)
   4c. 更新全局模型 (apply_aggregated_model)
5. 重复 2-4 直到收敛
```

### 8.3 多AGV蜂群协同

```python
from src.embodied import AGVSwarmCoordinator

coordinator = AGVSwarmCoordinator(
    num_agvs=4,
    scene_type=SceneType.FACTORY,
    topology="mesh"  # mesh / star / chain
)

# 蜂群任务分配
assignments = coordinator.assign_tasks([
    TaskRequest(type="transport", target="A"),
    TaskRequest(type="transport", target="B"),
    TaskRequest(type="patrol", zone="zone_1"),
])

# 碰撞避免
safe_velocity = coordinator.compute_collision_free_velocity(
    agv_id="AGV-1",
    current_vel=(0.5, 0.3),
    obstacles=[other_agv_positions]
)
```

---

## 9. 状态机与错误处理

### 9.1 Pipeline 状态机

```
                    ┌─────────────┐
        start() ───►│  IDLE       │
                    └──────┬──────┘
                           │ initialize()
                    ┌──────▼──────┐
          ┌────────│ INITIALIZING │
          │        └──────┬──────┘
          │               │ 初始化成功
          │        ┌──────▼──────┐
          │        │    READY    │◄─────────────────┐
          │        └──────┬──────┘                  │
          │               │ execute_task()          │
          │        ┌──────▼──────┐                  │
          │        │   RUNNING    │                  │
          │        └──────┬──────┘                  │
          │               │ 任务完成                  │
          │        ┌──────▼──────┐                  │
          │        │  COMPLETED   │──────────────────┘
          │        └─────────────┘  reset()
          │
          │  错误发生
          │  ┌──────▼──────┐
          └─►│    ERROR    │
               └──────┬──────┘
                      │ reset() / fix()
               ┌──────▼──────┐
               │  EMERGENCY  │──stop()
               │    STOP     │
               └─────────────┘
```

### 9.2 错误处理策略

| 错误类型 | 处理策略 | 降级行为 |
|---------|---------|---------|
| 传感器超时 | 切换到估计值 | IMU用积分估计，视觉用历史帧 |
| 行为树失败 | 重试N次 | 降级到简单行为树 |
| 电池低 | 中断任务返回充电 | 暂停非关键任务 |
| 通信中断 | 本地缓存决策 | 切换到离线模式 |
| 碰撞检测 | 立即停止 | 进入安全避障行为树 |
| 模型推理超时 | 使用规则备选 | 降级到PID控制 |

---

## 10. 五级AGV规格适配对照

### 10.1 各等级 Pipeline 配置

| 规格 | grade | 场景智能 | 技能数量 | 记忆容量 | 联邦学习 |
|-----|-------|---------|---------|---------|---------|
| 超标S | S | 完全 | 20+ | 5000条 | 支持 |
| 高级M | M | 完全 | 14+ | 2000条 | 支持 |
| 标准L | L | 简化 | 8+ | 1000条 | 可选 |
| 基础XL | XL | 基础 | 4+ | 500条 | 不可用 |
| 入门XXL | XXL | 极简 | 2+ | 200条 | 不可用 |

### 10.2 各等级传感器配置

| 规格 | 视觉 | 听觉 | 触觉 | 力觉 | IMU | SLAM |
|-----|------|------|------|------|-----|------|
| S | 双目深度 | 4-mic | 全身电子皮肤 | 六维力矩 | 高精度 | 激光+视觉 |
| M | 单目+深度 | 4-mic | 手腕+足底 | 六维力矩 | 标准 | 激光 |
| L | 单目 | 双mic | 手腕 | 一维力 | 标准 | 激光 |
| XL | 广角单目 | 单mic | 无 | 一维力 | 基础 | 激光 |
| XXL | 广角单目 | 无 | 无 | 无 | 基础 | 里程计 |

---

## 附录: 快速参考

### Pipeline API 速查

```python
# 创建
pipeline = EmbodiedPipeline(grade="M", scene=SceneType.WAREHOUSE)

# 启动/停止
pipeline.start()
pipeline.stop()

# 执行任务
result = pipeline.execute_task("transport", target="station_A")

# 查询
status = pipeline.get_status()
scene_state = pipeline.get_scene_state()
memory_summary = pipeline.get_memory_summary()

# 仿真步骤 (SIMULATION 模式)
pipeline.run_simulation_step(dt=0.01)
```

### 关键配置参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `grade` | "M" | AGV规格等级 |
| `mode` | SIMULATION | 运行模式 |
| `max_concurrent_tasks` | 4 | 最大并发任务数 |
| `task_timeout_s` | 600 | 任务超时(秒) |
| `health_check_interval_s` | 5 | 健康检查间隔 |
| `simulation_timestep` | 0.01 | 仿真时间步长(秒) |
