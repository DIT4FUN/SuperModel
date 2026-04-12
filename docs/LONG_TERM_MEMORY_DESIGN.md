# SuperModel 长期记忆系统设计文档 (v2.87.0)

## 概述
长期记忆系统是 SuperModel 具身智能大脑的核心认知组件，模拟人类记忆的三层结构：工作记忆（短期）、情景记忆（经历）、语义记忆（知识）、程序记忆（技能），实现记忆的**存储、检索、巩固、遗忘**四大核心功能，支持机器人终身学习、经验复用、自主决策。

---

## 系统架构
```
┌───────────────────────────────────────────────────────────┐
│                     统一记忆接口 (LongTermMemory)          │
├───────────────┬───────────────┬───────────────┬────────────┤
│  情景记忆     │  语义记忆     │  程序记忆     │ 工作记忆   │
│ (Episodic)    │ (Semantic)    │ (Procedural)  │ (Working)  │
├───────────────┴───────────────┴───────────────┴────────────┤
│  记忆存储层 (MemoryStore)      │ 记忆检索引擎 (MemoryRetrieval) │
├───────────────────────────────┼───────────────────────────────┤
│  向量数据库 (ChromaDB)        │  混合检索策略 (向量+关键词+语义) │
│  结构化存储 (SQLite)          │  召回排序算法 (BM25 + 余弦相似度) │
├───────────────────────────────┴───────────────────────────────┤
│                     记忆巩固引擎 (MemoryConsolidation)         │
└───────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. 情景记忆 (Episodic Memory)
**功能**: 存储机器人的经历、交互事件、任务执行过程
**数据结构**:
```python
@dataclass
class Episode:
    episode_id: str
    timestamp: float
    summary: str                  # 事件摘要
    context: Dict[str, Any]       # 上下文环境数据
    entities: List[str]           # 涉及的实体/物体
    emotional_tag: EmotionalTag   # 情感标签 (POSITIVE/NEGATIVE/NEUTRAL)
    importance: ImportanceLevel   # 重要性等级 (LOW/MEDIUM/HIGH/CRITICAL)
    accessibility: float          # 可访问性分数 (0-1，用于遗忘)
    retrieval_count: int          # 被检索次数
```
**核心功能**:
- 按时间/实体/重要性检索经历
- 自动计算记忆可访问性（随时间衰减，被检索则提升）
- 支持基于 Ebbinghaus 遗忘曲线的自动遗忘

### 2. 语义记忆 (Semantic Memory)
**功能**: 存储客观知识、概念、事实、规则、环境信息
**数据结构**:
```python
@dataclass
class Concept:
    concept_id: str
    name: str
    category: str
    attributes: Dict[str, Any]    # 属性字典
    relations: List[str]          # 与其他概念的关系
    confidence: float             # 置信度 (0-1)
    source: KnowledgeSource       # 知识来源 (OBSERVATION/TAUGHT/INFERRED)
    last_accessed: float
```
**核心功能**:
- 概念层次化组织
- 关系推理与知识图谱构建
- 置信度动态更新（验证正确则提升，错误则降低）

### 3. 程序记忆 (Procedural Memory)
**功能**: 存储技能、动作序列、策略、操作流程
**数据结构**:
```python
@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    preconditions: List[str]      # 执行前置条件
    steps: List[Dict[str, Any]]   # 动作步骤序列
    success_count: int
    total_attempts: int
    success_rate: float           # 成功率
    last_used: float
    skill_level: SkillLevel       # 技能等级 (NOVICE/INTERMEDIATE/EXPERT/MASTER)
```
**核心功能**:
- 技能熟练度自动升级（成功次数越多等级越高）
- 前置条件检查与技能推荐
- 技能参数自适应优化

### 4. 工作记忆 (Working Memory)
**功能**: 短期存储当前任务相关的焦点信息，模拟人脑工作记忆容量限制
**核心特性**:
- 容量限制：默认最多同时存储7±2个激活项
- 激活衰减：未被访问的项随时间自动失活
- 焦点切换：支持主动设置/移除当前焦点信息
- 绑定机制：支持跨模态信息临时绑定

---

## 核心机制实现

### 1. 记忆存储 (MemoryStore)
**存储架构**: 混合存储方案
| 存储类型 | 存储介质 | 存储内容 |
|---------|---------|---------|
| 向量存储 | ChromaDB | 记忆文本 embedding、检索索引 |
| 结构化存储 | SQLite | 记忆元数据、属性、关系、统计信息 |
| 二进制存储 | 文件系统 | 大容量传感数据、图像/点云等非结构化数据 |

**持久化策略**:
- 新记忆立即写入事务日志
- 批量异步刷入持久化存储
- 定期备份与压缩

### 2. 记忆检索 (MemoryRetrieval)
**混合检索策略**:
1. **语义检索**: 基于向量相似度匹配相关记忆 (余弦相似度)
2. **关键词检索**: 基于 BM25 算法匹配关键词
3. **结构化检索**: 基于属性过滤（时间范围、重要性、实体等）

**召回排序**:
> 最终得分 = 0.6*向量相似度 + 0.3*BM25得分 + 0.1*记忆重要性 + 0.05*检索次数

**检索流程**:
```
用户查询 → 查询嵌入生成 → 多路召回（向量+关键词+结构化）→ 结果去重 → 排序 → 返回Top-K结果
```

### 3. 记忆巩固 (MemoryConsolidation)
**巩固触发条件**:
- 定期自动巩固（默认每小时一次）
- 任务结束后触发巩固
- 记忆容量达到阈值时触发

**巩固流程**:
1. **知识提取**: 从情景记忆中提取通用知识、规则、技能
2. **冲突检测**: 检查新知识与现有语义记忆是否冲突
3. **记忆更新**: 合并重复记忆，更新置信度与熟练度
4. **遗忘清理**: 删除低价值、低访问率的记忆

### 4. 记忆遗忘 (Forgetting Mechanism)
**遗忘策略**: 基于价值的选择性遗忘，而非随机删除
- **保留高价值记忆**: CRITICAL 重要性的记忆永久保留
- **衰减低价值记忆**: 基于时间衰减、访问频率、重要性计算保留分数
- **定期清理**: 每次巩固时自动删除保留分数低于阈值的记忆，释放存储空间

**遗忘公式**:
> 保留分数 = 重要性权重 * (0.8^（天数/30）) * min(1.0, log2(检索次数 + 1))
> 保留分数 < 0.2 → 遗忘删除

---

## 接口规范

### 统一接口使用示例
```python
from src.memory.long_term_memory import LongTermMemory, MemoryConfig

# 初始化记忆系统
config = MemoryConfig(
    store_path="./memory_data",
    max_episodes=10000,
    max_concepts=5000,
    max_skills=1000
)
ltm = LongTermMemory(config)

# 存储记忆
ltm.store_episode(
    summary="成功抓取红色盒子放置到货架",
    context={"object": "red_box", "location": "shelf_1"},
    importance="HIGH"
)

ltm.store_knowledge(
    name="红色盒子",
    category="物体",
    attributes={"weight": "500g", "color": "red"}
)

ltm.store_skill(
    name="抓取盒子",
    steps=[...]
)

# 检索记忆
results = ltm.retrieve(query="抓取红色盒子的经验", top_k=5)

# 触发记忆巩固
consolidation_result = ltm.consolidate()

# 获取记忆系统状态
status = ltm.get_status()
```

---

## 性能指标
| 指标 | 规格 |
|------|------|
| 存储容量 | 支持最多 10 万条记忆条目 |
| 检索延迟 | < 100ms / 次查询 |
| 巩固速度 | 处理 1000 条记忆 < 10s |
| 内存占用 | 运行时内存 < 512MB |
| 检索准确率 | > 92% （Top-5 召回率） |

---

## 测试覆盖
所有记忆模块已通过 32 项单元测试，覆盖：
- ✅ 记忆增删改查基本操作
- ✅ 检索准确性与排序逻辑
- ✅ 巩固与知识提取功能
- ✅ 遗忘机制正确性
- ✅ 边界条件与异常处理
- ✅ 性能与稳定性测试

---

## 适配AGV五级规格
| 等级 | 记忆容量 | 检索速度 | 巩固频率 | 遗忘开关 |
|------|---------|---------|---------|---------|
| S | 1000条 | <500ms | 每天一次 | 开启 |
| M | 5000条 | <200ms | 每4小时一次 | 开启 |
| L | 20000条 | <100ms | 每小时一次 | 开启 |
| XL | 100000条 | <50ms | 每30分钟一次 | 可配置 |
| XXL | 无限制 | <20ms | 每10分钟一次 | 可配置 |

---

*文档版本: v2.87.0 | 最后更新: 2026-04-12*
