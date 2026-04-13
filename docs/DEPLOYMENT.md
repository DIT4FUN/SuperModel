# SuperModel 具身智能部署文档

> 真实AGV机器人部署指南 | 硬件接口 | 环境配置 | 调试 | 运行监控

## 目录

- [概述](#概述)
- [硬件要求](#硬件要求)
- [AGV五级规格硬件对照](#agv五级规格硬件对照)
- [软件环境配置](#软件环境配置)
- [传感器接线](#传感器接线)
- [CAN总线配置](#can总线配置)
- [部署步骤](#部署步骤)
- [启动与验证](#启动与验证)
- [健康监控](#健康监控)
- [应急处理](#应急处理)
- [故障排查](#故障排查)

---

## 概述

本文档提供 SuperModel 超模态大模型具身智能大脑在真实 AGV 机器人上的部署指南。涵盖硬件选型、接线、软件配置、启动验证、运行监控和应急处理全流程。

部署架构:

```
传感器 → CAN Bus/USB → RK3588 NPU → SuperModel → CAN Bus → 电机驱动器 → AGV运动
                         ↓
                    健康监控 → 日志/报警
```

---

## 硬件要求

### 核心计算单元

推荐: **Rockchip RK3588** (8核 Cortex-A76/A55, 8MB L3, 6TOPS NPU)

- NPU 用于跨模态融合网络推理
- 支持 INT8 量化模型加速
- 功耗低 (10W-15W) 适合车载
- 提供 PCIe/MIPI/USB3.0 接口

备选:

- Jetson Orin NX 8GB (更高性能, 更高功耗)
- Raspberry Pi 5 (S级小负载AGV开发测试)

### 电源要求

| AGV等级 | 推荐电源 | 电压 | 容量 |
|---------|----------|------|------|
| S | 12V 10Ah | 12V | 120Wh |
| M | 24V 20Ah | 24V | 480Wh |
| L | 24V 35Ah | 24V | 840Wh |
| XL | 48V 35Ah | 48V | 1680Wh |
| XXL | 48V 50Ah | 48V | 2400Wh |

要求:
- 带过流保护
- 带欠压保护
- 支持PD快充充电

---

## AGV五级规格硬件对照

完整规格见 [AGV_FIVE_LEVEL_CONSOLIDATED_SPEC.md](./design/AGV_FIVE_LEVEL_CONSOLIDATED_SPEC.md)

### 传感器配置

| 组件 | S | M | L | XL | XXL |
|------|---|---|---|----|-----|
| 激光雷达 | 10m 360° | 20m 360° | 25m 360° | 25m 360° | 40m 360° |
| IMU | ✅ 6轴 | ✅ 6轴 | ✅ 6轴 | ✅ 6轴 | ✅ 6轴 |
| RGB相机 | - | ✅ | ✅ | ✅ | ✅ |
| 深度相机 | - | - | ✅ | ✅ | ✅ |
| 触觉阵列 | 4×4 | 8×8 | 16×8 | 16×16 | 32×16 |
| 六维力传感器 | - | ✅ | ✅ | ✅ | ✅ |
| 编码器 | 轮子 | 轮子 | 轮子 | 轮子 | 轮子 |

### 驱动配置

| AGV等级 | 驱动器 | 电机 | 控制方式 |
|---------|--------|------|----------|
| S | TB6612 | 57步进 | PWM |
| M | 中菱 ZLAC8015D ×1 | 150W ×2 | CANopen |
| L | 中菱 ZLAC8015D ×1 | 150W ×4 | CANopen |
| XL | 中菱 ZLAC8015D ×2 | 300W ×4 | CANopen |
| XXL | 中菱 ZLAC8015D ×2 | 500W ×4 | CANopen |

**ZLAC8015D 支持:**
- 一拖二轮毂伺服驱动
- 位置/速度/力矩控制模式
- CANopen / RS485 通信
- 24V-48V 宽电压输入
- 15A/30A 峰值输出

---

## 软件环境配置

### 基础系统 (RK3588)

推荐镜像: `Ubuntu 22.04 Server for RK3588`

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3-pip python3-numpy python3-opencv \
    python3-serial can-utils net-tools i2c-tools

# 安装 Python 依赖
pip3 install -r requirements.txt

# 开启 CAN 总线 (请提前配置 /boot/firmware/extlinux/extlinux.conf 添加 overlay)
sudo ip link set can0 up type can bitrate 500000
```

### 使能 CAN 总线

RK3588 设备树配置:

```dts
&can0 {
    status = "okay";
    pinctrl-names = "default";
    pinctrl-0 = <&can0_pins>;
};
```

验证 CAN 总线:

```bash
# 查看 CAN 接口
ip link show can0

# 监听 CAN 报文
candump can0
```

---

## 传感器接线

### ZLAC8015D 驱动器接线

| 引脚 | 功能 | 连接到 |
|------|------|--------|
| VCC | 电源正极 | 24V/48V 电源 |
| GND | 电源地 | 电源地 |
| CAN_H | CAN High | RK3588 CAN0_H |
| CAN_L | CAN Low | RK3588 CAN0_L |
| A+ | 电机A+ | 左电机 A+ |
| A- | 电机A- | 左电机 A- |
| B+ | 电机B+ | 右电机 B+ |
| B- | 电机B- | 右电机 B- |
| EN | 使能 | 接高或保留 |

### 镭神 N10P 激光雷达接线

- 接口: Ethernet
- 静态IP: `192.168.1.100` (出厂默认)
- 连接到 RK3588 网口

### ETT10A-PW IMU 接线

- 接口: UART / TTL
- 波特率: 115200
- VCC → 5V
- GND → GND
- TX → RK3588 UART_RX

### 触觉阵列电子皮肤

- 接口: SPI
- VCC → 3.3V
- GND → GND
- SCK → SPI_SCK
- MISO → SPI_MISO
- MOSI → SPI_MOSI

### 六维力传感器

- 接口: RS485 / CAN
- 按照传感器手册配置波特率和地址

---

## CAN总线配置

### 节点地址规划

| 设备 | CAN 地址 (十进制) | CAN ID |
|------|-----------------|--------|
| 左驱动器 | 1 | 0x601 + |
| 右驱动器 | 2 | 0x602 + |
| IMU | 3 | N/A |
| 触觉阵列 | 4 | N/A |
| 力觉传感器 | 5 | N/A |

### 波特率

推荐: `500 kbps`

ZLAC8015D 默认: `500 kbps`

### 驱动节点测试

```python
from src.embodied.real_agv_interface import ZLAC8015DController

# 创建控制器
controller = ZLAC8015DController(channel='can0', bitrate=500000)
controller.open()

# 设置速度
controller.set_wheel_speeds(left=0.5, right=0.5)  # m/s

# 读取编码器
pos = controller.get_encoder_positions()
print(f"Encoder positions: {pos}")

controller.close()
```

---

## 部署步骤

### 1. 配置硬件

按照上述接线图连接所有传感器和驱动器

### 2. 克隆代码

```bash
cd ~
git clone https://github.com/DIT4FUN/SuperModel.git
cd SuperModel
pip3 install -r requirements.txt
```

### 3. 配置等级

根据你的AGV等级修改配置:

```python
from src.embodied.real_agv_interface import AGVHardwareConfig
from src.embodied.deployment import DeploymentConfig, DeploymentManager

# 根据等级创建配置
config = AGVHardwareConfig.from_grade('M')  # S, M, L, XL, XXL
deployment_config = DeploymentConfig(
    grade='M',
    can_channel='can0',
    lidar_ip='192.168.1.100',
    imu_port='/dev/ttyUSB0',
    enable_tactile=True,
    enable_force=True,
)
```

### 4. 预部署检查

```python
from src.embodied.deployment import DeploymentValidator

validator = DeploymentValidator()
result = validator.validate_config(deployment_config)
if result.is_healthy():
    print("配置检查通过")
else:
    print(f"检查不通过: {result.message}")
```

### 5. 创建部署管理器

```python
manager = DeploymentManager(deployment_config)
manager.deploy()
```

部署管理器会:

1. 打开所有传感器
2. 初始化驱动器
3. 启动健康监控线程
4. 启动安全护盾

### 6. 开始运行

```python
# 主循环 @ 50Hz
while True:
    manager.step()  # 读取传感器 + 融合 + 决策 + 执行
    time.sleep(0.02)
```

完整示例见 `examples/real_agv_deploy.py`

---

## 启动与验证

### 快速启动检查清单

- [ ] 电源电压正常
- [ ] CAN 总线接口已启用
- [ ] 所有传感器通电
- [ ] 紧急停止按钮可正常触发
- [ ] 驱动器通信正常
- [ ] 激光雷达IP可达
- [ ] 所有传感器读数正常
- [ ] 健康监控报告 healthy

### 运行命令

```bash
cd ~/SuperModel
python3 examples/real_agv_deploy.py --grade M
```

### 预期输出

```
[INFO]  Deployment: Starting deployment for grade M
[INFO]  CAN: Opening can0 @ 500000 bps
[INFO]  Lidar: Connected to 192.168.1.100
[INFO]  IMU: Opened /dev/ttyUSB0 @ 115200 baud
[INFO]  Tactile: SPI opened
[INFO]  Force: RS485 opened
[INFO]  Motors: ZLAC8015D initialized
[INFO]  Health: All checks passed
[INFO]  Deployment: READY
```

---

## 健康监控

`HealthMonitor` 持续监控以下指标:

- 电池电压
- 控制器温度
- 传感器连接状态
- 通信超时计数
- 驱动器错误状态

### 配置报警阈值

```python
config = DeploymentConfig(
    ...,
    battery_critical_voltage=22.0,  # 24V系统
    battery_warning_voltage=23.0,
    max_temperature_critical=80.0,
    max_temperature_warning=60.0,
    max_communication_losses=5,
)
```

### 回调通知

```python
def on_health_event(result):
    if result.is_critical():
        # 发送报警到飞书/手机
        send_alert(result.message)

health_monitor.add_callback(on_health_event)
```

### 健康报告

```python
summary = health_monitor.get_health_summary()
print(summary)
```

输出:

```json
{
  "state": "healthy",
  "battery_voltage": 24.2,
  "controller_temperature": 42.1,
  "sensor_status": {
    "lidar": "ok",
    "imu": "ok",
    "tactile": "ok",
    "force": "ok"
  },
  "communication_losses": 0
}
```

---

## 应急处理

`EmergencyProcedure` 处理以下紧急情况:

1. **碰撞检测** → 立即停止
2. **电池电压过低** → 减速并报警
3. **严重过热** → 停止并报警
4. **严重倾斜** → 停止防止倾倒
5. **紧急停止按钮按下** → 立即切断动力

### 默认应急流程

1. 切断电机动力
2. 发送报警通知
3. 保持传感器读取
4. 等待人工复位

### 自定义应急处理

```python
emergency.set_custom_callback(EmergencyType.COLLISION, my_collision_handler)
```

---

## 故障排查

### 常见问题

**Q: CAN总线找不到**
```
A: 检查设备树是否使能CAN0，检查引脚复用，检查重启后是否加载了can模块。
   sudo modprobe can
   sudo modprobe can_raw
```

**Q: 驱动器不响应**
```
A: 1. 检查CAN地址匹配 2. 检查波特率匹配 3. 检查终端电阻(120Ω)
```

**Q: 激光雷达连接不上**
```
A: ping 192.168.1.100 检查是否通，检查网线连接，检查静态IP设置。
```

**Q: 健康报告警告**
```
A: 查看 HealthCheckResult.message 获取详细原因。检查对应传感器接线。
```

**Q: 电机抖动**
```
A: 降低PID参数，检查编码器接线，检查电流限制设置是否正确。
```

**Q: 倾斜报警误触发**
```
A: 调整倾斜阈值在 DeploymentConfig 中。IMU需要校准。
```

---

## 日志

默认日志输出到: `/var/log/supermodel/`

配置日志路径:

```python
deployment_config.log_dir = "/home/agv/supermodel/logs"
```

日志轮转自动开启，保留最近 10 个日志文件。

---

## 版本

本文档对应 SuperModel v2.69.0+

更新: 2026-04-11


# =============================================================================
# 高级功能部署 (v3.1.0+)
# =============================================================================

## Gymnasium强化学习环境集成

SuperModel支持gymnasium格式的仿真环境，可用于强化学习训练。

### 安装依赖

```bash
pip install gymnasium
```

### 使用仿真环境

```python
from embodiment.simulation import GymnasiumAGVEnv, SimSceneConfig

# 创建单AGV环境
env = GymnasiumAGVEnv(
    scene_config=SimSceneConfig(scene_type="warehouse"),
    num_agvs=1,
    max_steps=1000
)

# 标准gymnasium接口
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### 向量化环境（多进程训练）

```python
from embodiment.simulation import GymnasiumVectorEnv

vec_env = GymnasiumVectorEnv(
    num_envs=8,
    scene_config=SimSceneConfig(scene_type="warehouse"),
    num_agvs_per_env=1
)

# 并行step
actions = vec_env.action_space.sample()  # (8, 2)
obs_batch, rewards, terms, truncs, infos = vec_env.step(actions)
vec_env.close()
```

---

## 市场拍卖任务分配

多AGV蜂群支持基于市场机制的拍卖分配策略。

### 启动拍卖

```python
from embodiment.multi_agv_coordinator import (
    MultiAGVCoordinator, AGVTask, MarketAuctionAllocator, MarketAuctionConfig
)

coordinator = MultiAGVCoordinator()
allocator = MarketAuctionAllocator(coordinator, MarketAuctionConfig(auction_timeout=5.0))

# 创建任务并启动拍卖
task = AGVTask(task_id="delivery_001", task_type="transport")
auction_id = allocator.start_auction(task)

# AGV提交出价（出价=成本估计，越低越容易获胜）
allocator.submit_bid(auction_id, "agv_001", bid_value=10.0)
allocator.submit_bid(auction_id, "agv_002", bid_value=8.0)  # 更低价

# 关闭拍卖，确定winner
winner = allocator.close_auction(auction_id)
print(f"Winner: {winner}")  # agv_002 (最低价)
```

---

## 编队控制

支持多种编队几何形状：直线、矩形、菱形、楔形。

### 基本编队控制

```python
from embodiment.multi_agv_coordinator import FormationController, MultiAGVCoordinator, FormationController

coordinator = MultiAGVCoordinator()
coordinator.add_agv(0, position=(0.0, 0.0))  # 领队
coordinator.add_agv(1, position=(1.0, 0.0))
coordinator.add_agv(2, position=(2.0, 0.0))

controller = FormationController(coordinator)
controller.set_leader(0)
controller.set_formation(FormationController.FormationType.LINE, spacing=1.5)

# 计算编队中每个AGV的目标位置
positions = controller.compute_formation_positions()

# 计算每个AGV的速度控制量
controls = controller.maintain_formation()
# controls = {1: (v, omega), 2: (v, omega), ...}
```

### 编队类型

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| LINE | 直线纵队 | 狭长通道、走廊 |
| RECTANGLE | 矩形方阵 | 物资搬运、区域覆盖 |
| DIAMOND | 菱形编队 | 探索、搜索 |
| WEDGE | 楔形/箭头 | 队形突击、包围 |

---

## 扩展行为树节点

### 新增节点类型

```python
from embodiment.behavior_tree_engine import (
    ParallelNode, StateMachineNode, RetryNode, TimeoutNode,
    InverterNode, AlwaysSuccessNode, AlwaysFailureNode,
    AGVTaskTrees, NodeStatus
)

# 并行执行（多子节点同时运行）
parallel = ParallelNode("parallel", ParallelNode.Policy.REQUIRE_ONE)
parallel.add_child(TaskNode("t1", lambda ctx: {"success": True}))
parallel.add_child(TaskNode("t2", lambda ctx: {"success": False}))
status = parallel.tick({})  # REQUIRE_ONE: 一个成功即成功

# 状态机（互斥状态管理）
sm = StateMachineNode("AGVMode")
sm.add_state("IDLE", TaskNode("idle_task", lambda ctx: {"success": True}))
sm.add_state("MOVING", TaskNode("move_task", lambda ctx: {"success": True}))
sm.add_transition("IDLE", "MOVING", lambda ctx: ctx.get("should_move"))
status = sm.tick({})

# 重试节点
retry = RetryNode("Retry3", TaskNode("risky", risky_task), max_retries=3)

# 超时节点
timeout = TimeoutNode("MoveTimeout", MoveTask, timeout=5.0)
```

### 预建AGV任务树

```python
from embodiment.behavior_tree_engine import AGVTaskTrees

# 巡逻任务
patrol_tree = AGVTaskTrees.build_patrol_tree()

# 物料运输任务
transport_tree = AGVTaskTrees.build_transport_tree()

# 应急处理任务
emergency_tree = AGVTaskTrees.build_emergency_tree()
```

---

## 版本更新

- v3.1.0 (2026-04-13): 新增Gymnasium集成、市场拍卖、编队控制、扩展行为树节点
- v3.0.1 (2026-04-11): 完善长期记忆系统
- v3.0.0 (2026-04-11): 最终正式版发布
