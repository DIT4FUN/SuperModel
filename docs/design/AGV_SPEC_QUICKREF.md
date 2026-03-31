# SuperModel AGV五级快速参考卡

> 快速查阅 SuperModel 超模态机器人从 S 级到 XXL 级的核心规格差异
> 版本: v1.31.0 | 更新: 2026-04-01

---

## 一目了然

| 等级 | 定位 | 典型场景 | 算力 | 价格 |
|------|------|----------|------|------|
| **S** | 教育/实验 | 实验室研究、教学 | <5 TOPS | ¥5-15K |
| **M** | 标准助手 | 室内服务、轻工业 | 5-20 TOPS | ¥15-50K |
| **L** | 专业级 | 工业制造、物流 | 20-100 TOPS | ¥50-150K |
| **XL** | 高性能 | 复杂装配、精密操作 | 100-300 TOPS | ¥150-500K |
| **XXL** | 旗舰全功能 | 多机协作、户外全地形 | >300 TOPS | >¥500K |

---

## 感知系统核心规格

### 触觉 (Tactile)

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 阵列 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| 采样 | 50 Hz | 100 Hz | 200 Hz | 500 Hz | 1000 Hz |
| 分辨率 | 12 bit | 12 bit | 14 bit | 14 bit | 16 bit |
| 压力范围 | 500 kPa | 1000 kPa | 2000 kPa | 5000 kPa | 10000 kPa |
| 温度感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 接近觉 | ✗ | ✗ | ✓ | ✓ | ✓ |

### 力觉 (Force)

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 轴数 | 3 | 6 | 6 | 6 | 6 |
| 力范围 | ±100 N | ±200 N | ±500 N | ±1000 N | ±5000 N |
| 力矩范围 | ±10 N·m | ±20 N·m | ±50 N·m | ±100 N·m | ±500 N·m |
| 采样 | 100 Hz | 500 Hz | 1000 Hz | 2000 Hz | 5000 Hz |

### IMU

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| 采样 | 100 Hz | 200 Hz | 500 Hz | 1000 Hz | 2000 Hz |
| 噪声密度 | 400 μg/√Hz | 120 μg/√Hz | 60 μg/√Hz | 20 μg/√Hz | 10 μg/√Hz |
| 零偏稳定 | ±1°/s | ±0.5°/s | ±0.2°/s | ±0.05°/s | ±0.02°/s |

### 视觉 (Vision)

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 类型 | 单目 | 双目 | 双目 | 双目+事件相机 | 多目+3D LiDAR |
| 分辨率 | 640×480 | 1280×720 | 1280×720 | 1920×1080 | 1920×1080×4 |
| 帧率 | 30 fps | 30 fps | 60 fps | 90 fps | 120 fps |
| 基线 | - | 50 mm | 50 mm | 75 mm | 100-200 mm |
| 深度范围 | - | 0.2-5 m | 0.2-8 m | 0.1-10 m | 0.1-30 m |

### 听觉 (Audio)

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 麦克风 | 1 | 2 | 4阵列 | 6阵列 | 8阵列 |
| 采样率 | 16 kHz | 16 kHz | 22 kHz | 32 kHz | 44 kHz |
| 拾音范围 | 1 m | 3 m | 5 m | 8 m | 10 m |
| 声源定位精度 | - | ±15° | ±5° | ±2° | ±1° |

---

## 控制与执行规格

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 最大线速度 | 0.5 m/s | 1.0 m/s | 2.0 m/s | 3.0 m/s | 5.0 m/s |
| 最大角速度 | 1.5 rad/s | 2.0 rad/s | 2.5 rad/s | 3.0 rad/s | 3.5 rad/s |
| 控制频率 | 50 Hz | 100 Hz | 200 Hz | 500 Hz | 1000 Hz |
| 驱动类型 | 差速 | 差速 | 麦克纳姆 | 麦克纳姆 | 麦克纳姆 |
| 安全监控 | 基础 | 标准 | 增强 | 高级 | 全面 |

---

## 融合与认知规格

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 融合策略 | Early | Hybrid | Hybrid | Late | Late |
| 隐层维度 | 256 | 512 | 1024 | 2048 | 4096 |
| 注意力头数 | 4 | 8 | 16 | 16 | 32 |
| Transformer层数 | 2 | 4 | 6 | 8 | 12 |
| 世界模型更新频率 | 10 Hz | 20 Hz | 50 Hz | 100 Hz | 200 Hz |

---

## 硬件平台规格

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 推荐SoC | RK3566 | RK3588 | RK3588M | RK3588M×2 | RK3588M×4 |
| CPU | 4×A55 | 4×A76+4×A55 | 4×A76+4×A55 | 8×A76 | 8×A76 |
| NPU | 1 TOPS | 6 TOPS | 6 TOPS | 12 TOPS | 24 TOPS |
| 内存 | 2 GB | 4 GB | 8 GB | 16 GB | 32 GB |
| 存储 | 32 GB | 64 GB | 128 GB | 256 GB | 512 GB |

---

## 软件功能对比

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 视觉感知 | ✓ 基础 | ✓ 标准 | ✓ 增强 | ✓ 高级 | ✓ 全面 |
| 听觉感知 | ✓ ASR | ✓ ASR+定位 | ✓ 增强 | ✓ 高级 | ✓ 多波束 |
| 触觉感知 | ✗ | ✓ 压力 | ✓ 压力+接近 | ✓ 全功能 | ✓ 全功能 |
| 力觉控制 | ✗ | ✓ 基础 | ✓ 力位混合 | ✓ 阻抗 | ✓ 高级阻抗 |
| IMU稳定 | ✓ | ✓ | ✓ 增强 | ✓ 融合 | ✓ 融合+地图 |
| ROS2接口 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 多机协调 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 户外导航 | ✗ | ✗ | ✗ | ✓ | ✓ |

---

## 选型速查

```
场景                              推荐等级
──────────────────────────────────────────
教学演示/算法研究                    S
室内服务机器人                       M
工业分拣/物流                       L
精密装配/手术辅助                    XL
多机协同/户外作业                    XXL
```

---

## 快速代码示例

### 初始化 AGV 等级传感器

```python
from sensors.manager import SensorManager, SensorManagerConfig

config = SensorManagerConfig(grade="M")
manager = SensorManager(config)
manager.open_all()

frame = manager.capture_all()
print(f"可用模态: {frame.get_modalities()}")
print(f"健康状态: {frame.is_healthy()}")
```

### 获取 AGV 规格

```python
from control.agv import get_agv_spec

spec = get_agv_spec("XXL")
print(f"最大线速度: {spec.max_linear_speed} m/s")
print(f"控制频率: {spec.control_frequency} Hz")
```

### 传感器规格速查

```python
from sensors.tactile import get_tactile_spec
from sensors.force import get_force_spec
from sensors.imu import get_imu_spec

for grade in ["S", "M", "L", "XL", "XXL"]:
    t = get_tactile_spec(grade)
    f = get_force_spec(grade)
    i = get_imu_spec(grade)
    print(f"{grade}: 触觉{t['array']} {t['freq_hz']}Hz | "
          f"力觉{t['axes']}轴 {f['sampling_hz']}Hz | "
          f"IMU {i['sample_hz']}Hz")
```

### 触觉伺服控制

```python
from control.tactile_control import TactileServoController, TactileServoParams

params = TactileServoParams.from_grade("M")
ctrl = TactileServoController(params)

contacts = tactile.detect_contacts(frame)
if contacts:
    wrench = ctrl.compute_force_control(contacts[0], frame)
    print(f"接触力: {wrench.magnitude:.2f} N")
```

### 力觉导纳控制

```python
from control.force_control import ForceController, ForceControlParams

params = ForceControlParams.from_grade("L")
ctrl = ForceController(params)
ctrl.set_target_force(np.array([0, 0, -10.0, 0, 0, 0]))

wrench = force_sensor.capture()
admittance = ctrl.compute_admittance(wrench)
print(f"导纳位移: {admittance}")
```

### IMU 姿态稳定

```python
from control.imu_control import AttitudeStabilizer, AttitudeStabilizerParams

params = AttitudeStabilizerParams.from_grade("XL")
stabilizer = AttitudeStabilizer(params)
stabilizer.set_target_orientation(0.0, 0.0, 0.0)

imu_frame = imu_sensor.capture()
stabilizer.update(imu_frame)
tilt = stabilizer.get_tilt_state()
print(f"倾角状态: {tilt}")
```

---

## 版本历史

- **v1.31.0** (2026-04-01): 新增快速参考卡，汇总所有五级规格
- **v1.29.0** (2026-04-01): AGV五级完整规格总表首次发布

---

> 完整规格见: `docs/design/AGV_GRADE_SPEC.md`
> 模块接口见: `docs/design/MODULE_INTERFACE.md`
> 实战指南见: `docs/architecture/SENSOR_CONTROL_PRACTICAL_GUIDE.md`
