# SuperModel AGV五级规格快速对照表

> **版本**: v1.0.0 | **更新**: 2026-04-10 | **用途**: 快速查阅 / 选型参考

---

## 一、整车参数

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **尺寸** | 0.4×0.3m | 0.6×0.4m | 0.8×0.6m | 1.0×0.7m | 1.2×0.9m |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |

---

## 二、传感器配置

| 传感器 | S | M | L | XL | XXL |
|--------|:--:|:--:|:--:|:--:|:--:|
| **视觉** | 单目640×480 | 双目D435i | 双目D455 | 双目+事件相机 | 多目+3D LiDAR |
| **听觉** | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 |
| **触觉** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **力觉** | 3轴±100N | 6轴±200N | 6轴±500N | 6轴±1000N | 6轴±5000N |
| **IMU** | MPU6050 | BMI088 | BMI088×2 | ADIS16470×2 | ADIS16470×4 |
| **编码器** | 128d | 256d | 512d | 768d | 1024d |

---

## 三、控制参数

| 控制参数 | S | M | L | XL | XXL |
|----------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **实时内核** | ✗ | ✗ | ✓Xenomai | ✓RT-PREEMPT | ✓Xenomai+FPGA |

---

## 四、计算资源

| 资源 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **处理器** | RPi 4B | RK3588/Nano | Orin NX | Orin AGX | Orin AGX×2+GPU |
| **AI算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **内存** | 4GB | 8GB | 16-32GB | 64-128GB | 256+GB |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |

---

## 五、感知→控制延迟

| 阶段 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **传感器采样** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **特征提取** | 80ms | 30ms | 15ms | 5ms | 2ms |
| **融合推理** | 30ms | 10ms | 5ms | 2ms | 1ms |
| **决策规划** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制计算** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **电机响应** | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | **<200ms** | **<80ms** | **<35ms** | **<15ms** | **<7ms** |

---

## 六、融合网络

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **融合策略** | 拼接 | 拼接+门控 | 跨模态注意力 | 多头跨模态 | 多头+层级 |
| **隐层维度** | 128 | 256 | 512 | 768 | 1024 |
| **注意力头数** | 2 | 4 | 8 | 12 | 16 |
| **编码器总数** | 3 | 5 | 8 | 12 | 16 |
| **参数量** | ~10M | ~50M | ~200M | ~500M | ~1B |

---

## 七、具身任务能力

| 任务 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **抓取** | ✗ | ✓简单 | ✓中等 | ✓复杂 | ✓精密 |
| **放置** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **推** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **拉** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **表面追踪** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **插入** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **抛光** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **自主学习** | ✗ | 简单 | 中等 | 高级 | 在线 |

---

## 八、多机协同与部署

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **有线通信** | USB | USB/ETH | Ethernet | EtherCAT | EtherCAT+光纤 |
| **无线通信** | WiFi | WiFi | WiFi+5G | 5G+LoRa | 5G+卫星 |
| **多机协同** | ✗ | ✗ | ✗ | ✓5台 | ✓20台+ |
| **典型场景** | 实验室研究 | 仓储物流 | 柔性制造 | 重载车间 | 无人化工厂 |

---

## 九、快速选型指南

```
场景                          推荐等级   核心模块组合
─────────────────────────────────────────────────────
实验室研究                    S/M       Vision + IMU + AGV
仓储物流搬运                  M/L       Vision + IMU + Navigation + Patrol
柔性制造装配                  L/XL      Vision + Force + IMU + Assembly + MPC
重载车间搬运                  XL/XXL    Vision + Force + Tactile + IMU + MPC + MultiAgent
无人化工厂                    XXL       全模态 + CrossModalFusion + Dreamer + WorldModel
```

---

## 十、代码初始化速查

```python
from src.sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from src.sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
from src.sensors.imu import IMUSensor, IMUSensorType, get_imu_spec, PoseEstimator
from src.fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from src.control.sensorimotor import SensorimotorIntegration, SensorimotorConfig
from src.control.supervisor import ControlSupervisor, get_supervisor_config

GRADE = 'L'  # 选定为L级

tactile = TactileArray(array_size=get_tactile_spec(GRADE)['array'], ...)
force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, ...)
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=get_imu_spec(GRADE)['sample_hz'])

fusion = CrossModalFusion(FusionConfig(hidden_dim=512, num_heads=8, grade=GRADE))
sensorimotor = SensorimotorIntegration(SensorimotorConfig.from_grade(GRADE))
supervisor = ControlSupervisor(config=get_supervisor_config(GRADE))
```
