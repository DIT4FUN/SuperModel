# SuperModel AGV五级规格总表 (Master Spec)
> **版本**: v2.62.0 | **更新**: 2026-04-10 | **维护**: SuperModel开发团队

---

## 一、项目状态概览

| 模块 | 状态 | 文件 | 测试 |
|------|------|------|------|
| 触觉感知 | ✅ 完成 | `src/sensors/tactile.py` | sensor_tests.py |
| 力觉感知 | ✅ 完成 | `src/sensors/force.py` | sensor_tests.py |
| IMU感知 | ✅ 完成 | `src/sensors/imu.py` | sensor_tests.py |
| 传感器管理器 | ✅ 完成 | `src/sensors/manager.py` | sensor_tests.py |
| 跨模态融合网络 | ✅ 完成 | `fusion/` | fusion_tests.py |
| 控制模块 | ✅ 完成 | `src/control/` | control_tests.py |
| 具身仿真环境 | ✅ 完成 | `src/control/embodied_sim.py` | embodied_sim_tests.py |
| 核心目标系统 | ✅ 完成 | `src/core/` | core_tests.py |
| 自主学习框架 | ✅ 完成 | `src/learning/` | autonomous_learning_tests.py |
| AGV五级规格表 | ✅ 完成 | 本文档 | 五级集成测试 |

---

## 二、AGV五级物理规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **负载** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **最大总重** | 45kg | 135kg | 380kg | 750kg | 1500kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **轮子配置** | 2轮差速 | 2轮差速 | 4轮差速 | 4轮差速 | 4轮差速 |
| **轮子直径** | 4寸(100mm) | 5.5寸(140mm) | 5.5寸(140mm) | 6.5寸(165mm) | 7.5寸(200mm) |
| **驱动电机** | 57步进 | 5.5寸轮毂150W×2 | 5.5寸轮毂150W×2 | 6.5寸轮毂200W×2 | 7.5寸轮毂300W×4 |
| **从动轮** | 2寸万向轮 | 2.5寸静音万向轮 | 2.5寸静音万向轮 | 3寸重型万向轮 | 4寸重型万向轮 |
| **最大速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大角速度** | 1.5rad/s | 2.0rad/s | 2.5rad/s | 3.0rad/s | 3.5rad/s |
| **最大扭矩** | 50Nm | 100Nm | 200Nm | 500Nm | 1000Nm |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |
| **典型场景** | 教育/实验室 | 室内服务/轻工业 | 工业制造/物流 | 复杂装配/精密 | 多机协作/户外 |

---

## 三、AGV五级控制规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制周期** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **仿真步长** | 2ms | 1ms | 0.5ms | 0.2ms | 0.1ms |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **碰撞响应** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **实时控制** | ✗ | ✗ | ✓ Xenomai | ✓ RT-PREEMPT | ✓ Xenomai+FPGA |

---

## 四、AGV五级感知规格总表

### 4.1 触觉 (Tactile)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **阵列尺寸** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **采样频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **ADC分辨率** | 12bit | 12bit | 14bit | 14bit | 16bit |
| **压力范围** | 500kPa | 1000kPa | 2000kPa | 5000kPa | 10000kPa |
| **空间分辨率** | 5mm | 3mm | 2mm | 1.5mm | 1mm |
| **温度感知** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **接近觉** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **滑移检测** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **通信接口** | I2C | SPI | USB | USB/Ethernet | EtherCAT |
| **编码器维度** | 32 | 64 | 64 | 128 | 128 |
| **传感器类型** | 电阻式 | 电容式 | 电容式 | 电容式 | 压电式 |

### 4.2 力觉 (Force)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **轴数** | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| **力范围** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **力矩范围** | ±10N·m | ±20N·m | ±50N·m | ±100N·m | ±500N·m |
| **分辨率** | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| **采样频率** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| **通信接口** | USB | USB/Ethernet | Ethernet | EtherCAT | EtherCAT |
| **编码器维度** | 16 | 32 | 32 | 64 | 64 |
| **传感器类型** | 三维力 | 六维力矩 | 六维力矩 | 六维力矩 | 六维力矩 |

### 4.3 IMU

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| **加速度量程** | ±8g | ±16g | ±24g | ±40g | ±80g |
| **陀螺量程** | ±1000°/s | ±2000°/s | ±4000°/s | ±4000°/s | ±8000°/s |
| **采样频率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **噪声密度** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **零偏稳定性** | ±1°/s | ±0.5°/s | ±0.2°/s | ±0.05°/s | ±0.02°/s |
| **磁力计** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **编码器维度** | 32 | 32 | 64 | 64 | 128 |
| **接口** | I2C | SPI/I2C | SPI/I2C | SPI | SPI |

### 4.4 视觉 (Vision)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **类型** | 单目USB | 双目D435i | 双目D455 | 双目+事件相机 | 多目+3D LiDAR |
| **分辨率** | 640×480 | 1280×720 | 1280×720 | 1920×1080 | 1920×1080×4 |
| **帧率** | 30fps | 30fps | 60fps | 90fps | 120fps |
| **基线** | — | 50mm | 50mm | 75mm | 100-200mm |
| **深度范围** | — | 0.2-5m | 0.2-8m | 0.1-10m | 0.1-30m |
| **视场角** | 60° | 85° | 87° | 91° | 180° |
| **编码器维度** | — | 256 | 512 | 768 | 1024 |

### 4.5 听觉 (Audio)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **麦克风数量** | 1 | 2(双耳) | 4阵列 | 6阵列 | 8阵列 |
| **采样率** | 16kHz | 16kHz | 22kHz | 32kHz | 44kHz |
| **拾音范围** | 1m | 3m | 5m | 8m | 10m |
| **波束形成** | ✗ | ✓ | ✓ | ✓ | ✓(多波束) |
| **声源定位精度** | — | ±15° | ±5° | ±2° | ±1° |
| **编码器维度** | 64 | 128 | 128 | 256 | 256 |

---

## 五、AGV五级融合与认知规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **融合策略** | Early | Hybrid | Hybrid | Late | Late |
| **隐层维度** | 256 | 512 | 1024 | 2048 | 4096 |
| **注意力头数** | 4 | 8 | 16 | 16 | 32 |
| **Transformer层数** | 2 | 4 | 6 | 8 | 12 |
| **世界模型更新频率** | 10Hz | 20Hz | 50Hz | 100Hz | 200Hz |
| **轨迹预测horizon** | 0.5s | 1.0s | 2.0s | 5.0s | 10.0s |

---

## 六、AGV五级硬件平台规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **推荐SoC** | RK3566 | RK3588 | RK3588M | RK3588M×2 | RK3588M×4+GPU |
| **AI算力** | 5 TOPS | 20 TOPS | 100 TOPS | 300 TOPS | 500+ TOPS |
| **内存** | 2GB | 4GB | 8GB | 16GB | 32GB |
| **存储** | 32GB | 64GB | 128GB | 256GB | 512GB |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |
| **通信** | WiFi5 | WiFi6 | WiFi6+BT5.2 | 5G | 5G+LoRa |

---

## 七、AGV五级传感器异常降级策略总表

| 传感器 | 异常类型 | S | M | L | XL | XXL |
|--------|---------|:--:|:--:|:--:|:--:|:--:|
| **IMU** | 通信中断 | 停机 | 降速50% | 保持速度+报警 | 降级到编码器融合 | 降级+预测+告警 |
| **IMU** | 偏置漂移 | — | 偏置重校准 | 自动偏置补偿 | 在线漂移补偿 | 在线漂移补偿+告警 |
| **力觉** | 通信中断 | 停机 | 碰撞检测失效 | 速度限制 | 触觉+视觉替代 | 传感器融合冗余 |
| **力觉** | 饱和/过载 | 停机 | 报警 | 限幅+报警 | 限幅+记录+告警 | 限幅+预测+告警 |
| **触觉** | 部分失效 | — | 忽略失效区 | 忽略+补全 | 忽略+补偿+告警 | 局部失效降级+告警 |
| **编码器** | 通信中断 | 停机 | 停机 | 停机 | IMU航位推算 | IMU航位推算+告警 |
| **激光雷达** | 降质 | 停机 | 停机 | 减速+避障 | 减速+告警 | 多传感器补偿 |

---

## 八、模块接口快速索引

| 模块 | 主文件 | 关键类 | 工厂函数 |
|------|--------|--------|---------|
| 触觉 | `src/sensors/tactile.py` | `TactileArray`, `VirtualTactileSensor`, `PressureProcessor` | `get_tactile_spec()` |
| 力觉 | `src/sensors/force.py` | `ForceTorqueSensor`, `VirtualForceSensor`, `WrenchProcessor` | `get_force_spec()` |
| IMU | `src/sensors/imu.py` | `IMUSensor`, `VirtualIMUSensor`, `PoseEstimator` | `get_imu_spec()` |
| 传感器管理 | `src/sensors/manager.py` | `SensorManager`, `TimeSynchronizer` | — |
| 编码器 | `src/sensors/encoders.py` | `SensorEncoder`, `MultiModalEncoder` | `get_encoder_spec()` |
| 传感器融合 | `fusion/` | `SensorFusion`, `CrossModalFusion` | — |
| 控制-速度 | `src/control/velocity_control.py` | `AGVVelocityController` | `get_velocity_control_spec()` |
| 控制-力控 | `src/control/force_control.py` | `ForceController` | — |
| 控制-阻抗 | `src/control/impedance.py` | `ImpedanceController` | — |
| 控制-轨迹 | `src/control/trajectory.py` | `TrajectoryPlanner` | — |
| 控制-具身 | `src/control/embodied_control.py` | `EmbodiedController` | — |
| 控制-具身仿真 | `src/control/embodied_sim.py` | `EmbodiedSimulator` | — |
| 控制-分级 | `src/control/grade_control.py` | `GradeAwareController` | `get_control_grade_spec()` |
| 控制-蜂群 | `src/control/swarm_control.py` | `SwarmController` | — |
| 仿真 | `src/control/simulation.py` | `AGVSimulator` | — |
| 核心目标 | `src/core/core_brain.py` | `CoreBrain` | — |
| 自主学习 | `src/learning/autonomous_learning.py` | `AutonomousLearning` | — |
| 世界模型 | `src/learning/world_model.py` | `WorldModel` | — |
| Dreamer | `src/learning/dreamer_agent.py` | `DreamerAgent` | — |

---

## 九、快速使用示例

### 9.1 传感器初始化

```python
from src.sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from src.sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
from src.sensors.imu import IMUSensor, IMUSensorType, get_imu_spec

# M级AGV规格
tactile_spec = get_tactile_spec('M')  # 16x16, 100Hz
force_spec = get_force_spec('M')        # 6轴, ±200N, 500Hz
imu_spec = get_imu_spec('M')            # BMI088, 200Hz

# 创建传感器
tactile = TactileArray(array_size=(16, 16), sensor_type=TactileSensorType.CAPACITIVE)
force = ForceTorqueSensor(ForceSensorType.SIX_AXIS)
imu = IMUSensor(IMUSensorType.BMI088, sample_rate=200)

# 使用上下文管理器
with tactile, force, imu:
    for _ in range(100):
        t_frame = tactile.capture()
        f_wrench = force.capture()
        i_frame = imu.capture()
        print(f"触觉: {t_frame.pressure_map.shape}, 力觉: {f_wrench.magnitude:.2f}N, IMU: {i_frame.accel_magnitude:.2f}m/s²")
```

### 9.2 姿态估计

```python
from src.sensors.imu import IMUSensor, PoseEstimator, IMUSensorType

imu = IMUSensor(IMUSensorType.BMI088, sample_rate=200)
pose_estimator = PoseEstimator(algorithm="madgwick", sample_rate=200.0)

with imu:
    for _ in range(100):
        frame = imu.capture()
        pose = pose_estimator.update(frame.accel, frame.gyro, dt=1/200)
        euler = pose.to_euler()
        print(f"Roll: {euler[0]:.3f}, Pitch: {euler[1]:.3f}, Yaw: {euler[2]:.3f}")
```

### 9.3 具身仿真

```python
from src.control.embodied_sim import EmbodiedSimulator

sim = EmbodiedSimulator(grade='M', dt=0.01)
obs = sim.reset()

for step in range(1000):
    # 随机动作
    action = np.random.randn(2) * 0.5
    obs, reward, done, info = sim.step(action)
    if done:
        obs = sim.reset()
```

---

## 十、测试覆盖

| 测试文件 | 测试数量 | 覆盖模块 |
|----------|---------|---------|
| `tests/sensor_tests.py` | 347 | tactile, force, imu, manager, encoders |
| `tests/fusion_tests.py` | 79 | cross-modal fusion, sensor fusion, EKF |
| `tests/control_tests.py` | 100+ | velocity, force, impedance, trajectory |
| `tests/embodied_sim_tests.py` | 50+ | embodied simulation |
| `tests/core_tests.py` | 53 | core goals, safety shield |
| **总计** | **600+** | **全模块覆盖** |

---

*文档版本: v2.62.0 | 更新日期: 2026-04-10*
