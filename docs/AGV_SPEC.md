# SuperModel AGV 规格说明

## 5.5寸轮毂电机参数

| 参数 | 数值 |
|------|------|
| 直径 | 140mm (5.5英寸) |
| 电压 | 24V DC |
| 功率 | 150W |
| 额定扭矩 | 15Nm |
| 转速 | 400RPM |
| 空载转速 | 500RPM |
| 额定电流 | 8A |
| 效率 | 85% |
| 重量 | 1.5kg |
| 防护等级 | IP65 |

## 从动轮参数 (ESUN 2.5寸)

| 参数 | 数值 |
|------|------|
| 型号 | ESUN JQR25310-80A |
| 轮径 | 63.5mm (2.5寸) |
| 材质 | 聚氨酯 PU 80A |
| 单轮承重 | 135kg |
| 减震行程 | 10mm |
| 总高度 | 106mm |
| 旋转 | 360° 全向 |

## 电机驱动器参数 (中菱 ZLAC8015D)

| 参数 | 数值 |
|------|------|
| 型号 | ZLAC8015D |
| 类型 | 一拖二轮毂伺服驱动器 |
| 工作电压 | 24~48VDC |
| 输出电流 | 15A均值 / 30A峰值 |
| 控制方式 | CANopen / RS485 |
| 尺寸 | 150×97×31mm |

## AGV 五级规格

| 等级 | 描述 | 负载 | 轮子配置 | 电机规格 |
|------|------|------|----------|----------|
| **S** | 小型AGV | 30kg | 2轮 | 57步进电机 |
| **M** | 中型AGV | 100kg | 2轮 | 5.5寸轮毂150W |
| **L** | 大型AGV | 300kg | 4轮 | 5.5寸轮毂150W x2 |
| **XL** | 超大型AGV | 600kg | 4轮 | 6.5寸轮毂200W x2 |
| **XXL** | 重型AGV | 1200kg | 4轮 | 7.5寸轮毂300W x4 |

### AGV五级详细规格对照表

#### 整车规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载能力** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **最大总重** | 45kg | 135kg | 380kg | 750kg | 1500kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **轮子配置** | 2轮驱动 | 2轮驱动 | 4轮驱动 | 4轮驱动 | 4轮驱动 |
| **轮子直径** | 100mm | 140mm | 140mm | 165mm | 200mm |
| **电机类型** | 57步进 | 5.5寸轮毂150W | 5.5寸轮毂150W×2 | 6.5寸轮毂200W×2 | 7.5寸轮毂300W×4 |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大扭矩** | 5Nm | 15Nm | 30Nm | 60Nm | 120Nm |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |

#### 感知子系统规格

| 模态 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **相机** | 单目640×480 | 双目D435i 720p | 双目D455 60fps | 双目+事件相机 | 多目+3D LiDAR |
| **麦克风** | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 |
| **触觉阵列** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **力觉** | 3轴±100N | 6轴±200N | 6轴±500N | 6轴±1000N | 6轴±5000N |
| **IMU** | MPU6050 100Hz | BMI088 200Hz | BMI088 500Hz | ADIS16470 1kHz | ADIS16470 2kHz |
| **融合编码器** | 128d | 256d | 512d | 768d | 1024d |

#### 控制子系统规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **碰撞响应** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |

#### 计算与通信规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **处理器** | RPi 4B | RK3588/Nano | Orin NX | Orin AGX | Orin AGX×2+GPU |
| **AI算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **内存** | 4GB | 8GB | 16-32GB | 64-128GB | 256+GB |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |
| **实时控制** | ✗ | ✗ | ✓ Xenomai | ✓ RT-PREEMPT | ✓ Xenomai+FPGA |
| **有线通信** | USB | USB/ETH | Ethernet | EtherCAT | EtherCAT+光纤 |
| **无线通信** | WiFi | WiFi | WiFi+5G | 5G+LoRa | 5G+卫星 |
| **多机协同** | ✗ | ✗ | ✗ | ✓ 5台 | ✓ 20台+ |

#### 感知→控制闭环延迟规格

| 阶段 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **传感器采样** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **特征提取** | 80ms | 30ms | 15ms | 5ms | 2ms |
| **融合推理** | 30ms | 10ms | 5ms | 2ms | 1ms |
| **决策规划** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制计算** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **电机响应** | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |

### M级 AGV 详细参数

```
车体参数:
  尺寸: 0.6 x 0.4 x 0.15 m
  自重: 35 kg
  负载: 100 kg

轮子参数:
  直径: 140 mm (5.5寸)
  宽度: 50 mm
  轮毂电机: 150W / 24V / 15Nm

运动参数:
  最高速度: 1.5 m/s
  最大扭矩: 15 Nm
  轨道宽度: 0.35 m
```

## 传感器配置

每个AGV配备:

1. **IMU (惯性测量单元)**
   - 位置: 车体中心上方
   - 采样率: 100Hz

2. **深度相机**
   - 位置: 车体前方
   - 分辨率: 640x480
   - 视场角: 60°

3. **激光雷达 (可选)**
   - 位置: 车体顶部中央
   - 范围: 360° / 12m

## 物理仿真参数

```python
# PyBullet 仿真配置
GRADE_CONFIGS = {
    'M': {
        'body_length': 0.6,      # m
        'body_width': 0.4,        # m
        'body_height': 0.15,      # m
        'mass': 35,               # kg
        'wheel_radius': 0.07,      # m (5.5寸)
        'wheel_width': 0.05,      # m
        'track_width': 0.35,      # m
        'max_speed': 1.5,         # m/s
        'rated_torque': 15,       # Nm
    }
}
```

## URDF 模型

AGV 模型使用 URDF (Universal Robot Description Format) 描述:

```bash
# 生成 URDF 文件
python3 -c "
from simulation.agv_model_generator import generate_agv_urdf_detailed
urdf_path = generate_agv_urdf_detailed('M', '2轮')
print(f'URDF: {urdf_path}')
"
```

## 使用示例

```python
from simulation.pybullet_sim import PyBulletSimulator, generate_agv_urdf

# 创建仿真器
sim = PyBulletSimulator(gui=True, grade='M')

# 加载AGV模型
agv_id = sim.load_agv_model()

# 设置电机速度 (rad/s)
sim.set_motor_velocities([5.0, 5.0])  # 前进
sim.set_motor_velocities([5.0, -5.0])  # 原地转向

# 获取AGV状态
state = sim.get_agv_state()
# state = {'x': ..., 'y': ..., 'theta': ..., 'vx': ..., 'vy': ..., 'omega': ...}
```

---

## 实用集成示例

### 快速创建五级AGV系统

```python
# 选择AGV等级
AGV_GRADE = 'M'  # 可选: S, M, L, XL, XXL

# 传感器初始化
from sensors.tactile import TactileArray, get_tactile_spec
from sensors.force import ForceTorqueSensor, get_force_spec
from sensors.imu import IMUSensor, get_imu_spec

tactile_spec = get_tactile_spec(AGV_GRADE)
force_spec = get_force_spec(AGV_GRADE)
imu_spec = get_imu_spec(AGV_GRADE)

print(f"触觉阵列: {tactile_spec['array']} @ {tactile_spec['sample_hz']}Hz")
print(f"力觉: {force_spec['axes']}轴 @ {force_spec['sample_hz']}Hz")
print(f"IMU: {imu_spec['model']} @ {imu_spec['sample_hz']}Hz")

# 传感器实例化
tactile = TactileArray(array_size=tactile_spec['array'])
force = ForceTorqueSensor()
imu = IMUSensor(sensor_type=imu_spec['model'], sample_rate=imu_spec['sample_hz'])

# 打开传感器
tactile.open()
force.open()
imu.open()

# 校准
imu.calibrate_gyro_bias(samples=100)
imu.calibrate_accel_bias(samples=100)

# 读取数据
import time
while True:
    t = time.time()
    
    # 触觉
    tactile_frame = tactile.capture()
    contacts = tactile.detect_contacts(tactile_frame)
    
    # 力觉
    wrench = force.capture()
    contact = force.detect_contact(wrench)
    
    # IMU
    imu_frame = imu.capture()
    
    print(f"t={t:.3f} | 触觉={len(contacts)}接触 | 力={wrench.force} | 姿态=({imu_frame.euler[0]:.2f}, {imu_frame.euler[1]:.2f}, {imu_frame.euler[2]:.2f})")
    
    time.sleep(0.01)  # 100Hz
```

### 五级配置快速查询

| 等级 | 传感器配置 | 控制频率 | 适用场景 |\n|------|-----------|---------|---------|\n| **S** | 8×8触觉 + 3轴力 + MPU6050 | 50Hz | 教学/实验 |\n| **M** | 16×16触觉 + 6轴力 + BMI088 | 100Hz | 标准AGV |\n| **L** | 24×24触觉 + 6轴力 + BMI088 500Hz | 200Hz | 工业精密 |\n| **XL** | 32×32触觉 + 6轴力 + ADIS16470 | 500Hz | 高性能 |\n| **XXL** | 48×48触觉 + 6轴力 + ADIS16470 2kHz | 1000Hz | 重载/特种 |\n

### 五级AGV完整规格对照表 (v1.53.0)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载能力** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **闭环延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |
| **触觉阵列** | 8x8 | 16x16 | 24x24 | 32x32 | 48x48 |
| **触觉采样** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **力觉轴数** | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| **力觉范围** | +/-100N | +/-200N | +/-500N | +/-1000N | +/-5000N |
| **力觉采样** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| **IMU型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| **IMU采样** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **加速度范围** | +/-8g | +/-16g | +/-24g | +/-40g | +/-80g |
| **陀螺仪范围** | +/-1000dps | +/-2000dps | +/-4000dps | +/-4000dps | +/-8000dps |
| **定位精度** | +/-10mm | +/-5mm | +/-3mm | +/-1mm | +/-0.5mm |
| **导航方式** | 磁条/二维码 | 激光导航 | SLAM视觉 | 多传感器融合 | 超模态具身智能 |
| **典型价格** | 5-15K | 15-50K | 50-150K | 150-500K | >500K |

### 集成示例 (v1.53.0)

新增完整传感器-控制集成示例:

```bash
# 运行单级AGV演示
python3 examples/sensor_control_integration_example.py M

# 运行所有五级AGV演示
python3 examples/sensor_control_integration_example.py
```

功能:
1. 按AGV等级(S/M/L/XL/XXL)自动配置传感器阵列
2. 多传感器同步采集 (触觉+力觉+IMU)
3. 互补滤波+扩展卡尔曼滤波融合
4. 感知-控制闭环验证
5. 安全监控 (速度/力/边界检查)
