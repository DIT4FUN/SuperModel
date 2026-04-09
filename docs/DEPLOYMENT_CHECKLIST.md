# SuperModel 部署清单 / Deployment Checklist

> **文档版本**: v1.0.0
> **更新**: 2026-04-09 (v2.05.0)
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档提供 SuperModel 部署的完整清单，按阶段组织，确保无遗漏。

---

## 阶段一: 硬件验收

### 1.1 传感器清单

| 检查项 | S级 | M级 | L级 | XL级 | XXL级 | 状态 |
|--------|:---:|:---:|:---:|:----:|:-----:|:----:|
| IMU 传感器 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS×2 | ☐ |
| 相机 | 单目640×480 | D435i | D455 | 双目+事件 | 多目+LiDAR | ☐ |
| 麦克风 | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 | ☐ |
| 触觉阵列 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 | ☐ |
| 力觉传感器 | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 | ☐ |
| 编码器 | 128d | 256d | 512d | 768d | 1024d | ☐ |
| 电机驱动器 | CAN/RS485 | CANopen | EtherCAT | EtherCAT×2 | EtherCAT×4 | ☐ |

### 1.2 物理连接检查

- [ ] 所有线缆连接牢固，无松动
- [ ] 电源电压在规定范围内 (24V±5%)
- [ ] CAN总线终端电阻已安装 (120Ω)
- [ ] USB/GigE连接正常
- [ ] GPIO限位开关已连接并测试
- [ ] 急停按钮功能正常

---

## 阶段二: 软件环境

### 2.1 系统依赖

```bash
# 基础依赖
python3 --version  # >= 3.10
pip install -r requirements.txt

# 传感器驱动
pip install pyrealsense2  # RealSense
pip install smbus2        # I2C/SPI IMU
pip install can          # CAN总线

# ROS2 Humble (如需)
ros2 --version  # Humble

# RK3588 NPU (如部署到RK3588)
pip install rknn-toolkit2
```

### 2.2 代码部署

```bash
# 克隆项目
git clone https://github.com/DIT4FUN/SuperModel.git
cd SuperModel

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# 验证安装
python -c "from src.control.agv import AGVMotionController; print('OK')"
```

---

## 阶段三: 传感器标定

### 3.1 IMU 标定

```python
# 1. 陀螺仪偏置标定 (静止状态)
from sensors.imu import IMUSensor, IMUSensorType
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="imu_0")
imu.open()
imu.calibrate_gyro_bias(num_samples=500, duration_sec=5.0)

# 2. 加速度计标定
imu.calibrate_accel(known_orientation="level")
imu.calibrate_accel(known_orientation="up")

# 3. 保存标定参数
imu.save_calibration("calib/imu_calib.yaml")
imu.close()
```

- [ ] 陀螺仪偏置 < 5°/h
- [ ] 加速度计零偏 < 10mg
- [ ] 姿态角静态误差 < 1°

### 3.2 相机标定

```bash
# 双目内参标定
ros2 run camera_calibration cameracalibrator.py \
    --size 8x6 --square 0.025 \
    left:=/camera/left/image_raw \
    right:=/camera/right/image_raw

# 保存标定结果
ros2 run camera_calibration camercalibrator.py \
    --size 8x6 --square 0.025 \
    left:=/camera/left/image_raw \
    right:=/camera/right/image_raw \
    --calibration_file ~/calib/stereo_calib.yaml
```

- [ ] 重投影误差 < 0.5 像素
- [ ] 基线距离误差 < 1mm
- [ ] 畸变系数 < 0.01

### 3.3 力传感器标定

```python
from sensors.force import ForceTorqueSensor, ForceSensorType
ft = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="ft_0")
ft.open()

# 偏置校准 (无负载)
ft.calibrate_bias(num_samples=100)

# 工具中心设置
ft.set_tool_center(tool_mass=0.55, tool_com=np.array([0.0, 0.0, 0.05]))

# 力范围校准 (已知砝码)
ft.calibrate_force_range()
ft.save_calibration("calib/ft_calib.yaml")
ft.close()
```

- [ ] 六轴偏置稳定性 < 0.5N/0.05Nm
- [ ] 力矩非线性误差 < 1%
- [ ] 温漂 < 0.1N/°C

### 3.4 触觉传感器标定

```python
from sensors.tactile import TactileArray, TactileSensorType
tactile = TactileArray(array_size=(16, 16), sensor_id="tactile_0")
tactile.open()

# 零压力基准
import numpy as np
frames = [tactile.capture() for _ in range(50)]
zero_pressure = np.mean([f.pressure_map for f in frames], axis=0)

# 力标定
tactile.calibrate(zero_pressure=zero_pressure)
tactile.save_calibration("calib/tactile_calib.yaml")
tactile.close()
```

- [ ] 零压力基准稳定性
- [ ] 压力范围线性度
- [ ] 空间均匀性误差 < 5%

---

## 阶段四: 控制参数配置

### 4.1 五级控制规格速查

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| 控制频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| PID Kp | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
| PID Ki | 0.01 | 0.05 | 0.1 | 0.5 | 1.0 |
| PID Kd | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
| 最大速度 | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| 最大加速度 | 0.5m/s² | 1.0m/s² | 2.0m/s² | 3.0m/s² | 5.0m/s² |

### 4.2 安全参数配置

```python
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel

safety_config = SafetyConfig(
    safety_level=SafetyLevel.STANDARD,  # 根据等级调整
    joint_limits=[...],
    velocity_limits=[...],
    torque_limits=[...],
    collision_force_threshold=50.0,
    emergency_stop_enabled=True,
)
safety = SafetyController(config=safety_config)
```

- [ ] 关节限位已设置
- [ ] 速度限制已设置
- [ ] 力矩限制已设置
- [ ] 碰撞检测阈值已设置
- [ ] 急停功能已测试

---

## 阶段五: 仿真验证

### 5.1 仿真测试清单

```bash
# 运行全部传感器测试
pytest tests/sensor_tests.py -v --tb=short

# 运行融合测试
pytest tests/fusion_tests.py -v --tb=short

# 运行控制集成测试
pytest tests/control_integration_tests.py -v --tb=short

# 运行五级基准测试
pytest tests/five_grade_pipeline_tests.py -v --tb=short

# 运行完整集成测试
pytest tests/full_pipeline_integration_tests.py -v --tb=short
```

- [ ] 所有传感器测试通过 (S/M/L/XL/XXL)
- [ ] 融合网络测试通过
- [ ] 控制模块测试通过
- [ ] 五级规格验证通过

### 5.2 仿真-实机差异分析

```python
SIM_TO_REAL_COMPENSATION = {
    "S": {"torque_scale": 0.80, "velocity_scale": 0.90},
    "M": {"torque_scale": 0.85, "velocity_scale": 0.90},
    "L": {"torque_scale": 0.90, "velocity_scale": 0.95},
    "XL": {"torque_scale": 0.95, "velocity_scale": 0.98},
    "XXL": {"torque_scale": 0.98, "velocity_scale": 0.99},
}
```

- [ ] 摩擦力补偿已调整
- [ ] 质量重心已测量
- [ ] 电机特性已实测

---

## 阶段六: 实机部署

### 6.1 部署前检查

- [ ] 所有传感器供电正常
- [ ] 通信链路畅通 (CAN/USB/Ethernet)
- [ ] 安全系统已激活
- [ ] 急停按钮功能正常
- [ ] 标定参数已加载
- [ ] 控制参数已配置

### 6.2 低速功能验证

```python
# 1. 关节原点复现
from control.motor import MotorController
motor = MotorController(num_joints=4)
motor.find_home()  # 寻找原点

# 2. 低速点到点运动
motor.go_to_position(joint_positions=[0, 0, 0, 0], speed=0.1)
motor.go_to_position(joint_positions=[0.5, 0.5, 0.5, 0.5], speed=0.1)

# 3. 力控模式切换
motor.set_control_mode(ControlMode.FORCE)
motor.apply_force(ft_sensor_reading=ft.read())

# 4. 触觉反馈测试
tactile = TactileArray(array_size=(16, 16))
frame = tactile.capture()
print(f"触觉压力分布: min={frame.pressure_map.min():.2f}, max={frame.pressure_map.max():.2f}")
```

- [ ] 关节原点复现成功
- [ ] 低速运动平稳
- [ ] 力控模式切换正常
- [ ] 触觉反馈正常

### 6.3 自主运行测试

```python
from control.embodied_control import EmbodiedController, create_for_grade

# 加载五级配置
controller = create_for_grade("M")

# 仿真运行
controller.run(num_steps=10000, dt=0.01)

# 真实运行
controller.run_real(
    sensor_manager=sensor_manager,
    control_rate=100,
    duration=300,  # 5分钟
)
```

- [ ] 5分钟连续运行无故障
- [ ] 传感器数据稳定
- [ ] 控制响应正常
- [ ] 轨迹跟踪误差 < 5cm

---

## 阶段七: 长期稳定性验证

### 7.1 24小时老化测试

- [ ] 连续运行24小时无异常
- [ ] 传感器漂移在允许范围内
- [ ] 控制性能无明显衰减
- [ ] 通信无丢包

### 7.2 性能回归测试

```python
# 定期执行基准测试
from evaluation.benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark(grade="M")
baseline = benchmark.establish_baseline()
current = benchmark.run()

for metric, (base, curr) in zip(baseline.items(), current.items()):
    regression = abs(curr - base) / base
    if regression > 0.1:  # 10% regression threshold
        print(f"WARNING: {metric} regressed by {regression*100:.1f}%")
```

---

## 总结检查表

### 交付清单

- [ ] 传感器标定报告
- [ ] 控制参数配置文件
- [ ] 安全参数验证报告
- [ ] 仿真验证报告
- [ ] 实机功能测试报告
- [ ] 24小时老化测试报告
- [ ] 用户操作手册
- [ ] 维护保养指南

### 版本信息

```
SuperModel: v2.05.0
传感器固件: 最新版本
控制固件: 最新版本
ROS2 Humble: 最新LTS
```

---

_本文档与 SuperModel v2.05.0 同步更新_
