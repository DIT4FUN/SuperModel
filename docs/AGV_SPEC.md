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

## AGV 五级规格

| 等级 | 描述 | 负载 | 轮子配置 | 电机规格 |
|------|------|------|----------|----------|
| **S** | 小型AGV | 30kg | 2轮 | 57步进电机 |
| **M** | 中型AGV | 100kg | 2轮 | 5.5寸轮毂150W |
| **L** | 大型AGV | 300kg | 4轮 | 5.5寸轮毂150W x2 |
| **XL** | 超大型AGV | 600kg | 4轮 | 6.5寸轮毂200W x2 |
| **XXL** | 重型AGV | 1200kg | 4轮 | 7.5寸轮毂300W x4 |

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
