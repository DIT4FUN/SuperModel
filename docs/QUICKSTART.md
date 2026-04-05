# SuperModel 快速上手指南

## 环境准备

```bash
# Python >= 3.10
python3 --version

# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install numpy pybullet pytest
```

## 目录结构

```
SuperModel/
├── src/                    # 源代码
│   ├── sensors/            # 传感器模块
│   ├── perception/         # 感知融合
│   ├── fusion/             # 跨模态融合
│   ├── learning/          # 自主学习
│   ├── control/           # 运动控制
│   └── simulation/        # 仿真环境
│       ├── pybullet_sim.py        # PyBullet仿真
│       └── agv_model_generator.py  # AGV URDF模型生成
├── sim_demos/              # PyBullet可视化仿真演示
│   ├── base_sim.py        # 仿真基类
│   ├── run_gui.py         # S形路径避障
│   ├── run_warehouse.py    # 仓库物流仿真
│   ├── run_multi_agv.py  # 多AGV协同
│   └── run_agv_showcase.py # AGV等级展示
├── tests/                  # 测试用例
├── docs/                  # 文档
└── examples/              # 示例脚本
```

## 模块速查

### 1. 传感器模块 `src/sensors/`

| 类 | 用途 | 关键方法 |
|----|------|----------|
| `BinocularCamera` | 双目RGBD相机 | `capture()`, `get_depth_map()`, `get_point_cloud()` |
| `DepthProcessor` | 深度图像处理 | `filter_depth()`, `project_to_3d()`, `depth_to_pointcloud()` |
| `BinauralMic` | 双耳麦克风 | `capture()`, `get_sound_direction()`, `localize_sources()` |
| `SoundLocalizer` | 声源定位 | `estimate_tdoa()`, `localize()`, `beamform()` |
| `TactileArray` | 电子皮肤触觉 | `capture()`, `detect_contacts()`, `get_slip_signal()` |
| `ForceTorqueSensor` | 六维力矩 | `capture()`, `detect_contact()`, `estimate_payload()` |
| `IMUSensor` | IMU惯性测量 | `capture()`, `self_test()`, `calibrate_gyro_bias()` |
| `PoseEstimator` | 姿态解算 | `update()`, `get_pose()`, `get_euler()` |

### 2. 融合模块 `src/perception/`

```python
from src.perception import CrossModalFusion, MultimodalInput, FusionConfig

# 创建融合网络
config = FusionConfig(
    vision_dim=512, audio_dim=128,
    tactile_dim=64, force_dim=32, imu_dim=64,
    hidden_dim=256, num_heads=4
)
fusion = CrossModalFusion(config)

# 前向传播
multimodal = MultimodalInput(
    vision=torch.randn(2, 512),
    audio=torch.randn(2, 128)
)
features = fusion(multimodal)  # shape: (2, 256)
```

### 3. 控制模块 `src/control/`

```python
from src.control.motion import MotionController, ControlMode
from src.control.impedance import ImpedanceController, ImpedanceParams

# 关节位置控制
controller = MotionController(num_joints=6)
torque = controller.compute_joint_torque(
    target_position=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
)

# 阻抗控制
imp = ImpedanceController(ImpedanceParams.default_6d())
```

### 4. 仿真模块 `src/simulation/`

```python
from src.simulation.environment import RobotSimulator, SensorSimulator, SimConfig

sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
sensor_sim = SensorSimulator(sim)

state = sim.step(np.zeros(6))  # 执行一步
imu_data = sensor_sim.get_imu_data()
```

## 运行测试

```bash
# 所有测试
python3 -m pytest tests/ -v

# PyBullet仿真测试
python3 -m pytest tests/pybullet_sim_tests.py -v

# 传感器测试
python3 -m pytest tests/sensor_tests.py -v

# 融合测试
python3 -m pytest tests/fusion_tests.py -v

# 控制模块测试
python3 -m pytest tests/control_tests.py -v
```

## PyBullet 仿真

```bash
cd sim_demos

# S形路径避障仿真
python3 run_gui.py

# 仓库物流仿真
python3 run_warehouse.py

# 多AGV协同仿真
python3 run_multi_agv.py

# AGV等级展示
python3 run_agv_showcase.py
```

### PyBullet AGV模型生成

```python
from simulation.agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS

# 生成M级AGV URDF
urdf_path = generate_agv_urdf_detailed('M', '2轮')

# 查看配置
print(GRADE_CONFIGS['M'])
# {'description': '中型AGV (100kg负载)', 'wheel_config': '2轮', ...}
```

### PyBullet 仿真控制

```python
import pybullet as p
from simulation.agv_model_generator import generate_agv_urdf_detailed

# 连接仿真
client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)

# 加载AGV
urdf = generate_agv_urdf_detailed('M', '2轮')
p.loadURDF('plane.urdf')
agv_id = p.loadURDF(urdf, basePosition=[0, 0, 0.15])

# 仿真控制
for i in range(1000):
    p.stepSimulation()

p.disconnect()
```

## 配置 AGV 等级

修改 `configs/default.yaml` 中的 `agv_grade`:

```yaml
agv_grade: "M"  # S / M / L / XL / XXL
```

或在代码中直接查询规格:

```python
from src.sensors.vision import get_stereo_spec
from src.sensors.tactile import get_tactile_spec
from src.sensors.force import get_force_spec
from src.sensors.imu import get_imu_spec

spec = get_stereo_spec("XL")
print(spec)
```

## 典型开发流程

### 1. 添加新传感器

```python
# 在 src/sensors/ 下创建新文件
# 实现 open(), close(), capture() 接口
# 在 __init__.py 中导出
```

### 2. 添加新技能

```python
from src.control.skill import Skill, SkillConfig, SkillResult

class MySkill(Skill):
    def can_execute(self, context):
        return True
    
    def execute(self, context):
        # 实现技能逻辑
        return SkillResult(success=True, status=SkillStatus.SUCCEEDED)
```

### 3. 添加仿真场景

```python
from src.simulation.environment import PRESET_SCENES, create_scene

scene = create_scene("tabletop")
obstacles = scene["obstacles"]
```

## 常见问题

**Q: 提示 `No module named 'src'`**
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Q: 传感器采集返回全是零**
- 检查 `open()` 是否调用
- 使用上下文管理器 `with ... as` 确保资源释放

**Q: 测试失败**
```bash
python3 -m pytest tests/sensor_tests.py::TestTactileArray -v
```

---

*文档版本: v0.1.0*
