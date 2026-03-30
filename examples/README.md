# SuperModel 示例

本目录包含 SuperModel 超模态机器人具身智能大脑的示例代码。

## 示例列表

### complete_system_demo.py

完整系统演示，展示从传感器采集到运动控制的端到端工作流程：

```bash
python examples/complete_system_demo.py
```

**演示内容:**

1. **传感器模块初始化**
   - BinocularCamera (双目视觉)
   - BinauralMic (双耳听觉)
   - TactileArray (触觉阵列)
   - ForceTorqueSensor (六维力矩)
   - IMUSensor (惯性测量)
   - SensorManager (统一管理器)

2. **跨模态融合网络**
   - CrossModalFusion 配置与初始化
   - MultimodalInput 构建
   - UnifiedRepresentation 生成

3. **感知与场景理解**
   - SceneUnderstanding 占据栅格
   - 物体检测与跟踪
   - 场景图谱构建

4. **运动控制系统**
   - MotionController 关节控制
   - AGVMotionController AGV控制
   - TrajectoryGenerator 轨迹生成
   - RRTPlanner 路径规划
   - ImpedanceController 阻抗控制
   - JointSpaceMPC 模型预测控制
   - SafetyController 安全监控

5. **仿真环境**
   - RobotSimulator 物理仿真
   - SuperModelGymEnv Gymnasium环境
   - VirtualTactileSensor 虚拟触觉
   - VirtualForceSensor 虚拟力觉
   - VirtualIMUSensor 虚拟IMU

6. **自主学习框架**
   - SelfSupervisedLearner 自主学习
   - WorldModelAgent 世界模型
   - DreamerAgent Dreamer智能体

## 快速开始

### 1. 传感器演示

```python
from sensors.vision import BinocularCamera
from sensors.tactile import TactileArray
from sensors.force import ForceTorqueSensor
from sensors.imu import IMUSensor

# 创建并使用传感器
cam = BinocularCamera()
cam.open()
frame = cam.capture()
cam.close()
```

### 2. 融合网络演示

```python
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
import torch

config = FusionConfig(hidden_dim=256)
fusion = CrossModalFusion(config)

multimodal = MultimodalInput(
    vision=torch.randn(2, 512),
    audio=torch.randn(2, 128)
)
features = fusion(multimodal)  # shape: (2, 256)
```

### 3. 控制演示

```python
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose

spec = AGVSpec.from_grade(AGVGrade.M)
agv = AGVMotionController(spec)

agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
wheel_cmds = agv.compute_wheel_commands(AGVPose(x=1.0, y=0.5, theta=0.0), dt=0.01)
```

### 4. 仿真演示

```python
from simulation.gym_env import make_env

env = make_env(scenario='reach', grade='M', seed=42)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

env.close()
```

## AGV 五级配置示例

```python
from sensors.manager import SensorManager, SensorManagerConfig, SensorGrade
from control.agv import AGVSpec, AGVGrade
from control.mpc import MPCConfig

# S级配置 (教育/实验)
sensors_s = SensorManagerConfig(grade='S')
agv_s = AGVSpec.from_grade(AGVGrade.S)
mpc_s = MPCConfig.for_grade('S', num_joints=6)

# M级配置 (标准助手)
sensors_m = SensorManagerConfig(grade='M')
agv_m = AGVSpec.from_grade(AGVGrade.M)
mpc_m = MPCConfig.for_grade('M', num_joints=6)

# XXL级配置 (旗舰全功能)
sensors_xxl = SensorManagerConfig(grade='XXL')
agv_xxl = AGVSpec.from_grade(AGVGrade.XXL)
mpc_xxl = MPCConfig.for_grade('XXL', num_joints=7)
```

---

*文档版本: v1.3.0*
*最后更新: 2026-03-30*
