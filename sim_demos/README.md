# SuperModel PyBullet 仿真演示

## 目录结构

```
sim_demos/
├── base_sim.py           # 可视化基类
├── run_gui.py           # S形穿插避障仿真
├── run_warehouse.py      # 仓库物流仿真
├── run_multi_agv.py     # 多AGV协同仿真
├── run_agv_showcase.py  # AGV等级展示
└── run_agv_grades.py   # AGV五级规格演示
```

## 快速开始

```bash
cd ~/.openclaw/workspace/projects/SuperModel/sim_demos

# 激活虚拟环境
../venv/bin/python3 run_xxx.py
```

## 演示说明

### 1. S形穿插避障仿真 (run_gui.py)
- AGV 沿 S 形路径运动
- 势场法避障算法
- 可动态添加障碍物
- 碰撞检测
- **默认速度**: 3x

### 2. 仓库物流仿真 (run_warehouse.py)
- 单通道仓库布局
- 3台 AGV 自动调度
- 货架取货/送货任务
- 动态障碍物
- **默认速度**: 3x

### 3. AGV等级展示 (run_agv_showcase.py)
- S/M/L 三种规格
- 尺寸对比可视化
- 5.5寸轮毂电机 + ESUN从动轮
- 真实物理参数

### 4. 多AGV协同仿真 (run_multi_agv.py)
- 4台 AGV 同时运行
- 任务分配系统
- 多目标点导航
- 避障协调

### 5. AGV五级规格演示 (run_agv_grades.py)
- S/M/L/XL/XXL 五种规格
- 尺寸对比可视化
- 浮动动画效果

## 操作说明

### 键盘控制 (终端窗口)
| 按键 | 功能 |
|------|------|
| SPACE | 暂停/继续 |
| PAGE_UP | 加速 |
| PAGE_DOWN | 减速 |
| ↑ | 视角拉近 |
| ↓ | 视角拉远 |
| ← | 视角左转 |
| → | 视角右转 |
| INSERT | 俯视 |
| DELETE | 斜视 |

### 鼠标控制 (PyBullet窗口)
| 操作 | 功能 |
|------|------|
| 滚轮 | 缩放视角 |
| 左键拖拽 | 旋转视角 |
| 左键单击 | 添加障碍物 (部分演示) |

## AGV 硬件配置

### 驱动系统
- **电机**: 5.5寸轮毂电机 (140mm, 24V/150W/15Nm)
- **驱动器**: 中菱 ZLAC8015D (一拖二, CANopen/RS485)
- **从动轮**: ESUN 2.5寸静音避震万向轮 (聚氨酯80A)

### 传感器
- **激光雷达**: 镭神 N10P (360°, 25m, TOF)
- **IMU**: ETT10A-PW (6轴, IP67防水)
- **RGB相机**: 奥比中光 C100 (1080P, FOV 112°)
- **深度相机**: 奥比中光 Astra Pro Plus (可选)

## 依赖

```bash
pip install pybullet numpy
```

## 基类使用

```python
from base_sim import BaseSimulation

class MyDemo(BaseSimulation):
    def setup(self):
        super().setup()
        # 初始化场景
        
    def onUpdate(self):
        # 每帧更新逻辑
        pass

demo = MyDemo("我的演示")
demo.setup()
demo.run()
```
