# SuperModel 可视化仿真演示

## 目录结构

```
sim_demos/
├── base_sim.py          # 可视化基类
├── run_gui.py           # S形穿插避障仿真
├── run_warehouse.py     # 仓库物流仿真
├── run_agv_grades.py   # AGV五级规格演示
└── run_multi_agv.py     # 多AGV协同仿真
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

### 2. 仓库物流仿真 (run_warehouse.py)
- 单通道仓库布局
- 3台 AGV 自动调度
- 货架取货/送货任务
- 动态障碍物

### 3. AGV五级规格演示 (run_agv_grades.py)
- S/M/L/XL/XXL 五种规格
- 尺寸对比可视化
- 浮动动画效果

### 4. 多AGV协同仿真 (run_multi_agv.py)
- 4台 AGV 同时运行
- 任务分配系统
- 多目标点导航

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
