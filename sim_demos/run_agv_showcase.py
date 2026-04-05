#!/usr/bin/env python3
"""
SuperModel AGV模型展示
=====================
展示AGV五级规格和5.5寸轮毂电机的可视化仿真
"""

import os
import sys
import time
import math

os.environ.setdefault('DISPLAY', ':0')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

try:
    from sim_demos.base_sim import BaseSimulation
except ImportError:
    from base_sim import BaseSimulation

import pybullet as p
import pybullet_data
import numpy as np


class AGVShowcaseDemo(BaseSimulation):
    """AGV模型展示"""
    
    def __init__(self):
        super().__init__("SuperModel AGV模型展示 - 5.5寸轮毂电机")
        self.agvs = []
        self.show_info = True
        
    def setup(self):
        super().setup()
        
        # 导入模型生成器
        from simulation.agv_model_generator import (
            generate_agv_urdf_detailed,
            GRADE_CONFIGS,
            MOTOR_55_SPECS
        )
        
        self.motor_specs = MOTOR_55_SPECS
        self.grade_configs = GRADE_CONFIGS
        
        # 创建AGV展示
        positions = [
            (-4, 2),    # S级
            (0, 2),     # M级
            (4, 2),     # L级
            (-4, -2),   # XL级
            (0, -2),    # XXL级
        ]
        
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        colors = [
            [0.2, 0.8, 0.4, 1],  # 绿
            [0.2, 0.4, 0.9, 1],  # 蓝
            [0.9, 0.6, 0.2, 1],  # 橙
            [0.8, 0.3, 0.6, 1],  # 粉
            [0.5, 0.3, 0.8, 1],  # 紫
        ]
        
        for i, (grade, pos) in enumerate(zip(grades, positions)):
            cfg = self.grade_configs[grade]
            wheel_config = cfg['wheel_config']
            
            # 生成URDF
            urdf_path = generate_agv_urdf_detailed(grade, wheel_config)
            
            # 加载AGV
            agv_id = p.loadURDF(
                urdf_path,
                basePosition=[pos[0], pos[1], 0.2],
                physicsClientId=self.client
            )
            
            # 添加标签
            label_text = f"{grade}级"
            if 'wheel_diameter' in cfg:
                wheel_inch = cfg['wheel_diameter'] / 0.0254
                label_text += f" {wheel_inch:.1f}\""
            
            p.addUserDebugText(
                label_text,
                textPosition=[pos[0], pos[1] + 0.5, 0.5],
                textColorRGB=[1, 1, 1],
                textSize=1.2,
                lifeTime=0,
                physicsClientId=self.client
            )
            
            # 显示负载
            payload_text = f"负载:{cfg['payload']}kg"
            p.addUserDebugText(
                payload_text,
                textPosition=[pos[0], pos[1] + 0.3, 0.5],
                textColorRGB=[0.8, 0.8, 0.8],
                textSize=0.8,
                lifeTime=0,
                physicsClientId=self.client
            )
            
            self.agvs.append({
                'id': agv_id,
                'grade': grade,
                'x': pos[0],
                'y': pos[1],
                'color': colors[i],
                'wheel_config': wheel_config
            })
        
        # 显示标题
        p.addUserDebugText(
            "SuperModel AGV - 5.5寸轮毂电机",
            textPosition=[0, 5, 1],
            textColorRGB=[0, 1, 0.5],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.client
        )
        
        # 显示电机规格
        spec_text = f"电机: 24V/150W/15Nm/400RPM"
        p.addUserDebugText(
            spec_text,
            textPosition=[0, -5, 1],
            textColorRGB=[1, 1, 0],
            textSize=1.0,
            lifeTime=0,
            physicsClientId=self.client
        )
        
        self.camera_distance = 15
        self.camera_pitch = -50
        self.camera_yaw = 45
        self.camera_target = [0, 0, 0]
        
        print(f"创建了 {len(self.agvs)} 个AGV")
    
    def onUpdate(self):
        t = self.sim_time
        
        # 让每个AGV原地旋转展示
        for agv in self.agvs:
            # 差速驱动实现原地旋转
            if agv['wheel_config'] == '2轮':
                # 2轮AGV
                for joint_idx in range(2):
                    p.setJointMotorControl2(
                        agv['id'],
                        joint_idx,
                        p.VELOCITY_CONTROL,
                        targetVelocity=5.0 if joint_idx == 0 else -5.0,
                        force=50,
                        physicsClientId=self.client
                    )
            else:
                # 4轮AGV
                for joint_idx in range(4):
                    p.setJointMotorControl2(
                        agv['id'],
                        joint_idx,
                        p.VELOCITY_CONTROL,
                        targetVelocity=5.0 if joint_idx % 2 == 0 else -5.0,
                        force=50,
                        physicsClientId=self.client
                    )
        
        # 显示操作提示
        if self.show_info and t > 1.0:
            p.addUserDebugText(
                "SPACE=暂停 PAGE_UP/DOWN=速度",
                textPosition=[0, 6, 0.5],
                textColorRGB=[0.5, 0.5, 0.5],
                textSize=0.8,
                lifeTime=0,
                physicsClientId=self.client
            )
            self.show_info = False


def main():
    print("="*60)
    print("SuperModel AGV模型展示 - 5.5寸轮毂电机")
    print("="*60)
    
    demo = AGVShowcaseDemo()
    demo.setup()
    demo.run()


if __name__ == '__main__':
    main()
