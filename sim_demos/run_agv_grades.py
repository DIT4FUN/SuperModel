#!/usr/bin/env python3
"""
AGV 五级规格演示
================
展示 AGV 从 S 级到 XXL 级的尺寸差异
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sim_demos.base_sim import BaseSimulation, screenToWorld
except ImportError:
    from base_sim import BaseSimulation, screenToWorld
import pybullet as p
import pybullet_data
import math
import numpy as np


class AGVFiveGradeDemo(BaseSimulation):
    """AGV 五级规格演示"""
    
    GRADES = {
        'S': {'size': (0.4, 0.3, 0.15), 'color': [0.2, 0.8, 0.3, 1], 'label': 'S级 (30kg)'},
        'M': {'size': (0.6, 0.4, 0.2), 'color': [0.3, 0.6, 0.9, 1], 'label': 'M级 (100kg)'},
        'L': {'size': (0.8, 0.6, 0.25), 'color': [0.9, 0.6, 0.2, 1], 'label': 'L级 (300kg)'},
        'XL': {'size': (1.0, 0.8, 0.3), 'color': [0.8, 0.3, 0.6, 1], 'label': 'XL级 (600kg)'},
        'XXL': {'size': (1.2, 1.0, 0.35), 'color': [0.5, 0.3, 0.8, 1], 'label': 'XXL级 (1200kg)'},
    }
    
    def __init__(self):
        super().__init__("AGV 五级规格演示")
        self.agvs = {}
        self.current_grade = 'S'
        
    def setup(self):
        super().setup()
        
        # 创建各级 AGV
        positions = [(i * 2 - 4, 0) for i in range(5)]
        grades = list(self.GRADES.keys())
        
        for i, (grade, pos) in enumerate(zip(grades, positions)):
            cfg = self.GRADES[grade]
            sx, sy, sz = cfg['size']
            
            agv_id = p.createMultiBody(
                baseMass=10,
                basePosition=[pos[0], pos[1], sz],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2], 
                                                        rgbaColor=cfg['color']),
                physicsClientId=self.client
            )
            
            # 标签
            p.addUserDebugText(
                cfg['label'],
                textPosition=[pos[0], pos[1] + 1, sz + 0.2],
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                physicsClientId=self.client
            )
            
            self.agvs[grade] = {
                'id': agv_id,
                'x': pos[0],
                'y': pos[1],
                'size': cfg['size']
            }
        
        self.camera_target = [0, 0, 0]
        self.camera_distance = 12
        self.camera_pitch = -40
        
    def onUpdate(self):
        # 上下浮动动画
        t = self.sim_time
        for grade, agv in self.agvs.items():
            sx, sy, sz = agv['size']
            z = sz + 0.05 * math.sin(t * 2 + list(self.GRADES.keys()).index(grade))
            p.resetBasePositionAndOrientation(
                agv['id'], [agv['x'], agv['y'], z], [0, 0, 0, 1],
                physicsClientId=self.client
            )


def main():
    demo = AGVFiveGradeDemo()
    demo.setup()
    demo.run()


if __name__ == '__main__':
    main()
