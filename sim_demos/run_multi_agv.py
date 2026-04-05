#!/usr/bin/env python3
"""
多 AGV 协同仿真
================
多个 AGV 协同完成任务
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
import random


class MultiAGVDemo(BaseSimulation):
    """多 AGV 协同演示"""
    
    def setup(self):
        super().setup()
        
        self.num_agvs = 4
        self.agvs = []
        self.targets = []
        self.completed = 0
        
        # 创建 AGV
        colors = [
            [0.2, 0.8, 0.4, 1],
            [0.8, 0.4, 0.2, 1],
            [0.2, 0.4, 0.8, 1],
            [0.8, 0.6, 0.2, 1],
        ]
        
        for i in range(self.num_agvs):
            x = (i % 2) * 3 - 1.5
            y = (i // 2) * 3 - 1.5
            
            agv_id = p.createMultiBody(
                baseMass=5,
                basePosition=[x, y, 0.15],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.18, 0.1]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.18, 0.1], 
                                                      rgbaColor=colors[i]),
                physicsClientId=self.client
            )
            
            self.agvs.append({
                'id': agv_id,
                'x': x,
                'y': y,
                'tx': x,
                'ty': y,
                'vx': 0,
                'vy': 0,
                'color': colors[i],
                'task': None
            })
        
        # 创建目标点
        self.createTargets()
        
        self.camera_distance = 10
        self.camera_pitch = -60
        print(f"✅ 创建了 {self.num_agvs} 个 AGV")
    
    def createTargets(self):
        """创建任务目标点"""
        target_positions = [
            (3, 3), (-3, 3), (3, -3), (-3, -3),
            (5, 0), (-5, 0), (0, 5), (0, -5),
        ]
        
        for i, (tx, ty) in enumerate(target_positions[:8]):
            tid = p.createMultiBody(
                baseMass=0,
                basePosition=[tx, ty, 0.05],
                baseCollisionShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.05]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.05],
                                                      rgbaColor=[1, 0.8, 0, 0.7]),
                physicsClientId=self.client
            )
            self.targets.append({'id': tid, 'x': tx, 'y': ty, 'taken': False})
    
    def assignTask(self, agv):
        """分配任务"""
        if agv['task'] is None:
            free_targets = [t for t in self.targets if not t['taken']]
            if free_targets:
                target = random.choice(free_targets)
                agv['tx'] = target['x']
                agv['ty'] = target['y']
                agv['task'] = target
                target['taken'] = True
    
    def onUpdate(self):
        for agv in self.agvs:
            # 分配任务
            self.assignTask(agv)
            
            # 计算方向
            dx = agv['tx'] - agv['x']
            dy = agv['ty'] - agv['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.2:
                # 到达目标
                if agv['task'] is not None:
                    agv['task']['taken'] = False
                    agv['task'] = None
                    self.completed += 1
                    print(f"✅ 任务完成! 总计: {self.completed}")
            else:
                # 移动向目标
                speed = 0.8 * self.speed
                target_vx = (dx / dist) * speed
                target_vy = (dy / dist) * speed
                
                agv['vx'] = agv['vx'] * 0.8 + target_vx * 0.2
                agv['vy'] = agv['vy'] * 0.8 + target_vy * 0.2
            
            agv['x'] += agv['vx'] * self.dt
            agv['y'] += agv['vy'] * self.dt
            
            # 角度
            if abs(agv['vx']) > 0.01 or abs(agv['vy']) > 0.01:
                theta = math.atan2(agv['vy'], agv['vx'])
            else:
                theta = 0
            
            p.resetBasePositionAndOrientation(
                agv['id'],
                [agv['x'], agv['y'], 0.15],
                p.getQuaternionFromEuler([0, 0, theta]),
                physicsClientId=self.client
            )
        
        # 显示状态
        if int(self.sim_time) % 2 == 0 and self.sim_time % 2 < 0.1:
            statuses = [f"AGV{i+1}:{'✓' if a['task'] is None else '→'} " for i, a in enumerate(self.agvs)]
            print(f"t={self.sim_time:.1f}s | {' '.join(statuses)} | 完成:{self.completed}")


def main():
    demo = MultiAGVDemo()
    demo.setup()
    demo.run()


if __name__ == '__main__':
    main()
