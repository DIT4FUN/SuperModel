#!/usr/bin/env python3
"""
多 AGV 协同仿真
================
多个 AGV 协同完成任务，带碰撞检测和避障
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
    """多 AGV 协同演示 - 带避障"""
    
    def setup(self):
        super().setup()
        
        self.num_agvs = 4
        self.agvs = []
        self.targets = []
        self.completed = 0
        self.collisions = 0
        self.last_target_id = {}  # 记录上次目标，避免重复
        
        # AGV 参数
        self.agv_radius = 0.35
        self.safe_distance = 0.8
        self.repulsion_gain = 1.5
        self.min_target_distance = 1.5  # 最小目标距离
        
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
                'task': None,
                'collision_count': 0,
                'arrive_cooldown': 0  # 到达冷却时间
            })
            self.last_target_id[i] = -1
        
        # 创建目标点
        self.createTargets()
        
        self.camera_distance = 12
        self.camera_pitch = -60
        print(f"创建了 {self.num_agvs} 个 AGV")
    
    def createTargets(self):
        """创建任务目标点"""
        target_positions = [
            (4, 4), (-4, 4), (4, -4), (-4, -4),
            (6, 0), (-6, 0), (0, 6), (0, -4),
        ]
        
        for i, (tx, ty) in enumerate(target_positions[:8]):
            tid = p.createMultiBody(
                baseMass=0,
                basePosition=[tx, ty, 0.05],
                baseCollisionShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.05]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.05],
                                                      rgbaColor=[1, 0.8, 0, 0.7]),
                physicsClientId=self.client
            )
            self.targets.append({'id': tid, 'x': tx, 'y': ty, 'taken': False, 'idx': i})
    
    def compute_repulsion(self, agv, all_agvs):
        """计算来自其他AGV的斥力"""
        repulsion = np.array([0.0, 0.0])
        
        for other in all_agvs:
            if other is agv:
                continue
            
            dx = agv['x'] - other['x']
            dy = agv['y'] - other['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < self.safe_distance and dist > 0.01:
                force_mag = self.repulsion_gain * (1.0/dist - 1.0/self.safe_distance) / (dist + 0.1)
                repulsion[0] += force_mag * dx / dist
                repulsion[1] += force_mag * dy / dist
        
        return repulsion
    
    def check_collision(self, agv, all_agvs):
        """检测与其他AGV的碰撞"""
        for other in all_agvs:
            if other is agv:
                continue
            
            dx = agv['x'] - other['x']
            dy = agv['y'] - other['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < self.agv_radius * 1.5:
                return True, other
        return False, None
    
    def assignTask(self, agv, agv_idx):
        """分配任务"""
        if agv['task'] is not None:
            return
        
        # 优先选择空闲目标，且距离足够远
        free_targets = [t for t in self.targets if not t['taken'] and t['idx'] != self.last_target_id.get(agv_idx, -1)]
        
        if free_targets:
            # 计算到每个目标距离，优先选择远的
            best_target = None
            best_dist = -1
            for t in free_targets:
                d = math.sqrt((agv['x'] - t['x'])**2 + (agv['y'] - t['y'])**2)
                if d > best_dist:
                    best_dist = d
                    best_target = t
            
            target = best_target
            agv['tx'] = target['x']
            agv['ty'] = target['y']
            agv['task'] = target
            target['taken'] = True
            self.last_target_id[agv_idx] = target['idx']
        else:
            # 所有目标被占用，巡逻模式
            patrol_targets = [(4, 4), (-4, 4), (4, -4), (-4, -4), 
                            (6, 0), (-6, 0), (0, 6), (0, -4)]
            
            # 选择离当前AGV足够远的巡逻点
            best_patrol = None
            best_dist = -1
            for pt in patrol_targets:
                d = math.sqrt((agv['x'] - pt[0])**2 + (agv['y'] - pt[1])**2)
                if d > best_dist and d > self.min_target_distance:
                    best_dist = d
                    best_patrol = pt
            
            if best_patrol:
                agv['tx'] = best_patrol[0] + random.uniform(-0.3, 0.3)
                agv['ty'] = best_patrol[1] + random.uniform(-0.3, 0.3)
            else:
                # 找不到足够远的目标，随机一个方向走
                angle = random.uniform(0, 2 * math.pi)
                agv['tx'] = agv['x'] + math.cos(angle) * 3
                agv['ty'] = agv['y'] + math.sin(angle) * 3
                # 限制范围
                agv['tx'] = max(-7, min(7, agv['tx']))
                agv['ty'] = max(-5, min(7, agv['ty']))
    
    def onUpdate(self):
        for idx, agv in enumerate(self.agvs):
            # 更新冷却时间
            if agv['arrive_cooldown'] > 0:
                agv['arrive_cooldown'] -= self.dt
            
            # 分配任务（冷却时间结束后）
            if agv['task'] is None and agv['arrive_cooldown'] <= 0:
                self.assignTask(agv, idx)
            
            # 计算方向
            dx = agv['tx'] - agv['x']
            dy = agv['ty'] - agv['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            # 吸引力
            if dist > 0.1:
                attraction = np.array([dx/dist * 0.6, dy/dist * 0.6])
            else:
                attraction = np.array([0.0, 0.0])
            
            # 斥力
            repulsion = self.compute_repulsion(agv, self.agvs)
            
            # 合力
            force = attraction + repulsion * self.speed
            
            # 速度限制
            speed = np.linalg.norm(force)
            max_speed = 0.8 * self.speed
            if speed > max_speed:
                force = force / speed * max_speed
            elif speed < 0.01 and dist > 0.5:
                force = attraction * 0.5
            
            # 平滑速度
            agv['vx'] = agv['vx'] * 0.8 + force[0] * 0.2
            agv['vy'] = agv['vy'] * 0.8 + force[1] * 0.2
            
            # 防止完全停止
            speed_now = math.sqrt(agv['vx']**2 + agv['vy']**2)
            if speed_now < 0.01 and dist > 0.3:
                agv['vx'] += random.uniform(-0.05, 0.05)
                agv['vy'] += random.uniform(-0.05, 0.05)
            
            # 更新位置
            agv['x'] += agv['vx'] * self.dt
            agv['y'] += agv['vy'] * self.dt
            
            # 边界限制
            agv['x'] = max(-7, min(7, agv['x']))
            agv['y'] = max(-5, min(7, agv['y']))
            
            # 碰撞检测
            collision, other = self.check_collision(agv, self.agvs)
            if collision:
                agv['collision_count'] += 1
                self.collisions += 1
                if agv['collision_count'] == 1:
                    print(f"AGV碰撞 @ ({agv['x']:.1f}, {agv['y']:.1f})")
            
            # 到达目标
            if dist < 0.3:
                if agv['task'] is not None:
                    agv['task']['taken'] = False
                    agv['task'] = None
                    self.completed += 1
                    agv['arrive_cooldown'] = 0.5  # 0.5秒冷却
                    if self.completed % 5 == 0:
                        print(f"任务完成! 总计: {self.completed}")
            
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
        if int(self.sim_time * 2) % 2 == 0 and self.sim_time % 1 < 0.02:
            total_collision = sum(a['collision_count'] for a in self.agvs)
            avoiding = sum(1 for a in self.agvs if np.linalg.norm(self.compute_repulsion(a, self.agvs)) > 0.1)
            status = f"完成:{self.completed} 碰撞:{total_collision} 避障:{avoiding}"
            print(f"t={self.sim_time:.1f}s | {status}")


def main():
    demo = MultiAGVDemo()
    demo.setup()
    demo.run()


if __name__ == '__main__':
    main()
