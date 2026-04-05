#!/usr/bin/env python3
"""
SuperModel 仿真可视化基类
=========================
提供通用的 PyBullet 可视化框架
"""

import os
import sys
import time
import math
import pybullet as p
import pybullet_data
import numpy as np


class BaseSimulation:
    """仿真基类"""
    
    def __init__(self, title="SuperModel Simulation"):
        self.title = title
        self.client = None
        self.running = True
        self.paused = False
        self.speed = 3.0
        self.camera_distance = 15.0
        self.camera_yaw = 45
        self.camera_pitch = -50
        self.camera_target = [0, 0, 0]
        self.sim_time = 0.0
        self.dt = 1.0 / 60.0
        
        # 鼠标状态
        self.is_dragging = False
        self.last_mouse = (0, 0)
        
    def setup(self):
        """初始化仿真环境"""
        os.environ.setdefault('DISPLAY', ':0')
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf', physicsClientId=self.client)
        self.resetCamera()
        print(f"✅ {self.title} - 初始化完成")
        
    def resetCamera(self):
        """重置相机"""
        p.resetDebugVisualizerCamera(
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            self.camera_target,
            physicsClientId=self.client
        )
    
    def processInput(self):
        """处理输入"""
        keys = p.getKeyboardEvents(self.client)
        
        # SPACE 暂停
        if 32 in keys and keys[32] == p.KEY_IS_DOWN:
            if not hasattr(self, '_space_was') or not self._space_was:
                self.paused = not self.paused
                print(f"{'⏸️ 暂停' if self.paused else '▶️ 继续'}")
            self._space_was = True
        else:
            self._space_was = False
        
        # 速度调节
        if 65299 in keys and keys[65299] == p.KEY_IS_DOWN:  # PAGE_UP
            self.speed = min(5.0, self.speed + 0.01)
        if 65300 in keys and keys[65300] == p.KEY_IS_DOWN:  # PAGE_DOWN
            self.speed = max(0.1, self.speed - 0.01)
        
        # 视角调整
        if 65297 in keys and keys[65297] == p.KEY_IS_DOWN:  # UP
            self.camera_distance = max(5, self.camera_distance - 0.1)
        if 65298 in keys and keys[65298] == p.KEY_IS_DOWN:  # DOWN
            self.camera_distance = min(50, self.camera_distance + 0.1)
        if 65295 in keys and keys[65295] == p.KEY_IS_DOWN:  # LEFT
            self.camera_yaw = (self.camera_yaw - 0.5) % 360
        if 65296 in keys and keys[65296] == p.KEY_IS_DOWN:  # RIGHT
            self.camera_yaw = (self.camera_yaw + 0.5) % 360
        
        # INSERT 俯视
        if 65303 in keys and keys[65303] == p.KEY_IS_DOWN:
            self.camera_pitch = -89
            self.camera_yaw = 0
        
        # DELETE 斜视
        if 65304 in keys and keys[65304] == p.KEY_IS_DOWN:
            self.camera_pitch = -50
            self.camera_yaw = 45
        
        # 鼠标事件
        for event in p.getMouseEvents(self.client):
            etype, _, _, mx, my = event
            
            if etype == 2:  # 滚轮
                self.camera_distance = max(5, min(50, self.camera_distance - event[2] * 0.3))
            
            elif etype == 3:  # 左键按下
                self.is_dragging = True
                self.last_mouse = (mx, my)
            
            elif etype == 4:  # 左键释放
                self.is_dragging = False
            
            elif etype == 5 and self.is_dragging:  # 拖动
                dx, dy = mx - self.last_mouse[0], my - self.last_mouse[1]
                self.camera_yaw = (self.camera_yaw + dx * 0.3) % 360
                self.camera_pitch = max(-89, min(-10, self.camera_pitch + dy * 0.3))
                self.last_mouse = (mx, my)
    
    def updateCamera(self):
        """更新相机"""
        p.resetDebugVisualizerCamera(
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            self.camera_target,
            physicsClientId=self.client
        )
    
    def step(self):
        """仿真步进 - 子类重写"""
        if not self.paused:
            p.stepSimulation(self.client)
            self.sim_time += self.dt * self.speed
    
    def run(self, duration=float('inf')):
        """运行仿真"""
        print("\n" + "=" * 60)
        print(f"🏁 {self.title}")
        print("=" * 60)
        print("""
┌─────────────────────────────────────────────────────────────┐
│  SPACE=暂停  PAGE_UP/DOWN=速度  ↑↓←→=视角  INSERT=俯视    │
│  鼠标滚轮=缩放  拖拽=旋转  Ctrl+C=退出                       │
└─────────────────────────────────────────────────────────────┘
""")
        
        try:
            while self.running and self.sim_time < duration:
                self.processInput()
                
                if not self.paused:
                    p.stepSimulation(self.client)
                    self.sim_time += self.dt * self.speed
                
                self.updateCamera()
                self.onUpdate()
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n收到退出信号")
        finally:
            self.cleanup()
    
    def onUpdate(self):
        """每帧更新 - 子类重写"""
        pass
    
    def cleanup(self):
        """清理 - 子类重写"""
        p.disconnect(self.client)
        print("再见!")


def screenToWorld(mx, my, state):
    """屏幕坐标转世界坐标"""
    w, h = 640, 480
    yr, pr = math.radians(state.camera_yaw), math.radians(state.camera_pitch)
    
    nx = 2 * mx / w - 1
    ny = -(2 * my / h - 1)
    
    fov = math.radians(30)
    rx, ry, rz = nx * math.tan(fov), ny * math.tan(fov), -1
    
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)
    
    wx = rx * cy - rz * sy
    wy = rx * sy + rz * cy
    wz = ry * cp
    
    d = state.camera_distance
    cx = state.camera_target[0] - d * sy * cp
    cy = state.camera_target[1] + d * cy * cp
    cz = state.camera_target[2] - d * sp
    
    if abs(wz) > 0.001:
        t = -cz / wz
        if t > 0:
            return cx + t * wx, cy + t * wy
    return None, None


if __name__ == '__main__':
    print("SuperModel 仿真可视化基类")
    print("子类化并重写 setup(), step(), onUpdate() 方法")
