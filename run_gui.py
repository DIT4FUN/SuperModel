#!/usr/bin/env python3
"""
SuperModel AGV S形穿插避障仿真 - 增强版
=========================================
鼠标+键盘控制 + 动态添加障碍物 + 实时坐标显示
"""

import os
import sys
import time
import math

os.environ.setdefault('DISPLAY', ':0')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import pybullet as p
import pybullet_data
import numpy as np


class SimState:
    """仿真全局状态"""
    def __init__(self):
        self.speed_multiplier = 1.0
        self.camera_distance = 15.0
        self.camera_yaw = 60
        self.camera_pitch = -50
        self.camera_target = [0, 3, 0]
        self.paused = False
        self.collision_count = 0
        self.obstacles = []
        self.obstacle_ids = []
        self.last_collision_time = -10
        self.client = None
        self.running = True
        
        # 鼠标状态
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_click_pos = None
        
        # 坐标文本ID（用于更新）
        self.agv_text_id = None
        self.mouse_text_id = None
        self.speed_text_id = None


class SAvoidController:
    """S形穿插避障控制器"""
    
    def __init__(self, sim_state):
        self.sim_state = sim_state
        
        self.path_amplitude_x = 4.0
        self.path_frequency_x = 0.3
        self.path_frequency_y = 0.8
        self.path_speed = 0.4
        
        self.safe_distance = 0.8
        self.warning_distance = 2.5
        self.repulsion_gain = 2.0
        self.avoidance_strength = 1.5
        
        self.position = np.array([0.0, -3.0])
        self.angle = 0.0
        self.velocity = np.array([0.0, 0.0])
        
    def compute_s_path(self, t):
        x = self.path_amplitude_x * math.sin(self.path_frequency_x * t)
        y = t * self.path_speed
        x += 1.5 * math.sin(self.path_frequency_y * t * 0.5)
        return np.array([x, y])
    
    def compute_repulsion(self, obstacles):
        repulsion = np.array([0.0, 0.0])
        for obs_pos in obstacles:
            dx = self.position[0] - obs_pos[0]
            dy = self.position[1] - obs_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < self.warning_distance and dist > 0.01:
                if dist < self.safe_distance:
                    force_mag = self.repulsion_gain * 3.0 / (dist + 0.1)
                else:
                    force_mag = self.repulsion_gain * (1.0/dist - 1.0/self.warning_distance) / (dist * 0.5)
                
                repulsion[0] += force_mag * dx / (dist + 0.1)
                repulsion[1] += force_mag * dy / (dist + 0.1)
        return repulsion
    
    def update(self, t, dt, obstacles):
        speed = self.sim_state.speed_multiplier
        path_target = self.compute_s_path(t * speed)
        
        attraction = (path_target - self.position) * 2.0 * speed
        repulsion = self.compute_repulsion(obstacles) * self.avoidance_strength
        
        force = attraction + repulsion
        
        self.velocity = self.velocity * 0.8 + force * dt * 0.5
        
        speed_norm = np.linalg.norm(self.velocity)
        max_speed = 2.0 * speed
        if speed_norm > max_speed:
            self.velocity = self.velocity / speed_norm * max_speed
        
        self.position += self.velocity * dt
        
        if speed_norm > 0.05:
            target_angle = math.atan2(self.velocity[1], self.velocity[0])
            angle_diff = target_angle - self.angle
            while angle_diff > math.pi: angle_diff -= 2 * math.pi
            while angle_diff < -math.pi: angle_diff += 2 * math.pi
            self.angle += angle_diff * 0.3
        
        return self.position.copy(), self.angle
    
    def check_collision(self, obstacles):
        for obs_pos in obstacles:
            dist = np.linalg.norm(self.position - np.array(obs_pos))
            if dist < self.safe_distance * 0.7:
                return True
        return False


def add_obstacle_at_position(position, sim_state):
    """添加障碍物"""
    colors = [
        (0.9, 0.3, 0.3, 1), (0.3, 0.9, 0.3, 1), (0.3, 0.3, 0.9, 1),
        (0.9, 0.6, 0.1, 1), (0.7, 0.3, 0.9, 1), (0.1, 0.9, 0.9, 1),
    ]
    
    color = colors[len(sim_state.obstacles) % len(colors)]
    size = 0.3
    
    obs_id = p.createMultiBody(
        baseMass=0,
        basePosition=[position[0], position[1], size],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[size]*3),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[size]*3, rgbaColor=color),
        physicsClientId=sim_state.client
    )
    
    # 安全区
    p.createMultiBody(
        baseMass=0,
        basePosition=[position[0], position[1], 0.02],
        baseCollisionShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.8),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.8, rgbaColor=[1, 0, 0, 0.15]),
        physicsClientId=sim_state.client
    )
    
    sim_state.obstacles.append((position[0], position[1]))
    sim_state.obstacle_ids.append(obs_id)
    
    print(f"➕ 障碍物 #{len(sim_state.obstacles)} @ ({position[0]:.1f}, {position[1]:.1f})")


def screen_to_world(mouse_x, mouse_y, sim_state):
    """将屏幕坐标转换为世界坐标"""
    width, height = 640, 480
    
    yaw_rad = math.radians(sim_state.camera_yaw)
    pitch_rad = math.radians(sim_state.camera_pitch)
    
    nx = (2.0 * mouse_x / width - 1.0)
    ny = -(2.0 * mouse_y / height - 1.0)
    
    fov = 60.0
    fov_rad = fov * math.pi / 180.0
    tan_h = math.tan(fov_rad / 2.0)
    tan_w = tan_h * (width / height)
    
    ray_x = nx * tan_w
    ray_y = ny * tan_h
    ray_z = -1.0
    
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    
    world_x = ray_x * cy - ray_z * sy
    world_y = ray_x * sy + ray_z * cy
    world_z = ray_y * cp
    
    dist = sim_state.camera_distance
    cam_x = sim_state.camera_target[0] - dist * sy * cp
    cam_y = sim_state.camera_target[1] + cy * cp
    cam_z = sim_state.camera_target[2] - dist * sp
    
    if abs(world_z) > 0.001:
        t = -cam_z / world_z
        if t > 0:
            wx = cam_x + t * world_x
            wy = cam_y + t * world_y
            return wx, wy
    return None, None


def main():
    print("=" * 60)
    print("🏁 SuperModel AGV 增强版避障仿真")
    print("=" * 60)
    
    sim_state = SimState()
    
    client = p.connect(p.GUI)
    sim_state.client = client
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 地面
    plane_id = p.loadURDF('plane.urdf', physicsClientId=client)
    
    # 初始障碍物
    initial_obstacles = [
        (-3.5, 1.0), (-3.5, 4.0), (-3.5, 7.0),
        (-1.5, 2.5), (-1.5, 5.5), (-1.5, 8.5),
        (1.5, 1.5), (1.5, 4.5), (1.5, 7.5),
        (3.5, 2.0), (3.5, 5.0), (3.5, 8.0),
    ]
    
    print(f"创建 {len(initial_obstacles)} 个初始障碍物...")
    
    colors = [
        (0.9, 0.2, 0.2, 1), (0.2, 0.8, 0.2, 1), (0.2, 0.2, 0.9, 1),
        (0.9, 0.5, 0.1, 1), (0.5, 0.2, 0.8, 1), (0.2, 0.8, 0.8, 1),
    ]
    
    for i, (x, y) in enumerate(initial_obstacles):
        color = colors[i % len(colors)]
        size = 0.3
        
        obs_id = p.createMultiBody(
            baseMass=0,
            basePosition=[x, y, size],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[size]*3),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[size]*3, rgbaColor=color),
            physicsClientId=client
        )
        sim_state.obstacle_ids.append(obs_id)
        sim_state.obstacles.append((x, y))
        
        p.createMultiBody(
            baseMass=0,
            basePosition=[x, y, 0.02],
            baseCollisionShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.8),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.8, rgbaColor=[1, 0, 0, 0.1]),
            physicsClientId=client
        )
    
    # AGV
    agv_id = p.createMultiBody(
        baseMass=5.0,
        basePosition=[0, -3, 0.15],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.18, 0.1]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.18, 0.1], 
                                              rgbaColor=[0.2, 0.85, 0.35, 1]),
        physicsClientId=client
    )
    
    # 轮子
    wheel_positions = [(0.2, 0.15), (0.2, -0.15), (-0.2, 0.15), (-0.2, -0.15)]
    wheel_ids = []
    for x, y in wheel_positions:
        wid = p.createMultiBody(
            baseMass=0.3,
            basePosition=[x, y - 3, 0.06],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.1]*3+[1]),
            physicsClientId=client
        )
        wheel_ids.append(wid)
    
    # 路径可视化
    for i in range(100):
        t = i * 0.3
        x = 4.0 * math.sin(0.3 * t) + 1.5 * math.sin(0.4 * t * 0.5)
        y = t * 0.4
        p.createMultiBody(
            baseMass=0,
            basePosition=[x, y, 0.01],
            baseCollisionShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0.5]*3+[0.3]),
            physicsClientId=client
        )
    
    # 起点/终点
    p.createMultiBody(
        baseMass=0,
        basePosition=[0, -3, 0.2],
        baseCollisionShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01], rgbaColor=[0, 1, 0, 0.8]),
        physicsClientId=client
    )
    p.createMultiBody(
        baseMass=0,
        basePosition=[0, 10, 0.2],
        baseCollisionShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01], rgbaColor=[1, 0, 0, 0.8]),
        physicsClientId=client
    )
    
    # 相机
    p.resetDebugVisualizerCamera(
        sim_state.camera_distance,
        sim_state.camera_yaw,
        sim_state.camera_pitch,
        sim_state.camera_target,
        physicsClientId=client
    )
    
    # 控制器
    controller = SAvoidController(sim_state)
    
    print("\n" + "=" * 60)
    print("🏁 仿真开始！")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                          操作说明                                       │
├─────────────────────────────────────────────────────────────────────┤
│  🖱️ PyBullet窗口:                                                      │
│     滚轮         - 缩放视角                                              │
│     左键拖拽     - 旋转视角                                              │
│     左键单击     - 添加障碍物                                            │
│                                                                      │
│  ⌨️ 终端窗口:                                                          │
│     SPACE        - 暂停/继续                                           │
│     PAGE_UP      - 加速                                               │
│     PAGE_DOWN    - 减速                                               │
│     ↑↓←→        - 调整视角                                            │
│     INSERT      - 俯视                                               │
│     DELETE      - 斜视                                               │
│     Ctrl+C      - 退出                                               │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    t = 0.0
    dt = 1.0 / 60.0
    last_print = 0
    last_speed = 1.0
    last_collision_time = -10
    
    # 按键状态追踪
    space_was_pressed = False
    insert_was_pressed = False
    delete_was_pressed = False
    
    try:
        while sim_state.running:
            # 获取键盘事件
            keys = p.getKeyboardEvents(client)
            
            # SPACE 暂停/继续
            space_now_pressed = 32 in keys and keys[32] == p.KEY_IS_DOWN
            if space_now_pressed and not space_was_pressed:
                sim_state.paused = not sim_state.paused
                print(f"{'⏸️ 暂停' if sim_state.paused else '▶️ 继续'}")
            space_was_pressed = space_now_pressed
            
            # 速度调节
            if 65299 in keys and keys[65299] == p.KEY_IS_DOWN:  # PAGE_UP
                sim_state.speed_multiplier = min(5.0, sim_state.speed_multiplier + 0.01)
            if 65300 in keys and keys[65300] == p.KEY_IS_DOWN:  # PAGE_DOWN
                sim_state.speed_multiplier = max(0.1, sim_state.speed_multiplier - 0.01)
            
            # 视角调整
            if 65297 in keys and keys[65297] == p.KEY_IS_DOWN:  # UP
                sim_state.camera_distance = max(3.0, sim_state.camera_distance - 0.1)
            if 65298 in keys and keys[65298] == p.KEY_IS_DOWN:  # DOWN
                sim_state.camera_distance = min(30.0, sim_state.camera_distance + 0.1)
            if 65295 in keys and keys[65295] == p.KEY_IS_DOWN:  # LEFT
                sim_state.camera_yaw = (sim_state.camera_yaw - 0.5) % 360
            if 65296 in keys and keys[65296] == p.KEY_IS_DOWN:  # RIGHT
                sim_state.camera_yaw = (sim_state.camera_yaw + 0.5) % 360
            
            # INSERT 俯视
            insert_now_pressed = 65303 in keys and keys[65303] == p.KEY_IS_DOWN
            if insert_now_pressed and not insert_was_pressed:
                sim_state.camera_pitch = -89
                sim_state.camera_yaw = 0
            insert_was_pressed = insert_now_pressed
            
            # DELETE 斜视
            delete_now_pressed = 65304 in keys and keys[65304] == p.KEY_IS_DOWN
            if delete_now_pressed and not delete_was_pressed:
                sim_state.camera_pitch = -50
                sim_state.camera_yaw = 60
            delete_was_pressed = delete_now_pressed
            
            # 获取鼠标事件
            mouse_events = p.getMouseEvents(client)
            
            for event in mouse_events:
                event_type = event[0]
                mouse_x = event[3]
                mouse_y = event[4]
                
                # 滚轮
                if event_type == 2:
                    wheel_delta = event[2]
                    sim_state.camera_distance = max(3.0, min(30.0, 
                        sim_state.camera_distance - wheel_delta * 0.3))
                
                # 左键按下
                elif event_type == 3:
                    sim_state.is_dragging = True
                    sim_state.last_mouse_x = mouse_x
                    sim_state.last_mouse_y = mouse_y
                    sim_state.mouse_click_pos = (mouse_x, mouse_y)
                
                # 左键释放 - 如果没有拖动则是点击添加障碍物
                elif event_type == 4:
                    if sim_state.is_dragging:
                        # 检测是否为点击（而非拖动）
                        dx = abs(mouse_x - sim_state.last_mouse_x)
                        dy = abs(mouse_y - sim_state.last_mouse_y)
                        if dx < 5 and dy < 5:
                            # 点击 - 添加障碍物
                            wx, wy = screen_to_world(mouse_x, mouse_y, sim_state)
                            if wx is not None and wy is not None:
                                if -5 < wx < 5 and -5 < wy < 12:
                                    add_obstacle_at_position([wx, wy], sim_state)
                    sim_state.is_dragging = False
                
                # 鼠标移动 - 拖动旋转视角
                elif event_type == 5 and sim_state.is_dragging:
                    dx = mouse_x - sim_state.last_mouse_x
                    dy = mouse_y - sim_state.last_mouse_y
                    
                    sim_state.camera_yaw = (sim_state.camera_yaw + dx * 0.5) % 360
                    sim_state.camera_pitch = max(-89, min(-10, sim_state.camera_pitch + dy * 0.5))
                    
                    sim_state.last_mouse_x = mouse_x
                    sim_state.last_mouse_y = mouse_y
            
            # 暂停控制
            if not sim_state.paused:
                p.stepSimulation(physicsClientId=client)
            
            # 更新相机
            p.resetDebugVisualizerCamera(
                sim_state.camera_distance,
                sim_state.camera_yaw,
                sim_state.camera_pitch,
                sim_state.camera_target,
                physicsClientId=client
            )
            
            # 更新控制器
            if not sim_state.paused:
                new_pos, new_angle = controller.update(t, dt, sim_state.obstacles)
                t += dt * sim_state.speed_multiplier
            else:
                new_pos, new_angle = controller.position, controller.angle
            
            # 边界限制
            new_pos[0] = max(-5.5, min(5.5, new_pos[0]))
            new_pos[1] = max(-4.0, min(11.0, new_pos[1]))
            
            # 更新AGV
            p.resetBasePositionAndOrientation(
                agv_id,
                [new_pos[0], new_pos[1], 0.15],
                p.getQuaternionFromEuler([0, 0, new_angle]),
                physicsClientId=client
            )
            
            # 更新轮子
            for i, (wx, wy) in enumerate(wheel_positions):
                wdx = new_pos[0] + wx * math.cos(new_angle) - wy * math.sin(new_angle)
                wdy = new_pos[1] + wx * math.sin(new_angle) + wy * math.cos(new_angle)
                p.resetBasePositionAndOrientation(
                    wheel_ids[i], [wdx, wdy, 0.06], [0, 0, 0, 1], physicsClientId=client
                )
            
            # 碰撞检测
            collision = controller.check_collision(sim_state.obstacles)
            if collision and (t - last_collision_time) > 1.5:
                sim_state.collision_count += 1
                last_collision_time = t
                print(f"⚠️  碰撞! #{sim_state.collision_count} @ ({new_pos[0]:+.1f}, {new_pos[1]:+.1f})")
            
            # 到达终点后重置
            if new_pos[1] > 10.5:
                print("🏆 到达终点! 重置到起点...")
                t = 0.0
                controller.position = np.array([0.0, -3.0])
                controller.velocity = np.array([0.0, 0.0])
                sim_state.collision_count = 0
            
            # 在窗口显示AGV坐标
            if sim_state.agv_text_id is not None:
                p.removeUserDebugItem(sim_state.agv_text_id)
            speed = np.linalg.norm(controller.velocity)
            agv_text = f"AGV: ({new_pos[0]:+.2f}, {new_pos[1]:+.2f}) v={speed:.2f} {'⏸️' if sim_state.paused else ''}"
            sim_state.agv_text_id = p.addUserDebugText(
                agv_text,
                textPosition=[-5.5, 10.5, 1.5],
                textColorRGB=[0, 1, 0],
                textSize=1.2,
                lifeTime=0,
                parentObjectUniqueId=-1,
                physicsClientId=client
            )
            
            # 显示速度
            if sim_state.speed_text_id is not None:
                p.removeUserDebugItem(sim_state.speed_text_id)
            speed_text = f"Speed: {sim_state.speed_multiplier:.2f}x | Obstacles: {len(sim_state.obstacles)}"
            sim_state.speed_text_id = p.addUserDebugText(
                speed_text,
                textPosition=[-5.5, 10.0, 1.5],
                textColorRGB=[1, 1, 0],
                textSize=1.0,
                lifeTime=0,
                parentObjectUniqueId=-1,
                physicsClientId=client
            )
            
            # 打印状态
            if t - last_print >= 0.5:
                avoiding = any(
                    np.linalg.norm(new_pos - np.array(obs)) < controller.warning_distance 
                    for obs in sim_state.obstacles
                )
                status = "🟡 避障" if avoiding else "🟢 直行"
                
                if abs(sim_state.speed_multiplier - last_speed) > 0.05:
                    print(f"⚡ 速度: {sim_state.speed_multiplier:.2f}x")
                    last_speed = sim_state.speed_multiplier
                
                print(f"{'⏸️ ' if sim_state.paused else ''}t={t:5.1f}s | 位置=({new_pos[0]:+.1f}, {new_pos[1]:+.1f}) | "
                      f"v={speed:.2f} | {status} | 碰撞={sim_state.collision_count} | 障碍物={len(sim_state.obstacles)}")
                last_print = t
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n收到退出信号")
    
    print(f"\n📊 最终统计:")
    print(f"  - 碰撞次数: {sim_state.collision_count}")
    print(f"  - 障碍物数量: {len(sim_state.obstacles)}")
    print(f"  - 最高速度倍数: {sim_state.speed_multiplier:.1f}x")
    print("正在关闭...")
    p.disconnect(physicsClientId=client)
    print("再见!")


if __name__ == '__main__':
    main()
