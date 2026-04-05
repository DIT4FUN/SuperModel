#!/usr/bin/env python3
"""
SuperModel 仓库物流仿真
======================
仓库场景可视化 + AGV调度 + 避障测试
"""

import os
import sys
import time
import math
import random

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
        self.speed_multiplier = 3.0
        self.camera_distance = 20.0
        self.camera_yaw = 45
        self.camera_pitch = -70
        self.camera_target = [0, 10, 0]
        self.paused = False
        self.collision_count = 0
        self.completed_tasks = 0
        self.client = None
        self.running = True
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.agv_text_id = None
        self.speed_text_id = None
        self.stats_text_id = None


def create_warehouse_scene(client, layout="single_aisle"):
    """创建仓库场景"""
    objects = {}
    
    # 地面
    plane_id = p.loadURDF('plane.urdf', physicsClientId=client)
    objects['plane'] = plane_id
    
    # 导入AGV模型
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from simulation.agv_model_generator import generate_agv_urdf_detailed
    
    # 货架布局参数
    if layout == "single_aisle":
        shelf_configs = [
            # (x, y, width, depth, height, levels)
            (-6, 3, 1, 3, 3, 3),
            (-6, 8, 1, 3, 3, 3),
            (-6, 13, 1, 3, 3, 3),
            (6, 3, 1, 3, 3, 3),
            (6, 8, 1, 3, 3, 3),
            (6, 13, 1, 3, 3, 3),
        ]
        agv_starts = [(0, 0), (0, 5), (0, 10)]
        charging_station = (0, 18)
        
    elif layout == "multi_aisle":
        shelf_configs = [
            (-8, 4, 1, 2, 3, 3), (-8, 8, 1, 2, 3, 3), (-8, 12, 1, 2, 3, 3),
            (-4, 4, 1, 2, 3, 3), (-4, 8, 1, 2, 3, 3), (-4, 12, 1, 2, 3, 3),
            (4, 4, 1, 2, 3, 3), (4, 8, 1, 2, 3, 3), (4, 12, 1, 2, 3, 3),
            (8, 4, 1, 2, 3, 3), (8, 8, 1, 2, 3, 3), (8, 12, 1, 2, 3, 3),
        ]
        agv_starts = [(0, 0), (-2, 6), (2, 6)]
        charging_station = (0, 18)
        
    else:  # u_shape
        shelf_configs = [
            # 左侧
            (-8, 4, 1, 2, 3, 3), (-8, 8, 1, 2, 3, 3), (-8, 12, 1, 2, 3, 3),
            # 中间
            (-4, 12, 1, 2, 3, 3), (0, 12, 1, 2, 3, 3), (4, 12, 1, 2, 3, 3),
            # 右侧
            (8, 4, 1, 2, 3, 3), (8, 8, 1, 2, 3, 3), (8, 12, 1, 2, 3, 3),
        ]
        agv_starts = [(0, 2), (-2, 6), (2, 6)]
        charging_station = (0, 18)
    
    # 创建货架
    shelves = []
    for x, y, w, d, h, levels in shelf_configs:
        for level in range(levels):
            shelf_h = (level + 1) * (h / levels)
            shelf_id = p.createMultiBody(
                baseMass=0,
                basePosition=[x, y, shelf_h/2],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[w/2, d/2, shelf_h/2]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[w/2, d/2, shelf_h/2], 
                                                      rgbaColor=[0.6, 0.4, 0.2, 0.9]),
                physicsClientId=client
            )
            shelves.append(shelf_id)
    
    objects['shelves'] = shelves
    
    # 创建墙壁/边界
    wall_color = [0.5, 0.5, 0.5, 0.5]
    
    # 上墙
    p.createMultiBody(
        baseMass=0,
        basePosition=[0, 20, 1.5],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[12, 0.2, 1.5]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[12, 0.2, 1.5], rgbaColor=wall_color),
        physicsClientId=client
    )
    # 下墙
    p.createMultiBody(
        baseMass=0,
        basePosition=[0, -2, 1.5],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[12, 0.2, 1.5]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[12, 0.2, 1.5], rgbaColor=wall_color),
        physicsClientId=client
    )
    # 左墙
    p.createMultiBody(
        baseMass=0,
        basePosition=[-11, 9, 1.5],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 12, 1.5]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 12, 1.5], rgbaColor=wall_color),
        physicsClientId=client
    )
    # 右墙
    p.createMultiBody(
        baseMass=0,
        basePosition=[11, 9, 1.5],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 12, 1.5]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 12, 1.5], rgbaColor=wall_color),
        physicsClientId=client
    )
    
    # 充电站
    charging_id = p.createMultiBody(
        baseMass=0,
        basePosition=[charging_station[0], charging_station[1], 0.3],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.3]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.3], rgbaColor=[0.1, 0.8, 0.2, 0.8]),
        physicsClientId=client
    )
    objects['charging_station'] = charging_id
    
    # 创建AGV (使用M级5.5寸轮毂电机)
    agvs = []
    for i, (sx, sy) in enumerate(agv_starts):
        urdf_path = generate_agv_urdf_detailed('M', '2轮')
        agv_id = p.loadURDF(urdf_path, basePosition=[sx, sy, 0.15], physicsClientId=client)
        agvs.append({
            'id': agv_id,
            'x': sx,
            'y': sy,
            'theta': 0,
            'target_x': sx,
            'target_y': sy,
            'status': 'idle',
            'task': None,
            'vx': 0,
            'vy': 0
        })
    
    objects['agvs'] = agvs
    
    # 货物颜色
    item_colors = [
        [0.9, 0.2, 0.2, 1],  # 红
        [0.2, 0.8, 0.2, 1],  # 绿
        [0.2, 0.2, 0.9, 1],  # 蓝
        [0.9, 0.6, 0.1, 1],  # 橙
    ]
    
    # 任务点标记
    task_markers = []
    for x, y, w, d, h, levels in shelf_configs:
        for level in range(levels):
            marker_id = p.createMultiBody(
                baseMass=0,
                basePosition=[x, y, (level + 1) * (h / levels) + 0.1],
                baseCollisionShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15], 
                                                        rgbaColor=item_colors[level]),
                physicsClientId=client
            )
            task_markers.append(marker_id)
    
    objects['task_markers'] = task_markers
    
    return objects


def update_agv(agv, dt, speed_mult):
    """更新单个AGV"""
    # 目标方向
    dx = agv['target_x'] - agv['x']
    dy = agv['target_y'] - agv['y']
    dist = math.sqrt(dx*dx + dy*dy)
    
    if dist < 0.1:
        agv['vx'] *= 0.9
        agv['vy'] *= 0.9
        if agv['status'] == 'moving':
            agv['status'] = 'idle'
    else:
        # 势场法避障
        speed = 0.8 * speed_mult
        target_vx = (dx / dist) * speed
        target_vy = (dy / dist) * speed
        
        agv['vx'] = agv['vx'] * 0.8 + target_vx * 0.2
        agv['vy'] = agv['vy'] * 0.8 + target_vy * 0.2
        
        agv['theta'] = math.atan2(agv['vy'], agv['vx'])
        agv['status'] = 'moving'
    
    # 更新位置
    agv['x'] += agv['vx'] * dt
    agv['y'] += agv['vy'] * dt
    
    return dist


def assign_new_task(agv, shelves, completed_tasks):
    """分配新任务"""
    if agv['status'] == 'idle' and agv['task'] is None:
        # 随机选择一个货架位置
        shelf = random.choice(shelves)
        agv['target_x'] = shelf[0]
        agv['target_y'] = shelf[1]
        agv['task'] = completed_tasks + 1
        agv['status'] = 'moving'
        return True
    return False


def screen_to_world(mouse_x, mouse_y, sim_state):
    """屏幕坐标转世界坐标"""
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


def add_obstacle_at_position(position, sim_state, scene):
    """添加动态障碍物"""
    obs_id = p.createMultiBody(
        baseMass=0,
        basePosition=[position[0], position[1], 0.3],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.3),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.9, 0.3, 0.9, 0.9]),
        physicsClientId=sim_state.client
    )
    scene['obstacles'].append({
        'id': obs_id,
        'x': position[0],
        'y': position[1],
        'vx': random.uniform(-0.3, 0.3),
        'vy': random.uniform(-0.3, 0.3)
    })
    print(f"➕ 动态障碍物 #{len(scene['obstacles'])} @ ({position[0]:.1f}, {position[1]:.1f})")


def main():
    print("=" * 60)
    print("🏭 SuperModel 仓库物流仿真")
    print("=" * 60)
    
    sim_state = SimState()
    
    # 可选择布局: single_aisle, multi_aisle, u_shape
    layout = "single_aisle"
    
    client = p.connect(p.GUI)
    sim_state.client = client
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 创建仓库场景
    print(f"创建仓库场景: {layout}")
    scene = create_warehouse_scene(client, layout)
    scene['obstacles'] = []
    
    # 相机
    p.resetDebugVisualizerCamera(
        sim_state.camera_distance,
        sim_state.camera_yaw,
        sim_state.camera_pitch,
        sim_state.camera_target,
        physicsClientId=client
    )
    
    print("\n" + "=" * 60)
    print("🏭 仓库物流仿真开始！")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                          操作说明                                       │
├─────────────────────────────────────────────────────────────────────┤
│  🖱️ PyBullet窗口:                                                      │
│     滚轮         - 缩放视角                                              │
│     左键拖拽     - 旋转视角                                              │
│     左键单击     - 添加动态障碍物                                          │
│                                                                      │
│  ⌨️ 终端窗口:                                                          │
│     SPACE        - 暂停/继续                                           │
│     PAGE_UP      - 加速                                               │
│     PAGE_DOWN    - 减速                                               │
│     ↑↓←→        - 调整视角                                            │
│     Ctrl+C      - 退出                                               │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    dt = 1.0 / 60.0
    t = 0.0
    last_print = 0
    space_was_pressed = False
    
    try:
        while sim_state.running:
            # 键盘事件
            keys = p.getKeyboardEvents(client)
            
            # SPACE暂停
            space_now = 32 in keys and keys[32] == p.KEY_IS_DOWN
            if space_now and not space_was_pressed:
                sim_state.paused = not sim_state.paused
                print(f"{'⏸️ 暂停' if sim_state.paused else '▶️ 继续'}")
            space_was_pressed = space_now
            
            # 速度调节
            if 65299 in keys and keys[65299] == p.KEY_IS_DOWN:
                sim_state.speed_multiplier = min(3.0, sim_state.speed_multiplier + 0.005)
            if 65300 in keys and keys[65300] == p.KEY_IS_DOWN:
                sim_state.speed_multiplier = max(0.1, sim_state.speed_multiplier - 0.005)
            
            # 视角调整
            if 65297 in keys and keys[65297] == p.KEY_IS_DOWN:
                sim_state.camera_distance = max(10, sim_state.camera_distance - 0.1)
            if 65298 in keys and keys[65298] == p.KEY_IS_DOWN:
                sim_state.camera_distance = min(40, sim_state.camera_distance + 0.1)
            if 65295 in keys and keys[65295] == p.KEY_IS_DOWN:
                sim_state.camera_yaw = (sim_state.camera_yaw - 0.3) % 360
            if 65296 in keys and keys[65296] == p.KEY_IS_DOWN:
                sim_state.camera_yaw = (sim_state.camera_yaw + 0.3) % 360
            
            # INSERT俯视
            if 65303 in keys and keys[65303] == p.KEY_IS_DOWN:
                sim_state.camera_pitch = -89
                sim_state.camera_yaw = 0
            
            # 鼠标事件
            mouse_events = p.getMouseEvents(client)
            for event in mouse_events:
                event_type = event[0]
                mouse_x = event[3]
                mouse_y = event[4]
                
                if event_type == 2:  # 滚轮
                    sim_state.camera_distance = max(10, min(40, 
                        sim_state.camera_distance - event[2] * 0.3))
                
                elif event_type == 3:  # 左键按下
                    sim_state.is_dragging = True
                    sim_state.last_mouse_x = mouse_x
                    sim_state.last_mouse_y = mouse_y
                
                elif event_type == 4:  # 左键释放
                    if not sim_state.is_dragging:
                        pass
                    sim_state.is_dragging = False
                
                elif event_type == 5 and sim_state.is_dragging:  # 拖动
                    dx = mouse_x - sim_state.last_mouse_x
                    dy = mouse_y - sim_state.last_mouse_y
                    sim_state.camera_yaw = (sim_state.camera_yaw + dx * 0.3) % 360
                    sim_state.camera_pitch = max(-89, min(-10, sim_state.camera_pitch + dy * 0.3))
                    sim_state.last_mouse_x = mouse_x
                    sim_state.last_mouse_y = mouse_y
            
            # 暂停
            if not sim_state.paused:
                p.stepSimulation(client)
            
            # 更新相机
            p.resetDebugVisualizerCamera(
                sim_state.camera_distance,
                sim_state.camera_yaw,
                sim_state.camera_pitch,
                sim_state.camera_target,
                physicsClientId=client
            )
            
            # 更新AGV
            if not sim_state.paused:
                shelf_positions = []
                for x, y, w, d, h, levels in [
                    (-6, 3, 1, 3, 3, 3), (-6, 8, 1, 3, 3, 3), (-6, 13, 1, 3, 3, 3),
                    (6, 3, 1, 3, 3, 3), (6, 8, 1, 3, 3, 3), (6, 13, 1, 3, 3, 3),
                ]:
                    for level in range(levels):
                        shelf_positions.append((x, y))
                
                for agv in scene['agvs']:
                    dist = update_agv(agv, dt, sim_state.speed_multiplier)
                    
                    # 到达目标，分配新任务
                    if dist < 0.1 and agv['task'] is not None:
                        agv['task'] = None
                        sim_state.completed_tasks += 1
                        print(f"✅ 任务完成! 总计: {sim_state.completed_tasks}")
                    
                    # 分配新任务
                    if agv['status'] == 'idle' and agv['task'] is None:
                        assign_new_task(agv, shelf_positions, sim_state.completed_tasks)
                    
                    # 更新位置
                    q = p.getQuaternionFromEuler([0, 0, agv['theta']])
                    p.resetBasePositionAndOrientation(
                        agv['id'], [agv['x'], agv['y'], 0.15], q, physicsClientId=client
                    )
                
                # 更新动态障碍物
                for obs in scene['obstacles']:
                    obs['x'] += obs['vx'] * dt * sim_state.speed_multiplier
                    obs['y'] += obs['vy'] * dt * sim_state.speed_multiplier
                    
                    # 边界反弹
                    if obs['x'] < -10 or obs['x'] > 10:
                        obs['vx'] = -obs['vx']
                    if obs['y'] < -1 or obs['y'] > 19:
                        obs['vy'] = -obs['vy']
                    
                    p.resetBasePositionAndOrientation(
                        obs['id'], [obs['x'], obs['y'], 0.3], [0, 0, 0, 1], physicsClientId=client
                    )
                
                t += dt * sim_state.speed_multiplier
            
            # 显示状态
            if sim_state.agv_text_id is not None:
                p.removeUserDebugItem(sim_state.agv_text_id)
            
            avg_x = sum(a['x'] for a in scene['agvs']) / len(scene['agvs'])
            avg_y = sum(a['y'] for a in scene['agvs']) / len(scene['agvs'])
            
            status = "⏸️ " if sim_state.paused else ""
            agv_text = f"{status}AGV: ({avg_x:+.1f}, {avg_y:+.1f}) | 完成: {sim_state.completed_tasks} | 障碍物: {len(scene['obstacles'])}"
            sim_state.agv_text_id = p.addUserDebugText(
                agv_text,
                textPosition=[-10, 19, 2],
                textColorRGB=[0, 1, 0],
                textSize=1.0,
                lifeTime=0,
                physicsClientId=client
            )
            
            # 打印
            if t - last_print >= 0.5:
                statuses = [a['status'] for a in scene['agvs']]
                print(f"{status}t={t:.1f}s | AGV状态: {statuses} | 完成: {sim_state.completed_tasks} | 速度: {sim_state.speed_multiplier:.1f}x")
                last_print = t
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n收到退出信号")
    
    print(f"\n📊 统计:")
    print(f"  - 完成任务: {sim_state.completed_tasks}")
    print(f"  - 障碍物数量: {len(scene['obstacles'])}")
    print("正在关闭...")
    p.disconnect(client)
    print("再见!")


if __name__ == '__main__':
    main()
