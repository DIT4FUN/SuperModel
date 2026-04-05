#!/usr/bin/env python3
"""
SuperModel AGV S形穿插避障仿真 - 增强版
=========================================
使用5.5寸轮毂电机AGV模型 + 严格S路径跟随
"""

import os
import sys
import time
import math

os.environ.setdefault('DISPLAY', ':0')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'src'))

import pybullet as p
import pybullet_data
import numpy as np


class SimState:
    """仿真全局状态"""
    def __init__(self):
        self.speed_multiplier = 3.0
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
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_click_pos = None
        self.agv_text_id = None
        self.speed_text_id = None


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
    print(f"添加障碍物 #{len(sim_state.obstacles)} @ ({position[0]:.1f}, {position[1]:.1f})")


def screen_to_world(mx, my, state):
    """屏幕坐标转世界坐标"""
    w, h = 640, 480
    yr, pr = math.radians(state.camera_yaw), math.radians(state.camera_pitch)
    
    nx = 2 * mx / w - 1
    ny = -(2 * my / h - 1)
    
    fov = math.radians(60)
    rx = nx * math.tan(fov/2)
    ry = ny * math.tan(fov/2)
    rz = -1
    
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)
    
    wx = rx * cy - rz * sy
    wy = rx * sy + rz * cy
    wz = ry * cp
    
    d = state.camera_distance
    cx = state.camera_target[0] - d * sy * cp
    cy2 = state.camera_target[1] + d * cy * cp
    cz = state.camera_target[2] - d * sp
    
    if abs(wz) > 0.001:
        t = -cz / wz
        if t > 0:
            return cx + t * wx, cy2 + t * wy
    return None, None


def main():
    print("=" * 60)
    print("SuperModel AGV S形穿插避障仿真")
    print("=" * 60)
    
    sim_state = SimState()
    
    client = p.connect(p.GUI)
    sim_state.client = client
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf', physicsClientId=client)
    
    # 使用新的AGV模型生成器
    sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'src'))
    from simulation.agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS
    
    # 创建AGV (使用M级)
    print("创建AGV (5.5寸轮毂电机 M级)...")
    cfg = GRADE_CONFIGS['M']
    urdf_path = generate_agv_urdf_detailed('M', '2轮')
    agv_id = p.loadURDF(urdf_path, basePosition=[0, -3, 0.15], physicsClientId=client)
    
    # 获取关节信息
    num_joints = p.getNumJoints(agv_id)
    print(f"AGV关节数: {num_joints}")
    joint_indices = list(range(num_joints))  # 所有关节
    
    # 初始障碍物
    initial_obstacles = [
        (-3.5, 1.0), (-3.5, 4.0), (-3.5, 7.0),
        (-1.5, 2.5), (-1.5, 5.5), (-1.5, 8.5),
        (1.5, 1.5), (1.5, 4.5), (1.5, 7.5),
        (3.5, 2.0), (3.5, 5.0), (3.5, 8.0),
    ]
    
    for i, (x, y) in enumerate(initial_obstacles):
        color = [(0.9, 0.2, 0.2, 1), (0.2, 0.8, 0.2, 1), (0.2, 0.2, 0.9, 1)][i % 3]
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
    
    # S形路径可视化
    print("绘制S形路径...")
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
    
    # 起点终点
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
    
    p.resetDebugVisualizerCamera(
        sim_state.camera_distance, sim_state.camera_yaw, sim_state.camera_pitch,
        sim_state.camera_target, physicsClientId=client
    )
    
    print("\n" + "=" * 60)
    print("仿真开始!")
    print("=" * 60)
    print("""
SPACE=暂停 PAGE_UP/DOWN=速度 鼠标滚轮=缩放
左键拖拽=旋转  INSERT=俯视  DELETE=斜视
""")
    
    # S路径参数
    path_amp_x = 4.0
    path_freq_x = 0.3
    path_freq_y = 0.8
    path_base_speed = 0.4
    
    # AGV状态
    pos_x, pos_y = 0.0, -3.0
    vel_x, vel_y = 0.0, 0.0
    theta = 0.0
    
    # 避障参数
    safe_dist = 0.8
    warning_dist = 2.5
    repulsion_gain = 2.0
    
    t = 0.0
    dt = 1.0 / 60.0
    last_print = 0
    space_was = False
    insert_was = False
    delete_was = False
    
    try:
        while sim_state.running:
            keys = p.getKeyboardEvents(client)
            
            # SPACE
            space_now = 32 in keys and keys[32] == p.KEY_IS_DOWN
            if space_now and not space_was:
                sim_state.paused = not sim_state.paused
                print(f"{'⏸️ 暂停' if sim_state.paused else '▶️ 继续'}")
            space_was = space_now
            
            # 速度
            if 65299 in keys and keys[65299] == p.KEY_IS_DOWN:
                sim_state.speed_multiplier = min(5.0, sim_state.speed_multiplier + 0.005)
            if 65300 in keys and keys[65300] == p.KEY_IS_DOWN:
                sim_state.speed_multiplier = max(0.1, sim_state.speed_multiplier - 0.005)
            
            # 视角
            if 65297 in keys and keys[65297] == p.KEY_IS_DOWN:
                sim_state.camera_distance = max(3.0, sim_state.camera_distance - 0.1)
            if 65298 in keys and keys[65298] == p.KEY_IS_DOWN:
                sim_state.camera_distance = min(30.0, sim_state.camera_distance + 0.1)
            if 65295 in keys and keys[65295] == p.KEY_IS_DOWN:
                sim_state.camera_yaw = (sim_state.camera_yaw - 0.5) % 360
            if 65296 in keys and keys[65296] == p.KEY_IS_DOWN:
                sim_state.camera_yaw = (sim_state.camera_yaw + 0.5) % 360
            
            insert_now = 65303 in keys and keys[65303] == p.KEY_IS_DOWN
            if insert_now and not insert_was:
                sim_state.camera_pitch = -89
                sim_state.camera_yaw = 0
            insert_was = insert_now
            
            delete_now = 65304 in keys and keys[65304] == p.KEY_IS_DOWN
            if delete_now and not delete_was:
                sim_state.camera_pitch = -50
                sim_state.camera_yaw = 60
            delete_was = delete_now
            
            # 鼠标事件
            for event in p.getMouseEvents(client):
                etype, _, _, mx, my = event
                if etype == 2:
                    sim_state.camera_distance = max(3.0, min(30.0, sim_state.camera_distance - event[2] * 0.3))
                elif etype == 3:
                    sim_state.is_dragging = True
                    sim_state.last_mouse_x = mx
                    sim_state.last_mouse_y = my
                elif etype == 4:
                    sim_state.is_dragging = False
                elif etype == 5 and sim_state.is_dragging:
                    dx = mx - sim_state.last_mouse_x
                    dy = my - sim_state.last_mouse_y
                    sim_state.camera_yaw = (sim_state.camera_yaw + dx * 0.3) % 360
                    sim_state.camera_pitch = max(-89, min(-10, sim_state.camera_pitch + dy * 0.3))
                    sim_state.last_mouse_x = mx
                    sim_state.last_mouse_y = my
            
            if not sim_state.paused:
                p.stepSimulation(client)
            
            p.resetDebugVisualizerCamera(
                sim_state.camera_distance, sim_state.camera_yaw, sim_state.camera_pitch,
                sim_state.camera_target, physicsClientId=client
            )
            
            if not sim_state.paused:
                speed_mult = sim_state.speed_multiplier
                
                # 计算S路径上的目标点
                path_t = t * speed_mult
                target_x = path_amp_x * math.sin(path_freq_x * path_t) + 1.5 * math.sin(path_freq_y * path_t * 0.5)
                target_y = path_t * path_base_speed
                
                # 吸引力 - 朝目标点移动
                dx = target_x - pos_x
                dy = target_y - pos_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0.01:
                    attraction_x = dx / dist * 0.8
                    attraction_y = dy / dist * 0.8
                else:
                    attraction_x, attraction_y = 0.0, 0.0
                
                # 斥力 - 避开障碍物
                repulsion_x, repulsion_y = 0.0, 0.0
                for obs in sim_state.obstacles:
                    odx = pos_x - obs[0]
                    ody = pos_y - obs[1]
                    odist = math.sqrt(odx*odx + ody*ody)
                    if odist < warning_dist and odist > 0.01:
                        force_mag = repulsion_gain * (1.0/odist - 1.0/warning_dist) / (odist + 0.1)
                        repulsion_x += force_mag * odx / odist
                        repulsion_y += force_mag * ody / odist
                
                # 合力
                force_x = attraction_x + repulsion_x
                force_y = attraction_y + repulsion_y
                
                # 速度限制
                force_mag = math.sqrt(force_x*force_x + force_y*force_y)
                max_force = 0.8 * speed_mult
                if force_mag > max_force:
                    force_x = force_x / force_mag * max_force
                    force_y = force_y / force_mag * max_force
                
                # 平滑速度
                vel_x = vel_x * 0.8 + force_x * 0.2
                vel_y = vel_y * 0.8 + force_y * 0.2
                
                # 防止停止
                vel_mag = math.sqrt(vel_x*vel_x + vel_y*vel_y)
                if vel_mag < 0.01 and dist > 0.3:
                    vel_x += (target_x - pos_x) * 0.1
                    vel_y += (target_y - pos_y) * 0.1
                
                # 更新位置
                pos_x += vel_x * dt
                pos_y += vel_y * dt
                
                # 边界
                pos_x = max(-5.5, min(5.5, pos_x))
                pos_y = max(-4.0, min(11.0, pos_y))
                
                # 计算朝向
                if vel_mag > 0.01:
                    target_theta = math.atan2(vel_y, vel_x)
                    theta_diff = target_theta - theta
                    while theta_diff > math.pi: theta_diff -= 2 * math.pi
                    while theta_diff < -math.pi: theta_diff += 2 * math.pi
                    theta += theta_diff * 0.3
                
                # 碰撞检测
                for obs in sim_state.obstacles:
                    odist = math.sqrt((pos_x-obs[0])**2 + (pos_y-obs[1])**2)
                    if odist < safe_dist * 0.7 and (t - sim_state.last_collision_time) > 1.5:
                        sim_state.collision_count += 1
                        sim_state.last_collision_time = t
                        print(f"⚠️ 碰撞! #{sim_state.collision_count} @ ({pos_x:+.1f}, {pos_y:+.1f})")
                
                # 到达终点重置
                if pos_y > 10.5:
                    print("到达终点! 重置...")
                    pos_x, pos_y = 0.0, -3.0
                    vel_x, vel_y = 0.0, 0.0
                    sim_state.collision_count = 0
                    t = 0.0
                
                t += dt * speed_mult
                
                # 更新AGV位置
                q = p.getQuaternionFromEuler([0, 0, theta])
                p.resetBasePositionAndOrientation(agv_id, [pos_x, pos_y, 0.15], q, physicsClientId=client)
                
                # 设置电机速度 - 差速驱动
                wheel_vel = 5.0 * speed_mult
                # 左轮和右轮
                for joint_idx in joint_indices[:2]:  # 只控制前两个关节(驱动轮)
                    p.setJointMotorControl2(
                        agv_id, joint_idx, p.VELOCITY_CONTROL,
                        targetVelocity=wheel_vel, force=50,
                        physicsClientId=client
                    )
            
            # 显示AGV坐标
            if sim_state.agv_text_id:
                p.removeUserDebugItem(sim_state.agv_text_id)
            speed = math.sqrt(vel_x**2 + vel_y**2)
            avoiding = any(math.sqrt((pos_x-o[0])**2+(pos_y-o[1])**2) < warning_dist for o in sim_state.obstacles)
            status = "避障" if avoiding else "直行"
            paused = "⏸️ " if sim_state.paused else ""
            agv_text = f"{paused}AGV: ({pos_x:+.1f}, {pos_y:+.1f}) {status} 速度:{sim_state.speed_multiplier:.1f}x"
            sim_state.agv_text_id = p.addUserDebugText(
                agv_text, textPosition=[-5.5, 10.5, 1.5],
                textColorRGB=[0, 1, 0], textSize=1.0, lifeTime=0,
                physicsClientId=client
            )
            
            # 打印
            if t - last_print >= 0.5:
                print(f"t={t:.1f}s | 位置=({pos_x:+.1f}, {pos_y:+.1f}) | {status} | 碰撞:{sim_state.collision_count}")
                last_print = t
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n退出")
    
    p.disconnect(client)
    print("再见!")


if __name__ == '__main__':
    main()
