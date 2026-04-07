#!/usr/bin/env python3
"""
SuperModel 传感器融合控制仿真
============================
展示: IMU + 力觉 + 触觉 → 融合 → 运动控制 完整闭环

功能:
- 三种AGV等级 (S/M/L) 可选
- 实时传感器数据采集与显示
- 互补滤波/EKF姿态估计
- 力控导纳控制可视化
- 触觉滑移检测与响应
- 轨迹跟踪与碰撞回避

运行:
    cd ~/.openclaw/workspace/projects/SuperModel/sim_demos
    python run_sensor_fusion.py
"""

import os
import sys
import time
import math
import numpy as np

os.environ.setdefault('DISPLAY', ':0')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import pybullet as p
import pybullet_data


class SensorFusionSimulator:
    """
    传感器融合控制仿真器
    
    集成:
    - IMU 姿态估计 (互补滤波)
    - 力觉导纳控制 (ForceTorqueSensor)
    - 触觉滑移检测 (TactileArray)
    - AGV 运动控制 (AGVMotionController)
    """

    GRADE_CONFIGS = {
        'S': {
            'color': [0.2, 0.8, 0.3, 1],
            'size': (0.4, 0.3, 0.12),
            'mass': 15.0,
            'max_speed': 0.5,
            'imu_noise': 0.01,
            'force_range': 100,
        },
        'M': {
            'color': [0.3, 0.6, 0.9, 1],
            'size': (0.6, 0.4, 0.15),
            'mass': 35.0,
            'max_speed': 1.5,
            'imu_noise': 0.005,
            'force_range': 200,
        },
        'L': {
            'color': [0.9, 0.6, 0.2, 1],
            'size': (0.8, 0.6, 0.2),
            'mass': 80.0,
            'max_speed': 2.0,
            'imu_noise': 0.002,
            'force_range': 500,
        },
    }

    def __init__(self, grade: str = 'M', use_gui: bool = True):
        self.grade = grade
        self.cfg = self.GRADE_CONFIGS[grade]
        self.use_gui = use_gui
        self.dt = 1.0 / 240.0

        # PyBullet 客户端
        self.client = None
        self.agv_id = None

        # 传感器仿真
        self.virtual_imu = None
        self.virtual_force = None
        self.virtual_tactile = None

        # 控制器
        self.pose_estimator = None
        self.force_controller = None
        self.tactile_controller = None
        self.agv_controller = None

        # 状态
        self.time = 0.0
        self.step_count = 0
        self.running = True

        # 数据记录
        self.pose_history = []
        self.force_history = []
        self.tactile_history = []

    def setup(self):
        """初始化仿真环境"""
        if self.use_gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # 地面
        p.loadURDF('plane.urdf', physicsClientId=self.client)

        # 创建 AGV
        self._create_agv()

        # 初始化传感器
        self._setup_sensors()

        # 初始化控制器
        self._setup_controllers()

        # 添加障碍物
        self._add_obstacles()

        print(f"[SensorFusionSim] Grade={self.grade}, "
              f"Size={self.cfg['size']}, MaxSpeed={self.cfg['max_speed']}m/s")
        print("[SensorFusionSim] Setup complete")

    def _create_agv(self):
        """创建AGV模型"""
        sx, sy, sz = self.cfg['size']
        half_extents = [sx/2, sy/2, sz/2]

        # 车身
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=self.cfg['color']
        )

        self.agv_id = p.createMultiBody(
            baseMass=self.cfg['mass'],
            basePosition=[0, 0, sz/2 + 0.01],
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            physicsClientId=self.client
        )

        # 轮子
        wheel_radius = 0.05
        wheel_half = 0.02
        wheel_color = [0.15, 0.15, 0.15, 1]

        wheel_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_half,
            rgbaColor=wheel_color
        )

        wheel_positions = [
            (sx/2 - 0.05, sy/2 - 0.05),
            (sx/2 - 0.05, -sy/2 + 0.05),
        ]

        for wx, wy in wheel_positions:
            p.createMultiBody(
                baseMass=0.5,
                basePosition=[wx, wy, wheel_radius],
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_half
                ),
                baseVisualShapeIndex=wheel_visual,
                baseParentObjectUniqueId=self.agv_id,
                baseJointType=p.JOINT_REVOLUTE,
                jointLimits=[-1e10, 1e10],
                physicsClientId=self.client
            )

    def _setup_sensors(self):
        """初始化传感器"""
        from src.sensors.imu import VirtualIMUSensor, PoseEstimator
        from src.sensors.force import VirtualForceSensor
        from src.sensors.tactile import VirtualTactileSensor

        # IMU
        self.virtual_imu = VirtualIMUSensor(
            sensor_id=f"imu_{self.grade}",
            accel_noise=self.cfg['imu_noise'],
            gyro_noise=self.cfg['imu_noise'] * 0.1,
        )
        self.virtual_imu.open()

        # 姿态估计器
        self.pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=240)
        self.pose_estimator.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # 力觉
        self.virtual_force = VirtualForceSensor(
            sensor_id=f"force_{self.grade}",
            noise_level=0.02,
        )
        self.virtual_force.open()

        # 触觉
        self.virtual_tactile = VirtualTactileSensor(
            array_size=(16, 16),
            sensor_id=f"tactile_{self.grade}",
        )
        self.virtual_tactile.open()

    def _setup_controllers(self):
        """初始化控制器"""
        from src.control.force_control import ForceController, ForceControlParams
        from src.control.tactile_control import TactileServoController, TactileServoParams
        from src.control.agv import AGVMotionController

        # AGV 运动控制器
        self.agv_controller = AGVMotionController(grade=self.grade)

        # 力控
        force_params = ForceControlParams.from_grade(self.grade)
        self.force_controller = ForceController(self.virtual_force, force_params)

        # 触觉伺服
        tactile_params = TactileServoParams.from_grade(self.grade)
        self.tactile_controller = TactileServoController(self.virtual_tactile, tactile_params)

    def _add_obstacles(self):
        """添加障碍物"""
        obstacle_positions = [
            (2.0, 0.0, 0.15),
            (4.0, 1.5, 0.15),
            (4.0, -1.5, 0.15),
            (6.0, 0.0, 0.15),
            (8.0, 1.0, 0.15),
            (8.0, -1.0, 0.15),
        ]

        for pos in obstacle_positions:
            size = 0.2 + np.random.rand() * 0.2
            p.createMultiBody(
                baseMass=0,
                basePosition=pos,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[size]*3
                ),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[size]*3,
                    rgbaColor=[0.8, 0.3, 0.3, 1]
                ),
                physicsClientId=self.client
            )

    def get_agv_state(self):
        """获取AGV状态"""
        pos, orn = p.getBasePositionAndOrientation(self.agv_id, physicsClientId=self.client)
        vel = p.getBaseVelocity(self.agv_id, physicsClientId=self.client)

        # 欧拉角
        euler = p.getEulerFromQuaternion(orn)

        return {
            'position': np.array(pos),
            'orientation': np.array(orn),
            'euler': np.array(euler),
            'linear_velocity': np.array(vel[0]),
            'angular_velocity': np.array(vel[1]),
        }

    def update_sensors(self, agv_state):
        """更新传感器数据"""
        # IMU 数据
        roll, pitch, yaw = agv_state['euler']
        imu_frame = self.virtual_imu.simulate_static((roll, pitch, yaw))

        # 姿态估计
        pose = self.pose_estimator.update(
            imu_frame.accel, imu_frame.gyro, imu_frame.mag
        )

        # 力觉数据 (模拟接触力)
        base_vel = agv_state['linear_velocity']
        speed = np.linalg.norm(base_vel[:2])

        if speed > 0.1:
            # 运动时模拟空气阻力/轮毂力
            contact_force = self.virtual_force.simulate_contact(
                force=(0.0, 0.0, -speed * 5.0),
                torque=(0.0, 0.0, 0.0),
                add_noise=True
            )
        else:
            contact_force = self.virtual_force.simulate_contact(
                force=(0.0, 0.0, -self.cfg['mass'] * 9.81),
                torque=(0.0, 0.0, 0.0),
                add_noise=False
            )

        # 触觉数据 (模拟车轮与地面接触)
        if speed > 0.01:
            contact_pos = (0.5 + 0.1 * np.sin(self.time * 2), 0.5 + 0.1 * np.cos(self.time * 2))
            tactile_frame = self.virtual_tactile.simulate_contact(
                contact_pos=contact_pos,
                contact_radius=0.2,
                contact_force=speed * 10.0,
                noise_level=0.02
            )
        else:
            tactile_frame = self.virtual_tactile.simulate_contact(
                contact_pos=(0.5, 0.5),
                contact_radius=0.1,
                contact_force=5.0,
                noise_level=0.01
            )

        return {
            'imu': imu_frame,
            'pose': pose,
            'force': contact_force,
            'tactile': tactile_frame,
        }

    def step_control(self, sensor_data):
        """执行控制步骤"""
        # 姿态估计
        euler = sensor_data['pose'].to_euler()
        pose_error = np.array([0.0 - euler[0], 0.0 - euler[1], 0.0 - euler[2]])

        # 力控导纳
        desired_force = np.array([0.0, 0.0, -self.cfg['mass'] * 9.81 * 0.1])
        force_adj = self.force_controller.compute_admittance(
            desired_force, sensor_data['force'], dt=self.dt
        )

        # 触觉伺服
        tactile_sig = self.tactile_controller.compute_control_signal(
            target_force=5.0, current_frame=sensor_data['tactile']
        )

        # AGV 速度控制
        target_speed = self.cfg['max_speed'] * 0.5
        from src.control.agv import AGVTwist
        twist = AGVTwist(vx=target_speed, vy=0.0, omega=0.0)
        self.agv_controller.set_target_twist(twist)
        wheel_vel = self.agv_controller.step(dt=self.dt)

        return {
            'pose_error': pose_error,
            'force_adjustment': force_adj,
            'tactile_signal': tactile_sig,
            'wheel_velocity': wheel_vel,
        }

    def apply_action(self, wheel_vel):
        """应用动作到AGV"""
        # 简化: 直接设置角速度
        # 差速驱动: left_vel, right_vel
        left_vel = wheel_vel[0] if len(wheel_vel) >= 1 else 0.0
        right_vel = wheel_vel[1] if len(wheel_vel) >= 2 else left_vel

        # 应用到关节
        for _ in range(2):
            p.setJointMotorControl2(
                bodyUniqueId=self.agv_id,
                jointIndex=_ + 1,  # 跳过base (index 0)
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=left_vel if _ == 0 else right_vel,
                force=50,
                physicsClientId=self.client
            )

    def render_overlay(self, sensor_data, control_data):
        """渲染信息叠加"""
        if not self.use_gui:
            return

        info_lines = [
            f"=== SuperModel 传感器融合仿真 ===",
            f"等级: {self.grade} | 速度: {self.cfg['max_speed']}m/s",
            f"",
            f"--- IMU 姿态 ---",
            f"Roll:  {sensor_data['pose'].to_euler()[0]*180/math.pi:7.2f}°",
            f"Pitch: {sensor_data['pose'].to_euler()[1]*180/math.pi:7.2f}°",
            f"Yaw:   {sensor_data['pose'].to_euler()[2]*180/math.pi:7.2f}°",
            f"",
            f"--- 力觉 ---",
            f"Fz: {sensor_data['force'].force[2]:7.2f} N",
            f"Tx: {sensor_data['force'].torque[0]:7.3f} N·m",
            f"",
            f"--- 触觉 ---",
            f"Peak:  {np.max(sensor_data['tactile'].pressure_map):7.3f}",
            f"Mean:  {np.mean(sensor_data['tactile'].pressure_map):7.3f}",
            f"",
            f"--- 控制 ---",
            f"VL: {control_data['wheel_velocity'][0]:7.3f} rad/s",
            f"VR: {control_data['wheel_velocity'][1]:7.3f} rad/s",
            f"",
            f"时间: {self.time:7.2f}s | 步数: {self.step_count}",
        ]

        for i, line in enumerate(info_lines):
            p.addUserDebugText(
                line,
                textPosition=[0.01, 0, 0],
                textColorRGB=[0.2, 1.0, 0.2],
                textSize=1.0,
                lifeTime=0,
                parentObjectUniqueId=self.agv_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=1000 + i,
                physicsClientId=self.client
            )

    def run(self, max_steps: int = 10000, record_interval: int = 10):
        """运行仿真"""
        print(f"[SensorFusionSim] Starting simulation (max_steps={max_steps})...")

        while self.running and self.step_count < max_steps:
            # 步进物理
            p.stepSimulation(physicsClientId=self.client)

            # 获取AGV状态
            agv_state = self.get_agv_state()

            # 更新传感器
            sensor_data = self.update_sensors(agv_state)

            # 执行控制
            control_data = self.step_control(sensor_data)

            # 应用动作
            self.apply_action(control_data['wheel_velocity'])

            # 记录数据
            if self.step_count % record_interval == 0:
                self.pose_history.append(sensor_data['pose'].to_euler().copy())
                self.force_history.append(sensor_data['force'].to_vector().copy())
                self.tactile_history.append({
                    'peak': float(np.max(sensor_data['tactile'].pressure_map)),
                    'mean': float(np.mean(sensor_data['tactile'].pressure_map)),
                })

            # 渲染信息
            if self.step_count % 5 == 0:
                self.render_overlay(sensor_data, control_data)

            self.time += self.dt
            self.step_count += 1

            # 速度控制
            if self.use_gui:
                time.sleep(max(self.dt / 3, 0.001))

        print(f"[SensorFusionSim] Finished: {self.step_count} steps, {self.time:.2f}s")

    def get_summary(self):
        """获取仿真结果摘要"""
        if not self.pose_history:
            return {}

        pose_arr = np.array(self.pose_history)
        force_arr = np.array(self.force_history)

        return {
            'steps': self.step_count,
            'duration': self.time,
            'pose_final': pose_arr[-1] if len(pose_arr) > 0 else np.zeros(3),
            'pose_std': np.std(pose_arr, axis=0) if len(pose_arr) > 0 else np.zeros(3),
            'force_z_mean': np.mean(force_arr[:, 2]) if len(force_arr) > 0 else 0.0,
            'tactile_peaks': [t['peak'] for t in self.tactile_history[-10:]],
        }

    def cleanup(self):
        """清理资源"""
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
        if self.virtual_imu:
            self.virtual_imu.close()
        if self.virtual_force:
            self.virtual_force.close()
        if self.virtual_tactile:
            self.virtual_tactile.close()
        print("[SensorFusionSim] Cleanup complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SuperModel 传感器融合仿真')
    parser.add_argument('--grade', '-g', choices=['S', 'M', 'L'], default='M',
                        help='AGV等级 (S/M/L)')
    parser.add_argument('--steps', '-s', type=int, default=5000,
                        help='最大仿真步数')
    parser.add_argument('--headless', action='store_true',
                        help='无GUI模式')
    args = parser.parse_args()

    sim = SensorFusionSimulator(
        grade=args.grade,
        use_gui=not args.headless
    )

    try:
        sim.setup()
        sim.run(max_steps=args.steps)
        summary = sim.get_summary()

        print("\n" + "="*50)
        print("仿真结果摘要")
        print("="*50)
        print(f"等级: {args.grade}")
        print(f"步数: {summary.get('steps', 0)}")
        print(f"时长: {summary.get('duration', 0):.2f}s")
        print(f"姿态稳定度(Std): {summary.get('pose_std', np.zeros(3))} rad")
        print(f"触觉峰值(末10): {summary.get('tactile_peaks', [])}")

    finally:
        sim.cleanup()


if __name__ == '__main__':
    main()
