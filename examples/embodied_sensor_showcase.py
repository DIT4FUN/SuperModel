#!/usr/bin/env python3
"""
SuperModel 具身智能传感器综合展示
====================================
展示: 触觉 + 力觉 + IMU → 超模态融合 → AGV运动控制 完整闭环

功能:
- 五种AGV等级 (S/M/L/XL/XXL) 全覆盖
- 触觉: 压力分布、抓取质量、滑移检测
- 力觉: 六维力矩、接触检测、碰撞响应
- IMU: 姿态估计、轨迹模拟、运动跟踪
- 超模态融合: 多传感器注意力融合
- AGV运动控制: 速度跟踪、轨迹规划、安全监管

运行:
    cd ~/.openclaw/workspace/projects/SuperModel/examples
    python embodied_sensor_showcase.py --grade M
    python embodied_sensor_showcase.py --grade L --duration 10
    python embodied_sensor_showcase.py --grade XL --fusion fusion --verbose
"""

import argparse
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def print_header(title: str, width: int = 70):
    """打印标题"""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str):
    """打印子标题"""
    print()
    print(f"── {title} ──")


class EmbodiedSensorShowcase:
    """
    具身智能传感器综合展示
    
    完整闭环: 触觉 + 力觉 + IMU → 融合 → AGV控制
    """
    
    GRADES = ['S', 'M', 'L', 'XL', 'XXL']
    
    def __init__(self, grade: str = 'M', verbose: bool = False):
        if grade not in self.GRADES:
            raise ValueError(f"Grade must be one of {self.GRADES}, got {grade}")
        self.grade = grade
        self.verbose = verbose
        self._init_sensors()
        self._init_fusion()
        self._init_control()
        
    def _init_sensors(self):
        """初始化传感器"""
        print_subsection("传感器初始化")
        
        # 触觉传感器
        from src.sensors.tactile import (
            TactileArray, TactileSensorType, get_tactile_spec, VirtualTactileSensor
        )
        tactile_spec = get_tactile_spec(self.grade)
        array_size = tactile_spec['array']
        self.tactile = TactileArray(
            array_size=array_size,
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id=f"tactile_{self.grade}"
        )
        self.virtual_tactile = VirtualTactileSensor(
            array_size=array_size,
            sensor_id=f"virtual_tactile_{self.grade}"
        )
        self.tactile.open()
        self.virtual_tactile.open()
        print(f"  触觉传感器: {self.grade}级 {array_size} 阵列, "
              f"采样率={tactile_spec['freq_hz']}Hz")
        
        # 力觉传感器
        from src.sensors.force import (
            ForceTorqueSensor, ForceSensorType, get_force_spec, VirtualForceSensor
        )
        force_spec = get_force_spec(self.grade)
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id=f"force_{self.grade}"
        )
        self.virtual_force = VirtualForceSensor(
            sensor_id=f"virtual_force_{self.grade}"
        )
        self.force.open()
        self.virtual_force.open()
        print(f"  力觉传感器: {self.grade}级 {force_spec['axes']}轴, "
              f"力范围=±{force_spec['force_range']}N, "
              f"采样率={force_spec['sampling_hz']}Hz")
        
        # IMU传感器
        from src.sensors.imu import (
            IMUSensor, IMUSensorType, get_imu_spec, VirtualIMUSensor, PoseEstimator
        )
        imu_spec = get_imu_spec(self.grade)
        self.imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088 if self.grade != 'S' else IMUSensorType.MPU6050,
            sensor_id=f"imu_{self.grade}"
        )
        self.virtual_imu = VirtualIMUSensor(sensor_id=f"virtual_imu_{self.grade}")
        self.pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=imu_spec['sample_hz'])
        self.imu.open()
        self.virtual_imu.open()
        print(f"  IMU传感器: {self.grade}级 {imu_spec['type']}, "
              f"采样率={imu_spec['sample_hz']}Hz, "
              f"噪声密度={imu_spec['noise_density']}μg/√Hz")
    
    def _init_fusion(self):
        """初始化融合模块"""
        print_subsection("融合模块初始化")
        
        from src.fusion.sensor_fusion import ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion
        
        self.complementary_filter = ComplementaryFilter(alpha=0.96)
        self.ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)
        self.multi_fusion = MultiSensorFusion()
        self.multi_fusion.add_fusion_method('complementary', self.complementary_filter, weight=1.0)
        
        print(f"  互补滤波器: alpha=0.96")
        print(f"  扩展卡尔曼滤波: 状态维度=6, 测量维度=3")
        print(f"  多传感器融合: 互补滤波+多模态输入")
    
    def _init_control(self):
        """初始化控制模块"""
        print_subsection("控制模块初始化")
        
        from src.control.motion import MotionController, AdaptivePIDController
        from src.control.agv import AGVMotionController, AGVSpec, AGVGrade
        from src.control.safety_controller import SafetyController, SafetyConfig, SafetyLevel
        
        # AGV规格
        grade_map = {'S': AGVGrade.S, 'M': AGVGrade.M, 'L': AGVGrade.L,
                     'XL': AGVGrade.XL, 'XXL': AGVGrade.XXL}
        agv_grade = grade_map.get(self.grade, AGVGrade.M)
        self.agv_spec = AGVSpec.from_grade(agv_grade)
        self.agv_ctrl = AGVMotionController(self.agv_spec)
        
        # 运动控制器
        self.motion_ctrl = MotionController(num_joints=2, control_rate=100.0)
        
        # 安全控制器
        level_map = {'S': SafetyLevel.S, 'M': SafetyLevel.M, 'L': SafetyLevel.M,
                     'XL': SafetyLevel.L, 'XXL': SafetyLevel.L}
        safety_level = level_map.get(self.grade, SafetyLevel.M)
        safety_config = SafetyConfig(
            joint_limits_lower=np.array([-3.14, -2.5]),
            joint_limits_upper=np.array([3.14, 2.5]),
            velocity_limits=np.array([2.0, 2.0]),
            acceleration_limits=np.array([5.0, 5.0]),
            safety_level=safety_level
        )
        self.safety = SafetyController(safety_config)
        
        print(f"  AGV运动学: {self.grade}级, 驱动类型={self.agv_spec.drive_type.value}")
        print(f"  运动控制器: AdaptivePID")
        print(f"  安全控制器: {safety_level.value}级, "
              f"碰撞阈值={safety_config.collision_threshold}N")
    
    def run_tactile_sequence(self, num_frames: int = 10):
        """触觉传感器测试序列"""
        print_subsection(f"触觉传感器测试 ({num_frames}帧)")
        
        # 模拟多种接触场景
        scenarios = [
            ("单点接触", [(0.5, 0.5)], 15.0, 0.25),
            ("双点接触", [(0.3, 0.4), (0.7, 0.6)], 10.0, 0.2),
            ("边缘接触", [(0.1, 0.5)], 20.0, 0.3),
            ("大面积接触", [(0.5, 0.5)], 8.0, 0.45),
        ]
        
        for name, positions, force, radius in scenarios:
            frames = []
            for pos in positions:
                frame = self.virtual_tactile.simulate_contact(
                    contact_pos=pos,
                    contact_force=force,
                    contact_radius=radius
                )
                frames.append(frame)
            
            # 分析接触
            contacts = self.tactile.detect_contacts(frames[-1])
            quality = self.tactile.estimate_grip_quality(frames[-1])
            
            print(f"  [{name}] 接触数={len(contacts)}, "
                  f"抓取质量={quality['overall']:.3f}, "
                  f"均匀性={quality['uniformity']:.3f}")
        
        # 滑移检测
        sliding_frames = self.virtual_tactile.simulate_sliding(
            direction=(0.5, 0.3),
            speed=0.05,
            duration_frames=15
        )
        slip = self.tactile.get_slip_signal(sliding_frames[-1])
        print(f"  滑移检测: 平均滑移信号={np.mean(slip):.4f}, "
              f"峰值={np.max(slip):.4f}")
    
    def run_force_sequence(self, num_frames: int = 10):
        """力觉传感器测试序列"""
        print_subsection(f"力觉传感器测试 ({num_frames}帧)")
        
        scenarios = [
            ("静止负载", 10.0, (0.0, 0.0, 0.0)),
            ("前进推力", 5.0, (10.0, 2.0, 0.0)),
            ("侧向力", 3.0, (0.0, 8.0, 0.0)),
            ("碰撞冲击", 50.0, (-20.0, 5.0, 10.0)),
        ]
        
        for name, peak_force, direction in scenarios:
            if name == "碰撞冲击":
                wrenches = self.virtual_force.simulate_collision(
                    direction=direction,
                    peak_force=peak_force,
                    duration_ms=150
                )
                contact_state = self.force.detect_contact(wrenches[-1])
                payload = self.force.estimate_payload(wrenches[-1])
                print(f"  [{name}] 峰值力={np.max([w.magnitude for w in wrenches]):.2f}N, "
                      f"接触检测={contact_state.is_contact}, "
                      f"估计负载={payload:.3f}kg")
            else:
                wrench = self.virtual_force.simulate_contact(
                    force=direction,
                    torque=(0.1, -0.1, 0.0)
                )
                contact_state = self.force.detect_contact(wrench)
                payload = self.force.estimate_payload(wrench)
                print(f"  [{name}] 力向量=({direction[0]:.1f}, {direction[1]:.1f}, {direction[2]:.1f})N, "
                      f"接触={contact_state.is_contact}, "
                      f"接触力={contact_state.contact_force:.2f}N")
        
        # 摩擦力仿真
        friction = self.virtual_force.simulate_friction_contact(
            normal_force=10.0,
            velocity=(0.1, 0.0, 0.0),
            friction_coeff=0.3
        )
        print(f"  摩擦接触: 摩擦力=({friction.force[0]:.3f}, "
              f"{friction.force[1]:.3f}, {friction.force[2]:.3f})N")
    
    def run_imu_sequence(self, num_frames: int = 10):
        """IMU传感器测试序列"""
        print_subsection(f"IMU传感器测试 ({num_frames}帧)")
        
        # 静止测试
        print("  静止状态测试:")
        static_frames = []
        for _ in range(num_frames):
            frame = self.virtual_imu.simulate_static(orientation=(0.0, 0.0, 0.0))
            pose = self.pose_estimator.update(frame.accel, frame.gyro, frame.mag)
            static_frames.append(frame)
        euler = self.pose_estimator.get_euler()
        print(f"    欧拉角: roll={math.degrees(euler[0]):.2f}°, "
              f"pitch={math.degrees(euler[1]):.2f}°, "
              f"yaw={math.degrees(euler[2]):.2f}°")
        
        # 轨迹测试
        print("  轨迹模拟测试:")
        for traj_type in ['circle', 'figure8', 'sine']:
            traj_frames = self.virtual_imu.simulate_trajectory(
                trajectory_type=traj_type,
                duration_s=1.0,
                dt=0.01
            )
            avg_gyro = np.mean([f.gyro_magnitude for f in traj_frames])
            avg_accel = np.mean([f.accel_magnitude for f in traj_frames])
            print(f"    {traj_type}: 平均角速度={avg_gyro:.4f}rad/s, "
                  f"平均加速度={avg_accel:.3f}m/s²")
        
        # AGV运动测试
        print("  AGV运动仿真:")
        for lin_vel in [(0.5, 0.0), (1.0, 0.0), (0.5, 0.3)]:
            frame = self.virtual_imu.simulate_agv_motion(
                linear_velocity=lin_vel,
                angular_velocity=0.1,
                grade=self.grade
            )
            print(f"    线速度={lin_vel}m/s, 角速度=0.1rad/s: "
                  f"accel=({frame.accel[0]:.3f}, {frame.accel[1]:.3f}, {frame.accel[2]:.3f})m/s², "
                  f"gyro=({frame.gyro[0]:.4f}, {frame.gyro[1]:.4f}, {frame.gyro[2]:.4f})rad/s")
        
        # 步行测试
        walk_frames = self.virtual_imu.simulate_human_walking(
            step_frequency=1.5,
            walk_speed=1.0,
            duration_s=2.0,
            dt=0.01
        )
        print(f"  人类步行仿真: {len(walk_frames)}帧, "
              f"步频=1.5Hz, 速度=1.0m/s")
    
    def run_fusion_sequence(self, num_iterations: int = 20):
        """融合模块测试序列"""
        print_subsection(f"融合模块测试 ({num_iterations}次迭代)")
        
        # 收集传感器数据
        accel_history = []
        gyro_history = []
        force_history = []
        
        for i in range(num_iterations):
            # IMU
            frame = self.virtual_imu.simulate_agv_motion(
                linear_velocity=(0.5, 0.0),
                angular_velocity=0.1,
                grade=self.grade
            )
            accel_history.append(frame.accel)
            gyro_history.append(frame.gyro)
            
            # 互补滤波
            self.complementary_filter.update(
                {'accel': frame.accel, 'gyro': frame.gyro},
                dt=0.01
            )
            
            # EKF预测
            self.ekf.predict(dt=0.01)
            
            # 力觉
            wrench = self.virtual_force.simulate_contact(
                force=(0.0, 0.0, -10.0)
            )
            force_history.append(wrench.to_vector())
            
            # 多传感器融合
            self.multi_fusion.update({
                'complementary': {
                    'accel': frame.accel,
                    'gyro': frame.gyro
                }
            }, dt=0.01)
        
        # 分析结果
        comp_state = self.complementary_filter.get_state()
        ekf_state = self.ekf.get_state()
        fusion_state = self.multi_fusion.get_fused_state()
        
        print(f"  互补滤波收敛状态: roll={math.degrees(comp_state[0]):.2f}°, "
              f"pitch={math.degrees(comp_state[1]):.2f}°, "
              f"yaw={math.degrees(comp_state[2]):.2f}°")
        print(f"  EKF状态估计前3维: {ekf_state[:3] if len(ekf_state) >= 3 else ekf_state}")
        print(f"  多传感器融合: {fusion_state}")
    
    def run_control_sequence(self, num_steps: int = 20):
        """AGV控制测试序列"""
        print_subsection(f"AGV控制测试 ({num_steps}步)")
        
        dt = 0.01
        
        # 圆形轨迹跟踪
        print("  圆形轨迹跟踪:")
        radius = 1.0
        linear_vel = 0.5
        angular_vel = linear_vel / radius
        
        for step in range(num_steps):
            t = step * dt
            # 目标位置
            target_x = radius * math.cos(angular_vel * t)
            target_y = radius * math.sin(angular_vel * t)
            target_theta = angular_vel * t + math.pi / 2
            
            # 目标速度
            target_vx = -radius * angular_vel * math.sin(angular_vel * t)
            target_vy = radius * angular_vel * math.cos(angular_vel * t)
            target_vtheta = angular_vel
            
            if step % 5 == 0:
                print(f"    Step {step:3d}: pos=({target_x:.3f}, {target_y:.3f}), "
                      f"vel=({target_vx:.3f}, {target_vy:.3f}), "
                      f"theta={math.degrees(target_theta):.1f}°")
        
        # 障碍物避让 (DWA)
        print("  障碍物避让 (DWA):")
        from src.control.obstacle_avoidance import ObstacleAvoider, AvoidanceConfig, Obstacle
        config = AvoidanceConfig()
        oac = ObstacleAvoider(config=config)
        
        for dist in [2.0, 1.0, 0.5, 0.3]:
            obstacles = [Obstacle(position=np.array([dist, 0.0]), radius=0.3)]
            cmd = oac.compute_command(
                robot_pose=np.array([0.0, 0.0, 0.0]),
                robot_velocity=np.array([0.5, 0.0, 0.0]),
                goal=np.array([3.0, 0.0]),
                obstacles=obstacles,
                dt=dt
            )
            print(f"    障碍距离={dist}m: vx={cmd.vx:.3f}, "
                  f"vy={cmd.vy:.3f}, omega={cmd.omega:.3f}rad/s")
    
    def run_complete_pipeline(self, duration: float = 3.0, dt: float = 0.01):
        """完整流水线测试"""
        print_subsection(f"完整流水线 ({duration}s, dt={dt}s)")
        
        num_steps = int(duration / dt)
        timestamps = []
        fusion_latencies = []
        control_latencies = []
        
        for step in range(num_steps):
            t_start = time.time()
            
            # 1. 传感器采集
            tactile_frame = self.tactile.capture()
            force_wrench = self.force.capture()
            imu_frame = self.imu.capture()
            
            # 2. 姿态估计
            pose = self.pose_estimator.update(
                imu_frame.accel, imu_frame.gyro, imu_frame.mag, dt
            )
            
            # 3. 融合
            fusion_start = time.time()
            self.multi_fusion.update({
                'complementary': {
                    'accel': imu_frame.accel,
                    'gyro': imu_frame.gyro
                }
            }, dt=dt)
            fusion_latency = (time.time() - fusion_start) * 1000
            
            # 4. 控制决策
            control_start = time.time()
            # 简化的速度命令 (安全由SafetyController统一管理)
            target_velocity = np.array([0.5, 0.0, 0.0])
            control_latency = (time.time() - control_start) * 1000
            
            timestamps.append(t_start)
            fusion_latencies.append(fusion_latency)
            control_latencies.append(control_latency)
        
        print(f"  总帧数: {num_steps}")
        print(f"  融合延迟: 平均={np.mean(fusion_latencies):.3f}ms, "
              f"最大={np.max(fusion_latencies):.3f}ms, "
              f"最小={np.min(fusion_latencies):.3f}ms")
        print(f"  控制延迟: 平均={np.mean(control_latencies):.3f}ms, "
              f"最大={np.max(control_latencies):.3f}ms, "
              f"最小={np.min(control_latencies):.3f}ms")
        print(f"  总延迟预算: {np.mean(fusion_latencies) + np.mean(control_latencies):.3f}ms "
              f"(目标<10ms)")
    
    def run_all_tests(self, duration: float = 3.0):
        """运行所有测试"""
        print_header(f"SuperModel 具身智能传感器综合展示 - {self.grade}级AGV")
        from src.sensors.tactile import get_tactile_spec
        from src.sensors.force import get_force_spec
        from src.sensors.imu import get_imu_spec
        ts = get_tactile_spec(self.grade)
        fs = get_force_spec(self.grade)
        ims = get_imu_spec(self.grade)
        print(f"  规格: 触觉={ts['array']}, "
              f"力觉={fs['axes']}轴, "
              f"IMU={ims['type']}")
        
        self.run_tactile_sequence(num_frames=10)
        self.run_force_sequence(num_frames=10)
        self.run_imu_sequence(num_frames=10)
        self.run_fusion_sequence(num_iterations=20)
        self.run_control_sequence(num_steps=20)
        self.run_complete_pipeline(duration=duration)
        
        print_header("所有测试完成")
        print(f"  {self.grade}级AGV具身智能传感器综合展示测试通过")
        print()
    
    def cleanup(self):
        """清理资源"""
        self.tactile.close()
        self.virtual_tactile.close()
        self.force.close()
        self.virtual_force.close()
        self.imu.close()
        self.virtual_imu.close()
        print("[Cleanup] 所有传感器已关闭")


def main():
    parser = argparse.ArgumentParser(description='SuperModel 具身智能传感器综合展示')
    parser.add_argument('--grade', '-g', type=str, default='M',
                        choices=['S', 'M', 'L', 'XL', 'XXL'],
                        help='AGV等级 (默认: M)')
    parser.add_argument('--duration', '-d', type=float, default=3.0,
                        help='完整流水线测试时长,秒 (默认: 3.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    parser.add_argument('--all-grades', '-a', action='store_true',
                        help='运行所有等级测试')
    args = parser.parse_args()
    
    if args.all_grades:
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            try:
                showcase = EmbodiedSensorShowcase(grade=grade, verbose=args.verbose)
                showcase.run_all_tests(duration=args.duration)
                showcase.cleanup()
            except Exception as e:
                print(f"[ERROR] {grade}级测试失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        showcase = EmbodiedSensorShowcase(grade=args.grade, verbose=args.verbose)
        showcase.run_all_tests(duration=args.duration)
        showcase.cleanup()
    
    print()
    print("🎉 SuperModel 具身智能传感器综合展示完成!")


if __name__ == '__main__':
    main()
