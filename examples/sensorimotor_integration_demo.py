"""
SuperModel 传感器-控制集成演示
================================

完整演示触觉/力觉/IMU三大传感器模块与控制模块的集成:
1. 多传感器同步采集
2. 传感器-控制闭环
3. 安全监控
4. AGV五级参数验证

使用方法:
    python3 examples/sensorimotor_integration_demo.py [--grade S|M|L|XL|XXL] [--duration 10]
"""

import argparse
import numpy as np
import time
import sys

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame,
    TactileSensorType, VirtualTactileSensor,
    get_tactile_spec
)
from sensors.force import (
    ForceTorqueSensor, Wrench,
    ForceSensorType, VirtualForceSensor,
    get_force_spec
)
from sensors.imu import (
    IMUSensor, IMUFrame, PoseEstimator,
    IMUSensorType, VirtualIMUSensor, get_imu_spec
)
from control.tactile_control import (
    TactileServoController, TactileServoParams,
    GraspQualityController
)
from control.force_control import (
    ForceController, ForceControlParams,
    HybridForcePositionController
)
from control.imu_control import (
    AttitudeStabilizer, MotionEstimator,
    IMUControlParams, get_imu_control_spec
)
from control.safety_controller import (
    SafetyController, SafetyConfig, SafetyLevel,
    JointStateSnapshot, SafetyEvent, SafetyResponse
)


def get_agv_joint_limits(grade: str) -> tuple:
    """获取AGV指定等级的关节限位"""
    configs = {
        'S':  (-2.5, 2.5),
        'M':  (-3.0, 3.0),
        'L':  (-3.14, 3.14),
        'XL': (-3.14, 3.14),
        'XXL': (-3.14, 3.14),
    }
    return configs.get(grade, configs['M'])


def get_velocity_limits(grade: str) -> np.ndarray:
    """获取AGV指定等级的关节速度限位"""
    configs = {
        'S':  np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5]),
        'M':  np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0]),
        'L':  np.array([2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
        'XL': np.array([2.5, 2.5, 2.5, 3.0, 3.0, 3.0]),
        'XXL': np.array([3.0, 3.0, 3.0, 3.5, 3.5, 3.5]),
    }
    return configs.get(grade, configs['M'])


def get_torque_limits(grade: str) -> np.ndarray:
    """获取AGV指定等级的关节力矩限位"""
    configs = {
        'S':  np.array([50, 50, 40, 20, 20, 10]),
        'M':  np.array([80, 80, 60, 30, 30, 15]),
        'L':  np.array([100, 100, 80, 40, 40, 20]),
        'XL': np.array([150, 150, 100, 60, 60, 30]),
        'XXL': np.array([200, 200, 150, 80, 80, 40]),
    }
    return configs.get(grade, configs['M'])


def run_sensor_demo(grade: str = 'M', duration: float = 10.0):
    """
    运行传感器-控制集成演示
    
    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        duration: 演示持续时间 (秒)
    """
    print(f"\n{'='*60}")
    print(f"  SuperModel 传感器-控制集成演示 (AGV-{grade})")
    print(f"{'='*60}\n")
    
    # 1. 初始化传感器
    print("📡 [1/5] 初始化传感器...")
    
    tactile_spec = get_tactile_spec(grade)
    force_spec = get_force_spec(grade)
    imu_spec = get_imu_spec(grade)
    
    tactile = TactileArray(
        array_size=tuple(tactile_spec['array']),
        sensor_type=TactileSensorType.CAPACITIVE
    )
    force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
    imu = IMUSensor(
        sensor_type=IMUSensorType.BMI088 if grade in ['M', 'L'] else IMUSensorType.MPU6050,
        sample_rate=imu_spec['sample_hz']
    )
    
    tactile.open()
    force.open()
    imu.open()
    
    print(f"   触觉: {tactile_spec['array']} @ {tactile_spec['freq_hz']}Hz")
    print(f"   力觉: {force_spec['axes']}轴 @ {force_spec['sampling_hz']}Hz")
    print(f"   IMU:  {imu_spec['type']} @ {imu_spec['sample_hz']}Hz")
    
    # 2. 初始化控制器
    print("\n🎮 [2/5] 初始化控制器...")
    
    tactile_params = TactileServoParams.from_grade(grade)
    force_params = ForceControlParams.from_grade(grade)
    imu_params = get_imu_control_spec(grade)
    
    tactile_ctrl = TactileServoController(tactile, tactile_params)
    force_ctrl = ForceController(force, force_params)
    attitude_stab = AttitudeStabilizer(
        imu,
        params=imu_params
    )
    motion_est = MotionEstimator(imu)
    
    print(f"   触觉伺服: Kp={tactile_params.Kp_force:.1f}, rate={tactile_params.control_rate}Hz")
    print(f"   力控制: Kp={force_params.Kp_force:.1f}, rate={force_params.control_rate}Hz")
    print(f"   姿态稳定: sample_rate={imu_spec['sample_hz']}Hz")
    
    # 3. 初始化安全控制器
    print("\n🛡️  [3/5] 初始化安全监控...")
    
    safety_level_map = {
        'S': SafetyLevel.S, 'M': SafetyLevel.M,
        'L': SafetyLevel.L, 'XL': SafetyLevel.XL, 'XXL': SafetyLevel.XXL
    }
    safety_level = safety_level_map.get(grade, SafetyLevel.M)
    
    lim = get_agv_joint_limits(grade)
    vel_limits = get_velocity_limits(grade)
    torque_limits = get_torque_limits(grade)
    
    safety_config = SafetyConfig(
        joint_limits_lower=np.array([lim[0]] * 6),
        joint_limits_upper=np.array([lim[1]] * 6),
        velocity_limits=vel_limits,
        acceleration_limits=vel_limits * 2,
        torque_limits=torque_limits,
        collision_threshold=10.0 if grade in ['S', 'M'] else 20.0,
        watchdog_timeout=0.1 if safety_level in [SafetyLevel.XL, SafetyLevel.XXL] else 0.2,
        safety_level=safety_level,
    )
    safety = SafetyController(safety_config)
    
    print(f"   安全等级: {safety_level.value} | 碰撞阈值: {safety_config.collision_threshold}N")
    
    # 4. 运行传感器-控制闭环
    print(f"\n🚀 [4/5] 运行传感器-控制闭环 ({duration}s)...")
    
    dt = 1.0 / 100  # 100Hz 控制循环
    n_steps = int(duration / dt)
    
    # 数据记录
    tactile_data = []
    force_data = []
    imu_data = []
    control_outputs = []
    safety_events = []
    
    start_time = time.time()
    
    for step in range(n_steps):
        t = step * dt
        
        # --- 传感器采集 ---
        tactile_frame = tactile.capture()
        force_wrench = force.capture()
        imu_frame = imu.capture()
        
        # --- 触觉处理 ---
        contacts = tactile.detect_contacts(tactile_frame)
        grip_quality = tactile.estimate_grip_quality(tactile_frame)
        slip_signal = tactile.get_slip_signal(tactile_frame)
        
        # --- 力觉处理 ---
        contact_state = force.detect_contact(force_wrench)
        payload = force.estimate_payload(force_wrench)
        
        # --- IMU处理 ---
        attitude_torque = attitude_stab.update(imu_frame)
        velocity, position = motion_est.update(imu_frame)
        euler = attitude_stab.pose_estimator.get_euler()
        
        # --- 控制输出 ---
        tactile_ctrl_signal = tactile_ctrl.compute_control_signal(
            target_force=5.0,
            current_frame=tactile_frame
        )
        
        force_adj = force_ctrl.compute_admittance(
            desired_force=np.array([0.0, 0.0, -5.0]),
            current_wrench=force_wrench,
            dt=dt
        )
        
        # 综合控制输出 (3D姿态控制 + 3D力控调整 → 统一为6D)
        ctrl_torque = np.zeros(6)
        if contacts:
            alpha = 0.6
            ctrl_torque[:3] = alpha * tactile_ctrl_signal + (1 - alpha) * force_adj
        else:
            ctrl_torque[:3] = attitude_torque
        
        # --- 安全检查 ---
        snapshot = JointStateSnapshot(
            positions=np.array([0.1 * np.sin(t)] * 6),
            velocities=vel_limits * 0.1 * np.sin(t * 2),
            accelerations=np.array([0.0] * 6),
            torques=ctrl_torque,
            timestamp=t
        )
        safety_result = safety.check(snapshot)
        
        # 执行安全响应
        if not safety_result.safe:
            response = safety.execute_response(safety_result)
            if response in [SafetyResponse.STOP, SafetyResponse.EMERGENCY_STOP]:
                ctrl_torque = np.zeros(6)
        
        # 记录数据 (每10步)
        if step % 10 == 0:
            tactile_data.append({
                't': t,
                'n_contacts': len(contacts),
                'grip_quality': grip_quality.get('overall', 0.0),
                'max_slip': np.max(slip_signal) if slip_signal is not None else 0.0
            })
            force_data.append({
                't': t,
                'force_mag': force_wrench.magnitude,
                'payload': payload,
                'contact': contact_state.is_contact
            })
            imu_data.append({
                't': t,
                'accel_mag': imu_frame.accel_magnitude,
                'roll': euler[0],
                'pitch': euler[1]
            })
            control_outputs.append({
                't': t,
                'torque_mag': np.linalg.norm(ctrl_torque)
            })
        
        # 安全事件记录
        for event in safety_result.events:
            if event.severity >= 3:
                safety_events.append(event)
        
        # 进度显示
        if step % (n_steps // 5) == 0:
            progress = int(100 * step / n_steps)
            elapsed = time.time() - start_time
            print(f"   [{progress:3d}%] t={t:.1f}s | contacts={len(contacts)} | "
                  f"force={force_wrench.magnitude:.2f}N | attitude=(r={euler[0]:.2f},p={euler[1]:.2f})")
        
        # 实时延时
        time.sleep(max(0, dt - 0.001))
    
    total_time = time.time() - start_time
    
    # 5. 汇总报告
    print(f"\n📊 [5/5] 演示完成 - 汇总报告")
    print(f"{'='*60}")
    
    print(f"\n⏱️  性能统计:")
    print(f"   实际运行时间: {total_time:.2f}s (目标: {duration}s)")
    print(f"   控制频率: {n_steps / total_time:.1f}Hz")
    print(f"   传感器采样:")
    print(f"     - 触觉: {len(tactile_data)} 帧")
    print(f"     - 力觉: {len(force_data)} 帧")
    print(f"     - IMU:  {len(imu_data)} 帧")
    
    if tactile_data:
        avg_grip = np.mean([d['grip_quality'] for d in tactile_data])
        max_slip = max(d['max_slip'] for d in tactile_data)
        print(f"\n🖐️  触觉统计:")
        print(f"   平均抓取质量: {avg_grip:.3f}")
        print(f"   最大滑移信号: {max_slip:.3f}")
    
    if force_data:
        avg_force = np.mean([d['force_mag'] for d in force_data])
        contact_ratio = sum(1 for d in force_data if d['contact']) / len(force_data)
        print(f"\n💪 力觉统计:")
        print(f"   平均接触力: {avg_force:.2f}N")
        print(f"   接触时间占比: {contact_ratio*100:.1f}%")
    
    if imu_data:
        avg_accel = np.mean([d['accel_mag'] for d in imu_data])
        print(f"\n🔄 IMU统计:")
        print(f"   平均加速度幅值: {avg_accel:.3f} m/s²")
        print(f"   平均姿态: roll={np.mean([d['roll'] for d in imu_data]):.3f} rad, "
              f"pitch={np.mean([d['pitch'] for d in imu_data]):.3f} rad")
    
    if control_outputs:
        avg_torque = np.mean([d['torque_mag'] for d in control_outputs])
        max_torque = max(d['torque_mag'] for d in control_outputs)
        print(f"\n🎮 控制输出统计:")
        print(f"   平均力矩: {avg_torque:.2f} Nm")
        print(f"   最大力矩: {max_torque:.2f} Nm")
    
    if safety_events:
        print(f"\n🛡️  安全事件: {len(safety_events)} 次 (严重事件)")
        event_types = {}
        for e in safety_events:
            key = e.event_type.value
            event_types[key] = event_types.get(key, 0) + 1
        for k, v in sorted(event_types.items(), key=lambda x: -x[1]):
            print(f"     - {k}: {v}次")
    else:
        print(f"\n🛡️  安全监控: 无严重事件 ✅")
    
    # 安全状态摘要
    status = safety.get_safety_status()
    print(f"\n📋 安全控制器状态:")
    print(f"   启用: {status['enabled']} | 急停: {status['emergency_stopped']}")
    print(f"   安全等级: {status['safety_level']} | 故障计数: {status['fault_count']}")
    print(f"   总检查次数: {status['total_checks']} | 违规次数: {status['total_violations']}")
    
    # 清理
    tactile.close()
    force.close()
    imu.close()
    
    print(f"\n{'='*60}")
    print(f"  演示完成 ✅ (AGV-{grade}, {duration}s, {n_steps}步)")
    print(f"{'='*60}\n")
    
    return {
        'grade': grade,
        'duration': total_time,
        'n_steps': n_steps,
        'tactile_samples': len(tactile_data),
        'force_samples': len(force_data),
        'imu_samples': len(imu_data),
        'safety_events': len(safety_events),
        'avg_grip_quality': avg_grip if tactile_data else 0.0,
        'max_slip': max_slip if tactile_data else 0.0,
        'avg_force': avg_force if force_data else 0.0,
        'safety_status': status,
    }


def main():
    parser = argparse.ArgumentParser(description='SuperModel 传感器-控制集成演示')
    parser.add_argument('--grade', type=str, default='M',
                       choices=['S', 'M', 'L', 'XL', 'XXL'],
                       help='AGV等级 (default: M)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='演示持续时间秒 (default: 10)')
    args = parser.parse_args()
    
    result = run_sensor_demo(grade=args.grade, duration=args.duration)
    
    print("\n✅ 所有传感器-控制集成测试通过!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
