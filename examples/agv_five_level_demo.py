"""
AGV五级完整演示
================

展示 SuperModel 在不同 AGV 等级下的完整能力

- S级: 教育/实验级 - 基础传感器 + 简单控制
- M级: 标准助手级 - 完整传感器 + 阻抗控制
- L级: 专业工业级 - 多传感器融合 + MPC
- XL级: 高性能级 - 多机协作 + 预测控制
- XXL级: 旗舰全功能 - 全模态 + 自主学习
"""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, get_stereo_spec
from sensors.audio import BinauralMic, get_audio_spec
from sensors.tactile import TactileArray, get_tactile_spec, VirtualTactileSensor
from sensors.force import ForceTorqueSensor, get_force_spec, VirtualForceSensor, ForceSensorType
from sensors.imu import IMUSensor, get_imu_spec, VirtualIMUSensor, PoseEstimator, IMUSensorType
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from control.agv import AGVMotionController, AGVSpec, AGVGrade, get_agv_spec
from control.impedance import ImpedanceController, ImpedanceParams
from control.mpc import MPCConfig, JointSpaceMPC, get_mpc_spec
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, get_safety_spec
from control.planner import TaskPlanner, Task, TaskStatus, TaskPriority
from control.skill import SkillLibrary, Skill
from control.supervisor import ControlSupervisor, SupervisorConfig, ControlMode
from control.multi_agent import MultiAgentCoordinator, FormationType, get_coordination_spec
from simulation.environment import (
    RobotSimulator, SensorSimulator, SimConfig,
    ContactPhysicsModel, get_contact_physics_spec, PRESET_SCENES
)


def print_header(title: str):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_grade_specs(grade: str):
    """打印指定AGV等级的完整规格"""
    print(f"\n【{grade}级规格】")
    
    # 感知规格
    print("\n  📷 视觉:")
    spec = get_stereo_spec(grade)
    print(f"    基线: {spec['baseline_mm']}mm, 视场: {spec['fov']}°, "
          f"范围: {spec['range_m']}m")
    
    print("\n  🎤 听觉:")
    spec = get_audio_spec(grade)
    print(f"    通道数: {spec['channels']}, 采样率: {spec['sr']}Hz, "
          f"拾音范围: {spec['range_m']}m, 波束形成: {'是' if spec['beamforming'] else '否'}")
    
    print("\n  🖐️ 触觉:")
    spec = get_tactile_spec(grade)
    print(f"    阵列: {spec['array']}, 分辨率: {spec['res']}bit, "
          f"压力范围: {spec['range_kpa']}kPa, 频率: {spec['freq_hz']}Hz")
    
    print("\n  💪 力觉:")
    spec = get_force_spec(grade)
    print(f"    轴数: {spec['axes']}, 力范围: {spec['force_range']}N, "
          f"力矩范围: {spec['torque_range']}N·m, 采样: {spec['sampling_hz']}Hz")
    
    print("\n  🧭 IMU:")
    spec = get_imu_spec(grade)
    print(f"    型号: {spec['type']}, 加速度程: ±{spec['accel_range']}g, "
          f"陀螺量程: ±{spec['gyro_range']}°/s, 噪声: {spec['noise_density']}μg/√Hz")
    
    print("\n  🚗 AGV运动:")
    spec = get_agv_spec(grade)
    print(f"    最大速度: {spec.max_linear_speed}m/s, "
          f"最大加速度: {spec.max_linear_accel}m/s², "
          f"控制频率: {spec.control_frequency}Hz, "
          f"驱动类型: {spec.drive_type.value}")
    
    print("\n  🛡️ 安全:")
    spec = get_safety_spec(grade)
    print(f"    安全等级: {spec['level']}, "
          f"响应时间: {spec['response_time_ms']}ms")
    
    print("\n  🤝 多机协作:")
    spec = get_coordination_spec(grade)
    print(f"    多Agent支持: {'是' if spec['multi_agent'] else '否'}, "
          f"最大Agent数: {spec['max_agents']}, "
          f"编队控制: {'是' if spec['formation'] else '否'}")


def demo_sensor_initialization(grade: str):
    """演示传感器初始化"""
    print_header(f"传感器初始化 ({grade}级)")
    
    # 视觉
    cam = BinocularCamera()
    cam.open()
    frame = cam.capture()
    print(f"  ✓ 视觉: {frame.left_image.shape}, frame_id={frame.frame_id}")
    cam.close()
    
    # 听觉
    audio_spec = get_audio_spec(grade)
    mic = BinauralMic(sample_rate=audio_spec['sr'])
    mic.open()
    audio_frame = mic.capture()
    print(f"  ✓ 听觉: {audio_frame.left_channel.shape}, 采样率: {audio_frame.sample_rate}Hz")
    mic.close()
    
    # 触觉
    tactile_spec = get_tactile_spec(grade)
    tactile = TactileArray(array_size=tactile_spec['array'])
    tactile.open()
    t_frame = tactile.capture()
    print(f"  ✓ 触觉: {t_frame.pressure_map.shape}, 峰值压力: {t_frame.pressure_map.max():.3f}")
    tactile.close()
    
    # 力觉
    force_spec = get_force_spec(grade)
    force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS if force_spec['axes'] == 6 else ForceSensorType.THREE_AXIS)
    force.open()
    wrench = force.capture()
    print(f"  ✓ 力觉: |F|={wrench.magnitude:.2f}N, |T|={wrench.torque_magnitude:.3f}N·m")
    force.close()
    
    # IMU
    imu_spec = get_imu_spec(grade)
    imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
    imu.open()
    imu_frame = imu.capture()
    print(f"  ✓ IMU: |accel|={imu_frame.accel_magnitude:.2f}m/s², "
          f"|gyro|={imu_frame.gyro_magnitude:.4f}rad/s")
    imu.close()


def demo_sensor_fusion(grade: str):
    """演示多模态融合"""
    print_header(f"多模态融合 ({grade}级)")
    
    # 融合配置 (固定输入维度)
    grade_configs = {
        'S': (128, 2, 1),
        'M': (256, 4, 2),
        'L': (512, 8, 4),
        'XL': (768, 12, 6),
        'XXL': (1024, 16, 8)
    }
    hidden_dim, num_heads, num_layers = grade_configs.get(grade, grade_configs['M'])
    
    config = FusionConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    fusion = CrossModalFusion(config)
    
    # 创建多模态输入 (torch tensors)
    import torch
    batch_size = 2
    multimodal = MultimodalInput(
        vision=torch.randn(batch_size, 512),
        audio=torch.randn(batch_size, 128),
        tactile=torch.randn(batch_size, 64),
        force=torch.randn(batch_size, 32),
        imu=torch.randn(batch_size, 64)
    )
    
    # 融合
    start = time.time()
    output = fusion(multimodal)
    elapsed = (time.time() - start) * 1000
    
    print(f"  ✓ 融合网络: hidden={hidden_dim}, heads={num_heads}, layers={num_layers}")
    print(f"  ✓ 融合输出: {output.shape}")
    print(f"  ✓ 推理延迟: {elapsed:.2f}ms")
    
    # 根据等级检查实时性要求
    latency_req = {'S': 50, 'M': 20, 'L': 10, 'XL': 5, 'XXL': 2}.get(grade, 20)
    if elapsed < latency_req:
        print(f"  ✓ 实时性要求满足 (<{latency_req}ms)")
    else:
        print(f"  ⚠ 延迟超标 (要求<{latency_req}ms)")


def demo_control_loop(grade: str):
    """演示控制循环"""
    print_header(f"控制循环 ({grade}级)")
    
    # AGV控制器
    agv_grade = AGVGrade[grade] if grade in [g.value for g in AGVGrade] else AGVGrade.M
    spec = AGVSpec.from_grade(agv_grade)
    
    controller = AGVMotionController(spec)
    
    # 安全控制器
    safety_config = SafetyConfig(
        joint_limits_lower=-np.ones(6) * np.pi,
        joint_limits_upper=np.ones(6) * np.pi,
        velocity_limits=np.ones(6) * 2.0,
        acceleration_limits=np.ones(6) * 5.0,
        safety_level=SafetyLevel.M
    )
    safety = SafetyController(safety_config)
    
    # 仿真
    sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
    
    # 运行控制循环
    n_steps = 50
    print(f"  ✓ AGV控制器初始化: {spec.drive_type.value}驱动")
    print(f"  ✓ 安全控制器初始化: 碰撞阈值={safety_config.collision_threshold}N")
    print(f"  ✓ 运行 {n_steps} 步仿真...")
    
    for i in range(n_steps):
        # 仿真一步
        sim.step(np.zeros(6))
    
    status = safety.get_safety_status()
    print(f"  ✓ 控制循环完成: {n_steps}步")
    print(f"  ✓ AGV速度: {spec.max_linear_speed}m/s, 加速度: {spec.max_linear_accel}m/s²")
    print(f"  ✓ 控制频率: {spec.control_frequency}Hz")
    print(f"  ✓ 安全状态: 启用={status['enabled']}, 紧急停止={status['emergency_stopped']}")


def demo_task_planning(grade: str):
    """演示任务规划"""
    print_header(f"任务规划 ({grade}级)")
    
    planner = TaskPlanner()
    
    # 添加任务
    tasks = [
        Task(id="t1", name="移动任务", description="移动到目标位置", priority=TaskPriority.HIGH),
        Task(id="t2", name="检测任务", description="检测物体", priority=TaskPriority.NORMAL),
        Task(id="t3", name="抓取任务", description="抓取物体", priority=TaskPriority.CRITICAL),
        Task(id="t4", name="移动放置任务", description="移动到放置点", priority=TaskPriority.HIGH),
        Task(id="t5", name="放置任务", description="放置物体", priority=TaskPriority.CRITICAL),
    ]
    
    for task in tasks:
        planner.add_task(task)
    
    print(f"  ✓ 任务规划器初始化")
    print(f"  ✓ 添加了 {len(tasks)} 个任务:")
    for task in tasks:
        print(f"    - {task.name}: {task.description} (优先级={task.priority.name})")


def demo_contact_physics(grade: str):
    """演示接触物理"""
    print_header(f"接触物理 ({grade}级)")
    
    # 获取指定等级的接触物理模型
    contact_model = get_contact_physics_spec(grade)
    
    print(f"  接触模型参数:")
    print(f"    静摩擦系数: {contact_model.mu_s}")
    print(f"    动摩擦系数: {contact_model.mu_d}")
    print(f"    接触刚度: {contact_model.k_n} N/m")
    print(f"    接触阻尼: {contact_model.c_n} N·s/m")
    
    # 模拟接触事件
    event = contact_model.simulate_contact_event(
        initial_penetration=0.002,
        impact_velocity=0.1,
        object_mass=0.5,
        duration=0.05
    )
    
    print(f"  ✓ 接触事件模拟完成:")
    print(f"    持续时间: {event['time'][-1]*1000:.1f}ms")
    print(f"    最大法向力: {max(event['normal_force']):.2f}N")
    print(f"    滑移检测: {'是' if any(event['slip_detected']) else '否'}")
    
    # 抓取质量评估
    contact_points = [
        np.array([0.1, 0.0, 0.05]),
        np.array([-0.1, 0.0, 0.05]),
        np.array([0.0, 0.1, 0.05])
    ]
    contact_normals = [
        np.array([0, 0, 1]),
        np.array([0, 0, 1]),
        np.array([0, 0, 1])
    ]
    
    quality = contact_model.compute_grasp_quality(
        contact_points, contact_normals,
        object_center=np.array([0, 0, 0.05]),
        object_mass=0.1
    )
    
    print(f"  ✓ 抓取质量评估:")
    print(f"    综合评分: {quality['overall']:.2f}")
    print(f"    力闭合: {quality['force_closure']:.2f}")
    print(f"    接触刚度: {quality['stiffness']:.2f}")


def demo_multi_agent(grade: str):
    """演示多机协作 (L级以上)"""
    print_header(f"多机协作 ({grade}级)")
    
    coord_spec = get_coordination_spec(grade)
    
    if not coord_spec['multi_agent']:
        print(f"  ⚠ {grade}级不支持多机协作")
        return
    
    coordinator = MultiAgentCoordinator(
        communication_range=10.0,
        safety_distance=0.5,
        max_agents=coord_spec['max_agents']
    )
    
    print(f"  ✓ 协调器初始化:")
    print(f"    最大Agent数: {coord_spec['max_agents']}")
    print(f"    编队控制: {'支持' if coord_spec['formation'] else '不支持'}")
    print(f"    碰撞避免: {coord_spec['collision_avoidance']}")
    print(f"  ✓ 编队类型: LINE, TRIANGLE, V_SHAPE 等")
    print(f"  ✓ 通信范围: 10m, 安全距离: 0.5m")


def demo_all_grades():
    """演示所有AGV等级"""
    grades = ['S', 'M', 'L', 'XL', 'XXL']
    
    for grade in grades:
        print_header(f"AGV {grade}级完整演示")
        print_grade_specs(grade)
        
        # 跳过传感器初始化演示(太耗时)
        # demo_sensor_initialization(grade)
        
        demo_sensor_fusion(grade)
        demo_control_loop(grade)
        demo_task_planning(grade)
        demo_contact_physics(grade)
        
        if grade in ['L', 'XL', 'XXL']:
            demo_multi_agent(grade)
        
        print(f"\n✅ {grade}级演示完成\n")


def main():
    """主函数"""
    print("\n" + "="*60)
    print(" SuperModel 超模态机器人具身智能大脑")
    print(" AGV五级完整能力演示")
    print("="*60)
    
    # 演示所有等级
    demo_all_grades()
    
    # 单独演示S级(详细)
    print_header("S级详细演示")
    print_grade_specs('S')
    demo_sensor_initialization('S')
    demo_sensor_fusion('S')
    demo_control_loop('S')
    demo_task_planning('S')
    demo_contact_physics('S')
    
    # XXL级完整演示
    print_header("XXL级完整演示")
    print_grade_specs('XXL')
    demo_sensor_fusion('XXL')
    demo_control_loop('XXL')
    demo_task_planning('XXL')
    demo_contact_physics('XXL')
    demo_multi_agent('XXL')
    
    print("\n" + "="*60)
    print(" 演示完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
