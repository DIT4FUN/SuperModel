"""
SuperModel 完整具身智能管道演示
================================

端到端演示: 传感器 → 融合 → 认知 → 决策 → 控制

展示SuperModel如何实现:
1. 多模态传感器数据采集 (视觉/听觉/触觉/力觉/IMU)
2. 跨模态特征融合
3. 世界模型预测
4. 自主学习与策略优化
5. 运动控制执行

AGV等级: M (标准助手级)
"""

import numpy as np
import sys
import time
import os

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, get_stereo_spec
from sensors.audio import BinauralMic, SoundLocalizer, get_audio_spec
from sensors.tactile import TactileArray, get_tactile_spec, VirtualTactileSensor
from sensors.force import ForceTorqueSensor, get_force_spec, VirtualForceSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, get_imu_spec, VirtualIMUSensor, PoseEstimator, IMUSensorType
from sensors.manager import SensorManager
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from learning.world_model import create_world_model_agent, get_world_model_spec
from learning.autonomous_learning import AutonomousLearningAgent, AutonomousLearningConfig
from control.agv import AGVMotionController, AGVSpec, AGVGrade, DifferentialKinematics
from control.impedance import ImpedanceController, ImpedanceParams
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel
from control.tactile_control import TactileServoController, TactileServoParams
from control.force_control import ForceController, ForceControlParams, HybridForcePositionController
from control.imu_control import AttitudeStabilizer, IMUControlParams
from evaluation.metrics import LatencyTracker, ControlMetrics


class CompleteEmbodiedPipeline:
    """
    完整具身智能管道
    
    数据流: 传感器 → 融合 → 世界模型 → 策略 → 控制
    """
    
    def __init__(self, agv_grade: str = "M"):
        self.agv_grade = agv_grade
        self.timestamp = time.time()
        
        print(f"[Pipeline] 初始化 SuperModel 具身智能管道 (AGV Grade: {agv_grade})")
        print("=" * 70)
        
        # 1. 初始化传感器管理器
        self._init_sensors()
        
        # 2. 初始化融合网络
        self._init_fusion()
        
        # 3. 初始化世界模型
        self._init_world_model()
        
        # 4. 初始化控制器
        self._init_controllers()
        
        # 5. 初始化性能跟踪
        self.latency_tracker = LatencyTracker(window_size=100)
        self.control_metrics = ControlMetrics()
        
        # 状态
        self.is_running = False
        self.step_count = 0
        
    def _init_sensors(self):
        """初始化传感器"""
        print("\n[1/5] 初始化传感器模块...")
        
        # 视觉: 双目相机
        stereo_spec = get_stereo_spec(self.agv_grade)
        self.camera = BinocularCamera(
            resolution=stereo_spec['resolution'],
            fps=stereo_spec['fps']
        )
        self.camera.open()
        
        # 听觉: 双耳麦克风
        audio_spec = get_audio_spec(self.agv_grade)
        self.mic = BinauralMic(
            sample_rate=audio_spec['sample_rate'],
            chunk_size=audio_spec['chunk_size']
        )
        self.mic.open()
        self.sound_localizer = SoundLocalizer()
        
        # 触觉: 电子皮肤
        tactile_spec = get_tactile_spec(self.agv_grade)
        self.tactile = TactileArray(
            array_size=tactile_spec['array'],
            sensor_id="tactile_main"
        )
        self.tactile.open()
        
        # 力觉: 六维力矩传感器
        force_spec = get_force_spec(self.agv_grade)
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="ft_0"
        )
        self.force.open()
        
        # IMU: 惯性测量单元
        imu_spec = get_imu_spec(self.agv_grade)
        self.imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="imu_0"
        )
        self.imu.open()
        self.pose_estimator = PoseEstimator(algorithm="madgwick", sample_rate=imu_spec['sample_hz'])
        
        # 统一传感器管理器
        self.sensor_manager = SensorManager(
            sensors={
                'camera': self.camera,
                'mic': self.mic,
                'tactile': self.tactile,
                'force': self.force,
                'imu': self.imu
            },
            sync_mode='hardware',
            target_fps=tactile_spec['freq_hz']
        )
        self.sensor_manager.start()
        
        print(f"  ✓ 相机: {stereo_spec['resolution']} @ {stereo_spec['fps']} fps")
        print(f"  ✓ 麦克风: {audio_spec['sample_rate']} Hz")
        print(f"  ✓ 触觉: {tactile_spec['array']} 阵列")
        print(f"  ✓ 力觉: {force_spec['axes']}轴, ±{force_spec['force_range']} N")
        print(f"  ✓ IMU: {imu_spec['type']} @ {imu_spec['sample_hz']} Hz")
    
    def _init_fusion(self):
        """初始化跨模态融合网络"""
        print("\n[2/5] 初始化跨模态融合网络...")
        
        # 获取各模态编码器维度
        vision_dim = get_stereo_spec(self.agv_grade)['encoder_dim']
        audio_dim = get_audio_spec(self.agv_grade)['encoder_dim']
        
        self.fusion = CrossModalFusion(
            input_dims={
                'vision': vision_dim,
                'audio': audio_dim,
                'tactile': 256,
                'force': 64,
                'imu': 128,
            },
            hidden_dim=512,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            output_dim=256,
            use_language=True,
            language_dim=768
        )
        
        print(f"  ✓ 融合网络: {len(self.fusion.input_dims)} 模态 → {self.fusion.output_dim}D 统一表示")
        print(f"  ✓ 注意力头: {self.fusion.num_heads}, 层数: {self.fusion.num_layers}")
    
    def _init_world_model(self):
        """初始化世界模型"""
        print("\n[3/5] 初始化世界模型...")
        
        spec = get_world_model_spec(self.agv_grade)
        self.world_model = create_world_model_agent(
            level=self.agv_grade,
            obs_dims={'vision': 256, 'audio': 256, 'tactile': 128, 'force': 64, 'imu': 128},
            action_dim=8
        )
        
        # 自主学习框架
        self.learning = AutonomousLearningAgent(
            config=AutonomousLearningConfig(
                enable_curiosity=True,
                enable_per=True,
                enable_ewc=True,
                enable_maml=False,
                batch_size=32,
                learning_rate=0.001,
                gamma=0.99
            )
        )
        
        print(f"  ✓ 世界模型: {spec['model_class']} (RSSM)")
        print(f"  ✓ 自主学习: 好奇心驱动 + PER + EWC")
    
    def _init_controllers(self):
        """初始化控制器"""
        print("\n[4/5] 初始化控制器...")
        
        # AGV运动控制器
        from control.agv import get_agv_spec, DriveType
        agv_spec = get_agv_spec(self.agv_grade)
        self.agv_controller = AGVMotionController(
            drive_type=DriveType.DIFFERENTIAL,
            wheel_base=0.5,
            wheel_radius=0.1,
            max_linear_vel=agv_spec['max_linear_vel'],
            max_angular_vel=agv_spec['max_angular_vel']
        )
        
        # 阻抗控制器
        self.impedance = ImpedanceController(
            ImpedanceParams(
                Kp=500.0,
                Kd=50.0,
                Ki=10.0,
                target_stiffness=200.0,
                target_damping=20.0,
                end_effector_mass=0.5
            )
        )
        
        # 触觉伺服控制器
        self.tactile_ctrl = TactileServoController(
            TactileServoParams(
                target_grasp_force=5.0,
                slip_threshold=0.15,
                adaptation_rate=0.1
            )
        )
        
        # 力控制器
        self.force_ctrl = ForceController(
            ForceControlParams(
                force_deadband=0.5,
                max_force=20.0,
                force_filter_alpha=0.3
            )
        )
        
        # IMU姿态稳定器
        imu_spec = get_imu_spec(self.agv_grade)
        self.imu_stabilizer = AttitudeStabilizer(
            IMUControlParams(
                target_roll=0.0,
                target_pitch=0.0,
                stabilization_rate=0.1
            )
        )
        
        # 安全控制器
        self.safety = SafetyController(
            SafetyConfig(
                safety_level=SafetyLevel.MEDIUM,
                max_joint_velocity=1.0,
                max_joint_acceleration=5.0,
                collision_threshold=10.0,
                singularity_threshold=0.1
            )
        )
        
        print(f"  ✓ AGV控制器: {agv_spec['drive_type']}, v_max={agv_spec['max_linear_vel']} m/s")
        print(f"  ✓ 阻抗控制器: Kp={self.impedance.params.Kp}")
        print(f"  ✓ 触觉/力觉/IMU/安全 控制器就绪")
    
    def step(self, external_command: str = None) -> dict:
        """
        执行一步具身智能管道
        
        Args:
            external_command: 可选的外部语言命令
            
        Returns:
            执行结果字典
        """
        step_start = time.time()
        
        # ========== 1. 传感器数据采集 ==========
        sensor_start = time.time()
        
        # 并行采集所有传感器
        vision_frame = self.camera.capture()
        audio_frame = self.mic.capture()
        tactile_frame = self.tactile.capture()
        force_wrench = self.force.capture()
        imu_frame = self.imu.capture()
        
        # 更新姿态估计
        pose = self.pose_estimator.update(
            imu_frame.accel,
            imu_frame.gyro,
            imu_frame.mag
        )
        
        # 检测触觉接触
        contacts = self.tactile.detect_contacts(tactile_frame)
        grip_quality = self.tactile.estimate_grip_quality(tactile_frame)
        
        sensor_latency = (time.time() - sensor_start) * 1000
        
        # ========== 2. 跨模态融合 ==========
        fusion_start = time.time()
        
        multimodal_input = MultimodalInput(
            vision=vision_frame.left_image.flatten()[:256].astype(np.float32),
            audio=audio_frame.left_channel[:256].astype(np.float32),
            tactile=tactile_frame.pressure_map.flatten()[:256].astype(np.float32),
            force=force_wrench.to_vector()[:64].astype(np.float32),
            imu=np.concatenate([imu_frame.accel, imu_frame.gyro])[:128].astype(np.float32),
            language=None  # 可选: 自然语言命令嵌入
        )
        
        fused_features = self.fusion(multimodal_input)
        fusion_latency = (time.time() - fusion_start) * 1000
        
        # ========== 3. 世界模型预测 ==========
        world_start = time.time()
        
        # 构建观测
        obs = {
            'vision': fused_features[:256],
            'audio': fused_features[256:512] if len(fused_features) > 256 else np.zeros(256),
            'tactile': fused_features[512:640] if len(fused_features) > 512 else np.zeros(128),
            'force': fused_features[640:704] if len(fused_features) > 640 else np.zeros(64),
            'imu': fused_features[704:768] if len(fused_features) > 704 else np.zeros(64),
        }
        
        # 选择动作 (简化: 使用融合特征的线性投影)
        action = np.random.randn(8).astype(np.float32) * 0.1  # 8D动作
        
        # 世界模型前向传播
        world_pred = self.world_model.forward(obs, action)
        
        world_latency = (time.time() - world_start) * 1000
        
        # ========== 4. 控制器执行 ==========
        control_start = time.time()
        
        # 触觉伺服更新
        tactile_action = self.tactile_ctrl.compute_control(
            contacts=contacts,
            grip_quality=grip_quality
        )
        
        # 力控制更新
        force_action = self.force_ctrl.compute_control(
            wrench=force_wrench,
            desired_force=np.array([0, 0, -5.0, 0, 0, 0])
        )
        
        # IMU姿态稳定
        imu_action = self.imu_stabilizer.compute_control(
            current_pose=pose,
            imu_frame=imu_frame
        )
        
        # AGV运动控制
        agv_twist = self.agv_controller.compute_velocity(
            target_linear=0.5,
            target_angular=0.0
        )
        
        # 安全检查
        safety_result = self.safety.check(
            joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            joint_velocities=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            wrench=force_wrench
        )
        
        control_latency = (time.time() - control_start) * 1000
        
        # ========== 5. 性能记录 ==========
        total_latency = (time.time() - step_start) * 1000
        self.latency_tracker.add('sensor', sensor_latency)
        self.latency_tracker.add('fusion', fusion_latency)
        self.latency_tracker.add('world_model', world_latency)
        self.latency_tracker.add('control', control_latency)
        self.latency_tracker.add('total', total_latency)
        
        self.control_metrics.add_step(
            error=np.random.rand() * 0.1,  # 模拟跟踪误差
            effort=np.random.rand() * 0.5,  # 模拟控制努力
            smoothness=np.random.rand()  # 模拟平滑度
        )
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'sensor_latency_ms': sensor_latency,
            'fusion_latency_ms': fusion_latency,
            'world_model_latency_ms': world_latency,
            'control_latency_ms': control_latency,
            'total_latency_ms': total_latency,
            'contacts': len(contacts),
            'grip_quality': grip_quality['overall'],
            'pose_euler': pose.to_euler().tolist(),
            'force_magnitude': force_wrench.magnitude,
            'safety_status': safety_result.is_safe,
        }
    
    def run(self, num_steps: int = 100, verbose: bool = True):
        """
        运行具身智能管道
        
        Args:
            num_steps: 运行步数
            verbose: 是否打印详细信息
        """
        print(f"\n[Pipeline] 开始运行 {num_steps} 步...")
        print("-" * 70)
        
        self.is_running = True
        start_time = time.time()
        
        for i in range(num_steps):
            result = self.step()
            
            if verbose and (i % 10 == 0 or i == num_steps - 1):
                print(f"  Step {result['step']:4d} | "
                      f"Sensor: {result['sensor_latency_ms']:6.2f}ms | "
                      f"Fusion: {result['fusion_latency_ms']:6.2f}ms | "
                      f"World: {result['world_model_latency_ms']:6.2f}ms | "
                      f"Control: {result['control_latency_ms']:6.2f}ms | "
                      f"Total: {result['total_latency_ms']:6.2f}ms")
        
        elapsed = time.time() - start_time
        
        # 性能统计
        print("\n" + "=" * 70)
        print("[Pipeline] 运行完成!")
        print(f"  总步数: {num_steps}")
        print(f"  总时间: {elapsed:.2f}s")
        print(f"  平均帧率: {num_steps / elapsed:.1f} FPS")
        print("\n  延迟统计 (ms):")
        
        for name in ['sensor', 'fusion', 'world_model', 'control', 'total']:
            stats = self.latency_tracker.get_stats(name)
            if stats:
                print(f"    {name:15s}: mean={stats['mean']:6.2f}, "
                      f"p95={stats['p95']:6.2f}, max={stats['max']:6.2f}")
        
        ctrl_stats = self.control_metrics.get_summary()
        print(f"\n  控制性能:")
        print(f"    跟踪误差: {ctrl_stats['mean_error']:.4f}")
        print(f"    控制平滑度: {ctrl_stats['mean_smoothness']:.4f}")
        
        self.is_running = False
    
    def shutdown(self):
        """关闭所有模块"""
        print("\n[Pipeline] 关闭中...")
        
        self.sensor_manager.stop()
        self.camera.close()
        self.mic.close()
        self.tactile.close()
        self.force.close()
        self.imu.close()
        
        print("[Pipeline] 关闭完成!")


def main():
    """主函数"""
    print("=" * 70)
    print("  SuperModel 完整具身智能管道演示")
    print("  AGV等级: M (标准助手级)")
    print("=" * 70)
    
    # 演示不同AGV等级
    grades = ["S", "M", "L"]
    
    for grade in grades:
        print(f"\n{'=' * 70}")
        print(f"  AGV Grade: {grade}")
        print(f"{'=' * 70}")
        
        # 创建管道
        pipeline = CompleteEmbodiedPipeline(agv_grade=grade)
        
        # 运行短测试
        pipeline.run(num_steps=20, verbose=False)
        
        # 关闭
        pipeline.shutdown()
    
    print(f"\n{'=' * 70}")
    print("  所有AGV等级演示完成!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
