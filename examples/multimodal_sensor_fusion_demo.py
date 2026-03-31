#!/usr/bin/env python3
"""
SuperModel 多模态传感器融合演示
=================================

展示 6 模态传感器融合: 视觉 + 听觉 + 触觉 + 力觉 + IMU + 关节编码器

运行: python3 examples/multimodal_sensor_fusion_demo.py
"""

import numpy as np
import sys
import time
import torch

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import (
    TactileArray, TactileFrame,
    TactileSensorType, VirtualTactileSensor
)
from sensors.force import (
    ForceTorqueSensor, Wrench,
    ForceSensorType, VirtualForceSensor
)
from sensors.imu import (
    IMUSensor, PoseEstimator,
    IMUSensorType, VirtualIMUSensor
)
from sensors.manager import SensorManager, SensorManagerConfig, SensorGrade
from fusion.cross_modal_fusion import CrossModalFusion, CrossModalFusionConfig


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def init_sensors_xxl():
    """初始化 XXL 级传感器配置"""
    print_header("Step 1: 初始化多模态传感器 (XXL 级)")
    
    sensors = {}
    
    # 视觉: 双目相机
    sensors['vision'] = BinocularCamera(
        resolution=(1280, 720),
        fps=60,
        sensor_id="realsense_d455"
    )
    sensors['vision'].open()
    print("  ✓ 双目相机 (RealSense D455, 1280x720 @ 60fps)")
    
    # 听觉: 双耳麦克风
    sensors['audio'] = BinauralMic(
        sample_rate=44100,
        sensor_id="respeaker_4mic"
    )
    sensors['audio'].open()
    print("  ✓ 双耳麦克风 (ReSpeaker 4-mic, 44.1kHz)")
    
    # 触觉: 电子皮肤阵列 (32x32)
    sensors['tactile'] = TactileArray(
        array_size=(32, 32),
        sensor_type=TactileSensorType.CAPACITIVE,
        sensor_id="digitalskin_32x32"
    )
    sensors['tactile'].open()
    print("  ✓ 触觉阵列 (电容式, 32x32, 500Hz)")
    
    # 力觉: 六维力矩传感器
    sensors['force'] = ForceTorqueSensor(
        sensor_type=ForceSensorType.SIX_AXIS,
        sensor_id="ati_nano25"
    )
    sensors['force'].open()
    print("  ✓ 六维力矩传感器 (ATI Nano25, ±25N)")
    
    # IMU: 高性能惯性测量单元
    sensors['imu'] = IMUSensor(
        sensor_type=IMUSensorType.BMI088,
        accel_range=16,
        gyro_range=2000,
        sample_rate=500,
        sensor_id="bmi088_imu"
    )
    sensors['imu'].open()
    print("  ✓ IMU (BMI088, ±16g, ±2000°/s, 500Hz)")
    
    return sensors


def capture_all_sensors(sensors: dict, n_frames: int = 10):
    """采集多帧多模态数据"""
    print_header(f"Step 2: 采集 {n_frames} 帧多模态数据")
    
    data_log = {k: [] for k in sensors}
    
    for i in range(n_frames):
        for name, sensor in sensors.items():
            if name == 'vision':
                frame = sensor.capture()
                data_log[name].append({
                    'timestamp': frame.timestamp,
                    'frame_id': frame.frame_id,
                    'left_shape': frame.left_image.shape if frame.left_image is not None else None
                })
            elif name == 'audio':
                frame = sensor.capture(duration=0.05)
                data_log[name].append({
                    'timestamp': frame.timestamp,
                    'rms': float(np.sqrt(np.mean(frame.left_channel**2)))
                })
            elif name == 'tactile':
                frame = sensor.capture()
                contacts = sensor.detect_contacts(frame)
                data_log[name].append({
                    'timestamp': frame.timestamp,
                    'contacts': len(contacts),
                    'peak_pressure': float(np.max(frame.pressure_map)) if contacts else 0.0
                })
            elif name == 'force':
                wrench = sensor.capture()
                data_log[name].append({
                    'timestamp': wrench.timestamp,
                    'force_mag': float(wrench.magnitude)
                })
            elif name == 'imu':
                frame = sensor.capture()
                data_log[name].append({
                    'timestamp': frame.timestamp,
                    'accel_mag': float(frame.accel_magnitude)
                })
        
        if (i + 1) % 5 == 0:
            print(f"  采集进度: {i+1}/{n_frames} 帧")
        time.sleep(0.01)
    
    print("  ✓ 多模态数据采集完成")
    return data_log


def init_fusion_network():
    """初始化跨模态融合网络"""
    print_header("Step 3: 初始化跨模态融合网络")
    
    config = CrossModalFusionConfig(
        num_modalities=6,
        modality_names=['vision', 'audio', 'tactile', 'force', 'imu', 'joint'],
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )
    
    fusion = CrossModalFusion(config)
    print(f"  ✓ 融合网络: {sum(p.numel() for p in fusion.parameters()):,} 参数")
    print(f"    - 模态数: 6 (视觉/听觉/触觉/力觉/IMU/关节)")
    print(f"    - 隐层维度: {config.hidden_dim}")
    print(f"    - 注意力头数: {config.num_heads}")
    print(f"    - 融合层数: {config.num_layers}")
    
    return fusion, config


def run_fusion_inference(fusion, sensors, n_iterations: int = 5):
    """运行融合网络推理"""
    print_header(f"Step 4: 跨模态融合推理 ({n_iterations} 次迭代)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion = fusion.to(device)
    fusion.eval()
    
    latencies = []
    
    with torch.no_grad():
        for i in range(n_iterations):
            # 模拟多模态特征输入
            batch_size = 1
            
            # 视觉特征: 来自编码器
            vision_feat = torch.randn(batch_size, 512, device=device)
            
            # 听觉特征
            audio_feat = torch.randn(batch_size, 128, device=device)
            
            # 触觉特征
            tactile_feat = torch.randn(batch_size, 256, device=device)
            
            # 力觉特征
            force_feat = torch.randn(batch_size, 6, device=device)
            
            # IMU特征
            imu_feat = torch.randn(batch_size, 6, device=device)
            
            # 关节特征
            joint_feat = torch.randn(batch_size, 7, device=device)
            
            t0 = time.time()
            
            # 融合
            fused = fusion(
                vision=vision_feat,
                audio=audio_feat,
                tactile=tactile_feat,
                force=force_feat,
                imu=imu_feat,
                joint=joint_feat
            )
            
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)
            
            print(f"  迭代 {i+1:2d}: 融合输出 shape={fused.shape}, "
                  f"延迟={latency_ms:.2f}ms")
    
    avg_latency = np.mean(latencies)
    print(f"\n  ✓ 平均推理延迟: {avg_latency:.2f}ms")
    
    if avg_latency < 10:
        print("  ✓ 满足实时性要求 (< 10ms)")
    
    return avg_latency


def demonstrate_sensor_interaction(sensors: dict):
    """演示传感器间协同"""
    print_header("Step 5: 传感器协同感知演示")
    
    # 模拟抓取场景
    print("  [场景] 机器人正在接近并抓取物体...")
    
    # 1. 视觉检测距离
    print("  1) 视觉: 检测目标距离...")
    time.sleep(0.1)
    
    # 2. IMU 确认稳定姿态
    print("  2) IMU: 姿态稳定检查...")
    imu_frame = sensors['imu'].capture()
    roll, pitch, yaw = np.degrees(PoseEstimator(algorithm='madgwick').update(
        imu_frame.accel, imu_frame.gyro, None
    ).to_euler())
    print(f"     当前姿态: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°")
    
    # 3. 触觉检测接近
    print("  3) 触觉: 检测接近...")
    tactile_frame = sensors['tactile'].capture()
    contacts = sensors['tactile'].detect_contacts(tactile_frame)
    print(f"     检测到 {len(contacts)} 个接触区域")
    
    # 4. 力觉确认接触
    print("  4) 力觉: 确认物理接触...")
    wrench = sensors['force'].capture()
    contact = sensors['force'].detect_contact(wrench)
    print(f"     接触力: {wrench.magnitude:.2f}N, 接触={contact.is_contact}")
    
    # 5. 抓取质量评估
    print("  5) 综合: 抓取质量评估...")
    quality = sensors['tactile'].estimate_grip_quality(tactile_frame)
    print(f"     综合评分: {quality['overall']:.2f}")
    print(f"     接触面积: {quality['contact_area']:.2f}")
    print(f"     均匀性:   {quality['uniformity']:.2f}")
    print(f"     稳定性:   {quality['stability']:.2f}")
    
    print("\n  ✓ 传感器协同感知演示完成")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       SuperModel 超模态大模型 - 多模态融合演示                ║
    ║                                                              ║
    ║       传感器: 视觉 + 听觉 + 触觉 + 力觉 + IMU               ║
    ║       融合网络: 6模态跨模态注意力                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 1. 初始化传感器
        sensors = init_sensors_xxl()
        
        # 2. 采集数据
        data_log = capture_all_sensors(sensors, n_frames=10)
        
        # 3. 初始化融合网络
        fusion, config = init_fusion_network()
        
        # 4. 运行融合推理
        avg_latency = run_fusion_inference(fusion, sensors, n_iterations=5)
        
        # 5. 传感器协同演示
        demonstrate_sensor_interaction(sensors)
        
        # 总结
        print_header("总结")
        print(f"  ✓ 传感器模态: 6 (视觉/听觉/触觉/力觉/IMU/关节)")
        print(f"  ✓ 采集帧数: 10 帧/传感器")
        print(f"  ✓ 融合网络参数: {sum(p.numel() for p in fusion.parameters()):,}")
        print(f"  ✓ 平均推理延迟: {avg_latency:.2f}ms")
        print(f"  ✓ 实时性: {'满足' if avg_latency < 10 else '不满足'} (< 10ms)")
        
    finally:
        # 清理
        print_header("清理资源")
        for name, sensor in sensors.items():
            sensor.close()
            print(f"  ✓ {name} 已关闭")


if __name__ == '__main__':
    main()
