#!/usr/bin/env python3
"""
SuperModel 多传感器数据采集脚本
==============================

用于采集、记录和分析多模态传感器数据
支持实时数据流、离线回放和数据统计

运行: python3 scripts/multi_sensor_data_collection.py --grade M --duration 60 --output data/
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sensors.vision import BinocularCamera, DepthProcessor
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from sensors.force import ForceTorqueSensor, ForceSensorType, Wrench
from sensors.imu import IMUSensor, IMUSensorType, PoseEstimator, get_imu_spec


class MultiSensorCollector:
    """
    多传感器数据采集器
    
    功能:
    - 同步采集多模态传感器数据
    - 数据缓存和批量写入
    - 实时统计和监控
    - 数据回放支持
    """
    
    def __init__(
        self,
        grade: str = "M",
        output_dir: str = "./data",
        buffer_size: int = 100,
        capture_rate: int = None
    ):
        self.grade = grade
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        
        # 采样率配置
        self.sample_rates = {
            'vision': 30,
            'audio': 16000,
            'tactile': get_tactile_spec(grade)['freq_hz'],
            'force': get_force_spec(grade)['sampling_hz'],
            'imu': get_imu_spec(grade)['sample_hz']
        }
        if capture_rate:
            self.capture_rate = capture_rate
        else:
            # 使用最低采样率
            self.capture_rate = min(self.sample_rates.values())
        
        # 采集会话ID
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = None
        
        # 传感器实例
        self.vision_cam: Optional[BinocularCamera] = None
        self.audio_mic: Optional[BinauralMic] = None
        self.tactile: Optional[TactileArray] = None
        self.force: Optional[ForceTorqueSensor] = None
        self.imu: Optional[IMUSensor] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        
        # 数据缓冲
        self.data_buffer: List[Dict] = []
        self.stats = {
            'frames_collected': 0,
            'vision_frames': 0,
            'audio_frames': 0,
            'tactile_frames': 0,
            'force_frames': 0,
            'imu_frames': 0,
            'dropped_frames': 0
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 元数据文件
        self.metadata_file = self.session_dir / "metadata.json"
        
    def _create_metadata(self) -> Dict:
        """创建采集元数据"""
        from sensors.vision import get_stereo_spec
        from sensors.audio import get_audio_spec
        from sensors.tactile import get_tactile_spec
        from sensors.force import get_force_spec
        from sensors.imu import get_imu_spec
        
        return {
            'session_id': self.session_id,
            'grade': self.grade,
            'start_time': datetime.now().isoformat(),
            'capture_rate_hz': self.capture_rate,
            'sample_rates': self.sample_rates,
            'specs': {
                'vision': get_stereo_spec(self.grade),
                'audio': get_audio_spec(self.grade),
                'tactile': get_tactile_spec(self.grade),
                'force': get_force_spec(self.grade),
                'imu': get_imu_spec(self.grade)
            },
            'sensors': {
                'vision': {'enabled': True},
                'audio': {'enabled': True},
                'tactile': {'array_size': list(get_tactile_spec(self.grade)['array'])},
                'force': {'axes': get_force_spec(self.grade)['axes']},
                'imu': {'type': get_imu_spec(self.grade)['type']}
            }
        }
    
    def open(self):
        """打开所有传感器"""
        print(f"[Collector] Opening sensors for AGV-{self.grade}...")
        
        # 元数据
        metadata = self._create_metadata()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 视觉
        print("  Opening BinocularCamera...")
        self.vision_cam = BinocularCamera()
        self.vision_cam.open()
        
        # 音频
        print("  Opening BinauralMic...")
        self.audio_mic = BinauralMic(sample_rate=16000)
        self.audio_mic.open()
        
        # 触觉
        tactile_spec = get_tactile_spec(self.grade)
        print(f"  Opening TactileArray {tactile_spec['array']}...")
        self.tactile = TactileArray(
            array_size=tactile_spec['array'],
            sensor_type=TactileSensorType.CAPACITIVE
        )
        self.tactile.open()
        
        # 力矩
        force_spec = get_force_spec(self.grade)
        print(f"  Opening ForceTorqueSensor ({force_spec['axes']}-axis)...")
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS if force_spec['axes'] == 6 else ForceSensorType.THREE_AXIS
        )
        self.force.open()
        
        # IMU
        imu_spec = get_imu_spec(self.grade)
        print(f"  Opening IMUSensor ({imu_spec['type']})...")
        self.imu = IMUSensor(
            sensor_type=IMUSensorType[imu_spec['type'].upper().replace('-', '_')]
        )
        self.imu.open()
        
        # 姿态估计器
        self.pose_estimator = PoseEstimator(
            algorithm='madgwick',
            sample_rate=imu_spec['sample_hz']
        )
        
        print(f"[Collector] All sensors opened")
        print(f"[Collector] Session: {self.session_id}")
        print(f"[Collector] Output: {self.session_dir}")
        
        return True
    
    def close(self):
        """关闭所有传感器"""
        print("[Collector] Closing sensors...")
        
        if self.vision_cam:
            self.vision_cam.close()
        if self.audio_mic:
            self.audio_mic.close()
        if self.tactile:
            self.tactile.close()
        if self.force:
            self.force.close()
        if self.imu:
            self.imu.close()
        
        # 保存缓冲数据
        self._flush_buffer()
        
        # 更新元数据
        metadata = json.loads(self.metadata_file.read_text())
        metadata['end_time'] = datetime.now().isoformat()
        metadata['stats'] = self.stats
        metadata['duration_sec'] = time.time() - self.start_time if self.start_time else 0
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Collector] Session ended. Collected {self.stats['frames_collected']} frames")
    
    def capture_frame(self) -> Dict:
        """采集一帧多模态数据"""
        frame_data = {
            'timestamp': time.time(),
            'frame_id': self.stats['frames_collected']
        }
        
        # 视觉 (降采样以匹配capture_rate)
        if self.vision_cam and self.stats['frames_collected'] % (30 // self.capture_rate) == 0:
            try:
                stereo = self.vision_cam.capture()
                frame_data['vision'] = {
                    'left_shape': list(stereo.left_image.shape),
                    'right_shape': list(stereo.right_image.shape),
                    'timestamp': stereo.timestamp
                }
                self.stats['vision_frames'] += 1
            except Exception as e:
                frame_data['vision'] = {'error': str(e)}
        
        # 音频
        if self.audio_mic:
            try:
                audio = self.audio_mic.capture()
                frame_data['audio'] = {
                    'left_samples': len(audio.left_channel),
                    'right_samples': len(audio.right_channel),
                    'sample_rate': audio.sample_rate,
                    'timestamp': audio.timestamp
                }
                self.stats['audio_frames'] += 1
            except Exception as e:
                frame_data['audio'] = {'error': str(e)}
        
        # 触觉
        if self.tactile:
            try:
                tac_frame = self.tactile.capture()
                contacts = self.tactile.detect_contacts(tac_frame)
                frame_data['tactile'] = {
                    'pressure_shape': list(tac_frame.pressure_map.shape),
                    'num_contacts': len(contacts),
                    'max_pressure': float(np.max(tac_frame.pressure_map)) if tac_frame.pressure_map.size > 0 else 0.0,
                    'timestamp': tac_frame.timestamp
                }
                self.stats['tactile_frames'] += 1
            except Exception as e:
                frame_data['tactile'] = {'error': str(e)}
        
        # 力矩
        if self.force:
            try:
                wrench = self.force.capture()
                frame_data['force'] = {
                    'force': wrench.force.tolist(),
                    'torque': wrench.torque.tolist(),
                    'force_magnitude': float(wrench.magnitude),
                    'timestamp': wrench.timestamp
                }
                self.stats['force_frames'] += 1
            except Exception as e:
                frame_data['force'] = {'error': str(e)}
        
        # IMU
        if self.imu:
            try:
                imu_frame = self.imu.capture()
                pose = self.pose_estimator.update(imu_frame.accel, imu_frame.gyro)
                euler = pose.to_euler()
                frame_data['imu'] = {
                    'accel': imu_frame.accel.tolist(),
                    'gyro': imu_frame.gyro.tolist(),
                    'accel_magnitude': float(imu_frame.accel_magnitude),
                    'euler_deg': (euler * 180 / np.pi).tolist(),
                    'timestamp': imu_frame.timestamp
                }
                self.stats['imu_frames'] += 1
            except Exception as e:
                frame_data['imu'] = {'error': str(e)}
        
        self.stats['frames_collected'] += 1
        return frame_data
    
    def collect(
        self,
        duration_sec: float = 60.0,
        progress_interval: float = 5.0,
        save_interval: int = 100
    ):
        """
        执行数据采集
        
        Args:
            duration_sec: 采集时长 (秒)
            progress_interval: 进度报告间隔 (秒)
            save_interval: 缓冲保存间隔 (帧数)
        """
        self.start_time = time.time()
        end_time = self.start_time + duration_sec
        
        print(f"\n[Collector] Starting collection for {duration_sec}s...")
        print(f"[Collector] Capture rate: {self.capture_rate} Hz")
        print(f"[Collector] Buffer size: {self.buffer_size}")
        
        last_progress = self.start_time
        frame_times = []
        
        try:
            while time.time() < end_time:
                loop_start = time.time()
                
                # 采集
                frame = self.capture_frame()
                self.data_buffer.append(frame)
                
                # 定期保存
                if len(self.data_buffer) >= self.buffer_size:
                    self._flush_buffer()
                
                # 进度报告
                current_time = time.time()
                if current_time - last_progress >= progress_interval:
                    elapsed = current_time - self.start_time
                    fps = self.stats['frames_collected'] / elapsed if elapsed > 0 else 0
                    progress = elapsed / duration_sec * 100
                    print(f"  [{progress:5.1f}%] {elapsed:.1f}s | {fps:.1f} fps | "
                          f"frames={self.stats['frames_collected']} | "
                          f"vision={self.stats['vision_frames']} | "
                          f"imu={self.stats['imu_frames']}")
                    last_progress = current_time
                
                # 控制采集速率
                loop_time = time.time() - loop_start
                target_interval = 1.0 / self.capture_rate
                if loop_time < target_interval:
                    time.sleep(target_interval - loop_time)
                
                frame_times.append(time.time() - loop_start)
                
        except KeyboardInterrupt:
            print("\n[Collector] Interrupted by user")
        
        # 最终统计
        if frame_times:
            avg_frame_time = np.mean(frame_times)
            print(f"\n[Collector] Average frame time: {avg_frame_time*1000:.2f} ms")
            print(f"[Collector] Estimated max rate: {1.0/avg_frame_time:.1f} fps")
    
    def _flush_buffer(self):
        """将缓冲数据写入磁盘"""
        if not self.data_buffer:
            return
        
        # 写入JSONL文件
        data_file = self.session_dir / f"data_{len(self.data_buffer)}.jsonl"
        with open(data_file, 'w') as f:
            for frame in self.data_buffer:
                f.write(json.dumps(frame) + '\n')
        
        print(f"[Collector] Flushed {len(self.data_buffer)} frames to {data_file.name}")
        self.data_buffer = []
    
    def print_summary(self):
        """打印采集统计摘要"""
        duration = time.time() - self.start_time if self.start_time else 0
        print("\n" + "=" * 50)
        print("  数据采集统计摘要")
        print("=" * 50)
        print(f"  会话ID: {self.session_id}")
        print(f"  AGV等级: {self.grade}")
        print(f"  采集时长: {duration:.1f} 秒")
        print(f"  总帧数: {self.stats['frames_collected']}")
        print(f"  视觉帧: {self.stats['vision_frames']}")
        print(f"  音频帧: {self.stats['audio_frames']}")
        print(f"  触觉帧: {self.stats['tactile_frames']}")
        print(f"  力矩帧: {self.stats['force_frames']}")
        print(f"  IMU帧: {self.stats['imu_frames']}")
        print(f"  数据目录: {self.session_dir}")
        print("=" * 50)
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.print_summary()
        self.close()


def main():
    parser = argparse.ArgumentParser(description='SuperModel 多传感器数据采集')
    parser.add_argument('--grade', type=str, default='M', choices=['S', 'M', 'L', 'XL', 'XXL'],
                        help='AGV等级')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='采集时长 (秒)')
    parser.add_argument('--output', type=str, default='./data',
                        help='输出目录')
    parser.add_argument('--buffer-size', type=int, default=100,
                        help='缓冲大小')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     SuperModel 多传感器数据采集工具                           ║
║     AGV-{args.grade} 级 | {args.duration}秒 | 输出: {args.output}                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    collector = MultiSensorCollector(
        grade=args.grade,
        output_dir=args.output,
        buffer_size=args.buffer_size
    )
    
    with collector:
        collector.collect(duration_sec=args.duration)
    
    print("\n✅ 数据采集完成!")


if __name__ == '__main__':
    main()
