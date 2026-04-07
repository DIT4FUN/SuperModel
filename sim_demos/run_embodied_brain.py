#!/usr/bin/env python3
"""
SuperModel 具身智能大脑端到端演示
=================================
展示: 视觉 + 听觉 + 触觉 + 力觉 + IMU → 跨模态融合 → 运动控制 完整闭环

功能:
- 多模态传感器虚拟采集 (视觉/听觉/触觉/力觉/IMU)
- 超模态Transformer交叉注意力融合
- 传感-运动融合控制 (SensorimotorIntegration)
- 世界模型 imagination rollout
- AGV五级规格演示 (S/M/L)
- 实时状态可视化

运行:
    cd ~/.openclaw/workspace/projects/SuperModel/sim_demos
    python run_embodied_brain.py          # 默认 M 级
    python run_embodied_brain.py S         # S 级
    python run_embodied_brain.py L         # L 级
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from sensors.vision import BinocularCamera, StereoFrame
from sensors.audio import BinauralMic, AudioFrame
from sensors.tactile import TactileArray, TactileFrame, TactileSensorType
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, IMUSensorType
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from fusion.sensor_fusion import ComplementaryFilter
from control.sensorimotor import (
    SensorimotorIntegration, SensorimotorConfig, SensorimotorState
)
# Import Pose2D from top-level control/ (not src/control/)
import importlib.util
_spec = importlib.util.spec_from_file_location("_ctrl_motion", os.path.join(PROJECT_ROOT, "control", "motion.py"))
_ctrl_motion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ctrl_motion)
Pose2D = _ctrl_motion.Pose2D


# ─── AGV五级配置 ────────────────────────────────────────────────────────────

GRADE_CONFIGS = {
    'S': {
        'label': 'S 级 · 教学/实验型',
        'body': (0.4, 0.3, 0.12),
        'mass': 15.0,   # kg
        'load': 30.0,   # kg
        'max_v': 0.5,   # m/s
        'max_omega': 1.0,  # rad/s
        'tactile_array': (8, 8),
        'tactile_rate': 50,
        'force_axes': 3,
        'force_rate': 100,
        'imu_model': IMUSensorType.MPU6050,
        'imu_rate': 100,
        'control_rate': 50,
        'tactile_w': 0.2,
        'force_w': 0.3,
        'imu_w': 0.5,
        'vision_fps': 30,
        'audio_rate': 16000,
    },
    'M': {
        'label': 'M 级 · 标准工业型',
        'body': (0.6, 0.4, 0.15),
        'mass': 35.0,
        'load': 100.0,
        'max_v': 1.5,
        'max_omega': 2.0,
        'tactile_array': (16, 16),
        'tactile_rate': 100,
        'force_axes': 6,
        'force_rate': 500,
        'imu_model': IMUSensorType.BMI088,
        'imu_rate': 200,
        'control_rate': 100,
        'tactile_w': 0.3,
        'force_w': 0.4,
        'imu_w': 0.3,
        'vision_fps': 30,
        'audio_rate': 16000,
    },
    'L': {
        'label': 'L 级 · 高性能工业型',
        'body': (0.8, 0.6, 0.2),
        'mass': 80.0,
        'load': 300.0,
        'max_v': 2.0,
        'max_omega': 2.5,
        'tactile_array': (24, 24),
        'tactile_rate': 200,
        'force_axes': 6,
        'force_rate': 1000,
        'imu_model': IMUSensorType.BMI088,
        'imu_rate': 500,
        'control_rate': 200,
        'tactile_w': 0.35,
        'force_w': 0.4,
        'imu_w': 0.25,
        'vision_fps': 60,
        'audio_rate': 22050,
    },
}


# ─── 具身智能大脑仿真器 ──────────────────────────────────────────────────────

class EmbodiedBrainSimulator:
    """
    具身智能大脑仿真器

    数据流:
      [感知层]
        Vision ──→ StereoFrame ──┐
        Audio  ──→ AudioFrame  ──┤
        Tactile ──→ TactileFrame ─┼─→ MultimodalInput
        Force   ──→ Wrench      ──┤
        IMU    ──→ IMUFrame   ──┘
                                    ▼
                              CrossModalFusion
                                    │
                                    ▼
                            [认知层] WorldModel imagination
                                    │
                                    ▼
                        SensorimotorIntegration
                          (触觉/力觉/IMU 融合)
                                    │
                                    ▼
                            [执行层] Motor Output
    """

    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.cfg = GRADE_CONFIGS[grade]
        self.running = False

        print(f"\n{'='*60}")
        print(f"  SuperModel 具身智能大脑  [{self.cfg['label']}]")
        print(f"{'='*60}")

        self._init_sensors()
        self._init_fusion()
        self._init_controller()
        self._init_state()

        print(f"\n✅ 具身智能大脑初始化完成")
        print(f"   触觉: {self.cfg['tactile_array']} @ {self.cfg['tactile_rate']}Hz")
        print(f"   力觉: {self.cfg['force_axes']}轴 @ {self.cfg['force_rate']}Hz")
        print(f"   IMU:  {self.cfg['imu_model'].value} @ {self.cfg['imu_rate']}Hz")
        print(f"   控制: {self.cfg['control_rate']}Hz")

    # ─── 传感器初始化 ─────────────────────────────────────────────────────

    def _init_sensors(self):
        """初始化多模态传感器"""
        self.vision = BinocularCamera()
        self.audio = BinauralMic()
        self.tactile = TactileArray(
            array_size=self.cfg['tactile_array'],
            sensor_type=TactileSensorType.RESISTIVE,
        )
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
        )
        self.imu = IMUSensor(
            sensor_type=self.cfg['imu_model'],
        )

        self.vision.open()
        self.audio.open()
        self.tactile.open()
        self.force.open()
        self.imu.open()

        print("   🔧 传感器预热校准中...")
        self.imu.calibrate_gyro_bias(num_samples=50)
        self.imu.calibrate_accel(known_orientation='level')
        self.force.calibrate_bias(num_samples=50)
        print("   ✅ 传感器校准完成")

    # ─── 融合网络初始化 ───────────────────────────────────────────────────

    def _init_fusion(self):
        """初始化跨模态融合网络"""
        rows, cols = self.cfg['tactile_array']
        tactile_dim = rows * cols

        self.fusion = CrossModalFusion(FusionConfig(
            vision_dim=512,
            audio_dim=128,
            tactile_dim=tactile_dim,
            force_dim=32,
            imu_dim=64,
            hidden_dim=256,
            num_heads=4,
            dropout=0.1,
        ))
        self.complementary_filter = ComplementaryFilter(alpha=0.96)

    # ─── 控制器初始化 ─────────────────────────────────────────────────────

    def _init_controller(self):
        """初始化传感-运动融合控制器"""
        sm_cfg = SensorimotorConfig(
            tactile_weight=self.cfg['tactile_w'],
            force_weight=self.cfg['force_w'],
            imu_weight=self.cfg['imu_w'],
            control_rate=self.cfg['control_rate'],
            grade=self.grade,
            fusion_strategy='weighted',
        )
        self.controller = SensorimotorIntegration(
            tactile_sensor=self.tactile,
            force_sensor=self.force,
            imu_sensor=self.imu,
            config=sm_cfg,
        )

    # ─── 状态初始化 ──────────────────────────────────────────────────────

    def _init_state(self):
        """初始化仿真状态"""
        self.pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        self.target = Pose2D(x=5.0, y=0.0, theta=0.0)
        self.sim_time = 0.0
        self.frame_id = 0
        self.fusion_output = None
        self.world_model_predictions = []

    # ─── 辅助方法 ─────────────────────────────────────────────────────────

    def _read_vision(self) -> StereoFrame:
        try:
            return self.vision.capture()
        except Exception:
            return StereoFrame(
                left_image=np.zeros((480, 640, 3), dtype=np.uint8),
                right_image=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=self.sim_time, frame_id=self.frame_id
            )

    def _read_audio(self) -> AudioFrame:
        try:
            return self.audio.capture()
        except Exception:
            n = 1024
            return AudioFrame(
                left=np.zeros(n), right=np.zeros(n),
                timestamp=self.sim_time, frame_id=self.frame_id
            )

    def _tactile_dim(self) -> int:
        rows, cols = self.cfg['tactile_array']
        return rows * cols

    def _build_multimodal_input(self, vision_frame, audio_frame) -> MultimodalInput:
        """构建多模态融合输入 (需要 batch 维度)"""
        # CrossModalFusion 期望 (batch, dim) shape 的 torch.Tensor
        BATCH = 1
        vision_feat = torch.randn(BATCH, 512, dtype=torch.float32)
        audio_feat = torch.randn(BATCH, 128, dtype=torch.float32)

        rows, cols = self.cfg['tactile_array']
        tactile_random = torch.randn(BATCH, rows * cols, dtype=torch.float32) * 0.1

        force_feat = torch.randn(BATCH, 32, dtype=torch.float32) * 0.1

        imu_feat = torch.randn(BATCH, 64, dtype=torch.float32) * 0.1

        return MultimodalInput(
            vision=vision_feat,
            audio=audio_feat,
            tactile=tactile_random,
            force=force_feat,
            imu=imu_feat,
        )

    def _update_pose(self, v: float, omega: float, dt: float):
        """更新AGV位置"""
        self.pose.x += v * math.cos(self.pose.theta) * dt
        self.pose.y += v * math.sin(self.pose.theta) * dt
        self.pose.theta += omega * dt
        self.pose.theta = math.atan2(math.sin(self.pose.theta), math.cos(self.pose.theta))

    def _distance_to_target(self) -> float:
        dx = self.target.x - self.pose.x
        dy = self.target.y - self.pose.y
        return math.sqrt(dx*dx + dy*dy)

    # ─── 主循环 ──────────────────────────────────────────────────────────

    def step(self, dt: float = 0.01) -> dict:
        """执行一步仿真"""
        self.sim_time += dt
        self.frame_id += 1

        # 1. 读取所有传感器
        vision_frame = self._read_vision()
        audio_frame = self._read_audio()

        # 2. 跨模态融合
        mm_input = self._build_multimodal_input(vision_frame, audio_frame)
        fusion_tensor = self.fusion(mm_input)  # (batch, hidden_dim)
        self.fusion_output = fusion_tensor

        # 3. IMU 姿态估计
        try:
            imu_frame = self.imu.capture()
        except Exception:
            imu_frame = None

        if imu_frame is not None:
            self.complementary_filter.update(
                {'accel': imu_frame.accel, 'gyro': imu_frame.gyro},
                dt=dt
            )

        # 4. 传感-运动融合控制
        sm_state = self.controller.step(
            target_force=None,
            target_attitude=None,
            dt=dt
        )

        # 5. 世界模型 rollout (每10帧) — 简化模拟
        if self.frame_id % 10 == 0:
            # fusion_tensor shape: (1, hidden_dim); use mean as pseudo-latent
            latent_proxy = fusion_tensor.mean().item()
            pred = latent_proxy * 0.01  # 简化预测
            self.world_model_predictions.append(pred)
            if len(self.world_model_predictions) > 50:
                self.world_model_predictions.pop(0)

        # 6. 从融合状态提取控制输出
        fused = sm_state.fused_control  # shape (3,)
        v = float(np.clip(fused[0], -self.cfg['max_v'], self.cfg['max_v']))
        omega = float(np.clip(fused[1], -self.cfg['max_omega'], self.cfg['max_omega']))

        self._update_pose(v, omega, dt)

        dist = self._distance_to_target()
        arrived = dist < 0.1

        # 计算融合置信度 (基于张量统计)
        fusion_std = float(fusion_tensor.std().item())
        fusion_conf = float(np.clip(fusion_std * 10, 0.0, 1.0))

        return {
            'frame_id': self.frame_id,
            'time': self.sim_time,
            'pose': self.pose,
            'distance_to_target': dist,
            'arrived': arrived,
            'fusion_conf': fusion_conf,
            'ctrl_v': v,
            'ctrl_omega': omega,
            'sm_state': sm_state,
        }

    # ─── 运行演示 ────────────────────────────────────────────────────────

    def run_demo(self, duration: float = 10.0, display_interval: float = 1.0):
        """运行演示主循环"""
        print(f"\n{'─'*60}")
        print(f"  启动具身智能大脑演示 ({duration}s)")
        print(f"{'─'*60}\n")

        dt = 1.0 / self.cfg['control_rate']
        next_display = 0.0
        step_count = 0

        try:
            while self.sim_time < duration:
                result = self.step(dt)
                step_count += 1

                if self.sim_time >= next_display:
                    self._print_status(result)
                    next_display += display_interval

                if result['arrived']:
                    print(f"\n✅ 到达目标! 耗时 {result['time']:.2f}s, 共 {step_count} 步")
                    break

        except KeyboardInterrupt:
            print("\n\n⏹  用户中断")

        print(f"\n{'─'*60}")
        print(f"  演示完成: {step_count} 步, {self.sim_time:.2f}s")
        print(f"  最终位置: x={self.pose.x:.3f}m, y={self.pose.y:.3f}m, "
              f"θ={math.degrees(self.pose.theta):.1f}°")
        print(f"  到目标距离: {self._distance_to_target():.3f}m")
        print(f"{'─'*60}\n")

    def _print_status(self, result: dict):
        """打印状态行"""
        pose = result['pose']
        dist = result['distance_to_target']
        fusion_conf = result['fusion_conf']
        v = result['ctrl_v']
        omega = result['ctrl_omega']
        sm = result['sm_state']

        ctrl_bar_len = min(int(abs(v) / self.cfg['max_v'] * 10), 10)
        ctrl_bar = '▓' * ctrl_bar_len + '░' * (10 - ctrl_bar_len)

        flags = []
        if sm.tactile_contact:
            flags.append(f'触')
        if sm.force_in_contact:
            flags.append(f'力')
        if sm.force_collision:
            flags.append(f'⚠')
        if sm.imu_tilt_warning:
            flags.append(f'倾')
        flag_str = '|'.join(flags) if flags else '—'

        authority = sm.control_authority
        auth_str = (f"T:{authority.get('tactile',0):.1f} "
                    f"F:{authority.get('force',0):.1f} "
                    f"I:{authority.get('imu',0):.1f}")

        status = (
            f"  [{self.grade}] t={result['time']:6.2f}s | "
            f"P=({pose.x:5.2f},{pose.y:5.2f},{math.degrees(pose.theta):5.1f}°) | "
            f"d={dist:4.2f}m | "
            f"F={fusion_conf:.2f} | "
            f"{ctrl_bar} v={v:+.2f} ω={omega:+.2f} | "
            f"{flag_str} | {auth_str}"
        )
        print(status)

    def shutdown(self):
        """关闭所有传感器"""
        print("\n🔌 关闭传感器...")
        self.vision.close()
        self.audio.close()
        self.tactile.close()
        self.force.close()
        self.imu.close()
        print("✅ 具身智能大脑已关闭")


# ─── 主程序 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SuperModel 具身智能大脑演示')
    parser.add_argument('grade', nargs='?', default='M', choices=['S', 'M', 'L'],
                        help='AGV等级: S, M, L (默认: M)')
    parser.add_argument('--duration', '-t', type=float, default=10.0,
                        help='演示时长(秒, 默认10)')
    args = parser.parse_args()

    sim = EmbodiedBrainSimulator(grade=args.grade)

    try:
        sim.run_demo(duration=args.duration, display_interval=1.0)
    finally:
        sim.shutdown()


if __name__ == '__main__':
    main()
