"""
vla_inference.py - VLA 推理管道 + 具身Pipeline集成
SuperModel 超模态大模型具身智能系统

VLA推理管道:
- 感知数据预处理
- VLA模型推理
- 动作后处理
- 闭环执行监控
- 安全_shield集成
- 具身Pipeline桥接

功能:
- 实时VLA推理循环
- 动作平滑 (指数移动平均)
- 安全检查与覆盖
- 传感器反馈闭环
- 联邦学习VLA聚合
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .vla_model import (
    VLAModel, VLAConfig, VLAInput, VLAOutput, VLAAction, VLAPerceptionFrame,
    VLAActionSpace, VLAGrade, create_vla_model,
)
from .embodied_skill import EmbodiedSkillRegistry, get_global_skill_registry

logger = logging.getLogger(__name__)

__all__ = [
    'InferencePolicy',
    'ActionSmoothingMode',
    'VLAInferencePipeline',
    'VLAPipelineConfig',
    'create_vla_inference_pipeline',
]


# ============================================================
# 推理策略
# ============================================================

class InferencePolicy(Enum):
    """推理策略"""
    SINGLE_SHOT = "single_shot"       # 单步推理
    CONTINUOUS = "continuous"          # 连续推理循环
    TRIGGERED = "triggered"            # 指令触发推理
    PREDICTIVE = "predictive"         # 预测性推理 (提前多步)


class ActionSmoothingMode(Enum):
    """动作平滑模式"""
    NONE = "none"
    EMA = "ema"                       # 指数移动平均
    LOW_PASS = "low_pass"            # 一阶低通滤波
    KALMAN = "kalman"                # 卡尔曼滤波


@dataclass
class VLAPipelineConfig:
    """VLA推理管道配置"""
    # 模型配置
    grade: str = "M"
    action_space: VLAActionSpace = VLAActionSpace.TWIST
    inference_policy: InferencePolicy = InferencePolicy.CONTINUOUS
    smoothing_mode: ActionSmoothingMode = ActionSmoothingMode.EMA
    
    # 频率配置
    inference_hz: float = 10.0       # 推理频率
    max_inference_hz: float = 30.0   # 最大推理频率
    
    # 平滑参数
    ema_alpha: float = 0.7           # EMA平滑因子
    low_pass_cutoff: float = 5.0     # 低通滤波截止频率 (Hz)
    
    # 安全配置
    safety_enabled: bool = True
    max_linear_speed: float = 2.0    # m/s
    max_angular_speed: float = math.pi / 2  # rad/s
    min_clearance: float = 0.3       # m, 最小安全距离
    
    # 动作执行
    action_timeout_s: float = 1.0    # 单步动作超时
    max_queue_size: int = 10          # 动作队列大小
    
    # 传感器配置
    use_camera: bool = True
    use_lidar: bool = True
    use_proprioception: bool = True
    
    # Pipeline集成
    integrate_with_pipeline: bool = True
    enable_feedback_loop: bool = True
    
    def __post_init__(self):
        pass


# ============================================================
# 动作平滑器
# ============================================================

class ActionSmoother:
    """
    动作平滑器 - 减少动作抖动
    
    支持:
    - EMA (指数移动平均)
    - 一阶低通滤波
    - 卡尔曼滤波 (简化版)
    """
    
    def __init__(self, mode: ActionSmoothingMode = ActionSmoothingMode.EMA, alpha: float = 0.7):
        self.mode = mode
        self.alpha = alpha
        
        self._ema_prev: Optional[VLAAction] = None
        self._lp_prev: Optional[VLAAction] = None
        self._kalman_x: Optional[VLAAction] = None  # 状态估计
        self._kalman_p: float = 1.0  # 协方差
        
        # 噪声参数 (卡尔曼)
        self._q = 0.01  # 过程噪声
        self._r = 0.1   # 观测噪声
    
    def smooth(self, action: VLAAction) -> VLAAction:
        """平滑动作"""
        if self.mode == ActionSmoothingMode.NONE:
            return action
        
        elif self.mode == ActionSmoothingMode.EMA:
            return self._smooth_ema(action)
        
        elif self.mode == ActionSmoothingMode.LOW_PASS:
            return self._smooth_low_pass(action)
        
        elif self.mode == ActionSmoothingMode.KALMAN:
            return self._smooth_kalman(action)
        
        return action
    
    def _smooth_ema(self, action: VLAAction) -> VLAAction:
        """指数移动平均"""
        if self._ema_prev is None:
            self._ema_prev = action
            return action
        
        smoothed = VLAAction()
        smoothed.action_space = action.action_space
        
        # EMA: smoothed = alpha * current + (1-alpha) * prev
        smoothed.vx = self.alpha * action.vx + (1 - self.alpha) * self._ema_prev.vx
        smoothed.vy = self.alpha * action.vy + (1 - self.alpha) * self._ema_prev.vy
        smoothed.vz = self.alpha * action.vz + (1 - self.alpha) * self._ema_prev.vz
        smoothed.rx = self.alpha * action.rx + (1 - self.alpha) * self._ema_prev.rx
        smoothed.ry = self.alpha * action.ry + (1 - self.alpha) * self._ema_prev.ry
        smoothed.rz = self.alpha * action.rz + (1 - self.alpha) * self._ema_prev.rz
        smoothed.gripper_position = self.alpha * action.gripper_position + (1 - self.alpha) * self._ema_prev.gripper_position
        smoothed.gripper_force = self.alpha * action.gripper_force + (1 - self.alpha) * self._ema_prev.gripper_force
        smoothed.confidence = action.confidence
        smoothed.reasoning = action.reasoning
        
        self._ema_prev = smoothed
        return smoothed
    
    def _smooth_low_pass(self, action: VLAAction) -> VLAAction:
        """一阶低通滤波 (与EMA等价)"""
        return self._smooth_ema(action)  # 本质相同
    
    def _smooth_kalman(self, action: VLAAction) -> VLAAction:
        """简化卡尔曼滤波"""
        if self._kalman_x is None:
            self._kalman_x = action
            return action
        
        # 预测步
        self._kalman_p = self._kalman_p + self._q
        
        # 更新步
        meas_noise = 1.0 - action.confidence + 0.01  # 观测噪声
        k = self._kalman_p / (self._kalman_p + meas_noise * self._r)
        
        smoothed = VLAAction()
        smoothed.action_space = action.action_space
        
        # 状态更新
        smoothed.vx = self._kalman_x.vx + k * (action.vx - self._kalman_x.vx)
        smoothed.vy = self._kalman_x.vy + k * (action.vy - self._kalman_x.vy)
        smoothed.vz = self._kalman_x.vz + k * (action.vz - self._kalman_x.vz)
        smoothed.rx = self._kalman_x.rx + k * (action.rx - self._kalman_x.rx)
        smoothed.ry = self._kalman_x.ry + k * (action.ry - self._kalman_x.ry)
        smoothed.rz = self._kalman_x.rz + k * (action.rz - self._kalman_x.rz)
        smoothed.gripper_position = self._kalman_x.gripper_position + k * (action.gripper_position - self._kalman_x.gripper_position)
        smoothed.gripper_force = self._kalman_x.gripper_force + k * (action.gripper_force - self._kalman_x.gripper_force)
        smoothed.confidence = action.confidence
        
        # 协方差更新
        self._kalman_p = (1 - k) * self._kalman_p
        
        self._kalman_x = smoothed
        return smoothed
    
    def reset(self):
        """重置平滑器状态"""
        self._ema_prev = None
        self._lp_prev = None
        self._kalman_x = None
        self._kalman_p = 1.0


# ============================================================
# VLA推理管道
# ============================================================

class VLAInferencePipeline:
    """
    VLA推理管道 - 端到端具身智能执行循环
    
    架构:
    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
    │  Sensor     │───▶│ VLA Model   │───▶│ Safety Check │
    │  Input      │    │ Inference   │    │ & Clipping   │
    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                  │
    ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
    │  Action     │◀───│ Smoother    │◀───│ Action      │
    │  Executor   │    │             │    │ PostProcess │
    └─────────────┘    └─────────────┘    └─────────────┘
    
    使用示例:
        pipeline = VLAInferencePipeline(grade="M")
        pipeline.start()
        
        # 注册传感器数据回调
        pipeline.register_sensor_callback('camera', my_camera_cb)
        pipeline.register_sensor_callback('lidar', my_lidar_cb)
        
        # 发送指令
        pipeline.set_instruction("go to station A")
        
        # 获取动作
        action = pipeline.get_latest_action()
        
        pipeline.stop()
    """
    
    def __init__(
        self,
        config: Optional[VLAPipelineConfig] = None,
        vla_model: Optional[VLAModel] = None,
        skill_registry: Optional[EmbodiedSkillRegistry] = None,
    ):
        self.config = config or VLAPipelineConfig()
        self.vla = vla_model or create_vla_model(grade=self.config.grade)
        self.skill_registry = skill_registry or get_global_skill_registry()
        
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 传感器回调
        self._sensor_callbacks: Dict[str, Callable] = {}
        self._latest_sensor_data: Dict[str, Any] = {}
        
        # 当前指令
        self._current_instruction: str = ""
        
        # 动作输出
        self._latest_action: Optional[VLAAction] = None
        self._action_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        
        # 平滑器
        self.smoother = ActionSmoother(
            mode=self.config.smoothing_mode,
            alpha=self.config.ema_alpha,
        )
        
        # 统计
        self._stats = {
            'total_inferences': 0,
            'total_safety_overrides': 0,
            'avg_inference_time_ms': 0.0,
            'last_inference_time_ms': 0.0,
            'fps': 0.0,
        }
        self._stats_lock = threading.Lock()
        
        # Pipeline集成回调
        self._pipeline_callback: Optional[Callable] = None
        
        # 动作执行回调 (发送给机器人)
        self._action_executor_cb: Optional[Callable[[VLAAction], None]] = None
        
        # 反馈闭环
        self._feedback_history: List[Tuple[VLAAction, VLAPerceptionFrame]] = []
        self._feedback_max_len = 100
        
        logger.info(f"VLAInferencePipeline created: grade={self.config.grade}, "
                    f"policy={self.config.inference_policy.value}")
    
    # ============================================================
    # 生命周期
    # ============================================================
    
    def start(self):
        """启动推理管道"""
        if self._is_running:
            logger.warning("VLAInferencePipeline already running")
            return
        
        self._is_running = True
        self.vla.start()
        self.smoother.reset()
        
        if self.config.inference_policy == InferencePolicy.CONTINUOUS:
            self._thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._thread.start()
        
        logger.info("VLAInferencePipeline started")
    
    def stop(self):
        """停止推理管道"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        self.vla.stop()
        logger.info(f"VLAInferencePipeline stopped. Stats: {self._stats}")
    
    # ============================================================
    # 传感器数据输入
    # ============================================================
    
    def register_sensor_callback(self, sensor_type: str, callback: Callable[[], Any]):
        """
        注册传感器数据回调
        
        Args:
            sensor_type: 'camera', 'lidar', 'imu', 'tactile', 'force', 'joint_states'
            callback: 返回最新传感器数据的回调函数
        """
        self._sensor_callbacks[sensor_type] = callback
        logger.debug(f"Registered sensor callback: {sensor_type}")
    
    def push_sensor_data(self, sensor_type: str, data: Any):
        """推送传感器数据 (替代回调方式)"""
        self._latest_sensor_data[sensor_type] = data
    
    def set_instruction(self, instruction: str):
        """设置当前指令"""
        with self._lock:
            self._current_instruction = instruction
        logger.debug(f"Instruction set: {instruction[:50]}...")
    
    def set_perception_frame(self, perception: VLAPerceptionFrame):
        """设置感知帧 (直接接口)"""
        with self._lock:
            if self._current_instruction:
                perception.instruction = self._current_instruction
            self._latest_sensor_data['perception'] = perception
    
    # ============================================================
    # 动作输出
    # ============================================================
    
    def get_latest_action(self, timeout: Optional[float] = None) -> Optional[VLAAction]:
        """获取最新动作 (阻塞或非阻塞)"""
        try:
            if timeout is None:
                return self._latest_action
            return self._action_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def register_action_executor(self, executor: Callable[[VLAAction], None]):
        """注册动作执行回调"""
        self._action_executor_cb = executor
    
    # ============================================================
    # 推理循环
    # ============================================================
    
    def _inference_loop(self):
        """连续推理循环"""
        import math
        
        dt = 1.0 / self.config.inference_hz
        next_tick = time.time()
        
        while self._is_running:
            try:
                # 等待节拍
                sleep_time = next_tick - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_tick += dt
                
                # 限制最大频率
                if time.time() - next_tick > 1.0:
                    next_tick = time.time() + dt
                
                # 获取传感器数据
                perception = self._build_perception_frame()
                
                if perception is None:
                    continue
                
                # VLA推理
                vla_input = VLAInput(
                    perception=perception,
                    history_actions=self.vla._history[-5:],
                )
                
                output = self.vla.step(vla_input)
                action = output.action
                
                # 动作后处理
                action = self._post_process_action(action, perception)
                
                # 动作平滑
                action = self.smoother.smooth(action)
                
                # 安全检查
                action, overridden = self._safety_check(action, perception)
                
                # 更新最新动作
                self._latest_action = action
                
                # 放入队列 (非阻塞)
                try:
                    self._action_queue.put_nowait(action)
                except queue.Full:
                    try:
                        self._action_queue.get_nowait()
                        self._action_queue.put_nowait(action)
                    except queue.Empty:
                        pass
                
                # 反馈记录
                self._feedback_history.append((action, perception))
                if len(self._feedback_history) > self._feedback_max_len:
                    self._feedback_history.pop(0)
                
                # 发送到执行器
                if self._action_executor_cb:
                    self._action_executor_cb(action)
                
                # Pipeline集成
                if self._pipeline_callback:
                    self._pipeline_callback(output)
                
                # 更新统计
                self._update_stats(output, overridden)
                
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                if not self._is_running:
                    break
                time.sleep(0.1)
    
    def _build_perception_frame(self) -> Optional[VLAPerceptionFrame]:
        """从传感器回调构建感知帧"""
        perception = VLAPerceptionFrame()
        
        with self._lock:
            self._current_instruction = self._current_instruction
        
        # 相机
        if self.config.use_camera and 'camera' in self._sensor_callbacks:
            try:
                rgb = self._sensor_callbacks['camera']()
                if rgb is not None:
                    perception.rgb_image = rgb
            except Exception as e:
                logger.warning(f"Camera callback error: {e}")
        
        if 'camera' in self._latest_sensor_data:
            perception.rgb_image = self._latest_sensor_data.get('camera')
        
        # 激光雷达
        if self.config.use_lidar and 'lidar' in self._sensor_callbacks:
            try:
                lidar_data = self._sensor_callbacks['lidar']()
                if lidar_data is not None:
                    if isinstance(lidar_data, tuple):
                        perception.lidar_scan, perception.lidar_angles = lidar_data
                    else:
                        perception.lidar_scan = lidar_data
            except Exception as e:
                logger.warning(f"Lidar callback error: {e}")
        
        if 'lidar' in self._latest_sensor_data:
            lidar_data = self._latest_sensor_data.get('lidar')
            if isinstance(lidar_data, tuple):
                perception.lidar_scan, perception.lidar_angles = lidar_data
            else:
                perception.lidar_scan = lidar_data
        
        # 本体感知
        if self.config.use_proprioception:
            if 'joint_states' in self._sensor_callbacks:
                try:
                    perception.joint_states = self._sensor_callbacks['joint_states']()
                except Exception:
                    pass
            if 'joint_states' in self._latest_sensor_data:
                perception.joint_states = self._latest_sensor_data.get('joint_states')
            
            if 'base_pose' in self._latest_sensor_data:
                perception.base_pose = self._latest_sensor_data.get('base_pose')
        
        # 深度图像
        if 'depth' in self._latest_sensor_data:
            perception.depth_image = self._latest_sensor_data.get('depth')
        
        # 指令
        with self._lock:
            perception.instruction = self._current_instruction
        
        # 如果没有任何传感器数据，跳过
        if perception.get_modalities() == ['language'] or perception.get_modalities() == []:
            if not perception.instruction:
                return None
        
        return perception
    
    def _post_process_action(self, action: VLAAction, perception: VLAPerceptionFrame) -> VLAAction:
        """动作后处理"""
        import math
        
        # 速度裁剪
        speed = math.sqrt(action.vx**2 + action.vy**2)
        if speed > self.config.max_linear_speed:
            scale = self.config.max_linear_speed / speed
            action.vx *= scale
            action.vy *= scale
        
        # 角速度裁剪
        rot_speed = abs(action.rz)
        if rot_speed > self.config.max_angular_speed:
            action.rz = math.copysign(self.config.max_angular_speed, action.rz)
        
        # 夹爪范围
        action.gripper_position = max(-1.0, min(1.0, action.gripper_position))
        action.gripper_force = max(0.0, min(100.0, action.gripper_force))
        
        return action
    
    def _safety_check(self, action: VLAAction, perception: VLAPerceptionFrame) -> Tuple[VLAAction, bool]:
        """安全检查"""
        import math
        
        overridden = False
        
        if not self.config.safety_enabled:
            return action, overridden
        
        # 激光雷达障碍物检测
        if perception.lidar_scan is not None:
            min_range = perception.lidar_scan.min()
            
            if min_range < self.config.min_clearance:
                # 紧急停止
                action.vx = 0.0
                action.vy = 0.0
                action.vz = 0.0
                action.rx = 0.0
                action.ry = 0.0
                action.rz = 0.0
                action.reasoning = f"Emergency stop: obstacle at {min_range:.2f}m"
                overridden = True
                
            elif min_range < self.config.min_clearance * 2:
                # 减速
                action.vx *= 0.3
                action.vy *= 0.3
                action.reasoning = f"Slow down: obstacle at {min_range:.2f}m"
                overridden = True
        
        # 低电量保护
        if perception.battery_level < 0.1:
            action.vx *= 0.5
            action.reasoning = f"Low battery: {perception.battery_level*100:.0f}%"
        
        return action, overridden
    
    def _update_stats(self, output: VLAOutput, overridden: bool):
        """更新统计信息"""
        with self._stats_lock:
            self._stats['total_inferences'] += 1
            if overridden:
                self._stats['total_safety_overrides'] += 1
            
            t = output.inference_time_ms
            n = self._stats['total_inferences']
            prev_avg = self._stats['avg_inference_time_ms']
            self._stats['avg_inference_time_ms'] = (prev_avg * (n - 1) + t) / n
            self._stats['last_inference_time_ms'] = t
            
            # FPS
            if t > 0:
                self._stats['fps'] = 1000.0 / t
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    def trigger_inference(self, perception: VLAPerceptionFrame) -> VLAOutput:
        """
        触发单次推理 (用于TRIGGERED策略)
        
        Args:
            perception: 感知帧
            
        Returns:
            VLAOutput: 推理输出
        """
        if self._current_instruction:
            perception.instruction = self._current_instruction
        
        vla_input = VLAInput(
            perception=perception,
            history_actions=self.vla._history[-5:],
        )
        
        output = self.vla.step(vla_input)
        action = self._post_process_action(output.action, perception)
        action = self.smoother.smooth(action)
        action, overridden = self._safety_check(action, perception)
        
        self._latest_action = action
        self._update_stats(output, overridden)
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管道统计"""
        with self._stats_lock:
            stats = dict(self._stats)
        
        stats.update(self.vla.get_stats())
        stats['queue_size'] = self._action_queue.qsize()
        stats['is_running'] = self._is_running
        stats['feedback_history_len'] = len(self._feedback_history)
        
        return stats
    
    def get_feedback_history(self) -> List[Tuple[VLAAction, VLAPerceptionFrame]]:
        """获取反馈历史"""
        return list(self._feedback_history)
    
    def reset(self):
        """重置管道状态"""
        self.vla.reset_history()
        self.smoother.reset()
        self._feedback_history.clear()
        
        with self._lock:
            self._current_instruction = ""
            self._latest_sensor_data.clear()
        
        try:
            while True:
                self._action_queue.get_nowait()
        except queue.Empty:
            pass
        
        logger.info("VLAInferencePipeline reset")


# ============================================================
# 工厂函数
# ============================================================

def create_vla_inference_pipeline(
    grade: str = "M",
    policy: InferencePolicy = InferencePolicy.CONTINUOUS,
    smoothing: ActionSmoothingMode = ActionSmoothingMode.EMA,
    integrate_with_pipeline: bool = True,
) -> VLAInferencePipeline:
    """创建VLA推理管道"""
    config = VLAPipelineConfig(
        grade=grade.upper(),
        inference_policy=policy,
        smoothing_mode=smoothing,
        integrate_with_pipeline=integrate_with_pipeline,
    )
    return VLAInferencePipeline(config=config)
