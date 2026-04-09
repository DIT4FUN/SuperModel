"""
自适应增益调度模块
=================

根据运行状态、负载变化、环境条件自动调整控制器增益

功能:
- 基于误差的自适应增益调度
- 基于负载估计的增益调整
- 基于温度的补偿增益
- 基于速度/加速度的前馈增益
- 多模态增益融合

支持等级: M/L/XL/XXL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Callable
from enum import Enum


class AdaptationStrategy(Enum):
    """自适应策略"""
    ERROR_BASED = "error_based"       # 基于跟踪误差
    LOAD_BASED = "load_based"         # 基于负载估计
    TEMP_BASED = "temp_based"         # 基于温度补偿
    VELOCITY_BASED = "velocity_based" # 基于速度
    MULTI_MODAL = "multi_modal"       # 多模态融合


@dataclass
class GainSchedule:
    """增益调度配置"""
    kp_base: float = 10.0
    ki_base: float = 1.0
    kd_base: float = 2.0
    kf_base: float = 0.5             # 前馈增益
    schedule_type: str = "linear"     # linear/polynomial/exponential
    bounds: Tuple[float, float] = (0.1, 10.0)  # 增益缩放范围


@dataclass
class AdaptationState:
    """自适应状态"""
    current_kp: float = 10.0
    current_ki: float = 1.0
    current_kd: float = 2.0
    current_kf: float = 0.5
    error_integral: float = 0.0
    error_derivative: float = 0.0
    last_error: float = 0.0
    adaptation_ratio: float = 1.0    # 当前自适应比例
    confidence: float = 1.0          # 增益置信度
    timestamp: float = 0.0


class AdaptiveGainScheduler:
    """
    自适应增益调度器

    根据系统状态自动调整 PID 增益:
    - 误差大时增大 P，误差小时减小 P
    - 负载变化时调整积分增益
    - 高速运动时调整 D 和前馈增益
    - 温度变化时补偿增益漂移
    """

    def __init__(
        self,
        strategy: AdaptationStrategy = AdaptationStrategy.MULTI_MODAL,
        schedule: Optional[GainSchedule] = None,
        scheduler_id: str = "adaptive_gain_0"
    ):
        self.strategy = strategy
        self.schedule = schedule or GainSchedule()
        self.scheduler_id = scheduler_id

        # 状态
        self._state = AdaptationState()
        self._is_enabled = True

        # 权重配置 (多模态策略下各因子权重)
        self._weights = {
            'error': 0.4,
            'load': 0.2,
            'temperature': 0.1,
            'velocity': 0.3
        }

        # 缓存
        self._history: List[AdaptationState] = []
        self._max_history = 100

        # 回调
        self._on_gain_change: Optional[Callable] = None

    def update(
        self,
        error: float,
        dt: float,
        load_estimate: float = 1.0,
        temperature: float = 25.0,
        velocity: float = 0.0,
        acceleration: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        更新增益调度

        Args:
            error: 跟踪误差
            dt: 时间步长
            load_estimate: 负载估计 (相对值, 1.0=额定负载)
            temperature: 当前温度 (摄氏度)
            velocity: 当前速度 (m/s)
            acceleration: 当前加速度 (m/s^2)

        Returns:
            (kp, ki, kd, kf): 调整后的增益
        """
        if not self._is_enabled:
            return (self.schedule.kp_base, self.schedule.ki_base,
                    self.schedule.kd_base, self.schedule.kf_base)

        # 更新误差积分和微分
        self._state.error_integral += error * dt
        self._state.error_derivative = (error - self._state.last_error) / max(dt, 1e-6)
        self._state.last_error = error

        # 计算各因子增益调整
        if self.strategy == AdaptationStrategy.ERROR_BASED:
            ratio = self._compute_error_ratio(error)
        elif self.strategy == AdaptationStrategy.LOAD_BASED:
            ratio = self._compute_load_ratio(load_estimate)
        elif self.strategy == AdaptationStrategy.TEMP_BASED:
            ratio = self._compute_temp_ratio(temperature)
        elif self.strategy == AdaptationStrategy.VELOCITY_BASED:
            ratio = self._compute_velocity_ratio(velocity, acceleration)
        else:  # MULTI_MODAL
            r_error = self._compute_error_ratio(error)
            r_load = self._compute_load_ratio(load_estimate)
            r_temp = self._compute_temp_ratio(temperature)
            r_vel = self._compute_velocity_ratio(velocity, acceleration)
            ratio = (self._weights['error'] * r_error +
                     self._weights['load'] * r_load +
                     self._weights['temperature'] * r_temp +
                     self._weights['velocity'] * r_vel)

        # 应用边界约束
        ratio = np.clip(ratio, self.schedule.bounds[0], self.schedule.bounds[1])
        self._state.adaptation_ratio = ratio

        # 计算调整后增益
        kp = self.schedule.kp_base * ratio
        ki = self.schedule.ki_base * ratio
        kd = self.schedule.kd_base * ratio
        kf = self.schedule.kf_base * (1.0 / max(ratio, 0.1))  # 前馈与反馈反向

        # 更新状态
        self._state.current_kp = kp
        self._state.current_ki = ki
        self._state.current_kd = kd
        self._state.current_kf = kf

        # 计算置信度
        self._state.confidence = min(ratio / self.schedule.bounds[1] * 2, 1.0)

        # 历史记录
        import time
        self._state.timestamp = time.time()
        self._history.append(self._state)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # 触发回调
        if self._on_gain_change:
            self._on_gain_change(kp, ki, kd, kf)

        return kp, ki, kd, kf

    def _compute_error_ratio(self, error: float) -> float:
        """基于误差计算增益比例"""
        abs_error = abs(error)
        # 误差越大，增益越高
        if self.schedule.schedule_type == "linear":
            ratio = 1.0 + np.clip(abs_error * 2.0, 0, 5.0)
        elif self.schedule.schedule_type == "exponential":
            ratio = np.exp(np.clip(abs_error, 0, 2.0))
        else:  # polynomial
            ratio = 1.0 + abs_error**1.5
        return np.clip(ratio, self.schedule.bounds[0], self.schedule.bounds[1])

    def _compute_load_ratio(self, load: float) -> float:
        """基于负载计算增益比例"""
        # 负载越大，需要更大的增益
        ratio = 1.0 + (load - 1.0) * 0.5
        return np.clip(ratio, self.schedule.bounds[0], self.schedule.bounds[1])

    def _compute_temp_ratio(self, temperature: float) -> float:
        """基于温度计算增益比例"""
        # 温度偏离25°C越多，补偿越大
        delta_t = abs(temperature - 25.0)
        # 典型热膨胀系数导致的增益漂移约0.1%/°C
        ratio = 1.0 - 0.001 * delta_t
        return np.clip(ratio, self.schedule.bounds[0], self.schedule.bounds[1])

    def _compute_velocity_ratio(self, velocity: float, acceleration: float) -> float:
        """基于速度计算增益比例"""
        v_mag = abs(velocity)
        a_mag = abs(acceleration)
        # 高速时增强微分和前馈
        ratio = 1.0 + v_mag * 0.1 + a_mag * 0.05
        return np.clip(ratio, self.schedule.bounds[0], self.schedule.bounds[1])

    def get_gains(self) -> Dict[str, float]:
        """获取当前增益"""
        return {
            'kp': self._state.current_kp,
            'ki': self._state.current_ki,
            'kd': self._state.current_kd,
            'kf': self._state.current_kf,
            'ratio': self._state.adaptation_ratio,
            'confidence': self._state.confidence
        }

    def reset(self):
        """重置自适应状态"""
        self._state = AdaptationState()
        self._history.clear()

    def enable(self):
        """启用自适应调度"""
        self._is_enabled = True

    def disable(self):
        """禁用自适应调度（使用基础增益）"""
        self._is_enabled = False

    def set_weights(self, error: float = 0.4, load: float = 0.2,
                    temperature: float = 0.1, velocity: float = 0.3):
        """设置多模态策略下各因子权重"""
        total = error + load + temperature + velocity
        self._weights = {
            'error': error / total,
            'load': load / total,
            'temperature': temperature / total,
            'velocity': velocity / total
        }

    def on_gain_change(self, callback: Callable):
        """注册增益变化回调"""
        self._on_gain_change = callback

    def get_history(self, n: int = 10) -> List[Dict]:
        """获取最近 n 次增益记录"""
        history = self._history[-n:]
        return [
            {'kp': s.current_kp, 'ki': s.current_ki, 'kd': s.current_kd,
             'kf': s.current_kf, 'ratio': s.adaptation_ratio,
             'timestamp': s.timestamp}
            for s in history
        ]


class GainBlendController:
    """
    增益混合控制器

    在多个增益配置之间平滑切换:
    - 抓取 vs 释放
    - 高速 vs 低速
    - 柔顺 vs 刚性
    """

    def __init__(
        self,
        controller_id: str = "blend_0",
        blend_time: float = 0.5
    ):
        self.controller_id = controller_id
        self.blend_time = blend_time

        # 增益配置
        self._configs: Dict[str, GainSchedule] = {}
        self._current_config: Optional[str] = None
        self._target_config: Optional[str] = None

        # 混合状态
        self._blend_alpha: float = 1.0  # 0=当前, 1=目标
        self._blend_start_frame: int = -1  # 确定性帧计数，避免 wall-clock 抖动
        self._is_blending: bool = False

        self._current_gains = GainSchedule()
        self._target_gains = GainSchedule()

    def register_config(self, name: str, schedule: GainSchedule):
        """注册增益配置"""
        self._configs[name] = schedule
        if self._current_config is None:
            self._current_config = name
            self._current_gains = schedule

    def switch_config(self, name: str, blend: bool = True):
        """切换增益配置"""
        if name not in self._configs:
            raise ValueError(f"Unknown config: {name}")

        if not blend or name == self._current_config:
            self._current_config = name
            self._current_gains = self._configs[name]
            self._is_blending = False
            self._blend_alpha = 1.0
            return

        self._target_config = name
        self._target_gains = self._configs[name]
        self._is_blending = True
        self._blend_alpha = 0.0
        self._blend_start_frame = -1  # 首次 update 时设置

    def update(self, dt: float) -> GainSchedule:
        """更新混合状态，返回当前插值增益
        
        使用 perf_counter_ns (纳秒精度) 避免毫秒级 timing 歧义，
        确保 blend 持续时间精确可预测。
        """
        import time as _time_module
        if self._is_blending:
            if self._blend_start_frame < 0:
                # 首次调用：记录纳秒级起始时间，本次不推进 blend
                self._blend_start_frame = _time_module.perf_counter_ns()
                return self._current_gains

            elapsed_ns = _time_module.perf_counter_ns() - self._blend_start_frame
            elapsed_s = elapsed_ns / 1e9

            if elapsed_s >= self.blend_time:
                # blend 完成
                self._blend_alpha = 1.0
                self._current_config = self._target_config
                self._current_gains = self._target_gains
                self._is_blending = False
                self._blend_start_frame = -1
            else:
                self._blend_alpha = min(elapsed_s / self.blend_time, 1.0)
                t = self._ease_in_out(self._blend_alpha)
                self._current_gains = GainSchedule(
                    kp_base=self._lerp(self._current_gains.kp_base, self._target_gains.kp_base, t),
                    ki_base=self._lerp(self._current_gains.ki_base, self._target_gains.ki_base, t),
                    kd_base=self._lerp(self._current_gains.kd_base, self._target_gains.kd_base, t),
                    kf_base=self._lerp(self._current_gains.kf_base, self._target_gains.kf_base, t),
                    schedule_type=self._target_gains.schedule_type,
                    bounds=self._target_gains.bounds
                )

        return self._current_gains

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _ease_in_out(self, t: float) -> float:
        """缓入缓出插值"""
        return t * t * (3 - 2 * t)

    def get_current_config_name(self) -> Optional[str]:
        """获取当前配置名称"""
        return self._current_config

    def is_blending(self) -> bool:
        """是否正在混合切换"""
        return self._is_blending


# AGV五级自适应增益规格
AGV_ADAPTIVE_GAIN_GRADES = {
    'S':  {'enabled': False, 'strategy': None, 'max_rate': 0.0},
    'M':  {'enabled': True, 'strategy': 'error_based', 'max_rate': 10.0},
    'L':  {'enabled': True, 'strategy': 'multi_modal', 'max_rate': 20.0},
    'XL': {'enabled': True, 'strategy': 'multi_modal', 'max_rate': 50.0},
    'XXL': {'enabled': True, 'strategy': 'multi_modal', 'max_rate': 100.0},
}


def get_adaptive_gain_spec(grade: str) -> dict:
    """获取AGV指定等级的自适应增益规格"""
    return AGV_ADAPTIVE_GAIN_GRADES.get(grade, AGV_ADAPTIVE_GAIN_GRADES['M'])


class ModelReferenceAdaptiveController:
    """
    模型参考自适应控制 (MRAC)

    基于参考模型的参数自适应控制:
    - 适用于非线性系统
    - 参数在线估计
    - 稳定性保证 (Lyapunov)

    支持等级: L/XL/XXL
    """

    def __init__(
        self,
        reference_model: Callable,
        controller_id: str = "mrac_0",
        adaptation_gain: float = 10.0
    ):
        self.reference_model = reference_model
        self.controller_id = controller_id
        self.adaptation_gain = adaptation_gain

        # 参数估计
        self._theta = np.zeros(3)  # 3个可调参数
        self._phi = np.zeros(3)    # 回归向量

        # 状态
        self._reference_state = 0.0
        self._system_state = 0.0
        self._adaptation_rate = adaptation_gain

    def compute_control(
        self,
        system_state: float,
        reference_input: float,
        dt: float
    ) -> float:
        """
        计算自适应控制量

        Args:
            system_state: 系统当前状态
            reference_input: 参考输入
            dt: 时间步长

        Returns:
            控制输入
        """
        self._system_state = system_state

        # 1. 参考模型输出
        self._reference_state = self.reference_model(reference_input, dt)

        # 2. 跟踪误差
        error = self._reference_state - system_state

        # 3. 回归向量 (简化的线性参数化)
        self._phi = np.array([
            system_state,
            reference_input,
            1.0
        ])

        # 4. 参数自适应律 (梯度法)
        theta_dot = self.adaptation_gain * error * self._phi
        self._theta += theta_dot * dt

        # 5. 控制量
        control = self._theta @ self._phi

        return float(control)

    def get_parameters(self) -> np.ndarray:
        """获取当前参数估计"""
        return self._theta.copy()

    def reset(self):
        """重置参数"""
        self._theta = np.zeros(3)
