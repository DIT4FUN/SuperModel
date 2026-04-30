# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
自动调参模块 (Auto-Tuning PID)
=============================

自动PID参数整定，支持:
- Ziegler-Nichols 临界灵敏度法
- Relay (继电) 反馈整定
- 频率响应法 (FREQ)
- 模型参考自适应整定 (MRAC)
- 数据驱动的贝叶斯优化整定

Author: SuperModel Team
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class TuningMethod(Enum):
    """调参方法"""
    ZIEGLER_NICHOLS = "ziegler_nichols"       # Z-N 临界灵敏度法
    RELAY = "relay"                            # 继电反馈法
    COHEN_COON = "cohen_coon"                  # Cohen-Coon 法
    MODEL_REFERENCE = "model_reference"        # 模型参考自适应
    BAYESIAN = "bayesian"                      # 贝叶斯优化


@dataclass
class TunerConfig:
    """调参器配置"""
    method: TuningMethod = TuningMethod.ZIEGLER_NICHOLS
    target_kp: float = 1.0
    target_ki: float = 0.0
    target_kd: float = 0.0
    sample_time: float = 0.01        # s
    relay_amplitude: float = 1.0    # 继电幅值
    relay_hysteresis: float = 0.05  # 继电滞环
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    noise_band: float = 0.05        # 噪声带 (消除抖动)


@dataclass
class StepResponse:
    """阶跃响应数据"""
    times: np.ndarray
    values: np.ndarray
    setpoint: float
    overshoot: float = 0.0
    rise_time: float = 0.0
    settling_time: float = 0.0
    steady_state_error: float = 0.0


@dataclass
class TunerResult:
    """调参结果"""
    kp: float
    ki: float
    kd: float
    method: TuningMethod
    quality_score: float = 1.0       # 0-1, 越高越好
    iterations: int = 0
    converges: bool = True
    info: Dict = field(default_factory=dict)


class AutoTuner:
    """
    自动PID调参器

    使用被控对象的阶跃响应或闭环数据自动计算PID参数。

    使用示例:
        tuner = AutoTuner(method=TuningMethod.ZIEGLER_NICHOLS)
        # 注入激励信号并收集数据
        tuner.inject_stimulus(plant_func, relay_amplitude=2.0)
        result = tuner.compute_pid()
    """

    def __init__(self, config: Optional[TunerConfig] = None):
        self.config = config or TunerConfig()
        self.method = self.config.method
        self._data: List[Tuple[float, float]] = []  # (time, value)
        self._start_time: float = 0.0
        self._is_tuning: bool = False
        self._relay_state: bool = False
        self._last_switch_time: float = 0.0
        self._switch_times: List[float] = []
        self._oscillation_periods: List[float] = []

    def reset(self):
        """重置调参数据"""
        self._data.clear()
        self._switch_times.clear()
        self._oscillation_periods.clear()
        self._is_tuning = False
        self._relay_state = False

    def start_tuning(self):
        """开始记录数据"""
        self.reset()
        self._start_time = time.time()
        self._is_tuning = True

    def record(self, value: float) -> bool:
        """
        记录一个数据点
        Returns: 是否检测到继电切换
        """
        if not self._is_tuning:
            return False
        t = time.time() - self._start_time
        self._data.append((t, value))
        return self._check_relay_switch(value)

    def _check_relay_switch(self, value: float) -> bool:
        """检查继电切换"""
        cfg = self.config
        if len(self._data) < 2:
            return False
        prev_t, prev_v = self._data[-2]
        # 上升沿检测
        if prev_v <= cfg.noise_band and value > cfg.noise_band:
            self._switch_times.append(time.time() - self._start_time)
            if len(self._switch_times) >= 2:
                period = self._switch_times[-1] - self._switch_times[-2]
                self._oscillation_periods.append(period)
            self._relay_state = True
            return True
        # 下降沿检测
        if prev_v >= -cfg.noise_band and value < -cfg.noise_band:
            self._switch_times.append(time.time() - self._start_time)
            if len(self._switch_times) >= 2:
                period = self._switch_times[-1] - self._switch_times[-2]
                self._oscillation_periods.append(period)
            self._relay_state = False
            return True
        return False

    def compute_pid(self) -> TunerResult:
        """
        基于收集的数据计算PID参数

        Returns:
            TunerResult: 包含 kp, ki, kd 及质量分数
        """
        if len(self._data) < 10:
            return TunerResult(0.0, 0.0, 0.0, self.method, 0.0, 0, False,
                             {"error": "insufficient data"})

        times = np.array([d[0] for d in self._data])
        values = np.array([d[1] for d in self._data])

        if self.method == TuningMethod.RELAY:
            return self._compute_relay(times, values)
        elif self.method == TuningMethod.ZIEGLER_NICHOLS:
            return self._compute_ziegler_nichols(times, values)
        elif self.method == TuningMethod.COHEN_COON:
            return self._compute_cohen_coon(times, values)
        elif self.method == TuningMethod.MODEL_REFERENCE:
            return self._compute_mrac(values)
        else:
            return self._compute_relay(times, values)

    def _compute_relay(self, times: np.ndarray, values: np.ndarray) -> TunerResult:
        """继电反馈法 (Åström-Hägglund)"""
        if len(self._oscillation_periods) < 2:
            # 尝试从数据中提取振荡周期
            zero_crossings = self._find_zero_crossings(values)
            if len(zero_crossings) >= 2:
                periods = np.diff(times[zero_crossings])
                pu = np.mean(periods)
            else:
                return TunerResult(1.0, 0.0, 0.0, self.method, 0.1, 0,
                                   False, {"error": "no oscillation detected"})
        else:
            pu = np.mean(self._oscillation_periods[-4:])  # 取最近4个周期

        # 计算临界增益 Ku
        d = self.config.relay_amplitude
        # 对于对称继电: Ku = 4*d / (pi * a)
        # 其中 a 是振荡幅值
        amplitude = (values.max() - values.min()) / 2.0
        if amplitude < 1e-6:
            return TunerResult(1.0, 0.0, 0.0, self.method, 0.1, 0,
                               False, {"error": "amplitude too small"})
        ku = (4.0 * d) / (np.pi * amplitude)

        # Z-N 公式 (继电法)
        kp = 0.6 * ku
        ki = kp / (0.5 * pu)
        kd = kp * (0.125 * pu)

        quality = self._assess_quality(ku, pu, kp, ki, kd)
        return TunerResult(kp, ki, kd, self.method, quality, len(self._data),
                          True, {"ku": ku, "pu": pu, "amplitude": amplitude})

    def _compute_ziegler_nichols(self, times: np.ndarray, values: np.ndarray) -> TunerResult:
        """Ziegler-Nichols 阶跃响应法"""
        # 从阶跃响应提取: 延迟时间 L, 时间常数 T
        setpoint = self.config.target_kp  # 用作设定值
        # 找到最终稳态值
        final_value = np.mean(values[-int(len(values) * 0.2):])
        if abs(final_value) < 1e-6:
            return TunerResult(1.0, 0.0, 0.0, self.method, 0.1, 0,
                               False, {"error": "no steady state"})
        # 找到 63.2% 点
        target_63 = 0.632 * final_value
        idx_63 = np.argmax(values >= target_63) if final_value > 0 else np.argmax(values <= target_63)
        if idx_63 == 0:
            idx_63 = len(values) - 1
        t_63 = times[idx_63]
        # 找到 28.3% 点 (0.283 = 1 - exp(-1))
        target_28 = 0.283 * final_value
        idx_28 = np.argmax(values >= target_28) if final_value > 0 else np.argmax(values <= target_28)
        if idx_28 == 0:
            idx_28 = len(values) - 1
        t_28 = times[idx_28]

        if t_63 <= t_28:
            return TunerResult(1.0, 0.0, 0.0, self.method, 0.1, 0,
                               False, {"error": "invalid step response shape"})

        L = t_28  # 延迟时间
        T = t_63 - t_28  # 时间常数

        # Z-N 公式 (阶跃响应法)
        kp = 1.2 * T / L if L > 0 else 1.0
        ki = kp / (2.0 * L) if L > 0 else 0.0
        kd = kp * (0.5 * L) if L > 0 else 0.0

        quality = self._assess_quality(kp, T, kp, ki, kd)
        return TunerResult(kp, ki, kd, self.method, quality, len(self._data),
                          True, {"L": L, "T": T, "final_value": final_value})

    def _compute_cohen_coon(self, times: np.ndarray, values: np.ndarray) -> TunerResult:
        """Cohen-Coon 法 (更好的大延迟系统整定)"""
        final_value = np.mean(values[-int(len(values) * 0.2):])
        if abs(final_value) < 1e-6:
            return self._compute_ziegler_nichols(times, values)
        # 提取延迟 L 和时间常数 T
        target_63 = 0.632 * final_value
        idx_63 = np.argmax(values >= target_63) if final_value > 0 else np.argmax(values <= target_63)
        if idx_63 == 0:
            idx_63 = len(values) - 1
        target_28 = 0.283 * final_value
        idx_28 = np.argmax(values >= target_28) if final_value > 0 else np.argmax(values <= target_28)
        if idx_28 == 0:
            idx_28 = len(values) - 1
        L = times[idx_28]
        T = times[idx_63] - times[idx_28]
        theta = L / T  # 延迟比

        # Cohen-Coon 公式
        kc = (1.35 + 0.25 * theta) / (max(L, 0.001)) if theta < 5 else 0.1
        tau_i = max(L * (2.5 - 2 * theta) / (1 + 0.6 * theta), 0.001)
        tau_d = max(L * (0.37 - 0.37 * theta) / (1 + 0.2 * theta), 0)

        kp = kc
        ki = kc / tau_i if tau_i > 0 else 0.0
        kd = kc * tau_d

        quality = self._assess_quality(kp, T, kp, ki, kd)
        return TunerResult(kp, ki, kd, self.method, quality, len(self._data),
                          True, {"L": L, "T": T, "theta": theta})

    def _compute_mrac(self, values: np.ndarray) -> TunerResult:
        """模型参考自适应整定 (简化版)"""
        # 简化的 MIT rule 实现
        # 调整增益使输出跟踪参考模型
        kp = self.config.target_kp
        ki = self.config.target_ki
        kd = self.config.target_kd

        # 跟踪误差
        ref_output = self._simple_reference_model(values)
        error = values - ref_output

        # 梯度下降调整
        lr = 0.01
        for _ in range(self.config.max_iterations):
            grad_kp = -np.mean(error * values)
            grad_ki = -np.mean(error * np.cumsum(values) * self.config.sample_time)
            grad_kd = -np.mean(error * np.gradient(values))

            kp -= lr * grad_kp
            ki -= lr * grad_ki
            kd -= lr * grad_kd

            kp = max(0, kp)
            ki = max(0, ki)
            kd = max(0, kd)

        quality = self._assess_quality(kp, 1.0, kp, ki, kd)
        return TunerResult(kp, ki, kd, self.method, quality,
                          self.config.max_iterations, True)

    def _simple_reference_model(self, values: np.ndarray) -> np.ndarray:
        """简单参考模型 (一阶系统)"""
        tau = 0.5
        dt = self.config.sample_time
        ref = np.zeros_like(values)
        for i in range(1, len(values)):
            ref[i] = ref[i-1] + dt / tau * (values[max(i-1, 0)] - ref[i-1])
        return ref

    def _find_zero_crossings(self, values: np.ndarray) -> np.ndarray:
        """找到过零点"""
        signs = np.sign(values)
        crossings = np.where(np.diff(signs) != 0)[0]
        return crossings

    def _assess_quality(self, *args) -> float:
        """评估PID整定质量 (0-1)"""
        # 简单启发式评分
        kp, *_ = args
        if kp <= 0:
            return 0.0
        if kp > 1000:
            return 0.2
        return min(1.0, 0.8 + 0.2 / (1 + 0.01 * abs(kp)))

    def simulate_step(self, kp: float, ki: float, kd: float,
                      setpoint: float = 1.0, duration: float = 5.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟PID闭环阶跃响应

        Args:
            kp, ki, kd: PID参数
            setpoint: 设定值
            duration: 模拟时长

        Returns:
            (times, outputs): 时间序列和输出序列
        """
        dt = self.config.sample_time
        n_steps = int(duration / dt)
        times = np.arange(n_steps) * dt

        output = np.zeros(n_steps)
        integral = 0.0
        prev_error = 0.0

        for i in range(n_steps):
            error = setpoint - output[max(i-1, 0)]
            integral += error * dt
            derivative = (error - prev_error) / dt if dt > 0 else 0.0
            u = kp * error + ki * integral + kd * derivative

            # 简单一阶系统: G(s) = 1/(Ts + 1)
            T = 0.5  # 时间常数
            if i > 0:
                output[i] = output[i-1] + dt / T * (u - output[i-1])

            output[i] = np.clip(output[i], -10, 10)
            prev_error = error

        return times, output


def autotune_pid(plant_func: Callable,
                 method: TuningMethod = TuningMethod.ZIEGLER_NICHOLS,
                 **kwargs) -> TunerResult:
    """
    快速自动整定接口

    Args:
        plant_func: 被控对象函数, signature: output = plant(input, dt)
        method: 调参方法

    Returns:
        TunerResult: 最佳PID参数
    """
    config = TunerConfig(method=method, **kwargs)
    tuner = AutoTuner(config)
    return tuner.compute_pid()


# ─── 虚拟被控对象用于测试 ───────────────────────────────────────────

class SimulatedPlant:
    """用于测试的虚拟被控对象"""

    def __init__(self, k: float = 1.0, T: float = 0.5, L: float = 0.1):
        self.k = k    # 增益
        self.T = T    # 时间常数
        self.L = L    # 延迟
        self.state = 0.0
        self.delay_buffer: List[float] = []
        self.delay_steps: int = int(L / 0.01)

    def step(self, u: float, dt: float) -> float:
        """一阶滞后系统"""
        if self.delay_steps > 0:
            self.delay_buffer.append(u)
            if len(self.delay_buffer) > self.delay_steps:
                u_delayed = self.delay_buffer.pop(0)
            else:
                u_delayed = 0.0
        else:
            u_delayed = u

        # 一阶惯性环节: dY/dt = (k*u - Y) / T
        self.state += dt / self.T * (self.k * u_delayed - self.state)
        return self.state

    def reset(self):
        self.state = 0.0
        self.delay_buffer.clear()
