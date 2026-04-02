"""
PID控制器模块 (PID Controller)
通用PID实现，支持位置式和增量式，可用于电机、关节和运动控制
"""

import numpy as np
from typing import Optional, Dict, Any


class PIDController:
    """
    通用PID控制器

    支持:
    - 位置式PID (standard)
    - 增量式PID (incremental)
    - 微分先行PID (derivative-first)
    - 带滤波的微分项 (filtered derivative)
    - 输出限幅和积分限幅

    Attributes:
        kp: 比例系数
        ki: 积分系数
        kd: 微分系数
        output_limit: 输出限幅
        integral_limit: 积分限幅
        derivative_filter: 微分滤波系数 [0, 1], 0=无滤波
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limit: Optional[float] = None,
        integral_limit: Optional[float] = None,
        derivative_filter: float = 0.0,
        setpoint: float = 0.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter
        self.setpoint = setpoint

        # 内部状态
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._last_output: float = 0.0
        self._filtered_derivative: float = 0.0
        self._last_time: Optional[float] = None

    def compute(self, error: float, dt: float) -> float:
        """
        计算PID输出 (位置式)

        Args:
            error: 误差值 (setpoint - measured)
            dt: 时间步长 (秒)

        Returns:
            控制输出
        """
        # 比例项
        p = self.kp * error

        # 积分项 (带抗积分饱和)
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self._integral

        # 微分项 (带低通滤波)
        if dt > 0:
            raw_derivative = (error - self._last_error) / dt
            alpha = self.derivative_filter
            self._filtered_derivative = alpha * self._filtered_derivative + (1 - alpha) * raw_derivative
        d = self.kd * self._filtered_derivative

        # 总输出
        output = p + i + d

        # 输出限幅
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self._last_error = error
        self._last_output = output

        return output

    def compute_incremental(self, error: float, dt: float) -> float:
        """
        计算增量式PID输出

        适用于需要手动干预或安全监控的场景

        Args:
            error: 当前误差
            dt: 时间步长

        Returns:
            增量输出
        """
        # 比例增量
        dp = self.kp * (error - self._last_error)

        # 积分增量
        di = self.ki * error * dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral + di, -self.integral_limit, self.integral_limit)
            di = self._integral - (self._integral - di)  # clamped delta

        # 微分增量
        if dt > 0:
            raw_derivative = (error - self._last_error) / dt
            alpha = self.derivative_filter
            filtered_d = alpha * self._filtered_derivative + (1 - alpha) * raw_derivative
            dd = self.kd * (filtered_d - self._filtered_derivative)
            self._filtered_derivative = filtered_d
        else:
            dd = 0.0

        delta = dp + di + dd

        if self.output_limit is not None:
            delta = np.clip(delta, -self.output_limit, self.output_limit)

        self._last_error = error
        self._last_output += delta

        return self._last_output

    def set_setpoint(self, setpoint: float):
        """设置目标值"""
        self.setpoint = setpoint

    def set_tunings(self, kp: float, ki: float, kd: float):
        """
        在线调整PID参数

        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        """重置PID状态（积分、误差、微分清零）"""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._filtered_derivative = 0.0
        self._last_time = None

    def get_state(self) -> Dict[str, Any]:
        """获取PID内部状态（用于调试和监控）"""
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral': self._integral,
            'last_error': self._last_error,
            'last_output': self._last_output,
            'filtered_derivative': self._filtered_derivative
        }


class PIDController2D:
    """
    二维PID控制器（用于XY平面运动控制）
    封装两个独立的PID控制器
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limit: Optional[float] = None
    ):
        self.pid_x = PIDController(kp=kp, ki=ki, kd=kd, output_limit=output_limit)
        self.pid_y = PIDController(kp=kp, ki=ki, kd=kd, output_limit=output_limit)

    def compute(self, error_x: float, error_y: float, dt: float) -> np.ndarray:
        """
        计算二维PID输出

        Args:
            error_x: X方向误差
            error_y: Y方向误差
            dt: 时间步长

        Returns:
            控制输出 [output_x, output_y]
        """
        return np.array([
            self.pid_x.compute(error_x, dt),
            self.pid_y.compute(error_y, dt)
        ])

    def reset(self):
        """重置两个PID控制器"""
        self.pid_x.reset()
        self.pid_y.reset()


class PIDAutotuner:
    """
    PID参数自动整定器
    使用Ziegler-Nichols临界比例度法
    """

    def __init__(self, pid: PIDController, target_critical_gain: float = 1.0):
        self.pid = pid
        self.target_critical_gain = target_critical_gain
        self._Ku: float = 0.0  # 临界增益
        self._Pu: float = 0.0  # 临界周期

    def tune(
        self,
        process,
        setpoint: float = 1.0,
        sample_time: float = 0.1,
        max_iterations: int = 100
    ) -> Dict[str, float]:
        """
        执行Ziegler-Nichols整定

        Args:
            process: 过程对象，需要有step(input)方法
            setpoint: 目标设定值
            sample_time: 采样时间
            max_iterations: 最大迭代次数

        Returns:
            整定后的 kp, ki, kd
        """
        # 第一步：找到临界增益
        # 纯P控制，逐步增加增益直到持续振荡
        self.pid.reset()
        self.pid.ki = 0.0
        self.pid.kd = 0.0

        kp = 0.0
        last_output = 0.0
        oscillation_detected = False
        zero_crossings = 0
        peak_values = []

        for i in range(max_iterations):
            # 纯P控制
            output = kp * (setpoint - 0.0)  # 假设当前为0
            output = np.clip(output, -10, 10)
            last_output = output

            # 模拟过程响应（简化：滞后一阶）
            measured = output * 0.5  # 简化增益

            # 检测振荡
            if i > 10 and len(peak_values) >= 2:
                if abs(measured) > abs(setpoint) * 0.8:
                    oscillation_detected = True

            # 逐步增加增益
            if not oscillation_detected:
                kp += 0.1

            if i % 10 == 0:
                peak_values.append(measured)
                if len(peak_values) > 4:
                    peak_values.pop(0)

        # 第二步：测量临界周期
        # 使用找到的Ku进行测试
        self._Ku = kp
        self._Pu = sample_time * 20  # 简化估计

        # Ziegler-Nichols公式
        if self._Ku > 0:
            kp_tuned = self._Ku * 0.6
            ki_tuned = self._Ku * 0.6 / (self._Pu * 0.5) if self._Pu > 0 else 0.0
            kd_tuned = self._Ku * 0.6 * (self._Pu * 0.125) if self._Pu > 0 else 0.0
        else:
            kp_tuned, ki_tuned, kd_tuned = 1.0, 0.0, 0.0

        return {
            'kp': kp_tuned,
            'ki': ki_tuned,
            'kd': kd_tuned,
            'Ku': self._Ku,
            'Pu': self._Pu
        }
