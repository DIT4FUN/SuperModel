"""
SuperModel 预测性维护模块 v1.0.0
================================

AGV 预测性维护与健康管理系统
功能:
  - 电机轴承磨损检测 (电流 signature 分析)
  - 电池健康状态 (SOH) 估计
  - 电机绕组温度预测
  - 车轮打滑检测与校正
  - 整体 AGV 健康评分
  - 故障分类与根因分析

Author: SuperModel Dev Team
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from collections import deque


class HealthLevel(IntEnum):
    """健康等级"""
    CRITICAL = 0   # 严重故障, 需立即停机
    FAULT = 1      # 故障, 需维修
    WARNING = 2     # 警告, 需关注
    DEGRADED = 3   # 性能下降
    HEALTHY = 4    # 健康


class FaultType(IntEnum):
    """故障类型"""
    NONE = 0
    MOTOR_BEARING_WEAR = 1
    MOTOR_OVERHEATING = 2
    MOTOR_STALL = 3
    BATTERY_SOH_LOW = 4
    BATTERY_OVER_TEMP = 5
    WHEEL_SLIP = 6
    WHEEL_MISALIGNMENT = 7
    ENCODER_DRIFT = 8
    SENSOR_CALIBRATION_DRIFT = 9
    COMMUNICATION_LATENCY = 10


@dataclass
class MotorHealthMetrics:
    """电机健康指标"""
    bearing_wear_index: float = 0.0      # 0=新, 1=需更换
    winding_temp: float = 25.0           # 摄氏度
    winding_temp_predicted: float = 25.0 # 预测温度
    current_rms: float = 0.0            # RMS 电流 (A)
    vibration_index: float = 0.0         # 振动指数
    stall_probability: float = 0.0      # 堵转概率
    efficiency: float = 1.0             # 效率 0-1
    health_level: HealthLevel = HealthLevel.HEALTHY
    fault_type: FaultType = FaultType.NONE


@dataclass
class BatteryHealthMetrics:
    """电池健康指标"""
    soh: float = 100.0                  # State of Health (%)
    cycle_count: int = 0                # 充放电循环次数
    capacity_loss: float = 0.0           # 容量损失 (%)
    internal_resistance: float = 0.05    # 内阻 (Ohm)
    temperature: float = 25.0           # 电池温度 (C)
    voltage: float = 48.0               # 当前电压 (V)
    current: float = 0.0                # 当前电流 (A)
    estimated_remaining_cycles: int = 1000  # 预估剩余循环
    health_level: HealthLevel = HealthLevel.HEALTHY
    fault_type: FaultType = FaultType.NONE


@dataclass
class WheelHealthMetrics:
    """车轮健康指标"""
    slip_ratio: float = 0.0             # 打滑率 0-1
    alignment_error: float = 0.0         # 对中误差 (deg)
    odometry_drift: float = 0.0         # 里程计漂移 (m)
    load_distribution: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # 各轮负载
    health_level: HealthLevel = HealthLevel.HEALTHY
    fault_type: FaultType = FaultType.NONE


@dataclass
class AGVHealthReport:
    """AGV 健康报告"""
    timestamp: float
    overall_score: float = 100.0         # 0-100 健康分
    health_level: HealthLevel = HealthLevel.HEALTHY
    motor_metrics: Dict[str, MotorHealthMetrics] = field(default_factory=dict)
    battery_metrics: BatteryHealthMetrics = field(default_factory=BatteryHealthMetrics)
    wheel_metrics: WheelHealthMetrics = field(default_factory=WheelHealthMetrics)
    active_faults: List[Tuple[FaultType, float]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# AGV 五级规格: 预测性维护参数
# ============================================================================

AGV_PREDICTIVE_MAINTENANCE_GRADES = {
    "S": dict(
        motor_current_sample_rate=100,       # Hz
        battery_soh_update_interval=300,      # s
        bearing_wear_window=500,              # samples for analysis
        temp_prediction_horizon=60,          # s
        wheel_odometry_window=200,
        vibration_analysis=False,
        health_score_baseline=80.0,
    ),
    "M": dict(
        motor_current_sample_rate=200,
        battery_soh_update_interval=120,
        bearing_wear_window=800,
        temp_prediction_horizon=120,
        wheel_odometry_window=400,
        vibration_analysis=True,
        health_score_baseline=85.0,
    ),
    "L": dict(
        motor_current_sample_rate=500,
        battery_soh_update_interval=60,
        bearing_wear_window=1000,
        temp_prediction_horizon=180,
        wheel_odometry_window=600,
        vibration_analysis=True,
        health_score_baseline=90.0,
    ),
    "XL": dict(
        motor_current_sample_rate=1000,
        battery_soh_update_interval=30,
        bearing_wear_window=1500,
        temp_prediction_horizon=300,
        wheel_odometry_window=800,
        vibration_analysis=True,
        health_score_baseline=92.0,
    ),
    "XXL": dict(
        motor_current_sample_rate=2000,
        battery_soh_update_interval=10,
        bearing_wear_window=2000,
        temp_prediction_horizon=600,
        wheel_odometry_window=1000,
        vibration_analysis=True,
        health_score_baseline=95.0,
    ),
}


def get_predictive_maintenance_spec(grade: str) -> dict:
    """获取指定 AGV 等级的预测性维护规格"""
    return AGV_PREDICTIVE_MAINTENANCE_GRADES.get(grade, AGV_PREDICTIVE_MAINTENANCE_GRADES["M"])


# ============================================================================
# 电机健康监测器
# ============================================================================

class MotorHealthMonitor:
    """
    电机健康状态监测器
    
    通过电机电流 signature 分析检测:
    - 轴承磨损
    - 绕组过热
    - 堵转风险
    """

    def __init__(
        self,
        motor_id: str,
        rated_current: float = 10.0,
        rated_power: float = 500.0,
        thermal_time_constant: float = 300.0,
        sample_rate: float = 200.0,
        bearing_l10_life: float = 20000.0,  # 小时
    ):
        self.motor_id = motor_id
        self.rated_current = rated_current
        self.rated_power = rated_power
        self.thermal_time_constant = thermal_time_constant  # 秒
        self.sample_rate = sample_rate
        self.bearing_l10_life = bearing_l10_life

        # 状态
        self._current_buffer = deque(maxlen=2000)
        self._temp_buffer = deque(maxlen=500)
        self._time_buffer = deque(maxlen=500)
        self._start_time = None

        # 累积统计
        self._total_operating_hours = 0.0
        self._total_current_rms_samples = 0.0
        self._cumulative_heating = 0.0

        # 健康指标
        self.metrics = MotorHealthMetrics()
        self._fault_history: deque = deque(maxlen=100)

    def update(
        self,
        current: float,
        voltage: float,
        speed: float,
        dt: float,
        ambient_temp: float = 25.0,
    ) -> MotorHealthMetrics:
        """更新电机状态并返回健康指标"""
        import time
        if self._start_time is None:
            self._start_time = time.time()

        self._total_operating_hours += dt / 3600.0

        # 记录电流
        self._current_buffer.append(current)
        self._temp_buffer.append(self.metrics.winding_temp)
        self._time_buffer.append(time.time())

        # 计算 RMS 电流
        if len(self._current_buffer) > 10:
            self.metrics.current_rms = float(np.sqrt(np.mean(np.array(self._current_buffer) ** 2)))

        # 轴承磨损检测 (电流 signature 分析)
        self._analyze_bearing_wear(speed, current)

        # 绕组温度计算 (热模型)
        self._update_winding_temperature(current, ambient_temp, dt)

        # 振动指数 (基于电流纹波)
        self._compute_vibration_index(current, speed)

        # 堵转概率
        self._evaluate_stall_risk(current, speed)

        # 效率估计
        self._estimate_efficiency(current, voltage, speed)

        # 健康等级判定
        self._evaluate_motor_health()

        return self.metrics

    def _analyze_bearing_wear(self, speed: float, current: float) -> None:
        """分析轴承磨损 (电流 signature 方法)"""
        if len(self._current_buffer) < 100 or speed < 0.1:
            return

        # 简化的轴承磨损指数计算
        # 实际应用中需要 FFT 分析电流 signature 的特定频率成分
        current_arr = np.array(self._current_buffer)
        if len(current_arr) > 50:
            # 检测电流波动 (高频成分指示轴承问题)
            detrended = current_arr - np.mean(current_arr)
            ac_variance = np.var(detrended)
            dc_level = np.abs(np.mean(current_arr))

            # 归一化振动指标
            if dc_level > 0.1:
                ripple_ratio = np.sqrt(ac_variance) / dc_level
            else:
                ripple_ratio = 0.0

            # 轴承磨损进度 = 运行时间 / L10 寿命
            wear_progress = min(1.0, self._total_operating_hours / self.bearing_l10_life)

            # 结合电流纹波综合评估
            # 正常磨损进度 + 异常纹波加成
            self.metrics.bearing_wear_index = np.clip(
                wear_progress + 0.3 * ripple_ratio,
                0.0, 1.0
            )

    def _update_winding_temperature(
        self,
        current: float,
        ambient_temp: float,
        dt: float,
    ) -> None:
        """
        更新绕组温度 (一阶热模型)
        
        dT/dt = (P_loss - (T - T_amb) / R_th) / C_th
        
        简化为离散形式:
        T_new = T_old + dt * (P_loss*R_th - (T_old - T_amb)) / (R_th * C_th)
        """
        # 铜损 (I^2 * R), 假设 R = 0.1 Ohm
        r_winding = 0.1
        p_loss = current ** 2 * r_winding

        # 热阻和热容 (归一化参数)
        r_th = 1.0  # 热阻 (归一化)
        c_th = self.thermal_time_constant

        # 温度变化
        temp_diff = self.metrics.winding_temp - ambient_temp
        temp_change = dt * (p_loss * r_th - temp_diff) / c_th

        self.metrics.winding_temp += temp_change

        # 简单的温度预测 (向前看 horizon 秒)
        horizon = 60.0  # 默认 60 秒
        if len(self._current_buffer) > 0:
            avg_current = float(np.mean(np.array(self._current_buffer)[-min(100, len(self._current_buffer)):]))
            future_p_loss = avg_current ** 2 * r_winding
            future_temp_change = horizon * (future_p_loss * r_th - (self.metrics.winding_temp - ambient_temp)) / c_th
            self.metrics.winding_temp_predicted = self.metrics.winding_temp + future_temp_change

    def _compute_vibration_index(self, current: float, speed: float) -> None:
        """计算振动指数"""
        if len(self._current_buffer) < 20 or speed < 1.0:
            return

        current_arr = np.array(self._current_buffer)
        # 电流变化率 (反映转矩波动)
        current_diff = np.diff(current_arr)
        self.metrics.vibration_index = float(np.std(current_diff) / (np.abs(np.mean(current_arr)) + 0.01))

    def _evaluate_stall_risk(self, current: float, speed: float) -> None:
        """评估堵转风险"""
        # 高电流 + 低转速 = 堵转风险
        current_ratio = current / (self.rated_current + 1e-6)
        speed_ratio = speed / (self.rated_power / (self.rated_current + 1e-6) + 1e-6)

        if current_ratio >= 1.5 and speed_ratio < 0.1:
            self.metrics.stall_probability = min(1.0, self.metrics.stall_probability + 0.1)
        else:
            self.metrics.stall_probability = max(0.0, self.metrics.stall_probability - 0.02)

    def _estimate_efficiency(self, current: float, voltage: float, speed: float) -> None:
        """估计电机效率"""
        # 机械功率 = 转矩 * 角速度 ~ speed * current (简化)
        p_mech = speed * abs(current) * 0.8  # 简化模型
        p_elec = abs(current * voltage) + 1e-6
        self.metrics.efficiency = np.clip(p_mech / p_elec, 0.0, 1.0)

        # 如果轴承磨损严重, 效率下降
        if self.metrics.bearing_wear_index > 0.3:
            efficiency_penalty = 0.1 * self.metrics.bearing_wear_index
            self.metrics.efficiency *= (1.0 - efficiency_penalty)

    def _evaluate_motor_health(self) -> None:
        """综合评估电机健康等级"""
        score = 1.0
        faults = []

        # 轴承磨损 (权重 0.3)
        if self.metrics.bearing_wear_index > 0.8:
            self.metrics.health_level = HealthLevel.CRITICAL
            self.metrics.fault_type = FaultType.MOTOR_BEARING_WEAR
            faults.append(FaultType.MOTOR_BEARING_WEAR)
        elif self.metrics.bearing_wear_index > 0.5:
            self.metrics.health_level = HealthLevel.WARNING
            score *= 0.8
        elif self.metrics.bearing_wear_index > 0.3:
            self.metrics.health_level = HealthLevel.DEGRADED
            score *= 0.9

        # 绕组温度 (权重 0.3)
        max_temp = 120.0  # F 级绝缘
        if self.metrics.winding_temp > max_temp:
            self.metrics.health_level = HealthLevel.CRITICAL
            self.metrics.fault_type = FaultType.MOTOR_OVERHEATING
            faults.append(FaultType.MOTOR_OVERHEATING)
            score *= 0.5
        elif self.metrics.winding_temp > 100.0:
            self.metrics.health_level = HealthLevel.FAULT
            faults.append(FaultType.MOTOR_OVERHEATING)
            score *= 0.7
        elif self.metrics.winding_temp > 80.0:
            if self.metrics.health_level.value > HealthLevel.WARNING.value:
                self.metrics.health_level = HealthLevel.WARNING
            score *= 0.9

        # 堵转概率 (权重 0.2)
        if self.metrics.stall_probability > 0.7:
            self.metrics.health_level = HealthLevel.FAULT
            self.metrics.fault_type = FaultType.MOTOR_STALL
            faults.append(FaultType.MOTOR_STALL)
            score *= 0.6

        # 振动指数 (权重 0.2)
        if self.metrics.vibration_index > 0.3:
            score *= 0.85

        if not faults:
            self.metrics.fault_type = FaultType.NONE


# ============================================================================
# 电池 SOH 估计器
# ============================================================================

class BatterySOHEstimator:
    """
    电池健康状态 (SOH) 估计器
    
    估计方法:
    - 容量衰减模型 (循环次数 + 温度影响)
    - 内阻增长模型
    - 电压跌落分析
    """

    def __init__(
        self,
        nominal_capacity: float = 40.0,    # Ah
        nominal_voltage: float = 48.0,      # V
        chemistry: str = "Li-ion",
        initial_soh: float = 100.0,
    ):
        self.nominal_capacity = nominal_capacity
        self.nominal_voltage = nominal_voltage
        self.chemistry = chemistry
        self.metrics = BatteryHealthMetrics(soh=initial_soh)

        # 累计数据
        self._charge_throughput = 0.0      # Ah (累计充入容量)
        self._discharge_throughput = 0.0    # Ah (累计放出容量)
        self._cycle_count = 0
        self._last_soc = 1.0
        self._temp_history = deque(maxlen=1000)
        self._voltage_history = deque(maxlen=500)
        self._current_history = deque(maxlen=500)
        self._soc_history = deque(maxlen=500)
        self._total_time_seconds = 0.0

    def update(
        self,
        voltage: float,
        current: float,
        soc: float,
        temperature: float,
        dt: float,
    ) -> BatteryHealthMetrics:
        """更新电池状态并返回 SOH 指标"""
        self.metrics.voltage = voltage
        self.metrics.current = current
        self.metrics.temperature = temperature
        self._temp_history.append(temperature)
        self._voltage_history.append(voltage)
        self._current_history.append(current)
        self._soc_history.append(soc)
        self._total_time_seconds += dt

        # 检测充放电循环 (SOC 从低到高再从高到低算一个完整循环)
        self._detect_cycle(soc)

        # 估计 SOH
        self._estimate_soh(temperature)

        # 更新内阻估计
        self._update_internal_resistance(voltage, current)

        # 预估剩余循环
        self._estimate_remaining_cycles()

        # 评估电池健康等级
        self._evaluate_battery_health()

        self._last_soc = soc
        return self.metrics

    def _detect_cycle(self, soc: float) -> None:
        """检测完整的充放电循环"""
        if self._last_soc < 0.2 and soc > 0.8:
            # 从低 SOC 充到高 SOC, 算半个充电循环
            self._cycle_count += 0.5
        elif self._last_soc > 0.8 and soc < 0.2:
            # 从高 SOC 放到低 SOC, 算半个放电循环
            self._cycle_count += 0.5

        self.metrics.cycle_count = self._cycle_count

    def _estimate_soh(self, temperature: float) -> None:
        """
        基于循环次数、时间和温度估计 SOH
        
        SOH 模型:
        SOH = 100 * (1 - cycle_factor - calendar_factor) * temp_factor
        
        其中:
        - cycle_factor = 循环衰减 (k * cycle_count / EOL_cycles)
        - calendar_factor = 日历衰减 (随时间累积, 与温度相关)
        - temp_factor = 温度衰减因子 (高温加速衰减)
        """
        # 循环衰减
        eol_cycles = 800.0  # 假设 800 循环后 SOH=80% (EOL)
        cycle_factor = 0.8 * (self._cycle_count / eol_cycles)

        # 日历衰减: 假设每年约 2% 衰减, 按时间比例计算
        # 归一化: 1 年 = 365*24*3600 ≈ 31,536,000 秒
        seconds_per_year = 365.0 * 24.0 * 3600.0
        calendar_loss_rate_per_year = 0.02  # 每年 2%
        calendar_factor = calendar_loss_rate_per_year * (self._total_time_seconds / seconds_per_year)

        # 温度因子 (高温加速衰减, 基准 25C)
        if temperature > 35.0:
            temp_factor = 1.0 - 0.001 * (temperature - 35.0) ** 2
        elif temperature < 10.0:
            temp_factor = 1.0 - 0.0005 * (10.0 - temperature) ** 2
        else:
            temp_factor = 1.0

        temp_factor = max(0.6, min(1.0, temp_factor))

        # 总衰减
        total_decay = min(0.99, cycle_factor + calendar_factor)

        # 容量损失
        self.metrics.capacity_loss = 100.0 * total_decay
        self.metrics.soh = 100.0 * (1.0 - total_decay) * temp_factor
        self.metrics.soh = max(0.0, min(100.0, self.metrics.soh))

    def _update_internal_resistance(self, voltage: float, current: float) -> None:
        """更新电池内阻估计"""
        if abs(current) > 0.5:  # 需要足够的电流才能估计内阻
            # 简化: 通过电压跌落估算内阻
            voc = self._estimate_open_circuit_voltage()
            v_drop = abs(voltage - voc)
            r = v_drop / (abs(current) + 1e-6)
            # 滑动平均
            self.metrics.internal_resistance = 0.9 * self.metrics.internal_resistance + 0.1 * r
            self.metrics.internal_resistance = min(0.5, max(0.01, self.metrics.internal_resistance))

    def _estimate_open_circuit_voltage(self) -> float:
        """估算开路电压 (通过 SOC 查表, 简化线性模型)"""
        if len(self._soc_history) == 0:
            return self.nominal_voltage
        soc = self._soc_history[-1]
        # 简化: 48V 电池, SOC 从 0-100% 对应 42-54V
        return 42.0 + 0.12 * soc * self.nominal_voltage

    def _estimate_remaining_cycles(self) -> None:
        """预估剩余可用循环"""
        # 线性外推
        if self._cycle_count > 10:
            soh_per_cycle = self.metrics.soh / max(1, self._cycle_count)
            if soh_per_cycle > 0:
                remaining = self._cycle_count * (100.0 - 80.0) / (100.0 - self.metrics.soh + 1e-6)
                self.metrics.estimated_remaining_cycles = int(remaining)
            else:
                self.metrics.estimated_remaining_cycles = 500
        else:
            self.metrics.estimated_remaining_cycles = 500

    def _evaluate_battery_health(self) -> None:
        """评估电池健康等级"""
        if self.metrics.soh < 60.0:
            self.metrics.health_level = HealthLevel.CRITICAL
            self.metrics.fault_type = FaultType.BATTERY_SOH_LOW
        elif self.metrics.soh < 75.0:
            self.metrics.health_level = HealthLevel.FAULT
            self.metrics.fault_type = FaultType.BATTERY_SOH_LOW
        elif self.metrics.soh < 85.0:
            self.metrics.health_level = HealthLevel.WARNING
        elif self.metrics.soh < 95.0:
            self.metrics.health_level = HealthLevel.DEGRADED
        else:
            self.metrics.health_level = HealthLevel.HEALTHY
            self.metrics.fault_type = FaultType.NONE

        if self.metrics.temperature > 55.0:
            self.metrics.health_level = HealthLevel.CRITICAL
            self.metrics.fault_type = FaultType.BATTERY_OVER_TEMP


# ============================================================================
# 车轮健康监测器
# ============================================================================

class WheelHealthMonitor:
    """
    车轮健康状态监测器
    
    检测:
    - 车轮打滑
    - 车轮对中误差
    - 里程计漂移
    """

    def __init__(
        self,
        wheel_base: float = 0.5,        # m
        wheel_radius: float = 0.1,       # m
        num_wheels: int = 4,
        drive_type: str = "DIFFERENTIAL",
    ):
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.num_wheels = num_wheels
        self.drive_type = drive_type

        self.metrics = WheelHealthMetrics()

        # 里程计数据
        self._encoder_counts = deque(maxlen=1000)  #  encoder ticks
        self._positions = deque(maxlen=1000)        # 位置估计 (x, y, theta)
        self._references = deque(maxlen=1000)        # 参考位置 (GPS/RTK 等)
        self._wheel_speeds = deque(maxlen=1000)     # 各轮速度

    def update(
        self,
        encoder_counts: List[int],
        wheel_speeds: List[float],
        position: Tuple[float, float, float],  # x, y, theta
        reference_position: Optional[Tuple[float, float, float]] = None,
        dt: float = 0.01,
    ) -> WheelHealthMetrics:
        """更新车轮状态并返回健康指标"""
        self._encoder_counts.append(encoder_counts)
        self._positions.append(position)
        self._wheel_speeds.append(wheel_speeds)
        if reference_position is not None:
            self._references.append(reference_position)

        # 检测打滑
        self._detect_slip(wheel_speeds, encoder_counts, dt)

        # 检测对中误差
        self._detect_misalignment(wheel_speeds)

        # 估计里程计漂移
        self._estimate_odometry_drift(position, reference_position)

        # 评估车轮健康等级
        self._evaluate_wheel_health()

        return self.metrics

    def _detect_slip(
        self,
        wheel_speeds: List[float],
        encoder_counts: List[int],
        dt: float,
    ) -> None:
        """检测车轮打滑"""
        if len(self._wheel_speeds) < 2:
            return

        # 计算车轮速度理论值与编码器测量值之比
        prev_counts = self._encoder_counts[-2] if len(self._encoder_counts) > 1 else encoder_counts
        if len(encoder_counts) == 4 and len(prev_counts) == 4:
            # 各轮转速
            count_diff = [c - p for c, p in zip(encoder_counts, prev_counts)]
            measured_speeds = [cd / dt / 1000.0 for cd in count_diff]  # 假设 1000 ticks/rev

            # 理论速度 (从编码器平滑估计)
            expected_speeds = wheel_speeds

            # 打滑率 = |测量 - 理论| / max(测量, 理论)
            slip_ratios = []
            for m, e in zip(measured_speeds, expected_speeds):
                if max(abs(m), abs(e)) > 0.01:
                    slip = abs(m - e) / (max(abs(m), abs(e)) + 1e-6)
                else:
                    slip = 0.0
                slip_ratios.append(slip)

            self.metrics.slip_ratio = float(np.mean(slip_ratios))

            # 突发打滑检测 (打滑率突增)
            if self.metrics.slip_ratio > 0.5:
                self.metrics.fault_type = FaultType.WHEEL_SLIP

    def _detect_misalignment(self, wheel_speeds: List[float]) -> None:
        """检测车轮对中误差"""
        if len(wheel_speeds) < 4 or len(self._wheel_speeds) < 10:
            return

        # 差速驱动: 正常情况下同侧车轮速度相近
        # 检测速度不对称性
        if self.drive_type == "DIFFERENTIAL":
            left_avg = np.mean([ws[0] for ws in list(self._wheel_speeds)[-10:]])
            right_avg = np.mean([ws[1] for ws in list(self._wheel_speeds)[-10:]])
            if left_avg > 0.01 or right_avg > 0.01:
                ratio = min(left_avg, right_avg) / (max(left_avg, right_avg) + 1e-6)
                # 不对称度
                asymmetry = 1.0 - ratio
                self.metrics.alignment_error = asymmetry * 5.0  # 转换为度估计

    def _estimate_odometry_drift(
        self,
        position: Tuple[float, float, float],
        reference_position: Optional[Tuple[float, float, float]],
    ) -> None:
        """估计里程计漂移"""
        if reference_position is None or len(self._references) < 10:
            # 无参考, 使用方差估计漂移
            if len(self._positions) > 20:
                pos_arr = np.array(list(self._positions)[-20:])
                drift = float(np.std(pos_arr[:, 0]) + np.std(pos_arr[:, 1]))
                self.metrics.odometry_drift = min(1.0, drift)
            return

        # 有参考 (RTK GPS), 直接计算漂移
        ref = self._references[-1]
        dx = position[0] - ref[0]
        dy = position[1] - ref[1]
        self.metrics.odometry_drift = np.sqrt(dx**2 + dy**2)

    def _evaluate_wheel_health(self) -> None:
        """评估车轮健康等级"""
        score = 1.0

        if self.metrics.slip_ratio > 0.7:
            self.metrics.health_level = HealthLevel.FAULT
            self.metrics.fault_type = FaultType.WHEEL_SLIP
            score *= 0.5
        elif self.metrics.slip_ratio > 0.4:
            self.metrics.health_level = HealthLevel.WARNING
            score *= 0.8

        if self.metrics.alignment_error > 5.0:
            self.metrics.health_level = HealthLevel.FAULT
            self.metrics.fault_type = FaultType.WHEEL_MISALIGNMENT
            score *= 0.6
        elif self.metrics.alignment_error > 2.0:
            if self.metrics.health_level.value > HealthLevel.WARNING.value:
                self.metrics.health_level = HealthLevel.WARNING
            score *= 0.85

        if self.metrics.odometry_drift > 0.5:
            if self.metrics.health_level.value > HealthLevel.WARNING.value:
                self.metrics.health_level = HealthLevel.WARNING
            score *= 0.9

        if self.metrics.health_level == HealthLevel.HEALTHY:
            self.metrics.fault_type = FaultType.NONE


# ============================================================================
# AGV 整体健康管理系统
# ============================================================================

class PredictiveMaintenanceSystem:
    """
    AGV 预测性维护系统
    
    整合所有健康监测器, 提供:
    - 整体健康评分 (0-100)
    - 故障预测与根因分析
    - 维护建议生成
    - AGV 五级规格支持
    """

    def __init__(self, grade: str = "M"):
        self.grade = grade
        self.spec = get_predictive_maintenance_spec(grade)

        # 子系统监测器
        self._motor_monitors: Dict[str, MotorHealthMonitor] = {}
        self._battery_estimator: Optional[BatterySOHEstimator] = None
        self._wheel_monitor: Optional[WheelHealthMonitor] = None

        # 历史报告
        self._report_history: deque = deque(maxlen=100)

        # 健康评分基线
        self._baseline_score = self.spec["health_score_baseline"]

    def add_motor(self, motor_id: str, **kwargs) -> MotorHealthMonitor:
        """添加电机监测器"""
        monitor = MotorHealthMonitor(motor_id=motor_id, **kwargs)
        self._motor_monitors[motor_id] = monitor
        return monitor

    def set_battery(self, **kwargs) -> BatterySOHEstimator:
        """设置电池 SOH 估计器"""
        self._battery_estimator = BatterySOHEstimator(**kwargs)
        return self._battery_estimator

    def set_wheel_monitor(self, **kwargs) -> WheelHealthMonitor:
        """设置车轮健康监测器"""
        self._wheel_monitor = WheelHealthMonitor(**kwargs)
        return self._wheel_monitor

    def update(self, timestamp: Optional[float] = None) -> AGVHealthReport:
        """更新所有健康指标并生成报告"""
        import time
        if timestamp is None:
            timestamp = time.time()

        report = AGVHealthReport(timestamp=timestamp)

        # 电机健康
        for motor_id, monitor in self._motor_monitors.items():
            report.motor_metrics[motor_id] = monitor.metrics

        # 电池健康
        if self._battery_estimator is not None:
            report.battery_metrics = self._battery_estimator.metrics

        # 车轮健康
        if self._wheel_monitor is not None:
            report.wheel_metrics = self._wheel_monitor.metrics

        # 计算整体健康分
        self._compute_overall_score(report)

        # 收集活跃故障
        self._collect_faults(report)

        # 生成维护建议
        self._generate_recommendations(report)

        # 保存报告
        self._report_history.append(report)

        return report

    def _compute_overall_score(self, report: AGVHealthReport) -> None:
        """计算整体健康评分"""
        weights = {"motor": 0.4, "battery": 0.3, "wheel": 0.3}
        scores = {}

        # 电机评分
        if report.motor_metrics:
            motor_scores = []
            for m in report.motor_metrics.values():
                s = 100.0
                s *= (1.0 - m.bearing_wear_index * 0.3)
                s *= (1.0 - max(0, (m.winding_temp - 80.0) / 80.0) * 0.2)
                s *= (1.0 - m.stall_probability * 0.3)
                s *= m.efficiency
                motor_scores.append(max(0.0, s))
            scores["motor"] = float(np.mean(motor_scores)) if motor_scores else self._baseline_score
        else:
            scores["motor"] = self._baseline_score

        # 电池评分
        if self._battery_estimator is not None:
            scores["battery"] = report.battery_metrics.soh
        else:
            scores["battery"] = self._baseline_score

        # 车轮评分
        if self._wheel_monitor is not None:
            wm = report.wheel_metrics
            scores["wheel"] = 100.0 * (1.0 - wm.slip_ratio * 0.3 - min(wm.alignment_error / 10.0, 1.0) * 0.2)
            scores["wheel"] = max(0.0, scores["wheel"])
        else:
            scores["wheel"] = self._baseline_score

        # 加权总分
        report.overall_score = (
            scores["motor"] * weights["motor"] +
            scores["battery"] * weights["battery"] +
            scores["wheel"] * weights["wheel"]
        )
        report.overall_score = max(0.0, min(100.0, report.overall_score))

        # 健康等级
        if report.overall_score < 50.0:
            report.health_level = HealthLevel.CRITICAL
        elif report.overall_score < 65.0:
            report.health_level = HealthLevel.FAULT
        elif report.overall_score < 80.0:
            report.health_level = HealthLevel.WARNING
        elif report.overall_score < 90.0:
            report.health_level = HealthLevel.DEGRADED
        else:
            report.health_level = HealthLevel.HEALTHY

    def _collect_faults(self, report: AGVHealthReport) -> None:
        """收集所有活跃故障"""
        # 电机故障
        for motor_id, metrics in report.motor_metrics.items():
            if metrics.fault_type != FaultType.NONE:
                report.active_faults.append((metrics.fault_type, metrics.health_level.value))

        # 电池故障
        if report.battery_metrics.fault_type != FaultType.NONE:
            report.active_faults.append(
                (report.battery_metrics.fault_type, report.battery_metrics.health_level.value)
            )

        # 车轮故障
        if report.wheel_metrics.fault_type != FaultType.NONE:
            report.active_faults.append(
                (report.wheel_metrics.fault_type, report.wheel_metrics.health_level.value)
            )

    def _generate_recommendations(self, report: AGVHealthReport) -> None:
        """生成维护建议"""
        for fault_type, level in report.active_faults:
            if level > HealthLevel.WARNING.value:
                continue  # 只对 WARNING 及以上生成建议

            rec = self._fault_to_recommendation(fault_type, report)
            if rec:
                report.recommendations.append(rec)

        # 基于健康分的通用建议
        if report.overall_score < 60.0:
            report.recommendations.append("AGV 健康状态较差, 建议安排计划维护")
        elif report.overall_score < 80.0:
            report.recommendations.append("AGV 健康状态下降, 建议近期检查关键部件")

    def _fault_to_recommendation(self, fault_type: FaultType, report: AGVHealthReport) -> str:
        """故障类型转维护建议"""
        recommendations = {
            FaultType.MOTOR_BEARING_WEAR: "电机轴承磨损检测, 建议在最近维护窗口更换轴承",
            FaultType.MOTOR_OVERHEATING: "电机绕组温度过高, 检查散热系统和负载情况",
            FaultType.MOTOR_STALL: "电机堵转风险, 检查传动系统和机械卡滞",
            FaultType.BATTERY_SOH_LOW: f"电池 SOH 低至 {report.battery_metrics.soh:.1f}%, 考虑更换电池",
            FaultType.BATTERY_OVER_TEMP: "电池温度过高, 检查散热和充电策略",
            FaultType.WHEEL_SLIP: "车轮打滑检测, 检查地面条件和轮胎状态",
            FaultType.WHEEL_MISALIGNMENT: "车轮对中误差, 建议进行四轮定位校正",
            FaultType.ENCODER_DRIFT: "编码器漂移, 建议重新标定里程计",
        }
        return recommendations.get(fault_type, "")

    def get_trend(self, metric_name: str, hours: int = 24) -> Dict:
        """获取指定指标的历史趋势"""
        if len(self._report_history) < 2:
            return {}

        reports = list(self._report_history)
        timestamps = [r.timestamp for r in reports]
        scores = [r.overall_score for r in reports]

        return {
            "timestamps": timestamps,
            "scores": scores,
            "trend": "stable" if abs(scores[-1] - scores[0]) < 5.0 else ("improving" if scores[-1] > scores[0] else "declining"),
            "min": min(scores),
            "max": max(scores),
            "avg": float(np.mean(scores)),
        }


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_predictive_maintenance_system(grade: str = "M") -> PredictiveMaintenanceSystem:
    """创建指定 AGV 等级的预测性维护系统"""
    system = PredictiveMaintenanceSystem(grade=grade)

    # 根据等级添加监测器
    if grade in ("S", "M"):
        system.add_motor("drive_left", rated_current=10.0, rated_power=500.0)
        system.add_motor("drive_right", rated_current=10.0, rated_power=500.0)
    else:
        for i in range(4):
            system.add_motor(f"wheel_{i}", rated_current=15.0, rated_power=750.0)

    system.set_battery(nominal_capacity=40.0, nominal_voltage=48.0)
    system.set_wheel_monitor(num_wheels=4, drive_type="DIFFERENTIAL" if grade in ("S", "M") else "MECANUM")

    return system
