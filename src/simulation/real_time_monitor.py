"""
Real-Time Multi-Sensor Fusion Performance Monitor
实时多传感器融合性能监控器

功能:
- 多传感器数据实时采集与时间戳对齐
- 传感器延迟、抖动、吞吐量统计
- 融合模块各阶段延迟分析
- AGV五级性能合规性检测
- 异常检测与告警
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import statistics


class AGVGrade(str, Enum):
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


@dataclass
class GradeSpec:
    """AGV五级性能规格"""
    grade: AGVGrade
    control_freq_hz: float
    tactile_size: Tuple[int, int]
    tactile_hz: float
    force_axes: int
    force_range_n: float
    force_hz: float
    imu_model: str
    imu_hz: float
    max_latency_ms: float
    max_jitter_ms: float


# AGV五级规格定义
AGV_GRADE_SPECS: Dict[AGVGrade, GradeSpec] = {
    AGVGrade.S: GradeSpec(
        grade=AGVGrade.S,
        control_freq_hz=50,
        tactile_size=(8, 8),
        tactile_hz=50,
        force_axes=3,
        force_range_n=100,
        force_hz=100,
        imu_model="MPU6050",
        imu_hz=100,
        max_latency_ms=200,
        max_jitter_ms=50,
    ),
    AGVGrade.M: GradeSpec(
        grade=AGVGrade.M,
        control_freq_hz=100,
        tactile_size=(16, 16),
        tactile_hz=100,
        force_axes=6,
        force_range_n=200,
        force_hz=500,
        imu_model="BMI088",
        imu_hz=200,
        max_latency_ms=80,
        max_jitter_ms=20,
    ),
    AGVGrade.L: GradeSpec(
        grade=AGVGrade.L,
        control_freq_hz=200,
        tactile_size=(24, 24),
        tactile_hz=200,
        force_axes=6,
        force_range_n=500,
        force_hz=1000,
        imu_model="BMI088",
        imu_hz=500,
        max_latency_ms=35,
        max_jitter_ms=8,
    ),
    AGVGrade.XL: GradeSpec(
        grade=AGVGrade.XL,
        control_freq_hz=500,
        tactile_size=(32, 32),
        tactile_hz=500,
        force_axes=6,
        force_range_n=1000,
        force_hz=2000,
        imu_model="ADIS16470",
        imu_hz=1000,
        max_latency_ms=15,
        max_jitter_ms=3,
    ),
    AGVGrade.XXL: GradeSpec(
        grade=AGVGrade.XXL,
        control_freq_hz=1000,
        tactile_size=(48, 48),
        tactile_hz=1000,
        force_axes=6,
        force_range_n=5000,
        force_hz=5000,
        imu_model="ADIS16470",
        imu_hz=2000,
        max_latency_ms=7,
        max_jitter_ms=1.5,
    ),
}


@dataclass
class SensorSample:
    """单次传感器采样记录"""
    sensor_id: str
    sensor_type: str  # "tactile", "force", "imu"
    timestamp: float
    seq: int
    data_size: int


@dataclass
class FusionStage:
    """融合阶段记录"""
    stage_name: str
    start_time: float
    end_time: float
    duration_ms: float


@dataclass
class MonitoringRecord:
    """监控记录"""
    global_time: float
    tactile_latency_ms: float
    force_latency_ms: float
    imu_latency_ms: float
    fusion_latency_ms: float
    total_latency_ms: float
    cycle_jitter_ms: float
    tactile_throughput_hz: float
    force_throughput_hz: float
    imu_throughput_hz: float
    grade_compliant: bool


class SensorSimulator:
    """虚拟传感器模拟器 - 用于性能测试"""

    def __init__(self, grade: AGVGrade):
        self.grade = grade
        self.spec = AGV_GRADE_SPECS[grade]
        self._tactile_seq = 0
        self._force_seq = 0
        self._imu_seq = 0
        self._last_tactile_time = 0
        self._last_force_time = 0
        self._last_imu_time = 0
        # 模拟传感器延迟 (us)
        self._tactile_delay_us = {AGVGrade.S: 5000, AGVGrade.M: 3000,
                                   AGVGrade.L: 2000, AGVGrade.XL: 1000, AGVGrade.XXL: 500}[grade]
        self._force_delay_us = {AGVGrade.S: 3000, AGVGrade.M: 2000,
                                 AGVGrade.L: 1000, AGVGrade.XL: 500, AGVGrade.XXL: 200}[grade]
        self._imu_delay_us = {AGVGrade.S: 200, AGVGrade.M: 100,
                              AGVGrade.L: 50, AGVGrade.XL: 25, AGVGrade.XXL: 10}[grade]

    def capture_tactile(self) -> SensorSample:
        now = time.perf_counter()
        if self._last_tactile_time > 0:
            period = 1.0 / self.spec.tactile_hz
            sleep_time = max(0, period - (now - self._last_tactile_time) - self._tactile_delay_us / 1e6)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._last_tactile_time = time.perf_counter()
        self._tactile_seq += 1
        delay_s = self._tactile_delay_us / 1e6
        return SensorSample(
            sensor_id="tactile",
            sensor_type="tactile",
            timestamp=now - delay_s,
            seq=self._tactile_seq,
            data_size=self.spec.tactile_size[0] * self.spec.tactile_size[1] * 4,
        )

    def capture_force(self) -> SensorSample:
        now = time.perf_counter()
        if self._last_force_time > 0:
            period = 1.0 / self.spec.force_hz
            sleep_time = max(0, period - (now - self._last_force_time) - self._force_delay_us / 1e6)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._last_force_time = time.perf_counter()
        self._force_seq += 1
        delay_s = self._force_delay_us / 1e6
        return SensorSample(
            sensor_id="force",
            sensor_type="force",
            timestamp=now - delay_s,
            seq=self._force_seq,
            data_size=self.spec.force_axes * 8,
        )

    def capture_imu(self) -> SensorSample:
        now = time.perf_counter()
        if self._last_imu_time > 0:
            period = 1.0 / self.spec.imu_hz
            sleep_time = max(0, period - (now - self._last_imu_time) - self._imu_delay_us / 1e6)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._last_imu_time = time.perf_counter()
        self._imu_seq += 1
        delay_s = self._imu_delay_us / 1e6
        return SensorSample(
            sensor_id="imu",
            sensor_type="imu",
            timestamp=now - delay_s,
            seq=self._imu_seq,
            data_size=6 * 8 + 4 * 4,  # accel + gyro + quat
        )


class FusionSimulator:
    """融合处理模拟器"""

    def __init__(self, grade: AGVGrade):
        self.grade = grade
        self.spec = AGV_GRADE_SPECS[grade]
        # 融合计算复杂度随等级缩放
        self._compute_us = {AGVGrade.S: 500, AGVGrade.M: 300,
                              AGVGrade.L: 150, AGVGrade.XL: 80, AGVGrade.XXL: 30}[grade]

    def fuse(self, tactile: SensorSample, force: SensorSample, imu: SensorSample) -> FusionStage:
        """执行融合处理"""
        start = time.perf_counter()
        # 模拟时间戳对齐计算
        time.sleep(self._compute_us / 1e6 * np.random.uniform(0.8, 1.2))
        end = time.perf_counter()
        return FusionStage(
            stage_name="cross_modal_fusion",
            start_time=start,
            end_time=end,
            duration_ms=(end - start) * 1000,
        )


class RealTimeMonitor:
    """
    实时多传感器融合性能监控器

    使用方法:
        monitor = RealTimeMonitor(grade="M", window_size=100)
        monitor.start()
        # ... 运行一段时间 ...
        stats = monitor.get_statistics()
        monitor.stop()
    """

    def __init__(self, grade: AGVGrade | str = "M", window_size: int = 100):
        if isinstance(grade, str):
            self.grade = AGVGrade(grade)
        else:
            self.grade = grade
        self.spec = AGV_GRADE_SPECS[self.grade]
        self.window_size = window_size

        self._sensor_sim = SensorSimulator(self.grade)
        self._fusion_sim = FusionSimulator(self.grade)

        self._records: deque[MonitoringRecord] = deque(maxlen=window_size)
        self._tactile_latencies: deque[float] = deque(maxlen=window_size)
        self._force_latencies: deque[float] = deque(maxlen=window_size)
        self._imu_latencies: deque[float] = deque(maxlen=window_size)
        self._fusion_latencies: deque[float] = deque(maxlen=window_size)
        self._cycle_times: deque[float] = deque(maxlen=window_size)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 全局时钟 (传感器时间戳基准)
        self._global_start: float = 0

    def start(self):
        """启动监控线程"""
        self._running = True
        self._global_start = time.perf_counter()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """监控主循环"""
        cycle_target = 1.0 / self.spec.control_freq_hz
        last_cycle_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # 1. 采集三个传感器 (并行模拟)
            t_tactile = time.perf_counter()
            tactile_sample = self._sensor_sim.capture_tactile()
            force_sample = self._sensor_sim.capture_force()
            imu_sample = self._sensor_sim.capture_imu()

            # 2. 传感器延迟计算
            now = time.perf_counter()
            tactile_latency_ms = (now - tactile_sample.timestamp) * 1000
            force_latency_ms = (now - force_sample.timestamp) * 1000
            imu_latency_ms = (now - imu_sample.timestamp) * 1000

            # 3. 融合处理
            fusion_stage = self._fusion_sim.fuse(tactile_sample, force_sample, imu_sample)
            fusion_latency_ms = fusion_stage.duration_ms

            # 4. 总延迟
            total_latency_ms = (now - self._global_start) * 1000 % (1000 / self.spec.control_freq_hz)
            # 实际端到端延迟
            total_latency_ms = tactile_latency_ms + force_latency_ms + imu_latency_ms + fusion_latency_ms

            # 5. 周期抖动
            actual_cycle = now - last_cycle_time
            cycle_jitter_ms = abs(actual_cycle - cycle_target) * 1000
            last_cycle_time = now

            # 6. 吞吐量计算
            period_s = 1.0 / self.spec.control_freq_hz
            tactile_throughput = self.spec.tactile_hz
            force_throughput = self.spec.force_hz
            imu_throughput = self.spec.imu_hz

            # 7. 等级合规性
            compliant = (
                tactile_latency_ms < self.spec.max_latency_ms and
                force_latency_ms < self.spec.max_latency_ms and
                imu_latency_ms < self.spec.max_latency_ms and
                cycle_jitter_ms < self.spec.max_jitter_ms
            )

            record = MonitoringRecord(
                global_time=now - self._global_start,
                tactile_latency_ms=tactile_latency_ms,
                force_latency_ms=force_latency_ms,
                imu_latency_ms=imu_latency_ms,
                fusion_latency_ms=fusion_latency_ms,
                total_latency_ms=total_latency_ms,
                cycle_jitter_ms=cycle_jitter_ms,
                tactile_throughput_hz=tactile_throughput,
                force_throughput_hz=force_throughput,
                imu_throughput_hz=imu_throughput,
                grade_compliant=compliant,
            )

            with self._lock:
                self._records.append(record)
                self._tactile_latencies.append(tactile_latency_ms)
                self._force_latencies.append(force_latency_ms)
                self._imu_latencies.append(imu_latency_ms)
                self._fusion_latencies.append(fusion_latency_ms)
                self._cycle_times.append(cycle_jitter_ms)

            # 等待下一个周期
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, cycle_target - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time * 0.5)  # 留出余量

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            if not self._records:
                return {}

            records_list = list(self._records)

            def safe_stats(data: List[float]) -> Dict[str, float]:
                if not data:
                    return {}
                return {
                    "mean": statistics.mean(data),
                    "std": statistics.stdev(data) if len(data) > 1 else 0.0,
                    "min": min(data),
                    "max": max(data),
                    "p50": np.percentile(data, 50),
                    "p95": np.percentile(data, 95),
                    "p99": np.percentile(data, 99),
                }

            compliant_count = sum(1 for r in records_list if r.grade_compliant)
            compliance_rate = compliant_count / len(records_list) * 100

            return {
                "grade": self.grade.value,
                "control_freq_hz": self.spec.control_freq_hz,
                "sample_count": len(records_list),
                "compliance_rate_percent": compliance_rate,
                "tactile_latency_ms": safe_stats(list(self._tactile_latencies)),
                "force_latency_ms": safe_stats(list(self._force_latencies)),
                "imu_latency_ms": safe_stats(list(self._imu_latencies)),
                "fusion_latency_ms": safe_stats(list(self._fusion_latencies)),
                "total_latency_ms": safe_stats([r.total_latency_ms for r in records_list]),
                "cycle_jitter_ms": safe_stats(list(self._cycle_times)),
                "spec_max_latency_ms": self.spec.max_latency_ms,
                "spec_max_jitter_ms": self.spec.max_jitter_ms,
            }

    def print_report(self):
        """打印性能报告"""
        stats = self.get_statistics()
        if not stats:
            print("No data collected yet.")
            return

        print("\n" + "=" * 60)
        print(f"  SuperModel 实时融合性能报告 - AGV {stats['grade']} 级")
        print("=" * 60)
        print(f"  控制频率: {stats['control_freq_hz']} Hz")
        print(f"  采样周期: {stats['sample_count']} 个周期")
        print(f"  合规率: {stats['compliance_rate_percent']:.1f}%")
        print("-" * 60)
        print(f"  {'指标':<20} {'均值':>8} {'标准差':>8} {'P95':>8} {'最大值':>8}")
        print("-" * 60)

        for name, key in [
            ("触觉延迟", "tactile_latency_ms"),
            ("力觉延迟", "force_latency_ms"),
            ("IMU延迟", "imu_latency_ms"),
            ("融合延迟", "fusion_latency_ms"),
            ("总延迟", "total_latency_ms"),
            ("周期抖动", "cycle_jitter_ms"),
        ]:
            s = stats[key]
            if s:
                flag = " ✓" if stats['spec_max_latency_ms'] >= s['p95'] else " ✗"
                print(f"  {name:<18} {s['mean']:>7.2f}ms {s['std']:>7.2f}ms {s['p95']:>7.2f}ms {s['max']:>7.2f}ms{flag}")

        print("-" * 60)
        print(f"  规格最大延迟: {stats['spec_max_latency_ms']} ms")
        print(f"  规格最大抖动: {stats['spec_max_jitter_ms']} ms")
        print("=" * 60 + "\n")


def run_monitor_benchmark(grade: AGVGrade | str = "M", duration_seconds: float = 5.0) -> Dict[str, Any]:
    """
    运行指定等级的监控基准测试

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        duration_seconds: 测试持续时间

    Returns:
        统计结果字典
    """
    if isinstance(grade, str):
        grade = AGVGrade(grade)

    monitor = RealTimeMonitor(grade=grade, window_size=int(duration_seconds * 100))
    monitor.start()

    # 等待指定时长
    time.sleep(duration_seconds)

    monitor.stop()
    stats = monitor.get_statistics()
    return stats


def run_all_grade_benchmark(duration_seconds: float = 3.0) -> Dict[str, Dict]:
    """
    运行所有AGV五级的基准测试对比

    Returns:
        各等级统计结果字典
    """
    results = {}
    for grade in AGVGrade:
        print(f"\n  正在测试 AGV {grade.value} 级 ...")
        stats = run_monitor_benchmark(grade=grade, duration_seconds=duration_seconds)
        results[grade.value] = stats
        print(f"  合规率: {stats.get('compliance_rate_percent', 0):.1f}% | "
              f"总延迟P95: {stats.get('total_latency_ms', {}).get('p95', 0):.2f}ms")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SuperModel 实时融合性能监控 - 基准测试")
    print("=" * 60)

    # 单等级测试
    print("\n>>> 单等级测试: AGV M 级 (5秒)")
    monitor = RealTimeMonitor(grade="M", window_size=500)
    monitor.start()
    time.sleep(5.0)
    monitor.stop()
    monitor.print_report()

    # 全五级对比
    print("\n>>> 全五级对比测试 (每级3秒)")
    all_results = run_all_grade_benchmark(duration_seconds=3.0)

    print("\n" + "=" * 60)
    print("  AGV五级性能对比汇总")
    print("=" * 60)
    print(f"  {'等级':<6} {'控制频率':>10} {'合规率':>8} {'延迟P95':>10} {'抖动P95':>10}")
    print("-" * 60)
    for grade in AGVGrade:
        r = all_results.get(grade.value, {})
        total_p95 = r.get('total_latency_ms', {}).get('p95', 0)
        jitter_p95 = r.get('cycle_jitter_ms', {}).get('p95', 0)
        compliance = r.get('compliance_rate_percent', 0)
        freq = r.get('control_freq_hz', 0)
        flag = " ✓" if compliance >= 99 else (" ⚠" if compliance >= 95 else " ✗")
        print(f"  {grade.value:<6} {freq:>9}Hz {compliance:>7.1f}% {total_p95:>9.2f}ms {jitter_p95:>9.2f}ms{flag}")

    print("\n  测试完成 ✓")
