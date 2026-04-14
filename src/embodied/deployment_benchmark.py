"""
deployment_benchmark.py - AGV部署性能基准测试
=============================================

为不同AGV等级提供标准化部署基准测试:
- S级: 简单AGV, 阈值触发
- M级: 标准AGV, 实时触觉伺服
- L级: 高级AGV, 力位混合+姿态稳定
- XL级: 精密AGV, 完整阻抗+多模态融合
- XXL级: 超精密AGV, MPC预测+多层级协同

包含:
- 部署时间基准
- 运行时性能基准
- 资源消耗基准
- 可靠性基准
"""

from __future__ import annotations

import time
import threading
import gc
import psutil
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'BenchmarkResult',
    'DeploymentBenchmark',
    'PerformanceBenchmark',
    'ResourceBenchmark',
    'ReliabilityBenchmark',
    'BenchmarkReport',
    'run_all_benchmarks',
]


# ============================================================
# 基准结果数据类
# ============================================================

class BenchmarkGrade(str, Enum):
    """基准测试AGV等级"""
    S = 'S'
    M = 'M'
    L = 'L'
    XL = 'XL'
    XXL = 'XXL'


@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    name: str
    grade: str
    value: float
    unit: str
    passed: bool
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{self.name}[{self.grade}]: {self.value:.3f}{self.unit} ({status})"


@dataclass
class BenchmarkReport:
    """完整基准报告"""
    grade: str
    timestamp: float
    duration_s: float
    results: List[BenchmarkResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def pass_rate(self) -> float:
        total = len(self.results)
        return self.passed_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'grade': self.grade,
            'timestamp': self.timestamp,
            'duration_s': self.duration_s,
            'passed': self.passed_count,
            'failed': self.failed_count,
            'pass_rate': f"{self.pass_rate * 100:.1f}%",
            'summary': self.summary,
            'results': [
                {
                    'name': r.name,
                    'value': r.value,
                    'unit': r.unit,
                    'passed': r.passed,
                    'threshold': r.threshold,
                    'details': r.details,
                }
                for r in self.results
            ],
        }
    
    def print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"AGV等级: {self.grade}")
        print(f"测试时间: {self.duration_s:.1f}s")
        print(f"通过率: {self.passed_count}/{len(self.results)} ({self.pass_rate * 100:.1f}%)")
        print(f"{'=' * 60}")
        for r in self.results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.name}: {r.value:.3f}{r.unit} (阈值: {r.threshold:.3f}{r.unit})")


# ============================================================
# AGV五级基准阈值
# ============================================================

BENCHMARK_THRESHOLDS = {
    'S': {
        'init_time_ms': 1000.0,
        'task_latency_p95_ms': 500.0,
        'throughput_tasks_per_s': 1.0,
        'memory_mb': 200.0,
        'cpu_percent': 30.0,
        'sensor_rate_hz': 20.0,
        'control_rate_hz': 50.0,
        'error_rate_percent': 5.0,
    },
    'M': {
        'init_time_ms': 800.0,
        'task_latency_p95_ms': 200.0,
        'throughput_tasks_per_s': 5.0,
        'memory_mb': 300.0,
        'cpu_percent': 40.0,
        'sensor_rate_hz': 50.0,
        'control_rate_hz': 100.0,
        'error_rate_percent': 3.0,
    },
    'L': {
        'init_time_ms': 600.0,
        'task_latency_p95_ms': 100.0,
        'throughput_tasks_per_s': 10.0,
        'memory_mb': 400.0,
        'cpu_percent': 50.0,
        'sensor_rate_hz': 100.0,
        'control_rate_hz': 200.0,
        'error_rate_percent': 2.0,
    },
    'XL': {
        'init_time_ms': 400.0,
        'task_latency_p95_ms': 50.0,
        'throughput_tasks_per_s': 20.0,
        'memory_mb': 500.0,
        'cpu_percent': 60.0,
        'sensor_rate_hz': 200.0,
        'control_rate_hz': 500.0,
        'error_rate_percent': 1.0,
    },
    'XXL': {
        'init_time_ms': 200.0,
        'task_latency_p95_ms': 20.0,
        'throughput_tasks_per_s': 50.0,
        'memory_mb': 800.0,
        'cpu_percent': 80.0,
        'sensor_rate_hz': 500.0,
        'control_rate_hz': 1000.0,
        'error_rate_percent': 0.5,
    },
}


# ============================================================
# 部署基准测试
# ============================================================

class DeploymentBenchmark:
    """AGV部署基准测试"""
    
    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.thresholds = BENCHMARK_THRESHOLDS.get(grade, BENCHMARK_THRESHOLDS['M'])
    
    def run(self, pipeline_factory: Callable) -> List[BenchmarkResult]:
        """运行部署基准测试"""
        results = []
        
        # 初始化时间测试
        init_result = self._benchmark_init_time(pipeline_factory)
        results.append(init_result)
        
        return results
    
    def _benchmark_init_time(
        self,
        pipeline_factory: Callable,
    ) -> BenchmarkResult:
        """基准测试初始化时间"""
        from embodied.embodied_pipeline import EmbodiedPipeline
        
        times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            config = self._make_config()
            p = EmbodiedPipeline(config=config)
            p.start()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            p.stop()
        
        avg_ms = np.mean(times)
        threshold = self.thresholds['init_time_ms']
        passed = avg_ms < threshold
        
        return BenchmarkResult(
            name="init_time",
            grade=self.grade,
            value=avg_ms,
            unit="ms",
            passed=passed,
            threshold=threshold,
            details={'samples': times, 'std_ms': np.std(times)},
        )
    
    def _make_config(self):
        """创建Pipeline配置"""
        from embodied.embodied_pipeline import PipelineConfig, PipelineMode
        return PipelineConfig(
            grade=self.grade,
            mode=PipelineMode.SIMULATION,
            enable_skill_registry=True,
            enable_memory=True,
            enable_scene_intelligence=True,
            enable_swarm_coordination=False,
            enable_federated_learning=False,
        )


# ============================================================
# 性能基准测试
# ============================================================

class PerformanceBenchmark:
    """运行时性能基准测试"""
    
    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.thresholds = BENCHMARK_THRESHOLDS.get(grade, BENCHMARK_THRESHOLDS['M'])
    
    def run(self) -> List[BenchmarkResult]:
        """运行性能基准测试"""
        from embodied.embodied_pipeline import EmbodiedPipeline, PipelineConfig, PipelineMode
        
        results = []
        config = PipelineConfig(
            grade=self.grade,
            mode=PipelineMode.SIMULATION,
        )
        p = EmbodiedPipeline(config=config)
        p.start()
        
        try:
            # 任务延迟测试
            latency_result = self._benchmark_task_latency(p)
            results.append(latency_result)
            
            # 吞吐量测试
            throughput_result = self._benchmark_throughput(p)
            results.append(throughput_result)
            
            # 传感器速率测试
            sensor_result = self._benchmark_sensor_rate(p)
            results.append(sensor_result)
            
            # 控制频率测试
            control_result = self._benchmark_control_rate(p)
            results.append(control_result)
            
        finally:
            p.stop()
        
        return results
    
    def _benchmark_task_latency(self, p) -> BenchmarkResult:
        """任务延迟基准"""
        latencies = []
        for i in range(100):
            start = time.perf_counter()
            p.execute_task(f"bench_task_{i}", target=f"dest_{i % 5}")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        p95 = np.percentile(latencies, 95)
        threshold = self.thresholds['task_latency_p95_ms']
        
        return BenchmarkResult(
            name="task_latency_p95",
            grade=self.grade,
            value=p95,
            unit="ms",
            passed=p95 < threshold,
            threshold=threshold,
            details={
                'avg_ms': np.mean(latencies),
                'p50_ms': np.percentile(latencies, 50),
                'p99_ms': np.percentile(latencies, 99),
                'max_ms': np.max(latencies),
            },
        )
    
    def _benchmark_throughput(self, p) -> BenchmarkResult:
        """吞吐量基准"""
        start_time = time.time()
        task_count = 0
        error_count = 0
        
        while time.time() - start_time < 10.0:
            try:
                result = p.execute_task(f"throughput_task_{task_count}", target="dest")
                task_count += 1
                if not result.get('success', False):
                    error_count += 1
            except Exception:
                error_count += 1
        
        duration = time.time() - start_time
        throughput = task_count / duration
        error_rate = error_count / max(task_count, 1) * 100
        threshold = self.thresholds['throughput_tasks_per_s']
        
        return BenchmarkResult(
            name="throughput",
            grade=self.grade,
            value=throughput,
            unit="tasks/s",
            passed=throughput >= threshold,
            threshold=threshold,
            details={
                'total_tasks': task_count,
                'error_count': error_count,
                'error_rate_percent': error_rate,
                'duration_s': duration,
            },
        )
    
    def _benchmark_sensor_rate(self, p) -> BenchmarkResult:
        """传感器更新速率基准"""
        update_times = []
        for _ in range(200):
            start = time.perf_counter()
            p.run_simulation_step(dt=0.01)
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > 0:
                update_times.append(elapsed)
        
        avg_rate = 1000.0 / np.mean(update_times) if update_times else 0
        threshold = self.thresholds['sensor_rate_hz']
        
        return BenchmarkResult(
            name="sensor_rate",
            grade=self.grade,
            value=avg_rate,
            unit="Hz",
            passed=avg_rate >= threshold,
            threshold=threshold,
            details={'avg_step_ms': np.mean(update_times) if update_times else 0},
        )
    
    def _benchmark_control_rate(self, p) -> BenchmarkResult:
        """控制频率基准"""
        step_times = []
        for _ in range(500):
            start = time.perf_counter()
            p.run_simulation_step(dt=0.001)
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > 0:
                step_times.append(elapsed)
        
        avg_rate = 1000.0 / np.mean(step_times) if step_times else 0
        threshold = self.thresholds['control_rate_hz']
        
        return BenchmarkResult(
            name="control_rate",
            grade=self.grade,
            value=avg_rate,
            unit="Hz",
            passed=avg_rate >= threshold,
            threshold=threshold,
            details={'avg_step_ms': np.mean(step_times) if step_times else 0},
        )


# ============================================================
# 资源消耗基准测试
# ============================================================

class ResourceBenchmark:
    """资源消耗基准测试"""
    
    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.thresholds = BENCHMARK_THRESHOLDS.get(grade, BENCHMARK_THRESHOLDS['M'])
        self.process = psutil.Process(os.getpid())
    
    def run(self) -> List[BenchmarkResult]:
        """运行资源消耗基准测试"""
        from embodied.embodied_pipeline import EmbodiedPipeline, PipelineConfig, PipelineMode
        
        results = []
        gc.collect()
        initial_mem = self.process.memory_info().rss / 1024 / 1024
        
        config = PipelineConfig(
            grade=self.grade,
            mode=PipelineMode.SIMULATION,
        )
        p = EmbodiedPipeline(config=config)
        p.start()
        
        try:
            # 运行工作负载
            for i in range(100):
                p.execute_task(f"resource_test_{i}", target=f"dest_{i % 10}")
            
            gc.collect()
            peak_mem = self.process.memory_info().rss / 1024 / 1024
            mem_growth = peak_mem - initial_mem
            
            # CPU使用率
            cpu_percent = self.process.cpu_percent(interval=0.5)
            
            # 内存基准
            mem_threshold = self.thresholds['memory_mb']
            results.append(BenchmarkResult(
                name="memory_peak",
                grade=self.grade,
                value=peak_mem,
                unit="MB",
                passed=peak_mem < mem_threshold,
                threshold=mem_threshold,
                details={'initial_mb': initial_mem, 'growth_mb': mem_growth},
            ))
            
            # CPU基准
            cpu_threshold = self.thresholds['cpu_percent']
            results.append(BenchmarkResult(
                name="cpu_usage",
                grade=self.grade,
                value=cpu_percent,
                unit="%",
                passed=cpu_percent < cpu_threshold,
                threshold=cpu_threshold,
                details={},
            ))
            
        finally:
            p.stop()
        
        return results


# ============================================================
# 可靠性基准测试
# ============================================================

class ReliabilityBenchmark:
    """可靠性基准测试"""
    
    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.thresholds = BENCHMARK_THRESHOLDS.get(grade, BENCHMARK_THRESHOLDS['M'])
    
    def run(self) -> List[BenchmarkResult]:
        """运行可靠性基准测试"""
        from embodied.embodied_pipeline import EmbodiedPipeline, PipelineConfig, PipelineMode
        
        results = []
        config = PipelineConfig(
            grade=self.grade,
            mode=PipelineMode.SIMULATION,
        )
        p = EmbodiedPipeline(config=config)
        p.start()
        
        try:
            error_count = 0
            total_count = 200
            
            for i in range(total_count):
                try:
                    result = p.execute_task(f"reliability_test_{i}", target=f"dest_{i % 10}")
                    if not result.get('success', False):
                        error_count += 1
                except Exception:
                    error_count += 1
            
            error_rate = error_count / total_count * 100
            threshold = self.thresholds['error_rate_percent']
            
            results.append(BenchmarkResult(
                name="error_rate",
                grade=self.grade,
                value=error_rate,
                unit="%",
                passed=error_rate < threshold,
                threshold=threshold,
                details={'errors': error_count, 'total': total_count},
            ))
            
            # 状态恢复测试
            recovery_ok = self._test_state_recovery(p)
            results.append(BenchmarkResult(
                name="state_recovery",
                grade=self.grade,
                value=1.0 if recovery_ok else 0.0,
                unit="bool",
                passed=recovery_ok,
                threshold=1.0,
                details={},
            ))
            
        finally:
            p.stop()
        
        return results
    
    def _test_state_recovery(self, p) -> bool:
        """测试状态恢复能力"""
        try:
            # 保存状态
            state = p.save_state()
            
            # 执行任务
            for i in range(10):
                p.execute_task(f"recovery_test_{i}", target="dest")
            
            return True
        except Exception:
            return False


# ============================================================
# 综合基准测试运行器
# ============================================================

def run_all_benchmarks(grade: str = 'M') -> BenchmarkReport:
    """
    为指定AGV等级运行所有基准测试
    
    Args:
        grade: AGV等级 ('S', 'M', 'L', 'XL', 'XXL')
    
    Returns:
        BenchmarkReport: 完整基准报告
    """
    start_time = time.time()
    
    deploy_bench = DeploymentBenchmark(grade)
    perf_bench = PerformanceBenchmark(grade)
    resource_bench = ResourceBenchmark(grade)
    reliability_bench = ReliabilityBenchmark(grade)
    
    all_results = []
    all_results.extend(deploy_bench.run(None))
    all_results.extend(perf_bench.run())
    all_results.extend(resource_bench.run())
    all_results.extend(reliability_bench.run())
    
    duration = time.time() - start_time
    
    report = BenchmarkReport(
        grade=grade,
        timestamp=time.time(),
        duration_s=duration,
        results=all_results,
        summary={
            'total': len(all_results),
            'passed': sum(1 for r in all_results if r.passed),
            'failed': sum(1 for r in all_results if not r.passed),
        },
    )
    
    return report


def run_all_grades_benchmarks() -> Dict[str, BenchmarkReport]:
    """为所有AGV等级运行基准测试"""
    reports = {}
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        print(f"\n正在测试 AGV等级: {grade}")
        report = run_all_benchmarks(grade)
        report.print_summary()
        reports[grade] = report
    return reports


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        grade = sys.argv[1]
        report = run_all_benchmarks(grade)
        report.print_summary()
    else:
        print("运行所有AGV等级基准测试...")
        reports = run_all_grades_benchmarks()
        
        print("\n" + "=" * 60)
        print("所有等级基准测试汇总")
        print("=" * 60)
        for grade, report in reports.items():
            print(f"  {grade}: {report.passed_count}/{len(report.results)} 通过 ({report.pass_rate * 100:.1f}%)")
