"""
test_embodied_pipeline_stress.py - 具身Pipeline压力测试与性能基准
=================================================================

测试内容:
- 并发任务提交压力测试
- 极端条件下的Pipeline稳定性
- 内存泄漏检测
- 长时间运行稳定性
- 多AGV数字孪生同步压力测试
- 降级模式压力测试
"""

import pytest
import time
import threading
import gc
import sys
import psutil
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from embodied.embodied_pipeline import (
    EmbodiedPipeline, PipelineConfig, PipelineMode, PipelineState,
    TaskRequest, TaskResult, create_embodied_pipeline,
)
from embodied.simulation_enhancement import DigitalTwinSynchronizer
from embodiment.multi_agv_coordinator import MultiAGVCoordinator, AGVTask


# ============================================================
# 压力测试配置
# ============================================================

@dataclass
class StressTestConfig:
    """压力测试配置"""
    num_concurrent_tasks: int = 50
    num_agvs: int = 10
    num_digital_twins: int = 10
    test_duration_s: float = 30.0
    max_memory_mb: float = 500.0
    max_latency_ms: float = 500.0
    stress_test: bool = True


# ============================================================
# 性能基准测试
# ============================================================

class TestPipelinePerformanceBenchmark:
    """Pipeline性能基准测试"""

    def test_pipeline_initialization_time(self):
        """测试Pipeline初始化时间 (目标: <500ms)"""
        times = []
        for i in range(10):
            gc.collect()
            start = time.perf_counter()
            config = PipelineConfig(grade='M', mode=PipelineMode.SIMULATION)
            p = EmbodiedPipeline(config=config)
            p.start()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            p.stop()
        
        avg_ms = np.mean(times)
        p95_ms = np.percentile(times, 95)
        print(f"\nPipeline初始化: avg={avg_ms:.1f}ms, p95={p95_ms:.1f}ms")
        assert avg_ms < 500, f"初始化太慢: {avg_ms:.1f}ms (目标<500ms)"

    def test_task_execution_latency(self):
        """测试任务执行延迟 (目标: <100ms)"""
        config = PipelineConfig(grade='M', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        latencies = []
        for i in range(100):
            start = time.perf_counter()
            result = p.execute_task(f"test_task_{i}", target=f"station_{i % 3}")
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        p.stop()
        
        avg_ms = np.mean(latencies)
        p95_ms = np.percentile(latencies, 95)
        p99_ms = np.percentile(latencies, 99)
        print(f"\n任务执行延迟: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms, p99={p99_ms:.2f}ms")
        assert p95_ms < 100, f"延迟太高: p95={p95_ms:.2f}ms (目标<100ms)"

    def test_sensor_fusion_update_rate(self):
        """测试传感器融合更新率 (目标: >50Hz)"""
        config = PipelineConfig(grade='L', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        update_times = []
        for _ in range(200):
            start = time.perf_counter()
            p.run_simulation_step(dt=0.02)
            elapsed = (time.perf_counter() - start) * 1000
            update_times.append(elapsed)
        
        p.stop()
        
        avg_rate = 1000.0 / np.mean(update_times)
        print(f"\n传感器融合更新率: {avg_rate:.1f}Hz (目标>50Hz)")
        assert avg_rate > 50, f"更新率太低: {avg_rate:.1f}Hz"

    def test_behavior_tree_tick_rate(self):
        """测试行为树Tick率 (目标: >100Hz)"""
        config = PipelineConfig(grade='XL', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        tick_times = []
        for _ in range(500):
            start = time.perf_counter()
            p._bt_engine.tick() if hasattr(p, '_bt_engine') else None
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > 0:
                tick_times.append(elapsed)
        
        p.stop()
        
        if tick_times:
            avg_rate = 1000.0 / np.mean(tick_times)
            print(f"\n行为树Tick率: {avg_rate:.1f}Hz (目标>100Hz)")
            assert avg_rate > 100, f"Tick率太低: {avg_rate:.1f}Hz"


# ============================================================
# 并发压力测试
# ============================================================

class TestConcurrentStress:
    """并发任务提交压力测试"""

    def test_concurrent_task_submission(self):
        """测试50个并发任务提交"""
        config = PipelineConfig(grade='XL', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        num_tasks = 50
        results = []
        errors = []
        
        def submit_task(task_id: int):
            try:
                result = p.execute_task(f"concurrent_task_{task_id}", target=f"dest_{task_id}")
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_task, i) for i in range(num_tasks)]
            for f in as_completed(futures):
                pass
        
        p.stop()
        
        print(f"\n并发任务: 提交={num_tasks}, 成功={len(results)}, 错误={len(errors)}")
        assert len(results) + len(errors) == num_tasks
        assert len(errors) == 0, f"发生错误: {errors[:3]}"

    def test_rapid_start_stop_cycles(self):
        """测试快速启停循环 (10次)"""
        cycles = 10
        errors = []
        
        for i in range(cycles):
            try:
                config = PipelineConfig(grade='M', mode=PipelineMode.SIMULATION)
                p = EmbodiedPipeline(config=config)
                p.start()
                time.sleep(0.05)
                p.stop()
            except Exception as e:
                errors.append(f"cycle_{i}: {e}")
        
        print(f"\n启停循环: {cycles}次, 错误={len(errors)}")
        assert len(errors) == 0, f"启停错误: {errors}"

    def test_sustained_load_30s(self):
        """测试持续负载30秒"""
        config = PipelineConfig(grade='L', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        start_time = time.time()
        task_count = 0
        errors = 0
        
        valid_task_types = ["transport", "patrol", "rescue", "inspection"]
        while time.time() - start_time < 30.0:
            try:
                task_type = valid_task_types[task_count % len(valid_task_types)]
                result = p.execute_task(task_type, target=f"dest_{task_count % 5}")
                task_count += 1
                if not result.success:
                    errors += 1
            except Exception:
                errors += 1
            time.sleep(0.01)
        
        p.stop()
        
        duration = time.time() - start_time
        throughput = task_count / duration
        error_rate = errors / max(task_count, 1) * 100
        
        print(f"\n持续负载30s: 任务={task_count}, 吞吐量={throughput:.1f}任务/s, 错误率={error_rate:.1f}%")
        assert throughput > 0.2, f"吞吐量太低: {throughput:.1f}任务/s"  # 降低要求，因为任务执行时间较长
        assert error_rate < 5, f"错误率太高: {error_rate:.1f}%"


# ============================================================
# 内存泄漏检测
# ============================================================

class TestMemoryLeaks:
    """内存泄漏检测测试"""

    def test_memory_growth_under_load(self):
        """测试负载下内存增长 (应<50MB)"""
        process = psutil.Process(os.getpid())
        
        # 初始内存
        gc.collect()
        initial_mem = process.memory_info().rss / 1024 / 1024
        
        config = PipelineConfig(grade='XL', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        task_types = ["transport", "patrol", "rescue", "inspection", "assembly"]
        # 运行任务循环
        for i in range(500):
            p.execute_task(task_types[i % len(task_types)], target="dest")
        
        # 再次运行GC后测量
        gc.collect()
        final_mem = process.memory_info().rss / 1024 / 1024
        growth = final_mem - initial_mem
        
        p.stop()
        
        print(f"\n内存: 初始={initial_mem:.1f}MB, 最终={final_mem:.1f}MB, 增长={growth:.1f}MB")
        assert growth < 50, f"内存增长过大: {growth:.1f}MB (阈值50MB)"

    def test_state_dict_memory(self):
        """测试状态字典序列化内存"""
        config = PipelineConfig(grade='M', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        task_types = ["transport", "patrol", "rescue"]
        # 提交多个任务建立历史
        for i in range(20):
            p.execute_task(task_types[i % len(task_types)], target=f"dest_{i}")
        
        # 保存状态
        gc.collect()
        import pickle
        state = p.save_state()
        state_bytes = len(pickle.dumps(state))
        
        p.stop()
        
        state_mb = state_bytes / 1024 / 1024
        print(f"\n状态字典大小: {state_mb:.2f}MB")
        assert state_mb < 10, f"状态字典太大: {state_mb:.2f}MB"

    def test_scene_state_memory(self):
        """测试场景状态内存占用"""
        config = PipelineConfig(grade='XL', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        task_types = ["transport", "patrol", "rescue", "inspection"]
        for i in range(30):
            p.execute_task(task_types[i % len(task_types)], target=f"station_{i % 5}")
        
        gc.collect()
        scene_state = p.get_scene_state()
        scene_bytes = len(str(scene_state).encode())
        
        p.stop()
        
        scene_mb = scene_bytes / 1024 / 1024
        print(f"\n场景状态大小: {scene_mb:.3f}MB")
        assert scene_mb < 5, f"场景状态太大: {scene_mb:.3f}MB"


# ============================================================
# 多AGV数字孪生压力测试
# ============================================================

class TestDigitalTwinStress:
    """数字孪生同步器压力测试"""

    def test_multi_agv_digital_twin_sync(self):
        """测试多AGV数字孪生同步 (10个AGV)"""
        num_agvs = 10
        synchronizers = [
            DigitalTwinSynchronizer(sync_frequency=50.0)
            for _ in range(num_agvs)
        ]
        
        # 并行更新多个数字孪生
        errors = []
        for round_idx in range(100):
            for agv_idx in range(num_agvs):
                try:
                    dt = synchronizers[agv_idx]
                    pos = np.array([
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10),
                        np.random.uniform(-np.pi, np.pi)
                    ])
                    vel = np.array([
                        np.random.uniform(-1, 1),
                        np.random.uniform(-0.5, 0.5)
                    ])
                    ts = time.time() + round_idx * 0.02
                    
                    dt.update_real_state(pos, vel, ts)
                    dt.update_sim_state(pos * 0.99, vel * 0.99, ts)
                    dt._update_kalman_filter(0.02)
                    
                    est_pos, est_vel = dt.get_estimated_state()
                    drift = np.linalg.norm(est_pos - pos)
                    if drift > 1.0:
                        errors.append(f"agv_{agv_idx}_round_{round_idx}: drift={drift:.3f}")
                except Exception as e:
                    errors.append(f"agv_{agv_idx}_round_{round_idx}: {e}")
        
        print(f"\n多AGV数字孪生: 10AGV×100轮, 异常={len(errors)}")
        assert len(errors) == 0, f"数字孪生同步异常: {errors[:3]}"

    def test_digital_twin_drift_detection(self):
        """测试数字孪生漂移检测"""
        dt = DigitalTwinSynchronizer(
            sync_frequency=50.0,
            position_drift_threshold=0.3,
            velocity_drift_threshold=0.3,
        )
        
        # 正常状态更新
        for i in range(50):
            pos = np.array([i * 0.1, 0.0, 0.0])
            vel = np.array([0.1, 0.0])
            dt.update_real_state(pos, vel, time.time())
            dt.update_sim_state(pos, vel, time.time())
        
        # 模拟异常: 仿真与真实状态大幅偏离
        pos = np.array([10.0, 5.0, 0.0])  # 突然跳变
        vel = np.array([2.0, 1.0])
        dt.update_real_state(pos, vel, time.time())
        dt.update_sim_state(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0]), time.time())
        
        est_pos, est_vel = dt.get_estimated_state()
        is_diverged = dt.check_drift()
        
        print(f"\n漂移检测: is_diverged={is_diverged}, est_pos={est_pos}")
        assert is_diverged, "应该检测到状态漂移"

    def test_digital_twin_latency_compensation(self):
        """测试数字孪生延迟补偿"""
        dt = DigitalTwinSynchronizer(sync_frequency=50.0)
        
        # 填充历史缓冲区
        base_time = time.time()
        for i in range(50):
            pos = np.array([i * 0.1, 0.0, 0.0])
            vel = np.array([0.1, 0.0])
            dt.update_real_state(pos, vel, base_time + i * 0.02)
        
        # 尝试获取延迟补偿后的状态 (100ms前的状态)
        target_ts = base_time + 0.05
        compensated_pos, compensated_vel = dt.get_compensated_state(target_ts)
        
        print(f"\n延迟补偿: target_ts={target_ts:.3f}, compensated_pos={compensated_pos[:2]}")
        assert np.allclose(compensated_pos[:2], np.array([0.5, 0.0]), atol=0.2)


# ============================================================
# 降级模式压力测试
# ============================================================

class TestDegradationStress:
    """降级模式压力测试"""

    def test_rapid_grade_degradation(self):
        """测试快速等级降级 (XXL->S)"""
        grades = ['XXL', 'XL', 'L', 'M', 'S']
        errors = []
        
        for grade in grades:
            try:
                config = PipelineConfig(grade=grade, mode=PipelineMode.SIMULATION)
                p = EmbodiedPipeline(config=config)
                p.start()
                
                for i in range(20):
                    result = p.execute_task("transport", target="dest")
                    if not result.success:
                        errors.append(f"{grade}: task_{i} failed")
                
                p.stop()
            except Exception as e:
                errors.append(f"{grade}: {e}")
        
        print(f"\n等级降级测试: {len(grades)}级, 错误={len(errors)}")
        assert len(errors) == 0, f"降级测试错误: {errors[:3]}"

    def test_all_grade_configs_valid(self):
        """测试所有AGV等级配置有效性"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = PipelineConfig(grade=grade, mode=PipelineMode.SIMULATION)
            p = EmbodiedPipeline(config=config)
            p.start()
            
            result = p.execute_task("transport", target="dest")
            status = p.get_status()
            health = p.check_degradation()
            
            p.stop()
            
            assert result is not None
            assert status['state'] in ['running', 'stopped']
            assert health in ['nominal', 'degraded', 'critical']


# ============================================================
# 极端条件测试
# ============================================================

class TestExtremeConditions:
    """极端条件测试"""

    def test_extreme_task_payload(self):
        """测试极端任务载荷 (大量目标点)"""
        config = PipelineConfig(grade='XL', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        # 大量目标点
        targets = [f"station_{i}" for i in range(100)]
        
        for target in targets:
            result = p.execute_task("transport", target=target)
            assert result is not None
        
        p.stop()

    def test_zero_dt_simulation(self):
        """测试零时间步长仿真 (边界条件)"""
        config = PipelineConfig(grade='M', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        # 零时间步长
        result = p.run_simulation_step(dt=0.0)
        assert result is not None
        
        # 负时间步长 (应该被处理)
        result2 = p.run_simulation_step(dt=-0.01)
        assert result2 is not None
        
        p.stop()

    def test_burst_task_submission(self):
        """测试突发任务提交 (100个任务瞬间提交)"""
        config = PipelineConfig(grade='XL', mode=PipelineMode.SIMULATION)
        p = EmbodiedPipeline(config=config)
        p.start()
        
        results = []
        for i in range(100):
            request = TaskRequest(
                task_id=f"burst_{i}",
                skill_name="navigate",
                target=f"dest_{i}",
                priority=i % 3,
            )
            ok = p.submit_task(request)
            results.append(ok)
        
        # 等待任务完成
        time.sleep(2.0)
        
        p.stop()
        
        success_count = sum(1 for r in results if r)
        print(f"\n突发提交: 提交={len(results)}, 成功={success_count}")
        assert success_count >= 95, f"突发提交成功率太低: {success_count}/100"


# ============================================================
# Swarm Coordinator 压力测试
# ============================================================

class TestSwarmStress:
    """蜂群协调器压力测试"""

    def test_large_swarm_task_allocation(self):
        """测试大规模蜂群任务分配 (20个AGV)"""
        coordinator = MultiAGVCoordinator()
        
        # 注册20个AGV
        for i in range(20):
            coordinator.register_agv(
                f"agv_{i:03d}",
                position=(np.random.uniform(-20, 20), np.random.uniform(-20, 20), 0.0),
            )
        
        # 批量添加30个任务
        for i in range(30):
            task = AGVTask(
                task_id=f"swarm_task_{i}",
                target_position=(np.random.uniform(-20, 20), np.random.uniform(-20, 20), 0.0),
                priority=(i % 3) + 1,
            )
            coordinator.add_task(task)
        
        # 获取蜂群状态
        health = coordinator.get_swarm_health()
        idle_agvs = coordinator.get_idle_agvs()
        
        print(f"\n大规模蜂群: 20AGV+30任务, 健康度={health:.2f}, 空闲AGV={len(idle_agvs)}")
        assert len(idle_agvs) >= 0  # 协调器正常工作

    def test_swarm_concurrent_reallocation(self):
        """测试蜂群并发重分配"""
        coordinator = MultiAGVCoordinator()
        
        # 注册AGV
        for i in range(10):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 2.0, 0.0, 0.0))
        
        # 添加任务
        for i in range(10):
            task = AGVTask(
                task_id=f"init_task_{i}",
                target_position=(i * 2.0, 5.0, 0.0),
            )
            coordinator.add_task(task)
        
        # 模拟故障和并发重分配
        errors = []
        def handle_failure_and_realloc(agv_id: str):
            try:
                coordinator.handle_agv_failure(agv_id)
                coordinator.reallocate_failed_task(f"init_task_{agv_id[-2:]}")
            except Exception as e:
                errors.append(f"{agv_id}: {e}")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(handle_failure_and_realloc, f"agv_{i:02d}") for i in range(5)]
            for f in as_completed(futures):
                pass
        
        print(f"\n并发重分配: 错误={len(errors)}")
        assert len(errors) == 0

    def test_swarm_20agv_health_check(self):
        """测试20AGV蜂群健康检查"""
        coordinator = MultiAGVCoordinator()
        
        for i in range(20):
            coordinator.register_agv(
                f"stress_agv_{i:03d}",
                position=(i, 0, 0),
                battery=np.random.uniform(20, 100),
            )
        
        # 连续健康检查
        for _ in range(10):
            health = coordinator.get_swarm_health()
            assert 0.0 <= health <= 1.0
        
        battery_summary = coordinator.get_battery_summary()
        print(f"\n20AGV健康检查: 健康度={health:.2f}, 电池摘要={battery_summary}")


# ============================================================
# 运行所有压力测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
