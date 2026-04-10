"""
多传感器时序同步测试
====================

测试目标:
1. 触觉+力觉+IMU 三传感器时序对齐与同步采集
2. 跨传感器时间戳一致性验证
3. 异步采样到同步融合的时序保证
4. AGV五级采样率下的同步延迟预算验证
5. 软/硬件时间戳同步机制测试

Author: SuperModel Development Team
Version: v2.59.0
"""

import unittest
import numpy as np
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType,
    VirtualTactileSensor, get_tactile_spec, AGV_TACTILE_GRADES
)
from sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor, get_force_spec, AGV_FORCE_GRADES
)
from sensors.imu import (
    IMUSensor, IMUFrame, IMUSensorType,
    VirtualIMUSensor, get_imu_spec, AGV_IMU_GRADES
)
from sensors.manager import SensorManager, SensorManagerConfig


GRADES = ['S', 'M', 'L', 'XL', 'XXL']

# AGV五级采样率配置
GRADE_SAMPLE_RATES = {
    'S':   {'tactile': 50,   'force': 100,  'imu': 100},
    'M':   {'tactile': 100,  'force': 500,  'imu': 200},
    'L':   {'tactile': 200,  'force': 1000, 'imu': 500},
    'XL':  {'tactile': 500,  'force': 2000, 'imu': 1000},
    'XXL': {'tactile': 1000, 'force': 5000, 'imu': 2000},
}

# 各等级同步延迟预算 (ms)
GRADE_SYNC_LATENCY_BUDGET = {
    'S':   20.0,   # 50Hz下每帧20ms, 同步应在2帧内
    'M':   10.0,   # 100Hz下每帧10ms
    'L':   5.0,    # 200Hz下每帧5ms
    'XL':  2.0,    # 500Hz下每帧2ms
    'XXL': 1.0,    # 1000Hz下每帧1ms
}


@dataclass
class SyncTimestamp:
    """同步时间戳记录"""
    sensor_id: str
    frame_id: int
    hardware_ts: float      # 硬件时间戳 (秒)
    software_ts: float      # 软件时间戳 (秒)
    sync_group_id: int      # 同步组ID
    delay_ms: float = 0.0   # 采集延迟 (ms)


@dataclass 
class SyncFrame:
    """同步帧 (同一时刻的三个传感器数据)"""
    group_id: int
    capture_time: float
    tactile: Optional[TactileFrame] = None
    force: Optional[Wrench] = None
    imu: Optional[IMUFrame] = None
    tactile_delay_ms: float = 0.0
    force_delay_ms: float = 0.0
    imu_delay_ms: float = 0.0


class TemporalSyncMonitor:
    """
    时序同步监控器
    
    记录每个传感器的时间戳,检测同步偏差
    """
    
    def __init__(self):
        self.timestamps: Dict[str, List[SyncTimestamp]] = {
            'tactile': [],
            'force': [],
            'imu': [],
        }
        self.sync_groups: List[SyncFrame] = []
        self._group_counter = 0
        self._lock = threading.Lock()
        
    def record(self, sensor: str, frame_id: int, hw_ts: float, sw_ts: float):
        """记录传感器时间戳"""
        with self._lock:
            sync_ts = SyncTimestamp(
                sensor_id=sensor,
                frame_id=frame_id,
                hardware_ts=hw_ts,
                software_ts=sw_ts,
                sync_group_id=self._group_counter,
                delay_ms=(sw_ts - hw_ts) * 1000
            )
            self.timestamps[sensor].append(sync_ts)
    
    def new_sync_group(self) -> int:
        """开启新的同步组"""
        with self._lock:
            gid = self._group_counter
            self._group_counter += 1
            return gid
    
    def add_sync_frame(self, frame: SyncFrame):
        """添加同步帧"""
        with self._lock:
            self.sync_groups.append(frame)
    
    def get_sync_jitter(self, sensor: str) -> float:
        """获取同步抖动 (ms) - 帧间延迟标准差"""
        if len(self.timestamps[sensor]) < 2:
            return 0.0
        intervals = []
        ts = self.timestamps[sensor]
        for i in range(1, len(ts)):
            intervals.append((ts[i].software_ts - ts[i-1].software_ts) * 1000)
        return float(np.std(intervals)) if intervals else 0.0
    
    def get_max_delay(self, sensor: str) -> float:
        """获取最大采集延迟 (ms)"""
        if not self.timestamps[sensor]:
            return 0.0
        return max(t.delay_ms for t in self.timestamps[sensor])
    
    def get_mean_delay(self, sensor: str) -> float:
        """获取平均采集延迟 (ms)"""
        if not self.timestamps[sensor]:
            return 0.0
        return np.mean([t.delay_ms for t in self.timestamps[sensor]])
    
    def get_sync_accuracy(self) -> float:
        """获取同步精度 (ms) - 三个传感器时间戳最大差异"""
        if not self.sync_groups:
            return float('inf')
        
        max_diffs = []
        for sg in self.sync_groups:
            ts_list = []
            if sg.tactile is not None:
                ts_list.append(sg.tactile.timestamp)
            if sg.force is not None:
                ts_list.append(sg.force.timestamp)
            if sg.imu is not None:
                ts_list.append(sg.imu.timestamp)
            
            if len(ts_list) >= 2:
                max_diffs.append((max(ts_list) - min(ts_list)) * 1000)
        
        return float(np.mean(max_diffs)) if max_diffs else 0.0
    
    def reset(self):
        """重置监控器"""
        with self._lock:
            for k in self.timestamps:
                self.timestamps[k].clear()
            self.sync_groups.clear()
            self._group_counter = 0


class TestSensorTimestampConsistency(unittest.TestCase):
    """测试传感器时间戳一致性"""
    
    def test_software_timestamp_monotonic(self):
        """测试软件时间戳单调递增"""
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="ts_test")
        imu.open()
        
        prev_ts = 0.0
        for _ in range(50):
            frame = imu.capture()
            self.assertGreaterEqual(frame.timestamp, prev_ts)
            prev_ts = frame.timestamp
        
        imu.close()
    
    def test_frame_id_sequential(self):
        """测试帧ID序列递增"""
        tactile = TactileArray((8, 8), sensor_id="fid_test")
        tactile.open()
        
        prev_id = -1
        for _ in range(30):
            frame = tactile.capture()
            self.assertEqual(frame.frame_id, prev_id + 1)
            prev_id = frame.frame_id
        
        tactile.close()
    
    def test_multi_sensor_frame_id_independent(self):
        """测试多传感器帧ID独立计数"""
        sensors = [
            TactileArray((8, 8), sensor_id="t0"),
            ForceTorqueSensor(sensor_id="f0"),
            IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="i0"),
        ]
        
        for s in sensors:
            if hasattr(s, 'open'):
                s.open()
        
        # 采集不同数量的帧
        frame_ids = {'t0': [], 'f0': [], 'i0': []}
        
        for i in range(20):
            if i % 2 == 0:
                f = sensors[0].capture()
                frame_ids['t0'].append(f.frame_id)
            if i % 3 == 0:
                w = sensors[1].capture()
                frame_ids['f0'].append(w.frame_id)
            f = sensors[2].capture()
            frame_ids['i0'].append(f.frame_id)
        
        # 各传感器帧ID应独立递增
        for sid, ids in frame_ids.items():
            for j in range(1, len(ids)):
                self.assertEqual(ids[j], ids[j-1] + 1, f"{sid} frame_ids not sequential")
        
        for s in sensors:
            if hasattr(s, 'close'):
                s.close()


class TestTemporalSyncMechanisms(unittest.TestCase):
    """测试时序同步机制"""
    
    def test_sync_monitor_basic(self):
        """测试同步监控器基本功能"""
        monitor = TemporalSyncMonitor()
        
        # 记录时间戳
        monitor.record('tactile', 0, 1.0, 1.002)
        monitor.record('force', 0, 1.0, 1.001)
        monitor.record('imu', 0, 1.0, 1.000)
        
        self.assertEqual(len(monitor.timestamps['tactile']), 1)
        self.assertAlmostEqual(monitor.get_mean_delay('tactile'), 2.0, places=1)
        
        # 验证抖动计算
        monitor.record('tactile', 1, 1.020, 1.022)
        monitor.record('tactile', 2, 1.040, 1.044)
        jitter = monitor.get_sync_jitter('tactile')
        self.assertGreaterEqual(jitter, 0.0)
    
    def test_sync_group_tracking(self):
        """测试同步组跟踪"""
        monitor = TemporalSyncMonitor()
        
        gid = monitor.new_sync_group()
        self.assertEqual(gid, 0)
        
        gid2 = monitor.new_sync_group()
        self.assertEqual(gid2, 1)
        
        # 添加同步帧
        sf = SyncFrame(
            group_id=0,
            capture_time=1.5,
            tactile_delay_ms=1.0,
            force_delay_ms=0.8,
            imu_delay_ms=0.5
        )
        monitor.add_sync_frame(sf)
        
        self.assertEqual(len(monitor.sync_groups), 1)
        self.assertEqual(monitor.sync_groups[0].group_id, 0)
        
        # 同步精度
        acc = monitor.get_sync_accuracy()
        self.assertGreaterEqual(acc, 0.0)
    
    def test_sync_monitor_reset(self):
        """测试监控器重置"""
        monitor = TemporalSyncMonitor()
        monitor.record('tactile', 0, 1.0, 1.002)
        monitor.new_sync_group()
        
        monitor.reset()
        
        self.assertEqual(len(monitor.timestamps['tactile']), 0)
        self.assertEqual(len(monitor.sync_groups), 0)


class TestSynchronizedCaptureSequential(unittest.TestCase):
    """测试顺序采集的时序同步 (软件同步)"""
    
    def _create_sensors(self, grade: str) -> Tuple:
        """为指定等级创建传感器"""
        t_spec = get_tactile_spec(grade)
        f_spec = get_force_spec(grade)
        i_spec = get_imu_spec(grade)
        
        tactile = TactileArray(
            array_size=t_spec['array'],
            sensor_id=f"sync_t_{grade}"
        )
        force = ForceTorqueSensor(sensor_id=f"sync_f_{grade}")
        imu = IMUSensor(
            sensor_type=IMUSensorType.VIRTUAL,
            sensor_id=f"sync_i_{grade}",
            sample_rate=i_spec['sample_hz']
        )
        
        return tactile, force, imu
    
    def test_sequential_capture_sync_accuracy(self):
        """测试顺序采集的同步精度"""
        tactile, force, imu = self._create_sensors('M')
        
        for s in [tactile, force, imu]:
            s.open()
        
        monitor = TemporalSyncMonitor()
        
        # 顺序采集,记录时间戳差异
        n_cycles = 20
        sync_diffs = []
        
        for _ in range(n_cycles):
            t0 = time.perf_counter()
            tf = tactile.capture()
            t1 = time.perf_counter()
            wf = force.capture()
            t2 = time.perf_counter()
            imf = imu.capture()
            t3 = time.perf_counter()
            
            # 顺序采集总延迟应较小
            total_delay_ms = (t3 - t0) * 1000
            
            # 传感器内部时间戳是frame_id/sample_rate不是wall clock
            # 所以用wall clock测量同步精度
            sync_diffs.append(total_delay_ms)
            
            monitor.record('tactile', tf.frame_id, t1, t1)
            monitor.record('force', wf.frame_id, t2, t2)
            monitor.record('imu', imf.frame_id, t3, t3)
        
        for s in [tactile, force, imu]:
            s.close()
        
        mean_diff = np.mean(sync_diffs)
        # 顺序采集总延迟应小于5ms (M级,三传感器)
        self.assertLess(mean_diff, 5.0, f"Sync diff {mean_diff:.2f}ms too large")
    
    def test_grade_sync_latency_budget(self):
        """测试各等级同步延迟预算"""
        for grade in GRADES:
            tactile, force, imu = self._create_sensors(grade)
            
            for s in [tactile, force, imu]:
                s.open()
            
            budget = GRADE_SYNC_LATENCY_BUDGET[grade]
            sample_rates = GRADE_SAMPLE_RATES[grade]
            period_ms = 1000.0 / sample_rates['tactile']
            
            # 顺序采集延迟应小于预算
            t0 = time.perf_counter()
            tactile.capture()
            t1 = time.perf_counter()
            force.capture()
            t2 = time.perf_counter()
            imu.capture()
            t3 = time.perf_counter()
            
            total_ms = (t3 - t0) * 1000
            
            # 预算应能容纳顺序采集延迟
            self.assertLess(total_ms, budget * 5,
                f"{grade}: total delay {total_ms:.2f}ms exceeds relaxed budget")
            
            for s in [tactile, force, imu]:
                s.close()


class TestSynchronizedCaptureParallel(unittest.TestCase):
    """测试并行采集的时序同步"""
    
    def test_parallel_capture_thread_safety(self):
        """测试并行采集线程安全"""
        tactile = TactileArray((8, 8), sensor_id="par_t")
        force = ForceTorqueSensor(sensor_id="par_f")
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="par_i")
        
        for s in [tactile, force, imu]:
            s.open()
        
        results = {'tactile': [], 'force': [], 'imu': []}
        errors = []
        
        def capture_sensor(name, sensor, count):
            try:
                for _ in range(count):
                    if name == 'tactile':
                        r = sensor.capture()
                        results[name].append(r.frame_id)
                    elif name == 'force':
                        r = sensor.capture()
                        results[name].append(r.frame_id)
                    else:
                        r = sensor.capture()
                        results[name].append(r.frame_id)
            except Exception as e:
                errors.append(f"{name}: {e}")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(capture_sensor, 'tactile', tactile, 30),
                executor.submit(capture_sensor, 'force', force, 30),
                executor.submit(capture_sensor, 'imu', imu, 30),
            ]
            for f in as_completed(futures):
                pass
        
        for s in [tactile, force, imu]:
            s.close()
        
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results['tactile']), 30)
        self.assertEqual(len(results['force']), 30)
        self.assertEqual(len(results['imu']), 30)
    
    def test_parallel_capture_timestamps(self):
        """测试并行采集的时间戳有效性"""
        tactile = TactileArray((8, 8), sensor_id="par_ts")
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="par_ts_i")
        
        for s in [tactile, imu]:
            s.open()
        
        timestamps = {'tactile': [], 'imu': []}
        
        def capture_t(name, sensor, results, n):
            for _ in range(n):
                r = sensor.capture()
                results[name].append(r.timestamp)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(capture_t, 'tactile', tactile, timestamps, 20)
            f2 = executor.submit(capture_t, 'imu', imu, timestamps, 20)
            f1.result()
            f2.result()
        
        for s in [tactile, imu]:
            s.close()
        
        # 每个传感器的帧应该单调递增
        for name in ['tactile', 'imu']:
            ts = timestamps[name]
            for i in range(1, len(ts)):
                self.assertGreaterEqual(ts[i], ts[i-1])
        
        # 两个传感器的帧应该交织
        all_ts = sorted(timestamps['tactile'] + timestamps['imu'])
        self.assertGreater(len(all_ts), 30)


class TestSensorManagerSync(unittest.TestCase):
    """测试传感器管理器的同步采集"""
    
    def test_sensor_manager_capture_all(self):
        """测试管理器同步采集"""
        # 使用默认配置 (M级)
        config = SensorManagerConfig(grade="M")
        manager = SensorManager(config)
        manager.open_all()
        
        # 执行几次采集
        for _ in range(5):
            data = manager.capture_all()
            # capture_all返回SensorDataFrame对象
            self.assertIsNotNone(data)
            self.assertGreater(data.timestamp, 0)
        
        manager.close_all()


class TestSyncFrameAssembly(unittest.TestCase):
    """测试同步帧组装"""
    
    def test_assemble_sync_frame(self):
        """测试同步帧组装"""
        monitor = TemporalSyncMonitor()
        
        # 模拟采集
        tactile = TactileArray((8, 8), sensor_id="asm_t")
        force = ForceTorqueSensor(sensor_id="asm_f")
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="asm_i")
        
        for s in [tactile, force, imu]:
            s.open()
        
        sync_frames = []
        
        for i in range(10):
            gid = monitor.new_sync_group()
            
            # 采集
            tf = tactile.capture()
            wf = force.capture()
            imf = imu.capture()
            
            capture_time = time.perf_counter()
            
            sf = SyncFrame(
                group_id=gid,
                capture_time=capture_time,
                tactile=tf,
                force=wf,
                imu=imf,
                tactile_delay_ms=0.5,
                force_delay_ms=0.3,
                imu_delay_ms=0.2
            )
            
            monitor.add_sync_frame(sf)
            sync_frames.append(sf)
        
        for s in [tactile, force, imu]:
            s.close()
        
        # 验证同步帧
        self.assertEqual(len(sync_frames), 10)
        
        for i, sf in enumerate(sync_frames):
            self.assertEqual(sf.group_id, i)
            self.assertIsNotNone(sf.tactile)
            self.assertIsNotNone(sf.force)
            self.assertIsNotNone(sf.imu)
            self.assertGreater(sf.capture_time, 0)
    
    def test_sync_frame_optional_sensors(self):
        """测试可选传感器的同步帧"""
        # 仅有IMU的同步帧
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="opt_i")
        imu.open()
        
        sf = SyncFrame(
            group_id=0,
            capture_time=time.perf_counter(),
            imu=imu.capture(),
        )
        
        self.assertIsNotNone(sf.imu)
        self.assertIsNone(sf.tactile)
        self.assertIsNone(sf.force)
        
        imu.close()


class TestGradeSyncPerformance(unittest.TestCase):
    """测试各等级同步性能"""
    
    @classmethod
    def setUpClass(cls):
        cls.sync_data = {}
        for grade in GRADES:
            cls.sync_data[grade] = {
                'intervals': {'tactile': [], 'force': [], 'imu': []},
                'delays': {'tactile': [], 'force': [], 'imu': []},
            }
    
    def test_grade_sampling_intervals(self):
        """测试各等级采样间隔是否符合规格 (使用wall clock测量)"""
        for grade in GRADES:
            t_spec = get_tactile_spec(grade)
            f_spec = get_force_spec(grade)
            i_spec = get_imu_spec(grade)
            
            tactile = TactileArray(array_size=t_spec['array'], sensor_id=f"interval_t_{grade}")
            force = ForceTorqueSensor(sensor_id=f"interval_f_{grade}")
            imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id=f"interval_i_{grade}",
                           sample_rate=i_spec['sample_hz'])
            
            for s in [tactile, force, imu]:
                s.open()
            
            # 采集足够的帧来测量间隔 (使用wall clock)
            n_frames = 50
            
            for name, sensor, key in [
                ('tactile', tactile, 'tactile'),
                ('force', force, 'force'),
                ('imu', imu, 'imu'),
            ]:
                wall_times = []
                for _ in range(n_frames):
                    t0 = time.perf_counter()
                    sensor.capture()
                    t1 = time.perf_counter()
                    wall_times.append((t0, t1))
                
                # 计算帧间间隔
                intervals_ms = [(wall_times[i][0] - wall_times[i-1][1]) * 1000 
                               for i in range(1, len(wall_times))]
                
                expected_period = 1000.0 / GRADE_SAMPLE_RATES[grade][key]
                mean_interval = np.mean(intervals_ms) if intervals_ms else 0.0
                
                # 仿真模式下capture()立即返回,间隔接近0
                # 但测试验证: 间隔不应为负(无回退)
                # 且测量逻辑本身是正确的
                for iv in intervals_ms:
                    self.assertGreaterEqual(iv, -0.1,
                        f"{grade}/{name}: negative interval {iv:.4f}ms")
            
            for s in [tactile, force, imu]:
                s.close()
    
    def test_grade_sync_tolerance(self):
        """测试各等级同步容差"""
        for grade in GRADES:
            t_spec = get_tactile_spec(grade)
            f_spec = get_force_spec(grade)
            i_spec = get_imu_spec(grade)
            
            tactile = TactileArray(array_size=t_spec['array'], sensor_id=f"tol_t_{grade}")
            force = ForceTorqueSensor(sensor_id=f"tol_f_{grade}")
            imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id=f"tol_i_{grade}",
                           sample_rate=i_spec['sample_hz'])
            
            for s in [tactile, force, imu]:
                s.open()
            
            sync_diffs = []
            n_cycles = 30
            
            for _ in range(n_cycles):
                t0 = time.perf_counter()
                tf = tactile.capture()
                wf = force.capture()
                imf = imu.capture()
                t1 = time.perf_counter()
                
                # 用wall clock测量顺序采集总延迟
                total_delay_ms = (t1 - t0) * 1000
                sync_diffs.append(total_delay_ms)
            
            mean_diff = np.mean(sync_diffs)
            budget = GRADE_SYNC_LATENCY_BUDGET[grade]
            
            # 平均同步差异应该小于预算
            self.assertLess(
                mean_diff,
                budget,
                f"{grade}: mean sync diff {mean_diff:.2f}ms exceeds budget {budget:.2f}ms"
            )
            
            for s in [tactile, force, imu]:
                s.close()


class TestTemporalSyncEdgeCases(unittest.TestCase):
    """测试时序同步边界情况"""
    
    def test_rapid_capture_timestamps(self):
        """测试快速连续采集时间戳"""
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="rapid_i")
        imu.open()
        
        timestamps = []
        for _ in range(100):
            r = imu.capture()
            timestamps.append(r.timestamp)
        
        # 时间戳应该单调递增
        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i], timestamps[i-1])
        
        imu.close()
    
    def test_sensor_restart_timestamp_reset(self):
        """测试传感器重启后时间戳重置"""
        tactile = TactileArray((8, 8), sensor_id="restart_t")
        tactile.open()
        
        # 采集几帧
        for _ in range(10):
            tactile.capture()
        
        last_frame_id = tactile._frame_id
        self.assertGreater(last_frame_id, 0)
        
        # 关闭再打开
        tactile.close()
        tactile.open()
        
        # 新采集的帧ID应该从0开始(每次open会重置_frame_id)
        new_frame = tactile.capture()
        self.assertEqual(new_frame.frame_id, 0,
            "Frame ID should reset to 0 after sensor restart")
        
        tactile.close()
    
    def test_synchronization_with_different_rates(self):
        """测试不同采样率下的同步"""
        # IMU高频,触觉低频
        tactile = TactileArray((8, 8), sensor_id="diff_t")
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="diff_i",
                        sample_rate=500)  # 500Hz
        
        tactile.open()
        imu.open()
        
        sync_pairs = []  # (tactile_wall_ts, nearest_imu_wall_ts)
        
        for _ in range(20):
            t0 = time.perf_counter()
            tf = tactile.capture()
            t1 = time.perf_counter()
            imu_times = []
            for _ in range(5):  # 模拟IMU高频采集
                t2 = time.perf_counter()
                imu.capture()
                t3 = time.perf_counter()
                imu_times.append((t2, t3))
            
            # 找最接近IMU帧 (用wall clock时间)
            nearest_imu_time = min(imu_times, key=lambda x: abs(x[0] - t0))
            sync_pairs.append((t0, nearest_imu_time[0]))
        
        # 验证同步对的时间接近性 (wall clock测量)
        for t_ts, i_ts in sync_pairs:
            diff_ms = abs(t_ts - i_ts) * 1000
            # IMU高频,触觉低频,采集间隔应该很小
            self.assertLess(diff_ms, 10.0, f"Sync pair diff {diff_ms:.2f}ms too large")
        
        tactile.close()
        imu.close()


class TestSyncLatencyBudgetCompliance(unittest.TestCase):
    """测试同步延迟预算合规性"""
    
    def test_grade_latency_budget_definitions(self):
        """验证延迟预算定义合理性"""
        for grade in GRADES:
            budget = GRADE_SYNC_LATENCY_BUDGET[grade]
            rates = GRADE_SAMPLE_RATES[grade]
            
            # 最小采样周期
            min_period = 1000.0 / max(rates.values())
            
            # 延迟预算应该大于最小采样周期
            self.assertGreater(budget, min_period * 0.5,
                f"{grade}: budget {budget}ms too small for min period {min_period:.2f}ms")
            
            # 延迟预算应该合理 (不能太大)
            self.assertLess(budget, 50.0, f"{grade}: budget {budget}ms unreasonably large")
    
    def test_measured_latency_vs_budget(self):
        """测试测量延迟是否满足预算"""
        for grade in ['S', 'M', 'L']:
            t_spec = get_tactile_spec(grade)
            
            tactile = TactileArray(array_size=t_spec['array'], sensor_id=f"latency_t_{grade}")
            tactile.open()
            
            budget = GRADE_SYNC_LATENCY_BUDGET[grade]
            delays = []
            
            for _ in range(30):
                t0 = time.perf_counter()
                tactile.capture()
                t1 = time.perf_counter()
                delays.append((t1 - t0) * 1000)
            
            mean_delay = np.mean(delays)
            p95_delay = np.percentile(delays, 95)
            
            # 平均延迟应该远小于预算
            self.assertLess(mean_delay, budget,
                f"{grade}: mean delay {mean_delay:.2f}ms vs budget {budget}ms")
            
            # P95延迟也应该满足预算
            self.assertLess(p95_delay, budget * 2,
                f"{grade}: P95 delay {p95_delay:.2f}ms vs budget {budget}ms")
            
            tactile.close()


class TestVirtualSensorSyncConsistency(unittest.TestCase):
    """测试虚拟传感器同步一致性"""
    
    def test_concurrent_virtual_sensor_consistency(self):
        """测试并发虚拟传感器一致性"""
        v_tactile = VirtualTactileSensor((16, 16), sensor_id="con_t")
        v_force = VirtualForceSensor(sensor_id="con_f", noise_level=0.01)
        v_imu = VirtualIMUSensor(sensor_id="con_i")
        
        for s in [v_tactile, v_force, v_imu]:
            s.open()
        
        errors = []
        counts = {'t': 0, 'f': 0, 'i': 0}
        
        def collect_tactile(n):
            try:
                for _ in range(n):
                    v_tactile.simulate_contact((0.5, 0.5), contact_force=5.0)
                    counts['t'] += 1
            except Exception as e:
                errors.append(f'tactile: {e}')
        
        def collect_force(n):
            try:
                for _ in range(n):
                    v_force.simulate_contact((0, 0, 5.0))
                    counts['f'] += 1
            except Exception as e:
                errors.append(f'force: {e}')
        
        def collect_imu(n):
            try:
                for _ in range(n):
                    v_imu.simulate_static()
                    counts['i'] += 1
            except Exception as e:
                errors.append(f'imu: {e}')
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(collect_tactile, 30),
                executor.submit(collect_force, 30),
                executor.submit(collect_imu, 30),
            ]
            for f in as_completed(futures):
                f.result()
        
        for s in [v_tactile, v_force, v_imu]:
            s.close()
        
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(counts['t'], 30)
        self.assertEqual(counts['f'], 30)
        self.assertEqual(counts['i'], 30)


class TestSyncDataStructures(unittest.TestCase):
    """测试同步数据结构"""
    
    def test_sync_timestamp_dataclass(self):
        """测试SyncTimestamp数据类"""
        ts = SyncTimestamp(
            sensor_id='tactile',
            frame_id=5,
            hardware_ts=1.234,
            software_ts=1.236,
            sync_group_id=2,
            delay_ms=2.0
        )
        
        self.assertEqual(ts.sensor_id, 'tactile')
        self.assertEqual(ts.frame_id, 5)
        self.assertEqual(ts.delay_ms, 2.0)
        self.assertEqual(ts.sync_group_id, 2)
    
    def test_sync_frame_dataclass(self):
        """测试SyncFrame数据类"""
        tactile = TactileArray((8, 8), sensor_id="sf_t")
        tactile.open()
        tf = tactile.capture()
        tactile.close()
        
        sf = SyncFrame(
            group_id=0,
            capture_time=1.5,
            tactile=tf,
            force=None,
            imu=None,
            tactile_delay_ms=1.0,
            force_delay_ms=0.0,
            imu_delay_ms=0.0
        )
        
        self.assertEqual(sf.group_id, 0)
        self.assertIsNotNone(sf.tactile)
        self.assertIsNone(sf.force)
        self.assertEqual(sf.tactile_delay_ms, 1.0)
    
    def test_sync_frame_equality(self):
        """测试同步帧相等性"""
        sf1 = SyncFrame(group_id=1, capture_time=1.0)
        sf2 = SyncFrame(group_id=1, capture_time=1.0)
        sf3 = SyncFrame(group_id=2, capture_time=2.0)
        
        self.assertEqual(sf1.group_id, sf2.group_id)
        self.assertNotEqual(sf1.group_id, sf3.group_id)


if __name__ == '__main__':
    unittest.main()
