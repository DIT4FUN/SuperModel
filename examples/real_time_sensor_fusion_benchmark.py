"""
SuperModel 实时多传感器融合性能基准测试
=========================================

测试目标:
1. 触觉/力觉/IMU 传感器数据采集延迟
2. 互补滤波 + EKF 融合实时性能
3. 端到端传感器→融合→控制管道延迟

AGV等级: L (工业级)
运行: python examples/real_time_sensor_fusion_benchmark.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileSensorType, VirtualTactileSensor,
    TactileContact, get_tactile_spec, AGV_TACTILE_GRADES
)
from sensors.force import (
    ForceTorqueSensor, ForceSensorType, VirtualForceSensor,
    Wrench, ForceCalibration, ContactState, get_force_spec, AGV_FORCE_GRADES
)
from sensors.imu import (
    IMUSensor, IMUSensorType, VirtualIMUSensor,
    IMUFrame, Pose, PoseEstimator, get_imu_spec, AGV_IMU_GRADES
)
from fusion.sensor_fusion import (
    ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion
)


class SensorBenchmark:
    """传感器性能基准测试"""

    def __init__(self, num_iterations: int = 1000, warmup: int = 50):
        self.num_iterations = num_iterations
        self.warmup = warmup

    def benchmark_tactile(self, array_size=(16, 16), grade='M'):
        """触觉传感器延迟测试"""
        print(f"\n{'='*60}")
        print(f"[触觉] 基准测试: {array_size}, AGV Grade={grade}")
        print(f"{'='*60}")

        spec = get_tactile_spec(grade)
        sensor = TactileArray(
            array_size=(spec['array'][0], spec['array'][1]),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id=f"bench_tactile_{grade}"
        )

        with sensor:
            # 预热
            for _ in range(self.warmup):
                sensor.capture()

            # 基准测试
            latencies = []
            for i in range(self.num_iterations):
                t0 = time.perf_counter()
                frame = sensor.capture()
                contacts = sensor.detect_contacts(frame)
                slip = sensor.get_slip_signal(frame)
                quality = sensor.estimate_grip_quality(frame)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # ms

        latencies = np.array(latencies)
        print(f"  采集延迟:     {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
        print(f"  最小延迟:     {np.min(latencies):.4f} ms")
        print(f"  最大延迟:     {np.max(latencies):.4f} ms")
        print(f"  95th百分位:   {np.percentile(latencies, 95):.4f} ms")
        print(f"  99th百分位:   {np.percentile(latencies, 99):.4f} ms")
        print(f"  接触检测数:   {len(contacts)} 个")
        print(f"  抓取质量:     {quality['overall']:.3f}")
        return latencies

    def benchmark_force(self, grade='M'):
        """力觉传感器延迟测试"""
        print(f"\n{'='*60}")
        print(f"[力觉] 基准测试: AGV Grade={grade}")
        print(f"{'='*60}")

        spec = get_force_spec(grade)
        sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id=f"bench_force_{grade}"
        )

        with sensor:
            # 预热
            for _ in range(self.warmup):
                sensor.capture()

            # 基准测试
            latencies = []
            contact_forces = []
            for i in range(self.num_iterations):
                t0 = time.perf_counter()
                wrench = sensor.capture()
                contact = sensor.detect_contact(wrench)
                payload = sensor.estimate_payload(wrench)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                contact_forces.append(wrench.magnitude)

        latencies = np.array(latencies)
        contact_forces = np.array(contact_forces)
        print(f"  采集延迟:     {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
        print(f"  最小延迟:     {np.min(latencies):.4f} ms")
        print(f"  最大延迟:     {np.max(latencies):.4f} ms")
        print(f"  95th百分位:   {np.percentile(latencies, 95):.4f} ms")
        print(f"  力范围:       {np.min(contact_forces):.2f} ~ {np.max(contact_forces):.2f} N")
        print(f"  平均力:       {np.mean(contact_forces):.2f} N")
        print(f"  采样频率:     {spec['sampling_hz']} Hz")
        return latencies

    def benchmark_imu(self, grade='M'):
        """IMU传感器延迟测试"""
        print(f"\n{'='*60}")
        print(f"[IMU] 基准测试: AGV Grade={grade}")
        print(f"{'='*60}")

        spec = get_imu_spec(grade)
        sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088 if grade in ['M', 'L'] else IMUSensorType.MPU6050,
            sensor_id=f"bench_imu_{grade}",
            accel_range=spec['accel_range'],
            gyro_range=spec['gyro_range'],
            sample_rate=spec['sample_hz']
        )
        pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=spec['sample_hz'])

        with sensor:
            # 预热
            for _ in range(self.warmup):
                frame = sensor.capture()

            # 基准测试
            latencies = []
            roll_vals, pitch_vals, yaw_vals = [], [], []
            for i in range(self.num_iterations):
                t0 = time.perf_counter()
                frame = sensor.capture()
                pose = pose_estimator.update(frame.accel, frame.gyro, frame.mag)
                euler = pose.to_euler()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                roll_vals.append(euler[0])
                pitch_vals.append(euler[1])
                yaw_vals.append(euler[2])

        latencies = np.array(latencies)
        roll_vals, pitch_vals, yaw_vals = np.array(roll_vals), np.array(pitch_vals), np.array(yaw_vals)
        print(f"  采集+姿态估计延迟: {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
        print(f"  最小延迟:          {np.min(latencies):.4f} ms")
        print(f"  最大延迟:          {np.max(latencies):.4f} ms")
        print(f"  95th百分位:        {np.percentile(latencies, 95):.4f} ms")
        print(f"  Roll范围:          {np.min(roll_vals):.3f} ~ {np.max(roll_vals):.3f} rad")
        print(f"  Pitch范围:         {np.min(pitch_vals):.3f} ~ {np.max(pitch_vals):.3f} rad")
        print(f"  Yaw范围:           {np.min(yaw_vals):.3f} ~ {np.max(yaw_vals):.3f} rad")
        print(f"  IMU型号:           {spec['type']}")
        print(f"  采样频率:          {spec['sample_hz']} Hz")
        return latencies

    def benchmark_sensor_fusion(self, grade='M'):
        """传感器融合性能测试"""
        print(f"\n{'='*60}")
        print(f"[融合] 基准测试: AGV Grade={grade}")
        print(f"{'='*60}")

        spec_imu = get_imu_spec(grade)
        sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id=f"bench_fusion_{grade}",
            sample_rate=spec_imu['sample_hz']
        )

        # 互补滤波
        comp_filter = ComplementaryFilter(alpha=0.96)

        # 扩展卡尔曼滤波
        ekf = ExtendedKalmanFilter(
            state_dim=6,  # [x, y, theta, vx, vy, omega]
            measurement_dim=3,
            process_noise=0.01,
            measurement_noise=0.1
        )
        ekf.initialize(np.zeros(6))
        F = np.eye(6)
        F[0, 3] = 0.01; F[1, 4] = 0.01; F[2, 5] = 0.01
        H = np.zeros((3, 6)); H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        ekf.set_matrices(F, H)

        comp_latencies, ekf_latencies = [], []

        with sensor:
            for _ in range(self.warmup):
                frame = sensor.capture()

            for i in range(self.num_iterations):
                frame = sensor.capture()

                # 互补滤波
                t0 = time.perf_counter()
                comp_state = comp_filter.update({
                    'accel': frame.accel,
                    'gyro': frame.gyro
                }, dt=1.0/spec_imu['sample_hz'])
                t1 = time.perf_counter()
                comp_latencies.append((t1 - t0) * 1000)

                # EKF
                t0 = time.perf_counter()
                ekf.predict(dt=0.01)
                ekf.correct(comp_state)
                ekf_state = ekf.get_state()
                t1 = time.perf_counter()
                ekf_latencies.append((t1 - t0) * 1000)

        comp_latencies = np.array(comp_latencies)
        ekf_latencies = np.array(ekf_latencies)

        print(f"  互补滤波延迟:  {np.mean(comp_latencies):.4f} ± {np.std(comp_latencies):.4f} ms")
        print(f"  EKF延迟:       {np.mean(ekf_latencies):.4f} ± {np.std(ekf_latencies):.4f} ms")
        print(f"  EKF 95th:      {np.percentile(ekf_latencies, 95):.4f} ms")
        print(f"  EKF 99th:      {np.percentile(ekf_latencies, 99):.4f} ms")
        return comp_latencies, ekf_latencies

    def benchmark_end_to_end(self, grade='M'):
        """端到端管道延迟测试"""
        print(f"\n{'='*60}")
        print(f"[E2E管道] 基准测试: AGV Grade={grade}")
        print(f"{'='*60}")

        spec_t = get_tactile_spec(grade)
        spec_f = get_force_spec(grade)
        spec_i = get_imu_spec(grade)

        tactile = TactileArray(
            array_size=(spec_t['array'][0], spec_t['array'][1]),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id="e2e_tactile"
        )
        force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="e2e_force"
        )
        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="e2e_imu",
            sample_rate=spec_i['sample_hz']
        )
        pose_est = PoseEstimator(algorithm='madgwick', sample_rate=spec_i['sample_hz'])
        msf = MultiSensorFusion()
        msf.add_fusion_method("imu_comp", ComplementaryFilter(alpha=0.96), weight=1.0)

        all_latencies = []

        with tactile, force, imu:
            for _ in range(self.warmup):
                tactile.capture(); force.capture(); imu.capture()

            for i in range(self.num_iterations):
                t0 = time.perf_counter()

                # 1. 传感器采集
                tf = tactile.capture()
                contacts = tactile.detect_contacts(tf)
                fw = force.capture()
                contact = force.detect_contact(fw)
                iframe = imu.capture()
                pose = pose_est.update(iframe.accel, iframe.gyro, iframe.mag)

                # 2. 融合
                fused_state = msf.get_fused_state()

                # 3. 传感器→融合管道
                sensor_data = {
                    "imu_comp": {
                        'accel': iframe.accel,
                        'gyro': iframe.gyro
                    }
                }
                results = msf.update(sensor_data, dt=0.01)

                t1 = time.perf_counter()
                all_latencies.append((t1 - t0) * 1000)

        all_latencies = np.array(all_latencies)
        print(f"  端到端延迟:   {np.mean(all_latencies):.4f} ± {np.std(all_latencies):.4f} ms")
        print(f"  最小延迟:     {np.min(all_latencies):.4f} ms")
        print(f"  最大延迟:     {np.max(all_latencies):.4f} ms")
        print(f"  95th百分位:   {np.percentile(all_latencies, 95):.4f} ms")
        print(f"  99th百分位:   {np.percentile(all_latencies, 99):.4f} ms")
        return all_latencies

    def run_all(self, grades=None):
        """运行所有基准测试"""
        if grades is None:
            grades = ['S', 'M', 'L']

        print("=" * 70)
        print("SuperModel 实时多传感器融合性能基准测试")
        print(f"迭代次数: {self.num_iterations}, 预热: {self.warmup}")
        print("=" * 70)

        summary = {}

        for grade in grades:
            print(f"\n{'#'*70}")
            print(f"### AGV等级: {grade}")
            print(f"{'#'*70}")

            summary[grade] = {}

            # 触觉
            t_lat = self.benchmark_tactile(grade=grade)
            summary[grade]['tactile_mean'] = np.mean(t_lat)
            summary[grade]['tactile_p95'] = np.percentile(t_lat, 95)

            # 力觉
            f_lat = self.benchmark_force(grade=grade)
            summary[grade]['force_mean'] = np.mean(f_lat)
            summary[grade]['force_p95'] = np.percentile(f_lat, 95)

            # IMU
            i_lat = self.benchmark_imu(grade=grade)
            summary[grade]['imu_mean'] = np.mean(i_lat)
            summary[grade]['imu_p95'] = np.percentile(i_lat, 95)

            # 融合
            c_lat, e_lat = self.benchmark_sensor_fusion(grade=grade)
            summary[grade]['comp_fusion_mean'] = np.mean(c_lat)
            summary[grade]['ekf_fusion_mean'] = np.mean(e_lat)

            # E2E
            e2e_lat = self.benchmark_end_to_end(grade=grade)
            summary[grade]['e2e_mean'] = np.mean(e2e_lat)
            summary[grade]['e2e_p95'] = np.percentile(e2e_lat, 95)

        # 汇总
        print(f"\n{'='*70}")
        print("基准测试汇总")
        print(f"{'='*70}")
        print(f"{'等级':<6} {'触觉ms':<10} {'力觉ms':<10} {'IMUms':<10} {'融合ms':<10} {'E2E ms':<10}")
        print("-" * 60)
        for grade, s in summary.items():
            print(f"{grade:<6} "
                  f"{s['tactile_mean']:<10.4f} "
                  f"{s['force_mean']:<10.4f} "
                  f"{s['imu_mean']:<10.4f} "
                  f"{s['ekf_fusion_mean']:<10.4f} "
                  f"{s['e2e_mean']:<10.4f}")

        print(f"\n所有测试完成。迭代次数={self.num_iterations}")
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SuperModel 传感器融合基准测试')
    parser.add_argument('--iterations', '-n', type=int, default=500,
                        help='每项测试迭代次数 (默认500)')
    parser.add_argument('--warmup', '-w', type=int, default=30,
                        help='预热迭代次数 (默认30)')
    parser.add_argument('--grades', '-g', type=str, default='S,M,L',
                        help='AGV等级列表，逗号分隔 (默认S,M,L)')
    args = parser.parse_args()

    grades = [g.strip() for g in args.grades.split(',')]
    bench = SensorBenchmark(num_iterations=args.iterations, warmup=args.warmup)
    bench.run_all(grades=grades)


if __name__ == '__main__':
    main()
