#!/usr/bin/env python3
"""
SuperModel 具身智能实战演示
===========================
真实机器人传感器-控制集成示例

演示内容:
- 多传感器初始化与数据采集
- 传感器-控制闭环集成
- 五级AGV配置加载
- 实时监控与安全

Usage:
    python3 examples/real_robot_integration.py [--grade M] [--duration 10]

AGV等级选项: S / M / L / XL / XXL
"""

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, '.')


def parse_args():
    parser = argparse.ArgumentParser(description='SuperModel 具身智能实战演示')
    parser.add_argument('--grade', type=str, default='M',
                        choices=['S', 'M', 'L', 'XL', 'XXL'],
                        help='AGV等级 (默认: M)')
    parser.add_argument('--duration', type=int, default=10,
                        help='演示持续时间秒数 (默认: 10)')
    parser.add_argument('--sensors', type=str, default='all',
                        choices=['all', 'tactile', 'force', 'imu', 'vision'],
                        help='启用的传感器 (默认: all)')
    parser.add_argument('--rate', type=int, default=None,
                        help='控制频率 Hz (默认: 等级默认值)')
    return parser.parse_args()


def create_sensor_manager(grade: str, enabled_sensors: list):
    """创建传感器管理器"""
    from sensors.manager import SensorManager
    from sensors.vision import CameraSensor
    from sensors.audio import AudioSensor
    from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
    from sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
    from sensors.imu import IMUSensor, IMUSensorType, get_imu_spec

    manager = SensorManager()

    # 视觉 (S级及以上)
    if 'vision' in enabled_sensors and grade in ['S', 'M', 'L', 'XL', 'XXL']:
        try:
            camera = CameraSensor(
                sensor_type='realsense' if grade != 'S' else 'astra',
                sensor_id='front_camera',
                resolution=(1280, 720) if grade != 'S' else (640, 480),
                frame_rate=30 if grade != 'XXL' else 60
            )
            manager.add_sensor(camera, modalities=['vision'])
            print(f"  [Vision] {camera.sensor_id} 已添加 (grade={grade})")
        except Exception as e:
            print(f"  [Vision] 跳过: {e}")

    # 触觉 (M级及以上)
    if 'tactile' in enabled_sensors and grade in ['M', 'L', 'XL', 'XXL']:
        t_spec = get_tactile_spec(grade)
        tactile = TactileArray(
            array_size=t_spec['array'],
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id=f'tactile_{grade}'
        )
        manager.add_sensor(tactile, modalities=['tactile'])
        print(f"  [Tactile] {t_spec['array']} @ {t_spec['freq_hz']}Hz 已添加 (grade={grade})")

    # 力觉 (M级及以上)
    if 'force' in enabled_sensors and grade in ['M', 'L', 'XL', 'XXL']:
        f_spec = get_force_spec(grade)
        force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS if f_spec['axes'] == 6 else ForceSensorType.THREE_AXIS,
            sensor_id=f'force_{grade}'
        )
        manager.add_sensor(force, modalities=['force'])
        print(f"  [Force] {f_spec['axes']}轴 @ {f_spec['sampling_hz']}Hz 已添加 (grade={grade})")

    # IMU (所有等级)
    if 'imu' in enabled_sensors:
        i_spec = get_imu_spec(grade)
        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088 if 'BMI' in i_spec['type'] else IMUSensorType.MPU6050,
            sensor_id=f'imu_{grade}',
            sample_rate=i_spec['sample_hz']
        )
        manager.add_sensor(imu, modalities=['imu'])
        print(f"  [IMU] {i_spec['type']} @ {i_spec['sample_hz']}Hz 已添加 (grade={grade})")

    return manager


def create_controller_stack(grade: str, sensor_manager):
    """创建控制栈"""
    from control.tactile_control import TactileServoController, TactileServoParams
    from control.force_control import ForceController, ForceControlParams
    from control.imu_control import AttitudeStabilizer, IMUControlParams
    from control.agv import AGVMotionController, AGVControlParams
    from control.safety_controller import SafetyController, SafetyParams

    controllers = {}

    # 触觉控制器 (M级以上)
    if grade in ['M', 'L', 'XL', 'XXL']:
        tactile = sensor_manager.get_sensor_by_modality('tactile')
        if tactile:
            params = TactileServoParams.from_grade(grade)
            controllers['tactile'] = TactileServoController(tactile, params)
            print(f"  [TactileController] 已创建 (grade={grade})")

    # 力控制器 (M级以上)
    if grade in ['M', 'L', 'XL', 'XXL']:
        force = sensor_manager.get_sensor_by_modality('force')
        if force:
            params = ForceControlParams.from_grade(grade)
            controllers['force'] = ForceController(force, params)
            print(f"  [ForceController] 已创建 (grade={grade})")

    # IMU控制器 (所有等级)
    imu = sensor_manager.get_sensor_by_modality('imu')
    if imu:
        params = IMUControlParams.from_grade(grade)
        controllers['imu'] = AttitudeStabilizer(imu, params)
        print(f"  [AttitudeStabilizer] 已创建 (grade={grade})")

    # AGV运动控制器 (所有等级)
    agv_params = AGVControlParams.from_grade(grade)
    controllers['agv'] = AGVMotionController(agv_params)
    print(f"  [AGVMotionController] 已创建 (grade={grade})")

    # 安全控制器 (所有等级)
    safety_params = SafetyParams.from_grade(grade)
    controllers['safety'] = SafetyController(safety_params)
    print(f"  [SafetyController] 已创建 (grade={grade})")

    return controllers


class RealRobotDemo:
    """真实机器人集成演示"""

    def __init__(self, grade: str = 'M', duration: int = 10, rate: int = None):
        self.grade = grade
        self.duration = duration
        self.rate = rate or self._default_rate(grade)
        self.dt = 1.0 / self.rate

        # 传感器和控制器的控制频率
        self.sensor_rates = {
            'S': 50, 'M': 100, 'L': 200, 'XL': 500, 'XXL': 1000
        }
        self.sensor_rate = self.sensor_rates.get(grade, 100)
        self.sensor_dt = 1.0 / self.sensor_rate

        print("\n" + "=" * 60)
        print(f"SuperModel 具身智能实战演示")
        print(f"  AGV等级: {grade}")
        print(f"  控制频率: {self.rate} Hz")
        print(f"  传感器频率: {self.sensor_rate} Hz")
        print(f"  演示时长: {duration} 秒")
        print("=" * 60)

        self.sensor_manager = None
        self.controllers = {}
        self.running = False
        self.stats = {
            'loop_count': 0,
            'sensor_updates': 0,
            'control_updates': 0,
            'safety_checks': 0,
            'errors': 0
        }

    def _default_rate(self, grade: str) -> int:
        rates = {'S': 50, 'M': 100, 'L': 200, 'XL': 500, 'XXL': 1000}
        return rates.get(grade, 100)

    def setup(self, enabled_sensors='all'):
        """初始化传感器和控制器"""
        print("\n[1/4] 初始化传感器...")
        self.sensor_manager = create_sensor_manager(self.grade, enabled_sensors.split(','))

        # 打开所有传感器
        for sensor in self.sensor_manager.sensors.values():
            try:
                sensor.open()
                print(f"  ✓ {sensor.sensor_id} 已打开")
            except Exception as e:
                print(f"  ✗ {sensor.sensor_id} 打开失败: {e}")

        print("\n[2/4] 初始化控制器...")
        self.controllers = create_controller_stack(self.grade, self.sensor_manager)

        print("\n[3/4] 传感器预热 (200 frames)...")
        warmup_end = min(200, int(self.sensor_rate * 2))
        for i in range(warmup_end):
            self.sensor_manager.capture_all()
        print(f"  ✓ 完成 {warmup_end} 帧预热")

        print("\n[4/4] 初始化完成")
        self.running = True

    def run(self):
        """运行主循环"""
        print(f"\n开始实时控制循环 ({self.rate} Hz)...")

        loop_start = time.time()
        sensor_accum = 0.0
        control_accum = 0.0

        iteration = 0
        max_iterations = self.duration * self.rate

        while self.running and iteration < max_iterations:
            t_loop_start = time.time()

            # --- 传感器更新 (高频) ---
            sensor_accum += self.dt
            sensor_steps = int(self.sensor_rate * self.dt)
            for _ in range(sensor_steps):
                self.sensor_manager.capture_all()
                self.stats['sensor_updates'] += 1

            # --- 控制更新 (低频) ---
            control_accum += self.dt
            control_steps = int(self.rate * self.dt)
            for _ in range(max(1, control_steps)):
                self._control_step()
                self.stats['control_updates'] += 1

            # --- 安全检查 ---
            self._safety_step()

            # --- 统计输出 (每5秒) ---
            if iteration % (self.rate * 5) == 0 and iteration > 0:
                self._print_stats()

            # --- 限速 ---
            elapsed = time.time() - t_loop_start
            sleep_time = max(0, self.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            iteration += 1
            self.stats['loop_count'] += 1

        total_time = time.time() - loop_start
        print(f"\n演示完成! 总运行时间: {total_time:.2f}秒")
        self._print_stats()

    def _control_step(self):
        """执行控制步骤"""
        # 获取当前传感器数据
        imu_data = self.sensor_manager.get_by_modality('imu')
        tactile_data = self.sensor_manager.get_by_modality('tactile')
        force_data = self.sensor_manager.get_by_modality('force')

        # IMU姿态控制
        if 'imu' in self.controllers and imu_data:
            try:
                self.controllers['imu'].update(imu_data, self.dt)
            except Exception as e:
                self.stats['errors'] += 1

        # 触觉伺服控制
        if 'tactile' in self.controllers and tactile_data:
            try:
                contacts = tactile_data.detect_contacts()
                grip_quality = tactile_data.estimate_grip_quality()
            except Exception as e:
                self.stats['errors'] += 1

        # 力控
        if 'force' in self.controllers and force_data:
            try:
                wrench = force_data.capture()
                contact_state = force_data.detect_contact(wrench)
            except Exception as e:
                self.stats['errors'] += 1

        # AGV运动控制
        if 'agv' in self.controllers:
            try:
                self.controllers['agv'].step(self.dt)
            except Exception as e:
                self.stats['errors'] += 1

    def _safety_step(self):
        """执行安全检查"""
        if 'safety' not in self.controllers:
            return

        safety = self.controllers['safety']

        # 检查IMU倾斜
        if 'imu' in self.controllers:
            imu_data = self.sensor_manager.get_by_modality('imu')
            if imu_data:
                try:
                    tilt_status = self.controllers['imu'].get_tilt_status()
                    if tilt_status.get('tilt_critical', False):
                        safety.emergency_stop('IMU tilt critical')
                except Exception:
                    pass

        self.stats['safety_checks'] += 1

    def _print_stats(self):
        """打印统计信息"""
        elapsed = self.stats['loop_count'] * self.dt
        actual_rate = self.stats['loop_count'] / elapsed if elapsed > 0 else 0
        print(f"\n  [{self.grade}] 统计 (t={elapsed:.1f}s):")
        print(f"    控制循环: {self.stats['loop_count']} 次, 实际频率: {actual_rate:.1f} Hz")
        print(f"    传感器更新: {self.stats['sensor_updates']} 次")
        print(f"    控制更新: {self.stats['control_updates']} 次")
        print(f"    安全检查: {self.stats['safety_checks']} 次")
        print(f"    错误: {self.stats['errors']} 次")

    def shutdown(self):
        """关闭所有传感器和控制器"""
        print("\n关闭中...")
        self.running = False

        if self.sensor_manager:
            for sensor in self.sensor_manager.sensors.values():
                try:
                    sensor.close()
                except Exception:
                    pass

        print("演示结束。")


def quick_test(grade: str = 'M'):
    """快速功能测试 (不涉及真实硬件)"""
    print(f"\n{'='*60}")
    print(f"SuperModel 快速功能测试 (grade={grade})")
    print(f"{'='*60}")

    from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
    from sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
    from sensors.imu import IMUSensor, IMUSensorType, get_imu_spec, VirtualIMUSensor
    from control.imu_control import AttitudeStabilizer, IMUControlParams
    from control.tactile_control import TactileServoController, TactileServoParams
    from control.force_control import ForceController, ForceControlParams

    print("\n[触觉传感器测试]")
    t_spec = get_tactile_spec(grade)
    tactile = TactileArray(
        array_size=t_spec['array'],
        sensor_type=TactileSensorType.CAPACITIVE,
        sensor_id='test_tactile'
    )
    tactile.open()
    for i in range(5):
        frame = tactile.capture()
        contacts = tactile.detect_contacts(frame)
        quality = tactile.estimate_grip_quality(frame)
    tactile.close()
    print(f"  ✓ TactileArray {t_spec['array']} @ {t_spec['freq_hz']}Hz: {len(contacts)} contacts, quality={quality['overall']:.2f}")

    print("\n[力觉传感器测试]")
    f_spec = get_force_spec(grade)
    force = ForceTorqueSensor(
        sensor_type=ForceSensorType.SIX_AXIS,
        sensor_id='test_force'
    )
    force.open()
    force.calibrate_bias(10)
    for i in range(5):
        wrench = force.capture()
        state = force.detect_contact(wrench)
        payload = force.estimate_payload(wrench)
    force.close()
    print(f"  ✓ ForceTorqueSensor {f_spec['axes']}轴 @ {f_spec['sampling_hz']}Hz: F={wrench.magnitude:.2f}N, payload={payload:.2f}kg")

    print("\n[IMU传感器测试]")
    i_spec = get_imu_spec(grade)
    imu = IMUSensor(
        sensor_type=IMUSensorType.BMI088,
        sensor_id='test_imu',
        sample_rate=i_spec['sample_hz']
    )
    imu.open()
    imu.self_test()
    imu.calibrate_gyro_bias(50)
    for i in range(5):
        frame = imu.capture()
    imu.close()
    print(f"  ✓ IMUSensor {i_spec['type']} @ {i_spec['sample_hz']}Hz: accel={frame.accel_magnitude:.2f} m/s²")

    print("\n[控制器创建测试]")
    imu2 = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id='ctrl_test_imu')
    imu2.open()
    imu_params = IMUControlParams.from_grade(grade)
    stabilizer = AttitudeStabilizer(imu2, imu_params)
    for i in range(10):
        frame = imu2.capture()
        torque = stabilizer.update(frame, 0.01)
    imu2.close()
    print(f"  ✓ AttitudeStabilizer (grade={grade}): Kp={imu_params.Kp_attitude}")

    print(f"\n[Grade={grade}] 全部测试通过!")
    print(f"  触觉: {t_spec['array']}, {t_spec['freq_hz']}Hz")
    print(f"  力觉: {f_spec['axes']}轴, {f_spec['sampling_hz']}Hz")
    print(f"  IMU: {i_spec['type']}, {i_spec['sample_hz']}Hz")
    print(f"  控制频率: {imu_params.control_rate}Hz")


def main():
    args = parse_args()

    print(f"""
    ╔═══════════════════════════════════════════╗
    ║   SuperModel 具身智能实战演示               ║
    ║   AGV等级: {args.grade}                             ║
    ╚═══════════════════════════════════════════╝
    """)

    # 先运行快速测试
    quick_test(args.grade)

    # 然后尝试真实传感器演示 (需要硬件)
    print(f"\n{'='*60}")
    print("注意: 下面的真实传感器演示需要连接实际硬件")
    print("      如无硬件，可使用仿问模式或跳过")
    print(f"{'='*60}")

    demo = None
    try:
        demo = RealRobotDemo(
            grade=args.grade,
            duration=args.duration,
            rate=args.rate
        )
        demo.setup(enabled_sensors=args.sensors)
        demo.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if demo:
            demo.shutdown()


if __name__ == '__main__':
    main()
