#!/usr/bin/env python3
"""
SuperModel 自主巡逻演示
======================

展示 SuperModel AGV 的自主巡逻能力:
- 多点巡逻路线规划
- DWA/APF 动态避障
- 传感器融合感知
- 异常检测与自主恢复
- 五级AGV能力对比

使用方法:
    python autonomous_patrol_demo.py              # 默认M级演示
    python autonomous_patrol_demo.py --grade S    # S级演示
    python autonomous_patrol_demo.py --all-grades # 五级对比
    python autonomous_patrol_demo.py --visualize  # 可视化模式
"""

import argparse
import sys
import time
import os

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.patrol_control import (
    PatrolController, PatrolRoute, PatrolPoint, PatrolState,
    Obstacle, create_patrol_controller, run_patrol_benchmark,
    get_patrol_spec, PatrolGrade, PatrolMetrics,
)
import numpy as np


def print_banner(text: str, width: int = 60):
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_patrol_state(controller: PatrolController, step: int):
    """打印巡逻状态"""
    pose = controller.get_pose()
    metrics = controller.get_metrics()
    state = controller.get_state()

    print(f"\n--- Step {step} ---")
    print(f"  State:    {state.value}")
    print(f"  Pose:     x={pose[0]:.3f}m  y={pose[1]:.3f}m  θ={np.degrees(pose[2]):.1f}°")
    print(f"  Velocity: vx={controller.velocity[0]:.3f}m/s  vy={controller.velocity[1]:.3f}m/s  ω={controller.velocity[2]:.3f}rad/s")
    print(f"  Metrics:  distance={metrics['total_distance']:.3f}m  time={metrics['total_time']:.2f}s  avg_speed={metrics['avg_speed']:.3f}m/s")
    print(f"  Alerts:   {metrics['alerts_triggered']} obstacles_avoided={metrics['obstacles_avoided']}  points={metrics['points_completed']}/{metrics['points_total']}")


def run_single_patrol_demo(grade: str, steps: int = 200, verbose: bool = True):
    """运行单级巡逻演示"""
    spec = get_patrol_spec(grade)
    print_banner(f"AGV巡逻演示 - {grade}级 ({spec['avoidance_strategy']})")
    print(f"  最大速度: {spec['max_patrol_speed']} m/s")
    print(f"  控制频率: {spec['control_frequency']} Hz")
    print(f"  避障策略: {spec['avoidance_strategy']}")
    print(f"  传感器: {', '.join(spec['sensor_modalities'])}")

    # 创建巡逻路线 (仓库环境)
    route = PatrolRoute(
        name="warehouse_patrol",
        points=[
            PatrolPoint(x=0.0, y=0.0, name="充电桩", priority=5),
            PatrolPoint(x=3.0, y=0.0, name="货架A", priority=3),
            PatrolPoint(x=3.0, y=3.0, name="货架B", priority=3),
            PatrolPoint(x=0.0, y=3.0, name="质检区", priority=4),
            PatrolPoint(x=1.5, y=1.5, name="中央通道", priority=2),
        ],
        loop=True,
    )

    # 创建动态障碍物
    obstacles = [
        Obstacle(position=np.array([1.5, 0.0]), radius=0.3, type="static"),
        Obstacle(position=np.array([2.5, 2.0]), radius=0.4, type="person"),
        Obstacle(position=np.array([0.5, 2.5]), radius=0.25, type="static"),
    ]

    # 创建控制器
    controller = create_patrol_controller(
        grade=grade,
        pose=(0.0, 0.0, 0.0),
        route=route,
    )

    # 添加额外障碍物
    for obs in obstacles:
        controller.obstacles.append(obs)

    # 启动巡逻
    controller.start_patrol()

    print(f"\n启动巡逻: {route.name}")
    print(f"巡逻点: {[p.name for p in route.points]}")
    print(f"障碍物: {len(controller.obstacles)} 个")
    print(f"\n开始模拟 ({steps} 步)...")

    # 模拟巡逻
    start_time = time.time()
    for step in range(steps):
        dt = 1.0 / spec['control_frequency']
        controller.update(dt=dt)

        if verbose and step % 20 == 0:
            print_patrol_state(controller, step)

        # 检查是否到达终点
        if controller.state == PatrolState.ARRIVED:
            print("\n✓ 巡逻路线完成!")
            break

    elapsed = time.time() - start_time
    print_banner("巡逻完成")
    metrics = controller.get_metrics()
    print(f"  总距离:   {metrics['total_distance']:.3f} m")
    print(f"  总时间:   {metrics['total_time']:.2f} s")
    print(f"  平均速度: {metrics['avg_speed']:.3f} m/s")
    print(f"  避障次数: {metrics['obstacles_avoided']}")
    print(f"  告警次数: {metrics['alerts_triggered']}")
    print(f"  完成点数: {metrics['points_completed']}/{metrics['points_total']}")
    print(f"  急停次数: {metrics['emergency_stops']}")
    print(f"  模拟耗时: {elapsed:.3f} s")
    print(f"  目标达成: {'✓' if metrics['points_completed'] > 0 else '✗'}")

    controller.stop_patrol()
    return controller


def run_multi_obstacle_demo(grade: str = 'L', duration: float = 5.0):
    """多障碍物密集场景演示"""
    print_banner(f"密集障碍物场景 - {grade}级")
    spec = get_patrol_spec(grade)

    route = PatrolRoute(
        name="dense_obstacles",
        points=[
            PatrolPoint(x=0.0, y=0.0, name="起点"),
            PatrolPoint(x=5.0, y=0.0, name="终点"),
        ],
        loop=False,
    )

    # 创建密集障碍物
    obstacles = []
    for i in range(8):
        x = 0.5 + i * 0.6
        y = np.sin(i * 0.8) * 0.5
        obstacles.append(Obstacle(
            position=np.array([x, y]),
            radius=0.25,
            type="static" if i % 3 == 0 else "person"
        ))

    controller = create_patrol_controller(grade=grade, pose=(0.0, 0.0, 0.0), route=route)
    for obs in obstacles:
        controller.obstacles.append(obs)

    controller.start_patrol()

    steps = int(duration * spec['control_frequency'])
    print(f"障碍物数量: {len(obstacles)}")
    print(f"模拟时长: {duration}s ({steps}步)")

    for step in range(steps):
        dt = 1.0 / spec['control_frequency']
        state, _ = controller.update(dt=dt)
        if step % 25 == 0:
            pose = controller.get_pose()
            print(f"  Step {step:3d}: pos=({pose[0]:.2f}, {pose[1]:.2f}) state={state.value}")

    metrics = controller.get_metrics()
    print(f"\n结果: 距离={metrics['total_distance']:.2f}m  避障={metrics['obstacles_avoided']}次  到达={'✓' if controller.state == PatrolState.ARRIVED else '✗'}")

    controller.stop_patrol()
    return metrics


def run_five_grade_benchmark():
    """五级AGV巡逻能力基准测试"""
    print_banner("AGV五级巡逻能力基准测试")

    grades = ['S', 'M', 'L', 'XL', 'XXL']
    results = {}

    for grade in grades:
        spec = get_patrol_spec(grade)
        print(f"\n测试 {grade}级 ({spec['avoidance_strategy']})...")

        route = PatrolRoute(
            name=f"bench_{grade}",
            points=[
                PatrolPoint(x=0.0, y=0.0, name="start"),
                PatrolPoint(x=2.0, y=0.0, name="p1"),
                PatrolPoint(x=2.0, y=2.0, name="p2"),
                PatrolPoint(x=0.0, y=2.0, name="p3"),
            ],
            loop=True,
        )

        obstacles = [
            Obstacle(position=np.array([1.0, 0.5]), radius=0.3),
            Obstacle(position=np.array([1.5, 1.5]), radius=0.35),
        ]

        controller = create_patrol_controller(grade=grade, pose=(0.0, 0.0, 0.0), route=route)
        for obs in obstacles:
            controller.obstacles.append(obs)

        controller.start_patrol()
        steps = int(8 * spec['control_frequency'])

        for _ in range(steps):
            controller.update(dt=1.0 / spec['control_frequency'])

        metrics = controller.get_metrics()
        results[grade] = metrics
        print(f"  {grade}: 距离={metrics['total_distance']:.2f}m  速度={metrics['avg_speed']:.2f}m/s  避障={metrics['obstacles_avoided']}  告警={metrics['alerts_triggered']}")

        controller.stop_patrol()

    # 打印对比表
    print_banner("五级巡逻能力对比")
    print(f"{'等级':<6} {'最大速度':<10} {'控制频率':<10} {'避障策略':<10} {'避障次数':<8} {'平均速度':<10} {'告警次数':<8}")
    print("-" * 70)
    for grade in grades:
        spec = get_patrol_spec(grade)
        m = results[grade]
        print(f"{grade:<6} {spec['max_patrol_speed']:<10.1f} {spec['control_frequency']:<10.0f} {spec['avoidance_strategy']:<10} {m['obstacles_avoided']:<8} {m['avg_speed']:<10.3f} {m['alerts_triggered']:<8}")

    return results


def run_emergency_recovery_demo(grade: str = 'L'):
    """紧急避障恢复演示"""
    print_banner(f"紧急避障与自主恢复 - {grade}级")
    spec = get_patrol_spec(grade)

    route = PatrolRoute(
        name="emergency_test",
        points=[PatrolPoint(x=0.0, y=0.0, name="start"), PatrolPoint(x=3.0, y=0.0, name="end")],
        loop=False,
    )

    controller = create_patrol_controller(grade=grade, pose=(0.0, 0.0, 0.0), route=route)

    # 模拟突然出现的障碍物
    controller.start_patrol()
    for step in range(50):
        dt = 1.0 / spec['control_frequency']
        controller.update(dt=dt)

    # 在路径上突然添加障碍物
    print(f"\n[Step 50] 检测到突然出现的障碍物!")
    controller.obstacles.append(Obstacle(position=np.array([1.0, 0.0]), radius=0.4, type="person"))

    for step in range(50, 100):
        dt = 1.0 / spec['control_frequency']
        state, _ = controller.update(dt=dt)
        if step % 10 == 0:
            pose = controller.get_pose()
            print(f"  Step {step}: pos=({pose[0]:.2f}, {pose[1]:.2f}) state={state.value}")

    metrics = controller.get_metrics()
    print(f"\n避障成功: {'✓' if metrics['obstacles_avoided'] > 0 else '✗'}  避障次数: {metrics['obstacles_avoided']}")
    controller.stop_patrol()


def main():
    parser = argparse.ArgumentParser(description="SuperModel 自主巡逻演示")
    parser.add_argument('--grade', default='M', choices=['S', 'M', 'L', 'XL', 'XXL'], help='AGV等级')
    parser.add_argument('--all-grades', action='store_true', help='五级对比模式')
    parser.add_argument('--steps', type=int, default=200, help='模拟步数')
    parser.add_argument('--dense', action='store_true', help='密集障碍物场景')
    parser.add_argument('--emergency', action='store_true', help='紧急避障恢复演示')
    parser.add_argument('--quiet', action='store_true', help='静默模式(不打印中间状态)')

    args = parser.parse_args()

    if args.all_grades:
        run_five_grade_benchmark()
    elif args.dense:
        run_multi_obstacle_demo(args.grade)
    elif args.emergency:
        run_emergency_recovery_demo(args.grade)
    else:
        run_single_patrol_demo(args.grade, steps=args.steps, verbose=not args.quiet)


if __name__ == '__main__':
    main()
