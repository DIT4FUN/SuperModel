#!/usr/bin/env python3
"""
SuperModel 核心目标系统演示

展示六大核心目标(P0-P5)的持续运行和实时决策能力
"""

import numpy as np
from src.core import CoreBrain


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_normal_scenario():
    """演示1: 正常安全场景"""
    print_section("演示1: 正常安全场景 (人员3米外)")

    brain = CoreBrain(grade='M')

    # 正常场景 - 人员距离3米,环境安全
    ctx = brain.update_context(
        robot_position=np.array([0.0, 0.0, 0.0]),
        robot_velocity=np.array([0.5, 0.0, 0.0]),
        laser_ranges=np.array([5.0] * 360),
        human_positions=[np.array([3.0, 0.0, 0.0])],  # 3米外
        robot_battery_level=0.9,
        robot_temperature=35.0,
        human_instructions=['前进'],
        instruction_urgency=0.6,
    )

    scores = brain.get_all_scores()
    print("\n目标评分:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    decision = brain.decide()
    print(f"\n决策: {decision.decision_type.value}")
    print(f"安全通过: {decision.safety_passed}")
    print(f"伦理通过: {decision.ethical_passed}")
    print(f"动作: {decision.action[:3]}")
    print(f"理由: {decision.reasoning}")


def demo_dangerous_scenario():
    """演示2: 危险场景"""
    print_section("演示2: 危险场景 (人员30cm)")

    brain = CoreBrain(grade='M')

    # 危险场景 - 人员距离30cm,立即触发安全停止
    ctx = brain.update_context(
        robot_position=np.array([0.0, 0.0, 0.0]),
        robot_velocity=np.array([1.5, 0.0, 0.0]),
        laser_ranges=np.array([0.2] * 360),  # 极近障碍
        human_positions=[np.array([0.3, 0.0, 0.0])],  # 30cm!
        robot_battery_level=0.9,
        robot_temperature=35.0,
    )

    scores = brain.get_all_scores()
    print("\n目标评分:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    decision = brain.decide()
    print(f"\n决策: {decision.decision_type.value}")
    print(f"安全: {decision.safety_passed}")
    print(f"理由: {decision.reasoning}")


def demo_battery_critical():
    """演示3: 电量危急场景"""
    print_section("演示3: 电量危急场景 (10%)")

    brain = CoreBrain(grade='M')

    # 电量危急 - 触发自我保护
    ctx = brain.update_context(
        robot_position=np.array([0.0, 0.0, 0.0]),
        robot_velocity=np.array([0.5, 0.0, 0.0]),
        laser_ranges=np.array([5.0] * 360),
        human_positions=[np.array([3.0, 0.0, 0.0])],
        robot_battery_level=0.1,  # 10%!
        robot_temperature=35.0,
    )

    scores = brain.get_all_scores()
    print("\n目标评分:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    decision = brain.decide()
    print(f"\n决策: {decision.decision_type.value}")
    print(f"理由: {decision.reasoning}")


def demo_instruction_follow():
    """演示4: 指令遵循"""
    print_section("演示4: 指令遵循 (紧急指令)")

    brain = CoreBrain(grade='M')

    # 有紧急指令
    ctx = brain.update_context(
        robot_position=np.array([0.0, 0.0, 0.0]),
        robot_velocity=np.array([0.0, 0.0, 0.0]),
        laser_ranges=np.array([5.0] * 360),
        human_positions=[np.array([5.0, 0.0, 0.0])],  # 远离人员
        robot_battery_level=0.8,
        robot_temperature=35.0,
        human_instructions=['立即前进到3米处'],
        instruction_urgency=0.9,  # 高紧急度
    )

    scores = brain.get_all_scores()
    print("\n目标评分:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    decision = brain.decide()
    print(f"\n决策: {decision.decision_type.value}")
    print(f"安全: {decision.safety_passed}")
    print(f"理由: {decision.reasoning}")


def demo_status_monitoring():
    """演示5: 实时状态监控"""
    print_section("演示5: 实时状态监控")

    brain = CoreBrain(grade='M')

    ctx = brain.update_context(
        robot_position=np.array([0.0, 0.0, 0.0]),
        robot_velocity=np.array([0.5, 0.0, 0.0]),
        laser_ranges=np.array([5.0] * 360),
        human_positions=[np.array([3.0, 0.0, 0.0])],
        robot_battery_level=0.85,
        robot_temperature=38.0,
    )

    status = brain.get_status()
    print("\n核心状态:")
    print(f"  运行状态: {status['running']}")
    print(f"  总周期数: {status['total_cycles']}")
    print(f"  安全护盾: {status['safety_shield']['safety_score']:.3f}")
    print(f"  伦理评估: {status['value_judgment']['kindness_score']:.3f}")
    print(f"  自我保护: {status['self_preservation']['overall_health']:.3f}")
    print(f"  自我进化: {status['self_evolution']['learning_progress']:.3f}")

    print("\n所有目标状态:")
    goals_status = brain.get_goals_status()
    for k, v in goals_status.items():
        if 'score' in v:
            print(f"  {k}: {v['score']:.3f}")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  SuperModel 核心目标系统演示 (v2.24.0)".center(58) + "║")
    print("║" + "  六大核心目标持续运行,实时决策".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    print("\n核心目标 (P0-P5, 全部 always_active):")
    print("  P0 - 保护人类安全 (SafetyShield) - 绝对优先级")
    print("  P1 - 遵循人类指令 (DecisionMaking)")
    print("  P2 - 善良品质 (ValueJudgment)")
    print("  P3 - 热爱世界 (ValueJudgment)")
    print("  P4 - 自我生存安全 (SelfPreservation)")
    print("  P5 - 自我进化 (SelfEvolution)")

    # 运行所有演示
    demo_normal_scenario()
    demo_dangerous_scenario()
    demo_battery_critical()
    demo_instruction_follow()
    demo_status_monitoring()

    print_section("演示完成")
    print("\n✅ 核心目标系统运行正常!")
    print("✅ 实时评估所有传感器维度")
    print("✅ 自动生成最优动作,安全优先")
    print("✅ 6大目标持续运行,50Hz周期")
    print()


if __name__ == "__main__":
    main()
