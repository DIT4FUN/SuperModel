#!/usr/bin/env python3
"""
run_domain_scenes_demo.py - 医疗/工业/联邦学习场景演示
Healthcare, Industrial & Federated Learning Scene Demo
SuperModel 超模态大模型具身智能系统

运行方式:
    python sim_demos/run_domain_scenes_demo.py
"""

import time
import sys
import os

# 确保src在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def healthcare_scene_demo():
    print_header("🏥 医疗场景化具身智能演示")
    
    from src.embodied import (
        HealthcareZone, HealthcareRiskLevel, PatientCallPriority,
        MedicationType, SpecimenCategory, HealthcareTaskLibrary,
        HealthcareSceneController, InfectionControlMonitor,
        PatientCallHandler, MedicationDeliveryPlanner, SpecimenTransportManager,
    )
    
    # 1. 创建医疗任务
    print("\n[1] 医疗任务库")
    lib = HealthcareTaskLibrary()
    
    # 常规口服药配送
    routine = lib.create_medication_task(
        'routine_oral', HealthcareZone.PHARMACY, HealthcareZone.WARD, 'P001', 'MED001'
    )
    print(f"  常规口服药: {routine.task_id}, 优先级={routine.priority.name}, 时限={routine.time_constraint}s")
    
    # 急救注射药配送
    urgent = lib.create_medication_task(
        'urgent_injection', HealthcareZone.PHARMACY, HealthcareZone.ICU, 'P002', 'MED002'
    )
    print(f"  急救注射药: {urgent.task_id}, 优先级={urgent.priority.name}, 时限={urgent.time_constraint}s, 管制={urgent.requires_controlled_access}")
    
    # 标本运输
    specimen = lib.create_specimen_task(
        'blood_emergency', HealthcareZone.EMERGENCY, HealthcareZone.LABORATORY, 'SPEC001'
    )
    print(f"  急救血液标本: {specimen.task_id}, 优先级={specimen.priority.name}, 时限={specimen.time_constraint}s")
    
    # 2. 感染控制监控
    print("\n[2] 感染控制系统")
    monitor = InfectionControlMonitor()
    
    # 各区域风险等级
    zones = [HealthcareZone.ISOLATION, HealthcareZone.ICU, HealthcareZone.WARD, HealthcareZone.CORRIDOR]
    for zone in zones:
        risk = monitor.get_risk_level(zone)
        status = monitor.get_decontamination_status(zone)
        print(f"  {zone.name}: 风险={risk}, 需消毒={status['needs_decontamination']}")
    
    # 3. 患者呼叫处理
    print("\n[3] 患者呼叫处理")
    handler = PatientCallHandler(lib)
    
    # 处理多个呼叫
    handler.handle_call('CALL001', 'P001', HealthcareZone.WARD, PatientCallPriority.routine)
    handler.handle_call('CALL002', 'P002', HealthcareZone.ICU, PatientCallPriority.urgent)
    handler.handle_call('CALL003', 'P003', HealthcareZone.ICU, PatientCallPriority.emergency)
    
    active = handler.get_active_calls()
    print(f"  活跃呼叫数: {len(active)}")
    for call in active[:3]:
        print(f"    {call['call_id']}: {call['task'].priority.name}, 等待{call['waiting_seconds']:.1f}s")
    
    # 4. 医疗场景总控制器
    print("\n[4] 医疗场景总控制器")
    ctrl = HealthcareSceneController(agv_grade='M')
    ctrl.add_task(routine)
    ctrl.add_task(urgent)
    
    status = ctrl.get_scene_status()
    print(f"  场景状态: {status}")
    
    # 生成场景报告
    report = ctrl.generate_scene_report()
    print(f"  报告: 总配送={report['total_deliveries']}, 准时率={report['on_time_rate']:.1%}")
    
    print("\n✅ 医疗场景演示完成")


def industrial_scene_demo():
    print_header("🏭 工业制造场景化具身智能演示")
    
    from src.embodied import (
        ProductionLineType, WorkstationType, MaterialType,
        QualityGrade, ToolType, ProductionTask,
        ProductionLineController, QualityInspectionStation,
        PredictiveMaintenanceMonitor, ToolManagementSystem,
        SafetyMonitoringSystem, MaterialFlowCoordinator,
    )
    
    # 1. 质量检测工位
    print("\n[1] 质量检测工位")
    station = QualityInspectionStation("QI_FINAL")
    
    # 执行多次检测
    results_summary = {'pass': 0, 'rework': 0, 'reject': 0}
    for i in range(20):
        value = np.random.uniform(-0.1, 0.2)
        result = station.perform_inspection(f'PART_{i:04d}', 'diameter', value)
        results_summary[result['result']] = results_summary.get(result['result'], 0) + 1
    
    stats = station.get_station_stats()
    print(f"  检测总数: {stats['total_inspections']}")
    print(f"  首次通过率: {stats['first_pass_rate']:.1%}")
    print(f"  不良品: {stats['defect_count']}, 返工: {stats['rework_count']}")
    
    # 2. 预测性维护监控
    print("\n[2] 预测性维护监控")
    monitor = PredictiveMaintenanceMonitor()
    
    # 注册设备
    equip_ids = ['CNC_01', 'PRESS_01', 'WELD_01', 'ROBOT_01']
    for eid in equip_ids:
        monitor.register_equipment(eid, 'cnc', {})
    
    # 模拟遥测数据
    for eid in equip_ids[:2]:
        monitor.update_telemetry(eid, {'temperature_c': 72.0 + np.random.uniform(0, 15)})
        monitor.update_telemetry(eid, {'vibration_mm_s': 3.0 + np.random.uniform(0, 5)})
    
    health = monitor.get_overall_health_score()
    print(f"  设备健康评分: {health['health_score']:.1f}/100 ({health['status']})")
    print(f"  健康/警告/危急: {health['healthy_count']}/{health['warning_count']}/{health['critical_count']}")
    
    # 3. 工具管理系统
    print("\n[3] 工具管理系统")
    tms = ToolManagementSystem()
    
    # 注册和安装工具
    tools = [
        ('TM_END_10', ToolType.END_MILL),
        ('TM_DRL_5', ToolType.DRILL_BIT),
        ('TM_INS_45', ToolType.INSERT),
    ]
    for tid, ttype in tools:
        tms.register_tool(tid, ttype, {'diameter': 10.0})
        tms.install_tool(tid, WorkstationType.CNC_MACHINE)
    
    # 模拟使用
    for tid in ['TM_END_10', 'TM_DRL_5']:
        usage = tms.record_tool_usage(tid, hours=50.0, parts=500)
        status = tms.get_tool_status(tid)
        print(f"  {tid}: {status['status']}, 寿命剩余: {status['life_remaining_percent']:.0f}%")
    
    # 4. 安全监控系统
    print("\n[4] 工业安全监控")
    sms = SafetyMonitoringSystem()
    
    # 注册人员位置
    sms.register_person('WORKER_01', (2.0, 3.0))
    sms.register_person('WORKER_02', (10.0, 15.0))
    
    # 检查各区域
    for zone in ['assembly_line', 'welding_area', 'heavy_load_zone']:
        result = sms.check_zone_entry('AGV_01', (5.0, 5.0), zone)
        print(f"  {zone}: 允许={result['allowed']}, 风险={result['risk']}")
    
    safety = sms.get_safety_status()
    print(f"  安全状态: 紧急停止={safety['emergency_stop_active']}, 在岗人员={safety['active_personnel']}")
    
    # 5. 生产线总控制器
    print("\n[5] 生产线总控制器 (柔性制造)")
    ctrl = ProductionLineController(ProductionLineType.FLEXIBLE, grade='L')
    
    # 创建生产任务
    for i in range(5):
        task = ctrl.create_production_task(
            'material_supply',
            WorkstationType.LOADING_STATION,
            WorkstationType.ASSEMBLY_STATION,
            priority=np.random.randint(1, 10),
            material_id=f'MAT_{i:03d}',
        )
        ctrl.add_task(task)
    
    oee = ctrl.get_oee()
    print(f"  OEE: {oee['oee']:.1f}% = 可用率{oee['availability']:.1%} × 性能{oee['performance']:.1%} × 质量{oee['quality']:.1%}")
    
    report = ctrl.get_production_report()
    print(f"  设备健康: {report['equipment_health']['health_score']:.1f}/100")
    print(f"  生产任务: {report['total_production_tasks']}个")
    
    print("\n✅ 工业场景演示完成")


def federated_learning_demo():
    print_header("🤖 联邦学习多AGV协同演示")
    
    from src.embodied import (
        FLClientState, FLRoundResult, LocalTrainingResult,
        FederatedClient, FederatedServer, DifferentialPrivacy,
        ByzantineFilter, AdaptiveAggregator, FederatedLearningCoordinator,
        create_federated_learning_system,
    )
    
    # 1. 差分隐私
    print("\n[1] 差分隐私 (Gaussian Mechanism)")
    dp = DifferentialPrivacy(epsilon=2.0, delta=1e-5)
    print(f"  ε={dp.epsilon}, δ={dp.delta:.0e}, 噪声乘数={dp.noise_multiplier:.4f}")
    
    gradient = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    noisy = dp.add_noise_to_gradient(gradient)
    print(f"  原始梯度: {gradient}")
    print(f"  加噪梯度: {noisy}")
    spent_e, spent_d = dp.compute_privacy_spent(5)
    print(f"  5轮后隐私消耗: ε={spent_e:.1f}, δ={spent_d:.0e}")
    
    # 2. 拜占庭容错过滤器
    print("\n[2] 拜占庭容错过滤器")
    bf = ByzantineFilter(f=1, n=10)
    
    # 模拟正常客户端
    results = [
        LocalTrainingResult(
            client_id=f'AGV_{i:02d}', round_number=1, num_samples=200,
            training_loss=0.5 + np.random.uniform(-0.05, 0.05),
            validation_accuracy=0.88 + np.random.uniform(-0.03, 0.03),
            gradients={}, model_update_hash=f"h{i}",
            training_time_seconds=10, communication_bytes=100000,
            client_state=FLClientState.IDLE,
        )
        for i in range(5)
    ]
    # 添加一个异常的拜占庭客户端
    byzantine_result = LocalTrainingResult(
        client_id='AGV_BYZ', round_number=1, num_samples=200,
        training_loss=5.0,  # 明显异常
        validation_accuracy=0.1,
        gradients={}, model_update_hash='byz',
        training_time_seconds=10, communication_bytes=100000,
        client_state=FLClientState.IDLE,
    )
    results.append(byzantine_result)
    
    filtered = bf.filter_byzantine_clients(results)
    print(f"  客户端数: {len(results)}, 被过滤: {len(filtered)}")
    if filtered:
        print(f"  被过滤: {filtered}")
    
    # 3. 自适应聚合器
    print("\n[3] 自适应聚合权重")
    agg = AdaptiveAggregator()
    test_results = [
        LocalTrainingResult(
            client_id='AGV_01', round_number=1, num_samples=300,
            training_loss=0.4, validation_accuracy=0.92,
            gradients={}, model_update_hash='h1',
            training_time_seconds=8, communication_bytes=80000,
            client_state=FLClientState.IDLE,
        ),
        LocalTrainingResult(
            client_id='AGV_02', round_number=1, num_samples=200,
            training_loss=0.6, validation_accuracy=0.85,
            gradients={}, model_update_hash='h2',
            training_time_seconds=12, communication_bytes=120000,
            client_state=FLClientState.IDLE,
        ),
    ]
    weights = agg.compute_adaptive_weights(test_results)
    print(f"  AGV_01权重: {weights.get('AGV_01', 0):.4f}")
    print(f"  AGV_02权重: {weights.get('AGV_02', 0):.4f}")
    
    # 4. 联邦学习服务器
    print("\n[4] 联邦学习服务器")
    server = FederatedServer(
        model_config={'num_layers': 4, 'gradient_shape': (128,)},
        num_rounds=20, min_clients_per_round=2,
        use_differential_privacy=True, dp_epsilon=3.0,
    )
    
    # 注册3个AGV客户端
    for i in range(3):
        client = FederatedClient(
            client_id=f'FL_C{i:02d}', agv_id=f'AGV_{i:02d}',
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            local_epochs=3,
        )
        server.register_client(client)
    
    print(f"  已注册客户端: {len(server._clients)}")
    
    # 执行一轮训练
    selected = server.select_clients(min_count=2)
    print(f"  选中参与本轮: {selected}")
    
    result = server.execute_round(selected)
    print(f"  第{result.round_number}轮完成:")
    print(f"    参与客户端: {result.num_participants}")
    print(f"    全局损失: {result.global_loss:.4f}")
    print(f"    全局精度: {result.global_accuracy:.4f}")
    print(f"    隐私保护: ε={result.epsilon}")
    print(f"    通信量: {result.total_communication_bytes / 1e3:.1f}KB")
    
    # 5. 联邦学习协调器
    print("\n[5] 联邦学习协调器 (5 AGV编队)")
    coord = create_federated_learning_system(num_agvs=5, grade='L')
    
    status = coord.get_system_status()
    print(f"  活跃AGV: {status['active_agvs']}")
    print(f"  注册客户端: {status['registered_clients']}")
    
    # 执行3轮训练
    for round_num in range(1, 4):
        selected = coord._server.select_clients(min_count=3)
        if len(selected) >= 3:
            result = coord._server.execute_round(selected)
            print(f"  第{result.round_number}轮: 损失={result.global_loss:.4f}, 精度={result.global_accuracy:.4f}")
    
    final_summary = coord._server.get_training_summary()
    print(f"  训练摘要: 第{final_summary['current_round']}轮, "
          f"平均精度={final_summary['avg_accuracy']:.4f}, "
          f"最优精度={final_summary['best_accuracy']:.4f}")
    
    print("\n✅ 联邦学习演示完成")


def main():
    print("\n" + "="*60)
    print("  SuperModel 超模态大模型 - 领域场景化具身智能")
    print("  Healthcare / Industrial / Federated Learning")
    print("="*60)
    
    healthcare_scene_demo()
    industrial_scene_demo()
    federated_learning_demo()
    
    print("\n" + "="*60)
    print("  🎉 所有场景演示完成!")
    print("="*60)


if __name__ == '__main__':
    main()
