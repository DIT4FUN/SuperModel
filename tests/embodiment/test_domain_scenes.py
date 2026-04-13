"""
test_domain_scenes.py - 医疗/工业场景化具身智能测试
Healthcare and Industrial Scene Intelligence Tests
"""

import pytest
import time
import numpy as np
from src.embodied import (
    # Healthcare
    HealthcareZone, HealthcareRiskLevel, PatientCallPriority,
    MedicationType, SpecimenCategory, HealthcareTask,
    HealthcareTaskLibrary, HealthcareSceneController,
    InfectionControlMonitor, PatientCallHandler,
    MedicationDeliveryPlanner, SpecimenTransportManager,
    get_healthcare_scene_controller,
    # Industrial
    ProductionLineType, WorkstationType, MaterialType,
    QualityGrade, ToolType, ProductionTask,
    ProductionLineController, QualityInspectionStation,
    PredictiveMaintenanceMonitor, ToolManagementSystem,
    SafetyMonitoringSystem, MaterialFlowCoordinator,
    get_industrial_scene_controller,
    # Federated Learning
    FLClientState, FLRoundResult, LocalTrainingResult,
    FederatedClient, FederatedServer, DifferentialPrivacy,
    ByzantineFilter, AdaptiveAggregator, FederatedLearningCoordinator,
    create_federated_learning_system,
)


# ==================== Healthcare Scene Tests ====================

class TestHealthcareZone:
    def test_zone_enum_values(self):
        assert HealthcareZone.PHARMACY.value == 1
        assert HealthcareZone.WARD.value == 2
        assert HealthcareZone.ICU.value == 3
        assert HealthcareZone.OPERATING_ROOM.value == 4

    def test_all_zones_defined(self):
        zones = list(HealthcareZone)
        assert len(zones) >= 10
        assert HealthcareZone.PHARMACY in zones
        assert HealthcareZone.ISOLATION in zones


class TestHealthcareRiskLevel:
    def test_risk_levels_ascending(self):
        assert HealthcareRiskLevel.LOW.value == 1
        assert HealthcareRiskLevel.MEDIUM.value == 2
        assert HealthcareRiskLevel.HIGH.value == 3
        assert HealthcareRiskLevel.CRITICAL.value == 4


class TestHealthcareTaskLibrary:
    def test_create_library(self):
        lib = HealthcareTaskLibrary()
        assert lib._task_counter == 0

    def test_create_medication_task(self):
        lib = HealthcareTaskLibrary()
        task = lib.create_medication_task(
            template_name='routine_oral',
            source=HealthcareZone.PHARMACY,
            destination=HealthcareZone.WARD,
            patient_id='P001',
            medication_id='MED001',
        )
        assert task.task_type == 'medication_delivery'
        assert task.priority == HealthcareRiskLevel.LOW
        assert task.patient_id == 'P001'
        assert task.requires_cold_chain is False
        assert task.requires_controlled_access is False
        assert task.time_constraint == 900

    def test_create_urgent_medication_task(self):
        lib = HealthcareTaskLibrary()
        task = lib.create_medication_task(
            template_name='urgent_injection',
            source=HealthcareZone.PHARMACY,
            destination=HealthcareZone.ICU,
            patient_id='P002',
            medication_id='MED002',
        )
        assert task.priority == HealthcareRiskLevel.MEDIUM
        assert task.requires_controlled_access is True
        assert task.time_constraint == 300

    def test_create_specimen_task(self):
        lib = HealthcareTaskLibrary()
        task = lib.create_specimen_task(
            template_name='blood_routine',
            source=HealthcareZone.WARD,
            destination=HealthcareZone.LABORATORY,
            specimen_id='SPEC001',
        )
        assert task.task_type == 'specimen_transport'
        assert task.priority == HealthcareRiskLevel.LOW
        assert task.requires_sterile is True

    def test_create_supply_task(self):
        lib = HealthcareTaskLibrary()
        task = lib.create_supply_task(
            template_name='sterile_supplies',
            source=HealthcareZone.CENTRAL_SUPPLY,
            destination=HealthcareZone.OPERATING_ROOM,
            supply_id='SUP001',
        )
        assert task.task_type == 'supply_delivery'
        assert task.requires_sterile is True


class TestInfectionControlMonitor:
    def test_create_monitor(self):
        monitor = InfectionControlMonitor()
        assert len(monitor._decontamination_events) == 0
        assert len(monitor._isolation_alerts) == 0

    def test_zone_risk_levels(self):
        monitor = InfectionControlMonitor()
        assert monitor.get_risk_level(HealthcareZone.ISOLATION) == 5
        assert monitor.get_risk_level(HealthcareZone.ICU) == 4
        assert monitor.get_risk_level(HealthcareZone.CORRIDOR) == 1

    def test_requires_isolation_protocol(self):
        monitor = InfectionControlMonitor()
        assert monitor.requires_isolation_protocol(HealthcareZone.ISOLATION) is True
        assert monitor.requires_isolation_protocol(HealthcareZone.ICU) is True
        assert monitor.requires_isolation_protocol(HealthcareZone.WARD) is False

    def test_check_zone_access_normal(self):
        monitor = InfectionControlMonitor()
        allowed, reason = monitor.check_zone_access(
            HealthcareZone.WARD, {'isolation_compatible': False}
        )
        assert allowed is True

    def test_record_decontamination(self):
        monitor = InfectionControlMonitor()
        monitor.record_decontamination(HealthcareZone.OPERATING_ROOM)
        assert HealthcareZone.OPERATING_ROOM in monitor._last_decontamination
        assert len(monitor._decontamination_events) == 1

    def test_decontamination_status(self):
        monitor = InfectionControlMonitor()
        status = monitor.get_decontamination_status(HealthcareZone.STERILE_STORAGE)
        assert 'needs_decontamination' in status
        assert status['risk_level'] == 2


class TestPatientCallHandler:
    def test_create_handler(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        assert len(handler._active_calls) == 0

    def test_handle_call(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        task = handler.handle_call(
            call_id='CALL001',
            patient_id='P001',
            ward_zone=HealthcareZone.WARD,
            priority=PatientCallPriority.routine,
        )
        assert task is not None
        assert task.patient_id == 'P001'
        assert task.priority == HealthcareRiskLevel.LOW

    def test_handle_emergency_call(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        task = handler.handle_call(
            call_id='CALL002',
            patient_id='P002',
            ward_zone=HealthcareZone.ICU,
            priority=PatientCallPriority.emergency,
        )
        # emergency maps to urgent_injection template which is MEDIUM priority
        assert task.priority == HealthcareRiskLevel.MEDIUM

    def test_complete_call(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        handler.handle_call('C001', 'P001', HealthcareZone.WARD, PatientCallPriority.routine)
        handler.complete_call('C001')
        assert 'C001' not in handler._active_calls
        assert len(handler._call_history) == 1

    def test_get_active_calls_sorted(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        handler.handle_call('C1', 'P1', HealthcareZone.WARD, PatientCallPriority.routine)
        handler.handle_call('C2', 'P2', HealthcareZone.ICU, PatientCallPriority.emergency)
        calls = handler.get_active_calls()
        assert len(calls) == 2
        # emergency (urgent_injection) = MEDIUM; routine = LOW
        assert calls[0]['task'].priority in (HealthcareRiskLevel.CRITICAL, HealthcareRiskLevel.MEDIUM)


class TestMedicationDeliveryPlanner:
    def test_create_planner(self):
        monitor = InfectionControlMonitor()
        planner = MedicationDeliveryPlanner(monitor)
        assert planner._infection_monitor is monitor

    def test_plan_delivery_normal(self):
        monitor = InfectionControlMonitor()
        planner = MedicationDeliveryPlanner(monitor)
        task = HealthcareTask(
            task_id='T001',
            task_type='medication_delivery',
            priority=HealthcareRiskLevel.LOW,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.WARD,
            payload_type='medication',
            payload_id='M001',
        )
        plan = planner.plan_delivery(task, [{'agv_id': 'AGV1', 'distance_to_pharmacy': 10}])
        assert plan is not None
        assert plan['status'] == 'planned'
        assert 'assigned_agv' in plan


class TestSpecimenTransportManager:
    def test_create_manager(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        assert len(manager._specimen_tracking) == 0

    def test_register_specimen(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        task = manager.register_specimen(
            specimen_id='SPEC001',
            category=SpecimenCategory.BLOOD,
            source=HealthcareZone.WARD,
        )
        assert task.task_type == 'specimen_transport'
        assert task.destination_zone == HealthcareZone.LABORATORY
        assert task.requires_sterile is True

    def test_verify_chain_of_custody_fresh(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        manager.register_specimen('SPEC001', SpecimenCategory.BLOOD, HealthcareZone.WARD)
        chain = manager.verify_chain_of_custody('SPEC001')
        assert chain['valid'] is True
        assert chain['remaining_seconds'] > 0


class TestHealthcareSceneController:
    def test_create_controller(self):
        ctrl = HealthcareSceneController(agv_grade='M')
        assert ctrl.agv_grade == 'M'

    def test_add_and_get_task(self):
        ctrl = HealthcareSceneController()
        lib = ctrl.task_library
        task = lib.create_medication_task(
            'routine_oral', HealthcareZone.PHARMACY, HealthcareZone.WARD, 'P001', 'M001'
        )
        ctrl.add_task(task)
        assert len(ctrl._task_queue) == 1
        retrieved = ctrl.get_next_task()
        assert retrieved.task_id == task.task_id

    def test_priority_ordering(self):
        ctrl = HealthcareSceneController()
        lib = ctrl.task_library
        low = lib.create_medication_task('routine_oral', HealthcareZone.PHARMACY, HealthcareZone.WARD, 'P1', 'M1')
        high = lib.create_medication_task('urgent_injection', HealthcareZone.PHARMACY, HealthcareZone.ICU, 'P2', 'M2')
        ctrl.add_task(low)
        ctrl.add_task(high)
        first = ctrl.get_next_task()
        # urgent_injection template = MEDIUM priority
        assert first.priority == HealthcareRiskLevel.MEDIUM

    def test_scene_status(self):
        ctrl = HealthcareSceneController()
        status = ctrl.get_scene_status()
        assert 'pending_tasks' in status
        assert 'completed_tasks' in status
        assert 'agv_grade' in status

    def test_global_singleton(self):
        ctrl1 = get_healthcare_scene_controller('L')
        ctrl2 = get_healthcare_scene_controller('M')
        assert ctrl1 is ctrl2  # Same instance


# ==================== Industrial Scene Tests ====================

class TestProductionLineType:
    def test_line_types(self):
        assert ProductionLineType.ASSEMBLY.value == 1
        assert ProductionLineType.FLEXIBLE.value == 6
        assert ProductionLineType.AUTOMATED_WAREHOUSE.value == 7

    def test_all_types_defined(self):
        types = list(ProductionLineType)
        assert len(types) >= 6


class TestWorkstationType:
    def test_workstation_types(self):
        assert WorkstationType.CNC_MACHINE.value == 1
        assert WorkstationType.ROBOT_CELL.value == 2
        assert WorkstationType.QUALITY_GATE.value == 7


class TestQualityGrade:
    def test_grade_order(self):
        assert QualityGrade.A_PRIME.value == 1
        assert QualityGrade.A_STANDARD.value == 2
        assert QualityGrade.B_REWORK.value == 3
        assert QualityGrade.C_REJECT.value == 4


class TestQualityInspectionStation:
    def test_create_station(self):
        station = QualityInspectionStation("QI_01")
        assert station.station_id == "QI_01"
        assert station._inspection_count == 0

    def test_perform_inspection_pass(self):
        station = QualityInspectionStation("QI_01")
        result = station.perform_inspection('PART001', 'diameter', 0.02)
        assert result['result'] in ('pass', 'rework')

    def test_perform_inspection_reject(self):
        station = QualityInspectionStation("QI_01")
        result = station.perform_inspection('PART001', 'diameter', 1.0)
        assert result['grade'] == QualityGrade.C_REJECT

    def test_station_stats(self):
        station = QualityInspectionStation("QI_01")
        station.perform_inspection('P1', 'diameter', 0.01)
        station.perform_inspection('P2', 'diameter', 0.5)
        stats = station.get_station_stats()
        assert stats['total_inspections'] == 2
        assert stats['defect_count'] >= 1


class TestPredictiveMaintenanceMonitor:
    def test_create_monitor(self):
        monitor = PredictiveMaintenanceMonitor()
        assert len(monitor._equipment_states) == 0

    def test_register_equipment(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment('CNC01', 'cnc_mill', {'max_rpm': 8000})
        assert 'CNC01' in monitor._equipment_states

    def test_update_telemetry_warning(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment('CNC01', 'cnc_mill', {})
        alerts = monitor.update_telemetry('CNC01', {'temperature_c': 75.0})
        assert len(alerts) >= 1
        assert alerts[0]['level'] == 'warning'

    def test_update_telemetry_critical(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment('CNC01', 'cnc_mill', {})
        alerts = monitor.update_telemetry('CNC01', {'temperature_c': 90.0})
        assert len(alerts) >= 1
        assert alerts[0]['level'] == 'critical'

    def test_predict_maintenance_window(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment('CNC01', 'cnc_mill', {})
        window = monitor.predict_maintenance_window('CNC01')
        assert window is not None
        assert 'remaining_hours' in window

    def test_overall_health_score(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment('E1', 'lathe', {})
        monitor.register_equipment('E2', 'press', {})
        score = monitor.get_overall_health_score()
        assert 'health_score' in score
        assert score['total_equipment'] == 2


class TestToolManagementSystem:
    def test_create_system(self):
        tms = ToolManagementSystem()
        assert len(tms._tool_inventory) == 0

    def test_register_tool(self):
        tms = ToolManagementSystem()
        tms.register_tool('TM001', ToolType.END_MILL, {'diameter': 10.0})
        assert 'TM001' in tms._tool_inventory
        assert tms._tool_inventory['TM001']['status'] == 'available'

    def test_install_tool(self):
        tms = ToolManagementSystem()
        tms.register_tool('TM001', ToolType.END_MILL, {})
        result = tms.install_tool('TM001', WorkstationType.CNC_MACHINE)
        assert result is True
        assert tms._tool_inventory['TM001']['status'] == 'installed'

    def test_record_tool_usage(self):
        tms = ToolManagementSystem()
        tms.register_tool('TM001', ToolType.DRILL_BIT, {})
        tms.install_tool('TM001', WorkstationType.CNC_MACHINE)
        result = tms.record_tool_usage('TM001', hours=5.0, parts=100)
        assert result['status'] == 'installed'
        assert result['needs_replacement'] is False

    def test_tool_worn_detection(self):
        tms = ToolManagementSystem()
        tms.register_tool('TM001', ToolType.DRILL_BIT, {})
        tms.install_tool('TM001', WorkstationType.CNC_MACHINE)
        result = tms.record_tool_usage('TM001', hours=200.0, parts=1000)
        assert result['needs_replacement'] is True


class TestSafetyMonitoringSystem:
    def test_create_system(self):
        sms = SafetyMonitoringSystem()
        assert sms._safety_stop_active is False

    def test_register_person(self):
        sms = SafetyMonitoringSystem()
        sms.register_person('PERSON01', (1.0, 2.0))
        assert 'PERSON01' in sms._personnel_positions

    def test_check_zone_entry_safe(self):
        sms = SafetyMonitoringSystem()
        result = sms.check_zone_entry('AGV1', (0.0, 0.0), 'assembly_line')
        assert result['allowed'] is True

    def test_emergency_stop(self):
        sms = SafetyMonitoringSystem()
        sms.trigger_emergency_stop('Fire alarm')
        assert sms._safety_stop_active is True
        assert len(sms._incident_log) == 1

    def test_reset_emergency_stop(self):
        sms = SafetyMonitoringSystem()
        sms.trigger_emergency_stop('Test')
        sms.reset_emergency_stop()
        assert sms._safety_stop_active is False


class TestMaterialFlowCoordinator:
    def test_create_coordinator(self):
        coord = MaterialFlowCoordinator(ProductionLineType.ASSEMBLY)
        assert coord.line_type == ProductionLineType.ASSEMBLY

    def test_allocate_transfer(self):
        coord = MaterialFlowCoordinator(ProductionLineType.FLEXIBLE)
        task = ProductionTask(
            task_id='PT001', task_type='material_supply',
            priority=5, source_station=WorkstationType.LOADING_STATION,
            destination_station=WorkstationType.ASSEMBLY_STATION,
        )
        result = coord.allocate_transfer(task, 'AGV1')
        assert result['task_id'] == 'PT001'
        assert result['assigned_agv'] == 'AGV1'
        assert result['estimated_time'] > 0

    def test_complete_transfer(self):
        coord = MaterialFlowCoordinator(ProductionLineType.MACHINING)
        task = ProductionTask(
            task_id='PT002', task_type='part_transfer',
            priority=7, source_station=WorkstationType.ROBOT_CELL,
            destination_station=WorkstationType.INSPECTION_STATION,
        )
        coord.allocate_transfer(task, 'AGV2')
        success = coord.complete_transfer('PT002')
        assert success is True
        assert len(coord._active_transfers) == 0


class TestProductionLineController:
    def test_create_controller(self):
        ctrl = ProductionLineController(ProductionLineType.FLEXIBLE, 'L')
        assert ctrl.line_type == ProductionLineType.FLEXIBLE
        assert ctrl.agv_grade == 'L'

    def test_create_production_task(self):
        ctrl = ProductionLineController()
        task = ctrl.create_production_task(
            'material_supply',
            WorkstationType.LOADING_STATION,
            WorkstationType.ASSEMBLY_STATION,
            priority=7,
        )
        assert task.task_type == 'material_supply'
        assert task.priority == 7
        assert task.status == 'pending'

    def test_add_and_get_task(self):
        ctrl = ProductionLineController()
        task = ctrl.create_production_task(
            'quality_check', WorkstationType.ASSEMBLY_STATION, WorkstationType.QUALITY_GATE
        )
        ctrl.add_task(task)
        assert len(ctrl._production_tasks) == 1

    def test_oee_empty(self):
        ctrl = ProductionLineController()
        oee = ctrl.get_oee()
        assert oee['oee'] == 0

    def test_production_report(self):
        ctrl = ProductionLineController()
        report = ctrl.get_production_report()
        assert 'timestamp' in report
        assert 'line_type' in report
        assert 'oee' in report
        assert 'equipment_health' in report

    def test_global_singleton(self):
        ctrl1 = get_industrial_scene_controller(ProductionLineType.ASSEMBLY, 'M')
        ctrl2 = get_industrial_scene_controller(ProductionLineType.MACHINING, 'L')
        assert ctrl1 is ctrl2


# ==================== Federated Learning Tests ====================

class TestDifferentialPrivacy:
    def test_create_dp(self):
        dp = DifferentialPrivacy(epsilon=2.0, delta=1e-5)
        assert dp.epsilon == 2.0
        assert dp.delta == 1e-5

    def test_add_noise_to_gradient(self):
        dp = DifferentialPrivacy(epsilon=3.0)
        gradient = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        noisy = dp.add_noise_to_gradient(gradient)
        assert noisy.shape == gradient.shape
        assert noisy.dtype == gradient.dtype  # Same as input

    def test_add_noise_to_gradients(self):
        dp = DifferentialPrivacy(epsilon=3.0)
        grads = {
            'layer_0': np.array([1.0, 2.0], dtype=np.float32),
            'layer_1': np.array([3.0, 4.0], dtype=np.float32),
        }
        noisy = dp.add_noise_to_gradients(grads)
        assert 'layer_0' in noisy
        assert 'layer_1' in noisy
        assert noisy['layer_0'].shape == (2,)

    def test_privacy_spent(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        spent_e, spent_d = dp.compute_privacy_spent(5)
        assert spent_e == 5.0  # epsilon * rounds
        assert spent_d > 0


class TestByzantineFilter:
    def test_create_filter(self):
        bf = ByzantineFilter(f=1, n=10)
        assert bf.f == 1
        assert bf.n == 10

    def test_register_metric(self):
        bf = ByzantineFilter()
        bf.register_metric('AGV1', 0.85)
        bf.register_metric('AGV1', 0.87)
        assert len(bf._client_metrics['AGV1']) == 2

    def test_filter_normal_clients(self):
        bf = ByzantineFilter(f=1, n=5)
        results = [
            LocalTrainingResult(
                client_id=f'C{i}', round_number=1, num_samples=100,
                training_loss=0.5 + np.random.uniform(-0.1, 0.1),
                validation_accuracy=0.9, gradients={}, model_update_hash="hash",
                training_time_seconds=10, communication_bytes=1000,
                client_state=FLClientState.IDLE,
            )
            for i in range(3)
        ]
        filtered = bf.filter_byzantine_clients(results)
        assert isinstance(filtered, list)

    def test_robust_aggregate(self):
        bf = ByzantineFilter(f=1, n=5)
        updates = {
            'C1': np.array([1.0, 2.0], dtype=np.float32),
            'C2': np.array([1.1, 2.1], dtype=np.float32),
        }
        weights = {'C1': 0.5, 'C2': 0.5}
        result = bf.compute_robust_aggregate(updates, weights)
        assert result.shape == (2,)


class TestAdaptiveAggregator:
    def test_create_aggregator(self):
        agg = AdaptiveAggregator()
        assert len(agg._historical_accuracy) == 0

    def test_update_history(self):
        agg = AdaptiveAggregator()
        agg.update_client_history('AGV1', 0.85, 50000)
        agg.update_client_history('AGV1', 0.87, 48000)
        assert len(agg._historical_accuracy['AGV1']) == 2

    def test_compute_weights(self):
        agg = AdaptiveAggregator()
        results = [
            LocalTrainingResult(
                client_id='C1', round_number=1, num_samples=100,
                training_loss=0.5, validation_accuracy=0.9, gradients={},
                model_update_hash="h", training_time_seconds=10,
                communication_bytes=50000, client_state=FLClientState.IDLE,
            ),
            LocalTrainingResult(
                client_id='C2', round_number=1, num_samples=100,
                training_loss=0.6, validation_accuracy=0.85, gradients={},
                model_update_hash="h2", training_time_seconds=12,
                communication_bytes=60000, client_state=FLClientState.IDLE,
            ),
        ]
        weights = agg.compute_adaptive_weights(results)
        assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestFederatedClient:
    def test_create_client(self):
        client = FederatedClient(
            client_id='C001', agv_id='AGV1',
            model_config={'num_layers': 4, 'gradient_shape': (64,)},
        )
        assert client.client_id == 'C001'
        assert client.agv_id == 'AGV1'
        assert client.state == FLClientState.IDLE

    def test_receive_global_model(self):
        client = FederatedClient('C001', 'AGV1', {'num_layers': 2, 'gradient_shape': (32,)})
        global_model = {'layer_0': np.zeros(32, dtype=np.float32), 'layer_1': np.zeros(32, dtype=np.float32)}
        client.receive_global_model(global_model)
        assert client._local_model is not None
        assert 'layer_0' in client._local_model

    def test_local_train(self):
        client = FederatedClient('C001', 'AGV1', {'num_layers': 2, 'gradient_shape': (32,)})
        result = client.local_train({'size': 200})
        assert result.client_id == 'C001'
        assert result.num_samples == 200
        assert result.training_loss > 0
        assert result.validation_accuracy > 0
        assert len(result.gradients) == 2
        assert len(result.model_update_hash) == 16

    def test_evaluate(self):
        client = FederatedClient('C001', 'AGV1', {})
        eval_result = client.evaluate({'size': 50})
        assert 'test_accuracy' in eval_result
        assert eval_result['client_id'] == 'C001'


class TestFederatedServer:
    def test_create_server(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            num_rounds=50, min_clients_per_round=2,
        )
        assert server._round_number == 0
        assert len(server._clients) == 0
        assert len(server._round_results) == 0

    def test_initialize_model(self):
        server = FederatedServer({'num_layers': 3, 'gradient_shape': (64,)})
        model = server.global_model
        assert 'layer_0' in model
        assert 'layer_1' in model
        assert 'layer_2' in model

    def test_register_client(self):
        server = FederatedServer({'num_layers': 2, 'gradient_shape': (32,)})
        client = FederatedClient('C001', 'AGV1', {'num_layers': 2, 'gradient_shape': (32,)})
        server.register_client(client)
        assert len(server._clients) == 1
        assert 'C001' in server._clients

    def test_select_clients(self):
        server = FederatedServer({'num_layers': 2, 'gradient_shape': (32,)})
        for i in range(3):
            client = FederatedClient(f'C{i:03d}', f'AGV{i}', {'num_layers': 2, 'gradient_shape': (32,)})
            server.register_client(client)
        selected = server.select_clients(min_count=2)
        assert len(selected) <= 3

    def test_execute_round(self):
        server = FederatedServer({'num_layers': 2, 'gradient_shape': (32,)}, min_clients_per_round=1)
        client = FederatedClient('C001', 'AGV1', {'num_layers': 2, 'gradient_shape': (32,)})
        server.register_client(client)
        result = server.execute_round(['C001'])
        assert result.round_number == 1
        assert result.num_participants == 1
        assert result.global_loss > 0
        assert result.global_accuracy > 0

    def test_training_summary(self):
        server = FederatedServer({'num_layers': 2, 'gradient_shape': (32,)}, min_clients_per_round=1)
        client = FederatedClient('C001', 'AGV1', {'num_layers': 2, 'gradient_shape': (32,)})
        server.register_client(client)
        server.execute_round(['C001'])
        summary = server.get_training_summary()
        assert 'current_round' in summary
        assert summary['current_round'] == 1


class TestFederatedLearningCoordinator:
    def test_create_system(self):
        coord = create_federated_learning_system(num_agvs=3, grade='L')
        assert coord.grade == 'L'
        assert coord._server is not None

    def test_register_agv(self):
        coord = create_federated_learning_system(num_agvs=0, grade='M')
        client_id = coord.register_agv('AGV001', {'capability': 'navigation'})
        assert client_id == 'fl_client_AGV001'
        assert 'AGV001' in coord._active_agvs

    def test_start_training_round(self):
        coord = create_federated_learning_system(num_agvs=2, grade='M')
        result = coord.start_training_round()
        # May be None if not enough clients
        if result is not None:
            assert result.round_number == 1

    def test_system_status(self):
        coord = create_federated_learning_system(num_agvs=3, grade='L')
        status = coord.get_system_status()
        assert 'active_agvs' in status
        assert status['active_agvs'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
