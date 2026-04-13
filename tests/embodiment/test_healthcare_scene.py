"""
test_healthcare_scene.py - 医疗场景模块完整测试
HealthcareSceneController 医疗AGV场景: 药房/病房/ICU/手术室/检验科/血库/急诊
"""
import pytest
import time
from src.embodied.healthcare_scene import (
    HealthcareZone,
    HealthcareRiskLevel,
    PatientCallPriority,
    MedicationType,
    SpecimenCategory,
    HealthcareTask,
    HealthcareTaskLibrary,
    InfectionControlMonitor,
    PatientCallHandler,
    MedicationDeliveryPlanner,
    SpecimenTransportManager,
    HealthcareSceneController,
    get_healthcare_scene_controller,
)


class TestHealthcareEnums:
    """医疗场景枚举完整性测试"""

    def test_healthcare_zones_complete(self):
        assert len(HealthcareZone) == 12
        assert HealthcareZone.PHARMACY
        assert HealthcareZone.ICU
        assert HealthcareZone.OPERATING_ROOM
        assert HealthcareZone.BLOOD_BANK
        assert HealthcareZone.ISOLATION

    def test_healthcare_risk_levels_values(self):
        assert HealthcareRiskLevel.LOW.value == 1
        assert HealthcareRiskLevel.MEDIUM.value == 2
        assert HealthcareRiskLevel.HIGH.value == 3
        assert HealthcareRiskLevel.CRITICAL.value == 4
        assert HealthcareRiskLevel.LOW.value < HealthcareRiskLevel.CRITICAL.value

    def test_patient_call_priority(self):
        assert PatientCallPriority.routine.value == 1
        assert PatientCallPriority.urgent.value == 2
        assert PatientCallPriority.emergency.value == 3

    def test_medication_types(self):
        assert MedicationType.ORAL
        assert MedicationType.INJECTION
        assert MedicationType.INFUSION
        assert MedicationType.COLD_CHAIN
        assert MedicationType.CONTROLLED
        assert MedicationType.RADIOPHARMA
        assert MedicationType.BIOLOGICAL
        assert len(MedicationType) == 7

    def test_specimen_categories(self):
        assert SpecimenCategory.BLOOD
        assert SpecimenCategory.TISSUE
        assert SpecimenCategory.URINE
        assert SpecimenCategory.STOOL
        assert SpecimenCategory.CSF
        assert SpecimenCategory.BACTERIA
        assert SpecimenCategory.PATHOLOGY
        assert len(SpecimenCategory) == 7


class TestHealthcareTask:
    """医疗任务创建和属性测试"""

    def test_task_creation_minimal(self):
        task = HealthcareTask(
            task_id="t1",
            task_type="medication_delivery",
            priority=HealthcareRiskLevel.LOW,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.WARD,
            payload_type="medication",
            payload_id="MED001",
        )
        assert task.task_id == "t1"
        assert task.status == "pending"
        assert task.created_at > 0
        assert task.completed_at is None

    def test_task_creation_full(self):
        task = HealthcareTask(
            task_id="t2",
            task_type="specimen_transport",
            priority=HealthcareRiskLevel.HIGH,
            source_zone=HealthcareZone.LABORATORY,
            destination_zone=HealthcareZone.LABORATORY,  # valid zone
            payload_type="blood",
            payload_id="SPEC001",
            patient_id="P12345",
            requires_cold_chain=True,
            requires_controlled_access=True,
            requires_sterile=True,
            time_constraint=300.0,
        )
        assert task.patient_id == "P12345"
        assert task.requires_cold_chain is True
        assert task.requires_controlled_access is True
        assert task.requires_sterile is True
        assert task.time_constraint == 300.0

    def test_task_status_transitions(self):
        task = HealthcareTask(
            task_id="t3",
            task_type="supply_delivery",
            priority=HealthcareRiskLevel.MEDIUM,
            source_zone=HealthcareZone.CENTRAL_SUPPLY,
            destination_zone=HealthcareZone.WARD,
            payload_type="supplies",
            payload_id="SUP001",
        )
        task.status = "in_transit"
        assert task.status == "in_transit"
        task.completed_at = time.time()
        task.status = "completed"
        assert task.status == "completed"
        assert task.completed_at is not None


class TestHealthcareTaskLibrary:
    """医疗任务模板库测试"""

    def test_library_has_templates(self):
        lib = HealthcareTaskLibrary()
        assert 'routine_oral' in lib.MEDICATION_DELIVERY_TEMPLATES
        assert 'urgent_injection' in lib.MEDICATION_DELIVERY_TEMPLATES
        assert 'critical_infusion' in lib.MEDICATION_DELIVERY_TEMPLATES
        assert 'cold_chain_biological' in lib.MEDICATION_DELIVERY_TEMPLATES

    def test_specimen_templates(self):
        lib = HealthcareTaskLibrary()
        assert 'blood_routine' in lib.SPECIMEN_TRANSPORT_TEMPLATES
        assert 'blood_emergency' in lib.SPECIMEN_TRANSPORT_TEMPLATES
        assert 'tissue_pathology' in lib.SPECIMEN_TRANSPORT_TEMPLATES

    def test_supply_templates(self):
        lib = HealthcareTaskLibrary()
        assert 'sterile_supplies' in lib.SUPPLY_DELIVERY_TEMPLATES
        assert 'linen' in lib.SUPPLY_DELIVERY_TEMPLATES

    def test_cold_chain_template(self):
        lib = HealthcareTaskLibrary()
        tmpl = lib.MEDICATION_DELIVERY_TEMPLATES['cold_chain_biological']
        assert tmpl['cold_chain'] is True
        assert tmpl['priority'].value >= HealthcareRiskLevel.MEDIUM.value

    def test_controlled_template(self):
        lib = HealthcareTaskLibrary()
        tmpl = lib.MEDICATION_DELIVERY_TEMPLATES['urgent_injection']
        assert tmpl['controlled'] is True
        assert tmpl['priority'].value >= HealthcareRiskLevel.MEDIUM.value

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
        assert task.source_zone == HealthcareZone.PHARMACY
        assert task.destination_zone == HealthcareZone.WARD
        assert task.patient_id == 'P001'
        assert task.priority == HealthcareRiskLevel.LOW


class TestInfectionControlMonitor:
    """感染控制监控测试"""

    def test_monitor_initialization(self):
        monitor = InfectionControlMonitor()
        assert len(monitor._isolation_alerts) == 0
        assert len(monitor._decontamination_events) == 0

    def test_zone_risk_levels(self):
        monitor = InfectionControlMonitor()
        assert monitor.get_risk_level(HealthcareZone.ISOLATION) == 5
        assert monitor.get_risk_level(HealthcareZone.ICU) == 4
        assert monitor.get_risk_level(HealthcareZone.CORRIDOR) == 1

    def test_check_zone_access_normal(self):
        monitor = InfectionControlMonitor()
        allowed, reason = monitor.check_zone_access(
            HealthcareZone.WARD, {'isolation_compatible': True}
        )
        assert allowed is True

    def test_check_zone_access_isolation_requires_cert(self):
        monitor = InfectionControlMonitor()
        allowed, reason = monitor.check_zone_access(
            HealthcareZone.ISOLATION, {'isolation_compatible': False}
        )
        assert allowed is False
        assert '隔离' in reason

    def test_check_zone_access_isolation_allowed(self):
        monitor = InfectionControlMonitor()
        allowed, reason = monitor.check_zone_access(
            HealthcareZone.ISOLATION, {'isolation_compatible': True}
        )
        assert allowed is True

    def test_record_decontamination(self):
        monitor = InfectionControlMonitor()
        monitor.record_decontamination(HealthcareZone.OPERATING_ROOM)
        assert len(monitor._decontamination_events) == 1
        assert monitor._decontamination_events[0]['zone'] == HealthcareZone.OPERATING_ROOM

    def test_add_isolation_alert(self):
        monitor = InfectionControlMonitor()
        monitor.add_isolation_alert(HealthcareZone.ISOLATION, "Confirmed infection")
        assert len(monitor._isolation_alerts) == 1
        assert monitor._isolation_alerts[0]['zone'] == HealthcareZone.ISOLATION
        assert monitor._isolation_alerts[0]['reason'] == "Confirmed infection"

    def test_requires_isolation_protocol(self):
        monitor = InfectionControlMonitor()
        assert monitor.requires_isolation_protocol(HealthcareZone.ISOLATION) is True
        assert monitor.requires_isolation_protocol(HealthcareZone.ICU) is True
        assert monitor.requires_isolation_protocol(HealthcareZone.OPERATING_ROOM) is True
        assert monitor.requires_isolation_protocol(HealthcareZone.WARD) is False

    def test_get_decontamination_status(self):
        monitor = InfectionControlMonitor()
        status = monitor.get_decontamination_status(HealthcareZone.OPERATING_ROOM)
        assert 'zone' in status
        assert 'needs_decontamination' in status
        assert 'risk_level' in status
        assert status['zone'] == HealthcareZone.OPERATING_ROOM
        assert status['risk_level'] == 4  # OR has risk level 4


class TestPatientCallHandler:
    """患者呼叫处理测试"""

    def test_handler_initialization(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        assert len(handler._active_calls) == 0

    def test_handle_call(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        task = handler.handle_call(
            call_id="call_001",
            patient_id="P12345",
            ward_zone=HealthcareZone.WARD,
            priority=PatientCallPriority.routine,
        )
        assert task is not None
        assert task.patient_id == "P12345"
        assert task.destination_zone == HealthcareZone.WARD
        assert '常规' in task.notes or 'routine' in task.notes.lower()

    def test_handle_call_emergency(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        task = handler.handle_call(
            call_id="call_002",
            patient_id="P54321",
            ward_zone=HealthcareZone.ICU,
            priority=PatientCallPriority.emergency,
        )
        assert task is not None
        # urgent_injection template gives MEDIUM even for emergency calls
        assert task.priority == HealthcareRiskLevel.MEDIUM

    def test_handle_call_duplicate(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        handler.handle_call(
            call_id="call_003",
            patient_id="P11111",
            ward_zone=HealthcareZone.WARD,
            priority=PatientCallPriority.urgent,
        )
        # Duplicate call_id
        result = handler.handle_call(
            call_id="call_003",
            patient_id="P22222",
            ward_zone=HealthcareZone.ICU,
            priority=PatientCallPriority.emergency,
        )
        assert result is None

    def test_complete_call(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        handler.handle_call(
            call_id="call_004",
            patient_id="P44444",
            ward_zone=HealthcareZone.WARD,
            priority=PatientCallPriority.routine,
        )
        handler.complete_call("call_004")
        assert "call_004" not in handler._active_calls
        assert len(handler._call_history) == 1

    def test_get_active_calls(self):
        lib = HealthcareTaskLibrary()
        handler = PatientCallHandler(lib)
        handler.handle_call("c1", "P1", HealthcareZone.WARD, PatientCallPriority.routine)
        handler.handle_call("c2", "P2", HealthcareZone.ICU, PatientCallPriority.emergency)
        calls = handler.get_active_calls()
        assert len(calls) == 2
        # Emergency should be first (highest priority)
        assert calls[0]['call_id'] == "c2"


class TestMedicationDeliveryPlanner:
    """药品配送规划测试"""

    def test_planner_initialization(self):
        monitor = InfectionControlMonitor()
        planner = MedicationDeliveryPlanner(monitor)
        assert planner._infection_monitor is monitor

    def test_plan_delivery_simple(self):
        monitor = InfectionControlMonitor()
        planner = MedicationDeliveryPlanner(monitor)
        task = HealthcareTask(
            task_id="med1", task_type="medication_delivery",
            priority=HealthcareRiskLevel.LOW,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.WARD,
            payload_type="medication", payload_id="MED001",
        )
        result = planner.plan_delivery(task, [{'agv_id': 'AGV_01', 'distance_to_pharmacy': 10}])
        assert result is not None
        assert result['status'] == 'planned'

    def test_plan_delivery_controlled_requires_cert(self):
        monitor = InfectionControlMonitor()
        planner = MedicationDeliveryPlanner(monitor)
        task = HealthcareTask(
            task_id="med2", task_type="medication_delivery",
            priority=HealthcareRiskLevel.HIGH,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.OPERATING_ROOM,
            payload_type="medication", payload_id="MED002",
            requires_controlled_access=True,
        )
        # No certified AGV available
        result = planner.plan_delivery(task, [{'agv_id': 'AGV_01', 'distance_to_pharmacy': 10}])
        assert result is None  # No suitable AGV

    def test_plan_delivery_controlled_with_cert(self):
        monitor = InfectionControlMonitor()
        # Record decontamination so OR is accessible
        monitor.record_decontamination(HealthcareZone.OPERATING_ROOM)
        planner = MedicationDeliveryPlanner(monitor)
        task = HealthcareTask(
            task_id="med3", task_type="medication_delivery",
            priority=HealthcareRiskLevel.HIGH,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.OPERATING_ROOM,
            payload_type="medication", payload_id="MED003",
            requires_controlled_access=True,
        )
        result = planner.plan_delivery(
            task,
            [{'agv_id': 'AGV_01', 'distance_to_pharmacy': 10, 'controlled_drug_certified': True}]
        )
        assert result is not None
        assert result['status'] == 'planned'
        assert result['assigned_agv'] == 'AGV_01'


class TestSpecimenTransportManager:
    """标本运输管理测试"""

    def test_manager_initialization(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        assert len(manager._specimen_tracking) == 0

    def test_register_specimen_blood(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        task = manager.register_specimen(
            specimen_id="SPEC_B001",
            category=SpecimenCategory.BLOOD,
            source=HealthcareZone.WARD,
        )
        assert task.task_type == 'specimen_transport'
        assert task.priority == HealthcareRiskLevel.LOW
        assert task.destination_zone == HealthcareZone.LABORATORY
        assert task.requires_sterile is True

    def test_register_specimen_pathology(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        task = manager.register_specimen(
            specimen_id="SPEC_P001",
            category=SpecimenCategory.PATHOLOGY,
            source=HealthcareZone.OPERATING_ROOM,
        )
        assert task.task_type == 'specimen_transport'
        # Pathology has lower time limit (900s)
        assert task.time_constraint == 900

    def test_track_specimen(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        manager.register_specimen("SPEC_T001", SpecimenCategory.BLOOD, HealthcareZone.WARD)
        tracked = manager.track_specimen("SPEC_T001")
        assert tracked is not None
        assert tracked['category'] == SpecimenCategory.BLOOD
        assert 'container' in tracked
        assert tracked['container'] == 'vacutainer'

    def test_track_nonexistent_specimen(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        assert manager.track_specimen("NONEXISTENT") is None

    def test_verify_chain_of_custody_valid(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        manager.register_specimen("SPEC_V001", SpecimenCategory.BLOOD, HealthcareZone.WARD)
        result = manager.verify_chain_of_custody("SPEC_V001")
        assert result['valid'] is True
        assert result['specimen_id'] == "SPEC_V001"

    def test_verify_chain_of_custody_not_found(self):
        monitor = InfectionControlMonitor()
        manager = SpecimenTransportManager(monitor)
        result = manager.verify_chain_of_custody("NONEXISTENT")
        assert result['valid'] is False
        assert result['reason'] == "Specimen not found"

    def test_transport_container_mapping(self):
        manager = SpecimenTransportManager(InfectionControlMonitor())
        assert manager.TRANSPORT_CONTAINERS[SpecimenCategory.BLOOD] == 'vacutainer'
        assert manager.TRANSPORT_CONTAINERS[SpecimenCategory.TISSUE] == 'formalin_container'
        assert manager.TRANSPORT_CONTAINERS[SpecimenCategory.PATHOLOGY] == 'histology cassette'


class TestHealthcareSceneController:
    """医疗场景总控制器集成测试"""

    def test_controller_initialization(self):
        controller = HealthcareSceneController(agv_grade="L")
        assert controller.agv_grade == "L"
        assert controller.task_library is not None
        assert controller.infection_monitor is not None

    def test_add_task(self):
        controller = HealthcareSceneController()
        task = HealthcareTask(
            task_id="ctrl1", task_type="medication_delivery",
            priority=HealthcareRiskLevel.HIGH,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.ICU,
            payload_type="medication", payload_id="MED001",
        )
        controller.add_task(task)
        status = controller.get_scene_status()
        assert status['pending_tasks'] == 1

    def test_task_priority_ordering(self):
        controller = HealthcareSceneController()
        low_task = HealthcareTask(
            task_id="c1", task_type="supply_delivery",
            priority=HealthcareRiskLevel.LOW,
            source_zone=HealthcareZone.CENTRAL_SUPPLY,
            destination_zone=HealthcareZone.WARD,
            payload_type="supplies", payload_id="S1",
        )
        crit_task = HealthcareTask(
            task_id="c2", task_type="medication_delivery",
            priority=HealthcareRiskLevel.CRITICAL,
            source_zone=HealthcareZone.PHARMACY,
            destination_zone=HealthcareZone.ICU,
            payload_type="medication", payload_id="M1",
        )
        controller.add_task(low_task)
        controller.add_task(crit_task)
        # Critical should be first (sorted descending by priority.value)
        next_task = controller.get_next_task()
        assert next_task.task_id == "c2"

    def test_get_next_task_empty(self):
        controller = HealthcareSceneController()
        assert controller.get_next_task() is None

    def test_scene_status(self):
        controller = HealthcareSceneController(agv_grade="XL")
        status = controller.get_scene_status()
        assert status['agv_grade'] == "XL"
        assert status['pending_tasks'] == 0

    def test_scene_report_structure(self):
        controller = HealthcareSceneController()
        report = controller.generate_scene_report()
        assert 'timestamp' in report
        assert 'total_deliveries' in report
        assert 'on_time_rate' in report
        assert 'task_breakdown' in report
        assert 'zone_status' in report
        assert 'risk_assessment' in report

    def test_scene_report_empty(self):
        controller = HealthcareSceneController()
        report = controller.generate_scene_report()
        assert report['total_deliveries'] == 0
        assert report['on_time_rate'] == 0

    def test_zone_status(self):
        controller = HealthcareSceneController()
        status = controller.get_scene_status()
        assert 'infection_alerts' in status

    def test_full_workflow(self):
        """完整工作流测试"""
        controller = HealthcareSceneController(agv_grade="L")
        monitor = controller.infection_monitor

        # Record decontamination
        monitor.record_decontamination(HealthcareZone.OPERATING_ROOM)

        # Add isolation alert
        monitor.add_isolation_alert(HealthcareZone.ISOLATION, "Test alert")

        # Handle patient call
        task = controller.call_handler.handle_call(
            call_id="wf1", patient_id="P999",
            ward_zone=HealthcareZone.WARD, priority=PatientCallPriority.urgent,
        )
        assert task is not None

        # Register specimen
        spec_task = controller.specimen_manager.register_specimen(
            "WF_SPEC_1", SpecimenCategory.BLOOD, HealthcareZone.WARD,
        )
        assert spec_task is not None

        # Status check
        status = controller.get_scene_status()
        assert status['infection_alerts'] == 1

        # Report
        report = controller.generate_scene_report()
        assert 'risk_assessment' in report

    def test_all_agv_grades(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            c = HealthcareSceneController(agv_grade=grade)
            assert c.agv_grade == grade


class TestGlobalSingleton:
    """全局单例测试"""

    def test_singleton_pattern(self):
        import src.embodied.healthcare_scene as hs
        hs._healthcare_controller = None

        c1 = get_healthcare_scene_controller(agv_grade="M")
        assert c1.agv_grade == "M"
        c2 = get_healthcare_scene_controller()
        assert c1 is c2  # Same instance
