"""
test_industrial_scene.py - 工业场景模块完整测试
ProductionLineController 工业AGV场景: 装配/加工/焊接/喷涂/包装/柔性制造
"""
import pytest
import time
from src.embodied.industrial_scene import (
    ProductionLineType,
    WorkstationType,
    MaterialType,
    QualityGrade,
    ToolType,
    ProductionTask,
    QualityInspectionStation,
    PredictiveMaintenanceMonitor,
    ToolManagementSystem,
    SafetyMonitoringSystem,
    MaterialFlowCoordinator,
    ProductionLineController,
)


class TestIndustrialEnums:
    """工业场景枚举完整性测试"""

    def test_production_line_types(self):
        assert ProductionLineType.ASSEMBLY
        assert ProductionLineType.MACHINING
        assert ProductionLineType.WELDING
        assert ProductionLineType.PAINTING
        assert ProductionLineType.PACKAGING
        assert ProductionLineType.FLEXIBLE
        assert ProductionLineType.AUTOMATED_WAREHOUSE
        assert len(ProductionLineType) == 7

    def test_workstation_types(self):
        assert WorkstationType.CNC_MACHINE
        assert WorkstationType.ROBOT_CELL
        assert WorkstationType.ASSEMBLY_STATION
        assert WorkstationType.INSPECTION_STATION
        assert WorkstationType.PACKING_STATION
        assert WorkstationType.LOADING_STATION
        assert WorkstationType.QUALITY_GATE
        assert len(WorkstationType) == 7

    def test_material_types(self):
        assert MaterialType.RAW_METAL
        assert MaterialType.RAW_PLASTIC
        assert MaterialType.COMPONENT
        assert MaterialType.SUBASSEMBLY
        assert MaterialType.FINISHED_GOOD
        assert MaterialType.PACKAGING
        assert MaterialType.HAZARDOUS
        assert len(MaterialType) == 7

    def test_quality_grades(self):
        assert QualityGrade.A_PRIME
        assert QualityGrade.A_STANDARD
        assert QualityGrade.B_REWORK
        assert QualityGrade.C_REJECT
        assert QualityGrade.UNKNOWN
        assert len(QualityGrade) == 5

    def test_tool_types(self):
        assert ToolType.END_MILL
        assert ToolType.DRILL_BIT
        assert ToolType.TAP
        assert ToolType.INSERT
        assert ToolType.CALIPER
        assert ToolType.GAUGE
        assert ToolType.WRENCH
        assert ToolType.FIXTURE
        assert len(ToolType) == 8


class TestProductionTask:
    """生产任务创建和属性测试"""

    def test_task_creation_minimal(self):
        task = ProductionTask(
            task_id="p1",
            task_type="material_supply",
            priority=5,
            source_station=WorkstationType.LOADING_STATION,
            destination_station=WorkstationType.CNC_MACHINE,
        )
        assert task.task_id == "p1"
        assert task.status == "pending"
        assert task.priority == 5

    def test_task_creation_full(self):
        task = ProductionTask(
            task_id="p2",
            task_type="part_transfer",
            priority=9,
            source_station=WorkstationType.ROBOT_CELL,
            destination_station=WorkstationType.ASSEMBLY_STATION,
            material_type=MaterialType.COMPONENT,
            material_id="COMP_001",
            quantity=10,
            cycle_time_target=30.0,
            quality_required=True,
        )
        assert task.material_type == MaterialType.COMPONENT
        assert task.quantity == 10
        assert task.cycle_time_target == 30.0
        assert task.quality_required is True

    def test_task_status_transitions(self):
        task = ProductionTask(
            task_id="p3",
            task_type="quality_check",
            priority=3,
            source_station=WorkstationType.INSPECTION_STATION,
            destination_station=WorkstationType.QUALITY_GATE,
        )
        task.status = "in_progress"
        assert task.status == "in_progress"
        task.completed_at = time.time()
        task.status = "completed"
        assert task.completed_at is not None


class TestQualityInspectionStation:
    """质量检测工位测试"""

    def test_station_initialization(self):
        station = QualityInspectionStation("QI_01")
        assert station.station_id == "QI_01"
        assert station._inspection_count == 0
        assert station._defect_count == 0

    def test_perform_inspection_diameter_pass(self):
        station = QualityInspectionStation("QI_01")
        result = station.perform_inspection(
            part_id="PART_001",
            inspection_type="diameter",
            measured_value=0.01,  # well within 0.05mm tolerance
        )
        assert result['result'] == 'pass'
        assert result['grade'] == QualityGrade.A_PRIME
        assert result['part_id'] == "PART_001"

    def test_perform_inspection_diameter_fail_rework(self):
        station = QualityInspectionStation("QI_01")
        result = station.perform_inspection(
            part_id="PART_002",
            inspection_type="diameter",
            measured_value=0.12,  # > 0.1 (tolerance_g) but < 0.2 → rework
        )
        assert result['result'] == 'rework'
        assert result['grade'] == QualityGrade.B_REWORK

    def test_perform_inspection_diameter_reject(self):
        station = QualityInspectionStation("QI_01")
        result = station.perform_inspection(
            part_id="PART_003",
            inspection_type="diameter",
            measured_value=0.25,  # >= 0.2 (= 0.1*2) → reject
        )
        assert result['result'] == 'reject'
        assert result['grade'] == QualityGrade.C_REJECT

    def test_perform_inspection_unknown_type(self):
        station = QualityInspectionStation("QI_01")
        result = station.perform_inspection(
            part_id="PART_004",
            inspection_type="unknown_type",
            measured_value=1.0,
        )
        assert result['status'] == 'unknown_type'

    def test_station_stats(self):
        station = QualityInspectionStation("QI_01")
        station.perform_inspection("P1", "diameter", 0.01)
        station.perform_inspection("P2", "diameter", 0.12)  # rework (0.12 > 0.1 but < 0.2)
        stats = station.get_station_stats()
        assert stats['total_inspections'] == 2
        assert stats['defect_count'] == 1
        assert stats['rework_count'] == 1
        assert 0 <= stats['first_pass_rate'] <= 1


class TestPredictiveMaintenanceMonitor:
    """预测性维护监控器测试"""

    def test_monitor_initialization(self):
        monitor = PredictiveMaintenanceMonitor()
        assert len(monitor._equipment_states) == 0
        assert len(monitor._alert_history) == 0

    def test_register_equipment(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment("CNC_01", "cnc", {"model": "VMC850"})
        assert "CNC_01" in monitor._equipment_states
        assert monitor._equipment_states["CNC_01"]['type'] == "cnc"

    def test_update_telemetry_normal(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment("CNC_01", "cnc", {})
        alerts = monitor.update_telemetry("CNC_01", {'temperature_c': 50.0})
        assert alerts == []

    def test_update_telemetry_warning(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment("CNC_01", "cnc", {})
        alerts = monitor.update_telemetry("CNC_01", {'temperature_c': 75.0})
        assert len(alerts) == 1
        assert alerts[0]['level'] == 'warning'

    def test_update_telemetry_critical(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment("CNC_01", "cnc", {})
        alerts = monitor.update_telemetry("CNC_01", {'temperature_c': 90.0})
        assert len(alerts) == 1
        assert alerts[0]['level'] == 'critical'

    def test_update_telemetry_unknown_equipment(self):
        monitor = PredictiveMaintenanceMonitor()
        alerts = monitor.update_telemetry("UNKNOWN", {'temperature_c': 90.0})
        assert alerts == []

    def test_predict_maintenance_window(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment("CNC_01", "cnc", {})
        monitor._equipment_states["CNC_01"]["runtime_hours"] = 3000
        pred = monitor.predict_maintenance_window("CNC_01")
        assert pred['remaining_hours'] == 1000
        assert pred['predicted_failure_hours'] == 4000

    def test_predict_maintenance_window_not_found(self):
        monitor = PredictiveMaintenanceMonitor()
        assert monitor.predict_maintenance_window("NONEXISTENT") is None

    def test_overall_health_score_no_equipment(self):
        monitor = PredictiveMaintenanceMonitor()
        score = monitor.get_overall_health_score()
        assert score['health_score'] == 100
        assert score['status'] == 'no_equipment'

    def test_overall_health_score_healthy(self):
        monitor = PredictiveMaintenanceMonitor()
        monitor.register_equipment("E1", "type1", {})
        monitor.register_equipment("E2", "type1", {})
        score = monitor.get_overall_health_score()
        assert score['total_equipment'] == 2
        assert score['healthy_count'] == 2


class TestToolManagementSystem:
    """工具管理系统测试"""

    def test_system_initialization(self):
        system = ToolManagementSystem()
        assert len(system._tool_inventory) == 0

    def test_register_tool(self):
        system = ToolManagementSystem()
        system.register_tool("T001", ToolType.END_MILL, {"diameter": 10.0})
        assert "T001" in system._tool_inventory
        assert system._tool_inventory["T001"]['type'] == ToolType.END_MILL
        assert system._tool_inventory["T001"]['status'] == 'available'

    def test_install_tool(self):
        system = ToolManagementSystem()
        system.register_tool("T002", ToolType.DRILL_BIT, {})
        result = system.install_tool("T002", WorkstationType.CNC_MACHINE)
        assert result is True
        assert system._tool_inventory["T002"]['status'] == 'installed'

    def test_install_unknown_tool(self):
        system = ToolManagementSystem()
        result = system.install_tool("UNKNOWN", WorkstationType.CNC_MACHINE)
        assert result is False

    def test_install_already_used_tool(self):
        system = ToolManagementSystem()
        system.register_tool("T003", ToolType.TAP, {})
        system.install_tool("T003", WorkstationType.CNC_MACHINE)
        # Try to install again (should fail since already installed)
        result = system.install_tool("T003", WorkstationType.ROBOT_CELL)
        assert result is False

    def test_record_tool_usage(self):
        system = ToolManagementSystem()
        system.register_tool("T004", ToolType.END_MILL, {})
        system.install_tool("T004", WorkstationType.CNC_MACHINE)
        result = system.record_tool_usage("T004", hours=10.0, parts=50)
        assert result['status'] == 'installed'
        assert result['needs_replacement'] is False

    def test_record_tool_usage_unknown(self):
        system = ToolManagementSystem()
        result = system.record_tool_usage("UNKNOWN", hours=10.0)
        assert result['status'] == 'unknown_tool'

    def test_get_tool_status(self):
        system = ToolManagementSystem()
        system.register_tool("T005", ToolType.INSERT, {})
        system.install_tool("T005", WorkstationType.CNC_MACHINE)
        system.record_tool_usage("T005", hours=50.0, parts=200)
        status = system.get_tool_status("T005")
        assert status['tool_id'] == "T005"
        assert status['type'] == 'INSERT'
        assert status['status'] == 'installed'

    def test_get_tool_status_unknown(self):
        system = ToolManagementSystem()
        assert system.get_tool_status("NONEXISTENT") is None


class TestSafetyMonitoringSystem:
    """工业安全监控系统测试"""

    def test_system_initialization(self):
        system = SafetyMonitoringSystem()
        assert len(system._personnel_positions) == 0
        assert system._safety_stop_active is False

    def test_register_person(self):
        system = SafetyMonitoringSystem()
        system.register_person("W001", (1.0, 2.0))
        assert "W001" in system._personnel_positions
        assert system._personnel_positions["W001"] == (1.0, 2.0)

    def test_check_zone_entry_safe(self):
        system = SafetyMonitoringSystem()
        result = system.check_zone_entry("AGV_01", (0.0, 0.0), "assembly_line")
        assert result['allowed'] is True
        assert result['risk'] == 'low'

    def test_check_zone_entry_welding_critical(self):
        system = SafetyMonitoringSystem()
        # Register person near welding area
        system.register_person("W001", (1.0, 1.0))
        result = system.check_zone_entry("AGV_01", (1.5, 1.5), "welding_area")
        # Person is within 2m, welding is critical → blocked
        assert result['allowed'] is False
        assert result['risk'] == 'critical'

    def test_check_zone_entry_unknown_zone(self):
        system = SafetyMonitoringSystem()
        result = system.check_zone_entry("AGV_01", (0, 0), "unknown_zone")
        assert result['allowed'] is True

    def test_trigger_emergency_stop(self):
        system = SafetyMonitoringSystem()
        system.trigger_emergency_stop("Fire detected")
        assert system._safety_stop_active is True
        assert len(system._incident_log) == 1
        assert system._incident_log[0]['type'] == 'emergency_stop'

    def test_reset_emergency_stop(self):
        system = SafetyMonitoringSystem()
        system.trigger_emergency_stop("Test stop")
        system.reset_emergency_stop()
        assert system._safety_stop_active is False

    def test_get_safety_status(self):
        system = SafetyMonitoringSystem()
        system.register_person("W001", (1.0, 2.0))
        status = system.get_safety_status()
        assert status['active_personnel'] == 1
        assert 'emergency_stop_active' in status
        assert 'recent_incidents' in status


class TestMaterialFlowCoordinator:
    """物料流协调器测试"""

    def test_coordinator_initialization(self):
        coordinator = MaterialFlowCoordinator(ProductionLineType.FLEXIBLE)
        assert coordinator.line_type == ProductionLineType.FLEXIBLE
        assert len(coordinator._material_buffers) == len(WorkstationType)

    def test_request_material_with_buffer(self):
        coordinator = MaterialFlowCoordinator(ProductionLineType.ASSEMBLY)
        task = ProductionTask(
            task_id="m1", task_type="material_supply",
            priority=5, source_station=WorkstationType.LOADING_STATION,
            destination_station=WorkstationType.ASSEMBLY_STATION,
        )
        # Initially buffer is empty, request fails
        result = coordinator.request_material(task, WorkstationType.ASSEMBLY_STATION)
        assert result is False

    def test_allocate_transfer(self):
        coordinator = MaterialFlowCoordinator(ProductionLineType.MACHINING)
        task = ProductionTask(
            task_id="m2", task_type="part_transfer",
            priority=7, source_station=WorkstationType.LOADING_STATION,
            destination_station=WorkstationType.CNC_MACHINE,
        )
        result = coordinator.allocate_transfer(task, "AGV_01")
        assert result['task_id'] == "m2"
        assert result['assigned_agv'] == "AGV_01"
        assert 'estimated_time' in result
        assert 'route' in result

    def test_complete_transfer(self):
        coordinator = MaterialFlowCoordinator(ProductionLineType.MACHINING)
        task = ProductionTask(
            task_id="m3", task_type="material_supply",
            priority=5, source_station=WorkstationType.LOADING_STATION,
            destination_station=WorkstationType.CNC_MACHINE,
        )
        coordinator.allocate_transfer(task, "AGV_02")
        result = coordinator.complete_transfer("m3")
        assert result is True
        assert task.status == "completed"

    def test_complete_transfer_not_found(self):
        coordinator = MaterialFlowCoordinator(ProductionLineType.MACHINING)
        result = coordinator.complete_transfer("NONEXISTENT")
        assert result is False

    def test_get_buffer_status(self):
        coordinator = MaterialFlowCoordinator(ProductionLineType.ASSEMBLY)
        status = coordinator.get_buffer_status(WorkstationType.CNC_MACHINE)
        assert status['station'] == 'CNC_MACHINE'
        assert 'buffer_size' in status
        assert 'urgent_count' in status


class TestProductionLineController:
    """生产线总控制器集成测试"""

    def test_controller_initialization(self):
        controller = ProductionLineController(
            line_type=ProductionLineType.ASSEMBLY,
            grade="XL",
        )
        assert controller.line_type == ProductionLineType.ASSEMBLY
        assert controller.agv_grade == "XL"
        assert controller.quality_station is not None
        assert controller.maintenance_monitor is not None
        assert controller.tool_manager is not None
        assert controller.safety_monitor is not None
        assert controller.material_coordinator is not None

    def test_create_production_task(self):
        controller = ProductionLineController()
        task = controller.create_production_task(
            task_type="material_supply",
            source=WorkstationType.LOADING_STATION,
            destination=WorkstationType.CNC_MACHINE,
            priority=7,
            material_id="MAT_001",
            material_type=MaterialType.RAW_METAL,
        )
        assert task.task_type == "material_supply"
        assert task.priority == 7
        assert task.material_id == "MAT_001"
        assert task.material_type == MaterialType.RAW_METAL
        assert task.task_id == "prod_00001"

    def test_multiple_tasks_increment_id(self):
        controller = ProductionLineController()
        t1 = controller.create_production_task("type1", WorkstationType.LOADING_STATION, WorkstationType.CNC_MACHINE)
        t2 = controller.create_production_task("type2", WorkstationType.LOADING_STATION, WorkstationType.ASSEMBLY_STATION)
        t3 = controller.create_production_task("type3", WorkstationType.LOADING_STATION, WorkstationType.ROBOT_CELL)
        assert "00001" in t1.task_id
        assert "00002" in t2.task_id
        assert "00003" in t3.task_id

    def test_full_quality_inspection_workflow(self):
        """完整质量检测工作流"""
        controller = ProductionLineController()
        station = controller.quality_station

        # 检测合格
        r1 = station.perform_inspection("PART_A", "diameter", 0.01)
        assert r1['result'] == 'pass'

        # 检测不合格需返工
        r2 = station.perform_inspection("PART_B", "diameter", 0.12)  # rework
        assert r2['result'] == 'rework'

        stats = station.get_station_stats()
        assert stats['total_inspections'] == 2
        assert stats['defect_count'] == 1

    def test_full_maintenance_workflow(self):
        """完整维护工作流"""
        controller = ProductionLineController()
        monitor = controller.maintenance_monitor

        # 注册设备
        monitor.register_equipment("CNC_01", "cnc", {"model": "VMC850"})
        
        # 正常遥测
        alerts = monitor.update_telemetry("CNC_01", {'temperature_c': 50.0, 'runtime_hours': 1000})
        assert len(alerts) == 0
        
        # 警告遥测
        alerts = monitor.update_telemetry("CNC_01", {'temperature_c': 75.0})
        assert len(alerts) == 1
        
        # 健康评分
        score = monitor.get_overall_health_score()
        assert score['total_equipment'] == 1
        assert score['warning_count'] == 1

    def test_full_tool_workflow(self):
        """完整工具管理工作流"""
        controller = ProductionLineController()
        tools = controller.tool_manager

        # 注册并安装
        tools.register_tool("TOOL_01", ToolType.END_MILL, {"diameter": 8.0})
        tools.install_tool("TOOL_01", WorkstationType.CNC_MACHINE)
        
        # 记录使用
        result = tools.record_tool_usage("TOOL_01", hours=10.0, parts=100)
        assert result['needs_replacement'] is False
        
        # 状态查询
        status = tools.get_tool_status("TOOL_01")
        assert status['status'] == 'installed'

    def test_full_safety_workflow(self):
        """完整安全工作流"""
        controller = ProductionLineController()
        safety = controller.safety_monitor

        # 注册人员
        safety.register_person("W001", (1.0, 1.0))
        
        # 检查区域
        result = safety.check_zone_entry("AGV_01", (1.5, 1.5), "welding_area")
        assert result['allowed'] is False  # 有人，焊接区危险
        
        # 紧急停止
        safety.trigger_emergency_stop("Manual stop")
        assert safety._safety_stop_active is True
        
        # 复位
        safety.reset_emergency_stop()
        assert safety._safety_stop_active is False

    def test_full_material_flow_workflow(self):
        """完整物料流工作流"""
        controller = ProductionLineController()
        flow = controller.material_coordinator

        task = controller.create_production_task(
            "part_transfer",
            WorkstationType.LOADING_STATION,
            WorkstationType.ASSEMBLY_STATION,
            priority=8,
        )
        
        result = flow.allocate_transfer(task, "AGV_01")
        assert result['assigned_agv'] == "AGV_01"
        
        completed = flow.complete_transfer(task.task_id)
        assert completed is True

    def test_all_agv_grades(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            c = ProductionLineController(grade=grade)
            assert c.agv_grade == grade

    def test_all_production_line_types(self):
        for lt in ProductionLineType:
            c = ProductionLineController(line_type=lt)
            assert c.line_type == lt
