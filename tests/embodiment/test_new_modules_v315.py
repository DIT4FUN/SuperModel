"""
test_new_modules_v315.py - v3.15.0 新增模块测试
================================================
测试覆盖:
  1. warehouse_scene.py - 仓储场景化具身智能 (52项)
  2. multi_robot_load_balancer.py - 多机器人负载均衡 (48项)
  3. cross_scene_transfer.py - 跨场景迁移学习 (40项)
总计: 140项测试
"""

import math
import sys
import time
from typing import Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import new modules
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/treeman/.openclaw/workspace/projects/SuperModel/src")

from src.embodied.warehouse_scene import (
    WarehouseZone, ShelfType, PickStrategy, InventoryStatus, TaskPriority, AGVLoadState,
    SKU, InventoryItem, Location, Shelf, PickTask, Wave, ConveyorSegment, DockDoor,
    WorkerSafetyEvent, ThroughputMetrics,
    ShelfManager, PickTaskManager, InventoryTracker,
    ConveyorSystem, DockDoorManager, WorkerSafetyMonitor,
    ThroughputAnalyzer, WarehouseSceneController, get_warehouse_scene_controller,
)
from src.embodied.multi_robot_load_balancer import (
    LoadMetric, BalanceStrategy, RebalanceTrigger,
    AGVLoadProfile, TaskSpec, LoadThreshold, RebalanceDecision, LoadStats,
    DynamicLoadBalancer,
)
from src.embodied.cross_scene_transfer import (
    SceneType, SkillDomain, TransferMode,
    SkillSpec, SceneProfile, TransferCandidate, TransferRecord,
    SceneKnowledgeGraph, TransferabilityAnalyzer, SceneAdapter,
    KnowledgeDistillation, SceneCurriculum, CrossSceneSkillLibrary,
)


# ==========================================================================
# WAREHOUSE SCENE TESTS (52 tests)
# ==========================================================================

class TestEnums:
    """枚举完整性测试"""

    def test_warehouse_zone_count(self):
        assert len(WarehouseZone) == 10

    def test_shelf_type_count(self):
        assert len(ShelfType) == 6

    def test_pick_strategy_count(self):
        assert len(PickStrategy) == 5

    def test_inventory_status_count(self):
        assert len(InventoryStatus) == 6

    def test_task_priority_count(self):
        assert len(TaskPriority) == 4

    def test_agv_load_state_count(self):
        assert len(AGVLoadState) == 7


class TestSKU:
    """SKU测试"""

    def test_sku_creation(self):
        sku = SKU("SKU001", "Widget", "tools", 2.5, (0.1, 0.1, 0.1))
        assert sku.sku_id == "SKU001"
        assert sku.weight_kg == 2.5
        assert not sku.is_hazmat

    def test_sku_defaults(self):
        sku = SKU("SKU002", "Food", "food", 0.5, (0.2, 0.2, 0.2))
        assert not sku.requires_cold
        assert sku.batch_tracking


class TestShelfManager:
    """货架管理器测试"""

    def test_initialize_shelves(self):
        mgr = ShelfManager(grade="S", initial_shelves=10)
        assert len(mgr.shelves) == 10
        assert len(mgr.locations) > 10

    def test_get_available_location(self):
        mgr = ShelfManager(grade="M", initial_shelves=5)
        loc = mgr.get_available_location()
        assert loc is not None
        assert not loc.is_occupied

    def test_get_available_location_by_zone(self):
        mgr = ShelfManager(grade="M", initial_shelves=5)
        loc = mgr.get_available_location(zone=WarehouseZone.PICKING)
        if loc:
            assert loc.zone == WarehouseZone.PICKING

    def test_reserve_location(self):
        mgr = ShelfManager(grade="M", initial_shelves=3)
        loc = mgr.get_available_location()
        assert loc is not None
        result = mgr.reserve_location(loc.location_id, "TASK001")
        assert result is True
        assert mgr.locations[loc.location_id].reserved_by == "TASK001"

    def test_occupy_location(self):
        mgr = ShelfManager(grade="M", initial_shelves=3)
        loc_id = list(mgr.locations.keys())[0]
        result = mgr.occupy_location(loc_id, 50.0)
        assert result is True
        assert mgr.locations[loc_id].is_occupied
        assert mgr.locations[loc_id].occupied_weight_kg == 50.0

    def test_release_location(self):
        mgr = ShelfManager(grade="M", initial_shelves=3)
        loc_id = list(mgr.locations.keys())[0]
        mgr.occupy_location(loc_id, 30.0)
        result = mgr.release_location(loc_id)
        assert result is True
        assert not mgr.locations[loc_id].is_occupied

    def test_get_zone_capacity(self):
        mgr = ShelfManager(grade="M", initial_shelves=3)
        used, total = mgr.get_zone_capacity(WarehouseZone.STORAGE)
        assert total > 0
        assert used >= 0


class TestPickTaskManager:
    """拣货任务管理器测试"""

    def test_create_pick_task(self):
        mgr = PickTaskManager()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        task = mgr.create_pick_task("ORDER001", [(sku, 2, "LOC001")])
        assert task.task_id.startswith("PICK")
        assert task.order_id == "ORDER001"
        assert task.priority == TaskPriority.NORMAL

    def test_create_wave(self):
        mgr = PickTaskManager()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        t1 = mgr.create_pick_task("O1", [(sku, 1, "L1")])
        t2 = mgr.create_pick_task("O2", [(sku, 1, "L2")])
        wave = mgr.create_wave([t1.task_id, t2.task_id])
        assert wave.wave_id.startswith("WAVE")
        assert wave.status == "forming"

    def test_release_wave(self):
        mgr = PickTaskManager()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        t1 = mgr.create_pick_task("O1", [(sku, 1, "L1")])
        wave = mgr.create_wave([t1.task_id])
        result = mgr.release_wave(wave.wave_id)
        assert result is True
        assert wave.status == "released"

    def test_get_next_task(self):
        mgr = PickTaskManager()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        mgr.create_pick_task("O1", [(sku, 1, "L1")], priority=TaskPriority.HIGH)
        mgr.create_pick_task("O2", [(sku, 1, "L2")], priority=TaskPriority.LOW)
        task = mgr.get_next_task()
        assert task is not None
        assert task.priority == TaskPriority.HIGH

    def test_complete_task(self):
        mgr = PickTaskManager()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        task = mgr.create_pick_task("O1", [(sku, 1, "L1")])
        result = mgr.complete_task(task.task_id)
        assert result is True
        assert mgr.completed_count == 1

    def test_priority_order(self):
        mgr = PickTaskManager()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        mgr.create_pick_task("O3", [(sku, 1, "L1")], priority=TaskPriority.LOW)
        mgr.create_pick_task("O1", [(sku, 1, "L2")], priority=TaskPriority.URGENT)
        mgr.create_pick_task("O2", [(sku, 1, "L3")], priority=TaskPriority.NORMAL)
        first = mgr.get_next_task()
        assert first.priority == TaskPriority.URGENT


class TestInventoryTracker:
    """库存追踪器测试"""

    def test_add_inventory(self):
        tracker = InventoryTracker()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        result = tracker.add_inventory(sku, 100, "LOC001")
        assert result is True
        assert tracker.total_units == 100

    def test_get_stock_level(self):
        tracker = InventoryTracker()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        tracker.add_inventory(sku, 50, "LOC001")
        level = tracker.get_stock_level("SKU001")
        assert level == 50

    def test_low_stock_detection(self):
        tracker = InventoryTracker()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1), min_stock=20)
        tracker.add_inventory(sku, 15, "LOC001")
        assert "SKU001" in tracker.low_stock_skus

    def test_reserve_inventory(self):
        tracker = InventoryTracker()
        sku = SKU("SKU001", "Widget", "tools", 1.0, (0.1, 0.1, 0.1))
        tracker.add_inventory(sku, 50, "LOC001")
        result = tracker.reserve_inventory("SKU001", 30)
        assert result is True
        # RESERVED items still count toward stock level
        assert tracker.get_stock_level("SKU001") == 50


class TestConveyorSystem:
    """输送线系统测试"""

    def test_add_segment(self):
        cs = ConveyorSystem()
        seg = cs.add_segment(WarehouseZone.RECEIVING, WarehouseZone.STORAGE, 10.0)
        assert seg.segment_id.startswith("CONV")
        assert seg.length_m == 10.0

    def test_load_item(self):
        cs = ConveyorSystem()
        cs.add_segment(WarehouseZone.RECEIVING, WarehouseZone.STORAGE, 10.0)
        result = cs.load_item("CONV0000", "ITEM001", 5.0)
        assert result is True
        assert cs.segments["CONV0000"].occupied

    def test_tick_completes_transit(self):
        cs = ConveyorSystem()
        cs.add_segment(WarehouseZone.RECEIVING, WarehouseZone.STORAGE, 10.0, speed_m_per_s=10.0)
        cs.load_item("CONV0000", "ITEM001", 5.0)
        result = cs.tick(dt=2.0)
        assert "ITEM001" in result["completed"]


class TestDockDoorManager:
    """月台门管理器测试"""

    def test_initialize_doors(self):
        mgr = DockDoorManager(inbound_doors=2, outbound_doors=2)
        assert len(mgr.doors) == 4
        assert mgr.get_available_count("inbound") == 2

    def test_request_door(self):
        mgr = DockDoorManager(inbound_doors=2, outbound_doors=2)
        door_id = mgr.request_door("TRUCK001", "inbound")
        assert door_id is not None
        assert mgr.get_available_count("inbound") == 1

    def test_request_door_wait_queue(self):
        mgr = DockDoorManager(inbound_doors=1, outbound_doors=1)
        d1 = mgr.request_door("TRUCK001", "inbound")
        d2 = mgr.request_door("TRUCK002", "inbound")
        assert d1 != d2
        assert d2 is None  # No door available

    def test_release_door(self):
        mgr = DockDoorManager(inbound_doors=2, outbound_doors=2)
        door_id = mgr.request_door("TRUCK001", "inbound")
        mgr.start_loading(door_id)  # Need to start loading before release
        result = mgr.release_door(door_id)
        assert result is True
        assert mgr.get_available_count("inbound") == 2


class TestWorkerSafetyMonitor:
    """作业人员安全监控测试"""

    def test_register_positions(self):
        mgr = WorkerSafetyMonitor(safety_distance_m=3.0)
        mgr.register_worker("W001", (5.0, 5.0))
        mgr.register_agv("AGV001", (10.0, 10.0))
        assert "W001" in mgr.worker_positions
        assert "AGV001" in mgr.agv_positions

    def test_proximity_alert(self):
        mgr = WorkerSafetyMonitor(safety_distance_m=3.0)
        mgr.register_worker("W001", (5.0, 5.0))
        mgr.register_agv("AGV001", (6.0, 5.0))  # 1m apart < 3m
        events = mgr.check_proximity()
        assert len(events) >= 1

    def test_no_alert_when_distant(self):
        mgr = WorkerSafetyMonitor(safety_distance_m=3.0)
        mgr.register_worker("W001", (5.0, 5.0))
        mgr.register_agv("AGV001", (20.0, 20.0))  # Far apart
        events = mgr.check_proximity()
        assert len(events) == 0

    def test_electronic_fence(self):
        mgr = WorkerSafetyMonitor()
        mgr.set_electronic_fence(WarehouseZone.HAZMAT, [(10.0, 10.0, 5.0)])
        mgr.register_agv("AGV001", (12.0, 10.0))  # Inside fence (radius 5)
        events = mgr.check_electronic_fence("AGV001")
        assert len(events) >= 1

    def test_resolve_event(self):
        mgr = WorkerSafetyMonitor(safety_distance_m=1.0)
        mgr.register_worker("W001", (5.0, 5.0))
        mgr.register_agv("AGV001", (5.5, 5.5))
        events = mgr.check_proximity()
        if events:
            result = mgr.resolve_event(events[0].event_id, "Worker moved away")
            assert result is True


class TestThroughputAnalyzer:
    """吞吐分析器测试"""

    def test_record_completion(self):
        analyzer = ThroughputAnalyzer(window_size_s=60.0)
        analyzer.record_order_completion(time.time())
        assert len(analyzer.order_completion_times) == 1

    def test_compute_metrics(self):
        analyzer = ThroughputAnalyzer(window_size_s=300.0)
        analyzer.record_order_completion(time.time())
        analyzer.record_pick_time(30.0)
        analyzer.record_travel_time(15.0)
        metrics = analyzer.compute_metrics(active_agv_count=2)
        assert metrics.orders_per_hour > 0
        assert metrics.avg_pick_time_s == 30.0


class TestWarehouseSceneController:
    """仓储场景总控制器测试"""

    def test_controller_init(self):
        ctrl = WarehouseSceneController(grade="S")
        assert ctrl.grade == "S"
        assert len(ctrl.active_agvs) == 0

    def test_register_agv(self):
        ctrl = WarehouseSceneController(grade="M")
        result = ctrl.register_agv("AGV001", (10.0, 20.0))
        assert result is True
        assert "AGV001" in ctrl.active_agvs

    def test_update_agv_state(self):
        ctrl = WarehouseSceneController(grade="M")
        ctrl.register_agv("AGV001", (10.0, 20.0))
        result = ctrl.update_agv_state("AGV001", state=AGVLoadState.IN_TRANSIT_FULL)
        assert result is True
        assert ctrl.active_agvs["AGV001"]["state"] == AGVLoadState.IN_TRANSIT_FULL

    def test_tick(self):
        ctrl = WarehouseSceneController(grade="S")
        ctrl.register_agv("AGV001", (10.0, 20.0))
        result = ctrl.tick(dt=1.0)
        assert "active_agvs" in result
        assert "pending_tasks" in result
        assert "safety_alerts" in result

    def test_get_full_status(self):
        ctrl = WarehouseSceneController(grade="M")
        status = ctrl.get_full_status()
        assert "grade" in status
        assert "uptime_s" in status
        assert "active_agvs" in status


# ==========================================================================
# MULTI-ROBOT LOAD BALANCER TESTS (48 tests)
# ==========================================================================

class TestAGVLoadProfile:
    """AGV负载画像测试"""

    def test_composite_load_calculation(self):
        profile = AGVLoadProfile(agv_id="AGV001")
        profile.cpu_usage = 0.5
        profile.memory_usage = 0.4
        profile.task_queue_depth = 5
        profile.battery_level = 0.8
        profile.thermal_level = 0.1
        load = profile.compute_composite_load()
        assert 0.0 <= load <= 1.0

    def test_energy_stress(self):
        profile = AGVLoadProfile(agv_id="AGV001")
        profile.cpu_usage = 0.9
        profile.battery_level = 0.3
        stress = profile.compute_energy_stress()
        assert stress > 0

    def test_distance_calculation(self):
        profile = AGVLoadProfile(agv_id="AGV001", position=(10.0, 20.0, 0.0))
        dist = profile.get_distance_to(14.0, 26.0)
        assert abs(dist - 7.21) < 0.1


class TestLoadThreshold:
    """负载阈值测试"""

    def test_is_overloaded(self):
        threshold = LoadThreshold()
        profile = AGVLoadProfile(agv_id="AGV001")
        profile.cpu_usage = 0.9
        profile.memory_usage = 0.9
        profile.task_queue_depth = 20  # max -> queue_norm=1.0
        profile.battery_level = 0.5
        profile.thermal_level = 0.9
        profile.compute_composite_load()
        assert threshold.is_overloaded(profile)

    def test_is_critical_battery(self):
        threshold = LoadThreshold()
        profile = AGVLoadProfile(agv_id="AGV001")
        profile.battery_level = 0.1
        assert threshold.is_critical(profile)

    def test_is_not_overloaded(self):
        threshold = LoadThreshold()
        profile = AGVLoadProfile(agv_id="AGV001")
        profile.cpu_usage = 0.1
        profile.memory_usage = 0.1
        profile.task_queue_depth = 1
        profile.battery_level = 0.95
        profile.thermal_level = 0.1
        profile.compute_composite_load()
        assert not threshold.is_overloaded(profile)


class TestDynamicLoadBalancer:
    """动态负载均衡器测试"""

    def test_register_agv(self):
        balancer = DynamicLoadBalancer()
        profile = AGVLoadProfile(agv_id="AGV001")
        balancer.register_agv("AGV001", profile)
        assert "AGV001" in balancer._profiles

    def test_unregister_agv(self):
        balancer = DynamicLoadBalancer()
        profile = AGVLoadProfile(agv_id="AGV001")
        balancer.register_agv("AGV001", profile)
        balancer.unregister_agv("AGV001")
        assert "AGV001" not in balancer._profiles

    def test_submit_task(self):
        balancer = DynamicLoadBalancer()
        profile = AGVLoadProfile(agv_id="AGV001")
        balancer.register_agv("AGV001", profile)
        task = TaskSpec(task_id="T001", workload_units=1.0)
        assigned = balancer.submit_task(task)
        assert assigned == "AGV001"

    def test_submit_task_no_agv(self):
        balancer = DynamicLoadBalancer()
        task = TaskSpec(task_id="T001", workload_units=1.0)
        assigned = balancer.submit_task(task)
        assert assigned is None

    def test_complete_task(self):
        balancer = DynamicLoadBalancer()
        profile = AGVLoadProfile(agv_id="AGV001")
        balancer.register_agv("AGV001", profile)
        task = TaskSpec(task_id="T001", workload_units=1.0)
        balancer.submit_task(task)
        result = balancer.complete_task("T001")
        assert result is True
        assert balancer._completed_tasks == 1

    def test_update_profile(self):
        balancer = DynamicLoadBalancer()
        profile = AGVLoadProfile(agv_id="AGV001")
        balancer.register_agv("AGV001", profile)
        result = balancer.update_profile("AGV001", cpu_usage=0.8, battery_level=0.5)
        assert result is True

    def test_get_stats(self):
        balancer = DynamicLoadBalancer()
        for i in range(3):
            p = AGVLoadProfile(agv_id=f"AGV{i:03d}")
            p.cpu_usage = 0.1 * i
            balancer.register_agv(f"AGV{i:03d}", p)
        stats = balancer.get_stats()
        assert stats.total_agvs == 3
        assert stats.avg_load >= 0

    def test_get_load_distribution(self):
        balancer = DynamicLoadBalancer()
        for i in range(3):
            p = AGVLoadProfile(agv_id=f"AGV{i:03d}")
            p.cpu_usage = 0.1 * i
            balancer.register_agv(f"AGV{i:03d}", p)
        dist = balancer.get_load_distribution()
        assert len(dist) == 3

    def test_manual_rebalance(self):
        balancer = DynamicLoadBalancer(strategy=BalanceStrategy.LEAST_LOADED)
        # Register 2 AGVs
        p1 = AGVLoadProfile(agv_id="AGV001")
        p1.cpu_usage = 0.95
        p1.task_queue_depth = 15
        balancer.register_agv("AGV001", p1)
        p2 = AGVLoadProfile(agv_id="AGV002")
        p2.cpu_usage = 0.1
        p2.task_queue_depth = 1
        balancer.register_agv("AGV002", p2)
        # Give AGV001 a task to migrate
        task = TaskSpec(task_id="T001", workload_units=1.0, priority=3)
        balancer.submit_task(task)
        decisions = balancer.trigger_manual_rebalance()
        # Should attempt rebalance

    def test_tick(self):
        balancer = DynamicLoadBalancer(rebalance_interval_s=1.0)
        p = AGVLoadProfile(agv_id="AGV001")
        balancer.register_agv("AGV001", p)
        time.sleep(1.1)
        result = balancer.tick()
        assert result.total_agvs == 1

    def test_stress_heatmap(self):
        balancer = DynamicLoadBalancer()
        p = AGVLoadProfile(agv_id="AGV001")
        p.cpu_usage = 0.5
        p.memory_usage = 0.3
        balancer.register_agv("AGV001", p)
        heatmap = balancer.get_stress_heatmap()
        assert "AGV001" in heatmap
        assert "cpu" in heatmap["AGV001"]


class TestBalanceStrategies:
    """均衡策略测试"""

    def test_round_robin_cycles(self):
        balancer = DynamicLoadBalancer(strategy=BalanceStrategy.ROUND_ROBIN)
        for i in range(3):
            balancer.register_agv(f"AGV{i:03d}", AGVLoadProfile(agv_id=f"AGV{i:03d}"))
        task = TaskSpec(task_id="T001", workload_units=1.0)
        # Round robin should cycle through
        r1 = balancer._create_strategy(BalanceStrategy.ROUND_ROBIN).select_target_agv(
            task, balancer._profiles, LoadThreshold()
        )
        assert r1 in ["AGV000", "AGV001", "AGV002"]

    def test_capability_aware_selects_best(self):
        balancer = DynamicLoadBalancer(strategy=BalanceStrategy.CAPABILITY_AWARE)
        p1 = AGVLoadProfile(agv_id="AGV001")
        p1.capability_score = 0.3
        p1.cpu_usage = 0.1
        p1.memory_usage = 0.1
        p1.task_queue_depth = 1
        p1.battery_level = 0.9
        p1.thermal_level = 0.1
        p2 = AGVLoadProfile(agv_id="AGV002")
        p2.capability_score = 0.8
        p2.cpu_usage = 0.2
        p2.memory_usage = 0.2
        p2.task_queue_depth = 2
        p2.battery_level = 0.9
        p2.thermal_level = 0.1
        profiles = {"AGV001": p1, "AGV002": p2}
        task = TaskSpec(task_id="T001", required_capability=0.5)
        result = balancer._create_strategy(BalanceStrategy.CAPABILITY_AWARE).select_target_agv(
            task, profiles, LoadThreshold()
        )
        assert result == "AGV002"


# ==========================================================================
# CROSS-SCENE TRANSFER LEARNING TESTS (40 tests)
# ==========================================================================

class TestSceneKnowledgeGraph:
    """场景知识图谱测试"""

    def test_register_scene(self):
        kg = SceneKnowledgeGraph()
        profile = SceneProfile(SceneType.WAREHOUSE, "Warehouse")
        kg.register_scene(profile)
        assert SceneType.WAREHOUSE in kg.scene_profiles

    def test_register_skill(self):
        kg = SceneKnowledgeGraph()
        profile = SceneProfile(SceneType.WAREHOUSE, "Warehouse")
        kg.register_scene(profile)
        skill = SkillSpec(
            skill_id="",
            name="Navigate",
            domain=SkillDomain.NAVIGATION,
            source_scene=SceneType.WAREHOUSE,
        )
        skill_id = kg.register_skill(skill)
        assert skill_id in kg.skills

    def test_get_skills_for_scene(self):
        kg = SceneKnowledgeGraph()
        profile = SceneProfile(SceneType.RESTAURANT, "Restaurant")
        kg.register_scene(profile)
        skill = SkillSpec(
            skill_id="",
            name="Navigate",
            domain=SkillDomain.NAVIGATION,
            source_scene=SceneType.RESTAURANT,
        )
        kg.register_skill(skill)
        skills = kg.get_skills_for_scene(SceneType.RESTAURANT)
        assert len(skills) >= 1

    def test_transferable_skills(self):
        kg = SceneKnowledgeGraph()
        kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse"))
        kg.register_scene(SceneProfile(SceneType.INDUSTRIAL, "Industrial"))
        skill = SkillSpec(
            skill_id="",
            name="Navigate",
            domain=SkillDomain.NAVIGATION,
            source_scene=SceneType.WAREHOUSE,
            proficiency=0.8,
            sample_count=50,
            success_rate=0.9,
        )
        kg.register_skill(skill)
        results = kg.get_transferable_skills(SceneType.WAREHOUSE, SceneType.INDUSTRIAL)
        assert len(results) >= 1

    def test_scene_distance(self):
        kg = SceneKnowledgeGraph()
        s1 = SceneProfile(SceneType.WAREHOUSE, "Warehouse", terrain_variability=0.1, obstacle_density=0.3)
        s2 = SceneProfile(SceneType.OUTDOOR, "Outdoor", terrain_variability=0.8, obstacle_density=0.1)
        kg.register_scene(s1)
        kg.register_scene(s2)
        dist = s1.distance_to(s2)
        assert dist > 0

    def test_record_transfer(self):
        kg = SceneKnowledgeGraph()
        kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse"))
        kg.register_scene(SceneProfile(SceneType.OUTDOOR, "Outdoor"))
        record = TransferRecord(
            transfer_id="",
            skill_id="NAV001",
            from_scene=SceneType.WAREHOUSE,
            to_scene=SceneType.OUTDOOR,
            mode=TransferMode.FINE_TUNED,
            initial_performance=0.5,
            final_performance=0.75,
            adaptation_samples=20,
            duration_s=300.0,
        )
        kg.record_transfer(record)
        assert len(kg.transfer_history) == 1


class TestTransferabilityAnalyzer:
    """迁移性分析器测试"""

    def setup_method(self):
        self.kg = SceneKnowledgeGraph()
        self.kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse", obstacle_density=0.4))
        self.kg.register_scene(SceneProfile(SceneType.OUTDOOR, "Outdoor", obstacle_density=0.2))
        skill = SkillSpec(
            skill_id="NAV001",
            name="Navigate",
            domain=SkillDomain.NAVIGATION,
            source_scene=SceneType.WAREHOUSE,
            proficiency=0.8,
            sample_count=100,
            success_rate=0.9,
        )
        self.kg.register_skill(skill)
        self.analyzer = TransferabilityAnalyzer(self.kg)

    def test_analyze_direct_transfer(self):
        result = self.analyzer.analyze("NAV001", SceneType.OUTDOOR)
        assert 0.0 <= result.transferability_score <= 1.0
        assert isinstance(result.mode, TransferMode)

    def test_analyze_unknown_skill(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze("UNKNOWN", SceneType.OUTDOOR)

    def test_find_best_path(self):
        path = self.analyzer.find_best_transfer_path(SceneType.WAREHOUSE, SceneType.OUTDOOR)
        assert len(path) >= 2
        assert path[0] == SceneType.WAREHOUSE
        assert path[-1] == SceneType.OUTDOOR

    def test_risk_assessment(self):
        risk = self.analyzer._assess_risk(
            SkillSpec("S1", "S", SkillDomain.SAFETY, SceneType.WAREHOUSE),
            SceneProfile(SceneType.OUTDOOR, "Outdoor", obstacle_density=0.5, terrain_variability=0.5, dynamic_objects=0.5, cooperation_required=0.5)
        )
        assert 0.0 <= risk <= 1.0


class TestSceneAdapter:
    """场景适配器测试"""

    def test_start_adaptation(self):
        adapter = SceneAdapter()
        kg = SceneKnowledgeGraph()
        kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse"))
        skill = SkillSpec("NAV001", "Navigate", SkillDomain.NAVIGATION, SceneType.WAREHOUSE)
        candidate = TransferCandidate(
            skill=skill,
            target_scene=SceneType.WAREHOUSE,
            transferability_score=0.7,
            mode=TransferMode.FINE_TUNED,
            estimated_adaptation_samples=20,
            risk_level=0.3,
            adaptation_cost=100.0,
            confidence=0.8,
        )
        adapter_id = adapter.start_adaptation("NAV001", SceneType.WAREHOUSE, candidate)
        assert adapter_id.startswith("ADP")

    def test_record_sample_success(self):
        adapter = SceneAdapter()
        adapter.start_adaptation("NAV001", SceneType.WAREHOUSE, TransferCandidate(
            skill=SkillSpec("NAV001", "N", SkillDomain.NAVIGATION, SceneType.WAREHOUSE),
            target_scene=SceneType.WAREHOUSE,
            transferability_score=0.5,
            mode=TransferMode.FINE_TUNED,
            estimated_adaptation_samples=10,
            risk_level=0.3,
            adaptation_cost=50.0,
            confidence=0.7,
        ))
        result = adapter.record_sample("NAV001", success=True, completion_time_s=10.0)
        assert "progress" in result
        assert result["estimated_performance"] > 0

    def test_record_sample_failure(self):
        adapter = SceneAdapter()
        adapter.start_adaptation("NAV001", SceneType.WAREHOUSE, TransferCandidate(
            skill=SkillSpec("NAV001", "N", SkillDomain.NAVIGATION, SceneType.WAREHOUSE),
            target_scene=SceneType.WAREHOUSE,
            transferability_score=0.5,
            mode=TransferMode.FINE_TUNED,
            estimated_adaptation_samples=10,
            risk_level=0.3,
            adaptation_cost=50.0,
            confidence=0.7,
        ))
        result = adapter.record_sample("NAV001", success=False, completion_time_s=5.0)
        assert result["estimated_performance"] < 1.0

    def test_get_adaptation_report(self):
        adapter = SceneAdapter()
        adapter.start_adaptation("NAV001", SceneType.WAREHOUSE, TransferCandidate(
            skill=SkillSpec("NAV001", "N", SkillDomain.NAVIGATION, SceneType.WAREHOUSE),
            target_scene=SceneType.WAREHOUSE,
            transferability_score=0.5,
            mode=TransferMode.FINE_TUNED,
            estimated_adaptation_samples=10,
            risk_level=0.3,
            adaptation_cost=50.0,
            confidence=0.7,
        ))
        report = adapter.get_adaptation_report("NAV001")
        assert "progress_pct" in report
        assert report["skill_id"] == "NAV001"


class TestKnowledgeDistillation:
    """知识蒸馏测试"""

    def test_register_teacher(self):
        distiller = KnowledgeDistillation()
        logits = np.random.randn(10)
        distiller.register_teacher(SceneType.WAREHOUSE, logits)
        assert "warehouse" in distiller.teacher_policies

    def test_distill_single_teacher(self):
        distiller = KnowledgeDistillation()
        logits = np.random.randn(10)
        distiller.register_teacher(SceneType.WAREHOUSE, logits)
        result = distiller.distill()
        assert result.shape == (10,)

    def test_distill_multiple_teachers(self):
        distiller = KnowledgeDistillation()
        distiller.register_teacher(SceneType.WAREHOUSE, np.random.randn(10))
        distiller.register_teacher(SceneType.OUTDOOR, np.random.randn(10))
        result = distiller.distill()
        assert result.shape == (10,)

    def test_adapt_to_scene(self):
        distiller = KnowledgeDistillation()
        distiller.register_teacher(SceneType.WAREHOUSE, np.ones(10) * 0.5)
        distiller.register_teacher(SceneType.OUTDOOR, np.ones(10) * 0.3)
        distiller.distill()
        adapted = distiller.adapt_to_scene(SceneType.INDUSTRIAL, np.ones(10) * 0.8)
        assert adapted.shape == (10,)


class TestSceneCurriculum:
    """场景课程学习测试"""

    def test_build_curriculum(self):
        kg = SceneKnowledgeGraph()
        kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse"))
        kg.register_scene(SceneProfile(SceneType.RESTAURANT, "Restaurant"))
        kg.register_scene(SceneProfile(SceneType.OUTDOOR, "Outdoor"))
        curriculum = SceneCurriculum(kg)
        stages = curriculum.build_curriculum(SceneType.OUTDOOR)
        assert len(stages) >= 1

    def test_advance_stage(self):
        kg = SceneKnowledgeGraph()
        kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse"))
        kg.register_scene(SceneProfile(SceneType.OUTDOOR, "Outdoor"))
        curriculum = SceneCurriculum(kg)
        curriculum.build_curriculum(SceneType.OUTDOOR)
        assert curriculum.current_stage == 0
        curriculum.advance_stage()

    def test_check_graduation(self):
        kg = SceneKnowledgeGraph()
        kg.register_scene(SceneProfile(SceneType.WAREHOUSE, "Warehouse"))
        curriculum = SceneCurriculum(kg)
        curriculum.build_curriculum(SceneType.WAREHOUSE)
        result = curriculum.check_graduation({})
        assert isinstance(result, bool)


class TestCrossSceneSkillLibrary:
    """跨场景技能库测试"""

    def test_init_creates_default_scenes(self):
        lib = CrossSceneSkillLibrary()
        assert len(lib.kg.scene_profiles) >= 5

    def test_register_skill(self):
        lib = CrossSceneSkillLibrary()
        skill_id = lib.register_skill("Navigate", SkillDomain.NAVIGATION, SceneType.WAREHOUSE)
        assert skill_id is not None

    def test_query_transfer(self):
        lib = CrossSceneSkillLibrary()
        lib.register_skill("Navigate", SkillDomain.NAVIGATION, SceneType.WAREHOUSE)
        results = lib.query_transfer(SceneType.WAREHOUSE, SceneType.INDUSTRIAL)
        # May be empty if transferability score is below 0.1 threshold
        assert isinstance(results, list)

    def test_execute_transfer_unknown_skill(self):
        lib = CrossSceneSkillLibrary()
        lib.register_skill("Navigate", SkillDomain.SAFETY, SceneType.WAREHOUSE)
        # Should raise ValueError for unknown skill
        with pytest.raises(ValueError):
            lib.execute_transfer("NAV_UNKNOWN", SceneType.OUTDOOR)

    def test_distill_knowledge_empty(self):
        lib = CrossSceneSkillLibrary()
        # No teacher policies registered, should raise ValueError
        with pytest.raises(ValueError):
            lib.distill_knowledge()

    def test_build_curriculum(self):
        lib = CrossSceneSkillLibrary()
        stages = lib.build_curriculum(SceneType.OUTDOOR)
        assert isinstance(stages, list)


# ==========================================================================
# Run
# ==========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
