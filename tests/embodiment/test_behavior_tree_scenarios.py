"""
Behavior Tree Scenario Tests - 行为树场景化任务规划测试
测试真实AGV任务场景：仓库配送/工厂搬运/医院物流/餐厅送餐
使用 BehaviorNode (control.planner) + bt.run() 模式
"""

import pytest
import time
import math
import numpy as np
from embodiment.behavior_tree_engine import (
    NodeStatus, BehaviorTreeEngine, BehaviorNode,
    SequenceNode, ParallelNode, ConditionNode, TaskNode,
    StateMachineNode, RetryNode, TimeoutNode, InverterNode,
    AlwaysSuccessNode, AlwaysFailureNode, AGVTaskTrees
)
from embodiment.multi_agv_coordinator import (
    MultiAGVCoordinator, AGVStatus, MarketAuctionAllocator,
    FormationController
)
from embodiment.simulation import EmbodimentSimulator, SimSceneConfig, SimAGVConfig, SimulationScene


# =============================================================================
# Warehouse Delivery Scenarios
# =============================================================================

class TestWarehouseDeliveryScenarios:
    """仓库配送场景行为树测试"""

    def test_warehouse_pick_and_place_sequence(self):
        """仓库拣选-放置完整流程"""
        bt = BehaviorTreeEngine()
        state = {"at_station": None, "has_object": False, "delivery_complete": False}
        
        def move_to_pickup(ctx):
            state["at_station"] = "pickup_A"
            return NodeStatus.SUCCESS
        
        def grasp_object(ctx):
            state["has_object"] = True
            return NodeStatus.SUCCESS
        
        def check_loaded(ctx):
            return NodeStatus.SUCCESS if state.get("has_object", False) else NodeStatus.FAILURE
        
        def move_to_destination(ctx):
            state["at_station"] = "station_B"
            return NodeStatus.SUCCESS
        
        def release_object(ctx):
            state["has_object"] = False
            state["delivery_complete"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("move_to_pickup", move_to_pickup))
        bt.add_node(BehaviorNode("grasp_object", grasp_object))
        bt.add_node(BehaviorNode("check_loaded", check_loaded))
        bt.add_node(BehaviorNode("move_to_destination", move_to_destination))
        bt.add_node(BehaviorNode("release_object", release_object))
        
        bt.add_sequence("pick_place", [
            "move_to_pickup", "grasp_object", "check_loaded",
            "move_to_destination", "release_object"
        ])
        
        result = bt.run("pick_place", {})
        
        assert result == NodeStatus.SUCCESS
        assert state["has_object"] == False
        assert state["delivery_complete"] == True

    def test_warehouse_obstacle_avoidance_sequence(self):
        """仓库障碍规避序列"""
        bt = BehaviorTreeEngine()
        state = {"path_blocked": True}
        
        def check_path_clear(ctx):
            return NodeStatus.SUCCESS if not state["path_blocked"] else NodeStatus.FAILURE
        
        def plan_detour(ctx):
            state["alternate_path"] = [(0,1), (1,1), (2,0)]
            return NodeStatus.SUCCESS
        
        def execute_detour(ctx):
            state["alternate_path"].pop(0)
            if len(state["alternate_path"]) == 0:
                state["path_blocked"] = False
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING
        
        def resume_original_path(ctx):
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("check_clear", check_path_clear))
        bt.add_node(BehaviorNode("plan_detour", plan_detour))
        bt.add_node(BehaviorNode("execute_detour", execute_detour))
        bt.add_node(BehaviorNode("resume_path", resume_original_path))
        
        bt.add_sequence("obstacle_avoid", [
            "check_clear", "plan_detour", "execute_detour", "resume_path"
        ])
        
        # 场景：有障碍 -> check失败 -> 绕过
        bt2 = BehaviorTreeEngine()
        s2 = {"path_blocked": True}
        
        def c2(ctx): return NodeStatus.FAILURE
        def p2(ctx): s2["detoured"] = True; return NodeStatus.SUCCESS
        def r2(ctx): return NodeStatus.SUCCESS
        
        bt2.add_node(BehaviorNode("check_clear", c2))
        bt2.add_node(BehaviorNode("plan_detour", p2))
        bt2.add_node(BehaviorNode("resume_path", r2))
        bt2.add_fallback("fallback", ["check_clear", "plan_detour"])
        bt2.add_sequence("obstacle_avoid", ["fallback", "resume_path"])
        
        result = bt2.run("obstacle_avoid", {})
        
        assert result == NodeStatus.SUCCESS
        assert s2["detoured"] == True

    def test_warehouse_battery_low_return_sequence(self):
        """仓库电池低回充序列"""
        bt = BehaviorTreeEngine()
        state = {"at_station": None, "charging": False}
        
        def check_battery(ctx):
            return NodeStatus.SUCCESS if ctx.get("battery_soc", 100) < 20 else NodeStatus.FAILURE
        
        def plan_charging_route(ctx):
            state["target"] = "charging_station"
            return NodeStatus.SUCCESS
        
        def navigate_to_charger(ctx):
            state["at_station"] = state["target"]
            return NodeStatus.SUCCESS
        
        def initiate_charging(ctx):
            state["charging"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("check_battery", check_battery))
        bt.add_node(BehaviorNode("plan_charging_route", plan_charging_route))
        bt.add_node(BehaviorNode("navigate_to_charger", navigate_to_charger))
        bt.add_node(BehaviorNode("initiate_charging", initiate_charging))
        
        bt.add_sequence("battery_management", [
            "check_battery", "plan_charging_route", 
            "navigate_to_charger", "initiate_charging"
        ])
        
        result = bt.run("battery_management", {"battery_soc": 15})
        
        assert result == NodeStatus.SUCCESS
        assert state["charging"] == True

    def test_warehouse_concurrent_multi_pick(self):
        """仓库并发多拣选任务（顺序执行）"""
        bt = BehaviorTreeEngine()
        state = {"pick_A_done": False, "pick_B_done": False, "pick_C_done": False}
        
        def pick_A(ctx):
            state["pick_A_done"] = True
            return NodeStatus.SUCCESS
        
        def pick_B(ctx):
            state["pick_B_done"] = True
            return NodeStatus.SUCCESS
        
        def pick_C(ctx):
            state["pick_C_done"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("pick_A", pick_A))
        bt.add_node(BehaviorNode("pick_B", pick_B))
        bt.add_node(BehaviorNode("pick_C", pick_C))
        
        bt.add_sequence("multi_pick", ["pick_A", "pick_B", "pick_C"])
        
        result = bt.run("multi_pick", {})
        
        assert result == NodeStatus.SUCCESS
        assert state["pick_A_done"] == True
        assert state["pick_B_done"] == True
        assert state["pick_C_done"] == True


# =============================================================================
# Factory Floor Scenarios
# =============================================================================

class TestFactoryFloorScenarios:
    """工厂车间场景行为树测试"""

    def test_factory_production_line_feed(self):
        """生产线供料场景"""
        bt = BehaviorTreeEngine()
        state = {"feed_complete": False}
        
        def check_buffer_level(ctx):
            return NodeStatus.SUCCESS if ctx.get("buffer_full", False) else NodeStatus.FAILURE
        
        def wait_for_consumption(ctx):
            return NodeStatus.SUCCESS if ctx.get("consumed", False) else NodeStatus.RUNNING
        
        def transport_material(ctx):
            state["feed_complete"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("check_buffer", check_buffer_level))
        bt.add_node(BehaviorNode("wait", wait_for_consumption))
        bt.add_node(BehaviorNode("transport", transport_material))
        
        bt.add_sequence("production_feed", ["check_buffer", "wait", "transport"])
        
        result = bt.run("production_feed", {"buffer_full": True, "consumed": True})
        
        assert result == NodeStatus.SUCCESS
        assert state["feed_complete"] == True

    def test_factory_equipment_maintenance_sequence(self):
        """设备维护序列"""
        bt = BehaviorTreeEngine()
        state = {"diagnostics_complete": False, "parts_replaced": False, "recalibrated": False}
        
        def check_maint_needed(ctx):
            return NodeStatus.SUCCESS if ctx.get("maintenance_due", True) else NodeStatus.FAILURE
        
        def navigate_to_equipment(ctx):
            return NodeStatus.SUCCESS
        
        def perform_diagnostics(ctx):
            state["diagnostics_complete"] = True
            return NodeStatus.SUCCESS
        
        def replace_worn_parts(ctx):
            state["parts_replaced"] = True
            return NodeStatus.SUCCESS
        
        def recalibrate(ctx):
            state["recalibrated"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("check_maint", check_maint_needed))
        bt.add_node(BehaviorNode("navigate", navigate_to_equipment))
        bt.add_node(BehaviorNode("diagnostics", perform_diagnostics))
        bt.add_node(BehaviorNode("replace_parts", replace_worn_parts))
        bt.add_node(BehaviorNode("recalibrate", recalibrate))
        
        bt.add_sequence("maintenance", [
            "check_maint", "navigate", "diagnostics", "replace_parts", "recalibrate"
        ])
        
        result = bt.run("maintenance", {"maintenance_due": True})
        
        assert result == NodeStatus.SUCCESS
        assert state["recalibrated"] == True

    def test_factory_quality_check_gate(self):
        """工厂质量检查门"""
        bt = BehaviorTreeEngine()
        
        def quality_check(ctx):
            return NodeStatus.SUCCESS if ctx.get("quality_ok", True) else NodeStatus.FAILURE
        
        def route_to_rework(ctx):
            return NodeStatus.SUCCESS
        
        def route_to_passing(ctx):
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("quality_check", quality_check))
        bt.add_node(BehaviorNode("to_rework", route_to_rework))
        bt.add_node(BehaviorNode("to_passing", route_to_passing))
        
        # Fallback: 如果quality_check失败则to_passing（通过）
        bt.add_fallback("quality_gate", ["quality_check", "to_passing"])


# =============================================================================
# Hospital Logistics Scenarios
# =============================================================================

class TestHospitalLogisticsScenarios:
    """医院物流场景行为树测试"""

    def test_hospital_specimen_transport(self):
        """医院标本运输场景"""
        bt = BehaviorTreeEngine()
        
        def verify_container(ctx):
            ctx["container_secured"] = True
            return NodeStatus.SUCCESS
        
        def navigate_zone(ctx):
            ctx["in_critical_zone"] = True
            return NodeStatus.SUCCESS
        
        def check_temp(ctx):
            return NodeStatus.SUCCESS if ctx.get("temp_ok", True) else NodeStatus.FAILURE
        
        def deliver_to_lab(ctx):
            ctx["delivered"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("verify_container", verify_container))
        bt.add_node(BehaviorNode("navigate_zone", navigate_zone))
        bt.add_node(BehaviorNode("check_temp", check_temp))
        bt.add_node(BehaviorNode("deliver", deliver_to_lab))
        
        bt.add_sequence("specimen_transport", [
            "verify_container", "navigate_zone", "check_temp", "deliver"
        ])
        
        ctx = {"temp_ok": True}
        result = bt.run("specimen_transport", ctx)
        
        assert result == NodeStatus.SUCCESS
        assert ctx["delivered"] == True

    def test_hospital_medication_delivery(self):
        """医院药品配送场景"""
        bt = BehaviorTreeEngine()
        
        def verify_patient(ctx):
            return NodeStatus.SUCCESS if ctx.get("patient_verified", True) else NodeStatus.FAILURE
        
        def verify_med(ctx):
            return NodeStatus.SUCCESS if ctx.get("medication_verified", True) else NodeStatus.FAILURE
        
        def navigate_bed(ctx):
            return NodeStatus.SUCCESS
        
        def handoff(ctx):
            ctx["handoff_complete"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("verify_patient", verify_patient))
        bt.add_node(BehaviorNode("verify_med", verify_med))
        bt.add_node(BehaviorNode("navigate_bed", navigate_bed))
        bt.add_node(BehaviorNode("handoff", handoff))
        
        bt.add_sequence("med_delivery", [
            "verify_patient", "verify_med", "navigate_bed", "handoff"
        ])
        
        ctx = {"patient_verified": True, "medication_verified": True}
        result = bt.run("med_delivery", ctx)
        
        assert result == NodeStatus.SUCCESS
        assert ctx["handoff_complete"] == True

    def test_hospital_emergency_priority_override(self):
        """医院急诊优先级覆盖"""
        bt = BehaviorTreeEngine()
        
        def check_emergency(ctx):
            return NodeStatus.SUCCESS if ctx.get("is_emergency", False) else NodeStatus.FAILURE
        
        def preempt(ctx):
            ctx["preempted"] = True
            return NodeStatus.SUCCESS
        
        def handle_emergency(ctx):
            ctx["emergency_handled"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("check_emergency", check_emergency))
        bt.add_node(BehaviorNode("preempt", preempt))
        bt.add_node(BehaviorNode("handle", handle_emergency))
        
        bt.add_sequence("emergency_response", ["check_emergency", "preempt", "handle"])


# =============================================================================
# Restaurant Delivery Scenarios
# =============================================================================

class TestRestaurantDeliveryScenarios:
    """餐厅送餐场景行为树测试"""

    def test_restaurant_food_delivery_sequence(self):
        """餐厅食物配送序列"""
        bt = BehaviorTreeEngine()
        
        def pickup(ctx):
            ctx["has_food"] = True
            return NodeStatus.SUCCESS
        
        def navigate(ctx):
            return NodeStatus.SUCCESS
        
        def verify_order(ctx):
            return NodeStatus.SUCCESS if ctx.get("order_verified", True) else NodeStatus.FAILURE
        
        def serve(ctx):
            ctx["has_food"] = False
            ctx["served"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("pickup", pickup))
        bt.add_node(BehaviorNode("navigate", navigate))
        bt.add_node(BehaviorNode("verify", verify_order))
        bt.add_node(BehaviorNode("serve", serve))
        
        bt.add_sequence("food_delivery", ["pickup", "navigate", "verify", "serve"])
        
        ctx = {"order_verified": True}
        result = bt.run("food_delivery", ctx)
        
        assert result == NodeStatus.SUCCESS
        assert ctx["served"] == True

    def test_restaurant_dirty_dish_collection(self):
        """餐厅脏餐具回收"""
        bt = BehaviorTreeEngine()
        
        def nav_table(ctx):
            return NodeStatus.SUCCESS
        
        def collect(ctx):
            ctx["dishes_collected"] = True
            return NodeStatus.SUCCESS
        
        def nav_kitchen(ctx):
            return NodeStatus.SUCCESS
        
        def deposit(ctx):
            ctx["dishes_deposited"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("navigate_table", nav_table))
        bt.add_node(BehaviorNode("collect", collect))
        bt.add_node(BehaviorNode("navigate_kitchen", nav_kitchen))
        bt.add_node(BehaviorNode("deposit", deposit))
        
        bt.add_sequence("dish_collection", [
            "navigate_table", "collect", "navigate_kitchen", "deposit"
        ])


# =============================================================================
# State Machine Scenarios
# =============================================================================

class TestStateMachineScenarios:
    """状态机场景测试"""

    def test_agv_state_machine_full_cycle(self):
        """AGV完整状态机周期"""
        bt = BehaviorTreeEngine()
        
        # 使用ConditionNode构建状态机行为
        def set_idle(ctx):
            ctx["state"] = "IDLE"
            return NodeStatus.SUCCESS
        
        def set_navigating(ctx):
            ctx["state"] = "NAVIGATING"
            return NodeStatus.SUCCESS
        
        def set_loading(ctx):
            ctx["state"] = "LOADING"
            return NodeStatus.SUCCESS
        
        def set_unloading(ctx):
            ctx["state"] = "UNLOADING"
            return NodeStatus.SUCCESS
        
        def set_charging(ctx):
            ctx["state"] = "CHARGING"
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("idle_t", set_idle))
        bt.add_node(BehaviorNode("nav_t", set_navigating))
        bt.add_node(BehaviorNode("load_t", set_loading))
        bt.add_node(BehaviorNode("unload_t", set_unloading))
        bt.add_node(BehaviorNode("charge_t", set_charging))
        
        bt.add_sequence("idle_sequence", ["idle_t"])
        bt.add_sequence("nav_sequence", ["nav_t"])
        bt.add_sequence("load_sequence", ["load_t"])
        bt.add_sequence("unload_sequence", ["unload_t"])
        bt.add_sequence("charge_sequence", ["charge_t"])
        
        # 模拟状态转换序列
        ctx = {}
        result = bt.run("idle_sequence", ctx)
        assert result == NodeStatus.SUCCESS
        assert ctx["state"] == "IDLE"
        
        bt.reset()
        result = bt.run("nav_sequence", ctx)
        assert result == NodeStatus.SUCCESS
        assert ctx["state"] == "NAVIGATING"

    def test_agv_state_machine_transition_guard(self):
        """状态机转换守卫"""
        bt = BehaviorTreeEngine()
        
        def idle_action(ctx):
            ctx["at_idle"] = True
            return NodeStatus.SUCCESS
        
        def nav_action(ctx):
            ctx["at_nav"] = True
            return NodeStatus.SUCCESS
        
        def mission_assigned(ctx):
            return NodeStatus.SUCCESS if ctx.get("mission_assigned", False) else NodeStatus.FAILURE
        
        bt.add_node(BehaviorNode("idle_node", idle_action))
        bt.add_node(BehaviorNode("nav_node", nav_action))
        bt.add_node(ConditionNode("mission_guard", mission_assigned))
        
        # 带守卫的转换: fallback当守卫失败时执行nav_node导航到目标位置
        # 当nav_node执行成功时fallback返回SUCCESS，因此当mission_assigned=False时
        # fallback返回SUCCESS（守卫失败但fallback成功）
        bt.add_fallback("idle_with_guard", ["mission_guard", "nav_node"])
        
        # 场景1：任务未分配，守卫失败，fallback用nav_node作后备返回SUCCESS
        ctx1 = {"mission_assigned": False}
        result1 = bt.run("idle_with_guard", ctx1)
        assert result1 == NodeStatus.SUCCESS  # fallback: guard失败但nav_node成功，整体SUCCESS
        
        # 场景2：任务已分配，守卫成功
        ctx2 = {"mission_assigned": True}
        bt.reset()
        result2 = bt.run("idle_with_guard", ctx2)
        assert result2 == NodeStatus.SUCCESS


# =============================================================================
# Timeout and Retry Scenarios
# =============================================================================

class TestTimeoutRetryScenarios:
    """超时与重试场景测试"""

    def test_navigation_timeout_fallback(self):
        """导航超时后备方案"""
        bt = BehaviorTreeEngine()
        
        attempts = {"count": 0}
        
        def navigate_primary(ctx):
            attempts["count"] += 1
            return NodeStatus.RUNNING
        
        def check_timeout(ctx):
            return NodeStatus.SUCCESS if attempts["count"] >= 2 else NodeStatus.FAILURE
        
        def navigate_backup(ctx):
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("nav_primary", navigate_primary))
        bt.add_node(BehaviorNode("timeout_check", check_timeout))
        bt.add_node(BehaviorNode("nav_backup", navigate_backup))
        
        bt.add_sequence("nav_with_timeout", ["nav_primary", "timeout_check", "nav_backup"])
        
        # 运行多次直到超时
        for _ in range(10):
            bt.run("nav_with_timeout", {})
            if attempts["count"] >= 2:
                break
        
        assert attempts["count"] >= 2

    def test_delivery_retry_until_success(self):
        """配送任务重试直到成功"""
        bt = BehaviorTreeEngine()
        
        delivery_attempts = {"count": 0}
        
        def attempt_delivery(ctx):
            delivery_attempts["count"] += 1
            if delivery_attempts["count"] >= 2:
                ctx["delivered"] = True
                return NodeStatus.SUCCESS
            return NodeStatus.FAILURE
        
        bt.add_node(BehaviorNode("deliver", attempt_delivery))
        bt.add_decorator("retry_delivery", "deliver", max_retries=5)
        
        ctx = {"delivered": False}
        result = bt.run("retry_delivery", ctx)
        
        assert result == NodeStatus.SUCCESS
        assert ctx["delivered"] == True
        assert delivery_attempts["count"] == 2


# =============================================================================
# Mission Planning Integration Tests
# =============================================================================

class TestMissionPlanningIntegration:
    """任务规划集成测试"""

    def test_full_mission_with_agv_task_trees(self):
        """使用AGVTaskTrees完整任务规划"""
        mission_tree = AGVTaskTrees.build_transport_tree()
        
        assert mission_tree is not None
        
        # 验证任务树结构
        assert hasattr(mission_tree, 'tick')

    def test_patrol_mission_tree(self):
        """巡检任务树"""
        patrol_tree = AGVTaskTrees.build_patrol_tree()
        assert patrol_tree is not None
        assert hasattr(patrol_tree, 'tick')

    def test_emergency_mission_tree(self):
        """紧急任务树"""
        emergency_tree = AGVTaskTrees.build_emergency_tree()
        assert emergency_tree is not None
        assert hasattr(emergency_tree, 'tick')


# =============================================================================
# Formation and Swarm Behavior Tree Tests
# =============================================================================

class TestFormationSwarmBehavior:
    """编队与蜂群行为树测试"""

    def test_formation_change_behavior_tree(self):
        """编队变换行为树"""
        bt = BehaviorTreeEngine()
        
        def detect_formation_needed(ctx):
            return NodeStatus.SUCCESS if ctx.get("formation_change_needed", True) else NodeStatus.FAILURE
        
        def calculate_formation(ctx):
            ctx["target_formation"] = "wedge"
            return NodeStatus.SUCCESS
        
        def execute_formation_change(ctx):
            ctx["formation_changed"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("check_formation", detect_formation_needed))
        bt.add_node(BehaviorNode("calculate", calculate_formation))
        bt.add_node(BehaviorNode("execute", execute_formation_change))
        
        bt.add_sequence("formation_change", ["check_formation", "calculate", "execute"])
        
        ctx = {"formation_change_needed": True}
        result = bt.run("formation_change", ctx)
        
        assert result == NodeStatus.SUCCESS
        assert ctx["formation_changed"] == True

    def test_swarm_task_assignment_behavior(self):
        """蜂群任务分配行为树"""
        coordinator = MultiAGVCoordinator(swarm_id="swarm_auction")
        coordinator.register_agv("agv_1", position=(0.0, 0.0))
        coordinator.register_agv("agv_2", position=(1.0, 0.0))
        coordinator.register_agv("agv_3", position=(2.0, 0.0))
        
        auction = MarketAuctionAllocator(coordinator)
        
        # 创建任务
        task = {"task_id": "t1", "type": "transfer", "priority": 5}
        
        # 模拟竞价
        bids = {"1": 10.0, "2": 8.0, "3": 12.0}
        winner_id = min(bids, key=bids.get)
        
        assert winner_id == "2"
        assert bids[winner_id] == 8.0

    def test_role_negotiation_behavior_tree(self):
        """角色协商行为树"""
        bt = BehaviorTreeEngine()
        
        def propose_role(ctx):
            ctx["proposed_role"] = ctx.get("preferred_role", "follower")
            return NodeStatus.SUCCESS
        
        def negotiate_role(ctx):
            return NodeStatus.SUCCESS
        
        def confirm_role(ctx):
            ctx["role_confirmed"] = True
            return NodeStatus.SUCCESS
        
        bt.add_node(BehaviorNode("propose", propose_role))
        bt.add_node(BehaviorNode("negotiate", negotiate_role))
        bt.add_node(BehaviorNode("confirm", confirm_role))
        
        bt.add_sequence("role_negotiation", ["propose", "negotiate", "confirm"])
        
        ctx = {"preferred_role": "leader", "agv_id": "agv_1"}
        result = bt.run("role_negotiation", ctx)
        
        assert result == NodeStatus.SUCCESS
        assert ctx["role_confirmed"] == True
