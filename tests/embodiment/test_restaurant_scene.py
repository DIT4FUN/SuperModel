"""
test_restaurant_scene.py - 餐厅场景测试
"""

import pytest
import time
from src.embodied.restaurant_scene import (
    RestaurantZone,
    FoodCategory,
    FoodTemperature,
    OrderType,
    RestaurantTask,
    RestaurantTaskLibrary,
    RestaurantSceneController,
    TableManager,
    DishCollector,
    OrderTracker,
    HygieneMonitor,
    MenuManager,
    get_restaurant_scene_controller,
)


class TestRestaurantZone:
    def test_all_zones_defined(self):
        zones = list(RestaurantZone)
        assert len(zones) >= 10
        assert RestaurantZone.KITCHEN in zones
        assert RestaurantZone.DINING_AREA in zones
        assert RestaurantZone.BAR in zones

    def test_zone_string(self):
        assert RestaurantZone.KITCHEN.name == "KITCHEN"


class TestFoodCategory:
    def test_food_categories(self):
        cats = list(FoodCategory)
        assert FoodCategory.HOT_FOOD in cats
        assert FoodCategory.BEVERAGE in cats
        assert FoodCategory.DESSERT in cats


class TestFoodTemperature:
    def test_temperature_levels(self):
        temps = list(FoodTemperature)
        assert FoodTemperature.HOT_KEEP in temps
        assert FoodTemperature.COLD_CHILLED in temps
        assert FoodTemperature.FROZEN in temps


class TestTableManager:
    def test_create_table(self):
        tm = TableManager()
        tid = tm.create_table(capacity=4, zone=RestaurantZone.DINING_AREA)
        assert tid.startswith("table_")
        assert tm.get_table(tid)['capacity'] == 4

    def test_seat_customers(self):
        tm = TableManager()
        tid = tm.create_table(capacity=4)
        assert tm.seat_customers(tid, 3) is True
        assert tm.get_table(tid)['customer_count'] == 3

    def test_seat_overflow(self):
        tm = TableManager()
        tid = tm.create_table(capacity=2)
        assert tm.seat_customers(tid, 5) is False

    def test_update_state(self):
        tm = TableManager()
        tid = tm.create_table(capacity=4)
        assert tm.update_state(tid, 'seated') is True
        assert tm.get_table(tid)['state'] == 'seated'

    def test_available_tables(self):
        tm = TableManager()
        t1 = tm.create_table(capacity=4)
        t2 = tm.create_table(capacity=6)
        tm.seat_customers(t1, 2)
        avail = tm.get_available_tables(min_capacity=2)
        assert t2 in avail
        assert t1 not in avail


class TestDishCollector:
    def test_start_collection(self):
        dc = DishCollector()
        cid = dc.start_collection("table_001")
        assert cid.startswith("coll_")

    def test_add_dishes(self):
        dc = DishCollector()
        cid = dc.start_collection("table_001")
        dc.add_dish(cid, 'plate', 3)
        dc.add_dish(cid, 'cup', 2)
        result = dc.finalize_collection(cid)
        assert result['plate'] == 3
        assert result['cup'] == 2


class TestOrderTracker:
    def test_create_order(self):
        ot = OrderTracker()
        oid = ot.create_order("table_001", OrderType.DINE_IN, customer_count=3)
        assert oid.startswith("ord_")
        order = ot.get_order(oid)
        assert order['customer_count'] == 3
        assert order['status'] == 'created'

    def test_add_item(self):
        ot = OrderTracker()
        oid = ot.create_order("table_001")
        ot.add_item(oid, "宫保鸡丁", 1, 38.0, FoodCategory.MAIN_DISH)
        order = ot.get_order(oid)
        assert len(order['items']) == 1
        assert order['total_value'] == 38.0

    def test_mark_kitchen_ready(self):
        ot = OrderTracker()
        oid = ot.create_order("table_001")
        ot.mark_kitchen_ready(oid)
        assert ot.get_order(oid)['status'] == 'ready_for_pickup'

    def test_delivery_time(self):
        ot = OrderTracker()
        oid = ot.create_order("table_001")
        ot.mark_picked_up(oid)
        time.sleep(0.01)
        ot.mark_delivered(oid)
        dt = ot.get_delivery_time(oid)
        assert dt is not None
        assert dt >= 0.01

    def test_ready_orders(self):
        ot = OrderTracker()
        oid = ot.create_order("table_001")
        ot.mark_kitchen_ready(oid)
        ready = ot.get_ready_orders()
        assert len(ready) == 1


class TestHygieneMonitor:
    def test_sanitation_update(self):
        hm = HygieneMonitor()
        hm.update_sanitation(RestaurantZone.KITCHEN, 85.0)
        assert hm.get_sanitation_score(RestaurantZone.KITCHEN) == 85.0

    def test_temperature_check_hot(self):
        hm = HygieneMonitor()
        ok, msg = hm.check_food_temperature(65.0, FoodTemperature.HOT_KEEP)
        assert ok is True
        ok, msg = hm.check_food_temperature(55.0, FoodTemperature.HOT_KEEP)
        assert ok is False

    def test_temperature_check_cold(self):
        hm = HygieneMonitor()
        ok, msg = hm.check_food_temperature(2.0, FoodTemperature.COLD_CHILLED)
        assert ok is True
        ok, msg = hm.check_food_temperature(10.0, FoodTemperature.COLD_CHILLED)
        assert ok is False

    def test_alert_threshold(self):
        hm = HygieneMonitor()
        hm.update_sanitation(RestaurantZone.KITCHEN, 80.0)  # Below standard 95
        alerts = hm.get_alerts()
        assert len(alerts) == 1


class TestMenuManager:
    def test_add_item(self):
        mm = MenuManager()
        iid = mm.add_item("宫保鸡丁", FoodCategory.MAIN_DISH, FoodTemperature.HOT_KEEP, 38.0)
        assert iid.startswith("menu_")
        item = mm.get_item(iid)
        assert item['name'] == "宫保鸡丁"

    def test_set_availability(self):
        mm = MenuManager()
        iid = mm.add_item("红烧肉", FoodCategory.MAIN_DISH, FoodTemperature.HOT_KEEP, 58.0)
        mm.set_availability(iid, False)
        assert mm.get_item(iid)['is_available'] is False

    def test_get_by_category(self):
        mm = MenuManager()
        mm.add_item("宫保鸡丁", FoodCategory.MAIN_DISH, FoodTemperature.HOT_KEEP, 38.0)
        mm.add_item("凉拌黄瓜", FoodCategory.COLD_FOOD, FoodTemperature.ROOM_TEMP, 18.0)
        mm.add_item("可乐", FoodCategory.BEVERAGE, FoodTemperature.COLD_CHILLED, 8.0)
        beverages = mm.get_by_category(FoodCategory.BEVERAGE)
        assert len(beverages) == 1
        assert beverages[0]['name'] == '可乐'


class TestRestaurantSceneController:
    def test_init(self):
        ctrl = RestaurantSceneController(agv_grade="M")
        assert ctrl.agv_grade == "M"
        assert ctrl.table_manager is not None
        assert ctrl.dish_collector is not None

    def test_create_food_delivery_task(self):
        ctrl = RestaurantSceneController()
        oid = ctrl.order_tracker.create_order("table_001")
        task = ctrl.create_food_delivery_task(
            table_id="table_001",
            order_id=oid,
            food_categories=[FoodCategory.MAIN_DISH],
            temperature=FoodTemperature.HOT_KEEP,
        )
        assert task.task_type == 'food_delivery'
        assert task.priority == 5
        assert task.food_temperature == FoodTemperature.HOT_KEEP

    def test_create_dish_collection_task(self):
        ctrl = RestaurantSceneController()
        task = ctrl.create_dish_collection_task(table_id="table_001")
        assert task.task_type == 'dish_collection'
        assert task.source_zone == RestaurantZone.DINING_AREA

    def test_get_next_task(self):
        ctrl = RestaurantSceneController()
        t1 = ctrl.create_food_delivery_task("table_001", "ord_001", [FoodCategory.BEVERAGE], FoodTemperature.ROOM_TEMP)
        t2 = ctrl.create_dish_collection_task("table_002")
        next_t = ctrl.get_next_task()
        assert next_t.task_id == t1.task_id   # Higher priority (5 vs 2)

    def test_zone_distance(self):
        ctrl = RestaurantSceneController()
        d = ctrl.get_zone_distance(RestaurantZone.KITCHEN, RestaurantZone.CARRIER_AREA)
        assert d == 2.0

    def test_estimate_delivery_time(self):
        ctrl = RestaurantSceneController(agv_grade="M")
        t = ctrl.estimate_delivery_time(RestaurantZone.CARRIER_AREA, RestaurantZone.DINING_AREA)
        assert t > 0

    def test_scene_status(self):
        ctrl = RestaurantSceneController()
        status = ctrl.get_scene_status()
        assert 'pending_tasks' in status
        assert 'completed_tasks' in status
        assert status['agv_grade'] == 'M'

    def test_scene_report(self):
        ctrl = RestaurantSceneController()
        report = ctrl.generate_scene_report()
        assert 'total_deliveries' in report
        assert 'task_breakdown' in report

    def test_agv_registration(self):
        ctrl = RestaurantSceneController()
        ctrl.register_agv("agv_001", RestaurantZone.KITCHEN)
        assert ctrl.get_agv_zone("agv_001") == RestaurantZone.KITCHEN

    def test_default_tables_created(self):
        ctrl = RestaurantSceneController()
        tables = ctrl.table_manager.get_all_tables()
        assert len(tables) == 12  # 4 dining + 4 bar + 4 buffet

    def test_menu_initialized(self):
        ctrl = RestaurantSceneController()
        items = list(ctrl.menu_manager._menu_items.values())
        assert len(items) == 10


class TestRestaurantTaskLibrary:
    def test_get_template(self):
        tmpl = RestaurantTaskLibrary.get_template('hot_main_dish')
        assert tmpl['priority'] == 5
        assert tmpl['temperature'] == FoodTemperature.HOT_KEEP

    def test_get_nonexistent_template(self):
        tmpl = RestaurantTaskLibrary.get_template('nonexistent')
        assert tmpl == {}


class TestRestaurantSceneControllerGrades:
    def test_grade_s_speed(self):
        ctrl = RestaurantSceneController(agv_grade="S")
        t = ctrl.estimate_delivery_time(RestaurantZone.KITCHEN, RestaurantZone.DINING_AREA)
        # S grade is slowest, so time should be higher
        ctrl_m = RestaurantSceneController(agv_grade="M")
        tm = ctrl_m.estimate_delivery_time(RestaurantZone.KITCHEN, RestaurantZone.DINING_AREA)
        assert t >= tm


class TestRestaurantSceneEnums:
    def test_order_types(self):
        assert OrderType.DINE_IN in list(OrderType)
        assert OrderType.TAKEOUT in list(OrderType)
        assert OrderType.DELIVERY in list(OrderType)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
