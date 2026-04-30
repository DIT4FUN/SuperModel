# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
restaurant_scene.py - 餐厅场景化具身智能模块
SuperModel 超模态大模型具身智能系统

餐厅场景专项:
- 食物配送 (热食/冷食/饮料/甜点从厨房到餐桌)
- 餐具回收 (餐桌收拾/餐具分类/清理)
- 饮品补充 (水/饮料/咖啡补充)
- 餐桌清理 (桌面清洁/餐具摆放/椅凳整理)
- 自助区管理 (自助餐台/饮料机/调料台)
- 迎宾带位 (顾客迎接/座位引导/等位管理)
- 外卖配送 (打包/交接/外卖柜管理)
"""

from __future__ import annotations

import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)

__all__ = [
    'RestaurantZone',
    'FoodCategory',
    'FoodTemperature',
    'OrderType',
    'RestaurantTask',
    'RestaurantTaskLibrary',
    'RestaurantSceneController',
    'TableManager',
    'DishCollector',
    'OrderTracker',
    'HygieneMonitor',
    'MenuManager',
    'get_restaurant_scene_controller',
]


class RestaurantZone(Enum):
    """餐厅区域类型"""
    KITCHEN = auto()           # 后厨
    BAR = auto()              # 吧台
    DINING_AREA = auto()       # 用餐区
    COUNTER = auto()           # 收银台
    ENTRANCE = auto()          # 入口/迎宾区
    RESTROOM = auto()          # 洗手间
    STORAGE = auto()           # 储物间
    CARRIER_AREA = auto()      # 传菜通道
    BUFFET = auto()            # 自助餐台
    DRINKS_STATION = auto()    # 饮料区
    DESSERT_STATION = auto()   # 甜点区
    TAKEOUT_AREA = auto()      # 外卖区
    ELEVATOR = auto()          # 电梯


class FoodCategory(Enum):
    """食物类别"""
    HOT_FOOD = auto()          # 热菜
    COLD_FOOD = auto()         # 冷菜
    SOUP = auto()             # 汤类
    MAIN_DISH = auto()         # 主菜
    SIDE_DISH = auto()         # 小吃/配菜
    BREAD = auto()            # 面包/主食
    DESSERT = auto()           # 甜点
    BEVERAGE = auto()          # 饮料
    ALCOHOL = auto()           # 酒水
    FRUIT = auto()             # 水果


class FoodTemperature(Enum):
    """食物温度要求"""
    HOT_KEEP = auto()          # 保温 (>60°C)
    ROOM_TEMP = auto()         # 常温
    COLD_CHILLED = auto()      # 冷藏 (0-4°C)
    FROZEN = auto()            # 冷冻 (<-18°C)
    GRILL_HOT = auto()         # 现烤/热出锅 (>70°C)


class OrderType(Enum):
    """订单类型"""
    DINE_IN = auto()           # 堂食
    TAKEOUT = auto()           # 外卖打包
    DELIVERY = auto()          # 外卖配送
    BUFFET = auto()            # 自助餐
    BANQUET = auto()           # 宴席


@dataclass
class RestaurantTask:
    """餐厅任务"""
    task_id: str
    task_type: str                      # food_delivery/dish_collection/drink_refill/table_clear/bus_welcome/takeout
    priority: int                       # 1-5, 5最高
    source_zone: RestaurantZone
    destination_zone: RestaurantZone
    table_id: Optional[str] = None
    order_id: Optional[str] = None
    food_categories: List[FoodCategory] = field(default_factory=list)
    food_temperature: Optional[FoodTemperature] = None
    order_type: OrderType = OrderType.DINE_IN
    customer_count: int = 1
    requires_beverage_cart: bool = False
    time_constraint: Optional[float] = None    # 秒
    created_at: float = field(default_factory=time.time)
    picked_up_at: Optional[float] = None
    delivered_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"             # pending/picked_up/in_transit/delivered/completed/failed
    notes: str = ""
    rating: Optional[int] = None        # 1-5 customer rating


class RestaurantTaskLibrary:
    """餐厅任务模板库"""

    # 食物配送任务模板
    FOOD_DELIVERY_TEMPLATES = {
        'hot_main_dish': {
            'type': 'food_delivery',
            'priority': 5,
            'temperature': FoodTemperature.HOT_KEEP,
            'categories': [FoodCategory.HOT_FOOD, FoodCategory.MAIN_DISH],
            'time_limit': 180,         # 3分钟内送达
            'requires_cover': True,
        },
        'cold_dessert': {
            'type': 'food_delivery',
            'priority': 3,
            'temperature': FoodTemperature.COLD_CHILLED,
            'categories': [FoodCategory.COLD_FOOD, FoodCategory.DESSERT],
            'time_limit': 300,         # 5分钟内送达
            'requires_cover': True,
        },
        'beverage_refill': {
            'type': 'drink_refill',
            'priority': 2,
            'temperature': FoodTemperature.ROOM_TEMP,
            'categories': [FoodCategory.BEVERAGE],
            'time_limit': 120,          # 2分钟内送达
            'requires_cover': False,
        },
        'takeout_order': {
            'type': 'takeout',
            'priority': 4,
            'order_type': OrderType.TAKEOUT,
            'time_limit': 240,         # 4分钟内完成打包
            'requires_sealing': True,
        },
    }

    @classmethod
    def get_template(cls, template_name: str) -> Dict[str, Any]:
        """获取任务模板"""
        for library in [cls.FOOD_DELIVERY_TEMPLATES]:
            if template_name in library:
                return dict(library[template_name])
        return {}


class TableManager:
    """餐桌管理器"""

    TABLE_STATES = ['available', 'seated', 'ordering', 'eating', 'paying', 'clearing']

    def __init__(self):
        self._tables: Dict[str, Dict] = {}
        self._next_table_id = 1

    def create_table(self, capacity: int, zone: RestaurantZone = RestaurantZone.DINING_AREA) -> str:
        """创建餐桌"""
        table_id = f"table_{self._next_table_id:03d}"
        self._next_table_id += 1
        self._tables[table_id] = {
            'capacity': capacity,
            'zone': zone,
            'state': 'available',
            'customer_count': 0,
            'current_order_id': None,
            'last_service_time': None,
        }
        return table_id

    def seat_customers(self, table_id: str, count: int) -> bool:
        """顾客入座"""
        if table_id not in self._tables:
            return False
        table = self._tables[table_id]
        if count > table['capacity']:
            return False
        table['state'] = 'seated'
        table['customer_count'] = count
        return True

    def update_state(self, table_id: str, state: str) -> bool:
        """更新餐桌状态"""
        if table_id not in self._tables:
            return False
        if state not in self.TABLE_STATES:
            return False
        self._tables[table_id]['state'] = state
        self._tables[table_id]['last_service_time'] = time.time()
        return True

    def get_table(self, table_id: str) -> Optional[Dict]:
        """获取餐桌信息"""
        return self._tables.get(table_id)

    def get_available_tables(self, min_capacity: int = 1) -> List[str]:
        """获取可用餐桌"""
        return [
            tid for tid, t in self._tables.items()
            if t['state'] == 'available' and t['capacity'] >= min_capacity
        ]

    def get_all_tables(self) -> Dict[str, Dict]:
        """获取所有餐桌"""
        return dict(self._tables)

    def clear_table(self, table_id: str) -> bool:
        """清桌"""
        if table_id not in self._tables:
            return False
        self._tables[table_id].update({
            'state': 'available',
            'customer_count': 0,
            'current_order_id': None,
        })
        return True


class DishCollector:
    """餐具回收器"""

    DISH_TYPES = ['plate', 'bowl', 'cup', 'glass', 'utensil', 'napkin', 'tray']

    def __init__(self):
        self._collected: Dict[str, List[str]] = {}   # table_id -> [dish_ids]
        self._dish_counter = 0

    def start_collection(self, table_id: str) -> str:
        """开始回收"""
        collection_id = f"coll_{table_id}_{int(time.time())}"
        self._collected[collection_id] = []
        return collection_id

    def add_dish(self, collection_id: str, dish_type: str, count: int = 1) -> bool:
        """添加餐具"""
        if collection_id not in self._collected:
            return False
        for _ in range(count):
            self._dish_counter += 1
            dish_id = f"dish_{self._dish_counter:04d}"
            self._collected[collection_id].append(f"{dish_type}:{dish_id}")
        return True

    def finalize_collection(self, collection_id: str) -> Dict[str, int]:
        """完成回收，返回分类统计"""
        if collection_id not in self._collected:
            return {}
        result = {}
        for dish in self._collected[collection_id]:
            dtype = dish.split(':')[0]
            result[dtype] = result.get(dtype, 0) + 1
        self._collected[collection_id] = []
        return result

    def get_pending_collections(self) -> List[str]:
        """获取待处理回收"""
        return [
            cid for cid, dishes in self._collected.items()
            if len(dishes) > 0
        ]


class OrderTracker:
    """订单追踪器"""

    def __init__(self):
        self._orders: Dict[str, Dict] = {}
        self._order_counter = 0
        self._counter_locked = False

    def create_order(
        self,
        table_id: str,
        order_type: OrderType = OrderType.DINE_IN,
        customer_count: int = 1,
    ) -> str:
        """创建订单"""
        self._order_counter += 1
        order_id = f"ord_{self._order_counter:05d}"
        self._orders[order_id] = {
            'order_id': order_id,
            'table_id': table_id,
            'order_type': order_type,
            'customer_count': customer_count,
            'items': [],
            'status': 'created',
            'created_at': time.time(),
            'kitchen_ready_at': None,
            'picked_up_at': None,
            'delivered_at': None,
            'completed_at': None,
            'total_value': 0.0,
        }
        return order_id

    def add_item(self, order_id: str, item_name: str, quantity: int, price: float, category: FoodCategory) -> bool:
        """添加订单项"""
        if order_id not in self._orders:
            return False
        order = self._orders[order_id]
        order['items'].append({
            'name': item_name,
            'quantity': quantity,
            'price': price,
            'category': category,
            'ready': False,
        })
        order['total_value'] += price * quantity
        return True

    def mark_kitchen_ready(self, order_id: str) -> bool:
        """标记厨房已备好"""
        if order_id not in self._orders:
            return False
        self._orders[order_id]['kitchen_ready_at'] = time.time()
        self._orders[order_id]['status'] = 'ready_for_pickup'
        return True

    def mark_picked_up(self, order_id: str) -> bool:
        """标记已取餐"""
        if order_id not in self._orders:
            return False
        self._orders[order_id]['picked_up_at'] = time.time()
        self._orders[order_id]['status'] = 'in_delivery'
        return True

    def mark_delivered(self, order_id: str) -> bool:
        """标记已送达"""
        if order_id not in self._orders:
            return False
        self._orders[order_id]['delivered_at'] = time.time()
        self._orders[order_id]['status'] = 'delivered'
        return True

    def mark_completed(self, order_id: str, rating: Optional[int] = None) -> bool:
        """标记订单完成"""
        if order_id not in self._orders:
            return False
        self._orders[order_id]['completed_at'] = time.time()
        self._orders[order_id]['status'] = 'completed'
        if rating is not None:
            self._orders[order_id]['rating'] = rating
        return True

    def get_order(self, order_id: str) -> Optional[Dict]:
        """获取订单"""
        return self._orders.get(order_id)

    def get_ready_orders(self) -> List[Dict]:
        """获取待取订单"""
        return [
            o for o in self._orders.values()
            if o['status'] == 'ready_for_pickup'
        ]

    def get_delivery_time(self, order_id: str) -> Optional[float]:
        """计算配送时间(秒)"""
        if order_id not in self._orders:
            return None
        o = self._orders[order_id]
        if o['picked_up_at'] and o['delivered_at']:
            return o['delivered_at'] - o['picked_up_at']
        return None


class HygieneMonitor:
    """餐厅卫生监控器"""

    HYGIENE_STANDARDS = {
        RestaurantZone.KITCHEN: 95,     # 后厨要求最高
        RestaurantZone.DINING_AREA: 85,
        RestaurantZone.RESTROOM: 90,
        RestaurantZone.BUFFET: 95,
        RestaurantZone.DRINKS_STATION: 90,
    }

    def __init__(self):
        self._sanitation_scores: Dict[RestaurantZone, float] = {}
        self._cleaning_schedules: Dict[RestaurantZone, float] = {}
        self._hygiene_alerts: List[Dict] = []
        for zone in RestaurantZone:
            self._sanitation_scores[zone] = 100.0
            self._cleaning_schedules[zone] = time.time()

    def update_sanitation(self, zone: RestaurantZone, score: float) -> None:
        """更新卫生评分"""
        self._sanitation_scores[zone] = max(0.0, min(100.0, score))
        if score < self.HYGIENE_STANDARDS.get(zone, 80):
            self._hygiene_alerts.append({
                'zone': zone,
                'score': score,
                'standard': self.HYGIENE_STANDARDS.get(zone, 80),
                'timestamp': time.time(),
            })

    def get_sanitation_score(self, zone: RestaurantZone) -> float:
        """获取卫生评分"""
        return self._sanitation_scores.get(zone, 100.0)

    def get_decontamination_status(self, zone: RestaurantZone) -> Dict:
        """获取消毒状态"""
        elapsed = time.time() - self._cleaning_schedules.get(zone, 0)
        return {
            'zone': zone.name,
            'score': self._sanitation_scores.get(zone, 100.0),
            'standard': self.HYGIENE_STANDARDS.get(zone, 80),
            'cleaning_interval_s': elapsed,
            'needs_cleaning': elapsed > 1800,   # 30分钟需清洁
        }

    def schedule_cleaning(self, zone: RestaurantZone) -> None:
        """安排清洁"""
        self._cleaning_schedules[zone] = time.time()

    def get_alerts(self) -> List[Dict]:
        """获取卫生警报"""
        return list(self._hygiene_alerts[-10:])

    def check_food_temperature(self, temp: float, required: FoodTemperature) -> Tuple[bool, str]:
        """检查食物温度是否符合要求"""
        if required == FoodTemperature.HOT_KEEP:
            ok = temp >= 60.0
            return ok, "OK" if ok else f"温度{temp}°C低于60°C标准"
        elif required == FoodTemperature.COLD_CHILLED:
            ok = 0 <= temp <= 4.0
            return ok, "OK" if ok else f"温度{temp}°C超出0-4°C冷藏标准"
        elif required == FoodTemperature.FROZEN:
            ok = temp <= -18.0
            return ok, "OK" if ok else f"温度{temp}°C高于-18°C冷冻标准"
        elif required == FoodTemperature.GRILL_HOT:
            ok = temp >= 70.0
            return ok, "OK" if ok else f"温度{temp}°C低于70°C标准"
        return True, "OK"


class MenuManager:
    """菜单管理器"""

    def __init__(self):
        self._menu_items: Dict[str, Dict] = {}
        self._item_counter = 0

    def add_item(
        self,
        name: str,
        category: FoodCategory,
        temperature: FoodTemperature,
        price: float,
        prep_time: float = 60.0,
        is_available: bool = True,
    ) -> str:
        """添加菜单项"""
        self._item_counter += 1
        item_id = f"menu_{self._item_counter:04d}"
        self._menu_items[item_id] = {
            'item_id': item_id,
            'name': name,
            'category': category,
            'temperature': temperature,
            'price': price,
            'prep_time': prep_time,
            'is_available': is_available,
        }
        return item_id

    def get_item(self, item_id: str) -> Optional[Dict]:
        """获取菜单项"""
        return self._menu_items.get(item_id)

    def set_availability(self, item_id: str, available: bool) -> bool:
        """设置可用性"""
        if item_id not in self._menu_items:
            return False
        self._menu_items[item_id]['is_available'] = available
        return True

    def get_by_category(self, category: FoodCategory) -> List[Dict]:
        """按类别获取菜品"""
        return [
            item for item in self._menu_items.values()
            if item['category'] == category and item['is_available']
        ]


class RestaurantSceneController:
    """
    餐厅场景控制器
    
    协调餐厅环境中的所有AGV具身智能任务:
    - 食物从厨房到餐桌的配送
    - 餐具回收和餐桌清理
    - 饮品补充服务
    - 自助餐台管理
    - 卫生监控
    """

    def __init__(
        self,
        agv_grade: str = "M",
        enable_hygiene_monitor: bool = True,
        enable_order_tracker: bool = True,
    ):
        self.agv_grade = agv_grade
        self._table_manager = TableManager()
        self._dish_collector = DishCollector()
        self._order_tracker = OrderTracker()
        self._hygiene_monitor = HygieneMonitor()
        self._menu_manager = MenuManager()
        self._task_queue: List[RestaurantTask] = []
        self._completed_tasks: List[RestaurantTask] = []
        self._agv_positions: Dict[str, RestaurantZone] = {}

        # 初始化默认餐桌
        for i in range(1, 13):
            zone = RestaurantZone.DINING_AREA
            if i <= 4:
                zone = RestaurantZone.DINING_AREA
            elif i <= 8:
                zone = RestaurantZone.BAR
            else:
                zone = RestaurantZone.BUFFET
            self._table_manager.create_table(capacity=4, zone=zone)

        self._init_menu()

    def _init_menu(self):
        """初始化默认菜单"""
        menu_data = [
            ('宫保鸡丁', FoodCategory.MAIN_DISH, FoodTemperature.HOT_KEEP, 38.0),
            ('麻婆豆腐', FoodCategory.MAIN_DISH, FoodTemperature.HOT_KEEP, 32.0),
            ('清蒸鲈鱼', FoodCategory.MAIN_DISH, FoodTemperature.HOT_KEEP, 68.0),
            ('凉拌黄瓜', FoodCategory.COLD_FOOD, FoodTemperature.ROOM_TEMP, 18.0),
            ('番茄蛋汤', FoodCategory.SOUP, FoodTemperature.HOT_KEEP, 22.0),
            ('米饭', FoodCategory.BREAD, FoodTemperature.HOT_KEEP, 4.0),
            ('橙汁', FoodCategory.BEVERAGE, FoodTemperature.ROOM_TEMP, 12.0),
            ('可乐', FoodCategory.BEVERAGE, FoodTemperature.COLD_CHILLED, 8.0),
            ('提拉米苏', FoodCategory.DESSERT, FoodTemperature.COLD_CHILLED, 28.0),
            ('水果拼盘', FoodCategory.FRUIT, FoodTemperature.COLD_CHILLED, 32.0),
        ]
        for name, cat, temp, price in menu_data:
            self._menu_manager.add_item(name, cat, temp, price)

    # ----- 场景配置 -----
    def get_scene_zones(self) -> List[RestaurantZone]:
        """获取场景区域列表"""
        return list(RestaurantZone)

    def get_zone_distance(self, zone_a: RestaurantZone, zone_b: RestaurantZone) -> float:
        """估算两区域间的距离(米)"""
        distances = {
            (RestaurantZone.KITCHEN, RestaurantZone.CARRIER_AREA): 2.0,
            (RestaurantZone.CARRIER_AREA, RestaurantZone.DINING_AREA): 8.0,
            (RestaurantZone.CARRIER_AREA, RestaurantZone.BAR): 6.0,
            (RestaurantZone.DINING_AREA, RestaurantZone.RESTROOM): 12.0,
            (RestaurantZone.ENTRANCE, RestaurantZone.DINING_AREA): 5.0,
            (RestaurantZone.TAKEOUT_AREA, RestaurantZone.KITCHEN): 4.0,
        }
        for (za, zb), d in distances.items():
            if (za == zone_a and zb == zone_b) or (za == zone_b and zb == zone_a):
                return d
        return 10.0   # 默认10米

    def estimate_delivery_time(self, source: RestaurantZone, dest: RestaurantZone) -> float:
        """估算配送时间(秒)"""
        distance = self.get_zone_distance(source, dest)
        speeds = {'S': 0.5, 'M': 0.8, 'L': 1.2, 'XL': 1.5, 'XXL': 2.0}
        speed = speeds.get(self.agv_grade, 1.0)
        return distance / speed + 10.0   # 加上装卸时间

    # ----- 任务管理 -----
    def add_task(self, task: RestaurantTask) -> None:
        """添加餐厅任务"""
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)

    def create_food_delivery_task(
        self,
        table_id: str,
        order_id: str,
        food_categories: List[FoodCategory],
        temperature: FoodTemperature,
        order_type: OrderType = OrderType.DINE_IN,
    ) -> RestaurantTask:
        """创建食物配送任务"""
        source = RestaurantZone.CARRIER_AREA
        dest = RestaurantZone.DINING_AREA
        if order_type == OrderType.TAKEOUT:
            dest = RestaurantZone.TAKEOUT_AREA

        time_limit = self.estimate_delivery_time(source, dest) * 1.2
        task = RestaurantTask(
            task_id=f"rfd_{uuid.uuid4().hex[:8]}",
            task_type='food_delivery',
            priority=5 if temperature == FoodTemperature.HOT_KEEP else 3,
            source_zone=source,
            destination_zone=dest,
            table_id=table_id,
            order_id=order_id,
            food_categories=food_categories,
            food_temperature=temperature,
            order_type=order_type,
            time_constraint=time_limit,
        )
        self.add_task(task)
        return task

    def create_dish_collection_task(
        self,
        table_id: str,
    ) -> RestaurantTask:
        """创建餐具回收任务"""
        task = RestaurantTask(
            task_id=f"rdc_{uuid.uuid4().hex[:8]}",
            task_type='dish_collection',
            priority=2,
            source_zone=RestaurantZone.DINING_AREA,
            destination_zone=RestaurantZone.KITCHEN,
            table_id=table_id,
        )
        self.add_task(task)
        return task

    def get_next_task(self) -> Optional[RestaurantTask]:
        """获取下一个最高优先级任务"""
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    def complete_task(self, task_id: str, rating: Optional[int] = None) -> bool:
        """标记任务完成"""
        for task in self._completed_tasks:
            if task.task_id == task_id:
                task.completed_at = time.time()
                task.status = 'completed'
                if rating is not None:
                    task.rating = rating
                return True
        for task in self._task_queue:
            if task.task_id == task_id:
                task.completed_at = time.time()
                task.status = 'completed'
                self._completed_tasks.append(task)
                return True
        return False

    # ----- AGV位置管理 -----
    def register_agv(self, agv_id: str, zone: RestaurantZone) -> None:
        """注册AGV位置"""
        self._agv_positions[agv_id] = zone

    def get_agv_zone(self, agv_id: str) -> Optional[RestaurantZone]:
        """获取AGV当前区域"""
        return self._agv_positions.get(agv_id)

    # ----- 场景状态 -----
    def get_scene_status(self) -> Dict:
        """获取场景状态"""
        return {
            'pending_tasks': len(self._task_queue),
            'completed_tasks': len(self._completed_tasks),
            'ready_orders': len(self._order_tracker.get_ready_orders()),
            'hygiene_alerts': len(self._hygiene_monitor.get_alerts()),
            'agv_count': len(self._agv_positions),
            'agv_grade': self.agv_grade,
        }

    def generate_scene_report(self) -> Dict:
        """生成场景报告"""
        total = len(self._completed_tasks)
        rated = [t for t in self._completed_tasks if t.rating is not None]
        avg_rating = sum(t.rating for t in rated) / len(rated) if rated else 0.0
        timely = sum(
            1 for t in self._completed_tasks
            if t.time_constraint and t.completed_at
            and (t.completed_at - t.created_at) <= t.time_constraint
        )
        return {
            'timestamp': time.time(),
            'total_deliveries': total,
            'on_time_rate': timely / total if total > 0 else 0.0,
            'avg_rating': avg_rating,
            'task_breakdown': self._get_task_breakdown(),
            'zone_hygiene': {
                zone.name: self._hygiene_monitor.get_sanitation_score(zone)
                for zone in RestaurantZone
            },
        }

    def _get_task_breakdown(self) -> Dict:
        result = {}
        for t in self._completed_tasks:
            result[t.task_type] = result.get(t.task_type, 0) + 1
        return result

    # ----- 访问器 -----
    @property
    def table_manager(self) -> TableManager:
        return self._table_manager

    @property
    def dish_collector(self) -> DishCollector:
        return self._dish_collector

    @property
    def order_tracker(self) -> OrderTracker:
        return self._order_tracker

    @property
    def hygiene_monitor(self) -> HygieneMonitor:
        return self._hygiene_monitor

    @property
    def menu_manager(self) -> MenuManager:
        return self._menu_manager


# ============================================================
# 全局单例
# ============================================================

_restaurant_controller: Optional[RestaurantSceneController] = None


def get_restaurant_scene_controller(agv_grade: str = "M") -> RestaurantSceneController:
    """获取餐厅场景控制器全局实例"""
    global _restaurant_controller
    if _restaurant_controller is None:
        _restaurant_controller = RestaurantSceneController(agv_grade=agv_grade)
    return _restaurant_controller
