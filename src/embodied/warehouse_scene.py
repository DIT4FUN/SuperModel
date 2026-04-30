# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
warehouse_scene.py - 仓储物流场景化具身智能
=============================================

适用场景: AGV大型仓储中心 / 智能物流仓库 / 电商履约中心
功能模块:
  - WarehouseZone: 仓储分区管理 (入库区/出库区/存储区/分拣区/打包区/充电区)
  - ShelfManager: 货架生命周期管理 (入库/上架/移位/盘点/出库)
  - PickTaskManager: 拣货任务调度 (波次拣货/分区拣货/鱼骨拣货/货到人)
  - InventoryTracker: 库存实时追踪 (批次/LOT/效期/库位)
  - ConveyorSystem: 输送线协同 (皮带/滚筒/交叉带/推板)
  - DockDoorManager: 月台门管理 (装卸车/等候队列/流量控制)
  - WorkerSafetyMonitor: 作业人员安全 (电子围栏/人员检测/紧急制动)
  - ThroughputAnalyzer: 吞吐分析 (UPPH/等待时间/瓶颈识别)

AGV五级规格适配:
  - Grade-S: 小型仓储, 单AGV, 最多50个货位
  - Grade-M: 中型分拣中心, 2-3台AGV, 最多200个货位
  - Grade-L: 大型电商仓库, 5-10台AGV, 最多1000个货位
  - Grade-XL: 巨型物流中心, 10-30台AGV, 最多5000个货位
  - Grade-XXL: 超大型自动化立体仓, 30+台AGV, 10000+货位

Author: SuperModel Development Team
Version: 3.15.0
"""

from __future__ import annotations

import heapq
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WarehouseZone(Enum):
    """仓储分区类型"""
    RECEIVING = "receiving"        # 入库卸货区
    STORAGE = "storage"            # 存储区
    PICKING = "picking"            # 拣货区
    SORTING = "sorting"            # 分拣区
    PACKING = "packing"            # 打包区
    SHIPPING = "shipping"          # 出库装车区
    CHARGING = "charging"          # 充电区
    MAINTENANCE = "maintenance"    # 维护区
    COLD_STORAGE = "cold_storage" # 冷链区
    HAZMAT = "hazmat"              # 危险品区

class ShelfType(Enum):
    """货架类型"""
    STANDARD = "standard"          # 标准货架
    DRIVE_IN = "drive_in"         # 驶入式货架
    FLOW_RACK = "flow_rack"        # 重力流利货架
    MEZZANINE = "mezzanine"        # 阁楼货架
    CAROUSEL = "carousel"          # 旋转货架
    AUTOMATED = "automated"        # 自动化立体仓库

class PickStrategy(Enum):
    """拣货策略"""
    WAVE = "wave"                  # 波次拣货 (按批次集中)
    ZONE = "zone"                 # 分区拣货 (AGV固定区域)
    FISHBONE = "fishbone"         # 鱼骨拣货 (S型路线)
    BATCH = "batch"               # 批量拣货 (同类品合并)
    GOODS_TO_PERSON = "gtp"       # 货到人 (AGV搬运货架到工作站)

class InventoryStatus(Enum):
    """库存状态"""
    AVAILABLE = "available"       # 可用
    RESERVED = "reserved"          # 已预留
    QUARANTINE = "quarantine"      # 隔离/待检
    DAMAGED = "damaged"            # 损坏
    EXPIRED = "expired"            # 过期
    IN_TRANSIT = "in_transit"      # 运输中

class TaskPriority(Enum):
    """仓库任务优先级"""
    URGENT = 1                     # 紧急 (缺货/投诉)
    HIGH = 2                       # 高优先
    NORMAL = 3                     # 普通
    LOW = 4                        # 低优先

class AGVLoadState(Enum):
    """AGV负载状态"""
    IDLE = "idle"                  # 空闲
    LOADING = "loading"            # 装货中
    UNLOADING = "unloading"        # 卸货中
    IN_TRANSIT_FULL = "full"       # 重载运输中
    IN_TRANSIT_EMPTY = "empty"     # 空载运输中
    CHARGING = "charging"          # 充电中
    MAINTENANCE = "maintenance"    # 维护中

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SKU:
    """商品SKU"""
    sku_id: str
    name: str
    category: str
    weight_kg: float
    dimensions_m: Tuple[float, float, float]  # L x W x H
    is_hazmat: bool = False
    requires_cold: bool = False
    batch_tracking: bool = True
    min_stock: int = 10
    max_stock: int = 1000

@dataclass
class InventoryItem:
    """库存物品"""
    sku: SKU
    quantity: int
    location_id: str
    batch_id: Optional[str] = None
    lot_number: Optional[str] = None
    expiry_date: Optional[float] = None  # timestamp
    status: InventoryStatus = InventoryStatus.AVAILABLE
    inbound_time: float = field(default_factory=time.time)

@dataclass
class Location:
    """库位"""
    location_id: str
    zone: WarehouseZone
    position: Tuple[float, float, float]
    shelf_id: str
    level: int = 1  # 货架层
    capacity_kg: float = 1000.0
    occupied_weight_kg: float = 0.0
    is_occupied: bool = False
    reserved_by: Optional[str] = None  # task_id

@dataclass
class Shelf:
    """货架"""
    shelf_id: str
    shelf_type: ShelfType
    zone: WarehouseZone
    position: Tuple[float, float, float]
    levels: int
    locations_per_level: int
    locations: Dict[str, Location] = field(default_factory=dict)
    max_load_kg: float = 2000.0
    current_load_kg: float = 0.0

@dataclass
class PickTask:
    """拣货任务"""
    task_id: str
    order_id: str
    priority: TaskPriority
    sku_items: List[Tuple[SKU, int, str]]  # SKU, quantity, location_id
    created_time: float = field(default_factory=time.time)
    assigned_agv_id: Optional[str] = None
    status: str = "pending"  # pending/assigned/picking/completed/cancelled
    wave_id: Optional[str] = None
    zone_lock: Set[WarehouseZone] = field(default_factory=set)
    estimated_pick_time_s: float = 0.0
    actual_pick_time_s: float = 0.0
    picked_items: List[Tuple[str, int]] = field(default_factory=list)  # sku_id, qty
    sequence: List[str] = field(default_factory=list)  # 拣货顺序 location_ids

@dataclass
class Wave:
    """拣货波次"""
    wave_id: str
    task_ids: List[str]
    created_time: float = field(default_factory=time.time)
    strategy: PickStrategy = PickStrategy.WAVE
    status: str = "forming"  # forming/released/picking/completed
    release_time: Optional[float] = None

@dataclass
class ConveyorSegment:
    """输送线段"""
    segment_id: str
    from_zone: WarehouseZone
    to_zone: WarehouseZone
    length_m: float
    width_m: float
    speed_m_per_s: float
    max_load_kg: float
    occupied: bool = False
    item_on_line: Optional[str] = None  # item_id

@dataclass
class DockDoor:
    """月台门"""
    door_id: str
    door_type: str  # inbound/outbound
    zone: WarehouseZone
    position: Tuple[float, float]
    status: str = "available"  # available/loading/unloading/maintenance
    truck_id: Optional[str] = None
    current_task_id: Optional[str] = None
    avg_loading_time_s: float = 600.0
    wait_queue: List[str] = field(default_factory=list)  # truck_ids

@dataclass
class WorkerSafetyEvent:
    """作业人员安全事件"""
    event_id: str
    event_type: str  # zone_entry/zone_exit/emergency_stop/speed_violation/proximity_alert
    agv_id: str
    worker_id: Optional[str]
    zone: WarehouseZone
    position: Tuple[float, float]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None

@dataclass
class ThroughputMetrics:
    """吞吐指标"""
    timestamp: float = field(default_factory=time.time)
    orders_per_hour: float = 0.0
    lines_per_hour: float = 0.0
    units_per_hour: float = 0.0
    upph: float = 0.0  # Units Per Person Hour
    avg_pick_time_s: float = 0.0
    avg_travel_time_s: float = 0.0
    agv_utilization: float = 0.0  # 0.0-1.0
    conveyor_utilization: float = 0.0
    dock_door_utilization: float = 0.0
    queue_depth: int = 0
    bottleneck_zone: Optional[WarehouseZone] = None

# ---------------------------------------------------------------------------
# ShelfManager
# ---------------------------------------------------------------------------

class ShelfManager:
    """货架生命周期管理器"""

    def __init__(self, grade: str = "M", initial_shelves: int = 50):
        self.grade = grade
        self.shelves: Dict[str, Shelf] = {}
        self.locations: Dict[str, Location] = {}
        self._shelf_counter = 0
        self._location_counter = 0
        self._initialize_shelves(initial_shelves)

    def _initialize_shelves(self, count: int) -> None:
        """初始化货架"""
        rows = int(math.sqrt(count)) + 1
        for row in range(rows):
            for col in range(rows):
                if len(self.shelves) >= count:
                    return
                shelf_id = f"SH{self._shelf_counter:04d}"
                zone = WarehouseZone.STORAGE
                if row % 3 == 0:
                    zone = WarehouseZone.PICKING
                shelf = Shelf(
                    shelf_id=shelf_id,
                    shelf_type=ShelfType.STANDARD,
                    zone=zone,
                    position=(row * 3.0, col * 2.0, 0.0),
                    levels=4,
                    locations_per_level=5,
                )
                # 创建库位
                for level in range(1, shelf.levels + 1):
                    for pos in range(shelf.locations_per_level):
                        loc_id = f"{shelf_id}-L{level}P{pos+1}"
                        loc = Location(
                            location_id=loc_id,
                            zone=zone,
                            position=(shelf.position[0], shelf.position[1], level * 0.5),
                            shelf_id=shelf_id,
                            level=level,
                        )
                        shelf.locations[loc_id] = loc
                        self.locations[loc_id] = loc
                self.shelves[shelf_id] = shelf
                self._shelf_counter += 1

    def get_available_location(
        self,
        zone: Optional[WarehouseZone] = None,
        required_capacity_kg: float = 0.0,
    ) -> Optional[Location]:
        """获取可用库位"""
        for loc in self.locations.values():
            if loc.is_occupied or loc.reserved_by:
                continue
            if zone and loc.zone != zone:
                continue
            if loc.capacity_kg - loc.occupied_weight_kg < required_capacity_kg:
                continue
            return loc
        return None

    def reserve_location(self, location_id: str, task_id: str) -> bool:
        """预留库位"""
        loc = self.locations.get(location_id)
        if loc and not loc.is_occupied and not loc.reserved_by:
            loc.reserved_by = task_id
            return True
        return False

    def occupy_location(self, location_id: str, weight_kg: float) -> bool:
        """占用库位"""
        loc = self.locations.get(location_id)
        if not loc:
            return False
        if loc.reserved_by:
            loc.reserved_by = None
        loc.is_occupied = True
        loc.occupied_weight_kg += weight_kg
        return True

    def release_location(self, location_id: str) -> bool:
        """释放库位"""
        loc = self.locations.get(location_id)
        if not loc:
            return False
        loc.is_occupied = False
        loc.reserved_by = None
        return True

    def get_shelf_load(self, shelf_id: str) -> float:
        """获取货架当前负载"""
        shelf = self.shelves.get(shelf_id)
        if not shelf:
            return 0.0
        return sum(loc.occupied_weight_kg for loc in shelf.locations.values())

    def get_zone_capacity(self, zone: WarehouseZone) -> Tuple[float, float]:
        """获取分区容量 (used, total)"""
        total = 0.0
        used = 0.0
        for loc in self.locations.values():
            if loc.zone == zone:
                total += loc.capacity_kg
                used += loc.occupied_weight_kg
        return used, total

    def get_inventory_distribution(self) -> Dict[WarehouseZone, int]:
        """获取库存分布"""
        dist = {z: 0 for z in WarehouseZone}
        for loc in self.locations.values():
            if loc.is_occupied:
                dist[loc.zone] += 1
        return dist

# ---------------------------------------------------------------------------
# PickTaskManager
# ---------------------------------------------------------------------------

class PickTaskManager:
    """拣货任务调度器"""

    def __init__(self, strategy: PickStrategy = PickStrategy.ZONE):
        self.strategy = strategy
        self.tasks: Dict[str, PickTask] = {}
        self.waves: Dict[str, Wave] = {}
        self._task_counter = 0
        self._wave_counter = 0
        self.pending_queue: List[Tuple[int, str]] = []  # priority, task_id (heap)
        self.completed_count = 0
        self.cancelled_count = 0

    def create_pick_task(
        self,
        order_id: str,
        sku_items: List[Tuple[SKU, int, str]],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> PickTask:
        """创建拣货任务"""
        task_id = f"PICK{self._task_counter:06d}"
        self._task_counter += 1
        # 计算拣货顺序
        sequence = [loc_id for _, _, loc_id in sku_items]
        task = PickTask(
            task_id=task_id,
            order_id=order_id,
            priority=priority,
            sku_items=sku_items,
            sequence=sequence,
        )
        self.tasks[task_id] = task
        heapq.heappush(self.pending_queue, (priority.value, task_id))
        return task

    def create_wave(self, task_ids: List[str]) -> Wave:
        """创建拣货波次"""
        wave_id = f"WAVE{self._wave_counter:06d}"
        self._wave_counter += 1
        wave = Wave(wave_id=wave_id, task_ids=task_ids)
        self.waves[wave_id] = wave
        for tid in task_ids:
            task = self.tasks.get(tid)
            if task:
                task.wave_id = wave_id
                task.status = "wave_assigned"
        return wave

    def release_wave(self, wave_id: str) -> bool:
        """释放波次"""
        wave = self.waves.get(wave_id)
        if not wave or wave.status != "forming":
            return False
        wave.status = "released"
        wave.release_time = time.time()
        for tid in wave.task_ids:
            task = self.tasks.get(tid)
            if task:
                task.status = "pending"
                heapq.heappush(self.pending_queue, (task.priority.value, tid))
        return True

    def get_next_task(self) -> Optional[PickTask]:
        """获取下一个最高优先任务"""
        while self.pending_queue:
            _, task_id = heapq.heappop(self.pending_queue)
            task = self.tasks.get(task_id)
            if task and task.status in ("pending", "wave_assigned"):
                task.status = "assigned"
                return task
        return None

    def assign_task(self, task_id: str, agv_id: str) -> bool:
        """分配任务给AGV"""
        task = self.tasks.get(task_id)
        if not task or task.status != "pending":
            return False
        task.assigned_agv_id = agv_id
        task.status = "assigned"
        return True

    def complete_task(self, task_id: str) -> bool:
        """完成任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        task.status = "completed"
        self.completed_count += 1
        return True

    def get_pending_count(self) -> int:
        """获取待处理任务数"""
        return sum(1 for t in self.tasks.values() if t.status in ("pending", "wave_assigned"))

    def get_queue_depth_by_priority(self) -> Dict[TaskPriority, int]:
        """按优先级统计队列深度"""
        counts = {p: 0 for p in TaskPriority}
        for t in self.tasks.values():
            if t.status in ("pending", "wave_assigned"):
                counts[t.priority] += 1
        return counts

# ---------------------------------------------------------------------------
# InventoryTracker
# ---------------------------------------------------------------------------

class InventoryTracker:
    """库存实时追踪器"""

    def __init__(self):
        self.inventory: Dict[str, List[InventoryItem]] = {}  # sku_id -> items
        self.location_inventory: Dict[str, InventoryItem] = {}  # location_id -> item
        self.total_units = 0
        self.total_value = 0.0
        self.low_stock_skus: Set[str] = set()
        self.expiring_skus: Set[str] = set()

    def add_inventory(
        self,
        sku: SKU,
        quantity: int,
        location_id: str,
        batch_id: Optional[str] = None,
        expiry_date: Optional[float] = None,
    ) -> bool:
        """添加库存"""
        item = InventoryItem(
            sku=sku,
            quantity=quantity,
            location_id=location_id,
            batch_id=batch_id,
            expiry_date=expiry_date,
        )
        if sku.sku_id not in self.inventory:
            self.inventory[sku.sku_id] = []
        self.inventory[sku.sku_id].append(item)
        self.location_inventory[location_id] = item
        self.total_units += quantity
        self._check_low_stock(sku.sku_id)
        return True

    def reserve_inventory(self, sku_id: str, quantity: int) -> bool:
        """预留库存: 将指定数量标记为RESERVED, 剩余量仍计入库存水平"""
        items = self.inventory.get(sku_id, [])
        remaining = quantity
        for item in items:
            if item.status == InventoryStatus.AVAILABLE and remaining > 0:
                reserved = min(remaining, item.quantity)
                remaining -= reserved
                # 将部分或全部数量标记为RESERVED
                item.status = InventoryStatus.RESERVED
        return remaining == 0

    def get_stock_level(self, sku_id: str) -> int:
        """获取库存水平: AVAILABLE + RESERVED 数量 (排除隔离/损坏/过期)"""
        items = self.inventory.get(sku_id, [])
        return sum(
            it.quantity for it in items
            if it.status in (InventoryStatus.AVAILABLE, InventoryStatus.RESERVED)
        )

    def get_expiring_items(self, within_hours: float = 24.0) -> List[InventoryItem]:
        """获取即将过期物品"""
        threshold = time.time() + within_hours * 3600
        result = []
        for items in self.inventory.values():
            for item in items:
                if item.expiry_date and item.expiry_date <= threshold:
                    result.append(item)
        return result

    def _check_low_stock(self, sku_id: str) -> None:
        """检查低库存"""
        total = self.get_stock_level(sku_id)
        if sku_id in self.inventory and self.inventory[sku_id]:
            sku = self.inventory[sku_id][0].sku
            if total < sku.min_stock:
                self.low_stock_skus.add(sku_id)
            else:
                self.low_stock_skus.discard(sku_id)

# ---------------------------------------------------------------------------
# ConveyorSystem
# ---------------------------------------------------------------------------

class ConveyorSystem:
    """输送线协同系统"""

    def __init__(self):
        self.segments: Dict[str, ConveyorSegment] = {}
        self._segment_counter = 0
        self._item_counter = 0
        self.items_in_transit: Dict[str, Dict] = {}  # item_id -> {segment, progress, ...}

    def add_segment(
        self,
        from_zone: WarehouseZone,
        to_zone: WarehouseZone,
        length_m: float,
        speed_m_per_s: float = 0.5,
    ) -> ConveyorSegment:
        """添加输送线段"""
        seg_id = f"CONV{self._segment_counter:04d}"
        self._segment_counter += 1
        seg = ConveyorSegment(
            segment_id=seg_id,
            from_zone=from_zone,
            to_zone=to_zone,
            length_m=length_m,
            width_m=1.0,
            speed_m_per_s=speed_m_per_s,
            max_load_kg=50.0,
        )
        self.segments[seg_id] = seg
        return seg

    def load_item(self, segment_id: str, item_id: str, weight_kg: float) -> bool:
        """装载物品到输送线"""
        seg = self.segments.get(segment_id)
        if not seg or seg.occupied:
            return False
        seg.occupied = True
        seg.item_on_line = item_id
        self.items_in_transit[item_id] = {
            "segment_id": segment_id,
            "progress": 0.0,
            "weight_kg": weight_kg,
            "start_time": time.time(),
        }
        return True

    def tick(self, dt: float) -> Dict[str, List[str]]:
        """推进输送线状态 (返回完成传送的item_ids)"""
        completed = []
        for item_id, state in list(self.items_in_transit.items()):
            seg = self.segments.get(state["segment_id"])
            if not seg:
                continue
            progress_delta = seg.speed_m_per_s * dt / seg.length_m
            state["progress"] += progress_delta
            if state["progress"] >= 1.0:
                seg.occupied = False
                seg.item_on_line = None
                completed.append(item_id)
                del self.items_in_transit[item_id]
        return {"completed": completed}

    def get_utilization(self) -> float:
        """获取输送线利用率"""
        if not self.segments:
            return 0.0
        return sum(1 for s in self.segments.values() if s.occupied) / len(self.segments)

# ---------------------------------------------------------------------------
# DockDoorManager
# ---------------------------------------------------------------------------

class DockDoorManager:
    """月台门管理器"""

    def __init__(self, inbound_doors: int = 4, outbound_doors: int = 4):
        self.doors: Dict[str, DockDoor] = {}
        self._door_counter = 0
        self._initialize_doors(inbound_doors, outbound_doors)
        self.arrival_history: List[Dict] = []

    def _initialize_doors(self, inbound: int, outbound: int) -> None:
        for i in range(inbound):
            door_id = f"DOCK_IN_{i+1:02d}"
            self.doors[door_id] = DockDoor(
                door_id=door_id,
                door_type="inbound",
                zone=WarehouseZone.RECEIVING,
                position=(0.0, i * 5.0),
            )
        for i in range(outbound):
            door_id = f"DOCK_OUT_{i+1:02d}"
            self.doors[door_id] = DockDoor(
                door_id=door_id,
                door_type="outbound",
                zone=WarehouseZone.SHIPPING,
                position=(50.0, i * 5.0),
            )
        self._door_counter = inbound + outbound

    def request_door(self, truck_id: str, door_type: str) -> Optional[str]:
        """申请月台门"""
        for door in self.doors.values():
            if door.door_type == door_type and door.status == "available":
                door.status = "reserved"
                door.truck_id = truck_id
                return door.door_id
        # 加入等待队列
        for door in self.doors.values():
            if door.door_type == door_type:
                if truck_id not in door.wait_queue:
                    door.wait_queue.append(truck_id)
                return None
        return None

    def start_loading(self, door_id: str) -> bool:
        """开始装卸作业"""
        door = self.doors.get(door_id)
        if not door or door.status != "reserved":
            return False
        door.status = "loading" if door.door_type == "inbound" else "unloading"
        return True

    def release_door(self, door_id: str) -> bool:
        """释放月台门"""
        door = self.doors.get(door_id)
        if not door:
            return False
        self.arrival_history.append({
            "door_id": door_id,
            "truck_id": door.truck_id,
            "duration_s": time.time() - (door.current_task_id or 0),
        })
        door.status = "available"
        door.truck_id = None
        door.current_task_id = None
        # 唤醒等待队列
        if door.wait_queue:
            next_truck = door.wait_queue.pop(0)
            door.status = "reserved"
            door.truck_id = next_truck
        return True

    def get_available_count(self, door_type: str) -> int:
        """获取可用门数"""
        return sum(1 for d in self.doors.values() if d.door_type == door_type and d.status == "available")

# ---------------------------------------------------------------------------
# WorkerSafetyMonitor
# ---------------------------------------------------------------------------

class WorkerSafetyMonitor:
    """作业人员安全监控器"""

    def __init__(self, safety_distance_m: float = 2.0):
        self.safety_distance_m = safety_distance_m
        self.events: List[WorkerSafetyEvent] = []
        self.active_safety_events: Dict[str, WorkerSafetyEvent] = {}
        self._event_counter = 0
        self.worker_positions: Dict[str, Tuple[float, float]] = {}  # worker_id -> (x, y)
        self.agv_positions: Dict[str, Tuple[float, float]] = {}  # agv_id -> (x, y)
        self.electronic_fence: Dict[WarehouseZone, List[Tuple[float, float, float]]] = {}  # zone -> [(x,y,r), ...]
        self.total_events = 0
        self.resolved_events = 0

    def register_worker(self, worker_id: str, position: Tuple[float, float]) -> None:
        """注册作业人员位置"""
        self.worker_positions[worker_id] = position

    def register_agv(self, agv_id: str, position: Tuple[float, float]) -> None:
        """注册AGV位置"""
        self.agv_positions[agv_id] = position

    def set_electronic_fence(self, zone: WarehouseZone, circles: List[Tuple[float, float, float]]) -> None:
        """设置电子围栏 (x, y, radius)"""
        self.electronic_fence[zone] = circles

    def check_proximity(self) -> List[WorkerSafetyEvent]:
        """检查人员-AGV接近事件"""
        new_events = []
        for agv_id, agv_pos in self.agv_positions.items():
            for worker_id, worker_pos in self.worker_positions.items():
                dist = math.sqrt((agv_pos[0] - worker_pos[0])**2 + (agv_pos[1] - worker_pos[1])**2)
                if dist < self.safety_distance_m:
                    event_id = f"PROX{self._event_counter:06d}"
                    self._event_counter += 1
                    zone = self._get_zone_at_position(agv_pos)
                    event = WorkerSafetyEvent(
                        event_id=event_id,
                        event_type="proximity_alert",
                        agv_id=agv_id,
                        worker_id=worker_id,
                        zone=zone,
                        position=agv_pos,
                    )
                    new_events.append(event)
                    self.events.append(event)
                    self.active_safety_events[event_id] = event
                    self.total_events += 1
        return new_events

    def check_electronic_fence(self, agv_id: str) -> List[WorkerSafetyEvent]:
        """检查AGV是否进入电子围栏禁区"""
        agv_pos = self.agv_positions.get(agv_id)
        if not agv_pos:
            return []
        events = []
        for zone, circles in self.electronic_fence.items():
            for cx, cy, radius in circles:
                dist = math.sqrt((agv_pos[0] - cx)**2 + (agv_pos[1] - cy)**2)
                if dist < radius:
                    event_id = f"FENCE{self._event_counter:06d}"
                    self._event_counter += 1
                    event = WorkerSafetyEvent(
                        event_id=event_id,
                        event_type="zone_entry",
                        agv_id=agv_id,
                        worker_id=None,
                        zone=zone,
                        position=agv_pos,
                    )
                    events.append(event)
                    self.events.append(event)
                    self.active_safety_events[event_id] = event
                    self.total_events += 1
        return events

    def resolve_event(self, event_id: str, resolution: str) -> bool:
        """解决安全事件"""
        event = self.active_safety_events.get(event_id)
        if not event:
            return False
        event.resolved = True
        event.resolution = resolution
        del self.active_safety_events[event_id]
        self.resolved_events += 1
        return True

    def get_active_alert_count(self) -> int:
        """获取活跃告警数"""
        return len(self.active_safety_events)

    def _get_zone_at_position(self, pos: Tuple[float, float]) -> WarehouseZone:
        """根据位置判断所在分区"""
        x, y = pos
        if x < 10:
            return WarehouseZone.RECEIVING
        elif x > 40:
            return WarehouseZone.SHIPPING
        elif 10 <= x < 20:
            return WarehouseZone.PICKING
        else:
            return WarehouseZone.STORAGE

# ---------------------------------------------------------------------------
# ThroughputAnalyzer
# ---------------------------------------------------------------------------

class ThroughputAnalyzer:
    """仓储吞吐分析器"""

    def __init__(self, window_size_s: float = 300.0):
        self.window_size_s = window_size_s
        self.order_completion_times: List[float] = []
        self.pick_times: List[float] = []
        self.travel_times: List[float] = []
        self.agv_positions_history: List[Dict] = []
        self._order_counter = 0
        self._start_time = time.time()

    def record_order_completion(self, timestamp: float) -> None:
        """记录订单完成"""
        self.order_completion_times.append(timestamp)

    def record_pick_time(self, duration_s: float) -> None:
        """记录拣货耗时"""
        self.pick_times.append(duration_s)

    def record_travel_time(self, duration_s: float) -> None:
        """记录行驶时间"""
        self.travel_times.append(duration_s)

    def compute_metrics(self, active_agv_count: int) -> ThroughputMetrics:
        """计算吞吐指标"""
        now = time.time()
        cutoff = now - self.window_size_s

        # 过滤最近 window_size_s 的数据
        recent_orders = [t for t in self.order_completion_times if t >= cutoff]
        recent_picks = self.pick_times[-100:]  # 最近100条
        recent_travel = self.travel_times[-100:]

        orders_per_hour = len(recent_orders) * (3600.0 / self.window_size_s) if self.window_size_s > 0 else 0.0

        metrics = ThroughputMetrics(
            timestamp=now,
            orders_per_hour=orders_per_hour,
            lines_per_hour=orders_per_hour * 5.0,  # 假设每单5行
            units_per_hour=orders_per_hour * 20.0,  # 假设每单20件
            avg_pick_time_s=np.mean(recent_picks) if recent_picks else 0.0,
            avg_travel_time_s=np.mean(recent_travel) if recent_travel else 0.0,
        )

        # UPPH (假设5个工人)
        if metrics.avg_pick_time_s > 0:
            metrics.upph = 3600.0 / metrics.avg_pick_time_s * 5.0

        # AGV利用率 (简化估算)
        metrics.agv_utilization = min(1.0, orders_per_hour / (active_agv_count * 10.0)) if active_agv_count > 0 else 0.0

        return metrics

    def get_bottleneck_zone(self, zone_counts: Dict[WarehouseZone, int]) -> Optional[WarehouseZone]:
        """识别瓶颈分区"""
        if not zone_counts:
            return None
        return max(zone_counts, key=zone_counts.get)

# ---------------------------------------------------------------------------
# WarehouseSceneController
# ---------------------------------------------------------------------------

class WarehouseSceneController:
    """仓储场景总控制器"""

    def __init__(self, grade: str = "M"):
        self.grade = grade
        self.shelf_manager = ShelfManager(grade=grade, initial_shelves=self._grade_shelves(grade))
        self.pick_task_manager = PickTaskManager()
        self.inventory_tracker = InventoryTracker()
        self.conveyor_system = ConveyorSystem()
        self.dock_door_manager = DockDoorManager()
        self.worker_safety_monitor = WorkerSafetyMonitor()
        self.throughput_analyzer = ThroughputAnalyzer()
        self.active_agvs: Dict[str, Dict] = {}
        self.scene_start_time = time.time()
        self._total_orders = 0
        self._total_picks = 0

    @staticmethod
    def _grade_shelves(grade: str) -> int:
        mapping = {"S": 20, "M": 50, "L": 100, "XL": 200, "XXL": 500}
        return mapping.get(grade, 50)

    def register_agv(self, agv_id: str, position: Tuple[float, float]) -> bool:
        """注册AGV"""
        if agv_id in self.active_agvs:
            return False
        self.active_agvs[agv_id] = {
            "agv_id": agv_id,
            "position": list(position),
            "state": AGVLoadState.IDLE,
            "current_task_id": None,
            "battery_level": 0.8,
            "load_kg": 0.0,
        }
        return True

    def update_agv_state(
        self,
        agv_id: str,
        position: Optional[Tuple[float, float]] = None,
        state: Optional[AGVLoadState] = None,
        battery_level: Optional[float] = None,
    ) -> bool:
        """更新AGV状态"""
        agv = self.active_agvs.get(agv_id)
        if not agv:
            return False
        if position:
            agv["position"] = list(position)
            self.worker_safety_monitor.register_agv(agv_id, position)
        if state:
            agv["state"] = state
        if battery_level is not None:
            agv["battery_level"] = battery_level
        return True

    def assign_pick_task(self, agv_id: str) -> Optional[PickTask]:
        """为AGV分配拣货任务"""
        agv = self.active_agvs.get(agv_id)
        if not agv or agv["state"] != AGVLoadState.IDLE:
            return None
        task = self.pick_task_manager.get_next_task()
        if task:
            agv["state"] = AGVLoadState.LOADING
            agv["current_task_id"] = task.task_id
            task.assigned_agv_id = agv_id
        return task

    def safety_tick(self) -> Dict[str, Any]:
        """运行安全检查"""
        proximity_events = self.worker_safety_monitor.check_proximity()
        for agv_id in self.active_agvs:
            fence_events = self.worker_safety_monitor.check_electronic_fence(agv_id)
        return {
            "proximity_alerts": len(proximity_events),
            "active_alerts": self.worker_safety_monitor.get_active_alert_count(),
            "total_events": self.worker_safety_monitor.total_events,
        }

    def tick(self, dt: float = 1.0) -> Dict[str, Any]:
        """主循环 tick"""
        # 输送线推进
        conveyor_result = self.conveyor_system.tick(dt)

        # 安全检查
        safety = self.safety_tick()

        # 计算吞吐指标
        metrics = self.throughput_analyzer.compute_metrics(len(self.active_agvs))

        # 检查充电需求
        low_battery_agvs = [
            agv_id for agv_id, agv in self.active_agvs.items()
            if agv["battery_level"] < 0.2 and agv["state"] not in (AGVLoadState.CHARGING, AGVLoadState.MAINTENANCE)
        ]

        return {
            "active_agvs": len(self.active_agvs),
            "pending_tasks": self.pick_task_manager.get_pending_count(),
            "conveyor_utilization": self.conveyor_system.get_utilization(),
            "available_inbound_doors": self.dock_door_manager.get_available_count("inbound"),
            "available_outbound_doors": self.dock_door_manager.get_available_count("outbound"),
            "safety_alerts": safety["active_alerts"],
            "low_battery_agvs": len(low_battery_agvs),
            "orders_per_hour": metrics.orders_per_hour,
            "upph": metrics.upph,
            "inventory_distribution": self.shelf_manager.get_inventory_distribution(),
            "conveyor_completed": len(conveyor_result.get("completed", [])),
        }

    def get_full_status(self) -> Dict[str, Any]:
        """获取完整场景状态"""
        return {
            "grade": self.grade,
            "uptime_s": time.time() - self.scene_start_time,
            "active_agvs": len(self.active_agvs),
            "total_orders": self._total_orders,
            "total_picks": self._total_picks,
            "pending_tasks": self.pick_task_manager.get_pending_count(),
            "inventory_distribution": self.shelf_manager.get_inventory_distribution(),
            "conveyor_utilization": self.conveyor_system.get_utilization(),
            "dock_status": {
                "inbound_available": self.dock_door_manager.get_available_count("inbound"),
                "outbound_available": self.dock_door_manager.get_available_count("outbound"),
            },
            "safety_events": {
                "total": self.worker_safety_monitor.total_events,
                "active": self.worker_safety_monitor.get_active_alert_count(),
                "resolved": self.worker_safety_monitor.resolved_events,
            },
            "low_stock_skus": list(self.inventory_tracker.low_stock_skus),
        }


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_warehouse_controller: Optional[WarehouseSceneController] = None

def get_warehouse_scene_controller(grade: str = "M") -> WarehouseSceneController:
    """获取仓库场景控制器单例"""
    global _warehouse_controller
    if _warehouse_controller is None:
        _warehouse_controller = WarehouseSceneController(grade=grade)
    return _warehouse_controller
