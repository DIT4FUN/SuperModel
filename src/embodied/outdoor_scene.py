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
outdoor_scene.py - 户外场景化具身智能模块
SuperModel 超模态大模型具身智能系统

户外场景专项:
- 最后一公里配送 (社区/写字楼/学校/医院)
- 园区巡逻 (工业园区/校园/社区)
- 地形适应导航 (坡道/台阶/不平地面/草地)
- 恶劣天气应对 (雨天/雪天/雾天/大风)
- GPS/北斗定位融合
- 户外广告/环境监测
- 应急物资投递
- 包裹柜/驿站对接
"""

from __future__ import annotations

import time
import logging
import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)

__all__ = [
    'OutdoorZone',
    'TerrainType',
    'WeatherCondition',
    'WeatherSeverity',
    'OutdoorTask',
    'OutdoorTaskLibrary',
    'OutdoorSceneController',
    'TerrainNavigator',
    'WeatherMonitor',
    'GPSLocalizer',
    'DeliveryConfirmation',
    'EmergencyHandler',
    'get_outdoor_scene_controller',
]


class OutdoorZone(Enum):
    """户外区域类型"""
    RESIDENTIAL = auto()        # 住宅小区
    COMMERCIAL = auto()        # 商业区
    OFFICE = auto()            # 写字楼/办公区
    CAMPUS = auto()            # 校园/园区
    HOSPITAL = auto()          # 医院
    PARK = auto()              # 公园/绿地
    STREET = auto()            # 街道
    CROSSWALK = auto()         # 人行横道
    BUILDING_ENTRANCE = auto()  # 建筑物入口
    PARKING = auto()           # 停车场
    BIKE_LANE = auto()         # 自行车道
    SIDEWALK = auto()           # 人行道
    GRAVEL_PATH = auto()       # 碎石路
    RAMP = auto()              # 坡道
    STAIR = auto()             # 台阶
    DELIVERY_LOCKER = auto()   # 快递柜
    PICKUP_STATION = auto()    # 驿站/代收点


class TerrainType(Enum):
    """地形类型"""
    FLAT_PAVEMENT = auto()     # 平坦路面
    UNEVEN_PAVEMENT = auto()   # 不平路面
    GRAVEL = auto()            # 碎石地
    GRASS = auto()             # 草地
    SAND = auto()              # 沙地
    MUD = auto()               # 泥地
    RAMPDOWN = auto()          # 下坡
    RAMP_UP = auto()           # 上坡
    STAIR_UP = auto()          # 上台阶
    STAIR_DOWN = auto()        # 下台阶
    SPEED_BUMP = auto()        # 减速带
    PUDDLE = auto()            # 水洼
    DRAIN = auto()             # 排水沟
    SIDEWALK = auto()          # 人行道


class WeatherCondition(Enum):
    """天气状况"""
    CLEAR = auto()             # 晴朗
    CLOUDY = auto()            # 多云
    RAIN_LIGHT = auto()        # 小雨
    RAIN_HEAVY = auto()        # 大雨
    RAIN_STORM = auto()        # 暴雨
    SNOW_LIGHT = auto()        # 小雪
    SNOW_HEAVY = auto()        # 大雪
    FOG = auto()               # 雾
    WIND_LIGHT = auto()        # 微风
    WIND_STRONG = auto()       # 强风
    THUNDERSTORM = auto()      # 雷暴
    HAIL = auto()              # 冰雹


class WeatherSeverity(Enum):
    """天气恶劣程度"""
    NORMAL = 0        # 正常作业
    CAUTION = 1       # 注意
    WARNING = 2       # 警告
    SUSPENDED = 3     # 暂停户外作业


@dataclass
class OutdoorTask:
    """户外任务"""
    task_id: str
    task_type: str                          # last_mile_delivery/patrol/monitoring/collection
    priority: int                           # 1-5, 5最高
    pickup_zone: OutdoorZone
    destination_zone: OutdoorZone
    pickup_location: Optional[str] = None    # GPS坐标 "lat,lng"
    destination_location: Optional[str] = None
    package_type: Optional[str] = None      # small/medium/large/fragile/hazardous
    recipient_id: Optional[str] = None
    locker_id: Optional[str] = None         # 快递柜ID
    terrain_required: List[TerrainType] = field(default_factory=list)
    weather_sensitive: bool = False
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None
    estimated_distance: float = 0.0          # 米
    created_at: float = field(default_factory=time.time)
    picked_at: Optional[float] = None
    delivered_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"                  # pending/picked/in_transit/delivered/completed/failed
    weather_at_delivery: Optional[WeatherCondition] = None
    terrain_encountered: List[TerrainType] = field(default_factory=list)
    notes: str = ""


class OutdoorTaskLibrary:
    """户外任务模板库"""

    DELIVERY_TEMPLATES = {
        'last_mile_residential': {
            'type': 'last_mile_delivery',
            'priority': 4,
            'package_types': ['small', 'medium'],
            'time_limit': 1800,    # 30分钟
            'terrain': [TerrainType.FLAT_PAVEMENT, TerrainType.SIDEWALK],
        },
        'last_mile_office': {
            'type': 'last_mile_delivery',
            'priority': 4,
            'package_types': ['small', 'medium', 'large'],
            'time_limit': 1200,    # 20分钟
            'terrain': [TerrainType.FLAT_PAVEMENT, TerrainType.SPEED_BUMP],
        },
        'fragile_delivery': {
            'type': 'last_mile_delivery',
            'priority': 5,
            'package_types': ['fragile'],
            'time_limit': 2400,    # 40分钟，fragile需要更平稳
            'terrain': [TerrainType.FLAT_PAVEMENT],
            'max_slope': 5.0,      # 最大5度坡
        },
        'parcel_locker': {
            'type': 'last_mile_delivery',
            'priority': 3,
            'package_types': ['small', 'medium'],
            'time_limit': 900,
            'terrain': [TerrainType.FLAT_PAVEMENT],
        },
    }

    PATROL_TEMPLATES = {
        'campus_patrol': {
            'type': 'patrol',
            'priority': 3,
            'zones': [OutdoorZone.CAMPUS, OutdoorZone.PARK],
            'interval': 3600,     # 每小时巡逻
            'terrain': [TerrainType.FLAT_PAVEMENT, TerrainType.GRASS],
        },
        'industrial_patrol': {
            'type': 'patrol',
            'priority': 4,
            'zones': [OutdoorZone.COMMERCIAL, OutdoorZone.PARKING],
            'interval': 1800,     # 每30分钟
            'terrain': [TerrainType.FLAT_PAVEMENT, TerrainType.GRAVEL],
        },
    }

    @classmethod
    def get_delivery_template(cls, name: str) -> Dict[str, Any]:
        return dict(cls.DELIVERY_TEMPLATES.get(name, {}))

    @classmethod
    def get_patrol_template(cls, name: str) -> Dict[str, Any]:
        return dict(cls.PATROL_TEMPLATES.get(name, {}))


class TerrainNavigator:
    """地形导航器"""

    # 各地形的导航参数
    TERRAIN_PARAMS = {
        TerrainType.FLAT_PAVEMENT:   {'speed_factor': 1.0, 'stability': 1.0, 'energy_factor': 1.0},
        TerrainType.UNEVEN_PAVEMENT: {'speed_factor': 0.7, 'stability': 0.8, 'energy_factor': 1.3},
        TerrainType.GRAVEL:          {'speed_factor': 0.5, 'stability': 0.6, 'energy_factor': 1.8},
        TerrainType.GRASS:            {'speed_factor': 0.4, 'stability': 0.5, 'energy_factor': 2.2},
        TerrainType.SAND:             {'speed_factor': 0.3, 'stability': 0.4, 'energy_factor': 2.5},
        TerrainType.MUD:             {'speed_factor': 0.2, 'stability': 0.3, 'energy_factor': 3.0},
        TerrainType.RAMPDOWN:         {'speed_factor': 0.6, 'stability': 0.7, 'energy_factor': 1.2},
        TerrainType.RAMP_UP:          {'speed_factor': 0.5, 'stability': 0.6, 'energy_factor': 1.8},
        TerrainType.STAIR_UP:         {'speed_factor': 0.2, 'stability': 0.2, 'energy_factor': 4.0},
        TerrainType.STAIR_DOWN:       {'speed_factor': 0.3, 'stability': 0.3, 'energy_factor': 3.5},
        TerrainType.SPEED_BUMP:       {'speed_factor': 0.5, 'stability': 0.5, 'energy_factor': 1.5},
        TerrainType.PUDDLE:            {'speed_factor': 0.3, 'stability': 0.4, 'energy_factor': 2.0},
        TerrainType.DRAIN:             {'speed_factor': 0.4, 'stability': 0.5, 'energy_factor': 1.6},
    }

    def __init__(self):
        self._current_terrain = TerrainType.FLAT_PAVEMENT
        self._terrain_history: List[TerrainType] = []
        self._obstacle_avoidance_count = 0

    def set_terrain(self, terrain: TerrainType) -> None:
        """设置当前地形"""
        self._current_terrain = terrain
        self._terrain_history.append(terrain)
        if len(self._terrain_history) > 100:
            self._terrain_history = self._terrain_history[-100:]

    def get_terrain(self) -> TerrainType:
        """获取当前地形"""
        return self._current_terrain

    def get_speed_factor(self, terrain: Optional[TerrainType] = None) -> float:
        """获取地形速度因子"""
        t = terrain or self._current_terrain
        return self.TERRAIN_PARAMS.get(t, {}).get('speed_factor', 1.0)

    def get_stability_factor(self, terrain: Optional[TerrainType] = None) -> float:
        """获取地形稳定性因子"""
        t = terrain or self._current_terrain
        return self.TERRAIN_PARAMS.get(t, {}).get('stability', 1.0)

    def get_energy_factor(self, terrain: Optional[TerrainType] = None) -> float:
        """获取地形能耗因子"""
        t = terrain or self._current_terrain
        return self.TERRAIN_PARAMS.get(t, {}).get('energy_factor', 1.0)

    def get_recommended_speed(self, base_speed: float, terrain: Optional[TerrainType] = None) -> float:
        """获取推荐速度"""
        t = terrain or self._current_terrain
        return base_speed * self.get_speed_factor(t)

    def is_terrain_safe(self, terrain: TerrainType) -> bool:
        """判断地形是否安全"""
        unsafe = {TerrainType.MUD, TerrainType.SAND}
        return terrain not in unsafe

    def plan_terrain_route(
        self,
        start: OutdoorZone,
        end: OutdoorZone,
        avoid_terrain: Optional[List[TerrainType]] = None,
    ) -> Dict[str, Any]:
        """规划地形路线"""
        avoid = set(avoid_terrain or [])
        route_terrains = [TerrainType.FLAT_PAVEMENT]
        if start == OutdoorZone.RESIDENTIAL and end == OutdoorZone.STREET:
            route_terrains = [TerrainType.FLAT_PAVEMENT, TerrainType.SIDEWALK, TerrainType.SPEED_BUMP]
        elif start == OutdoorZone.PARK:
            route_terrains = [TerrainType.GRASS, TerrainType.FLAT_PAVEMENT]
        elif start == OutdoorZone.PARKING:
            route_terrains = [TerrainType.GRAVEL, TerrainType.FLAT_PAVEMENT]

        safe_terrains = [t for t in route_terrains if t not in avoid]
        return {
            'terrains': route_terrains,
            'safe_terrains': safe_terrains,
            'estimated_distance_m': len(route_terrains) * 10.0,
            'estimated_time_s': sum(self.get_speed_factor(t) * 10.0 for t in route_terrains),
        }


class WeatherMonitor:
    """天气监控器"""

    def __init__(self):
        self._current_weather = WeatherCondition.CLEAR
        self._severity = WeatherSeverity.NORMAL
        self._temperature = 20.0     # °C
        self._humidity = 60.0       # %
        self._wind_speed = 0.0      # m/s
        self._visibility = 1000.0  # 米
        self._weather_history: List[Tuple[float, WeatherCondition]] = []

    def update_weather(
        self,
        weather: WeatherCondition,
        temperature: float = 20.0,
        humidity: float = 60.0,
        wind_speed: float = 0.0,
        visibility: float = 1000.0,
    ) -> None:
        """更新天气状况"""
        self._current_weather = weather
        self._temperature = temperature
        self._humidity = humidity
        self._wind_speed = wind_speed
        self._visibility = visibility
        self._weather_history.append((time.time(), weather))
        self._update_severity()

    def _update_severity(self) -> None:
        """更新恶劣程度"""
        if self._current_weather in {WeatherCondition.THUNDERSTORM, WeatherCondition.HAIL}:
            self._severity = WeatherSeverity.SUSPENDED
        elif self._current_weather in {WeatherCondition.RAIN_HEAVY, WeatherCondition.SNOW_HEAVY}:
            self._severity = WeatherSeverity.WARNING
        elif self._current_weather in {WeatherCondition.RAIN_LIGHT, WeatherCondition.SNOW_LIGHT, WeatherCondition.FOG, WeatherCondition.WIND_STRONG}:
            self._severity = WeatherSeverity.CAUTION
        elif self._current_weather == WeatherCondition.CLEAR:
            self._severity = WeatherSeverity.NORMAL
        else:
            self._severity = WeatherSeverity.CAUTION

    def get_weather(self) -> WeatherCondition:
        return self._current_weather

    def get_severity(self) -> WeatherSeverity:
        return self._severity

    def can_operate(self) -> bool:
        """判断是否可以在当前天气下作业"""
        return self._severity != WeatherSeverity.SUSPENDED

    def get_operation_recommendation(self) -> str:
        """获取操作建议"""
        if self._severity == WeatherSeverity.SUSPENDED:
            return "暂停所有户外配送任务"
        elif self._severity == WeatherSeverity.WARNING:
            return "降低速度50%，避开低洼积水区域"
        elif self._severity == WeatherSeverity.CAUTION:
            return "减速20%，注意防滑"
        else:
            return "正常作业"

    def get_wind_speed(self) -> float:
        return self._wind_speed

    def get_temperature(self) -> float:
        return self._temperature

    def get_visibility(self) -> float:
        return self._visibility

    def get_slope_limit(self) -> float:
        """根据天气获取最大坡度限制(度)"""
        if self._current_weather in {WeatherCondition.RAIN_HEAVY, WeatherCondition.RAIN_STORM}:
            return 5.0
        elif self._current_weather in {WeatherCondition.SNOW_HEAVY}:
            return 3.0
        elif self._current_weather in {WeatherCondition.SNOW_LIGHT, WeatherCondition.FOG}:
            return 8.0
        return 12.0


class GPSLocalizer:
    """GPS/北斗定位融合器"""

    def __init__(self):
        self._current_lat = 0.0
        self._current_lng = 0.0
        self._altitude = 0.0
        self._heading = 0.0
        self._accuracy = 5.0   # 米
        self._lock_count = 0
        self._lost_lock_count = 0
        self._position_history: List[Tuple[float, float, float, float]] = []  # lat, lng, accuracy, time

    def update_position(self, lat: float, lng: float, accuracy: float = 5.0) -> None:
        """更新位置"""
        self._current_lat = lat
        self._current_lng = lng
        self._accuracy = accuracy
        self._lock_count += 1
        self._position_history.append((lat, lng, accuracy, time.time()))
        if len(self._position_history) > 1000:
            self._position_history = self._position_history[-1000:]

    def update_with_imu(self, heading: float, altitude: float = 0.0) -> None:
        """使用IMU数据更新"""
        self._heading = heading
        self._altitude = altitude

    def get_position(self) -> Tuple[float, float]:
        return (self._current_lat, self._current_lng)

    def get_position_3d(self) -> Tuple[float, float, float]:
        return (self._current_lat, self._current_lng, self._altitude)

    def get_heading(self) -> float:
        return self._heading

    def get_accuracy(self) -> float:
        return self._accuracy

    def calculate_distance(self, lat2: float, lng2: float) -> float:
        """计算到目标点的球面距离(米) - Haversine公式"""
        R = 6371000.0   # 地球半径(米)
        lat1, lng1 = math.radians(self._current_lat), math.radians(self._current_lng)
        lat2_r, lng2_r = math.radians(lat2), math.radians(lng2)
        dlat = lat2_r - lat1
        dlng = lng2_r - lng1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def has_gps_lock(self) -> bool:
        return self._accuracy < 20.0

    def get_statistics(self) -> Dict:
        return {
            'total_updates': self._lock_count,
            'lost_lock_events': self._lost_lock_count,
            'avg_accuracy_m': self._accuracy,
            'lock_rate': 1.0 - (self._lost_lock_count / max(1, self._lock_count)),
        }


class DeliveryConfirmation:
    """配送确认器"""

    def __init__(self):
        self._pending_deliveries: Dict[str, Dict] = {}
        self._completed_deliveries: Dict[str, Dict] = {}
        self._failed_deliveries: Dict[str, Dict] = {}

    def initiate_delivery(
        self,
        task_id: str,
        recipient_id: Optional[str],
        locker_id: Optional[str] = None,
        time_window_end: Optional[float] = None,
    ) -> str:
        """发起配送确认"""
        confirmation_id = f"conf_{task_id}"
        self._pending_deliveries[confirmation_id] = {
            'task_id': task_id,
            'recipient_id': recipient_id,
            'locker_id': locker_id,
            'time_window_end': time_window_end,
            'initiated_at': time.time(),
            'status': 'pending',
        }
        return confirmation_id

    def confirm_recipient(self, confirmation_id: str, recipient_confirmed: bool) -> bool:
        """确认收货人"""
        if confirmation_id not in self._pending_deliveries:
            return False
        entry = self._pending_deliveries[confirmation_id]
        if recipient_confirmed:
            entry['status'] = 'confirmed'
            entry['confirmed_at'] = time.time()
            self._completed_deliveries[confirmation_id] = entry
            del self._pending_deliveries[confirmation_id]
            return True
        else:
            entry['attempts'] = entry.get('attempts', 0) + 1
        return False

    def confirm_locker_drop(self, confirmation_id: str, locker_confirmed: bool) -> bool:
        """确认快递柜投递"""
        if confirmation_id not in self._pending_deliveries:
            return False
        entry = self._pending_deliveries[confirmation_id]
        entry['locker_confirmed'] = locker_confirmed
        entry['status'] = 'locker_confirmed'
        entry['confirmed_at'] = time.time()
        self._completed_deliveries[confirmation_id] = entry
        del self._pending_deliveries[confirmation_id]
        return True

    def mark_failed(self, confirmation_id: str, reason: str) -> bool:
        """标记配送失败"""
        if confirmation_id in self._pending_deliveries:
            entry = self._pending_deliveries[confirmation_id]
            entry['status'] = 'failed'
            entry['failure_reason'] = reason
            self._failed_deliveries[confirmation_id] = entry
            del self._pending_deliveries[confirmation_id]
            return True
        return False

    def get_pending(self) -> List[Dict]:
        return list(self._pending_deliveries.values())

    def get_completion_rate(self) -> float:
        total = len(self._completed_deliveries) + len(self._failed_deliveries)
        if total == 0:
            return 1.0
        return len(self._completed_deliveries) / total


class EmergencyHandler:
    """户外紧急情况处理器"""

    def __init__(self):
        self._emergency_types = {
            'vehicle_stuck', 'battery_low', 'gps_lost', 'weather_extreme',
            'obstacle_unpassable', 'delivery_failed', 'collision',
        }
        self._active_emergencies: List[Dict] = []
        self._emergency_log: List[Dict] = []

    def report_emergency(
        self,
        emergency_type: str,
        agv_id: str,
        location: Optional[Tuple[float, float]] = None,
        details: Optional[Dict] = None,
    ) -> str:
        """报告紧急情况"""
        if emergency_type not in self._emergency_types:
            return ""
        emergency_id = f"emerg_{len(self._emergency_log):04d}"
        entry = {
            'id': emergency_id,
            'type': emergency_type,
            'agv_id': agv_id,
            'location': location,
            'details': details or {},
            'reported_at': time.time(),
            'status': 'active',
        }
        self._active_emergencies.append(entry)
        self._emergency_log.append(entry)
        return emergency_id

    def resolve_emergency(self, emergency_id: str, resolution: str) -> bool:
        """解决紧急情况"""
        for em in self._active_emergencies:
            if em['id'] == emergency_id:
                em['status'] = 'resolved'
                em['resolution'] = resolution
                em['resolved_at'] = time.time()
                self._active_emergencies.remove(em)
                return True
        return False

    def get_active_emergencies(self) -> List[Dict]:
        return list(self._active_emergencies)

    def has_active_emergency(self) -> bool:
        return len(self._active_emergencies) > 0

    def get_recovery_action(self, emergency_type: str) -> str:
        """获取恢复动作"""
        recovery_actions = {
            'vehicle_stuck': '原地后退0.5m，重新规划路线',
            'battery_low': '寻找最近充电桩或安全停车点',
            'gps_lost': '切换惯性导航，等待GPS信号恢复',
            'weather_extreme': '暂停任务，寻找最近遮蔽位置',
            'obstacle_unpassable': '重新路径规划，绕过障碍物',
            'delivery_failed': '将包裹带回驿站，等待重新派单',
            'collision': '紧急停止，扫描周围环境，安全退出',
        }
        return recovery_actions.get(emergency_type, '人工介入处理')


class OutdoorSceneController:
    """
    户外场景控制器
    
    协调户外环境中的AGV具身智能任务:
    - 最后一公里配送
    - 园区巡逻
    - 地形适应导航
    - 天气监测与规避
    - GPS/北斗精确定位
    - 紧急情况处理
    """

    def __init__(
        self,
        agv_grade: str = "L",
        enable_gps: bool = True,
        enable_weather: bool = True,
    ):
        self.agv_grade = agv_grade
        self._terrain_navigator = TerrainNavigator()
        self._weather_monitor = WeatherMonitor()
        self._gps_localizer = GPSLocalizer()
        self._delivery_confirmer = DeliveryConfirmation()
        self._emergency_handler = EmergencyHandler()
        self._task_queue: List[OutdoorTask] = []
        self._completed_tasks: List[OutdoorTask] = []
        self._agv_positions: Dict[str, Tuple[float, float]] = {}

        # AGV速度基准(米/秒)，按等级
        self._base_speeds = {'S': 0.8, 'M': 1.2, 'L': 1.8, 'XL': 2.5, 'XXL': 3.0}

    def add_task(self, task: OutdoorTask) -> None:
        """添加户外任务"""
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)

    def create_delivery_task(
        self,
        pickup_zone: OutdoorZone,
        destination_zone: OutdoorZone,
        package_type: str = "medium",
        recipient_id: Optional[str] = None,
        locker_id: Optional[str] = None,
    ) -> OutdoorTask:
        """创建配送任务"""
        distance = self._estimate_zone_distance(pickup_zone, destination_zone)
        time_limit = distance / self._base_speeds.get(self.agv_grade, 1.5) + 300
        task = OutdoorTask(
            task_id=f"outdoor_{uuid.uuid4().hex[:8]}",
            task_type='last_mile_delivery',
            priority=4,
            pickup_zone=pickup_zone,
            destination_zone=destination_zone,
            package_type=package_type,
            recipient_id=recipient_id,
            locker_id=locker_id,
            terrain_required=[TerrainType.FLAT_PAVEMENT],
            time_window_end=time.time() + time_limit,
            estimated_distance=distance,
        )
        self.add_task(task)
        return task

    def create_patrol_task(
        self,
        zones: List[OutdoorZone],
        patrol_type: str = "routine",
    ) -> OutdoorTask:
        """创建巡逻任务"""
        task = OutdoorTask(
            task_id=f"patrol_{uuid.uuid4().hex[:8]}",
            task_type='patrol',
            priority=3,
            pickup_zone=zones[0] if zones else OutdoorZone.STREET,
            destination_zone=zones[-1] if zones else OutdoorZone.STREET,
        )
        self.add_task(task)
        return task

    def get_next_task(self) -> Optional[OutdoorTask]:
        """获取下一个任务"""
        if not self._task_queue:
            return None
        # 天气过滤
        if not self._weather_monitor.can_operate():
            return None
        return self._task_queue.pop(0)

    def complete_task(self, task_id: str) -> bool:
        """标记任务完成"""
        for task in list(self._task_queue) + self._completed_tasks:
            if task.task_id == task_id:
                task.completed_at = time.time()
                task.status = 'completed'
                task.weather_at_delivery = self._weather_monitor.get_weather()
                if task not in self._completed_tasks:
                    self._completed_tasks.append(task)
                return True
        return False

    def register_agv(self, agv_id: str, lat: float, lng: float) -> None:
        """注册AGV位置"""
        self._agv_positions[agv_id] = (lat, lng)
        self._gps_localizer.update_position(lat, lng)

    def update_agv_position(self, agv_id: str, lat: float, lng: float) -> None:
        """更新AGV位置"""
        self._agv_positions[agv_id] = (lat, lng)
        self._gps_localizer.update_position(lat, lng)

    def get_agv_position(self, agv_id: str) -> Optional[Tuple[float, float]]:
        return self._agv_positions.get(agv_id)

    def _estimate_zone_distance(self, zone_a: OutdoorZone, zone_b: OutdoorZone) -> float:
        """估算区域间距离(米)"""
        distances = {
            (OutdoorZone.RESIDENTIAL, OutdoorZone.DELIVERY_LOCKER): 50.0,
            (OutdoorZone.RESIDENTIAL, OutdoorZone.PICKUP_STATION): 200.0,
            (OutdoorZone.OFFICE, OutdoorZone.BUILDING_ENTRANCE): 30.0,
            (OutdoorZone.CAMPUS, OutdoorZone.PARK): 500.0,
            (OutdoorZone.STREET, OutdoorZone.SIDEWALK): 5.0,
        }
        for (za, zb), d in distances.items():
            if (za == zone_a and zb == zone_b) or (za == zone_b and zb == zone_a):
                return d
        return 100.0   # 默认100米

    def get_scene_status(self) -> Dict:
        """获取场景状态"""
        return {
            'pending_tasks': len(self._task_queue),
            'completed_tasks': len(self._completed_tasks),
            'active_emergencies': len(self._emergency_handler.get_active_emergencies()),
            'weather': self._weather_monitor.get_weather().name,
            'severity': self._weather_monitor.get_severity().name,
            'can_operate': self._weather_monitor.can_operate(),
            'agv_count': len(self._agv_positions),
            'gps_lock': self._gps_localizer.has_gps_lock(),
            'agv_grade': self.agv_grade,
        }

    def generate_scene_report(self) -> Dict:
        """生成场景报告"""
        total = len(self._completed_tasks)
        weather_breakdown = {}
        for t in self._completed_tasks:
            if t.weather_at_delivery:
                wname = t.weather_at_delivery.name
                weather_breakdown[wname] = weather_breakdown.get(wname, 0) + 1
        return {
            'timestamp': time.time(),
            'total_deliveries': total,
            'delivery_completion_rate': self._delivery_confirmer.get_completion_rate(),
            'weather_breakdown': weather_breakdown,
            'terrain_summary': dict(
                self._count_terrains(self._terrain_navigator._terrain_history)
            ),
            'gps_statistics': self._gps_localizer.get_statistics(),
            'emergency_count': len(self._emergency_handler._emergency_log),
            'active_emergencies': len(self._emergency_handler.get_active_emergencies()),
        }

    def _count_terrains(self, history: List[TerrainType]) -> Dict[str, int]:
        counts = {}
        for t in history:
            counts[t.name] = counts.get(t.name, 0) + 1
        return counts

    # ----- 访问器 -----
    @property
    def terrain_navigator(self) -> TerrainNavigator:
        return self._terrain_navigator

    @property
    def weather_monitor(self) -> WeatherMonitor:
        return self._weather_monitor

    @property
    def gps_localizer(self) -> GPSLocalizer:
        return self._gps_localizer

    @property
    def delivery_confirmer(self) -> DeliveryConfirmation:
        return self._delivery_confirmer

    @property
    def emergency_handler(self) -> EmergencyHandler:
        return self._emergency_handler

    def get_recommended_speed(self) -> float:
        """获取推荐速度(考虑天气+地形)"""
        base = self._base_speeds.get(self.agv_grade, 1.5)
        terrain_factor = self._terrain_navigator.get_speed_factor()
        if self._weather_monitor.get_severity() == WeatherSeverity.WARNING:
            weather_factor = 0.5
        elif self._weather_monitor.get_severity() == WeatherSeverity.CAUTION:
            weather_factor = 0.8
        else:
            weather_factor = 1.0
        return base * terrain_factor * weather_factor


# ============================================================
# 全局单例
# ============================================================

_outdoor_controller: Optional[OutdoorSceneController] = None


def get_outdoor_scene_controller(agv_grade: str = "L") -> OutdoorSceneController:
    """获取户外场景控制器全局实例"""
    global _outdoor_controller
    if _outdoor_controller is None:
        _outdoor_controller = OutdoorSceneController(agv_grade=agv_grade)
    return _outdoor_controller
