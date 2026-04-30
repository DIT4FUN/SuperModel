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
industrial_scene.py - 工业制造场景化具身智能模块
SuperModel 超模态大模型具身智能系统

工业场景专项:
- 柔性生产线 (生产节拍/工序调度/物料供应)
- 质量检测 (视觉检测/尺寸测量/缺陷识别)
- 设备预测性维护 (振动分析/温度监控/寿命预测)
- 物料搬运 (原材料/在制品/成品)
- 工具管理 (刀具管理/工装夹具/量具校验)
- 安全监控 (危险区域/人员检测/应急响应)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)

__all__ = [
    'ProductionLineType',
    'WorkstationType',
    'MaterialType',
    'QualityGrade',
    'ToolType',
    'ProductionTask',
    'ProductionLineController',
    'QualityInspectionStation',
    'PredictiveMaintenanceMonitor',
    'ToolManagementSystem',
    'SafetyMonitoringSystem',
    'MaterialFlowCoordinator',
    'get_industrial_scene_controller',
]


class ProductionLineType(Enum):
    """生产线类型"""
    ASSEMBLY = auto()          # 装配线
    MACHINING = auto()         # 机械加工
    WELDING = auto()           # 焊接线
    PAINTING = auto()          # 喷涂线
    PACKAGING = auto()         # 包装线
    FLEXIBLE = auto()          # 柔性制造单元
    AUTOMATED_WAREHOUSE = auto() # 自动仓储


class WorkstationType(Enum):
    """工位类型"""
    CNC_MACHINE = auto()       # CNC加工中心
    ROBOT_CELL = auto()        # 机器人工作站
    ASSEMBLY_STATION = auto()  # 装配工位
    INSPECTION_STATION = auto() # 检测工位
    PACKING_STATION = auto()   # 包装工位
    LOADING_STATION = auto()   # 上下料工位
    QUALITY_GATE = auto()      # 质量门


class MaterialType(Enum):
    """物料类型"""
    RAW_METAL = auto()         # 金属原材料
    RAW_PLASTIC = auto()       # 塑料原材料
    COMPONENT = auto()         # 零部件
    SUBASSEMBLY = auto()       # 子装配
    FINISHED_GOOD = auto()     # 成品
    PACKAGING = auto()         # 包装材料
    HAZARDOUS = auto()         # 危险品


class QualityGrade(Enum):
    """质量等级"""
    A_PRIME = auto()           # A+ 优等
    A_STANDARD = auto()        # A 标准
    B_REWORK = auto()          # B 需要返工
    C_REJECT = auto()          # C 报废
    UNKNOWN = auto()           # 未检测


class ToolType(Enum):
    """工具类型"""
    END_MILL = auto()          # 端铣刀
    DRILL_BIT = auto()         # 钻头
    TAP = auto()               # 丝锥
    INSERT = auto()            # 刀片
    CALIPER = auto()           # 卡尺
    GAUGE = auto()             # 量规
    WRENCH = auto()            # 扳手
    FIXTURE = auto()           # 工装夹具


@dataclass
class ProductionTask:
    """生产任务"""
    task_id: str
    task_type: str                     # material_supply/part_transfer/quality_check/tool_change
    priority: int                     # 1-10, 10最高
    source_station: WorkstationType
    destination_station: WorkstationType
    material_type: Optional[MaterialType] = None
    material_id: Optional[str] = None
    quantity: int = 1
    cycle_time_target: Optional[float] = None    # 目标节拍时间
    quality_required: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"           # pending/allocated/in_progress/completed/paused/failed
    quality_grade: Optional[QualityGrade] = None
    notes: str = ""


class QualityInspectionStation:
    """质量检测工位"""

    # 检测项目与公差
    INSPECTION_SPECS = {
        'diameter': {'tolerance_mm': 0.05, 'method': 'laser'},
        'flatness': {'tolerance_mm': 0.02, 'method': 'cmos'},
        'surface_defect': {'tolerance_mm': 0.1, 'method': 'vision'},
        'weight': {'tolerance_g': 5.0, 'method': 'scale'},
    }

    def __init__(self, station_id: str):
        self.station_id = station_id
        self._inspection_count = 0
        self._defect_count = 0
        self._rework_count = 0
        self._calibration_last: Optional[float] = None

    def perform_inspection(
        self,
        part_id: str,
        inspection_type: str,
        measured_value: float,
    ) -> Dict:
        """执行检测"""
        if inspection_type not in self.INSPECTION_SPECS:
            return {'status': 'unknown_type', 'grade': QualityGrade.UNKNOWN}

        spec = self.INSPECTION_SPECS[inspection_type]
        tolerance = spec['tolerance_mm'] if 'mm' in spec else spec.get('tolerance_g', 0.1)
        
        # 简化的判定逻辑
        is_within_tolerance = abs(measured_value) <= tolerance
        
        self._inspection_count += 1
        
        if is_within_tolerance:
            grade = QualityGrade.A_PRIME if measured_value < tolerance * 0.5 else QualityGrade.A_STANDARD
            result = 'pass'
        else:
            self._defect_count += 1
            if measured_value < tolerance * 2:
                grade = QualityGrade.B_REWORK
                self._rework_count += 1
                result = 'rework'
            else:
                grade = QualityGrade.C_REJECT
                result = 'reject'

        return {
            'part_id': part_id,
            'inspection_type': inspection_type,
            'measured_value': measured_value,
            'tolerance': tolerance,
            'result': result,
            'grade': grade,
            'station_id': self.station_id,
            'timestamp': time.time(),
        }

    def get_station_stats(self) -> Dict:
        """获取工位统计"""
        return {
            'station_id': self.station_id,
            'total_inspections': self._inspection_count,
            'defect_count': self._defect_count,
            'rework_count': self._rework_count,
            'first_pass_rate': (
                (self._inspection_count - self._defect_count) / self._inspection_count
                if self._inspection_count > 0 else 0
            ),
            'last_calibration': self._calibration_last,
            'needs_calibration': (
                time.time() - self._calibration_last > 28800
                if self._calibration_last else True
            ),
        }


class PredictiveMaintenanceMonitor:
    """预测性维护监控器"""

    # 设备关键参数阈值
    MAINTENANCE_THRESHOLDS = {
        'temperature_c': {'warning': 70, 'critical': 85, 'unit': 'celsius'},
        'vibration_mm_s': {'warning': 4.5, 'critical': 7.1, 'unit': 'mm/s'},
        'current_a': {'warning': 90, 'critical': 100, 'unit': 'percent'},
        'runtime_hours': {'warning': 2000, 'critical': 4000, 'unit': 'hours'},
    }

    def __init__(self):
        self._equipment_states: Dict[str, Dict] = {}
        self._maintenance_history: List[Dict] = []
        self._alert_history: List[Dict] = []

    def register_equipment(self, equip_id: str, equip_type: str, specs: Dict) -> None:
        """注册设备"""
        self._equipment_states[equip_id] = {
            'type': equip_type,
            'specs': specs,
            'install_date': time.time(),
            'runtime_hours': 0,
            'last_maintenance': None,
            'alerts': [],
        }

    def update_telemetry(self, equip_id: str, metrics: Dict) -> List[Dict]:
        """更新设备遥测数据"""
        if equip_id not in self._equipment_states:
            return []
        
        state = self._equipment_states[equip_id]
        alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.MAINTENANCE_THRESHOLDS:
                threshold = self.MAINTENANCE_THRESHOLDS[metric_name]
                
                if value >= threshold['critical']:
                    level = 'critical'
                    alert = {
                        'equip_id': equip_id,
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'level': level,
                        'timestamp': time.time(),
                    }
                    alerts.append(alert)
                    self._alert_history.append(alert)
                    state['alerts'].append(alert)
                    
                elif value >= threshold['warning']:
                    level = 'warning'
                    alert = {
                        'equip_id': equip_id,
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'level': level,
                        'timestamp': time.time(),
                    }
                    alerts.append(alert)
                    state['alerts'].append(alert)

        # 更新运行时长
        if 'runtime_hours' in metrics:
            state['runtime_hours'] = metrics['runtime_hours']
        
        return alerts

    def predict_maintenance_window(self, equip_id: str) -> Optional[Dict]:
        """预测维护窗口"""
        if equip_id not in self._equipment_states:
            return None
        
        state = self._equipment_states[equip_id]
        runtime = state['runtime_hours']
        
        # 基于运行时间的简单预测
        remaining_hours = max(0, 4000 - runtime)
        remaining_days = remaining_hours / 24
        
        # 基于告警的优先级
        critical_alerts = [a for a in state['alerts'] if a.get('level') == 'critical']
        
        return {
            'equip_id': equip_id,
            'current_runtime_hours': runtime,
            'predicted_failure_hours': 4000,
            'remaining_hours': remaining_hours,
            'remaining_days': remaining_days,
            'maintenance_priority': 'immediate' if critical_alerts else 'scheduled',
            'recommended_action': 'replace' if runtime > 3500 else 'inspect',
        }

    def get_overall_health_score(self) -> Dict:
        """获取整体设备健康评分"""
        total = len(self._equipment_states)
        if total == 0:
            return {'health_score': 100, 'status': 'no_equipment'}
        
        healthy = 0
        warning = 0
        critical = 0
        
        for equip_id, state in self._equipment_states.items():
            if any(a.get('level') == 'critical' for a in state.get('alerts', [])):
                critical += 1
            elif any(a.get('level') == 'warning' for a in state.get('alerts', [])):
                warning += 1
            else:
                healthy += 1
        
        score = (healthy * 100 + warning * 60 + critical * 20) / total
        
        return {
            'health_score': round(score, 1),
            'healthy_count': healthy,
            'warning_count': warning,
            'critical_count': critical,
            'total_equipment': total,
            'status': 'healthy' if score > 80 else 'warning' if score > 50 else 'critical',
        }


class ToolManagementSystem:
    """工具管理系统"""

    TOOL_LIFE_CYCLES = {
        ToolType.END_MILL: {'max_hours': 200, 'max_parts': 1000},
        ToolType.DRILL_BIT: {'max_hours': 100, 'max_parts': 500},
        ToolType.TAP: {'max_hours': 50, 'max_parts': 200},
        ToolType.INSERT: {'max_hours': 500, 'max_parts': 10000},
    }

    def __init__(self):
        self._tool_inventory: Dict[str, Dict] = {}
        self._tool_usage: Dict[str, Dict] = {}
        self._change_history: List[Dict] = []

    def register_tool(
        self,
        tool_id: str,
        tool_type: ToolType,
        specifications: Dict,
    ) -> None:
        """注册工具"""
        self._tool_inventory[tool_id] = {
            'type': tool_type,
            'specs': specifications,
            'assigned_station': None,
            'installed_at': None,
            'total_hours': 0,
            'parts_machined': 0,
            'status': 'available',      # available/assigned/installed/worn/failed
        }

    def install_tool(self, tool_id: str, station: WorkstationType) -> bool:
        """安装工具到工位"""
        if tool_id not in self._tool_inventory:
            return False
        
        tool = self._tool_inventory[tool_id]
        if tool['status'] not in ('available', 'worn'):
            return False
        
        tool['status'] = 'installed'
        tool['assigned_station'] = station
        tool['installed_at'] = time.time()
        
        if tool_id not in self._tool_usage:
            self._tool_usage[tool_id] = {'hours': 0, 'parts': 0}
        
        return True

    def record_tool_usage(self, tool_id: str, hours: float = 0, parts: int = 0) -> Dict:
        """记录工具使用"""
        if tool_id not in self._tool_inventory:
            return {'status': 'unknown_tool'}
        
        tool = self._tool_inventory[tool_id]
        tool['total_hours'] += hours
        tool['parts_machined'] += parts
        
        usage = self._tool_usage.get(tool_id, {'hours': 0, 'parts': 0})
        usage['hours'] += hours
        usage['parts'] += parts
        
        # 检查是否需要更换
        tool_type = tool['type']
        life = self.TOOL_LIFE_CYCLES.get(tool_type, {'max_hours': 100, 'max_parts': 500})
        
        needs_replacement = (
            usage['hours'] >= life['max_hours'] or
            usage['parts'] >= life['max_parts']
        )
        
        if needs_replacement and tool['status'] == 'installed':
            tool['status'] = 'worn'
            self._change_history.append({
                'tool_id': tool_id,
                'reason': 'life_expired',
                'hours_used': usage['hours'],
                'parts_machined': usage['parts'],
                'timestamp': time.time(),
            })
        
        return {
            'tool_id': tool_id,
            'status': tool['status'],
            'needs_replacement': needs_replacement,
            'remaining_hours': max(0, life['max_hours'] - usage['hours']),
            'remaining_parts': max(0, life['max_parts'] - usage['parts']),
        }

    def get_tool_status(self, tool_id: str) -> Optional[Dict]:
        """获取工具状态"""
        if tool_id not in self._tool_inventory:
            return None
        
        tool = self._tool_inventory[tool_id]
        usage = self._tool_usage.get(tool_id, {'hours': 0, 'parts': 0})
        life = self.TOOL_LIFE_CYCLES.get(tool['type'], {'max_hours': 100, 'max_parts': 500})
        
        return {
            'tool_id': tool_id,
            'type': tool['type'].name,
            'status': tool['status'],
            'assigned_station': tool['assigned_station'].name if tool['assigned_station'] else None,
            'total_hours': tool['total_hours'],
            'parts_machined': tool['parts_machined'],
            'life_remaining_percent': max(0, 100 * min(
                life['max_hours'] - usage['hours'],
                life['max_parts'] - usage['parts']
            ) / 100),
        }


class SafetyMonitoringSystem:
    """工业安全监控系统"""

    # 危险区域定义
    HAZARD_ZONES = {
        'forklift_area': {'risk': 'high', 'speed_limit': 1.0},
        'welding_area': {'risk': 'critical', 'speed_limit': 0},
        'press_area': {'risk': 'critical', 'speed_limit': 0},
        'painting_booth': {'risk': 'high', 'speed_limit': 0.5},
        'heavy_load_zone': {'risk': 'medium', 'speed_limit': 1.5},
        'assembly_line': {'risk': 'low', 'speed_limit': 2.0},
        'quality_gate': {'risk': 'low', 'speed_limit': 2.0},
    }

    def __init__(self):
        self._personnel_positions: Dict[str, Tuple[float, float]] = {}
        self._incident_log: List[Dict] = []
        self._zone_alerts: List[Dict] = []
        self._safety_stop_active = False

    def register_person(self, person_id: str, position: Tuple[float, float]) -> None:
        """注册人员位置"""
        self._personnel_positions[person_id] = position

    def check_zone_entry(self, agv_id: str, position: Tuple[float, float], zone: str) -> Dict:
        """检查区域进入安全性"""
        if zone not in self.HAZARD_ZONES:
            return {'allowed': True, 'zone': zone, 'risk': 'unknown'}
        
        hazard = self.HAZARD_ZONES[zone]
        
        # 检查是否有人员在危险区域
        nearby_person = False
        for pos in self._personnel_positions.values():
            distance = ((pos[0] - position[0])**2 + (pos[1] - position[1])**2)**0.5
            if distance < 2.0:  # 2米以内
                nearby_person = True
                break
        
        if hazard['risk'] == 'critical' and nearby_person:
            self._incident_log.append({
                'type': 'near_miss',
                'agv_id': agv_id,
                'zone': zone,
                'timestamp': time.time(),
            })
            return {
                'allowed': False,
                'zone': zone,
                'risk': hazard['risk'],
                'reason': 'Personnel in hazardous area',
            }
        
        return {
            'allowed': True,
            'zone': zone,
            'risk': hazard['risk'],
            'speed_limit': hazard['speed_limit'],
        }

    def trigger_emergency_stop(self, reason: str) -> None:
        """触发紧急停止"""
        self._safety_stop_active = True
        self._incident_log.append({
            'type': 'emergency_stop',
            'reason': reason,
            'timestamp': time.time(),
        })
        logger.warning(f"工业安全系统: 紧急停止触发 - {reason}")

    def reset_emergency_stop(self) -> None:
        """复位紧急停止"""
        self._safety_stop_active = False

    def get_safety_status(self) -> Dict:
        """获取安全状态"""
        critical_incidents = [
            i for i in self._incident_log[-100:]
            if i.get('type') in ('near_miss', 'emergency_stop')
        ]
        
        return {
            'emergency_stop_active': self._safety_stop_active,
            'active_personnel': len(self._personnel_positions),
            'recent_incidents': len(critical_incidents),
            'incident_count': len(self._incident_log),
            'zone_alerts': len(self._zone_alerts),
        }


class MaterialFlowCoordinator:
    """物料流协调器"""

    def __init__(self, line_type: ProductionLineType):
        self.line_type = line_type
        self._material_buffers: Dict[WorkstationType, List] = {
            ws: [] for ws in WorkstationType
        }
        self._active_transfers: List[ProductionTask] = []
        self._transfer_history: List[Dict] = []

    def request_material(
        self,
        task: ProductionTask,
        requesting_station: WorkstationType,
    ) -> bool:
        """请求物料"""
        # 简单FIFO逻辑
        buffer = self._material_buffers.get(requesting_station, [])
        if buffer:
            buffer.append(task)
            return True
        return False

    def allocate_transfer(
        self,
        task: ProductionTask,
        assigned_agv: str,
    ) -> Dict:
        """分配物料搬运任务"""
        self._active_transfers.append(task)
        task.status = 'allocated'
        task.started_at = time.time()
        
        return {
            'task_id': task.task_id,
            'assigned_agv': assigned_agv,
            'estimated_time': self._estimate_transfer_time(task),
            'route': self._plan_transfer_route(task),
        }

    def complete_transfer(self, task_id: str) -> bool:
        """完成物料搬运"""
        for task in self._active_transfers:
            if task.task_id == task_id:
                task.status = 'completed'
                task.completed_at = time.time()
                self._active_transfers.remove(task)
                
                self._transfer_history.append({
                    'task_id': task_id,
                    'duration': task.completed_at - task.started_at,
                    'timestamp': task.completed_at,
                })
                return True
        return False

    def _estimate_transfer_time(self, task: ProductionTask) -> float:
        """估算搬运时间"""
        # 简化: 固定节拍
        base_time = 30.0  # 秒
        
        if task.priority >= 8:
            base_time = 15.0  # 高优先级加速
        elif task.priority <= 3:
            base_time = 60.0  # 低优先级延长
        
        return base_time

    def _plan_transfer_route(self, task: ProductionTask) -> List[WorkstationType]:
        """规划搬运路线"""
        return [task.source_station, task.destination_station]

    def get_buffer_status(self, station: WorkstationType) -> Dict:
        """获取工位缓存状态"""
        buffer = self._material_buffers.get(station, [])
        return {
            'station': station.name,
            'buffer_size': len(buffer),
            'urgent_count': sum(1 for t in buffer if t.priority >= 8),
            'pending_count': sum(1 for t in buffer if t.status == 'pending'),
        }


class ProductionLineController:
    """生产线总控制器"""

    def __init__(self, line_type: ProductionLineType = ProductionLineType.FLEXIBLE, grade: str = "L"):
        self.line_type = line_type
        self.agv_grade = grade
        self._quality_station = QualityInspectionStation("QI_01")
        self._maintenance_monitor = PredictiveMaintenanceMonitor()
        self._tool_manager = ToolManagementSystem()
        self._safety_monitor = SafetyMonitoringSystem()
        self._material_coordinator = MaterialFlowCoordinator(line_type)
        self._task_counter = 0
        self._production_tasks: List[ProductionTask] = []

    @property
    def quality_station(self) -> QualityInspectionStation:
        return self._quality_station

    @property
    def maintenance_monitor(self) -> PredictiveMaintenanceMonitor:
        return self._maintenance_monitor

    @property
    def tool_manager(self) -> ToolManagementSystem:
        return self._tool_manager

    @property
    def safety_monitor(self) -> SafetyMonitoringSystem:
        return self._safety_monitor

    @property
    def material_coordinator(self) -> MaterialFlowCoordinator:
        return self._material_coordinator

    def create_production_task(
        self,
        task_type: str,
        source: WorkstationType,
        destination: WorkstationType,
        priority: int = 5,
        material_id: Optional[str] = None,
        material_type: Optional[MaterialType] = None,
    ) -> ProductionTask:
        """创建生产任务"""
        self._task_counter += 1
        return ProductionTask(
            task_id=f"prod_{self._task_counter:05d}",
            task_type=task_type,
            priority=priority,
            source_station=source,
            destination_station=destination,
            material_id=material_id,
            material_type=material_type,
        )

    def add_task(self, task: ProductionTask) -> None:
        """添加工件到生产线"""
        self._production_tasks.append(task)

    def get_oee(self) -> Dict:
        """
        计算OEE (Overall Equipment Effectiveness)
        OEE = Availability × Performance × Quality
        """
        total = len(self._production_tasks)
        if total == 0:
            return {'oee': 0, 'availability': 0, 'performance': 0, 'quality': 0}
        
        completed = sum(1 for t in self._production_tasks if t.status == 'completed')
        
        # 简化计算
        availability = 0.95  # 假设
        performance = completed / total if total > 0 else 0
        quality_grade_a = sum(
            1 for t in self._production_tasks
            if t.quality_grade in (QualityGrade.A_PRIME, QualityGrade.A_STANDARD)
        )
        quality = quality_grade_a / completed if completed > 0 else 0
        
        oee = availability * performance * quality * 100
        
        return {
            'oee': round(oee, 1),
            'availability': round(availability * 100, 1),
            'performance': round(performance * 100, 1),
            'quality': round(quality * 100, 1),
            'total_tasks': total,
            'completed_tasks': completed,
        }

    def get_production_report(self) -> Dict:
        """生成生产报告"""
        return {
            'timestamp': time.time(),
            'line_type': self.line_type.name,
            'agv_grade': self.agv_grade,
            'oee': self.get_oee(),
            'equipment_health': self._maintenance_monitor.get_overall_health_score(),
            'safety_status': self._safety_monitor.get_safety_status(),
            'active_transfers': len(self._material_coordinator._active_transfers),
            'total_production_tasks': len(self._production_tasks),
            'quality_inspections': self._quality_station._inspection_count,
        }


# 全局单例
_industrial_controller: Optional[ProductionLineController] = None


def get_industrial_scene_controller(
    line_type: ProductionLineType = ProductionLineType.FLEXIBLE,
    grade: str = "L"
) -> ProductionLineController:
    """获取工业场景控制器全局实例"""
    global _industrial_controller
    if _industrial_controller is None:
        _industrial_controller = ProductionLineController(line_type=line_type, grade=grade)
    return _industrial_controller
