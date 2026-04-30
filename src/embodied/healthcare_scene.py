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
healthcare_scene.py - 医疗场景化具身智能模块
SuperModel 超模态大模型具身智能系统

医疗场景专项:
- 医院物流 (药品配送/标本运输/物资运送)
- 手术室辅助 (器械传递/设备移动)
- 病房服务 (患者辅助/呼叫响应)
- 药房自动化 (药品分拣/库存管理)
- 感染控制 (消毒配送/隔离运输)
- 紧急物资运输 (急救药品/血液制品)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)

__all__ = [
    'HealthcareZone',
    'HealthcareRiskLevel',
    'PatientCallPriority',
    'MedicationType',
    'SpecimenCategory',
    'HealthcareTask',
    'HealthcareTaskLibrary',
    'HealthcareSceneController',
    'InfectionControlMonitor',
    'PatientCallHandler',
    'MedicationDeliveryPlanner',
    'SpecimenTransportManager',
    'get_healthcare_scene_controller',
]


class HealthcareZone(Enum):
    """医疗区域类型"""
    PHARMACY = auto()           # 药房
    WARD = auto()              # 病房
    ICU = auto()               # 重症监护室
    OPERATING_ROOM = auto()    # 手术室
    LABORATORY = auto()        # 检验科
    BLOOD_BANK = auto()        # 血库
    EMERGENCY = auto()         # 急诊
    CENTRAL_SUPPLY = auto()    # 中心供应
    STERILE_STORAGE = auto()   # 无菌物品存放
    ISOLATION = auto()         # 隔离区
    CORRIDOR = auto()          # 走廊
    ELEVATOR = auto()          # 电梯


class HealthcareRiskLevel(Enum):
    """医疗风险等级"""
    LOW = 1           # 低风险 (普通物资)
    MEDIUM = 2        # 中风险 (药品)
    HIGH = 3          # 高风险 (血液制品)
    CRITICAL = 4      # 危急 (急救药品/器官)


class PatientCallPriority(Enum):
    """患者呼叫优先级"""
    routine = 1      # 常规
    urgent = 2        # 紧急
    emergency = 3    # 急救


class MedicationType(Enum):
    """药品类型"""
    ORAL = auto()           # 口服药
    INJECTION = auto()      # 注射药
    INFUSION = auto()       # 输液
    COLD_CHAIN = auto()     # 冷链药品
    CONTROLLED = auto()      # 管制药品
    RADIOPHARMA = auto()    # 放射性药品
    BIOLOGICAL = auto()     # 生物制剂


class SpecimenCategory(Enum):
    """标本类别"""
    BLOOD = auto()          # 血液
    TISSUE = auto()         # 组织
    URINE = auto()          # 尿液
    STOOL = auto()          # 粪便
    CSF = auto()            # 脑脊液
    BACTERIA = auto()       # 细菌培养
    PATHOLOGY = auto()      # 病理切片


@dataclass
class HealthcareTask:
    """医疗任务"""
    task_id: str
    task_type: str                    # medication_delivery/specimen_transport/supply_delivery/patient_call
    priority: HealthcareRiskLevel
    source_zone: HealthcareZone
    destination_zone: HealthcareZone
    payload_type: str                 # medication/specimen/blood/supplies
    payload_id: str                  # 药品编号/标本编号
    patient_id: Optional[str] = None
    requires_cold_chain: bool = False
    requires_controlled_access: bool = False
    requires_sterile: bool = False
    time_constraint: Optional[float] = None   # 秒
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    status: str = "pending"          # pending/in_transit/delivered/completed/failed
    notes: str = ""


class HealthcareTaskLibrary:
    """医疗任务模板库"""

    # 药品配送任务模板
    MEDICATION_DELIVERY_TEMPLATES = {
        'routine_oral': {
            'type': 'medication_delivery',
            'priority': HealthcareRiskLevel.LOW,
            'cold_chain': False,
            'controlled': False,
            'sterile': False,
            'time_limit': 900,        # 15分钟
        },
        'urgent_injection': {
            'type': 'medication_delivery',
            'priority': HealthcareRiskLevel.MEDIUM,
            'cold_chain': False,
            'controlled': True,
            'sterile': True,
            'time_limit': 300,        # 5分钟
        },
        'critical_infusion': {
            'type': 'medication_delivery',
            'priority': HealthcareRiskLevel.HIGH,
            'cold_chain': False,
            'controlled': True,
            'sterile': True,
            'time_limit': 180,        # 3分钟
        },
        'cold_chain_biological': {
            'type': 'medication_delivery',
            'priority': HealthcareRiskLevel.HIGH,
            'cold_chain': True,
            'controlled': False,
            'sterile': True,
            'time_limit': 600,        # 10分钟
        },
    }

    # 标本运输任务模板
    SPECIMEN_TRANSPORT_TEMPLATES = {
        'blood_routine': {
            'type': 'specimen_transport',
            'category': SpecimenCategory.BLOOD,
            'priority': HealthcareRiskLevel.LOW,
            'time_limit': 1800,       # 30分钟
        },
        'blood_emergency': {
            'type': 'specimen_transport',
            'category': SpecimenCategory.BLOOD,
            'priority': HealthcareRiskLevel.CRITICAL,
            'time_limit': 300,        # 5分钟
        },
        'tissue_pathology': {
            'type': 'specimen_transport',
            'category': SpecimenCategory.PATHOLOGY,
            'priority': HealthcareRiskLevel.MEDIUM,
            'time_limit': 900,        # 15分钟
        },
        'bacteria_culture': {
            'type': 'specimen_transport',
            'category': SpecimenCategory.BACTERIA,
            'priority': HealthcareRiskLevel.MEDIUM,
            'time_limit': 1200,       # 20分钟
        },
    }

    # 物资运送任务模板
    SUPPLY_DELIVERY_TEMPLATES = {
        'sterile_supplies': {
            'type': 'supply_delivery',
            'sterile': True,
            'priority': HealthcareRiskLevel.MEDIUM,
            'time_limit': 600,
        },
        'linen': {
            'type': 'supply_delivery',
            'sterile': False,
            'priority': HealthcareRiskLevel.LOW,
            'time_limit': 1800,
        },
        'medical_devices': {
            'type': 'supply_delivery',
            'sterile': False,
            'priority': HealthcareRiskLevel.MEDIUM,
            'time_limit': 900,
        },
    }

    def __init__(self):
        self._task_counter = 0

    def create_medication_task(
        self,
        template_name: str,
        source: HealthcareZone,
        destination: HealthcareZone,
        patient_id: str,
        medication_id: str,
    ) -> HealthcareTask:
        """创建药品配送任务"""
        if template_name not in self.MEDICATION_DELIVERY_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.MEDICATION_DELIVERY_TEMPLATES[template_name]
        self._task_counter += 1
        
        return HealthcareTask(
            task_id=f"med_{self._task_counter:04d}",
            task_type='medication_delivery',
            priority=template['priority'],
            source_zone=source,
            destination_zone=destination,
            payload_type='medication',
            payload_id=medication_id,
            patient_id=patient_id,
            requires_cold_chain=template['cold_chain'],
            requires_controlled_access=template['controlled'],
            requires_sterile=template['sterile'],
            time_constraint=template['time_limit'],
        )

    def create_specimen_task(
        self,
        template_name: str,
        source: HealthcareZone,
        destination: HealthcareZone,
        specimen_id: str,
    ) -> HealthcareTask:
        """创建标本运输任务"""
        if template_name not in self.SPECIMEN_TRANSPORT_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.SPECIMEN_TRANSPORT_TEMPLATES[template_name]
        self._task_counter += 1
        
        return HealthcareTask(
            task_id=f"spec_{self._task_counter:04d}",
            task_type='specimen_transport',
            priority=template['priority'],
            source_zone=source,
            destination_zone=destination,
            payload_type='specimen',
            payload_id=specimen_id,
            requires_cold_chain=False,
            requires_controlled_access=False,
            requires_sterile=True,
            time_constraint=template['time_limit'],
        )

    def create_supply_task(
        self,
        template_name: str,
        source: HealthcareZone,
        destination: HealthcareZone,
        supply_id: str,
    ) -> HealthcareTask:
        """创建物资运送任务"""
        if template_name not in self.SUPPLY_DELIVERY_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.SUPPLY_DELIVERY_TEMPLATES[template_name]
        self._task_counter += 1
        
        return HealthcareTask(
            task_id=f"sup_{self._task_counter:04d}",
            task_type='supply_delivery',
            priority=template['priority'],
            source_zone=source,
            destination_zone=destination,
            payload_type='supplies',
            payload_id=supply_id,
            requires_cold_chain=False,
            requires_controlled_access=False,
            requires_sterile=template['sterile'],
            time_constraint=template['time_limit'],
        )


class InfectionControlMonitor:
    """感染控制监控器"""

    # 各区域感染风险等级
    ZONE_RISK_LEVELS: Dict[HealthcareZone, int] = {
        HealthcareZone.ISOLATION: 5,
        HealthcareZone.ICU: 4,
        HealthcareZone.OPERATING_ROOM: 4,
        HealthcareZone.EMERGENCY: 3,
        HealthcareZone.LABORATORY: 3,
        HealthcareZone.BLOOD_BANK: 3,
        HealthcareZone.STERILE_STORAGE: 2,
        HealthcareZone.PHARMACY: 2,
        HealthcareZone.CENTRAL_SUPPLY: 2,
        HealthcareZone.WARD: 2,
        HealthcareZone.CORRIDOR: 1,
        HealthcareZone.ELEVATOR: 1,
    }

    def __init__(self):
        self._decontamination_events: List[Dict] = []
        self._isolation_alerts: List[Dict] = []
        self._last_decontamination: Dict[HealthcareZone, float] = {}

    def check_zone_access(self, zone: HealthcareZone, agv_status: Dict) -> Tuple[bool, str]:
        """
        检查AGV进入某区域是否安全
        Returns: (allowed, reason)
        """
        risk_level = self.ZONE_RISK_LEVELS.get(zone, 1)
        
        # 隔离区需要专用AGV
        if zone == HealthcareZone.ISOLATION:
            if not agv_status.get('isolation_compatible', False):
                return False, "AGV不具备隔离区作业资质"
        
        # 无菌区域需要消毒
        if zone in (HealthcareZone.OPERATING_ROOM, HealthcareZone.STERILE_STORAGE):
            last_decon = self._last_decontamination.get(zone, 0)
            if time.time() - last_decon > 28800:  # 8小时未消毒
                return False, f"{zone.name}超过8小时未消毒"
        
        return True, "允许进入"

    def record_decontamination(self, zone: HealthcareZone) -> None:
        """记录消毒事件"""
        self._decontamination_events.append({
            'zone': zone,
            'timestamp': time.time(),
        })
        self._last_decontamination[zone] = time.time()
        logger.info(f"记录{zone.name}消毒事件")

    def add_isolation_alert(self, zone: HealthcareZone, reason: str) -> None:
        """添加隔离警报"""
        self._isolation_alerts.append({
            'zone': zone,
            'reason': reason,
            'timestamp': time.time(),
        })

    def requires_isolation_protocol(self, zone: HealthcareZone) -> bool:
        """判断区域是否需要隔离协议"""
        return zone in (
            HealthcareZone.ISOLATION,
            HealthcareZone.ICU,
            HealthcareZone.OPERATING_ROOM,
        )

    def get_risk_level(self, zone: HealthcareZone) -> int:
        """获取区域风险等级"""
        return self.ZONE_RISK_LEVELS.get(zone, 1)

    def get_decontamination_status(self, zone: HealthcareZone) -> Dict:
        """获取消毒状态"""
        last = self._last_decontamination.get(zone)
        elapsed = time.time() - last if last else float('inf')
        needs_decon = elapsed > 28800  # 8小时
        
        return {
            'zone': zone,
            'last_decontamination': last,
            'elapsed_seconds': elapsed,
            'needs_decontamination': needs_decon,
            'risk_level': self.ZONE_RISK_LEVELS.get(zone, 1),
        }


class PatientCallHandler:
    """患者呼叫处理器"""

    CALL_PRIORITY_MAP = {
        PatientCallPriority.routine: HealthcareRiskLevel.LOW,
        PatientCallPriority.urgent: HealthcareRiskLevel.MEDIUM,
        PatientCallPriority.emergency: HealthcareRiskLevel.CRITICAL,
    }

    def __init__(self, task_library: HealthcareTaskLibrary):
        self._task_library = task_library
        self._active_calls: Dict[str, Dict] = {}
        self._call_history: List[Dict] = []
        self._response_count = 0

    def handle_call(
        self,
        call_id: str,
        patient_id: str,
        ward_zone: HealthcareZone,
        priority: PatientCallPriority,
        callback: Optional[Callable] = None,
    ) -> Optional[HealthcareTask]:
        """处理患者呼叫"""
        if call_id in self._active_calls:
            return None  # 已处理的呼叫
        
        # 转换为医疗任务
        task = self._task_library.create_medication_task(
            template_name='routine_oral' if priority == PatientCallPriority.routine else 'urgent_injection',
            source=HealthcareZone.PHARMACY,
            destination=ward_zone,
            patient_id=patient_id,
            medication_id=f"med_c_{call_id}",
        )
        task.notes = f"来自患者{patient_id}的{'紧急' if priority == PatientCallPriority.emergency else '常规'}呼叫"
        
        self._active_calls[call_id] = {
            'task': task,
            'priority': priority,
            'callback': callback,
            'created_at': time.time(),
        }
        
        self._response_count += 1
        logger.info(f"处理患者呼叫 {call_id}, 优先级: {priority.name}")
        
        return task

    def complete_call(self, call_id: str) -> None:
        """完成呼叫处理"""
        if call_id in self._active_calls:
            call_info = self._active_calls.pop(call_id)
            self._call_history.append({
                'call_id': call_id,
                'task': call_info['task'].task_id,
                'completed_at': time.time(),
                'wait_time': time.time() - call_info['created_at'],
            })

    def get_active_calls(self, min_priority: PatientCallPriority = PatientCallPriority.routine) -> List[Dict]:
        """获取活跃呼叫列表"""
        result = []
        for call_id, info in self._active_calls.items():
            if info['priority'].value >= min_priority.value:
                result.append({
                    'call_id': call_id,
                    'task': info['task'],
                    'waiting_seconds': time.time() - info['created_at'],
                })
        return sorted(result, key=lambda x: x['task'].priority.value, reverse=True)


class MedicationDeliveryPlanner:
    """药品配送规划器"""

    # 药品兼容性矩阵 (某些药品不能同时运输)
    INCOMPATIBLE_PAIRS: Set[Tuple[str, str]] = {
        ('antibiotics', 'chemotherapy'),
        ('blood_products', 'insulin'),
        ('radioactive', 'any'),
    }

    def __init__(self, infection_monitor: InfectionControlMonitor):
        self._infection_monitor = infection_monitor
        self._active_deliveries: Dict[str, HealthcareTask] = {}
        self._delivery_history: List[Dict] = []

    def plan_delivery(
        self,
        task: HealthcareTask,
        available_agvs: List[Dict],
    ) -> Optional[Dict]:
        """规划药品配送"""
        if task.requires_controlled_access:
            # 管制药品需要专用AGV
            suitable = [a for a in available_agvs if a.get('controlled_drug_certified', False)]
        elif task.requires_cold_chain:
            # 冷链药品需要冷藏AGV
            suitable = [a for a in available_agvs if a.get('cold_chain_capable', False)]
        elif task.requires_sterile:
            # 无菌药品需要无菌AGV
            suitable = [a for a in available_agvs if a.get('sterile_transport', False)]
        else:
            suitable = available_agvs

        if not suitable:
            return None

        # 选择最近的AGV
        best_agv = min(suitable, key=lambda a: a.get('distance_to_pharmacy', float('inf')))

        # 检查感染控制
        allowed, reason = self._infection_monitor.check_zone_access(
            task.destination_zone, best_agv
        )
        if not allowed:
            return {'status': 'blocked', 'reason': reason}

        return {
            'status': 'planned',
            'assigned_agv': best_agv.get('agv_id'),
            'route': self._plan_route(task.source_zone, task.destination_zone),
            'time_estimate': self._estimate_delivery_time(task),
        }

    def _plan_route(self, source: HealthcareZone, dest: HealthcareZone) -> List[HealthcareZone]:
        """规划路径 (简化版)"""
        if source == HealthcareZone.PHARMACY:
            return [source, dest]
        return [source, HealthcareZone.CORRIDOR, dest]

    def _estimate_delivery_time(self, task: HealthcareTask) -> float:
        """估算配送时间"""
        base_times = {
            HealthcareZone.PHARMACY: 0,
            HealthcareZone.WARD: 180,
            HealthcareZone.ICU: 120,
            HealthcareZone.OPERATING_ROOM: 240,
            HealthcareZone.LABORATORY: 300,
            HealthcareZone.BLOOD_BANK: 150,
            HealthcareZone.EMERGENCY: 90,
            HealthcareZone.CENTRAL_SUPPLY: 200,
            HealthcareZone.STERILE_STORAGE: 200,
            HealthcareZone.ISOLATION: 180,
            HealthcareZone.CORRIDOR: 30,
            HealthcareZone.ELEVATOR: 60,
        }
        return base_times.get(task.destination_zone, 300)


class SpecimenTransportManager:
    """标本运输管理器"""

    TRANSPORT_CONTAINERS = {
        SpecimenCategory.BLOOD: 'vacutainer',
        SpecimenCategory.TISSUE: 'formalin_container',
        SpecimenCategory.URINE: 'sterile_container',
        SpecimenCategory.STOOL: 'sterile_container',
        SpecimenCategory.CSF: 'aseptic_container',
        SpecimenCategory.BACTERIA: 'culture_swab',
        SpecimenCategory.PATHOLOGY: 'histology cassette',
    }

    def __init__(self, infection_monitor: InfectionControlMonitor):
        self._infection_monitor = infection_monitor
        self._specimen_tracking: Dict[str, Dict] = {}
        self._transport_records: List[Dict] = []

    def register_specimen(
        self,
        specimen_id: str,
        category: SpecimenCategory,
        source: HealthcareZone,
    ) -> HealthcareTask:
        """注册标本并创建运输任务"""
        task = HealthcareTask(
            task_id=f"spec_{specimen_id}",
            task_type='specimen_transport',
            priority=HealthcareRiskLevel.LOW if category != SpecimenCategory.BACTERIA else HealthcareRiskLevel.MEDIUM,
            source_zone=source,
            destination_zone=HealthcareZone.LABORATORY,
            payload_type='specimen',
            payload_id=specimen_id,
            requires_sterile=True,
            time_constraint=1200 if category != SpecimenCategory.PATHOLOGY else 900,
        )
        
        self._specimen_tracking[specimen_id] = {
            'task': task,
            'category': category,
            'collected_at': time.time(),
            'container': self.TRANSPORT_CONTAINERS[category],
        }
        
        return task

    def track_specimen(self, specimen_id: str) -> Optional[Dict]:
        """追踪标本状态"""
        return self._specimen_tracking.get(specimen_id)

    def verify_chain_of_custody(self, specimen_id: str) -> Dict:
        """验证标本监管链"""
        record = self._specimen_tracking.get(specimen_id)
        if not record:
            return {'valid': False, 'reason': 'Specimen not found'}
        
        elapsed = time.time() - record['collected_at']
        
        # 检查运输时间限制
        category_time_limits = {
            SpecimenCategory.BLOOD: 1800,
            SpecimenCategory.TISSUE: 3600,
            SpecimenCategory.URINE: 1800,
            SpecimenCategory.STOOL: 3600,
            SpecimenCategory.CSF: 300,
            SpecimenCategory.BACTERIA: 7200,
            SpecimenCategory.PATHOLOGY: 14400,
        }
        
        limit = category_time_limits.get(record['category'], 3600)
        
        return {
            'valid': elapsed < limit,
            'specimen_id': specimen_id,
            'category': record['category'].name,
            'elapsed_seconds': elapsed,
            'time_limit': limit,
            'remaining_seconds': max(0, limit - elapsed),
        }


class HealthcareSceneController:
    """医疗场景总控制器"""

    def __init__(self, agv_grade: str = "M"):
        self.agv_grade = agv_grade
        self._task_library = HealthcareTaskLibrary()
        self._infection_monitor = InfectionControlMonitor()
        self._call_handler = PatientCallHandler(self._task_library)
        self._delivery_planner = MedicationDeliveryPlanner(self._infection_monitor)
        self._specimen_manager = SpecimenTransportManager(self._infection_monitor)
        self._task_queue: List[HealthcareTask] = []
        self._completed_tasks: List[HealthcareTask] = []

    @property
    def task_library(self) -> HealthcareTaskLibrary:
        return self._task_library

    @property
    def infection_monitor(self) -> InfectionControlMonitor:
        return self._infection_monitor

    @property
    def call_handler(self) -> PatientCallHandler:
        return self._call_handler

    @property
    def delivery_planner(self) -> MedicationDeliveryPlanner:
        return self._delivery_planner

    @property
    def specimen_manager(self) -> SpecimenTransportManager:
        return self._specimen_manager

    def add_task(self, task: HealthcareTask) -> None:
        """添加医疗任务"""
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda t: t.priority.value, reverse=True)

    def get_next_task(self) -> Optional[HealthcareTask]:
        """获取下一个最高优先级任务"""
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    def complete_task(self, task_id: str) -> bool:
        """标记任务完成"""
        for task in self._completed_tasks:
            if task.task_id == task_id:
                task.completed_at = time.time()
                task.status = 'completed'
                return True
        return False

    def get_scene_status(self) -> Dict:
        """获取场景状态"""
        return {
            'pending_tasks': len(self._task_queue),
            'completed_tasks': len(self._completed_tasks),
            'active_calls': len(self._call_handler._active_calls),
            'infection_alerts': len(self._infection_monitor._isolation_alerts),
            'agv_grade': self.agv_grade,
        }

    def generate_scene_report(self) -> Dict:
        """生成场景报告"""
        total = len(self._completed_tasks)
        on_time = sum(
            1 for t in self._completed_tasks
            if t.completed_at and t.time_constraint
            and (t.completed_at - t.created_at) <= t.time_constraint
        )
        
        return {
            'timestamp': time.time(),
            'total_deliveries': total,
            'on_time_rate': on_time / total if total > 0 else 0,
            'task_breakdown': self._get_task_breakdown(),
            'zone_status': self._get_zone_status(),
            'risk_assessment': self._get_risk_assessment(),
        }

    def _get_task_breakdown(self) -> Dict:
        result = {}
        for t in self._completed_tasks:
            result[t.task_type] = result.get(t.task_type, 0) + 1
        return result

    def _get_zone_status(self) -> Dict:
        return {
            zone.name: self._infection_monitor.get_decontamination_status(zone)
            for zone in HealthcareZone
        }

    def _get_risk_assessment(self) -> List[Dict]:
        risks = []
        for alert in self._infection_monitor._isolation_alerts[-10:]:
            risks.append({
                'zone': alert['zone'].name,
                'reason': alert['reason'],
                'timestamp': alert['timestamp'],
            })
        return risks


# 全局单例
_healthcare_controller: Optional[HealthcareSceneController] = None


def get_healthcare_scene_controller(agv_grade: str = "M") -> HealthcareSceneController:
    """获取医疗场景控制器全局实例"""
    global _healthcare_controller
    if _healthcare_controller is None:
        _healthcare_controller = HealthcareSceneController(agv_grade=agv_grade)
    return _healthcare_controller
