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
技能调度器 (Skill Dispatcher)
============================

跨模态技能协调执行器

功能:
- 多技能并发调度
- 技能依赖管理
- 资源冲突仲裁
- 技能执行监控
- AGV五级规格适配

使用示例:
    from src.control.skill_dispatcher import SkillDispatcher, SkillRequest, SkillPriority

    dispatcher = SkillDispatcher(grade='M')
    dispatcher.register_skill('grasp', grasp_controller)
    dispatcher.register_skill('navigate', nav_controller)

    # 提交技能请求
    request = SkillRequest(
        skill_name='grasp',
        params={'target': obj_pose},
        priority=SkillPriority.HIGH
    )
    result = dispatcher.dispatch(request)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Set, Callable
from enum import Enum
from collections import defaultdict
import threading
import time


class SkillPriority(Enum):
    """技能优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class SkillStatus(Enum):
    """技能执行状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """资源类型"""
    MOTOR = "motor"
    SENSOR_VISION = "sensor_vision"
    SENSOR_FORCE = "sensor_force"
    SENSOR_TACTILE = "sensor_tactile"
    SENSOR_IMU = "sensor_imu"
    POSITION = "position"
    GRIPPER = "gripper"


@dataclass
class SkillRequest:
    """技能请求"""
    skill_name: str
    params: Dict[str, Any]
    priority: SkillPriority = SkillPriority.NORMAL
    timeout_sec: float = 30.0
    request_id: str = ""
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"{self.skill_name}_{time.time():.6f}"


@dataclass
class SkillResult:
    """技能执行结果"""
    request_id: str
    skill_name: str
    status: SkillStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_sec: float = 0.0
    resources_used: Set[ResourceType] = field(default_factory=set)


@dataclass
class SkillDefinition:
    """技能定义"""
    name: str
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    required_resources: Set[ResourceType]
    estimated_duration_sec: float = 5.0
    max_retries: int = 3
    grade_requirement: str = "S"  # Minimum grade required


class SkillDispatcher:
    """
    技能调度器
    
    负责:
    - 注册和管理技能
    - 调度和执行技能请求
    - 资源冲突仲裁
    - 依赖关系管理
    - 执行监控
    """
    
    def __init__(
        self,
        grade: str = "M",
        enable_monitoring: bool = True,
        max_concurrent: Optional[int] = None
    ):
        """
        Args:
            grade: AGV等级 (S/M/L/XL/XXL)
            enable_monitoring: 是否启用执行监控
            max_concurrent: 最大并发技能数
        """
        self.grade = grade
        self.enable_monitoring = enable_monitoring
        self.max_concurrent = max_concurrent
        
        # 技能注册表
        self._skills: Dict[str, SkillDefinition] = {}
        
        # 资源锁定表
        self._resource_locks: Dict[ResourceType, str] = {}  # resource -> request_id
        self._request_resources: Dict[str, Set[ResourceType]] = defaultdict(set)
        
        # 执行状态
        self._running_requests: Dict[str, SkillRequest] = {}
        self._completed_requests: Dict[str, SkillResult] = {}
        self._skill_states: Dict[str, SkillStatus] = defaultdict(lambda: SkillStatus.IDLE)
        
        # 调度队列 (按优先级排序)
        self._pending_queue: List[SkillRequest] = []
        
        # 互斥锁
        self._lock = threading.RLock()
        
        # 监控统计
        self._stats = {
            'total_dispatched': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_cancelled': 0,
        }
        
        # AGV等级配置
        self._grade_configs = {
            'S': {'max_concurrent': 1, 'timeout_default': 30.0},
            'M': {'max_concurrent': 2, 'timeout_default': 20.0},
            'L': {'max_concurrent': 3, 'timeout_default': 15.0},
            'XL': {'max_concurrent': 4, 'timeout_default': 10.0},
            'XXL': {'max_concurrent': 6, 'timeout_default': 5.0},
        }
        config = self._grade_configs.get(grade, self._grade_configs['M'])
        # Use config's max_concurrent as default if not explicitly overridden
        if max_concurrent is None:
            self.max_concurrent = config['max_concurrent']
        elif max_concurrent > config['max_concurrent']:
            self.max_concurrent = config['max_concurrent']
        else:
            self.max_concurrent = max_concurrent
    
    def register_skill(self, skill: SkillDefinition) -> bool:
        """
        注册技能
        
        Args:
            skill: 技能定义
            
        Returns:
            是否注册成功
        """
        with self._lock:
            if skill.name in self._skills:
                print(f"[SkillDispatcher] Skill {skill.name} already registered")
                return False
            
            self._skills[skill.name] = skill
            print(f"[SkillDispatcher] Registered skill: {skill.name}, "
                  f"resources={skill.required_resources}")
            return True
    
    def unregister_skill(self, skill_name: str) -> bool:
        """注销技能"""
        with self._lock:
            if skill_name not in self._skills:
                return False
            del self._skills[skill_name]
            return True
    
    def dispatch(self, request: SkillRequest) -> SkillResult:
        """
        调度技能请求
        
        Args:
            request: 技能请求
            
        Returns:
            SkillResult: 执行结果
        """
        start_time = time.time()
        
        # 验证技能存在
        if request.skill_name not in self._skills:
            return SkillResult(
                request_id=request.request_id,
                skill_name=request.skill_name,
                status=SkillStatus.FAILED,
                error=f"Skill {request.skill_name} not found",
                execution_time_sec=time.time() - start_time
            )
        
        skill = self._skills[request.skill_name]
        
        # 检查资源冲突
        with self._lock:
            conflicting = self._check_resource_conflict(skill.required_resources, request.request_id)
            if conflicting:
                return SkillResult(
                    request_id=request.request_id,
                    skill_name=request.skill_name,
                    status=SkillStatus.FAILED,
                    error=f"Resource conflict with {conflicting}",
                    execution_time_sec=time.time() - start_time
                )
            
            # 检查并发限制
            active = len([r for r in self._running_requests.values() 
                        if r.request_id in self._resource_locks.values()])
            if active >= self.max_concurrent:
                # 加入等待队列
                self._pending_queue.append(request)
                self._pending_queue.sort(key=lambda r: r.priority.value, reverse=True)
                return SkillResult(
                    request_id=request.request_id,
                    skill_name=request.skill_name,
                    status=SkillStatus.IDLE,
                    error="Queued due to concurrent limit",
                    execution_time_sec=time.time() - start_time
                )
            
            # 锁定资源
            for res in skill.required_resources:
                self._resource_locks[res] = request.request_id
            self._request_resources[request.request_id] = skill.required_resources
            self._running_requests[request.request_id] = request
            self._skill_states[request.skill_name] = SkillStatus.RUNNING
        
        # 执行技能
        try:
            output = skill.execute_fn(request.params)
            status = SkillStatus.COMPLETED
            error = None
            self._stats['total_completed'] += 1
        except Exception as e:
            output = None
            status = SkillStatus.FAILED
            error = str(e)
            self._stats['total_failed'] += 1
        
        # 释放资源
        with self._lock:
            for res in skill.required_resources:
                if self._resource_locks.get(res) == request.request_id:
                    del self._resource_locks[res]
            if request.request_id in self._running_requests:
                del self._running_requests[request.request_id]
            self._skill_states[request.skill_name] = status
            
            # 处理等待队列
            self._process_queue()
        
        result = SkillResult(
            request_id=request.request_id,
            skill_name=request.skill_name,
            status=status,
            output=output,
            error=error,
            execution_time_sec=time.time() - start_time,
            resources_used=skill.required_resources
        )
        self._completed_requests[request.request_id] = result
        self._stats['total_dispatched'] += 1
        
        return result
    
    def cancel(self, request_id: str) -> bool:
        """取消技能请求"""
        with self._lock:
            if request_id in self._running_requests:
                request = self._running_requests[request_id]
                skill = self._skills.get(request.skill_name)
                if skill:
                    for res in skill.required_resources:
                        if self._resource_locks.get(res) == request_id:
                            del self._resource_locks[res]
                    del self._running_requests[request_id]
                self._skill_states[request.skill_name] = SkillStatus.CANCELLED
                self._stats['total_cancelled'] += 1
                return True
            return False
    
    def get_status(self, skill_name: str) -> SkillStatus:
        """获取技能状态"""
        return self._skill_states.get(skill_name, SkillStatus.IDLE)
    
    def get_result(self, request_id: str) -> Optional[SkillResult]:
        """获取执行结果"""
        return self._completed_requests.get(request_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        return self._stats.copy()
    
    def _check_resource_conflict(
        self,
        required: Set[ResourceType],
        request_id: str
    ) -> Optional[str]:
        """检查资源冲突"""
        for res in required:
            locked_by = self._resource_locks.get(res)
            if locked_by and locked_by != request_id:
                return locked_by
        return None
    
    def _process_queue(self):
        """处理等待队列"""
        while self._pending_queue and len(self._running_requests) < self.max_concurrent:
            request = self._pending_queue.pop(0)
            # 重新尝试调度 (递归会释放锁后重入)
            skill = self._skills.get(request.skill_name)
            if skill:
                conflicting = self._check_resource_conflict(skill.required_resources, request.request_id)
                if not conflicting:
                    for res in skill.required_resources:
                        self._resource_locks[res] = request.request_id
                    self._request_resources[request.request_id] = skill.required_resources
                    self._running_requests[request.request_id] = request
                    self._skill_states[request.skill_name] = SkillStatus.RUNNING


# ============================================================================
# AGV五级技能调度规格
# ============================================================================

AGV_SKILL_DISPATCHER_GRADES = {
    'S': {
        'max_concurrent': 1,
        'timeout_default': 30.0,
        'skill_count': 5,
        'monitoring': False,
    },
    'M': {
        'max_concurrent': 2,
        'timeout_default': 20.0,
        'skill_count': 10,
        'monitoring': True,
    },
    'L': {
        'max_concurrent': 3,
        'timeout_default': 15.0,
        'skill_count': 20,
        'monitoring': True,
    },
    'XL': {
        'max_concurrent': 4,
        'timeout_default': 10.0,
        'skill_count': 50,
        'monitoring': True,
    },
    'XXL': {
        'max_concurrent': 6,
        'timeout_default': 5.0,
        'skill_count': 100,
        'monitoring': True,
    },
}


def get_skill_dispatcher_spec(grade: str) -> dict:
    """获取AGV指定等级的技能调度规格"""
    return AGV_SKILL_DISPATCHER_GRADES.get(grade, AGV_SKILL_DISPATCHER_GRADES['M'])


def create_skill_dispatcher(grade: str = 'M') -> SkillDispatcher:
    """创建技能调度器"""
    config = get_skill_dispatcher_spec(grade)
    return SkillDispatcher(
        grade=grade,
        enable_monitoring=config['monitoring'],
        max_concurrent=config['max_concurrent']
    )


# ============================================================================
# 预定义技能工厂
# ============================================================================

def create_grasp_skill(
    force_controller,
    tactile_controller
) -> SkillDefinition:
    """创建抓取技能"""
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        target = params.get('target_position')
        force = params.get('grasp_force', 10.0)
        # 模拟抓取流程
        return {'success': True, 'grasp_force': force, 'object_pose': target}
    
    return SkillDefinition(
        name='grasp',
        execute_fn=execute,
        required_resources={ResourceType.MOTOR, ResourceType.SENSOR_FORCE, ResourceType.SENSOR_TACTILE},
        estimated_duration_sec=3.0,
        grade_requirement='M'
    )


def create_navigate_skill(
    nav_controller,
    obstacle_avoider
) -> SkillDefinition:
    """创建导航技能"""
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        target = params.get('target')
        return {'success': True, 'path_length': 2.5, 'target': target}
    
    return SkillDefinition(
        name='navigate',
        execute_fn=execute,
        required_resources={ResourceType.MOTOR, ResourceType.POSITION, ResourceType.SENSOR_VISION},
        estimated_duration_sec=10.0,
        grade_requirement='S'
    )


def create_place_skill(motor_controller) -> SkillDefinition:
    """创建放置技能"""
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        position = params.get('position')
        return {'success': True, 'placed_at': position}
    
    return SkillDefinition(
        name='place',
        execute_fn=execute,
        required_resources={ResourceType.MOTOR, ResourceType.GRIPPER},
        estimated_duration_sec=2.0,
        grade_requirement='M'
    )
