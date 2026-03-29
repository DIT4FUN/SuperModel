"""
安全控制器模块
==============

机器人安全监控与故障容忍
- 关节限位监控
- 速度/加速度限制
- 碰撞检测与响应
- 紧急停止
- 看门狗监控
- 故障恢复

AGV五级安全等级:
- S级: 基础软件限位
- M级: +速度监控
- L级: +碰撞检测
- XL级: +实时看门狗
- XXL级: +故障容忍与恢复
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable, Dict, Any, Set
from enum import Enum
import time
import threading


class SafetyLevel(Enum):
    """安全等级"""
    S = "S"   # 基础限位
    M = "M"   # 速度监控
    L = "L"   # 碰撞检测
    XL = "XL" # 看门狗
    XXL = "XXL"  # 故障容忍


class SafetyEvent(Enum):
    """安全事件类型"""
    JOINT_LIMIT = "joint_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACCELERATION_LIMIT = "acceleration_limit"
    COLLISION_DETECTED = "collision_detected"
    EMERGENCY_STOP = "emergency_stop"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    TORQUE_LIMIT = "torque_limit"
    TEMPERATURE_HIGH = "temperature_high"
    POWER_EXCEPTION = "power_exception"


class SafetyResponse(Enum):
    """安全响应策略"""
    WARNING = "warning"          # 仅警告
    SLOWDOWN = "slowdown"        # 减速
    STOP = "stop"                # 停止
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止
    FAULT_TOLERANT = "fault_tolerant"   # 故障容忍


@dataclass
class SafetyConfig:
    """安全配置"""
    # 关节限位
    joint_limits_lower: np.ndarray       # 关节下限 (rad)
    joint_limits_upper: np.ndarray       # 关节上限 (rad)
    
    # 速度限制
    velocity_limits: np.ndarray           # rad/s
    
    # 加速度限制
    acceleration_limits: np.ndarray      # rad/s^2
    
    # 力/力矩限制
    torque_limits: np.ndarray = field(default_factory=lambda: np.zeros(6))  # Nm
    force_limits: np.ndarray = field(default_factory=lambda: np.zeros(6))   # N
    
    # 碰撞检测
    collision_threshold: float = 10.0     # N, 碰撞力阈值
    collision_time_threshold: float = 0.1 # s, 碰撞判定时间
    
    # 看门狗
    watchdog_timeout: float = 0.1         # s, 超时阈值
    
    # 温度限制
    temperature_limits: Tuple[float, float] = (0.0, 80.0)  # 摄氏度
    
    # 安全等级
    safety_level: SafetyLevel = SafetyLevel.M
    
    # 故障容忍
    max_fault_count: int = 3             # 最大容错次数
    recovery_timeout: float = 5.0        # 恢复超时 (s)
    
    # 速度警告阈值
    velocity_warning_ratio: float = 0.8  # 警告阈值比例
    
    def __post_init__(self):
        if isinstance(self.joint_limits_lower, (list, tuple)):
            self.joint_limits_lower = np.array(self.joint_limits_lower, dtype=np.float32)
        if isinstance(self.joint_limits_upper, (list, tuple)):
            self.joint_limits_upper = np.array(self.joint_limits_upper, dtype=np.float32)
        if isinstance(self.velocity_limits, (list, tuple)):
            self.velocity_limits = np.array(self.velocity_limits, dtype=np.float32)
        if isinstance(self.acceleration_limits, (list, tuple)):
            self.acceleration_limits = np.array(self.acceleration_limits, dtype=np.float32)
        if isinstance(self.torque_limits, (list, tuple)):
            self.torque_limits = np.array(self.torque_limits, dtype=np.float32)
        if isinstance(self.force_limits, (list, tuple)):
            self.force_limits = np.array(self.force_limits, dtype=np.float32)


@dataclass
class SafetyEventRecord:
    """安全事件记录"""
    event_type: SafetyEvent
    timestamp: float
    severity: int                    # 1-5, 5最严重
    message: str
    joint_index: Optional[int] = None
    value: Optional[float] = None
    limit_value: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass 
class JointStateSnapshot:
    """关节状态快照"""
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: Optional[np.ndarray] = None
    torques: Optional[np.ndarray] = None
    timestamp: float = 0.0


class SafetyController:
    """
    安全控制器
    
    监控机器人状态，检测异常，执行安全响应。
    
    使用示例:
    ```python
    config = SafetyConfig(
        joint_limits_lower=np.array([-3.14, -2.5, -3.14, -3.14, -3.14, -3.14]),
        joint_limits_upper=np.array([3.14, 2.5, 3.14, 3.14, 3.14, 3.14]),
        velocity_limits=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
        torque_limits=np.array([100, 100, 80, 40, 40, 20]),
        safety_level=SafetyLevel.L,
    )
    safety = SafetyController(config)
    
    # 定期调用检查
    snapshot = JointStateSnapshot(positions=joints, velocities=vels, torques=torques)
    result = safety.check(snapshot)
    if not result.safe:
        safety.execute_response(result)
    ```
    """
    
    # 安全等级对应的最低功能要求
    LEVEL_FEATURES = {
        SafetyLevel.S: {"joint_limits", "velocity_limits"},
        SafetyLevel.M: {"joint_limits", "velocity_limits", "velocity_monitoring"},
        SafetyLevel.L: {"joint_limits", "velocity_limits", "collision_detection"},
        SafetyLevel.XL: {"joint_limits", "velocity_limits", "collision_detection", "watchdog"},
        SafetyLevel.XXL: {"joint_limits", "velocity_limits", "collision_detection", 
                         "watchdog", "fault_tolerance", "recovery"},
    }
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self._event_history: List[SafetyEventRecord] = []
        self._fault_count = 0
        self._last_check_time = time.time()
        self._watchdog_ok = True
        self._emergency_stopped = False
        self._enabled = True
        self._lock = threading.RLock()
        
        # 回调函数
        self._callbacks: Dict[SafetyEvent, List[Callable]] = {
            event: [] for event in SafetyEvent
        }
        
        # 状态
        self._last_state: Optional[JointStateSnapshot] = None
        self._collision_start_time: Optional[float] = None
        
        # 统计
        self._check_count = 0
        self._total_violations = 0
    
    @property
    def safety_level(self) -> SafetyLevel:
        return self.config.safety_level
    
    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped
    
    @property
    def fault_count(self) -> int:
        return self._fault_count
    
    @property
    def event_history(self) -> List[SafetyEventRecord]:
        return list(self._event_history)
    
    def enable(self):
        """启用安全监控"""
        with self._lock:
            self._enabled = True
    
    def disable(self):
        """禁用安全监控 (谨慎使用)"""
        with self._lock:
            self._enabled = False
    
    def register_callback(self, event: SafetyEvent, callback: Callable[[SafetyEventRecord], None]):
        """注册安全事件回调"""
        self._callbacks[event].append(callback)
    
    def check(self, state: JointStateSnapshot) -> 'SafetyCheckResult':
        """
        检查安全状态
        
        Args:
            state: 当前关节状态快照
            
        Returns:
            SafetyCheckResult: 检查结果
        """
        with self._lock:
            if not self._enabled:
                return SafetyCheckResult(safe=True, events=[])
            
            self._check_count += 1
            violations: List[SafetyEventRecord] = []
            now = time.time()
            
            joints = np.asarray(state.positions)
            vels = np.asarray(state.velocities)
            
            # 1. 关节限位检查 (所有等级)
            if "joint_limits" in self.LEVEL_FEATURES[self.config.safety_level]:
                for i in range(len(joints)):
                    if joints[i] < self.config.joint_limits_lower[i]:
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.JOINT_LIMIT,
                            timestamp=now,
                            severity=4,
                            message=f"关节{i}位置超下限: {joints[i]:.3f} < {self.config.joint_limits_lower[i]:.3f}",
                            joint_index=i,
                            value=float(joints[i]),
                            limit_value=float(self.config.joint_limits_lower[i])
                        ))
                    elif joints[i] > self.config.joint_limits_upper[i]:
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.JOINT_LIMIT,
                            timestamp=now,
                            severity=4,
                            message=f"关节{i}位置超上限: {joints[i]:.3f} > {self.config.joint_limits_upper[i]:.3f}",
                            joint_index=i,
                            value=float(joints[i]),
                            limit_value=float(self.config.joint_limits_upper[i])
                        ))
            
            # 2. 速度限制检查 (M级及以上)
            if "velocity_limits" in self.LEVEL_FEATURES[self.config.safety_level]:
                for i in range(len(vels)):
                    abs_vel = abs(vels[i])
                    if abs_vel > self.config.velocity_limits[i]:
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.VELOCITY_LIMIT,
                            timestamp=now,
                            severity=3,
                            message=f"关节{i}速度超限: {abs_vel:.3f} > {self.config.velocity_limits[i]:.3f}",
                            joint_index=i,
                            value=float(abs_vel),
                            limit_value=float(self.config.velocity_limits[i])
                        ))
                    elif abs_vel > self.config.velocity_limits[i] * self.config.velocity_warning_ratio:
                        # 警告级别
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.VELOCITY_LIMIT,
                            timestamp=now,
                            severity=1,
                            message=f"关节{i}速度接近限值: {abs_vel:.3f} > {self.config.velocity_limits[i] * self.config.velocity_warning_ratio:.3f}",
                            joint_index=i,
                            value=float(abs_vel),
                            limit_value=float(self.config.velocity_limits[i] * self.config.velocity_warning_ratio)
                        ))
            
            # 3. 加速度检查 (L级及以上)
            if state.accelerations is not None and "collision_detection" in self.LEVEL_FEATURES[self.config.safety_level]:
                accels = np.asarray(state.accelerations)
                for i in range(len(accels)):
                    if abs(accels[i]) > self.config.acceleration_limits[i]:
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.ACCELERATION_LIMIT,
                            timestamp=now,
                            severity=3,
                            message=f"关节{i}加速度超限: {abs(accels[i]):.3f} > {self.config.acceleration_limits[i]:.3f}",
                            joint_index=i,
                            value=float(abs(accels[i])),
                            limit_value=float(self.config.acceleration_limits[i])
                        ))
            
            # 4. 力矩限制检查 (所有等级)
            if state.torques is not None:
                torques = np.asarray(state.torques)
                for i in range(len(torques)):
                    if abs(torques[i]) > self.config.torque_limits[i]:
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.TORQUE_LIMIT,
                            timestamp=now,
                            severity=4,
                            message=f"关节{i}力矩超限: {abs(torques[i]):.3f} > {self.config.torque_limits[i]:.3f}",
                            joint_index=i,
                            value=float(abs(torques[i])),
                            limit_value=float(self.config.torque_limits[i])
                        ))
            
            # 5. 碰撞检测 (L级及以上)
            if "collision_detection" in self.LEVEL_FEATURES[self.config.safety_level] and state.torques is not None:
                contact_force = np.linalg.norm(state.torques[:3] if len(state.torques) >= 3 else state.torques)
                if contact_force > self.config.collision_threshold:
                    if self._collision_start_time is None:
                        self._collision_start_time = now
                    elif now - self._collision_start_time > self.config.collision_time_threshold:
                        violations.append(SafetyEventRecord(
                            event_type=SafetyEvent.COLLISION_DETECTED,
                            timestamp=now,
                            severity=5,
                            message=f"碰撞检测: 力={contact_force:.1f}N, 持续>{self.config.collision_time_threshold}s"
                        ))
                        self._collision_start_time = None
                else:
                    self._collision_start_time = None
            
            # 6. 看门狗检查 (XL级及以上)
            if "watchdog" in self.LEVEL_FEATURES[self.config.safety_level]:
                dt = now - self._last_check_time
                if dt > self.config.watchdog_timeout:
                    violations.append(SafetyEventRecord(
                        event_type=SafetyEvent.WATCHDOG_TIMEOUT,
                        timestamp=now,
                        severity=4,
                        message=f"看门狗超时: {dt:.3f}s > {self.config.watchdog_timeout:.3f}s"
                    ))
                    self._watchdog_ok = False
            
            self._last_check_time = now
            self._last_state = state
            
            # 记录事件
            for v in violations:
                self._event_history.append(v)
                if v.severity >= 3:
                    self._total_violations += 1
            
            # 保持历史记录在合理范围
            if len(self._event_history) > 1000:
                self._event_history = self._event_history[-500:]
            
            # 触发回调
            for v in violations:
                for cb in self._callbacks[v.event_type]:
                    try:
                        cb(v)
                    except Exception:
                        pass
            
            return SafetyCheckResult(
                safe=all(v.severity < 3 for v in violations),
                events=violations,
                emergency_stop=any(v.event_type == SafetyEvent.EMERGENCY_STOP for v in violations),
                watchdog_ok=self._watchdog_ok
            )
    
    def execute_response(self, result: 'SafetyCheckResult') -> SafetyResponse:
        """
        执行安全响应
        
        Args:
            result: 检查结果
            
        Returns:
            SafetyResponse: 执行的响应策略
        """
        if result.emergency_stop or self._emergency_stopped:
            self._emergency_stopped = True
            return SafetyResponse.EMERGENCY_STOP
        
        if not result.safe:
            max_severity = max(v.severity for v in result.events)
            
            if max_severity >= 5:
                self._fault_count += 1
                if "fault_tolerance" in self.LEVEL_FEATURES[self.config.safety_level]:
                    if self._fault_count > self.config.max_fault_count:
                        self._emergency_stopped = True
                        return SafetyResponse.EMERGENCY_STOP
                    return SafetyResponse.FAULT_TOLERANT
                self._emergency_stopped = True
                return SafetyResponse.EMERGENCY_STOP
            elif max_severity >= 4:
                return SafetyResponse.STOP
            elif max_severity >= 3:
                return SafetyResponse.SLOWDOWN
            else:
                return SafetyResponse.WARNING
        
        return SafetyResponse.WARNING
    
    def emergency_stop(self):
        """触发紧急停止"""
        with self._lock:
            self._emergency_stopped = True
            self._event_history.append(SafetyEventRecord(
                event_type=SafetyEvent.EMERGENCY_STOP,
                timestamp=time.time(),
                severity=5,
                message="手动触发紧急停止"
            ))
    
    def reset(self):
        """重置安全控制器"""
        with self._lock:
            self._emergency_stopped = False
            self._fault_count = 0
            self._watchdog_ok = True
            self._collision_start_time = None
            self._event_history.clear()
    
    def get_safety_status(self) -> Dict[str, Any]:
        """获取安全状态摘要"""
        with self._lock:
            recent_events = [e for e in self._event_history 
                           if time.time() - e.timestamp < 10.0]
            return {
                "enabled": self._enabled,
                "emergency_stopped": self._emergency_stopped,
                "safety_level": self.config.safety_level.value,
                "fault_count": self._fault_count,
                "watchdog_ok": self._watchdog_ok,
                "total_checks": self._check_count,
                "total_violations": self._total_violations,
                "recent_events": len(recent_events),
                "collision_in_progress": self._collision_start_time is not None,
            }
    
    def compute_safe_velocity(
        self, 
        current_vel: np.ndarray, 
        desired_vel: np.ndarray
    ) -> np.ndarray:
        """
        计算安全速度 (限幅)
        
        Args:
            current_vel: 当前速度
            desired_vel: 期望速度
            
        Returns:
            np.ndarray: 安全速度
        """
        safe_vel = np.clip(desired_vel, 
                         -self.config.velocity_limits,
                         self.config.velocity_limits)
        
        # 检查是否在限位附近, 减小速度
        if self._last_state is not None:
            positions = self._last_state.positions
            for i in range(len(positions)):
                dist_to_lower = positions[i] - self.config.joint_limits_lower[i]
                dist_to_upper = self.config.joint_limits_upper[i] - positions[i]
                min_dist = min(dist_to_lower, dist_to_upper)
                
                if min_dist < 0.1:  # 10cm以内
                    # 渐进减速
                    ratio = min(1.0, min_dist / 0.1)
                    safe_vel[i] *= ratio
        
        return safe_vel


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    safe: bool
    events: List[SafetyEventRecord]
    emergency_stop: bool = False
    watchdog_ok: bool = True
    
    @property
    def critical_events(self) -> List[SafetyEventRecord]:
        return [e for e in self.events if e.severity >= 3]
    
    @property
    def warnings(self) -> List[SafetyEventRecord]:
        return [e for e in self.events if e.severity < 3]


def get_safety_spec(level: SafetyLevel) -> Dict[str, Any]:
    """
    获取指定安全等级的技术规格
    
    Args:
        level: 安全等级
        
    Returns:
        Dict: 技术规格
    """
    specs = {
        SafetyLevel.S: {
            "level": "S",
            "description": "基础软件限位",
            "features": ["关节位置限位", "软件速度限幅"],
            "response_time_ms": 100,
            "redundancy": "无",
            "典型应用": "教育/实验",
            "典型价格": "¥5,000-15,000",
        },
        SafetyLevel.M: {
            "level": "M", 
            "description": "速度实时监控",
            "features": ["关节位置限位", "速度监控", "加速度监控", "警告系统"],
            "response_time_ms": 50,
            "redundancy": "单通道",
            "typical_application": "室内服务/轻工业",
            "typical_price": "¥15,000-50,000",
        },
        SafetyLevel.L: {
            "level": "L",
            "description": "碰撞检测与响应",
            "features": ["关节位置限位", "速度监控", "碰撞检测", "力矩监控", "自动减速"],
            "response_time_ms": 20,
            "redundancy": "双通道",
            "typical_application": "工业制造/物流",
            "typical_price": "¥50,000-150,000",
        },
        SafetyLevel.XL: {
            "level": "XL",
            "description": "实时看门狗与监控",
            "features": ["关节位置限位", "速度监控", "碰撞检测", "看门狗", "实时故障诊断", "自动停机"],
            "response_time_ms": 5,
            "redundancy": "双通道+独立监控",
            "typical_application": "复杂装配/精密操作",
            "typical_price": "¥150,000-500,000",
        },
        SafetyLevel.XXL: {
            "level": "XXL",
            "description": "故障容忍与恢复",
            "features": [
                "关节位置限位", "速度监控", "碰撞检测", "看门狗",
                "故障容忍", "自动恢复", "冗余传感", "预测性维护"
            ],
            "response_time_ms": 1,
            "redundancy": "全冗余",
            "typical_application": "多机协作/户外全地形",
            "typical_price": "> ¥500,000",
        },
    }
    return specs.get(level, specs[SafetyLevel.M])
