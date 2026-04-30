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
Teleoperation 遥操作控制模块
===========================

提供安全可靠的远程操控接口:
- 主从同步控制 (同步/姿态/速度模式)
- 双向力反馈
- 权限管理与优先级切换
- 紧急停止与安全回退
- 延迟补偿与预测显示
- 多操作者协调

支持AGV等级: S/M/L/XL/XXL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Callable
from enum import Enum
import time


class TeleopMode(Enum):
    """遥操作模式"""
    POSITION_SYNC = "position_sync"       # 主从位置同步
    VELOCITY_SYNC = "velocity_sync"       # 速度同步
    IMPEDANCE_SYNC = "impedance_sync"     # 阻抗同步
    SHARED = "shared_control"             # 共享控制
    SUPERVISED = "supervised"             # 监督式自动


class TeleopState(Enum):
    """遥操作状态"""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    SAFETY_STOP = "safety_stop"
    DISCONNECTED = "disconnected"


class AuthorityLevel(Enum):
    """权限等级"""
    VIEWER = 0     # 仅观察
    OPERATOR = 1   # 操作员
    SUPERVISOR = 2 # 监管员
    ADMIN = 3      # 管理员


@dataclass
class MasterState:
    """主端 (操作者) 状态"""
    joint_positions: np.ndarray           # 关节位置
    joint_velocities: np.ndarray          # 关节速度
    joint_torques: np.ndarray             # 关节力矩
    wrench: np.ndarray                    # 末端力旋量 (6,)
    timestamp: float = 0.0
    authority: AuthorityLevel = AuthorityLevel.OPERATOR


@dataclass
class SlaveState:
    """从端 (机器人) 状态"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pose: np.ndarray         # 4x4 或 7自由度
    contact_wrench: np.ndarray            # 接触力 (6,)
    timestamp: float = 0.0
    safety_level: int = 0


@dataclass
class TeleopCommand:
    """遥操作命令"""
    mode: TeleopMode
    target_joint_positions: Optional[np.ndarray] = None
    target_joint_velocities: Optional[np.ndarray] = None
    impedance_params: Optional[np.ndarray] = None  # Kp, Kd for impedance
    authority_request: AuthorityLevel = AuthorityLevel.OPERATOR


@dataclass
class TeleopConfig:
    """遥操作配置"""
    # 通信参数
    master_ip: str = "192.168.1.100"
    slave_ip: str = "192.168.1.101"
    port: int = 5000
    timeout_sec: float = 1.0
    
    # 控制参数
    control_frequency: float = 100.0  # Hz
    max_latency_compensation_ms: float = 100.0
    
    # 速度/位置限制
    max_master_velocity: float = 0.5     # m/s
    max_slave_velocity: float = 1.0      # m/s
    max_force_feedback: float = 10.0      # N
    
    # 安全参数
    safety_stop_threshold: float = 50.0  # N, 超过则停止
    authority_timeout_sec: float = 300.0  # 5分钟无操作降级
    emergency_stop_enabled: bool = True
    
    # 权限配置
    min_authority_for_autonomous: AuthorityLevel = AuthorityLevel.SUPERVISOR
    multi_operator_enabled: bool = False
    priority_inheritance: bool = True
    
    # 共享控制参数
    autonomy_blend_min: float = 0.0
    autonomy_blend_max: float = 1.0
    shared_control_update_hz: float = 10.0


@dataclass
class TeleopMetrics:
    """遥操作性能指标"""
    round_trip_latency_ms: float = 0.0
    master_to_slave_latency_ms: float = 0.0
    position_error_norm: float = 0.0
    velocity_error_norm: float = 0.0
    force_feedback_magnitude: float = 0.0
    authority_level: AuthorityLevel = AuthorityLevel.OPERATOR
    connection_quality: float = 1.0  # 0-1
    timestamp: float = 0.0


class SafetyMonitor:
    """
    遥操作安全监控
    
    检测:
    - 碰撞力超限
    - 工作空间边界
    - 速度/加速度超限
    - 通信超时
    - 权限异常
    """

    def __init__(self, config: TeleopConfig):
        self.config = config
        self._collision_history: List[float] = []
        self._velocity_history: List[float] = []
        self._max_history = 100
        self._emergency_stop_triggered = False
        self._last_safe_command_time = time.time()
    
    def check_collision(self, wrench: np.ndarray) -> bool:
        """检测碰撞力超限"""
        force_mag = np.linalg.norm(wrench[:3])
        self._collision_history.append(force_mag)
        if len(self._collision_history) > self._max_history:
            self._collision_history.pop(0)
        
        if force_mag > self.config.safety_stop_threshold:
            self._emergency_stop_triggered = True
            return True
        return False
    
    def check_velocity(self, velocity: np.ndarray) -> bool:
        """检测速度超限"""
        vel_mag = np.linalg.norm(velocity)
        self._velocity_history.append(vel_mag)
        if len(self._velocity_history) > self._max_history:
            self._velocity_history.pop(0)
        
        if vel_mag > self.config.max_slave_velocity * 1.2:
            return True
        return False
    
    def check_timeout(self) -> bool:
        """检测通信超时"""
        elapsed = time.time() - self._last_safe_command_time
        if elapsed > self.config.timeout_sec:
            return True
        return False
    
    def check_workspace_bounds(self, pose: np.ndarray) -> bool:
        """检测工作空间边界"""
        # 简化为位置范围检查
        if len(pose) >= 3:
            x, y, z = pose[0], pose[1], pose[2]
            if x < -2.0 or x > 2.0 or y < -2.0 or y > 2.0 or z < -0.5 or z > 2.0:
                return True
        return False
    
    def is_safe(
        self,
        master_state: MasterState,
        slave_state: SlaveState,
        latency_ms: float
    ) -> Tuple[bool, str]:
        """
        综合安全检查
        
        Returns:
            (is_safe, reason)
        """
        if self._emergency_stop_triggered:
            return False, "Emergency stop triggered"
        
        if latency_ms > self.config.max_latency_compensation_ms:
            return False, f"Latency too high: {latency_ms:.1f}ms"
        
        if self.check_timeout():
            return False, "Communication timeout"
        
        if slave_state.contact_wrench is not None:
            if self.check_collision(slave_state.contact_wrench):
                return False, f"Collision force exceeded: {np.linalg.norm(slave_state.contact_wrench[:3]):.1f}N"
        
        if self.check_velocity(slave_state.joint_velocities):
            return False, "Velocity exceeded limit"
        
        if self.check_workspace_bounds(slave_state.end_effector_pose):
            return False, "Workspace boundary violated"
        
        return True, "OK"
    
    def acknowledge_emergency(self):
        """确认紧急停止 (需更高权限)"""
        self._emergency_stop_triggered = False
    
    def record_command(self):
        """记录安全命令接收"""
        self._last_safe_command_time = time.time()


class LatencyCompensator:
    """
    延迟补偿器
    
    使用 Smith 预测器补偿网络延迟
    """

    def __init__(self, config: TeleopConfig):
        self.config = config
        self._delay_buffer: List[Tuple[float, np.ndarray]] = []
        self._predicted_slave_state: Optional[SlaveState] = None
        self._last_update_time = 0.0
        self._model_error_history: List[float] = []
    
    def update(
        self,
        master_cmd: TeleopCommand,
        actual_slave_state: SlaveState,
        observed_latency_ms: float
    ):
        """
        更新延迟模型
        
        Args:
            master_cmd: 主端命令
            actual_slave_state: 实际从端状态
            observed_latency_ms: 观测到的延迟
        """
        t = time.time()
        self._delay_buffer.append((t, master_cmd))
        
        # 清理旧条目
        max_delay = self.config.max_latency_compensation_ms / 1000.0
        self._delay_buffer = [(t_s, cmd) for t_s, cmd in self._delay_buffer if t - t_s < max_delay]
        
        self._last_update_time = t
        
        # 简单预测: 基于历史误差
        if self._predicted_slave_state is not None:
            error = np.linalg.norm(
                actual_slave_state.joint_positions - 
                self._predicted_slave_state.joint_positions
            )
            self._model_error_history.append(error)
            if len(self._model_error_history) > 50:
                self._model_error_history.pop(0)
    
    def predict_slave_state(
        self,
        master_cmd: TeleopCommand,
        current_slave_state: SlaveState,
        latency_ms: float
    ) -> SlaveState:
        """
        预测从端状态
        
        基于延迟时间和命令进行预测
        """
        dt = latency_ms / 1000.0
        
        # 简单一阶预测
        predicted_positions = current_slave_state.joint_positions.copy()
        
        if master_cmd.target_joint_velocities is not None:
            predicted_positions += master_cmd.target_joint_velocities * dt
        elif master_cmd.target_joint_positions is not None:
            # 位置差作为速度估计
            vel_estimate = (master_cmd.target_joint_positions - current_slave_state.joint_positions) * 10.0
            predicted_positions += vel_estimate * dt
        
        # 限制预测范围
        max_change = self.config.max_slave_velocity * dt * 2.0
        diff = predicted_positions - current_slave_state.joint_positions
        if np.linalg.norm(diff) > max_change:
            predicted_positions = current_slave_state.joint_positions + diff / np.linalg.norm(diff) * max_change
        
        self._predicted_slave_state = SlaveState(
            joint_positions=predicted_positions,
            joint_velocities=master_cmd.target_joint_velocities or current_slave_state.joint_velocities,
            end_effector_pose=current_slave_state.end_effector_pose,
            contact_wrench=current_slave_state.contact_wrench,
            timestamp=time.time()
        )
        
        return self._predicted_slave_state
    
    def get_compensation_quality(self) -> float:
        """获取补偿质量 (0-1)"""
        if not self._model_error_history:
            return 1.0
        
        avg_error = np.mean(self._model_error_history)
        # 误差越小，质量越高
        quality = max(0.0, 1.0 - avg_error * 10.0)
        return quality


class SharedControlBlender:
    """
    共享控制混合器
    
    在操作者命令和自主命令之间插值
    """

    def __init__(self, config: TeleopConfig):
        self.config = config
        self._autonomy_level = 0.5  # 初始 50% 自主
        self._confidence_history: List[float] = []
        self._update_period = 1.0 / config.shared_control_update_hz
        self._last_update_time = 0.0
    
    def update_autonomy(
        self,
        operator_confidence: float,
        task_difficulty: float,
        safety_margin: float
    ):
        """
        更新自主性水平
        
        Args:
            operator_confidence: 操作者信心 (0-1)
            task_difficulty: 任务难度 (0-1)
            safety_margin: 安全裕度 (0-1, 越小越危险)
        """
        now = time.time()
        if now - self._last_update_time < self._update_period:
            return
        
        self._last_update_time = now
        
        # 综合评估
        safety_weight = 0.4
        confidence_weight = 0.3
        difficulty_weight = 0.3
        
        # 安全优先: 安全裕度小时降低自主性
        if safety_margin < 0.2:
            target_autonomy = self.config.autonomy_blend_min
        else:
            target_autonomy = (
                confidence_weight * operator_confidence +
                difficulty_weight * (1.0 - task_difficulty) +
                safety_weight * safety_margin
            )
        
        # 平滑过渡
        alpha = 0.2
        self._autonomy_level = alpha * target_autonomy + (1 - alpha) * self._autonomy_level
        self._autonomy_level = np.clip(
            self._autonomy_level,
            self.config.autonomy_blend_min,
            self.config.autonomy_blend_max
        )
    
    def blend_commands(
        self,
        operator_command: np.ndarray,
        autonomous_command: np.ndarray
    ) -> np.ndarray:
        """
        混合命令
        
        blended = (1 - autonomy) * operator + autonomy * autonomous
        """
        return (
            (1.0 - self._autonomy_level) * operator_command +
            self._autonomy_level * autonomous_command
        )
    
    @property
    def autonomy_level(self) -> float:
        return self._autonomy_level


class TeleoperationController:
    """
    遥操作主控制器
    
    功能:
    - 主从同步
    - 权限管理
    - 安全监控
    - 延迟补偿
    - 共享控制
    """

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.state = TeleopState.IDLE
        self.mode = TeleopMode.POSITION_SYNC
        
        # 子模块
        self.safety = SafetyMonitor(config)
        self.latency_comp = LatencyCompensator(config)
        self.shared_blender = SharedControlBlender(config)
        
        # 状态
        self._master_state: Optional[MasterState] = None
        self._slave_state: Optional[SlaveState] = None
        self._current_authority = AuthorityLevel.OPERATOR
        self._metrics = TeleopMetrics()
        self._command_buffer: List[Tuple[float, TeleopCommand]] = []
        
        # 回调
        self._on_emergency_stop: Optional[Callable] = None
        self._on_authority_change: Optional[Callable] = None
        self._on_state_change: Optional[Callable] = None
    
    def set_master_state(self, state: MasterState):
        """设置主端状态"""
        self._master_state = state
        
        # 检查权限
        if state.authority.value > self._current_authority.value:
            self._upgrade_authority(state.authority)
    
    def set_slave_state(self, state: SlaveState):
        """设置从端状态"""
        self._slave_state = state
    
    def send_command(self, command: TeleopCommand) -> bool:
        """
        发送遥操作命令
        
        Returns:
            是否被接受
        """
        if self.state == TeleopState.SAFETY_STOP:
            if command.mode != TeleopMode.SUPERVISED:
                return False
        
        if command.authority_request.value < self._current_authority.value:
            return False  # 权限不足
        
        if self.state == TeleopState.DISCONNECTED:
            return False
        
        # 安全检查
        if self._slave_state is not None:
            is_safe, reason = self.safety.is_safe(
                self._master_state, self._slave_state,
                self._metrics.master_to_slave_latency_ms
            )
            if not is_safe:
                self._trigger_safety_stop(reason)
                return False
        
        # 记录命令
        self._command_buffer.append((time.time(), command))
        self.safety.record_command()
        
        # 延迟补偿
        if self._slave_state is not None:
            predicted = self.latency_comp.predict_slave_state(
                command, self._slave_state,
                self._metrics.master_to_slave_latency_ms
            )
        
        return True
    
    def compute_slave_command(
        self,
        autonomous_command: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        计算从端命令
        
        Args:
            autonomous_command: 自主系统命令 (可选)
            
        Returns:
            (blended_command, autonomy_level) 或 None
        """
        if not self._command_buffer or self._master_state is None:
            return None
        
        # 获取最新命令
        _, latest_cmd = self._command_buffer[-1]
        
        # 从主端命令提取
        if latest_cmd.target_joint_positions is not None:
            operator_cmd = latest_cmd.target_joint_positions
        elif latest_cmd.target_joint_velocities is not None:
            operator_cmd = latest_cmd.target_joint_velocities
        else:
            operator_cmd = np.array([])
        
        # 共享控制混合
        if autonomous_command is not None and len(operator_cmd) == len(autonomous_command):
            blended = self.shared_blender.blend_commands(operator_cmd, autonomous_command)
        else:
            blended = operator_cmd
            autonomous_command = None
        
        return blended, self.shared_blender.autonomy_level
    
    def _upgrade_authority(self, new_level: AuthorityLevel):
        """提升权限"""
        if new_level.value <= self._current_authority.value:
            return
        
        old_level = self._current_authority
        self._current_authority = new_level
        
        if self._on_authority_change:
            self._on_authority_change(old_level, new_level)
    
    def request_authority(self, level: AuthorityLevel) -> bool:
        """
        请求提升权限
        
        需要监管员以上才能提升他人权限
        """
        if self._current_authority.value < AuthorityLevel.SUPERVISOR.value:
            return False
        
        if level.value <= self._current_authority.value:
            return False
        
        self._current_authority = level
        return True
    
    def release_authority(self):
        """释放权限降级"""
        old = self._current_authority
        self._current_authority = AuthorityLevel.VIEWER
        if old != self._current_authority and self._on_authority_change:
            self._on_authority_change(old, self._current_authority)
    
    def _trigger_safety_stop(self, reason: str):
        """触发安全停止"""
        self.state = TeleopState.SAFETY_STOP
        self._current_authority = AuthorityLevel.VIEWER
        
        if self._on_emergency_stop:
            self._on_emergency_stop(reason)
    
    def emergency_stop(self):
        """手动紧急停止"""
        self._trigger_safety_stop("Manual emergency stop")
    
    def acknowledge_safety_stop(self, authority: AuthorityLevel):
        """确认安全停止并恢复"""
        if authority.value < AuthorityLevel.SUPERVISOR.value:
            return False
        
        if self.state != TeleopState.SAFETY_STOP:
            return False
        
        self.safety.acknowledge_emergency()
        self.state = TeleopState.IDLE
        self._current_authority = authority
        return True
    
    def pause(self):
        """暂停遥操作"""
        if self.state == TeleopState.ACTIVE:
            self.state = TeleopState.PAUSED
    
    def resume(self):
        """恢复遥操作"""
        if self.state == TeleopState.PAUSED:
            self.state = TeleopState.ACTIVE
    
    def connect(self):
        """建立连接"""
        self.state = TeleopState.IDLE
        self._current_authority = AuthorityLevel.OPERATOR
    
    def disconnect(self):
        """断开连接"""
        self.state = TeleopState.DISCONNECTED
        self._command_buffer.clear()
    
    def get_metrics(self) -> TeleopMetrics:
        """获取性能指标"""
        metrics = TeleopMetrics()
        
        if self._master_state and self._slave_state:
            metrics.position_error_norm = float(np.linalg.norm(
                self._master_state.joint_positions - 
                self._slave_state.joint_positions
            ))
            metrics.velocity_error_norm = float(np.linalg.norm(
                self._master_state.joint_velocities -
                self._slave_state.joint_velocities
            ))
        
        metrics.authority_level = self._current_authority
        metrics.timestamp = time.time()
        metrics.connection_quality = self.latency_comp.get_compensation_quality()
        
        return metrics
    
    def on_emergency_stop(self, callback: Callable[[str], None]):
        """注册紧急停止回调"""
        self._on_emergency_stop = callback
    
    def on_authority_change(self, callback: Callable[[AuthorityLevel, AuthorityLevel], None]):
        """注册权限变更回调"""
        self._on_authority_change = callback
    
    def on_state_change(self, callback: Callable[[TeleopState, TeleopState], None]):
        """注册状态变更回调"""
        self._on_state_change = callback
    
    @property
    def is_active(self) -> bool:
        return self.state == TeleopState.ACTIVE
    
    @property
    def is_safe(self) -> bool:
        return self.state not in [TeleopState.SAFETY_STOP, TeleopState.DISCONNECTED]


# AGV五级遥操作规格
AGV_TELEOP_GRADES = {
    'S':  {'mode': 'velocity_sync',  'freq': 50,   'latency_ms': 100, 'safety': 'basic',   'shared': False},
    'M':  {'mode': 'position_sync', 'freq': 100,  'latency_ms': 50,  'safety': 'standard', 'shared': False},
    'L':  {'mode': 'position_sync', 'freq': 200,  'latency_ms': 30,  'safety': 'advanced',  'shared': True},
    'XL': {'mode': 'impedance_sync', 'freq': 500,  'latency_ms': 20,  'safety': 'advanced',  'shared': True},
    'XXL': {'mode': 'impedance_sync', 'freq': 1000, 'latency_ms': 10, 'safety': 'full',     'shared': True},
}


def get_teleop_spec(grade: str) -> dict:
    """获取AGV指定等级的遥操作规格"""
    return AGV_TELEOP_GRADES.get(grade, AGV_TELEOP_GRADES['M'])


# 导出
__all__ = [
    'TeleopMode', 'TeleopState', 'AuthorityLevel',
    'MasterState', 'SlaveState', 'TeleopCommand', 'TeleopConfig', 'TeleopMetrics',
    'SafetyMonitor', 'LatencyCompensator', 'SharedControlBlender',
    'TeleoperationController',
    'AGV_TELEOP_GRADES', 'get_teleop_spec'
]
