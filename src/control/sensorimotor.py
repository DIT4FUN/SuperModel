"""
传感器-执行器融合控制模块
=========================

统一的传感-运动控制接口:
- TactileServoController → 触觉伺服
- ForceController → 力觉控制
- AttitudeStabilizer → IMU姿态稳定
- SensorimotorIntegration → 多模态融合控制

这是具身智能的核心: 将多模态感知直接映射到控制输出，
实现"感知-决策-执行"一体化闭环。

集成架构:
  触觉传感器 ─┐
  力觉传感器 ─┼─→ SensorimotorIntegration ──→ MotorController
  IMU传感器  ─┤       │
  视觉传感器 ─┘       ▼
               CrossModalFusion ──→ WorldModel (可选)
"""

import numpy as np
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, List, Dict, Any
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import TactileArray, TactileFrame, TactileContact
from sensors.force import ForceTorqueSensor, Wrench, ContactState
from sensors.imu import IMUSensor, IMUFrame, Pose
from control.tactile_control import TactileServoController, TactileServoParams
from control.force_control import ForceController, ForceControlParams
from control.imu_control import AttitudeStabilizer, IMUControlParams


@dataclass
class SensorimotorConfig:
    """传感-运动融合配置"""
    # 感知权重 (各模态对控制的贡献)
    tactile_weight: float = 0.3
    force_weight: float = 0.4
    imu_weight: float = 0.3
    
    # 融合策略
    fusion_strategy: str = "weighted"  # "weighted" / "adaptive" / "hierarchical"
    
    # 控制频率
    control_rate: float = 100.0  # Hz
    
    # AGV等级
    grade: str = 'M'
    
    # 各模态使能
    tactile_enabled: bool = True
    force_enabled: bool = True
    imu_enabled: bool = True
    
    @classmethod
    def from_grade(cls, grade: str) -> 'SensorimotorConfig':
        configs = {
            'S':  cls(tactile_weight=0.2, force_weight=0.3, imu_weight=0.5,
                       control_rate=50, grade='S'),
            'M':  cls(tactile_weight=0.3, force_weight=0.4, imu_weight=0.3,
                       control_rate=100, grade='M'),
            'L':  cls(tactile_weight=0.35, force_weight=0.4, imu_weight=0.25,
                       control_rate=200, grade='L'),
            'XL': cls(tactile_weight=0.4, force_weight=0.35, imu_weight=0.25,
                       control_rate=500, grade='XL'),
            'XXL': cls(tactile_weight=0.4, force_weight=0.4, imu_weight=0.2,
                        control_rate=1000, grade='XXL'),
        }
        return configs.get(grade, cls())


@dataclass
class SensorimotorState:
    """传感-运动融合状态"""
    timestamp: float = 0.0
    frame_id: int = 0
    
    # 各模态数据
    tactile_contact: bool = False
    tactile_contacts: List[TactileContact] = field(default_factory=list)
    tactile_slip_prob: float = 0.0
    tactile_grip_quality: float = 0.0
    
    force_magnitude: float = 0.0
    force_in_contact: bool = False
    force_collision: bool = False
    
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    imu_tilt_warning: bool = False
    imu_stable: bool = True
    
    # 融合输出
    fused_control: np.ndarray = field(default_factory=lambda: np.zeros(3))
    control_authority: Dict[str, float] = field(default_factory=dict)  # 各模态控制权
    
    # 健康状态
    tactile_healthy: bool = True
    force_healthy: bool = True
    imu_healthy: bool = True


class SensorimotorIntegration:
    """
    传感-运动融合控制器
    
    将触觉/力觉/IMU的多模态感知统一映射到机器人运动控制输出。
    
    工作流程:
    1. 采集各模态传感器数据
    2. 各模态独立计算控制信号
    3. 自适应融合各模态控制输出
    4. 输出最终运动控制指令
    
    支持:
    - 触觉导引 (接触检测/滑移反应/抓取质量)
    - 力觉控制 (碰撞检测/导纳控制/力位混合)
    - IMU稳定 (姿态保持/倾角保护/运动估计)
    - 多模态融合 (加权/自适应/分层)
    """
    
    def __init__(
        self,
        tactile_sensor: Optional[TactileArray] = None,
        force_sensor: Optional[ForceTorqueSensor] = None,
        imu_sensor: Optional[IMUSensor] = None,
        config: Optional[SensorimotorConfig] = None,
        grade: str = 'M'
    ):
        self.tactile = tactile_sensor
        self.force = force_sensor
        self.imu = imu_sensor
        self.config = config or SensorimotorConfig.from_grade(grade)
        self.grade = grade
        
        # 子控制器
        self._tactile_ctrl: Optional[TactileServoController] = None
        self._force_ctrl: Optional[ForceController] = None
        self._imu_ctrl: Optional[AttitudeStabilizer] = None
        
        # 状态
        self._state = SensorimotorState()
        self._frame_id = 0
        
        # 初始化子控制器
        if self.tactile and self.config.tactile_enabled:
            self._init_tactile_controller()
        if self.force and self.config.force_enabled:
            self._init_force_controller()
        if self.imu and self.config.imu_enabled:
            self._init_imu_controller()
    
    def _init_tactile_controller(self):
        params = TactileServoParams.from_grade(self.grade)
        self._tactile_ctrl = TactileServoController(self.tactile, params)
    
    def _init_force_controller(self):
        params = ForceControlParams.from_grade(self.grade)
        self._force_ctrl = ForceController(self.force, params)
    
    def _init_imu_controller(self):
        params = IMUControlParams.from_grade(self.grade)
        self._imu_ctrl = AttitudeStabilizer(self.imu, params)
    
    def open(self) -> bool:
        """打开所有传感器和控制器"""
        if self.tactile:
            self.tactile.open()
        if self.force:
            self.force.open()
        if self.imu:
            self.imu.open()
        return True
    
    def close(self):
        """关闭所有传感器"""
        if self.tactile:
            self.tactile.close()
        if self.force:
            self.force.close()
        if self.imu:
            self.imu.close()
    
    def step(
        self,
        target_force: Optional[float] = None,
        target_attitude: Optional[Tuple[float, float, float]] = None,
        dt: float = 0.01
    ) -> SensorimotorState:
        """
        执行一步传感-运动融合控制
        
        Args:
            target_force: 目标接触力 (N)
            target_attitude: 目标姿态 (roll, pitch, yaw) rad
            dt: 时间步长
            
        Returns:
            SensorimotorState: 融合状态
        """
        import time
        self._frame_id += 1
        self._state.timestamp = time.time()
        self._state.frame_id = self._frame_id
        
        # ── 1. 采集各模态数据 ─────────────────────────────────
        
        tactile_frame = None
        force_wrench = None
        imu_frame = None
        
        if self.tactile and self.config.tactile_enabled:
            try:
                tactile_frame = self.tactile.capture()
                self._state.tactile_healthy = True
            except Exception:
                self._state.tactile_healthy = False
        
        if self.force and self.config.force_enabled:
            try:
                if hasattr(self, '_force_wrench_override') and self._force_wrench_override is not None:
                    force_wrench = self._force_wrench_override
                    self._force_wrench_override = None  # consume it
                else:
                    force_wrench = self.force.capture()
                self._state.force_healthy = True
            except Exception:
                self._state.force_healthy = False
        
        if self.imu and self.config.imu_enabled:
            try:
                imu_frame = self.imu.capture()
                self._state.imu_healthy = True
            except Exception:
                self._state.imu_healthy = False
        
        # ── 2. 各模态独立计算控制信号 ──────────────────────────
        
        tactile_ctrl_output = np.zeros(3)
        force_ctrl_output = np.zeros(3)
        imu_ctrl_output = np.zeros(3)
        
        authority = {
            'tactile': 0.0,
            'force': 0.0,
            'imu': 0.0
        }
        
        # --- 触觉控制 ---
        if self._tactile_ctrl and tactile_frame:
            contacts = self.tactile.detect_contacts(tactile_frame)
            self._state.tactile_contact = len(contacts) > 0
            self._state.tactile_contacts = contacts
            
            if contacts:
                tactile_target = target_force or 5.0
                tactile_ctrl_output = self._tactile_ctrl.compute_control_signal(
                    tactile_target, tactile_frame
                )
                
                # 滑移检测
                slip = self.tactile.get_slip_signal(tactile_frame)
                self._state.tactile_slip_prob = float(np.max(slip))
                
                # 抓取质量
                quality = self.tactile.estimate_grip_quality(tactile_frame)
                self._state.tactile_grip_quality = quality['overall']
                
                authority['tactile'] = self.config.tactile_weight
            else:
                authority['tactile'] = 0.0
        
        # --- 力觉控制 ---
        if self._force_ctrl and force_wrench:
            self._state.force_magnitude = force_wrench.magnitude
            
            contact_state = self.force.detect_contact(force_wrench)
            self._state.force_in_contact = contact_state.is_contact
            
            collision, _ = self._force_ctrl.detect_collision(force_wrench)
            self._state.force_collision = collision
            
            if target_force is not None:
                target_f = np.array([0.0, 0.0, target_force])
                force_ctrl_output = self._force_ctrl.compute_admittance(
                    target_f, force_wrench, dt
                )
                
                if contact_state.is_contact:
                    authority['force'] = self.config.force_weight
            else:
                authority['force'] = 0.0
        
        # --- IMU控制 ---
        if self._imu_ctrl and imu_frame:
            roll, pitch, yaw = imu_frame.gyro
            self._state.imu_roll = float(np.arctan2(roll, np.abs(np.linalg.norm([roll, pitch]))))
            self._state.imu_pitch = float(np.arctan2(pitch, np.abs(np.linalg.norm([roll, pitch]))))
            self._state.imu_yaw = 0.0
            
            tilt_status = self._imu_ctrl.get_tilt_status()
            self._state.imu_tilt_warning = tilt_status['tilt_warning']
            self._state.imu_stable = tilt_status['is_stable']
            
            if target_attitude:
                self._imu_ctrl.set_target_attitude(*target_attitude)
            
            imu_ctrl_output = self._imu_ctrl.update(imu_frame, dt)
            
            if self._state.imu_stable:
                authority['imu'] = self.config.imu_weight
        
        # ── 3. 归一化权限 ────────────────────────────────────
        total_authority = sum(authority.values())
        if total_authority > 0:
            authority = {k: v / total_authority for k, v in authority.items()}
        
        self._state.control_authority = authority.copy()
        
        # ── 4. 融合控制输出 ─────────────────────────────────
        
        if self.config.fusion_strategy == "weighted":
            fused = (
                authority['tactile'] * tactile_ctrl_output +
                authority['force'] * force_ctrl_output +
                authority['imu'] * imu_ctrl_output
            )
        elif self.config.fusion_strategy == "adaptive":
            total = sum(authority.values())
            if total > 0:
                if self._state.force_in_contact:
                    adaptive_force = min(authority['force'] * 1.5, 1.0)
                    adaptive_total = adaptive_force + authority['tactile'] + authority['imu']
                    if adaptive_total > 0:
                        fused = (
                            authority['tactile'] / adaptive_total * tactile_ctrl_output +
                            adaptive_force / adaptive_total * force_ctrl_output +
                            authority['imu'] / adaptive_total * imu_ctrl_output
                        )
                    else:
                        fused = np.zeros(3)
                elif self._state.tactile_contact:
                    adaptive_tactile = min(authority['tactile'] * 1.5, 1.0)
                    adaptive_total = adaptive_tactile + authority['force'] + authority['imu']
                    if adaptive_total > 0:
                        fused = (
                            adaptive_tactile / adaptive_total * tactile_ctrl_output +
                            authority['force'] / adaptive_total * force_ctrl_output +
                            authority['imu'] / adaptive_total * imu_ctrl_output
                        )
                    else:
                        fused = np.zeros(3)
                else:
                    auth_ai = dict(authority)
                    auth_ai['imu'] = min(auth_ai['imu'] * 2.0, 1.0)
                    adaptive_total = sum(auth_ai.values())
                    if adaptive_total > 0:
                        fused = (
                            auth_ai['tactile'] / adaptive_total * tactile_ctrl_output +
                            auth_ai['force'] / adaptive_total * force_ctrl_output +
                            auth_ai['imu'] / adaptive_total * imu_ctrl_output
                        )
                    else:
                        fused = np.zeros(3)
            else:
                fused = np.zeros(3)
        else:
            # hierarchical: 优先级触觉 > 力觉 > IMU
            if self._state.force_collision:
                fused = force_ctrl_output
                authority = {'tactile': 0.0, 'force': 1.0, 'imu': 0.0}
            elif self._state.tactile_contact and self._state.tactile_slip_prob > 0.3:
                fused = tactile_ctrl_output
                authority = {'tactile': 1.0, 'force': 0.0, 'imu': 0.0}
            elif self._state.imu_tilt_warning:
                fused = imu_ctrl_output
                authority = {'tactile': 0.0, 'force': 0.0, 'imu': 1.0}
            else:
                total = sum(authority.values())
                if total > 0:
                    fused = (
                        authority['tactile'] / total * tactile_ctrl_output +
                        authority['force'] / total * force_ctrl_output +
                        authority['imu'] / total * imu_ctrl_output
                    )
                else:
                    fused = np.zeros(3)
        
        self._state.fused_control = fused.astype(np.float32)
        
        return self._state
    
    def get_state(self) -> SensorimotorState:
        """获取当前融合状态"""
        return self._state
    
    def is_safe(self) -> bool:
        """检查整体安全状态"""
        if not self._state.force_healthy:
            return False
        if self._state.force_collision:
            return False
        if self._state.imu_tilt_warning:
            return False
        return True
    
    def get_control_authority(self) -> Dict[str, float]:
        """获取各模态控制权限"""
        return self._state.control_authority.copy()
    
    def emergency_stop(self):
        """紧急停止 - 复位所有控制器"""
        self._state.fused_control = np.zeros(3)
        self._state.control_authority = {'tactile': 0.0, 'force': 0.0, 'imu': 0.0}
        print("[SensorimotorIntegration] Emergency stop triggered")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()


class SensorimotorSimulator:
    """
    传感器-执行器融合仿真器
    
    使用真实传感器接口 (TactileArray/ForceTorqueSensor/IMUSensor) 进行仿真，
    通过capture()方法获取模拟数据。用于:
    - 算法验证
    - 参数调优
    - 仿真环境中的具身智能训练
    """
    
    def __init__(self, grade: str = 'M'):
        tactile_size_map = {
            'S': (8, 8), 'M': (16, 16), 'L': (24, 24),
            'XL': (32, 32), 'XXL': (48, 48)
        }
        arr_size = tactile_size_map.get(grade, (16, 16))
        
        self.grade = grade
        # TactileArray/ForceTorqueSensor/IMUSensor 有 capture() 方法用于仿真
        self.virtual_tactile = TactileArray(array_size=arr_size, sensor_id="virtual_tactile")
        self.virtual_force = ForceTorqueSensor(sensor_id="virtual_force")
        self.virtual_imu = IMUSensor(sensor_id="virtual_imu")
        
        # 创建融合控制器
        self.integration = SensorimotorIntegration(
            tactile_sensor=self.virtual_tactile,
            force_sensor=self.virtual_force,
            imu_sensor=self.virtual_imu,
            grade=grade
        )
        
        self._is_running = False
        self._grasp_phase = 0  # 用于抓取阶段模拟
    
    def open(self) -> bool:
        self.virtual_tactile.open()
        self.virtual_force.open()
        self.virtual_imu.open()
        self.integration.open()
        self._is_running = True
        return True
    
    def close(self):
        self._is_running = False
        self.integration.close()
        self.virtual_tactile.close()
        self.virtual_force.close()
        self.virtual_imu.close()
    
    def _apply_grasp_contact(self, phase: float, object_pos: Tuple[float, float], object_force: float):
        """在 TactileArray 中应用模拟接触"""
        if phase < 0.2:
            # 接近阶段 - 无触觉接触,清除残留位置
            self.virtual_tactile._last_contact_pos = None
        elif phase < 0.4:
            # 接触开始
            self.virtual_tactile._last_contact_pos = object_pos
        elif phase < 0.6:
            # 夹取
            self.virtual_tactile._last_contact_pos = object_pos
        elif phase < 0.8:
            # 提起 - 滑移
            self.virtual_tactile._last_contact_pos = (
                object_pos[0] + 0.05,
                object_pos[1]
            )
        else:
            # 保持
            self.virtual_tactile._last_contact_pos = object_pos
    
    def simulate_grasp(
        self,
        object_pos: Tuple[float, float] = (0.5, 0.5),
        object_force: float = 10.0,
        num_steps: int = 100,
        dt: float = 0.01
    ) -> List[SensorimotorState]:
        """
        仿真抓取任务
        
        模拟一个完整的抓取序列:
        1. 接近 → 2. 接触 → 3. 夹取 → 4. 提起 → 5. 保持
        """
        states = []
        
        for step in range(num_steps):
            phase = step / num_steps
            
            # 设置接触位置以生成有意义的触觉数据
            self._apply_grasp_contact(phase, object_pos, object_force)
            
            # IMU: 模拟静止
            _ = self.virtual_imu.capture()
            
            # 执行融合控制
            state = self.integration.step(
                target_force=object_force,
                target_attitude=(0.0, 0.0, 0.0),
                dt=dt
            )
            states.append(replace(state))
        
        return states
    
    def simulate_agv_navigation(
        self,
        trajectory_type: str = "circle",
        duration_s: float = 5.0,
        dt: float = 0.01
    ) -> List[SensorimotorState]:
        """
        仿真AGV导航任务
        
        模拟AGV沿给定轨迹运动时的多模态传感-运动融合
        """
        states = []
        n_steps = int(duration_s / dt)
        
        for i in range(n_steps):
            t = i * dt
            
            # 根据轨迹类型计算运动状态
            if trajectory_type == "circle":
                omega = 0.5
                v_linear = 0.5
            elif trajectory_type == "figure8":
                omega = 0.5
                v_linear = 0.5
            elif trajectory_type == "linear":
                v_linear = 0.5
                omega = 0.0
            elif trajectory_type == "sine":
                v_linear = 0.5
                omega = 0.1
            else:
                v_linear = 0.0
                omega = 0.0
            
            # IMU 采集 - capture() 内部已有运动模拟
            _ = self.virtual_imu.capture()
            
            # 力觉采集 - capture() 内部已有重力模拟
            _ = self.virtual_force.capture()
            
            # 触觉采集 - 背景噪声
            _ = self.virtual_tactile.capture()
            
            # 执行融合控制
            state = self.integration.step(
                target_force=None,
                target_attitude=(0.0, 0.0, 0.0),
                dt=dt
            )
            states.append(replace(state))
        
        return states
    
    def simulate_collision_recovery(
        self,
        collision_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        collision_force: float = 50.0,
        recovery_steps: int = 50,
        dt: float = 0.01
    ) -> List[SensorimotorState]:
        """
        仿真碰撞检测与恢复
        
        模拟碰撞发生后的响应和恢复过程
        """
        states = []
        
        for step in range(recovery_steps):
            # 触觉无接触
            _ = self.virtual_tactile.capture()
            
            # IMU 静止
            _ = self.virtual_imu.capture()
            
            if step == 0:
                # 碰撞瞬间: 直接注入碰撞状态，绕过 capture() 覆盖问题
                collision_wrench = Wrench(
                    force=np.array([-collision_direction[i] * collision_force
                                   for i in range(3)], dtype=np.float32),
                    torque=np.zeros(3, dtype=np.float32),
                    timestamp=0.0, frame_id=step, sensor_id="virtual_force"
                )
                # 强制设置到 integration.step 可读的位置
                self.integration._force_wrench_override = collision_wrench
                self.integration._state.force_collision = True
                self.integration._force_ctrl._collision_history.append(True)
                self.integration._force_ctrl._collision_history.append(True)
                self.integration._force_ctrl._collision_history.append(True)
            elif step < 10:
                # 恢复过程: 力逐渐减小
                recovery_ratio = step / recovery_steps
                current_force = collision_force * (1 - recovery_ratio)
                recovery_wrench = Wrench(
                    force=np.array([-collision_direction[i] * current_force
                                   for i in range(3)], dtype=np.float32),
                    torque=np.zeros(3, dtype=np.float32),
                    timestamp=0.0, frame_id=step, sensor_id="virtual_force"
                )
                self.integration._force_wrench_override = recovery_wrench
                self.integration._force_ctrl._collision_history.append(False)
            else:
                # 恢复正常
                self.integration._force_wrench_override = None
                self.integration._state.force_collision = False
                self.integration._force_ctrl._collision_history.append(False)
            
            # 融合控制
            state = self.integration.step(dt=dt)
            states.append(replace(state))
        
        return states
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()


# AGV五级传感-运动融合规格
AGV_SENSORIMOTOR_GRADES = {
    'S':  SensorimotorConfig.from_grade('S'),
    'M':  SensorimotorConfig.from_grade('M'),
    'L':  SensorimotorConfig.from_grade('L'),
    'XL': SensorimotorConfig.from_grade('XL'),
    'XXL': SensorimotorConfig.from_grade('XXL'),
}


def get_sensorimotor_spec(grade: str) -> SensorimotorConfig:
    """获取AGV指定等级的传感-运动融合配置"""
    return AGV_SENSORIMOTOR_GRADES.get(grade, AGV_SENSORIMOTOR_GRADES['M'])
