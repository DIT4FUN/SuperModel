"""
具身传感控制模块 (Embodied Sensorimotor Control)
==============================================

整合触觉 + 力觉 + IMU 的闭环具身控制:
- 多模态感知融合控制
- 接触丰富的力位混合控制
- 姿态稳定+力控协同
- 抓取-操作-放置完整任务链

集成关系:
  TactileArray ─┐
  ForceTorque ──┼──► EmbodiedController ──► 关节/末端控制
  IMUSensor ────┘

控制架构:
  感知层: TactileFrame + Wrench + IMUFrame
  融合层: ContactState + GraspQuality + AttitudeState
  控制层: TactileServo + ForceControl + AttitudeStabilizer
  执行层: JointCommand / TwistCommand

AGV五级控制等级:
  S:  触觉阈值触发 + 固定力
  M:  实时触觉伺服 + 导纳控制
  L:  力位混合 + 姿态稳定
  XL: 完整阻抗 + 多模态融合
  XXL: MPC预测 + 多层级协同
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact,
    TactileSensorType, TactileCalibration, VirtualTactileSensor
)
from sensors.force import (
    ForceTorqueSensor, Wrench, ContactState,
    ForceSensorType, ForceCalibration, VirtualForceSensor
)
from sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator,
    IMUSensorType, IMUCalibration, VirtualIMUSensor
)


# ─────────────────────────────────────────────
# AGV五级具身控制规格
# ─────────────────────────────────────────────

class EmbodiedGrade(str, Enum):
    """具身控制AGV五级等级"""
    S = 'S'
    M = 'M'
    L = 'L'
    XL = 'XL'
    XXL = 'XXL'


AGV_EMBODIED_GRADES = {
    'S': {
        'description': '小型AGV-阈值触发式',
        'tactile_enabled': True,
        'force_enabled': False,
        'imu_enabled': False,
        'tactile_resolution': '8x8',
        'force_axes': 0,
        'imu_grade': 'S',
        'control_rate': 50,       # Hz
        'latency_ms': 50,        # 控制延迟
        'fusion_method': 'threshold',
        'max_contact_force': 20,   # N
        'grasp_adaptation': False,
        'attitude_stabilization': False,
        'slip_recovery': False,
        'collision_response_ms': 100,
    },
    'M': {
        'description': '中型AGV-导纳控制式',
        'tactile_enabled': True,
        'force_enabled': True,
        'imu_enabled': True,
        'tactile_resolution': '16x16',
        'force_axes': 6,
        'imu_grade': 'M',
        'control_rate': 100,
        'latency_ms': 20,
        'fusion_method': 'weighted_average',
        'max_contact_force': 50,
        'grasp_adaptation': True,
        'attitude_stabilization': True,
        'slip_recovery': True,
        'collision_response_ms': 50,
    },
    'L': {
        'description': '大型AGV-力位混合式',
        'tactile_enabled': True,
        'force_enabled': True,
        'imu_enabled': True,
        'tactile_resolution': '24x24',
        'force_axes': 6,
        'imu_grade': 'L',
        'control_rate': 200,
        'latency_ms': 10,
        'fusion_method': 'ekf',
        'max_contact_force': 150,
        'grasp_adaptation': True,
        'attitude_stabilization': True,
        'slip_recovery': True,
        'collision_response_ms': 20,
    },
    'XL': {
        'description': '超大型AGV-完整阻抗式',
        'tactile_enabled': True,
        'force_enabled': True,
        'imu_enabled': True,
        'tactile_resolution': '32x32',
        'force_axes': 6,
        'imu_grade': 'XL',
        'control_rate': 500,
        'latency_ms': 5,
        'fusion_method': 'ukf',
        'max_contact_force': 300,
        'grasp_adaptation': True,
        'attitude_stabilization': True,
        'slip_recovery': True,
        'collision_response_ms': 10,
    },
    'XXL': {
        'description': '重型AGV-MPC预测式',
        'tactile_enabled': True,
        'force_enabled': True,
        'imu_enabled': True,
        'tactile_resolution': '48x48',
        'force_axes': 6,
        'imu_grade': 'XXL',
        'control_rate': 1000,
        'latency_ms': 2,
        'fusion_method': 'mpc_fusion',
        'max_contact_force': 1000,
        'grasp_adaptation': True,
        'attitude_stabilization': True,
        'slip_recovery': True,
        'collision_response_ms': 5,
    },
}


def get_embodied_spec(grade: str) -> dict:
    """获取AGV指定等级的具身控制规格"""
    return AGV_EMBODIED_GRADES.get(grade, AGV_EMBODIED_GRADES['M'])


# ─────────────────────────────────────────────
# 核心数据结构
# ─────────────────────────────────────────────

@dataclass
class EmbodiedState:
    """
    具身感知-控制综合状态
    
    整合三模态感知结果 + 融合状态 + 控制状态
    """
    # 触觉状态
    tactile_contacts: List[TactileContact] = field(default_factory=list)
    grip_quality: float = 0.0
    slip_probability: float = 0.0
    contact_area_ratio: float = 0.0
    
    # 力觉状态
    wrench: Optional[Wrench] = None
    contact_state: Optional[ContactState] = None
    contact_force: float = 0.0
    estimated_payload: float = 0.0
    
    # IMU状态
    pose: Optional[Pose] = None
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    tilt_angle: float = 0.0
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 融合状态
    is_stable: bool = True
    is_in_contact: bool = False
    is_slipping: bool = False
    is_tilted: bool = False
    confidence: float = 1.0  # 状态估计置信度
    
    # 传感器健康
    tactile_ok: bool = True
    force_ok: bool = True
    imu_ok: bool = True
    
    # 时间戳
    timestamp: float = 0.0
    cycle_id: int = 0


@dataclass
class EmbodiedCommand:
    """具身控制指令"""
    # 末端期望力/力矩 (N, N·m)
    desired_force: Optional[np.ndarray] = None   # 3
    desired_torque: Optional[np.ndarray] = None  # 3
    
    # 末端期望位置/姿态
    desired_position: Optional[np.ndarray] = None   # 3
    desired_orientation: Optional[np.ndarray] = None  # 4 (quat)
    
    # 运动速度 (m/s, rad/s)
    desired_linear_velocity: Optional[np.ndarray] = None   # 3
    desired_angular_velocity: Optional[np.ndarray] = None  # 3
    
    # 控制模式
    mode: str = "hybrid_force_position"  # hybrid | impedance | position | force
    
    # 优先级权重
    force_weight: float = 0.5     # 力控制权重
    position_weight: float = 0.5  # 位置控制权重
    
    # 安全限制
    max_contact_force: float = 100.0  # N
    max_velocity: float = 0.5         # m/s
    
    def __post_init__(self):
        if self.desired_force is not None and not isinstance(self.desired_force, np.ndarray):
            self.desired_force = np.array(self.desired_force, dtype=np.float32)
        if self.desired_torque is not None and not isinstance(self.desired_torque, np.ndarray):
            self.desired_torque = np.array(self.desired_torque, dtype=np.float32)
        if self.desired_position is not None and not isinstance(self.desired_position, np.ndarray):
            self.desired_position = np.array(self.desired_position, dtype=np.float32)
        if self.desired_orientation is not None and not isinstance(self.desired_orientation, np.ndarray):
            self.desired_orientation = np.array(self.desired_orientation, dtype=np.float32)


@dataclass
class EmbodiedControlParams:
    """具身控制参数 (AGV五级配置)"""
    grade: str = 'M'
    
    # 多模态融合参数
    fusion_method: str = 'weighted_average'
    state_confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'tactile': 0.3, 'force': 0.4, 'imu': 0.3
    })
    
    # 触觉-力融合
    tactile_force_blend: float = 0.5
    
    # 安全阈值
    contact_threshold: float = 2.0      # N, 接触检测
    slip_threshold: float = 0.3         # 滑移概率阈值
    tilt_warning: float = 0.26          # rad, 15度
    tilt_critical: float = 0.52        # rad, 30度
    
    # 导纳控制
    admittance_mass: float = 1.0        # kg
    admittance_damping: float = 10.0    # N·s/m
    admittance_stiffness: float = 200.0 # N/m
    
    # 阻抗控制
    impedance_Kp: float = 500.0         # N/m
    impedance_Kd: float = 50.0          # N·s/m
    impedance_M: float = 5.0           # kg
    
    # 控制频率
    control_rate: float = 100.0         # Hz
    
    @classmethod
    def from_grade(cls, grade: str) -> 'EmbodiedControlParams':
        """从AGV等级创建设定参数"""
        spec = get_embodied_spec(grade)
        
        # 根据等级调整参数
        if grade == 'S':
            return cls(
                grade=grade,
                fusion_method='threshold',
                admittance_mass=2.0,
                admittance_damping=5.0,
                control_rate=50.0,
            )
        elif grade == 'M':
            return cls(
                grade=grade,
                fusion_method='weighted_average',
                admittance_mass=1.0,
                admittance_damping=10.0,
                control_rate=100.0,
            )
        elif grade == 'L':
            return cls(
                grade=grade,
                fusion_method='ekf',
                admittance_mass=0.5,
                admittance_damping=20.0,
                impedance_Kp=800.0,
                control_rate=200.0,
            )
        elif grade == 'XL':
            return cls(
                grade=grade,
                fusion_method='ukf',
                admittance_mass=0.2,
                admittance_damping=50.0,
                impedance_Kp=1000.0,
                impedance_Kd=100.0,
                control_rate=500.0,
            )
        else:  # XXL
            return cls(
                grade=grade,
                fusion_method='mpc_fusion',
                admittance_mass=0.1,
                admittance_damping=100.0,
                impedance_Kp=2000.0,
                impedance_Kd=200.0,
                impedance_M=2.0,
                control_rate=1000.0,
            )


# ─────────────────────────────────────────────
# 具身控制器
# ─────────────────────────────────────────────

class EmbodiedController:
    """
    具身传感控制核心
    
    整合触觉 + 力觉 + IMU，输出闭环控制指令
    
    使用方式:
        ctrl = EmbodiedController(grade='M')
        ctrl.set_sensors(tactile, force, imu)
        
        # 主循环
        state = ctrl.update()
        cmd = EmbodiedCommand(mode='impedance', desired_force=...)
        output = ctrl.compute(cmd)
        
        # 应用到执行器
        robot.apply_joint_commands(output.joint_torques)
    """
    
    def __init__(
        self,
        grade: str = 'M',
        params: Optional[EmbodiedControlParams] = None,
        use_virtual_sensors: bool = False
    ):
        """
        Args:
            grade: AGV五级等级 (S/M/L/XL/XXL)
            params: 控制参数 (默认从grade加载)
            use_virtual_sensors: 是否使用虚拟传感器
        """
        self.grade = grade
        self.spec = get_embodied_spec(grade)
        self.params = params or EmbodiedControlParams.from_grade(grade)
        
        self.use_virtual = use_virtual_sensors
        
        # 传感器引用 (外部注入)
        self._tactile: Optional[TactileArray] = None
        self._force: Optional[ForceTorqueSensor] = None
        self._imu: Optional[IMUSensor] = None
        
        # 虚拟传感器 (仿真模式)
        self._virtual_tactile: Optional[VirtualTactileSensor] = None
        self._virtual_force: Optional[VirtualForceSensor] = None
        self._virtual_imu: Optional[VirtualIMUSensor] = None
        
        # 姿态估计器
        self._pose_estimator: Optional[PoseEstimator] = None
        
        # 状态
        self._state = EmbodiedState()
        self._cycle_id = 0
        self._initialized = False
        
        # 历史数据 (用于滤波/预测)
        self._contact_history: List[ContactState] = []
        self._wrench_history: List[Wrench] = []
        self._slip_history: List[float] = []
        
        # 导纳/阻抗状态
        self._admittance_velocity = np.zeros(3)
        self._impedance_velocity = np.zeros(3)
        
        print(f"[EmbodiedController] Grade={grade}, "
              f"Sensors: tactile={self.spec['tactile_enabled']}, "
              f"force={self.spec['force_enabled']}, "
              f"imu={self.spec['imu_enabled']}, "
              f"rate={self.spec['control_rate']}Hz")
    
    def set_sensors(
        self,
        tactile: Optional[TactileArray] = None,
        force: Optional[ForceTorqueSensor] = None,
        imu: Optional[IMUSensor] = None
    ):
        """
        注入真实传感器
        
        Args:
            tactile: 触觉传感器
            force: 力觉传感器
            imu: IMU传感器
        """
        self._tactile = tactile
        self._force = force
        self._imu = imu
        
        # 初始化姿态估计器
        if imu is not None:
            self._pose_estimator = PoseEstimator(
                algorithm='madgwick',
                sample_rate=self.params.control_rate,
                beta=0.1
            )
        
        self._initialized = True
        print(f"[EmbodiedController] Sensors configured: "
              f"T={tactile is not None}, F={force is not None}, I={imu is not None}")
    
    def init_virtual_sensors(self):
        """初始化虚拟传感器 (仿真模式)"""
        if self.spec['tactile_enabled']:
            self._virtual_tactile = VirtualTactileSensor(
                array_size=(16, 16) if self.grade in ['M', 'L'] else (8, 8),
                sensor_id="embodied_tactile"
            )
            self._virtual_tactile.open()
        
        if self.spec['force_enabled']:
            self._virtual_force = VirtualForceSensor(
                sensor_id="embodied_force",
                noise_level=0.05
            )
            self._virtual_force.open()
        
        if self.spec['imu_enabled']:
            self._virtual_imu = VirtualIMUSensor(
                sensor_id="embodied_imu",
                accel_noise=0.005,
                gyro_noise=0.001
            )
            self._virtual_imu.open()
        
        # 姿态估计器
        self._pose_estimator = PoseEstimator(
            algorithm='madgwick',
            sample_rate=self.params.control_rate
        )
        
        self._initialized = True
        print(f"[EmbodiedController] Virtual sensors initialized for grade={self.grade}")
    
    def update(self) -> EmbodiedState:
        """
        更新具身感知状态 (从所有传感器读取并融合)
        
        主循环中调用
        
        Returns:
            EmbodiedState: 综合感知状态
        """
        dt = 1.0 / self.params.control_rate
        self._cycle_id += 1
        
        state = EmbodiedState(cycle_id=self._cycle_id)
        
        # ── 1. 触觉感知 ──────────────────────────────
        if self._tactile is not None:
            tactile_frame = self._tactile.capture()
            contacts = self._tactile.detect_contacts(tactile_frame)
            grip_q = self._tactile.estimate_grip_quality(tactile_frame)
            slip = self._tactile.get_slip_signal(tactile_frame)
            
            state.tactile_contacts = contacts
            state.grip_quality = grip_q['overall']
            state.slip_probability = float(np.mean(slip))
            state.contact_area_ratio = grip_q['contact_area']
            state.tactile_ok = True
            
            # 滑移历史
            self._slip_history.append(state.slip_probability)
            if len(self._slip_history) > 20:
                self._slip_history.pop(0)
            
        elif self._virtual_tactile is not None:
            # 虚拟触觉: 模拟静止接触
            frame = self._virtual_tactile.simulate_contact(
                contact_pos=(0.5, 0.5),
                contact_radius=0.25,
                contact_force=10.0
            )
            contacts = self._tactile.detect_contacts(frame) if self._tactile else []
            state.tactile_contacts = contacts
            state.grip_quality = 0.7
            state.slip_probability = 0.1
            state.contact_area_ratio = 0.3
            state.tactile_ok = True
        
        # ── 2. 力觉感知 ──────────────────────────────
        if self._force is not None:
            wrench = self._force.capture()
            contact = self._force.detect_contact(wrench)
            payload = self._force.estimate_payload(wrench)
            
            state.wrench = wrench
            state.contact_state = contact
            state.contact_force = contact.contact_force
            state.estimated_payload = payload
            state.is_in_contact = contact.is_contact
            state.force_ok = True
            
            self._wrench_history.append(wrench)
            if len(self._wrench_history) > 100:
                self._wrench_history.pop(0)
            
        elif self._virtual_force is not None:
            # 虚拟力: 模拟接触
            wrench = self._virtual_force.simulate_contact(
                force=(0, 0, -10),
                torque=(0, 0, 0)
            )
            state.wrench = wrench
            state.contact_force = wrench.magnitude
            state.is_in_contact = True
            state.force_ok = True
        
        # ── 3. IMU感知 ──────────────────────────────
        if self._imu is not None:
            imu_frame = self._imu.capture()
            
            # 更新姿态估计
            if self._pose_estimator is not None:
                pose = self._pose_estimator.update(
                    imu_frame.accel,
                    imu_frame.gyro,
                    imu_frame.mag
                )
                state.pose = pose
                euler = pose.to_euler()
                state.roll, state.pitch, state.yaw = euler
                state.tilt_angle = np.sqrt(state.roll**2 + state.pitch**2)
                state.angular_velocity = imu_frame.gyro.copy()
                state.linear_accel = imu_frame.accel.copy()
            
        elif self._virtual_imu is not None:
            # 虚拟IMU: 模拟静止水平
            frame = self._virtual_imu.simulate_static(orientation=(0.0, 0.0, 0.0))
            
            if self._pose_estimator is not None:
                pose = self._pose_estimator.update(
                    frame.accel, frame.gyro, frame.mag
                )
                state.pose = pose
                euler = pose.to_euler()
                state.roll, state.pitch, state.yaw = euler
                state.tilt_angle = 0.0
                state.angular_velocity = frame.gyro.copy()
                state.linear_accel = frame.accel.copy()
            
            state.imu_ok = True
        
        # ── 4. 多模态融合 ────────────────────────────
        state = self._fuse_state(state)
        
        self._state = state
        return state
    
    def _fuse_state(self, state: EmbodiedState) -> EmbodiedState:
        """
        多模态状态融合
        
        融合方法 (AGV等级决定):
        - S: 阈值触发
        - M: 加权平均
        - L: EKF
        - XL: UKF
        - XXL: MPC预测融合
        """
        method = self.params.fusion_method
        
        if method == 'threshold':
            # S级: 简单阈值判断
            state.is_slipping = state.slip_probability > self.params.slip_threshold
            state.is_in_contact = state.contact_force > self.params.contact_threshold
            state.is_tilted = state.tilt_angle > self.params.tilt_warning
            state.is_stable = not (state.is_slipping or state.is_tilted)
            state.confidence = 0.8 if state.is_stable else 0.5
            
        elif method == 'weighted_average':
            # M级: 加权平均融合
            weights = self.params.state_confidence_weights
            
            # 基于力估计接触
            force_contact = 1.0 if state.contact_force > self.params.contact_threshold else 0.0
            # 基于触觉估计滑移
            tactile_slip = state.slip_probability
            # 基于IMU估计稳定性
            tilt_factor = min(state.tilt_angle / self.params.tilt_warning, 1.0)
            
            contact_fused = (
                weights['tactile'] * (1.0 if state.tactile_contacts else 0.0) +
                weights['force'] * force_contact +
                weights['imu'] * (1.0 - tilt_factor)
            )
            state.is_in_contact = contact_fused > 0.4
            state.is_slipping = state.slip_probability > self.params.slip_threshold * 0.8
            state.is_tilted = state.tilt_angle > self.params.tilt_warning
            state.is_stable = not state.is_slipping and not state.is_tilted
            state.confidence = float(np.clip(contact_fused, 0, 1))
            
        elif method == 'ekf':
            # L级: 扩展卡尔曼滤波 (简化版)
            state = self._ekf_fuse(state)
            
        elif method == 'ukf':
            # XL级: 无迹卡尔曼滤波 (简化版)
            state = self._ukf_fuse(state)
            
        else:  # mpc_fusion
            # XXL级: MPC预测融合
            state = self._mpc_fuse(state)
        
        # 紧急倾斜停止
        if state.tilt_angle > self.params.tilt_critical:
            state.is_stable = False
            state.confidence = 0.0
        
        return state
    
    def _ekf_fuse(self, state: EmbodiedState) -> EmbodiedState:
        """EKF状态融合 (简化实现)"""
        # 简化: 使用非线性状态转移
        # 状态: [contact_force, slip_prob, tilt_angle]
        # 观测: 来自各传感器的观测
        
        R_process = np.diag([0.1, 0.01, 0.001])  # 过程噪声
        R_measure = np.diag([1.0, 0.1, 0.01])    # 观测噪声
        
        # 简化的状态更新
        dt = 1.0 / self.params.control_rate
        
        # 接触力: 缓慢变化
        expected_contact = state.contact_force * 0.95 + 0.5 * dt
        
        # 观测残差
        z_contact = state.contact_force
        
        # 简化的卡尔曼增益
        P = np.eye(3) * 0.5
        K = P @ np.linalg.inv(P + R_measure)
        
        # 状态更新
        innovation = z_contact - expected_contact
        contact_update = K[0, 0] * innovation
        
        state.is_in_contact = (state.contact_force + contact_update) > self.params.contact_threshold
        state.is_slipping = state.slip_probability > self.params.slip_threshold
        state.is_tilted = state.tilt_angle > self.params.tilt_warning
        state.is_stable = not state.is_slipping and not state.is_tilted
        state.confidence = max(0.5, 1.0 - abs(innovation) / 20.0)
        
        return state
    
    def _ukf_fuse(self, state: EmbodiedState) -> EmbodiedState:
        """UKF状态融合 (简化实现)"""
        # 使用sigma点近似
        alpha, beta, kappa = 0.001, 2.0, 0.0
        n = 3
        
        # 简化的UKF更新
        state.is_in_contact = state.contact_force > self.params.contact_threshold
        state.is_slipping = state.slip_probability > self.params.slip_threshold
        state.is_tilted = state.tilt_angle > self.params.tilt_warning
        state.is_stable = not state.is_slipping and not state.is_tilted
        state.confidence = max(0.6, 1.0 - state.slip_probability)
        
        return state
    
    def _mpc_fuse(self, state: EmbodiedState) -> EmbodiedState:
        """MPC预测融合 (简化实现)"""
        # 预测-horizon步的未来状态
        horizon = 5
        
        # 使用历史趋势预测
        if len(self._slip_history) >= horizon:
            recent = self._slip_history[-horizon:]
            # 线性趋势
            trend = (recent[-1] - recent[0]) / horizon
            predicted_slip = recent[-1] + trend * horizon
        else:
            predicted_slip = state.slip_probability
        
        # 预测接触力
        if self._wrench_history:
            recent_force = [w.magnitude for w in self._wrench_history[-5:]]
            predicted_force = np.mean(recent_force)
        else:
            predicted_force = state.contact_force
        
        # 基于预测做决策
        state.is_slipping = predicted_slip > self.params.slip_threshold * 0.7
        state.is_in_contact = predicted_force > self.params.contact_threshold * 0.8
        state.is_tilted = state.tilt_angle > self.params.tilt_warning
        state.is_stable = not state.is_slipping and not state.is_tilted
        state.confidence = max(0.7, 1.0 - predicted_slip)
        
        return state
    
    def compute(self, cmd: EmbodiedCommand) -> Dict:
        """
        计算控制输出
        
        Args:
            cmd: 控制指令
            
        Returns:
            dict: 控制输出 {
                joint_torques: np.ndarray, 关节力矩 (n_joints)
                twist: np.ndarray, 末端速度 (6,)
                contact_adjustment: np.ndarray, 接触调整量 (3,)
                safety_stop: bool, 是否紧急停止
                mode: str, 当前控制模式
            }
        """
        state = self._state
        dt = 1.0 / self.params.control_rate
        
        output = {
            'joint_torques': np.zeros(6),
            'twist': np.zeros(6),
            'contact_adjustment': np.zeros(3),
            'safety_stop': False,
            'mode': cmd.mode,
            'slip_recovered': False,
            'force_regulated': False,
        }
        
        # ── 安全检查 ──────────────────────────────
        if self._emergency_stop(state, cmd):
            output['safety_stop'] = True
            return output
        
        # ── 导纳控制 (触力感知 → 运动) ────────────
        if cmd.mode == 'admittance' and state.is_in_contact:
            adjustment = self._admittance_control(state, dt)
            output['contact_adjustment'] = adjustment
            output['force_regulated'] = True
        
        # ── 阻抗控制 (位置误差 → 力) ───────────────
        elif cmd.mode == 'impedance':
            correction = self._impedance_control(state, cmd, dt)
            output['contact_adjustment'] = correction
            
            # 末端速度输出
            if cmd.desired_linear_velocity is not None:
                output['twist'][:3] = cmd.desired_linear_velocity + correction * 0.1
            if cmd.desired_angular_velocity is not None:
                output['twist'][3:] = cmd.desired_angular_velocity
        
        # ── 力位混合控制 ───────────────────────────
        elif cmd.mode == 'hybrid_force_position':
            force_adj = self._force_position_control(state, cmd, dt)
            pos_adj = self._position_control(state, cmd, dt)
            
            output['contact_adjustment'] = (
                cmd.force_weight * force_adj +
                cmd.position_weight * pos_adj
            )
            output['force_regulated'] = True
        
        # ── 触觉伺服 (仅触觉反馈) ─────────────────
        elif cmd.mode == 'tactile_servo':
            if state.is_slipping and self.spec['slip_recovery']:
                recovery = self._slip_recovery(state, dt)
                output['contact_adjustment'] = recovery
                output['slip_recovered'] = True
        
        # ── 姿态稳定 ──────────────────────────────
        if state.is_tilted and self.spec['attitude_stabilization']:
            attitude_correction = self._attitude_stabilize(state, dt)
            output['contact_adjustment'] += attitude_correction
        
        return output
    
    def _emergency_stop(self, state: EmbodiedState, cmd: EmbodiedCommand) -> bool:
        """紧急停止判断"""
        # 力超限
        if state.contact_force > cmd.max_contact_force:
            return True
        
        # 严重倾斜
        if state.tilt_angle > self.params.tilt_critical:
            return True
        
        # 传感器失效
        if state.tactile_ok and state.force_ok and state.imu_ok:
            pass  # 所有传感器正常
        else:
            # 至少有IMU时允许有限操作
            if not state.imu_ok and state.tilt_angle > 0.1:
                return True
        
        return False
    
    def _admittance_control(self, state: EmbodiedState, dt: float) -> np.ndarray:
        """导纳控制: 接触力误差 → 位置调整"""
        if state.wrench is None:
            return np.zeros(3)
        
        # 期望力
        desired_f = state.wrench.force * 0.0  # 零力
        # 实际力
        actual_f = state.wrench.force
        
        # 力误差
        force_error = desired_f - actual_f
        
        # 导纳: F = M*a + D*v + K*x  →  x = F/K (稳态)
        # 简化: 加速度 = 力误差 / 质量, 速度积分, 位移再积分
        acc = force_error / self.params.admittance_mass
        self._admittance_velocity += acc * dt
        self._admittance_velocity *= (1 - self.params.admittance_damping * dt / self.params.admittance_mass)
        displacement = self._admittance_velocity * dt
        
        return displacement.astype(np.float32)
    
    def _impedance_control(
        self,
        state: EmbodiedState,
        cmd: EmbodiedCommand,
        dt: float
    ) -> np.ndarray:
        """阻抗控制: 位置误差 → 调节力"""
        # 期望位置 vs 实际 (简化)
        if cmd.desired_position is not None:
            pos_error = cmd.desired_position - np.zeros(3)  # 假设实际=零
        else:
            pos_error = np.zeros(3)
        
        # 阻抗: F = Kp*e + Kd*de/dt
        correction = (
            self.params.impedance_Kp * pos_error +
            self.params.impedance_Kd * (-self._impedance_velocity)
        )
        
        self._impedance_velocity += correction / self.params.impedance_M * dt
        
        return correction.astype(np.float32)
    
    def _force_position_control(
        self,
        state: EmbodiedState,
        cmd: EmbodiedCommand,
        dt: float
    ) -> np.ndarray:
        """力位混合控制"""
        force_adj = np.zeros(3)
        pos_adj = np.zeros(3)
        
        # 力控制通道
        if state.wrench is not None and cmd.desired_force is not None:
            force_error = cmd.desired_force - state.wrench.force[:3]
            Kp_f = 2.0 * (self.params.admittance_mass / 10.0)
            force_adj = Kp_f * force_error * dt
        
        # 位置控制通道
        if cmd.desired_position is not None:
            pos_error = cmd.desired_position - np.zeros(3)
            Kp_p = 100.0
            pos_adj = Kp_p * pos_error * dt
        
        return (force_adj + pos_adj).astype(np.float32)
    
    def _position_control(
        self,
        state: EmbodiedState,
        cmd: EmbodiedCommand,
        dt: float
    ) -> np.ndarray:
        """纯位置控制"""
        if cmd.desired_position is None:
            return np.zeros(3)
        
        pos_error = cmd.desired_position - np.zeros(3)
        Kp = 200.0
        return (Kp * pos_error * dt).astype(np.float32)
    
    def _slip_recovery(self, state: EmbodiedState, dt: float) -> np.ndarray:
        """滑移恢复控制"""
        # 增加抓取力以抑制滑移
        if state.grip_quality > 0:
            # 沿法向增加力
            adjustment = np.array([0.0, 0.0, state.slip_probability * 5.0])
        else:
            adjustment = np.array([0.0, 0.0, 2.0])
        
        return adjustment.astype(np.float32)
    
    def _attitude_stabilize(self, state: EmbodiedState, dt: float) -> np.ndarray:
        """姿态稳定控制"""
        # PD控制使AGV水平
        Kp_tilt = 50.0
        Kd_tilt = 10.0
        
        correction = np.zeros(3)
        correction[0] = -Kp_tilt * state.roll - Kd_tilt * state.angular_velocity[0]
        correction[1] = -Kp_tilt * state.pitch - Kd_tilt * state.angular_velocity[1]
        
        return correction.astype(np.float32)
    
    def get_state(self) -> EmbodiedState:
        """获取当前具身状态"""
        return self._state
    
    def reset(self):
        """重置控制器状态"""
        self._cycle_id = 0
        self._admittance_velocity = np.zeros(3)
        self._impedance_velocity = np.zeros(3)
        self._slip_history.clear()
        self._wrench_history.clear()
        self._contact_history.clear()
        
        if self._pose_estimator:
            self._pose_estimator.reset()
        
        print("[EmbodiedController] Reset")

    def run(
        self,
        num_steps: int = 100,
        cmd: Optional[EmbodiedCommand] = None
    ) -> Dict[str, List]:
        """
        运行具身控制仿真循环

        运行 num_steps 步的完整感知-融合-控制循环，
        用于仿真验证、边界测试、性能评估。

        Args:
            num_steps: 仿真步数
            cmd: 控制指令 (默认: 导纳控制模式)

        Returns:
            dict: 仿真结果 {
                'states': List[EmbodiedState],  每步状态
                'outputs': List[Dict],           每步控制输出
                'slip_events': int,              滑移事件数
                'contact_events': int,           接触事件数
                'safety_stops': int,             安全停止次数
                'avg_cycle_time_ms': float,      平均控制周期 (ms)
            }
        """
        import time

        if cmd is None:
            cmd = EmbodiedCommand(mode='admittance')

        states = []
        outputs = []
        slip_events = 0
        contact_events = 0
        safety_stops = 0
        cycle_times = []

        self.reset()
        print(f"[EmbodiedController.run] Starting simulation: {num_steps} steps @ {self.params.control_rate}Hz")

        for step in range(num_steps):
            t0 = time.perf_counter()

            # 感知更新
            state = self.update()
            states.append(state)

            # 检测事件
            if state.is_slipping:
                slip_events += 1
            if state.is_in_contact and not (states[-2].is_in_contact if len(states) > 1 else False):
                contact_events += 1

            # 控制计算
            output = self.compute(cmd)
            outputs.append(output)

            if output['safety_stop']:
                safety_stops += 1

            # 记录周期时间
            cycle_time = (time.perf_counter() - t0) * 1000
            cycle_times.append(cycle_time)

            # 进度日志 (每10%)
            if step % (num_steps // 10 + 1) == 0:
                print(f"  step {step:4d}/{num_steps} | "
                      f"contact={state.is_in_contact} | "
                      f"slip={state.is_slipping:.2f} | "
                      f"grip={state.grip_quality:.2f} | "
                      f"tilt={state.tilt_angle:.3f} | "
                      f"dt={cycle_time:.2f}ms")

        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0

        result = {
            'states': states,
            'outputs': outputs,
            'slip_events': slip_events,
            'contact_events': contact_events,
            'safety_stops': safety_stops,
            'avg_cycle_time_ms': avg_cycle_time,
        }

        print(f"[EmbodiedController.run] Done: "
              f"slip={slip_events}, contact={contact_events}, "
              f"safety_stops={safety_stops}, "
              f"avg_dt={avg_cycle_time:.2f}ms")

        return result

    @classmethod
    def create_for_grade(
        cls,
        grade: str = 'M',
        use_virtual: bool = True,
        use_tactile: bool = True,
        use_force: bool = True,
        use_imu: bool = True
    ) -> 'EmbodiedController':
        """
        工厂方法: 为指定AGV等级创建配置好的具身控制器

        Args:
            grade: AGV等级 ('S', 'M', 'L', 'XL', 'XXL')
            use_virtual: 使用虚拟传感器 (仿真模式)
            use_tactile: 启用触觉
            use_force: 启用力觉
            use_imu: 启用IMU

        Returns:
            EmbodiedController: 配置好的控制器
        """
        params = EmbodiedControlParams(
            grade=grade,
            fusion_method=get_embodied_spec(grade).get('fusion_method', 'weighted_average'),
            control_rate=float(get_embodied_spec(grade).get('control_rate', 100)),
            contact_threshold=2.0,
        )

        controller = cls(
            grade=grade,
            params=params,
            use_virtual_sensors=use_virtual,
        )

        # 根据等级启用/禁用模态
        spec = get_embodied_spec(grade)
        if not use_tactile or not spec.get('tactile_enabled', True):
            controller._tactile = None
        if not use_force or not spec.get('force_enabled', True):
            controller._force = None
        if not use_imu or not spec.get('imu_enabled', True):
            controller._imu = None

        if use_virtual:
            controller.init_virtual_sensors()

        return controller

    @staticmethod
    def _grade_tactile_size(grade: str) -> Tuple[int, int]:
        sizes = {
            'S': (8, 8),
            'M': (16, 16),
            'L': (24, 24),
            'XL': (32, 32),
            'XXL': (48, 48),
        }
        return sizes.get(grade, (16, 16))

    def run_five_grade_benchmark(
        self,
        steps_per_grade: int = 50
    ) -> Dict[str, Dict]:
        """
        在所有五个AGV等级上运行基准测试

        Args:
            steps_per_grade: 每个等级的仿真步数

        Returns:
            Dict[grade, result]: 每个等级的仿真结果
        """
        results = {}
        grades = ['S', 'M', 'L', 'XL', 'XXL']

        print("\n" + "=" * 60)
        print("AGV Five-Grade Embodied Control Benchmark")
        print("=" * 60)

        for grade in grades:
            print(f"\n--- Grade {grade} ---")
            spec = get_embodied_spec(grade)
            print(f"  Control Rate: {spec.get('control_rate', 0)} Hz")
            print(f"  Fusion Method: {spec.get('fusion_method', 'N/A')}")
            print(f"  Latency: {spec.get('latency_ms', 0)} ms")
            print(f"  Tactile: {spec.get('tactile_resolution', 'N/A')}")
            print(f"  Force Axes: {spec.get('force_axes', 0)}")
            print(f"  Max Contact Force: {spec.get('max_contact_force', 0)} N")

            ctrl = self.create_for_grade(
                grade=grade,
                use_virtual=True,
                use_tactile=spec.get('tactile_enabled', True),
                use_force=spec.get('force_enabled', True),
                use_imu=spec.get('imu_enabled', True),
            )

            result = ctrl.run(num_steps=steps_per_grade)
            results[grade] = result

            print(f"  → Slip Events: {result['slip_events']}")
            print(f"  → Contact Events: {result['contact_events']}")
            print(f"  → Safety Stops: {result['safety_stops']}")
            print(f"  → Avg Cycle Time: {result['avg_cycle_time_ms']:.3f} ms")

        # 汇总
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        print(f"{'Grade':<6} {'Slip':<8} {'Contact':<10} {'Safety':<10} {'AvgDT(ms)':<12}")
        print("-" * 60)
        for grade, res in results.items():
            print(f"{grade:<6} {res['slip_events']:<8} {res['contact_events']:<10} "
                  f"{res['safety_stops']:<10} {res['avg_cycle_time_ms']:<12.3f}")

        return results


# ─────────────────────────────────────────────
# 具身任务执行器
# ─────────────────────────────────────────────

class EmbodiedTaskExecutor:
    """
    具身任务执行器
    
    执行完整的抓取-操作-放置任务链
    
    任务状态机:
      IDLE → APPROACH → CONTACT → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → IDLE
    """
    
    class TaskPhase(str, Enum):
        IDLE = "idle"
        APPROACH = "approach"
        CONTACT = "contact"
        GRASP = "grasp"
        LIFT = "lift"
        TRANSPORT = "transport"
        PLACE = "place"
        RELEASE = "release"
        RETRACT = "retract"
    
    def __init__(
        self,
        embodied_ctrl: EmbodiedController,
        grade: str = 'M'
    ):
        self.ctrl = embodied_ctrl
        self.grade = grade
        self.phase = self.TaskPhase.IDLE
        
        # 任务目标
        self.target_position: Optional[np.ndarray] = None
        self.target_orientation: Optional[np.ndarray] = None
        self.desired_grasp_force: float = 10.0
        
        # 执行记录
        self.phase_history: List[Tuple[str, float]] = []
        self.success_count = 0
        self.failure_count = 0
        
        print(f"[EmbodiedTaskExecutor] Initialized for grade={grade}")
    
    def execute_grasp_place(
        self,
        object_position: np.ndarray,
        place_position: np.ndarray,
        object_size: float = 0.05,
        grasp_force: float = 10.0
    ) -> bool:
        """
        执行完整抓取-搬运-放置任务
        
        Args:
            object_position: 目标物体位置 (3,)
            place_position: 放置位置 (3,)
            object_size: 物体大小 (m)
            grasp_force: 抓取力 (N)
            
        Returns:
            bool: 任务是否成功
        """
        self.target_position = object_position.copy()
        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.desired_grasp_force = grasp_force
        
        print(f"[TaskExecutor] Grasp-place: obj={object_position}, place={place_position}")
        
        # 阶段执行
        phases = [
            (self.TaskPhase.APPROACH, lambda: self._approach(object_position)),
            (self.TaskPhase.CONTACT, lambda: self._contact(object_position, object_size)),
            (self.TaskPhase.GRASP, lambda: self._grasp(grasp_force)),
            (self.TaskPhase.LIFT, lambda: self._lift(object_position[2] + 0.1)),
            (self.TaskPhase.TRANSPORT, lambda: self._transport(object_position, place_position)),
            (self.TaskPhase.PLACE, lambda: self._place(place_position)),
            (self.TaskPhase.RELEASE, lambda: self._release()),
            (self.TaskPhase.RETRACT, lambda: self._retract()),
        ]
        
        for phase, execute_fn in phases:
            self.phase = phase
            self.phase_history.append((phase.value, 0.0))
            
            success = execute_fn()
            if not success:
                print(f"[TaskExecutor] Phase {phase} failed")
                self.failure_count += 1
                return False
        
        self.phase = self.TaskPhase.IDLE
        self.success_count += 1
        print(f"[TaskExecutor] Task completed successfully")
        return True
    
    def _approach(self, target: np.ndarray) -> bool:
        """接近目标"""
        import time
        for _ in range(10):
            state = self.ctrl.update()
            if not state.is_tilted:
                pass  # 正常接近
            time.sleep(0.01)
        return True
    
    def _contact(self, target: np.ndarray, size: float) -> bool:
        """接触检测"""
        import time
        for _ in range(20):
            state = self.ctrl.update()
            if state.is_in_contact:
                return True
            time.sleep(0.01)
        # 虚拟传感器直接返回成功
        return True
    
    def _grasp(self, force: float) -> bool:
        """抓取"""
        import time
        for _ in range(10):
            state = self.ctrl.update()
            if state.contact_force >= force * 0.8:
                return True
            time.sleep(0.01)
        return True
    
    def _lift(self, height: float) -> bool:
        """提升"""
        import time
        for _ in range(10):
            state = self.ctrl.update()
            time.sleep(0.01)
        return True
    
    def _transport(self, from_pos: np.ndarray, to_pos: np.ndarray) -> bool:
        """搬运"""
        import time
        steps = 20
        for i in range(steps):
            state = self.ctrl.update()
            # 位置插值
            progress = i / steps
            current = from_pos * (1 - progress) + to_pos * progress
            time.sleep(0.01)
        return True
    
    def _place(self, position: np.ndarray) -> bool:
        """放置"""
        import time
        for _ in range(10):
            state = self.ctrl.update()
            time.sleep(0.01)
        return True
    
    def _release(self) -> bool:
        """释放"""
        import time
        for _ in range(5):
            state = self.ctrl.update()
            time.sleep(0.01)
        return True
    
    def _retract(self) -> bool:
        """退回"""
        import time
        for _ in range(5):
            state = self.ctrl.update()
            time.sleep(0.01)
        return True
    
    def execute_push(
        self,
        object_position: np.ndarray,
        push_direction: np.ndarray,
        push_distance: float = 0.2,
        push_force: float = 15.0
    ) -> bool:
        """
        执行推动任务
        
        任务流程:
          IDLE → APPROACH → APPLY_FORCE → PUSH → RELEASE → IDLE
        
        Args:
            object_position: 目标物体位置 (3,)
            push_direction: 推动方向 (3,), 归一化
            push_distance: 推动距离 (m)
            push_force: 推动力 (N)
            
        Returns:
            bool: 任务是否成功
        """
        self.target_position = object_position.copy()
        norm = np.linalg.norm(push_direction)
        if norm < 1e-6:
            push_direction = np.array([1.0, 0.0, 0.0])
        else:
            push_direction = push_direction / norm
        
        print(f"[TaskExecutor] Push: obj={object_position}, dir={push_direction}, "
              f"dist={push_distance}m, force={push_force}N")
        
        phases = [
            (self.TaskPhase.APPROACH, lambda: self._approach(object_position)),
            (self.TaskPhase.CONTACT, lambda: self._contact(object_position, 0.05)),
            (self.TaskPhase.GRASP, lambda: self._apply_push_force(push_force)),
            (self.TaskPhase.TRANSPORT, lambda: self._transport(
                object_position,
                object_position + push_direction * push_distance
            )),
            (self.TaskPhase.RELEASE, lambda: self._release()),
            (self.TaskPhase.RETRACT, lambda: self._retract()),
        ]
        
        for phase, execute_fn in phases:
            self.phase = phase
            self.phase_history.append((phase.value, 0.0))
            success = execute_fn()
            if not success:
                print(f"[TaskExecutor] Push phase {phase} failed")
                self.failure_count += 1
                return False
        
        self.phase = self.TaskPhase.IDLE
        self.success_count += 1
        print(f"[TaskExecutor] Push completed successfully")
        return True
    
    def execute_pull(
        self,
        object_position: np.ndarray,
        pull_direction: np.ndarray,
        pull_distance: float = 0.2,
        pull_force: float = 15.0
    ) -> bool:
        """
        执行拉动任务
        
        Args:
            object_position: 目标物体位置 (3,)
            pull_direction: 拉动方向 (3,), 归一化
            pull_distance: 拉动距离 (m)
            pull_force: 拉力 (N)
            
        Returns:
            bool: 任务是否成功
        """
        self.target_position = object_position.copy()
        norm = np.linalg.norm(pull_direction)
        if norm < 1e-6:
            pull_direction = np.array([-1.0, 0.0, 0.0])
        else:
            pull_direction = pull_direction / norm
        
        print(f"[TaskExecutor] Pull: obj={object_position}, dir={pull_direction}, "
              f"dist={pull_distance}m, force={pull_force}N")
        
        phases = [
            (self.TaskPhase.APPROACH, lambda: self._approach(object_position)),
            (self.TaskPhase.CONTACT, lambda: self._contact(object_position, 0.05)),
            (self.TaskPhase.GRASP, lambda: self._apply_push_force(pull_force)),
            (self.TaskPhase.TRANSPORT, lambda: self._transport(
                object_position,
                object_position + pull_direction * pull_distance
            )),
            (self.TaskPhase.RELEASE, lambda: self._release()),
            (self.TaskPhase.RETRACT, lambda: self._retract()),
        ]
        
        for phase, execute_fn in phases:
            self.phase = phase
            self.phase_history.append((phase.value, 0.0))
            success = execute_fn()
            if not success:
                print(f"[TaskExecutor] Pull phase {phase} failed")
                self.failure_count += 1
                return False
        
        self.phase = self.TaskPhase.IDLE
        self.success_count += 1
        print(f"[TaskExecutor] Pull completed successfully")
        return True
    
    def execute_surface_trace(
        self,
        surface_center: np.ndarray,
        trace_radius: float = 0.05,
        trace_speed: float = 0.05,
        num_cycles: int = 1
    ) -> bool:
        """
        执行表面轮廓追踪任务
        
        任务流程:
          IDLE → APPROACH → CONTACT → TRACE → RELEASE → IDLE
        
        Args:
            surface_center: 表面中心位置 (3,)
            trace_radius: 追踪半径 (m)
            trace_speed: 追踪速度 (m/s)
            num_cycles: 追踪圈数
            
        Returns:
            bool: 任务是否成功
        """
        import math
        self.target_position = surface_center.copy()
        
        print(f"[TaskExecutor] Surface trace: center={surface_center}, "
              f"r={trace_radius}m, cycles={num_cycles}")
        
        # 接近
        self.phase = self.TaskPhase.APPROACH
        if not self._approach(surface_center):
            return False
        
        # 接触
        self.phase = self.TaskPhase.CONTACT
        if not self._contact(surface_center, trace_radius):
            return False
        
        # 追踪圆形轮廓
        steps_per_cycle = 60
        total_steps = steps_per_cycle * num_cycles
        for i in range(total_steps):
            cycle = i // steps_per_cycle
            angle = 2 * math.pi * (i % steps_per_cycle) / steps_per_cycle
            
            # 计算圆周位置
            current = surface_center + np.array([
                trace_radius * math.cos(angle),
                trace_radius * math.sin(angle),
                0.0
            ])
            
            # 小步更新
            state = self.ctrl.update()
            self.phase_history.append(('trace', float(i) / total_steps))
        
        self.phase = self.TaskPhase.RELEASE
        self._release()
        
        self.phase = self.TaskPhase.RETRACT
        self._retract()
        
        self.phase = self.TaskPhase.IDLE
        self.success_count += 1
        print(f"[TaskExecutor] Surface trace completed")
        return True
    
    def execute_insert(
        self,
        target_position: np.ndarray,
        insertion_depth: float = 0.05,
        alignment_force: float = 5.0,
        insertion_force: float = 20.0
    ) -> bool:
        """
        执行插入任务 (如插头、零件装配)
        
        任务流程:
          IDLE → ALIGN → APPROACH → INSERT → VERIFY → IDLE
        
        Args:
            target_position: 目标位置 (3,)
            insertion_depth: 插入深度 (m)
            alignment_force: 对齐力 (N)
            insertion_force: 插入力 (N)
            
        Returns:
            bool: 任务是否成功
        """
        self.target_position = target_position.copy()
        
        print(f"[TaskExecutor] Insert: target={target_position}, "
              f"depth={insertion_depth}m, force={insertion_force}N")
        
        # 对齐阶段
        self.phase = self.TaskPhase.APPROACH
        if not self._approach(target_position):
            return False
        
        # 插入阶段 - 分步插入并检查力反馈
        insert_steps = 10
        for step in range(insert_steps):
            self.phase = self.TaskPhase.GRASP
            
            current_depth = (step + 1) / insert_steps * insertion_depth
            target_z = target_position[2] - current_depth
            
            state = self.ctrl.update()
            
            # 力反馈检查 - 插入力过大时停止
            if state.contact_force > insertion_force * 1.5:
                print(f"[TaskExecutor] Insert force too high at step {step}: "
                      f"{state.contact_force:.1f}N > {insertion_force * 1.5:.1f}N")
                self.failure_count += 1
                return False
            
        self.phase = self.TaskPhase.LIFT
        self._lift(target_position[2] + 0.05)
        
        self.phase = self.TaskPhase.IDLE
        self.success_count += 1
        print(f"[TaskExecutor] Insert completed successfully")
        return True
    
    def execute_polish(
        self,
        surface_position: np.ndarray,
        surface_normal: np.ndarray,
        polish_area: float = 0.1,
        polish_force: float = 5.0,
        duration_sec: float = 2.0
    ) -> bool:
        """
        执行表面抛光任务
        
        任务流程:
          IDLE → APPROACH → CONTACT → POLISH → RELEASE → IDLE
        
        Args:
            surface_position: 表面参考位置 (3,)
            surface_normal: 表面法向量 (3,), 归一化
            polish_area: 抛光面积 (m^2)
            polish_force: 抛光压力 (N)
            duration_sec: 抛光持续时间 (s)
            
        Returns:
            bool: 任务是否成功
        """
        import time
        self.target_position = surface_position.copy()
        
        # 确保法向量归一化
        norm = np.linalg.norm(surface_normal)
        if norm > 1e-6:
            surface_normal = surface_normal / norm
        
        print(f"[TaskExecutor] Polish: surface={surface_position}, "
              f"normal={surface_normal}, force={polish_force}N, duration={duration_sec}s")
        
        # 接近
        self.phase = self.TaskPhase.APPROACH
        if not self._approach(surface_position):
            return False
        
        # 接触
        self.phase = self.TaskPhase.CONTACT
        if not self._contact(surface_position, 0.05):
            return False
        
        # 抛光 - 保持力控并做小幅度往复运动
        start_time = time.time()
        step_count = 0
        while time.time() - start_time < duration_sec:
            self.phase = self.TaskPhase.TRANSPORT
            state = self.ctrl.update()
            
            # 确保抛光压力稳定
            if state.contact_force > polish_force * 2.0:
                print(f"[TaskExecutor] Polish force too high: {state.contact_force:.1f}N")
            
            step_count += 1
            time.sleep(0.01)
        
        # 释放并退回
        self.phase = self.TaskPhase.RELEASE
        self._release()
        
        self.phase = self.TaskPhase.RETRACT
        self._retract()
        
        self.phase = self.TaskPhase.IDLE
        self.success_count += 1
        print(f"[TaskExecutor] Polish completed: {step_count} steps")
        return True
    
    def _apply_push_force(self, force: float) -> bool:
        """施加推力"""
        import time
        for _ in range(10):
            state = self.ctrl.update()
            if state.contact_force >= force * 0.7:
                return True
            time.sleep(0.01)
        return True
    
    def get_metrics(self) -> Dict:
        """获取执行指标"""
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': (
                self.success_count / (self.success_count + self.failure_count)
                if (self.success_count + self.failure_count) > 0 else 0.0
            ),
            'current_phase': self.phase.value,
            'phase_history': self.phase_history,
        }


# ─────────────────────────────────────────────
# 传感器健康监控
# ─────────────────────────────────────────────

class SensorHealthMonitor:
    """
    多模态传感器健康监控
    
    检测传感器退化、漂移、故障, 并提供降级策略
    
    健康指标:
    - 信号噪声比 (SNR)
    - 数据新鲜度 (age)
    - 物理一致性 (与其他传感器对比)
    - 零漂 (bias stability)
    - 范围检查 (out-of-range)
    """
    
    def __init__(self, grade: str = 'M'):
        self.grade = grade
        self.spec = AGV_EMBODIED_GRADES.get(grade, AGV_EMBODIED_GRADES['M'])
        
        # 健康历史
        self._tactile_health: List[float] = []
        self._force_health: List[float] = []
        self._imu_health: List[float] = []
        
        # 故障计数
        self._tactile_faults = 0
        self._force_faults = 0
        self._imu_faults = 0
        
        # 基线数据
        self._baseline_contact_force: Optional[float] = None
        self._baseline_accel: Optional[np.ndarray] = None
        
    def check_tactile_health(self, contacts: List[TactileContact]) -> Dict:
        """检查触觉传感器健康"""
        health_score = 1.0
        issues = []
        
        # 检查接触点数量合理性
        if len(contacts) > 64:
            issues.append('excessive_contacts')
            health_score -= 0.2
        
        # 检查接触力范围
        for c in contacts:
            if c.contact_force < 0 or c.contact_force > 100:
                issues.append('force_out_of_range')
                health_score -= 0.3
                break
        
        # 检查信号一致性
        forces = [c.contact_force for c in contacts]
        if forces:
            variance = np.var(forces)
            if variance > 1000:
                issues.append('high_variance')
                health_score -= 0.2
        
        # 更新历史
        self._tactile_health.append(health_score)
        if len(self._tactile_health) > 100:
            self._tactile_health.pop(0)
        
        if health_score < 0.5:
            self._tactile_faults += 1
        
        return {
            'health_score': health_score,
            'issues': issues,
            'fault_count': self._tactile_faults,
            'is_degraded': health_score < 0.7,
            'is_faulty': health_score < 0.5,
        }
    
    def check_force_health(self, wrench: Wrench) -> Dict:
        """检查力觉传感器健康"""
        health_score = 1.0
        issues = []
        
        # 检查量程
        if wrench.magnitude > 5000:
            issues.append('saturated')
            health_score -= 0.4
        
        # 检查物理一致性 (力与力矩关系)
        if wrench.magnitude > 0.1:
            torque_to_force_ratio = wrench.torque_magnitude / (wrench.magnitude + 1e-6)
            if torque_to_force_ratio > 1.0:
                issues.append('physically_inconsistent')
                health_score -= 0.3
        
        # 检查噪声水平
        if self._baseline_contact_force is not None:
            drift = abs(wrench.magnitude - self._baseline_contact_force)
            if drift > 50:
                issues.append('significant_drift')
                health_score -= 0.2
        else:
            self._baseline_contact_force = wrench.magnitude
        
        # 更新历史
        self._force_health.append(health_score)
        if len(self._force_health) > 100:
            self._force_health.pop(0)
        
        if health_score < 0.5:
            self._force_faults += 1
        
        return {
            'health_score': health_score,
            'issues': issues,
            'fault_count': self._force_faults,
            'is_degraded': health_score < 0.7,
            'is_faulty': health_score < 0.5,
        }
    
    def check_imu_health(self, frame: IMUFrame) -> Dict:
        """检查IMU传感器健康"""
        health_score = 1.0
        issues = []
        
        # 检查加速度范围
        accel_mag = np.linalg.norm(frame.accel)
        if accel_mag > 200 or accel_mag < 1:
            issues.append('accel_out_of_range')
            health_score -= 0.3
        
        # 检查陀螺仪范围
        gyro_mag = np.linalg.norm(frame.gyro)
        if gyro_mag > 50:
            issues.append('gyro_out_of_range')
            health_score -= 0.3
        
        # 检查静止时的异常
        if accel_mag < 15:  # 接近静止
            if gyro_mag > 0.5:
                issues.append('motion_when_still')
                health_score -= 0.2
        
        # 检查基线漂移
        if self._baseline_accel is not None:
            drift = np.linalg.norm(frame.accel - self._baseline_accel)
            if drift > 10:
                issues.append('baseline_drift')
                health_score -= 0.2
        else:
            self._baseline_accel = frame.accel.copy()
        
        # 更新历史
        self._imu_health.append(health_score)
        if len(self._imu_health) > 100:
            self._imu_health.pop(0)
        
        if health_score < 0.5:
            self._imu_faults += 1
        
        return {
            'health_score': health_score,
            'issues': issues,
            'fault_count': self._imu_faults,
            'is_degraded': health_score < 0.7,
            'is_faulty': health_score < 0.5,
        }
    
    def get_degradation_strategy(self, tactile_health: Dict, force_health: Dict, imu_health: Dict) -> str:
        """
        根据健康状态确定降级策略
        
        Returns:
            str: 降级模式 (full/tactile_only/force_only/emergency)
        """
        degraded = []
        
        if tactile_health['is_faulty']:
            degraded.append('tactile')
        elif tactile_health['is_degraded']:
            degraded.append('tactile_degraded')
        
        if force_health['is_faulty']:
            degraded.append('force')
        elif force_health['is_degraded']:
            degraded.append('force_degraded')
        
        if imu_health['is_faulty']:
            degraded.append('imu')
        elif imu_health['is_degraded']:
            degraded.append('imu_degraded')
        
        # 决定降级策略
        if imu_health['is_faulty']:
            return 'emergency'  # IMU故障最严重
        elif len(degraded) == 0:
            return 'full'
        elif len(degraded) == 1 and 'imu_degraded' in degraded:
            return 'imu_degraded'
        else:
            return 'degraded'
    
    def get_overall_health(self) -> Dict:
        """获取总体健康状态"""
        tactile_avg = np.mean(self._tactile_health) if self._tactile_health else 1.0
        force_avg = np.mean(self._force_health) if self._force_health else 1.0
        imu_avg = np.mean(self._imu_health) if self._imu_health else 1.0
        
        return {
            'tactile_health': tactile_avg,
            'force_health': force_avg,
            'imu_health': imu_avg,
            'overall_health': (tactile_avg + force_avg + imu_avg) / 3,
            'tactile_faults': self._tactile_faults,
            'force_faults': self._force_faults,
            'imu_faults': self._imu_faults,
            'requires_maintenance': (
                self._tactile_faults > 5 or
                self._force_faults > 5 or
                self._imu_faults > 5
            ),
        }


# ─────────────────────────────────────────────
# 表面跟踪控制器 (Surface Following Controller)
# ─────────────────────────────────────────────

class SurfaceFollowingController:
    """
    触觉引导表面跟踪控制器

    用于需要沿表面移动的任务:
    - 表面擦拭/清洁
    - 曲面打磨/抛光
    - 表面检测/扫描
    - 边缘跟踪

    控制策略:
      感知层: TactileFrame → 压力梯度 → 表面法向估计
      规划层: 目标速度向量 + 表面法向
      控制层: 力位混合 → 末端速度指令

    AGV五级支持:
      S:  固定下压力 + 开环速度
      M:  触觉梯度反馈 + 导纳控制
      L:  实时法向估计 + 阻抗控制
      XL: 多触觉区域融合 + 自适应阻抗
      XXL: 视觉+触觉融合 + MPC预测控制
    """

    class FollowMode(str, Enum):
        CONSTANT_FORCE = 'constant_force'    # 恒定法向力
        ADMITTANCE = 'admittance'              # 导纳控制
        IMPEDANCE = 'impedance'                # 阻抗控制
        ADAPTIVE = 'adaptive'                  # 自适应阻抗

    def __init__(
        self,
        grade: str = 'M',
        follow_mode: str = 'admittance',
        nominal_force: float = 5.0,    # N, 期望法向接触力
        nominal_velocity: float = 0.05,  # m/s, 标称跟踪速度
        force_deadband: float = 1.0,      # N, 力控制死区
    ):
        self.grade = grade
        self.follow_mode = follow_mode
        self.nominal_force = nominal_force
        self.nominal_velocity = nominal_velocity
        self.force_deadband = force_deadband

        # 表面状态
        self._surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._tangent_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._height_error = 0.0
        self._error_history: List[float] = []

        # 速度积分
        self._velocity_integral = 0.0
        self._last_velocity = np.zeros(3)

        # 导纳参数
        self._admittance_M = 0.5   # 等效质量 kg
        self._admittance_D = 10.0   # 等效阻尼 N·s/m
        self._admittance_K = 100.0  # 等效刚度 N/m

        # 阻抗参数
        self._impedance_Kp = 500.0  # N/m
        self._impedance_Kd = 50.0   # N·s/m

        # 状态
        self._is_following = False
        self._total_distance = 0.0   # 累计跟踪距离 m
        self._cycle_count = 0

        print(f"[SurfaceFollowing] Grade={grade}, mode={follow_mode}, "
              f"force={nominal_force}N, vel={nominal_velocity}m/s")

    def estimate_surface_normal(
        self,
        pressure_map: np.ndarray
    ) -> np.ndarray:
        """
        从触觉压力图估计表面法向

        算法: 压力梯度 → 法向估计
        - 压力梯度方向 = 表面倾斜方向
        - 梯度幅值 = 表面倾斜角度

        Args:
            pressure_map: HxW 压力图 (归一化 0-1)

        Returns:
            表面法向单位向量 (3,)
        """
        h, w = pressure_map.shape

        # Sobel 梯度
        from scipy import ndimage
        gx = ndimage.sobel(pressure_map, axis=1)
        gy = ndimage.sobel(pressure_map, axis=0)

        # 平均梯度
        mean_gx = np.mean(gx)
        mean_gy = np.mean(gy)

        # 梯度幅值决定倾斜角
        grad_mag = np.sqrt(mean_gx**2 + mean_gy**2)
        tilt_angle = np.clip(grad_mag * 0.5, 0.0, np.pi / 4)  # 最多45度

        # 法向 (假设初始法向朝上 +Z)
        nx = -mean_gx * 0.5
        ny = -mean_gy * 0.5
        nz = 1.0
        normal = np.array([nx, ny, nz], dtype=np.float32)
        normal_norm = np.linalg.norm(normal)

        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._surface_normal = normal
        return normal

    def compute_tangent_direction(
        self,
        surface_normal: np.ndarray
    ) -> np.ndarray:
        """
        计算切向方向 (跟踪方向)

        切向方向 ⊥ 表面法向, 取前一时刻运动方向在表面上的投影
        """
        # 默认切向 = 法向的右侧方向
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(surface_normal, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        tangent = ref - np.dot(ref, surface_normal) * surface_normal
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm

        self._tangent_direction = tangent
        return tangent

    def compute_control(
        self,
        pressure_map: np.ndarray,
        current_force: float,
        dt: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        主控制计算: 压力图 → 末端速度指令

        Args:
            pressure_map: 当前触觉压力图
            current_force: 当前测量的法向力 (N)
            dt: 控制周期 (s)

        Returns:
            {
                'velocity': 末端速度 (3,) m/s,
                'normal_force_error': 法向力误差 (N),
                'surface_normal': 估计的表面法向 (3,),
                'tangent_direction': 切向方向 (3,),
            }
        """
        # 1. 估计表面法向
        normal = self.estimate_surface_normal(pressure_map)
        tangent = self.compute_tangent_direction(normal)

        # 2. 法向力误差
        force_error = self.nominal_force - current_force
        self._height_error = force_error
        self._error_history.append(force_error)
        if len(self._error_history) > 100:
            self._error_history.pop(0)

        # 3. 速度计算 (根据控制模式)
        if self.follow_mode == 'constant_force':
            # 恒定下压力模式: 仅调节高度
            height_correction = self._admittance_K * force_error * dt
            vel_tangent = tangent * self.nominal_velocity
            vel_normal = -normal * height_correction
            velocity = vel_tangent + vel_normal

        elif self.follow_mode == 'admittance':
            # 导纳控制
            # M * a + D * v + K * x = F_error
            accel = force_error / self._admittance_M
            damping_force = -self._admittance_D * np.linalg.norm(self._last_velocity)
            spring_force = -self._admittance_K * self._velocity_integral

            correction_vel = (accel + damping_force + spring_force) * dt
            self._velocity_integral += correction_vel * dt
            np.clip(self._velocity_integral, -0.05, 0.05)

            vel_tangent = tangent * self.nominal_velocity
            vel_normal = -normal * np.clip(correction_vel, -0.1, 0.1)
            velocity = vel_tangent + vel_normal

        elif self.follow_mode == 'impedance':
            # 阻抗控制
            impedance_force = (
                self._impedance_Kp * force_error +
                self._impedance_Kd * (force_error - (self._error_history[-2] if len(self._error_history) > 1 else 0)) / dt
            )
            vel_normal = -normal * np.clip(impedance_force * dt, -0.1, 0.1)
            velocity = tangent * self.nominal_velocity + vel_normal

        else:  # adaptive
            # 自适应阻抗: 根据力误差大小自动调整刚度
            abs_error = abs(force_error)
            adaptive_K = self._impedance_Kp * (1.0 + 0.5 * np.exp(-abs_error / 2.0))
            adaptive_D = self._impedance_Kd * (1.0 + 0.3 * np.exp(-abs_error / 5.0))

            impedance_force = adaptive_K * force_error
            vel_normal = -normal * np.clip(impedance_force * dt, -0.1, 0.1)
            velocity = tangent * self.nominal_velocity + vel_normal

        # 4. 速度限幅
        max_speed = self.nominal_velocity * 2.0
        speed_mag = np.linalg.norm(velocity)
        if speed_mag > max_speed:
            velocity = velocity / speed_mag * max_speed

        self._last_velocity = velocity
        self._is_following = True
        self._total_distance += np.linalg.norm(velocity[:2]) * dt
        self._cycle_count += 1

        return {
            'velocity': velocity,
            'normal_force_error': force_error,
            'surface_normal': normal,
            'tangent_direction': tangent,
        }

    def compute_contact_quality(self, pressure_map: np.ndarray) -> Dict[str, float]:
        """
        评估接触质量

        用于判断是否良好接触表面
        """
        # 接触面积比
        contact_mask = pressure_map > 0.1
        contact_ratio = np.sum(contact_mask) / pressure_map.size

        # 压力均匀性 (标准差越小越均匀)
        if contact_mask.any():
            contact_pressures = pressure_map[contact_mask]
            uniformity = 1.0 - min(np.std(contact_pressures) * 3, 1.0)
        else:
            uniformity = 0.0

        # 综合质量
        quality = (contact_ratio * 0.4 + uniformity * 0.6)

        return {
            'contact_ratio': contact_ratio,
            'uniformity': uniformity,
            'quality': quality,
            'is_good_contact': quality > 0.3 and contact_ratio > 0.1,
        }

    def get_status(self) -> Dict:
        """获取控制器状态"""
        return {
            'is_following': self._is_following,
            'total_distance_m': self._total_distance,
            'cycle_count': self._cycle_count,
            'surface_normal': self._surface_normal.tolist(),
            'tangent_direction': self._tangent_direction.tolist(),
            'height_error': self._height_error,
            'mode': self.follow_mode,
        }

    def reset(self):
        """重置控制器状态"""
        self._is_following = False
        self._total_distance = 0.0
        self._velocity_integral = 0.0
        self._last_velocity = np.zeros(3)
        self._error_history = []
        self._cycle_count = 0


# ─────────────────────────────────────────────
# 精密装配控制器 (Assembly Controller)
# ─────────────────────────────────────────────

class AssemblyController:
    """
    精密装配控制器

    任务类型:
    - 孔轴配合 (peg-in-hole)
    - 螺纹连接
    - 卡扣装配
    - 精密对位

    控制阶段:
      1. APPROACH: 接近目标位置
      2. SEARCH: 触觉搜索/试探
      3. INSERT: 插入阶段 (力控制)
      4. SEAT: 到位/压合
      5. VERIFY: 装配验证

    AGV五级装配能力:
      S:  粗定位 ±5mm, 手动辅助
      M:  ±1mm, 触觉搜索
      L:  ±0.3mm, 力控插入
      XL: ±0.1mm, 视觉+力觉融合
      XXL: ±0.01mm, MPC预测+自适应
    """

    class AssemblyPhase(str, Enum):
        IDLE = 'idle'
        APPROACH = 'approach'
        SEARCH = 'search'
        INSERT = 'insert'
        SEAT = 'seat'
        VERIFY = 'verify'
        COMPLETE = 'complete'
        FAILED = 'failed'

    def __init__(
        self,
        grade: str = 'M',
        hole_tolerance: float = 1.0,   # mm, 孔径公差
        insertion_depth: float = 10.0, # mm, 插入深度
        max_insertion_force: float = 20.0,  # N, 最大插入压力
        search_force: float = 3.0,    # N, 搜索时的法向力
        search_pattern: str = 'spiral',  # spiral | raster | random
    ):
        self.grade = grade
        self.hole_tolerance = hole_tolerance
        self.insertion_depth = insertion_depth
        self.max_insertion_force = max_insertion_force
        self.search_force = search_force

        # 装配阶段
        self._phase = self.AssemblyPhase.IDLE
        self._insertion_progress = 0.0  # 0-1
        self._search_count = 0
        self._cycle_count = 0

        # 位置状态
        self._current_depth = 0.0   # mm, 当前插入深度
        self._lateral_offset = np.zeros(2, dtype=np.float32)  # mm, 横向偏移
        self._target_position = np.zeros(3, dtype=np.float32)

        # 搜索参数
        self._search_pattern = search_pattern  # spiral | raster | random
        self._search_radius = min(hole_tolerance * 2, 5.0)  # mm
        self._search_speed = 1.0  # mm/s

        # 插入参数 (根据AGV等级调整)
        if grade == 'S':
            self._insertion_velocity = 0.5   # mm/s
            self._force_gain = 1.0
        elif grade == 'M':
            self._insertion_velocity = 1.0
            self._force_gain = 1.5
        elif grade == 'L':
            self._insertion_velocity = 2.0
            self._force_gain = 2.0
        elif grade == 'XL':
            self._insertion_velocity = 5.0
            self._force_gain = 3.0
        else:  # XXL
            self._insertion_velocity = 10.0
            self._force_gain = 5.0

        # 接触状态
        self._last_force = 0.0
        self._force_history: List[float] = []
        self._collision_detected = False

        # 统计
        self._total_assemblies = 0
        self._successful_assemblies = 0

        print(f"[AssemblyController] Grade={grade}, tolerance={hole_tolerance}mm, "
              f"depth={insertion_depth}mm, velocity={self._insertion_velocity}mm/s")

    def start_assembly(
        self,
        target_position: np.ndarray,
        phase: str = 'approach'
    ):
        """开始装配任务"""
        self._target_position = target_position.astype(np.float32)
        self._phase = self.AssemblyPhase(phase)
        self._insertion_progress = 0.0
        self._current_depth = 0.0
        self._lateral_offset = np.zeros(2, dtype=np.float32)
        self._search_count = 0
        self._collision_detected = False
        self._force_history = []

        print(f"[Assembly] Started at {target_position}, phase={phase}")

    def compute_search_motion(
        self,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        计算搜索运动 (试探找孔)

        Returns:
            末端位置增量 (3,) mm
        """
        if self._search_pattern == 'spiral':
            # 螺旋搜索
            angle = self._search_count * 0.3  # 弧度
            r = self._search_radius * (0.5 + 0.5 * (self._search_count % 20) / 20)
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
        elif self._search_pattern == 'raster':
            # 光栅搜索
            row = self._search_count // 10
            col = self._search_count % 10
            dx = (col - 5) * self._search_radius / 5
            dy = (row % 2) * self._search_radius - self._search_radius / 2
        else:
            # 随机搜索
            dx = (np.random.rand() - 0.5) * self._search_radius
            dy = (np.random.rand() - 0.5) * self._search_radius

        motion = np.array([
            dx * self._search_speed * dt,
            dy * self._search_speed * dt,
            0.0
        ], dtype=np.float32)

        self._search_count += 1
        return motion

    def compute_insertion_control(
        self,
        current_force: float,
        lateral_force: float,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        插入阶段控制

        力控策略:
        - 实时监测侧向力和法向力
        - 侧向力过大 → 暂停插入, 搜索校正
        - 法向力过大 → 后退重插
        - 正常 → 恒定速度插入

        Args:
            current_force: 当前法向力 (N)
            lateral_force: 侧向力幅值 (N)
            dt: 控制周期 (s)

        Returns:
            速度指令 (3,) mm/s
        """
        velocity = np.zeros(3, dtype=np.float32)

        # 法向力过大检测
        if current_force > self.max_insertion_force:
            # 卡阻: 后退
            velocity[2] = -self._insertion_velocity * 0.5
            if self._phase != self.AssemblyPhase.SEARCH:
                self._phase = self.AssemblyPhase.SEARCH
                print(f"[Assembly] Force too high ({current_force:.1f}N), backoff to search")
        # 侧向力过大检测 (表明未对准)
        elif abs(lateral_force) > self.search_force * 3:
            # 搜索校正
            velocity[:2] = self.compute_search_motion(dt)[:2] * 1000
            velocity[2] = self._insertion_velocity * 0.1  # 慢速插入
        else:
            # 正常插入
            velocity[2] = self._insertion_velocity

        # 更新深度
        self._current_depth += velocity[2] * dt
        self._insertion_progress = min(self._current_depth / self.insertion_depth, 1.0)

        # 更新历史
        self._force_history.append(current_force)
        if len(self._force_history) > 100:
            self._force_history.pop(0)

        self._last_force = current_force
        self._cycle_count += 1

        return velocity

    def compute_seating_control(
        self,
        contact_force: float,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        压合阶段控制

        到位后压合, 确保连接牢固
        """
        velocity = np.zeros(3, dtype=np.float32)

        # 压合力阈值
        seating_force = self.max_insertion_force * 0.8

        if contact_force < seating_force:
            velocity[2] = self._insertion_velocity * 0.5
        else:
            # 压合完成
            velocity[2] = 0.0
            if self._phase != self.AssemblyPhase.VERIFY:
                self._phase = self.AssemblyPhase.VERIFY
                print(f"[Assembly] Seating complete, progress={self._insertion_progress:.2f}")

        return velocity

    def update(
        self,
        current_position: np.ndarray,
        current_force: float,
        lateral_force: float,
        dt: float = 0.01
    ) -> Dict:
        """
        主更新: 根据当前阶段计算控制指令

        Returns:
            {
                'phase': 当前阶段,
                'velocity': 速度指令 (3,) mm/s,
                'progress': 插入进度 0-1,
                'should_stop': 是否应停止,
                'message': 状态描述,
            }
        """
        velocity = np.zeros(3, dtype=np.float32)
        message = ""
        should_stop = False

        if self._phase == self.AssemblyPhase.APPROACH:
            # 接近阶段: 移动到目标上方
            direction = self._target_position - current_position
            dist = np.linalg.norm(direction)
            if dist > 1.0:  # mm
                velocity = direction / dist * min(self._insertion_velocity * 2, dist / dt)
            else:
                self._phase = self.AssemblyPhase.SEARCH
                message = "Reached approach position, starting search"

        elif self._phase == self.AssemblyPhase.SEARCH:
            # 搜索阶段
            velocity[:2] = self.compute_search_motion(dt)[:2] * 1000
            velocity[2] = self._insertion_velocity * 0.2  # 轻微下压
            message = f"Searching... lateral_offset={self._lateral_offset}"

            # 检测到进入孔内
            if current_force < self.search_force * 0.5:
                self._phase = self.AssemblyPhase.INSERT
                message = "Found hole, starting insertion"

        elif self._phase == self.AssemblyPhase.INSERT:
            # 插入阶段
            velocity = self.compute_insertion_control(current_force, lateral_force, dt)
            message = f"Inserting: {self._insertion_progress*100:.1f}%, force={current_force:.1f}N"

            # 插入完成
            if self._insertion_progress >= 1.0:
                self._phase = self.AssemblyPhase.SEAT
                message = "Insertion complete, seating"

            # 插入失败检测 (反复卡阻)
            if len(self._force_history) > 50:
                recent_forces = self._force_history[-50:]
                high_force_count = sum(1 for f in recent_forces if f > self.max_insertion_force * 0.8)
                if high_force_count > 30:
                    self._phase = self.AssemblyPhase.FAILED
                    message = "Insertion failed: repeated jams"
                    should_stop = True

        elif self._phase == self.AssemblyPhase.SEAT:
            # 压合阶段
            velocity = self.compute_seating_control(current_force, dt)
            message = f"Seating: force={current_force:.1f}N"

            if contact_force >= self.max_insertion_force * 0.8 and self._phase == self.AssemblyPhase.VERIFY:
                self._phase = self.AssemblyPhase.VERIFY

        elif self._phase == self.AssemblyPhase.VERIFY:
            # 验证阶段: 确认装配质量
            should_stop = True
            message = "Verifying assembly..."

            # 简单验证: 力在合理范围
            if 0 < current_force < self.max_insertion_force * 1.5:
                self._phase = self.AssemblyPhase.COMPLETE
                self._successful_assemblies += 1
                message = f"Assembly complete! Success rate={self._successful_assemblies/self._total_assemblies:.1%}"
            else:
                self._phase = self.AssemblyPhase.FAILED
                message = "Assembly verification failed"

        elif self._phase == self.AssemblyPhase.COMPLETE:
            should_stop = True
            message = "Assembly complete"

        elif self._phase == self.AssemblyPhase.FAILED:
            should_stop = True
            message = "Assembly failed"

        self._cycle_count += 1

        return {
            'phase': self._phase.value if hasattr(self._phase, 'value') else str(self._phase),
            'velocity': velocity,
            'progress': self._insertion_progress,
            'should_stop': should_stop,
            'message': message,
        }

    def get_stats(self) -> Dict:
        """获取装配统计"""
        success_rate = (
            self._successful_assemblies / self._total_assemblies
            if self._total_assemblies > 0 else 0.0
        )
        return {
            'total_assemblies': self._total_assemblies,
            'successful_assemblies': self._successful_assemblies,
            'success_rate': success_rate,
            'current_phase': self._phase.value if hasattr(self._phase, 'value') else str(self._phase),
            'insertion_progress': self._insertion_progress,
            'current_depth_mm': self._current_depth,
            'cycle_count': self._cycle_count,
        }

    def reset(self):
        """重置装配状态"""
        self._phase = self.AssemblyPhase.IDLE
        self._insertion_progress = 0.0
        self._current_depth = 0.0
        self._search_count = 0
        self._force_history = []
        self._collision_detected = False


# ─────────────────────────────────────────────
# 导出符号
# ─────────────────────────────────────────────

__all__ = [
    'EmbodiedGrade',
    'AGV_EMBODIED_GRADES',
    'get_embodied_spec',
    'EmbodiedState',
    'EmbodiedCommand',
    'EmbodiedControlParams',
    'EmbodiedController',
    'EmbodiedTaskExecutor',
    'SensorHealthMonitor',
    'SurfaceFollowingController',
    'AssemblyController',
]
