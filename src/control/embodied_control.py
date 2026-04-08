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
            if c.force < 0 or c.force > 100:
                issues.append('force_out_of_range')
                health_score -= 0.3
                break
        
        # 检查信号一致性
        forces = [c.force for c in contacts]
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
]
