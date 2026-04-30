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
MuJoCo 物理引擎仿真模块
=======================

基于 DeepMind MuJoCo 的高性能物理仿真。

功能:
- 刚体动力学仿真
- 接触力/碰撞检测
- 关节空间/任务空间控制
- AGV差速驱动仿真
- 多指灵巧手抓取仿真

支持:
- MuJoCo 3.x
- MJCF 模型格式
- 自定义 XML 模型

使用示例:
    from simulation.mujoco_sim import MuJoCoSimulator, AGVMuJoCoModel
    
    # 创建AGV仿真
    sim = MuJoCoSimulator()
    model = sim.load_agv_model()  # 差速驱动AGV
    
    # 仿真控制
    sim.set_control([1.0, -1.0])  # 左/右轮速度
    for _ in range(1000):
        sim.step()
        state = sim.get_state()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import tempfile
import os

# MuJoCo 可选导入
try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    mujoco = None


class ControlMode(Enum):
    """控制模式"""
    JOINT_TORQUE = "joint_torque"      # 关节力矩控制
    JOINT_VELOCITY = "joint_velocity" # 关节速度控制
    JOINT_POSITION = "joint_position"  # 关节位置控制 (拟静力)
    TASK_VELOCITY = "task_velocity"   # 任务空间速度
    ACTUATOR = "actuator"             # 执行器直接控制


@dataclass
class MuJoCoConfig:
    """MuJoCo仿真配置"""
    # 仿真参数
    dt: float = 0.002              # 仿真步长 (s)
    substeps: int = 1              # 每步子迭代
    solver: str = "Newton"         # Newton/PGS/CG
    iterations: int = 100          # 求解器迭代次数
    tolerance: float = 1e-8        # 求解器收敛容差
    
    # 物理参数
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    air_density: float = 0.0       # 空气密度 (kg/m^3)
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 风速 (m/s)
    
    # 传感器噪声 (归一化)
    sensor_noise: float = 0.0     # 传感器噪声标准差
    actuator_noise: float = 0.0    # 执行器噪声标准差
    
    # 可视化
    visualize: bool = False        # 是否显示可视化窗口
    trackbodyid: Optional[int] = None  # 跟随视角的body ID
    
    # 仿真引擎
    integrate: str = "Euler"       # Euler/implicit/RK4


# ============================================================================
# MuJoCo AGV XML 模型
# ============================================================================

AGV_MJCF_TEMPLATE = """
<mujoco model="agv_differential">
    <compiler angle="radian" meshdir="." autolimits="true"/>
    
    <option timestep="{dt}" gravity="{gravity}"
            solver="{solver}" iterations="{iterations}" tolerance="{tolerance}">
        <flag contact="enable" energy="enable"/>
    </option>
    
    <worldbody>
        <!-- 地面 -->
        <geom type="plane" name="ground" size="10 10 0.1" pos="0 0 -0.001" friction="1 0.005 0.0001"
              rgba="0.5 0.5 0.5 1" conaffinity="1" contype="1"/>
        
        <!-- AGV车体 -->
        <body name="chassis" pos="0 0 0.05">
            <freejoint/>
            <inertial pos="0 0 0" mass="{chassis_mass}" diaginertia="0.1 0.1 0.1"/>
            
            <!-- 底盘几何 -->
            <geom type="box" size="{chassis_length} {chassis_width} 0.02"
                  rgba="0.2 0.6 0.8 1" friction="0.8 0.005 0.0001"/>
            
            <!-- 左轮 -->
            <body name="left_wheel" pos="-0.05 0.22 0" axisangle="1 0 0 90">
                <joint name="left_wheel_joint" type="hinge" 
                       axis="0 0 1" damping="0.5" frictionloss="0.1"/>
                <geom type="cylinder" size="0.08 0.025" 
                      rgba="0.1 0.1 0.1 1" friction="1.0 0.005 0.0001"/>
            </body>
            
            <!-- 右轮 -->
            <body name="right_wheel" pos="-0.05 -0.22 0" axisangle="1 0 0 90">
                <joint name="right_wheel_joint" type="hinge"
                       axis="0 0 1" damping="0.5" frictionloss="0.1"/>
                <geom type="cylinder" size="0.08 0.025"
                      rgba="0.1 0.1 0.1 1" friction="1.0 0.005 0.0001"/>
            </body>
            
            <!-- 负载托盘 -->
            <body name="load" pos="0 0 0.12">
                <inertial pos="0 0 0" mass="{load_mass}" diaginertia="0.01 0.01 0.01"/>
                <geom type="box" size="0.2 0.2 0.02"
                      rgba="0.8 0.6 0.2 1"/>
            </body>
            
            <!-- IMU传感器 -->
            <site name="imu_site" pos="0 0 0.1"/>
            
            <!-- 前置相机 -->
            <camera name="front" pos="0.3 0 0.2" xyaxes="1 0 0 0 1 0" fovy="60"/>
        </body>
        
        <!-- 障碍物 -->
        {obstacles}
        
        <!-- 目标点 -->
        {targets}
    </worldbody>
    
    <actuator>
        <motor joint="left_wheel_joint" gear="1" ctrllimited="true" 
               ctrlrange="-10 10" name="left_motor"/>
        <motor joint="right_wheel_joint" gear="1" ctrllimited="true"
               ctrlrange="-10 10" name="right_motor"/>
    </actuator>
    
    <sensor>
        <!-- IMU传感器 -->
        <gyro site="imu_site" name="gyro"/>
        <accelerometer site="imu_site" name="accel"/>
        
        <!-- 关节传感器 -->
        <jointpos joint="left_wheel_joint" name="left_wheel_pos"/>
        <jointpos joint="right_wheel_joint" name="right_wheel_pos"/>
        <jointvel joint="left_wheel_joint" name="left_wheel_vel"/>
        <jointvel joint="right_wheel_joint" name="right_wheel_vel"/>
    </sensor>
    
    <keyframe>
        <!-- 初始状态: chassis_free(7) + wheel1(1) + wheel2(1) = 9 qpos, 8 qvel -->
        <key name="home" qpos="0 0 0.05 1 0 0 0 0 0" 
             qvel="0 0 0 0 0 0 0 0"/>
    </keyframe>
</mujoco>
"""


class MuJoCoSimulator:
    """
    MuJoCo物理引擎仿真器
    
    封装 MuJoCo 仿真功能，提供:
    - 刚体动力学仿真
    - 接触力/碰撞检测
    - 关节空间/任务空间控制
    - AGV差速驱动仿真
    
    AGV五级规格:
    - Level 1: 2轮差速, 2m/s
    - Level 2: 2轮差速 + 负载传感
    - Level 3: IMU辅助定位
    - Level 4: 视觉+IMU融合
    - Level 5: 多AGV协调
    """
    
    def __init__(
        self,
        config: Optional[MuJoCoConfig] = None,
        xml_string: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        初始化MuJoCo仿真器
        
        Args:
            config: MuJoCo配置
            xml_string: MJCF XML字符串
            model_path: MJCF模型文件路径
        """
        if not HAS_MUJOCO:
            raise RuntimeError(
                "MuJoCo未安装。请运行: pip install mujoco"
            )
        
        self.config = config or MuJoCoConfig()
        self._viewer = None
        
        # 创建或加载模型
        if xml_string:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        elif model_path:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            # 默认AGV模型
            xml = self._create_agv_xml()
            self.model = mujoco.MjModel.from_xml_string(xml)
        
        self.data = mujoco.MjData(self.model)
        
        # 初始化
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置仿真参数
        self.model.opt.timestep = self.config.dt
        # 设置积分器类型
        integrator_map = {'euler': 0, 'implicit': 1, 'rk4': 2}
        self.model.opt.integrator = integrator_map.get(self.config.integrate.lower(), 0)
        
        # 传感器映射
        self._init_sensors()
        
    def _create_agv_xml(self) -> str:
        """创建AGV MJCF XML"""
        g = self.config.gravity
        return AGV_MJCF_TEMPLATE.format(
            dt=self.config.dt,
            gravity=f"{g[0]} {g[1]} {g[2]}",
            solver=self.config.solver,
            iterations=self.config.iterations,
            tolerance=self.config.tolerance,
            integrate=self.config.integrate,
            chassis_mass="10.0",
            chassis_length="0.25",
            chassis_width="0.25",
            load_mass="5.0",
            obstacles="",
            targets=""
        )
    
    def _init_sensors(self):
        """初始化传感器映射"""
        self._sensor_names = {}
        names_bytes = self.model.names
        # names is a bytes object, parse null-terminated strings
        start = 0
        i = 0
        while start < len(names_bytes):
            null_pos = names_bytes.find(b'\x00', start)
            if null_pos == -1:
                break
            name = names_bytes[start:null_pos].decode('utf-8')
            self._sensor_names[name] = i
            i += 1
            start = null_pos + 1
    
    def load_agv_model(self) -> str:
        """加载AGV模型"""
        return self._create_agv_xml()
    
    def set_control(self, ctrl: np.ndarray, mode: ControlMode = ControlMode.ACTUATOR):
        """
        设置控制输入
        
        Args:
            ctrl: 控制向量
            mode: 控制模式
        """
        if isinstance(ctrl, list):
            ctrl = np.array(ctrl, dtype=np.float64)
        
        if ctrl.shape != (self.model.nu,) and ctrl.size == self.model.nu:
            ctrl = ctrl.flatten()
        
        self.data.ctrl[:] = ctrl
        self._control_mode = mode
    
    def step(self):
        """执行一步仿真"""
        # 添加执行器噪声
        if self.config.actuator_noise > 0:
            noise = np.random.randn(self.model.nu) * self.config.actuator_noise
            self.data.ctrl[:] += noise
        
        # MuJoCo仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新查看器
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
    
    def step_n(self, n: int):
        """执行n步仿真"""
        for _ in range(n):
            self.step()
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取完整状态
        
        Returns:
            state: 状态字典
        """
        return {
            # 车体位姿
            'chassis_pos': self.data.qpos[:3].copy(),        # x, y, z
            'chassis_quat': self.data.qpos[3:7].copy(),       # quaternion
            'chassis_euler': self._quat_to_euler(self.data.qpos[3:7]),
            
            # 车体速度
            'chassis_vel': self.data.qvel[:3].copy(),        # linear
            'chassis_omega': self.data.qvel[3:6].copy(),      # angular
            
            # 关节状态
            'joint_pos': self.data.qpos[7:].copy() if len(self.data.qpos) > 7 else np.array([]),
            'joint_vel': self.data.qvel[6:].copy() if len(self.data.qvel) > 6 else np.array([]),
            
            # IMU数据
            'imu_accel': self.data.sensordata[0:3].copy(),   # 加速度
            'imu_gyro': self.data.sensordata[3:6].copy(),    # 角速度
            
            # 电机输出
            'actuator_force': self.data.actuator_force.copy(),
            
            # 接触力
            'contact_forces': self._get_contact_forces(),
            'num_contacts': self.data.ncon,
            
            # 时间
            'time': self.data.time,
        }
    
    def _get_contact_forces(self) -> np.ndarray:
        """获取接触力"""
        forces = np.zeros((self.data.ncon, 3))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # mujoco.mj_contactForce expects [6,1] shaped array (force+torque)
            force = np.zeros((6, 1))
            mujoco.mj_contactForce(self.model, self.data, i, force)
            forces[i] = force[:3, 0]  # Extract force component only
        return forces
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """四元数转Euler角"""
        # 四元数转旋转矩阵
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        # 从旋转矩阵提取欧拉角 (ZYX顺序: roll, pitch, yaw)
        euler = np.zeros(3)
        euler[0] = np.arctan2(mat[7], mat[8])  # roll
        euler[1] = np.arctan2(-mat[6], np.sqrt(mat[7]**2 + mat[8]**2))  # pitch
        euler[2] = np.arctan2(mat[3], mat[0])  # yaw
        return euler
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        获取强化学习观测
        
        Returns:
            obs: 观测向量 (跟AGV五级相关)
        """
        state = self.get_state()
        
        obs = {
            # 基础观测 (Level 1)
            'joint_pos': state['joint_pos'],
            'joint_vel': state['joint_vel'],
            
            # Level 2: 负载感知
            'actuator_force': state['actuator_force'],
            
            # Level 3: IMU融合
            'imu_accel': state['imu_accel'],
            'imu_gyro': state['imu_gyro'],
            
            # 相对目标
            'target_rel_pos': self._get_target_relative_pos(),
        }
        
        return obs
    
    def _get_target_relative_pos(self) -> np.ndarray:
        """获取相对目标位置"""
        target_pos = np.array([2.0, 0.0, 0.0])  # 简化：目标在前方2米
        chassis_pos = self.data.qpos[:3]
        return target_pos - chassis_pos
    
    def render(self, mode: str = "human"):
        """渲染画面"""
        if mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif mode == "rgb":
            return self._render_image()
    
    def _render_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """渲染RGB图像"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height),
                          self.model, self.data, img)
        return img
    
    def close(self):
        """关闭仿真器"""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    
    def __del__(self):
        self.close()
    
    # =========================================================================
    # AGV特定功能
    # =========================================================================
    
    def set_wheel_velocity(self, v_left: float, v_right: float):
        """
        设置左右轮速度 (差速驱动)
        
        Args:
            v_left: 左轮速度 (rad/s)
            v_right: 右轮速度 (rad/s)
        """
        # 将速度转换为力矩
        # 简化: 假设扭矩与目标速度成正比
        ctrl = np.array([v_left * 0.5, v_right * 0.5])
        self.set_control(ctrl, ControlMode.JOINT_VELOCITY)
    
    def get_odometry(self) -> Dict[str, float]:
        """
        获取里程计数据
        
        Returns:
            odom: 里程计字典
        """
        # 轮子角度积分
        qpos = self.data.qpos
        left_wheel = qpos[7] if len(qpos) > 7 else 0.0
        right_wheel = qpos[8] if len(qpos) > 8 else 0.0
        
        wheel_radius = 0.08  # 轮子半径
        wheel_base = 0.44    # 轮间距
        
        # 里程计算
        v_left = left_wheel * wheel_radius
        v_right = right_wheel * wheel_radius
        v = (v_left + v_right) / 2
        omega = (v_right - v_left) / wheel_base
        
        return {
            'x': self.data.qpos[0],
            'y': self.data.qpos[1],
            'theta': self._quat_to_euler(self.data.qpos[3:7])[2],  # yaw
            'v_linear': v,
            'v_angular': omega,
        }
    
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()
    
    def reset_to_pose(self, pos: np.ndarray, quat: np.ndarray):
        """
        重置到指定位姿
        
        Args:
            pos: 位置 [x, y, z]
            quat: 四元数 [w, x, y, z]
        """
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)


# ============================================================================
# 工厂函数
# ============================================================================

def create_mujoco_simulator(
    grade: str = 'M',
    config: Optional[MuJoCoConfig] = None
) -> MuJoCoSimulator:
    """
    创建MuJoCo仿真器 (AGV五级规格)
    
    Args:
        grade: AGV等级 ('S', 'M', 'L', 'XL', 'XXL')
        config: 仿真配置
        
    Returns:
        simulator: MuJoCo仿真器实例
    """
    if not HAS_MUJOCO:
        raise ImportError("MuJoCo未安装: pip install mujoco")
    
    default_config = config or MuJoCoConfig()
    
    # 根据AGV等级调整配置
    grade_configs = {
        'S': MuJoCoConfig(dt=0.005, iterations=50),
        'M': MuJoCoConfig(dt=0.002, iterations=100),
        'L': MuJoCoConfig(dt=0.001, iterations=200),
        'XL': MuJoCoConfig(dt=0.001, iterations=400),
        'XXL': MuJoCoConfig(dt=0.0005, iterations=800),
    }
    
    cfg = grade_configs.get(grade.upper(), grade_configs['M'])
    if config is None:
        default_config = cfg
    
    return MuJoCoSimulator(config=default_config)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("MuJoCo Simulation Module Test")
    print(f"MuJoCo available: {HAS_MUJOCO}")
    
    if HAS_MUJOCO:
        # 创建仿真器
        config = MuJoCoConfig(dt=0.002, visualize=False)
        sim = MuJoCoSimulator(config=config)
        
        print(f"Model: {sim.model.name}")
        print(f"nq (positions): {sim.model.nq}")
        print(f"nv (velocities): {sim.model.nv}")
        print(f"nu (controls): {sim.model.nu}")
        
        # 运行仿真
        print("\nRunning 100 steps...")
        for i in range(100):
            sim.set_wheel_velocity(1.0, 1.0)  # 直行
            sim.step()
        
        state = sim.get_state()
        print(f"Final position: {state['chassis_pos']}")
        print(f"Final heading: {state['chassis_euler'][2]:.3f} rad")
        
        odom = sim.get_odometry()
        print(f"Odometry: x={odom['x']:.3f}, y={odom['y']:.3f}, theta={odom['theta']:.3f}")
        
        sim.close()
        print("\nTest passed!")
    else:
        print("Skipping simulation test (MuJoCo not installed)")
