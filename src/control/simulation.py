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
仿真控制接口
============

为具身智能大脑提供统一的仿真环境接口
- Gym/Gymnasium 环境封装
- MuJoCo 仿真接口
- PyBullet 仿真接口
- Gazebo/ROS2 仿真接口
- 五级AGV规格对应的仿真参数

使用示例:
    sim = SimulationInterface(grade='M', backend='gym')
    env = sim.create_agv_env()
    obs = env.reset()
    for _ in range(1000):
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import time


class SimulationBackend(Enum):
    """仿真后端类型"""
    GYM = "gym"              # Gymnasium标准接口
    MUJOCO = "mujoco"       # MuJoCo物理引擎
    PYBULLET = "pybullet"    # PyBullet物理引擎
    GAZEBO = "gazebo"       # Gazebo/ROS2仿真
    NONE = "none"            # 无仿真(仅硬件)


class SimulationGrade(Enum):
    """仿真精度等级"""
    S = "S"   # 简化模型, 50Hz, 低精度传感器仿真
    M = "M"   # 标准模型, 100Hz, 基础传感器噪声
    L = "L"   # 高保真模型, 200Hz, 完整传感器模型
    XL = "XL" # 极高保真, 500Hz, 物理精确 + 传感器精确
    XXL = "XXL" # 极致仿真, 1000Hz, 全物理精度 + 数字孪生


@dataclass
class SimulationConfig:
    """仿真配置"""
    backend: SimulationBackend = SimulationBackend.GYM
    grade: SimulationGrade = SimulationGrade.M
    dt: float = 0.01                      # 仿真步长 (秒)
    real_time_factor: float = 1.0          # 实时因子 (1.0=实时)
    enable_rendering: bool = True          # 启用渲染
    enable_sensor_noise: bool = True       # 传感器噪声
    enable_dynamics: bool = True           # 动力学模型
    enable_collision: bool = True           # 碰撞检测
    camera_config: Dict[str, Any] = field(default_factory=dict)
    domain_randomization: bool = False     # 领域随机化
    record_video: bool = False             # 录制视频
    video_output_dir: str = "videos/"      # 视频输出目录

    def __post_init__(self):
        # 根据等级自动设置dt
        grade_dt_map = {
            SimulationGrade.S: 0.02,
            SimulationGrade.M: 0.01,
            SimulationGrade.L: 0.005,
            SimulationGrade.XL: 0.002,
            SimulationGrade.XXL: 0.001,
        }
        if self.dt == 0.01:  # Only auto-set if using default
            self.dt = grade_dt_map.get(self.grade, 0.01)


@dataclass
class SimState:
    """仿真状态"""
    timestamp: float
    dt: float
    position: np.ndarray          # (3,) 位置 m
    orientation: np.ndarray      # (4,) 四元数
    velocity: np.ndarray          # (3,) 速度 m/s
    angular_velocity: np.ndarray  # (3,) 角速度 rad/s
    joint_positions: np.ndarray   # (n_joints,) 关节位置
    joint_velocities: np.ndarray  # (n_joints,) 关节速度
    joint_torques: np.ndarray     # (n_joints,) 关节力矩
    contact_forces: Dict[str, np.ndarray]  # 各接触点力
    sensor_readings: Dict[str, Any]        # 传感器读数
    render_frame: Optional[Any] = None     # 渲染帧


@dataclass
class AGVSimParams:
    """AGV仿真参数 (按等级)"""
    max_load_kg: float
    wheel_radius_m: float
    wheelbase_m: float
    track_width_m: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    max_linear_accel_mps2: float
    max_angular_accel_radps2: float
    motor_torque_constant: float
    motor_inertia: float
    vehicle_mass_kg: float
    vehicle_inertia_z: float
    friction_coefficient: float
    drive_type: str  # 'differential', 'mecanum', 'omni'


# AGV五级仿真参数
AGV_SIM_PARAMS: Dict[str, AGVSimParams] = {
    'S': AGVSimParams(
        max_load_kg=30, wheel_radius_m=0.05, wheelbase_m=0.3, track_width_m=0.25,
        max_linear_speed_mps=0.5, max_angular_speed_radps=1.5,
        max_linear_accel_mps2=0.5, max_angular_accel_radps2=2.0,
        motor_torque_constant=0.5, motor_inertia=0.001,
        vehicle_mass_kg=15, vehicle_inertia_z=0.5,
        friction_coefficient=0.3, drive_type='differential'
    ),
    'M': AGVSimParams(
        max_load_kg=100, wheel_radius_m=0.07, wheelbase_m=0.4, track_width_m=0.35,
        max_linear_speed_mps=1.5, max_angular_speed_radps=2.0,
        max_linear_accel_mps2=1.0, max_angular_accel_radps2=3.0,
        motor_torque_constant=1.0, motor_inertia=0.005,
        vehicle_mass_kg=35, vehicle_inertia_z=2.0,
        friction_coefficient=0.5, drive_type='differential'
    ),
    'L': AGVSimParams(
        max_load_kg=300, wheel_radius_m=0.07, wheelbase_m=0.6, track_width_m=0.5,
        max_linear_speed_mps=2.0, max_angular_speed_radps=1.5,
        max_linear_accel_mps2=1.5, max_angular_accel_radps2=2.0,
        motor_torque_constant=2.0, motor_inertia=0.01,
        vehicle_mass_kg=80, vehicle_inertia_z=8.0,
        friction_coefficient=0.6, drive_type='mecanum'
    ),
    'XL': AGVSimParams(
        max_load_kg=600, wheel_radius_m=0.0825, wheelbase_m=0.7, track_width_m=0.6,
        max_linear_speed_mps=2.5, max_angular_speed_radps=1.2,
        max_linear_accel_mps2=2.0, max_angular_accel_radps2=1.5,
        motor_torque_constant=3.0, motor_inertia=0.02,
        vehicle_mass_kg=150, vehicle_inertia_z=20.0,
        friction_coefficient=0.7, drive_type='mecanum'
    ),
    'XXL': AGVSimParams(
        max_load_kg=1200, wheel_radius_m=0.1, wheelbase_m=0.9, track_width_m=0.8,
        max_linear_speed_mps=3.0, max_angular_speed_radps=1.0,
        max_linear_accel_mps2=2.5, max_angular_accel_radps2=1.0,
        motor_torque_constant=5.0, motor_inertia=0.05,
        vehicle_mass_kg=300, vehicle_inertia_z=50.0,
        friction_coefficient=0.8, drive_type='mecanum'
    ),
}


def get_agv_sim_params(grade: str) -> AGVSimParams:
    """获取AGV指定等级的仿真参数"""
    return AGV_SIM_PARAMS.get(grade, AGV_SIM_PARAMS['M'])


class SimulationInterface:
    """
    统一仿真控制接口
    
    为具身智能大脑提供统一的仿真环境访问接口，
    支持 Gym/Gymnasium、MUJoCo、PyBullet、Gazebo 等多种仿真后端。
    
    五级规格说明:
    - S: 简化运动学模型, 低精度传感器仿真, 适合快速验证
    - M: 标准动力学模型, 基础传感器噪声, 适合开发测试
    - L: 高保真动力学, 完整传感器模型, 适合集成测试
    - XL: 极高保真, 物理精确 + 传感器精确, 适合性能评估
    - XXL: 极致仿真, 全精度 + 数字孪生, 适合部署前验证
    """
    
    def __init__(
        self,
        grade: str = 'M',
        backend: str = 'gym',
        config: Optional[SimulationConfig] = None
    ):
        """
        Args:
            grade: AGV五级等级 (S/M/L/XL/XXL)
            backend: 仿真后端 ('gym', 'mujoco', 'pybullet', 'gazebo', 'none')
            config: 仿真配置
        """
        self.grade = grade
        self.backend = SimulationBackend(backend)
        self.config = config or SimulationConfig(backend=self.backend, grade=SimulationGrade(grade))
        
        # 仿真参数
        self.params = get_agv_sim_params(grade)
        
        # 后端实例
        self._env = None
        self._mujoco_model = None
        self._pybullet_client = None
        self._gazebo_node = None
        
        # 仿真状态
        self._is_running = False
        self._step_count = 0
        self._sim_time = 0.0
        self._last_reset_time = None
        
        # 渲染器
        self._renderer = None
        
        # 初始化后端
        self._init_backend()
    
    def _init_backend(self):
        """根据后端类型初始化仿真环境"""
        if self.backend == SimulationBackend.GYM:
            self._init_gym()
        elif self.backend == SimulationBackend.MUJOCO:
            self._init_mujoco()
        elif self.backend == SimulationBackend.PYBULLET:
            self._init_pybullet()
        elif self.backend == SimulationBackend.GAZEBO:
            self._init_gazebo()
        elif self.backend == SimulationBackend.NONE:
            print("[SimulationInterface] No simulation backend (hardware only)")
    
    def _init_gym(self):
        """初始化Gym环境"""
        try:
            from src.simulation.gym_env import GymAGVEnv
            spec_params = get_agv_sim_params(self.grade)
            # 创建Gymnasium环境
            self._env = GymAGVEnv(
                grade=self.grade,
                dt=self.config.dt,
                enable_rendering=self.config.enable_rendering
            )
            print(f"[SimulationInterface] Gym backend initialized (grade={self.grade})")
        except ImportError as e:
            print(f"[SimulationInterface] Failed to init Gym: {e}. Using mock.")
            self._env = MockGymEnv(self.grade, self.params)
    
    def _init_mujoco(self):
        """初始化MuJoCo仿真"""
        try:
            import mujoco
            self._mujoco_model = mujoco.Model.from_xml_string(self._get_mjcf_xml())
            self._mujoco_data = mujoco.MjData(self._mujoco_model)
            print(f"[SimulationInterface] MuJoCo backend initialized (grade={self.grade})")
        except ImportError:
            print("[SimulationInterface] MuJoCo not installed. Using mock.")
            self._mujoco_model = None
    
    def _init_pybullet(self):
        """初始化PyBullet仿真"""
        try:
            import pybullet as p
            import pybullet_data
            self._pybullet_client = p.connect(p.DIRECT if not self.config.enable_rendering else p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            print(f"[SimulationInterface] PyBullet backend initialized (grade={self.grade})")
        except ImportError:
            print("[SimulationInterface] PyBullet not installed. Using mock.")
            self._pybullet_client = None
    
    def _init_gazebo(self):
        """初始化Gazebo/ROS2仿真"""
        try:
            import rclpy
            rclpy.init()
            self._gazebo_node = rclpy.node.Node('supermodel_gazebo_sim')
            print(f"[SimulationInterface] Gazebo backend initialized (grade={self.grade})")
        except ImportError:
            print("[SimulationInterface] ROS2/Gazebo not installed. Using mock.")
            self._gazebo_node = None
    
    def _get_mjcf_xml(self) -> str:
        """生成MuJoCo模型XML"""
        params = self.params
        return f"""<mujoco model="agv_{self.grade}">
  <option timestep="{self.config.dt}" integrator="RK4"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
    <body name="base" pos="0 0 {params.wheel_radius_m}">
      <freejoint/>
      <geom type="box" size="{0.2} {0.15} {0.05}" rgba=".3 .3 .8 1" mass="{params.vehicle_mass_kg}"/>
      <inertial pos="0 0 0" mass="{params.vehicle_mass_kg}" diaginertia="{params.vehicle_inertia_z} {params.vehicle_inertia_z} {params.vehicle_inertia_z}"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="base" ctrllimited="true" ctrlrange="-1 1" gain="{params.motor_torque_constant}"/>
  </actuator>
</mujoco>"""
    
    def reset(self) -> SimState:
        """重置仿真环境
        
        Returns:
            SimState: 初始仿真状态
        """
        self._step_count = 0
        self._sim_time = 0.0
        self._last_reset_time = time.time()
        
        if self.backend == SimulationBackend.GYM and self._env is not None:
            obs, info = self._env.reset()
            state = self._obs_to_state(obs, info)
        elif self.backend == SimulationBackend.MUJOCO and self._mujoco_data is not None:
            import mujoco
            mujoco.mj_resetData(self._mujoco_model, self._mujoco_data)
            state = self._mujoco_to_state()
        elif self.backend == SimulationBackend.PYBULLET and self._pybullet_client is not None:
            import pybullet as p
            p.resetSimulation()
            state = self._pybullet_to_state()
        elif self.backend == SimulationBackend.GAZEBO and self._gazebo_node is not None:
            state = self._gazebo_reset()
        else:
            state = self._mock_state()
        
        self._is_running = True
        return state
    
    def step(self, action: np.ndarray) -> Tuple[SimState, float, bool, Dict[str, Any]]:
        """执行一步仿真
        
        Args:
            action: 控制动作 (如电机电压/速度命令)
            
        Returns:
            Tuple[SimState, float, bool, Dict]: (新状态, 奖励, 是否结束, 信息字典)
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call reset() first.")
        
        if self.backend == SimulationBackend.GYM and self._env is not None:
            obs, reward, terminated, truncated, info = self._env.step(action)
            state = self._obs_to_state(obs, info)
            done = terminated or truncated
        elif self.backend == SimulationBackend.MUJOCO and self._mujoco_data is not None:
            import mujoco
            self._mujoco_data.ctrl[0] = float(action[0]) if hasattr(action, '__len__') else float(action)
            mujoco.mj_step(self._mujoco_model, self._mujoco_data)
            state = self._mujoco_to_state()
            reward = self._compute_reward(state)
            done = self._check_done(state)
            info = {}
        elif self.backend == SimulationBackend.PYBULLET and self._pybullet_client is not None:
            import pybullet as p
            if hasattr(action, '__len__') and len(action) >= 2:
                p.setJointMotorControlArray(
                    self._pybullet_client, 0,
                    [0, 1], 
                    p.VELOCITY_CONTROL,
                    targetVelocities=[float(action[0]), float(action[1])]
                )
            p.stepSimulation()
            state = self._pybullet_to_state()
            reward = self._compute_reward(state)
            done = self._check_done(state)
            info = {}
        elif self.backend == SimulationBackend.GAZEBO and self._gazebo_node is not None:
            state, reward, done, info = self._gazebo_step(action)
        else:
            state = self._mock_step(action)
            reward = self._compute_reward(state)
            done = self._check_done(state)
            info = {}
        
        self._step_count += 1
        self._sim_time += self.config.dt
        
        return state, reward, done, info
    
    def _compute_reward(self, state: SimState) -> float:
        """计算奖励函数"""
        # 速度奖励
        v = np.linalg.norm(state.velocity)
        speed_reward = v / max(self.params.max_linear_speed_mps, 0.01)
        
        # 能耗惩罚
        energy_penalty = -0.001 * np.sum(np.abs(state.joint_torques))
        
        # 安全惩罚
        safety_penalty = 0.0
        if np.linalg.norm(state.velocity) > self.params.max_linear_speed_mps:
            safety_penalty -= 1.0
        if np.any(np.abs(state.joint_torques) > self.params.motor_torque_constant * 10):
            safety_penalty -= 2.0
        
        # 任务奖励（偏置向目标）
        # 简化: 使用时间惩罚
        time_penalty = -0.01
        
        return speed_reward + energy_penalty + safety_penalty + time_penalty
    
    def _check_done(self, state: SimState) -> bool:
        """检查episode是否结束"""
        # 超时
        if self._sim_time > 60.0:  # 60秒超时
            return True
        # 翻倒检测
        ori = state.orientation
        up = np.array([2*ori[0]*ori[2] + 2*ori[1]*ori[3],
                       2*ori[1]*ori[2] - 2*ori[0]*ori[3],
                       1 - 2*(ori[1]**2 + ori[2]**2)])
        if up[2] < 0.3:  # 倾斜超过72度
            return True
        return False
    
    def _obs_to_state(self, obs, info) -> SimState:
        """Gym观察结果转换为SimState"""
        return SimState(
            timestamp=self._sim_time,
            dt=self.config.dt,
            position=info.get('position', np.zeros(3)),
            orientation=info.get('orientation', np.array([1., 0., 0., 0.])),
            velocity=info.get('velocity', np.zeros(3)),
            angular_velocity=info.get('angular_velocity', np.zeros(3)),
            joint_positions=info.get('joint_positions', np.zeros(2)),
            joint_velocities=info.get('joint_velocities', np.zeros(2)),
            joint_torques=info.get('joint_torques', np.zeros(2)),
            contact_forces=info.get('contact_forces', {}),
            sensor_readings=info.get('sensor_readings', {}),
        )
    
    def _mujoco_to_state(self) -> SimState:
        """MuJoCo数据转换为SimState"""
        import mujoco
        d = self._mujoco_data
        return SimState(
            timestamp=self._sim_time,
            dt=self.config.dt,
            position=d.qpos[:3].copy() if len(d.qpos) >= 3 else np.zeros(3),
            orientation=np.array([1., 0., 0., 0.]) if len(d.qpos) < 4 else np.array([d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]]),
            velocity=d.qvel[:3].copy() if len(d.qvel) >= 3 else np.zeros(3),
            angular_velocity=d.qvel[3:6].copy() if len(d.qvel) >= 6 else np.zeros(3),
            joint_positions=d.qpos[:] if len(d.qpos) > 0 else np.zeros(2),
            joint_velocities=d.qvel[:] if len(d.qvel) > 0 else np.zeros(2),
            joint_torques=d.qfrc_actuator.copy() if hasattr(d, 'qfrc_actuator') else np.zeros(2),
            contact_forces={},
            sensor_readings={},
        )
    
    def _pybullet_to_state(self) -> SimState:
        """PyBullet状态转换为SimState"""
        import pybullet as p
        pos, ori = p.getBasePositionOrientation(0)
        vel = p.getBaseVelocity(0)
        return SimState(
            timestamp=self._sim_time,
            dt=self.config.dt,
            position=np.array(pos),
            orientation=np.array(ori),
            velocity=np.array(vel[0]),
            angular_velocity=np.array(vel[1]),
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(2),
            contact_forces={},
            sensor_readings={},
        )
    
    def _gazebo_reset(self) -> SimState:
        """Gazebo重置"""
        return self._mock_state()
    
    def _gazebo_step(self, action) -> Tuple[SimState, float, bool, Dict]:
        """Gazebo一步仿真"""
        return self._mock_step(action)
    
    def _mock_state(self) -> SimState:
        """生成模拟状态"""
        return SimState(
            timestamp=self._sim_time,
            dt=self.config.dt,
            position=np.array([0.0, 0.0, self.params.wheel_radius_m]),
            orientation=np.array([1., 0., 0., 0.]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(2),
            contact_forces={},
            sensor_readings={},
        )
    
    def _mock_step(self, action) -> SimState:
        """模拟一步仿真"""
        v = float(action[0]) if hasattr(action, '__len__') else float(action)
        # step() will increment _sim_time after this call,
        # so use pre-computed next timestamp
        next_time = self._sim_time + self.config.dt
        return SimState(
            timestamp=next_time,
            dt=self.config.dt,
            position=np.array([self._sim_time * v, 0.0, self.params.wheel_radius_m]),
            orientation=np.array([1., 0., 0., 0.]),
            velocity=np.array([v, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            joint_positions=np.zeros(2),
            joint_velocities=np.array([v, v]),
            joint_torques=np.zeros(2),
            contact_forces={},
            sensor_readings={},
        )
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """渲染当前帧"""
        if self.backend == SimulationBackend.GYM and self._env is not None:
            return self._env.render(mode=mode)
        elif self.backend == SimulationBackend.MUJOCO and self._mujoco_model is not None:
            import mujoco
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._mujoco_model)
            self._renderer.update_scene(self._mujoco_data)
            return self._renderer.read_pixels()
        return None
    
    def close(self):
        """关闭仿真环境"""
        if self.backend == SimulationBackend.PYBULLET and self._pybullet_client is not None:
            import pybullet as p
            p.disconnect(self._pybullet_client)
        elif self.backend == SimulationBackend.GAZEBO and self._gazebo_node is not None:
            self._gazebo_node.destroy_node()
        elif self.backend == SimulationBackend.GYM and self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        
        self._is_running = False
        print(f"[SimulationInterface] Closed (grade={self.grade}, backend={self.backend.value})")
    
    def get_info(self) -> Dict[str, Any]:
        """获取仿真环境信息"""
        return {
            'grade': self.grade,
            'backend': self.backend.value,
            'params': {
                'max_linear_speed': self.params.max_linear_speed_mps,
                'max_angular_speed': self.params.max_angular_speed_radps,
                'vehicle_mass': self.params.vehicle_mass_kg,
                'max_load': self.params.max_load_kg,
                'drive_type': self.params.drive_type,
            },
            'step_count': self._step_count,
            'sim_time': self._sim_time,
            'is_running': self._is_running,
        }
    
    def __enter__(self):
        self.reset()
        return self
    
    def __exit__(self, *args):
        self.close()


class MockGymEnv:
    """Gym环境的Mock实现(当真实环境不可用时)"""
    
    def __init__(self, grade: str, params: AGVSimParams):
        self.grade = grade
        self.params = params
        self._step_count = 0
        self._state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'joint_positions': np.zeros(2),
        }
    
    def reset(self):
        self._step_count = 0
        obs = np.zeros(20)  # 简化的观察空间
        info = {
            'position': np.array([0., 0., self.params.wheel_radius_m]),
            'orientation': np.array([1., 0., 0., 0.]),
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'joint_positions': np.zeros(2),
            'joint_velocities': np.zeros(2),
            'joint_torques': np.zeros(2),
            'contact_forces': {},
            'sensor_readings': {},
        }
        return obs, info
    
    def step(self, action):
        v = float(action[0]) if hasattr(action, '__len__') else float(action)
        dt = 0.01
        self._step_count += 1
        self._state['position'][0] += v * dt
        self._state['velocity'][0] = v
        
        obs = np.zeros(20)
        info = {
            'position': self._state['position'] + np.array([0., 0., self.params.wheel_radius_m]),
            'orientation': np.array([1., 0., 0., 0.]),
            'velocity': self._state['velocity'],
            'angular_velocity': np.zeros(3),
            'joint_positions': np.array([v, v]),
            'joint_velocities': np.zeros(2),
            'joint_torques': np.zeros(2),
            'contact_forces': {},
            'sensor_readings': {},
        }
        reward = 0.1
        done = self._step_count > 1000
        return obs, reward, False, False, info
    
    def render(self, mode='human'):
        return None
    
    def close(self):
        pass


# 仿真五级规格
AGV_SIMULATION_GRADES = {
    'S':  {'backend': 'gym', 'dt': 0.02, 'freq': 50, 'fidelity': 'low', 'sensor_noise': False},
    'M':  {'backend': 'gym', 'dt': 0.01, 'freq': 100, 'fidelity': 'medium', 'sensor_noise': True},
    'L':  {'backend': 'mujoco', 'dt': 0.005, 'freq': 200, 'fidelity': 'high', 'sensor_noise': True},
    'XL': {'backend': 'mujoco', 'dt': 0.002, 'freq': 500, 'fidelity': 'ultra', 'sensor_noise': True},
    'XXL': {'backend': 'gazebo', 'dt': 0.001, 'freq': 1000, 'fidelity': 'digital_twin', 'sensor_noise': True},
}


def get_simulation_spec(grade: str) -> dict:
    """获取AGV指定等级的仿真规格"""
    return AGV_SIMULATION_GRADES.get(grade, AGV_SIMULATION_GRADES['M'])
