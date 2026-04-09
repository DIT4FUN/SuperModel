# SuperModel AGV 五级控制参数指南

> **版本**: v2.08.0
> **更新**: 2026-04-09
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档提供 SuperModel AGV 五级 (S/M/L/XL/XXL) 的控制参数完整参考，包括 PID、阻抗控制、MPC、轨迹跟踪、姿态稳定、力控和安全监控的全部调参指南。

---

## 1. 概述：五级控制架构差异

| 等级 | 控制频率 | 控制架构 | 核心算法 | 实时性 | 力控能力 |
|------|:--------:|:--------:|:--------:|:------:|:--------:|
| **S** | 50Hz | 位置环 | PID | 非实时 | 无 |
| **M** | 100Hz | 位置+速度环 | PID+前馈 | 非实时 | 碰撞检测 |
| **L** | 200Hz | 位置+速度+阻抗 | 阻抗+前馈 | Xenomai | 5Hz力控 |
| **XL** | 500Hz | 全模态闭环 | 阻抗+MPC | RT-PREEMPT | 20Hz力控 |
| **XXL** | 1000Hz | 全模态+MPC | MPC+自适应 | Xenomai+FPGA | 50Hz力控 |

---

## 2. 电机驱动参数

### 2.1 五级电机配置

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **电机类型** | 57步进 | 5.5寸轮毂150W | 5.5寸轮毂150W×2 | 6.5寸轮毂200W×2 | 7.5寸轮毂300W×4 |
| **驱动器** | L298N | ZLAC8015D | ZLAC8015D×2 | Copley 503 | Copley 506×2 |
| **控制接口** | PWM | CANopen | CANopen | CANopen/EtherCAT | EtherCAT |
| **额定电压** | 12V | 24V | 24V | 48V | 48V |
| **额定电流** | 2A | 15A均值/30A峰值 | 15A×2 | 20A×2 | 30A×4 |
| **编码器** | 开环 | 增量式256CPR | 增量式512CPR | 增量式768CPR | 增量式1024CPR |
| **减速比** | 1:30 | 直驱 | 直驱 | 1:10 | 1:10 |

### 2.2 电机 PID 参数

#### S级：简单位置 PID

```python
{
    "name": "position_pid",
    "Kp": 10.0,      # 比例增益
    "Ki": 0.5,        # 积分增益
    "Kd": 2.0,       # 微分增益
    "output_limit": 12.0,   # PWM输出限制 V
    "integral_limit": 5.0,   # 积分饱和
    "derivative_filter": 0.1, # 微分滤波 alpha
    "deadband": 0.02,       # 死区 rad/s
}
```

#### M级：级联 PID（位置-速度）

```python
{
    # 外环：位置环
    "position": {
        "Kp": 15.0,
        "Ki": 1.0,
        "Kd": 3.0,
        "output_limit": 1.5,   # m/s 速度上限
    },
    # 内环：速度环
    "velocity": {
        "Kp": 5.0,
        "Ki": 0.2,
        "Kd": 0.5,
        "output_limit": 15.0,  # A 电流限制
        "derivative_filter": 0.05,
    },
    # 前馈
    "feedforward": {
        "voltage_ff": 0.8,     # 速度前馈系数
        "torque_ff": 0.3,     # 力矩前馈系数
    },
}
```

#### L级：带观测器的 PID

```python
{
    "position": {
        "Kp": 20.0,
        "Ki": 2.0,
        "Kd": 5.0,
        "output_limit": 2.0,
        "anti_windup": "back_calculation",
        "Kd_filter": 0.02,
    },
    "velocity": {
        "Kp": 8.0,
        "Ki": 0.5,
        "Kd": 1.0,
        "output_limit": 20.0,
        "Kd_filter": 0.01,
    },
    # 扰动观测器
    "disturbance_observer": {
        "enabled": True,
        "cutoff_freq": 50.0,   # Hz
        "filter_order": 2,
    },
}
```

#### XL级：自适应 PID

```python
{
    "position": {
        "Kp_base": 25.0,
        "Ki_base": 3.0,
        "Kd_base": 6.0,
        "adaptive_gain_scheduling": {
            "enabled": True,
            "speed_threshold_high": 1.0,   # m/s
            "speed_threshold_low": 0.2,    # m/s
            "Kp_at_high_speed": 15.0,     # 高速降增益
            "Kp_at_low_speed": 30.0,      # 低速增增益
        },
    },
    "velocity": {
        "Kp_base": 10.0,
        "Ki_base": 0.8,
        "Kd_base": 1.5,
    },
    # 摩擦补偿
    "friction_compensation": {
        "enabled": True,
        "Coulomb_friction": 2.0,   # N·m
        "viscous_friction": 0.5,     # N·m/(rad/s)
        "Stribeck_velocity": 0.1,    # rad/s
    },
}
```

#### XXL级：自适应+MPC前馈

```python
{
    "position": {
        "Kp": 30.0,
        "Ki": 5.0,
        "Kd": 8.0,
        "output_limit": 3.0,
        "gain_scheduling": "load_dependent",
    },
    "velocity": {
        "Kp": 12.0,
        "Ki": 1.0,
        "Kd": 2.0,
    },
    # 多参数自适应
    "adaptive_control": {
        "enabled": True,
        "algorithm": "model_reference_adaptive",
        "adaptation_rate": 0.01,
        "reference_model": "2nd_order_butterworth",
        "bandwidth": 10.0,   # Hz
    },
    # MPC 前馈
    "mpc_feedforward": {
        "horizon": 20,       # 预测步数
        "weight_velocity": 0.1,
        "weight_acceleration": 0.05,
        "weight_control": 0.01,
    },
}
```

---

## 3. AGV 运动控制参数

### 3.1 差速驱动参数

#### M级：差速驱动 PID

```python
class AGVMotionParams:
    """M级 AGV 运动参数"""
    # 运动学
    wheelbase: float = 0.4          # m
    track_width: float = 0.35      # m
    wheel_radius: float = 0.07      # m  (140mm/2)
    
    # 速度限制
    max_linear_speed: float = 1.5   # m/s
    max_angular_speed: float = 3.0  # rad/s
    max_linear_accel: float = 1.0   # m/s^2
    max_angular_accel: float = 5.0  # rad/s^2
    
    # 差速 PID
    linear_pid = {"Kp": 8.0, "Ki": 0.5, "Kd": 1.0}
    angular_pid = {"Kp": 5.0, "Ki": 0.3, "Kd": 0.8}
    
    # 安全
    emergency_stop_distance: float = 0.3   # m
    collision_detection_threshold: float = 50.0  # N
```

### 3.2 全向移动参数 (麦克纳姆轮)

#### L级：全向运动控制

```python
class OmnidirectionalParams:
    """L级 全向AGV参数"""
    # 麦克纳姆轮运动学
    wheel_count: int = 4
    wheel_radius: float = 0.07     # m
    mecanum_angle: float = 45.0   # 滚轮安装角 °
    
    # 运动限制
    max_strafing_speed: float = 1.0   # m/s (横向)
    max_linear_speed: float = 2.0     # m/s
    max_angular_speed: float = 2.5    # rad/s
    
    # 轨迹跟踪
    trajectory_tracker = {
        "lookahead_distance": 0.3,     # m
        "lookahead_time": 0.5,         # s
        "kpheading": 3.0,
        "kpvelocity": 5.0,
        "kpvtheta": 4.0,
    }
```

---

## 4. 轨迹跟踪参数

### 4.1 Pure Pursuit（五级通用）

```python
pure_pursuit_params = {
    "S": {
        "lookahead_distance": 0.15,   # m
        "lookahead_gain": 2.0,
        "min_lookahead": 0.1,
        "max_lookahead": 0.3,
        "switch_speed_threshold": 0.5,  # m/s
    },
    "M": {
        "lookahead_distance": 0.25,
        "lookahead_gain": 2.5,
        "min_lookahead": 0.15,
        "max_lookahead": 0.5,
        "switch_speed_threshold": 1.0,
    },
    "L": {
        "lookahead_distance": 0.4,
        "lookahead_gain": 3.0,
        "min_lookahead": 0.2,
        "max_lookahead": 0.8,
        "switch_speed_threshold": 1.5,
        "adaptive_lookahead": True,
    },
    "XL": {
        "lookahead_distance": 0.5,
        "lookahead_gain": 3.5,
        "min_lookahead": 0.3,
        "max_lookahead": 1.0,
        "switch_speed_threshold": 2.0,
        "curvature_feedforward": True,
    },
    "XXL": {
        "lookahead_distance": 0.8,
        "lookahead_gain": 4.0,
        "min_lookahead": 0.4,
        "max_lookahead": 1.5,
        "switch_speed_threshold": 2.5,
        "curvature_feedforward": True,
        "velocity_dependent_lookahead": True,
    },
}
```

### 4.2 Stanley 方法（五级通用）

```python
stanley_params = {
    "S": {
        "k_soft": 1.0,           # 软化系数
        "k_gain": 2.0,           # 横摆角增益
        "velocity_filter": 0.1,   # 速度滤波
    },
    "M": {
        "k_soft": 2.0,
        "k_gain": 3.0,
        "velocity_filter": 0.08,
        "heading_filter": 0.05,
    },
    "L": {
        "k_soft": 3.0,
        "k_gain": 4.0,
        "velocity_filter": 0.05,
        "heading_filter": 0.03,
        "cross_track_error_gain": 1.5,
    },
    "XL": {
        "k_soft": 4.0,
        "k_gain": 5.0,
        "velocity_filter": 0.03,
        "heading_filter": 0.02,
        "cross_track_error_gain": 2.0,
        "yaw_rate_feedforward": True,
    },
    "XXL": {
        "k_soft": 5.0,
        "k_gain": 6.0,
        "velocity_filter": 0.02,
        "heading_filter": 0.01,
        "cross_track_error_gain": 2.5,
        "yaw_rate_feedforward": True,
        "load_adaptation": True,
    },
}
```

---

## 5. 阻抗/导纳控制参数

### 5.1 五级阻抗控制参数

```python
impedance_params = {
    "S": {
        # 无力控，仅位置控制
        "enabled": False,
    },
    "M": {
        "enabled": False,
        "collision_detection_only": True,
        "collision_threshold": 30.0,  # N
    },
    "L": {
        "enabled": True,
        "type": "impedance",         # 阻抗控制
        "M_mass": 2.0,              # 等效质量 kg
        "D_damping": 10.0,           # 等效阻尼 N·s/m
        "K_stiffness": 50.0,         # 等效刚度 N/m
        "force_bandwidth": 5.0,      # Hz
        "position_limit": 0.05,      # m
        "force_limit": 20.0,         # N
    },
    "XL": {
        "enabled": True,
        "type": "impedance",
        "M_mass": 1.5,
        "D_damping": 15.0,
        "K_stiffness": 100.0,
        "force_bandwidth": 20.0,     # Hz
        "position_limit": 0.02,
        "force_limit": 50.0,
        "adaptive_stiffness": True,
        "contact_detection_threshold": 2.0,  # N
    },
    "XXL": {
        "enabled": True,
        "type": "impedance_plus_mpc",
        "M_mass": 1.0,
        "D_damping": 20.0,
        "K_stiffness": 200.0,
        "force_bandwidth": 50.0,     # Hz
        "position_limit": 0.01,
        "force_limit": 100.0,
        "adaptive_stiffness": True,
        "task_adaptation": True,
        "learning_based_gain": True,  # 学习型增益
        "contact_detection_threshold": 1.0,
    },
}
```

### 5.2 导纳控制参数

```python
admittance_params = {
    "L": {
        "M_a": 5.0,          # 虚拟惯量
        "D_a": 15.0,         # 虚拟阻尼
        "K_a": 30.0,         # 虚拟刚度
        "input_filter_bw": 3.0,   # Hz
        "max_velocity": 0.2,     # m/s
    },
    "XL": {
        "M_a": 3.0,
        "D_a": 20.0,
        "K_a": 50.0,
        "input_filter_bw": 10.0,
        "max_velocity": 0.3,
        "force_bias_compensation": True,
    },
    "XXL": {
        "M_a": 2.0,
        "D_a": 30.0,
        "K_a": 80.0,
        "input_filter_bw": 25.0,
        "max_velocity": 0.5,
        "force_bias_compensation": True,
        "environment_impedance_estimation": True,
        "adaptive_admittance": True,
    },
}
```

---

## 6. MPC 参数（五级按复杂度）

### 6.1 运动学 MPC（L级起）

```python
mpc_params_kinematic = {
    "L": {
        "horizon": 10,               # 预测步数
        "dt": 0.005,                # 步长 s (200Hz)
        "max_iterations": 100,
        "weights": {
            "state": 1.0,
            "control": 0.1,
            "control_rate": 0.05,
            "terminal_state": 5.0,
        },
        "constraints": {
            "max_linear_velocity": 2.0,
            "max_angular_velocity": 2.5,
            "max_linear_acceleration": 1.0,
            "max_angular_acceleration": 3.0,
        },
    },
    "XL": {
        "horizon": 15,
        "dt": 0.002,                # s (500Hz)
        "max_iterations": 200,
        "weights": {
            "state": 1.0,
            "control": 0.05,
            "control_rate": 0.02,
            "terminal_state": 10.0,
        },
        "constraints": {
            "max_linear_velocity": 2.5,
            "max_angular_velocity": 3.0,
            "max_linear_acceleration": 2.0,
            "max_angular_acceleration": 5.0,
        },
        "warm_start": True,
    },
    "XXL": {
        "horizon": 20,
        "dt": 0.001,                # s (1000Hz)
        "max_iterations": 500,
        "weights": {
            "state": 1.0,
            "control": 0.01,
            "control_rate": 0.005,
            "terminal_state": 15.0,
            "smoothness": 0.02,
        },
        "constraints": {
            "max_linear_velocity": 3.0,
            "max_angular_velocity": 3.5,
            "max_linear_acceleration": 3.0,
            "max_angular_acceleration": 8.0,
        },
        "warm_start": True,
        "state_estimation_in_loop": True,
        "robust_optimization": True,
    },
}
```

### 6.2 动力学 MPC（XL级起）

```python
mpc_params_dynamic = {
    "XL": {
        "model": "dynamic_bicycle",  # 自行车模型
        "horizon": 20,
        "dt": 0.002,
        "mass": 150.0,              # kg
        "Iz": 50.0,                 # 转动惯量 kg·m²
        "Cf": 50000.0,              # 前轮侧偏刚度 N/rad
        "Cr": 50000.0,              # 后轮侧偏刚度 N/rad
        "weights": {
            "deviation": 1.0,
            "steering": 0.5,
            "acceleration": 0.3,
            "jerk": 0.1,
        },
    },
    "XXL": {
        "model": "dynamic_full",
        "horizon": 30,
        "dt": 0.001,
        "mass": 300.0,
        "Iz": 100.0,
        "Cf": 100000.0,
        "Cr": 100000.0,
        "weights": {
            "deviation": 1.0,
            "steering": 0.3,
            "acceleration": 0.2,
            "jerk": 0.05,
            "fuel_efficiency": 0.1,
        },
        "multi_fidelity_model": True,
        "online_parameter_update": True,
    },
}
```

---

## 7. 姿态稳定参数

### 7.1 五级姿态控制参数

```python
attitude_params = {
    "S": {
        "control_frequency": 50,     # Hz
        "roll_pid": {"Kp": 5.0, "Ki": 0.0, "Kd": 1.0},
        "pitch_pid": {"Kp": 5.0, "Ki": 0.0, "Kd": 1.0},
        "yaw_pid": {"Kp": 2.0, "Ki": 0.0, "Kd": 0.5},
        "max_tilt_angle": 15.0,     # °
    },
    "M": {
        "control_frequency": 100,
        "roll_pid": {"Kp": 8.0, "Ki": 0.5, "Kd": 2.0},
        "pitch_pid": {"Kp": 8.0, "Ki": 0.5, "Kd": 2.0},
        "yaw_pid": {"Kp": 3.0, "Ki": 0.2, "Kd": 0.8},
        "max_tilt_angle": 10.0,
        "gyro_qualification": True,
    },
    "L": {
        "control_frequency": 200,
        "roll_pid": {"Kp": 10.0, "Ki": 1.0, "Kd": 3.0},
        "pitch_pid": {"Kp": 10.0, "Ki": 1.0, "Kd": 3.0},
        "yaw_pid": {"Kp": 4.0, "Ki": 0.5, "Kd": 1.0},
        "max_tilt_angle": 5.0,
        "feedforward_acceleration": True,
        "disturbance_observer": True,
    },
    "XL": {
        "control_frequency": 500,
        "roll_pid": {"Kp": 15.0, "Ki": 2.0, "Kd": 4.0},
        "pitch_pid": {"Kp": 15.0, "Ki": 2.0, "Kd": 4.0},
        "yaw_pid": {"Kp": 5.0, "Ki": 1.0, "Kd": 1.5},
        "max_tilt_angle": 3.0,
        "feedforward_acceleration": True,
        "disturbance_observer": True,
        "adaptive_gain": True,
        "load_compensation": True,
    },
    "XXL": {
        "control_frequency": 1000,
        "roll_pid": {"Kp": 20.0, "Ki": 3.0, "Kd": 5.0},
        "pitch_pid": {"Kp": 20.0, "Ki": 3.0, "Kd": 5.0},
        "yaw_pid": {"Kp": 6.0, "Ki": 1.5, "Kd": 2.0},
        "max_tilt_angle": 1.5,
        "feedforward_acceleration": True,
        "disturbance_observer": True,
        "adaptive_gain": True,
        "load_compensation": True,
        "predictive_compensation": True,
        "redundant_imu_fusion": True,
    },
}
```

---

## 8. 安全监控参数

### 8.1 五级安全参数

```python
safety_params = {
    "S": {
        "control_frequency": 50,
        "emergency_stop_distance": 0.5,   # m
        "velocity_limit": 0.5,              # m/s
        "tilt_limit": 15.0,                # °
        "collision_force_threshold": 100.0,  # N
        "watchdog_timeout": 0.5,             # s
        "redundancy": 1,                   # 单通道
    },
    "M": {
        "control_frequency": 100,
        "emergency_stop_distance": 0.3,
        "velocity_limit": 1.5,
        "tilt_limit": 10.0,
        "collision_force_threshold": 50.0,
        "watchdog_timeout": 0.2,
        "redundancy": 1,
        "software_limits": True,
        "boundary_detection": True,
    },
    "L": {
        "control_frequency": 200,
        "emergency_stop_distance": 0.2,
        "velocity_limit": 2.0,
        "tilt_limit": 5.0,
        "collision_force_threshold": 30.0,
        "watchdog_timeout": 0.1,
        "redundancy": 2,
        "software_limits": True,
        "boundary_detection": True,
        "predictive_collision": True,
        "safe_stop_ramp": 0.5,              # m/s²
    },
    "XL": {
        "control_frequency": 500,
        "emergency_stop_distance": 0.1,
        "velocity_limit": 2.5,
        "tilt_limit": 3.0,
        "collision_force_threshold": 20.0,
        "watchdog_timeout": 0.05,
        "redundancy": 3,
        "software_limits": True,
        "boundary_detection": True,
        "predictive_collision": True,
        "safe_stop_ramp": 1.0,
        "redundant_imu_check": True,
        "safe_torque_off": True,
    },
    "XXL": {
        "control_frequency": 1000,
        "emergency_stop_distance": 0.05,
        "velocity_limit": 3.0,
        "tilt_limit": 1.5,
        "collision_force_threshold": 10.0,
        "watchdog_timeout": 0.02,
        "redundancy": 4,
        "software_limits": True,
        "boundary_detection": True,
        "predictive_collision": True,
        "safe_stop_ramp": 2.0,
        "redundant_imu_check": True,
        "safe_torque_off": True,
        "plausibility_checks": True,
        "cross_verification": True,
        "self_diagnostics": True,
    },
}
```

---

## 9. 触觉/力觉/IMU 控制参数

### 9.1 触觉伺服参数

```python
tactile_control_params = {
    "S": {"enabled": False},
    "M": {
        "enabled": True,
        "slip_detection_threshold": 0.3,
        "reaction_force": 2.0,        # N
        "grasp_force_target": 5.0,     # N
        "response_time": 0.1,          # s
    },
    "L": {
        "enabled": True,
        "slip_detection_threshold": 0.2,
        "reaction_force": 5.0,
        "grasp_force_target": 10.0,
        "response_time": 0.05,
        "adaptive_threshold": True,
    },
    "XL": {
        "enabled": True,
        "slip_detection_threshold": 0.15,
        "reaction_force": 10.0,
        "grasp_force_target": 20.0,
        "response_time": 0.02,
        "adaptive_threshold": True,
        "predictive_slip": True,
    },
    "XXL": {
        "enabled": True,
        "slip_detection_threshold": 0.1,
        "reaction_force": 20.0,
        "grasp_force_target": 50.0,
        "response_time": 0.01,
        "adaptive_threshold": True,
        "predictive_slip": True,
        "learning_based_detection": True,
        "cross_modal_slip_prediction": True,
    },
}
```

### 9.2 力控参数

```python
force_control_params = {
    "S": {"enabled": False},
    "M": {
        "enabled": True,
        "collision_detection_only": True,
        "collision_threshold": 30.0,   # N
        "impact_force_limit": 50.0,      # N
    },
    "L": {
        "enabled": True,
        "force_control_bandwidth": 5.0,   # Hz
        "hybrid_control": True,
        "force_limit": 50.0,                # N
        "impedance_control": True,
        "cartesian_impedance": {
            " translational_stiffness": 500.0,  # N/m
            "rotational_stiffness": 50.0,        # Nm/rad
        },
    },
    "XL": {
        "enabled": True,
        "force_control_bandwidth": 20.0,
        "force_limit": 100.0,
        "cartesian_impedance": {
            "translational_stiffness": 1000.0,
            "rotational_stiffness": 100.0,
        },
        "adaptive_impedance": True,
        "environment_classification": True,
    },
    "XXL": {
        "enabled": True,
        "force_control_bandwidth": 50.0,
        "force_limit": 200.0,
        "cartesian_impedance": {
            "translational_stiffness": 2000.0,
            "rotational_stiffness": 200.0,
        },
        "adaptive_impedance": True,
        "environment_classification": True,
        "learning_based_impedance": True,
        "task_context_adaptation": True,
    },
}
```

### 9.3 IMU 姿态估计参数

```python
imu_control_params = {
    "S": {
        "algorithm": "complementary",
        "alpha": 0.96,             # 陀螺仪权重
        "accel_filter": "low_pass",
        "accel_cutoff": 5.0,        # Hz
    },
    "M": {
        "algorithm": "madgwick",
        "beta": 0.1,
        "sample_rate": 200.0,
        "mag_fusion": False,
        "accel_filter": "low_pass",
        "accel_cutoff": 10.0,
    },
    "L": {
        "algorithm": "madgwick",
        "beta": 0.05,
        "sample_rate": 500.0,
        "mag_fusion": True,
        "mag_declination": 0.0,     # ° 磁偏角
        "accel_filter": "kalman",
    },
    "XL": {
        "algorithm": "ekf",
        "sample_rate": 1000.0,
        "process_noise": {"pos": 0.01, "vel": 0.1, "att": 0.001},
        "measurement_noise": {"accel": 0.1, "gyro": 0.01, "mag": 0.05},
        "mag_fusion": True,
        "external_pose_correction": True,
    },
    "XXL": {
        "algorithm": "ekf_multi_imu",
        "sample_rate": 2000.0,
        "process_noise": {"pos": 0.001, "vel": 0.01, "att": 0.0001},
        "measurement_noise": {"accel": 0.01, "gyro": 0.001, "mag": 0.005},
        "mag_fusion": True,
        "external_pose_correction": True,
        "redundant_sensor_fusion": True,
        "online_calibration": True,
    },
}
```

---

## 10. 避障参数

### 10.1 DWA（动态窗口法）参数

```python
dwa_params = {
    "S": {
        "max_speed": 0.5,
        "max_accel": 0.5,
        "max_angular_speed": 2.0,
        "angular_accel": 3.0,
        "prediction_horizon": 1.0,    # s
        "heading_weight": 0.7,
        "velocity_weight": 0.2,
        "obstacle_weight": 0.1,
        "resolution_joystick": 0.05,
    },
    "M": {
        "max_speed": 1.5,
        "max_accel": 1.0,
        "max_angular_speed": 3.0,
        "angular_accel": 5.0,
        "prediction_horizon": 1.5,
        "heading_weight": 0.6,
        "velocity_weight": 0.25,
        "obstacle_weight": 0.15,
        "resolution_joystick": 0.08,
    },
    "L": {
        "max_speed": 2.0,
        "max_accel": 1.5,
        "max_angular_speed": 3.5,
        "angular_accel": 6.0,
        "prediction_horizon": 2.0,
        "heading_weight": 0.5,
        "velocity_weight": 0.3,
        "obstacle_weight": 0.2,
        "resolution_joystick": 0.1,
        "trajectory_cost": True,
    },
    "XL": {
        "max_speed": 2.5,
        "max_accel": 2.0,
        "max_angular_speed": 4.0,
        "angular_accel": 8.0,
        "prediction_horizon": 2.5,
        "heading_weight": 0.4,
        "velocity_weight": 0.35,
        "obstacle_weight": 0.25,
        "resolution_joystick": 0.12,
        "trajectory_cost": True,
        "dynamic_obstacle_prediction": True,
    },
    "XXL": {
        "max_speed": 3.0,
        "max_accel": 3.0,
        "max_angular_speed": 4.5,
        "angular_accel": 10.0,
        "prediction_horizon": 3.0,
        "heading_weight": 0.35,
        "velocity_weight": 0.4,
        "obstacle_weight": 0.25,
        "resolution_joystick": 0.15,
        "trajectory_cost": True,
        "dynamic_obstacle_prediction": True,
        "multi_objective_optimization": True,
    },
}
```

---

## 11. 自动调参（Autotune）参数

```python
autotune_params = {
    "M": {
        "enabled": True,
        "method": "ziegler_nichols",
        "trial_duration": 10.0,     # s
        "oscillation_threshold": 0.01,
        "apply_results": True,
        "safety_margin": 0.8,
    },
    "L": {
        "enabled": True,
        "method": "relay_feedback",
        "trial_duration": 15.0,
        "oscillation_cycles": 5,
        "apply_results": True,
        "safety_margin": 0.8,
        "persist_results": True,
    },
    "XL": {
        "enabled": True,
        "method": "model_based",
        "identification_signal": "chirps",
        "trial_duration": 20.0,
        "frequency_range": [0.1, 10.0],  # Hz
        "apply_results": True,
        "safety_margin": 0.85,
        "persist_results": True,
        "online_adaptation": True,
    },
    "XXL": {
        "enabled": True,
        "method": "adaptive_model_based",
        "identification_signal": "multi_sine",
        "trial_duration": 30.0,
        "frequency_range": [0.01, 50.0],
        "apply_results": True,
        "safety_margin": 0.9,
        "persist_results": True,
        "online_adaptation": True,
        "multi_objective_tuning": True,
        "robustness_constraint": True,
    },
}
```

---

## 12. 参数速查表

### 控制频率速查

| 等级 | 控制周期 | 通信周期 | 传感器融合周期 | 规划周期 |
|------|:--------:|:--------:|:-------------:|:--------:|
| S | 20ms | 50ms | 50ms | 100ms |
| M | 10ms | 20ms | 20ms | 50ms |
| L | 5ms | 10ms | 10ms | 20ms |
| XL | 2ms | 5ms | 5ms | 10ms |
| XXL | 1ms | 2ms | 2ms | 5ms |

### 响应时间速查

| 等级 | 电机响应 | 碰撞检测 | 姿态稳定 | 避障响应 |
|------|:--------:|:--------:|:--------:|:--------:|
| S | 40ms | >100ms | <500ms | >200ms |
| M | 15ms | <50ms | <200ms | <100ms |
| L | 5ms | <20ms | <100ms | <50ms |
| XL | 2ms | <10ms | <50ms | <20ms |
| XXL | 1ms | <5ms | <20ms | <10ms |

### 定位精度与控制精度

| 等级 | 定位精度 | 重复定位精度 | 速度控制精度 | 姿态控制精度 |
|------|:--------:|:-----------:|:-----------:|:-----------:|
| S | ±10mm | ±5mm | ±0.05m/s | ±5° |
| M | ±5mm | ±2mm | ±0.02m/s | ±2° |
| L | ±3mm | ±1mm | ±0.01m/s | ±0.5° |
| XL | ±1mm | ±0.5mm | ±0.005m/s | ±0.2° |
| XXL | ±0.5mm | ±0.1mm | ±0.001m/s | ±0.05° |
