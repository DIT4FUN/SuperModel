#!/usr/bin/env python3
"""
SuperModel AGV URDF模型生成器
=============================
支持2轮和4轮差速驱动AGV，配置真实的硬件参数

硬件配置:
- 电机: 5.5寸轮毂电机 (140mm, 24V/150W/15Nm)
- 激光雷达: 镭神智能 N10P (360°, 25m, TOF)
- IMU: ETT10A-PW (6轴, 防水, IP67)
"""

import numpy as np
from typing import Optional, Dict, Any


# ============================================================================
# 5.5寸轮毂电机物理参数
# ============================================================================

# 5.5" = 139.7mm ≈ 140mm
WHEEL_DIAMETER_55_INCH = 0.14  # 140mm in meters
WHEEL_RADIUS_55_INCH = WHEEL_DIAMETER_55_INCH / 2  # 0.07m

# 电机规格参数
MOTOR_55_SPECS = {
    'voltage': 24,           # V
    'power': 150,            # W
    'rated_torque': 15,     # Nm
    'rated_speed': 400,     # RPM
    'no_load_speed': 500,    # RPM
    'rated_current': 8,     # A
    'efficiency': 85,        # %
    'weight': 1.5,          # kg
}

# 轮胎规格
TIRE_SPECS = {
    'diameter': 140,        # mm
    'width': 50,            # mm
    'material': 'polyurethane',
    'color': 'black',
    'load_capacity': 80,    # kg per wheel
}


# ============================================================================
# 镭神智能 N10P 激光雷达参数
# ============================================================================

LIDAR_N10P_SPECS = {
    'model': 'LSLIDAR N10P',
    'brand': 'Leishen Intelligent (镭神智能)',
    'principle': 'TOF (Time of Flight)',
    'scan_angle': 360,          # degrees
    'max_range': 25,           # meters
    'accuracy': 1.5,           # cm (±1.5cm)
    'ranging_frequency': 4500,  # Hz (4.5kHz)
    'scan_frequency': 10,       # Hz (configurable)
    'angular_resolution': 0.3,  # degrees
    'interface': 'UART (3.3V) / Ethernet',
    'voltage': '9-30V DC',
    'power': 8,             # W (typical)
    'operating_temp': '-10°C ~ 60°C',
    'ip_rating': 'IP65',
    'dimensions': 'φ60mm × H86.5mm',
    'weight': 240,           # g
}


# ============================================================================
# ETT10A-PW IMU 参数
# ============================================================================

IMU_ETT10A_PW_SPECS = {
    'model': 'ETT10A-PW',
    'type': '6-axis IMU (防水型)',
    'accelerometer': '3-axis MEMS',
    'gyroscope': '3-axis MEMS',
    'accel_range': '±16g (可选)',
    'gyro_range': '±2000°/s (可选)',
    'interface': 'RS485 / CAN (可选)',
    'voltage': '9-36V DC',
    'power': '<1W',
    'ip_rating': 'IP67',
    'connector': 'M8 6P 防水接头',
    'cable_length': '1.2m (屏蔽线)',
    'operating_temp': '-40°C ~ 85°C',
    'dimensions': '40 × 40 × 25mm',
    'weight': 50,           # g (含线缆)
}


# ============================================================================
# 奥比中光 Astra Pro Plus 深度相机
# ============================================================================

ASTRA_PRO_PLUS_SPECS = {
    'model': 'Astra Pro Plus',
    'brand': 'Orbbec (奥比中光)',
    'depth_technology': 'Structured Light (单目结构光)',
    'depth_resolution': '640 × 480 (VGA)',
    'rgb_resolution': '1280 × 960',
    'depth_range': '0.4m - 8m',
    'depth_fov': '60° H × 49.5° V',
    'rgb_framerate': '30fps',
    'depth_framerate': '30fps',
    'interface': 'USB 2.0',
    'laser_projection': 'Near-IR (近红外)',
    'power': '<2.5W',
    'dimensions': '165 × 40 × 30mm',
    'weight': 200,           # g
}


# ============================================================================
# 奥比中光 C100 / C70 RGB相机
# ============================================================================

ORBBEC_C100_SPECS = {
    'model': 'C100',
    'brand': 'Orbbec (奥比中光)',
    'type': 'RGB USB相机',
    'resolution': '1080P (1920×1080)',
    'fov': 'H112° / V80°',
    'framerate': '30fps',
    'sensor': 'CMOS',
    'exposure': '全局曝光',
    'interface': 'ZH 1.5-4PIN 转 USB 2.0',
    'cable_length': '150cm',
    'dimensions': '47 × 38 × 22.7mm',
    'weight': 40,           # g
}

ORBBEC_C70_SPECS = {
    'model': 'C70',
    'brand': 'Orbbec (奥比中光)',
    'type': 'RGB USB相机 (金属外壳)',
    'resolution': '720P (1280×720)',
    'fov': 'H85° / V47°',
    'framerate': '30fps',
    'sensor': 'CMOS',
    'exposure': '全局曝光',
    'interface': 'ZH 1.5-4PIN 转 USB 2.0',
    'cable_length': '150cm',
    'dimensions': '47 × 38 × 28.5mm',
    'weight': 50,           # g
}


# ============================================================================
# ESUN 2.5寸静音避震万向轮 (从动轮)
# ============================================================================

# ESUNcaster JQR25310-80A specs
CASTER_ESUN_25_specS = {
    'model': 'ESUN JQR25310-80A',
    'type': '静音避震万向轮',
    'wheel_diameter_inch': 2.5,     # 寸
    'wheel_diameter_mm': 63.5,      # mm (2.5 * 25.4)
    'wheel_radius': 0.03175,        # m
    'material': '聚氨酯 (PU 80A)',
    'load_capacity': 135,           # kg per wheel
    'shock_stroke': 0.010,         # 减震行程 10mm
    'overall_height': 0.106,        # 总高度 106mm (wheel bottom to mount top)
    'bracket_height': 0.07425,      # 支架高度 = overall_height - wheel_radius
    'rotation': '360°',             # 全向旋转
}

# 简写
CASTER_ESUN = CASTER_ESUN_25_specS


# ============================================================================
# AGV五级规格参数 (基于5.5寸轮毂电机)
# ============================================================================

GRADE_CONFIGS = {
    'S': {
        'description': '小型AGV (30kg负载)',
        'wheel_config': '2轮',  # 2轮差速
        'wheel_diameter': 0.10,     # 4寸等效
        'wheel_width': 0.04,
        'motor_spec': '57步进',
        'body_length': 0.4,
        'body_width': 0.3,
        'body_height': 0.12,
        'mass': 15,               # kg (自重)
        'payload': 30,            # kg (负载)
        'max_speed': 1.0,         # m/s
        'rated_torque': 5,        # Nm
        'track_width': 0.25,
    },
    'M': {
        'description': '中型AGV (100kg负载)',
        'wheel_config': '2轮',
        'wheel_diameter': WHEEL_DIAMETER_55_INCH,  # 5.5寸
        'wheel_width': 0.05,
        'motor_spec': '5.5寸轮毂150W',
        'body_length': 0.6,
        'body_width': 0.4,
        'body_height': 0.15,
        'mass': 35,
        'payload': 100,
        'max_speed': 1.5,
        'rated_torque': 15,
        'track_width': 0.35,
    },
    'L': {
        'description': '大型AGV (300kg负载)',
        'wheel_config': '4轮',
        'wheel_diameter': WHEEL_DIAMETER_55_INCH,
        'wheel_width': 0.05,
        'motor_spec': '5.5寸轮毂150W x2',
        'body_length': 0.8,
        'body_width': 0.6,
        'body_height': 0.2,
        'mass': 80,
        'payload': 300,
        'max_speed': 1.2,
        'rated_torque': 30,       # 2轮x15Nm
        'track_width': 0.5,
    },
    'XL': {
        'description': '超大型AGV (600kg负载)',
        'wheel_config': '4轮',
        'wheel_diameter': WHEEL_DIAMETER_55_INCH + 0.02,  # 6.5寸
        'wheel_width': 0.06,
        'motor_spec': '6.5寸轮毂200W x2',
        'body_length': 1.0,
        'body_width': 0.8,
        'body_height': 0.25,
        'mass': 150,
        'payload': 600,
        'max_speed': 1.0,
        'rated_torque': 50,
        'track_width': 0.7,
    },
    'XXL': {
        'description': '重型AGV (1200kg负载)',
        'wheel_config': '4轮',
        'wheel_diameter': WHEEL_DIAMETER_55_INCH + 0.04,  # 7.5寸
        'wheel_width': 0.08,
        'motor_spec': '7.5寸轮毂300W x4',
        'body_length': 1.2,
        'body_width': 1.0,
        'body_height': 0.3,
        'mass': 300,
        'payload': 1200,
        'max_speed': 0.8,
        'rated_torque': 100,
        'track_width': 0.9,
    },
}


def get_wheel_inertia(wheel_radius: float, wheel_width: float, wheel_mass: float) -> float:
    """计算轮子转动惯量 (cylinder about central axis)
    
    I = (1/2) * m * r^2
    """
    return 0.5 * wheel_mass * wheel_radius ** 2


def get_body_inertia(length: float, width: float, height: float, mass: float) -> Dict[str, float]:
    """计算车体转动惯量 (box about center)
    
    Ixx = (1/12) * m * (w^2 + h^2)
    Iyy = (1/12) * m * (l^2 + h^2)
    Izz = (1/12) * m * (l^2 + w^2)
    """
    ixx = (1/12) * mass * (width**2 + height**2)
    iyy = (1/12) * mass * (length**2 + height**2)
    izz = (1/12) * mass * (length**2 + width**2)
    return {'ixx': ixx, 'iyy': iyy, 'izz': izz, 'ixy': 0, 'ixz': 0, 'iyz': 0}


# ============================================================================
# AGV URDF模板 (详细版)
# ============================================================================

# 2轮差速驱动AGV URDF
AGV_2W_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="agv_2w_{grade}">

  <!-- ============================================================
       AGV 车体 (base_link)
       ============================================================ -->
  <link name="base_link">
    <inertial>
      <origin xyz="{com_x} {com_y} {com_z}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {body_height_2}" rpy="0 0 0"/>
      <geometry>
        <box size="{body_length} {body_width} {body_height}"/>
      </geometry>
      <material name="body_color">
        <color rgba="{body_color}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {body_height_2}" rpy="0 0 0"/>
      <geometry>
        <box size="{body_length} {body_width} {body_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- ============================================================
       左驱动轮 (Left Drive Wheel)
       5.5寸轮毂电机 + 聚氨酯轮胎
       ============================================================ -->
  <link name="left_wheel">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <!-- 轮毂电机外壳 -->
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="wheel_hub">
        <color rgba="0.2 0.2 0.2 1"/>
      </material>
    </visual>
    <visual>
      <!-- 轮胎外圈 -->
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="tire">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <margin value="0.001"/>
    </collision>
  </link>

  <!-- 右驱动轮 -->
  <link name="right_wheel">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="tire">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <margin value="0.001"/>
    </collision>
  </link>

  <!-- ============================================================
       从动轮 (ESUN 2.5寸静音避震万向轮)
       - 聚氨酯80A材质,单轮承重135kg
       - 减震行程10mm,360度旋转
       ============================================================ -->
  <link name="caster_front">
    <inertial>
      <mass value="{caster_mass}"/>
      <inertia ixx="{caster_ixx}" ixy="0" ixz="0" iyy="{caster_ixx}" iyz="0" izz="{caster_ixx}"/>
    </inertial>
    <visual>
      <!-- 聚氨酯轮子 (黄色) -->
      <origin xyz="0 0 -{caster_bracket_height}" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
      <material name="caster_color">
        <color rgba="0.9 0.7 0.2 1"/>  <!-- 聚氨酯黄色 -->
      </material>
    </visual>
    <visual>
      <!-- 安装支架 -->
      <origin xyz="0 0 {caster_radius}" rpy="0 0 0"/>
      <geometry>
        <box size="0.03 0.03 {caster_bracket_height}"/>
      </geometry>
      <material name="caster_bracket">
        <color rgba="0.3 0.3 0.3 1"/>  <!-- 金属灰 -->
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -{caster_bracket_height}" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
    </collision>
  </link>

  <link name="caster_back">
    <inertial>
      <mass value="{caster_mass}"/>
      <inertia ixx="{caster_ixx}" ixy="0" ixz="0" iyy="{caster_ixx}" iyz="0" izz="{caster_ixx}"/>
    </inertial>
    <visual>
      <!-- 聚氨酯轮子 (黄色) -->
      <origin xyz="0 0 -{caster_bracket_height}" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
      <material name="caster_color">
        <color rgba="0.9 0.7 0.2 1"/>  <!-- 聚氨酯黄色 -->
      </material>
    </visual>
    <visual>
      <!-- 安装支架 -->
      <origin xyz="0 0 {caster_radius}" rpy="0 0 0"/>
      <geometry>
        <box size="0.03 0.03 {caster_bracket_height}"/>
      </geometry>
      <material name="caster_bracket">
        <color rgba="0.3 0.3 0.3 1"/>  <!-- 金属灰 -->
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
    </collision>
  </link>

  <!-- ============================================================
       驱动轮关节 (Motor Joints)
       ============================================================ -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="{wheel_offset_x} -{track_width_half} -{body_height_plus_wheel_minus}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.05" joint_friction="{friction}"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="{wheel_offset_x} {track_width_half} -{body_height_plus_wheel_minus}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.05" joint_friction="{friction}"/>
  </joint>

  <!-- 从动轮关节 -->
  <joint name="caster_front_joint" type="continuous">
    <parent link="base_link"/>
    <child link="caster_front"/>
    <origin xyz="{caster_offset_x} 0 {body_height_plus_wheel}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="caster_back_joint" type="continuous">
    <parent link="base_link"/>
    <child link="caster_back"/>
    <origin xyz="-{caster_offset_x} 0 {body_height_plus_wheel}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- ============================================================
       IMU传感器: ETT10A-PW (6轴, 防水型)
       - 3轴加速度计 + 3轴陀螺仪
       - IP67防水, M8连接器
       ============================================================ -->
  <link name="imu_link">
    <inertial>
      <mass value="0.05"/>  <!-- 50g -->
      <inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="2e-5"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.04 0.04 0.025"/>  <!-- 40x40x25mm -->
      </geometry>
      <material name="imu_color">
        <color rgba="0.3 0.3 0.3 1"/>  <!-- 深灰色金属外壳 -->
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.04 0.04 0.025"/>
      </geometry>
    </collision>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 {body_height_2}" rpy="0 0 0"/>
  </joint>

  <!-- ============================================================
       前视深度相机: 奥比中光 Astra Pro Plus
       - 单目结构光, 640x480深度, 1280x960 RGB
       - 尺寸: 165 x 40 x 30mm (Astra Pro Plus)
       - C100: 47 x 38 x 22.7mm / C70: 47 x 38 x 28.5mm
       ============================================================ -->
  <link name="camera_link">
    <inertial>
      <mass value="0.04"/>  <!-- C100: 40g -->
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <!-- C100/C70 RGB相机: 47x38x22.7mm -->
      <geometry>
        <box size="0.047 0.038 0.0227"/>
      </geometry>
      <material name="camera_color">
        <color rgba="0.15 0.15 0.15 1"/>  <!-- 深灰色外壳 -->
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.047 0.038 0.0227"/>
      </geometry>
    </collision>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="{body_length_2_minus} 0 {body_height_minus_01}" rpy="0 0 0"/>
  </joint>

  <!-- ============================================================
       激光雷达: 镭神智能 N10P
       - 360°扫描, TOF测距, 最大25m
       - 尺寸: φ60mm × H86.5mm
       ============================================================ -->
  <link name="lidar_link">
    <inertial>
      <mass value="0.24"/>  <!-- 240g -->
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <!-- 镭神N10P 圆柱形外观 -->
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.0865"/>  <!-- φ60mm x H86.5mm -->
      </geometry>
      <material name="lidar_color">
        <color rgba="0.1 0.4 0.1 1"/>  <!-- 深绿色 -->
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.0865"/>
      </geometry>
    </collision>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 {body_height_plus_03}" rpy="0 0 0"/>
  </joint>

</robot>
"""


# 4轮差速驱动AGV URDF (双驱动轮配置)
AGV_4W_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="agv_4w_{grade}">

  <!-- ============================================================
       AGV 车体
       ============================================================ -->
  <link name="base_link">
    <inertial>
      <origin xyz="{com_x} {com_y} {com_z}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {body_height_2}" rpy="0 0 0"/>
      <geometry>
        <box size="{body_length} {body_width} {body_height}"/>
      </geometry>
      <material name="body_color">
        <color rgba="{body_color}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {body_height_2}" rpy="0 0 0"/>
      <geometry>
        <box size="{body_length} {body_width} {body_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- ============================================================
       左前驱动轮
       ============================================================ -->
  <link name="left_front_wheel">
    <inertial>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="tire">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <margin value="0.001"/>
    </collision>
  </link>

  <!-- 左后驱动轮 -->
  <link name="left_rear_wheel">
    <inertial>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="tire">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <margin value="0.001"/>
    </collision>
  </link>

  <!-- 右前驱动轮 -->
  <link name="right_front_wheel">
    <inertial>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="tire">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <margin value="0.001"/>
    </collision>
  </link>

  <!-- 右后驱动轮 -->
  <link name="right_rear_wheel">
    <inertial>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="tire">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <margin value="0.001"/>
    </collision>
  </link>

  <!-- ============================================================
       驱动轮关节
       ============================================================ -->
  <joint name="left_front_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_front_wheel"/>
    <origin xyz="{wheel_offset_x} -{track_width_half} -{body_height_plus_wheel_minus}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.05"/>
  </joint>

  <joint name="left_rear_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_rear_wheel"/>
    <origin xyz="-{wheel_offset_x} -{track_width_half} -{body_height_plus_wheel_minus}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.05"/>
  </joint>

  <joint name="right_front_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_front_wheel"/>
    <origin xyz="{wheel_offset_x} {track_width_half} -{body_height_plus_wheel_minus}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.05"/>
  </joint>

  <joint name="right_rear_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_rear_wheel"/>
    <origin xyz="-{wheel_offset_x} {track_width_half} -{body_height_plus_wheel_minus}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.05"/>
  </joint>

  <!-- IMU: ETT10A-PW (6轴防水) -->
  <link name="imu_link">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="2e-5"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.04 0.04 0.025"/>
      </geometry>
      <material name="imu_color">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.04 0.04 0.025"/>
      </geometry>
    </collision>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 {body_height_2}" rpy="0 0 0"/>
  </joint>

  <!-- RGB相机: 奥比中光 C100/C70 -->
  <link name="camera_link">
    <inertial>
      <mass value="0.04"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.047 0.038 0.0227"/>
      </geometry>
      <material name="camera_color">
        <color rgba="0.15 0.15 0.15 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.047 0.038 0.0227"/>
      </geometry>
    </collision>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="{body_length_2_minus} 0 {body_height_minus_01}" rpy="0 0 0"/>
  </joint>

  <!-- 激光雷达: 镭神智能 N10P -->
  <link name="lidar_link">
    <inertial>
      <mass value="0.24"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.0865"/>
      </geometry>
      <material name="lidar_color">
        <color rgba="0.1 0.4 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.0865"/>
      </geometry>
    </collision>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 {body_height_plus_03}" rpy="0 0 0"/>
  </joint>

</robot>
"""


def generate_agv_urdf_detailed(
    grade: str = 'M',
    wheel_config: str = '2轮',
    output_path: Optional[str] = None
) -> str:
    """
    生成详细的AGV URDF文件
    
    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        wheel_config: 轮子配置 ('2轮' 或 '4轮')
        output_path: 输出路径 (None=临时文件)
    
    Returns:
        URDF文件路径
    """
    import tempfile
    import os
    
    # 获取配置
    config = GRADE_CONFIGS.get(grade, GRADE_CONFIGS['M'])
    
    # 计算物理参数
    wheel_radius = config['wheel_diameter'] / 2
    wheel_width = config['wheel_width']
    wheel_mass = MOTOR_55_SPECS['weight']  # 1.5kg per wheel
    
    # 轮子转动惯量
    wheel_ixx = get_wheel_inertia(wheel_radius, wheel_width, wheel_mass)
    
    # 车体转动惯量
    body_inertia = get_body_inertia(
        config['body_length'],
        config['body_width'],
        config['body_height'],
        config['mass']
    )
    
    # 从动轮参数
    # ESUN 2.5寸静音避震万向轮参数
    caster_radius = CASTER_ESUN['wheel_radius']  # 0.03175m (2.5寸/2)
    caster_mass = 1.0  # 单轮承重135kg的万向轮,质量约1kg
    caster_bracket_height = CASTER_ESUN['bracket_height']  # 支架高度 0.07425m
    
    # 生成URDF
    if wheel_config == '4轮':
        template = AGV_4W_URDF_TEMPLATE
    else:
        template = AGV_2W_URDF_TEMPLATE
    
    # 构建参数字典（所有计算值）
    params = {
        'grade': grade,
        'description': config['description'],
        'body_length': config['body_length'],
        'body_width': config['body_width'],
        'body_height': config['body_height'],
        'body_height_2': config['body_height'] / 2,
        'mass': config['mass'],
        'max_speed': config['max_speed'],
        'rated_torque': config['rated_torque'],
        'wheel_config': config['wheel_config'],
        'track_width': config['track_width'],
        'track_width_half': config['track_width'] / 2,
        'wheel_radius': wheel_radius,
        'wheel_width': wheel_width,
        'wheel_mass': wheel_mass,
        'wheel_ixx': wheel_ixx,
        'wheel_offset_x': config['body_length'] * 0.3,
        'caster_radius': caster_radius,
        'caster_mass': caster_mass,
        'caster_ixx': get_wheel_inertia(caster_radius, caster_radius, caster_mass),
        'caster_bracket_height': caster_bracket_height,
        'caster_offset_x': config['body_length'] * 0.35,
        'body_length': config['body_length'],
        'body_length_2': config['body_length'] / 2,
        'body_length_2_minus': config['body_length'] / 2 - 0.02,
        'body_width': config['body_width'],
        'body_width_2': config['body_width'] / 2,
        'body_height': config['body_height'],
        'body_height_2': config['body_height'] / 2,
        'body_height_plus_03': config['body_height'] / 2 + 0.03,
        # 轮子位于车体下方,接触地面
        # wheel_joint_z (相对于base_link的z偏移): wheel应位于body下方
        # 轮子顶部接触body底部 => wheel_center_z = body_bottom - wheel_radius
        # body_bottom = 0 (base_link_z), so wheel_center_z = -wheel_radius
        # Template用负号: wheel_joint_z = -{body_height_plus_wheel_minus}
        # 所以 body_height_plus_wheel_minus = wheel_radius
        'body_height_plus_wheel': caster_radius,  # caster位置: sphere底部接触地面
        'body_height_plus_wheel_minus': wheel_radius,  # 驱动轮位置
        'body_height_minus_01': config['body_height'] / 2 - 0.01,
        'com_x': 0,
        'com_y': 0,
        'com_z': config['body_height'] * 0.3,
        'friction': 0.8,
        'body_color': "0.3 0.5 0.3 1",
        **body_inertia
    }
    
    urdf_content = template.format(**params)
    
    # 写入文件
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.urdf', prefix='agv_')
        os.write(fd, urdf_content.encode())
        os.close(fd)
    else:
        with open(output_path, 'w') as f:
            f.write(urdf_content)
    
    return output_path


def print_agv_specs(grade: str = 'M'):
    """打印AGV规格参数"""
    config = GRADE_CONFIGS.get(grade, GRADE_CONFIGS['M'])
    
    print(f"\n{'='*60}")
    print(f"AGV {grade}级规格 ({config['description']})")
    print(f"{'='*60}")
    print(f"\n【车体参数】")
    print(f"  尺寸: {config['body_length']} x {config['body_width']} x {config['body_height']} m")
    print(f"  自重: {config['mass']} kg")
    print(f"  负载: {config['payload']} kg")
    print(f"  轮子配置: {config['wheel_config']}")
    
    print(f"\n【电机参数】")
    print(f"  型号: {config['motor_spec']}")
    print(f"  直径: {config['wheel_diameter']*1000:.0f} mm ({config['wheel_diameter']/0.0254:.1f}寸)")
    print(f"  宽度: {config['wheel_width']*1000:.0f} mm")
    print(f"  额定扭矩: {config['rated_torque']} Nm")
    print(f"  最高速度: {config['max_speed']} m/s")
    
    print(f"\n【5.5寸轮毂电机规格】")
    for k, v in MOTOR_55_SPECS.items():
        print(f"  {k}: {v}")
    
    print(f"\n{'='*60}\n")


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # 打印所有等级规格
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        print_agv_specs(grade)
    
    # 生成URDF示例
    print("\n生成URDF文件...")
    
    for grade in ['S', 'M', 'L']:
        for config in ['2轮', '4轮']:
            if config == '2轮' and grade in ['L', 'XL', 'XXL']:
                continue  # 大型AGV只用4轮
                
            urdf_path = generate_agv_urdf_detailed(grade=grade, wheel_config=config)
            print(f"  {grade}{config}: {urdf_path}")
    
    print("\n完成!")
