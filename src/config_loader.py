"""
SuperModel 配置加载与验证器
===========================

从 configs/*.yaml 加载 AGV 各级配置
验证配置完整性与一致性
生成跨模态输入示例

支持等级: S / M / L / XL / XXL
"""

import os
import yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))


AGV_GRADES = ['S', 'M', 'L', 'XL', 'XXL']
CONFIG_DIR = Path(__file__).parent.parent / 'configs'


@dataclass
class SensorSpec:
    """传感器规格"""
    enabled: bool = False
    type: str = ""
    # 视觉
    resolution: Optional[List[int]] = None
    fps: int = 30
    baseline_mm: float = 0.0
    depth_range_m: Optional[List[float]] = None
    fov_deg: float = 0.0
    depth_accuracy_cm: float = 0.0
    # 听觉
    sample_rate: int = 16000
    channels: int = 1
    beamforming: bool = False
    pickup_range_m: float = 1.0
    source_localization_accuracy_deg: float = 0.0
    # 触觉
    array_size: Optional[List[int]] = None
    resolution_bits: int = 12
    pressure_range_kpa: Optional[List[int]] = None
    sampling_hz: int = 50
    temperature: bool = False
    proximity: bool = False
    # 力觉
    axes: int = 3
    force_range: float = 100.0
    torque_range: float = 10.0
    resolution_n: float = 0.1
    interface: str = "USB"
    # IMU
    accel_range: int = 8
    gyro_range: int = 1000
    noise_density: float = 400.0
    bias_stability_deg_s: float = 1.0


@dataclass
class FusionSpec:
    """融合规格"""
    strategy: str = "middle"
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    feature_dim: int = 64
    inference_latency_ms: float = 50.0


@dataclass
class ModelSpec:
    """模型规格"""
    world_model_enabled: bool = False
    world_model_grade: str = "S"
    latent_dim: int = 128
    hidden_dim: int = 256
    imagination_horizon: int = 10
    parameters: str = "~1M"
    lr: float = 1e-4
    batch_size: int = 16
    reinforcement_learning: str = "none"
    self_supervised: bool = False


@dataclass
class ControlSpec:
    """控制规格"""
    control_rate_hz: int = 50
    position_precision_mm: float = 5.0
    force_precision_n: float = 1.0
    impedance_control: bool = False
    impedance_type: str = "none"
    max_payload_kg: float = 2.0
    safety_soft_limits: bool = True
    safety_hard_limits: bool = False
    safety_collision_detection: bool = False
    safety_force_limited: bool = False
    safety_emergency_stop: bool = True


@dataclass
class HardwareSpec:
    """硬件规格"""
    platform: str = "Raspberry Pi 5"
    compute_device: str = "cpu"
    cpu: str = "4核 A76"
    gpu_tops: float = 1.5
    memory_gb: float = 4.0
    power_w: float = 5.0
    protection_class: str = "IP20"


@dataclass
class AGVConfig:
    """完整AGV配置"""
    name: str = "SuperModel-AGV-S"
    version: str = "0.1.0"
    grade: str = "S"
    description: str = ""
    sensors: SensorSpec = field(default_factory=SensorSpec)
    fusion: FusionSpec = field(default_factory=FusionSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    communication: Dict[str, str] = field(default_factory=dict)
    software: Dict[str, str] = field(default_factory=dict)
    dimensions: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    配置加载器
    
    从 YAML 文件加载 AGV 配置，支持验证和跨等级比较
    """
    
    _cache: Dict[str, AGVConfig] = {}
    
    @classmethod
    def load(cls, grade: str, config_dir: Path = CONFIG_DIR) -> AGVConfig:
        """
        加载指定等级的 AGV 配置
        
        Args:
            grade: AGV 等级 (S/M/L/XL/XXL)
            config_dir: 配置文件目录
            
        Returns:
            AGVConfig: 配置对象
        """
        grade = grade.upper()
        if grade in cls._cache:
            return cls._cache[grade]
        
        config_path = config_dir / f'agv_{grade.lower()}.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 解析配置
        cfg = cls._parse_config(data, grade)
        cls._cache[grade] = cfg
        return cfg
    
    @classmethod
    def load_all(cls) -> Dict[str, AGVConfig]:
        """加载所有等级的 AGV 配置"""
        configs = {}
        for grade in AGV_GRADES:
            try:
                configs[grade] = cls.load(grade)
            except FileNotFoundError:
                pass
        return configs
    
    @classmethod
    def _parse_config(cls, data: Dict, grade: str) -> AGVConfig:
        """解析 YAML 配置为 AGVConfig"""
        project = data.get('project', {})
        
        sensors_data = data.get('sensors', {})
        sensors = SensorSpec(
            enabled=sensors_data.get('vision', {}).get('enabled', False),
            type=sensors_data.get('vision', {}).get('type', ''),
            resolution=sensors_data.get('vision', {}).get('resolution'),
            fps=sensors_data.get('vision', {}).get('fps', 30),
            baseline_mm=sensors_data.get('vision', {}).get('baseline_mm', 0),
            depth_range_m=sensors_data.get('vision', {}).get('depth_range_m'),
            fov_deg=sensors_data.get('vision', {}).get('fov_deg', 0),
            depth_accuracy_cm=sensors_data.get('vision', {}).get('depth_accuracy_cm', 0),
            sample_rate=sensors_data.get('audio', {}).get('sample_rate', 16000),
            channels=sensors_data.get('audio', {}).get('channels', 1),
            beamforming=sensors_data.get('audio', {}).get('beamforming', False),
            pickup_range_m=sensors_data.get('audio', {}).get('pickup_range_m', 1.0),
            source_localization_accuracy_deg=sensors_data.get('audio', {}).get('source_localization_accuracy_deg', 0),
            array_size=sensors_data.get('tactile', {}).get('array_size'),
            resolution_bits=sensors_data.get('tactile', {}).get('resolution_bits', 12),
            pressure_range_kpa=sensors_data.get('tactile', {}).get('pressure_range_kpa'),
            sampling_hz=sensors_data.get('tactile', {}).get('sampling_hz', 50),
            temperature=sensors_data.get('tactile', {}).get('temperature', False),
            proximity=sensors_data.get('tactile', {}).get('proximity', False),
            axes=sensors_data.get('force', {}).get('axes', 3),
            force_range=sensors_data.get('force', {}).get('force_range', 100),
            torque_range=sensors_data.get('force', {}).get('torque_range', 10),
            resolution_n=sensors_data.get('force', {}).get('resolution_n', 0.1),
            interface=sensors_data.get('force', {}).get('interface', 'USB'),
            accel_range=sensors_data.get('imu', {}).get('accel_range', 8),
            gyro_range=sensors_data.get('imu', {}).get('gyro_range', 1000),
            noise_density=sensors_data.get('imu', {}).get('noise_density', 400),
            bias_stability_deg_s=sensors_data.get('imu', {}).get('bias_stability_deg_s', 1.0),
        )
        
        fusion_data = data.get('fusion', {})
        fusion = FusionSpec(
            strategy=fusion_data.get('strategy', 'middle'),
            hidden_dim=fusion_data.get('hidden_dim', 128),
            num_heads=fusion_data.get('num_heads', 4),
            num_layers=fusion_data.get('num_layers', 2),
            dropout=fusion_data.get('dropout', 0.1),
            feature_dim=fusion_data.get('feature_dim', 64),
            inference_latency_ms=fusion_data.get('inference_latency_ms', 50.0),
        )
        
        model_data = data.get('model', {})
        world_model_data = model_data.get('world_model', {})
        learning_data = model_data.get('learning', {})
        model = ModelSpec(
            world_model_enabled=world_model_data.get('enabled', False),
            world_model_grade=world_model_data.get('grade', 'S'),
            latent_dim=world_model_data.get('latent_dim', 128),
            hidden_dim=world_model_data.get('hidden_dim', 256),
            imagination_horizon=world_model_data.get('imagination_horizon', 10),
            parameters=world_model_data.get('parameters', '~1M'),
            lr=learning_data.get('lr', 1e-4),
            batch_size=learning_data.get('batch_size', 16),
            reinforcement_learning=learning_data.get('reinforcement_learning', 'none'),
            self_supervised=learning_data.get('self_supervised', False),
        )
        
        control_data = data.get('control', {})
        safety_data = control_data.get('safety', {})
        control = ControlSpec(
            control_rate_hz=control_data.get('control_rate_hz', 50),
            position_precision_mm=control_data.get('position_precision_mm', 5.0),
            force_precision_n=control_data.get('force_precision_n', 1.0),
            impedance_control=control_data.get('impedance_control', False),
            impedance_type=control_data.get('impedance_type', 'none'),
            max_payload_kg=control_data.get('max_payload_kg', 2.0),
            safety_soft_limits=safety_data.get('soft_limits', True),
            safety_hard_limits=safety_data.get('hard_limits', False),
            safety_collision_detection=safety_data.get('collision_detection', False),
            safety_force_limited=safety_data.get('force_limited', False),
            safety_emergency_stop=safety_data.get('emergency_stop', True),
        )
        
        hw_data = data.get('hardware', {})
        hardware = HardwareSpec(
            platform=hw_data.get('platform', 'Raspberry Pi 5'),
            compute_device=hw_data.get('compute_device', 'cpu'),
            cpu=hw_data.get('cpu', '4核 A76'),
            gpu_tops=float(hw_data.get('gpu_tops', 1.5)),
            memory_gb=float(hw_data.get('memory_gb', 4.0)),
            power_w=float(hw_data.get('power_w', 5.0)),
            protection_class=hw_data.get('protection_class', 'IP20'),
        )
        
        return AGVConfig(
            name=project.get('name', f'SuperModel-AGV-{grade}'),
            version=project.get('version', '0.1.0'),
            grade=grade,
            description=project.get('description', ''),
            sensors=sensors,
            fusion=fusion,
            model=model,
            control=control,
            hardware=hardware,
            communication=data.get('communication', {}),
            software=data.get('software', {}),
            dimensions=data.get('dimensions', {}),
        )
    
    @classmethod
    def validate(cls, config: AGVConfig) -> List[str]:
        """
        验证配置完整性
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # 检查传感器配置
        if config.sensors.enabled:
            if config.sensors.type == '':
                errors.append(f"[{config.grade}] Sensor type not specified")
        
        # 检查触觉分辨率
        if config.sensors.array_size:
            if config.sensors.array_size[0] * config.sensors.array_size[1] > 2304:
                if config.sensors.sampling_hz < 200:
                    errors.append(f"[{config.grade}] High-res tactile ({config.sensors.array_size}) requires sampling_hz >= 200")
        
        # 检查力觉轴数
        if config.sensors.axes == 6 and config.sensors.force_range < 200:
            errors.append(f"[{config.grade}] 6-axis force sensor should have force_range >= 200N")
        
        # 检查融合层数与隐层维度一致性
        if config.fusion.num_layers >= 4 and config.fusion.hidden_dim < 256:
            errors.append(f"[{config.grade}] num_layers >= 4 requires hidden_dim >= 256")
        
        # 检查模型规模与硬件匹配
        if 'M' in config.model.parameters:
            param_count = float(config.model.parameters.replace('~', '').replace('M', ''))
            if param_count > 50 and config.hardware.gpu_tops < 20:
                errors.append(f"[{config.grade}] Large model ({config.model.parameters}) requires GPU >= 20 TOPS")
        
        # 检查控制频率
        if config.control.control_rate_hz > 200 and config.hardware.compute_device == 'cpu':
            errors.append(f"[{config.grade}] High control rate ({config.control.control_rate_hz}Hz) requires GPU")
        
        return errors
    
    @classmethod
    def compare_grades(cls, grades: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        比较多个 AGV 等级的配置差异
        
        Returns:
            Dict with comparison results
        """
        if grades is None:
            grades = AGV_GRADES
        
        configs = {g: cls.load(g) for g in grades if g in AGV_GRADES}
        
        comparison = {
            'grades': grades,
            'sensors': {},
            'fusion': {},
            'model': {},
            'control': {},
            'hardware': {},
        }
        
        for category in ['sensors', 'fusion', 'model', 'control', 'hardware']:
            comp = comparison[category]
            for attr in configs[grades[0]].__dataclass_fields__.keys():
                values = {g: str(getattr(getattr(configs[g], category), attr)) for g in grades}
                if len(set(values.values())) > 1:
                    comp[attr] = values
        
        return comparison
    
    @classmethod
    def generate_spec_table(cls, grades: Optional[List[str]] = None) -> str:
        """
        生成 AGV 等级规格对比表 (Markdown 格式)
        """
        if grades is None:
            grades = AGV_GRADES
        
        configs = {g: cls.load(g) for g in grades if g in AGV_GRADES}
        
        lines = ["# AGV 等级规格对比表\n"]
        lines.append(f"| 参数 | {' | '.join(grades)} |")
        lines.append(f"|------|{'|'.join(['---'] * len(grades))}|")
        
        # 感知系统
        lines.append("\n## 感知系统\n")
        
        # 视觉
        lines.append("### 视觉\n")
        rows = [
            ("相机类型", lambda c: c.sensors.type),
            ("分辨率", lambda c: str(c.sensors.resolution)),
            ("帧率", lambda c: f"{c.sensors.fps} fps"),
            ("基线", lambda c: f"{c.sensors.baseline_mm} mm"),
            ("深度范围", lambda c: str(c.sensors.depth_range_m)),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        # 触觉
        lines.append("\n### 触觉\n")
        rows = [
            ("阵列尺寸", lambda c: str(c.sensors.array_size)),
            ("分辨率", lambda c: f"{c.sensors.resolution_bits} bit"),
            ("压力范围", lambda c: str(c.sensors.pressure_range_kpa)),
            ("采样频率", lambda c: f"{c.sensors.sampling_hz} Hz"),
            ("温度感知", lambda c: '✓' if c.sensors.temperature else '✗'),
            ("接近觉", lambda c: '✓' if c.sensors.proximity else '✗'),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        # 力觉
        lines.append("\n### 力觉\n")
        rows = [
            ("轴数", lambda c: str(c.sensors.axes)),
            ("力范围", lambda c: f"±{c.sensors.force_range} N"),
            ("力矩范围", lambda c: f"±{c.sensors.torque_range} N·m"),
            ("分辨率", lambda c: f"{c.sensors.resolution_n} N"),
            ("采样频率", lambda c: f"{c.control.control_rate_hz} Hz"),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        # IMU
        lines.append("\n### IMU\n")
        rows = [
            ("传感器型号", lambda c: c.sensors.type),
            ("加速度量程", lambda c: f"±{c.sensors.accel_range}g"),
            ("陀螺量程", lambda c: f"±{c.sensors.gyro_range}°/s"),
            ("噪声密度", lambda c: f"{c.sensors.noise_density} μg/√Hz"),
            ("采样频率", lambda c: f"{c.sensors.sampling_hz} Hz"),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        # 融合系统
        lines.append("\n## 融合系统\n")
        rows = [
            ("融合策略", lambda c: c.fusion.strategy),
            ("隐层维度", lambda c: str(c.fusion.hidden_dim)),
            ("注意力头数", lambda c: str(c.fusion.num_heads)),
            ("融合层数", lambda c: str(c.fusion.num_layers)),
            ("推理延迟", lambda c: f"{c.fusion.inference_latency_ms} ms"),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        # 硬件
        lines.append("\n## 硬件平台\n")
        rows = [
            ("平台", lambda c: c.hardware.platform),
            ("算力", lambda c: f"{c.hardware.gpu_tops} TOPS"),
            ("内存", lambda c: f"{c.hardware.memory_gb} GB"),
            ("功耗", lambda c: f"{c.hardware.power_w} W"),
            ("防护等级", lambda c: c.hardware.protection_class),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        # 执行系统
        lines.append("\n## 执行系统\n")
        rows = [
            ("控制频率", lambda c: f"{c.control.control_rate_hz} Hz"),
            ("位置精度", lambda c: f"±{c.control.position_precision_mm} mm"),
            ("力控精度", lambda c: f"±{c.control.force_precision_n} N"),
            ("最大负载", lambda c: f"{c.control.max_payload_kg} kg"),
            ("阻抗控制", lambda c: '✓' if c.control.impedance_control else '✗'),
        ]
        for label, getter in rows:
            vals = ' | '.join(getter(configs[g]) for g in grades)
            lines.append(f"| {label} | {vals} |")
        
        return '\n'.join(lines)


def load_agv_config(grade: str) -> AGVConfig:
    """便捷函数: 加载单个 AGV 配置"""
    return ConfigLoader.load(grade)


def load_all_agv_configs() -> Dict[str, AGVConfig]:
    """便捷函数: 加载所有 AGV 配置"""
    return ConfigLoader.load_all()


def validate_agv_config(grade: str) -> Tuple[bool, List[str]]:
    """
    便捷函数: 验证 AGV 配置
    
    Returns:
        (is_valid, error_messages)
    """
    config = ConfigLoader.load(grade)
    errors = ConfigLoader.validate(config)
    return len(errors) == 0, errors


if __name__ == '__main__':
    # 演示: 加载并验证所有配置
    print("Loading all AGV configurations...")
    configs = ConfigLoader.load_all()
    
    for grade, cfg in configs.items():
        valid, errors = ConfigLoader.validate(cfg)
        status = "✓ VALID" if valid else f"✗ ERRORS: {errors}"
        print(f"  [{grade}] {cfg.name}: {status}")
    
    print("\nGenerating spec table...")
    table = ConfigLoader.generate_spec_table()
    print(table)
