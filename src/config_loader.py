"""
SuperModel AGV 配置加载器
========================

支持加载 AGV 五级配置 (S/M/L/XL/XXL)
提供配置验证和参数查询功能
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class AGVConfig:
    """AGV 配置类"""
    
    # 配置目录
    CONFIG_DIR = Path(__file__).parent.parent.parent / 'configs'
    
    # 等级映射
    GRADE_FILES = {
        'S': 'agv_S.yaml',
        'M': 'agv_M.yaml',
        'L': 'agv_L.yaml',
        'XL': 'agv_XL.yaml',
        'XXL': 'agv_XXL.yaml'
    }
    
    def __init__(self, grade: str = 'M', config_path: Optional[str] = None):
        """
        初始化 AGV 配置
        
        Args:
            grade: AGV 等级 (S/M/L/XL/XXL)
            config_path: 自定义配置文件路径
        """
        self.grade = grade.upper()
        if self.grade not in self.GRADE_FILES:
            raise ValueError(f"Invalid grade: {grade}. Must be one of {list(self.GRADE_FILES.keys())}")
        
        self.config_path = config_path or str(self.CONFIG_DIR / self.GRADE_FILES[self.grade])
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            # 尝试加载项目默认配置
            default_path = self.CONFIG_DIR.parent / 'configs' / 'project_config.yaml'
            if os.path.exists(default_path):
                self.config_path = str(default_path)
            else:
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def grade(self) -> str:
        """获取 AGV 等级"""
        return self._config.get('project', {}).get('grade', self.grade)
    
    @property
    def name(self) -> str:
        """获取项目名称"""
        return self._config.get('project', {}).get('name', 'SuperModel')
    
    @property
    def sensors(self) -> Dict[str, Any]:
        """获取传感器配置"""
        return self._config.get('sensors', {})
    
    @property
    def fusion(self) -> Dict[str, Any]:
        """获取融合配置"""
        return self._config.get('fusion', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self._config.get('model', {})
    
    @property
    def control(self) -> Dict[str, Any]:
        """获取控制配置"""
        return self._config.get('control', {})
    
    @property
    def hardware(self) -> Dict[str, Any]:
        """获取硬件配置"""
        return self._config.get('hardware', {})
    
    @property
    def communication(self) -> Dict[str, Any]:
        """获取通信配置"""
        return self._config.get('communication', {})
    
    @property
    def software(self) -> Dict[str, Any]:
        """获取软件功能配置"""
        return self._config.get('software', {})
    
    @property
    def dimensions(self) -> Dict[str, Any]:
        """获取尺寸重量配置"""
        return self._config.get('dimensions', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_sensor_config(self, sensor_type: str) -> Dict[str, Any]:
        """获取指定传感器配置"""
        return self.sensors.get(sensor_type, {})
    
    def get_control_rate(self) -> int:
        """获取控制频率 (Hz)"""
        return self.control.get('control_rate_hz', 100)
    
    def get_compute_device(self) -> str:
        """获取计算设备"""
        return self.hardware.get('compute_device', 'cuda')
    
    def get_memory_gb(self) -> int:
        """获取内存大小 (GB)"""
        return self.hardware.get('memory_gb', 8)
    
    def is_world_model_enabled(self) -> bool:
        """是否启用世界模型"""
        return self.model.get('world_model', {}).get('enabled', False)
    
    def is_realtime(self) -> bool:
        """是否实时系统"""
        return self.communication.get('realtime', False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
    
    def summary(self) -> str:
        """生成配置摘要"""
        return f"""
AGV-{self.grade} 配置摘要
========================
项目: {self.name}
描述: {self._config.get('project', {}).get('description', 'N/A')}

传感器:
  - 视觉: {self.get_sensor_config('vision').get('type', 'N/A')}
  - 听觉: {self.get_sensor_config('audio').get('type', 'N/A')}
  - 触觉: {self.get_sensor_config('tactile').get('type', 'N/A')} ({self.get_sensor_config('tactile').get('array_size', 'N/A')})
  - 力觉: {self.get_sensor_config('force').get('type', 'N/A')} ({self.get_sensor_config('force').get('axes', 'N/A')}轴)
  - IMU: {self.get_sensor_config('imu').get('type', 'N/A')}

硬件:
  - 平台: {self.hardware.get('platform', 'N/A')}
  - 算力: {self.hardware.get('gpu_tops', 'N/A')} TOPS
  - 内存: {self.hardware.get('memory_gb', 'N/A')} GB
  - 功耗: {self.hardware.get('power_w', 'N/A')} W
  - 防护: {self.hardware.get('protection_class', 'N/A')}

控制:
  - 频率: {self.get_control_rate()} Hz
  - 位置精度: ±{self.control.get('position_precision_mm', 'N/A')} mm
  - 有效载荷: {self.control.get('max_payload_kg', 'N/A')} kg

软件功能:
  - SLAM: {self.software.get('slam', 'N/A')}
  - 物体识别: {self.software.get('object_recognition', 'N/A')}
  - 视觉伺服: {self.software.get('visual_servo', 'N/A')}
  - 世界模型: {'✓' if self.is_world_model_enabled() else '✗'}
  - 数字孪生: {'✓' if self.software.get('digital_twin') else '✗'}

尺寸重量:
  - 尺寸: {self.dimensions.get('typical_cm', 'N/A')} cm
  - 重量: {self.dimensions.get('weight_kg', 'N/A')} kg
  - 安装: {self.dimensions.get('installation', 'N/A')}
"""


def load_agv_config(grade: str = 'M') -> AGVConfig:
    """加载 AGV 配置"""
    return AGVConfig(grade=grade)


def get_all_grade_configs() -> Dict[str, AGVConfig]:
    """加载所有等级的 AGV 配置"""
    return {grade: AGVConfig(grade=grade) for grade in AGVConfig.GRADE_FILES.keys()}


def compare_grade_configs(grades: list = None) -> str:
    """对比多个 AGV 等级的配置差异"""
    if grades is None:
        grades = ['S', 'M', 'L', 'XL', 'XXL']
    
    configs = {g: AGVConfig(grade=g) for g in grades}
    
    lines = ["AGV 等级配置对比", "=" * 80]
    
    # 关键参数对比
    headers = ["参数"] + grades
    lines.append(" | ".join(f"{h:>12}" for h in headers))
    lines.append("-" * 80)
    
    comparisons = [
        ("算力 (TOPS)", lambda c: str(c.hardware.get('gpu_tops', 'N/A'))),
        ("内存 (GB)", lambda c: str(c.hardware.get('memory_gb', 'N/A'))),
        ("控制频率 (Hz)", lambda c: str(c.get_control_rate())),
        ("位置精度 (mm)", lambda c: str(c.control.get('position_precision_mm', 'N/A'))),
        ("最大负载 (kg)", lambda c: str(c.control.get('max_payload_kg', 'N/A'))),
        ("触觉阵列", lambda c: str(c.get_sensor_config('tactile').get('array_size', 'N/A'))),
        ("力觉轴数", lambda c: str(c.get_sensor_config('force').get('axes', 'N/A'))),
        ("IMU采样 (Hz)", lambda c: str(c.get_sensor_config('imu').get('sampling_hz', 'N/A'))),
        ("世界模型", lambda c: '✓' if c.is_world_model_enabled() else '✗'),
        ("数字孪生", lambda c: '✓' if c.software.get('digital_twin') else '✗'),
    ]
    
    for label, accessor in comparisons:
        row = [label] + [accessor(configs[g]) for g in grades]
        lines.append(" | ".join(f"{str(v):>12}" for v in row))
    
    return "\n".join(lines)


# 导出
__all__ = ['AGVConfig', 'load_agv_config', 'get_all_grade_configs', 'compare_grade_configs']
