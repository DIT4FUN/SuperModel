"""
地瓜机器人 RDK 系列支持
========================

地瓜机器人 (D-Robotics) RDK 开发板系列:

- **RDK X3**: 基于旭日X3派 (Sunrise X3 Pi)，5 TOPS NPU
- **RDK X5**: 基于旭日5 (Sunrise 5)，10 TOPS NPU
- **RDK S100**: 入门级，旭日系列芯片

注意: 旭日芯片来自地平线机器人 (D-Robotics，原地平线机器人)
官方文档: https://developer.d-robotics.cc/rdk_doc/
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

from .base import (
    BoardBase, BoardInfo, BoardType, ComputeCapability, PeripheralType
)
from .rk3588 import RK3588Platform


class DiguRobotSeries(Enum):
    """地瓜机器人系列"""
    RDK_X3 = "rdk_x3"
    RDK_X5_ULTRA = "rdk_x5_ultra"
    RDK_S100 = "rdk_s100"


# RDK 系列主板规格表
# 注意: 旭日(Sunrise)芯片来自地平线机器人(D-Robotics)
RDK_SPECS = {
    DiguRobotSeries.RDK_X3: {
        'name': '地瓜机器人 RDK X3 (旭日X3派)',
        'chip': '旭日X3 (Sunrise X3)',
        'cpu': '4x Cortex-A55',
        'cpu_cores': 4,
        'cpu_freq_mhz': 1500,
        'npu_tops': 5.0,
        'npu_type': 'BPU (地平线AI引擎)',
        'memory_mb': 4096,  # 4GB LPDDR4
        'gpu': 'G31',
        'video': '1080p@30fps 编码',
        'interfaces': ['HDMI 2.0', 'USB 3.0', 'MIPI-CSI x2', 'WiFi/BT'],
        'price_range': '¥299-499',
    },
    DiguRobotSeries.RDK_X5_ULTRA: {
        'name': '地瓜机器人 RDK X5 (旭日5)',
        'chip': '旭日5 (Sunrise 5)',
        'cpu': '4x Cortex-A55',
        'cpu_cores': 4,
        'cpu_freq_mhz': 1800,
        'npu_tops': 10.0,
        'npu_type': 'BPU (地平线AI引擎)',
        'memory_mb': 8192,  # 8GB LPDDR5
        'gpu': 'G57 MC1',
        'video': '4K@60fps 解码, 1080p@60fps 编码',
        'interfaces': ['HDMI 2.0', 'USB 3.0 x2', 'MIPI-CSI x4', '2x Ethernet', 'WiFi6'],
        'price_range': '¥549-699',
    },
    DiguRobotSeries.RDK_S100: {
        'name': '地瓜机器人 RDK S100',
        'chip': '旭日系列 (入门级)',
        'cpu': '2x Cortex-A55',
        'cpu_cores': 2,
        'cpu_freq_mhz': 1200,
        'npu_tops': 2.0,
        'npu_type': 'BPU (地平线AI引擎)',
        'memory_mb': 2048,  # 2GB LPDDR4
        'gpu': 'G31',
        'video': '1080p@30fps 解码',
        'interfaces': ['HDMI', 'USB 2.0', 'MIPI-CSI'],
        'price_range': '¥199-299',
    },
}


@dataclass
class RDKBoardInfo(BoardInfo):
    """RDK 主板信息扩展"""
    # RDK 特有字段
    npu_type: str = "RKNN"
    video_codec: str = ""
    interfaces: List[str] = None
    npu_freq_mhz: int = 0
    gpu_freq_mhz: int = 0
    
    def __post_init__(self):
        if self.interfaces is None:
            self.interfaces = []


class DiguRobotPlatform(RK3588Platform):
    """
    地瓜机器人 RDK 系列基类
    
    基于 RK3588 平台，添加地瓜特有的配置和功能。
    """
    
    SERIES: DiguRobotSeries = None
    
    def __init__(self, board_type: Optional[BoardType] = None, name: str = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化地瓜机器人平台
        
        Args:
            board_type: 可选，指定主板类型
            name: 可选，自定义名称
            config: 额外配置
        """
        super().__init__(config=config)
        
        if board_type and name:
            # 通用模式 (用于 x86_64 开发环境)
            self._custom_type = board_type
            self._custom_name = name
        else:
            self._custom_type = None
            self._custom_name = None
    
    @property
    def board_series(self) -> DiguRobotSeries:
        """获取主板系列"""
        return self.SERIES
    
    def _detect_info(self) -> RDKBoardInfo:
        """检测 RDK 主板信息"""
        # 如果是自定义类型，返回基本信息
        if self._custom_type:
            return RDKBoardInfo(
                board_type=self._custom_type,
                name=self._custom_name,
                chip="Unknown",
                cpu_cores=8,
                cpu_freq_mhz=2400,
                memory_mb=8192,
                npu_tops=0,
                arch='x86_64',
                peripherals=[],
            )
        
        # 读取设备树模型信息
        model = self._read_device_model()
        
        # 确定系列
        series = self.SERIES or self._detect_series_from_model(model)
        specs = RDK_SPECS.get(series, RDK_SPECS[DiguRobotSeries.RDK_X5_ULTRA])
        
        # 读取实际硬件信息
        freq_info = self._read_freq_info()
        mem_info = self._read_mem_info()
        peripherals = self._detect_peripherals()
        
        # 读取 NPU 实际频率
        npu_freq = freq_info.get('npu', 0)
        gpu_freq = freq_info.get('gpu', 0)
        
        # 估算 NPU 利用率
        npu_util = self.get_npu_utilization() if series != DiguRobotSeries.RDK_S100 else 0.0
        
        return RDKBoardInfo(
            board_type=BoardType.RDK_X5_ULTRA if series == DiguRobotSeries.RDK_X5_ULTRA 
                       else BoardType.RDK_X3 if series == DiguRobotSeries.RDK_X3
                       else BoardType.RDK_S100,
            name=specs['name'],
            chip=specs['chip'],
            cpu_cores=specs['cpu_cores'],
            cpu_freq_mhz=freq_info.get('cpu', specs['cpu_freq_mhz']),
            memory_mb=mem_info.get('total', specs['memory_mb']),
            npu_tops=specs['npu_tops'],
            gpu_tops=None,  # 集成 GPU 不单独计算
            arch='aarch64',
            os='linux',
            peripherals=peripherals,
            npu_type=specs['npu_type'],
            video_codec=specs['video'],
            interfaces=specs['interfaces'],
            npu_freq_mhz=npu_freq,
            gpu_freq_mhz=gpu_freq,
        )
    
    def _read_device_model(self) -> str:
        """读取设备模型名称"""
        if os.path.exists('/proc/device-tree/model'):
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    return f.read().strip()
            except:
                pass
        return ""
    
    def _detect_series_from_model(self, model: str) -> DiguRobotSeries:
        """从设备模型名称检测系列"""
        model_lower = model.lower()
        
        if 'x5' in model_lower or 'ultra' in model_lower:
            return DiguRobotSeries.RDK_X5_ULTRA
        elif 'x3' in model_lower:
            return DiguRobotSeries.RDK_X3
        elif 's100' in model_lower or 's-100' in model_lower:
            return DiguRobotSeries.RDK_S100
        
        # 默认返回 X5 Ultra
        return DiguRobotSeries.RDK_X5_ULTRA
    
    def initialize(self) -> bool:
        """初始化地瓜机器人平台"""
        if self._initialized:
            return True
        
        try:
            # 检查 RDK 特定服务
            self._check_rdk_services()
            
            # 初始化 NPU
            self._init_npu()
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"RDK initialization failed: {e}")
            return False
    
    def _check_rdk_services(self) -> None:
        """检查 RDK 特定服务状态"""
        # 检查 rknn_server (RKNN 推理服务)
        rknn_service_paths = [
            '/usr/bin/rknn_server',
            '/usr/local/bin/rknn_server',
        ]
        for path in rknn_service_paths:
            if os.path.exists(path):
                print(f"RKN Server found at {path}")
                break
    
    def _init_npu(self) -> bool:
        """初始化 NPU"""
        # 尝试启用 NPU
        return self.enable_npu()
    
    def get_rknn_model_path(self, model_name: str) -> str:
        """
        获取 RKNN 模型路径
        
        Args:
            model_name: 模型名称 (不含扩展名)
            
        Returns:
            模型文件路径
        """
        # 优先查找用户目录
        user_paths = [
            f'/home/root/.local/share/rknn/{model_name}.rknn',
            f'/home/root/rknn_models/{model_name}.rknn',
        ]
        
        # 系统路径
        system_paths = [
            f'/usr/share/rknn/{model_name}.rknn',
            f'/usr/local/share/rknn/{model_name}.rknn',
        ]
        
        for path in user_paths + system_paths:
            if os.path.exists(path):
                return path
        
        # 默认返回第一个路径
        return user_paths[0]
    
    def list_available_models(self) -> List[str]:
        """列出可用的 RKNN 模型"""
        models = []
        search_paths = [
            '/home/root/.local/share/rknn',
            '/home/root/rknn_models',
            '/usr/share/rknn',
            '/usr/local/share/rknn',
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                try:
                    for f in os.listdir(path):
                        if f.endswith('.rknn'):
                            models.append(f[:-5])  # 去掉 .rknn
                except:
                    pass
        
        return list(set(models))


class RDKX3(DiguRobotPlatform):
    """地瓜机器人 RDK X3"""
    
    BOARD_TYPE = BoardType.RDK_X3
    SERIES = DiguRobotSeries.RDK_X3
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.variant = 'rk3588'  # X3 使用 RK3588V2


class RDKX5Ultra(DiguRobotPlatform):
    """地瓜机器人 RDK X5 Ultra (旗舰)"""
    
    BOARD_TYPE = BoardType.RDK_X5_ULTRA
    SERIES = DiguRobotSeries.RDK_X5_ULTRA
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.variant = 'rk3588'


class RDKS100(DiguRobotPlatform):
    """地瓜机器人 RDK S100 (入门级)"""
    
    BOARD_TYPE = BoardType.RDK_S100
    SERIES = DiguRobotSeries.RDK_S100
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # S100 使用 RK3562，但基类按 RK3588 处理
    
    def _detect_info(self) -> RDKBoardInfo:
        """检测 RDK S100 主板信息"""
        # 读取基本信息
        freq_info = self._read_freq_info()
        mem_info = self._read_mem_info()
        peripherals = self._detect_peripherals()
        
        specs = RDK_SPECS[DiguRobotSeries.RDK_S100]
        
        return RDKBoardInfo(
            board_type=BoardType.RDK_S100,
            name=specs['name'],
            chip=specs['chip'],
            cpu_cores=specs['cpu_cores'],
            cpu_freq_mhz=freq_info.get('cpu', specs['cpu_freq_mhz']),
            memory_mb=mem_info.get('total', specs['memory_mb']),
            npu_tops=specs['npu_tops'],
            gpu_tops=None,
            arch='aarch64',
            os='linux',
            peripherals=peripherals,
            npu_type=specs['npu_type'],
            video_codec=specs['video'],
            interfaces=specs['interfaces'],
        )


# RDK 系列速查表
RDK_COMPARISON = """
地瓜机器人 RDK 系列对比表 (旭日芯片)
=====================================

| 规格       | RDK S100      | RDK X3 (X3派)   | RDK X5 (X5)       |
|-----------|---------------|-----------------|-------------------|
| 芯片      | 旭日系列      | 旭日X3          | 旭日5              |
| 架构      | 2x A55        | 4x A55          | 4x A55            |
| CPU主频   | 1.2 GHz       | 1.5 GHz         | 1.8 GHz           |
| NPU算力   | 2 TOPS        | 5 TOPS          | 10 TOPS           |
| NPU类型   | BPU           | BPU             | BPU               |
| 内存      | 2 GB LPDDR4   | 4 GB LPDDR4     | 8 GB LPDDR5       |
| GPU       | G31           | G31             | G57 MC1           |
| 视频解码  | 1080p@30fps   | 1080p@30fps     | 4K@60fps          |
| 视频编码  | -             | 1080p@30fps     | 1080p@60fps       |
| 视频接口  | HDMI          | HDMI 2.0        | HDMI 2.0          |
| USB       | USB 2.0       | USB 3.0         | USB 3.0 x2        |
| CSI       | MIPI-CSI      | MIPI-CSI x2     | MIPI-CSI x4       |
| 价格      | ¥199-299      | ¥299-499        | ¥549-699          |
| 定位      | 入门级        | 主流            | 高端              |

典型应用场景:
- RDK S100: 学习入门、物联网网关、简单视觉处理
- RDK X3 (旭日X3派): 边缘推理、ROS2 机器人、消费级 AI 设备
- RDK X5 (旭日5): 具身智能大脑、工业机器人、高端 AIoT

芯片来源: 地平线机器人 (D-Robotics) 旭日系列 AI 芯片
"""
