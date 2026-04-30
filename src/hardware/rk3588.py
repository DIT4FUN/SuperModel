# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
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
RK3588 平台支持
================

瑞芯微 Rockchip RK3588/RK3588S 系列芯片支持。

注意: 此模块用于瑞芯微 RK3588 芯片，
地瓜机器人 RDK 系列使用旭日(Sunrise)芯片，详见 digu_robot.py

RK3588 规格:
- CPU: 4x Cortex-A76 + 4x Cortex-A55 (big.LITTLE)
- NPU: 6 TOPS INT8 / 12 TOPS INT4
- GPU: Mali-G610 MP4 (Quad-core)
- 8K@30fps 视频编解码
- 支持 LPDDR5, eMMC 5.1, PCIe 3.0, USB 3.2
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .base import (
    BoardBase, BoardInfo, BoardType, ComputeCapability, PeripheralType
)


@dataclass
class RK3588BoardInfo(BoardInfo):
    """RK3588 主板信息扩展"""
    # RK3588 特有字段
    npu_freq_mhz: int = 0
    gpu_freq_mhz: int = 0
    npu_utilization: float = 0.0
    gpu_utilization: float = 0.0


class RK3588Platform(BoardBase):
    """
    RK3588 平台类
    
    支持:
    - RK3588 (带 NPU 6-12 TOPS)
    - RK3588S (紧凑版，NPU 3-6 TOPS)
    """
    
    BOARD_TYPE = BoardType.RK3588
    
    # RK3588 标准配置
    DEFAULT_SPECS = {
        'rk3588': {
            'name': 'Rockchip RK3588',
            'chip': 'RK3588',
            'cpu_cores': 8,  # 4x A76 + 4x A55
            'cpu_freq_mhz': 2400,  # A76 max
            'npu_tops': 6.0,  # INT8
            'gpu_tops': None,  # Mali-G610 无独立 TOPS
            'memory_mb': 8192,  # 典型 8GB
        },
        'rk3588s': {
            'name': 'Rockchip RK3588S',
            'chip': 'RK3588S',
            'cpu_cores': 8,
            'cpu_freq_mhz': 2200,
            'npu_tops': 3.0,
            'gpu_tops': None,
            'memory_mb': 4096,
        }
    }
    
    def __init__(self, variant: str = 'rk3588', config: Optional[Dict[str, Any]] = None):
        """
        初始化 RK3588 平台
        
        Args:
            variant: 芯片变体 ('rk3588' 或 'rk3588s')
            config: 额外配置
        """
        super().__init__(config)
        self.variant = variant.lower()
        self.specs = self.DEFAULT_SPECS.get(self.variant, self.DEFAULT_SPECS['rk3588'])
        
    def _detect_info(self) -> RK3588BoardInfo:
        """检测 RK3588 主板信息"""
        # 尝试读取设备树兼容信息
        chip_variant = self.variant
        
        if os.path.exists('/proc/device-tree/compatible'):
            try:
                with open('/proc/device-tree/compatible', 'r') as f:
                    compat = f.read().lower()
                    if 'rk3588s' in compat:
                        chip_variant = 'rk3588s'
                    elif 'rk3588' in compat:
                        chip_variant = 'rk3588'
            except:
                pass
        
        # 读取 CPU 信息
        cpu_info = self._read_cpu_info()
        
        # 读取内存信息
        mem_info = self._read_mem_info()
        
        # 读取频率信息
        freq_info = self._read_freq_info()
        
        # 读取外设信息
        peripherals = self._detect_peripherals()
        
        return RK3588BoardInfo(
            board_type=BoardType.RK3588 if chip_variant == 'rk3588' else BoardType.RK3588S,
            name=self.specs['name'],
            chip=chip_variant.upper(),
            cpu_cores=self.specs['cpu_cores'],
            cpu_freq_mhz=freq_info.get('cpu', self.specs['cpu_freq_mhz']),
            memory_mb=mem_info.get('total', self.specs['memory_mb']),
            npu_tops=self.specs['npu_tops'],
            gpu_tops=self.specs['gpu_tops'],
            arch='aarch64',
            os='linux',
            peripherals=peripherals,
            npu_freq_mhz=freq_info.get('npu', 0),
            gpu_freq_mhz=freq_info.get('gpu', 0),
        )
    
    def _read_cpu_info(self) -> Dict[str, Any]:
        """读取 CPU 信息"""
        info = {}
        try:
            # 读取 CPU 型号
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            info['model'] = line.split(':')[1].strip()
                            break
        except:
            pass
        return info
    
    def _read_mem_info(self) -> Dict[str, int]:
        """读取内存信息"""
        mem = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        mem['total'] = int(line.split()[1]) // 1024  # KB -> MB
                    elif line.startswith('MemAvailable'):
                        mem['available'] = int(line.split()[1]) // 1024
        except:
            pass
        return mem
    
    def _read_freq_info(self) -> Dict[str, int]:
        """读取频率信息"""
        freq = {}
        
        # CPU 频率
        cpu_freq_paths = [
            '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq',
            '/sys/bus/cpu/devices/cpu0/cpufreq/cpuinfo_max_freq',
        ]
        for path in cpu_freq_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        freq['cpu'] = int(f.read().strip()) // 1000  # KHz -> MHz
                    break
                except:
                    pass
        
        # NPU 频率
        npu_freq_paths = [
            '/sys/class/devfreq/ff9a0000.npu/cur_freq',
            '/sys/bus/platform/devices/ff9a0000.npu/devfreq/ff9a0000.npu/cur_freq',
        ]
        for path in npu_freq_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        freq['npu'] = int(f.read().strip()) // 1000000  # Hz -> MHz
                    break
                except:
                    pass
        
        # GPU 频率
        gpu_freq_paths = [
            '/sys/class/devfreq/ff9a0000.gpu/cur_freq',
            '/sys/bus/platform/devices/ff9a0000.gpu/devfreq/ff9a0000.gpu/cur_freq',
        ]
        for path in gpu_freq_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        freq['gpu'] = int(f.read().strip()) // 1000000
                    break
                except:
                    pass
        
        return freq
    
    def _detect_peripherals(self) -> List[PeripheralType]:
        """检测可用的外设"""
        peripherals = []
        
        # GPIO
        if os.path.exists('/sys/class/gpio'):
            peripherals.append(PeripheralType.GPIO)
        
        # I2C
        if os.path.exists('/dev/i2c-0') or os.path.exists('/dev/i2c-1'):
            peripherals.append(PeripheralType.I2C)
        
        # SPI
        if os.path.exists('/dev/spidev'):
            peripherals.append(PeripheralType.SPI)
        
        # PWM
        if os.path.exists('/sys/class/pwm'):
            peripherals.append(PeripheralType.PWM)
        
        # UART
        if os.path.exists('/dev/ttyS0') or os.path.exists('/dev/ttyUSB0'):
            peripherals.append(PeripheralType.UART)
        
        return peripherals
    
    def initialize(self) -> bool:
        """初始化 RK3588 平台"""
        if self._initialized:
            return True
        
        try:
            # 检查 NPU 驱动
            npu_ready = self._check_npu_driver()
            
            # 检查权限
            self._check_permissions()
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"RK3588 initialization failed: {e}")
            return False
    
    def _check_npu_driver(self) -> bool:
        """检查 NPU 驱动状态"""
        # 瑞芯微 NPU 设备节点
        npu_dev_paths = [
            '/dev/npu',
            '/dev/rknpu',
            '/dev/misc/rknpu',
        ]
        for path in npu_dev_paths:
            if os.path.exists(path):
                return True
        return False
    
    def _check_permissions(self) -> None:
        """检查并提示权限问题"""
        # 检查用户组
        try:
            result = subprocess.run(
                ['groups'],
                capture_output=True,
                text=True,
                timeout=5
            )
            groups = result.stdout.strip().split()
            
            # GPIO 权限检查
            if os.path.exists('/dev/gpiochip0'):
                if 'gpio' not in groups and 'dialout' not in groups:
                    print("Warning: Consider adding user to 'gpio' or 'dialout' group")
        except:
            pass
    
    def shutdown(self) -> None:
        """关闭 RK3588 平台"""
        self._initialized = False
        # RK3588 不需要特殊清理
    
    def get_npu_utilization(self) -> float:
        """获取 NPU 利用率 (0-1)"""
        # 瑞芯微 NPU 利用率接口
        npu_util_paths = [
            '/sys/class/devfreq/ff9a0000.npu/load',
            '/sys/bus/platform/devices/ff9a0000.npu/devfreq/ff9a0000.npu/load',
        ]
        for path in npu_util_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return int(f.read().strip()) / 100.0
                except:
                    pass
        return 0.0
    
    def get_gpu_utilization(self) -> float:
        """获取 GPU 利用率 (0-1)"""
        gpu_util_paths = [
            '/sys/class/devfreq/ff9a0000.gpu/load',
            '/sys/bus/platform/devices/ff9a0000.gpu/devfreq/ff9a0000.gpu/load',
        ]
        for path in gpu_util_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return int(f.read().strip()) / 100.0
                except:
                    pass
        return 0.0
    
    def enable_npu(self) -> bool:
        """启用 NPU"""
        try:
            # 设置 NPU 性能模式
            governor_path = '/sys/class/devfreq/ff9a0000.npu/governor'
            if os.path.exists(governor_path):
                with open(governor_path, 'w') as f:
                    f.write('performance')
                return True
        except:
            pass
        return False
    
    def set_npu_freq(self, freq_mhz: int) -> bool:
        """设置 NPU 频率 (MHz)"""
        try:
            freq_path = '/sys/class/devfreq/ff9a0000.npu/min_freq'
            if os.path.exists(freq_path):
                with open(freq_path, 'w') as f:
                    f.write(str(freq_mhz * 1000000))
                return True
        except:
            pass
        return False
