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
GPIO 控制器
============

提供统一的 GPIO 接口，支持:
- Linux sysfs GPIO
- 字符设备 GPIO (kernel 4.8+)
- 模拟 PWM 输出

注意: 需要 root 权限或用户加入 gpio 组
"""

import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable


class PinMode(Enum):
    """引脚模式"""
    INPUT = "in"
    OUTPUT = "out"
    INPUT_PULLUP = "in_pullup"
    INPUT_PULLDOWN = "in_pulldown"
    PWM = "pwm"
    DISABLED = "disabled"


class PinState(Enum):
    """引脚状态"""
    LOW = 0
    HIGH = 1
    UNKNOWN = -1


@dataclass
class PinInfo:
    """引脚信息"""
    pin: int
    name: str
    mode: PinMode
    state: PinState
    chip: Optional[str] = None


class GPIOController:
    """
    GPIO 控制器
    
    提供跨平台的 GPIO 操作接口。
    """
    
    def __init__(self, board: 'BoardBase'):
        """
        初始化 GPIO 控制器
        
        Args:
            board: 主板实例
        """
        self.board = board
        self._exported_pins: dict = {}
        self._pwm_enabled: dict = {}
        self._callbacks: dict = {}
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    @property
    def available(self) -> bool:
        """GPIO 是否可用"""
        return os.path.exists('/dev/gpiochip0') or os.path.exists('/sys/class/gpio')
    
    def export_pin(self, pin: int) -> bool:
        """
        导出 GPIO 引脚
        
        Args:
            pin: GPIO 编号
            
        Returns:
            是否成功
        """
        if pin in self._exported_pins:
            return True
        
        # 尝试 sysfs 方式
        export_path = '/sys/class/gpio/export'
        if os.path.exists(export_path):
            try:
                with open(export_path, 'w') as f:
                    f.write(str(pin))
                self._exported_pins[pin] = True
                return True
            except PermissionError:
                print(f"Permission denied: {export_path}")
                return False
            except Exception as e:
                print(f"Failed to export GPIO {pin}: {e}")
                return False
        
        return False
    
    def unexport_pin(self, pin: int) -> bool:
        """
        取消导出 GPIO 引脚
        
        Args:
            pin: GPIO 编号
            
        Returns:
            是否成功
        """
        if pin not in self._exported_pins:
            return True
        
        unexport_path = '/sys/class/gpio/unexport'
        if os.path.exists(unexport_path):
            try:
                with open(unexport_path, 'w') as f:
                    f.write(str(pin))
                del self._exported_pins[pin]
                return True
            except:
                return False
        
        return False
    
    def set_mode(self, pin: int, mode: PinMode) -> bool:
        """
        设置引脚模式
        
        Args:
            pin: GPIO 编号
            mode: 引脚模式
            
        Returns:
            是否成功
        """
        if not self.export_pin(pin):
            return False
        
        direction_path = f'/sys/class/gpio/gpio{pin}/direction'
        if os.path.exists(direction_path):
            try:
                if mode == PinMode.INPUT:
                    direction = 'in'
                elif mode == PinMode.OUTPUT:
                    direction = 'out'
                elif mode == PinMode.INPUT_PULLUP:
                    # Linux 不直接支持 pullup，设为 input 后软件上拉
                    direction = 'in'
                elif mode == PinMode.INPUT_PULLDOWN:
                    direction = 'in'
                else:
                    direction = 'in'
                
                with open(direction_path, 'w') as f:
                    f.write(direction)
                
                self._exported_pins[pin] = {'mode': mode}
                return True
            except:
                return False
        
        return False
    
    def write(self, pin: int, state: PinState) -> bool:
        """
        写入引脚状态
        
        Args:
            pin: GPIO 编号
            state: 引脚状态
            
        Returns:
            是否成功
        """
        if pin not in self._exported_pins:
            self.set_mode(pin, PinMode.OUTPUT)
        
        value_path = f'/sys/class/gpio/gpio{pin}/value'
        if os.path.exists(value_path):
            try:
                with open(value_path, 'w') as f:
                    f.write('1' if state == PinState.HIGH else '0')
                return True
            except:
                return False
        
        return False
    
    def read(self, pin: int) -> PinState:
        """
        读取引脚状态
        
        Args:
            pin: GPIO 编号
            
        Returns:
            引脚状态
        """
        if pin not in self._exported_pins:
            if not self.set_mode(pin, PinMode.INPUT):
                return PinState.UNKNOWN
        
        value_path = f'/sys/class/gpio/gpio{pin}/value'
        if os.path.exists(value_path):
            try:
                with open(value_path, 'r') as f:
                    value = int(f.read().strip())
                return PinState.HIGH if value == 1 else PinState.LOW
            except:
                return PinState.UNKNOWN
        
        return PinState.UNKNOWN
    
    def toggle(self, pin: int) -> Optional[PinState]:
        """
        切换引脚状态
        
        Args:
            pin: GPIO 编号
            
        Returns:
            切换后的状态
        """
        current = self.read(pin)
        new_state = PinState.HIGH if current == PinState.LOW else PinState.LOW
        if self.write(pin, new_state):
            return new_state
        return None
    
    def list_exported_pins(self) -> List[int]:
        """列出已导出的 GPIO"""
        gpio_path = '/sys/class/gpio'
        if os.path.exists(gpio_path):
            try:
                return [int(d.replace('gpio', '')) 
                       for d in os.listdir(gpio_path) 
                       if d.startswith('gpio')]
            except:
                pass
        return []
    
    def cleanup(self) -> None:
        """清理所有已导出的 GPIO"""
        for pin in list(self._exported_pins.keys()):
            self.unexport_pin(pin)
        self._exported_pins.clear()
        self._pwm_enabled.clear()
    
    def __del__(self):
        self.cleanup()


# PWM 控制 (软件 PWM via sysfs)
class PWMController:
    """
    PWM 控制器
    
    注意: 需要硬件 PWM 支持或 pwmchip 驱动
    """
    
    def __init__(self, board: 'BoardBase'):
        self.board = board
        self._enabled_channels = set()
    
    @property
    def available(self) -> bool:
        """PWM 是否可用"""
        return os.path.exists('/sys/class/pwm')
    
    def export_channel(self, channel: int) -> bool:
        """导出 PWM 通道"""
        export_path = '/sys/class/pwm/pwmchip0/export'
        if os.path.exists(export_path):
            try:
                with open(export_path, 'w') as f:
                    f.write(str(channel))
                return True
            except:
                return False
        return False
    
    def set_duty_cycle(self, channel: int, duty_ns: int, period_ns: int = 20000000) -> bool:
        """
        设置 PWM 占空比
        
        Args:
            channel: PWM 通道
            duty_ns: 高电平时间 (纳秒)
            period_ns: 周期 (纳秒), 默认 20ms (50Hz)
            
        Returns:
            是否成功
        """
        base = f'/sys/class/pwm/pwmchip0/pwm{channel}'
        
        # 设置周期
        period_path = f'{base}/period'
        if os.path.exists(period_path):
            try:
                with open(period_path, 'w') as f:
                    f.write(str(period_ns))
            except:
                return False
        
        # 设置占空比
        duty_path = f'{base}/duty_cycle'
        if os.path.exists(duty_path):
            try:
                with open(duty_path, 'w') as f:
                    f.write(str(duty_ns))
                return True
            except:
                return False
        
        return False
    
    def enable(self, channel: int) -> bool:
        """启用 PWM"""
        enable_path = f'/sys/class/pwm/pwmchip0/pwm{channel}/enable'
        if os.path.exists(enable_path):
            try:
                with open(enable_path, 'w') as f:
                    f.write('1')
                self._enabled_channels.add(channel)
                return True
            except:
                return False
        return False
    
    def disable(self, channel: int) -> bool:
        """禁用 PWM"""
        enable_path = f'/sys/class/pwm/pwmchip0/pwm{channel}/enable'
        if os.path.exists(enable_path):
            try:
                with open(enable_path, 'w') as f:
                    f.write('0')
                self._enabled_channels.discard(channel)
                return True
            except:
                return False
        return False
