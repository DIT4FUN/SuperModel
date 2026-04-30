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
触觉控制模块
============

基于触觉传感器的闭环控制:
- 触觉导引抓取
- 滑移检测与 reactive control
- 接触力估计与调节
- 灵巧手操作控制

集成:
- TactileArray → TactileServoController
- TactileFrame → GraspQualityController
- SlipSignal → ReactiveSlipController
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import TactileArray, TactileFrame, TactileContact


@dataclass
class TactileServoParams:
    """触觉伺服控制参数"""
    # 位置控制增益
    Kp_position: float = 2.0
    Kd_velocity: float = 0.5
    
    # 力控增益
    Kp_force: float = 1.0
    Kd_force: float = 0.2
    
    # 触觉阈值
    contact_threshold: float = 0.1     # 接触检测阈值
    slip_threshold: float = 0.3         # 滑移检测阈值
    max_force: float = 20.0             # 最大接触力 N
    
    # 抓取质量阈值
    grasp_quality_threshold: float = 0.6
    
    # 控制频率
    control_rate: float = 50.0  # Hz
    
    # AGV五级配置
    grade: str = 'M'
    
    @classmethod
    def from_grade(cls, grade: str) -> 'TactileServoParams':
        """从AGV等级获取默认参数"""
        configs = {
            'S':  cls(Kp_position=1.0, Kp_force=0.5, control_rate=30, grade='S'),
            'M':  cls(Kp_position=2.0, Kp_force=1.0, control_rate=50, grade='M'),
            'L':  cls(Kp_position=3.0, Kp_force=2.0, control_rate=100, grade='L'),
            'XL': cls(Kp_position=4.0, Kp_force=3.0, control_rate=200, grade='XL'),
            'XXL': cls(Kp_position=5.0, Kp_force=4.0, control_rate=500, grade='XXL'),
        }
        return configs.get(grade, cls())


class TactileServoController:
    """
    触觉伺服控制器
    
    功能:
    - 基于触觉反馈的位置/速度控制
    - 接触力估计与调节
    - 抓取质量监控
    """
    
    def __init__(
        self,
        tactile_sensor: TactileArray,
        params: Optional[TactileServoParams] = None
    ):
        self.tactile = tactile_sensor
        self.params = params or TactileServoParams()
        
        self._last_contact: Optional[TactileContact] = None
        self._target_position = np.zeros(3)
        self._current_position = np.zeros(3)
        self._last_error = np.zeros(3)
        
        # 抓取状态
        self._grasp_quality_history: List[float] = []
        self._is_grasping = False
        
    def compute_control_signal(
        self,
        target_force: float,
        current_frame: Optional[TactileFrame] = None
    ) -> np.ndarray:
        """
        计算控制信号 (位置增量)
        
        Args:
            target_force: 目标接触力 N
            current_frame: 当前触觉帧
            
        Returns:
            control_signal: 3D 控制增量 (dx, dy, dz)
        """
        if current_frame is None:
            current_frame = self.tactile.capture()
        
        contacts = self.tactile.detect_contacts(current_frame)
        
        if not contacts:
            # 无接触 - 搜索模式
            self._is_grasping = False
            return np.zeros(3)
        
        # 取最大接触
        best_contact = max(contacts, key=lambda c: c.peak_pressure)
        self._last_contact = best_contact
        self._is_grasping = True
        
        # 接触力误差
        force_error = target_force - best_contact.total_force
        
        # 接触中心位置 (归一化 → 世界坐标)
        cx, cy = best_contact.centroid  # row, col
        h, w = current_frame.pressure_map.shape
        contact_pos = np.array([
            cx / h - 0.5,   # x: row → x (归一化偏移)
            cy / w - 0.5,   # y: col → y
            0.0             # z: 深度未知，假设固定
        ], dtype=np.float32)
        
        # 位置误差
        pos_error = self._target_position - contact_pos
        
        # PD 控制
        d_error = pos_error - self._last_error
        control = (
            self.params.Kp_position * pos_error +
            self.params.Kd_velocity * d_error +
            self.params.Kp_force * force_error * np.array([0, 0, 1])
        )
        
        self._last_error = pos_error.copy()
        
        # 抓取质量监控
        quality = self.tactile.estimate_grip_quality(current_frame)
        self._grasp_quality_history.append(quality['overall'])
        if len(self._grasp_quality_history) > 100:
            self._grasp_quality_history.pop(0)
        
        return control.astype(np.float32)
    
    def detect_and_react_slip(self, current_frame: Optional[TactileFrame] = None) -> np.ndarray:
        """
        滑移检测与 reactive 控制
        
        检测到滑移时自动增加抓取力
        
        Returns:
            reactive_torque: 补偿滑移的力矩增量
        """
        if current_frame is None:
            current_frame = self.tactile.capture()
        
        slip_signal = self.tactile.get_slip_signal(current_frame)
        max_slip = np.max(slip_signal)
        
        if max_slip > self.params.slip_threshold:
            # 滑移检测 - 增加抓取力 (reactive)
            slip_ratio = (max_slip - self.params.slip_threshold) / (1.0 - self.params.slip_threshold)
            
            # 计算补偿力 - 沿接触法向增加力
            contacts = self.tactile.detect_contacts(current_frame)
            if contacts:
                best = max(contacts, key=lambda c: c.peak_pressure)
                normal_direction = np.array([0, 0, 1])  # 假设法向为Z
                compensation = slip_ratio * self.params.max_force * normal_direction
                return compensation.astype(np.float32)
        
        return np.zeros(3, dtype=np.float32)
    
    def monitor_grasp_quality(self) -> Dict[str, float]:
        """监控抓取质量"""
        if not self._grasp_quality_history:
            return {'current': 0.0, 'average': 0.0, 'trend': 0.0, 'stable': True}
        
        current = self._grasp_quality_history[-1]
        average = np.mean(self._grasp_quality_history[-10:])
        
        if len(self._grasp_quality_history) >= 5:
            trend = self._grasp_quality_history[-1] - self._grasp_quality_history[-5]
        else:
            trend = 0.0
        
        return {
            'current': float(current),
            'average': float(average),
            'trend': float(trend),
            'stable': abs(trend) < 0.05
        }
    
    def is_contact(self, current_frame: Optional[TactileFrame] = None) -> bool:
        """检测是否有接触"""
        if current_frame is None:
            current_frame = self.tactile.capture()
        contacts = self.tactile.detect_contacts(current_frame)
        return len(contacts) > 0

    def reset(self):
        """重置控制器状态"""
        self._last_contact = None
        self._target_position = np.zeros(3)
        self._current_position = np.zeros(3)
        self._last_error = np.zeros(3)
        self._grasp_quality_history.clear()
        self._is_grasping = False


class GraspQualityController:
    """
    抓取质量评估与控制
    
    实时评估抓取质量并调节:
    - 接触面积
    - 压力均匀性
    - 抓取稳定性
    """
    
    def __init__(
        self,
        tactile_sensor: TactileArray,
        target_quality: float = 0.7
    ):
        self.tactile = tactile_sensor
        self.target_quality = target_quality
        
        self._quality_history: List[float] = []
        
    def evaluate_and_regulate(
        self,
        current_frame: Optional[TactileFrame] = None
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        评估质量并返回调节指令
        
        Returns:
            quality_metrics: 质量指标
            adjustment: 调节量 (正=增加力, 负=减小力)
        """
        if current_frame is None:
            current_frame = self.tactile.capture()
        
        quality = self.tactile.estimate_grip_quality(current_frame)
        self._quality_history.append(quality['overall'])
        
        # 误差
        error = self.target_quality - quality['overall']
        
        # 调节量
        if quality['overall'] < self.target_quality:
            # 质量不足 - 增加接触面积或压力
            adjustment = np.array([0, 0, abs(error) * 10.0], dtype=np.float32)
        elif quality['contact_area'] > 0.8:
            # 接触面积过大 - 减小压力
            adjustment = np.array([0, 0, -error * 5.0], dtype=np.float32)
        else:
            adjustment = np.zeros(3, dtype=np.float32)
        
        return quality, adjustment


# AGV五级触觉控制规格
AGV_TACTILE_CONTROL_GRADES = {
    'S':  TactileServoParams.from_grade('S'),
    'M':  TactileServoParams.from_grade('M'),
    'L':  TactileServoParams.from_grade('L'),
    'XL': TactileServoParams.from_grade('XL'),
    'XXL': TactileServoParams.from_grade('XXL'),
}


def get_tactile_control_spec(grade: str) -> TactileServoParams:
    """获取AGV指定等级的触觉控制参数"""
    return AGV_TACTILE_CONTROL_GRADES.get(grade, AGV_TACTILE_CONTROL_GRADES['M'])
