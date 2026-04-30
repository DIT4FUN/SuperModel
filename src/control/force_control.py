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
力觉控制模块
============

基于力矩传感器的闭环控制:
- 恒力跟踪控制
- 力位混合控制
- 碰撞检测与响应
- 阻抗/导纳控制

集成:
- ForceTorqueSensor → ForceController
- Wrench → HybridForcePositionController
- ContactState → CollisionDetector
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.force import ForceTorqueSensor, Wrench, ContactState, ForceSensorType


@dataclass
class ForceControlParams:
    """力控参数"""
    # 导纳控制参数
    M_mass: float = 1.0        # 等效质量 kg
    D_damping: float = 5.0     # 等效阻尼 N·s/m
    K_stiffness: float = 100.0 # 等效刚度 N/m
    
    # 力控制增益
    Kp_force: float = 2.0
    Ki_force: float = 0.1
    Kd_force: float = 0.5
    
    # 碰撞检测
    collision_threshold: float = 50.0  # N
    collision_history_size: int = 50
    
    # 安全限制
    max_force: float = 100.0    # N
    max_torque: float = 20.0    # N·m
    
    # 控制频率
    control_rate: float = 100.0  # Hz
    
    # AGV等级
    grade: str = 'M'
    
    @classmethod
    def from_grade(cls, grade: str) -> 'ForceControlParams':
        configs = {
            'S':  cls(M_mass=0.5, D_damping=2.0, K_stiffness=50.0, Kp_force=1.0, control_rate=50, max_force=50, grade='S'),
            'M':  cls(M_mass=1.0, D_damping=5.0, K_stiffness=100.0, Kp_force=2.0, control_rate=100, max_force=100, grade='M'),
            'L':  cls(M_mass=2.0, D_damping=10.0, K_stiffness=200.0, Kp_force=3.0, control_rate=200, max_force=200, grade='L'),
            'XL': cls(M_mass=5.0, D_damping=20.0, K_stiffness=500.0, Kp_force=4.0, control_rate=500, max_force=500, grade='XL'),
            'XXL': cls(M_mass=10.0, D_damping=50.0, K_stiffness=1000.0, Kp_force=5.0, control_rate=1000, max_force=1000, grade='XXL'),
        }
        return configs.get(grade, cls())


class ForceController:
    """
    力觉控制器
    
    功能:
    - 恒力跟踪
    - 导纳控制
    - 碰撞检测
    """
    
    def __init__(
        self,
        force_sensor: ForceTorqueSensor,
        params: Optional[ForceControlParams] = None
    ):
        self.force_sensor = force_sensor
        self.params = params or ForceControlParams()
        
        self._force_error_integral = np.zeros(3)
        self._last_error = np.zeros(3)
        self._last_wrench: Optional[Wrench] = None
        self._collision_history: List[bool] = []
        
        # 碰撞状态
        self._in_collision = False
        self._collision_count = 0
        
    def compute_admittance(
        self,
        desired_force: np.ndarray,
        current_wrench: Optional[Wrench] = None,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        导纳控制: 将力误差转换为位置调整
        
        M * a + D * v + K * x = F_error
        
        Args:
            desired_force: 目标力向量 (3,)
            current_wrench: 当前力矩测量
            dt: 时间步长
            
        Returns:
            position_adjustment: 位置调整量 (3,)
        """
        if current_wrench is None:
            current_wrench = self.force_sensor.capture()
        
        # 当前力
        current_force = current_wrench.force.copy()
        
        # 力误差
        force_error = desired_force - current_force
        
        # 积分项
        self._force_error_integral += force_error * dt
        self._force_error_integral = np.clip(
            self._force_error_integral, -10, 10
        )
        
        # 微分项
        d_error = force_error - self._last_error
        self._last_error = force_error.copy()
        
        # PID 控制 (简化为 PD + 积分补偿)
        adjustment = (
            self.params.Kp_force * force_error +
            self.params.Ki_force * self._force_error_integral +
            self.params.Kd_force * d_error / dt
        )
        
        # 导纳模型: F = M*a + D*v + K*x
        # 简化: x = F / K (静态近似)
        position_adj = adjustment / (self.params.K_stiffness + 1e-6)
        
        return position_adj.astype(np.float32)
    
    def detect_collision(
        self,
        current_wrench: Optional[Wrench] = None,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        碰撞检测
        
        基于力信号突变检测碰撞
        
        Returns:
            is_collision: 是否检测到碰撞
            collision_magnitude: 碰撞力大小
        """
        if current_wrench is None:
            current_wrench = self.force_sensor.get_wrench()
        if current_wrench is None:
            return False, 0.0
        
        if threshold is None:
            threshold = self.params.collision_threshold
        
        magnitude = current_wrench.magnitude
        
        # 历史记录
        self._collision_history.append(magnitude > threshold)
        if len(self._collision_history) > self.params.collision_history_size:
            self._collision_history.pop(0)
        
        # 检测逻辑: 连续多次超阈值
        recent = self._collision_history[-5:]
        is_collision = sum(recent) >= 3
        
        if is_collision and not self._in_collision:
            self._collision_count += 1
            self._in_collision = True
        elif not is_collision:
            self._in_collision = False
        
        return is_collision, float(magnitude)
    
    def compute_collision_response(
        self,
        collision_direction: np.ndarray
    ) -> np.ndarray:
        """
        计算碰撞响应力
        
        碰撞时反向推力以减轻碰撞
        
        Args:
            collision_direction: 碰撞方向 (归一化)
            
        Returns:
            response_force: 响应力向量
        """
        magnitude = self._last_error[0] if np.any(self._last_error) else self.params.max_force
        
        # 反向推力 (减轻碰撞)
        response = -collision_direction * magnitude * 0.5
        
        return response.astype(np.float32)

    def reset(self):
        """重置控制器状态"""
        self._force_error_integral = np.zeros(3)
        self._last_error = np.zeros(3)
        self._last_wrench = None
        self._collision_history.clear()
        self._in_collision = False
        self._collision_count = 0


class HybridForcePositionController:
    """
    力位混合控制器
    
    同时控制力和位置:
    - 某些方向用力控
    - 其他方向用位置控
    """
    
    def __init__(
        self,
        force_sensor: ForceTorqueSensor,
        params: Optional[ForceControlParams] = None
    ):
        self.force = force_sensor
        self.params = params or ForceControlParams()
        self.force_controller = ForceController(force_sensor, params)
        
        # 混合控制配置
        # True = 力控, False = 位置控
        self.force_control_axes = np.array([True, True, False])  # x,y用力控, z用位置控
        
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        
    def compute_control(
        self,
        target_force: np.ndarray,
        target_position: np.ndarray,
        measured_wrench: Optional[Wrench] = None,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算混合控制输出
        
        Args:
            target_force: 目标力 (3,)
            target_position: 目标位置 (3,)
            measured_wrench: 当前力矩测量
            dt: 时间步长
            
        Returns:
            position_output: 位置控制输出
            force_output: 力控制输出
        """
        if measured_wrench is None:
            measured_wrench = self.force.capture()
        
        current_force = measured_wrench.force.copy()
        current_pos = self._position.copy()
        
        # --- 力控方向 ---
        force_adj = self.force_controller.compute_admittance(
            target_force, measured_wrench, dt
        )
        
        # --- 位置控方向 ---
        pos_error = target_position - current_pos
        pos_adj = self.params.Kp_force * pos_error
        
        # --- 混合 ---
        force_output = np.where(self.force_control_axes, force_adj, np.zeros(3))
        position_output = np.where(~self.force_control_axes, pos_adj, np.zeros(3))
        
        # 更新状态
        self._velocity = (target_position - current_pos) / (dt + 1e-6)
        self._position = target_position.copy()
        
        return position_output.astype(np.float32), force_output.astype(np.float32)

    def set_force_axes(self, axes: np.ndarray):
        """设置力控轴
        
        Args:
            axes: bool数组, True=力控, False=位置控
        """
        self.force_control_axes = np.asarray(axes, dtype=bool)

    def reset(self):
        """重置控制器状态"""
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self.force_controller.reset()


# AGV五级力控规格
AGV_FORCE_CONTROL_GRADES = {
    'S':  ForceControlParams.from_grade('S'),
    'M':  ForceControlParams.from_grade('M'),
    'L':  ForceControlParams.from_grade('L'),
    'XL': ForceControlParams.from_grade('XL'),
    'XXL': ForceControlParams.from_grade('XXL'),
}


def get_force_control_spec(grade: str) -> ForceControlParams:
    """获取AGV指定等级的力控参数"""
    return AGV_FORCE_CONTROL_GRADES.get(grade, AGV_FORCE_CONTROL_GRADES['M'])
