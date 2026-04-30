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
CrossModalCalibration: 跨模态联合标定
=======================================

功能:
- 触觉-力觉联合标定 (接触力与压力分布对应关系)
- IMU-视觉联合标定 (相机外参与IMU姿态同步)
- 力觉-IMU联合标定 (力矩与姿态角对应)
- 标定数据质量评估
- AGV五级标定规格

标定流程:
1. 静止标定 (重力方向, 零偏估计)
2. 已知姿态标定 (多位置力矩观测)
3. 接触力验证 (推拉力计对照)
4. 温度漂移建模
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass


@dataclass
class CalibrationDataPoint:
    """单个标定数据点"""
    # 传感器原始数据
    tactile_pressure: Optional[np.ndarray] = None   # 触觉压力图
    force_wrench: Optional[np.ndarray] = None       # 6D力旋量
    imu_accel: Optional[np.ndarray] = None          # 3D加速度
    imu_gyro: Optional[np.ndarray] = None          # 3D角速度
    imu_euler: Optional[np.ndarray] = None         # 3D欧拉角
    
    # 环境参数
    temperature: float = 25.0                      # 温度 (°C)
    timestamp: float = 0.0
    
    # 标定物真值 (如果有)
    known_force: Optional[np.ndarray] = None       # 已知力 (N)
    known_torque: Optional[np.ndarray] = None      # 已知力矩 (N·m)
    known_orientation: Optional[np.ndarray] = None # 已知姿态 (roll, pitch, yaw)


@dataclass
class CalibrationResult:
    """标定结果"""
    # 触觉→力觉转换矩阵
    tactile_to_force_matrix: Optional[np.ndarray] = None  # (6, N) N=触觉特征维
    
    # IMU→姿态矩阵
    imu_to_orientation_matrix: Optional[np.ndarray] = None
    
    # 零偏估计
    force_bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 比例因子
    force_scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    accel_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    # 温度补偿系数
    temp_coefficient_force: Optional[np.ndarray] = None  # (6,) 每度漂移
    temp_coefficient_accel: Optional[np.ndarray] = None  # (3,)
    
    # 标定质量
    residual_force: float = 0.0     # 力残差 RMS
    residual_orientation: float = 0.0  # 姿态残差 RMS
    r_squared_force: float = 0.0    # R² 决定系数
    r_squared_orientation: float = 0.0
    
    # 元数据
    num_samples: int = 0
    temperature_range: Tuple[float, float] = (25.0, 25.0)
    grade: str = 'M'
    timestamp: float = 0.0


class CrossModalCalibrator:
    """
    跨模态联合标定器
    
    解决的核心问题:
    1. 触觉阵列测量的压力分布 → 实际接触力/力矩
    2. IMU测量的加速度/角速度 → 精确姿态角
    3. 力传感器零偏和比例因子
    """
    
    def __init__(
        self,
        tactile_size: Tuple[int, int] = (16, 16),
        grade: str = 'M'
    ):
        self.tactile_size = tactile_size
        self.grade = grade
        self.data_points: List[CalibrationDataPoint] = []
        self._calibration_result: Optional[CalibrationResult] = None
        
    def add_static_calibration(
        self,
        force_wrench: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        temperature: float = 25.0
    ):
        """
        添加静止标定点 (用于零偏估计)
        
        Args:
            force_wrench: 6D力旋量 [Fx, Fy, Fz, Tx, Ty, Tz]
            accel: 3D加速度
            gyro: 3D角速度
            temperature: 当前温度
        """
        dp = CalibrationDataPoint(
            force_wrench=force_wrench.copy(),
            imu_accel=accel.copy(),
            imu_gyro=gyro.copy(),
            temperature=temperature,
            timestamp=0.0
        )
        self.data_points.append(dp)
    
    def add_oriented_calibration(
        self,
        tactile_pressure: np.ndarray,
        force_wrench: np.ndarray,
        imu_euler: np.ndarray,
        imu_accel: np.ndarray,
        known_force: Optional[np.ndarray] = None,
        known_torque: Optional[np.ndarray] = None,
        temperature: float = 25.0
    ):
        """
        添加已知姿态标定点 (用于比例因子标定)
        
        Args:
            tactile_pressure: 触觉压力图
            force_wrench: 6D力旋量
            imu_euler: IMU欧拉角
            imu_accel: IMU加速度
            known_force/torque: 推拉力计真值 (如果有)
            temperature: 当前温度
        """
        dp = CalibrationDataPoint(
            tactile_pressure=tactile_pressure.copy(),
            force_wrench=force_wrench.copy(),
            imu_accel=accel.copy() if (accel := imu_accel) is not None else None,
            imu_euler=imu_euler.copy() if imu_euler is not None else None,
            known_force=known_force.copy() if known_force is not None else None,
            known_torque=known_torque.copy() if known_torque is not None else None,
            temperature=temperature,
            timestamp=0.0
        )
        self.data_points.append(dp)
    
    def calibrate_force_bias(self) -> np.ndarray:
        """
        标定力传感器零偏 (静止数据)
        
        方法: 最小二乘法估计零偏
        """
        if not self.data_points:
            return np.zeros(6)
        
        # 取静止数据的均值作为零偏
        biases = []
        for dp in self.data_points:
            if dp.force_wrench is not None and dp.known_force is None:
                # 假设静止时应只有重力分量
                biases.append(dp.force_wrench)
        
        if not biases:
            return np.zeros(6)
        
        bias = np.mean(biases, axis=0)
        
        if self._calibration_result is None:
            self._calibration_result = CalibrationResult(grade=self.grade)
        
        self._calibration_result.force_bias = bias
        return bias
    
    def calibrate_tactile_to_force(
        self,
        regularization: float = 1e-6
    ) -> np.ndarray:
        """
        标定触觉→力觉转换矩阵
        
        方法: 线性回归 + 正则化
        
        物理意义:
        - 压力图 flatten() 后作为特征
        - 力旋量作为目标
        - M * tactile_features = force_wrench
        
        Returns:
            tactile_to_force_matrix: (6, N) 转换矩阵
        """
        if not self.data_points:
            return np.zeros((6, self.tactile_size[0] * self.tactile_size[1]))
        
        # 收集有效数据
        tactile_features = []
        force_targets = []
        
        for dp in self.data_points:
            if dp.tactile_pressure is not None and dp.force_wrench is not None:
                features = dp.tactile_pressure.flatten()
                tactile_features.append(features)
                force_targets.append(dp.force_wrench)
        
        if len(tactile_features) < 3:
            return np.zeros((6, self.tactile_size[0] * self.tactile_size[1]))
        
        X = np.array(tactile_features)  # (N, D)
        Y = np.array(force_targets)       # (N, 6)
        
        # 最小二乘法: M = (XᵀX + λI)⁻¹ XᵀY
        XtX = X.T @ X
        XtX_reg = XtX + regularization * np.eye(XtX.shape[0])
        XtY = X.T @ Y
        M = np.linalg.solve(XtX_reg, XtY).T  # (6, D)
        
        # 计算残差
        Y_pred = X @ M.T
        residuals = Y - Y_pred
        residual_rms = np.sqrt(np.mean(residuals**2))
        
        # R²
        Y_mean = np.mean(Y, axis=0)
        ss_tot = np.sum((Y - Y_mean)**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        if self._calibration_result is None:
            self._calibration_result = CalibrationResult(grade=self.grade)
        
        self._calibration_result.tactile_to_force_matrix = M
        self._calibration_result.residual_force = float(residual_rms)
        self._calibration_result.r_squared_force = float(np.mean(r_squared))
        self._calibration_result.num_samples = len(tactile_features)
        
        return M
    
    def calibrate_imu_orientation(
        self,
        gravity_reference: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        标定IMU→姿态转换
        
        方法: 多位置姿态观测回归
        
        Returns:
            (imu_to_euler_matrix, accel_bias)
        """
        if gravity_reference is None:
            gravity_reference = np.array([0.0, 0.0, 9.81])
        
        if not self.data_points:
            return np.eye(3), np.zeros(3)
        
        # 收集加速度和欧拉角数据
        accel_data = []
        euler_data = []
        
        for dp in self.data_points:
            if dp.imu_accel is not None and dp.imu_euler is not None:
                accel_data.append(dp.imu_accel)
                euler_data.append(dp.imu_euler)
        
        if len(accel_data) < 3:
            return np.eye(3), np.zeros(3)
        
        accel_arr = np.array(accel_data)    # (N, 3)
        euler_arr = np.array(euler_data)     # (N, 3)
        
        # 加速度计零偏估计 (静止数据的均值)
        accel_bias = np.mean(accel_arr, axis=0) - gravity_reference
        
        # 去偏
        accel_centered = accel_arr - accel_bias
        
        # 简化为比例因子估计
        accel_scale = np.linalg.norm(gravity_reference) / (
            np.linalg.norm(accel_centered, axis=1, keepdims=True) + 1e-6
        )
        accel_scale_mean = np.mean(accel_scale)
        
        accel_scale_vec = np.array([accel_scale_mean, accel_scale_mean, accel_scale_mean])
        
        # 计算残差
        euler_pred = np.zeros_like(euler_arr)
        for i in range(len(accel_data)):
            # 简化的加速度→姿态转换
            accel_n = accel_centered[i] / (np.linalg.norm(accel_centered[i]) + 1e-6)
            roll = np.arctan2(accel_n[1], accel_n[2])
            pitch = np.arcsin(-accel_n[0])
            euler_pred[i] = [roll, pitch, 0.0]
        
        residuals = euler_arr - euler_pred
        residual_rms = np.sqrt(np.mean(residuals**2))
        
        if self._calibration_result is None:
            self._calibration_result = CalibrationResult(grade=self.grade)
        
        self._calibration_result.accel_bias = accel_bias
        self._calibration_result.accel_scale = accel_scale_vec
        self._calibration_result.residual_orientation = float(residual_rms)
        
        return np.eye(3), accel_bias
    
    def calibrate_full(self) -> CalibrationResult:
        """
        执行完整标定流程
        
        1. 力零偏估计
        2. 触觉→力觉转换
        3. IMU→姿态转换
        4. 温度系数 (如果有温度变化数据)
        """
        print(f"[CrossModalCalibrator] Starting full calibration ({len(self.data_points)} samples)...")
        
        # 1. 零偏标定
        self.calibrate_force_bias()
        
        # 2. 触觉→力觉转换
        self.calibrate_tactile_to_force()
        
        # 3. IMU姿态标定
        self.calibrate_imu_orientation()
        
        # 4. 温度系数 (如果有温度变化)
        temps = [dp.temperature for dp in self.data_points]
        if len(set(temps)) > 1 and self._calibration_result is not None:
            self._calibrate_temp_coefficients()
        
        result = self._calibration_result
        print(f"[CrossModalCalibrator] Calibration complete:")
        print(f"  Force bias: {result.force_bias}")
        print(f"  Force residual RMS: {result.residual_force:.4f} N/Nm")
        print(f"  R² force: {result.r_squared_force:.4f}")
        print(f"  Orientation residual RMS: {result.residual_orientation:.4f} rad")
        
        return result
    
    def _calibrate_temp_coefficients(self):
        """标定温度漂移系数"""
        if len(self.data_points) < 10:
            return
        
        # 分离高低温度数据
        temps = np.array([dp.temperature for dp in self.data_points])
        temp_min, temp_max = np.min(temps), np.max(temps)
        
        if temp_max - temp_min < 5.0:  # 温度变化太小
            return
        
        # 估计温度系数 (简化)
        high_temp = temps > (temp_min + temp_max) / 2
        
        high_wrenches = np.array([dp.force_wrench for dp in self.data_points if dp.force_wrench is not None])
        if len(high_wrenches) < 2:
            return
        
        diff_per_degree = 0.001  # 简化估计: 每度漂移0.1%
        
        if self._calibration_result:
            self._calibration_result.temp_coefficient_force = np.ones(6) * diff_per_degree
            self._calibration_result.temperature_range = (float(temp_min), float(temp_max))
    
    def apply_calibration(
        self,
        tactile_pressure: Optional[np.ndarray] = None,
        force_wrench: Optional[np.ndarray] = None,
        imu_accel: Optional[np.ndarray] = None,
        temperature: float = 25.0
    ) -> Dict[str, np.ndarray]:
        """
        应用标定结果
        
        Returns:
            标定后的传感器数据
        """
        result = self._calibration_result
        if result is None:
            return {'error': 'No calibration result'}
        
        calibrated = {}
        
        # 触觉→力觉转换
        if tactile_pressure is not None and result.tactile_to_force_matrix is not None:
            features = tactile_pressure.flatten()
            force = result.tactile_to_force_matrix @ features
            calibrated['force_wrench'] = force
        
        # 力零偏补偿
        if force_wrench is not None:
            calibrated['force_wrench_calibrated'] = force_wrench - result.force_bias
        
        # IMU加速度补偿
        if imu_accel is not None:
            accel_cal = (imu_accel - result.accel_bias) * result.accel_scale
            
            # 温度补偿
            if result.temp_coefficient_accel is not None:
                delta_temp = temperature - 25.0
                accel_cal = accel_cal - result.temp_coefficient_accel * delta_temp
            
            calibrated['imu_accel_calibrated'] = accel_cal
        
        return calibrated
    
    def evaluate_quality(self) -> Dict[str, float]:
        """
        评估标定质量
        
        Returns:
            质量指标字典
        """
        if self._calibration_result is None:
            return {'error': 'No calibration result'}
        
        result = self._calibration_result
        
        # 综合评分
        score = 0.0
        
        # 力残差评分 (越小越好)
        if result.residual_force < 0.1:
            force_score = 1.0
        elif result.residual_force < 1.0:
            force_score = 1.0 - (result.residual_force - 0.1) / 0.9
        else:
            force_score = 0.0
        
        # 姿态残差评分
        if result.residual_orientation < 0.01:
            orient_score = 1.0
        elif result.residual_orientation < 0.1:
            orient_score = 1.0 - (result.residual_orientation - 0.01) / 0.09
        else:
            orient_score = 0.0
        
        # R² 评分
        r2_score = result.r_squared_force
        
        # 样本数量评分
        if result.num_samples >= 100:
            sample_score = 1.0
        elif result.num_samples >= 30:
            sample_score = 0.7
        else:
            sample_score = 0.3
        
        overall = 0.4 * force_score + 0.3 * orient_score + 0.3 * r2_score
        
        return {
            'overall_score': float(overall),
            'force_score': float(force_score),
            'orientation_score': float(orient_score),
            'r2_score': float(r2_score),
            'sample_score': float(sample_score),
            'residual_force': float(result.residual_force),
            'residual_orientation': float(result.residual_orientation),
            'r_squared_force': float(result.r_squared_force),
            'num_samples': result.num_samples,
        }
    
    def save(self, path: str):
        """保存标定结果到文件"""
        if self._calibration_result is None:
            raise ValueError("No calibration result to save")
        
        result = self._calibration_result
        data = {
            'tactile_to_force_matrix': result.tactile_to_force_matrix,
            'imu_to_orientation_matrix': result.imu_to_orientation_matrix,
            'force_bias': result.force_bias,
            'accel_bias': result.accel_bias,
            'gyro_bias': result.gyro_bias,
            'force_scale': result.force_scale,
            'accel_scale': result.accel_scale,
            'temp_coefficient_force': result.temp_coefficient_force,
            'temp_coefficient_accel': result.temp_coefficient_accel,
            'residual_force': result.residual_force,
            'residual_orientation': result.residual_orientation,
            'r_squared_force': result.r_squared_force,
            'num_samples': result.num_samples,
            'temperature_range': result.temperature_range,
            'grade': result.grade,
        }
        
        np.savez_compressed(path, **data)
        print(f"[CrossModalCalibrator] Saved calibration to {path}")
    
    @classmethod
    def load(cls, path: str, grade: str = 'M') -> 'CrossModalCalibrator':
        """从文件加载标定结果"""
        data = np.load(path)
        
        calibrator = cls(grade=grade)
        calibrator._calibration_result = CalibrationResult(
            tactile_to_force_matrix=data['tactile_to_force_matrix'],
            imu_to_orientation_matrix=data.get('imu_to_orientation_matrix'),
            force_bias=data['force_bias'],
            accel_bias=data['accel_bias'],
            gyro_bias=data.get('gyro_bias', np.zeros(3)),
            force_scale=data.get('force_scale', np.ones(6)),
            accel_scale=data.get('accel_scale', np.ones(3)),
            temp_coefficient_force=data.get('temp_coefficient_force'),
            temp_coefficient_accel=data.get('temp_coefficient_accel'),
            residual_force=float(data.get('residual_force', 0.0)),
            residual_orientation=float(data.get('residual_orientation', 0.0)),
            r_squared_force=float(data.get('r_squared_force', 0.0)),
            num_samples=int(data.get('num_samples', 0)),
            temperature_range=tuple(data.get('temperature_range', [25.0, 25.0])),
            grade=grade,
        )
        
        print(f"[CrossModalCalibrator] Loaded calibration from {path}")
        return calibrator


# ─── AGV五级标定规格 ────────────────────────────────────────────────────────

AGV_CALIBRATION_GRADES = {
    'S': {
        'min_static_samples': 50,
        'min_oriented_samples': 20,
        'force_accuracy_required': 1.0,     # N
        'orientation_accuracy_required': 0.1,  # rad
        'temp_range': 10.0,                # °C
        'calibration_time_min': 10,        # min
    },
    'M': {
        'min_static_samples': 100,
        'min_oriented_samples': 50,
        'force_accuracy_required': 0.5,
        'orientation_accuracy_required': 0.05,
        'temp_range': 15.0,
        'calibration_time_min': 15,
    },
    'L': {
        'min_static_samples': 200,
        'min_oriented_samples': 100,
        'force_accuracy_required': 0.2,
        'orientation_accuracy_required': 0.02,
        'temp_range': 20.0,
        'calibration_time_min': 20,
    },
    'XL': {
        'min_static_samples': 500,
        'min_oriented_samples': 200,
        'force_accuracy_required': 0.1,
        'orientation_accuracy_required': 0.01,
        'temp_range': 25.0,
        'calibration_time_min': 30,
    },
    'XXL': {
        'min_static_samples': 1000,
        'min_oriented_samples': 500,
        'force_accuracy_required': 0.05,
        'orientation_accuracy_required': 0.005,
        'temp_range': 30.0,
        'calibration_time_min': 60,
    },
}


def get_calibration_spec(grade: str) -> dict:
    """获取AGV指定等级的标定规格"""
    return AGV_CALIBRATION_GRADES.get(grade, AGV_CALIBRATION_GRADES['M'])
