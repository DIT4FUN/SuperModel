"""
工具模块测试
============

测试通用工具函数:
- 数据验证
- 坐标变换
- 信号处理
- 数值计算
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from utils import (
    validate_vector, validate_matrix, clamp, clamp_vector,
    euler_to_rotation_matrix, rotation_matrix_to_euler,
    quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    pose_to_transform_matrix, transform_pointcloud,
    moving_average, exponential_moving_average, lowpass_filter,
    normalize, derivative, integral,
    wrap_angle, wrap_angle_deg, interpolate_linear, interpolate_cubic,
    smooth_step, trajectory_generator_linear,
    point_to_line_distance, closest_point_on_trajectory,
    deg_to_rad, rad_to_deg, mps_to_rpm, rpm_to_mps,
    rpm_to_radps, radps_to_rpm
)


class TestDataValidation(unittest.TestCase):
    """测试数据验证工具"""
    
    def test_validate_vector(self):
        vec = np.array([1.0, 2.0, 3.0])
        result = validate_vector(vec, 3, "test_vec")
        self.assertEqual(result.shape, (3,))
        
        with self.assertRaises(ValueError):
            validate_vector(vec, 5, "test_vec")
    
    def test_validate_matrix(self):
        mat = np.eye(3)
        result = validate_matrix(mat, (3, 3), "test_mat")
        self.assertEqual(result.shape, (3, 3))
        
        with self.assertRaises(ValueError):
            validate_matrix(mat, (4, 4), "test_mat")
    
    def test_clamp(self):
        self.assertEqual(clamp(5.0, 0.0, 10.0), 5.0)
        self.assertEqual(clamp(-5.0, 0.0, 10.0), 0.0)
        self.assertEqual(clamp(15.0, 0.0, 10.0), 10.0)
    
    def test_clamp_vector(self):
        vec = np.array([3.0, 4.0])  # norm = 5
        result = clamp_vector(vec, 10.0)
        np.testing.assert_array_almost_equal(result, vec)  # 5 < 10, unchanged
        
        result = clamp_vector(vec, 3.0)
        np.testing.assert_array_almost_equal(result, vec * 0.6)


class TestCoordinateTransform(unittest.TestCase):
    """测试坐标变换工具"""
    
    def test_euler_roundtrip(self):
        """欧拉角 -> 旋转矩阵 -> 欧拉角"""
        euler_orig = np.array([0.5, 0.3, 0.8])
        R = euler_to_rotation_matrix(*euler_orig)
        euler_new = rotation_matrix_to_euler(R)
        np.testing.assert_array_almost_equal(euler_orig, euler_new, decimal=5)
    
    def test_quaternion_rotation_matrix(self):
        """四元数转旋转矩阵的基本性质"""
        # 使用单位四元数 (无旋转)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))
        
        # 使用90度绕Z轴的四元数
        q90 = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        R90 = quaternion_to_rotation_matrix(q90)
        # 验证90度旋转: R*[1,0,0] 应该接近 [0,1,0]
        v = np.array([1.0, 0.0, 0.0])
        v_rot = R90 @ v
        np.testing.assert_array_almost_equal(v_rot, np.array([0.0, 1.0, 0.0]), decimal=5)
    
    def test_pose_to_transform(self):
        """位姿 -> 变换矩阵"""
        pos = np.array([1.0, 2.0, 3.0])
        ori = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
        T = pose_to_transform_matrix(pos, ori)
        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
    
    def test_transform_pointcloud(self):
        """点云变换"""
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)
        T[0, 3] = 1.0  # 平移x+1
        result = transform_pointcloud(points, T)
        np.testing.assert_array_almost_equal(
            result,
            np.array([[2, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=np.float32)
        )


class TestSignalProcessing(unittest.TestCase):
    """测试信号处理工具"""
    
    def test_moving_average(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(data, 3)
        np.testing.assert_array_almost_equal(
            result,
            np.array([2.0, 3.0, 4.0])
        )
    
    def test_exponential_moving_average(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = exponential_moving_average(data, 0.5)
        self.assertEqual(len(result), len(data))
        self.assertGreater(result[-1], result[0])
    
    def test_lowpass_filter(self):
        """低通滤波 - 直流信号应该保持"""
        data = np.array([1.0, 1.0, 1.0, 1.0])
        result = lowpass_filter(data, 100.0, 1000.0)
        np.testing.assert_array_almost_equal(result, data, decimal=2)
    
    def test_normalize(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = normalize(data, -1.0, 1.0)
        self.assertAlmostEqual(np.min(result), -1.0)
        self.assertAlmostEqual(np.max(result), 1.0)
    
    def test_derivative(self):
        """导数 - 匀速运动应该得到常数速度"""
        t = np.linspace(0, 1, 11)
        position = 5.0 * t  # 速度=5
        velocity = derivative(position, 0.1)
        np.testing.assert_array_almost_equal(velocity, np.full(11, 5.0))
    
    def test_integral(self):
        """积分 - 常数应该得到线性增长"""
        velocity = np.ones(10)
        position = integral(velocity, 0.1)
        # 积分结果: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        np.testing.assert_array_almost_equal(position, expected)


class TestNumericalComputation(unittest.TestCase):
    """测试数值计算工具"""
    
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3.5 * np.pi), -0.5 * np.pi)
        self.assertAlmostEqual(wrap_angle(-3.5 * np.pi), 0.5 * np.pi)
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
    
    def test_wrap_angle_deg(self):
        self.assertAlmostEqual(wrap_angle_deg(270.0), -90.0)
        self.assertAlmostEqual(wrap_angle_deg(-270.0), 90.0)
    
    def test_interpolate_linear(self):
        result = interpolate_linear(0.0, 10.0, 0.5)
        self.assertAlmostEqual(result, 5.0)
    
    def test_interpolate_cubic(self):
        """三次插值 - 端点速度为零时应平滑通过端点"""
        result = interpolate_cubic(0.0, 10.0, 0.0, 0.0, 0.5)
        self.assertAlmostEqual(result, 5.0)  # 中点仍是中点
    
    def test_smooth_step(self):
        self.assertAlmostEqual(smooth_step(0.0), 0.0)
        self.assertAlmostEqual(smooth_step(1.0), 1.0)
        self.assertAlmostEqual(smooth_step(0.5), 0.5)
        # S曲线在0.5处应该等于0.5
        self.assertGreater(smooth_step(0.75), 0.75)  # 后期加速
    
    def test_trajectory_generator_linear(self):
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 10.0])
        traj = trajectory_generator_linear(start, end, 1.0, 0.1)
        self.assertEqual(len(traj), 11)
        np.testing.assert_array_almost_equal(traj[0], start)
        np.testing.assert_array_almost_equal(traj[-1], end)


class TestGeometry(unittest.TestCase):
    """测试几何计算工具"""
    
    def test_point_to_line_distance(self):
        point = np.array([0.0, 1.0])
        line_start = np.array([-1.0, 0.0])
        line_end = np.array([1.0, 0.0])
        dist = point_to_line_distance(point, line_start, line_end)
        self.assertAlmostEqual(dist, 1.0)  # 垂直距离=1
    
    def test_closest_point_on_trajectory(self):
        trajectory = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        point = np.array([1.5, 0.1])
        idx, dist = closest_point_on_trajectory(point, trajectory)
        self.assertEqual(idx, 1)  # 最接近 [1,0]


class TestUnitConversion(unittest.TestCase):
    """测试单位转换工具"""
    
    def test_deg_rad_conversion(self):
        self.assertAlmostEqual(deg_to_rad(180.0), np.pi)
        self.assertAlmostEqual(rad_to_deg(np.pi), 180.0)
    
    def test_mps_rpm_conversion(self):
        radius = 0.1  # 10cm
        mps = 1.0  # 1m/s
        rpm = mps_to_rpm(mps, radius)
        self.assertAlmostEqual(rpm_to_mps(rpm, radius), mps)
    
    def test_rpm_radps_conversion(self):
        rpm = 60.0
        radps = rpm_to_radps(rpm)
        self.assertAlmostEqual(radps, 2 * np.pi)  # 1 rev/s = 2π rad/s
        self.assertAlmostEqual(radps_to_rpm(radps), rpm)


if __name__ == '__main__':
    unittest.main()
