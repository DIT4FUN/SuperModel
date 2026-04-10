"""
阻抗控制模块测试
测试 ImpedanceController / AdmittanceController / ForceImpedanceController /
CollaborativeController / AdaptiveImpedanceController 及 AGV 五级规格
"""

import unittest
import numpy as np
import sys
import os

_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SrcPath = os.path.join(_ProjectRoot, 'src')
if _SrcPath not in sys.path:
    sys.path.insert(0, _SrcPath)
if _ProjectRoot not in sys.path:
    sys.path.insert(0, _ProjectRoot)

from src.control.impedance import (
    ImpedanceParams, ImpedanceController,
    AdmittanceController, ForceImpedanceController,
    CollaborativeController, AdaptiveImpedanceController,
    AGV_IMPEDANCE_GRADES, get_impedance_spec, list_impedance_capabilities,
)


class TestImpedanceParams(unittest.TestCase):
    """ImpedanceParams 数据类测试"""

    def test_default_6d(self):
        params = ImpedanceParams.default_6d()
        self.assertEqual(params.M.shape, (6, 6))
        self.assertEqual(params.D.shape, (6, 6))
        self.assertEqual(params.K.shape, (6, 6))
        self.assertTrue(np.allclose(np.diag(params.M), 5.0))
        self.assertTrue(np.allclose(np.diag(params.D), 50.0))
        self.assertTrue(np.allclose(np.diag(params.K), 200.0))

    def test_high_stiffness(self):
        params = ImpedanceParams.high_stiffness()
        self.assertTrue(np.allclose(np.diag(params.K), 1000.0))
        self.assertTrue(np.allclose(np.diag(params.D), 100.0))

    def test_from_lists(self):
        M_list = [[1, 0, 0, 0, 0, 0]] * 6
        D_list = [[2, 0, 0, 0, 0, 0]] * 6
        K_list = [[3, 0, 0, 0, 0, 0]] * 6
        params = ImpedanceParams(M=M_list, D=D_list, K=K_list)
        self.assertIsInstance(params.M, np.ndarray)
        self.assertEqual(params.M.dtype, np.float32)


class TestImpedanceController(unittest.TestCase):
    """ImpedanceController 测试"""

    def setUp(self):
        self.ctrl = ImpedanceController(
            impedance_params=ImpedanceParams.default_6d(),
            control_rate=100.0
        )

    def test_creation(self):
        self.assertIsNotNone(self.ctrl.params)
        self.assertEqual(self.ctrl.dt, 0.01)

    def test_set_impedance_params(self):
        new_params = ImpedanceParams.high_stiffness()
        self.ctrl.set_impedance_params(new_params)
        self.assertTrue(np.allclose(np.diag(self.ctrl.params.K), 1000.0))

    def test_compute_torque(self):
        jacobian = np.eye(6, 3)  # 3 joints
        torque = self.ctrl.compute_torque(
            desired_position=np.array([0.1, 0.0, 0.0]),
            desired_velocity=np.zeros(3),
            current_position=np.array([0.05, 0.0, 0.0]),
            current_velocity=np.zeros(3),
            external_wrench=np.zeros(6),
            jacobian=jacobian
        )
        self.assertEqual(torque.shape, (3,))

    def test_compute_cartesian_force(self):
        force = self.ctrl.compute_cartesian_force(
            desired_pose=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            desired_velocity=np.zeros(6),
            external_wrench=np.zeros(6),
        )
        self.assertEqual(force.shape, (6,))


class TestAdmittanceController(unittest.TestCase):
    """AdmittanceController 测试"""

    def test_creation(self):
        ctrl = AdmittanceController(M=10.0, D=50.0, K=200.0, control_rate=100.0)
        self.assertEqual(ctrl.M, 10.0)
        self.assertEqual(ctrl.D, 50.0)
        self.assertEqual(ctrl.K, 200.0)

    def test_update(self):
        ctrl = AdmittanceController()
        pos = ctrl.update(external_force=10.0, desired_position=0.0)
        self.assertIsInstance(pos, (float, np.floating))

    def test_reset(self):
        ctrl = AdmittanceController()
        ctrl.update(external_force=10.0, desired_position=1.0)
        ctrl.reset()
        self.assertEqual(ctrl._velocity, 0.0)
        self.assertEqual(ctrl._position, 0.0)


class TestForceImpedanceController(unittest.TestCase):
    """ForceImpedanceController 测试"""

    def test_creation(self):
        force_axes = np.array([0, 0, 1, 0, 0, 0])
        ctrl = ForceImpedanceController(force_axes=force_axes, Kp=100.0, Kf=1.0)
        self.assertTrue(ctrl.force_axes[2])
        self.assertFalse(ctrl.position_axes[2])

    def test_compute_torque(self):
        force_axes = np.array([0, 0, 1, 0, 0, 0])
        ctrl = ForceImpedanceController(force_axes=force_axes)
        # ForceImpedanceController expects 6D desired/current position, not 3D
        desired_position = np.zeros(6)
        desired_force = np.zeros(6)
        current_position = np.zeros(6)
        current_force = np.zeros(6)
        jacobian = np.eye(6, 3)
        torque = ctrl.compute_torque(
            desired_position=desired_position,
            desired_force=desired_force,
            current_position=current_position,
            current_force=current_force,
            jacobian=jacobian
        )
        self.assertEqual(torque.shape, (3,))


class TestCollaborativeController(unittest.TestCase):
    """CollaborativeController 测试"""

    def test_creation(self):
        ctrl = CollaborativeController(
            safety_force_limit=50.0,
            safety_velocity_limit=0.5,
            reaction_mode="pause"
        )
        self.assertEqual(ctrl.safety_force_limit, 50.0)

    def test_check_safety_pass(self):
        ctrl = CollaborativeController()
        safe, msg = ctrl.check_safety(
            external_force=np.array([5.0, 0.0, 0.0]),
            velocity=np.array([0.1, 0.0, 0.0])
        )
        self.assertTrue(safe)

    def test_check_safety_force_violation(self):
        ctrl = CollaborativeController(safety_force_limit=10.0)
        safe, msg = ctrl.check_safety(
            external_force=np.array([20.0, 0.0, 0.0]),
            velocity=np.zeros(3)
        )
        self.assertFalse(safe)
        self.assertIn("force_limit", msg)

    def test_get_reaction_torque_pause(self):
        ctrl = CollaborativeController(reaction_mode="pause")
        jacobian = np.eye(3, 3)
        torque = ctrl.get_reaction_torque(np.zeros(3), jacobian)
        self.assertTrue(np.allclose(torque, np.zeros(3)))


class TestAdaptiveImpedanceController(unittest.TestCase):
    """AdaptiveImpedanceController 测试"""

    def test_creation(self):
        ctrl = AdaptiveImpedanceController(control_rate=100.0)
        self.assertEqual(ctrl.dt, 0.01)
        self.assertEqual(ctrl._est_env_stiffness, 1000.0)
        self.assertTrue(ctrl.use_lyapunov)

    def test_estimated_env_params(self):
        ctrl = AdaptiveImpedanceController()
        params = ctrl.estimated_env_params
        self.assertIn("stiffness_N_per_m", params)
        self.assertIn("damping_Ns_per_m", params)
        self.assertIn("inertia_kg", params)

    def test_current_impedance_params(self):
        ctrl = AdaptiveImpedanceController()
        params = ctrl.current_impedance_params
        self.assertIsInstance(params, ImpedanceParams)

    def test_update(self):
        ctrl = AdaptiveImpedanceController(control_rate=100.0)
        jacobian = np.eye(6, 3)
        torque, info = ctrl.update(
            desired_position=np.array([0.1, 0.0, 0.0]),
            current_position=np.array([0.05, 0.0, 0.0]),
            current_velocity=np.zeros(3),
            external_wrench=np.zeros(6),
            jacobian=jacobian,
        )
        self.assertEqual(torque.shape, (3,))
        self.assertIn("est_env_K", info)
        self.assertIn("pos_error_norm", info)

    def test_reset(self):
        ctrl = AdaptiveImpedanceController()
        jacobian = np.eye(6, 3)
        ctrl.update(
            desired_position=np.array([0.1, 0.0, 0.0]),
            current_position=np.array([0.05, 0.0, 0.0]),
            current_velocity=np.zeros(3),
            external_wrench=np.zeros(6),
            jacobian=jacobian,
        )
        ctrl.reset()
        self.assertEqual(ctrl._est_env_stiffness, 1000.0)

    def test_convergence_metrics_initial(self):
        ctrl = AdaptiveImpedanceController()
        metrics = ctrl.get_convergence_metrics()
        self.assertFalse(metrics.get("converged", False))


# === AGV五级规格测试 ===

class TestAGVImpedanceGrades(unittest.TestCase):
    """AGV五级阻抗控制规格测试"""

    def test_all_grades_present(self):
        expected = {"S", "M", "L", "XL", "XXL"}
        actual = set(AGV_IMPEDANCE_GRADES.keys())
        self.assertEqual(expected, actual)

    def test_spec_completeness(self):
        required_keys = {
            "control_freq_hz", "stiffness_range", "damping_range",
            "inertia_range", "force_limit", "position_error_limit",
            "adaptation_rate", "convergence_time_s", "use_lyapunov", "use_mrac",
        }
        for grade, spec in AGV_IMPEDANCE_GRADES.items():
            missing = required_keys - set(spec.keys())
            self.assertFalse(missing, f"Grade {grade} missing keys: {missing}")

    def test_freq_monotonic(self):
        freqs = {g: s["control_freq_hz"] for g, s in AGV_IMPEDANCE_GRADES.items()}
        self.assertLess(freqs["S"], freqs["M"])
        self.assertLess(freqs["M"], freqs["L"])
        self.assertLess(freqs["L"], freqs["XL"])
        self.assertLess(freqs["XL"], freqs["XXL"])

    def test_error_limit_monotonic(self):
        limits = {g: s["position_error_limit"] for g, s in AGV_IMPEDANCE_GRADES.items()}
        self.assertGreater(limits["S"], limits["M"])
        self.assertGreater(limits["M"], limits["L"])
        self.assertGreater(limits["L"], limits["XL"])
        self.assertGreater(limits["XL"], limits["XXL"])

    def test_xxl_uses_all_features(self):
        spec = AGV_IMPEDANCE_GRADES["XXL"]
        self.assertTrue(spec["use_lyapunov"])
        self.assertTrue(spec["use_mrac"])
        self.assertEqual(spec["control_freq_hz"], 1000)

    def test_get_impedance_spec_default(self):
        spec = get_impedance_spec("INVALID")
        self.assertEqual(spec["control_freq_hz"], 100)

    def test_get_impedance_spec_xxl(self):
        spec = get_impedance_spec("XXL")
        self.assertEqual(spec["control_freq_hz"], 1000)
        self.assertEqual(spec["force_limit"], 500.0)

    def test_list_impedance_capabilities(self):
        caps = list_impedance_capabilities()
        self.assertEqual(len(caps), 5)
        self.assertIn("XXL", caps)


class TestImpedanceControllerGradeIntegration(unittest.TestCase):
    """ImpedanceController 与 AGV 五级规格集成测试"""

    def test_controller_with_grade_spec(self):
        for grade, spec in AGV_IMPEDANCE_GRADES.items():
            freq = spec["control_freq_hz"]
            stiffness_range = spec["stiffness_range"]
            K_default = (stiffness_range[0] + stiffness_range[1]) / 2
            D_default = spec["damping_range"][0]
            M_default = spec["inertia_range"][0]

            params = ImpedanceParams(
                M=np.eye(6) * M_default,
                D=np.eye(6) * D_default,
                K=np.eye(6) * K_default,
            )
            ctrl = ImpedanceController(params, control_rate=freq)
            self.assertEqual(ctrl.dt, 1.0 / freq)


if __name__ == "__main__":
    unittest.main(verbosity=2)
