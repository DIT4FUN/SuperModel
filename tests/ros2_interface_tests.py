"""
ROS2 接口模块测试
================

测试 ROS2 控制接口模块:
- ROS2JointTrajectoryInterface
- ROS2ActionInterface
- ROS2ParameterInterface
- ROS2ComponentInterface
"""

import numpy as np
import sys
import time
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.ros2_interface import (
    ROS2JointTrajectoryInterface, ROS2ActionInterface, ROS2ParameterInterface,
    ROS2ComponentInterface, ControlInterfaceMode, JointCommand, JointState,
    ActionGoalStatus, ActionFeedback, ActionResult
)


class TestROS2JointTrajectoryInterface(unittest.TestCase):
    """测试 ROS2 关节轨迹接口"""
    
    def setUp(self):
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.interface = ROS2JointTrajectoryInterface(
            joint_names=self.joint_names,
            interface_mode=ControlInterfaceMode.POSITION
        )
    
    def test_interface_initialization(self):
        self.assertEqual(self.interface.num_joints, 3)
        self.assertEqual(self.interface.mode, ControlInterfaceMode.POSITION)
        self.assertFalse(self.interface._is_active)
    
    def test_activate_deactivate(self):
        self.interface.activate()
        self.assertTrue(self.interface._is_active)
        
        self.interface.deactivate()
        self.assertFalse(self.interface._is_active)
    
    def test_send_single_point(self):
        self.interface.activate()
        
        cmd = JointCommand(
            positions=np.array([0.1, 0.2, 0.3]),
            time_from_start=0.0
        )
        result = self.interface.send_point(cmd)
        self.assertTrue(result)
        
        self.interface.deactivate()
    
    def test_send_trajectory(self):
        self.interface.activate()
        
        traj = [
            JointCommand(positions=np.array([0.1, 0.2, 0.3]), time_from_start=0.0),
            JointCommand(positions=np.array([0.5, 0.6, 0.7]), time_from_start=1.0),
            JointCommand(positions=np.array([1.0, 1.2, 1.5]), time_from_start=2.0),
        ]
        
        result = self.interface.send_trajectory(traj)
        self.assertTrue(result)
        
        self.interface.deactivate()
    
    def test_update_with_state(self):
        self.interface.activate()
        
        current_state = JointState(
            positions=np.array([0.0, 0.0, 0.0]),
            velocities=np.zeros(3),
            efforts=np.zeros(3),
            timestamp=time.time()
        )
        
        cmd = JointCommand(positions=np.array([0.1, 0.2, 0.3]))
        self.interface.send_point(cmd)
        
        next_cmd = self.interface.update(current_state)
        self.assertIsInstance(next_cmd, (JointCommand, type(None)))
        
        self.interface.deactivate()
    
    def test_cancel(self):
        self.interface.activate()
        
        # 发送一个轨迹后取消
        traj = [JointCommand(positions=np.array([0.1, 0.2, 0.3]))]
        self.interface.send_trajectory(traj)
        
        result = self.interface.cancel()
        self.assertTrue(result)
        
        self.interface.deactivate()
    
    def test_cancel_without_trajectory(self):
        self.interface.activate()
        
        # 没有活动轨迹时取消应返回 False
        result = self.interface.cancel()
        self.assertFalse(result)
        
        self.interface.deactivate()
    
    def test_velocity_mode(self):
        interface_vel = ROS2JointTrajectoryInterface(
            joint_names=self.joint_names,
            interface_mode=ControlInterfaceMode.VELOCITY
        )
        self.assertEqual(interface_vel.mode, ControlInterfaceMode.VELOCITY)
    
    def test_effort_mode(self):
        interface_effort = ROS2JointTrajectoryInterface(
            joint_names=self.joint_names,
            interface_mode=ControlInterfaceMode.EFFORT
        )
        self.assertEqual(interface_effort.mode, ControlInterfaceMode.EFFORT)
    
    def test_stats(self):
        self.interface.activate()
        
        # 注册回调以便计数
        command_count = [0]
        def cmd_callback(cmd):
            command_count[0] += 1
        self.interface.set_command_callback(cmd_callback)
        
        for _ in range(5):
            cmd = JointCommand(positions=np.array([0.1, 0.2, 0.3]))
            self.interface.send_point(cmd)
        
        stats = self.interface.get_stats()
        self.assertIn('sent_commands', stats)
        self.assertEqual(stats['sent_commands'], 5)
        
        self.interface.deactivate()
    
    def test_stats_no_callback(self):
        self.interface.activate()
        
        # 无回调时 sent_commands 不增加
        for _ in range(3):
            cmd = JointCommand(positions=np.array([0.1, 0.2, 0.3]))
            self.interface.send_point(cmd)
        
        stats = self.interface.get_stats()
        self.assertEqual(stats['sent_commands'], 0)
        
        self.interface.deactivate()
    
    def test_double_activate(self):
        self.interface.activate()
        self.interface.activate()  # 重复激活不应报错
        self.assertTrue(self.interface._is_active)
        self.interface.deactivate()
    
    def test_update_without_activate(self):
        state = JointState(
            positions=np.array([0.0, 0.0, 0.0]),
            velocities=np.zeros(2),
            efforts=np.zeros(2)
        )
        result = self.interface.update(state)
        self.assertIsNone(result)


class TestROS2ActionInterface(unittest.TestCase):
    """测试 ROS2 Action 接口"""
    
    def setUp(self):
        self.action = ROS2ActionInterface(action_name="test_action")
    
    def test_action_initialization(self):
        self.assertEqual(self.action.action_name, "test_action")
    
    def test_start_stop_server(self):
        self.action.start_server()
        self.assertTrue(self.action._is_server_active)
        
        self.action.stop_server()
        self.assertFalse(self.action._is_server_active)
    
    def test_send_goal_requires_active_server(self):
        # 服务器未启动时应抛出异常
        traj = [JointCommand(positions=np.array([0.1, 0.2]))]
        with self.assertRaises(RuntimeError):
            self.action.send_goal(traj)
    
    def test_send_goal(self):
        self.action.start_server()
        
        traj = [
            JointCommand(positions=np.array([0.1, 0.2])),
            JointCommand(positions=np.array([0.5, 0.6])),
        ]
        
        goal_id = self.action.send_goal(traj)
        self.assertIsInstance(goal_id, str)
        self.assertIn('goal_', goal_id)
        
        self.action.stop_server()
    
    def test_goal_status(self):
        self.action.start_server()
        
        traj = [JointCommand(positions=np.array([0.1, 0.2]))]
        goal_id = self.action.send_goal(traj)
        
        status = self.action.get_goal_status(goal_id)
        self.assertIsInstance(status, ActionGoalStatus)
        self.assertIn(status.value, [s.value for s in ActionGoalStatus])
        
        self.action.stop_server()
    
    def test_update_server(self):
        self.action.start_server()
        
        traj = [JointCommand(positions=np.array([0.1, 0.2]))]
        self.action.send_goal(traj)
        
        current_state = JointState(
            positions=np.array([0.0, 0.0]),
            velocities=np.zeros(2),
            efforts=np.zeros(2)
        )
        
        result = self.action.update_server(current_state)
        self.assertIsInstance(result, bool)
        
        self.action.stop_server()
    
    def test_cancel_goal(self):
        self.action.start_server()
        
        traj = [JointCommand(positions=np.array([0.1, 0.2]))]
        goal_id = self.action.send_goal(traj)
        
        result = self.action.cancel_goal(goal_id)
        self.assertTrue(result)
        
        self.action.stop_server()
    
    def test_send_goal_async(self):
        self.action.start_server()
        
        traj = [JointCommand(positions=np.array([0.1, 0.2]))]
        goal_id = self.action.send_goal_async(traj, timeout_sec=5.0)
        self.assertIsInstance(goal_id, str)
        
        self.action.stop_server()
    
    def test_wait_for_result(self):
        self.action.start_server()
        
        traj = [
            JointCommand(positions=np.array([0.1, 0.2])),
            JointCommand(positions=np.array([0.5, 0.6])),
        ]
        goal_id = self.action.send_goal(traj)
        
        result = self.action.wait_for_result(goal_id, timeout_sec=1.0)
        self.assertIsInstance(result, (ActionResult, type(None)))
        
        self.action.stop_server()
    
    def test_cancel_all_goals(self):
        self.action.start_server()
        
        for i in range(3):
            traj = [JointCommand(positions=np.array([0.1 * i, 0.2 * i]))]
            self.action.send_goal(traj)
        
        count = self.action.cancel_all_goals()
        self.assertEqual(count, 3)
        
        self.action.stop_server()
    
    def test_action_stats(self):
        self.action.start_server()
        
        for _ in range(3):
            traj = [JointCommand(positions=np.array([0.1, 0.2]))]
            self.action.send_goal(traj)
        
        stats = self.action.get_stats()
        self.assertIn('total_goals', stats)
        
        self.action.stop_server()


class TestROS2ParameterInterface(unittest.TestCase):
    """测试 ROS2 参数接口"""
    
    def setUp(self):
        self.param = ROS2ParameterInterface(node_name="test_node")
    
    def test_set_and_get_parameter(self):
        self.param.set_parameter("test_param", 1.5)
        value = self.param.get_parameter("test_param")
        self.assertEqual(value, 1.5)
    
    def test_get_parameter_with_default(self):
        value = self.param.get_parameter("nonexistent", default=42)
        self.assertEqual(value, 42)
    
    def test_get_multiple_parameters(self):
        self.param.set_parameter("p1", 1.0)
        self.param.set_parameter("p2", "hello")
        self.param.set_parameter("p3", True)
        
        values = self.param.get_parameters(["p1", "p2", "p3"])
        self.assertEqual(values['p1'], 1.0)
        self.assertEqual(values['p2'], "hello")
        self.assertEqual(values['p3'], True)
    
    def test_list_parameters(self):
        self.param.set_parameter("ns.param1", 1.0)
        self.param.set_parameter("ns.param2", 2.0)
        self.param.set_parameter("other.param", 3.0)
        
        all_params = self.param.list_parameters()
        self.assertGreaterEqual(len(all_params), 3)
        
        ns_params = self.param.list_parameters(prefix="ns")
        self.assertEqual(len(ns_params), 2)
    
    def test_declare_parameter(self):
        self.param.declare_parameter(
            "declared_param",
            value=3.14,
            descriptor={'type': 'double', 'description': 'A test parameter'}
        )
        value = self.param.get_parameter("declared_param")
        self.assertEqual(value, 3.14)
    
    def test_parameter_change_subscription(self):
        changes = []
        
        def callback(value):
            changes.append(value)
        
        self.param.subscribe_parameter_change("watched_param", callback)
        self.param.set_parameter("watched_param", 1.0)
        self.param.set_parameter("watched_param", 2.0)
        
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0], 1.0)
        self.assertEqual(changes[1], 2.0)
    
    def test_load_from_dict(self):
        params = {
            "control.Kp": 1.0,
            "control.Ki": 0.1,
            "control.Kd": 0.05,
        }
        
        self.param.load_from_dict(params)
        
        self.assertEqual(self.param.get_parameter("control.Kp"), 1.0)
        self.assertEqual(self.param.get_parameter("control.Ki"), 0.1)
        self.assertEqual(self.param.get_parameter("control.Kd"), 0.05)
    
    def test_to_dict(self):
        self.param.set_parameter("p1", 1.0)
        self.param.set_parameter("p2", "test")
        
        d = self.param.to_dict()
        self.assertIn('p1', d)
        self.assertIn('p2', d)
        self.assertEqual(d['p1'], 1.0)
    
    def test_parameter_overwrite(self):
        self.param.set_parameter("test", 1.0)
        self.param.set_parameter("test", 2.0)
        self.assertEqual(self.param.get_parameter("test"), 2.0)


class TestROS2ComponentInterface(unittest.TestCase):
    """测试 ROS2 组件接口"""
    
    def setUp(self):
        self.component = ROS2ComponentInterface(component_name="test_component")
    
    def test_component_initialization(self):
        self.assertEqual(self.component.component_name, "test_component")
        self.assertEqual(self.component.get_state(), "unconfigured")
    
    def test_lifecycle_state_machine(self):
        # unconfigured -> configure -> inactive
        result = self.component.configure()
        self.assertTrue(result)
        self.assertEqual(self.component.get_state(), "inactive")
        
        # inactive -> activate -> active
        result = self.component.activate()
        self.assertTrue(result)
        self.assertEqual(self.component.get_state(), "active")
        
        # active -> deactivate -> inactive
        result = self.component.deactivate()
        self.assertTrue(result)
        self.assertEqual(self.component.get_state(), "inactive")
        
        # inactive -> cleanup -> unconfigured
        result = self.component.cleanup()
        self.assertTrue(result)
        self.assertEqual(self.component.get_state(), "unconfigured")
    
    def test_shutdown(self):
        self.component.configure()
        self.component.activate()
        
        result = self.component.shutdown()
        self.assertTrue(result)
        self.assertEqual(self.component.get_state(), "shutdown")
    
    def test_callback_registration(self):
        configure_called = []
        
        def on_configure():
            configure_called.append(True)
            return True
        
        self.component.on_configure(on_configure)
        self.component.configure()
        
        self.assertEqual(len(configure_called), 1)
    
    def test_full_lifecycle_with_callbacks(self):
        states = []
        
        self.component.on_configure(lambda: (states.append('configure'), True))
        self.component.on_activate(lambda: (states.append('activate'), True))
        self.component.on_deactivate(lambda: (states.append('deactivate'), True))
        self.component.on_cleanup(lambda: (states.append('cleanup'), True))
        
        self.component.configure()
        self.assertIn('configure', states)
        
        self.component.activate()
        self.assertIn('activate', states)
        
        self.component.deactivate()
        self.assertIn('deactivate', states)
        
        self.component.cleanup()
        self.assertIn('cleanup', states)
        
        self.assertEqual(len(states), 4)
    
    def test_shutdown_from_any_state(self):
        component = ROS2ComponentInterface("test")
        # 从任意状态都可以shutdown
        component.shutdown()
        self.assertEqual(component.get_state(), "shutdown")


class TestJointCommandJointState(unittest.TestCase):
    """测试关节命令和状态数据类"""
    
    def test_joint_command_with_all_fields(self):
        cmd = JointCommand(
            positions=np.array([0.1, 0.2, 0.3]),
            velocities=np.array([0.0, 0.0, 0.0]),
            accelerations=np.array([0.0, 0.0, 0.0]),
            effort=np.array([0.0, 0.0, 0.0]),
            time_from_start=1.5
        )
        
        self.assertEqual(cmd.positions.shape, (3,))
        self.assertEqual(cmd.time_from_start, 1.5)
    
    def test_joint_command_minimal(self):
        cmd = JointCommand(positions=np.array([0.1, 0.2]))
        self.assertIsNone(cmd.velocities)
        self.assertEqual(cmd.time_from_start, 0.0)
    
    def test_joint_state(self):
        state = JointState(
            positions=np.array([0.1, 0.2, 0.3]),
            velocities=np.array([0.5, 0.5, 0.5]),
            efforts=np.array([1.0, 1.0, 1.0]),
            timestamp=123.456
        )
        
        self.assertEqual(state.positions.shape, (3,))
        self.assertEqual(state.velocities.shape, (3,))
        self.assertEqual(state.efforts.shape, (3,))
        self.assertEqual(state.timestamp, 123.456)


class TestActionFeedbackResult(unittest.TestCase):
    """测试 Action 反馈和结果数据类"""
    
    def test_action_feedback(self):
        fb = ActionFeedback(
            sequence=1,
            percent_complete=0.5,
            current_joint_positions=np.array([0.1, 0.2]),
            error=np.array([0.01, 0.01]),
            message="Halfway there"
        )
        
        self.assertEqual(fb.sequence, 1)
        self.assertEqual(fb.percent_complete, 0.5)
        self.assertEqual(fb.current_joint_positions.shape, (2,))
    
    def test_action_result(self):
        result = ActionResult(
            success=True,
            message="Completed",
            final_positions=np.array([1.0, 1.0]),
            execution_time=5.0,
            trajectory_length=10
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 5.0)
        self.assertEqual(result.trajectory_length, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
