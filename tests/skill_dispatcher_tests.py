"""
技能调度器测试用例
==================

测试 SkillDispatcher 的功能:
- 技能注册/注销
- 技能调度与资源锁定
- 资源冲突检测
- 优先级调度
- 并发限制
- AGV五级规格适配
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.skill_dispatcher import (
    SkillDispatcher, SkillRequest, SkillResult, SkillStatus,
    SkillPriority, SkillDefinition, ResourceType,
    AGV_SKILL_DISPATCHER_GRADES, get_skill_dispatcher_spec,
    create_skill_dispatcher,
    create_grasp_skill, create_navigate_skill, create_place_skill,
)


class TestSkillDispatcherBasics(unittest.TestCase):
    """技能调度器基础测试"""

    def test_creation_default(self):
        """测试默认创建"""
        dispatcher = SkillDispatcher()
        self.assertEqual(dispatcher.grade, 'M')
        self.assertEqual(dispatcher.max_concurrent, 2)
        self.assertTrue(dispatcher.enable_monitoring)

    def test_creation_by_grade(self):
        """测试按等级创建"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            dispatcher = SkillDispatcher(grade=grade)
            self.assertEqual(dispatcher.grade, grade)
    
    def test_creation_custom(self):
        """测试自定义参数"""
        dispatcher = SkillDispatcher(
            grade='XL',
            enable_monitoring=False,
            max_concurrent=3
        )
        self.assertEqual(dispatcher.grade, 'XL')
        self.assertFalse(dispatcher.enable_monitoring)
        self.assertEqual(dispatcher.max_concurrent, 3)

    def test_skill_registration(self):
        """测试技能注册"""
        dispatcher = SkillDispatcher(grade='M')
        
        def dummy_fn(params):
            return {'result': 'ok'}
        
        skill = SkillDefinition(
            name='test_skill',
            execute_fn=dummy_fn,
            required_resources={ResourceType.MOTOR},
        )
        
        self.assertTrue(dispatcher.register_skill(skill))
        self.assertIn('test_skill', dispatcher._skills)
    
    def test_skill_duplicate_registration(self):
        """测试重复注册"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill1 = SkillDefinition('dup', dummy_fn, {ResourceType.MOTOR})
        skill2 = SkillDefinition('dup', dummy_fn, {ResourceType.SENSOR_IMU})
        
        dispatcher.register_skill(skill1)
        self.assertFalse(dispatcher.register_skill(skill2))
    
    def test_skill_unregistration(self):
        """测试技能注销"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill = SkillDefinition('unreg', dummy_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        self.assertTrue(dispatcher.unregister_skill('unreg'))
        self.assertNotIn('unreg', dispatcher._skills)
        self.assertFalse(dispatcher.unregister_skill('nonexistent'))


class TestSkillDispatching(unittest.TestCase):
    """技能调度测试"""

    def test_dispatch_unknown_skill(self):
        """测试调度未知技能"""
        dispatcher = SkillDispatcher()
        request = SkillRequest(skill_name='nonexistent', params={})
        result = dispatcher.dispatch(request)
        self.assertEqual(result.status, SkillStatus.FAILED)
        self.assertIn('not found', result.error)

    def test_dispatch_success(self):
        """测试成功调度"""
        dispatcher = SkillDispatcher()
        
        def mock_fn(params):
            return {'output': params.get('value', 0) * 2}
        
        skill = SkillDefinition(
            name='multiply',
            execute_fn=mock_fn,
            required_resources={ResourceType.MOTOR},
        )
        dispatcher.register_skill(skill)
        
        request = SkillRequest(
            skill_name='multiply',
            params={'value': 5},
            priority=SkillPriority.HIGH
        )
        result = dispatcher.dispatch(request)
        
        self.assertEqual(result.status, SkillStatus.COMPLETED)
        self.assertEqual(result.output['output'], 10)
        self.assertIsNone(result.error)

    def test_dispatch_with_exception(self):
        """测试技能执行异常"""
        dispatcher = SkillDispatcher()
        
        def failing_fn(params):
            raise ValueError("Intentional failure")
        
        skill = SkillDefinition(
            name='fail_skill',
            execute_fn=failing_fn,
            required_resources={ResourceType.MOTOR},
        )
        dispatcher.register_skill(skill)
        
        request = SkillRequest(skill_name='fail_skill', params={})
        result = dispatcher.dispatch(request)
        
        self.assertEqual(result.status, SkillStatus.FAILED)
        self.assertIn('Intentional failure', result.error)

    def test_dispatch_execution_time(self):
        """测试执行时间记录"""
        dispatcher = SkillDispatcher()
        
        def slow_fn(params):
            time.sleep(0.05)
            return {}
        
        skill = SkillDefinition('slow', slow_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        request = SkillRequest(skill_name='slow', params={})
        result = dispatcher.dispatch(request)
        
        self.assertGreater(result.execution_time_sec, 0.04)
        self.assertLess(result.execution_time_sec, 1.0)


class TestResourceLocking(unittest.TestCase):
    """资源锁定测试"""

    def test_resource_lock_on_dispatch(self):
        """测试调度时资源锁定"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill = SkillDefinition(
            name='lock_test',
            execute_fn=dummy_fn,
            required_resources={ResourceType.MOTOR, ResourceType.SENSOR_FORCE},
        )
        dispatcher.register_skill(skill)
        
        request = SkillRequest(skill_name='lock_test', params={})
        result = dispatcher.dispatch(request)
        
        self.assertEqual(result.status, SkillStatus.COMPLETED)
        self.assertIn(ResourceType.MOTOR, result.resources_used)
        self.assertIn(ResourceType.SENSOR_FORCE, result.resources_used)

    def test_resource_conflict(self):
        """测试资源冲突检测"""
        dispatcher = SkillDispatcher(max_concurrent=2)
        
        def long_fn(params):
            time.sleep(0.5)
            return {}
        
        skill = SkillDefinition(
            name='long_skill',
            execute_fn=long_fn,
            required_resources={ResourceType.MOTOR},
        )
        dispatcher.register_skill(skill)
        
        # 第一个请求
        req1 = SkillRequest(skill_name='long_skill', params={})
        result1 = dispatcher.dispatch(req1)
        # 同步等待完成
        time.sleep(0.6)
        
        # 第二个请求相同资源
        req2 = SkillRequest(skill_name='long_skill', params={})
        result2 = dispatcher.dispatch(req2)
        self.assertEqual(result2.status, SkillStatus.COMPLETED)

    def test_resource_release_after_completion(self):
        """测试完成后资源释放"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill = SkillDefinition('release', dummy_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        request = SkillRequest(skill_name='release', params={})
        dispatcher.dispatch(request)
        
        # 资源应该已释放
        self.assertNotIn(ResourceType.MOTOR, dispatcher._resource_locks)


class TestPriorityScheduling(unittest.TestCase):
    """优先级调度测试"""

    def test_priority_normal(self):
        """测试普通优先级"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill = SkillDefinition('prio', dummy_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        request = SkillRequest(
            skill_name='prio',
            params={},
            priority=SkillPriority.NORMAL
        )
        result = dispatcher.dispatch(request)
        self.assertEqual(result.status, SkillStatus.COMPLETED)

    def test_priority_critical(self):
        """测试关键优先级"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {'critical': True}
        
        skill = SkillDefinition('critical', dummy_fn, {ResourceType.SENSOR_IMU})
        dispatcher.register_skill(skill)
        
        request = SkillRequest(
            skill_name='critical',
            params={},
            priority=SkillPriority.CRITICAL
        )
        result = dispatcher.dispatch(request)
        self.assertEqual(result.status, SkillStatus.COMPLETED)
        self.assertTrue(result.output['critical'])


class TestConcurrentLimit(unittest.TestCase):
    """并发限制测试"""

    def test_max_concurrent_by_grade(self):
        """测试各等级最大并发数"""
        expected = {'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 6}
        for grade, expected_max in expected.items():
            dispatcher = SkillDispatcher(grade=grade)
            self.assertEqual(dispatcher.max_concurrent, expected_max)

    def test_concurrent_limit_enforced(self):
        """测试并发限制强制执行"""
        dispatcher = SkillDispatcher(grade='M', max_concurrent=1)
        
        def long_fn(params):
            time.sleep(0.3)
            return {}
        
        skill = SkillDefinition('long', long_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        # 第一个请求会完成（因为是同步的）
        req1 = SkillRequest(skill_name='long', params={})
        result1 = dispatcher.dispatch(req1)
        self.assertEqual(result1.status, SkillStatus.COMPLETED)


class TestSkillStatus(unittest.TestCase):
    """技能状态测试"""

    def test_get_status_idle(self):
        """测试空闲状态"""
        dispatcher = SkillDispatcher()
        self.assertEqual(dispatcher.get_status('nonexistent'), SkillStatus.IDLE)

    def test_get_status_running(self):
        """测试运行状态"""
        dispatcher = SkillDispatcher()
        
        def long_fn(params):
            time.sleep(0.2)
            return {}
        
        skill = SkillDefinition('running', long_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        # 调度但不等待
        req = SkillRequest(skill_name='running', params={})
        # 同步调度
        result = dispatcher.dispatch(req)
        self.assertEqual(result.status, SkillStatus.COMPLETED)

    def test_get_result(self):
        """测试获取结果"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {'value': 42}
        
        skill = SkillDefinition('result_test', dummy_fn, {ResourceType.SENSOR_IMU})
        dispatcher.register_skill(skill)
        
        req = SkillRequest(skill_name='result_test', params={})
        dispatcher.dispatch(req)
        
        stored_result = dispatcher.get_result(req.request_id)
        self.assertIsNotNone(stored_result)
        self.assertEqual(stored_result.output['value'], 42)


class TestSkillDefinitions(unittest.TestCase):
    """技能定义工厂测试"""

    def test_create_grasp_skill(self):
        """测试创建抓取技能"""
        skill = create_grasp_skill(None, None)
        self.assertEqual(skill.name, 'grasp')
        self.assertIn(ResourceType.MOTOR, skill.required_resources)
        self.assertIn(ResourceType.SENSOR_FORCE, skill.required_resources)
        self.assertIn(ResourceType.SENSOR_TACTILE, skill.required_resources)

    def test_create_navigate_skill(self):
        """测试创建导航技能"""
        skill = create_navigate_skill(None, None)
        self.assertEqual(skill.name, 'navigate')
        self.assertIn(ResourceType.MOTOR, skill.required_resources)
        self.assertIn(ResourceType.POSITION, skill.required_resources)
        self.assertIn(ResourceType.SENSOR_VISION, skill.required_resources)

    def test_create_place_skill(self):
        """测试创建放置技能"""
        skill = create_place_skill(None)
        self.assertEqual(skill.name, 'place')
        self.assertIn(ResourceType.MOTOR, skill.required_resources)
        self.assertIn(ResourceType.GRIPPER, skill.required_resources)


class TestAGVGrades(unittest.TestCase):
    """AGV五级规格测试"""

    def test_skill_dispatcher_grades_complete(self):
        """测试五级规格完整性"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            self.assertIn(grade, AGV_SKILL_DISPATCHER_GRADES)
            spec = AGV_SKILL_DISPATCHER_GRADES[grade]
            self.assertIn('max_concurrent', spec)
            self.assertIn('timeout_default', spec)
            self.assertIn('skill_count', spec)
            self.assertIn('monitoring', spec)

    def test_skill_dispatcher_grades_monotonic(self):
        """测试五级规格递增性"""
        expected_concurrent = [1, 2, 3, 4, 6]
        expected_timeouts = [30.0, 20.0, 15.0, 10.0, 5.0]
        
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for i, grade in enumerate(grades):
            spec = get_skill_dispatcher_spec(grade)
            self.assertEqual(spec['max_concurrent'], expected_concurrent[i])
            self.assertEqual(spec['timeout_default'], expected_timeouts[i])

    def test_get_skill_dispatcher_spec(self):
        """测试获取规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_skill_dispatcher_spec(grade)
            self.assertIsInstance(spec, dict)
            self.assertGreater(spec['max_concurrent'], 0)

    def test_create_skill_dispatcher(self):
        """测试工厂函数"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            dispatcher = create_skill_dispatcher(grade)
            self.assertEqual(dispatcher.grade, grade)


class TestStats(unittest.TestCase):
    """调度统计测试"""

    def test_stats_initial(self):
        """测试初始统计"""
        dispatcher = SkillDispatcher()
        stats = dispatcher.get_stats()
        self.assertEqual(stats['total_dispatched'], 0)
        self.assertEqual(stats['total_completed'], 0)
        self.assertEqual(stats['total_failed'], 0)

    def test_stats_after_dispatch(self):
        """测试调度后统计"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill = SkillDefinition('stats_test', dummy_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        for _ in range(3):
            req = SkillRequest(skill_name='stats_test', params={})
            dispatcher.dispatch(req)
        
        stats = dispatcher.get_stats()
        self.assertEqual(stats['total_dispatched'], 3)
        self.assertEqual(stats['total_completed'], 3)


class TestCancel(unittest.TestCase):
    """取消功能测试"""

    def test_cancel_running_request(self):
        """测试取消运行中的请求"""
        dispatcher = SkillDispatcher()
        
        def dummy_fn(params):
            return {}
        
        skill = SkillDefinition('cancel_test', dummy_fn, {ResourceType.MOTOR})
        dispatcher.register_skill(skill)
        
        req = SkillRequest(skill_name='cancel_test', params={})
        # 先完成它
        dispatcher.dispatch(req)
        
        # 取消不存在的请求
        self.assertFalse(dispatcher.cancel('nonexistent_id'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
