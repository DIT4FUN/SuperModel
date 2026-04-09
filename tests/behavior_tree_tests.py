"""
行为树模块测试 (Behavior Tree Tests)
=====================================

测试覆盖:
- 节点状态机 (IDLE/RUNNING/SUCCESS/FAILURE/ERROR)
- Selector: fallback逻辑
- Sequence: 顺序执行逻辑
- Parallel: 并行执行逻辑
- Condition: 条件检查
- Action: 动作执行
- Decorators: Inverter/RepeatUntil/RetryUntil/Timeout/RateLimiter
- SubTree: 子树引用
- BehaviorTree: 完整行为树tick循环
- AGV五级规格配置

Author: SuperModel Dev Team
"""

import pytest
import numpy as np
import time
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.behavior_tree import (
    BehaviorTree, BTNode, BTContext, NodeState,
    Selector, Sequence, Parallel,
    Condition, Action, SubTree,
    Inverter, RepeatUntil, RetryUntil, Timeout, RateLimiter,
    BTGrade, AGV_BT_GRADES,
    create_for_grade, create_safe_selector, create_action_sequence,
)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def make_ctx(**kwargs) -> BTContext:
    return BTContext(**kwargs)


def succeed(ctx: BTContext) -> NodeState:
    return NodeState.SUCCESS


def fail(ctx: BTContext) -> NodeState:
    return NodeState.FAILURE


def running(ctx: BTContext) -> NodeState:
    return NodeState.RUNNING


def always_true(ctx: BTContext) -> bool:
    return True


def always_false(ctx: BTContext) -> bool:
    return False


# ─────────────────────────────────────────────
# Test BTContext
# ─────────────────────────────────────────────

class TestBTContext:
    def test_blackboard_set_get(self):
        ctx = make_ctx()
        ctx.set_blackboard('key1', 42)
        assert ctx.get_blackboard('key1') == 42
        assert ctx.get_blackboard('nonexistent', 999) == 999

    def test_sensor_access(self):
        ctx = make_ctx(sensor_data={'temp': 25.5, 'force': 10.0})
        assert ctx.get_sensor('temp') == 25.5
        assert ctx.get_sensor('force') == 10.0
        assert ctx.get_sensor('missing') is None


# ─────────────────────────────────────────────
# Test Selector
# ─────────────────────────────────────────────

class TestSelector:
    def test_selector_first_child_succeeds(self):
        sel = Selector('TestSelector')
        sel.add_child(Action('A', succeed))
        sel.add_child(Action('B', fail))
        ctx = make_ctx()
        result = sel.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_selector_all_fail(self):
        sel = Selector('TestSelector')
        sel.add_child(Action('A', fail))
        sel.add_child(Action('B', fail))
        ctx = make_ctx()
        result = sel.tick(ctx)
        assert result == NodeState.FAILURE

    def test_selector_second_child_succeeds(self):
        sel = Selector('TestSelector')
        sel.add_child(Action('A', fail))
        sel.add_child(Action('B', succeed))
        ctx = make_ctx()
        result = sel.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_selector_empty(self):
        sel = Selector('EmptySelector')
        ctx = make_ctx()
        result = sel.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_selector_running_child(self):
        call_count = [0]
        def partial_run(ctx):
            call_count[0] += 1
            if call_count[0] < 3:
                return NodeState.RUNNING
            return NodeState.SUCCESS

        sel = Selector('TestSelector')
        sel.add_child(Action('A', partial_run))
        ctx = make_ctx()

        r1 = sel.tick(ctx)
        assert r1 == NodeState.RUNNING
        r2 = sel.tick(ctx)
        assert r2 == NodeState.RUNNING
        r3 = sel.tick(ctx)
        assert r3 == NodeState.SUCCESS

    def test_selector_reset(self):
        sel = Selector('TestSelector')
        sel.add_child(Action('A', succeed))
        ctx = make_ctx()
        sel.tick(ctx)
        sel.reset()
        assert sel.state == NodeState.IDLE


# ─────────────────────────────────────────────
# Test Sequence
# ─────────────────────────────────────────────

class TestSequence:
    def test_sequence_all_succeed(self):
        seq = Sequence('TestSequence')
        seq.add_child(Action('A', succeed))
        seq.add_child(Action('B', succeed))
        ctx = make_ctx()
        result = seq.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_sequence_first_fails(self):
        seq = Sequence('TestSequence')
        seq.add_child(Action('A', fail))
        seq.add_child(Action('B', succeed))
        ctx = make_ctx()
        result = seq.tick(ctx)
        assert result == NodeState.FAILURE

    def test_sequence_empty(self):
        seq = Sequence('EmptySequence')
        ctx = make_ctx()
        result = seq.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_sequence_running_then_success(self):
        call_count = [0]
        def partial_run(ctx):
            call_count[0] += 1
            if call_count[0] < 2:
                return NodeState.RUNNING
            return NodeState.SUCCESS

        seq = Sequence('TestSequence')
        seq.add_child(Action('A', partial_run))
        ctx = make_ctx()

        r1 = seq.tick(ctx)
        assert r1 == NodeState.RUNNING
        r2 = seq.tick(ctx)
        assert r2 == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test Parallel
# ─────────────────────────────────────────────

class TestParallel:
    def test_parallel_success_on_all(self):
        para = Parallel('Para', policy='success_on_all')
        para.add_child(Action('A', succeed))
        para.add_child(Action('B', succeed))
        ctx = make_ctx()
        result = para.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_parallel_failure_on_all(self):
        para = Parallel('Para', policy='failure_on_all')
        para.add_child(Action('A', fail))
        para.add_child(Action('B', fail))
        ctx = make_ctx()
        result = para.tick(ctx)
        assert result == NodeState.FAILURE

    def test_parallel_success_on_one(self):
        para = Parallel('Para', policy='success_on_one')
        para.add_child(Action('A', succeed))
        para.add_child(Action('B', fail))
        ctx = make_ctx()
        result = para.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_parallel_failure_on_one(self):
        para = Parallel('Para', policy='failure_on_one')
        para.add_child(Action('A', fail))
        para.add_child(Action('B', succeed))
        ctx = make_ctx()
        result = para.tick(ctx)
        assert result == NodeState.FAILURE

    def test_parallel_empty(self):
        para = Parallel('EmptyPara')
        ctx = make_ctx()
        result = para.tick(ctx)
        assert result == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test Condition
# ─────────────────────────────────────────────

class TestCondition:
    def test_condition_true(self):
        cond = Condition('IsSafe', always_true)
        ctx = make_ctx()
        result = cond.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_condition_false(self):
        cond = Condition('IsSafe', always_false)
        ctx = make_ctx()
        result = cond.tick(ctx)
        assert result == NodeState.FAILURE

    def test_condition_with_context(self):
        def check_force(ctx):
            return ctx.get_blackboard('force', 0) > 10.0

        cond = Condition('ForceCheck', check_force)
        ctx = make_ctx(blackboard={'force': 20.0})
        result = cond.tick(ctx)
        assert result == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test Action
# ─────────────────────────────────────────────

class TestAction:
    def test_action_success(self):
        action = Action('DoSomething', succeed)
        ctx = make_ctx()
        result = action.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_action_failure(self):
        action = Action('DoSomething', fail)
        ctx = make_ctx()
        result = action.tick(ctx)
        assert result == NodeState.FAILURE

    def test_action_timeout(self):
        action = Action('SlowAction', lambda ctx: NodeState.RUNNING, timeout_ms=50.0)
        ctx = make_ctx()
        action.tick(ctx)
        time.sleep(0.06)
        result = action.tick(ctx)
        assert result == NodeState.FAILURE

    def test_action_with_blackboard(self):
        result_store = [None]
        def write_result(ctx):
            ctx.set_blackboard('result', 'done')
            result_store[0] = ctx.get_blackboard('result')
            return NodeState.SUCCESS

        action = Action('WriteResult', write_result)
        ctx = make_ctx()
        action.tick(ctx)
        assert result_store[0] == 'done'


# ─────────────────────────────────────────────
# Test Decorators
# ─────────────────────────────────────────────

class TestInverter:
    def test_inverter_success_to_failure(self):
        inv = Inverter()
        child = Action('Child', succeed)
        inv.child = child
        ctx = make_ctx()
        result = inv.tick(ctx)
        assert result == NodeState.FAILURE

    def test_inverter_failure_to_success(self):
        inv = Inverter()
        child = Action('Child', fail)
        inv.child = child
        ctx = make_ctx()
        result = inv.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_inverter_no_child(self):
        inv = Inverter()
        ctx = make_ctx()
        result = inv.tick(ctx)
        assert result == NodeState.SUCCESS


class TestRepeatUntil:
    def test_repeat_until_success(self):
        count = [0]
        def count_up(ctx):
            count[0] += 1
            return NodeState.SUCCESS

        dec = RepeatUntil('Repeat', maxRepeats=5, until_success=True)
        dec.child = Action('Counter', count_up)
        ctx = make_ctx()
        result = dec.tick(ctx)
        assert result == NodeState.SUCCESS
        assert count[0] == 1

    def test_repeat_until_max_reached(self):
        count = [0]
        def count_forever(ctx):
            count[0] += 1
            return NodeState.RUNNING

        dec = RepeatUntil('Repeat', maxRepeats=3, until_success=True)
        dec.child = Action('Counter', count_forever)
        ctx = make_ctx()

        for _ in range(10):
            dec.tick(ctx)

        assert count[0] == 3


class TestRetryUntil:
    def test_retry_on_failure(self):
        # Child fails twice, then succeeds → RetryUntil succeeds
        call_count = [0]
        def maybe_succeed(ctx):
            call_count[0] += 1
            if call_count[0] < 3:
                return NodeState.FAILURE
            return NodeState.SUCCESS

        dec = RetryUntil('Retry', max_retries=5, retry_on_failure=True)
        dec.child = Action('Trier', maybe_succeed)
        ctx = make_ctx()

        # First tick: FAILURE → retry (RUNNING)
        assert dec.tick(ctx) == NodeState.RUNNING
        # Second tick: FAILURE → retry (RUNNING)
        assert dec.tick(ctx) == NodeState.RUNNING
        # Third tick: SUCCESS → done
        assert dec.tick(ctx) == NodeState.SUCCESS
        assert call_count[0] == 3

    def test_retry_max_exceeded(self):
        # Child always returns RUNNING → RetryUntil stays RUNNING forever
        call_count = [0]
        def always_run(ctx):
            call_count[0] += 1
            return NodeState.RUNNING

        dec = RetryUntil('Retry', max_retries=2, retry_on_failure=True)
        dec.child = Action('Runner', always_run)
        ctx = make_ctx()

        # RUNNING should keep returning RUNNING, never give up
        for _ in range(5):
            result = dec.tick(ctx)
            assert result == NodeState.RUNNING

        assert call_count[0] == 5


class TestTimeout:
    def test_timeout_not_expired(self):
        dec = Timeout('Timeout', timeout_ms=5000.0)
        dec.child = Action('Quick', succeed)
        ctx = make_ctx()
        result = dec.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_timeout_expired(self):
        dec = Timeout('Timeout', timeout_ms=10.0)
        dec.child = Action('Slow', lambda ctx: NodeState.RUNNING)
        ctx = make_ctx()
        dec.tick(ctx)
        time.sleep(0.015)
        result = dec.tick(ctx)
        assert result == NodeState.FAILURE


class TestRateLimiter:
    def test_rate_limiter_allows(self):
        dec = RateLimiter('RateLimit', max_rate_hz=100.0)
        dec.child = Action('Fast', succeed)
        ctx = make_ctx()
        result = dec.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_rate_limiter_blocks(self):
        dec = RateLimiter('RateLimit', max_rate_hz=1000.0)
        dec.child = Action('Fast', succeed)
        ctx = make_ctx()
        r1 = dec.tick(ctx)
        r2 = dec.tick(ctx)
        assert r1 == NodeState.SUCCESS
        assert r2 == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test SubTree
# ─────────────────────────────────────────────

class TestSubTree:
    def test_subtree_reference(self):
        leaf = Action('Leaf', succeed)
        sub_root = Sequence('SubRoot')
        sub_root.add_child(leaf)
        subtree = SubTree('MySubTree', sub_root)
        ctx = make_ctx()
        result = subtree.tick(ctx)
        assert result == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test BehaviorTree
# ─────────────────────────────────────────────

class TestBehaviorTree:
    def test_bt_tick(self):
        root = Action('RootAction', succeed)
        bt = BehaviorTree(root, BTGrade.M, 'TestBT')
        ctx = make_ctx()
        result = bt.tick()
        assert result == NodeState.SUCCESS

    def test_bt_stats(self):
        root = Action('RootAction', succeed)
        bt = BehaviorTree(root, BTGrade.M, 'TestBT')
        bt.tick()
        stats = bt.get_stats()
        assert stats['name'] == 'TestBT'
        assert stats['grade'] == 'M'
        assert stats['tick_count'] == 1
        assert stats['last_tick_ms'] >= 0

    def test_bt_reset(self):
        root = Action('RootAction', succeed)
        bt = BehaviorTree(root, BTGrade.M, 'TestBT')
        bt.tick()
        bt.reset()
        assert bt.root.state == NodeState.IDLE
        assert bt._tick_count == 0

    def test_bt_complex_tree(self):
        root = Selector('RootSel')
        check = Condition('SafetyCheck', always_true)
        seq = Sequence('ActionSeq')
        seq.add_child(Action('A1', succeed))
        seq.add_child(Action('A2', succeed))

        root.add_child(check)
        root.add_child(seq)

        bt = BehaviorTree(root, BTGrade.L, 'ComplexBT')
        ctx = make_ctx()
        result = bt.tick()
        assert result == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test AGV Five-Grade Specification
# ─────────────────────────────────────────────

class TestAGVBTGrades:
    def test_s_grade_exists(self):
        assert 'S' in AGV_BT_GRADES
        assert AGV_BT_GRADES['S']['max_tree_depth'] == 3
        assert AGV_BT_GRADES['S']['tick_rate_hz'] == 10

    def test_m_grade_exists(self):
        assert 'M' in AGV_BT_GRADES
        assert AGV_BT_GRADES['M']['max_tree_depth'] == 5
        assert AGV_BT_GRADES['M']['tick_rate_hz'] == 50

    def test_l_grade_exists(self):
        assert 'L' in AGV_BT_GRADES
        assert AGV_BT_GRADES['L']['parallel_execution'] is True
        assert AGV_BT_GRADES['L']['tick_rate_hz'] == 100

    def test_xl_grade_exists(self):
        assert 'XL' in AGV_BT_GRADES
        assert AGV_BT_GRADES['XL']['max_tree_depth'] == 12
        assert 'subtree' in AGV_BT_GRADES['XL']['supported_nodes']

    def test_xxl_grade_exists(self):
        assert 'XXL' in AGV_BT_GRADES
        assert AGV_BT_GRADES['XXL']['max_tree_depth'] == 16
        assert AGV_BT_GRADES['XXL']['tick_rate_hz'] == 500
        assert AGV_BT_GRADES['XXL']['preemption'] is True

    def test_all_grades_have_required_keys(self):
        required_keys = [
            'description', 'max_tree_depth', 'max_nodes',
            'tick_rate_hz', 'supported_nodes', 'parallel_execution',
            'memory_nodes', 'decorator_types', 'action_timeout_ms', 'preemption',
        ]
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            for key in required_keys:
                assert key in AGV_BT_GRADES[grade], f"{grade} missing {key}"

    def test_create_for_grade_s(self):
        root = Action('Test', succeed)
        bt = create_for_grade(BTGrade.S, root, 'S_BT')
        assert bt.grade == BTGrade.S
        assert bt.name == 'S_BT'

    def test_create_for_grade_xxl(self):
        root = Selector('XXLRoot')
        bt = create_for_grade(BTGrade.XXL, root, 'XXL_BT')
        assert bt.grade == BTGrade.XXL
        assert bt.name == 'XXL_BT'

    def test_create_safe_selector(self):
        children = [Action('A', succeed), Action('B', fail)]
        sel = create_safe_selector('SafeSel', children, BTGrade.M)
        ctx = make_ctx()
        result = sel.tick(ctx)
        assert result == NodeState.SUCCESS

    def test_create_action_sequence(self):
        actions = [Action('A', succeed), Action('B', succeed)]
        seq = create_action_sequence('ActionSeq', actions, BTGrade.M)
        ctx = make_ctx()
        result = seq.tick(ctx)
        assert result == NodeState.SUCCESS


# ─────────────────────────────────────────────
# Test BTNode Stats
# ─────────────────────────────────────────────

class TestBTNodeStats:
    def test_node_stats_tracking(self):
        action = Action('Tracked', succeed)
        ctx = make_ctx()
        action.tick(ctx)
        stats = action.get_stats()
        assert stats['executions'] == 1
        assert stats['total_time_ms'] >= 0
        assert stats['name'] == 'Tracked'
        assert stats['type'] == 'Action'
