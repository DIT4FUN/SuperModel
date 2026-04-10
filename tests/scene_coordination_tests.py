"""
scene_coordination_tests.py - 场景感知多机协同测试
================================================

测试:
- 场景协调器
- AGV角色分配
- 多场景蜂群控制
- 场景自适应编队
"""

import pytest
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied.scene_coordination import (
    AGVSceneRole,
    SceneCoordinationConfig,
    AGVSceneState,
    SceneCoordinator,
    MultiSceneSwarmController,
)
from embodied.scene_intelligence import (
    SceneType,
    SceneContext,
    SceneFeatures,
    SceneIntelligence,
    SceneConfig,
)


# ============================================================
# 场景协调器测试
# ============================================================

class TestSceneCoordinator:
    """场景协调器测试"""

    def test_initialization(self):
        """测试初始化"""
        coord = SceneCoordinator("AGV_01")
        assert coord.get_role() == AGVSceneRole.FOLLOWER
        assert coord.get_my_state().agv_id == "AGV_01"

    def test_register_agv(self):
        """测试AGV注册"""
        coord = SceneCoordinator("AGV_01")
        coord.register_agv("AGV_02")
        coord.register_agv("AGV_03")
        states = coord.get_all_states()
        assert "AGV_02" in states
        assert "AGV_03" in states

    def test_update_my_state(self):
        """测试本机状态更新"""
        coord = SceneCoordinator("AGV_01")
        coord.update_my_state(
            position=np.array([1.0, 2.0, 0.0]),
            velocity=np.array([0.5, 0.0, 0.0]),
            task="delivery",
            battery_level=0.75,
        )
        state = coord.get_my_state()
        np.testing.assert_array_almost_equal(state.position, [1.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(state.velocity, [0.5, 0.0, 0.0])
        assert state.task == "delivery"
        assert state.battery_level == 0.75

    def test_update_agv_state(self):
        """测试其他AGV状态更新"""
        coord = SceneCoordinator("AGV_01")
        coord.register_agv("AGV_02")
        coord.update_agv_state(
            agv_id="AGV_02",
            position=np.array([3.0, 4.0, 0.0]),
            velocity=np.array([0.3, 0.0, 0.0]),
        )
        state = coord.get_all_states()["AGV_02"]
        np.testing.assert_array_almost_equal(state.position, [3.0, 4.0, 0.0])

    def test_get_nearby_healthy_agvs(self):
        """测试附近健康AGV获取"""
        coord = SceneCoordinator("AGV_01")
        coord.update_my_state(position=np.array([0.0, 0.0, 0.0]))
        coord.register_agv("AGV_02")
        coord.update_agv_state("AGV_02", position=np.array([2.0, 0.0, 0.0]))
        coord.register_agv("AGV_03")
        coord.update_agv_state("AGV_03", position=np.array([10.0, 0.0, 0.0]))  # 超出范围
        coord.register_agv("AGV_04")
        coord.update_agv_state("AGV_04", position=np.array([3.0, 0.0, 0.0]), is_healthy=False)

        nearby = coord.get_nearby_healthy_agvs(max_distance=5.0)
        assert "AGV_02" in nearby
        assert "AGV_03" not in nearby  # 超出范围
        assert "AGV_04" not in nearby  # 不健康

    def test_scene_adaptive_formation_params(self):
        """测试场景自适应编队参数"""
        coord = SceneCoordinator("AGV_01")
        # 默认场景
        params = coord.get_scene_adaptive_formation_params()
        assert 'max_agents' in params
        assert 'safe_distance' in params
        assert 'max_speed' in params  # 现在默认也包含 max_speed

    def test_leader_selection(self):
        """测试Leader选择"""
        coord = SceneCoordinator("AGV_01")
        coord.register_agv("AGV_02")
        coord.register_agv("AGV_03")
        # AGV_01成为Leader
        coord.update_agv_state("AGV_01", position=np.zeros(3), role=AGVSceneRole.LEADER)
        coord.update_agv_state("AGV_02", position=np.array([1, 0, 0]), role=AGVSceneRole.FOLLOWER)
        coord.update_agv_state("AGV_03", position=np.array([2, 0, 0]), role=AGVSceneRole.FOLLOWER)

        leader = coord.get_leader_id()
        assert leader == "AGV_01"


# ============================================================
# AGV场景角色测试
# ============================================================

class TestAGVSceneRole:
    """AGV场景角色测试"""

    def test_all_roles_defined(self):
        """测试所有角色已定义"""
        roles = list(AGVSceneRole)
        assert AGVSceneRole.LEADER in roles
        assert AGVSceneRole.FOLLOWER in roles
        assert AGVSceneRole.SCOUT in roles
        assert AGVSceneRole.GUARD in roles
        assert AGVSceneRole.COORDINATOR in roles

    def test_role_string_values(self):
        """测试角色字符串值"""
        assert AGVSceneRole.LEADER.value == "leader"
        assert AGVSceneRole.SCOUT.value == "scout"
        assert AGVSceneRole.GUARD.value == "guard"


# ============================================================
# 多场景蜂群控制器测试
# ============================================================

class TestMultiSceneSwarmController:
    """多场景蜂群控制器测试"""

    def test_initialization(self):
        """测试初始化"""
        ctrl = MultiSceneSwarmController("AGV_01")
        state = ctrl.get_coordination_state()
        assert state['my_id'] == "AGV_01"
        assert state['is_healthy'] is True

    def test_basic_update(self):
        """测试基本更新"""
        ctrl = MultiSceneSwarmController("AGV_01")
        result = ctrl.update(
            my_position=np.array([0.0, 0.0, 0.0]),
            my_velocity=np.array([0.5, 0.0, 0.0]),
            task="explore",
            battery_level=0.85,
        )
        assert result['my_id'] == "AGV_01"
        assert result['role'] in ['leader', 'follower', 'coordinator', 'scout', 'guard']
        assert result['safe_speed_limit'] > 0

    def test_received_states_processing(self):
        """测试接收状态处理"""
        ctrl = MultiSceneSwarmController("AGV_01")
        received = {
            "AGV_02": {
                "position": [5.0, 0.0, 0.0],
                "velocity": [0.3, 0.0, 0.0],
                "role": "FOLLOWER",
                "task": "follow",
                "battery_level": 0.7,
            }
        }
        result = ctrl.update(
            my_position=np.array([0.0, 0.0, 0.0]),
            received_states=received,
        )
        assert "AGV_02" in result['nearby_agvs']

    def test_scene_adaptive_speed_limit(self):
        """测试场景自适应速度限制"""
        ctrl = MultiSceneSwarmController("AGV_01")

        # 无场景信息
        result = ctrl.update(my_position=np.zeros(3))
        initial_speed = result['safe_speed_limit']

        # 有场景信息
        si = SceneIntelligence()
        si.update(location_hint="hospital")
        ctrl2 = MultiSceneSwarmController("AGV_02", scene_intelligence=si)
        result2 = ctrl2.update(my_position=np.zeros(3))
        assert result2['safe_speed_limit'] <= 1.0  # 医院限速

    def test_role_specific_commands(self):
        """测试角色特定指令"""
        ctrl = MultiSceneSwarmController("AGV_01")
        result = ctrl.update(my_position=np.zeros(3))
        # formation_role 在 update 的返回值里
        assert 'role' in result
        assert 'formation_role' in result

    def test_coordination_state(self):
        """测试协同状态"""
        ctrl = MultiSceneSwarmController("AGV_01")
        ctrl.update(
            my_position=np.array([1.0, 2.0, 0.0]),
            battery_level=0.65,
            is_healthy=True,
        )
        state = ctrl.get_coordination_state()
        assert state['battery'] == 0.65
        assert state['is_healthy'] is True
        assert state['nearby_count'] == 0


# ============================================================
# 场景协调配置测试
# ============================================================

class TestSceneCoordinationConfig:
    """场景协调配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = SceneCoordinationConfig()
        assert config.comm_range == 10.0
        assert config.enable_scene_formation is True
        assert config.enable_collaborative_safety is True
        assert config.grade == "M"

    def test_custom_config(self):
        """测试自定义配置"""
        config = SceneCoordinationConfig(
            grade="XL",
            scene_reassessment_interval=3.0,
            comm_range=20.0,
        )
        assert config.grade == "XL"
        assert config.scene_reassessment_interval == 3.0
        assert config.comm_range == 20.0


# ============================================================
# AGV场景状态测试
# ============================================================

class TestAGVSceneState:
    """AGV场景状态测试"""

    def test_initialization(self):
        """测试初始化"""
        state = AGVSceneState(agv_id="AGV_01")
        assert state.agv_id == "AGV_01"
        assert state.role == AGVSceneRole.FOLLOWER
        assert state.is_healthy is True
        np.testing.assert_array_equal(state.position, np.zeros(3))

    def test_update_position(self):
        """测试位置更新"""
        state = AGVSceneState(agv_id="AGV_01")
        state.update_position(
            pos=np.array([1.0, 2.0, 0.0]),
            vel=np.array([0.5, 0.0, 0.0]),
        )
        np.testing.assert_array_almost_equal(state.position, [1.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(state.velocity, [0.5, 0.0, 0.0])
        assert state.last_scene_update > 0


# ============================================================
# 集成测试
# ============================================================

class TestSceneCoordinationIntegration:
    """场景协同集成测试"""

    def test_multi_agv_scene_flow(self):
        """测试多AGV场景流程"""
        # 创建多个控制器
        ctrl1 = MultiSceneSwarmController("AGV_01")
        ctrl2 = MultiSceneSwarmController("AGV_02")
        ctrl3 = MultiSceneSwarmController("AGV_03")

        # 各AGV更新状态
        r1 = ctrl1.update(my_position=np.array([0.0, 0.0, 0.0]))
        r2 = ctrl2.update(my_position=np.array([2.0, 0.0, 0.0]))
        r3 = ctrl3.update(my_position=np.array([4.0, 0.0, 0.0]))

        # 模拟通信: AGV_02收到AGV_01的状态
        r2_with_neighbor = ctrl2.update(
            my_position=np.array([2.0, 0.0, 0.0]),
            received_states={"AGV_01": {"position": [0.0, 0.0, 0.0]}},
        )

        # AGV_02应该能看到AGV_01
        assert "AGV_01" in r2_with_neighbor['nearby_agvs']

    def test_scene_intelligence_integration(self):
        """测试场景智能集成"""
        si = SceneIntelligence()
        si.update(location_hint="warehouse")

        ctrl = MultiSceneSwarmController("AGV_01", scene_intelligence=si)
        result = ctrl.update(my_position=np.zeros(3))

        # 场景类型应该被传递
        assert result['scene_type'] == SceneType.WAREHOUSE.value
        # 速度限制应该反映场景
        assert result['safe_speed_limit'] >= 1.5

    def test_hospital_scene_high_safety(self):
        """测试医院场景高安全"""
        si = SceneIntelligence()
        si.update(location_hint="hospital")

        ctrl = MultiSceneSwarmController("AGV_01", scene_intelligence=si)
        result = ctrl.update(my_position=np.zeros(3))

        assert result['scene_type'] == SceneType.HOSPITAL.value
        assert result['safe_speed_limit'] <= 1.0
        # 警戒AGV应该有高优先级安全标志
        if result['avoidance_priority']:
            assert result['role'] in ['leader', 'guard', 'coordinator']

    def test_scene_change_triggers_reevaluation(self):
        """测试场景变化触发重评估"""
        si = SceneIntelligence(config=SceneConfig(detection_interval=0.05))
        ctrl = MultiSceneSwarmController("AGV_01", scene_intelligence=si)

        # 仓库场景
        si.update(location_hint="warehouse")
        result1 = ctrl.update(my_position=np.zeros(3))
        speed1 = result1['safe_speed_limit']

        # 短暂等待后切医院场景
        time.sleep(0.1)
        si.update(location_hint="hospital")
        result2 = ctrl.update(my_position=np.zeros(3))
        speed2 = result2['safe_speed_limit']

        # 医院速度应该更低
        assert speed2 < speed1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
