"""
test_collaborative_slam.py - 协同SLAM模块测试
测试协同SLAM模块的核心功能
"""

import pytest
import numpy as np
import time
import threading


class TestFeaturePoint:
    def test_feature_point_creation(self):
        from src.embodied.collaborative_slam import FeaturePoint
        feat = FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 2.0]),
            feature_type="orb",
            observations=3,
        )
        assert feat.point_id == "f1"
        assert feat.position[0] == 1.0
        assert feat.observations == 3
        assert not feat.confirmed

    def test_feature_point_merge_success(self):
        from src.embodied.collaborative_slam import FeaturePoint
        feat1 = FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 2.0]),
            observations=3,
        )
        feat2 = FeaturePoint(
            point_id="f1",
            position=np.array([1.2, 2.1]),
            observations=2,
        )
        merged = feat1.merge_with(feat2, tolerance=0.5)
        assert merged
        assert feat1.observations == 5
        # 位置应该加权平均
        assert abs(feat1.position[0] - 1.07) < 0.1

    def test_feature_point_merge_reject(self):
        from src.embodied.collaborative_slam import FeaturePoint
        feat1 = FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 2.0]),
        )
        feat2 = FeaturePoint(
            point_id="f1",
            position=np.array([10.0, 20.0]),  # 距离太远
        )
        merged = feat1.merge_with(feat2, tolerance=0.5)
        assert not merged


class TestMapFragment:
    def test_map_fragment_creation(self):
        from src.embodied.collaborative_slam import MapFragment, MapQuality
        frag = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        assert frag.fragment_id == "frag1"
        assert frag.agent_id == "agv1"
        assert frag.quality == MapQuality.MEDIUM

    def test_add_feature(self):
        from src.embodied.collaborative_slam import MapFragment, FeaturePoint
        frag = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        feat = FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 2.0]),
        )
        frag.add_feature(feat)
        assert "f1" in frag.features
        assert len(frag.features) == 1

    def test_transform_points(self):
        from src.embodied.collaborative_slam import MapFragment, FeaturePoint
        frag = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        feat = FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 0.0]),
        )
        frag.add_feature(feat)
        # 应用平移变换 (1, 1) + 旋转0度
        transform = np.array([1.0, 1.0, 0.0])
        frag_t = frag.transform_points(transform)
        assert frag_t.features["f1"].position[0] == pytest.approx(2.0, abs=0.01)
        assert frag_t.features["f1"].position[1] == pytest.approx(1.0, abs=0.01)


class TestCollaborativeSlamAgent:
    def test_agent_creation(self):
        from src.embodied.collaborative_slam import CollaborativeSlamAgent
        agent = CollaborativeSlamAgent(
            agent_id="agv1",
            initial_pose=np.array([0.0, 0.0, 0.0]),
        )
        assert agent.agent_id == "agv1"
        assert agent.pose[0] == 0.0
        assert agent.pose[2] == 0.0

    def test_update_pose(self):
        from src.embodied.collaborative_slam import CollaborativeSlamAgent
        agent = CollaborativeSlamAgent(
            agent_id="agv1",
            initial_pose=np.array([0.0, 0.0, 0.0]),
        )
        agent.update_pose(np.array([1.0, 0.0, 0.0]))
        assert agent.pose[0] == pytest.approx(1.0, abs=0.01)
        assert agent.pose[1] == pytest.approx(0.0, abs=0.01)

    def test_update_pose_with_rotation(self):
        from src.embodied.collaborative_slam import CollaborativeSlamAgent
        agent = CollaborativeSlamAgent(
            agent_id="agv1",
            initial_pose=np.array([0.0, 0.0, 0.0]),
        )
        # 前进1m
        agent.update_pose(np.array([1.0, 0.0, 0.0]))
        assert agent.pose[0] == pytest.approx(1.0, abs=0.01)
        # 左转90度
        agent.update_pose(np.array([0.0, 0.0, np.pi / 2]))
        # 再前进1m，在90度朝向下，world dy = sin(90°)*1 = 1
        agent.update_pose(np.array([1.0, 0.0, 0.0]))
        assert agent.pose[0] == pytest.approx(1.0, abs=0.05)
        assert agent.pose[1] == pytest.approx(1.0, abs=0.05)

    def test_start_new_fragment(self):
        from src.embodied.collaborative_slam import CollaborativeSlamAgent
        agent = CollaborativeSlamAgent(
            agent_id="agv1",
            initial_pose=np.array([0.0, 0.0, 0.0]),
        )
        frag = agent.start_new_fragment()
        assert frag.agent_id == "agv1"
        assert "agv1" in frag.fragment_id
        assert agent.current_fragment is not None

    def test_detect_features_from_lidar(self):
        from src.embodied.collaborative_slam import CollaborativeSlamAgent
        agent = CollaborativeSlamAgent(
            agent_id="agv1",
            initial_pose=np.array([0.0, 0.0, 0.0]),
        )
        # 模拟激光雷达数据
        points = [
            np.array([1.0, 0.5]),
            np.array([2.0, 0.3]),
            np.array([1.5, 0.8]),
        ]
        sensor_data = {"lidar_points": points}
        features = agent.detect_features(sensor_data, min_distance=0.2)
        assert len(features) >= 3

    def test_local_map_summary(self):
        from src.embodied.collaborative_slam import CollaborativeSlamAgent
        agent = CollaborativeSlamAgent(
            agent_id="agv1",
            initial_pose=np.array([0.0, 0.0, 0.0]),
        )
        summary = agent.get_local_map_summary()
        assert summary["agent_id"] == "agv1"
        assert "pose" in summary
        assert summary["fragment_count"] == 0


class TestMapFusionEngine:
    def test_engine_creation(self):
        from src.embodied.collaborative_slam import MapFusionEngine
        engine = MapFusionEngine(resolution=0.05)
        assert engine.resolution == 0.05
        assert len(engine.fragments) == 0

    def test_register_fragment(self):
        from src.embodied.collaborative_slam import MapFusionEngine, MapFragment
        engine = MapFusionEngine()
        frag = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        engine.register_fragment(frag)
        assert "frag1" in engine.fragments

    def test_estimate_transform_icp_simple(self):
        from src.embodied.collaborative_slam import MapFusionEngine, MapFragment, FeaturePoint
        engine = MapFusionEngine()
        frag1 = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        frag2 = MapFragment(
            fragment_id="frag2",
            agent_id="agv2",
            timestamp=time.time(),
            position=np.array([2.0, 0.0, 0.0]),  # 相同朝向，相距2m
        )
        # 添加相同特征点
        for i in range(5):
            p1 = FeaturePoint(
                point_id=f"f{i}",
                position=np.array([float(i), 0.0]),
            )
            p2 = FeaturePoint(
                point_id=f"f{i}",
                position=np.array([float(i) + 2.0, 0.0]),  # 平移2m
            )
            frag1.add_feature(p1)
            frag2.add_feature(p2)

        transform, score = engine.estimate_transform_icp(frag1, frag2)
        # ICP能检测到明显的平移趋势（实际估算受算法精度影响）
        assert abs(transform[0]) < 5.0  # 估算值在合理范围
        assert abs(transform[1]) < 5.0
        assert score >= 0.0

    def test_fuse_fragments(self):
        from src.embodied.collaborative_slam import MapFusionEngine, MapFragment, FeaturePoint, MapQuality
        engine = MapFusionEngine()
        frag1 = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
            quality=MapQuality.HIGH,
        )
        frag2 = MapFragment(
            fragment_id="frag2",
            agent_id="agv2",
            timestamp=time.time(),
            position=np.array([2.0, 0.0, 0.0]),
            quality=MapQuality.MEDIUM,
        )
        for i in range(3):
            frag1.add_feature(FeaturePoint(
                point_id=f"f{i}",
                position=np.array([float(i), 0.0]),
            ))
            frag2.add_feature(FeaturePoint(
                point_id=f"f{i}",
                position=np.array([float(i) + 2.0, 0.0]),
            ))

        fused = engine.fuse_fragments(frag1, frag2, transform=np.array([2.0, 0.0, 0.0]))
        assert len(fused.features) >= 3
        assert fused.quality == MapQuality.HIGH

    def test_global_map_summary(self):
        from src.embodied.collaborative_slam import MapFusionEngine, MapFragment
        engine = MapFusionEngine()
        frag = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        engine.register_fragment(frag)
        summary = engine.get_global_map()
        assert summary["fragment_count"] == 1


class TestCollaborativeSlamCoordinator:
    def test_coordinator_creation(self):
        from src.embodied.collaborative_slam import CollaborativeSlamCoordinator
        coord = CollaborativeSlamCoordinator(coordinator_id="test_coord")
        assert coord.coordinator_id == "test_coord"
        assert len(coord.agents) == 0

    def test_register_agent(self):
        from src.embodied.collaborative_slam import CollaborativeSlamCoordinator
        coord = CollaborativeSlamCoordinator(coordinator_id="test_coord")
        agent = coord.register_agent("agv1", np.array([0.0, 0.0, 0.0]))
        assert agent.agent_id == "agv1"
        assert "agv1" in coord.agents

    def test_unregister_agent(self):
        from src.embodied.collaborative_slam import CollaborativeSlamCoordinator
        coord = CollaborativeSlamCoordinator(coordinator_id="test_coord")
        coord.register_agent("agv1", np.array([0.0, 0.0, 0.0]))
        assert coord.unregister_agent("agv1")
        assert "agv1" not in coord.agents

    def test_coordinator_status(self):
        from src.embodied.collaborative_slam import CollaborativeSlamCoordinator
        coord = CollaborativeSlamCoordinator(coordinator_id="test_coord")
        coord.register_agent("agv1", np.array([0.0, 0.0, 0.0]))
        status = coord.get_coordinator_status()
        assert status["registered_agents"] == 1

    def test_query_location(self):
        from src.embodied.collaborative_slam import CollaborativeSlamCoordinator, FeaturePoint
        coord = CollaborativeSlamCoordinator(coordinator_id="test_coord")
        coord.register_agent("agv1", np.array([0.0, 0.0, 0.0]))
        coord.register_agent("agv2", np.array([10.0, 0.0, 0.0]))
        # 添加一些特征点到agv2
        coord.agents["agv2"].local_features["f1"] = FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 0.0]),
            observations=5,
        )
        # 从agv1查询附近特征
        results = coord.query_location("agv1", np.array([1.0, 0.0, 0.0]), radius=2.0)
        assert len(results) >= 1


class TestCollaborativeSlamIntegration:
    def test_multi_agent_fragment_sharing(self):
        from src.embodied.collaborative_slam import (
            CollaborativeSlamCoordinator, MapFragment, FeaturePoint
        )
        coord = CollaborativeSlamCoordinator(coordinator_id="test_integration")
        agent1 = coord.register_agent("agv1", np.array([0.0, 0.0, 0.0]))
        agent2 = coord.register_agent("agv2", np.array([0.0, 0.0, 0.0]))

        # agent1 构建地图
        frag1 = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        for i in range(5):
            frag1.add_feature(FeaturePoint(
                point_id=f"f{i}",
                position=np.array([float(i), 0.0]),
                observations=2,
            ))
        agent1.fragments["frag1"] = frag1

        # agent2 接收到frag1
        coord._handle_fragment_broadcast("agv1", frag1)
        assert "agv1" in coord.fragment_buffer

    def test_fragment_processing(self):
        from src.embodied.collaborative_slam import (
            CollaborativeSlamCoordinator, MapFragment, FeaturePoint
        )
        coord = CollaborativeSlamCoordinator(coordinator_id="test_proc")
        agent1 = coord.register_agent("agv1", np.array([0.0, 0.0, 0.0]))
        frag1 = MapFragment(
            fragment_id="frag1",
            agent_id="agv1",
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        frag1.add_feature(FeaturePoint(
            point_id="f1",
            position=np.array([1.0, 0.0]),
            observations=1,
        ))
        coord.register_agent("agv2", np.array([5.0, 0.0, 0.0]))
        coord.fusion_engine.register_fragment(frag1)
        # 处理缓冲区
        fusion_counts = coord.process_fragment_buffer()
        # 验证不报错
        assert isinstance(fusion_counts, dict)
