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
collaborative_slam.py - 协同SLAM模块
SuperModel 超模态大模型具身智能系统

多AGV协同SLAM:
- 分布式地图碎片管理
- 地图片段融合与对齐
- 基于ICP/特征匹配的地图注册
- 协同定位增强
- 冲突区域地图一致性维护
- 分布式图优化
"""

from __future__ import annotations

import time
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from collections import defaultdict
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

logger = logging.getLogger(__name__)

__all__ = [
    'MapFragment',
    'FeaturePoint',
    'PoseConstraint',
    'CollaborativeSlamAgent',
    'MapFusionEngine',
    'CollaborativeSlamCoordinator',
    'get_collaborative_slam_coordinator',
]


class MapQuality(Enum):
    """地图质量等级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


@dataclass
class FeaturePoint:
    """特征点"""
    point_id: str
    position: np.ndarray          # [x, y, z] or [x, y]
    descriptor: Optional[np.ndarray] = None
    feature_type: str = "orb"      # orb, sift, surf, akaze
    observations: int = 1
    last_seen: float = 0.0
    confirmed: bool = False

    def merge_with(self, other: "FeaturePoint", tolerance: float = 0.3) -> bool:
        """合并另一个特征点（距离在容忍范围内）"""
        dist = np.linalg.norm(self.position - other.position)
        if dist < tolerance:
            # 加权平均位置
            w = self.observations / (self.observations + other.observations)
            self.position = w * self.position + (1 - w) * other.position
            self.observations += other.observations
            self.last_seen = max(self.last_seen, other.last_seen)
            return True
        return False


@dataclass
class MapFragment:
    """地图碎片"""
    fragment_id: str
    agent_id: str
    timestamp: float
    position: np.ndarray           # 碎片原点世界坐标 [x, y, theta]
    features: Dict[str, FeaturePoint] = field(default_factory=dict)
    obstacles: List[np.ndarray] = field(default_factory=list)
    boundaries: np.ndarray = field(default_factory=lambda: np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    quality: MapQuality = MapQuality.MEDIUM
    coverage_radius: float = 5.0   # m
    resolution: float = 0.05       # m/cell
    parent_fragment_id: Optional[str] = None
    children_fragment_ids: List[str] = field(default_factory=list)
    verified_by: Set[str] = field(default_factory=set)  # 确认此碎片的agent集合

    def add_feature(self, feature: FeaturePoint) -> None:
        """添加特征点"""
        self.features[feature.point_id] = feature

    def add_obstacle(self, obstacle: np.ndarray) -> None:
        """添加障碍物"""
        self.obstacles.append(obstacle)

    def merge_fragment(self, other: "MapFragment", transform: np.ndarray) -> int:
        """
        合并另一个碎片（给定坐标变换矩阵）
        返回合并的特征点数量
        """
        merged_count = 0
        other_transformed = other.transform_points(transform)

        for pid, feat in other_transformed.features.items():
            if pid in self.features:
                if self.features[pid].merge_with(feat):
                    merged_count += 1
            else:
                self.features[pid] = feat
                merged_count += 1

        self.obstacles.extend(other_transformed.obstacles)
        self.quality = MapQuality(max(self.quality.value, other.quality.value))
        return merged_count

    def transform_points(self, transform: np.ndarray) -> MapFragment:
        """对碎片应用坐标变换"""
        import copy
        new_fragment = copy.deepcopy(self)
        new_fragment.fragment_id = f"{self.fragment_id}_t"
        # 应用2D变换
        cos_t = np.cos(transform[2])
        sin_t = np.sin(transform[2])
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        t = transform[:2]
        for feat in new_fragment.features.values():
            feat.position[:2] = R @ feat.position[:2] + t
        new_fragment.obstacles = [R @ obs[:2] + t for obs in self.obstacles]
        new_fragment.position = self.position + transform
        return new_fragment

    def to_grid_map(self, resolution: Optional[float] = None) -> np.ndarray:
        """转换为栅格地图"""
        if resolution is None:
            resolution = self.resolution
        bounds = self.boundaries
        min_xy = bounds.min(axis=0)
        max_xy = bounds.max(axis=0)
        size_xy = np.ceil((max_xy - min_xy) / resolution).astype(int) + 1
        grid = np.zeros(size_xy, dtype=np.uint8)
        # 障碍物标记
        for obs in self.obstacles:
            px = int((obs[0] - min_xy[0]) / resolution)
            py = int((obs[1] - min_xy[1]) / resolution)
            if 0 <= px < size_xy[0] and 0 <= py < size_xy[1]:
                grid[px, py] = 255
        return grid

    def get_overlap_score(self, other: "MapFragment", transform: np.ndarray) -> float:
        """计算与另一个碎片的重叠得分"""
        transformed = other.transform_points(transform)
        # 基于特征点匹配计算重叠
        matched = 0
        for pid in self.features:
            if pid in transformed.features:
                dist = np.linalg.norm(
                    self.features[pid].position - transformed.features[pid].position
                )
                if dist < 0.5:
                    matched += 1
        total = len(self.features)
        if total == 0:
            return 0.0
        return matched / total


@dataclass
class PoseConstraint:
    """位姿约束（用于图优化）"""
    constraint_id: str
    from_agent: str
    to_agent: str
    relative_pose: np.ndarray      # [dx, dy, dtheta]
    information_matrix: np.ndarray  # 6x6 协方差矩阵的逆
    timestamp: float


class CollaborativeSlamAgent:
    """
    协作SLAM的单个AGV代理
    管理本地地图碎片、特征检测、位姿估计
    """

    def __init__(
        self,
        agent_id: str,
        initial_pose: np.ndarray,
        comm_callback: Optional[Callable[[str, MapFragment], None]] = None,
    ):
        self.agent_id = agent_id
        self.pose = initial_pose.copy()  # [x, y, theta]
        self.uncertainty = np.eye(3) * 0.01
        self.fragments: Dict[str, MapFragment] = {}
        self.current_fragment: Optional[MapFragment] = None
        self.comm_callback = comm_callback
        self.local_features: Dict[str, FeaturePoint] = {}
        self.feature_counter = 0
        self._fragment_lock = threading.Lock()
        self._pose_lock = threading.Lock()

    def update_pose(self, delta: np.ndarray, measurement_noise: float = 0.01) -> None:
        """
        更新自身位姿
        delta: [dx, dy, dtheta] in body frame or world frame
        """
        with self._pose_lock:
            cos_t = np.cos(self.pose[2])
            sin_t = np.sin(self.pose[2])
            # 转换到世界坐标系
            dx_world = cos_t * delta[0] - sin_t * delta[1]
            dy_world = sin_t * delta[0] + cos_t * delta[1]
            self.pose[0] += dx_world
            self.pose[1] += dy_world
            self.pose[2] += delta[2]
            # 归一化角度
            self.pose[2] = np.arctan2(np.sin(self.pose[2]), np.cos(self.pose[2]))
            # 更新不确定性
            self.uncertainty += np.eye(3) * measurement_noise

    def set_pose(self, pose: np.ndarray, uncertainty: Optional[np.ndarray] = None) -> None:
        """设置绝对位姿（来自外部定位或融合结果）"""
        with self._pose_lock:
            self.pose = pose.copy()
            if uncertainty is not None:
                self.uncertainty = uncertainty

    def detect_features(
        self,
        sensor_data: Dict[str, Any],
        min_distance: float = 0.3,
    ) -> List[FeaturePoint]:
        """
        从传感器数据中检测特征点
        支持激光雷达点云、深度图像等
        返回新检测到的特征点列表
        """
        new_features = []

        # 模拟特征检测（实际实现需要接入具体传感器）
        if "lidar_points" in sensor_data:
            points = sensor_data["lidar_points"]
            for pt in points:
                # 简单的体素网格过滤
                pt_id = f"f{self.feature_counter}"
                self.feature_counter += 1
                feat = FeaturePoint(
                    point_id=pt_id,
                    position=pt.copy(),
                    feature_type="lidar_corner",
                    observations=1,
                    last_seen=time.time(),
                    confirmed=False,
                )
                new_features.append(feat)

        elif "depth_image" in sensor_data:
            # 从深度图像提取角点/边缘特征（模拟）
            pass

        # 去重：过滤距离过近的特征
        filtered = []
        for feat in new_features:
            is_dup = False
            for existing in self.local_features.values():
                if np.linalg.norm(feat.position - existing.position) < min_distance:
                    is_dup = True
                    break
            if not is_dup:
                self.local_features[feat.point_id] = feat
                filtered.append(feat)

        return filtered

    def start_new_fragment(self) -> MapFragment:
        """开始新的地图碎片"""
        with self._fragment_lock:
            frag_id = f"frag_{self.agent_id}_{int(time.time() * 1000)}"
            fragment = MapFragment(
                fragment_id=frag_id,
                agent_id=self.agent_id,
                timestamp=time.time(),
                position=self.pose.copy(),
                quality=MapQuality.LOW,
            )
            self.fragments[frag_id] = fragment
            self.current_fragment = fragment
            if self.current_fragment and self.current_fragment.parent_fragment_id is None:
                self.current_fragment.parent_fragment_id = frag_id
            return fragment

    def add_features_to_current_fragment(
        self,
        features: List[FeaturePoint],
        transform_to_world: Optional[np.ndarray] = None,
    ) -> int:
        """将特征点添加到当前碎片（自动转换到世界坐标）"""
        if self.current_fragment is None:
            return 0
        with self._fragment_lock:
            added = 0
            for feat in features:
                if transform_to_world is None:
                    world_pos = feat.position.copy()
                else:
                    cos_t = np.cos(transform_to_world[2])
                    sin_t = np.sin(transform_to_world[2])
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    world_pos = R @ feat.position[:2] + transform_to_world[:2]
                    world_pos = np.append(world_pos, feat.position[2] if len(feat.position) > 2 else 0)
                feat_world = FeaturePoint(
                    point_id=feat.point_id,
                    position=world_pos,
                    descriptor=feat.descriptor,
                    feature_type=feat.feature_type,
                    observations=feat.observations,
                    last_seen=time.time(),
                    confirmed=False,
                )
                self.current_fragment.add_feature(feat_world)
                added += 1
            return added

    def finalize_fragment(self, quality: MapQuality = MapQuality.MEDIUM) -> Optional[MapFragment]:
        """完成当前碎片，准备上报"""
        if self.current_fragment is None:
            return None
        with self._fragment_lock:
            self.current_fragment.quality = quality
            frag = self.current_fragment
            # 广播给其他代理
            if self.comm_callback:
                self.comm_callback(self.agent_id, frag)
            self.current_fragment = None
            return frag

    def receive_remote_fragment(
        self,
        remote_fragment: MapFragment,
        estimated_transform: np.ndarray,
    ) -> bool:
        """
        接收并融合远程地图碎片
        estimated_transform: remote -> self 的变换估计
        返回是否成功融合
        """
        transformed = remote_fragment.transform_points(estimated_transform)
        with self._fragment_lock:
            merged_features = 0
            for pid, feat in transformed.features.items():
                if pid in self.local_features:
                    if self.local_features[pid].merge_with(feat):
                        merged_features += 1
                else:
                    self.local_features[pid] = feat
                    merged_features += 1
            return merged_features > 0

    def get_local_map_summary(self) -> Dict[str, Any]:
        """获取本地地图摘要（用于广播）"""
        return {
            "agent_id": self.agent_id,
            "pose": self.pose.tolist(),
            "uncertainty": self.uncertainty.tolist(),
            "fragment_count": len(self.fragments),
            "local_feature_count": len(self.local_features),
            "timestamp": time.time(),
        }


class MapFusionEngine:
    """
    地图融合引擎
    实现多碎片地图的对齐、融合、一致性维护
    """

    def __init__(self, resolution: float = 0.05):
        self.resolution = resolution
        self.global_features: Dict[str, FeaturePoint] = {}
        self.global_obstacles: List[np.ndarray] = []
        self.fragments: Dict[str, MapFragment] = {}
        self.constraints: List[PoseConstraint] = []
        self._fusion_lock = threading.Lock()
        self._merge_history: List[Dict] = []

    def register_fragment(self, fragment: MapFragment) -> str:
        """注册新的地图碎片"""
        with self._fusion_lock:
            self.fragments[fragment.fragment_id] = fragment
            logger.info(
                f"[Fusion] Registered fragment {fragment.fragment_id} "
                f"from agent {fragment.agent_id}"
            )
            return fragment.fragment_id

    def estimate_transform_icp(
        self,
        source: MapFragment,
        target: MapFragment,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
    ) -> Tuple[np.ndarray, float]:
        """
        使用ICP估算两个碎片的相对变换
        返回: (transform [dx, dy, dtheta], fitness_score)
        """
        # 获取源和目标的2D特征点
        src_pts = np.array([f.position[:2] for f in source.features.values()]) if source.features else np.array([[0, 0]])
        tgt_pts = np.array([f.position[:2] for f in target.features.values()]) if target.features else np.array([[0, 0]])

        if len(src_pts) < 3 or len(tgt_pts) < 3:
            # 特征点不足，使用质心估计
            src_center = src_pts.mean(axis=0) if len(src_pts) > 0 else np.zeros(2)
            tgt_center = tgt_pts.mean(axis=0) if len(tgt_pts) > 0 else np.zeros(2)
            dxy = tgt_center - src_center
            return np.array([dxy[0], dxy[1], 0.0]), 0.0

        # 简化的ICP实现
        transform = np.zeros(3)  # [dx, dy, dtheta]
        prev_fitness = 0.0

        for it in range(max_iterations):
            # 1. 找到最近邻对应点
            cos_t = np.cos(transform[2])
            sin_t = np.sin(transform[2])
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            transformed_src = (R @ src_pts.T).T + transform[:2]

            distances = np.linalg.norm(
                transformed_src[:, np.newaxis] - tgt_pts[np.newaxis, :], axis=2
            )
            correspondences = np.argmin(distances, axis=1)

            # 2. 计算变换
            src_matched = transformed_src
            tgt_matched = tgt_pts[correspondences]

            # 减去质心
            src_mean = src_matched.mean(axis=0)
            tgt_mean = tgt_matched.mean(axis=0)
            src_centered = src_matched - src_mean
            tgt_centered = tgt_matched - tgt_mean

            # SVD求解旋转
            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T

            # 确保右手系
            if np.linalg.det(R_opt) < 0:
                Vt[-1] *= -1
                R_opt = Vt.T @ U.T

            theta_opt = np.arctan2(R_opt[1, 0], R_opt[0, 0])
            dxy_opt = tgt_mean - R_opt @ src_mean

            new_transform = np.array([dxy_opt[0], dxy_opt[1], theta_opt])
            transform = new_transform

            # 3. 计算fitness score
            fitness = np.mean(np.min(distances, axis=1))

            if abs(prev_fitness - fitness) < tolerance:
                break
            prev_fitness = fitness

        return transform, 1.0 / (1.0 + prev_fitness)

    def fuse_fragments(
        self,
        fragment_a: MapFragment,
        fragment_b: MapFragment,
        transform: Optional[np.ndarray] = None,
        use_icp: bool = True,
    ) -> MapFragment:
        """
        融合两个地图碎片
        返回融合后的新碎片
        """
        with self._fusion_lock:
            if use_icp and transform is None:
                transform, score = self.estimate_transform_icp(fragment_a, fragment_b)
                logger.info(
                    f"[Fusion] ICP transform: dx={transform[0]:.3f}, "
                    f"dy={transform[1]:.3f}, dtheta={np.degrees(transform[2]):.2f}°, "
                    f"score={score:.3f}"
                )

            # 合并特征
            merged_fragment = MapFragment(
                fragment_id=f"fused_{fragment_a.fragment_id}_{fragment_b.fragment_id}",
                agent_id="fusion_center",
                timestamp=time.time(),
                position=fragment_a.position.copy(),
                quality=MapQuality(
                    max(fragment_a.quality.value, fragment_b.quality.value)
                ),
            )

            # 合并A的特征
            for pid, feat in fragment_a.features.items():
                merged_fragment.add_feature(feat)

            # 合并B的特征（带变换）
            if transform is not None:
                fragment_b_transformed = fragment_b.transform_points(transform)
                for pid, feat in fragment_b_transformed.features.items():
                    if pid in merged_fragment.features:
                        merged_fragment.features[pid].merge_with(feat)
                    else:
                        merged_fragment.add_feature(feat)
                merged_fragment.obstacles = list(fragment_a.obstacles)
                merged_fragment.obstacles.extend(fragment_b_transformed.obstacles)
            else:
                for pid, feat in fragment_b.features.items():
                    if pid in merged_fragment.features:
                        merged_fragment.features[pid].merge_with(feat)
                    else:
                        merged_fragment.add_feature(feat)
                merged_fragment.obstacles = list(fragment_a.obstacles)
                merged_fragment.obstacles.extend(fragment_b.obstacles)

            # 更新全局特征库
            for pid, feat in merged_fragment.features.items():
                if pid not in self.global_features:
                    self.global_features[pid] = feat

            self._merge_history.append({
                "fragment_a": fragment_a.fragment_id,
                "fragment_b": fragment_b.fragment_id,
                "transform": transform.tolist() if transform is not None else None,
                "feature_count": len(merged_fragment.features),
                "timestamp": time.time(),
            })

            return merged_fragment

    def get_global_map(self) -> Dict[str, Any]:
        """获取全局地图摘要"""
        with self._fusion_lock:
            return {
                "fragment_count": len(self.fragments),
                "global_feature_count": len(self.global_features),
                "obstacle_count": len(self.global_obstacles),
                "constraint_count": len(self.constraints),
                "merge_events": len(self._merge_history),
            }


class CollaborativeSlamCoordinator:
    """
    协同SLAM协调器
    管理多个AGV的协作SLAM会话
    """

    def __init__(
        self,
        coordinator_id: Optional[str] = None,
        fusion_resolution: float = 0.05,
    ):
        self.coordinator_id = coordinator_id or f"cslam_{uuid.uuid4().hex[:8]}"
        self.agents: Dict[str, CollaborativeSlamAgent] = {}
        self.fusion_engine = MapFusionEngine(resolution=fusion_resolution)
        self.fragment_buffer: Dict[str, List[MapFragment]] = defaultdict(list)
        self._lock = threading.Lock()
        self.running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._fragment_age_threshold = 300.0  # 碎片最大保存时间(s)

        # 广播主题
        self.broadcast_topics = [
            "map_fragment",
            "pose_update",
            "localization_query",
            "map_merge_request",
        ]

        logger.info(f"[CoSLAM] Coordinator {self.coordinator_id} initialized")

    def register_agent(
        self,
        agent_id: str,
        initial_pose: np.ndarray,
    ) -> CollaborativeSlamAgent:
        """注册一个新的AGV代理"""
        def comm_callback(sender_id: str, fragment: MapFragment):
            self._handle_fragment_broadcast(sender_id, fragment)

        with self._lock:
            agent = CollaborativeSlamAgent(
                agent_id=agent_id,
                initial_pose=initial_pose,
                comm_callback=comm_callback,
            )
            self.agents[agent_id] = agent
            logger.info(f"[CoSLAM] Registered agent {agent_id} at pose {initial_pose}")
            return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """注销AGV代理"""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"[CoSLAM] Unregistered agent {agent_id}")
                return True
            return False

    def _handle_fragment_broadcast(self, sender_id: str, fragment: MapFragment) -> None:
        """处理来自其他代理的碎片广播"""
        with self._lock:
            self.fragment_buffer[sender_id].append(fragment)
            logger.debug(
                f"[CoSLAM] Buffering fragment {fragment.fragment_id} "
                f"from {sender_id}"
            )

    def process_fragment_buffer(self) -> Dict[str, int]:
        """
        处理碎片缓冲区，尝试融合
        返回每个发送者的融合数量
        """
        fusion_counts: Dict[str, int] = defaultdict(int)

        with self._lock:
            for sender_id, fragments in self.fragment_buffer.items():
                if sender_id not in self.agents:
                    continue
                agent = self.agents[sender_id]
                for fragment in fragments:
                    # 注册到融合引擎
                    self.fusion_engine.register_fragment(fragment)
                    # 尝试与本地地图融合
                    fusion_counts[sender_id] += self._try_fuse_with_local(agent, fragment)
                fragments.clear()

        return dict(fusion_counts)

    def _try_fuse_with_local(
        self,
        agent: CollaborativeSlamAgent,
        remote_fragment: MapFragment,
    ) -> int:
        """尝试将远程碎片与代理本地地图融合"""
        fused_count = 0
        for local_frag_id, local_frag in agent.fragments.items():
            # 估算变换
            transform, score = self.fusion_engine.estimate_transform_icp(
                remote_fragment, local_frag
            )
            if score > 0.3:  # 匹配得分阈值
                merged = self.fusion_engine.fuse_fragments(
                    remote_fragment, local_frag, transform
                )
                agent.fragments[merged.fragment_id] = merged
                fused_count += 1
                logger.info(
                    f"[CoSLAM] Fused fragment {remote_fragment.fragment_id} "
                    f"with {local_frag_id} (score={score:.3f})"
                )
        return fused_count

    def query_location(
        self,
        query_agent_id: str,
        query_pose: np.ndarray,
        radius: float = 5.0,
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        查询某位置附近已知的地图信息
        返回: [(agent_id, feature_position, confidence), ...]
        """
        results = []
        with self._lock:
            for agent_id, agent in self.agents.items():
                if agent_id == query_agent_id:
                    continue
                for feat in agent.local_features.values():
                    dist = np.linalg.norm(feat.position[:2] - query_pose[:2])
                    if dist < radius:
                        confidence = feat.observations / (1 + dist)
                        results.append((agent_id, feat.position, confidence))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:10]

    def distribute_map_update(
        self,
        recipient_agent_id: str,
        map_update: Dict[str, Any],
    ) -> bool:
        """向指定代理推送地图更新"""
        with self._lock:
            if recipient_agent_id not in self.agents:
                return False
            agent = self.agents[recipient_agent_id]
            # 模拟接收远程碎片
            # 实际实现通过comm_callback
            return True

    def start(self) -> None:
        """启动协调器"""
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info(f"[CoSLAM] Coordinator {self.coordinator_id} started")

    def stop(self) -> None:
        """停止协调器"""
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info(f"[CoSLAM] Coordinator {self.coordinator_id} stopped")

    def _worker_loop(self) -> None:
        """后台工作循环"""
        while self.running:
            try:
                self.process_fragment_buffer()
                self._cleanup_old_fragments()
            except Exception as e:
                logger.error(f"[CoSLAM] Worker loop error: {e}")
            time.sleep(1.0)

    def _cleanup_old_fragments(self) -> None:
        """清理过期的地图碎片"""
        current_time = time.time()
        for agent in self.agents.values():
            expired = [
                fid for fid, frag in agent.fragments.items()
                if current_time - frag.timestamp > self._fragment_age_threshold
            ]
            for fid in expired:
                del agent.fragments[fid]

    def get_coordinator_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        return {
            "coordinator_id": self.coordinator_id,
            "registered_agents": len(self.agents),
            "buffered_fragments": sum(len(v) for v in self.fragment_buffer.values()),
            "running": self.running,
            "global_map": self.fusion_engine.get_global_map(),
            "merge_history_count": len(self.fusion_engine._merge_history),
        }


# 全局协调器实例
_cslam_instance: Optional[CollaborativeSlamCoordinator] = None
_cslam_lock = threading.Lock()


def get_collaborative_slam_coordinator(
    coordinator_id: Optional[str] = None,
    fusion_resolution: float = 0.05,
    create: bool = True,
) -> CollaborativeSlamCoordinator:
    """获取协同SLAM协调器全局实例"""
    global _cslam_instance
    with _cslam_lock:
        if _cslam_instance is None and create:
            _cslam_instance = CollaborativeSlamCoordinator(
                coordinator_id=coordinator_id,
                fusion_resolution=fusion_resolution,
            )
            _cslam_instance.start()
        return _cslam_instance
