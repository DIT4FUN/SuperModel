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
场景理解模块
============

融合多传感器数据，构建统一场景表征:
- 3D 场景重建
- 物体检测与识别
- 空间关系图谱
- 场景语义分割
- 动态物体跟踪

依赖模块:
- sensors.vision: 深度/双目视觉
- sensors.force: 接触力反馈
- sensors.imu: 姿态/运动状态
- sensors.tactile: 触觉感知
- fusion: 跨模态融合网络
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class ObjectClass(Enum):
    """物体类别"""
    UNKNOWN = 0
    FLOOR = 1
    WALL = 2
    TABLE = 3
    CHAIR = 4
    ROBOT = 5
    HUMAN = 6
    OBSTACLE = 7
    TARGET = 8
    CONTAINER = 9


@dataclass
class SceneObject:
    """场景中的物体"""
    object_id: int
    class_id: ObjectClass
    class_name: str
    bounding_box_3d: np.ndarray      # 8x3, 3D包围盒 (8个角点)
    centroid_3d: np.ndarray          # 3, 质心位置 (m)
    pose: np.ndarray                 # 4x4, 变换矩阵
    velocity: Optional[np.ndarray] = None  # 3, 速度 (m/s)
    confidence: float = 1.0
    visual_features: Optional[np.ndarray] = None  # 视觉特征
    tactile_contact: bool = False    # 是否有触觉接触
    force_reading: Optional[np.ndarray] = None  # 3, 力向量


@dataclass
class SpatialRelation:
    """空间关系"""
    subject_id: int        # 关系主体
    object_id: int        # 关系客体
    relation_type: str     # "above", "on", "near", "inside", "left_of", "right_of"
    distance: float       # 距离 (m)
    confidence: float = 1.0


@dataclass
class OccupancyGrid:
    """3D占据栅格地图"""
    resolution: float      # 栅格大小 (m)
    size: Tuple[int, int, int]  # (nx, ny, nz)
    origin: np.ndarray    # 3, 原点位置 (m)
    data: np.ndarray      # nx*ny*nz, 占据概率 [0, 1]
    
    def __post_init__(self):
        self.data = self.data.reshape(self.size)
    
    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转栅格坐标"""
        g = ((point - self.origin) / self.resolution).astype(int)
        return (int(g[0]), int(g[1]), int(g[2]))
    
    def grid_to_world(self, gx: int, gy: int, gz: int) -> np.ndarray:
        """栅格坐标转世界坐标"""
        return self.origin + np.array([gx, gy, gz]) * self.resolution
    
    def set_occupied(self, point: np.ndarray, prob: float = 1.0):
        """设置占据"""
        g = self.world_to_grid(point)
        if 0 <= g[0] < self.size[0] and 0 <= g[1] < self.size[1] and 0 <= g[2] < self.size[2]:
            self.data[g[0], g[1], g[2]] = prob
    
    def is_occupied(self, point: np.ndarray, threshold: float = 0.5) -> bool:
        """检查是否占据"""
        g = self.world_to_grid(point)
        if 0 <= g[0] < self.size[0] and 0 <= g[1] < self.size[1] and 0 <= g[2] < self.size[2]:
            return self.data[g[0], g[1], g[2]] > threshold
        return False


@dataclass
class SceneGraph:
    """场景图谱"""
    objects: List[SceneObject]
    relations: List[SpatialRelation]
    timestamp: float = 0.0
    frame_id: int = 0
    
    def get_object(self, object_id: int) -> Optional[SceneObject]:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None
    
    def get_relations(self, object_id: int) -> List[SpatialRelation]:
        return [r for r in self.relations if r.subject_id == object_id or r.object_id == object_id]


@dataclass
class DynamicState:
    """动态状态"""
    is_moving: bool
    velocity: np.ndarray     # 3, 速度 (m/s)
    acceleration: np.ndarray # 3, 加速度 (m/s^2)
    angular_vel: np.ndarray  # 3, 角速度 (rad/s)
    trajectory: List[np.ndarray] = field(default_factory=list)  # 历史轨迹


@dataclass
class SceneState:
    """完整场景状态"""
    scene_graph: SceneGraph
    occupancy: OccupancyGrid
    robot_pose: np.ndarray          # 4x4, 机器人位姿
    robot_velocity: np.ndarray     # 3, 速度
    imu_data: Optional[np.ndarray] = None   # IMU原始数据
    tactile_data: Optional[Dict] = None    # 触觉数据
    force_data: Optional[np.ndarray] = None # 力数据
    dynamic_objects: List[int] = field(default_factory=list)  # 动态物体ID列表
    timestamp: float = 0.0
    frame_id: int = 0


class SceneUnderstanding:
    """
    场景理解核心引擎
    
    功能:
    - 深度图像 → 3D 场景重建
    - 物体检测与跟踪
    - 场景图谱构建
    - 动态/静态分离
    - 触觉反馈集成
    """
    
    def __init__(
        self,
        resolution: float = 0.05,
        grid_size: Tuple[int, int, int] = (100, 100, 20),
        origin: Optional[np.ndarray] = None,
        use_raycasting: bool = True,
        tracking_window: int = 30
    ):
        """
        Args:
            resolution: 占据栅格分辨率 (m)
            grid_size: 栅格地图尺寸
            origin: 地图原点世界坐标
            use_raycasting: 是否使用射线投射更新占据栅格
            tracking_window: 物体跟踪窗口大小
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.origin = origin if origin is not None else np.array([-2.5, -2.5, 0.0])
        
        # 占据栅格
        self.occupancy = OccupancyGrid(
            resolution=resolution,
            size=grid_size,
            origin=self.origin,
            data=np.zeros(grid_size[0] * grid_size[1] * grid_size[2])
        )
        
        self.use_raycasting = use_raycasting
        self.tracking_window = tracking_window
        
        # 物体跟踪
        self._next_object_id = 1
        self._tracked_objects: Dict[int, SceneObject] = {}
        self._object_trajectories: Dict[int, List[np.ndarray]] = {}
        self._last_scene_graph: Optional[SceneGraph] = None
        
        # 统计
        self._frame_count = 0
        
    def update_from_depth(
        self,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: Optional[np.ndarray] = None,
        depth_scale: float = 1000.0,
        max_depth: float = 10.0
    ) -> OccupancyGrid:
        """
        从深度图更新场景
        
        Args:
            depth_map: HxW 深度图 (米)
            intrinsics: 3x3 内参矩阵
            extrinsics: 4x4 外参 (相机到世界), 默认相机即为世界原点
            depth_scale: 深度缩放 (如果原始单位不是米)
            max_depth: 最大深度限制
            
        Returns:
            更新后的占据栅格
        """
        H, W = depth_map.shape
        
        # 生成点云
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        z = depth_map / depth_scale
        z = np.clip(z, 0, max_depth)
        
        # 反投影
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = (u_coords - cx) * z / fx
        y = (v_coords - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # Nx3
        
        # 应用外参
        if extrinsics is not None:
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
            points = (R @ points.T).T + t
        
        # 更新占据栅格
        if self.use_raycasting:
            self._update_occupancy_raycasting(points)
        else:
            self._update_occupancy_direct(points)
        
        self._frame_count += 1
        return self.occupancy
    
    def _update_occupancy_direct(self, points: np.ndarray):
        """直接占据更新 (点云所有点)"""
        for pt in points:
            if np.linalg.norm(pt) < 0.01:
                continue
            self.occupancy.set_occupied(pt, prob=0.7)
    
    def _update_occupancy_raycasting(self, points: np.ndarray):
        """射线投射占据更新"""
        # 简化: 使用光线末端点占据, 起点为空闲
        origin = np.array([0.0, 0.0, 0.0])
        
        for pt in points:
            if np.linalg.norm(pt) < 0.01:
                continue
            # 从原点到点的射线
            direction = pt / (np.linalg.norm(pt) + 1e-6)
            num_steps = int(np.linalg.norm(pt) / self.resolution)
            
            # 沿射线步进 (避开端点)
            for i in range(num_steps - 1):
                step_pt = origin + direction * self.resolution * (i + 1)
                self.occupancy.set_occupied(step_pt, prob=0.0)  # 空闲
    
    def update_from_pointcloud(self, pointcloud: np.ndarray):
        """
        从点云更新场景
        
        Args:
            pointcloud: Nx3 点云 (米)
        """
        self._update_occupancy_direct(pointcloud)
        self._frame_count += 1
    
    def detect_objects(
        self,
        pointcloud: Optional[np.ndarray] = None,
        use_euclidean_clustering: bool = True,
        cluster_tolerance: float = 0.05,
        min_cluster_size: int = 10
    ) -> List[SceneObject]:
        """
        从点云中检测物体
        
        使用欧式聚类将点云分割成物体
        
        Args:
            pointcloud: Nx3 点云
            use_euclidean_clustering: 是否使用欧式聚类
            cluster_tolerance: 聚类容差 (m)
            min_cluster_size: 最小簇大小 (点数)
            
        Returns:
            检测到的物体列表
        """
        if pointcloud is None or len(pointcloud) == 0:
            return []
        
        # 简化聚类实现 (使用栅格化方法)
        grid = self._voxelize(pointcloud, voxel_size=0.02)
        
        objects = []
        visited = set()
        
        for key in grid.keys():
            if key in visited:
                continue
            
            # 简单BFS聚类
            cluster_voxels = []
            queue = [key]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster_voxels.append(current)
                
                neighbors = self._get_voxel_neighbors(current)
                for n in neighbors:
                    if n in grid and n not in visited:
                        dist = np.linalg.norm(np.array(current) - np.array(n))
                        if dist < cluster_tolerance / 0.02:
                            queue.append(n)
            
            # 收集聚类中的所有点
            cluster_points = []
            for voxel_key in cluster_voxels:
                cluster_points.extend(grid[voxel_key])
            
            # 聚类点数太少则跳过
            if len(cluster_points) < min_cluster_size:
                continue
            
            # 聚类点数太少则跳过
            if len(cluster_points) < min_cluster_size:
                continue
            
            # 计算包围盒
            cluster_points_arr = np.array(cluster_points)
            min_pt = cluster_points_arr.min(axis=0)
            max_pt = cluster_points_arr.max(axis=0)
            
            centroid = (min_pt + max_pt) / 2
            
            # 简单物体分类 (基于高度)
            height = max_pt[2] - min_pt[2]
            if height < 0.05:
                class_id = ObjectClass.FLOOR
            elif height < 0.3:
                class_id = ObjectClass.TABLE
            elif height < 1.0:
                class_id = ObjectClass.CHAIR
            else:
                class_id = ObjectClass.UNKNOWN
            
            obj = SceneObject(
                object_id=self._next_object_id,
                class_id=class_id,
                class_name=class_id.name,
                bounding_box_3d=self._compute_bbox_corners(min_pt, max_pt),
                centroid_3d=centroid,
                pose=np.eye(4),
                confidence=0.8
            )
            objects.append(obj)
            self._next_object_id += 1
        
        return objects
    
    def _voxelize(self, points: np.ndarray, voxel_size: float) -> Dict[Tuple[int, int, int], List[np.ndarray]]:
        """体素化点云"""
        grid = {}
        for pt in points:
            key = tuple((pt / voxel_size).astype(int))
            if key not in grid:
                grid[key] = []
            grid[key].append(pt)
        return grid
    
    def _get_voxel_neighbors(self, key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """获取体素的26邻域"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((key[0]+dx, key[1]+dy, key[2]+dz))
        return neighbors
    
    def _compute_bbox_corners(self, min_pt: np.ndarray, max_pt: np.ndarray) -> np.ndarray:
        """计算包围盒8个角点"""
        corners = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    corners.append([
                        min_pt[0] if dx == 0 else max_pt[0],
                        min_pt[1] if dy == 0 else max_pt[1],
                        min_pt[2] if dz == 0 else max_pt[2]
                    ])
        return np.array(corners, dtype=np.float32)
    
    def build_scene_graph(
        self,
        objects: List[SceneObject],
        robot_position: np.ndarray
    ) -> SceneGraph:
        """
        构建场景图谱
        
        Args:
            objects: 检测到的物体列表
            robot_position: 机器人位置 (3,)
            
        Returns:
            场景图谱
        """
        relations = []
        
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:
                    continue
                
                # 计算空间关系
                rel = self._compute_relation(obj_a, obj_b)
                if rel is not None:
                    relations.append(rel)
        
        graph = SceneGraph(
            objects=objects,
            relations=relations,
            timestamp=self._frame_count / 30.0,
            frame_id=self._frame_count
        )
        
        self._last_scene_graph = graph
        self._frame_count += 1
        
        return graph
    
    def _compute_relation(self, obj_a: SceneObject, obj_b: SceneObject) -> Optional[SpatialRelation]:
        """计算两个物体间的空间关系"""
        dist = np.linalg.norm(obj_a.centroid_3d - obj_b.centroid_3d)
        
        if dist > 3.0:  # 超过3米不考虑
            return None
        
        # 基于位置的简单关系判断
        rel_types = []
        
        if obj_a.centroid_3d[2] > obj_b.centroid_3d[2] + 0.1:
            rel_types.append("above")
        if abs(obj_a.centroid_3d[2] - obj_b.centroid_3d[2]) < 0.1 and dist < 0.5:
            rel_types.append("near")
        
        if not rel_types:
            rel_types.append("near")
        
        return SpatialRelation(
            subject_id=obj_a.object_id,
            object_id=obj_b.object_id,
            relation_type=rel_types[0],
            distance=dist,
            confidence=1.0 / (1.0 + dist)
        )
    
    def track_objects(
        self,
        detected_objects: List[SceneObject],
        max_distance: float = 0.3
    ) -> List[SceneObject]:
        """
        物体跟踪 (简化版 - 基于最近邻匹配)
        
        Args:
            detected_objects: 当前帧检测到的物体
            max_distance: 最大匹配距离
            
        Returns:
            带跟踪ID的物体列表
        """
        tracked = []
        
        for det_obj in detected_objects:
            best_match_id = None
            best_distance = max_distance
            
            for obj_id, tracked_obj in self._tracked_objects.items():
                dist = np.linalg.norm(det_obj.centroid_3d - tracked_obj.centroid_3d)
                if dist < best_distance:
                    best_distance = dist
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # 更新已跟踪物体
                updated_obj = SceneObject(
                    object_id=best_match_id,
                    class_id=det_obj.class_id,
                    class_name=det_obj.class_name,
                    bounding_box_3d=det_obj.bounding_box_3d,
                    centroid_3d=det_obj.centroid_3d,
                    pose=det_obj.pose,
                    velocity=(det_obj.centroid_3d - self._tracked_objects[best_match_id].centroid_3d) / 0.033,
                    confidence=det_obj.confidence,
                    tactile_contact=det_obj.tactile_contact,
                    force_reading=det_obj.force_reading
                )
                self._tracked_objects[best_match_id] = updated_obj
                
                # 更新轨迹
                if best_match_id not in self._object_trajectories:
                    self._object_trajectories[best_match_id] = []
                self._object_trajectories[best_match_id].append(det_obj.centroid_3d.copy())
                if len(self._object_trajectories[best_match_id]) > self.tracking_window:
                    self._object_trajectories[best_match_id].pop(0)
                
                tracked.append(updated_obj)
            else:
                # 新物体
                det_obj.object_id = self._next_object_id
                self._tracked_objects[self._next_object_id] = det_obj
                self._object_trajectories[self._next_object_id] = [det_obj.centroid_3d.copy()]
                self._next_object_id += 1
                tracked.append(det_obj)
        
        # 移除长期未跟踪的物体
        stale_ids = []
        for obj_id in self._tracked_objects:
            if obj_id not in [o.object_id for o in detected_objects]:
                stale_ids.append(obj_id)
        
        for obj_id in stale_ids:
            if len(self._object_trajectories.get(obj_id, [])) > self.tracking_window * 3:
                del self._tracked_objects[obj_id]
                if obj_id in self._object_trajectories:
                    del self._object_trajectories[obj_id]
        
        return tracked
    
    def classify_dynamic_objects(
        self,
        objects: List[SceneObject],
        velocity_threshold: float = 0.05
    ) -> List[int]:
        """
        区分动态/静态物体
        
        Args:
            objects: 物体列表
            velocity_threshold: 速度阈值 (m/s)
            
        Returns:
            动态物体ID列表
        """
        dynamic_ids = []
        
        for obj in objects:
            if obj.velocity is not None:
                speed = np.linalg.norm(obj.velocity)
                if speed > velocity_threshold:
                    dynamic_ids.append(obj.object_id)
        
        return dynamic_ids
    
    def integrate_tactile_contact(
        self,
        objects: List[SceneObject],
        tactile_contact_point: np.ndarray,
        contact_force: float,
        sensor_id: str = "default"
    ) -> List[SceneObject]:
        """
        集成触觉接触信息到场景
        
        将触觉传感器的接触事件与场景物体关联
        
        Args:
            objects: 当前场景物体列表
            tactile_contact_point: 接触点世界坐标 (3,)
            contact_force: 接触力大小 (N)
            sensor_id: 传感器ID
            
        Returns:
            更新后的物体列表
        """
        for obj in objects:
            if np.linalg.norm(obj.centroid_3d - tactile_contact_point) < 0.2:
                obj.tactile_contact = True
                obj.force_reading = np.array([0, 0, contact_force])
        
        return objects
    
    def get_scene_state(
        self,
        robot_pose: np.ndarray,
        robot_velocity: np.ndarray,
        imu_data: Optional[np.ndarray] = None,
        tactile_data: Optional[Dict] = None,
        force_data: Optional[np.ndarray] = None
    ) -> SceneState:
        """
        获取完整场景状态
        
        Args:
            robot_pose: 4x4 机器人位姿
            robot_velocity: 3 机器人速度
            imu_data: IMU原始数据
            tactile_data: 触觉数据字典
            force_data: 力数据 (6,)
            
        Returns:
            完整场景状态
        """
        objects = list(self._tracked_objects.values())
        scene_graph = self.build_scene_graph(objects, robot_pose[:3, 3])
        dynamic_ids = self.classify_dynamic_objects(objects)
        
        return SceneState(
            scene_graph=scene_graph,
            occupancy=self.occupancy,
            robot_pose=robot_pose,
            robot_velocity=robot_velocity,
            imu_data=imu_data,
            tactile_data=tactile_data,
            force_data=force_data,
            dynamic_objects=dynamic_ids,
            timestamp=self._frame_count / 30.0,
            frame_id=self._frame_count
        )
    
    def reset(self):
        """重置场景"""
        self.occupancy.data.fill(0)
        self._tracked_objects.clear()
        self._object_trajectories.clear()
        self._frame_count = 0
        self._last_scene_graph = None


# AGV五级场景理解规格
AGV_SCENE_UNDERSTANDING_GRADES = {
    'S':  {'resolution': 0.10, 'range': 3.0,  'max_objects': 10,  'tracking': False, 'semantic': False},
    'M':  {'resolution': 0.05, 'range': 5.0,  'max_objects': 30,  'tracking': True,  'semantic': False},
    'L':  {'resolution': 0.02, 'range': 8.0,  'max_objects': 50,  'tracking': True,  'semantic': True},
    'XL': {'resolution': 0.01, 'range': 10.0, 'max_objects': 100, 'tracking': True,  'semantic': True},
    'XXL': {'resolution': 0.005, 'range': 15.0, 'max_objects': 200, 'tracking': True, 'semantic': True},
}


def get_scene_spec(grade: str) -> dict:
    """获取AGV指定等级的场景理解规格"""
    return AGV_SCENE_UNDERSTANDING_GRADES.get(grade, AGV_SCENE_UNDERSTANDING_GRADES['M'])
