"""
场景理解模块测试
================

测试 SceneUnderstanding 及相关组件:
- OccupancyGrid
- SceneObject / SceneGraph
- 深度图 → 占据栅格
- 物体检测与跟踪
- 动态/静态分离
- 触觉反馈集成
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from perception.scene_understanding import (
    SceneUnderstanding, SceneObject, SceneGraph, SceneState,
    SpatialRelation, OccupancyGrid, ObjectClass,
    get_scene_spec, AGV_SCENE_UNDERSTANDING_GRADES
)


class TestOccupancyGrid(unittest.TestCase):
    """测试占据栅格"""
    
    def test_grid_creation(self):
        grid = OccupancyGrid(
            resolution=0.05,
            size=(10, 10, 5),
            origin=np.array([-0.25, -0.25, 0.0]),
            data=np.zeros(10 * 10 * 5)
        )
        self.assertEqual(grid.data.shape, (10, 10, 5))
        self.assertAlmostEqual(grid.resolution, 0.05)
    
    def test_world_to_grid(self):
        grid = OccupancyGrid(
            resolution=0.1,
            size=(20, 20, 10),
            origin=np.array([-1.0, -1.0, 0.0]),
            data=np.zeros(20 * 20 * 10)
        )
        gx, gy, gz = grid.world_to_grid(np.array([-1.0, -1.0, 0.0]))
        self.assertEqual((gx, gy, gz), (0, 0, 0))
        
        gx, gy, gz = grid.world_to_grid(np.array([0.0, 0.0, 0.0]))
        self.assertEqual((gx, gy, gz), (10, 10, 0))
    
    def test_grid_to_world(self):
        grid = OccupancyGrid(
            resolution=0.1,
            size=(20, 20, 10),
            origin=np.array([-1.0, -1.0, 0.0]),
            data=np.zeros(20 * 20 * 10)
        )
        world = grid.grid_to_world(10, 10, 0)
        np.testing.assert_array_almost_equal(world, np.array([0.0, 0.0, 0.0]))
    
    def test_set_occupied(self):
        grid = OccupancyGrid(
            resolution=0.1,
            size=(20, 20, 10),
            origin=np.array([-1.0, -1.0, 0.0]),
            data=np.zeros(20 * 20 * 10)
        )
        grid.set_occupied(np.array([0.0, 0.0, 0.0]), prob=0.8)
        self.assertGreater(grid.data[10, 10, 0], 0.5)
    
    def test_is_occupied(self):
        grid = OccupancyGrid(
            resolution=0.1,
            size=(20, 20, 10),
            origin=np.array([-1.0, -1.0, 0.0]),
            data=np.zeros(20 * 20 * 10)
        )
        grid.set_occupied(np.array([0.0, 0.0, 0.0]), prob=0.8)
        self.assertTrue(grid.is_occupied(np.array([0.0, 0.0, 0.0])))
        self.assertFalse(grid.is_occupied(np.array([0.5, 0.5, 0.0])))
    
    def test_boundary_check(self):
        grid = OccupancyGrid(
            resolution=0.1,
            size=(5, 5, 5),
            origin=np.array([-0.25, -0.25, 0.0]),
            data=np.zeros(5 * 5 * 5)
        )
        # 超出边界的点应返回 False
        self.assertFalse(grid.is_occupied(np.array([10.0, 10.0, 0.0])))


class TestSceneObject(unittest.TestCase):
    """测试场景物体"""
    
    def test_scene_object_creation(self):
        obj = SceneObject(
            object_id=1,
            class_id=ObjectClass.TABLE,
            class_name="TABLE",
            bounding_box_3d=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
            centroid_3d=np.array([0.5, 0.5, 0.5]),
            pose=np.eye(4),
            confidence=0.95
        )
        self.assertEqual(obj.object_id, 1)
        self.assertEqual(obj.class_id, ObjectClass.TABLE)
        self.assertAlmostEqual(obj.confidence, 0.95)
    
    def test_scene_object_with_velocity(self):
        obj = SceneObject(
            object_id=2,
            class_id=ObjectClass.HUMAN,
            class_name="HUMAN",
            bounding_box_3d=np.zeros((8, 3)),
            centroid_3d=np.array([1.0, 2.0, 0.5]),
            pose=np.eye(4),
            velocity=np.array([0.1, 0.0, 0.0]),
            confidence=0.9
        )
        self.assertIsNotNone(obj.velocity)
        self.assertAlmostEqual(obj.velocity[0], 0.1)


class TestSceneGraph(unittest.TestCase):
    """测试场景图谱"""
    
    def test_scene_graph_creation(self):
        obj1 = SceneObject(1, ObjectClass.FLOOR, "FLOOR",
                          np.zeros((8,3)), np.zeros(3), np.eye(4))
        obj2 = SceneObject(2, ObjectClass.TABLE, "TABLE",
                          np.zeros((8,3)), np.array([1,1,0.5]), np.eye(4))
        
        rel = SpatialRelation(1, 2, "above", 0.5, 0.9)
        
        graph = SceneGraph(
            objects=[obj1, obj2],
            relations=[rel],
            timestamp=1.5,
            frame_id=10
        )
        
        self.assertEqual(len(graph.objects), 2)
        self.assertEqual(len(graph.relations), 1)
        self.assertAlmostEqual(graph.timestamp, 1.5)
    
    def test_get_object(self):
        obj1 = SceneObject(1, ObjectClass.FLOOR, "FLOOR",
                          np.zeros((8,3)), np.zeros(3), np.eye(4))
        obj2 = SceneObject(2, ObjectClass.TABLE, "TABLE",
                          np.zeros((8,3)), np.array([1,1,0.5]), np.eye(4))
        
        graph = SceneGraph([obj1, obj2], [])
        
        found = graph.get_object(2)
        self.assertIsNotNone(found)
        self.assertEqual(found.class_id, ObjectClass.TABLE)
        
        not_found = graph.get_object(999)
        self.assertIsNone(not_found)
    
    def test_get_relations(self):
        obj1 = SceneObject(1, ObjectClass.FLOOR, "FLOOR",
                          np.zeros((8,3)), np.zeros(3), np.eye(4))
        obj2 = SceneObject(2, ObjectClass.TABLE, "TABLE",
                          np.zeros((8,3)), np.array([1,1,0.5]), np.eye(4))
        obj3 = SceneObject(3, ObjectClass.CHAIR, "CHAIR",
                          np.zeros((8,3)), np.array([2,1,0.4]), np.eye(4))
        
        rel1 = SpatialRelation(2, 1, "above", 0.5, 0.9)   # TABLE on FLOOR
        rel2 = SpatialRelation(3, 1, "above", 0.6, 0.8)   # CHAIR on FLOOR
        
        graph = SceneGraph([obj1, obj2, obj3], [rel1, rel2])
        
        # FLOOR 的关系
        floor_rels = graph.get_relations(1)
        self.assertEqual(len(floor_rels), 2)
        
        # TABLE 的关系
        table_rels = graph.get_relations(2)
        self.assertEqual(len(table_rels), 1)


class TestSceneUnderstanding(unittest.TestCase):
    """测试场景理解引擎"""
    
    def setUp(self):
        self.scene = SceneUnderstanding(
            resolution=0.05,
            grid_size=(40, 40, 10),
            origin=np.array([-1.0, -1.0, 0.0])
        )
    
    def test_scene_initialization(self):
        self.assertEqual(self.scene._frame_count, 0)
        self.assertIsNotNone(self.scene.occupancy)
        self.assertEqual(self.scene.occupancy.resolution, 0.05)
    
    def test_update_from_depth_simple(self):
        # 创建简单深度图 (320x240) - 使用0.3m确保在栅格z范围内 (0~0.5m)
        # 深度值以米为单位 (depth_scale=1.0)
        H, W = 240, 320
        depth = np.ones((H, W)) * 0.3  # 0.3米深度
        
        # 简单内参 (FOV 60度)
        fx = fy = W / (2 * np.tan(np.radians(60) / 2))
        intrinsics = np.array([
            [fx, 0, W/2],
            [0, fy, H/2],
            [0, 0, 1]
        ])
        
        # 关闭射线投射以直接占据; depth_scale=1.0 表示深度已是米
        self.scene.use_raycasting = False
        occupancy = self.scene.update_from_depth(depth, intrinsics, depth_scale=1.0)
        self.assertIsNotNone(occupancy)
        self.assertGreater(np.sum(occupancy.data > 0), 0)
    
    def test_update_from_depth_with_objects(self):
        H, W = 120, 160
        depth = np.ones((H, W)) * 3.0
        
        # 中心区域放置一个"物体" (距离更近)
        center_y, center_x = H // 2, W // 2
        radius = 20
        y_coords, x_coords = np.ogrid[:H, :W]
        mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
        depth[mask] = 1.5  # 物体距离1.5米
        
        fx = fy = W / (2 * np.tan(np.radians(60) / 2))
        intrinsics = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]])
        
        occupancy = self.scene.update_from_depth(depth, intrinsics)
        self.assertIsNotNone(occupancy)
    
    def test_update_from_pointcloud(self):
        # 模拟桌面点云 (在栅格范围内: x/y: -1~1, z: 0~0.5)
        x = np.random.uniform(-0.2, 0.2, 100)
        y = np.random.uniform(-0.2, 0.2, 100)
        z = np.random.uniform(0.1, 0.3, 100)  # 在栅格z范围内
        points = np.stack([x, y, z], axis=1)
        
        self.scene.update_from_pointcloud(points)
        
        # 检查占据栅格中有占据点
        occupied_cells = np.sum(self.scene.occupancy.data > 0)
        self.assertGreater(occupied_cells, 0)
    
    def test_detect_objects(self):
        # 模拟场景点云 (桌面 + 物体, 在栅格范围内)
        # 桌面点 (在栅格 x/y: -1~1, z: 0~0.5)
        table_x = np.random.uniform(-0.2, 0.2, 200)
        table_y = np.random.uniform(-0.2, 0.2, 200)
        table_z = np.random.uniform(0.1, 0.3, 200)
        table_points = np.stack([table_x, table_y, table_z], axis=1)
        
        # 另一个物体
        obj_x = np.random.uniform(0.3, 0.5, 100)
        obj_y = np.random.uniform(-0.1, 0.1, 100)
        obj_z = np.random.uniform(0.1, 0.25, 100)
        obj_points = np.stack([obj_x, obj_y, obj_z], axis=1)
        
        all_points = np.vstack([table_points, obj_points])
        
        objects = self.scene.detect_objects(all_points, min_cluster_size=5)
        self.assertGreater(len(objects), 0)
    
    def test_detect_objects_empty(self):
        objects = self.scene.detect_objects(np.zeros((0, 3)))
        self.assertEqual(len(objects), 0)
    
    def test_track_objects(self):
        # 第一帧检测
        obj1 = SceneObject(0, ObjectClass.TABLE, "TABLE",
                          np.zeros((8,3)), np.array([0.5, 0.5, 0.75]), np.eye(4))
        obj2 = SceneObject(0, ObjectClass.CHAIR, "CHAIR",
                          np.zeros((8,3)), np.array([1.0, 0.5, 0.4]), np.eye(4))
        
        tracked = self.scene.track_objects([obj1, obj2])
        self.assertEqual(len(tracked), 2)
        self.assertEqual(tracked[0].object_id, 1)  # 分配了新ID
        self.assertEqual(tracked[1].object_id, 2)
        
        # 第二帧 (物体略微移动)
        obj1_new = SceneObject(0, ObjectClass.TABLE, "TABLE",
                              np.zeros((8,3)), np.array([0.52, 0.5, 0.75]), np.eye(4))
        obj2_new = SceneObject(0, ObjectClass.CHAIR, "CHAIR",
                              np.zeros((8,3)), np.array([1.02, 0.5, 0.4]), np.eye(4))
        
        tracked2 = self.scene.track_objects([obj1_new, obj2_new])
        self.assertEqual(tracked2[0].object_id, 1)  # 应该保持相同ID
        self.assertEqual(tracked2[1].object_id, 2)
    
    def test_build_scene_graph(self):
        obj1 = SceneObject(1, ObjectClass.FLOOR, "FLOOR",
                          np.zeros((8,3)), np.zeros(3), np.eye(4))
        obj2 = SceneObject(2, ObjectClass.TABLE, "TABLE",
                          np.zeros((8,3)), np.array([0, 0, 0.75]), np.eye(4))
        
        graph = self.scene.build_scene_graph([obj1, obj2], np.zeros(3))
        
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.objects), 2)
        self.assertGreaterEqual(len(graph.relations), 0)
    
    def test_classify_dynamic_objects(self):
        static_obj = SceneObject(1, ObjectClass.TABLE, "TABLE",
                                np.zeros((8,3)), np.array([0, 0, 0.75]), np.eye(4),
                                velocity=np.array([0.01, 0.0, 0.0]))  # 慢速
        moving_obj = SceneObject(2, ObjectClass.HUMAN, "HUMAN",
                                np.zeros((8,3)), np.array([1, 1, 0.5]), np.eye(4),
                                velocity=np.array([0.2, 0.0, 0.0]))  # 快速
        
        dynamic_ids = self.scene.classify_dynamic_objects(
            [static_obj, moving_obj], velocity_threshold=0.05
        )
        self.assertIn(2, dynamic_ids)
        self.assertNotIn(1, dynamic_ids)
    
    def test_integrate_tactile_contact(self):
        obj = SceneObject(1, ObjectClass.TABLE, "TABLE",
                         np.zeros((8,3)), np.array([0.5, 0.5, 0.75]), np.eye(4))
        
        updated = self.scene.integrate_tactile_contact(
            [obj],
            tactile_contact_point=np.array([0.5, 0.5, 0.75]),
            contact_force=5.0
        )
        
        self.assertTrue(updated[0].tactile_contact)
        self.assertIsNotNone(updated[0].force_reading)
    
    def test_get_scene_state(self):
        robot_pose = np.eye(4)
        robot_pose[:3, 3] = [0, 0, 0]
        robot_velocity = np.array([0.1, 0.0, 0.0])
        
        imu_data = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        force_data = np.array([0, 0, 5, 0, 0, 0])
        
        state = self.scene.get_scene_state(
            robot_pose, robot_velocity,
            imu_data=imu_data,
            force_data=force_data
        )
        
        self.assertIsInstance(state, SceneState)
        self.assertIsNotNone(state.scene_graph)
        self.assertIsNotNone(state.occupancy)
        np.testing.assert_array_equal(state.robot_velocity, robot_velocity)
        np.testing.assert_array_equal(state.imu_data, imu_data)
        np.testing.assert_array_equal(state.force_data, force_data)
    
    def test_reset(self):
        # 添加一些数据
        points = np.random.randn(100, 3)
        self.scene.update_from_pointcloud(points)
        
        self.assertGreater(self.scene._frame_count, 0)
        
        self.scene.reset()
        
        self.assertEqual(self.scene._frame_count, 0)
        self.assertEqual(len(self.scene._tracked_objects), 0)
        self.assertAlmostEqual(np.sum(self.scene.occupancy.data), 0.0)


class TestSceneUnderstandingGrades(unittest.TestCase):
    """测试AGV五级场景理解规格"""
    
    def test_scene_spec_grades(self):
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_scene_spec(grade)
            self.assertIn('resolution', spec)
            self.assertIn('range', spec)
            self.assertIn('max_objects', spec)
            self.assertIn('tracking', spec)
            self.assertIn('semantic', spec)
    
    def test_scene_spec_defaults(self):
        spec = get_scene_spec('UNKNOWN')
        self.assertEqual(spec, get_scene_spec('M'))
    
    def test_scene_spec_values(self):
        # S级: 低分辨率, 无跟踪
        s_spec = get_scene_spec('S')
        self.assertFalse(s_spec['tracking'])
        self.assertFalse(s_spec['semantic'])
        
        # XXL级: 高分辨率, 全功能
        xxl_spec = get_scene_spec('XXL')
        self.assertTrue(xxl_spec['tracking'])
        self.assertTrue(xxl_spec['semantic'])
        self.assertLess(xxl_spec['resolution'], s_spec['resolution'])
        self.assertGreater(xxl_spec['max_objects'], s_spec['max_objects'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
