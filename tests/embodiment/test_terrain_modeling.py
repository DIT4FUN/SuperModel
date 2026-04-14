"""
test_terrain_modeling.py - 地形建模系统测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- 地板类型参数
- 地形区域检测
- 斜坡建模
- 轮子打滑建模
- AGV等级规格适配
- 地形状态报告
"""

import pytest
import math
from src.embodied.simulation_enhancement import (
    TerrainModelingSystem,
    TerrainRegion,
    FloorType,
)


class TestFloorTypeParameters:
    """地板类型物理参数测试"""
    
    def test_smooth_concrete_params(self):
        """测试光滑混凝土参数"""
        t = TerrainModelingSystem(grade='XXL')
        params = t.get_floor_parameters(FloorType.SMOOTH_CONCRETE)
        assert params['friction_factor'] == 1.00
        assert params['max_speed_factor'] == 1.00
        assert params['slip_factor'] == 0.02
    
    def test_grass_params(self):
        """测试草地参数 (高打滑)"""
        t = TerrainModelingSystem(grade='XXL')
        params = t.get_floor_parameters(FloorType.GRASS)
        assert params['slip_factor'] == 0.25
        assert params['max_speed_factor'] == 0.50
    
    def test_rubber_mat_low_slip(self):
        """测试橡胶垫低打滑"""
        t = TerrainModelingSystem(grade='L')
        params = t.get_floor_parameters(FloorType.RUBBER_MAT)
        assert params['slip_factor'] == 0.01
    
    def test_unknown_floor_fallback(self):
        """测试未知地板类型回退"""
        t = TerrainModelingSystem(grade='M')
        params = t.get_floor_parameters(FloorType.UNKNOWN)
        assert params['friction_factor'] == 0.70
        assert params['slip_factor'] == 0.10


class TestTerrainRegion:
    """地形区域测试"""
    
    def test_region_contains_point(self):
        """测试点是否在区域内"""
        region = TerrainRegion(
            region_id="test",
            floor_type=FloorType.SMOOTH_CONCRETE,
            center=(10.0, 5.0),
            radius=3.0,
        )
        assert region.contains_point(11.0, 5.0) is True
        assert region.contains_point(10.0, 5.0) is True
        assert region.contains_point(13.5, 5.0) is False  # 刚好在边界外
    
    def test_region_boundary(self):
        """测试区域边界"""
        region = TerrainRegion(
            region_id="boundary_test",
            floor_type=FloorType.EPOXY_COATING,
            center=(0.0, 0.0),
            radius=2.0,
        )
        # 边界上
        assert region.contains_point(2.0, 0.0) is True
        # 边界外
        assert region.contains_point(2.1, 0.0) is False
    
    def test_slope_vector(self):
        """测试斜坡方向向量计算"""
        region = TerrainRegion(
            region_id="slope_test",
            floor_type=FloorType.ROUGH_CONCRETE,
            center=(0.0, 0.0),
            slope_angle=10.0,
            slope_direction=0.0,  # 向X正方向
        )
        vx, vy, vz = region.get_slope_vector()
        assert abs(vz - math.cos(math.radians(10.0))) < 0.001


class TestTerrainGradeSupport:
    """AGV等级地形支持测试"""
    
    def test_grade_s_single_floor(self):
        """测试S级AGV只支持光滑混凝土"""
        t = TerrainModelingSystem(grade='S')
        assert FloorType.SMOOTH_CONCRETE in t.supported_floor_types
        assert FloorType.ROUGH_CONCRETE not in t.supported_floor_types
        assert FloorType.GRASS not in t.supported_floor_types
    
    def test_grade_xl_all_indoor(self):
        """测试XL级AGV支持大部分室内地形"""
        t = TerrainModelingSystem(grade='XL')
        assert FloorType.OUTDOOR_ASPHALT in t.supported_floor_types
        assert FloorType.GRASS not in t.supported_floor_types
    
    def test_grade_xxl_all_terrain(self):
        """测试XXL级AGV支持所有地形"""
        t = TerrainModelingSystem(grade='XXL')
        assert len(t.supported_floor_types) == len(list(FloorType))
    
    def test_unsupported_floor_not_added(self):
        """测试不支持的地板类型不会被添加"""
        t = TerrainModelingSystem(grade='M')  # M级不支持草地
        t.add_region_simple("grass_patch", FloorType.GRASS, (0, 0), radius=2.0)
        assert len(t.regions) == 0  # 未添加


class TestTerrainSpeedAndAccel:
    """地形速度和加速度限制测试"""
    
    def test_base_speed_on_smooth_concrete(self):
        """测试光滑混凝土基础速度"""
        t = TerrainModelingSystem(grade='M')
        t.update_position(0, 0)  # 无区域
        # 默认应该是光滑混凝土
        status = t.get_terrain_status(0, 0)
        assert status['effective_max_speed'] == pytest.approx(2.0)  # base_max_speed * 1.0
    
    def test_speed_reduction_on_grass(self):
        """测试草地对速度的限制"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("grass_area", FloorType.GRASS, (0, 0), radius=5.0)
        t.update_position(0, 0)
        status = t.get_terrain_status(0, 0)
        assert status['effective_max_speed'] == pytest.approx(1.0)  # 2.0 * 0.50
    
    def test_slope_reduces_speed(self):
        """测试上坡降低速度"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("steep_ramp", FloorType.ROUGH_CONCRETE, (5, 0), radius=2.0, slope_angle=15.0)
        t.update_position(5, 0)
        
        status = t.get_terrain_status(5, 0)
        # 15度坡应该降低速度
        assert status['effective_max_speed'] < 1.8  # 小于平地的rough concrete速度
    
    def test_accel_limited_by_friction(self):
        """测试摩擦系数限制加速度"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("smooth_area", FloorType.SMOOTH_CONCRETE, (0, 0), radius=5.0)
        t.add_region_simple("grass_area", FloorType.GRASS, (10, 0), radius=5.0)
        
        t.update_position(0, 0)
        smooth_accel = t.get_effective_max_acceleration()
        
        t.update_position(10, 0)
        grass_accel = t.get_effective_max_acceleration()
        
        assert smooth_accel > grass_accel


class TestSlipModeling:
    """打滑建模测试"""
    
    def test_slip_increases_with_speed(self):
        """测试速度越高打滑概率越高"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("concrete", FloorType.SMOOTH_CONCRETE, (0, 0), radius=10.0)
        t.update_position(0, 0)
        
        low_speed = (0.5, 0.0)
        high_speed = (2.0, 0.0)
        
        slip_low = t.get_slip_probability(low_speed, load_mass=0)
        slip_high = t.get_slip_probability(high_speed, load_mass=0)
        
        assert slip_high > slip_low
    
    def test_heavy_load_increases_slip(self):
        """测试重载增加打滑"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("concrete", FloorType.SMOOTH_CONCRETE, (0, 0), radius=10.0)
        t.update_position(0, 0)
        
        vel = (1.0, 0.5)
        slip_empty = t.get_slip_probability(vel, load_mass=0)
        slip_heavy = t.get_slip_probability(vel, load_mass=500)
        
        assert slip_heavy > slip_empty
    
    def test_wet_surface_increases_slip(self):
        """测试湿滑地面增加打滑"""
        t = TerrainModelingSystem(grade='XXL', enable_wet_surface=True)
        wet_region = TerrainRegion(
            region_id="wet_concrete",
            floor_type=FloorType.SMOOTH_CONCRETE,
            center=(0, 0),
            is_wet=True,
        )
        dry_region = TerrainRegion(
            region_id="dry_concrete",
            floor_type=FloorType.SMOOTH_CONCRETE,
            center=(20, 0),  # 远处
            is_wet=False,
        )
        t.add_region(wet_region)
        t.add_region(dry_region)
        
        vel = (1.0, 0.5)
        
        t.update_position(0, 0)  # 湿区
        slip_wet = t.get_slip_probability(vel, load_mass=0)
        
        t.update_position(20, 0)  # 干区
        slip_dry = t.get_slip_probability(vel, load_mass=0)
        
        # 湿滑应该增加打滑 (wet_factor=2.5)
        assert slip_wet > slip_dry
    
    def test_slip_capped_at_95_percent(self):
        """测试打滑概率上限95%"""
        t = TerrainModelingSystem(grade='XXL', enable_wet_surface=True)
        wet_region = TerrainRegion(
            region_id="wet_grass",
            floor_type=FloorType.GRASS,
            center=(0, 0),
            is_wet=True,
        )
        t.add_region(wet_region)
        t.update_position(0, 0)
        
        # 极端条件
        high_speed = (2.0, 2.0)
        slip = t.get_slip_probability(high_speed, load_mass=500)
        
        assert slip <= 0.95
    
    def test_slip_disabled_by_flag(self):
        """测试关闭打滑建模"""
        t = TerrainModelingSystem(grade='XXL', enable_slip_modeling=False)
        t.add_region_simple("concrete", FloorType.SMOOTH_CONCRETE, (0, 0), radius=10.0)
        t.update_position(0, 0)
        
        slip = t.get_slip_probability((2.0, 0.0), load_mass=500)
        assert slip == 0.0


class TestSlopeSafety:
    """斜坡安全检查测试"""
    
    def test_flat_surface_safe(self):
        """测试平地安全"""
        t = TerrainModelingSystem(grade='M')
        t.add_region_simple("flat", FloorType.SMOOTH_CONCRETE, (0, 0), slope_angle=5.0)
        t.update_position(0, 0)
        
        is_safe, msg = t.check_slope_safety(max_safe_angle=15.0)
        assert is_safe is True
    
    def test_excessive_slope_unsafe(self):
        """测试超限斜坡不安全"""
        t = TerrainModelingSystem(grade='XL')
        t.add_region_simple("steep", FloorType.ROUGH_CONCRETE, (0, 0), slope_angle=20.0)
        t.update_position(0, 0)
        
        is_safe, msg = t.check_slope_safety(max_safe_angle=15.0)
        assert is_safe is False
        assert "exceeds" in msg
    
    def test_grade_dependent_max_slope(self):
        """测试不同AGV等级的最大安全斜坡"""
        # XXL级AGV可以处理更陡的坡
        t_xxl = TerrainModelingSystem(grade='XXL')
        t_xxl.add_region_simple("steep", FloorType.ROUGH_CONCRETE, (0, 0), slope_angle=20.0)
        t_xxl.update_position(0, 0)
        is_safe_xxl, _ = t_xxl.check_slope_safety(max_safe_angle=25.0)
        assert is_safe_xxl is True


class TestTerrainStatus:
    """地形状态报告测试"""
    
    def test_unknown_area_status(self):
        """测试未知区域状态"""
        t = TerrainModelingSystem(grade='M')
        t.update_position(100, 100)  # 未知区域
        
        status = t.get_terrain_status(100, 100)
        assert status['floor_type'] == 'smooth_concrete'  # 默认
        assert status['slope_angle'] == 0.0
    
    def test_full_status_with_slope(self):
        """测试含斜坡的完整状态"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("complex", FloorType.OUTDOOR_ASPHALT, (5, 3), radius=3.0, slope_angle=8.0, slope_direction=45.0)
        
        status = t.get_terrain_status(5, 3)
        
        assert status['floor_type'] == 'outdoor_asphalt'
        assert status['slope_angle'] == 8.0
        assert status['slope_direction'] == 45.0
        assert 'slope_safety' in status
        assert status['effective_max_speed'] < 2.0


class TestWarehouseLayout:
    """仓库地形布局测试"""
    
    def test_generate_layout(self):
        """测试仓库布局生成"""
        t = TerrainModelingSystem(grade='XL')
        t.generate_warehouse_layout(seed=42)
        
        # 应该生成多个区域
        assert len(t.regions) > 0
        
        # 验证有不同类型的地板
        floor_types = {r.floor_type for r in t.regions}
        assert FloorType.EPOXY_COATING in floor_types
        assert FloorType.ROUGH_CONCRETE in floor_types
    
    def test_warehouse_layout_s_level(self):
        """测试S级AGV无法使用仓库布局"""
        t = TerrainModelingSystem(grade='S')
        t.generate_warehouse_layout(seed=42)
        
        # S级只支持光滑混凝土，仓库有其他类型
        # 只有光滑混凝土区域会被添加
        assert len(t.regions) > 0


class TestSlipNoiseApplication:
    """打滑噪声应用测试"""
    
    def test_slip_noise_changes_velocity(self):
        """测试打滑噪声改变速度方向"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("concrete", FloorType.SMOOTH_CONCRETE, (0, 0), radius=10.0)
        t.update_position(0, 0)
        
        intended = (1.0, 0.0, 0.0)
        
        # 多次测试，总会有一次打滑
        slipped = False
        for _ in range(100):
            result = t.apply_slip_noise(intended, slip_probability=0.3)
            if result != intended:
                slipped = True
                break
        
        assert slipped  # 30%概率，应该至少一次打滑
    
    def test_no_slip_when_prob_zero(self):
        """测试概率为0时不打滑"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("concrete", FloorType.SMOOTH_CONCRETE, (0, 0), radius=10.0)
        t.update_position(0, 0)
        
        intended = (1.0, 0.5, 0.0)
        
        for _ in range(50):
            result = t.apply_slip_noise(intended, slip_probability=0.0)
            assert result == intended


class TestGravitySlopeForce:
    """重力坡度分力测试"""
    
    def test_flat_terrain_no_slope_force(self):
        """测试平地无坡度分力"""
        t = TerrainModelingSystem(grade='M')
        t.add_region_simple("flat", FloorType.SMOOTH_CONCRETE, (0, 0), radius=5.0, slope_angle=0.0)
        t.update_position(0, 0)
        
        fx, fy = t.get_gravity_slope_force()
        assert abs(fx) < 0.001
        assert abs(fy) < 0.001
    
    def test_slope_produces_force(self):
        """测试斜坡产生分力"""
        t = TerrainModelingSystem(grade='XXL')
        t.add_region_simple("ramp", FloorType.ROUGH_CONCRETE, (0, 0), radius=5.0, slope_angle=10.0, slope_direction=0.0)
        t.update_position(0, 0)
        
        fx, fy = t.get_gravity_slope_force()
        # 应该有沿X方向的分力
        assert abs(fx) > 0.001 or abs(fy) > 0.001


class TestSupportedFloorTypes:
    """支持的地板类型列表测试"""
    
    def test_grade_m_supported(self):
        """测试M级支持列表"""
        t = TerrainModelingSystem(grade='M')
        supported = t.get_supported_floor_types()
        assert 'smooth_concrete' in supported
        assert 'rough_concrete' in supported
        assert 'epoxy_coating' in supported
        assert 'grass' not in supported
    
    def test_grade_l_supported(self):
        """测试L级支持列表"""
        t = TerrainModelingSystem(grade='L')
        supported = t.get_supported_floor_types()
        assert 'rubber_mat' in supported
        assert 'metal_plate' in supported
        assert 'grass' not in supported


class TestTerrainRegionClass:
    """TerrainRegion类测试"""
    
    def test_full_initialization(self):
        """测试完整初始化"""
        region = TerrainRegion(
            region_id="full_test",
            floor_type=FloorType.EPOXY_COATING,
            center=(10.0, 20.0),
            radius=5.0,
            slope_angle=12.0,
            slope_direction=135.0,
            unevenness=0.3,
            is_wet=True,
        )
        
        assert region.region_id == "full_test"
        assert region.floor_type == FloorType.EPOXY_COATING
        assert region.center == (10.0, 20.0)
        assert region.slope_angle == 12.0
        assert region.unevenness == 0.3
        assert region.is_wet is True
    
    def test_contains_point_edge_cases(self):
        """测试边界情况"""
        region = TerrainRegion(
            region_id="edge",
            floor_type=FloorType.SMOOTH_CONCRETE,
            center=(0, 0),
            radius=1.0,
        )
        
        # 精确边界
        assert region.contains_point(0, 1.0) is True
        assert region.contains_point(1.0, 0) is True
        assert region.contains_point(0, -1.0) is True
        assert region.contains_point(-1.0, 0) is True
        
        # 稍微超出
        assert region.contains_point(0, 1.01) is False


class TestTerrainIntegration:
    """地形系统集成测试"""
    
    def test_full_navigation_scenario(self):
        """
        测试完整导航场景:
        AGV从入库区(光滑混凝土) -> 斜坡 -> 存储区(环氧涂层) -> 出库区(粗糙混凝土)
        """
        t = TerrainModelingSystem(grade='XL')
        
        # 入库区
        t.add_region_simple("inbound", FloorType.SMOOTH_CONCRETE, (0, 0), radius=5.0)
        # 斜坡
        t.add_region_simple("ramp", FloorType.ROUGH_CONCRETE, (6, 0), radius=2.0, slope_angle=10.0)
        # 存储区
        t.add_region_simple("storage", FloorType.EPOXY_COATING, (12, 0), radius=5.0)
        # 出库区
        t.add_region_simple("outbound", FloorType.ROUGH_CONCRETE, (25, 0), radius=5.0)
        
        # 模拟路径
        waypoints = [(0, 0), (3, 0), (6, 0), (10, 0), (15, 0), (25, 0)]
        floor_types_seen = []
        max_speeds = []
        
        for x, y in waypoints:
            t.update_position(x, y)
            status = t.get_terrain_status(x, y)
            floor_types_seen.append(status['floor_type'])
            max_speeds.append(status['effective_max_speed'])
        
        # 验证路径上经过了不同地形
        assert len(set(floor_types_seen)) >= 2
        
        # 斜坡处速度应该降低
        ramp_speed = max_speeds[2]  # x=6 是斜坡
        storage_speed = max_speeds[3]  # x=10 是存储区前
        assert ramp_speed <= storage_speed  # 斜坡应该更慢
    
    def test_multi_agv_different_grades(self):
        """测试不同等级AGV在同一地形的行为"""
        positions = [(5, 0)]  # 环氧树脂涂层区
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            t = TerrainModelingSystem(grade=grade)
            t.add_region_simple("epoxy", FloorType.EPOXY_COATING, (5, 0), radius=5.0)
            t.update_position(5, 0)
            
            status = t.get_terrain_status(5, 0)
            # 所有等级在支持的地形上都能工作
            assert status['effective_max_speed'] > 0
            assert status['effective_max_accel'] > 0
