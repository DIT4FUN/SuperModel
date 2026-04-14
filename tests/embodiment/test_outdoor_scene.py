"""
test_outdoor_scene.py - 户外场景测试
"""

import pytest
import time
from src.embodied.outdoor_scene import (
    OutdoorZone,
    TerrainType,
    WeatherCondition,
    WeatherSeverity,
    OutdoorTask,
    OutdoorTaskLibrary,
    OutdoorSceneController,
    TerrainNavigator,
    WeatherMonitor,
    GPSLocalizer,
    DeliveryConfirmation,
    EmergencyHandler,
    get_outdoor_scene_controller,
)


class TestOutdoorZone:
    def test_zones_defined(self):
        zones = list(OutdoorZone)
        assert len(zones) >= 12
        assert OutdoorZone.RESIDENTIAL in zones
        assert OutdoorZone.COMMERCIAL in zones
        assert OutdoorZone.PARK in zones
        assert OutdoorZone.DELIVERY_LOCKER in zones

    def test_zone_name(self):
        assert OutdoorZone.RESIDENTIAL.name == "RESIDENTIAL"


class TestTerrainType:
    def test_terrain_types(self):
        terrains = list(TerrainType)
        assert TerrainType.FLAT_PAVEMENT in terrains
        assert TerrainType.GRASS in terrains
        assert TerrainType.SIDEWALK in terrains  # Added to fix

    def test_sidewalk_added(self):
        assert hasattr(TerrainType, 'SIDEWALK')


class TestWeatherCondition:
    def test_weather_conditions(self):
        conds = list(WeatherCondition)
        assert WeatherCondition.CLEAR in conds
        assert WeatherCondition.RAIN_HEAVY in conds
        assert WeatherCondition.SNOW_HEAVY in conds
        assert WeatherCondition.THUNDERSTORM in conds


class TestTerrainNavigator:
    def test_init(self):
        tn = TerrainNavigator()
        assert tn.get_terrain() == TerrainType.FLAT_PAVEMENT

    def test_set_terrain(self):
        tn = TerrainNavigator()
        tn.set_terrain(TerrainType.GRASS)
        assert tn.get_terrain() == TerrainType.GRASS

    def test_speed_factor_flat(self):
        tn = TerrainNavigator()
        assert tn.get_speed_factor(TerrainType.FLAT_PAVEMENT) == 1.0

    def test_speed_factor_grass(self):
        tn = TerrainNavigator()
        assert tn.get_speed_factor(TerrainType.GRASS) == 0.4

    def test_speed_factor_gravel(self):
        tn = TerrainNavigator()
        assert tn.get_speed_factor(TerrainType.GRAVEL) == 0.5

    def test_stability_factor(self):
        tn = TerrainNavigator()
        assert tn.get_stability_factor(TerrainType.FLAT_PAVEMENT) == 1.0
        assert tn.get_stability_factor(TerrainType.MUD) == 0.3

    def test_energy_factor(self):
        tn = TerrainNavigator()
        assert tn.get_energy_factor(TerrainType.FLAT_PAVEMENT) == 1.0
        assert tn.get_energy_factor(TerrainType.SAND) == 2.5

    def test_recommended_speed(self):
        tn = TerrainNavigator()
        speed = tn.get_recommended_speed(1.5, TerrainType.GRASS)
        assert speed == 1.5 * 0.4

    def test_terrain_safe(self):
        tn = TerrainNavigator()
        assert tn.is_terrain_safe(TerrainType.FLAT_PAVEMENT) is True
        assert tn.is_terrain_safe(TerrainType.MUD) is False

    def test_plan_terrain_route(self):
        tn = TerrainNavigator()
        route = tn.plan_terrain_route(OutdoorZone.RESIDENTIAL, OutdoorZone.STREET)
        assert 'terrains' in route
        assert 'safe_terrains' in route
        assert 'estimated_distance_m' in route


class TestWeatherMonitor:
    def test_init(self):
        wm = WeatherMonitor()
        assert wm.get_weather() == WeatherCondition.CLEAR
        assert wm.can_operate() is True

    def test_update_weather_clear(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.CLEAR)
        assert wm.get_severity() == WeatherSeverity.NORMAL

    def test_update_weather_rain_heavy(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.RAIN_HEAVY)
        assert wm.get_severity() == WeatherSeverity.WARNING

    def test_update_weather_thunderstorm(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.THUNDERSTORM)
        assert wm.get_severity() == WeatherSeverity.SUSPENDED
        assert wm.can_operate() is False

    def test_update_weather_fog(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.FOG)
        assert wm.get_severity() == WeatherSeverity.CAUTION

    def test_operation_recommendation(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.THUNDERSTORM)
        rec = wm.get_operation_recommendation()
        assert "暂停" in rec

    def test_slope_limit_clear(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.CLEAR)
        assert wm.get_slope_limit() == 12.0

    def test_slope_limit_rain_heavy(self):
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.RAIN_HEAVY)
        assert wm.get_slope_limit() == 5.0


class TestGPSLocalizer:
    def test_init(self):
        gps = GPSLocalizer()
        assert gps.get_accuracy() == 5.0

    def test_update_position(self):
        gps = GPSLocalizer()
        gps.update_position(31.2304, 121.4737, accuracy=3.0)
        lat, lng = gps.get_position()
        assert abs(lat - 31.2304) < 0.0001
        assert abs(lng - 121.4737) < 0.0001

    def test_has_gps_lock(self):
        gps = GPSLocalizer()
        assert gps.has_gps_lock() is True
        gps.update_position(0, 0, accuracy=50.0)
        assert gps.has_gps_lock() is False

    def test_calculate_distance(self):
        gps = GPSLocalizer()
        gps.update_position(31.2304, 121.4737)
        # ~111m per degree lat/lng roughly
        d = gps.calculate_distance(31.2314, 121.4737)
        assert 100 < d < 200  # ~111m

    def test_statistics(self):
        gps = GPSLocalizer()
        gps.update_position(31.23, 121.47, accuracy=2.0)
        stats = gps.get_statistics()
        assert 'total_updates' in stats
        assert stats['total_updates'] >= 1


class TestDeliveryConfirmation:
    def test_initiate_delivery(self):
        dc = DeliveryConfirmation()
        cid = dc.initiate_delivery("task_001", recipient_id="user_001")
        assert cid.startswith("conf_")

    def test_confirm_recipient(self):
        dc = DeliveryConfirmation()
        cid = dc.initiate_delivery("task_001", recipient_id="user_001")
        assert dc.confirm_recipient(cid, True) is True
        assert dc.get_completion_rate() == 1.0

    def test_confirm_locker_drop(self):
        dc = DeliveryConfirmation()
        cid = dc.initiate_delivery("task_001", recipient_id=None, locker_id="locker_001")
        assert dc.confirm_locker_drop(cid, True) is True

    def test_mark_failed(self):
        dc = DeliveryConfirmation()
        cid = dc.initiate_delivery("task_001", recipient_id="user_001")
        dc.mark_failed(cid, "recipient_not_home")
        assert dc.get_completion_rate() == 0.0

    def test_pending_deliveries(self):
        dc = DeliveryConfirmation()
        dc.initiate_delivery("task_001", recipient_id="user_001")
        pending = dc.get_pending()
        assert len(pending) == 1


class TestEmergencyHandler:
    def test_report_emergency(self):
        eh = EmergencyHandler()
        eid = eh.report_emergency('battery_low', 'agv_001', location=(31.23, 121.47))
        assert eid.startswith("emerg_")

    def test_resolve_emergency(self):
        eh = EmergencyHandler()
        eid = eh.report_emergency('vehicle_stuck', 'agv_001')
        assert eh.resolve_emergency(eid, 'manually_cleared') is True
        assert eh.has_active_emergency() is False

    def test_recovery_action(self):
        eh = EmergencyHandler()
        action = eh.get_recovery_action('battery_low')
        assert '充电' in action or '停车' in action

    def test_multiple_emergencies(self):
        eh = EmergencyHandler()
        eh.report_emergency('gps_lost', 'agv_001')
        eh.report_emergency('weather_extreme', 'agv_002')
        assert len(eh.get_active_emergencies()) == 2


class TestOutdoorSceneController:
    def test_init(self):
        ctrl = OutdoorSceneController(agv_grade="L")
        assert ctrl.agv_grade == "L"
        assert ctrl.terrain_navigator is not None
        assert ctrl.weather_monitor is not None
        assert ctrl.gps_localizer is not None

    def test_create_delivery_task(self):
        ctrl = OutdoorSceneController()
        task = ctrl.create_delivery_task(
            pickup_zone=OutdoorZone.RESIDENTIAL,
            destination_zone=OutdoorZone.DELIVERY_LOCKER,
            package_type="medium",
            recipient_id="user_001",
        )
        assert task.task_type == 'last_mile_delivery'
        assert task.package_type == 'medium'
        assert task.estimated_distance > 0

    def test_create_patrol_task(self):
        ctrl = OutdoorSceneController()
        task = ctrl.create_patrol_task(
            zones=[OutdoorZone.CAMPUS, OutdoorZone.PARK],
            patrol_type="routine",
        )
        assert task.task_type == 'patrol'

    def test_get_next_task(self):
        ctrl = OutdoorSceneController()
        ctrl.create_delivery_task(OutdoorZone.RESIDENTIAL, OutdoorZone.DELIVERY_LOCKER)
        next_t = ctrl.get_next_task()
        assert next_t is not None

    def test_weather_blocks_task(self):
        ctrl = OutdoorSceneController()
        ctrl.weather_monitor.update_weather(WeatherCondition.THUNDERSTORM)
        ctrl.create_delivery_task(OutdoorZone.RESIDENTIAL, OutdoorZone.DELIVERY_LOCKER)
        next_t = ctrl.get_next_task()
        assert next_t is None  # Blocked by weather

    def test_complete_task(self):
        ctrl = OutdoorSceneController()
        task = ctrl.create_delivery_task(OutdoorZone.RESIDENTIAL, OutdoorZone.DELIVERY_LOCKER)
        assert ctrl.complete_task(task.task_id) is True

    def test_register_agv(self):
        ctrl = OutdoorSceneController()
        ctrl.register_agv("agv_001", 31.2304, 121.4737)
        pos = ctrl.get_agv_position("agv_001")
        assert pos is not None
        assert abs(pos[0] - 31.2304) < 0.001

    def test_scene_status(self):
        ctrl = OutdoorSceneController()
        status = ctrl.get_scene_status()
        assert 'weather' in status
        assert 'can_operate' in status
        assert status['agv_grade'] == 'L'

    def test_scene_report(self):
        ctrl = OutdoorSceneController()
        report = ctrl.generate_scene_report()
        assert 'total_deliveries' in report
        assert 'gps_statistics' in report

    def test_recommended_speed(self):
        ctrl = OutdoorSceneController(agv_grade="L")
        ctrl.terrain_navigator.set_terrain(TerrainType.GRASS)
        speed = ctrl.get_recommended_speed()
        assert speed < 1.8  # Should be reduced by grass factor

    def test_zone_distance_estimation(self):
        ctrl = OutdoorSceneController()
        d = ctrl._estimate_zone_distance(OutdoorZone.RESIDENTIAL, OutdoorZone.DELIVERY_LOCKER)
        assert d == 50.0

    def test_default_grade_l(self):
        ctrl = OutdoorSceneController()
        assert ctrl.agv_grade == "L"


class TestOutdoorTaskLibrary:
    def test_get_delivery_template(self):
        tmpl = OutdoorTaskLibrary.get_delivery_template('last_mile_residential')
        assert tmpl['priority'] == 4
        assert 'TerrainType.FLAT_PAVEMENT' in str(tmpl.get('terrain', []))

    def test_get_patrol_template(self):
        tmpl = OutdoorTaskLibrary.get_patrol_template('campus_patrol')
        assert tmpl['type'] == 'patrol'


class TestOutdoorSceneWeatherOperation:
    def test_severity_escalation(self):
        """Test weather severity escalation from normal to suspended"""
        wm = WeatherMonitor()
        wm.update_weather(WeatherCondition.CLEAR)
        assert wm.get_severity() == WeatherSeverity.NORMAL
        wm.update_weather(WeatherCondition.RAIN_LIGHT)
        assert wm.get_severity() == WeatherSeverity.CAUTION
        wm.update_weather(WeatherCondition.RAIN_HEAVY)
        assert wm.get_severity() == WeatherSeverity.WARNING
        wm.update_weather(WeatherCondition.THUNDERSTORM)
        assert wm.get_severity() == WeatherSeverity.SUSPENDED


class TestOutdoorSceneGrades:
    def test_grade_xxl_speed(self):
        ctrl_xxl = OutdoorSceneController(agv_grade="XXL")
        ctrl_s = OutdoorSceneController(agv_grade="S")
        speed_xxl = ctrl_xxl.get_recommended_speed()
        speed_s = ctrl_s.get_recommended_speed()
        assert speed_xxl > speed_s


class TestOutdoorSceneControllerTaskCompletion:
    def test_complete_nonexistent_task(self):
        ctrl = OutdoorSceneController()
        assert ctrl.complete_task("nonexistent_task_id") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
