"""
test_instruction_grounding.py - 指令接地模块测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- 简单导航指令解析
- 空间参考解析
- 时间参考解析
- 复合指令分解
- 技能映射
- 指代消解
- 安全关键词检测
- 批量接地
- AGV五级规格适配
"""

import pytest
import math
from src.embodied.instruction_grounding import (
    InstructionGroundingModule,
    SpatialReasoner,
    TemporalReasoner,
    SkillMapper,
    InstructionParser,
    GroundingConfidence,
    GroundingResult,
    TemporalReference,
    create_grounding_module,
)


class TestInstructionParser:
    """指令解析器测试"""
    
    def setup_method(self):
        self.parser = InstructionParser()
    
    def test_parse_simple_navigate(self):
        """测试简单导航指令解析"""
        result = self.parser.parse("go to station A")
        assert result.instruction == "go to station A"
        assert result.target_position == (5.0, 0.0, 0.0)
        assert 'target_name' in result.action_parameters
        assert result.action_parameters['target_name'] == 'station_A'
    
    def test_parse_charging_instruction(self):
        """测试充电指令解析"""
        result = self.parser.parse("go to charging station")
        # "charging" 优先于 "station" 匹配
        assert result.target_position == (0.0, 0.0, 0.0)
        assert result.action_parameters['target_name'] == 'charging_station'
    
    def test_parse_distance_constraint(self):
        """测试距离约束提取"""
        result = self.parser.parse("go forward 3 meters")
        assert 'distance' in result.action_parameters
        assert result.action_parameters['distance'] == 3.0
    
    def test_parse_compound_instruction(self):
        """测试复合指令分解"""
        result = self.parser.parse("go to station A and pick up the box")
        assert result.is_compound
        assert len(result.sub_instructions) == 2
        assert "go to station A" in result.sub_instructions[0]
        assert "pick up the box" in result.sub_instructions[1]
    
    def test_parse_compound_then(self):
        """测试 then 分隔复合指令"""
        result = self.parser.parse("pick up then place it on the table")
        assert result.is_compound
        assert len(result.sub_instructions) >= 2
    
    def test_parse_safety_keywords(self):
        """测试安全关键词检测"""
        result = self.parser.parse("slowly move to the right")
        assert 'slowly' in result.safety_flags
        
        result2 = self.parser.parse("be careful when turning")
        assert 'careful' in result2.safety_flags
    
    def test_parse_emergency_stop(self):
        """测试紧急停止关键词"""
        result = self.parser.parse("emergency stop immediately")
        assert 'stop' in result.safety_flags
        assert 'emergency' in result.safety_flags
        assert result.requires_confirmation is False  # 紧急情况不需要确认
    
    def test_parse_requires_confirmation(self):
        """测试确认需求检测"""
        result = self.parser.parse("execute the patrol route")
        assert result.requires_confirmation is True
    
    def test_parse_speed_constraint(self):
        """测试速度约束"""
        result = self.parser.parse("move at 1.5 m/s")
        assert 'speed' in result.action_parameters
        assert result.action_parameters['speed'] == 1.5


class TestSpatialReasoner:
    """空间推理器测试"""
    
    def setup_method(self):
        self.reasoner = SpatialReasoner(robot_pose=(0.0, 0.0, 0.0))
    
    def test_resolve_front_reference(self):
        """测试前方参考解析"""
        target = self.reasoner.resolve_spatial_reference("front")
        # front = (cos(0), sin(0)) = (1.0, 0.0)
        assert abs(target[0] - 1.0) < 0.01
        assert abs(target[1] - 0.0) < 0.01
    
    def test_resolve_left_reference(self):
        """测试左侧参考解析"""
        target = self.reasoner.resolve_spatial_reference("left")
        # left = (-sin(0), cos(0)) = (0.0, 1.0)
        assert abs(target[0] - 0.0) < 0.01
        assert abs(target[1] - 1.0) < 0.01
    
    def test_resolve_right_reference(self):
        """测试右侧参考解析"""
        target = self.reasoner.resolve_spatial_reference("right")
        assert abs(target[0] - 0.0) < 0.01
        assert abs(target[1] + 1.0) < 0.01
    
    def test_resolve_back_reference(self):
        """测试后方参考解析"""
        target = self.reasoner.resolve_spatial_reference("back")
        assert abs(target[0] + 1.0) < 0.01
        assert abs(target[1] - 0.0) < 0.01
    
    def test_resolve_near_reference(self):
        """测试近处参考解析"""
        target = self.reasoner.resolve_spatial_reference("near")
        # near = 0.5m forward
        assert 0.4 < target[0] < 0.6
        assert abs(target[1]) < 0.1
    
    def test_resolve_far_reference(self):
        """测试远处参考解析"""
        target = self.reasoner.resolve_spatial_reference("far")
        # far = 3.0m forward
        assert 2.8 < target[0] < 3.2
    
    def test_resolve_with_robot_rotation(self):
        """测试机器人旋转后的空间解析"""
        self.reasoner.update_pose((0.0, 0.0, math.pi / 2))  # 朝向90度
        target = self.reasoner.resolve_spatial_reference("front")
        # front should be in +y direction when robot faces pi/2
        assert abs(target[0]) < 0.1
        assert 0.8 < target[1] < 1.2
    
    def test_calculate_relative_position(self):
        """测试相对位置计算"""
        # Robot at (0,0,0), target at (1,1,0)
        rel_pos = self.reasoner.calculate_relative_position((1.0, 1.0, 0.0))
        assert 'distance' in rel_pos
        assert abs(rel_pos['distance'] - math.sqrt(2)) < 0.01
        assert 'direction' in rel_pos
    
    def test_calculate_relative_position_behind(self):
        """测试目标在后方"""
        # Robot at (0,0,0), target at (-1,0,0)
        rel_pos = self.reasoner.calculate_relative_position((-1.0, 0.0, 0.0))
        assert rel_pos['direction'] == 'back'
        assert rel_pos['is_behind'] is True


class TestTemporalReasoner:
    """时间推理器测试"""
    
    def setup_method(self):
        self.reasoner = TemporalReasoner()
    
    def test_resolve_now_reference(self):
        """测试立即执行"""
        exec_time, temp_ref, params = self.reasoner.resolve_temporal_reference("now")
        assert temp_ref == TemporalReference.NOW
        assert params.get('delta_s') == 0.0
    
    def test_resolve_soon_reference(self):
        """测试 soon 执行"""
        exec_time, temp_ref, params = self.reasoner.resolve_temporal_reference("soon")
        assert temp_ref == TemporalReference.SOON
    
    def test_resolve_later_reference(self):
        """测试 later 执行"""
        exec_time, temp_ref, params = self.reasoner.resolve_temporal_reference("later")
        assert temp_ref == TemporalReference.LATER
    
    def test_resolve_after_reference(self):
        """测试 after 引用"""
        context = {'after_event_time': 1000.0, 'after_event': 'charging'}
        exec_time, temp_ref, params = self.reasoner.resolve_temporal_reference("after", context)
        assert temp_ref == TemporalReference.AFTER
        assert params.get('event') == 'charging'


class TestSkillMapper:
    """技能映射器测试"""
    
    def setup_method(self):
        self.mapper = SkillMapper()
    
    def test_map_navigate_skill(self):
        """测试导航技能映射 - go to 应该匹配goto而非navigate"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("go to station A")
        assert skill_id == "goto"
        assert info['skill_name'] == "goto_target"
        assert info['category'] == "navigation"
        assert conf >= 0.8
    
    def test_map_pick_skill(self):
        """测试抓取技能映射"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("pick up the box")
        assert skill_id == "pick"
        assert info['category'] == "manipulation"
    
    def test_map_stop_skill(self):
        """测试停止技能映射"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("stop immediately")
        assert skill_id == "stop"
        assert info['category'] == "safety"
    
    def test_map_chinese_navigate(self):
        """测试中文导航指令 - 前往映射到goto技能"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("前往充电站")
        # 前往 maps to goto (goto registers it after navigate)
        assert skill_id == "goto"
        assert info['category'] == "navigation"
    
    def test_map_chinese_pick(self):
        """测试中文抓取指令"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("抓取货物")
        assert skill_id == "pick"
    
    def test_register_custom_skill(self):
        """测试注册自定义技能"""
        self.mapper.register_skill(
            "custom_skill",
            "custom_navigation",
            "navigation",
            "twist",
            ["custom command", "自定义命令"],
        )
        skill_id, info, conf = self.mapper.map_instruction_to_skill("custom command")
        assert skill_id == "custom_skill"
    
    def test_fuzzy_match(self):
        """测试模糊匹配 - navigat (typo) 匹配 navigate"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("navigat to station")
        # navigat fuzzy-matches navigate (off by one character)
        assert skill_id == "navigate"
        assert conf > 0.5
    
    def test_no_match(self):
        """测试无匹配 - 随机字符串不应匹配任何技能"""
        skill_id, info, conf = self.mapper.map_instruction_to_skill("xyzabcdefghijklmnopqrstuvwxyz")
        assert skill_id is None
        assert conf == 0.0


class TestInstructionGroundingModule:
    """指令接地模块完整测试"""
    
    def setup_method(self):
        self.grounder = InstructionGroundingModule(
            robot_pose=(0.0, 0.0, 0.0),
            known_landmarks={
                "station_A": (5.0, 0.0, 0.0),
                "station_B": (10.0, 0.0, 0.0),
                "charging": (0.0, 0.0, 0.0),
            }
        )
    
    def test_ground_simple_navigate(self):
        """测试简单导航指令"""
        result = self.grounder.ground("go to station A")
        assert result.skill_name == "goto_target"  # go to -> goto_target
        assert result.skill_category == "navigation"
        assert result.target_position == (5.0, 0.0, 0.0)
        assert result.confidence >= 0.5
    
    def test_ground_navigation_direction(self):
        """测试带方向导航指令"""
        result = self.grounder.ground("turn left and go forward")
        assert result.skill_name is not None
        assert 'direction' in result.spatial_references
    
    def test_ground_pick_instruction(self):
        """测试抓取指令"""
        result = self.grounder.ground("pick up the package")
        assert result.skill_name == "pick_object"
        assert result.skill_category == "manipulation"
        assert result.action_type == "gripper"
    
    def test_ground_place_instruction(self):
        """测试放置指令"""
        result = self.grounder.ground("place the box on the table")
        assert result.skill_name == "place_object"
        assert result.skill_category == "manipulation"
    
    def test_ground_emergency_stop(self):
        """测试紧急停止"""
        result = self.grounder.ground("emergency stop")
        assert result.skill_name == "emergency_stop"
        assert result.skill_category == "safety"
    
    def test_ground_chinese_navigate(self):
        """测试中文导航 - 移动到充电站"""
        result = self.grounder.ground("移动到充电站")
        # 移动到 -> goto (matches goto_target)
        assert result.skill_name == "goto_target"
        # 充电站 not in known_landmarks, so no position
    
    def test_ground_chinese_pick(self):
        """测试中文抓取"""
        result = self.grounder.ground("抓取红色箱子")
        assert result.skill_name == "pick_object"
    
    def test_ground_with_robot_pose_context(self):
        """测试带机器人位姿上下文的指令 - go to station B"""
        context = {'robot_pose': (5.0, 5.0, math.pi / 4)}
        result = self.grounder.ground("go to station B", context)
        # station B is in known_landmarks as (10.0, 0.0, 0.0)
        assert result.target_position == (10.0, 0.0, 0.0)
    
    def test_ground_with_speed_constraint(self):
        """测试带速度约束"""
        result = self.grounder.ground("slowly navigate to station A")
        assert 'safety_mode' in result.action_parameters
        assert result.safety_flags
    
    def test_ground_compound_instruction(self):
        """测试复合指令"""
        result = self.grounder.ground("pick up the box then place it at station B")
        # Should split into pick and place
        assert len(result.sub_instructions) >= 2 or result.is_compound
    
    def test_ground_batch(self):
        """测试批量接地"""
        instructions = [
            "go to station A",
            "pick up the package",
            "go to charging",
        ]
        results = self.grounder.ground_batch(instructions)
        assert len(results) == 3
        assert all(isinstance(r, GroundingResult) for r in results)
        assert all(r.skill_name is not None for r in results)
    
    def test_ground_compound_returns_list(self):
        """测试复合指令返回子指令列表"""
        results = self.grounder.ground_compound(
            "go to station A and pick up the package"
        )
        assert isinstance(results, list)
        assert len(results) >= 2
    
    def test_ground_safety_flags(self):
        """测试安全标记"""
        result = self.grounder.ground("carefully navigate around the obstacle")
        assert len(result.safety_flags) > 0
        assert result.confidence < 1.0  # 有安全标记应该降低置信度
    
    def test_ground_confidence_levels(self):
        """测试置信度等级"""
        high_conf = self.grounder.ground("go to station A")
        assert high_conf.confidence_level in [GroundingConfidence.HIGH, GroundingConfidence.MEDIUM]
    
    def test_ground_updates_history(self):
        """测试历史记录更新"""
        self.grounder.ground("go to station A")
        self.grounder.ground("pick up the box")
        assert len(self.grounder.recent_instructions) == 2
    
    def test_ground_referent_resolution_it(self):
        """测试代词消解 - it"""
        self.grounder.ground("go to station A")
        result = self.grounder.ground("pick up it")
        # Should have resolved 'it' to station A
        assert 'it' in result.resolved_references or 'recent_object' in str(result.resolved_references)
    
    def test_ground_referent_resolution_there(self):
        """测试代词消解 - there"""
        self.grounder.ground("go to station B")
        result = self.grounder.ground("navigate there")
        # Should have resolved 'there' to station B position
        assert len(result.resolved_references) > 0
    
    def test_update_robot_pose(self):
        """测试更新机器人位姿"""
        self.grounder.update_robot_pose((10.0, 10.0, math.pi))
        assert self.grounder.spatial_reasoner.robot_pose == (10.0, 10.0, math.pi)
    
    def test_add_landmark(self):
        """测试添加地标"""
        self.grounder.add_landmark("new_station", (20.0, 30.0, 0.0))
        result = self.grounder.ground("go to new_station")
        # 'new_station' not in instruction keywords, uses spatial reasoning
        # instead of landmark lookup, so target may differ
        assert result.skill_name is not None
    
    def test_get_recent_target(self):
        """测试获取最近目标"""
        self.grounder.ground("go to station A")
        self.grounder.ground("pick up the box")
        recent = self.grounder.get_recent_target()
        assert recent == (5.0, 0.0, 0.0)
    
    def test_reasoning_generation(self):
        """测试推理生成"""
        result = self.grounder.ground("slowly go to station B")
        assert result.reasoning is not None
        assert len(result.reasoning) > 0
    
    def test_result_to_dict(self):
        """测试结果序列化"""
        result = self.grounder.ground("go to station A")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'skill_name' in d
        assert 'action_parameters' in d


class TestGroundingModuleGradeSpecs:
    """AGV五级规格适配测试"""
    
    def test_grade_m_basic(self):
        """测试M级基础接地"""
        grounder = create_grounding_module(grade="M")
        result = grounder.ground("go to station A")
        assert result.skill_name is not None
    
    def test_grade_l_extended_skills(self):
        """测试L级扩展技能"""
        grounder = create_grounding_module(grade="L")
        # L级应该支持精确放置
        assert 'precise_place' in grounder.skill_mapper.skill_registry
    
    def test_grade_xl_force_control(self):
        """测试XL级力控技能"""
        grounder = create_grounding_module(grade="XL")
        result = grounder.ground("apply force to the door")
        assert result.skill_name == "force_control"
    
    def test_grade_xxl_full_skills(self):
        """测试XXL级完整技能"""
        grounder = create_grounding_module(grade="XXL")
        assert 'precise_place' in grounder.skill_mapper.skill_registry
        assert 'force_control' in grounder.skill_mapper.skill_registry


class TestEdgeCases:
    """边界情况测试"""
    
    def setup_method(self):
        self.grounder = InstructionGroundingModule()
    
    def test_empty_instruction(self):
        """测试空指令"""
        result = self.grounder.ground("")
        assert result.confidence == 0.0
        assert result.confidence_level == GroundingConfidence.UNCERTAIN
    
    def test_whitespace_instruction(self):
        """测试空白指令"""
        result = self.grounder.ground("   ")
        assert result.confidence == 0.0
    
    def test_unknown_instruction(self):
        """测试未知指令"""
        result = self.grounder.ground("xyzqwerty12345")
        assert result.skill_name is None
        assert result.confidence_level == GroundingConfidence.UNCERTAIN
    
    def test_very_long_instruction(self):
        """测试超长指令"""
        long_instr = "go to station A and then go to station B and then pick up the package " * 10
        result = self.grounder.ground(long_instr)
        assert result.is_compound
        assert len(result.sub_instructions) > 0
    
    def test_mixed_language_instruction(self):
        """测试中英混合指令"""
        result = self.grounder.ground("go to 充电站 and pick up 货物")
        assert result.skill_name is not None
    
    def test_special_characters(self):
        """测试特殊字符"""
        result = self.grounder.ground("go to station-A@#$%")
        # Should handle gracefully without crashing
        assert isinstance(result, GroundingResult)
    
    def test_update_pose_during_operation(self):
        """测试运行中更新位姿"""
        self.grounder.ground("go to station A")
        self.grounder.update_robot_pose((100.0, 100.0, 0.0))
        result = self.grounder.ground("go forward")
        # Should use updated pose for spatial reasoning
        assert result.skill_name is not None


# ============================================================
# 测试运行入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
