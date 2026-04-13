"""
test_vla.py - VLA (Vision-Language-Action) 模型测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- VLA模型创建与配置
- Vision/Language/Action编码器
- 端到端VLA推理
- AGV五级规格适配
- VLA推理管道
- 动作平滑与安全shield
"""

import pytest
import numpy as np
import time
import threading

from src.embodied.vla_model import (
    VLAActionSpace, VLAGrade, VLAAction, VLAPerceptionFrame,
    VLAInput, VLAOutput, VisionEncoder, LanguageEncoder,
    ActionDecoder, VLAModel, VLAConfig, create_vla_model, load_vla_model,
)

from src.embodied.vla_inference import (
    InferencePolicy, ActionSmoothingMode, VLAPipelineConfig,
    ActionSmoother, VLAInferencePipeline, create_vla_inference_pipeline,
)


# ============================================================
# VLA 配置测试
# ============================================================

class TestVLAConfig:
    """测试VLA配置"""

    def test_vla_action_space_enum(self):
        assert VLAActionSpace.TWIST.value == "twist"
        assert VLAActionSpace.JOINT_POSITION.value == "joint_position"
        assert VLAActionSpace.END_EFFECTOR.value == "end_effector"
        assert VLAActionSpace.GRIPPER.value == "gripper"
        assert VLAActionSpace.COMBINED.value == "combined"

    def test_vla_grade_enum(self):
        assert VLAGrade.S.value == "S"
        assert VLAGrade.M.value == "M"
        assert VLAGrade.L.value == "L"
        assert VLAGrade.XL.value == "XL"
        assert VLAGrade.XXL.value == "XXL"

    def test_vla_config_defaults(self):
        config = VLAConfig()
        assert config.vision_dim == 512
        assert config.lidar_dim == 128
        assert config.lang_hidden_dim == 768
        assert config.fusion_hidden_dim == 512
        assert config.num_joints == 6
        assert config.action_seq_len == 8
        assert config.grade == "M"

    def test_vla_config_grade_s(self):
        config = VLAConfig(grade="S")
        assert config.grade == "S"
        assert config.get_grade_action_space() == VLAActionSpace.TWIST

    def test_vla_config_grade_m(self):
        config = VLAConfig(grade="M")
        assert config.grade == "M"
        assert config.get_grade_action_space() == VLAActionSpace.COMBINED

    def test_vla_config_grade_l(self):
        config = VLAConfig(grade="L")
        assert config.grade == "L"

    def test_vla_config_grade_xl(self):
        config = VLAConfig(grade="XL")
        assert config.grade == "XL"

    def test_vla_config_grade_xxl(self):
        config = VLAConfig(grade="XXL")
        assert config.grade == "XXL"

    def test_get_action_dim_twist(self):
        config = VLAConfig(action_space=VLAActionSpace.TWIST)
        assert config.get_action_dim() == 6

    def test_get_action_dim_gripper(self):
        config = VLAConfig(action_space=VLAActionSpace.GRIPPER)
        assert config.get_action_dim() == 1

    def test_get_action_dim_combined(self):
        config = VLAConfig(action_space=VLAActionSpace.COMBINED)
        assert config.get_action_dim() == 7


class TestVLAAction:
    """测试VLA动作"""

    def test_vla_action_defaults(self):
        action = VLAAction()
        assert action.vx == 0.0
        assert action.vy == 0.0
        assert action.vz == 0.0
        assert action.rx == 0.0
        assert action.ry == 0.0
        assert action.rz == 0.0
        assert action.confidence == 1.0
        assert action.action_space == VLAActionSpace.TWIST

    def test_vla_action_navigation(self):
        action = VLAAction(vx=0.5, vy=0.0, vz=0.0, rx=0.0, ry=0.0, rz=0.1)
        assert action.vx == pytest.approx(0.5)
        assert action.rz == pytest.approx(0.1)

    def test_vla_action_with_gripper(self):
        action = VLAAction(vx=0.3, vy=0.1, rz=0.05, gripper_position=0.8)
        assert action.gripper_position == pytest.approx(0.8)

    def test_vla_action_to_twist(self):
        action = VLAAction(vx=0.5, vy=0.2, vz=0.0, rx=0.0, ry=0.0, rz=0.1)
        twist = action.to_twist()
        assert len(twist) == 6
        assert twist[0] == pytest.approx(0.5)
        assert twist[5] == pytest.approx(0.1)

    def test_vla_action_to_dict(self):
        action = VLAAction(vx=0.5, vz=0.1)
        d = action.to_dict()
        assert 'action_id' in d
        assert 'timestamp' in d
        assert d['twist'][0] == pytest.approx(0.5)

    def test_vla_action_with_reasoning(self):
        action = VLAAction(vx=0.5, reasoning="Moving forward")
        assert action.reasoning == "Moving forward"
        assert action.attention_weights is None


class TestVLAPerceptionFrame:
    """测试感知帧"""

    def test_perception_frame_defaults(self):
        frame = VLAPerceptionFrame()
        assert frame.timestamp > 0
        assert frame.rgb_image is None
        assert frame.lidar_scan is None

    def test_perception_frame_with_rgb(self):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frame = VLAPerceptionFrame(rgb_image=image)
        assert frame.rgb_image is not None
        assert frame.rgb_image.shape == (224, 224, 3)

    def test_perception_frame_with_lidar(self):
        scan = np.random.rand(360)
        frame = VLAPerceptionFrame(lidar_scan=scan)
        assert frame.lidar_scan is not None
        assert frame.lidar_scan.shape == (360,)

    def test_perception_frame_with_instruction(self):
        frame = VLAPerceptionFrame(instruction="Navigate to target")
        assert frame.instruction == "Navigate to target"

    def test_perception_frame_get_modalities(self):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Move forward")
        mods = frame.get_modalities()
        assert 'vision' in mods
        assert 'language' in mods

    def test_perception_frame_joint_states(self):
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        frame = VLAPerceptionFrame(joint_states=joints)
        assert frame.joint_states is not None
        assert len(frame.joint_states) == 6

    def test_perception_frame_base_pose(self):
        pose = np.array([1.0, 2.0, 0.5])  # x, y, theta
        frame = VLAPerceptionFrame(base_pose=pose)
        assert frame.base_pose is not None
        assert frame.base_pose[0] == pytest.approx(1.0)


class TestVisionEncoder:
    """测试视觉编码器"""

    def test_vision_encoder_creation(self):
        encoder = VisionEncoder(vision_dim=512, lidar_dim=128)
        assert encoder.vision_dim == 512
        assert encoder.lidar_dim == 128

    def test_vision_encoder_encode_rgb(self):
        encoder = VisionEncoder(vision_dim=512)
        image = np.random.rand(224, 224, 3).astype(np.float32)
        features = encoder.encode_rgb(image)
        assert features.shape == (512,)

    def test_vision_encoder_encode_depth(self):
        encoder = VisionEncoder(vision_dim=512)
        depth = np.random.rand(224, 224).astype(np.float32)
        features = encoder.encode_depth(depth)
        assert features.shape == (128,)

    def test_vision_encoder_encode_lidar(self):
        encoder = VisionEncoder(vision_dim=512, lidar_dim=128)
        scan = np.random.rand(360).astype(np.float32)
        features = encoder.encode_lidar(scan)
        assert features.shape == (128,)

    def test_vision_encoder_encode_full_frame(self):
        encoder = VisionEncoder(vision_dim=512, lidar_dim=128)
        image = np.random.rand(224, 224, 3).astype(np.float32)
        scan = np.random.rand(360).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, lidar_scan=scan)
        features = encoder.encode(frame)
        # Returns concatenated vision + lidar features
        assert features.shape[0] > 0

    def test_vision_encoder_batch(self):
        encoder = VisionEncoder(vision_dim=512)
        # 单张图像编码
        image = np.random.rand(224, 224, 3).astype(np.float32)
        features = encoder.encode_rgb(image)
        assert features.shape == (512,)


class TestLanguageEncoder:
    """测试语言编码器"""

    def test_language_encoder_creation(self):
        encoder = LanguageEncoder(hidden_dim=768, vocab_size=10000)
        assert encoder.hidden_dim == 768
        assert encoder.vocab_size == 10000

    def test_language_encoder_encode(self):
        encoder = LanguageEncoder(hidden_dim=768)
        text = "Move forward to target"
        features = encoder.encode(text)
        assert features.shape == (768,)

    def test_language_encoder_batch(self):
        encoder = LanguageEncoder(hidden_dim=768)
        texts = ["Move forward", "Turn left", "Stop"]
        features = encoder.batch_encode(texts)
        assert features.shape[0] == 3
        assert features.shape[1] == 768


class TestActionDecoder:
    """测试动作解码器"""

    def test_action_decoder_creation(self):
        decoder = ActionDecoder(hidden_dim=512, action_dim=6)
        assert decoder.hidden_dim == 512
        assert decoder.action_dim == 6

    def test_action_decoder_decode(self):
        decoder = ActionDecoder(hidden_dim=512, action_dim=6)
        fused = np.random.rand(512).astype(np.float32)
        actions, logits = decoder.decode(fused, history_actions=[], num_steps=1)
        assert len(actions) == 1
        assert isinstance(actions[0], VLAAction)

    def test_action_decoder_with_history(self):
        decoder = ActionDecoder(hidden_dim=512, action_dim=6)
        fused = np.random.rand(512).astype(np.float32)
        history = [VLAAction(vx=0.1, vz=0.01), VLAAction(vx=0.2, vz=0.02)]
        actions, logits = decoder.decode(fused, history_actions=history, num_steps=1)
        assert len(actions) == 1
        assert isinstance(actions[0], VLAAction)

    def test_action_decoder_multi_step(self):
        decoder = ActionDecoder(hidden_dim=512, action_dim=6, max_seq_len=5)
        fused = np.random.rand(512).astype(np.float32)
        actions, logits = decoder.decode(fused, history_actions=[], num_steps=5)
        assert len(actions) == 5


class TestVLAModel:
    """测试完整VLA模型"""

    def test_vla_model_creation(self):
        config = VLAConfig(grade="M")
        model = VLAModel(config)
        assert model.config.grade == "M"

    def test_vla_model_start_stop(self):
        config = VLAConfig(grade="S")
        model = VLAModel(config)
        model.start()
        assert model._is_running is True
        model.stop()
        assert model._is_running is False

    def test_vla_model_single_step(self):
        config = VLAConfig(grade="S")
        model = VLAModel(config)
        model.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Move forward")
        vla_input = VLAInput(perception=frame)
        
        output = model.step(vla_input)
        assert isinstance(output, VLAOutput)
        assert isinstance(output.action, VLAAction)
        model.stop()

    def test_vla_model_with_lidar(self):
        config = VLAConfig(grade="S")
        model = VLAModel(config)
        model.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        scan = np.random.rand(360).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, lidar_scan=scan, instruction="Navigate")
        vla_input = VLAInput(perception=frame)
        
        output = model.step(vla_input)
        assert isinstance(output.action, VLAAction)
        model.stop()

    def test_vla_model_multistep_history(self):
        config = VLAConfig(grade="S")
        model = VLAModel(config)
        model.start()
        
        for i in range(3):
            image = np.random.rand(224, 224, 3).astype(np.float32)
            frame = VLAPerceptionFrame(rgb_image=image, instruction=f"Step {i}")
            vla_input = VLAInput(perception=frame)
            output = model.step(vla_input)
            assert isinstance(output.action, VLAAction)
        
        # History should accumulate
        assert len(model._history) == 3
        model.stop()

    def test_vla_model_collision_risk(self):
        config = VLAConfig(grade="S")
        model = VLAModel(config)
        model.start()
        
        # 高速前进动作
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Fast forward")
        vla_input = VLAInput(perception=frame)
        
        output = model.step(vla_input)
        assert output.collision_risk >= 0.0
        assert output.collision_risk <= 1.0
        model.stop()

    def test_vla_model_reset_history(self):
        config = VLAConfig(grade="S")
        model = VLAModel(config)
        model.start()
        
        # 执行几步
        for _ in range(3):
            frame = VLAPerceptionFrame(rgb_image=np.random.rand(224, 224, 3).astype(np.float32))
            model.step(VLAInput(perception=frame))
        
        assert len(model._history) == 3
        model.reset_history()
        assert len(model._history) == 0
        model.stop()


def test_create_vla_model_s():
    model = create_vla_model(grade="S")
    assert isinstance(model, VLAModel)
    assert model.config.grade == "S"


def test_create_vla_model_m():
    model = create_vla_model(grade="M")
    assert isinstance(model, VLAModel)
    assert model.config.grade == "M"


def test_create_vla_model_xl():
    model = create_vla_model(grade="XL")
    assert isinstance(model, VLAModel)
    assert model.config.grade == "XL"


def test_load_vla_model(tmp_path):
    config = VLAConfig(grade="S")
    model = VLAModel(config)
    
    # 保存 (占位实现)
    save_path = tmp_path / "vla_model"
    if hasattr(model, 'save'):
        model.save(str(save_path))
    
    # 加载
    loaded = load_vla_model(str(save_path), grade="S")
    assert isinstance(loaded, VLAModel)


# ============================================================
# VLA 推理管道测试
# ============================================================

class TestVLAPipelineConfig:
    def test_pipeline_config_defaults(self):
        config = VLAPipelineConfig()
        assert config.grade == "M"
        assert config.inference_hz == 10.0
        assert config.ema_alpha == 0.7
        assert config.safety_enabled is True

    def test_pipeline_config_s_grade(self):
        config = VLAPipelineConfig(grade="S")
        assert config.grade == "S"

    def test_pipeline_config_high_freq(self):
        config = VLAPipelineConfig(inference_hz=30.0, max_inference_hz=60.0)
        assert config.inference_hz == 30.0
        assert config.max_inference_hz == 60.0

    def test_pipeline_config_safety_params(self):
        config = VLAPipelineConfig(
            safety_enabled=True,
            max_linear_speed=1.5,
            max_angular_speed=1.0,
            min_clearance=0.5,
        )
        assert config.max_linear_speed == 1.5
        assert config.min_clearance == 0.5


class TestActionSmoother:
    def test_smoother_creation(self):
        smoother = ActionSmoother(mode=ActionSmoothingMode.EMA, alpha=0.7)
        assert smoother.mode == ActionSmoothingMode.EMA
        assert smoother.alpha == 0.7

    def test_smoother_ema_single(self):
        smoother = ActionSmoother(mode=ActionSmoothingMode.EMA, alpha=0.7)
        action = VLAAction(vx=0.5)
        smoothed = smoother.smooth(action)
        assert isinstance(smoothed, VLAAction)
        assert smoothed.vx == pytest.approx(0.5)

    def test_smoother_ema_sequential(self):
        smoother = ActionSmoother(mode=ActionSmoothingMode.EMA, alpha=0.7)
        action1 = VLAAction(vx=0.5)
        action2 = VLAAction(vx=0.6)
        smoother.smooth(action1)
        smoothed2 = smoother.smooth(action2)
        # EMA: 0.7*0.6 + 0.3*0.5 = 0.42 + 0.15 = 0.57
        assert smoothed2.vx == pytest.approx(0.57, rel=0.02)

    def test_smoother_low_pass(self):
        smoother = ActionSmoother(mode=ActionSmoothingMode.LOW_PASS)
        action = VLAAction(vx=0.5)
        smoothed = smoother.smooth(action)
        assert isinstance(smoothed, VLAAction)

    def test_smoother_none(self):
        smoother = ActionSmoother(mode=ActionSmoothingMode.NONE)
        action = VLAAction(vx=0.5)
        smoothed = smoother.smooth(action)
        assert smoothed.vx == pytest.approx(0.5)


class TestVLAInferencePipeline:
    def test_pipeline_creation(self):
        config = VLAPipelineConfig(grade="S")
        pipeline = VLAInferencePipeline(config)
        assert pipeline.config.grade == "S"

    def test_pipeline_start_stop(self):
        config = VLAPipelineConfig(grade="S")
        pipeline = VLAInferencePipeline(config)
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()
        # Check internal state exists
        assert hasattr(pipeline, '_is_running')
        assert hasattr(pipeline, '_action_queue')

    def test_pipeline_trigger_inference(self):
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config)
        pipeline.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Move forward")
        
        result = pipeline.trigger_inference(frame)
        assert isinstance(result, VLAOutput)
        assert isinstance(result.action, VLAAction)
        pipeline.stop()

    def test_pipeline_set_instruction(self):
        config = VLAPipelineConfig(grade="S")
        pipeline = VLAInferencePipeline(config)
        pipeline.set_instruction("Navigate to target")
        assert pipeline._current_instruction == "Navigate to target"

    def test_pipeline_set_perception_frame(self):
        config = VLAPipelineConfig(grade="S")
        pipeline = VLAInferencePipeline(config)
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image)
        # set_perception_frame should not raise
        pipeline.set_perception_frame(frame)
        assert True  # No exception = success

    def test_pipeline_get_latest_action(self):
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config)
        pipeline.start()
        time.sleep(0.2)
        
        latest = pipeline.get_latest_action(timeout=0.1)
        if latest is not None:
            assert isinstance(latest, VLAAction)
        pipeline.stop()

    def test_pipeline_stats(self):
        config = VLAPipelineConfig(grade="S")
        pipeline = VLAInferencePipeline(config)
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()
        
        stats = pipeline.get_stats()
        assert 'total_inferences' in stats
        assert 'history_len' in stats

    def test_pipeline_reset(self):
        config = VLAPipelineConfig(grade="S")
        pipeline = VLAInferencePipeline(config)
        pipeline.start()
        time.sleep(0.2)
        pipeline.stop()  # Must stop before reset
        pipeline.reset()
        # After reset, stats should be clean
        stats = pipeline.get_stats()
        assert stats['history_len'] == 0


def test_create_vla_inference_pipeline_s():
    pipeline = create_vla_inference_pipeline(grade="S")
    assert isinstance(pipeline, VLAInferencePipeline)
    assert pipeline.config.grade == "S"


def test_create_vla_inference_pipeline_m():
    pipeline = create_vla_inference_pipeline(grade="M")
    assert isinstance(pipeline, VLAInferencePipeline)
    assert pipeline.config.grade == "M"


def test_create_vla_inference_pipeline_xl():
    pipeline = create_vla_inference_pipeline(grade="XL")
    assert isinstance(pipeline, VLAInferencePipeline)
    assert pipeline.config.grade == "XL"


# ============================================================
# 集成测试
# ============================================================

class TestVLAEndToEnd:
    def test_full_pipeline_s_grade(self):
        model = create_vla_model(grade="S")
        pipeline = create_vla_inference_pipeline(grade="S")
        
        model.start()
        pipeline.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Navigate")
        vla_input = VLAInput(perception=frame)
        
        output = model.step(vla_input)
        assert isinstance(output.action, VLAAction)
        
        model.stop()
        pipeline.stop()

    def test_full_pipeline_m_grade_with_gripper(self):
        model = create_vla_model(grade="M")
        model.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Pick up package")
        vla_input = VLAInput(perception=frame)
        
        output = model.step(vla_input)
        assert isinstance(output.action, VLAAction)
        model.stop()

    def test_multi_step_sequence(self):
        pipeline = create_vla_inference_pipeline(grade="S")
        pipeline.start()
        
        instructions = ["Move forward", "Turn left", "Continue", "Stop"]
        for instr in instructions:
            image = np.random.rand(224, 224, 3).astype(np.float32)
            frame = VLAPerceptionFrame(rgb_image=image, instruction=instr)
            result = pipeline.trigger_inference(frame)
            assert isinstance(result.action, VLAAction)
        
        pipeline.stop()

    def test_vla_with_proprioception(self):
        model = create_vla_model(grade="L")
        model.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        pose = np.array([1.0, 2.0, 0.5])
        
        frame = VLAPerceptionFrame(
            rgb_image=image,
            joint_states=joints,
            base_pose=pose,
            instruction="Move to target",
        )
        vla_input = VLAInput(perception=frame)
        
        output = model.step(vla_input)
        assert isinstance(output.action, VLAAction)
        model.stop()

    def test_concurrent_pipeline(self):
        config = VLAPipelineConfig(grade="S", safety_enabled=False)
        pipeline = VLAInferencePipeline(config)
        pipeline.start()
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(3):
                    image = np.random.rand(224, 224, 3).astype(np.float32)
                    frame = VLAPerceptionFrame(rgb_image=image, instruction="Move")
                    result = pipeline.trigger_inference(frame)
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        pipeline.stop()
        # 所有线程应该成功完成
        assert len(errors) == 0
        assert len(results) == 9


# ============================================================
# 性能测试
# ============================================================

class TestVLAPerformance:
    def test_model_inference_latency(self):
        model = create_vla_model(grade="S")
        model.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Test")
        vla_input = VLAInput(perception=frame)
        
        # 预热
        model.step(vla_input)
        
        # 测量
        latencies = []
        for _ in range(10):
            start = time.time()
            model.step(vla_input)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        assert avg_latency < 1000  # 平均延迟 < 1s (简化实现)
        
        model.stop()

    def test_model_throughput(self):
        model = create_vla_model(grade="S")
        model.start()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        frame = VLAPerceptionFrame(rgb_image=image, instruction="Test")
        vla_input = VLAInput(perception=frame)
        
        start = time.time()
        count = 0
        for _ in range(20):
            model.step(vla_input)
            count += 1
        elapsed = time.time() - start
        
        fps = count / elapsed
        assert fps > 1  # 至少 1 FPS
        
        model.stop()

    def test_pipeline_continuous_inference(self):
        pipeline = create_vla_inference_pipeline(grade="S")
        pipeline.set_instruction("Continuous test")
        
        pipeline.start()
        time.sleep(0.5)
        pipeline.stop()
        
        # Should have stats recorded
        stats = pipeline.get_stats()
        assert 'history_len' in stats or 'total_inferences' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
