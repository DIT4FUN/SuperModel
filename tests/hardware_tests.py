"""
硬件支持测试
============

测试 RK3588 和地瓜机器人 RDK 系列主板支持。

运行: pytest tests/hardware_tests.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestHardwareModule:
    """硬件模块基础测试"""
    
    def test_import_hardware(self):
        """测试硬件模块导入"""
        from hardware import (
            BoardType, BoardInfo, BoardBase,
            create_board, detect_board,
            DiguRobotPlatform, RDKX3, RDKX5Ultra, RDKS100,
            GPIOController, NPUAccelerator
        )
        assert True
    
    def test_board_type_enum(self):
        """测试主板类型枚举"""
        from hardware import BoardType
        
        assert BoardType.RDK_X3.value == "rdk_x3"
        assert BoardType.RDK_X5_ULTRA.value == "rdk_x5_ultra"
        assert BoardType.RDK_S100.value == "rdk_s100"
    
    def test_peripheral_type_enum(self):
        """测试外设类型枚举"""
        from hardware import PeripheralType
        
        assert PeripheralType.GPIO.value == "gpio"
        assert PeripheralType.I2C.value == "i2c"
        assert PeripheralType.SPI.value == "spi"
        assert PeripheralType.UART.value == "uart"


class TestRDKX5Ultra:
    """RDK X5 Ultra 旗舰主板测试"""
    
    def test_create_rdk_x5_ultra(self):
        """测试创建 RDK X5 Ultra 主板"""
        from hardware import create_board, BoardType
        
        board = create_board(BoardType.RDK_X5_ULTRA)
        assert board.BOARD_TYPE == BoardType.RDK_X5_ULTRA
    
    def test_rdk_x5_ultra_specs(self):
        """测试 RDK X5 Ultra 规格"""
        from hardware import create_board, BoardType
        
        board = create_board(BoardType.RDK_X5_ULTRA)
        info = board.info
        
        assert info.chip == "RK3588"
        assert info.cpu_cores == 8
        assert info.npu_tops == 12.0
        # memory_mb 从实际系统检测，不检查具体值


class TestRDKX3:
    """RDK X3 主板测试"""
    
    def test_create_rdk_x3(self):
        """测试创建 RDK X3 主板"""
        from hardware import create_board, BoardType
        
        board = create_board(BoardType.RDK_X3)
        assert board.BOARD_TYPE == BoardType.RDK_X3
    
    def test_rdk_x3_specs(self):
        """测试 RDK X3 规格"""
        from hardware import create_board, BoardType
        
        board = create_board(BoardType.RDK_X3)
        info = board.info
        
        assert info.chip == "RK3588V2"
        assert info.cpu_cores == 8
        assert info.npu_tops == 6.0
        # memory_mb 从实际系统检测，不检查具体值


class TestRDKS100:
    """RDK S100 主板测试"""
    
    def test_create_rdk_s100(self):
        """测试创建 RDK S100 主板"""
        from hardware import create_board, BoardType
        
        board = create_board(BoardType.RDK_S100)
        assert board.BOARD_TYPE == BoardType.RDK_S100
    
    def test_rdk_s100_specs(self):
        """测试 RDK S100 规格"""
        from hardware import create_board, BoardType
        
        board = create_board(BoardType.RDK_S100)
        info = board.info
        
        assert info.chip == "RK3562"
        assert info.cpu_cores == 4
        assert info.npu_tops == 3.0
        # memory_mb 从实际系统检测，不检查具体值


class TestGPIOController:
    """GPIO 控制器测试"""
    
    def test_create_gpio_controller(self):
        """测试创建 GPIO 控制器"""
        from hardware import create_board, BoardType, GPIOController
        
        board = create_board(BoardType.RDK_X5_ULTRA)
        gpio = GPIOController(board)
        assert gpio is not None


class TestNPUAccelerator:
    """NPU 加速器测试"""
    
    def test_create_npu_context(self):
        """测试创建 NPU 上下文"""
        from hardware import create_board, BoardType, get_npu_context
        
        board = create_board(BoardType.RDK_X5_ULTRA)
        ctx = get_npu_context(board)
        assert ctx is not None


class TestAutonomousLearning:
    """自主学习框架测试"""
    
    def test_import_autonomous_learning(self):
        """测试导入自主学习模块"""
        from learning import (
            AutonomousLearningConfig,
            Experience,
            PrioritizedReplayBuffer,
            EWC,
            MetaLearner,
            CuriosityModule,
            SkillLibrary,
            AutonomousLearningAgent
        )
        assert True
    
    def test_experience_dataclass(self):
        """测试 Experience 数据类"""
        from learning import Experience
        import numpy as np
        
        exp = Experience(
            state={'vision': np.zeros(10)},
            action=np.zeros(6),
            reward=1.0,
            next_state={'vision': np.zeros(10)},
            done=False,
            priority=1.0
        )
        assert exp.reward == 1.0
        assert exp.priority == 1.0
    
    def test_prioritized_replay_buffer(self):
        """测试优先经验回放"""
        from learning import Experience, PrioritizedReplayBuffer
        import numpy as np
        
        buffer = PrioritizedReplayBuffer(capacity=100)
        exp = Experience(
            state={'v': np.zeros(3)},
            action=np.zeros(6),
            reward=1.0,
            next_state={'v': np.zeros(3)},
            done=False
        )
        buffer.push(exp)
        assert len(buffer) == 1
    
    def test_curiosity_module(self):
        """测试好奇心模块"""
        from learning import CuriosityModule
        
        curiosity = CuriosityModule(
            state_dim=10,
            action_dim=6,
            hidden_dim=32
        )
        assert curiosity is not None
    
    def test_skill_library(self):
        """测试技能库"""
        from learning import SkillLibrary
        
        lib = SkillLibrary(skill_dim=32, max_skills=10)
        assert lib.max_skills == 10
        assert len(lib.skills) == 0


class TestRDKComparison:
    """RDK 系列对比测试"""
    
    def test_all_rdk_boards_have_unique_specs(self):
        """测试所有 RDK 主板有不同规格"""
        from hardware import create_board, BoardType
        
        x3 = create_board(BoardType.RDK_X3)
        x5 = create_board(BoardType.RDK_X5_ULTRA)
        s100 = create_board(BoardType.RDK_S100)
        
        # NPU 算力应该不同
        assert x3.info.npu_tops != x5.info.npu_tops
        assert x3.info.npu_tops != s100.info.npu_tops
        assert x5.info.npu_tops != s100.info.npu_tops
        
        # X5 Ultra 应该最高
        assert x5.info.npu_tops == 12.0
    
    def test_rdk_specs_table(self):
        """测试 RDK 规格表"""
        from hardware.digu_robot import RDK_SPECS, DiguRobotSeries
        
        assert DiguRobotSeries.RDK_X3 in RDK_SPECS
        assert DiguRobotSeries.RDK_X5_ULTRA in RDK_SPECS
        assert DiguRobotSeries.RDK_S100 in RDK_SPECS
        
        # X5 Ultra 应该最贵
        x3_price = RDK_SPECS[DiguRobotSeries.RDK_X3]['price_range']
        x5_price = RDK_SPECS[DiguRobotSeries.RDK_X5_ULTRA]['price_range']
        s100_price = RDK_SPECS[DiguRobotSeries.RDK_S100]['price_range']
        
        assert '1200' in x5_price  # X5 Ultra 最贵
        assert '200' in s100_price  # S100 最便宜
