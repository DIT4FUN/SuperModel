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
NPU 加速器模块
==============

瑞芯微 RKNN (Rockchip Neural Network) 加速支持。

提供:
- RKNN 模型加载与推理
- NPU 利用率监控
- 算子支持查询
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union


class NPUPlugin(Enum):
    """NPU 插件类型"""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    AUTO = "auto"


@dataclass
class NPUContext:
    """NPU 上下文"""
    board: 'BoardBase'
    device_id: int = 0
    priority: int = 0
    model_path: Optional[str] = None
    is_initialized: bool = False


class NPUAccelerator:
    """
    NPU 加速器抽象类
    
    提供统一的 NPU 加速接口。
    """
    
    def __init__(self, board: 'BoardBase'):
        """
        初始化 NPU 加速器
        
        Args:
            board: 主板实例
        """
        self.board = board
        self._context: Optional[NPUContext] = None
        self._model_loaded = False
        self._inference_count = 0
        self._total_inference_time = 0.0
        
    @property
    def is_available(self) -> bool:
        """NPU 是否可用"""
        if not hasattr(self.board.info, 'npu_tops'):
            return False
        return self.board.info.npu_tops > 0
    
    @property
    def utilization(self) -> float:
        """NPU 利用率 (0-1)"""
        if isinstance(self.board, type):
            return 0.0
        return getattr(self.board, 'get_npu_utilization', lambda: 0.0)()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """推理统计"""
        avg_time = self._total_inference_time / self._inference_count if self._inference_count > 0 else 0
        return {
            'inference_count': self._inference_count,
            'total_time_ms': self._total_inference_time * 1000,
            'avg_time_ms': avg_time * 1000,
            'throughput_fps': 1.0 / avg_time if avg_time > 0 else 0,
        }
    
    def load_model(self, model_path: str) -> bool:
        """
        加载 RKNN 模型
        
        Args:
            model_path: 模型文件路径 (.rknn)
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return False
        
        self._model_loaded = True
        self._context = NPUContext(
            board=self.board,
            model_path=model_path,
            is_initialized=True
        )
        return True
    
    @abstractmethod
    def infer(
        self, 
        inputs: Dict[str, Any], 
        timeout_ms: int = 1000
    ) -> Optional[Dict[str, Any]]:
        """
        执行推理
        
        Args:
            inputs: 输入数据字典 {name: numpy array}
            timeout_ms: 超时毫秒
            
        Returns:
            输出数据字典
        """
        pass
    
    def unload_model(self) -> None:
        """卸载模型"""
        self._model_loaded = False
        self._context = None
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._inference_count = 0
        self._total_inference_time = 0.0


class MockNPUAccelerator(NPUAccelerator):
    """
    模拟 NPU 加速器 (用于开发/x86_64 环境)
    """
    
    def __init__(self, board: 'BoardBase'):
        super().__init__(board)
        self._mock_latency_ms = 10  # 模拟 10ms 延迟
    
    def load_model(self, model_path: str) -> bool:
        """加载模型 (模拟)"""
        print(f"[MockNPU] Loading model: {model_path}")
        return super().load_model(model_path)
    
    def infer(
        self, 
        inputs: Dict[str, Any], 
        timeout_ms: int = 1000
    ) -> Optional[Dict[str, Any]]:
        """执行推理 (模拟)"""
        if not self._model_loaded:
            return None
        
        # 模拟推理延迟
        time.sleep(self._mock_latency_ms / 1000.0)
        
        self._inference_count += 1
        self._total_inference_time += self._mock_latency_ms / 1000.0
        
        # 返回模拟输出
        return {
            'output': inputs.get('input', [[0.0]]),
        }


def get_npu_context(board: 'BoardBase') -> NPUContext:
    """
    获取 NPU 上下文
    
    Args:
        board: 主板实例
        
    Returns:
        NPUContext
    """
    return NPUContext(
        board=board,
        is_initialized=board.is_initialized
    )


def create_npu_accelerator(board: 'BoardBase') -> NPUAccelerator:
    """
    创建 NPU 加速器实例
    
    Args:
        board: 主板实例
        
    Returns:
        NPUAccelerator
    """
    # 检测是否为真实 RK3588 平台
    if board.info.arch == 'aarch64' and board.info.npu_tops > 0:
        # 真实 RK3588 平台
        try:
            # 尝试导入 rknn 库
            import rknn
            return RKNNAccelerator(board)
        except ImportError:
            print("Warning: rknn package not found, using mock NPU")
            return MockNPUAccelerator(board)
    else:
        # x86_64 或无 NPU 平台
        return MockNPUAccelerator(board)


class RKNNAccelerator(NPUAccelerator):
    """
    RKNN 加速器 (需要 rknn_api 安装)
    
    用法:
        from hardware import create_board
        board = create_board(BoardType.RDK_X5_ULTRA)
        acc = create_npu_accelerator(board)
        acc.load_model('model.rknn')
        output = acc.infer({'input': data})
    """
    
    def __init__(self, board: 'BoardBase'):
        super().__init__(board)
        self._rknn_model = None
        self._input_nodes = []
        self._output_nodes = []
    
    def load_model(self, model_path: str) -> bool:
        """加载 RKNN 模型"""
        try:
            from rknn.api import rknn
            self._rknn_model = rknn()
            
            # 初始化 RKNN 运行时
            ret = self._rknn_model.load_rknn(model_path)
            if ret != 0:
                print(f"Failed to load RKNN model: {ret}")
                return False
            
            # 初始化运行时
            ret = self._rknn_model.init_runtime()
            if ret != 0:
                print(f"Failed to init runtime: {ret}")
                return False
            
            return super().load_model(model_path)
            
        except ImportError:
            print("Error: rknn package not installed")
            print("Install: pip install rknn_api")
            return False
        except Exception as e:
            print(f"RKNN load error: {e}")
            return False
    
    def infer(
        self, 
        inputs: Dict[str, Any], 
        timeout_ms: int = 1000
    ) -> Optional[Dict[str, Any]]:
        """执行 RKNN 推理"""
        if not self._model_loaded or self._rknn_model is None:
            return None
        
        start_time = time.time()
        
        try:
            # 格式化为列表
            input_list = list(inputs.values())
            
            # 执行推理
            outputs = self._rknn_model.inference(inputs=input_list)
            
            # 统计
            elapsed = time.time() - start_time
            self._inference_count += 1
            self._total_inference_time += elapsed
            
            # 转换为字典
            return {'output': outputs[0] if len(outputs) == 1 else outputs}
            
        except Exception as e:
            print(f"RKNN inference error: {e}")
            return None
    
    def unload_model(self) -> None:
        """卸载 RKNN 模型"""
        if self._rknn_model:
            try:
                self._rknn_model.release()
            except:
                pass
            self._rknn_model = None
        super().unload_model()


# RKNN 模型转换工具函数
def convert_tflite_to_rknn(
    tflite_path: str,
    output_path: str,
    dataset_path: Optional[str] = None
) -> bool:
    """
    将 TensorFlow Lite 模型转换为 RKNN 模型
    
    Args:
        tflite_path: 输入 .tflite 文件路径
        output_path: 输出 .rknn 文件路径
        dataset_path: 校准数据集路径 (可选)
        
    Returns:
        是否转换成功
    """
    try:
        from rknn.api import rknn
        
        rknn_model = rknn()
        
        # 加载 TFLite 模型
        print(f"Loading TFLite model: {tflite_path}")
        ret = rknn_model.load_tflite(tflite_path)
        if ret != 0:
            print(f"Failed to load TFLite: {ret}")
            return False
        
        # 构建模型
        print("Building RKNN model...")
        ret = rknn_model.build(do_quantization=True, dataset=dataset_path)
        if ret != 0:
            print(f"Failed to build: {ret}")
            return False
        
        # 导出模型
        print(f"Exporting to: {output_path}")
        ret = rknn_model.export_rknn(output_path)
        if ret != 0:
            print(f"Failed to export: {ret}")
            return False
        
        rknn_model.release()
        return True
        
    except ImportError:
        print("Error: rknn package not installed")
        return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def query_rknn_ops() -> List[str]:
    """
    查询 RKNN 支持的算子列表
    
    Returns:
        算子名称列表
    """
    try:
        from rknn.api import rknn
        ops = rknn.api.rknn_runtime.query_rknn_ops()
        return ops if ops else []
    except:
        return [
            'Conv2D', 'DepthwiseConv2D', 'FullyConnected',
            'ReLU', 'ReLU6', 'LeakyReLU',
            'MaxPool2D', 'AvgPool2D',
            'BatchNorm', 'Add', 'Concat',
            'Softmax', 'Sigmoid',
            'Reshape', 'Transpose',
        ]
