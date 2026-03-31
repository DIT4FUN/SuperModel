"""
技能库模块
==========

层次化技能库管理
- 基础技能注册
- 技能参数化
- 技能序列编排
- 从演示中学习 (LfD)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import time


class SkillStatus(Enum):
    """技能执行状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SkillResult:
    """技能执行结果"""
    success: bool
    status: SkillStatus
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration: float = 0.0


@dataclass
class SkillConfig:
    """技能配置"""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0  # 秒
    max_retries: int = 0


class Skill:
    """
    技能基类
    
    所有具体技能继承此类
    """
    
    def __init__(self, config: SkillConfig):
        self.config = config
        self.status = SkillStatus.IDLE
        self.start_time: Optional[float] = None
        self._is_cancelled = False
        
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """
        检查是否可以执行
        
        Args:
            context: 执行上下文 (传感器数据/环境状态等)
            
        Returns:
            True if skill can be executed
        """
        return True
    
    def execute(self, context: Dict[str, Any]) -> SkillResult:
        """
        执行技能
        
        Args:
            context: 执行上下文
            
        Returns:
            SkillResult: 执行结果
        """
        raise NotImplementedError("Subclass must implement execute()")
    
    def cancel(self):
        """请求取消技能"""
        self._is_cancelled = True
        self.status = SkillStatus.CANCELLED
    
    def check_timeout(self) -> bool:
        """检查是否超时"""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.config.timeout


class PrimitiveSkill(Skill):
    """原子技能 - 不可再分的基础动作"""
    pass


class CompositeSkill(Skill):
    """组合技能 - 由多个子技能组成"""
    def __init__(self, config: SkillConfig, sub_skills: Optional[List[Skill]] = None, execution_policy: str = "sequential"):
        super().__init__(config)
        self.sub_skills = sub_skills or []
        self.execution_policy = execution_policy
    
    def execute(self, context: Dict[str, Any]) -> SkillResult:
        """顺序或并行执行子技能"""
        start = time.time()
        
        if self.execution_policy == "sequential":
            for skill in self.sub_skills:
                if self._is_cancelled:
                    return SkillResult(False, SkillStatus.CANCELLED, duration=time.time()-start)
                
                if not skill.can_execute(context):
                    continue
                    
                result = skill.execute(context)
                if not result.success:
                    return result
        
        elif self.execution_policy == "parallel":
            return ParallelExecutor.execute_parallel(
                self.sub_skills, context, timeout=self.config.timeout
            )
        
        elif self.execution_policy == "parallel_any":
            # 任意一个成功即可
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(self.sub_skills)) as executor:
                futures = {executor.submit(s.execute, context): s for s in self.sub_skills}
                for future in as_completed(futures, timeout=self.config.timeout):
                    try:
                        result = future.result()
                        if result.success:
                            return result
                    except Exception:
                        continue
        
        return SkillResult(True, SkillStatus.SUCCEEDED, duration=time.time()-start)


class ParallelExecutor:
    """并行执行器"""
    
    @staticmethod
    def execute_parallel(skills: List[Skill], context: Dict[str, Any], timeout: float = 30.0) -> SkillResult:
        """
        并行执行多个技能
        
        Args:
            skills: 技能列表
            context: 执行上下文
            timeout: 超时时间
            
        Returns:
            汇总的执行结果
        """
        import threading
        import queue
        
        results = {}
        errors = {}
        
        def run_skill(skill, result_queue, error_queue):
            try:
                if skill.can_execute(context):
                    result = skill.execute(context)
                    result_queue.put((skill.config.name, result))
                else:
                    result_queue.put((skill.config.name, SkillResult(False, SkillStatus.FAILED, error_message="Prerequisites not met")))
            except Exception as e:
                error_queue.put((skill.config.name, str(e)))
        
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        threads = []
        
        start_time = time.time()
        
        # 启动所有线程
        for skill in skills:
            t = threading.Thread(target=run_skill, args=(skill, result_queue, error_queue))
            t.daemon = True
            t.start()
            threads.append(t)
        
        # 等待完成或超时
        for t in threads:
            remaining = max(0.1, timeout - (time.time() - start_time))
            t.join(timeout=remaining)
            if time.time() - start_time > timeout:
                break
        
        # 收集结果
        while not result_queue.empty():
            name, result = result_queue.get_nowait()
            results[name] = result
        
        while not error_queue.empty():
            name, error = error_queue.get_nowait()
            errors[name] = error
        
        duration = time.time() - start_time
        
        # 判断整体成功
        if errors:
            return SkillResult(False, SkillStatus.FAILED, output={'errors': errors}, duration=duration)
        
        all_success = all(r.success for r in results.values()) if results else False
        status = SkillStatus.SUCCEEDED if all_success else SkillStatus.FAILED
        
        return SkillResult(all_success, status, output=results, duration=duration)


class SkillLibrary:
    """
    技能库
    
    管理所有可用技能
    """
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skill_factories: Dict[str, Callable[[Dict], Skill]] = {}
        
        # 注册内置技能
        self._register_builtin_skills()
    
    def _register_builtin_skills(self):
        """注册内置技能"""
        # 基础运动技能
        self.register_skill_factory("move_to", self._create_move_to_skill)
        self.register_skill_factory("follow_trajectory", self._create_follow_trajectory_skill)
        self.register_skill_factory("grasp", self._create_grasp_skill)
        self.register_skill_factory("place", self._create_place_skill)
        
        # 感知技能
        self.register_skill_factory("look_at", self._create_look_at_skill)
        self.register_skill_factory("localize", self._create_localize_skill)
        
        # 交互技能
        self.register_skill_factory("push", self._create_push_skill)
        self.register_skill_factory("pull", self._create_pull_skill)
    
    def register_skill_factory(
        self,
        name: str,
        factory: Callable[[Dict], Skill]
    ):
        """注册技能工厂"""
        self._skill_factories[name] = factory
    
    def create_skill(self, name: str, config: Dict) -> Optional[Skill]:
        """创建技能实例"""
        if name not in self._skill_factories:
            return None
        return self._skill_factories[name](config)
    
    def register_skill(self, skill: Skill):
        """注册技能实例"""
        self._skills[skill.config.name] = skill
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self._skills.get(name)
    
    def list_skills(self) -> List[str]:
        """列出所有技能"""
        return list(self._skills.keys())
    
    def _create_move_to_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="move_to",
            description="移动到目标位置",
            parameters=config
        ))
    
    def _create_follow_trajectory_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="follow_trajectory",
            description="沿轨迹运动",
            parameters=config
        ))
    
    def _create_grasp_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="grasp",
            description="抓取物体",
            parameters=config
        ))
    
    def _create_place_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="place",
            description="放置物体",
            parameters=config
        ))
    
    def _create_look_at_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="look_at",
            description="注视目标",
            parameters=config
        ))
    
    def _create_localize_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="localize",
            description="定位",
            parameters=config
        ))
    
    def _create_push_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="push",
            description="推动物体",
            parameters=config
        ))
    
    def _create_pull_skill(self, config: Dict) -> Skill:
        return PrimitiveSkill(SkillConfig(
            name="pull",
            description="拉动物体",
            parameters=config
        ))


class SkillRegistry:
    """
    技能注册中心
    
    全局单例，管理所有技能库
    """
    _instance: Optional['SkillRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.library = SkillLibrary()
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'SkillRegistry':
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# 预定义技能配置
PRESET_GRASP_CONFIGS = {
    "top_grasp": {
        "approach_height": 0.1,
        "gripper_width": 0.08,
        "force_threshold": 5.0
    },
    "side_grasp": {
        "approach_angle": 0.0,
        "gripper_width": 0.05,
        "force_threshold": 3.0
    }
}


# =============================================================================
# AGV五级技能库规格
# =============================================================================

class SkillGrade(Enum):
    """技能库等级"""
    S = "S"   # 教育/实验
    M = "M"   # 标准助手
    L = "L"   # 专业工业
    XL = "XL"  # 高性能
    XXL = "XXL"  # 旗舰全功能


@dataclass
class SkillLibrarySpec:
    """技能库规格参数"""
    grade: SkillGrade
    max_primitive_skills: int       # 最大基础技能数量
    max_composite_skills: int       # 最大组合技能数量
    max_learning_rate: float        # 最大学习率
    supports_adaptation: bool        # 支持技能自适应
    supports_learning: bool         # 支持技能学习
    supports_composition: bool      # 支持技能组合
    supports_imitation: bool        # 支持模仿学习
    skill_library_size: int         # 技能库总容量
    control_frequency: float        # 技能执行频率 Hz
    
    @classmethod
    def from_grade(cls, grade: SkillGrade) -> "SkillLibrarySpec":
        specs = {
            SkillGrade.S: cls(
                grade=SkillGrade.S,
                max_primitive_skills=5, max_composite_skills=2,
                max_learning_rate=0.001, supports_adaptation=False,
                supports_learning=False, supports_composition=False,
                supports_imitation=False, skill_library_size=5,
                control_frequency=10.0
            ),
            SkillGrade.M: cls(
                grade=SkillGrade.M,
                max_primitive_skills=20, max_composite_skills=10,
                max_learning_rate=0.01, supports_adaptation=True,
                supports_learning=False, supports_composition=True,
                supports_imitation=False, skill_library_size=30,
                control_frequency=20.0
            ),
            SkillGrade.L: cls(
                grade=SkillGrade.L,
                max_primitive_skills=50, max_composite_skills=30,
                max_learning_rate=0.05, supports_adaptation=True,
                supports_learning=True, supports_composition=True,
                supports_imitation=True, skill_library_size=80,
                control_frequency=50.0
            ),
            SkillGrade.XL: cls(
                grade=SkillGrade.XL,
                max_primitive_skills=100, max_composite_skills=50,
                max_learning_rate=0.1, supports_adaptation=True,
                supports_learning=True, supports_composition=True,
                supports_imitation=True, skill_library_size=150,
                control_frequency=100.0
            ),
            SkillGrade.XXL: cls(
                grade=SkillGrade.XXL,
                max_primitive_skills=200, max_composite_skills=100,
                max_learning_rate=0.2, supports_adaptation=True,
                supports_learning=True, supports_composition=True,
                supports_imitation=True, skill_library_size=300,
                control_frequency=200.0
            ),
        }
        return specs[grade]


def get_skill_library_spec(grade: str) -> SkillLibrarySpec:
    """获取AGV指定等级的技能库规格"""
    try:
        g = SkillGrade(grade)
    except ValueError:
        g = SkillGrade.M
    return SkillLibrarySpec.from_grade(g)


# AGV五级技能库等级规格快速表
AGV_SKILL_GRADES = {
    'S':  {'primitive': 5,   'composite': 2,   'learning': False, 'adaptation': False, 'composition': False},
    'M':  {'primitive': 20,  'composite': 10,  'learning': False, 'adaptation': True,  'composition': True},
    'L':  {'primitive': 50,  'composite': 30,  'learning': True,  'adaptation': True,  'composition': True},
    'XL': {'primitive': 100, 'composite': 50,  'learning': True,  'adaptation': True,  'composition': True},
    'XXL': {'primitive': 200, 'composite': 100, 'learning': True,  'adaptation': True,  'composition': True},
}


def get_skill_spec(grade: str) -> dict:
    """获取AGV指定等级的技能规格"""
    return AGV_SKILL_GRADES.get(grade, AGV_SKILL_GRADES['M'])
