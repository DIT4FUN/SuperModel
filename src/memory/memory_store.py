"""
Memory Store - 记忆存储层
=========================

持久化存储接口，支持多种后端。

职责:
- 数据序列化/反序列化
- 文件系统存储
- 数据库存储 (可选)
- 备份和恢复
- 存储压缩
"""

from __future__ import annotations

import os
import json
import shutil
import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import time
import threading


class StorageBackend(Enum):
    """存储后端类型"""
    JSON_FILE = "json_file"
    COMPRESSED_JSON = "compressed_json"
    # SQLITE = "sqlite"  # 未来扩展


class MemoryStore:
    """
    统一存储接口
    
    支持:
    - JSON 文件存储
    - 压缩 JSON 存储
    - 增量备份
    - 自动保存
    """
    
    def __init__(
        self,
        base_path: str,
        backend: str = "compressed_json",
        auto_save: bool = True,
        save_interval_s: float = 60.0,
        max_backups: int = 5,
    ):
        """
        Args:
            base_path: 存储基础路径
            backend: 存储后端
            auto_save: 自动保存
            save_interval_s: 自动保存间隔
            max_backups: 最大备份数
        """
        self.base_path = Path(base_path)
        self.backend = backend
        self.auto_save = auto_save
        self.save_interval = save_interval_s
        self.max_backups = max_backups
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self._cache: Dict[str, Any] = {}
        self._dirty: set = set()
        self._cache_lock = threading.RLock()
        
        # 自动保存线程
        self._save_thread: Optional[threading.Thread] = None
        self._stop_save_thread = threading.Event()
        
        if auto_save:
            self._start_auto_save()
    
    def _start_auto_save(self) -> None:
        """启动自动保存线程"""
        def auto_save_loop():
            while not self._stop_save_thread.wait(self.save_interval):
                with self._cache_lock:
                    if self._dirty:
                        self._save_dirty()
        
        self._save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        self._save_thread.start()
    
    def stop_auto_save(self) -> None:
        """停止自动保存"""
        self._stop_save_thread.set()
        if self._save_thread:
            self._save_thread.join(timeout=5.0)
    
    # ==================== 核心操作 ====================
    
    def save(self, key: str, data: Dict[str, Any]) -> bool:
        """
        保存数据
        
        Args:
            key: 存储键
            data: 数据
            
        Returns:
            是否成功
        """
        with self._cache_lock:
            self._cache[key] = data
            self._dirty.add(key)
            
            if not self.auto_save:
                return self._save_key(key)
        
        return True
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        加载数据
        
        Args:
            key: 存储键
            
        Returns:
            数据或None
        """
        with self._cache_lock:
            # 先检查缓存
            if key in self._cache:
                return self._cache[key]
            
            # 从磁盘加载
            data = self._load_key(key)
            if data is not None:
                self._cache[key] = data
            
            return data
    
    def delete(self, key: str) -> bool:
        """删除数据"""
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
            
            self._dirty.discard(key)
            
            # 删除文件
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
        
        return False
    
    def exists(self, key: str) -> bool:
        """检查是否存在"""
        with self._cache_lock:
            if key in self._cache:
                return True
            
            return self._get_file_path(key).exists()
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        列出所有键
        
        Args:
            pattern: 可选的模式过滤
            
        Returns:
            键列表
        """
        with self._cache_lock:
            keys = list(self._cache.keys())
        
        # 扫描文件
        if self.base_path.exists():
            for f in self.base_path.iterdir():
                if f.is_file():
                    suffix = f".{self.backend}.json"
                    if f.suffix == suffix or f.suffix == ".json":
                        key = f.stem.replace(f".{self.backend}", "")
                        if key not in keys:
                            keys.append(key)
        
        # 模式过滤
        if pattern:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        
        return keys
    
    # ==================== 文件操作 ====================
    
    def _get_file_path(self, key: str) -> Path:
        """获取文件路径"""
        return self.base_path / f"{key}.{self.backend}.json"
    
    def _save_key(self, key: str) -> bool:
        """保存单个键到磁盘"""
        if key not in self._cache:
            return False
        
        file_path = self._get_file_path(key)
        
        try:
            data = self._cache[key]
            
            # 序列化
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            
            # 压缩
            if self.backend == "compressed_json":
                json_bytes = json_str.encode('utf-8')
                compressed = gzip.compress(json_bytes)
                
                with open(file_path, 'wb') as f:
                    f.write(compressed)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            return True
            
        except Exception as e:
            print(f"Failed to save {key}: {e}")
            return False
    
    def _load_key(self, key: str) -> Optional[Dict[str, Any]]:
        """从磁盘加载单个键"""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            # 尝试其他后缀
            for backend in ["compressed_json", "json_file"]:
                alt_path = self.base_path / f"{key}.{backend}.json"
                if alt_path.exists():
                    file_path = alt_path
                    break
            else:
                return None
        
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            return data
            
        except Exception as e:
            print(f"Failed to load {key}: {e}")
            return None
    
    def _save_dirty(self) -> None:
        """保存所有脏数据"""
        for key in list(self._dirty):
            self._save_key(key)
        self._dirty.clear()
    
    def save_all(self) -> bool:
        """保存所有数据"""
        with self._cache_lock:
            self._save_dirty()
        return True
    
    # ==================== 备份 ====================
    
    def create_backup(self, name: Optional[str] = None) -> str:
        """
        创建备份
        
        Args:
            name: 备份名称，默认使用时间戳
            
        Returns:
            备份路径
        """
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_path = self.base_path / "backups" / name
        
        with self._cache_lock:
            # 保存当前状态
            self._save_dirty()
            
            # 复制所有文件
            backup_path.mkdir(parents=True, exist_ok=True)
            
            for f in self.base_path.iterdir():
                if f.is_file() and f.suffix == ".json":
                    shutil.copy2(f, backup_path / f.name)
        
        # 清理旧备份
        self._cleanup_backups()
        
        return str(backup_path)
    
    def _cleanup_backups(self) -> None:
        """清理旧备份"""
        backup_dir = self.base_path / "backups"
        if not backup_dir.exists():
            return
        
        backups = sorted(
            [(p, p.stat().st_mtime) for p in backup_dir.iterdir() if p.is_dir()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for path, _ in backups[self.max_backups:]:
            shutil.rmtree(path)
    
    def restore_backup(self, name: str) -> bool:
        """
        恢复备份
        
        Args:
            name: 备份名称
            
        Returns:
            是否成功
        """
        backup_path = self.base_path / "backups" / name
        
        if not backup_path.exists():
            return False
        
        with self._cache_lock:
            # 清空当前
            self._cache.clear()
            self._dirty.clear()
            
            # 从备份恢复
            for f in backup_path.iterdir():
                if f.is_file():
                    key = f.stem.replace(f".{self.backend}", "")
                    data = self._load_key(key)
                    if data:
                        self._cache[key] = data
            
            # 强制保存
            self._save_dirty()
        
        return True
    
    # ==================== 存储信息 ====================
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        with self._cache_lock:
            cached_keys = len(self._cache)
            dirty_count = len(self._dirty)
        
        files = list(self.base_path.iterdir()) if self.base_path.exists() else []
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        return {
            'base_path': str(self.base_path),
            'backend': self.backend,
            'cached_keys': cached_keys,
            'dirty_keys': dirty_count,
            'file_count': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
        }
    
    def clear_cache(self) -> None:
        """清除缓存"""
        with self._cache_lock:
            self._cache.clear()
            self._dirty.clear()
    
    def close(self) -> None:
        """关闭存储"""
        self.stop_auto_save()
        self.save_all()


# 辅助函数
def compute_hash(data: Any) -> str:
    """计算数据哈希"""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def merge_memory_data(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
    strategy: str = "new_wins",
) -> Dict[str, Any]:
    """
    合并记忆数据
    
    Args:
        old_data: 旧数据
        new_data: 新数据
        strategy: 合并策略
            - "new_wins": 新数据优先
            - "old_wins": 旧数据优先
            - "merge": 合并
        
    Returns:
        合并后的数据
    """
    if strategy == "new_wins":
        return {**old_data, **new_data}
    
    if strategy == "old_wins":
        return {**new_data, **old_data}
    
    if strategy == "merge":
        result = {**old_data, **new_data}
        
        # 对于列表，合并
        for key in set(old_data.keys()) & set(new_data.keys()):
            if isinstance(old_data[key], list) and isinstance(new_data[key], list):
                result[key] = old_data[key] + new_data[key]
        
        return result
    
    return new_data
