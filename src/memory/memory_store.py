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
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from enum import Enum
import time
import threading
import numpy as np

# 可选依赖
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

BOTO3_AVAILABLE = False
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    pass


class StorageBackend(Enum):
    """存储后端类型"""
    JSON_FILE = "json_file"
    COMPRESSED_JSON = "compressed_json"
    VECTOR_DB = "vector_db"  # 向量数据库
    # SQLITE = "sqlite"  # 未来扩展


class CloudStorageProvider(Enum):
    """云存储提供商"""
    S3 = "s3"          # AWS S3/兼容S3的服务 (MinIO/OSS/COS等)
    # GCS = "gcs"     # Google Cloud Storage (未来扩展)
    # AZURE = "azure" # Azure Blob Storage (未来扩展)


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
        # 向量数据库配置
        vector_dim: int = 1536,  # 嵌入维度 (默认OpenAI embedding维度)
        vector_index_type: str = "IVFFlat",
        # 云存储配置
        cloud_provider: Optional[str] = None,
        cloud_endpoint: Optional[str] = None,
        cloud_access_key: Optional[str] = None,
        cloud_secret_key: Optional[str] = None,
        cloud_bucket: Optional[str] = None,
        cloud_sync_interval_s: float = 3600.0,  # 云同步间隔
    ):
        """
        Args:
            base_path: 存储基础路径
            backend: 存储后端
            auto_save: 自动保存
            save_interval_s: 自动保存间隔
            max_backups: 最大备份数
            vector_dim: 向量嵌入维度
            vector_index_type: 向量索引类型
            cloud_provider: 云存储提供商 (None 表示不启用云存储)
            cloud_endpoint: 云存储端点
            cloud_access_key: 云存储访问密钥
            cloud_secret_key: 云存储秘密密钥
            cloud_bucket: 云存储桶名
            cloud_sync_interval_s: 云同步间隔 (秒)
        """
        self.base_path = Path(base_path)
        self.backend = backend
        self.auto_save = auto_save
        self.save_interval = save_interval_s
        self.max_backups = max_backups
        
        # 向量数据库配置
        self.vector_dim = vector_dim
        self.vector_index_type = vector_index_type
        self._vector_index: Optional[Any] = None
        self._vector_id_map: Dict[str, int] = {}  # memory_id -> 向量ID
        self._next_vector_id = 0
        self._vector_lock = threading.RLock()
        
        # 云存储配置
        self.cloud_provider = CloudStorageProvider(cloud_provider) if cloud_provider else None
        self.cloud_endpoint = cloud_endpoint
        self.cloud_access_key = cloud_access_key
        self.cloud_secret_key = cloud_secret_key
        self.cloud_bucket = cloud_bucket
        self.cloud_sync_interval = cloud_sync_interval_s
        self._cloud_client: Optional[Any] = None
        self._cloud_sync_thread: Optional[threading.Thread] = None
        self._stop_cloud_sync = threading.Event()
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "vectors").mkdir(parents=True, exist_ok=True)
        (self.base_path / "backups").mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self._cache: Dict[str, Any] = {}
        self._dirty: set = set()
        self._cache_lock = threading.RLock()
        
        # 初始化向量索引
        if backend == StorageBackend.VECTOR_DB.value:
            self._init_vector_index()
        
        # 初始化云存储客户端
        if self.cloud_provider:
            self._init_cloud_client()
        
        # 自动保存线程
        self._save_thread: Optional[threading.Thread] = None
        self._stop_save_thread = threading.Event()
        
        if auto_save:
            self._start_auto_save()
        
        # 云同步线程
        if self.cloud_provider and cloud_sync_interval_s > 0:
            self._start_cloud_sync()
    
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
    
    # ==================== 向量数据库方法 ====================
    
    def _init_vector_index(self) -> None:
        """初始化向量索引"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed, please install it with 'pip install faiss-cpu' or 'faiss-gpu'")
        
        vector_index_path = self.base_path / "vectors" / "index.faiss"
        id_map_path = self.base_path / "vectors" / "id_map.pkl"
        
        if vector_index_path.exists() and id_map_path.exists():
            # 加载现有索引
            self._vector_index = faiss.read_index(str(vector_index_path))
            with open(id_map_path, 'rb') as f:
                self._vector_id_map = pickle.load(f)
            self._next_vector_id = max(self._vector_id_map.values()) + 1 if self._vector_id_map else 0
        else:
            # 创建新索引
            if self.vector_index_type == "IVFFlat":
                quantizer = faiss.IndexFlatL2(self.vector_dim)
                nlist = min(4096, max(64, len(self._vector_id_map) // 10))  # 自适应聚类数
                self._vector_index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)
                self._vector_index.train(np.zeros((1, self.vector_dim), dtype=np.float32))  # 空训练
            else:  # 默认Flat
                self._vector_index = faiss.IndexFlatL2(self.vector_dim)
        
        self._vector_index.make_direct_map()
    
    def add_vector(self, memory_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加向量到索引
        
        Args:
            memory_id: 记忆ID
            vector: 嵌入向量
            metadata: 元数据
            
        Returns:
            是否成功
        """
        if self.backend != StorageBackend.VECTOR_DB.value:
            return False
        
        with self._vector_lock:
            if memory_id in self._vector_id_map:
                # 更新现有向量
                vec_id = self._vector_id_map[memory_id]
                self._vector_index.remove_ids(np.array([vec_id], dtype=np.int64))
            else:
                # 新增向量
                vec_id = self._next_vector_id
                self._next_vector_id += 1
                self._vector_id_map[memory_id] = vec_id
            
            # 添加向量
            vector = vector.reshape(1, -1).astype(np.float32)
            self._vector_index.add_with_ids(vector, np.array([vec_id], dtype=np.int64))
            
            # 存储元数据
            if metadata:
                self.save(f"vector_meta_{memory_id}", metadata)
            
            return True
    
    def search_vectors(self, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.8) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个结果
            threshold: 相似度阈值 (L2距离越小越相似，阈值为最大允许距离)
            
        Returns:
            列表: (memory_id, score, metadata)
        """
        if self.backend != StorageBackend.VECTOR_DB.value or self._vector_index is None:
            return []
        
        with self._vector_lock:
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            distances, indices = self._vector_index.search(query_vector, top_k)
            
            results = []
            id_to_memory = {v: k for k, v in self._vector_id_map.items()}
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                if dist > threshold:
                    continue
                
                memory_id = id_to_memory.get(idx)
                if not memory_id:
                    continue
                
                # 加载元数据
                metadata = self.load(f"vector_meta_{memory_id}") or {}
                # 转换为相似度分数 (0-1，越高越相似)
                similarity = 1.0 / (1.0 + dist)
                results.append((memory_id, similarity, metadata))
            
            return results
    
    def delete_vector(self, memory_id: str) -> bool:
        """删除向量"""
        if self.backend != StorageBackend.VECTOR_DB.value or memory_id not in self._vector_id_map:
            return False
        
        with self._vector_lock:
            vec_id = self._vector_id_map.pop(memory_id)
            self._vector_index.remove_ids(np.array([vec_id], dtype=np.int64))
            self.delete(f"vector_meta_{memory_id}")
            return True
    
    def _save_vector_index(self) -> None:
        """保存向量索引到磁盘"""
        if self.backend != StorageBackend.VECTOR_DB.value or self._vector_index is None:
            return
        
        vector_index_path = self.base_path / "vectors" / "index.faiss"
        id_map_path = self.base_path / "vectors" / "id_map.pkl"
        
        with self._vector_lock:
            faiss.write_index(self._vector_index, str(vector_index_path))
            with open(id_map_path, 'wb') as f:
                pickle.dump(self._vector_id_map, f)
    
    # ==================== 云存储方法 ====================
    
    def _init_cloud_client(self) -> None:
        """初始化云存储客户端"""
        if not BOTO3_AVAILABLE:
            raise RuntimeError("Boto3 is not installed, please install it with 'pip install boto3'")
        
        if self.cloud_provider == CloudStorageProvider.S3:
            self._cloud_client = boto3.client(
                's3',
                endpoint_url=self.cloud_endpoint,
                aws_access_key_id=self.cloud_access_key,
                aws_secret_access_key=self.cloud_secret_key,
            )
            # 确保桶存在
            try:
                self._cloud_client.head_bucket(Bucket=self.cloud_bucket)
            except ClientError:
                self._cloud_client.create_bucket(Bucket=self.cloud_bucket)
    
    def _start_cloud_sync(self) -> None:
        """启动云同步线程"""
        def cloud_sync_loop():
            while not self._stop_cloud_sync.wait(self.cloud_sync_interval):
                try:
                    self.sync_to_cloud()
                except Exception as e:
                    print(f"Cloud sync failed: {e}")
        
        self._cloud_sync_thread = threading.Thread(target=cloud_sync_loop, daemon=True)
        self._cloud_sync_thread.start()
    
    def stop_cloud_sync(self) -> None:
        """停止云同步"""
        self._stop_cloud_sync.set()
        if self._cloud_sync_thread:
            self._cloud_sync_thread.join(timeout=10.0)
    
    def sync_to_cloud(self) -> bool:
        """同步本地数据到云存储"""
        if not self._cloud_client or not self.cloud_bucket:
            return False
        
        try:
            # 先创建备份
            backup_path = self.create_backup()
            backup_name = Path(backup_path).name
            
            # 上传备份
            for f in Path(backup_path).iterdir():
                if f.is_file():
                    key = f"backups/{backup_name}/{f.name}"
                    self._cloud_client.upload_file(str(f), self.cloud_bucket, key)
            
            # 上传最新状态
            for f in self.base_path.iterdir():
                if f.is_file() and f.suffix in ['.json', '.faiss', '.pkl']:
                    key = f"latest/{f.name}"
                    self._cloud_client.upload_file(str(f), self.cloud_bucket, key)
            
            # 上传向量文件
            vector_dir = self.base_path / "vectors"
            if vector_dir.exists():
                for f in vector_dir.iterdir():
                    if f.is_file():
                        key = f"latest/vectors/{f.name}"
                        self._cloud_client.upload_file(str(f), self.cloud_bucket, key)
            
            return True
        except Exception as e:
            print(f"Sync to cloud failed: {e}")
            return False
    
    def restore_from_cloud(self, backup_name: Optional[str] = None) -> bool:
        """从云存储恢复"""
        if not self._cloud_client or not self.cloud_bucket:
            return False
        
        try:
            restore_path = self.base_path / "cloud_restore"
            restore_path.mkdir(parents=True, exist_ok=True)
            
            prefix = f"backups/{backup_name}/" if backup_name else "latest/"
            
            # 列出文件
            paginator = self._cloud_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.cloud_bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        local_path = restore_path / key[len(prefix):]
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        self._cloud_client.download_file(self.cloud_bucket, key, str(local_path))
            
            # 恢复数据
            self._cache.clear()
            self._dirty.clear()
            
            # 复制文件到主目录
            for f in restore_path.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.base_path / f.name)
            
            # 恢复向量文件
            vector_restore_dir = restore_path / "vectors"
            if vector_restore_dir.exists():
                vector_dir = self.base_path / "vectors"
                shutil.rmtree(vector_dir, ignore_errors=True)
                shutil.copytree(vector_restore_dir, vector_dir)
                # 重新加载向量索引
                if self.backend == StorageBackend.VECTOR_DB.value:
                    self._init_vector_index()
            
            return True
        except Exception as e:
            print(f"Restore from cloud failed: {e}")
            return False
    
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
            deleted = False
            if key in self._cache:
                del self._cache[key]
                deleted = True
            
            if key in self._dirty:
                self._dirty.discard(key)
                deleted = True
            
            # 删除文件
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                deleted = True
        
        return deleted
    
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
        
        # 保存向量索引
        if self.backend == StorageBackend.VECTOR_DB.value:
            self._save_vector_index()
        
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
        self.stop_cloud_sync()
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
