"""两级存储：索引层 + 详情层

索引层：各维度向量 + 摘要（用于快速算分）
详情层：完整原始文本（按需拉取）

使用 SQLite 持久化，支持增量插入。
"""
import os
import json
import sqlite3
import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .vocabulary import Dimension


def _vec_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _bytes_to_vec(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


@dataclass
class MemoryEntry:
    """一条记忆"""
    id: str
    summary: str                          # 一句话摘要（用于消歧展示）
    content: str                          # 完整长文本（详情层）
    created_at: float = 0.0               # 绝对时间戳
    mentioned_time: str = ""              # 信息中提到的时间（原文）
    dim_vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    dim_terms: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """两级记忆存储"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                mentioned_time TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS dim_vectors (
                memory_id TEXT NOT NULL,
                dimension TEXT NOT NULL,
                vector BLOB NOT NULL,
                terms TEXT DEFAULT '[]',
                PRIMARY KEY (memory_id, dimension),
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at);
        """)
        self._conn.commit()

    # ==================== 写入 ====================

    def add(self, entry: MemoryEntry):
        """添加一条记忆"""
        if not entry.created_at:
            entry.created_at = time.time()

        self._conn.execute(
            "INSERT OR REPLACE INTO memories (id, summary, content, created_at, mentioned_time, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry.id, entry.summary, entry.content, entry.created_at,
             entry.mentioned_time, json.dumps(entry.metadata, ensure_ascii=False))
        )

        for dim_key, vec in entry.dim_vectors.items():
            terms = entry.dim_terms.get(dim_key, [])
            self._conn.execute(
                "INSERT OR REPLACE INTO dim_vectors (memory_id, dimension, vector, terms) "
                "VALUES (?, ?, ?, ?)",
                (entry.id, dim_key, _vec_to_bytes(vec),
                 json.dumps(terms, ensure_ascii=False))
            )
        self._conn.commit()

    def batch_add(self, entries: List[MemoryEntry]):
        """批量添加"""
        for e in entries:
            self.add(e)

    # ==================== 索引层读取 ====================

    def get_all_index(self) -> List[Dict[str, Any]]:
        """获取所有记忆的索引数据（向量+摘要，不含详情）"""
        rows = self._conn.execute(
            "SELECT id, summary, created_at, mentioned_time FROM memories"
        ).fetchall()

        result = []
        for mem_id, summary, created_at, mentioned_time in rows:
            vectors = {}
            dim_rows = self._conn.execute(
                "SELECT dimension, vector FROM dim_vectors WHERE memory_id = ?",
                (mem_id,)
            ).fetchall()
            for dim_key, vec_bytes in dim_rows:
                vectors[dim_key] = _bytes_to_vec(vec_bytes)
            vectors['_id'] = mem_id
            vectors['_summary'] = summary
            vectors['_created_at'] = created_at
            vectors['_mentioned_time'] = mentioned_time
            result.append(vectors)

        return result

    # ==================== 详情层读取 ====================

    def get_detail(self, memory_id: str) -> Optional[MemoryEntry]:
        """按ID获取完整记忆（包含详情）"""
        row = self._conn.execute(
            "SELECT id, summary, content, created_at, mentioned_time, metadata "
            "FROM memories WHERE id = ?",
            (memory_id,)
        ).fetchone()

        if not row:
            return None

        mem_id, summary, content, created_at, mentioned_time, meta_str = row
        entry = MemoryEntry(
            id=mem_id, summary=summary, content=content,
            created_at=created_at, mentioned_time=mentioned_time,
            metadata=json.loads(meta_str),
        )

        # 加载向量和词条
        dim_rows = self._conn.execute(
            "SELECT dimension, vector, terms FROM dim_vectors WHERE memory_id = ?",
            (mem_id,)
        ).fetchall()
        for dim_key, vec_bytes, terms_str in dim_rows:
            entry.dim_vectors[dim_key] = _bytes_to_vec(vec_bytes)
            entry.dim_terms[dim_key] = json.loads(terms_str)

        return entry

    def get_details(self, memory_ids: List[str]) -> List[MemoryEntry]:
        """批量获取完整记忆"""
        return [e for mid in memory_ids if (e := self.get_detail(mid)) is not None]

    # ==================== 删除 ====================

    def delete(self, memory_id: str):
        self._conn.execute("DELETE FROM dim_vectors WHERE memory_id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    # ==================== 统计 ====================

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self):
        self._conn.close()
