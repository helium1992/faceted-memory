"""记忆中间件：自动提取 + 存储 + 检索

作为通信管道的中间层，对消息进行透明处理：
- 收到消息 → 自动检索相关记忆 → 注入到消息中
- 发出消息 → 自动提取关键词 → 写入记忆库

使用方式：
    mw = MemoryMiddleware(db_path="data/memory.db", vocab_dir="data/vocab/")
    enriched = mw.on_incoming("张三的桥梁优化进展如何")
    # enriched = 原始消息 + Top3相关记忆
    mw.on_outgoing("张三完成了拱桥优化，承重效率2.17N/g")
    # 自动写入记忆库
"""
import os
import time
import uuid
import json
from typing import List, Optional, Dict, Tuple

from .retriever import FacetedRetriever, SearchResult
from .vocabulary import VocabularyManager, Dimension
from .engine import Embedder


class MemoryMiddleware:
    """记忆中间件"""

    def __init__(self, db_path: str, vocab_dir: str = None,
                 embedder: Embedder = None,
                 top_k: int = 3,
                 min_score: float = 0.1,
                 auto_store: bool = True):
        """
        Args:
            db_path: 记忆数据库路径
            vocab_dir: 词库目录
            embedder: 向量化器（None用默认）
            top_k: 检索返回条数
            min_score: 最低匹配分数（低于此分数不注入）
            auto_store: 是否自动存储消息
        """
        self.retriever = FacetedRetriever(
            db_path=db_path,
            vocab_dir=vocab_dir,
            embedder=embedder,
        )
        self.top_k = top_k
        self.min_score = min_score
        self.auto_store = auto_store

    # ==================== 入站处理 ====================

    def on_incoming(self, message: str, sender: str = "user") -> str:
        """处理入站消息：检索相关记忆并注入

        Args:
            message: 用户发来的原始消息
            sender: 发送者标识

        Returns:
            增强后的消息（原始消息 + 相关记忆上下文）
        """
        # 1. 检索相关记忆
        results = self.retriever.search(message, top_k=self.top_k)
        relevant = [r for r in results if r.score >= self.min_score]

        if not relevant:
            return message

        # 2. 拉取详情
        for r in relevant:
            entry = self.retriever.store.get_detail(r.memory_id)
            if entry:
                r.content = entry.content

        # 3. 构建增强消息
        memory_block = self._format_memory_block(relevant)
        enriched = f"{message}\n\n{memory_block}"

        return enriched

    def search_only(self, message: str) -> List[SearchResult]:
        """只检索不注入，返回原始结果"""
        results = self.retriever.search(message, top_k=self.top_k)
        return [r for r in results if r.score >= self.min_score]

    # ==================== 出站处理 ====================

    def on_outgoing(self, message: str, sender: str = "ai",
                    summary: str = None,
                    metadata: dict = None) -> str:
        """处理出站消息：自动提取关键词并存入记忆库

        Args:
            message: AI发出的消息
            sender: 发送者标识
            summary: 可选的摘要（默认取前100字符）
            metadata: 额外元数据

        Returns:
            memory_id（新创建的记忆ID）
        """
        if not self.auto_store:
            return ""

        # 检查是否有足够的词库匹配（至少匹配到1个维度的词条）
        dim_terms = self.retriever.vocab.extract_dimensions(message)
        active_dims = {d.value: v for d, v in dim_terms.items() if v}

        if not active_dims:
            return ""  # 没有匹配到任何维度词条，不存储

        # 生成摘要
        if not summary:
            summary = message[:150].replace('\n', ' ').strip()
            if len(message) > 150:
                summary += "..."

        # 存储
        mem_id = self.retriever.add_memory(
            summary=summary,
            content=message,
            metadata={
                "sender": sender,
                "matched_dims": active_dims,
                **(metadata or {}),
            }
        )
        return mem_id

    # ==================== 手动操作 ====================

    def store(self, content: str, summary: str = "",
              metadata: dict = None) -> str:
        """手动存入一条记忆"""
        if not summary:
            summary = content[:150].replace('\n', ' ').strip()
        return self.retriever.add_memory(
            summary=summary,
            content=content,
            metadata=metadata or {},
        )

    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """手动检索"""
        return self.retriever.search(query, top_k=top_k or self.top_k)

    # ==================== 内部方法 ====================

    def _format_memory_block(self, results: List[SearchResult]) -> str:
        """格式化记忆块用于注入消息"""
        lines = ["---", "📚 相关记忆（自动检索）:"]
        for i, r in enumerate(results, 1):
            score_pct = int(r.score * 100)
            lines.append(f"[{i}] ({score_pct}%匹配) {r.summary}")
            if r.content and r.content != r.summary:
                # 截取详情前200字符
                detail = r.content[:200].replace('\n', ' ')
                if len(r.content) > 200:
                    detail += "..."
                lines.append(f"    → {detail}")
        lines.append("---")
        return "\n".join(lines)

    # ==================== 工具方法 ====================

    @property
    def vocab(self) -> VocabularyManager:
        return self.retriever.vocab

    def stats(self) -> dict:
        return self.retriever.stats()

    def close(self):
        self.retriever.close()
