"""统一检索入口

将词库匹配、向量比对、时间衰减、消歧交互整合为一个简洁的API。

使用流程：
    retriever = FacetedRetriever(db_path="memories.db", vocab_dir="vocab/")
    retriever.add_memory("m1", summary="...", content="...", raw_text="...")
    results = retriever.search("张三昨天说的那件事", top_k=3)
"""
import uuid
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from .vocabulary import VocabularyManager, Dimension
from .engine import FacetedEngine, Embedder, CharNGramEmbedder, ScoredResult
from .storage import MemoryStore, MemoryEntry
from .time_decay import TimeDecay


@dataclass
class SearchResult:
    """最终检索结果"""
    memory_id: str
    summary: str
    score: float
    dim_scores: Dict[str, float]
    active_dims: List[str]
    masked_dims: List[str]
    content: Optional[str] = None  # 仅在拉取详情后填充


class FacetedRetriever:
    """分面多向量记忆检索器"""

    def __init__(self, db_path: str = ":memory:",
                 vocab_dir: str = None,
                 embedder: Embedder = None,
                 confidence_gap: float = 0.15):
        """
        Args:
            db_path: SQLite数据库路径（":memory:"为内存模式）
            vocab_dir: 词库目录（None则需手动添加词条）
            embedder: 向量化器（None则使用默认CharNGram）
            confidence_gap: 自信跳过阈值（Top1与Top2分差超过此值则跳过消歧）
        """
        self.vocab = VocabularyManager()
        self.engine = FacetedEngine(embedder=embedder)
        self.store = MemoryStore(db_path=db_path)
        self.time_decay = TimeDecay()
        self.confidence_gap = confidence_gap

        if vocab_dir:
            self.vocab.load(vocab_dir)

        # 将时间标签自动加入NOUN_TIME维度词库
        for label in self.time_decay.get_all_time_terms():
            self.vocab.add_term(Dimension.NOUN_TIME, label)

    # ==================== 写入 ====================

    def add_memory(self, memory_id: str = None,
                   summary: str = "",
                   content: str = "",
                   raw_text: str = None,
                   mentioned_time: str = "",
                   created_at: float = None,
                   metadata: dict = None) -> str:
        """添加一条记忆

        Args:
            memory_id: 唯一ID（None则自动生成）
            summary: 一句话摘要
            content: 完整长文本
            raw_text: 用于词库匹配的原文（默认=summary+content）
            mentioned_time: 信息中提到的时间（原文）
            created_at: 时间戳（默认=now）
            metadata: 额外元数据

        Returns:
            memory_id
        """
        if not memory_id:
            memory_id = uuid.uuid4().hex[:12]
        if not created_at:
            created_at = time.time()

        # 从原文提取各维度词条
        text_for_match = raw_text or (summary + " " + content)
        dim_terms = self.vocab.extract_dimensions(text_for_match)

        # 合并时间维度（自动衰减标签 + 文本中提到的时间）
        mentioned_terms = dim_terms.get(Dimension.NOUN_TIME, [])
        when_terms = self.time_decay.enrich_when_dimension(
            created_at, mentioned_terms)
        dim_terms[Dimension.NOUN_TIME] = when_terms

        # 向量化（embed_terms接受Dimension键，返回str键）
        dim_vectors = self.engine.embed_terms(dim_terms)

        # 转为字符串键
        str_terms = {d.value: v for d, v in dim_terms.items() if v}
        str_vectors = dim_vectors  # already str keys from embed_terms

        entry = MemoryEntry(
            id=memory_id,
            summary=summary,
            content=content,
            created_at=created_at,
            mentioned_time=mentioned_time,
            dim_vectors=str_vectors,
            dim_terms=str_terms,
            metadata=metadata or {},
        )
        self.store.add(entry)
        return memory_id

    # ==================== 检索 ====================

    def search(self, query: str, top_k: int = 5,
               now: float = None) -> List[SearchResult]:
        """检索记忆

        Args:
            query: 用户查询文本
            top_k: 返回前K条
            now: 当前时间（默认=time.time()，用于时间衰减）

        Returns:
            按相关度排序的结果列表
        """
        if now is None:
            now = time.time()

        # 1. 词库匹配：提取查询中各维度词条
        query_dim_terms = self.vocab.extract_dimensions(query)

        # 2. 向量化查询
        query_vectors = self.engine.embed_terms(query_dim_terms)

        # 3. 获取所有记忆索引（含预存向量）
        all_index = self.store.get_all_index()

        # 4. 更新记忆的When维度向量（基于当前时间重新计算相对标签）
        for mem_idx in all_index:
            created_at = mem_idx.get('_created_at', 0)
            when_labels = self.time_decay.get_relative_labels(created_at, now)
            if when_labels:
                mem_idx['noun_time'] = self.engine.embedder.embed(when_labels)

        # 5. 排序
        scored = self.engine.rank(query_vectors, all_index, top_k=top_k)

        # 6. 组装结果
        results = []
        for s in scored:
            mem_idx = next((m for m in all_index if m.get('_id') == s.memory_id), {})
            results.append(SearchResult(
                memory_id=s.memory_id,
                summary=mem_idx.get('_summary', ''),
                score=s.total_score,
                dim_scores=s.dim_scores,
                active_dims=s.active_dims,
                masked_dims=s.masked_dims,
            ))

        return results

    def search_with_detail(self, query: str, top_k: int = 3,
                           now: float = None) -> List[SearchResult]:
        """检索并拉取详情"""
        results = self.search(query, top_k=top_k, now=now)
        for r in results:
            entry = self.store.get_detail(r.memory_id)
            if entry:
                r.content = entry.content
        return results

    # ==================== 消歧 ====================

    def disambiguate(self, results: List[SearchResult],
                     callback: Callable[[List[str]], Optional[int]] = None
                     ) -> Optional[SearchResult]:
        """交互式消歧

        Args:
            results: search()返回的结果列表
            callback: 用户选择回调 fn(summaries) -> 选中的索引(0-based)，None表示都不是

        Returns:
            用户确认的记忆，或None
        """
        if not results:
            return None

        # 自信跳过：Top1远超Top2
        if len(results) == 1 or (
            len(results) >= 2 and
            results[0].score - results[1].score >= self.confidence_gap
        ):
            return results[0]

        # 需要消歧
        if callback is None:
            return results[0]  # 无回调时返回Top1

        # 游标翻页：每次展示3条
        page_size = 3
        cursor = 0
        while cursor < len(results):
            page = results[cursor:cursor + page_size]
            summaries = [f"[{i+1}] (得分{r.score:.2f}) {r.summary}"
                         for i, r in enumerate(page)]
            choice = callback(summaries)
            if choice is not None and 0 <= choice < len(page):
                selected = page[choice]
                selected.content = self.store.get_detail(selected.memory_id).content
                return selected
            cursor += page_size

        return None  # 所有页都被否决

    # ==================== 工具方法 ====================

    def stats(self) -> dict:
        return {
            "memory_count": self.store.count(),
            "vocab_stats": self.vocab.stats(),
        }

    def close(self):
        self.store.close()
