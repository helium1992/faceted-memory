"""分面多向量比对引擎

核心逻辑：
1. 维度对齐：只比较同维度的向量
2. 动态遮蔽：查询中缺失的维度自动跳过
3. 加权打分：各维度可配置权重
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .vocabulary import Dimension


# 默认维度权重（17维度）
DEFAULT_WEIGHTS: Dict[Dimension, float] = {
    # 名词类 — 较高权重（实体信息最具区分度）
    Dimension.NOUN_PERSON: 0.12,
    Dimension.NOUN_PLACE: 0.08,
    Dimension.NOUN_ORG: 0.08,
    Dimension.NOUN_OBJECT: 0.07,
    Dimension.NOUN_TIME: 0.05,
    Dimension.NOUN_CONCEPT: 0.08,
    Dimension.NOUN_EVENT: 0.07,
    Dimension.NOUN_PROJECT: 0.10,
    # 动词类 — 中等权重（行为信息）
    Dimension.VERB_DAILY: 0.05,
    Dimension.VERB_SOCIAL: 0.05,
    Dimension.VERB_WORK: 0.05,
    Dimension.VERB_TECH: 0.06,
    Dimension.VERB_CONSUME: 0.04,
    Dimension.VERB_COGNITION: 0.04,
    # 形容词类 — 较低权重（修饰性信息）
    Dimension.ADJ_EMOTION: 0.03,
    Dimension.ADJ_EVAL: 0.02,
    Dimension.ADJ_STATE: 0.01,
}


@dataclass
class ScoredResult:
    """带分数的检索结果"""
    memory_id: str
    total_score: float
    dim_scores: Dict[str, float]   # 各维度得分明细
    active_dims: List[str]         # 参与计算的维度
    masked_dims: List[str]         # 被遮蔽的维度


class Embedder:
    """向量化接口（抽象基类）

    子类需实现 embed() 方法。
    默认提供基于字符重叠的简单实现用于测试。
    """

    def embed(self, terms: List[str]) -> np.ndarray:
        """将一组词条转换为单个向量

        Args:
            terms: 同一维度的词条列表
        Returns:
            归一化后的向量 (1D numpy array)
        """
        raise NotImplementedError

    def vector_dim(self) -> int:
        """返回向量维度"""
        raise NotImplementedError


class CharNGramEmbedder(Embedder):
    """基于字符n-gram的简易向量化器（无需外部模型）

    将词条拆成字符级n-gram，用哈希映射到固定维度向量。
    适合demo和测试，生产环境建议替换为sentence-transformers等。
    """

    def __init__(self, dim: int = 256, ngram_range: Tuple[int, int] = (1, 3)):
        self._dim = dim
        self._ngram_range = ngram_range

    def vector_dim(self) -> int:
        return self._dim

    def embed(self, terms: List[str]) -> np.ndarray:
        if not terms:
            return np.zeros(self._dim)

        vec = np.zeros(self._dim)
        for term in terms:
            for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
                for i in range(len(term) - n + 1):
                    ngram = term[i:i + n]
                    idx = hash(ngram) % self._dim
                    vec[idx] += 1.0

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class FacetedEngine:
    """分面多向量比对引擎"""

    def __init__(self, embedder: Optional[Embedder] = None,
                 weights: Optional[Dict[Dimension, float]] = None):
        self.embedder = embedder or CharNGramEmbedder()
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def set_weight(self, dim: Dimension, weight: float):
        self.weights[dim] = weight

    def embed_terms(self, dim_terms: Dict[Dimension, List[str]]) -> Dict[str, np.ndarray]:
        """将各维度词条转换为向量字典"""
        vectors = {}
        for dim in Dimension:
            terms = dim_terms.get(dim, [])
            if terms:
                vectors[dim.value] = self.embedder.embed(terms)
            # 没有词条的维度不生成向量（查询时会被mask）
        return vectors

    def score(self, query_vectors: Dict[str, np.ndarray],
              memory_vectors: Dict[str, np.ndarray]) -> ScoredResult:
        """计算查询与单条记忆的多维度加权得分

        Args:
            query_vectors: 查询各维度向量
            memory_vectors: 记忆各维度向量（包含memory_id键）

        Returns:
            ScoredResult
        """
        memory_id = memory_vectors.get('_id', 'unknown')
        dim_scores = {}
        active_dims = []
        masked_dims = []

        for dim in Dimension:
            dv = dim.value
            q_vec = query_vectors.get(dv)
            m_vec = memory_vectors.get(dv)

            if q_vec is None:
                # 查询中没有这个维度 → 遮蔽
                masked_dims.append(dv)
                continue

            if m_vec is None:
                # 记忆中没有这个维度 → 该维度得0分但参与计算
                dim_scores[dv] = 0.0
                active_dims.append(dv)
                continue

            # 余弦相似度（向量已归一化）
            sim = float(np.dot(q_vec, m_vec))
            sim = max(0.0, sim)  # 截断负值
            dim_scores[dv] = sim
            active_dims.append(dv)

        # 加权汇总（只对活跃维度加权）
        if not active_dims:
            return ScoredResult(
                memory_id=memory_id, total_score=0.0,
                dim_scores=dim_scores, active_dims=active_dims,
                masked_dims=masked_dims
            )

        total_weight = sum(self.weights.get(Dimension(d), 0.1) for d in active_dims)
        weighted_sum = sum(
            dim_scores[d] * self.weights.get(Dimension(d), 0.1)
            for d in active_dims
        )
        total_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return ScoredResult(
            memory_id=memory_id,
            total_score=total_score,
            dim_scores=dim_scores,
            active_dims=active_dims,
            masked_dims=masked_dims,
        )

    def rank(self, query_vectors: Dict[str, np.ndarray],
             memory_list: List[Dict[str, np.ndarray]],
             top_k: int = 10) -> List[ScoredResult]:
        """对多条记忆排序，返回Top-K"""
        results = [self.score(query_vectors, m) for m in memory_list]
        results.sort(key=lambda r: r.total_score, reverse=True)
        return results[:top_k]
