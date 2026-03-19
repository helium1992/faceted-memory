"""分面多向量记忆检索系统

AI离线建库 → 纯算法在线匹配 → 零推理成本

核心模块：
- vocabulary: 垂直词库管理 + 双向最大匹配
- engine: 分面多向量比对引擎
- storage: 两级存储（索引+详情）
- time_decay: 时间自动衰减
- retriever: 统一检索入口
"""

from .retriever import FacetedRetriever
from .vocabulary import VocabularyManager, Dimension
from .storage import MemoryStore, MemoryEntry
from .engine import FacetedEngine
from .time_decay import TimeDecay
from .middleware import MemoryMiddleware

__all__ = [
    'FacetedRetriever',
    'VocabularyManager', 'Dimension',
    'MemoryStore', 'MemoryEntry',
    'FacetedEngine',
    'TimeDecay',
    'MemoryMiddleware',
]
