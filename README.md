# 17维分面记忆持久化系统

## 概述

零推理成本的本地记忆检索系统。AI 离线建库 → 纯算法在线匹配 → 无需调用任何 LLM API。

通过 17 个语义维度（人物、地点、时间、事件、情感等）对记忆进行分面索引，实现高精度、低延迟的记忆检索。

## 架构

```
消息输入 → 双向最大匹配分词 → 17维向量化 → 多向量余弦相似 → 时间衰减加权 → Top-K 结果
```

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| **词库管理** | `vocabulary.py` | 17维垂直词库 + 双向最大匹配分词 |
| **向量引擎** | `engine.py` | 分面多向量比对（CharNGram + 余弦相似） |
| **两级存储** | `storage.py` | SQLite：索引表（轻量）+ 详情表（按需加载） |
| **时间衰减** | `time_decay.py` | 指数衰减，让近期记忆权重更高 |
| **统一检索** | `retriever.py` | 整合所有模块的统一 API 入口 |
| **中间件** | `middleware.py` | 通信管道中间层：自动检索+注入+存储 |
| **词库生成** | `gen_vocab.py` | AI 一次性生成垂直词库（离线） |

### 17 个语义维度

```
person      — 人物/角色
location    — 地点/位置
time        — 时间/日期
event       — 事件/活动
emotion     — 情感/态度
object      — 物品/对象
action      — 动作/操作
topic       — 话题/主题
relation    — 关系/关联
quantity    — 数量/度量
attribute   — 属性/特征
cause       — 原因/因果
method      — 方法/方式
status      — 状态/进度
domain      — 领域/学科
goal        — 目标/意图
constraint  — 约束/条件
```

---

## 快速接入（新项目）

### 第 1 步：复制模块

将整个 `faceted_memory/` 文件夹复制到你的项目中。

### 第 2 步：安装依赖

```bash
pip install numpy
```

仅依赖 numpy + Python 标准库（sqlite3、json、os、time、uuid 等）。

### 第 3 步：生成词库（首次）

```python
# 让 AI 调用 gen_vocab.py 生成 17 维词库
# 或手动创建词库目录
python -m faceted_memory.gen_vocab --output data/vocab/
```

词库文件结构：
```
data/vocab/
├── person.json       # {"terms": ["张三", "李四", ...]}
├── location.json
├── time.json
├── event.json
├── ... (17个维度各一个文件)
```

### 第 4 步：使用

#### 方式 A：中间件模式（推荐）

最简单的接入方式，适合通信管道/聊天系统：

```python
from faceted_memory.middleware import MemoryMiddleware

mw = MemoryMiddleware(
    db_path="data/memory.db",
    vocab_dir="data/vocab/",
    top_k=3,
    min_score=0.1,
)

# 收到用户消息 → 自动检索相关记忆 → 注入到消息中
enriched_message = mw.on_incoming("张三的桥梁优化进展如何")
# enriched_message = 原始消息 + Top3相关记忆上下文

# AI 回复后 → 自动提取关键词 → 写入记忆库
mw.on_outgoing("张三完成了拱桥优化，承重效率2.17N/g")
```

#### 方式 B：检索器模式

更灵活的控制：

```python
from faceted_memory import FacetedRetriever

retriever = FacetedRetriever(
    db_path="data/memory.db",
    vocab_dir="data/vocab/",
)

# 存入记忆
retriever.add_memory(
    memory_id="m1",
    summary="张三完成拱桥优化",
    content="张三完成了拱桥优化，承重效率2.17N/g，使用MILP求解器",
    raw_text="张三完成了拱桥优化，承重效率2.17N/g",
)

# 检索记忆
results = retriever.search("张三的桥梁进展", top_k=3)
for r in results:
    print(f"{r.summary} (score={r.score:.2f})")
```

#### 方式 C：与 IDE Claw 集成

在 `dialog.py` 或 MCP Server 中使用（已内置支持）：

```python
from faceted_memory.middleware import MemoryMiddleware

mw = MemoryMiddleware(db_path="data/memory.db", vocab_dir="data/vocab/")

# 用户发消息时，自动检索相关记忆并注入
user_msg = "之前讨论的优化方案怎么样了"
enriched = mw.on_incoming(user_msg)
# enriched 包含原始消息 + 相关历史记忆

# AI 回复时，自动存储到记忆库
ai_reply = "上次讨论的MILP优化方案已经实现..."
mw.on_outgoing(ai_reply)
```

---

## API 参考

### MemoryMiddleware

```python
mw = MemoryMiddleware(
    db_path: str,          # SQLite 数据库路径
    vocab_dir: str = None, # 词库目录（None 用默认分词）
    top_k: int = 3,        # 返回前 K 条结果
    min_score: float = 0.1, # 最低匹配分数
    auto_store: bool = True, # 是否自动存储消息
)

mw.on_incoming(message, sender="user") -> str   # 检索+注入
mw.on_outgoing(message, sender="ai") -> str     # 提取+存储
mw.search_only(message) -> List[SearchResult]   # 只检索不注入
```

### FacetedRetriever

```python
retriever = FacetedRetriever(
    db_path: str,
    vocab_dir: str = None,
)

retriever.add_memory(memory_id, summary, content, raw_text) -> None
retriever.search(query, top_k=5) -> List[SearchResult]
```

### SearchResult

```python
@dataclass
class SearchResult:
    memory_id: str
    summary: str
    score: float           # 综合得分 (0~1)
    dim_scores: Dict[str, float]  # 每个维度的得分
    active_dims: List[str]        # 激活的维度
    masked_dims: List[str]        # 被屏蔽的维度
    content: Optional[str]        # 详细内容（按需加载）
```

---

## 数据存储

使用 SQLite 两级存储：

- **索引表** `memory_index`：轻量级，包含 memory_id、summary、17维向量、时间戳
- **详情表** `memory_detail`：完整内容，仅在需要时按 ID 加载

数据库文件默认位于 `data/memory.db`。

---

## 文件结构

```
faceted_memory/
├── README.md           ← 本文件
├── __init__.py         ← 模块入口
├── vocabulary.py       ← 17维词库管理 + 双向最大匹配
├── engine.py           ← 分面多向量比对引擎
├── storage.py          ← SQLite 两级存储
├── time_decay.py       ← 时间衰减
├── retriever.py        ← 统一检索入口
├── middleware.py        ← 通信中间件
├── gen_vocab.py        ← 词库生成工具
├── check_db.py         ← 数据库检查工具
├── THIRD_PARTY_LICENSES.md  ← 第三方许可证
└── test_*.py           ← 测试文件
```

---

## Acknowledgments

This project uses the following open-source library:

- **[NumPy](https://numpy.org/)** — BSD-3-Clause License

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full details.
