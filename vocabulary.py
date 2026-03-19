"""垂直词库管理 + 双向最大匹配

每个维度(Dimension)拥有独立词库，互不干扰。
匹配采用正向最大匹配，优先匹配最长词。
"""
import os
import json
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


class Dimension(str, Enum):
    """检索维度 — 基于词性+语义细分的17维度分类"""
    # 名词类 (8)
    NOUN_PERSON = "noun_person"      # 人物名词：家人/朋友/职业/角色
    NOUN_PLACE = "noun_place"        # 地点名词：城市/场所/自然地理
    NOUN_ORG = "noun_org"            # 组织机构：公司/品牌/机构/学校
    NOUN_OBJECT = "noun_object"      # 物品名词：食物/电子/交通/服装
    NOUN_TIME = "noun_time"          # 时间名词：节日/季节/时段/周期
    NOUN_CONCEPT = "noun_concept"    # 概念名词：技术/学科/抽象概念
    NOUN_EVENT = "noun_event"        # 事件名词：考试/活动/比赛
    NOUN_PROJECT = "noun_project"    # 项目专有：类名/文件名/模块名
    # 动词类 (6)
    VERB_DAILY = "verb_daily"        # 日常动词：饮食/起居/出行/家务
    VERB_SOCIAL = "verb_social"      # 社交动词：沟通/关系/礼仪
    VERB_WORK = "verb_work"          # 工作动词：职场/求职/商务
    VERB_TECH = "verb_tech"          # 技术动词：开发/系统/数据操作
    VERB_CONSUME = "verb_consume"    # 消费动词：购买/支付/理财
    VERB_COGNITION = "verb_cognition"  # 认知动词：思考/学习/表达
    # 形容词类 (3)
    ADJ_EMOTION = "adj_emotion"      # 情感形容词：正面/负面/中性情感
    ADJ_EVAL = "adj_eval"            # 评价形容词：品质/价值/难度
    ADJ_STATE = "adj_state"          # 状态形容词：身体/忙闲/天气/进度


@dataclass
class MatchResult:
    """单次匹配结果"""
    dimension: Dimension
    term: str           # 匹配到的词
    start: int          # 在原文中的起始位置
    end: int            # 在原文中的结束位置


@dataclass
class DimensionVocab:
    """单个维度的词库"""
    dimension: Dimension
    terms: Set[str] = field(default_factory=set)
    aliases: Dict[str, str] = field(default_factory=dict)  # 别名 → 标准词
    max_term_len: int = 0

    def add(self, term: str, aliases: List[str] = None):
        """添加词条（可附带别名）"""
        self.terms.add(term)
        self.max_term_len = max(self.max_term_len, len(term))
        if aliases:
            for alias in aliases:
                self.aliases[alias] = term
                self.terms.add(alias)
                self.max_term_len = max(self.max_term_len, len(alias))

    def normalize(self, term: str) -> str:
        """将别名转换为标准词"""
        return self.aliases.get(term, term)


class VocabularyManager:
    """多维度词库管理器"""

    def __init__(self):
        self._vocabs: Dict[Dimension, DimensionVocab] = {}
        for dim in Dimension:
            self._vocabs[dim] = DimensionVocab(dimension=dim)

    def get_vocab(self, dim: Dimension) -> DimensionVocab:
        return self._vocabs[dim]

    def add_term(self, dim: Dimension, term: str, aliases: List[str] = None):
        """向指定维度添加词条"""
        self._vocabs[dim].add(term, aliases)

    def add_terms(self, dim: Dimension, terms: List[str]):
        """批量添加词条"""
        for t in terms:
            self._vocabs[dim].add(t)

    # ==================== 双向最大匹配 ====================

    def match(self, text: str) -> List[MatchResult]:
        """对输入文本进行多维度正向最大匹配

        优先匹配更长的词。跨维度时，长词优先。
        """
        results = []
        i = 0
        text_len = len(text)

        while i < text_len:
            best_match: Optional[MatchResult] = None
            best_len = 0

            # 在所有维度中寻找从位置i开始的最长匹配
            for dim, vocab in self._vocabs.items():
                if vocab.max_term_len == 0:
                    continue
                # 从最长到最短尝试
                max_len = min(vocab.max_term_len, text_len - i)
                for length in range(max_len, 0, -1):
                    candidate = text[i:i + length]
                    if candidate in vocab.terms:
                        if length > best_len:
                            best_len = length
                            best_match = MatchResult(
                                dimension=dim,
                                term=vocab.normalize(candidate),
                                start=i,
                                end=i + length,
                            )
                        break  # 这个维度找到最长的了，继续下一个维度

            if best_match:
                results.append(best_match)
                i += best_len  # 跳过已匹配的部分
            else:
                i += 1  # 未匹配，前进一个字符

        return results

    def extract_dimensions(self, text: str) -> Dict[Dimension, List[str]]:
        """提取文本中各维度的词条（去重，返回标准词）"""
        matches = self.match(text)
        dim_terms: Dict[Dimension, List[str]] = {d: [] for d in Dimension}
        seen: Dict[Dimension, Set[str]] = {d: set() for d in Dimension}

        for m in matches:
            if m.term not in seen[m.dimension]:
                seen[m.dimension].add(m.term)
                dim_terms[m.dimension].append(m.term)

        return dim_terms

    # ==================== 持久化 ====================

    def save(self, directory: str):
        """保存词库到目录"""
        os.makedirs(directory, exist_ok=True)
        for dim, vocab in self._vocabs.items():
            data = {
                "dimension": dim.value,
                "terms": sorted(vocab.terms - set(vocab.aliases.keys())),
                "aliases": vocab.aliases,
            }
            path = os.path.join(directory, f"{dim.value}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, directory: str):
        """从目录加载词库"""
        for dim in Dimension:
            path = os.path.join(directory, f"{dim.value}.json")
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            vocab = self._vocabs[dim]
            for term in data.get("terms", []):
                vocab.add(term)
            for alias, standard in data.get("aliases", {}).items():
                vocab.add(standard, aliases=[alias])

    def stats(self) -> Dict[str, int]:
        """各维度词条数统计"""
        return {dim.value: len(vocab.terms) for dim, vocab in self._vocabs.items()}
