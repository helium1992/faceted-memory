"""时间自动衰减：绝对时间戳 → 相对时间标签

两层时间维度：
1. 事件发生时间（created_at）→ 自动计算相对标签
2. 信息中提到的时间（mentioned_time）→ 原文保留

相对标签规则（基于当前时间）：
- 0天: 今天
- 1天: 昨天
- 2天: 前天
- 3-6天: 前几天（重叠：也标记为本周）
- 7-13天: 上一周
- 14-29天: 上个月（近似）
- 30-89天: 前几个月
- 90-364天: 今年（如果同年）/ 去年
- 365-729天: 去年
- 730+天: 前年 / 很久以前

重叠窗口：边界日期同时标记多个标签，避免硬边界遗漏。
"""
import time
from typing import List
from datetime import datetime, timedelta


# 时间区间定义：(最小天数, 最大天数, 标签列表)
# 重叠窗口通过多个区间的交叉实现
TIME_BRACKETS = [
    (0, 0, ["今天"]),
    (1, 1, ["昨天"]),
    (2, 2, ["前天"]),
    (3, 6, ["前几天"]),
    (5, 8, ["这周", "上一周"]),       # 重叠窗口
    (7, 13, ["上一周"]),
    (12, 15, ["上一周", "上个月"]),    # 重叠窗口
    (14, 29, ["上个月"]),
    (28, 32, ["上个月", "前几个月"]),  # 重叠窗口
    (30, 89, ["前几个月"]),
    (90, 364, ["今年"]),
    (365, 729, ["去年"]),
    (730, 99999, ["前年", "很久以前"]),
]


class TimeDecay:
    """时间自动衰减管理器"""

    def __init__(self, custom_brackets: List = None):
        self.brackets = custom_brackets or TIME_BRACKETS

    def get_relative_labels(self, created_at: float,
                            now: float = None) -> List[str]:
        """根据绝对时间戳计算相对时间标签

        Args:
            created_at: 记忆创建时间戳
            now: 当前时间戳（默认time.time()）

        Returns:
            相对时间标签列表（可能有多个，因为重叠窗口）
        """
        if now is None:
            now = time.time()

        days_ago = (now - created_at) / 86400.0

        labels = set()
        for min_days, max_days, bracket_labels in self.brackets:
            if min_days <= days_ago <= max_days:
                labels.update(bracket_labels)

        # 补充年份信息
        created_dt = datetime.fromtimestamp(created_at)
        now_dt = datetime.fromtimestamp(now)
        if created_dt.year == now_dt.year:
            labels.add("今年")
        elif created_dt.year == now_dt.year - 1:
            labels.add("去年")

        return sorted(labels) if labels else ["很久以前"]

    def get_all_time_terms(self) -> List[str]:
        """获取所有可能的时间标签（用于构建When维度词库）"""
        terms = set()
        for _, _, bracket_labels in self.brackets:
            terms.update(bracket_labels)
        terms.update(["今年", "去年"])
        return sorted(terms)

    def enrich_when_dimension(self, created_at: float,
                              mentioned_time_terms: List[str] = None,
                              now: float = None) -> List[str]:
        """合并两层时间维度的词条

        Args:
            created_at: 事件发生时间
            mentioned_time_terms: 信息中提到的时间词条
            now: 当前时间

        Returns:
            合并后的When维度词条列表
        """
        labels = self.get_relative_labels(created_at, now)
        if mentioned_time_terms:
            labels = list(set(labels) | set(mentioned_time_terms))
        return labels
