"""测试日常中文对话的记忆检索效果"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceted_memory.middleware import MemoryMiddleware

VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vocab')


def test_daily():
    mw = MemoryMiddleware(db_path=":memory:", vocab_dir=VOCAB_DIR, top_k=3, min_score=0.05)

    s = mw.stats()
    total = sum(s['vocab_stats'].values())
    print(f"词库: {s['vocab_stats']}  合计: {total}")

    # 模拟日常对话
    convos = [
        ("user", "今天下雨了，我在家看了一天电影，看了三部科幻片"),
        ("ai", "推荐你看星际穿越和银翼杀手2049，都是经典科幻电影"),
        ("user", "老板让我加班到10点，太累了，明天还要出差去上海"),
        ("ai", "注意休息，出差记得提前订酒店，上海这几天天气不错"),
        ("user", "同事推荐了一家杭州的火锅店，周末约朋友去吃"),
        ("ai", "杭州的火锅确实不错，海底捞和大龙燚都很受欢迎"),
        ("user", "孩子考试考了第一名，太开心了，奖励他玩游戏"),
        ("user", "最近A股跌得厉害，基金也亏了不少，要不要割肉"),
        ("ai", "建议长期持有，不要追涨杀跌。可以考虑定投分散风险"),
        ("user", "女朋友生日快到了，想在北京找一家好餐厅庆祝"),
        ("ai", "北京的新荣记和大董烤鸭都很适合庆祝，提前预约"),
        ("user", "周末带孩子去公园遛狗，顺便拍了很多照片"),
    ]

    print("\n写入记忆...")
    for sender, msg in convos:
        mid = mw.on_outgoing(msg, sender=sender)
        if mid:
            dims = mw.retriever.vocab.extract_dimensions(msg)
            active = {k.value: v for k, v in dims.items() if v}
            print(f"  [{sender}] {active}")

    count = mw.stats()['memory_count']
    print(f"\n记忆库: {count}条")

    # 检索测试
    print("\n" + "=" * 60)
    queries = [
        "下雨天看什么电影",
        "加班出差的事",
        "杭州吃火锅",
        "孩子考试成绩",
        "股票基金亏了",
        "朋友周末约饭",
        "女朋友生日",
        "北京餐厅推荐",
        "遛狗拍照",
        "上海天气怎么样",
    ]

    for q in queries:
        dims = mw.retriever.vocab.extract_dimensions(q)
        active = {k.value: v for k, v in dims.items() if v}
        results = mw.search_only(q)
        print(f"\n「{q}」 维度: {active}")
        for i, r in enumerate(results[:2]):
            pct = int(r.score * 100)
            print(f"  #{i+1} ({pct}%) {r.summary[:70]}")
        if not results:
            print(f"  (无匹配)")

    # 注入测试
    print("\n" + "=" * 60)
    print("注入测试:")
    enriched = mw.on_incoming("孩子最近学习怎么样")
    print(enriched)

    print("\n" + "=" * 60)
    print("混合测试(项目+日常):")
    # 加一条项目记忆
    mw.on_outgoing("AppBridge的推送功能修复了，支持conversation_id", sender="ai")
    enriched2 = mw.on_incoming("推送功能修好了吗")
    print(enriched2)

    print("\n✅ 日常对话测试完成！")


if __name__ == "__main__":
    test_daily()
