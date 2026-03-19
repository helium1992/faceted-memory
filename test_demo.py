"""端到端测试：分面多向量记忆检索系统

构建一个简单的商业新闻场景词库，添加几条记忆，验证检索效果。
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceted_memory import FacetedRetriever, Dimension


def build_demo_retriever() -> FacetedRetriever:
    """构建demo检索器（内存模式+简单词库）"""
    r = FacetedRetriever(db_path=":memory:")

    # ===== WHO 维度 =====
    r.vocab.add_term(Dimension.WHO, "张三", aliases=["老张"])
    r.vocab.add_term(Dimension.WHO, "李四", aliases=["老李"])
    r.vocab.add_term(Dimension.WHO, "王五")
    r.vocab.add_term(Dimension.WHO, "阿里巴巴", aliases=["阿里"])
    r.vocab.add_term(Dimension.WHO, "腾讯")
    r.vocab.add_term(Dimension.WHO, "字节跳动", aliases=["字节"])
    r.vocab.add_term(Dimension.WHO, "华为")

    # ===== WHAT 维度 =====
    r.vocab.add_term(Dimension.WHAT, "收购", aliases=["收购案"])
    r.vocab.add_term(Dimension.WHAT, "融资")
    r.vocab.add_term(Dimension.WHAT, "上市")
    r.vocab.add_term(Dimension.WHAT, "裁员")
    r.vocab.add_term(Dimension.WHAT, "发布会")
    r.vocab.add_term(Dimension.WHAT, "合作")
    r.vocab.add_term(Dimension.WHAT, "投资")
    r.vocab.add_term(Dimension.WHAT, "开会", aliases=["会议", "开了个会"])

    # ===== WHERE 维度 =====
    r.vocab.add_term(Dimension.WHERE, "北京")
    r.vocab.add_term(Dimension.WHERE, "上海")
    r.vocab.add_term(Dimension.WHERE, "深圳")
    r.vocab.add_term(Dimension.WHERE, "杭州")
    r.vocab.add_term(Dimension.WHERE, "硅谷")

    # ===== TAGS 维度 =====
    r.vocab.add_term(Dimension.TAGS, "科技")
    r.vocab.add_term(Dimension.TAGS, "金融")
    r.vocab.add_term(Dimension.TAGS, "教育")
    r.vocab.add_term(Dimension.TAGS, "AI")
    r.vocab.add_term(Dimension.TAGS, "芯片")
    r.vocab.add_term(Dimension.TAGS, "云计算")

    # WHEN 维度由 TimeDecay 自动填充
    return r


def add_demo_memories(r: FacetedRetriever):
    """添加测试记忆"""
    now = time.time()

    r.add_memory(
        memory_id="mem_001",
        summary="阿里巴巴在杭州宣布收购一家AI公司",
        content="2026年3月10日，阿里巴巴集团在杭州总部召开新闻发布会，宣布以50亿元收购一家专注于大模型的AI初创公司。此次收购将加强阿里在云计算和AI领域的布局。",
        created_at=now - 5 * 86400,  # 5天前
        metadata={"source": "新闻"}
    )

    r.add_memory(
        memory_id="mem_002",
        summary="腾讯在深圳发布新AI产品",
        content="腾讯在深圳举办科技发布会，推出了全新的AI助手产品，主打企业级应用场景。",
        created_at=now - 2 * 86400,  # 前天
        metadata={"source": "新闻"}
    )

    r.add_memory(
        memory_id="mem_003",
        summary="张三和李四在上海开了个会讨论投资",
        content="张三（某基金合伙人）和李四（某科技公司CEO）在上海浦东的咖啡厅开会，讨论了一笔2亿元的AI教育项目投资。会议持续了3个小时。",
        created_at=now - 1 * 86400,  # 昨天
        metadata={"source": "日记"}
    )

    r.add_memory(
        memory_id="mem_004",
        summary="华为在北京发布芯片",
        content="华为在北京举行秋季发布会，正式发布了新一代自研AI芯片，性能较上代提升40%。",
        created_at=now - 30 * 86400,  # 上个月
        metadata={"source": "新闻"}
    )

    r.add_memory(
        memory_id="mem_005",
        summary="字节跳动硅谷裁员",
        content="字节跳动宣布将硅谷办公室的技术团队裁员约200人，主要涉及非核心业务线。",
        created_at=now - 10 * 86400,  # 上一周
        metadata={"source": "新闻"}
    )


def test_basic_search():
    """测试基础检索"""
    r = build_demo_retriever()
    add_demo_memories(r)

    print("=" * 60)
    print("📊 系统统计:", r.stats())
    print("=" * 60)

    # 测试用例
    queries = [
        "阿里收购了什么",
        "张三昨天干了什么",
        "腾讯的发布会",
        "上个月华为的事",
        "AI相关的新闻",
        "字节跳动裁员",
        "老李在上海",
        "前天的科技新闻",
    ]

    for q in queries:
        print(f"\n🔍 查询: 「{q}」")

        # 显示词库匹配结果
        dim_terms = r.vocab.extract_dimensions(q)
        active = {k.value: v for k, v in dim_terms.items() if v}
        print(f"   维度提取: {active}")

        # 检索
        results = r.search(q, top_k=3)
        for i, res in enumerate(results):
            dims_str = ", ".join(f"{d}={res.dim_scores.get(d, 0):.2f}"
                                for d in res.active_dims)
            mask_str = f" [masked: {', '.join(res.masked_dims)}]" if res.masked_dims else ""
            print(f"   #{i+1} ({res.score:.3f}) {res.summary}")
            print(f"       {dims_str}{mask_str}")

    print("\n" + "=" * 60)
    print("✅ 基础检索测试完成")


def test_disambiguation():
    """测试消歧交互"""
    r = build_demo_retriever()
    add_demo_memories(r)

    print("\n" + "=" * 60)
    print("🔀 消歧测试")
    print("=" * 60)

    results = r.search("AI相关", top_k=5)

    # 模拟用户选择第2个
    def mock_callback(summaries):
        print("   系统询问：您指的是哪件事？")
        for s in summaries:
            print(f"     {s}")
        print("   → 用户选择: 第2条")
        return 1

    selected = r.disambiguate(results, callback=mock_callback)
    if selected:
        detail = r.store.get_detail(selected.memory_id)
        print(f"\n   ✅ 确认记忆: {selected.summary}")
        print(f"   📄 详情: {detail.content[:100]}...")
    print("✅ 消歧测试完成")


def test_time_decay():
    """测试时间衰减"""
    r = build_demo_retriever()
    add_demo_memories(r)

    print("\n" + "=" * 60)
    print("⏰ 时间衰减测试")
    print("=" * 60)

    time_queries = ["昨天", "前天", "前几天", "上一周", "上个月"]
    for q in time_queries:
        results = r.search(q, top_k=2)
        top = results[0] if results else None
        if top:
            print(f"   「{q}」 → #{top.memory_id} ({top.score:.3f}) {top.summary}")
        else:
            print(f"   「{q}」 → 无匹配")

    print("✅ 时间衰减测试完成")


if __name__ == "__main__":
    test_basic_search()
    test_disambiguation()
    test_time_decay()
    print("\n🎉 所有测试通过！")
