"""端到端测试：17维度分类系统"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceted_memory.retriever import FacetedRetriever

VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vocab')


def test_vocab_loading():
    """测试17维度词库加载"""
    r = FacetedRetriever(db_path=":memory:", vocab_dir=VOCAB_DIR)
    stats = r.vocab.stats()
    print("=== 词库加载统计 ===")
    total = 0
    for dim_name, count in sorted(stats.items()):
        print(f"  {dim_name:18s}: {count:4d} 词")
        total += count
    print(f"  {'总计':18s}: {total:4d} 词")
    assert len(stats) == 17, f"应有17个维度，实际{len(stats)}"
    assert total > 1500, f"总词数应>1500，实际{total}"
    print("  ✓ 17维度词库加载正常\n")
    return r


def test_dimension_extraction(r):
    """测试多维度词条提取"""
    print("=== 维度提取测试 ===")
    test_cases = [
        ("妈妈做的红烧肉太好吃了", {
            "noun_person": ["妈妈"],
            "noun_object": ["红烧肉"],
        }),
        ("周末约朋友去杭州吃火锅", {
            "noun_time": ["周末"],
            "noun_person": ["朋友"],
            "noun_place": ["杭州"],
            "noun_object": ["火锅"],
        }),
        ("AppBridge推送修复了，部署到服务器", {
            "noun_project": ["AppBridge"],
            "verb_tech": ["推送", "修复", "部署"],
        }),
        ("最近加班太多，感觉特别累", {
            "verb_work": ["加班"],
            "adj_state": ["累"],
        }),
        ("在星巴克喝咖啡看书", {
            "noun_org": ["星巴克"],
            "verb_daily": ["喝咖啡"],
            "verb_cognition": ["看书"],
        }),
    ]

    from faceted_memory.vocabulary import Dimension
    for text, expected in test_cases:
        dims = r.vocab.extract_dimensions(text)
        matched = {d.value: v for d, v in dims.items() if v}
        print(f"  输入: '{text}'")
        print(f"  匹配: {matched}")
        for dim_key, expected_terms in expected.items():
            actual = matched.get(dim_key, [])
            for t in expected_terms:
                assert t in actual, f"    ✗ 期望 {dim_key}=[{t}]，实际={actual}"
        print(f"  ✓ 通过")
    print()


def test_memory_store_and_retrieve(r):
    """测试记忆存储和检索"""
    print("=== 存储+检索测试 ===")

    # 存入几条记忆
    memories = [
        ("妈妈做的红烧肉太好吃了，下次还要吃", "妈妈红烧肉"),
        ("周末约朋友去杭州吃火锅，西湖边那家很不错", "杭州火锅"),
        ("AppBridge推送功能修复了，测试通过", "AppBridge推送修复"),
        ("最近加班太多感觉特别累，想请假休息", "加班累请假"),
        ("在星巴克喝咖啡看了一本心理学的书", "星巴克看书"),
        ("高考成绩出来了，考上了985大学", "高考成绩"),
        ("新买的耳机音质很不错，降噪效果也好", "新耳机评价"),
        ("和女朋友吵架了，冷战三天了", "吵架冷战"),
    ]

    for content, summary in memories:
        r.add_memory(summary=summary, content=content, raw_text=content)
    print(f"  已存入 {len(memories)} 条记忆")

    # 检索测试
    queries = [
        ("妈妈做的菜", "妈妈红烧肉"),
        ("杭州旅游吃东西", "杭州火锅"),
        ("推送功能bug", "AppBridge推送修复"),
        ("太累了想休息", "加班累请假"),
        ("咖啡读书", "星巴克看书"),
        ("考试成绩", "高考成绩"),
        ("耳机好不好", "新耳机评价"),
        ("女朋友生气了", "吵架冷战"),
    ]

    correct = 0
    for query, expected_summary in queries:
        results = r.search(query, top_k=3)
        top1 = results[0] if results else None
        hit = top1 and top1.summary == expected_summary
        if hit:
            correct += 1
        status = "✓" if hit else "✗"
        top1_info = f"{top1.summary} ({top1.score:.2f})" if top1 else "无结果"
        print(f"  {status} '{query}' → {top1_info} (期望: {expected_summary})")

    accuracy = correct / len(queries) * 100
    print(f"\n  准确率: {correct}/{len(queries)} = {accuracy:.0f}%")
    assert accuracy >= 60, f"准确率过低: {accuracy:.0f}%"
    print("  ✓ 检索测试通过\n")


def test_injection_format(r):
    """测试记忆注入格式"""
    print("=== 注入格式测试 ===")
    results = r.search_with_detail("杭州吃火锅", top_k=2)
    print(f"  返回 {len(results)} 条结果")
    for i, res in enumerate(results):
        print(f"  [{i+1}] ({res.score*100:.0f}%匹配) {res.summary}")
        print(f"      维度: {res.active_dims}")
        if res.content:
            print(f"      详情: {res.content[:60]}...")
    assert len(results) > 0, "应返回结果"
    assert results[0].content is not None, "详情应被拉取"
    print("  ✓ 注入格式正常\n")


def main():
    print("=" * 60)
    print("17维度分类系统 端到端测试")
    print("=" * 60 + "\n")

    r = test_vocab_loading()
    test_dimension_extraction(r)
    test_memory_store_and_retrieve(r)
    test_injection_format(r)

    print("=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)
    r.close()


if __name__ == "__main__":
    main()
