"""中间件端到端测试：17维度系统 + 真实middleware流程"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceted_memory.middleware import MemoryMiddleware

VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vocab')
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'memory.db')


def test_full_flow():
    print("=" * 60)
    print("17维度 中间件全流程测试")
    print("=" * 60 + "\n")

    mw = MemoryMiddleware(db_path=DB_PATH, vocab_dir=VOCAB_DIR)

    # 1. 词库统计
    stats = mw.stats()
    print(f"词库: {sum(stats['vocab_stats'].values())} 词 / {len(stats['vocab_stats'])} 维度")
    print(f"记忆: {stats['memory_count']} 条\n")

    # 2. 模拟对话存储 (on_outgoing)
    conversations = [
        ("ai", "AppBridge推送功能已修复，测试全部通过"),
        ("user", "妈妈做的红烧肉太好吃了，吃了三碗饭"),
        ("ai", "周末去杭州西湖边吃了火锅，味道非常不错"),
        ("user", "最近加班太多了，感觉特别累，想请假休息"),
        ("ai", "高考成绩出来了，孩子考上了985大学，全家很高兴"),
        ("user", "在星巴克喝咖啡看了一本心理学的书，很有收获"),
        ("ai", "遗传算法收敛速度优化后提升了3倍，适应度从0.7到0.95"),
        ("user", "和女朋友吵架了，冷战三天了，很难过"),
    ]

    print("=== 存储对话 ===")
    for sender, msg in conversations:
        mid = mw.on_outgoing(msg, sender=sender)
        dims = mw.retriever.vocab.extract_dimensions(msg)
        active = {d.value: v for d, v in dims.items() if v}
        dim_names = list(active.keys())
        print(f"  [{sender}] {msg[:40]}...")
        print(f"    → 维度: {dim_names}")
        print(f"    → id: {mid}")

    stats = mw.stats()
    print(f"\n  已存储 {stats['memory_count']} 条记忆\n")

    # 3. 模拟入站检索 (on_incoming)
    print("=== 检索测试 ===")
    queries = [
        "推送功能还有bug吗",
        "妈妈做的菜好不好吃",
        "遗传算法优化效果怎么样",
        "我和女朋友还在冷战",
        "杭州有什么好吃的",
        "最近工作压力大",
    ]

    for q in queries:
        enriched = mw.on_incoming(q)
        has_memory = "📚 相关记忆" in enriched
        print(f"  查询: '{q}'")
        if has_memory:
            # 提取记忆块
            mem_start = enriched.index("---")
            mem_block = enriched[mem_start:]
            lines = [l for l in mem_block.split('\n') if l.startswith('[')]
            for l in lines:
                print(f"    {l}")
        else:
            print(f"    (无匹配记忆)")
        print()

    # 4. 验证注入格式
    print("=== 注入格式验证 ===")
    enriched = mw.on_incoming("妈妈做的红烧肉")
    print(f"  原始消息: '妈妈做的红烧肉'")
    print(f"  增强消息:")
    for line in enriched.split('\n'):
        print(f"    {line}")

    assert "📚 相关记忆" in enriched, "应包含记忆注入"
    assert "妈妈" in enriched, "应包含相关内容"
    print("  ✓ 注入格式正常\n")

    print("=" * 60)
    print("全部测试通过 ✓")
    print(f"数据库: {DB_PATH}")
    print("=" * 60)

    mw.close()


if __name__ == "__main__":
    test_full_flow()
