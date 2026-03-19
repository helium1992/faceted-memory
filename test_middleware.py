"""端到端测试：记忆中间件 + 项目词库"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceted_memory.middleware import MemoryMiddleware

DB_PATH = ":memory:"
VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vocab')


def test_middleware():
    mw = MemoryMiddleware(db_path=DB_PATH, vocab_dir=VOCAB_DIR, top_k=3, min_score=0.05)

    print("=" * 60)
    print("📊 词库统计:", mw.stats())
    print("=" * 60)

    # 模拟一系列对话，自动存入记忆
    conversations = [
        ("ai", "AppBridge的推送功能已经修复了，现在push_message和push_screenshot都支持conversation_id参数，Qwen Agent推送的消息会显示为新对话。"),
        ("user", "看到了，qwen推送确实显示为新对话了"),
        ("ai", "Qwen Agent自主完成了拱桥优化任务，承重效率2.17 N/g，最大承重448.1N，自重206.78g，安全系数4.48。Phase系统创建了bridge_span phase。"),
        ("user", "优化结果不错，杆件重叠问题解决了吗"),
        ("ai", "已实现分面多向量记忆检索系统，包含词库匹配、向量比对、时间衰减、动态遮蔽和消歧确认功能。存储使用SQLite两级架构。"),
        ("user", "记忆系统集成到dual_dialog.py了吗"),
        ("ai", "dual_dialog.py已集成记忆中间件，每次对话自动检索相关记忆并注入到响应文件中，同时自动存储AI和用户的消息。"),
    ]

    print("\n📝 写入模拟对话记忆...")
    for sender, msg in conversations:
        mem_id = mw.on_outgoing(msg, sender=sender)
        if mem_id:
            # 显示匹配到的维度
            dims = mw.retriever.vocab.extract_dimensions(msg)
            active = {k.value: v for k, v in dims.items() if v}
            print(f"  [{sender}] → mem_{mem_id[:8]}  维度: {active}")
        else:
            print(f"  [{sender}] → (无匹配词条，未存储)")

    print(f"\n📊 记忆库: {mw.stats()}")

    # 测试检索
    print("\n" + "=" * 60)
    print("🔍 检索测试")
    print("=" * 60)

    queries = [
        "AppBridge推送功能",
        "Qwen优化结果怎么样",
        "杆件重叠",
        "记忆系统",
        "dual_dialog.py的改动",
        "承重效率",
        "词库匹配怎么工作的",
    ]

    for q in queries:
        print(f"\n🔍 「{q}」")
        dims = mw.retriever.vocab.extract_dimensions(q)
        active = {k.value: v for k, v in dims.items() if v}
        print(f"   维度: {active}")

        results = mw.search_only(q)
        if results:
            for i, r in enumerate(results[:3]):
                pct = int(r.score * 100)
                print(f"   #{i+1} ({pct}%) {r.summary[:80]}")
        else:
            print(f"   (无匹配)")

    # 测试消息注入
    print("\n" + "=" * 60)
    print("💉 消息注入测试")
    print("=" * 60)

    enriched = mw.on_incoming("AppBridge推送功能修好了吗")
    print(enriched)

    print("\n✅ 中间件测试完成！")


if __name__ == "__main__":
    test_middleware()
