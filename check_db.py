"""检查记忆数据库内容

实际表结构:
  memories: id, summary, content, created_at, mentioned_time, metadata
  dim_vectors: memory_id, dimension, vector(BLOB), terms
"""
import sqlite3
import os
import json
import time

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'memory.db')

def main():
    if not os.path.exists(DB_PATH):
        print("数据库文件不存在!")
        return

    size = os.path.getsize(DB_PATH)
    print(f"数据库文件: {DB_PATH} ({size} bytes)")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 查看所有表
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print(f"表: {[t[0] for t in tables]}")

    # === memories 表 (索引层+详情层合一) ===
    cur.execute("SELECT COUNT(*) FROM memories")
    mem_count = cur.fetchone()[0]
    print(f"\nmemories 条目数: {mem_count}")

    if mem_count > 0:
        cur.execute("SELECT id, summary, content, created_at, metadata FROM memories ORDER BY created_at DESC")
        rows = cur.fetchall()
        print("\n=== memories 表（按时间倒序）===")
        for i, (mid, summary, content, ts, meta) in enumerate(rows):
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
            print(f"[{i+1}] id={mid[:12]}...")
            print(f"    【索引层】summary: {summary[:100]}")
            print(f"    【详情层】content长度: {len(content)}字符")
            content_preview = content[:120].replace('\n', ' ')
            print(f"    【详情层】content预览: {content_preview}")
            print(f"    created: {ts_str}")
            if meta:
                try:
                    m = json.loads(meta)
                    sender = m.get("sender", "?")
                    dims = m.get("matched_dims", {})
                    print(f"    sender: {sender}")
                    print(f"    matched_dims: {dims}")
                except:
                    print(f"    metadata: {meta[:100]}")
            print()

    # === dim_vectors 表 ===
    cur.execute("SELECT COUNT(*) FROM dim_vectors")
    vec_count = cur.fetchone()[0]
    print(f"dim_vectors 条目数: {vec_count}")

    if vec_count > 0 and mem_count > 0:
        # 展示第一条记忆的向量维度
        first_id = cur.execute("SELECT id FROM memories ORDER BY created_at DESC LIMIT 1").fetchone()[0]
        cur.execute("SELECT dimension, terms FROM dim_vectors WHERE memory_id = ?", (first_id,))
        dim_rows = cur.fetchall()
        print(f"\n=== 最新记忆的维度向量 (id={first_id[:12]}...) ===")
        for dim, terms_str in dim_rows:
            terms = json.loads(terms_str)
            print(f"  {dim}: {terms}")

    print(f"\n{'='*50}")
    print(f"总结:")
    print(f"  memories表: {mem_count}条 (每条含summary索引 + content详情)")
    print(f"  dim_vectors表: {vec_count}条 (每条记忆多个维度向量)")
    if mem_count > 0:
        avg_dims = vec_count / mem_count
        print(f"  平均每条记忆: {avg_dims:.1f}个维度向量")

    conn.close()


if __name__ == "__main__":
    main()
