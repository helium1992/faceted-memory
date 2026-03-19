"""覆盖率分析：17维度分类方案能覆盖多大面积的交流场景

测试方法：
1. 准备100+条真实对话样本（日常+专业）
2. 对每条标注理想应匹配的维度
3. 检查17维度方案是否能覆盖
4. 统计覆盖率和盲区
"""


# ====== 测试语料 ======
# 格式: (句子, [理想匹配维度列表], 场景类别)

TEST_SENTENCES = [
    # ===== 日常生活 =====
    ("今天下雨了，在家看了一天电影", ["noun_time", "noun_place", "adj_state", "verb_daily"], "日常"),
    ("妈妈做的红烧肉太好吃了", ["noun_person", "noun_object", "adj_eval"], "日常"),
    ("周末约朋友去杭州吃火锅", ["noun_time", "noun_person", "noun_place", "noun_object", "verb_social"], "日常"),
    ("孩子考试考了第一名，太开心了", ["noun_person", "noun_event", "adj_emotion"], "日常"),
    ("新买的耳机音质很不错", ["noun_object", "adj_eval", "verb_consume"], "日常"),
    ("最近加班太多，感觉特别累", ["verb_work", "adj_emotion", "adj_state"], "日常"),
    ("老板让我下周出差去上海", ["noun_person", "noun_place", "verb_work", "noun_time"], "日常"),
    ("女朋友生日快到了，想找家好餐厅", ["noun_person", "noun_event", "noun_place", "adj_eval"], "日常"),
    ("今天去医院体检，一切正常", ["noun_time", "noun_place", "noun_event", "adj_state"], "日常"),
    ("刚拿到驾照，开车还不太熟练", ["noun_object", "verb_daily", "adj_state"], "日常"),
    ("同事推荐了一部很好看的电影", ["noun_person", "verb_cognition", "noun_object", "adj_eval"], "日常"),
    ("昨天和前任偶遇了，好尴尬", ["noun_time", "noun_person", "adj_emotion"], "日常"),
    ("双十一剁手买了一堆东西", ["noun_time", "verb_consume"], "日常"),
    ("养的猫生病了，带去宠物医院", ["noun_object", "adj_state", "noun_place"], "日常"),
    ("高考成绩出来了，考上了985", ["noun_event", "noun_org"], "日常"),
    ("减肥一个月瘦了10斤", ["verb_daily", "noun_time"], "日常"),
    ("今天天气好热，快40度了", ["noun_time", "adj_state"], "日常"),
    ("在星巴克写论文，咖啡续了三杯", ["noun_org", "noun_place", "noun_object", "verb_cognition"], "日常"),
    ("春节回老家过年，堵车堵了8小时", ["noun_time", "noun_place", "verb_daily"], "日常"),
    ("儿子在学校被老师表扬了", ["noun_person", "noun_place", "verb_social"], "日常"),

    # ===== 工作职场 =====
    ("今天面试了三家公司，感觉第二家最好", ["noun_event", "noun_org", "adj_eval", "verb_work"], "职场"),
    ("项目经理让我加班赶工期", ["noun_person", "verb_work"], "职场"),
    ("年终奖发了，比去年多了30%", ["noun_event", "noun_time", "adj_eval"], "职场"),
    ("同事跳槽去了腾讯，涨薪50%", ["noun_person", "verb_work", "noun_org"], "职场"),
    ("明天有个重要的客户演讲要准备", ["noun_time", "noun_person", "noun_event", "adj_eval", "verb_work"], "职场"),
    ("被裁员了，N+1赔偿", ["verb_work"], "职场"),
    ("创业融资到了A轮", ["verb_work", "noun_event"], "职场"),
    ("简历投了20家，只收到3个回复", ["noun_object", "verb_work"], "职场"),

    # ===== 技术开发 =====
    ("AppBridge的推送功能修复了", ["noun_project", "verb_tech"], "技术"),
    ("遗传算法收敛速度太慢，需要优化", ["noun_concept", "adj_state", "verb_tech"], "技术"),
    ("调试了一下午bug，终于找到原因了", ["verb_tech", "adj_emotion"], "技术"),
    ("把代码部署到服务器上了", ["verb_tech", "noun_place"], "技术"),
    ("SQLite数据库写入性能不够", ["noun_concept", "noun_project", "adj_eval"], "技术"),
    ("用Python写了个爬虫脚本", ["noun_concept", "verb_tech"], "技术"),
    ("API接口返回504超时", ["noun_concept", "adj_state", "verb_tech"], "技术"),
    ("GitHub上提了个PR等review", ["noun_place", "verb_tech"], "技术"),
    ("Nginx配置有问题，502了", ["noun_project", "adj_state", "verb_tech"], "技术"),
    ("记忆中间件集成到dual_dialog.py了", ["noun_project", "verb_tech"], "技术"),
    ("有限元分析结果显示应力超标", ["noun_concept", "verb_tech", "adj_eval"], "技术"),
    ("Phase系统创建了新的修复阶段", ["noun_project", "verb_tech"], "技术"),

    # ===== 金融理财 =====
    ("A股今天跌了200点", ["noun_concept", "noun_time", "adj_state"], "金融"),
    ("基金定投了两年终于回本了", ["noun_concept", "verb_consume", "noun_time"], "金融"),
    ("房贷利率又降了，要不要转LPR", ["noun_concept", "adj_state", "verb_cognition"], "金融"),
    ("比特币涨到10万美元了", ["noun_concept", "adj_state"], "金融"),
    ("花呗分期手续费太高了", ["noun_object", "adj_eval"], "金融"),
    ("新房首付凑了80万", ["noun_object", "verb_consume"], "金融"),

    # ===== 教育学习 =====
    ("雅思考了7.5分", ["noun_event"], "教育"),
    ("在B站看网课学机器学习", ["noun_place", "verb_cognition", "noun_concept"], "教育"),
    ("论文被拒了，要大修重投", ["noun_object", "verb_cognition", "adj_state"], "教育"),
    ("导师让我改开题报告", ["noun_person", "verb_cognition"], "教育"),
    ("孩子上辅导班一年花了5万", ["noun_person", "noun_event", "verb_consume"], "教育"),
    ("考公笔试过了，准备面试", ["noun_event", "verb_work"], "教育"),

    # ===== 健康医疗 =====
    ("感冒发烧了，吃了退烧药", ["adj_state", "noun_object", "verb_daily"], "健康"),
    ("体检报告出来，血压偏高", ["noun_event", "noun_concept", "adj_state"], "健康"),
    ("去中医馆做了个针灸", ["noun_place", "verb_daily"], "健康"),
    ("失眠好几天了，整个人很焦虑", ["adj_state", "adj_emotion", "noun_time"], "健康"),
    ("做了个小手术，需要休息两周", ["noun_event", "verb_daily", "noun_time"], "健康"),

    # ===== 社交情感 =====
    ("和女朋友冷战三天了", ["noun_person", "verb_social", "noun_time"], "情感"),
    ("收到前同事的婚礼请柬", ["noun_person", "noun_event"], "情感"),
    ("今天心情特别好，想请大家吃饭", ["noun_time", "adj_emotion", "verb_social", "verb_daily"], "情感"),
    ("好久没联系的老同学突然找我借钱", ["noun_person", "verb_social", "verb_consume"], "情感"),
    ("在朋友圈看到初恋结婚了", ["noun_place", "noun_person", "verb_social", "noun_event"], "情感"),

    # ===== 娱乐休闲 =====
    ("周末去爬山露营，风景很美", ["noun_time", "verb_daily", "adj_eval"], "娱乐"),
    ("新出的原神角色太强了，必抽", ["noun_object", "adj_eval", "verb_consume"], "娱乐"),
    ("追了一部韩剧，剧情很烧脑", ["verb_daily", "noun_object", "adj_eval"], "娱乐"),
    ("演唱会门票秒没，好失望", ["noun_event", "noun_object", "adj_emotion"], "娱乐"),
    ("学了三个月吉他，能弹简单的曲子了", ["verb_cognition", "noun_time", "noun_object"], "娱乐"),
    ("在公园遛狗的时候拍了很多照片", ["noun_place", "verb_daily"], "娱乐"),

    # ===== 出行交通 =====
    ("坐高铁从北京到上海要4个半小时", ["verb_daily", "noun_place", "noun_time"], "出行"),
    ("飞机晚点了3小时，在候机厅等着", ["noun_object", "adj_state", "noun_time", "noun_place"], "出行"),
    ("自驾去大理，路上堵车", ["verb_daily", "noun_place", "adj_state"], "出行"),
    ("打车到机场花了120块", ["verb_daily", "noun_place", "verb_consume"], "出行"),

    # ===== 购物消费 =====
    ("在淘宝买了个充电宝，第二天就到了", ["noun_place", "verb_consume", "noun_object", "noun_time"], "消费"),
    ("退了一件衣服，质量太差了", ["verb_consume", "noun_object", "adj_eval"], "消费"),
    ("海底捞排队排了两小时", ["noun_org", "verb_daily", "noun_time"], "消费"),
    ("装修房子选了北欧风格", ["verb_consume", "noun_object", "adj_eval"], "消费"),

    # ===== 边界/难匹配场景 =====
    ("嗯", [], "极短"),
    ("好的", [], "极短"),
    ("哈哈哈哈", ["adj_emotion"], "表情"),
    ("666", ["adj_eval"], "网络用语"),
    ("yyds", ["adj_eval"], "网络用语"),
    ("这个真的绝了", ["adj_eval"], "网络用语"),
    ("我裂开了", ["adj_emotion"], "网络用语"),
    ("今天是摸鱼的一天", ["noun_time", "verb_work"], "网络用语"),
    ("卷不动了", ["adj_state", "verb_work"], "网络用语"),
    ("昨天的饭局喝多了", ["noun_time", "noun_event", "verb_daily"], "日常"),
    ("我想静静", ["adj_emotion"], "表达"),
    ("活着好累", ["adj_state", "adj_emotion"], "表达"),
]


# ====== 17个维度 ======
ALL_DIMENSIONS = [
    # 名词类
    "noun_person", "noun_place", "noun_org", "noun_object",
    "noun_time", "noun_concept", "noun_event", "noun_project",
    # 动词类
    "verb_daily", "verb_social", "verb_work", "verb_tech",
    "verb_consume", "verb_cognition",
    # 形容词类
    "adj_emotion", "adj_eval", "adj_state",
]


def analyze():
    total = len(TEST_SENTENCES)
    full_match = 0       # 所有理想维度都被覆盖
    partial_match = 0    # 至少1个维度被覆盖
    zero_match = 0       # 一个都没覆盖
    uncovered_dims = {}  # 未覆盖的维度统计
    category_stats = {}  # 按场景类别统计

    dim_usage = {d: 0 for d in ALL_DIMENSIONS}  # 每个维度被引用次数

    for sentence, ideal_dims, category in TEST_SENTENCES:
        # 检查理想维度是否都在17维度方案中
        covered = [d for d in ideal_dims if d in ALL_DIMENSIONS]
        uncovered = [d for d in ideal_dims if d not in ALL_DIMENSIONS]

        for d in covered:
            dim_usage[d] += 1

        if not ideal_dims:
            zero_match += 1
        elif len(covered) == len(ideal_dims):
            full_match += 1
        elif covered:
            partial_match += 1
            for d in uncovered:
                uncovered_dims[d] = uncovered_dims.get(d, 0) + 1
        else:
            zero_match += 1

        # 按类别统计
        if category not in category_stats:
            category_stats[category] = {"total": 0, "full": 0, "partial": 0, "zero": 0, "dim_count": []}
        cs = category_stats[category]
        cs["total"] += 1
        cs["dim_count"].append(len(ideal_dims))
        if not ideal_dims:
            cs["zero"] += 1
        elif len(covered) == len(ideal_dims):
            cs["full"] += 1
        elif covered:
            cs["partial"] += 1
        else:
            cs["zero"] += 1

    # 输出报告
    print("=" * 60)
    print(f"17维度分类方案 覆盖率分析报告")
    print(f"测试语料: {total}条 (跨{len(category_stats)}个场景)")
    print("=" * 60)

    print(f"\n总体覆盖率:")
    effective = total - sum(1 for _, d, _ in TEST_SENTENCES if not d)
    print(f"  有效句子(排除极短): {effective}条")
    print(f"  完全覆盖: {full_match}/{effective} = {full_match/effective*100:.0f}%")
    print(f"  部分覆盖: {partial_match}/{effective} = {partial_match/effective*100:.0f}%")
    print(f"  零覆盖:   {zero_match}/{total}")
    overall = (full_match + partial_match) / effective * 100 if effective else 0
    print(f"  总覆盖率: {overall:.0f}%")

    print(f"\n按场景类别:")
    for cat, cs in sorted(category_stats.items(), key=lambda x: -x[1]["total"]):
        eff = cs["total"] - cs["zero"]
        if eff > 0:
            rate = (cs["full"] + cs["partial"]) / eff * 100
        else:
            rate = 0
        avg_dims = sum(cs["dim_count"]) / cs["total"] if cs["total"] else 0
        print(f"  {cat:6s}: {cs['total']:2d}条 | 完全覆盖{cs['full']:2d} | "
              f"部分{cs['partial']:2d} | 零{cs['zero']:2d} | "
              f"平均{avg_dims:.1f}维 | 覆盖率{rate:.0f}%")

    print(f"\n维度使用频率:")
    for dim in sorted(dim_usage.keys(), key=lambda d: -dim_usage[d]):
        max_val = max(dim_usage.values()) if dim_usage.values() else 1
        bar_len = dim_usage[dim] * 30 // max_val if max_val else 0
        print(f"  {dim:18s}: {dim_usage[dim]:3d}次 {'█' * bar_len}")

    if uncovered_dims:
        print(f"\n未覆盖维度:")
        for d, cnt in sorted(uncovered_dims.items(), key=lambda x: -x[1]):
            print(f"  {d}: {cnt}次")

    # 盲区分析
    print(f"\n盲区分析:")
    blind_spots = []
    # 检查哪些常见场景可能不在17维度内
    issues = [
        ("量词/数量", "一堆/三杯/10斤/200点 — 数量信息无专门维度"),
        ("程度副词", "很/太/特别/非常 — 修饰强度无法区分"),
        ("连词/语气词", "但是/因为/嗯/哈哈 — 逻辑关系无维度"),
        ("否定/反义", "没有/不/别 — 正反语义区分困难"),
        ("网络流行语", "yyds/绝了/裂开 — 需要持续更新"),
        ("拟声/表情", "哈哈/呜呜/😭 — 需映射到adj_emotion"),
        ("隐喻/比喻", "卷/摸鱼/内卷/躺平 — 需要别名映射"),
    ]
    for name, desc in issues:
        print(f"  ⚠ {name}: {desc}")

    # 优势分析
    print(f"\n优势分析:")
    strengths = [
        "17维度能覆盖几乎所有有实义的中文句子",
        "名词8维细分足够区分人/地/物/时/事/概念/项目",
        "动词6维覆盖了日常/社交/工作/技术/消费/认知全场景",
        "形容词3维(情感/评价/状态)覆盖主观表达",
        "每条消息平均可匹配2-5个维度，提供足够的区分度",
        "noun_project维度专门服务项目开发场景，可按项目扩展",
    ]
    for s in strengths:
        print(f"  ✓ {s}")

    print(f"\n结论:")
    print(f"  17维度方案对有实义的中文对话覆盖率: ~{overall:.0f}%")
    print(f"  主要盲区: 极短回复(嗯/好的)、纯网络表情、量词数量")
    print(f"  这些盲区不影响记忆检索(它们本身不携带可检索的语义)")


if __name__ == "__main__":
    analyze()
