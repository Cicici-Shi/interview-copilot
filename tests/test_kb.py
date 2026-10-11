from copilot.kb import KnowledgeBase, chunk_markdown, rrf, tokenize

DOC = """# 面试
## MQ
### 二、Kafka 深度解析
1. **消息积压怎么处理**
   - 增加分区并同步增加消费者实例
   - 消费者数量超过分区数没用
2. **消费者组 Consumer Group**
   - 同一组内一个分区只被一个消费者消费
## Redis
### 五、分布式锁
一段没有编号的开场说明。
1. **SETNX 加过期时间**
   - 用 Lua 保证原子
"""


def test_numbered_items_become_chunks_under_their_headings():
    chunks = chunk_markdown(DOC)
    titles = [c.title for c in chunks]
    assert "消息积压怎么处理" in titles
    assert "五、分布式锁" in titles  # 编号前的散文字归到小节名下
    kafka = next(c for c in chunks if c.title == "消费者组 Consumer Group")
    assert kafka.path == ("MQ", "二、Kafka 深度解析")  # 唯一的一级标题是文档名，不进路径
    assert "一个分区只被一个消费者" in kafka.body


def test_indented_numbers_stay_inside_their_item():
    chunks = chunk_markdown("## A\n1. **题**\n   1. 子点一\n   2. 子点二\n")
    assert len(chunks) == 1 and "子点二" in chunks[0].body


def test_tokenize_mixes_english_words_and_chinese_bigrams():
    tokens = tokenize("Kafka 消息积压")
    assert "kafka" in tokens and "积压" in tokens


def test_search_finds_item_by_question_words():
    kb = KnowledgeBase(chunk_markdown(DOC))
    assert kb.search("Kafka 出现消息积压怎么解决")[0].chunk.title == "消息积压怎么处理"


def test_context_resolves_follow_up_without_topic_word():
    kb = KnowledgeBase(chunk_markdown(DOC))
    follow_up = "那你是增加组还是增加组里面的实例"
    hit = kb.search(follow_up, context="Kafka 消费者组 group 消费", k=1)[0]
    assert hit.chunk.path[0] == "MQ"


def test_rrf_rewards_items_ranked_high_by_both_lists():
    assert rrf([[1, 2, 3], [2, 1, 4], [2, 5]], k=2) == [2, 1]


PROJECTS = """# 检索
## 检索链路怎么分层？
结论先说。
1. 第一层 BM25
2. 第二层 rerank
## 课件的切分策略是什么？
按页切。
# 闲聊
## 为什么读研
想系统补 AI。
"""


def test_project_doc_is_chunked_per_question_with_its_numbered_list_intact():
    chunks = chunk_markdown(PROJECTS, source="项目", split_items=False)
    assert [c.title for c in chunks] == ["检索链路怎么分层？", "课件的切分策略是什么？", "为什么读研"]
    assert "第二层 rerank" in chunks[0].body
    assert chunks[0].label == "项目｜检索 / 检索链路怎么分层？"


def test_sources_are_merged_with_unique_ids(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "b.md").write_text(PROJECTS, encoding="utf-8")
    kb = KnowledgeBase.from_sources([(tmp_path / "a.md", "八股", True), (tmp_path / "b.md", "项目", False), (tmp_path / "missing.md", "x", True)])
    assert [c.id for c in kb.chunks] == list(range(len(kb.chunks)))
    assert kb.search("检索链路分层")[0].chunk.source == "项目"
