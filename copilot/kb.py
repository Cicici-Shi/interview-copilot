"""知识库：把面试总文档切成条目卡片，用 BM25 检索。

文档本身按「## 分类 → ### 小节 → 1. **条目标题** + 要点」组织，
一个编号条目就是一个能直接拿来答题的单位，所以按条目切，不按字数切。
不用 embedding：口语和文档用词对不上的情况交给慢答的查询改写。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
NUMBERED_ITEM = re.compile(r"^(\d+)[.、]\s+(.*)$")
MAX_BODY = 1800


@dataclass(frozen=True)
class Chunk:
    id: int
    path: tuple[str, ...]  # 所在标题链，如 ("MQ", "二、Kafka 深度解析")
    title: str
    body: str
    source: str = "八股"

    @property
    def label(self) -> str:
        parts = [p for p in self.path if p]
        return f"{self.source}｜" + " / ".join(parts if parts and parts[-1] == self.title else [*parts, self.title])


def clean(text: str) -> str:
    return re.sub(r"\*\*|__|`", "", text).strip()


def chunk_markdown(markdown: str, source: str = "八股", split_items: bool = True, first_id: int = 0) -> list[Chunk]:
    """编号条目各成一块；小节里编号条目之前的散文字也成一块，标题用小节名。

    项目文档是「## 问题 + 答案」，答案里的编号列表属于同一个答案，所以 split_items=False，只按标题切。
    """
    # 全文只有一个一级标题时它是文档名（如「# 面试」），不进路径；有多个时它们是分类（如「# 检索」）
    title_level = 1 if sum(1 for line in markdown.splitlines() if line.startswith("# ")) == 1 else 0
    chunks: list[Chunk] = []
    path: list[str] = []
    title: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if title is None and not body:
            return
        name = title or (path[-1] if path else "文档开头")
        for start in range(0, max(len(body), 1), MAX_BODY):
            part = body[start : start + MAX_BODY]
            if not part and start > 0:
                break
            chunks.append(Chunk(first_id + len(chunks), tuple(path), name, part, source))

    for raw in markdown.splitlines():
        heading = HEADING.match(raw)
        item = split_items and NUMBERED_ITEM.match(raw)  # 只认顶格编号，缩进的是条目内部要点
        if heading:
            flush()
            level = len(heading.group(1)) - title_level
            if level == 0:
                title, lines = None, []
                continue
            del path[level - 1 :]
            path.extend([""] * (level - 1 - len(path)))
            path.append(clean(heading.group(2)))
            title, lines = None, []
        elif item:
            flush()
            title, lines = clean(item.group(2)), []
        else:
            lines.append(raw)
    flush()
    return [c for c in chunks if c.body or c.title]


def tokenize(text: str) -> list[str]:
    """英文按词，中文按字的二元组。二元组对「消息积压」「幂等」这类术语足够。"""
    text = clean(text)
    # 驼峰词额外拆开：HashMap 也算 hash、map。实时转写常把它写成「哈希 map」，不拆就一个词都对不上
    parts = [p.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", text) for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", w) if p.lower() != w.lower() and len(p) > 1]
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9_+#.]*[a-z0-9+#]|[a-z]|\d+", text) + parts
    for run in re.findall(r"[一-鿿]+", text):
        if len(run) == 1:
            tokens.append(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Field:
    """一个字段的 BM25 统计。"""

    def __init__(self, docs: list[list[str]], k1: float, b: float):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        n = len(docs)
        df = Counter(t for doc in self.docs for t in doc)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.avg = sum(self.lengths) / max(n, 1) or 1.0

    def scores(self, terms: set[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


@dataclass
class KnowledgeBase:
    """标题和正文分字段打分再相加（BM25F 的简化）：标题就是题目本身，
    混进正文的话，长答案（项目口径动辄上千字）的标题命中会被长度归一化稀释掉。"""

    chunks: list[Chunk]
    k1: float = 1.4
    b: float = 0.75
    title_weight: float = 1.0
    _title: Field = field(init=False)
    _body: Field = field(init=False)

    def __post_init__(self) -> None:
        self._title = Field([tokenize(c.title) for c in self.chunks], self.k1, self.b)
        self._body = Field([tokenize(" ".join(c.path)) + tokenize(c.body) for c in self.chunks], self.k1, self.b)

    @classmethod
    def from_sources(cls, sources: list[tuple[str | Path, str, bool]]) -> "KnowledgeBase":
        """sources 是 (文件, 来源名, 是否按编号条目切)，不存在的文件跳过。"""
        chunks: list[Chunk] = []
        for path, source, split_items in sources:
            if Path(path).exists():
                chunks += chunk_markdown(Path(path).read_text(encoding="utf-8"), source, split_items, len(chunks))
        return cls(chunks)

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        return [self.title_weight * t + b for t, b in zip(self._title.scores(terms), self._body.scores(terms))]

    def search(self, query: str, k: int = 3, context: str = "", context_weight: float = 0.4) -> list[Hit]:
        """context 是前几轮面试官的话：追问「那增加 group 是增加消费者吗」本身不提 Kafka，靠它补上。"""
        total = self.scores(query)
        if context:
            total = [a + context_weight * c for a, c in zip(total, self.scores(context))]
        ranked = sorted(range(len(total)), key=lambda i: total[i], reverse=True)
        return [Hit(self.chunks[i], total[i]) for i in ranked[:k] if total[i] > 0]

    def search_many(self, queries: list[str], k: int = 5) -> list[Hit]:
        """多个改写查询轮流取，保证每个查询的第一名先进来（和黑客松快答的合并方式一样）。"""
        per = [self.search(q, k) for q in queries if q.strip()]
        merged: dict[int, Hit] = {}
        for rank in range(k):
            for hits in per:
                if rank < len(hits) and hits[rank].chunk.id not in merged:
                    merged[hits[rank].chunk.id] = hits[rank]
        return list(merged.values())[:k]

    @staticmethod
    def outline_key(c: Chunk) -> str:
        return c.source + "｜" + " / ".join(p for p in c.path[:2] if p)

    def section(self, key: str) -> list[Chunk]:
        """目录里的一行对应的全部切块；模型抄错一个字就是空列表。"""
        return [c for c in self.chunks if c.path and self.outline_key(c) == key.strip()]

    def outline(self) -> str:
        """二、三级标题目录，给慢答改写查询时对齐文档用词。"""
        seen: list[str] = []
        for c in self.chunks:
            if not c.path:
                continue
            key = self.outline_key(c)
            if key not in seen:
                seen.append(key)
        return "\n".join(seen)


def rrf(rankings: list[list[int]], k: int, c: int = 60) -> list[int]:
    """Reciprocal Rank Fusion：只看名次不看分数，BM25 分和余弦分不用对齐量纲。"""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1 / (c + rank + 1)
    return sorted(score, key=score.get, reverse=True)[:k]


class DenseIndex:
    """切块向量。按全部切块文本的 hash 缓存，文档没改就不重新 embed。"""

    def __init__(self, kb: KnowledgeBase, vectors):
        import numpy as np

        self.kb = kb
        v = np.asarray(vectors, dtype=np.float32)
        self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)

    @staticmethod
    def chunk_text(c: Chunk) -> str:
        return f"{c.label}\n{c.body}"[:2000]

    @classmethod
    async def build(cls, kb: KnowledgeBase, embedder, cache_dir: str | Path, batch: int = 64) -> "DenseIndex":
        import hashlib

        import numpy as np

        texts = [cls.chunk_text(c) for c in kb.chunks]
        digest = hashlib.sha256(("\x00".join(texts) + embedder.model).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"dense-{digest}.npy"
        if cache.exists():
            return cls(kb, np.load(cache))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            vectors.extend(await embedder.embed(texts[start : start + batch]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.asarray(vectors, dtype=np.float32))
        return cls(kb, vectors)

    def rank(self, query_vector, k: int) -> list[int]:
        import numpy as np

        q = np.asarray(query_vector, dtype=np.float32)
        sims = self.vectors @ (q / np.linalg.norm(q))
        return [int(i) for i in np.argsort(-sims)[:k]]

    async def hybrid(self, queries: list[str], embedder, k: int = 5, depth: int = 10) -> list[Hit]:
        """每个查询各出一条 BM25 排名和一条向量排名，全部 RRF 合并。"""
        queries = [q for q in queries if q.strip()]
        vectors = await embedder.embed(queries)
        rankings = [[h.chunk.id for h in self.kb.search(q, depth)] for q in queries]
        rankings += [self.rank(v, depth) for v in vectors]
        return [Hit(self.kb.chunks[i], 0.0) for i in rrf(rankings, k)]
