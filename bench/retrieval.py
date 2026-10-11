"""知识库检索对比，以人工标注 bench/labels/kb-985.txt 为准。

  python bench/retrieval.py pool     # 导出每题的候选池（各方法前 5 名的并集），供人工标注
  python bench/retrieval.py run      # 跑各方法（需要 embedding / LLM 的现调）并出表
"""

import asyncio
import json
import math
import re
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from copilot.assistant import INTERVIEWER  # noqa: E402
from copilot.kb import DenseIndex, KnowledgeBase, rrf  # noqa: E402
from copilot.llm import Embedder, OpenAICompatible  # noqa: E402
from gate import labels as gate_labels, turns  # noqa: E402

OUT = ROOT / "bench/out"
LABELS = ROOT / "bench/labels/kb-985.txt"
TAGS = list(string.ascii_uppercase)


def kb() -> KnowledgeBase:
    return KnowledgeBase.from_sources([(ROOT / "kb/interview.md", "八股", True), (ROOT / "kb/projects.md", "项目", False)])


def slow_plans() -> dict[str, tuple[str, list[str]]]:
    """v4 回放里慢答改写出的独立问题和检索词。"""
    out = {}
    for block in (ROOT / "reports/v4-full.md").read_text().split("\n## ")[1:]:
        m = re.search(r"慢答改写（[^）]*）：(.*?)；查询 (\[.*?\])", block)
        if m:
            out[block[1:6]] = (m.group(1), json.loads(m.group(2).replace("'", '"')))
    return out


def cases():
    gold = gate_labels()
    plans = slow_plans()
    out = []
    for t in turns():
        if gold.get(t["at"]) != "Y":
            continue
        prior = " ".join(u.text for u in t["history"][-6:] if u.speaker == INTERVIEWER)
        question, queries = plans.get(t["at"], (t["text"], []))
        out.append({"at": t["at"], "text": t["text"], "prior": prior, "question": question, "queries": queries})
    return out


async def methods(base: KnowledgeBase, dense: DenseIndex, embedder: Embedder, judge: OpenAICompatible | None, c: dict) -> dict[str, tuple[list[int], float]]:
    """每种方法返回（排好序的切块 id，耗时 ms）。"""
    res = {}

    def timed(name, fn):
        t = time.monotonic()
        ids = fn()
        res[name] = (ids, (time.monotonic() - t) * 1000)

    timed("快·BM25", lambda: [h.chunk.id for h in base.search(c["text"], 10)])
    timed("快·BM25+上文", lambda: [h.chunk.id for h in base.search(c["text"], 10, context=c["prior"])])
    t = time.monotonic()
    [qv] = await embedder.embed([c["text"]])
    emb_ms = (time.monotonic() - t) * 1000
    res["快·向量"] = (dense.rank(qv, 10), emb_ms)
    t = time.monotonic()
    fused = rrf([[h.chunk.id for h in base.search(c["text"], 10, context=c["prior"])], dense.rank(qv, 10)], 10)
    res["快·RRF(BM25+上文, 向量)"] = (fused, emb_ms + (time.monotonic() - t) * 1000)
    qs = [*c["queries"], c["question"]]
    timed("慢·BM25 多路", lambda: [h.chunk.id for h in base.search_many(qs, 10)])
    t = time.monotonic()
    vecs = await embedder.embed(qs)
    rankings = [[h.chunk.id for h in base.search(q, 10)] for q in qs] + [dense.rank(v, 10) for v in vecs]
    res["慢·RRF 多路(现用)"] = (rrf(rankings, 10), (time.monotonic() - t) * 1000)
    if judge:
        for name, pool_name in (("快·BM25+上文→重排", "快·BM25+上文"), ("慢·BM25 多路→重排", "慢·BM25 多路")):
            cands = res[pool_name][0][:10]
            ids, ms = await rerank(base, judge, c["question"] if name.startswith("慢") else c["text"], c["prior"], cands)
            res[name] = (ids, res[pool_name][1] + ms)
    return res


async def rerank(base, judge, question, prior, cands) -> tuple[list[int], float]:
    """一次调用：候选编字母，读「哪条最能回答」的分布；模型提到的排前，其余按原顺序补齐。"""
    body = "\n\n".join(f"{TAGS[i]}. 【{base.chunks[cid].label}】{base.chunks[cid].body[:300]}" for i, cid in enumerate(cands))
    messages = [
        {"role": "system", "content": "下面是面试官的问题和若干知识库条目。哪一条最能帮候选人回答这个问题？只回答条目前的字母；都不相关回答 0。"},
        {"role": "user", "content": f"前文里面试官说过：{prior[-200:] or '（无）'}\n\n面试官的问题：{question}\n\n条目：\n{body}"},
    ]
    t = time.monotonic()
    for _ in range(20):
        try:
            r = await judge.client.post(judge.url, headers=judge.headers, json=judge._body(messages, 1, False) | {"temperature": 0, "logprobs": True, "top_logprobs": 20})
            if r.status_code == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    ms = (time.monotonic() - t) * 1000
    probs = {x["token"].strip(): math.exp(x["logprob"]) for x in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    scored = sorted(((probs.get(TAGS[i], 0.0), cid) for i, cid in enumerate(cands)), key=lambda x: -x[0])
    ranked = [cid for p, cid in scored if p > 0.02]
    return ranked + [cid for cid in cands if cid not in ranked], ms


def load_labels(base: KnowledgeBase) -> dict[str, set[int] | None]:
    by_label: dict[str, set[int]] = {}
    for c in base.chunks:  # 过长被切成几块的条目标签相同，几块都算
        by_label.setdefault(c.label, set()).add(c.id)
    out = {}
    for line in LABELS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        at, rest = line.split(" ", 1)
        names = [n.strip() for n in rest.split("||") if n.strip()]
        if names == ["无"]:
            out[at] = None
            continue
        missing = [n for n in names if n not in by_label]
        if missing:
            raise SystemExit(f"{at} 标注的条目不存在：{missing}")
        out[at] = set().union(*(by_label[n] for n in names))
    return out


async def main():
    base = kb()
    embedder = Embedder.from_env()
    dense = await DenseIndex.build(base, embedder, ROOT / "kb/.cache")
    if sys.argv[1] == "pool":
        lines = []
        for c in cases():
            res = await methods(base, dense, embedder, None, c)
            pool = []
            for ids, _ in res.values():
                for cid in ids[:5]:
                    if cid not in pool:
                        pool.append(cid)
            lines.append(f"### {c['at']} {c['text'][:90]}\n改写：{c['question']}\n" + "\n".join(f"  - {base.chunks[cid].label}" for cid in pool))
        (OUT / "kb-pool-985.md").write_text("\n\n".join(lines))
        print(OUT / "kb-pool-985.md")
        return
    judge = OpenAICompatible.from_env(sys.argv[2] if len(sys.argv) > 2 else "qwen3.8:27b")
    gold = load_labels(base)
    rows = []
    for c in cases():
        if c["at"] not in gold:
            continue
        rows.append((c, gold[c["at"]], await methods(base, dense, embedder, judge, c)))
    answerable = [r for r in rows if r[1]]
    print(f"有标准条目 {len(answerable)} 题，标「无」{len(rows) - len(answerable)} 题")
    print("| 方法 | hit@1 | hit@2 | hit@5 | MRR@10 | 延迟 p50 |\n|---|---|---|---|---|---|")
    for name in rows[0][2]:
        h1 = h2 = h5 = mrr = 0
        ms = sorted(r[2][name][1] for r in rows)
        for _, g, res in answerable:
            ids = res[name][0]
            rank = next((i for i, cid in enumerate(ids[:10]) if cid in g), None)
            h1 += rank is not None and rank < 1
            h2 += rank is not None and rank < 2
            h5 += rank is not None and rank < 5
            mrr += 1 / (rank + 1) if rank is not None else 0
        n = len(answerable)
        print(f"| {name} | {h1 / n:.0%} | {h2 / n:.0%} | {h5 / n:.0%} | {mrr / n:.2f} | {ms[len(ms) // 2]:.0f}ms |")


if __name__ == "__main__":
    asyncio.run(main())
