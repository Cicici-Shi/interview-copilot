"""logprob 判官 vs Jev：黑客松检索重排基准（会议问题 × BM25 前 30 候选）。

判官都给每个候选一个「有助于回答」的概率，按黑客松 run.ts 的规则排：≥0.4 留下，≥0.7 的排前面，
组内保持 BM25 顺序，取前 6。Jev 与 qwen-JSON 的分数取自黑客松缓存 judgements.json。

  python bench/search_rerank.py pointwise qwen3.8:27b   # 每个候选一次调用，读「yes/no」
  python bench/search_rerank.py listwise qwen3.8:27b    # 一次调用给 30 个候选，读「哪个最有用」的分布
"""

import argparse
import asyncio
import json
import math
import os
import re
import string
import time
from pathlib import Path

import httpx

HACK = Path.home() / "Desktop/git/personal_agent_long_context/eval/search/out"
OUT = Path(__file__).resolve().parent / "out"
TOP, KEEP, SURE = 6, 0.4, 0.7
JUDGE_QUESTION = "The record contains information that helps answer the question, about the people, project, incident or event the question refers to."
LABELS = list(string.ascii_uppercase) + ["a", "b", "c", "d"]  # 30 个候选各一个单 token 标签


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def load():
    rows = [r for r in json.loads((HACK / "results-meeting.json").read_text())]
    cache = json.loads((HACK / "judgements.json").read_text())
    chunks: dict[str, list[dict]] = {}
    for c in json.loads((HACK / "corpus.json").read_text()):
        chunks.setdefault(c["sourceId"], []).append(c)
    words = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))

    def card(question: str, source_id: str, i: int) -> dict:
        # 缓存只记了 sourceId；同一来源有多个切块时取和问题重词最多的那块（BM25 命中的多半就是它）
        best = max(chunks[source_id], key=lambda c: len(words(question) & words(c["title"] + " " + c["content"])))
        out = {"i": i, "id": source_id, "type": best["sourceType"], "title": clip(best["title"], 160), "excerpt": clip(re.sub(r"\s+", " ", best["content"]), 420)}
        if best.get("date"):
            out["date"] = best["date"]
        return out

    items = []
    for r in rows:
        q = r["item"]["question"]
        jev = [cache.get(f"jev\x00{q}\x00{s}") for s in r["pool"]]
        if any(v is None for v in jev):
            continue
        items.append({
            "id": r["item"]["id"], "question": q, "gold": r["item"]["gold"], "pool": r["pool"],
            "cards": [card(q, s, i) for i, s in enumerate(r["pool"])],
            "jev": jev, "llm": [cache.get(f"llm\x00{q}\x00{s}", 0.0) for s in r["pool"]],
        })
    return items


async def post(client, body):
    for attempt in range(10):
        started = time.monotonic()
        try:
            r = await client.post(os.environ["SOCLAAS_BASE_URL"] + "/chat/completions", headers={"authorization": "Bearer " + os.environ["SOCLAAS_API_KEY"]}, json=body)
        except httpx.HTTPError:
            await asyncio.sleep(1 + attempt)
            continue
        if r.status_code == 200 and "choices" in r.json():
            return r.json(), (time.monotonic() - started) * 1000
        await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"failed: {r.status_code} {r.text[:200]}")


def top_probs(data) -> dict[str, float]:
    return {x["token"].strip(): math.exp(x["logprob"]) for x in data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}


def base_body(model: str) -> dict:
    return {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}


async def pointwise(client, model, item, gate):
    async def one(c):
        async with gate:
            data, ms = await post(client, base_body(model) | {"messages": [
                {"role": "system", "content": f"Judge one statement about the state. Statement: {JUDGE_QUESTION} Reply with yes or no only."},
                {"role": "user", "content": json.dumps({"question": item["question"], "record": c}, ensure_ascii=False)},
            ]})
        p = top_probs(data)
        yes = sum(v for k, v in p.items() if k.lower() == "yes")
        no = sum(v for k, v in p.items() if k.lower() == "no")
        return (yes / (yes + no) if yes + no else 0.0), ms

    got = await asyncio.gather(*(one(c) for c in item["cards"]))
    # 逐条调用若全部并发，延迟 ≈ 最慢那条；这里记最慢那条，现实里受限流所限做不到
    return [g[0] for g in got], max(g[1] for g in got), sum(g[1] for g in got)


POOL = 30


async def listwise(client, model, item, gate):
    cards = [c | {"label": LABELS[i]} for i, c in enumerate(item["cards"][:POOL])]
    for c in cards:
        del c["i"]
    async with gate:
        data, ms = await post(client, base_body(model) | {"messages": [
            {"role": "system", "content": "Candidates are records, each with a label. Which one best contains information that helps answer the question, about the people, project, incident or event it refers to? If none of them does, answer 0. Reply with the label only."},
            {"role": "user", "content": json.dumps({"question": item["question"], "candidates": cards}, ensure_ascii=False)},
        ]})
    p = top_probs(data)
    scores = [p.get(LABELS[i], 0.0) for i in range(len(cards))] + [0.0] * (len(item["cards"]) - len(cards))
    return scores, ms, ms, p.get("0", 0.0)


def arrange(pool, scores, keep=KEEP, sure=SURE):
    first = [s for s, v in zip(pool, scores) if v >= sure]
    second = [s for s, v in zip(pool, scores) if keep <= v < sure]
    return first + second


def score(ranked, gold):
    top = ranked[:TOP]
    idx = next((i for i, s in enumerate(top) if s in set(gold)), -1)
    return (1 if idx >= 0 else 0), (1 / (idx + 1) if idx >= 0 else 0), (1 if not top else 0)


def report(items, arms):
    ans = [i for i in items if i["gold"]]
    absent = [i for i in items if not i["gold"]]
    lines = [f"可答 {len(ans)} 题，无答案 {len(absent)} 题"]
    for name, rank in arms.items():
        h = [score(rank(i), i["gold"]) for i in ans]
        e = [score(rank(i), []) for i in absent]
        lines.append(f"  {name:28s} hit@6 {sum(x[0] for x in h) / len(h):.0%}  MRR {sum(x[1] for x in h) / len(h):.2f}  可答题却空 {sum(x[2] for x in h)}  无答案题正确给空 {sum(x[2] for x in e)}/{len(e)}")
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pointwise", "listwise", "report"])
    ap.add_argument("model", nargs="?", default="qwen3.8:27b")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", type=int, default=30, help="listwise 只给前 N 个候选，量候选数对延迟的影响")
    args = ap.parse_args()
    items = load()
    if args.limit:
        items = items[: args.limit]
    OUT.mkdir(exist_ok=True)
    global POOL
    POOL = args.pool
    tag = f"{args.mode}_{args.model.replace(':', '_')}" + (f"_pool{args.pool}" if args.pool != 30 else "")
    if args.mode != "report":
        client = httpx.AsyncClient(timeout=90)
        gate = asyncio.Semaphore(args.concurrency)
        fn = pointwise if args.mode == "pointwise" else listwise
        res = []
        for n, item in enumerate(items):  # 题目之间串行，便于量单题延迟
            res.append(await fn(client, args.model, item, gate))
            if n % 20 == 19:
                print(f"  {n + 1}/{len(items)}", flush=True)
        saved = {i["id"]: list(r) for i, r in zip(items, res)}
        (OUT / f"search-{tag}.json").write_text(json.dumps(saved))
        ms = sorted(r[1] for r in res)
        print(f"{tag} 单题延迟 p50 {ms[len(ms) // 2]:.0f}ms p90 {ms[int(len(ms) * .9)]:.0f}ms")
    arms = {
        "bm25": lambda i: i["pool"],
        "bm25+jev（存档）": lambda i: arrange(i["pool"], i["jev"]),
        "bm25+qwen JSON（存档）": lambda i: arrange(i["pool"], i["llm"]),
    }
    for f in sorted(OUT.glob("search-*.json")):
        got = json.loads(f.read_text())
        name = f.stem.removeprefix("search-")
        if name.startswith("listwise"):
            # 分布加起来是 1，不能套 0.4/0.7：按概率排序；「0」（都不相关）概率过半就给空
            arms[name] = lambda i, g=got: [] if g[i["id"]][3] > 0.5 else [s for s, _ in sorted(zip(i["pool"], g[i["id"]][0]), key=lambda x: -x[1]) if _ > 0]
            # 补齐版：模型给了概率的排前面，其余按 BM25 原顺序接上，不因为模型没提到就丢
            arms[name + "+补齐"] = lambda i, g=got: [] if g[i["id"]][3] > 0.5 else (
                lambda ranked: ranked + [s for s in i["pool"] if s not in ranked]
            )([s for s, _ in sorted(zip(i["pool"], g[i["id"]][0]), key=lambda x: -x[1]) if _ > 0.02])
        else:
            arms[name] = lambda i, g=got: arrange(i["pool"], g[i["id"]][0])
    items = [i for i in items if all(f"{i['id']}" in json.loads(f.read_text()) for f in OUT.glob("search-*.json"))]
    print(report(items, arms))


if __name__ == "__main__":
    asyncio.run(main())
