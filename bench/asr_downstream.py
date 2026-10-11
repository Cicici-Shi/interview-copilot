"""ASR 文本上的下游效果：同一批人工标签，比较「人工整理的逐字稿」和「豆包实时转写」喂给门控和检索的结果。

  python bench/asr_downstream.py reports/asr-985-full-D.jsonl
"""

import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from copilot.assistant import GATE_SYSTEM, GATE_THRESHOLD, INTERVIEWER, ME, PLAN_SYSTEM, Utterance, render_history  # noqa: E402
from copilot.llm import OpenAICompatible  # noqa: E402
from gate import TRANSCRIPT, labels as gate_labels  # noqa: E402
from retrieval import kb as load_kb, load_labels  # noqa: E402

LINE = re.compile(r"^\*\*\[(\d+):(\d+)\]\s*(面试官|候选人)\*\*[：:]\s*(.*)$")


def reference():
    out = []
    for line in TRANSCRIPT.read_text().splitlines():
        m = LINE.match(line.strip())
        if m:
            out.append((int(m.group(1)) * 60 + int(m.group(2)), INTERVIEWER if m.group(3) == "面试官" else ME))
    return out


def asr_finals(path: Path):
    """定稿句，按 FinalTracker 去掉「合并重放」的重复句（早期跑的 jsonl 没去重，新的去过也不影响）。"""
    from copilot.doubao import FinalTracker

    tracker = FinalTracker()
    finals = [json.loads(l) for l in path.read_text().splitlines()]
    return [f for f in finals if f["kind"] == "final" and tracker.should_emit(f.get("speaker"), f["start_ms"], f["end_ms"])]


def oracle_speakers(finals, ref):
    """模拟双声道：说话人不靠服务端分离，按时间和逐字稿的说话人区间重叠最多的那个定。
    逐字稿只有每句开始秒数，区间取到下一句开始。"""
    spans = [(s, ref[i + 1][0] if i + 1 < len(ref) else s + 10, who) for i, (s, who) in enumerate(ref)]
    for f in finals:
        a, b = f["start_ms"] / 1000, f["end_ms"] / 1000
        overlap: Counter = Counter()
        for s, e, who in spans:
            overlap[who] += max(0.0, min(b, e) - max(a, s))
        f["speaker"] = "oracle-" + (overlap.most_common(1)[0][0] if overlap else ME)
    return finals


def map_speakers(finals, ref):
    """每句 ASR 按开始时间找逐字稿里最近的一句，按说话人编号多数投票定是谁。"""
    votes: dict[str, Counter] = defaultdict(Counter)
    for f in finals:
        t = f["start_ms"] / 1000
        near = min(ref, key=lambda r: abs(r[0] - t))
        votes[f.get("speaker") or "?"][near[1]] += 1
    return {spk: c.most_common(1)[0][0] for spk, c in votes.items()}, votes


def asr_turns(finals, who):
    turns, cur = [], None
    for i, f in enumerate(finals):
        role = who[f.get("speaker") or "?"]
        if role == INTERVIEWER:
            if cur is None:
                cur = {"start": f["start_ms"] / 1000, "end": f["end_ms"] / 1000, "text": f["text"], "first": i}
            else:
                cur["end"], cur["text"] = f["end_ms"] / 1000, cur["text"] + f["text"]
        elif cur is not None:
            turns.append(cur)
            cur = None
    if cur:
        turns.append(cur)
    history = lambda t: [Utterance(who[f.get("speaker") or "?"], f["text"]) for f in finals[: t["first"]]]
    return turns, history


def gaps(finals, who):
    """面试官同一轮内相邻两句的停顿（下一句开始 − 上一句结束），决定静音阈值能不能再短。"""
    inner = []
    for a, b in zip(finals, finals[1:]):
        if who[a.get("speaker") or "?"] == INTERVIEWER and who[b.get("speaker") or "?"] == INTERVIEWER:
            inner.append((b["start_ms"] - a["end_ms"]) / 1000)
    return sorted(inner)


async def main():
    path = Path(sys.argv[1])
    ref = reference()
    if "--dual" in sys.argv:
        # 双声道：面试官声道、候选人声道分别转写，说话人由声道决定，按开始时间合并
        mine = Path(sys.argv[sys.argv.index("--dual") + 1])
        finals = sorted(
            [f | {"speaker": "ch-interviewer"} for f in asr_finals(path)] + [f | {"speaker": "ch-me"} for f in asr_finals(mine)],
            key=lambda f: f["start_ms"],
        )
    else:
        finals = asr_finals(path)
    if "--oracle" in sys.argv:
        finals = oracle_speakers(finals, ref)
    who, votes = map_speakers(finals, ref)
    if "--dual" in sys.argv:
        who = {"ch-interviewer": INTERVIEWER, "ch-me": ME}
    print("说话人映射：", {k: (v, dict(votes[k])) for k, v in who.items()})
    turns, history = asr_turns(finals, who)
    g = gaps(finals, who)
    if g:
        print(f"面试官同轮内停顿 {len(g)} 处：p10 {g[len(g) // 10]:.2f}s p50 {g[len(g) // 2]:.2f}s p90 {g[int(len(g) * .9)]:.2f}s；"
              f"≤0.5s {sum(x <= .5 for x in g)}，≤0.8s {sum(x <= .8 for x in g)}")

    gold = gate_labels()
    base = load_kb()
    kbgold = load_labels(base)

    def to_sec(at):
        m, s = at.split(":")
        return int(m) * 60 + int(s)

    # 每个人工标签对到开始时间最接近的 ASR 面试官轮次（3 秒内），对不上记为 ASR 漏掉
    aligned = {}
    for at, lab in gold.items():
        if lab == "A":
            continue
        cand = [t for t in turns if abs(t["start"] - to_sec(at)) <= 3 or t["start"] - 1 <= to_sec(at) <= t["end"]]
        aligned[at] = min(cand, key=lambda t: abs(t["start"] - to_sec(at))) if cand else None
    lost = [at for at, t in aligned.items() if t is None]
    print(f"人工标签 {len(aligned)} 条，ASR 里找不到对应面试官轮次 {len(lost)} 条：{lost[:10]}")

    gate_model = OpenAICompatible.from_env("qwen3.8:27b")
    plan_model = OpenAICompatible.from_env("qwen3.8:27b")

    async def retry(fn):
        for _ in range(20):
            try:
                return await fn()
            except Exception:
                await asyncio.sleep(8)
        raise RuntimeError("retry exhausted")

    miss = extra = 0
    fast_hit = slow_hit = n_kb = 0
    rows = []
    for at, t in aligned.items():
        lab = gold[at]
        if t is None:
            miss += lab == "Y"
            if kbgold.get(at):
                n_kb += 1
            continue
        hist = history(t)
        p = await retry(lambda: gate_model.yes_prob([
            {"role": "system", "content": GATE_SYSTEM},
            {"role": "user", "content": f"前文：\n{render_history(hist[-4:]) or '（无）'}\n\n面试官刚说：{t['text']}"},
        ]))
        answered = p >= GATE_THRESHOLD
        miss += lab == "Y" and not answered
        extra += lab == "N" and answered
        if kbgold.get(at):
            n_kb += 1
            prior = " ".join(u.text for u in hist[-6:] if u.speaker == INTERVIEWER)
            fh = {h.chunk.id for h in base.search(t["text"], 3, context=prior)}
            fast_hit += bool(fh & kbgold[at])
            plan = await retry(lambda: plan_model.json([
                {"role": "system", "content": PLAN_SYSTEM.replace("{outline}", base.outline())},
                {"role": "user", "content": f"对话：\n{render_history(hist[-14:]) or '（无）'}\n\n面试官最后这段话：{t['text']}"},
            ], max_tokens=200))
            qs = [str(x) for x in plan.get("queries") or []][:4] + [str(plan.get("question") or t["text"])]
            sh = {h.chunk.id for h in base.search_many(qs, 5)}
            slow_hit += bool(sh & kbgold[at])
        rows.append((at, lab, round(p, 2), t["text"][:60]))
    ys = sum(1 for at in aligned if gold[at] == "Y")
    ns = sum(1 for at in aligned if gold[at] == "N")
    print(f"门控（27b，阈值 {GATE_THRESHOLD}）：漏答 {miss}/{ys}，多答 {extra}/{ns}")
    print(f"快答检索 前 3 条命中 {fast_hit}/{n_kb} = {fast_hit / n_kb:.0%}；慢答 前 5 条命中 {slow_hit}/{n_kb} = {slow_hit / n_kb:.0%}")
    (ROOT / "bench/out" / f"asr-downstream-{path.stem}{'-oracle' if '--oracle' in sys.argv else ''}{'-dual' if '--dual' in sys.argv else ''}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
