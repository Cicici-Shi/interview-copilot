"""用带时间戳的面试逐字稿回放，测问题识别、快答、慢答。

  python -m copilot.replay 逐字稿.md                # 每轮面试官的话单独跑一遍，出报告
  python -m copilot.replay 逐字稿.md --live --from 00:00 --to 03:00   # 按真实时间流式回放，看触发和取消
  --no-punct  去掉标点，模拟没有标点的实时 ASR 输出
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .assistant import INTERVIEWER, ME, Assistant, Event, Utterance
from .kb import DenseIndex, KnowledgeBase
from .llm import Embedder, OpenAICompatible

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^\*\*\[(\d+):(\d+)\]\s*(面试官|候选人)\*\*[：:]\s*(.*)$")


@dataclass
class Segment:
    start: float
    speaker: str
    text: str


def parse_transcript(path: Path, no_punct: bool) -> list[Segment]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        text = re.sub(r"\s*⚠️\s*", "", m.group(4)).strip()
        if no_punct:
            text = re.sub(r"[，。？！、；：,.?!;:—…“”\"（）()]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        speaker = INTERVIEWER if m.group(3) == "面试官" else ME
        out.append(Segment(int(m.group(1)) * 60 + int(m.group(2)), speaker, text))
    return out


def stamp(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


@dataclass
class Record:
    at: float
    question: str
    events: list[Event] = field(default_factory=list)

    def first(self, kind: str) -> Event | None:
        return next((e for e in self.events if e.kind == kind), None)


async def build(args) -> tuple[KnowledgeBase, DenseIndex | None, Embedder | None, OpenAICompatible, OpenAICompatible]:
    kb = KnowledgeBase.from_sources([(args.kb, "八股", True), (args.projects, "项目", False)])
    embedder = Embedder.from_env() if args.embed else None
    dense = await DenseIndex.build(kb, embedder, ROOT / "kb" / ".cache") if embedder else None
    fast = OpenAICompatible.from_env(args.fast_model)
    slow = OpenAICompatible.from_env(args.slow_model)
    gate = OpenAICompatible.from_env(args.gate_model)
    return kb, dense, embedder, fast, slow, gate


async def run_isolated(segments: list[Segment], args) -> list[Record]:
    """面试官每一轮（连续的面试官句子）当一个候选问题，历史是它之前的全部对话。"""
    kb, dense, embedder, fast, slow, gate = await build(args)
    turns: list[tuple[int, int]] = []  # (起, 止) 下标
    i = 0
    while i < len(segments):
        if segments[i].speaker == INTERVIEWER:
            j = i
            while j + 1 < len(segments) and segments[j + 1].speaker == INTERVIEWER:
                j += 1
            turns.append((i, j))
            i = j + 1
        else:
            i += 1
    lo, hi = parse_time(args.start), parse_time(args.end)
    turns = [t for t in turns if lo <= segments[t[0]].start < hi]
    gate = asyncio.Semaphore(args.concurrency)

    async def one(start: int, end: int) -> Record:
        text = "".join(s.text for s in segments[start : end + 1])
        record = Record(segments[start].start, text)
        async with gate:
            a = Assistant(kb, fast, slow, record.events.append, dense=dense, embedder=embedder, gate_model=gate)
            history = [Utterance(s.speaker, s.text) for s in segments[:start]]
            if len(text) >= 4:
                a.ask(text, history)
                await a.wait()
        return record

    return await asyncio.gather(*(one(s, e) for s, e in turns))


async def run_live(segments: list[Segment], args) -> None:
    """按逐字稿时间流式喂：句子开始时发「开始说话」，下一句开始时发这句的定稿。"""
    kb, dense, embedder, fast, slow, gate = await build(args)
    lo, hi = parse_time(args.start), parse_time(args.end)
    window = [s for s in segments if lo <= s.start < hi]
    t0 = time.monotonic()

    def show(e: Event) -> None:
        clock = stamp(lo + (time.monotonic() - t0) * args.speed)
        if e.kind in ("fast_token", "slow_token"):
            return
        text = e.text if e.kind != "kb_hit" else "命中 " + e.text
        print(f"  [{clock}] Q{e.qid} +{e.elapsed:.2f}s {e.kind:10s} {text[:160]}", flush=True)

    a = Assistant(kb, fast, slow, show, dense=dense, embedder=embedder, gate_model=gate)
    for k, seg in enumerate(window):
        end = window[k + 1].start if k + 1 < len(window) else seg.start + 3
        await asyncio.sleep(max(0.0, (seg.start - lo) / args.speed - (time.monotonic() - t0)))
        a.on_speech_start(seg.speaker)
        await asyncio.sleep(max(0.0, (end - lo) / args.speed - (time.monotonic() - t0)))
        print(f"[{stamp(seg.start)}] {'面试官' if seg.speaker == INTERVIEWER else '我'}：{seg.text}", flush=True)
        a.on_final(seg.speaker, seg.text)
    await asyncio.sleep(1.5)
    await a.wait()


def parse_time(value: str) -> float:
    m, s = value.split(":")
    return int(m) * 60 + int(s)


def pct(values: list[float], p: float) -> str:
    if not values:
        return "-"
    values = sorted(values)
    return f"{values[min(len(values) - 1, int(p * len(values)))]:.2f}s"


def report(records: list[Record], args) -> str:
    fired = [r for r in records if r.first("question")]
    fast_answered = [r for r in fired if r.first("fast_done")]
    slow_answered = [r for r in fired if r.first("slow_done")]
    kb_ms = [r.first("kb_hit").elapsed for r in fired if r.first("kb_hit")]
    fast_first = [r.first("fast_token").elapsed for r in fast_answered]
    fast_done = [r.first("fast_done").elapsed for r in fast_answered]
    plan = [r.first("slow_plan").elapsed for r in slow_answered]
    slow_first = [r.first("slow_token").elapsed for r in slow_answered]
    slow_done = [r.first("slow_done").elapsed for r in slow_answered]
    skip_disagree = [r for r in fired if bool(r.first("fast_skip")) != bool(r.first("slow_skip"))]
    lines = [
        f"# 回放报告：{Path(args.transcript).name}",
        "",
        f"快答模型 `{args.fast_model}`，慢答模型 `{args.slow_model}`，慢答检索 {'BM25 + bge-m3' if args.embed else 'BM25 多路'}，标点 {'去掉' if args.no_punct else '保留'}。",
        "",
        f"- 面试官轮次 {len(records)}，触发 {len(fired)}；快答作答 {len(fast_answered)}、判 SKIP {sum(1 for r in fired if r.first('fast_skip'))}；慢答作答 {len(slow_answered)}、判非问题 {sum(1 for r in fired if r.first('slow_skip'))}；快慢判断不一致 {len(skip_disagree)}",
        f"- 知识库命中出现：p50 {pct(kb_ms, .5)}",
        f"- 快答首字 p50 {pct(fast_first, .5)} / p90 {pct(fast_first, .9)}；写完 p50 {pct(fast_done, .5)} / p90 {pct(fast_done, .9)}",
        f"- 慢答改写完 p50 {pct(plan, .5)}；首字 p50 {pct(slow_first, .5)} / p90 {pct(slow_first, .9)}；写完 p50 {pct(slow_done, .5)} / p90 {pct(slow_done, .9)}",
        f"- 出错 {sum(1 for r in fired if r.first('error'))}",
        "",
    ]
    for r in records:
        lines.append(f"## [{stamp(r.at)}] {r.question}")
        if not r.first("question"):
            lines += ["太短，未触发", ""]
            continue
        hit = r.first("kb_hit")
        lines.append(f"- 快答命中：{hit.text if hit else '-'}")
        if r.first("fast_skip"):
            lines.append(f"- 快答：SKIP（+{r.first('fast_skip').elapsed:.2f}s）")
        elif r.first("fast_done"):
            f = r.first("fast_done")
            lines.append(f"- 快答（首字 +{r.first('fast_token').elapsed:.2f}s，完 +{f.elapsed:.2f}s）：")
            lines += [f"  > {line}" for line in f.text.splitlines() if line.strip()]
        if r.first("slow_skip"):
            lines.append(f"- 慢答：判为非问题 {r.first('slow_skip').text}")
        elif r.first("slow_plan"):
            p = r.first("slow_plan")
            lines.append(f"- 慢答改写（+{p.elapsed:.2f}s，{p.data.get('type')}）：{p.text}；查询 {p.data.get('queries')}")
            lines.append(f"- 慢答命中：{' | '.join(p.data.get('hits', []))}")
            d = r.first("slow_done")
            if d:
                lines.append(f"- 慢答（首字 +{r.first('slow_token').elapsed:.2f}s，完 +{d.elapsed:.2f}s）：")
                lines.append(f"  > {d.text}".replace("\n", "\n  > "))
        for e in r.events:
            if e.kind == "error":
                lines.append(f"- 出错：{e.text}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--kb", default=str(ROOT / "kb" / "interview.md"))
    ap.add_argument("--projects", default=str(ROOT / "kb" / "projects.md"))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--from", dest="start", default="00:00")
    ap.add_argument("--to", dest="end", default="99:59")
    ap.add_argument("--no-punct", action="store_true")
    ap.add_argument("--embed", action="store_true", help="慢答混入 bge-m3 向量（默认关，见 bench/retrieval.py）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--fast-model", default="qwen3.8:27b")
    ap.add_argument("--slow-model", default="qwen3.8:27b")
    ap.add_argument("--gate-model", default="qwen3.8:27b")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    segments = parse_transcript(Path(args.transcript), args.no_punct)
    if args.live:
        asyncio.run(run_live(segments, args))
        return
    records = asyncio.run(run_isolated(segments, args))
    out = Path(args.out or ROOT / "reports" / f"replay-{time.strftime('%m%d-%H%M')}{'-nopunct' if args.no_punct else ''}.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(report(records, args), encoding="utf-8")
    print(out)
    print("\n".join(report(records, args).splitlines()[2:9]))


if __name__ == "__main__":
    main()
