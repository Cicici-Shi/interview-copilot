"""「要不要答」门控对比，以人工标签 bench/labels/gate-985.txt 为准（A 类不计分）。

  python bench/gate.py run qwen3.6:35b      # 现跑 logprob 门控并存结果
  python bench/gate.py report               # 汇总所有方案
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from copilot.assistant import GATE_SYSTEM, INTERVIEWER, Utterance, render_history  # noqa: E402
from copilot.llm import OpenAICompatible  # noqa: E402
from copilot.replay import parse_transcript  # noqa: E402

TRANSCRIPT = Path.home() / "Downloads/面试视频逐字稿-985本硕自学一年.md"
OUT = ROOT / "bench/out"
QUESTION_RULE = re.compile(r"[?？]|吗|呢|什么|怎么|为什么|哪|有没有|是不是|能不能|会不会|讲一下|说一下|介绍|展开|举个例子")


def stamp(s: float) -> str:
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def turns(no_punct: bool = False):
    segs = parse_transcript(TRANSCRIPT, no_punct)
    out, i = [], 0
    while i < len(segs):
        if segs[i].speaker != INTERVIEWER:
            i += 1
            continue
        j = i
        while j + 1 < len(segs) and segs[j + 1].speaker == INTERVIEWER:
            j += 1
        out.append({"at": stamp(segs[i].start), "text": "".join(s.text for s in segs[i : j + 1]), "history": [Utterance(s.speaker, s.text) for s in segs[:i]]})
        i = j + 1
    return out


def labels() -> dict[str, str]:
    out = {}
    for line in (ROOT / "bench/labels/gate-985.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            at, lab, *_ = line.split()
            out[at] = lab
    return out


def from_report(path: Path, skip_marker: str) -> dict[str, bool]:
    """回放报告里某路是否作答（True=答）。"""
    out = {}
    for block in path.read_text().split("\n## ")[1:]:
        at = block[1:6]
        out[at] = skip_marker not in block and "太短，未触发" not in block
    return out


async def run(model_name: str) -> None:
    model = OpenAICompatible.from_env(model_name)
    rows = {}
    for t in turns():
        messages = [
            {"role": "system", "content": GATE_SYSTEM},
            {"role": "user", "content": f"前文：\n{render_history(t['history'][-4:]) or '（无）'}\n\n面试官刚说：{t['text']}"},
        ]
        for attempt in range(20):  # 批量跑时配额会被打满，等久一点；只记成功那次的耗时
            started = time.monotonic()
            try:
                p = await model.yes_prob(messages)
                break
            except Exception:
                await asyncio.sleep(10)
        else:
            continue
        rows[t["at"]] = {"p": p, "ms": round((time.monotonic() - started) * 1000)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"gate-lp-{model_name.replace(':', '_')}.json").write_text(json.dumps(rows))
    print("done", model_name)


def score(name: str, answered: dict[str, bool], gold: dict[str, str], ms: list[int] | None = None) -> str:
    ys = [at for at, g in gold.items() if g == "Y" and at in answered]
    ns = [at for at, g in gold.items() if g == "N" and at in answered]
    miss = [at for at in ys if not answered[at]]
    extra = [at for at in ns if answered[at]]
    lat = f"p50 {sorted(ms)[len(ms) // 2]}ms p90 {sorted(ms)[int(len(ms) * .9)]}ms" if ms else "-"
    return f"| {name} | {len(miss)}/{len(ys)} | {len(extra)}/{len(ns)} | {lat} |"


def report() -> None:
    gold = labels()
    lines = ["| 方案 | 漏答真问题 | 多答非问题 | 延迟 |", "|---|---|---|---|"]
    for no_punct in (False, True):
        rule = {t["at"]: bool(QUESTION_RULE.search(t["text"])) for t in turns(no_punct)}
        lines.append(score(f"规则（疑问词/问号{'，去标点' if no_punct else ''}）", rule, gold))
    lines.append(score("快答首 token SKIP（v2）", from_report(ROOT / "reports/v2-full.md", "快答：SKIP"), gold))
    lines.append(score("慢答 JSON 判断（v4）", from_report(ROOT / "reports/v4-full.md", "慢答：判为非问题"), gold))
    for f in sorted(OUT.glob("gate-lp-*.json")):
        rows = json.loads(f.read_text())
        ms = [r["ms"] for r in rows.values()]
        for th in (0.1, 0.2, 0.35, 0.5):
            lines.append(score(f"logprob {f.stem.removeprefix('gate-lp-')} 阈值 {th}", {at: r["p"] >= th for at, r in rows.items()}, gold, ms))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        asyncio.run(run(sys.argv[2]))
    else:
        report()
