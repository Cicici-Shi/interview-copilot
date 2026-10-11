"""评估豆包 ASR 回放结果：字错率（CER）、技术术语命中、说话人分离准确率、定稿延迟。

  python -m copilot.asr_eval reports/asr-985-full-sp-ew300.jsonl reports/asr-985-full-sp-ew100.jsonl reports/asr-985-full-nosp-ew300.jsonl \
      --transcript ~/Downloads/面试视频逐字稿-985本硕自学一年.md --out reports/asr-eval.md

每个 jsonl 是 asr_replay.py 的输出（第一行是 config，其余是 partial/final/error）。
逐字稿本身是「音频自动转写后人工校正」得到的（见文件头说明），CER 是和这份参照比，
不是和真实音频内容比，报告里会注明这一点。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .doubao import FinalTracker

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^\*\*\[(\d+):(\d+)\]\s*(面试官|候选人)\*\*[：:]\s*(.*)$")

# CER 前先去掉的东西：括号补字（转写者自己按上下文猜的，不是原声）、⚠️ 标记本身、
# 各种中英文标点和空白。字符级切分对中文合理，但整段英文/数字（Kafka、offset、0.75）
# 按一个词算一个 token，不然一个词错一个字母就要扣好几次。
PAREN = re.compile(r"[（(][^（）()]*[）)]")
WARN = re.compile(r"⚠️")
PUNCT = re.compile(r"[\s，。？！、；：,.?!;:—–\-…“”\"'‘’《》〈〉·\[\]【】\n]")
TOKEN = re.compile(r"[A-Za-z0-9]+|[一-鿿]")

# 术语表：题目里给的那批 + 逐字稿里其余高频技术词；最终只保留在逐字稿里真出现过的
TERM_CANDIDATES = [
    "Kafka", "ZSet", "HashMap", "consumer", "group", "topic", "offset", "Redis", "JVM",
    "幂等", "限流", "滑动窗口", "红黑树", "负载因子", "TCP", "UDP", "JDK", "put", "hash",
    "哈希表", "扩容", "消息队列", "削峰", "分区", "副本",
]


def normalize(text: str) -> str:
    text = PAREN.sub("", text)
    text = WARN.sub("", text)
    text = PUNCT.sub("", text)
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


def count_term(text: str, term: str) -> int:
    """英文/数字术语按词边界数，避免 hash 把 HashMap 里的 Hash 也算进去；中文直接数子串。"""
    if re.fullmatch(r"[A-Za-z0-9]+", term):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        return len(pattern.findall(text))
    return text.count(term)


def edit_distance(a: list[str], b: list[str]) -> int:
    """标准 Levenshtein，双行滚动省内存（转写文本长，全矩阵放不下也没必要）。"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


@dataclass
class RefSegment:
    start: float
    speaker: str  # 面试官 | 候选人
    text: str


def parse_transcript(path: Path) -> list[RefSegment]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        out.append(RefSegment(int(m.group(1)) * 60 + int(m.group(2)), m.group(3), m.group(4)))
    return out


@dataclass
class Run:
    name: str
    config: dict
    events: list[dict]

    @property
    def finals(self) -> list[dict]:
        return [e for e in self.events if e["kind"] == "final"]

    @property
    def partials(self) -> list[dict]:
        return [e for e in self.events if e["kind"] == "partial" and e["text"]]

    @property
    def errors(self) -> list[dict]:
        return [e for e in self.events if e["kind"] == "error"]

    @property
    def full_text(self) -> str:
        return "".join(f["text"] for f in self.finals)


def rededuplicate_finals(events: list[dict]) -> list[dict]:
    """doubao.py 里 (start_ms, end_ms) 精确去重的旧版本有个漏洞：enable_nonstream 有时会把
    同一说话人已经出过 final 的好几句合并重放成一条跨度更大的新分句，精确去重漏判（时间戳区间
    变了），这批 985-full-* 数据就是用旧版本跑的。这里按 FinalTracker 现在的重叠判断规则，
    照原始到达顺序（jsonl 里本来就是按到达顺序写的）重放一遍，去掉那些重叠重放，不用重新跑
    一遍音频（豆包按时长计费）。之后重新用新版 doubao.py 采集的数据跑这个函数是幂等的
    （不会有重叠的 final，等于不丢东西）。"""
    tracker = FinalTracker()
    out = []
    for e in events:
        if e["kind"] != "final":
            out.append(e)
            continue
        if tracker.should_emit(e.get("speaker"), e["start_ms"], e["end_ms"]):
            out.append(e)
    return out


def load_run(path: Path) -> Run:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    config = events[0] if events and events[0]["kind"] == "config" else {}
    events = rededuplicate_finals(events)
    return Run(path.stem.removeprefix("asr-"), config, events)


def cer(run: Run, ref_text: str) -> tuple[float, int, int]:
    ref_tokens = tokenize(normalize(ref_text))
    hyp_tokens = tokenize(normalize(run.full_text))
    dist = edit_distance(hyp_tokens, ref_tokens)
    return dist / len(ref_tokens), dist, len(ref_tokens)


def term_table(runs: list[Run], ref_text: str) -> list[tuple[str, int, list[int]]]:
    terms = [t for t in TERM_CANDIDATES if count_term(ref_text, t) > 0]
    rows = []
    for term in terms:
        ref_count = count_term(ref_text, term)
        run_counts = [count_term(run.full_text, term) for run in runs]
        rows.append((term, ref_count, run_counts))
    return rows


def speaker_accuracy(run: Run, ref_segments: list[RefSegment]) -> tuple[dict[str, str], float, float, int] | None:
    """把 final 按时间中点对齐到逐字稿的说话人区间，找 speaker_id -> 面试官/候选人 的最佳映射，
    分别按句数和按时长算准确率（多数投票，同一批 id 允许多对一）。"""
    finals = [f for f in run.finals if f.get("speaker") is not None]
    if not finals:
        return None
    starts = [s.start for s in ref_segments]

    def true_label(f: dict) -> str | None:
        mid = (f["start_ms"] + f["end_ms"]) / 2000
        i = bisect_right(starts, mid) - 1
        return ref_segments[i].speaker if i >= 0 else None

    by_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_dur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for f in finals:
        label = true_label(f)
        if label is None:
            continue
        dur = max(0, f["end_ms"] - f["start_ms"]) / 1000
        by_count[f["speaker"]][label] += 1
        by_dur[f["speaker"]][label] += dur

    mapping = {sid: max(counts, key=counts.get) for sid, counts in by_count.items()}
    correct_n = sum(by_count[sid][mapping[sid]] for sid in mapping)
    total_n = sum(sum(c.values()) for c in by_count.values())
    correct_dur = sum(by_dur[sid][mapping[sid]] for sid in mapping)
    total_dur = sum(sum(c.values()) for c in by_dur.values())
    return mapping, correct_n / total_n if total_n else 0.0, correct_dur / total_dur if total_dur else 0.0, total_n


def pct(values: list[float], p: float) -> str:
    if not values:
        return "-"
    values = sorted(values)
    return f"{values[min(len(values) - 1, int(p * len(values)))]:.2f}s"


def finalize_latency(run: Run) -> list[float]:
    """本地收到 final 的时刻 - 该句 end_ms 换算成的本地时钟时刻（都相对推流开始，speed 已换算）。"""
    speed = run.config.get("speed", 1.0) or 1.0
    return [f["elapsed_s"] - f["end_ms"] / 1000 / speed for f in run.finals]


def partial_first_delay(run: Run) -> list[float]:
    """每句 final 对应的第一个 partial，相对这句音频开始被发送时刻的延迟。"""
    speed = run.config.get("speed", 1.0) or 1.0
    finals = sorted(run.finals, key=lambda f: f["elapsed_s"])
    partials = sorted(run.partials, key=lambda p: p["elapsed_s"])
    delays = []
    prev_elapsed = 0.0
    pi = 0
    for f in finals:
        sentence_start_wall = f["start_ms"] / 1000 / speed
        # 这句 final 收到之前、上一句 final 收到之后出现的第一个非空 partial
        while pi < len(partials) and partials[pi]["elapsed_s"] <= prev_elapsed:
            pi += 1
        if pi < len(partials) and partials[pi]["elapsed_s"] <= f["elapsed_s"]:
            delays.append(partials[pi]["elapsed_s"] - sentence_start_wall)
        prev_elapsed = f["elapsed_s"]
    return [d for d in delays if d >= -0.05]  # 允许一点时钟误差，明显负的丢掉（对齐错了）


def report(runs: list[Run], ref_segments: list[RefSegment], ref_text: str, transcript_path: Path) -> str:
    lines = [
        "# 豆包流式 ASR 评估报告",
        "",
        f"逐字稿：`{transcript_path}`（音频自动转写后人工校正，本身不是绝对真值，CER 是相对这份参照算的）。",
        "",
        "## 配置概览",
        "",
        "| 配置 | speakers | end_window_ms | speed | final 句数 | 错误 |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        c = r.config
        lines.append(f"| {r.name} | {c.get('speakers')} | {c.get('end_window_ms')} | {c.get('speed')} | {len(r.finals)} | {len(r.errors)} |")

    lines += ["", "## 字错率 CER（去标点、去空格、去逐字稿括号补字后，按字符/英文单词 token 算编辑距离）", "",
               "| 配置 | CER | 编辑距离 | 参照 token 数 |", "|---|---|---|---|"]
    for r in runs:
        rate, dist, n = cer(r, ref_text)
        lines.append(f"| {r.name} | {rate:.1%} | {dist} | {n} |")

    lines += ["", "## 技术术语命中（逐字稿里出现过的术语，次数对比）", "",
               "| 术语 | 逐字稿 | " + " | ".join(r.name for r in runs) + " |",
               "|---|---|" + "---|" * len(runs)]
    for term, ref_count, run_counts in term_table(runs, ref_text):
        lines.append(f"| {term} | {ref_count} | " + " | ".join(str(c) for c in run_counts) + " |")

    lines += ["", "## 说话人分离准确率（按时间中点对齐逐字稿说话人，多数投票找 speaker_id 映射）", "",
               "| 配置 | 映射 | 按句数准确率 | 按时长准确率 | 参与判定的 final 句数 |", "|---|---|---|---|---|"]
    for r in runs:
        result = speaker_accuracy(r, ref_segments)
        if result is None:
            lines.append(f"| {r.name} | （未开分离或无 speaker 字段） | - | - | 0 |")
            continue
        mapping, acc_n, acc_dur, total_n = result
        mapping_str = "、".join(f"{k}→{v}" for k, v in sorted(mapping.items()))
        lines.append(f"| {r.name} | {mapping_str} | {acc_n:.1%} | {acc_dur:.1%} | {total_n} |")

    lines += ["", "## 定稿延迟：本地收到 final 时刻 − 该句音频 end_ms（换算到本地时钟）", "",
               "| 配置 | p50 | p90 | 样本数 |", "|---|---|---|---|"]
    for r in runs:
        lat = finalize_latency(r)
        lines.append(f"| {r.name} | {pct(lat, .5)} | {pct(lat, .9)} | {len(lat)} |")

    lines += ["", "## partial 首次出现延迟：相对这句音频开始被推流的时刻", "",
               "| 配置 | p50 | p90 | 样本数 |", "|---|---|---|---|"]
    for r in runs:
        delay = partial_first_delay(r)
        lines.append(f"| {r.name} | {pct(delay, .5)} | {pct(delay, .9)} | {len(delay)} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports" / "asr-eval.md"))
    args = ap.parse_args()
    runs = [load_run(Path(p)) for p in args.jsonl]
    transcript_path = Path(args.transcript).expanduser()
    ref_segments = parse_transcript(transcript_path)
    ref_text = "".join(s.text for s in ref_segments)
    out = report(runs, ref_segments, ref_text, transcript_path)
    Path(args.out).write_text(out, encoding="utf-8")
    print(args.out)
    print("\n".join(out.splitlines()[:4]))


if __name__ == "__main__":
    main()
