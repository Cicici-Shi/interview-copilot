"""从一场实时日志抽「我说过实话之后面试官再开口」的边界，给新问题/补充判定做评测集。

  python -m bench.relation.extract reports/live-20261010-105744.jsonl --offset 158 > bench/relation/cases-1010.json
轮次切分用现行规则（附和不切轮）；每个边界给：前文、上一轮面试官原话、我在中间说的话、面试官新这一轮原话。
"""

import argparse
import json
import re

from copilot.assistant import BACKCHANNEL


def read_events(path):
    """实时日志 jsonl，或 copilot.replay 格式的逐字稿（**[mm:ss] 面试官**：……，来自千问转写）。"""
    if path.endswith(".jsonl"):
        yield from (json.loads(line) for line in open(path, encoding="utf-8"))
        return
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\*\*\[(\d+):(\d+)\]\s*(面试官|候选人)\*\*[：:]\s*(.*)$", line.strip())
        if m:
            yield {"t": int(m[1]) * 60 + int(m[2]), "type": "final", "speaker": "interviewer" if m[3] == "面试官" else "me", "text": m[4]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--start", type=float, default=0.0, help="日志里从第几秒算起（跳过开场前自测）")
    args = ap.parse_args()
    utts, turns, in_turn, mine_since, first_mine = [], [], False, [], 0.0
    for e in read_events(args.log):
        if e["t"] < args.start or e["type"] != "final":
            continue
        utts.append({"t": e["t"], "who": e["speaker"], "text": e["text"]})
        if e["speaker"] == "me":
            if not mine_since:
                first_mine = e["t"] - len(e["text"]) * 0.3 - 2  # 定稿时刻减估计时长，放音从这里起
            mine_since.append(e["text"])
            if len(BACKCHANNEL.sub("", e["text"])) >= 2:
                in_turn = False
            continue
        if not in_turn:
            turns.append({"t": e["t"], "idx": len(utts) - 1, "texts": [], "mine_before": mine_since,
                          "play_from": (first_mine if mine_since else e["t"] - 10)})
            mine_since = []
        turns[-1]["texts"].append(e["text"])
        in_turn = True
    cases = []
    for prev, cur in zip(turns, turns[1:]):
        s = round(cur["t"] - args.offset)
        cases.append({
            "id": f"{s // 60:02d}:{s % 60:02d}",
            "history": [f"{'面试官' if u['who'] == 'interviewer' else '候选人'}：{u['text']}" for u in utts[max(0, cur["idx"] - 30) : cur["idx"]]],
            "prev_turn": "".join(prev["texts"]),
            "mine_between": "".join(cur["mine_before"]),
            "new_turn": "".join(cur["texts"]),
            "play_from": round(max(0.0, cur["play_from"] - args.offset), 1),  # 录音时间，从我开口前一点放
        })
    json.dump(cases, __import__("sys").stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
