"""把单声道面试录音按千问转写的说话人切成两路，给 copilot.live 回放用。

  python3 -m copilot.split_dual "~/Library/Mobile Documents/com~apple~CloudDocs/Recording 15.m4a" \\
      ~/Desktop/git/interview-vault/transcripts/2026-09-24_binance-final-interview.md --interviewer 1 --name binance
  → reports/.tmp/binance-interviewer.wav / binance-me.wav / binance.md（逐字稿，copilot.replay 的格式）

千问转写每段只有开始时间（`发言人1   00:24`），一段算到下一段开始为止，边界有 ±1 秒误差：
会截掉个别句尾或漏进对方的话。另一方说话的时段填静音，模拟会议里两路分开采的样子。
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = re.compile(r"^发言人(\d+)\s+(?:(\d+):)?(\d+):(\d+)\s*$")
SAMPLE_RATE = 16000


def parse(path: Path) -> list[tuple[float, str, str]]:
    """[(开始秒, 发言人编号, 文本)]"""
    out: list[tuple[float, str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        m = HEAD.match(line.strip())
        if not m:
            continue
        h, mi, s = int(m.group(2) or 0), int(m.group(3)), int(m.group(4))
        text = next((x.strip() for x in lines[i + 1 :] if x.strip()), "")
        out.append((h * 3600 + mi * 60 + s, m.group(1), text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("transcript")
    ap.add_argument("--interviewer", required=True, help="转写里面试官是发言人几（看转写稿头部说明）")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    segments = parse(Path(args.transcript).expanduser())
    out = ROOT / "reports" / ".tmp"
    out.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(Path(args.audio).expanduser()), "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"],
        check=True, capture_output=True,
    ).stdout
    tracks = {"interviewer": bytearray(len(raw)), "me": bytearray(len(raw))}
    for i, (start, who, _) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < len(segments) else len(raw) / 2 / SAMPLE_RATE
        a, b = int(start * SAMPLE_RATE) * 2, min(len(raw), int(end * SAMPLE_RATE) * 2)
        track = tracks["interviewer" if who == args.interviewer else "me"]
        track[a:b] = raw[a:b]
    for key, pcm in tracks.items():
        wav = out / f"{args.name}-{key}.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-", str(wav)],
            input=bytes(pcm), check=True,
        )
        print(wav)
    md = out / f"{args.name}.md"
    md.write_text(
        "\n".join(
            f"**[{int(t) // 60:02d}:{int(t) % 60:02d}] {'面试官' if who == args.interviewer else '候选人'}**：{text}"
            for t, who, text in segments
        ),
        encoding="utf-8",
    )
    print(md, f"{len(segments)} 段")


if __name__ == "__main__":
    main()
