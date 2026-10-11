"""用真实录音按时间节奏喂给豆包流式识别，记录 partial/final 事件测延迟和识别质量。

  python -m copilot.asr_replay 音频.mp3 --name short --start 0 --duration 90 --speed 2 --speakers
  python -m copilot.asr_replay 音频.mp3 --name full-800 --end-window 800 --speakers

先用 ffmpeg 转成 16kHz 单声道 s16le，按真实时间节奏（每 200ms 推 200ms 音频，--speed 倍速
只压缩推流之间的等待，不改变每块音频的时长）推给豆包；每个 partial/final 事件带上本地时钟
时间戳（相对推流开始的秒数）写成 reports/asr-<name>.jsonl；final 再存一份 markdown，格式和
人工逐字稿一致（`**[mm:ss] 说话人X**：文本`），方便直接比对。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

from .doubao import DoubaoAsrClient, Final, Partial

ROOT = Path(__file__).resolve().parent.parent
CHUNK_MS = 200
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_MS // 1000


def stamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def extract_pcm(source: Path, raw_path: Path, start: float, duration: float | None) -> None:
    """ffmpeg 转 16kHz 单声道 s16le，落一个临时原始 PCM 文件，方便按固定块顺序读。"""
    cmd = ["ffmpeg", "-y", "-i", str(source)]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", str(raw_path)]
    subprocess.run(cmd, check=True, capture_output=True)


async def _sender(client: DoubaoAsrClient, raw_path: Path, speed: float, t0: float) -> None:
    """按真实时间节奏顺序读块、发送；--speed 只压缩发送间隔，块本身仍是 200ms 音频。"""
    with raw_path.open("rb") as f:
        sent_ms = 0
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            await client.send(chunk)
            sent_ms += CHUNK_MS
            target = t0 + sent_ms / 1000 / speed
            await asyncio.sleep(max(0.0, target - time.monotonic()))
    await client.finish()


async def run(args: argparse.Namespace) -> Path:
    source = Path(args.audio).expanduser()
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else ROOT / "reports" / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_dir / f"asr-replay-{args.name}.raw"
    extract_pcm(source, raw_path, args.start, args.duration)

    options = {}
    if args.hotwords or args.scene:
        from .hotwords import asr_options
        from .kb import KnowledgeBase

        kb = KnowledgeBase.from_sources([(ROOT / "kb/interview.md", "八股", True), (ROOT / "kb/projects.md", "项目", False)])
        options = asr_options(kb, args.hotwords, args.scene)
    client = DoubaoAsrClient.from_env(speakers=args.speakers, end_window_ms=args.end_window, ddc=args.ddc, **options)
    await client.connect()

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / f"asr-{args.name}.jsonl"
    md_path = out_dir / f"asr-{args.name}.md"

    events: list[dict] = [
        {
            "kind": "config",
            "elapsed_s": 0.0,
            "speed": args.speed,
            "speakers": args.speakers,
            "end_window_ms": args.end_window,
            "start_offset_s": args.start,
            "duration_s": args.duration,
            "source": source.name,
        }
    ]
    t0 = time.monotonic()

    async def receiver() -> None:
        async for event in client.events():
            elapsed = round(time.monotonic() - t0, 3)
            if isinstance(event, Partial):
                events.append({"kind": "partial", "elapsed_s": elapsed, "text": event.text})
            elif isinstance(event, Final):
                events.append(
                    {
                        "kind": "final",
                        "elapsed_s": elapsed,
                        "text": event.text,
                        "start_ms": event.start_ms,
                        "end_ms": event.end_ms,
                        "speaker": event.speaker,
                    }
                )

    receiver_task = asyncio.create_task(receiver())
    try:
        await _sender(client, raw_path, args.speed, t0)
    except Exception as e:  # 服务端错误帧会从 send/events 里冒出来；先记下来再往下走
        events.append({"kind": "error", "elapsed_s": round(time.monotonic() - t0, 3), "message": repr(e)})
    # finish() 已发出去，给服务端一点时间把最后几句定稿
    await asyncio.sleep(args.tail_wait)
    await client.close()
    try:
        await receiver_task
    except Exception as e:
        events.append({"kind": "error", "elapsed_s": round(time.monotonic() - t0, 3), "message": repr(e)})

    jsonl_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")

    finals = [e for e in events if e["kind"] == "final"]
    lines = []
    for f in finals:
        when = stamp(args.start + f["start_ms"] / 1000)
        who = f"说话人{f['speaker']}" if f.get("speaker") is not None else "说话人?"
        lines.append(f"**[{when}] {who}**：{f['text']}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    raw_path.unlink(missing_ok=True)
    print(f"{jsonl_path}\n{md_path}")
    print(f"final 句数 {len(finals)}，错误 {sum(1 for e in events if e['kind'] == 'error')}")
    return jsonl_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--name", required=True, help="报告文件名后缀：reports/asr-<name>.jsonl / .md")
    ap.add_argument("--start", type=float, default=0.0, help="从音频这一秒开始截取")
    ap.add_argument("--duration", type=float, default=None, help="截取这么多秒；不填就到结尾")
    ap.add_argument("--speed", type=float, default=1.0, help="推流倍速，只压缩发送间隔")
    ap.add_argument("--speakers", action="store_true", help="开启服务端说话人分离")
    ap.add_argument("--end-window", type=int, default=300, help="end_window_size 毫秒，黑客松默认 300")
    ap.add_argument("--hotwords", type=int, default=0, help="从知识库取这么多个热词（0 不用）")
    ap.add_argument("--scene", action="store_true", help="把知识库标题里的英文术语写成上下文场景描述")
    ap.add_argument("--ddc", action="store_true", help="开官方顺滑：删口水词和重复")
    ap.add_argument("--tail-wait", type=float, default=5.0, help="发完最后一帧后等这么多秒再关连接")
    ap.add_argument("--tmp-dir", default="", help="临时 PCM 文件目录，默认 reports/.tmp")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
