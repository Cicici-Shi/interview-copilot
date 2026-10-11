"""实时提词：网页采两路音频 → 本地 VAD + 豆包转写 → Assistant 快慢答 → 推回网页显示。

  set -a; . ~/Downloads/soclaas.env; set +a
  export $(grep ^VOLC_ ~/Desktop/git/personal_agent_long_context/.env)
  python3 -m copilot.live                    # 打开 http://localhost:8766 ，点「开始」
  python3 -m copilot.live --replay-interviewer a.wav --replay-me b.wav --from 0 --duration 180
                                             # 不用网页采音，拿录音按真实时间喂，网页只看

网页发来的二进制帧：第 1 字节是通道（0 面试官 = 共享的会议声音，1 我 = 麦克风），后面是 16kHz 单声道 s16le。
文本帧：{"type": "start"} / {"type": "stop"}。服务端发给网页的都是 JSON，见 Session.out 的调用处。
一场下来所有输出另存 reports/live-<时间>.jsonl，复盘用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import websockets
from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from .assistant import INTERVIEWER, ME, Assistant, Event
from .coding import CodingCoach
from .doubao import DoubaoAsrClient, Final, Partial
from .hotwords import asr_options
from .kb import KnowledgeBase
from .llm import OpenAICompatible
from .vad import SAMPLE_RATE, EnergyVad

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "web" / "index.html"
CHANNELS = {0: INTERVIEWER, 1: ME}
# 我这一路要连续响久一点才算开口：开口会立刻结束面试官这一轮，键盘声、清嗓子误触发代价大
VAD_MIN_MS = {INTERVIEWER: 100, ME: 250}
RECONNECT_S = 1.0


SCENE = ROOT / "kb/scene.md"  # 面试场景描述，见 SPEC 第 5 节

class Channel:
    """一路音频：VAD 报开口，豆包出 partial/final。豆包连接断了就重连，断开期间的音频丢掉。"""

    def __init__(self, speaker: str, session: "Session", asr_factory):
        self.speaker = speaker
        self.session = session
        self.asr_factory = asr_factory
        self.vad = EnergyVad(VAD_MIN_MS[speaker])
        self.client: DoubaoAsrClient | None = None
        self.connecting: asyncio.Task | None = None
        self.reader: asyncio.Task | None = None
        self.stopping = False

    def start(self) -> None:
        self.connecting = asyncio.create_task(self._connect())

    async def _connect(self) -> None:
        while not self.stopping:
            client = self.asr_factory()
            try:
                await client.connect()
            except Exception as e:
                self.session.out({"type": "status", "speaker": self.speaker, "text": f"转写连接失败：{e!r}"[:200]})
                await asyncio.sleep(RECONNECT_S)
                continue
            self.client = client
            self.reader = asyncio.create_task(self._read(client))
            self.session.out({"type": "status", "speaker": self.speaker, "text": "转写已连接"})
            return

    async def _read(self, client: DoubaoAsrClient) -> None:
        try:
            async for event in client.events():
                if isinstance(event, Partial):
                    self.session.out({"type": "partial", "speaker": self.speaker, "text": event.text})
                    self.session.assistant.on_partial(self.speaker, event.text)
                elif isinstance(event, Final):
                    self.session.out({"type": "final", "speaker": self.speaker, "text": event.text})
                    self.session.assistant.on_final(self.speaker, event.text)
                    self.session.coach.on_final(self.speaker, event.text)
        except Exception as e:
            self.session.out({"type": "status", "speaker": self.speaker, "text": f"转写出错：{e!r}"[:200]})
        if self.client is client:
            self.client = None
        await client.close()
        if not self.stopping:  # 服务端断开（超时、出错）：重连接着转
            self.session.out({"type": "status", "speaker": self.speaker, "text": "转写断开，重连中"})
            await asyncio.sleep(RECONNECT_S)
            if not self.stopping and self.client is None:
                self.start()

    async def feed(self, pcm: bytes) -> None:
        if self.vad.feed(pcm):
            self.session.out({"type": "speech_start", "speaker": self.speaker})
            self.session.assistant.on_speech_start(self.speaker)
        client = self.client
        if client is None:
            return
        try:
            await client.send(pcm)
        except websockets.exceptions.ConnectionClosed:
            pass  # 读循环会发现断开并重连

    async def stop(self, tail_s: float = 1.5) -> None:
        self.stopping = True
        if self.connecting:
            self.connecting.cancel()
        client = self.client
        if client is not None:
            try:
                await client.finish()
                await asyncio.wait_for(asyncio.shield(self.reader), tail_s)  # 最后一句定稿
            except Exception:
                pass
            await client.close()
        if self.reader:
            self.reader.cancel()


class Session:
    """一场面试：两路 Channel 共用一个 Assistant。输出先存一份，后连上的网页能补看。"""

    def __init__(self, kb: KnowledgeBase, models: tuple, asr_factory, send, log_dir: Path = ROOT / "reports"):
        fast, slow, gate = models
        self.t0 = time.monotonic()
        self.send = send
        self.outputs: list[dict] = []
        log_dir.mkdir(exist_ok=True)
        self.log = (log_dir / f"live-{time.strftime('%Y%m%d-%H%M%S')}.jsonl").open("a", encoding="utf-8")
        scene = SCENE.read_text(encoding="utf-8") if SCENE.exists() else ""
        self.assistant = Assistant(kb, fast, slow, self._event, gate_model=gate, scene=scene)
        self.coach = CodingCoach(slow, self.out, scene=scene)
        self.assistant.paused = lambda: self.coach.active
        self.assistant.on_coding = self.coach.start
        self.channels = {s: Channel(s, self, asr_factory) for s in (INTERVIEWER, ME)}

    def start(self) -> None:
        self.out({"type": "session", "text": "开始"})
        for ch in self.channels.values():
            ch.start()

    def _event(self, e: Event) -> None:
        self.out({"type": "event"} | asdict(e))

    def out(self, msg: dict) -> None:
        msg = {"t": round(time.monotonic() - self.t0, 3)} | msg
        self.outputs.append(msg)
        self.log.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.log.flush()
        self.send(json.dumps(msg, ensure_ascii=False))

    async def feed(self, speaker: str, pcm: bytes) -> None:
        await self.channels[speaker].feed(pcm)

    async def stop(self) -> None:
        await asyncio.gather(*(ch.stop() for ch in self.channels.values()))
        self.assistant.close()
        self.coach.stop()
        self.out({"type": "session", "text": "结束"})
        self.log.close()


class Server:
    def __init__(self, kb: KnowledgeBase | None, models: tuple, asr_factory, replay: dict | None = None,
                 audio: str | None = None, review: dict | None = None):
        self.kb, self.models, self.asr_factory = kb, models, asr_factory
        self.audio = audio  # 原始混音，回放/复看时网页跟着放
        self.review = review  # 复看：{"offset": 秒, "messages": [...]}，不跑模型，整场输出一次发给网页按录音时间放
        self.viewers: set[ServerConnection] = set()
        self.session: Session | None = None
        self.source: ServerConnection | None = None  # 正在送音频的那个网页
        self.replay = replay
        self.replay_task: asyncio.Task | None = None

    def broadcast(self, message: str) -> None:
        broadcast(self.viewers, message)

    async def start_session(self, source: ServerConnection | None) -> Session:
        await self.stop_session()
        self.session = Session(self.kb, self.models, self.asr_factory, self.broadcast)
        self.source = source
        self.session.start()
        return self.session

    async def stop_session(self) -> None:
        session, self.session, self.source = self.session, None, None
        if session:
            await session.stop()

    def page(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.path in ("/ws", "/page"):
            return None
        if request.path in ("/", "/index.html"):
            mode = "replay" if self.replay or self.review else "live"
            body = PAGE.read_bytes().replace(b"__MODE__", mode.encode())  # 每次现读，改页面不用重启
            return Response(200, "OK", Headers([("Content-Type", "text/html; charset=utf-8"), ("Cache-Control", "no-store")]), body)
        if request.path == "/audio" and self.audio:
            body = Path(self.audio).read_bytes()
            head = [("Content-Type", "audio/mp4"), ("Accept-Ranges", "bytes")]
            rng = request.headers.get("Range", "")  # 不支持 Range 浏览器就不让跳转，只能从头放
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start, end = int(a or 0), min(int(b) if b else len(body) - 1, len(body) - 1)
                head += [("Content-Range", f"bytes {start}-{end}/{len(body)}"), ("Content-Length", str(end - start + 1))]
                return Response(206, "Partial Content", Headers(head), body[start : end + 1])
            return Response(200, "OK", Headers(head + [("Content-Length", str(len(body)))]), body)
        return connection.respond(404, "not found\n")

    async def handle(self, ws: ServerConnection) -> None:
        if ws.request.path == "/page":
            return await self.handle_page(ws)
        if self.review:
            await ws.send(json.dumps({"type": "playback", "offset": self.review["offset"], "audio": bool(self.audio)}))
            await ws.send(json.dumps({"type": "tape", "messages": self.review["messages"]}, ensure_ascii=False))
            await ws.wait_closed()
            return
        self.viewers.add(ws)
        try:
            if self.session:  # 中途打开的网页先补看
                for msg in self.session.outputs:
                    await ws.send(json.dumps(msg, ensure_ascii=False))
            if self.replay and (self.replay_task is None or self.replay_task.done()):  # 放完了再打开就重放
                self.replay_task = asyncio.create_task(self.run_replay())
            async for message in ws:
                if isinstance(message, bytes):
                    speaker = CHANNELS.get(message[0])
                    if self.session and self.source is ws and speaker:
                        await self.session.feed(speaker, message[1:])
                    continue
                kind = json.loads(message).get("type")
                if kind == "start" and not self.replay:
                    await self.start_session(ws)
                elif kind == "stop" and self.source is ws:
                    await self.stop_session()
                elif kind == "coding_stop" and self.session:
                    self.session.coach.stop()
        finally:
            self.viewers.discard(ws)
            if self.source is ws:  # 采音的网页关了，这场就结束
                await self.stop_session()

    async def handle_page(self, ws: ServerConnection) -> None:
        """Chrome 扩展推来的编程页面内容（extension/），交给手撕教练。"""
        async for message in ws:
            m = json.loads(message)
            if m.get("type") == "page" and self.session:
                self.session.coach.on_page(m)

    async def run_replay(self) -> None:
        """两路录音按真实时间喂，每 100ms 一块。给了 segments 就只喂那几段，段与段之间插几秒静音。"""
        r = self.replay
        session = await self.start_session(None)
        if r["segments"][0][0] > 0:  # 从中间开始放：面试官早就说过话了，别把我说的当成问题
            session.assistant.heard_interviewer = True
        for start, duration in r["segments"]:
            tracks = {s: decode(p, start, duration) for s, p in ((INTERVIEWER, r["interviewer"]), (ME, r["me"])) if p}
            # 解码完、开喂前再记，复看时用它把日志时间对到录音时间
            session.out({"type": "segment", "text": f"{stamp(start)} 起", "start": start, "wall": time.time(), "audio": bool(self.audio)})
            await self.feed_tracks(session, tracks)
            await self.feed_tracks(session, {s: bytes(SAMPLE_RATE * 2 * GAP_S) for s in tracks})
        await asyncio.sleep(3)  # 让最后一个问题的慢答写完
        await session.assistant.wait()
        await self.stop_session()


    @staticmethod
    async def feed_tracks(session: Session, tracks: dict[str, bytes]) -> None:
        chunk = SAMPLE_RATE * 2 // 10
        longest = max(len(pcm) for pcm in tracks.values())
        t0 = time.monotonic()
        for i, offset in enumerate(range(0, longest, chunk)):
            for speaker, pcm in tracks.items():
                if offset < len(pcm):
                    await session.feed(speaker, pcm[offset : offset + chunk])
            await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.1 - time.monotonic()))


GAP_S = 3


def load_review(path: Path, offset: float | None) -> dict:
    """复看用：只留网页要画的消息（partial 太多，按 0.5 秒抽一条）。
    回放生成的日志按段换算成录音时间（各段之间有间隔，一个固定偏移对不上），这时 offset 为 0。"""
    messages, last_partial, seg = [], {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if m["type"] == "segment" and offset is None and "start" in m:
            seg = (m["t"], m["start"])
        if seg is not None:
            m["t"] = m["t"] - seg[0] + seg[1]
        if m["type"] == "partial":
            if m["t"] - last_partial.get(m["speaker"], -1) < 0.5:
                continue
            last_partial[m["speaker"]] = m["t"]
        if m["type"] in ("partial", "final", "event", "segment"):
            messages.append(m)
    return {"offset": 0.0 if seg is not None else (offset or 0.0), "messages": messages}


def stamp(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def parse_segments(text: str) -> list[tuple[float, float]]:
    """「08:15-09:45,12:15-17:00」→ [(起秒, 时长秒)]"""
    def sec(x: str) -> float:
        parts = [float(p) for p in x.strip().split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    out = []
    for part in text.split(","):
        a, b = part.split("-")
        out.append((sec(a), sec(b) - sec(a)))
    return out


def decode(path: str, start: float, duration: float | None) -> bytes:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(Path(path).expanduser())]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


async def main(args: argparse.Namespace) -> None:
    if args.review:  # 复看不跑模型，不读知识库和 key
        server = Server(None, (), None, audio=args.replay_audio, review=load_review(Path(args.review), args.offset))
        async with serve(server.handle, args.host, args.port, process_request=server.page, max_size=None):
            print(f"http://localhost:{args.port}")
            await asyncio.get_running_loop().create_future()
        return
    kb = KnowledgeBase.from_sources([(ROOT / "kb/interview.md", "八股", True), (ROOT / "kb/projects.md", "项目", False), (ROOT / "kb/bagu-2026-10.md", "项目", False)])
    options = asr_options(kb, scene=SCENE.read_text(encoding="utf-8") if SCENE.exists() else "")
    models = (OpenAICompatible.from_env(args.fast_model), OpenAICompatible.from_env(args.slow_model), OpenAICompatible.from_env(args.gate_model))

    def asr_factory() -> DoubaoAsrClient:
        return DoubaoAsrClient.from_env(end_window_ms=300, ddc=True, **options)

    replay = None
    if args.replay_interviewer or args.replay_me:
        segments = parse_segments(args.segments) if args.segments else [(args.start, args.duration)]
        replay = {"interviewer": args.replay_interviewer, "me": args.replay_me, "segments": segments}
    server = Server(kb, models, asr_factory, replay, audio=args.replay_audio)
    async with serve(server.handle, args.host, args.port, process_request=server.page, max_size=None):
        print(f"http://localhost:{args.port}")
        await asyncio.get_running_loop().create_future()


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--fast-model", default="qwen3.8:27b")
    ap.add_argument("--slow-model", default="qwen3.8:27b")
    ap.add_argument("--gate-model", default="qwen3.8:27b")
    ap.add_argument("--replay-interviewer", help="拿这段录音当面试官那一路")
    ap.add_argument("--replay-me", help="拿这段录音当我这一路")
    ap.add_argument("--replay-audio", help="原始混音录音，网页上点「播放声音」跟着回放进度放（时间轴要和两路一致）")
    ap.add_argument("--review", help="复看一场存下来的 reports/live-*.jsonl，配 --replay-audio 带进度条放")
    ap.add_argument("--offset", type=float, default=None, help="复看：日志时间减它等于录音时间；不给就用日志里回放开喂的时刻")
    ap.add_argument("--from", dest="start", type=float, default=0.0, help="录音从第几秒开始")
    ap.add_argument("--duration", type=float, default=None, help="只喂这么多秒")
    ap.add_argument("--segments", help="只喂这几段，如 08:15-09:45,12:15-17:00（挑要提醒的题，跳过自我介绍）")
    args = ap.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
