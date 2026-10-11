"""豆包（火山引擎）双向流式语音识别，走二进制 WebSocket 协议。

协议和黑客松 TS 版（personal_agent_long_context/src/meetings/doubao.ts）一致：
每帧 4 字节头 + 4 字节序号（可选）+ 4 字节长度 + gzip 过的 JSON payload。
音频进：16kHz 单声道 16bit PCM，几百毫秒一段。出来的是还在说的那句（partial，
每次整句替换）和定稿的句子（final：一段静音之后，服务端二次识别给出的最终文本，
带 start_ms/end_ms；按时间区间是否和同一说话人前面的分句重叠去重，见 FinalTracker）。

协议文档：https://docs.volcengine.com/docs/DoubaoVoice/bidirectional-streaming-automatic-speech-recognition-websocket
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import struct
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import certifi
import websockets

# python.org 官方安装包的 OpenSSL 不带任何根证书（经典的「运行一下 Install
# Certificates.command」坑），任何 https/wss 都会报 CERTIFICATE_VERIFY_FAILED。
# 这里补一个默认值：没设过 SSL_CERT_FILE 就指到 certifi 自带的证书包，和密钥无关，
# 不影响用户已有的环境变量。
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"

# 消息类型和 flag，和 TS 版的 FULL_REQUEST / AUDIO_ONLY / SERVER_RESPONSE / SERVER_ERROR 对齐
FULL_REQUEST = 0x1
AUDIO_ONLY = 0x2
SERVER_RESPONSE = 0x9
SERVER_ERROR = 0xF
POSITIVE_SEQUENCE = 0x1
LAST_WITH_SEQUENCE = 0x3


def _frame(msg_type: int, flags: int, payload: bytes, sequence: int) -> bytes:
    """构造一个请求帧：协议版本 1、4 字节头，payload 一律 gzip。"""
    body = gzip.compress(payload)
    header = bytes([0x11, (msg_type << 4) | flags, 0x11, 0])  # 0x11 = JSON + gzip
    return header + struct.pack(">i", sequence) + struct.pack(">I", len(body)) + body


@dataclass
class _Utterance:
    text: str = ""
    definite: bool = False
    start_time: int = 0
    end_time: int = 0
    speaker: str | None = None
    source: str | None = None


@dataclass
class _ParseResult:
    error: str | None = None
    utterances: list[_Utterance] = field(default_factory=list)


def parse_frame(data: bytes) -> _ParseResult:
    """解析服务端返回的一帧。错误帧和正常帧的字段布局不一样，分开处理。"""
    msg_type = data[1] >> 4
    flags = data[1] & 0x0F
    gzipped = (data[2] & 0x0F) == 1
    offset = 4
    if msg_type == SERVER_ERROR:
        code = struct.unpack_from(">I", data, offset)[0]
        size = struct.unpack_from(">I", data, offset + 4)[0]
        message = data[offset + 8 : offset + 8 + size].decode("utf-8", "replace")
        return _ParseResult(error=f"{code}: {message}")
    if msg_type != SERVER_RESPONSE:
        return _ParseResult()
    if flags & 0x1:
        offset += 4  # 带序号的帧多 4 字节
    size = struct.unpack_from(">I", data, offset)[0]
    body = data[offset + 4 : offset + 4 + size]
    if gzipped and body:
        body = gzip.decompress(body)
    if not body:
        return _ParseResult()
    payload = json.loads(body.decode("utf-8"))
    raw = (payload.get("result") or {}).get("utterances") or []
    utterances = [
        _Utterance(
            text=u.get("text") or "",
            definite=bool(u.get("definite")),
            start_time=u.get("start_time") or 0,
            end_time=u.get("end_time") or 0,
            speaker=(u.get("additions") or {}).get("speaker_id"),
            source=(u.get("additions") or {}).get("source"),
        )
        for u in raw
    ]
    return _ParseResult(utterances=utterances)


@dataclass
class Partial:
    """正在说的这句话，每次替换（不是追加）。"""

    text: str


@dataclass
class Final:
    """一句话定稿：静音一段时间后，服务端二次识别给出的最终文本。"""

    text: str
    start_ms: int
    end_ms: int
    speaker: str | None = None


@dataclass
class FinalTracker:
    """判断一条 definite=true 的分句算不算「新的一句话」。

    开 enable_nonstream 后（文档：「系统重新对这一小段音频进行识别，用更稳定、准确的
    结果替换之前实时显示的文字」），实测发现服务端有时不是逐句替换，而是把同一个说话人
    连续说的好几句已经出过 final 的短句，重新合并识别成一条跨度更大、start_time 对齐
    到其中第一句的新 definite 分句——TS 版按 (start_time, end_time) 精确去重会漏判这种
    情况（时间戳区间变了），导致同一段话在 on_final 里被喂给下游两遍，而且合并后的文本
    反而丢了原来每句之间的标点（实测：分开出的 6 句都带标点，合并后的一整句一个标点都
    没有），对我们「转写完就吐给下游」的用法没有帮助。
    这里按时间区间是否和「这个说话人已经出过的最晚 end_time」重叠来判断：重叠就是这种
    合并重放，丢弃；不重叠才当作新内容。按说话人分开跟踪，避免打断/交叉说话时把另一个
    说话人的正常发言误判成重叠（这种粗略处理仍可能在极少数打断场景漏判或误判，见评估
    报告里的说明）。
    """

    _max_end_ms: dict[str | None, int] = field(default_factory=dict)

    def should_emit(self, speaker: str | None, start_ms: int, end_ms: int) -> bool:
        prev_end = self._max_end_ms.get(speaker, -1)
        if start_ms < prev_end:
            return False
        self._max_end_ms[speaker] = max(prev_end, end_ms)
        return True


def events_from(result: _ParseResult, tracker: FinalTracker) -> list[Partial | Final]:
    """把一帧解析结果转成 partial/final 事件。抽成纯函数方便不连网络单测；
    _read_loop 和这里共用同一份逻辑。"""
    out: list[Partial | Final] = []
    partial = ""
    for u in result.utterances:
        if not u.text:
            continue
        if not u.definite:
            partial = u.text
            continue
        if not tracker.should_emit(u.speaker, u.start_time, u.end_time):
            continue
        out.append(Final(text=u.text, start_ms=u.start_time, end_ms=u.end_time, speaker=u.speaker))
    # 和 TS 版一样，每条消息都上报一次 partial（可能是空字符串，代表清空显示）
    out.append(Partial(text=partial))
    return out


class DoubaoAsrError(RuntimeError):
    """服务端返回的错误帧（握手失败、参数不支持等）。"""


class DoubaoAsrClient:
    """一条 ASR 会话：connect() 一次 → 反复 send(pcm) → finish() → close()。

    partial/final 事件从 events() 异步迭代拿。发音频和收事件要各自起一个协程并发跑
    （asr_replay.py 就是这么用的），别在一个协程里先等 events() 再发下一段音频。
    """

    def __init__(
        self,
        api_key: str,
        resource_id: str,
        *,
        speakers: bool = False,
        end_window_ms: int = 300,
        hotwords: list[str] | None = None,
        scene: str = "",
        ddc: bool = False,
    ):
        self.api_key = api_key
        self.resource_id = resource_id
        self.speakers = speakers
        self.end_window_ms = end_window_ms
        # 热词直传（双向流式限 100 token）和上下文（800 token）都放在 corpus.context 里，
        # 用来纠正英文术语（Kafka 被听成「卡不卡」）；ddc 是官方的「顺滑」，删口水词和重复
        self.hotwords = hotwords or []
        self.scene = scene
        self.ddc = ddc
        self._ws: websockets.ClientConnection | None = None
        self._sequence = 1
        self._final_tracker = FinalTracker()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._closed = False

    @classmethod
    def from_env(cls, *, speakers: bool = False, end_window_ms: int = 300, **options) -> "DoubaoAsrClient":
        return cls(
            os.environ["VOLC_ASR_API_KEY"],
            os.environ.get("VOLC_ASR_RESOURCE_ID", "volc.seedasr.sauc.duration"),
            speakers=speakers,
            end_window_ms=end_window_ms,
            **options,
        )

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            URL,
            additional_headers={
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Connect-Id": str(uuid.uuid4()),
            },
            max_size=None,
        )
        request = {
            "user": {"uid": "interview-copilot"},
            "audio": {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
                "result_type": "full",
                # 一句话后静这么久算说完，第二遍重读这句给出定稿文本（同 TS 版 endWindowMs）
                "end_window_size": self.end_window_ms,
                "enable_nonstream": True,
                "enable_speaker_info": self.speakers,
                **({"ssd_version": "200"} if self.speakers else {}),  # 文档：分离要配这个版本号
                **({"enable_ddc": True} if self.ddc else {}),
                **({"corpus": {"context": json.dumps(self._context(), ensure_ascii=False)}} if self.hotwords or self.scene else {}),
            },
        }
        await self._ws.send(_frame(FULL_REQUEST, POSITIVE_SEQUENCE, json.dumps(request).encode("utf-8"), self._sequence))
        # 第一条回复只是握手确认，之后才能发真正的音频（同 TS 版的 ready 状态，
        # 这里直接 await 一条消息，不用像 TS 那样维护一个等待队列）
        ack = parse_frame(await self._ws.recv())
        if ack.error:
            raise DoubaoAsrError(f"握手失败：{ack.error}")
        self._reader = asyncio.create_task(self._read_loop())

    def _context(self) -> dict:
        ctx: dict = {}
        if self.hotwords:
            ctx["hotwords"] = [{"word": w} for w in self.hotwords]
        if self.scene:
            ctx["context_type"] = "dialog_ctx"
            ctx["context_data"] = [{"text": self.scene}]
        return ctx

    async def send(self, pcm: bytes) -> None:
        if self._closed or self._ws is None:
            return
        self._sequence += 1
        await self._ws.send(_frame(AUDIO_ONLY, POSITIVE_SEQUENCE, pcm, self._sequence))

    async def finish(self) -> None:
        """没有更多音频了：发最后一帧（序号取负），服务端把最后一句定稿后关闭连接。"""
        if self._closed or self._ws is None:
            return
        self._sequence += 1
        await self._ws.send(_frame(AUDIO_ONLY, LAST_WITH_SEQUENCE, b"", -self._sequence))

    async def events(self) -> AsyncIterator[Partial | Final]:
        while True:
            item = await self._queue.get()
            if item is None:  # 读循环结束：连接关闭或出错
                return
            if isinstance(item, Exception):
                raise item
            yield item

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for data in self._ws:
                result = parse_frame(data)
                if result.error:
                    await self._queue.put(DoubaoAsrError(result.error))
                    return
                for event in events_from(result, self._final_tracker):
                    await self._queue.put(event)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._queue.put(None)

    async def close(self) -> None:
        self._closed = True
        if self._reader:
            self._reader.cancel()
        if self._ws is not None:
            await self._ws.close()
