import asyncio
import json

import numpy as np

from copilot.assistant import INTERVIEWER, ME
from copilot.doubao import Final, Partial
from copilot.live import Session
from copilot.vad import EnergyVad
from test_assistant import KB, FakeModel


def tone(ms: int, amp: float = 0.2) -> bytes:
    t = np.arange(16 * ms) / 16000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def noise(ms: int, amp: float = 0.002, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(16 * ms) * amp * 32767).astype("<i2").tobytes()


def test_vad_reports_each_speech_onset_once():
    vad = EnergyVad(min_ms=100)
    assert not vad.feed(noise(1000))
    assert vad.feed(tone(300))
    assert not vad.feed(tone(300))  # 同一段话不重复报
    assert not vad.feed(noise(500))
    assert vad.feed(tone(200))  # 安静够久后再开口，再报一次


def test_vad_ignores_bursts_shorter_than_min():
    vad = EnergyVad(min_ms=250)
    vad.feed(noise(1000))
    assert not vad.feed(tone(100))  # 键盘一下
    assert not vad.feed(noise(600))


def test_vad_catches_speech_with_gaps_between_syllables():
    vad = EnergyVad(min_ms=250)
    vad.feed(noise(1000))
    syllables = (tone(120) + noise(80, seed=1)) * 5  # 一个字一停，没有连续 250ms 响的
    assert vad.feed(syllables)


def test_vad_handles_odd_chunk_sizes():
    vad = EnergyVad(min_ms=100)
    vad.feed(noise(1000))
    audio = tone(300)
    hits = [vad.feed(audio[i : i + 333]) for i in range(0, len(audio), 333)]  # 奇数字节也不能错位
    assert sum(hits) == 1


class FakeAsr:
    """假的豆包客户端：测试往 queue 里塞事件，记录收到的音频。"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sent = []

    async def connect(self):
        pass

    async def send(self, pcm):
        self.sent.append(pcm)

    async def finish(self):
        await self.queue.put(None)

    async def events(self):
        while (item := await self.queue.get()) is not None:
            yield item

    async def close(self):
        pass


async def test_session_routes_channels_and_triggers_question(tmp_path):
    clients = []

    def factory():
        clients.append(FakeAsr())
        return clients[-1]

    sent = []
    session = Session(KB, (FakeModel(), FakeModel(), FakeModel()), factory, sent.append, log_dir=tmp_path)
    session.assistant.silence_s = 0.05
    session.start()
    await asyncio.sleep(0.01)
    them, me = session.channels[INTERVIEWER].client, session.channels[ME].client
    assert them is not me

    await session.feed(INTERVIEWER, noise(500))
    assert them.sent and not me.sent

    await them.queue.put(Partial("Kafka 消息"))
    await them.queue.put(Final("Kafka 消息积压怎么办", 0, 1000))
    await asyncio.sleep(0.2)
    await session.assistant.wait()
    await session.stop()

    msgs = [json.loads(m) for m in sent]
    assert any(m["type"] == "partial" and m["speaker"] == INTERVIEWER and m["text"] == "Kafka 消息" for m in msgs)
    kinds = [m.get("kind") for m in msgs if m["type"] == "event"]
    assert kinds[0] == "question" and "fast_done" in kinds and "slow_done" in kinds
    log = (next(tmp_path.glob("live-*.jsonl"))).read_text().splitlines()
    assert len(log) == len(sent)


async def test_my_speech_onset_fires_question_immediately(tmp_path):
    session = Session(KB, (FakeModel(), FakeModel(), FakeModel()), FakeAsr, lambda m: None, log_dir=tmp_path)
    session.assistant.silence_s = 10  # 静音计时器不会先到
    session.start()
    await asyncio.sleep(0.01)
    await session.channels[INTERVIEWER].client.queue.put(Final("消息积压怎么处理", 0, 1000))
    await asyncio.sleep(0.01)
    await session.feed(ME, noise(1000))
    assert session.assistant.current is None
    await session.feed(ME, tone(400))  # 我开口 → 面试官这一轮结束，立刻触发
    assert session.assistant.current and session.assistant.current.text == "消息积压怎么处理"
    await session.stop()
