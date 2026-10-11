"""豆包 ASR 二进制协议编解码的单测：不连网络，只测帧构造/解析/去重逻辑。"""

import gzip
import json
import struct

from copilot.doubao import (
    AUDIO_ONLY,
    FULL_REQUEST,
    LAST_WITH_SEQUENCE,
    POSITIVE_SEQUENCE,
    SERVER_ERROR,
    SERVER_RESPONSE,
    Final,
    FinalTracker,
    Partial,
    _frame,
    events_from,
    parse_frame,
)


def build_response_frame(payload: dict, *, with_sequence: bool, sequence: int = 1) -> bytes:
    """按协议手工拼一个服务端响应帧，模拟 parse_frame 要处理的输入。"""
    body = gzip.compress(json.dumps(payload).encode("utf-8"))
    flags = POSITIVE_SEQUENCE if with_sequence else 0
    header = bytes([0x11, (SERVER_RESPONSE << 4) | flags, 0x11, 0])
    out = header
    if with_sequence:
        out += struct.pack(">i", sequence)
    out += struct.pack(">I", len(body)) + body
    return out


def build_error_frame(code: int, message: str) -> bytes:
    # 错误帧没有序号字段，紧跟 4 字节头的是 code + size + message（同 TS 版 parse()）
    header = bytes([0x11, (SERVER_ERROR << 4) | 0, 0x11, 0])
    encoded = message.encode("utf-8")
    return header + struct.pack(">I", code) + struct.pack(">I", len(encoded)) + encoded


def test_frame_header_and_gzip_payload_roundtrip():
    payload = {"user": {"uid": "x"}}
    out = _frame(FULL_REQUEST, POSITIVE_SEQUENCE, json.dumps(payload).encode("utf-8"), 1)
    assert out[0] == 0x11  # 协议版本 1，4 字节头
    assert out[1] == (FULL_REQUEST << 4) | POSITIVE_SEQUENCE
    assert out[2] == 0x11  # JSON + gzip
    sequence = struct.unpack_from(">i", out, 4)[0]
    assert sequence == 1
    size = struct.unpack_from(">I", out, 8)[0]
    body = out[12 : 12 + size]
    assert len(body) == size
    assert json.loads(gzip.decompress(body).decode("utf-8")) == payload


def test_frame_last_audio_uses_negative_sequence():
    out = _frame(AUDIO_ONLY, LAST_WITH_SEQUENCE, b"", -5)
    assert out[1] == (AUDIO_ONLY << 4) | LAST_WITH_SEQUENCE
    assert struct.unpack_from(">i", out, 4)[0] == -5
    # 空 payload 也会被 gzip，长度不是 0
    size = struct.unpack_from(">I", out, 8)[0]
    assert size == len(gzip.compress(b""))


def test_parse_frame_with_sequence_number_offsets_correctly():
    payload = {"result": {"utterances": [{"text": "你好", "definite": False}]}}
    frame = build_response_frame(payload, with_sequence=True, sequence=7)
    result = parse_frame(frame)
    assert result.error is None
    assert len(result.utterances) == 1
    assert result.utterances[0].text == "你好"
    assert result.utterances[0].definite is False


def test_parse_frame_without_sequence_number():
    payload = {"result": {"utterances": [{"text": "在", "definite": True, "start_time": 100, "end_time": 500}]}}
    frame = build_response_frame(payload, with_sequence=False)
    result = parse_frame(frame)
    assert result.utterances[0].start_time == 100
    assert result.utterances[0].end_time == 500


def test_parse_frame_reads_speaker_id_from_additions():
    payload = {
        "result": {
            "utterances": [
                {"text": "这段", "definite": True, "start_time": 0, "end_time": 900, "additions": {"speaker_id": "1"}}
            ]
        }
    }
    frame = build_response_frame(payload, with_sequence=True)
    result = parse_frame(frame)
    assert result.utterances[0].speaker == "1"


def test_parse_frame_missing_speaker_is_none():
    payload = {"result": {"utterances": [{"text": "x", "definite": True, "start_time": 0, "end_time": 1}]}}
    frame = build_response_frame(payload, with_sequence=True)
    result = parse_frame(frame)
    assert result.utterances[0].speaker is None


def test_parse_frame_empty_body_returns_no_utterances_and_no_error():
    header = bytes([0x11, (SERVER_RESPONSE << 4) | 0, 0x11, 0])
    frame = header + struct.pack(">I", 0)
    result = parse_frame(frame)
    assert result.error is None
    assert result.utterances == []


def test_parse_frame_server_error_decodes_code_and_message():
    frame = build_error_frame(45000001, "invalid audio format")
    result = parse_frame(frame)
    assert result.error == "45000001: invalid audio format"


def test_parse_frame_ignores_unknown_message_type():
    header = bytes([0x11, (0x0 << 4) | 0, 0x11, 0])
    frame = header + struct.pack(">I", 0)
    result = parse_frame(frame)
    assert result.error is None
    assert result.utterances == []


def test_events_from_emits_partial_text():
    payload = {"result": {"utterances": [{"text": "在讲", "definite": False}]}}
    result = parse_frame(build_response_frame(payload, with_sequence=True))
    events = events_from(result, FinalTracker())
    assert events == [Partial(text="在讲")]


def test_events_from_emits_final_and_dedupes_exact_repeat():
    payload = {"result": {"utterances": [{"text": "第一句", "definite": True, "start_time": 0, "end_time": 800}]}}
    result = parse_frame(build_response_frame(payload, with_sequence=True))
    tracker = FinalTracker()
    first = events_from(result, tracker)
    assert first == [Final(text="第一句", start_ms=0, end_ms=800, speaker=None), Partial(text="")]
    # 服务端把同一句（时间戳不变）重复推一遍，二次识别文本可能变了，但不应该再出一次 final
    payload2 = {"result": {"utterances": [{"text": "第一句改", "definite": True, "start_time": 0, "end_time": 800}]}}
    second = events_from(parse_frame(build_response_frame(payload2, with_sequence=True)), tracker)
    assert second == [Partial(text="")]


def test_events_from_drops_a_later_merged_restatement_that_overlaps_earlier_finals():
    """实测发现的行为：服务端把同一说话人已经出过 final 的好几句合并重放成一条更长、
    start_time 对齐到其中第一句的新分句；这种重叠的重放要丢弃，不然下游会拿到重复内容。"""
    tracker = FinalTracker()
    first = {"result": {"utterances": [{"text": "第一句", "definite": True, "start_time": 0, "end_time": 800, "additions": {"speaker_id": "0"}}]}}
    second = {"result": {"utterances": [{"text": "第二句", "definite": True, "start_time": 900, "end_time": 1600, "additions": {"speaker_id": "0"}}]}}
    merged = {"result": {"utterances": [{"text": "第一句第二句", "definite": True, "start_time": 0, "end_time": 1600, "additions": {"speaker_id": "0"}}]}}
    events = []
    for payload in (first, second, merged):
        events += events_from(parse_frame(build_response_frame(payload, with_sequence=True)), tracker)
    finals = [e for e in events if isinstance(e, Final)]
    assert [f.text for f in finals] == ["第一句", "第二句"]


def test_events_from_does_not_drop_a_genuinely_new_utterance_from_a_different_speaker():
    tracker = FinalTracker()
    a = {"result": {"utterances": [{"text": "面试官问", "definite": True, "start_time": 0, "end_time": 2000, "additions": {"speaker_id": "0"}}]}}
    # 候选人在面试官这句话还没定稿完之前就插话，时间上和面试官那句重叠，但说话人不同，不该被丢
    b = {"result": {"utterances": [{"text": "候选人插话", "definite": True, "start_time": 1000, "end_time": 1500, "additions": {"speaker_id": "1"}}]}}
    events = []
    for payload in (a, b):
        events += events_from(parse_frame(build_response_frame(payload, with_sequence=True)), tracker)
    finals = [e for e in events if isinstance(e, Final)]
    assert [f.text for f in finals] == ["面试官问", "候选人插话"]


def test_events_from_mixed_partial_and_final_in_one_message():
    payload = {
        "result": {
            "utterances": [
                {"text": "定稿句", "definite": True, "start_time": 0, "end_time": 500},
                {"text": "正在说的", "definite": False},
            ]
        }
    }
    result = parse_frame(build_response_frame(payload, with_sequence=True))
    events = events_from(result, FinalTracker())
    assert Final(text="定稿句", start_ms=0, end_ms=500, speaker=None) in events
    assert Partial(text="正在说的") in events
