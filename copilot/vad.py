"""按能量判断「开始说话」，给 Assistant.on_speech_start 用。

只需要「开口」这一个信号，而且要快：面试官停顿后又开口，要在静音计时器（0.8s）到点前把它停住；
ASR 的 partial 要晚 1 秒左右，来不及。所以不判断说完，也不追求准确切句。

做法：20ms 一帧算 dBFS；底噪用不说话时的帧跟（往下快、往上约 10 秒才追上，字间停顿和键盘声拉不高它）；
比底噪高 MARGIN_DB 的帧，在最近 2×min_ms 里累计够 min_ms 就算开口——真人说话字与字之间有停顿，
按「连续响」算，250ms 的门槛真实语音几乎达不到（985 录音实测漏掉整句）；
之后安静 RELEASE_MS 才重新待命，同一段话只报一次。
"""

from __future__ import annotations

from collections import deque

import numpy as np

SAMPLE_RATE = 16000
FRAME = 320  # 20ms
MARGIN_DB = 12.0
FLOOR_DB = -50.0  # 底噪再低也按这个算，免得全静音时一点电流声就触发
RELEASE_MS = 400


class EnergyVad:
    def __init__(self, min_ms: int = 100):
        self.min_frames = max(1, min_ms // 20)
        self.release_frames = RELEASE_MS // 20
        self.noise_db = FLOOR_DB
        self.window: deque[bool] = deque(maxlen=self.min_frames * 2)  # 最近这些帧响不响
        self.quiet = 0  # 说话中连续安静的帧数
        self.speaking = False
        self._rest = b""  # 不够一帧的尾巴，下次接上（按字节存，奇数字节的块也不会错位）

    def feed(self, pcm: bytes) -> bool:
        """喂一段 16kHz 单声道 s16le，这段里有人开口就返回 True。"""
        data = self._rest + pcm
        whole = len(data) // (FRAME * 2) * FRAME * 2
        self._rest = data[whole:]
        started = False
        for frame in np.frombuffer(data[:whole], dtype="<i2").reshape(-1, FRAME):
            started |= self._frame(frame)
        return started

    def _frame(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        db = 20 * np.log10(max(rms, 1.0) / 32768)
        loud = db > max(self.noise_db, FLOOR_DB) + MARGIN_DB
        if not self.speaking:
            # 底噪：比当前低就快跟下去，高就慢慢涨（约 2 秒追上），说话时不更新
            rate = 0.3 if db < self.noise_db else 0.002
            if not loud:
                self.noise_db += rate * (db - self.noise_db)
            self.window.append(loud)
            if sum(self.window) >= self.min_frames:
                self.speaking, self.quiet = True, 0
                return True
            return False
        self.quiet = 0 if loud else self.quiet + 1
        if self.quiet >= self.release_frames:
            self.speaking = False
            self.window.clear()
        return False
