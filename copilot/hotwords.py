"""从知识库里挑 ASR 热词。

中文文档里用英文写的术语，面试时多半也是用英文念的，夹在中文里最容易被听错
（Kafka 被写成「卡不卡」，topic 被写成「爬取型」）。热词额度很小（双向流式 100 token），
要给「主题词」而不是「高频词」：HashMap 只在一条里集中讲，按正文频率会被 Key、list 挤掉。
主题词最集中在条目标题和小节标题里（「HashMap 底层原理」「Topic / Partition / Offset」），
所以先取标题里的英文词，额度还有剩再按正文里夹在中文中的英文词补（跳过代码）。
从知识库取词而不是从某场面试的录音里挑，换一场面试也照样适用。
"""

from __future__ import annotations

import re
from collections import Counter

from .kb import KnowledgeBase

CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
# 前后 3 个字符内有汉字，才算「夹在中文里念的英文」
TERM = re.compile(r"(?<=[一-鿿（）：，、 ]{1})([A-Za-z][A-Za-z0-9+#]*[A-Za-z0-9+#])(?=[一-鿿（）：，、 ]{1})")
NOISE = re.compile(r"^P\d+$|\d{3,}|^[A-Za-z0-9]{20,}$")  # 页码 P11、图片 token 之类


def _titles(kb: KnowledgeBase) -> list[str]:
    heads: set[str] = set()
    for c in kb.chunks:
        heads.update(c.path)
        heads.add(c.title)
    out: Counter[str] = Counter()
    spelled: dict[str, Counter[str]] = {}
    for h in heads:
        for w in set(re.findall(r"[A-Za-z][A-Za-z0-9+#]*[A-Za-z0-9+#]", h)):
            if NOISE.search(w) or len(w) < 3:
                continue
            out[w.lower()] += 1
            spelled.setdefault(w.lower(), Counter())[w] += 1
    return [spelled[k].most_common(1)[0][0] for k, _ in out.most_common()]


def pick(kb: KnowledgeBase, limit: int = 40) -> list[str]:
    chosen = _titles(kb)[:limit]
    taken = {w.lower() for w in chosen}
    return chosen + [w for w in _prose(kb) if w.lower() not in taken][: max(0, limit - len(chosen))]


def _prose(kb: KnowledgeBase) -> list[str]:
    seen: Counter[str] = Counter()
    spelled: dict[str, Counter[str]] = {}
    for c in kb.chunks:
        prose = CODE.sub(" ", c.title + "\n" + c.body)
        words = set()
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9+#]*[A-Za-z0-9+#]", prose):
            w = m.group(0)
            around = prose[max(0, m.start() - 3) : m.start()] + prose[m.end() : m.end() + 3]
            if not re.search(r"[一-鿿]", around) or NOISE.search(w) or len(w) < 3:
                continue
            key = w.lower()
            words.add(key)
            spelled.setdefault(key, Counter())[w] += 1
        seen.update(words)
    return [spelled[k].most_common(1)[0][0] for k, _ in seen.most_common()]


def asr_options(kb: KnowledgeBase, hotwords: int = 40, scene: bool | str = True) -> dict:
    """给 DoubaoAsrClient 的热词和场景描述，现行配置见 bench/RESULTS.md「现行配置」。
    scene 给字符串（kb/scene.md 的面试场景描述）就用它：一段讲清候选人做过什么的话，比拼起来的标题词更像场景。"""
    options: dict = {}
    if hotwords:
        options["hotwords"] = pick(kb, hotwords)
    if isinstance(scene, str) and scene.strip():
        options["scene"] = "这是一场中文技术面试的录音，面试官和候选人会夹杂英文技术术语。" + scene.strip()
    elif scene:
        options["scene"] = "这是一场中文技术面试的录音，面试官和候选人会夹杂英文技术术语，可能涉及：" + "、".join(_titles(kb))
    return options
