"""OpenAI 兼容的流式调用。默认走 NUS SoC LaaS（~/Downloads/soclaas.env）。"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import AsyncIterator
from typing import Protocol

import asyncio

import httpx

RETRY_DELAYS = (1.0, 2.0, 4.0)  # 只在还没开始出字前重试
RETRY_STATUS = {429, 502, 503}  # SoC LaaS 限流回 429，后端临时不可用回 503


async def _post(client: httpx.AsyncClient, url: str, headers: dict, body: dict) -> httpx.Response:
    for delay in (*RETRY_DELAYS, None):
        r = await client.post(url, headers=headers, json=body)
        if r.status_code not in RETRY_STATUS or delay is None:
            r.raise_for_status()
            return r
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")


class ChatModel(Protocol):
    def stream(self, messages: list[dict], max_tokens: int) -> AsyncIterator[str]: ...
    async def json(self, messages: list[dict], max_tokens: int) -> dict: ...
    async def yes_prob(self, messages: list[dict], yes: str = "是", no: str = "否") -> float: ...


class OpenAICompatible:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"authorization": f"Bearer {api_key}"}
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenAICompatible":
        return cls(os.environ["SOCLAAS_BASE_URL"], os.environ["SOCLAAS_API_KEY"], model or os.environ.get("COPILOT_MODEL", "qwen3.8:27b"))

    def _body(self, messages: list[dict], max_tokens: int, stream: bool) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2, "stream": stream}
        if "qwen" in self.model.lower():
            # 思考模式首 token 要多等几秒，实时场景关掉
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    async def stream(self, messages: list[dict], max_tokens: int) -> AsyncIterator[str]:
        for delay in (*RETRY_DELAYS, None):
            async with self.client.stream("POST", self.url, headers=self.headers, json=self._body(messages, max_tokens, True)) as r:
                if r.status_code in RETRY_STATUS and delay is not None:
                    await asyncio.sleep(delay)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"LLM {r.status_code}: {(await r.aread())[:300]!r}")
                async for delta in _deltas(r):
                    yield delta
                return

    async def json(self, messages: list[dict], max_tokens: int) -> dict:
        r = await _post(self.client, self.url, self.headers, self._body(messages, max_tokens, False))
        return parse_json(r.json()["choices"][0]["message"]["content"])

    async def yes_prob(self, messages: list[dict], yes: str = "是", no: str = "否") -> float:
        """只让模型输出一个字，读「是」「否」的 logprob 换成概率（Jev 那种只答判断题的用法）。
        比让模型自由输出 SKIP 稳：同一句话不会这次答那次跳，还能按阈值调偏向。"""
        body = self._body(messages, 1, False) | {"temperature": 0, "logprobs": True, "top_logprobs": 5}
        r = await _post(self.client, self.url, self.headers, body)
        top = {x["token"].strip(): math.exp(x["logprob"]) for x in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        p_yes, p_no = top.get(yes, 0.0), top.get(no, 0.0)
        return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5


async def _deltas(r: httpx.Response):
    async for line in r.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError):
            continue
        if delta:
            yield delta


class Embedder:
    def __init__(self, base_url: str, api_key: str, model: str = "bge-m3", timeout: float = 60.0):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.headers = {"authorization": f"Bearer {api_key}"}
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls) -> "Embedder":
        return cls(os.environ["SOCLAAS_BASE_URL"], os.environ["SOCLAAS_API_KEY"], os.environ.get("COPILOT_EMBED_MODEL", "bge-m3"))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        r = await _post(self.client, self.url, self.headers, {"model": self.model, "input": texts})
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def parse_json(text: str) -> dict:
    """模型偶尔包一层 ```json，取第一个完整对象。"""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
