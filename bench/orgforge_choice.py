"""logprob 选择题 vs Jev：黑客松 OrgForge 257 句会议句子分类（11 类）。

同样的题、同样的选项说明、同样的前 3 句上下文。Jev 和 qwen-JSON 的答案取自黑客松存档
（personal_agent_long_context/eval/meetings/orgforge/results.json），这里只新跑 logprob 版本。

  python bench/orgforge_choice.py qwen3.6:35b [--concurrency 3]
"""

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import httpx

HACK = Path.home() / "Desktop/git/personal_agent_long_context/eval/meetings/orgforge"
OUT = Path(__file__).resolve().parent / "out"

# 原样照抄 run.ts 的 KINDS 与 CODE
KINDS = {
    "none": "not a commitment: discussion, an idea being floated, a report of work already done, small talk, or a question asking colleagues for their opinion",
    "task": "someone takes on, or is given, a piece of their own work (build, test, update, review, draft a design) that needs nothing sent on their behalf",
    "answer_question": "a direct question about a fact recorded in company systems (a past ticket, incident, version, customer, or decision) that someone asks now",
    "email_draft": "someone promises to send an email or a written follow-up or summary to people",
    "message_draft": "someone promises a quick chat message (WhatsApp, Teams) carrying a specific date, number or decision to someone not in the meeting",
    "calendar_draft": "someone commits to setting up or sending an invite for a meeting",
    "ticket_draft": "someone commits to opening a ticket or task for work to be done",
    "doc_draft": "someone promises to write up a new internal document such as notes, a spec or a checklist",
    "sheet_draft": "someone promises to put together a new table or spreadsheet",
    "hiring_request": "someone says the company needs to hire a new person",
    "escalation": "someone offers or commits money: a discount, refund, credit, payment, price change, or signing a contract",
}
CODE = {"N": "none", "T": "task", "Q": "answer_question", "E": "email_draft", "M": "message_draft", "C": "calendar_draft", "K": "ticket_draft", "D": "doc_draft", "S": "sheet_draft", "H": "hiring_request", "X": "escalation"}
INSTRUCTIONS = "What is the last line of the meeting (the line field), given the lines before it?"

SYSTEM_LETTER = (
    "Answer the question about the state by choosing one option.\n"
    "Options:\n" + "\n".join(f"{code}: {KINDS[kind]}" for code, kind in CODE.items()) + "\n"
    "Reply with the option letter only."
)
# 选项名版：输出选项名本身，读首 token。字母代码没有语义，一步作答时模型得先把 T 映射回「task」才能判断
SYSTEM_KEY = (
    "Answer the question about the state by choosing one option.\n"
    "Options:\n" + "\n".join(f"{kind}: {desc}" for kind, desc in KINDS.items()) + "\n"
    "Reply with the option name only, exactly as written before the colon."
)


def items() -> list[dict]:
    meetings = json.loads((HACK / "lines.json").read_text())
    labels = {}
    for line in (HACK / "labels.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            key, *codes = line.split(" ")
            labels[key] = [CODE[c] for c in codes]
    out = []
    for mi, m in enumerate(meetings):
        for li, line in enumerate(m["lines"]):
            out.append({
                "key": f"{mi}.{li}",
                "accepted": labels[f"{mi}.{li}"],
                "state": {
                    "meeting": m["title"],
                    "before": [f"{l['speaker']}: {l['text']}" for l in m["lines"][max(0, li - 3) : li]],
                    "line": f"{line['speaker']}: {line['text']}",
                },
            })
    return out


def first_token_owner(token: str, fmt: str) -> str | None:
    token = token.strip()
    if fmt == "letter":
        return CODE.get(token.upper())
    owners = [kind for kind in KINDS if token and kind.startswith(token.lower())]
    return owners[0] if len(owners) == 1 else None


async def classify(client: httpx.AsyncClient, model: str, state: dict, fmt: str) -> dict:
    body = {
        "model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": SYSTEM_LETTER if fmt == "letter" else SYSTEM_KEY},
            {"role": "user", "content": json.dumps({"state": state, "question": INSTRUCTIONS}, ensure_ascii=False)},
        ],
    }
    started = time.monotonic()
    for attempt in range(8):
        r = await client.post(os.environ["SOCLAAS_BASE_URL"] + "/chat/completions", headers={"authorization": "Bearer " + os.environ["SOCLAAS_API_KEY"]}, json=body)
        if r.status_code == 200 and "choices" in r.json():
            break
        await asyncio.sleep(1 + attempt)
    else:
        return {"choice": "ERROR", "p": 0, "ms": 0}
    ms = round((time.monotonic() - started) * 1000)
    probs: dict[str, float] = {}
    for x in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        kind = first_token_owner(x["token"], fmt)
        if kind:
            probs[kind] = probs.get(kind, 0) + math.exp(x["logprob"])
    total = sum(probs.values())
    if not total:
        return {"choice": "ERROR", "p": 0, "ms": ms}
    probs = {k: v / total for k, v in probs.items()}
    choice = max(probs, key=probs.get)
    return {"choice": choice, "p": probs[choice], "ms": ms, "probs": probs}


def summarize(rows: list[dict], who: str) -> str:
    card = lambda k: k not in ("none", "task", "answer_question", "ERROR", "")
    ok = [r for r in rows if r[who]["choice"] != "ERROR"]
    right = [r for r in ok if r[who]["choice"] in r["accepted"]]
    ms = sorted(r[who]["ms"] for r in ok)
    want_card = [r for r in rows if card(r["accepted"][0])]
    got_card = [r for r in ok if card(r[who]["choice"])]
    false_card = [r for r in got_card if not any(card(a) for a in r["accepted"])]
    lines = [f"{who}: 出错 {len(rows) - len(ok)}，准确率 {len(right)}/{len(ok)} = {len(right) / len(ok):.1%}，p50 {ms[len(ms) // 2]}ms，p90 {ms[int(len(ms) * .9)]}ms；"
             f"该开卡 {len(want_card)} 抓到 {sum(r[who]['choice'] in r['accepted'] for r in want_card)}，误开卡 {len(false_card)}"]
    if all("p" in r[who] for r in ok) and who != "qwen":
        for t in (0.6, 0.8, 0.9):
            sure = [r for r in ok if r[who]["p"] >= t]
            if sure:
                lines.append(f"    p≥{t}: 覆盖 {len(sure)}/{len(ok)}，其中对 {sum(r[who]['choice'] in r['accepted'] for r in sure) / len(sure):.1%}")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--format", choices=["letter", "key"], default="key")
    args = ap.parse_args()
    archived = {r["key"]: r for r in json.loads((HACK / "results.json").read_text())["rows"]}
    todo = items()
    client = httpx.AsyncClient(timeout=60)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(item):
        async with gate:
            return await classify(client, args.model, item["state"], args.format)

    answers = await asyncio.gather(*(one(i) for i in todo))
    tag = f"lp_{args.format}_" + args.model.replace(":", "_")
    rows = [{"key": i["key"], "line": i["state"]["line"], "accepted": i["accepted"], "jev": archived[i["key"]]["jev"], "qwen": archived[i["key"]]["qwen"], tag: a} for i, a in zip(todo, answers)]
    OUT.mkdir(exist_ok=True)
    (OUT / f"orgforge-{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for who in ("jev", "qwen", tag):
        print(summarize(rows, who))


if __name__ == "__main__":
    asyncio.run(main())
