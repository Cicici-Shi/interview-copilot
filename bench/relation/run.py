"""新问题/补充判定离线评测：在慢答第一步的 JSON 判定里加 relation 字段，对照人工标签。

  set -a; . ~/Downloads/soclaas.env; set +a
  python -m bench.relation.run bench/relation/cases-1010.json bench/relation/labels-1010.json --concurrency 4
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from copilot.assistant import HISTORY_FOR_PLAN, PLAN_SYSTEM, RELATION_FIELD, RELATION_RULE
from copilot.kb import KnowledgeBase
from copilot.llm import OpenAICompatible

ROOT = Path(__file__).resolve().parents[2]

RELATION = """
另外判断面试官最后这段话和「上一张卡片的问题」是什么关系，加一个字段：
 "relation": "补充" | "新问题"
补充 = 还在问上一张卡片那个问题：回答候选人刚才的追问或确认、换个说法再问一遍、给同一题加条件或提示、点名要候选人展开那个问题里已经问到的部分、上一段没说完接着说。
新问题 = 问了上一张卡片没问到的东西：换了话题，或在候选人的回答基础上往深一层问一个新的点。
候选人插话时要看他说了什么：他在追问细节、确认题意，面试官接下来的话多半是补充；他已经在作答，面试官接下来的多半是新问题。
is_question 为 false 时 relation 填 null。"""

# v2：对照物换成整张卡（主问题 + 已并进来的补充）；按 10-10 两类错补规则，规则按机制写，例子和那场无关
RELATION_V2 = """
另外判断面试官最后这段话和「当前卡片」是什么关系，加一个字段：
 "relation": "补充" | "新问题"
当前卡片 = 候选人正在回答的那个主问题，以及之后已经并进来的补充。按面试官这段话「要候选人回答的东西」判断，不按话题像不像：
补充 = 要的答案还在当前卡片的主问题里，包括：
- 面试官给提示、举例、补条件、给出部分答案或思路，帮候选人把这题答下去（例如问"怎么设计限流"，接着说"比如假设峰值是平时十倍"）
- 回答或确认候选人的追问、复述；候选人复述题目或问"是指 X 吗"之后，面试官接着解释或展开
- 面试官自己插问一个背景或前提（项目规模、是不是新系统、用的什么版本），问完还会回到主问题
- 换个说法再问一遍、教候选人按什么顺序答、点名要候选人展开主问题里已经问到的那部分
- 候选人只答了一部分，面试官追问"还有别的吗""只有这个吗"
新问题 = 要的答案超出了当前卡片的主问题：换了话题；或候选人已经答了主问题，面试官在这个回答上往深一层要一个新的东西（例如答完"用了哪个框架"，再问"那它和另一个框架比有什么区别"）。
面试官嘴上说"换个问题"不一定是新问题，看他实际问的是不是主问题里已经包含的部分。
is_question 为 false 时 relation 填 null；面试官在收尾、问候选人有没有问题要问他、或回答候选人的反问时，is_question 为 false。"""


def cards(cases, labels):
    """按标签切卡：每处给出它面对的「当前卡片」（主问题 + 已并进来的补充）。假设前面每一步都判对。"""
    turns = [cases[0]["prev_turn"]] + [c["new_turn"] for c in cases]
    out, card = [], [turns[0]]
    for k, c in enumerate(cases):
        out.append(card[:])
        g = labels[c["id"]]
        if g == "新问题":
            card = [turns[k + 1]]
        elif g == "补充":
            card.append(turns[k + 1])
    return out


async def judge(model, outline, case, sem, prompt, card):
    async with sem:
        if prompt == "v1":
            ref = f"上一张卡片的问题（面试官上一轮原话）：{case['prev_turn']}"
        else:
            ref = f"当前卡片的主问题：{card[0]}" + "".join(f"\n已并进来的补充：{x}" for x in card[1:])
        rule = {"v1": RELATION, "v2": RELATION_V2, "v3": RELATION_RULE}[prompt]
        system = PLAN_SYSTEM.replace("{outline}", outline)
        if prompt == "v2":  # 放进 JSON 格式里当第一个字段，放在末尾时模型常漏写
            system = system.replace('{"is_question":', '{"relation": "补充" | "新问题" | null（见文末说明，必须写）,\n "is_question":', 1)
        if prompt == "v3":  # 和线上同一份提示（copilot/assistant.py）
            system = system.replace('{"is_question":', RELATION_FIELD, 1)
        user = f"对话：\n" + "\n".join(case["history"][-HISTORY_FOR_PLAN:]) + f"\n\n{ref}\n\n面试官最后这段话：{case['new_turn']}"
        for attempt in range(3):
            try:
                return await model.json([{"role": "system", "content": system + rule},
                                         {"role": "user", "content": user}], max_tokens=320)
            except Exception as e:
                if attempt == 2:
                    return {"error": repr(e)[:200]}
                await asyncio.sleep(2 * (attempt + 1))


def verdict(plan):
    if "error" in plan:
        return "出错"
    if not plan.get("is_question", True):
        return "不用答"
    return plan.get("relation") if plan.get("relation") in ("补充", "追问", "新问题") else "缺失"


def severity(gold, got):
    """标签只有 补充/新问题/不用答 三类，新问题在这里指「要完整回答」，追问和新问题都算对。"""
    if gold == got or (gold == "新问题" and got == "追问"):
        return "对"
    if gold == "补充" and got == "追问":
        return "轻微：写多了"
    if gold == "补充" and got == "新问题":
        return "严重：挤掉原卡"
    if gold == "新问题" and got == "补充":
        return "严重：答薄了"
    return "严重：要不要答判反" if "不用答" in (gold, got) else "严重：" + got


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("labels")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt", default="v3", choices=["v1", "v2", "v3"])
    ap.add_argument("--skip", default="", help="不计分的 id（逗号分隔），比如标注有分歧的；标签为「排除」的也不计分")
    args = ap.parse_args()
    cases = json.load(open(args.cases))
    labels = {k: v for k, v in json.load(open(args.labels)).items() if not k.startswith("_")}
    kb = KnowledgeBase.from_sources([(ROOT / "kb/interview.md", "八股", True), (ROOT / "kb/projects.md", "项目", False), (ROOT / "kb/bagu-2026-10.md", "项目", False)])
    model = OpenAICompatible.from_env(args.model)
    sem = asyncio.Semaphore(args.concurrency)
    plans = await asyncio.gather(*(judge(model, kb.outline(), c, sem, args.prompt, card) for c, card in zip(cases, cards(cases, labels))))
    confusion, rows = Counter(), []
    skip = set(filter(None, args.skip.split(",")))
    for c, p in zip(cases, plans):
        gold, got = labels[c["id"]], verdict(p)
        if gold == "排除" or c["id"] in skip:
            continue
        confusion[(gold, got)] += 1
        rows.append({"id": c["id"], "gold": gold, "got": got, "plan": p, "new_turn": c["new_turn"][:80], "mine": c["mine_between"][:80]})
    n = len(rows)
    sev = Counter(severity(r["gold"], r["got"]) for r in rows)
    print(f"共 {n} 处：" + "，".join(f"{k} {v}" for k, v in sorted(sev.items())))
    for g in ("新问题", "补充", "不用答"):
        print(g, {k[1]: v for k, v in confusion.items() if k[0] == g})
    print("\n不一致：")
    for r in rows:
        if severity(r["gold"], r["got"]) != "对":
            print(f"{r['id']} 标 {r['gold']} 判 {r['got']} | 我：{r['mine'][:40]} | 新：{r['new_turn'][:60]}")
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
