import asyncio

import pytest

from copilot.assistant import INTERVIEWER, ME, Assistant
from copilot.kb import KnowledgeBase, chunk_markdown

KB = KnowledgeBase(chunk_markdown("## MQ\n### Kafka\n1. **消息积压**\n   - 加分区加消费者\n"))


class FakeModel:
    def __init__(self, tokens=("加分区", "，加消费者"), plan=None, delay=0.0, p_yes=0.9, gate_delay=0.0):
        self.tokens, self.plan, self.delay = tokens, plan or {"is_question": True, "kind": "八股", "question": "q", "queries": ["消息积压"]}, delay
        self.p_yes, self.gate_delay = p_yes, gate_delay
        self.prompts = []

    async def yes_prob(self, messages, yes="是", no="否"):
        await asyncio.sleep(self.gate_delay)
        if isinstance(self.p_yes, Exception):
            raise self.p_yes
        return self.p_yes

    async def stream(self, messages, max_tokens):
        self.prompts.append(messages)
        for t in self.tokens:
            await asyncio.sleep(self.delay)
            yield t

    async def json(self, messages, max_tokens):
        self.prompts.append(messages)
        await asyncio.sleep(self.delay)
        return self.plan


def make(fast=None, slow=None, silence=0.05):
    events = []
    a = Assistant(KB, fast or FakeModel(), slow or FakeModel(), events.append, silence_s=silence)
    return a, events


def kinds(events, qid=None):
    return [e.kind for e in events if qid is None or e.qid == qid]


async def test_silence_after_interviewer_fires_fast_and_slow():
    a, events = make()
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    await asyncio.sleep(0.1)
    await a.wait()
    k = kinds(events)
    assert k[0] == "question" and "kb_hit" in k and "fast_done" in k and "slow_plan" in k and "slow_done" in k
    fast_done = next(e for e in events if e.kind == "fast_done")
    assert fast_done.text == "加分区，加消费者"


async def test_my_speech_start_fires_without_waiting_for_silence():
    a, events = make(silence=10)
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    a.on_speech_start(ME)
    assert kinds(events)[:1] == ["question"]
    await a.wait()


async def test_interviewer_pieces_in_one_turn_are_asked_as_one_question():
    a, events = make(silence=10)
    a.on_final(INTERVIEWER, "那我先问一个问题，")
    a.on_final(INTERVIEWER, "一个 topic 可以有多个 group 消费吗？")
    a.on_speech_start(ME)
    q = [e for e in events if e.kind == "question"]
    assert [e.text for e in q] == ["那我先问一个问题，一个 topic 可以有多个 group 消费吗？"]
    await a.wait()


async def test_interviewer_continuing_after_pause_appends_instead_of_rewriting():
    """面试官停顿后补充：已经给出的骨架不取消、不重写，第二次只针对新增的部分补。"""
    slow = FakeModel(delay=0.05)
    a, events = make(slow=slow, silence=0.02)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    await asyncio.sleep(0.04)  # 已触发，慢答还在跑
    a.on_final(INTERVIEWER, "比如消费者挂了的情况")
    await asyncio.sleep(0.04)
    await a.wait()
    questions = [e for e in events if e.kind == "question"]
    assert [e.text for e in questions] == ["消息积压怎么办？", "消息积压怎么办？比如消费者挂了的情况"]
    first, second = questions[0].qid, questions[1].qid
    assert "cancelled" not in kinds(events) and "slow_done" in kinds(events, first)
    assert questions[1].data.get("append") is True
    # 第二次等第一版写完，带着第一版去写补充
    done_first = next(i for i, e in enumerate(events) if e.qid == first and e.kind == "slow_done")
    plan_second = next(i for i, e in enumerate(events) if e.qid == second and e.kind == "slow_plan")
    assert events[plan_second].data["mode"] == "append" and plan_second > done_first
    prompt = slow.prompts[-1][1]["content"]
    assert "加分区，加消费者" in prompt and "已经给出的骨架" in prompt


class PlanSequence(FakeModel):
    def __init__(self, plans, **kw):
        super().__init__(**kw)
        self.plans = list(plans)

    async def json(self, messages, max_tokens):
        self.prompts.append(messages)
        return self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]


async def test_append_after_a_skipped_first_version_answers_in_full():
    """第一版只听到半句、被判为不是问题：补充后照常完整作答。"""
    no = {"is_question": False, "kind": "其他", "question": "", "queries": []}
    yes = {"is_question": True, "kind": "八股", "question": "q", "queries": ["消息积压"]}
    slow = PlanSequence([no, yes])
    a, events = make(slow=slow, silence=0.02)
    a.on_final(INTERVIEWER, "那那那那")
    await asyncio.sleep(0.04)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    await asyncio.sleep(0.04)
    await a.wait()
    plan = next(e for e in events if e.kind == "slow_plan")
    assert plan.data["mode"] == "full" and "已经给出的骨架" not in slow.prompts[-1][1]["content"]


async def test_append_skips_the_fast_answer_already_shown():
    fast = FakeModel(delay=0.01)
    a, events = make(fast=fast, silence=0.02)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    await asyncio.sleep(0.06)
    a.on_final(INTERVIEWER, "比如消费者挂了的情况")
    await asyncio.sleep(0.04)
    await a.wait()
    second = [e for e in events if e.kind == "question"][1].qid
    assert "fast_token" not in kinds(events, second)


async def test_append_still_gets_a_fast_answer_when_the_first_was_gated_out():
    """第一版是「那那那」被门控拦下，没上屏：补充成真问题后要有开口句。"""
    fast = FakeModel(delay=0.01, p_yes=0.1)
    a, events = make(fast=fast, silence=0.02)
    a.on_final(INTERVIEWER, "那那那那那")
    await asyncio.sleep(0.06)
    fast.p_yes = 0.9
    a.on_final(INTERVIEWER, "为什么不都用 ES？")
    await asyncio.sleep(0.06)
    await a.wait()
    second = [e for e in events if e.kind == "question"][1].qid
    assert "fast_token" in kinds(events, second)


async def test_repeated_trigger_with_same_text_does_not_reask():
    a, events = make(silence=0.02)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    await asyncio.sleep(0.05)
    a.on_speech_start(ME)  # 已按静音触发过，同样的话不再问
    await a.wait()
    assert kinds(events).count("question") == 1


async def test_gate_saying_no_hides_the_fast_answer():
    a, events = make(fast=FakeModel(p_yes=0.05))
    a.on_final(INTERVIEWER, "明白了明白了")
    a.on_speech_start(ME)
    await a.wait()
    assert "fast_skip" in kinds(events) and "fast_token" not in kinds(events)


async def test_tokens_wait_for_a_slow_gate_then_come_out_together():
    a, events = make(fast=FakeModel(gate_delay=0.05))
    a.on_final(INTERVIEWER, "慢查询怎么排查？")
    a.on_speech_start(ME)
    await a.wait()
    k = kinds(events)
    assert k.index("gate") < k.index("fast_token")
    assert [e.text for e in events if e.kind == "fast_token"] == ["加分区，加消费者"]


async def test_gate_failure_still_shows_the_fast_answer():
    a, events = make(fast=FakeModel(p_yes=RuntimeError("gateway down")))
    a.on_final(INTERVIEWER, "慢查询怎么排查？")
    a.on_speech_start(ME)
    await a.wait()
    assert next(e for e in events if e.kind == "fast_done").text == "加分区，加消费者"


async def test_slow_stops_when_plan_says_not_a_question():
    slow = FakeModel(plan={"is_question": False})
    a, events = make(slow=slow)
    a.on_final(INTERVIEWER, "对，他的观感不好。")
    a.on_speech_start(ME)
    await a.wait()
    assert "slow_skip" in kinds(events) and "slow_token" not in kinds(events)


async def test_short_backchannel_never_calls_the_model():
    fast = FakeModel()
    a, events = make(fast=fast)
    a.on_final(INTERVIEWER, "好。")
    a.on_speech_start(ME)
    await a.wait()
    assert events == [] and fast.prompts == []


async def test_one_path_failing_leaves_the_other_running():
    class Broken(FakeModel):
        async def json(self, messages, max_tokens):
            raise RuntimeError("gateway down")

    a, events = make(slow=Broken())
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    a.on_speech_start(ME)
    await a.wait()
    assert "error" in kinds(events) and "fast_done" in kinds(events)


async def test_question_i_started_answering_is_locked_and_anchors_the_next():
    slow = FakeModel(delay=0.05)
    a, events = make(slow=slow, silence=10)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    a.on_speech_start(ME)
    a.on_final(ME, "加消费者")
    a.on_final(INTERVIEWER, "那分区数不够呢？")  # 我还在照着第一问说，面试官插进来
    a.on_speech_start(ME)
    await a.wait()
    first, second = [e.qid for e in events if e.kind == "question"]
    # 我照着说的那一问不取消，写完
    assert "locked" in kinds(events, first) and "cancelled" not in kinds(events, first)
    assert "slow_done" in kinds(events, first) and "slow_done" in kinds(events, second)
    # 第二问的口述稿带着我说过的话和第一问那版稿子，要求接着说、有错圆过去
    prompt = slow.prompts[-1][1]["content"]
    assert "加消费者" in prompt and "上一问：消息积压怎么办？" in prompt and "加分区，加消费者" in prompt
    assert "我补充一点" in prompt
    # 两问是面试官的两轮
    turns = {e.qid: e.turn for e in events if e.kind == "question"}
    assert turns[first] != turns[second]


async def test_reask_within_a_turn_keeps_the_turn():
    a, events = make(fast=FakeModel(delay=0.05), slow=FakeModel(delay=0.05), silence=0.02)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    await asyncio.sleep(0.04)
    a.on_final(INTERVIEWER, "线上真遇到过吗？")
    await asyncio.sleep(0.04)
    a.on_speech_start(ME)
    a.on_partial(ME, "遇到过，大促的时候")
    await a.wait()
    questions = [e for e in events if e.kind == "question"]
    assert len(questions) == 2 and questions[0].turn == questions[1].turn
    assert "locked" in kinds(events, questions[1].qid)


MIXED = KnowledgeBase(
    chunk_markdown("## MQ\n### Kafka\n1. **消息积压**\n   - 加分区加消费者\n")
    + chunk_markdown("# 检索\n## 检索链路怎么分层？\n三层。\n# 车辆\n## 分配系统解决什么问题？\n派车。\n", "项目", False, first_id=1)
)


async def run_project(ref):
    slow = FakeModel(plan={"is_question": True, "kind": "项目", "question": "检索怎么分层", "queries": ["检索 分层"], "project_ref": ref})
    events = []
    a = Assistant(MIXED, FakeModel(), slow, events.append, silence_s=10)
    a.on_final(INTERVIEWER, "你们检索链路怎么分层的？")
    a.on_speech_start(ME)
    await a.wait()
    plan = next(e for e in events if e.kind == "slow_plan")
    return plan, slow.prompts[-1]


async def test_project_question_with_a_real_ref_uses_that_answer_and_the_five_part_style():
    plan, prompt = await run_project("项目｜检索 / 检索链路怎么分层？")
    assert plan.data["type"] == "项目"
    assert plan.data["hits"][0] == "项目｜检索 / 检索链路怎么分层？"
    assert "方案对比" in prompt[0]["content"]


async def test_project_question_with_a_misquoted_or_missing_ref_gets_no_template_and_no_other_projects():
    for ref in ("项目｜检索 / 检索链路如何分层", None):
        plan, prompt = await run_project(ref)
        assert plan.data["type"] == "项目无口径"
        assert not any(label.startswith("项目｜") for label in plan.data["hits"])
        assert "方案对比" not in prompt[0]["content"]


async def test_interviewer_starting_the_next_sentence_holds_the_trigger():
    a, events = make(silence=0.03)
    a.on_final(INTERVIEWER, "那我先问一个问题，")
    a.on_speech_start(INTERVIEWER)  # VAD：面试官紧接着又开口
    await asyncio.sleep(0.06)  # 超过静音阈值也不触发，因为还在说
    assert "question" not in kinds(events)
    a.on_final(INTERVIEWER, "一个 topic 可以有多个 group 消费吗？")
    await asyncio.sleep(0.06)
    await a.wait()
    assert [e.text for e in events if e.kind == "question"] == ["那我先问一个问题，一个 topic 可以有多个 group 消费吗？"]


async def test_skipped_question_is_not_used_as_anchor():
    """我开口时锁定的是一句残句（慢答判为不是问题），后面的问题要依托再前面那张真答过的。"""
    no = FakeModel(plan={"is_question": False, "kind": "其他", "question": "", "queries": []})
    yes = FakeModel(tokens=("过渡：刚才说了加分区\n", "· 加消费者"))
    a, events = make(slow=yes, silence=10)
    a.on_final(INTERVIEWER, "消息积压怎么办？")
    a.on_speech_start(ME)
    a.on_final(ME, "加分区加消费者")
    await a.wait()
    a.slow_model = no
    a.on_final(INTERVIEWER, "项目里面实际占用这一块")
    a.on_speech_start(ME)
    a.on_final(ME, "占用大概三成")
    await a.wait()
    a.slow_model = yes
    a.on_final(INTERVIEWER, "那分区数不够呢？")
    a.on_speech_start(ME)
    await a.wait()
    prompt = yes.prompts[-1][1]["content"]
    assert "上一问：消息积压怎么办？" in prompt and "实际占用" not in prompt.split("上一问")[1]
    assert "过渡：刚才说了加分区" not in prompt  # 那版自己的过渡句不给，免得照抄


async def test_my_speech_is_the_question_until_the_interviewer_has_spoken():
    # 没共享会议、自己对着麦克风练：面试官那路一个字没出过，我说的话就当问题
    a, events = make()
    a.on_speech_start(ME)
    a.on_final(ME, "Kafka 消息积压怎么处理？")
    await asyncio.sleep(0.1)
    await a.wait()
    assert kinds(events)[:1] == ["question"]
    assert a.history[-1].speaker == INTERVIEWER
    # 面试官一开口就回到正常：我再说话是在回答，不出题
    a.on_final(INTERVIEWER, "说说 Kafka 分区。")
    await asyncio.sleep(0.1)
    await a.wait()
    n = len([e for e in events if e.kind == "question"])
    a.on_final(ME, "分区是并行的单位，加分区加消费者。")
    await asyncio.sleep(0.1)
    await a.wait()
    assert len([e for e in events if e.kind == "question"]) == n
    assert a.history[-1].speaker == ME


async def test_backchannel_or_noise_does_not_end_the_turn():
    """VAD 报我开口但只转出附和（或什么都没转出）：面试官接着补的话归同一轮，原卡片不被挤掉，也不锁定。"""
    a, events = make(silence=10)
    a.on_final(INTERVIEWER, "AI coding 的整个流程是什么样的？")
    a.on_speech_start(ME)  # 杂音
    a.on_final(ME, "嗯嗯，对对")
    a.on_final(INTERVIEWER, "然后你是怎么做这件事情的？")
    a.on_speech_start(ME)
    a.on_partial(ME, "嗯，首先我们拿到需求")
    await a.wait()
    questions = [e for e in events if e.kind == "question"]
    assert len(questions) == 2 and questions[0].turn == questions[1].turn
    assert questions[1].data["append"]
    assert "locked" not in kinds(events, questions[0].qid) and "locked" in kinds(events, questions[1].qid)


def relation_model(relation, tokens=("加分区", "，加消费者")):
    return FakeModel(tokens=tokens, plan={"relation": relation, "is_question": True, "kind": "八股", "question": "q", "queries": ["消息积压"]})


async def answer_then_interviewer_again(slow):
    """面试官问 → 我答出实话（锁定）→ 面试官再开口。"""
    a, events = make(slow=slow, silence=10)
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    a.on_speech_start(ME)
    a.on_final(ME, "加分区加消费者")
    await a.wait()
    a.on_final(INTERVIEWER, "那分区数已经很多了呢？")
    a.on_speech_start(ME)
    await a.wait()
    first, second = [e for e in events if e.kind == "question"]
    return events, first, second


async def test_followup_after_my_answer_merges_into_the_card_and_is_written_in_full():
    slow = relation_model("追问")
    events, first, second = await answer_then_interviewer_again(slow)
    assert second.data["provisional"] and second.turn != first.turn  # 先上临时卡
    rel = next(e for e in events if e.kind == "relation")
    assert rel.text == "追问" and rel.data["merge_into"] == first.turn
    after = [e for e in events if e.qid == second.qid and events.index(e) > events.index(rel)]
    assert after and all(e.turn == first.turn for e in after)  # 之后的字都写进原卡
    plan = next(e for e in after if e.kind == "slow_plan")
    assert plan.data["mode"] == "followup"
    prompt = slow.prompts[-1][1]["content"]
    assert "追问了一层" in prompt and "加分区，加消费者" in prompt  # 看着原卡写，按完整答案的规格
    judge = slow.prompts[-2]
    # 原卡给改写后的问题（已按上下文纠正错字）和原话
    assert "当前卡片的主问题：q（原话：Kafka 消息积压怎么处理？）" in judge[1]["content"] and '"relation"' in judge[0]["content"]


async def test_new_topic_after_my_answer_keeps_its_own_card():
    events, first, second = await answer_then_interviewer_again(relation_model("新问题"))
    rel = next(e for e in events if e.kind == "relation")
    assert rel.text == "新问题" and "merge_into" not in rel.data
    assert all(e.turn == second.turn for e in events if e.qid == second.qid)
    assert next(e for e in events if e.kind == "slow_plan" and e.qid == second.qid).data["mode"] == "full"


async def test_first_question_has_no_card_to_compare_with():
    slow = relation_model(None)
    a, events = make(slow=slow, silence=10)
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    a.on_speech_start(ME)
    await a.wait()
    assert not next(e for e in events if e.kind == "question").data["provisional"]
    assert '"relation"' not in slow.prompts[0][0]["content"] and "relation" not in kinds(events)


async def test_interviewer_filler_on_a_provisional_card_does_not_swallow_the_real_follow_up():
    """我答过后面试官先说一句附和（临时卡被判为不是问题），接着才问：后面这段要重新和原卡比，能并回原卡。"""
    class Judge(FakeModel):
        async def json(self, messages, max_tokens):
            self.prompts.append(messages)
            said = messages[1]["content"].split("面试官最后这段话：")[-1]
            if "了解" in said and "分区" not in said:
                return {"relation": "补充", "is_question": False}
            return {"relation": "追问", "is_question": True, "kind": "八股", "question": "q", "queries": ["消息积压"]}
    a, events = make(slow=Judge(), silence=0.02)
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    a.on_speech_start(ME)
    a.on_final(ME, "加分区加消费者")
    await a.wait()
    a.on_final(INTERVIEWER, "嗯，好的好的，了解")
    await asyncio.sleep(0.05)
    await a.wait()
    a.on_final(INTERVIEWER, "那分区数已经很多了呢？")
    await asyncio.sleep(0.05)
    await a.wait()
    first, filler, real = [e for e in events if e.kind == "question"]
    assert filler.data["provisional"] and real.data["provisional"] and not real.data["append"]
    rel = next(e for e in events if e.kind == "relation" and e.qid == real.qid)
    assert rel.data["merge_into"] == first.turn


async def test_follow_up_arriving_before_the_filler_card_is_judged_still_gets_merged():
    """临时卡还没判完面试官就接着问了：等它判完，作废了就让后面这段自己再和原卡比一次。"""
    class Judge(FakeModel):
        async def json(self, messages, max_tokens):
            self.prompts.append(messages)
            said = messages[1]["content"].split("面试官最后这段话：")[-1]
            await asyncio.sleep(0.05)
            if "分区" not in said and "了解" in said:
                return {"relation": "补充", "is_question": False}
            return {"relation": "追问", "is_question": True, "kind": "八股", "question": "q", "queries": ["消息积压"]}
    a, events = make(slow=Judge(), silence=0.01)
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    a.on_speech_start(ME)
    a.on_final(ME, "加分区加消费者")
    await a.wait()
    a.on_final(INTERVIEWER, "嗯，好的好的，了解")
    await asyncio.sleep(0.02)  # 已触发，慢答还没判完
    a.on_final(INTERVIEWER, "那分区数已经很多了呢？")
    await asyncio.sleep(0.03)
    await a.wait()
    first, filler, real = [e for e in events if e.kind == "question"]
    assert real.data["append"]  # 先接在临时卡后面
    rel = next(e for e in events if e.kind == "relation" and e.qid == real.qid)
    assert rel.data["merge_into"] == first.turn
    assert next(e for e in events if e.kind == "slow_plan" and e.qid == real.qid).data["mode"] == "followup"


async def test_interviewer_finishing_the_same_sentence_is_not_answered_twice():
    """同一轮里面试官把一段话分几次说完：后面几版按补充写，不按追问重写一遍。"""
    slow = relation_model("追问")
    a, events = make(slow=slow, silence=0.01)
    a.on_final(INTERVIEWER, "Kafka 消息积压怎么处理？")
    a.on_speech_start(ME)
    a.on_final(ME, "加分区加消费者")
    await a.wait()
    a.on_final(INTERVIEWER, "那分区数已经很多了呢？")
    await asyncio.sleep(0.03)
    a.on_final(INTERVIEWER, "还能怎么办？")
    await asyncio.sleep(0.03)
    await a.wait()
    modes = [e.data["mode"] for e in events if e.kind == "slow_plan"]
    assert modes == ["full", "followup", "append"]


class Rechecker(FakeModel):
    """第一次判定给原理解，复核时按 changed 返回。"""

    def __init__(self, changed, **kw):
        super().__init__(**kw)
        self.changed = changed

    async def json(self, messages, max_tokens):
        self.prompts.append(messages)
        if "候选人已经开始回答" in messages[1]["content"]:
            return {"changed": self.changed, "is_question": True, "kind": "八股", "question": "Superpowers 和 Matt skill 的区别", "queries": ["消息积压"]}
        return {"is_question": True, "kind": "八股", "question": "SDD 和 Serverless 的区别", "queries": ["消息积压"]}


async def answer_long(a):
    a.on_final(INTERVIEWER, "你有没有对比过它和 Serverless 的区别？")
    a.on_speech_start(ME)
    a.on_final(ME, "有的，super policy 是我们部门常用的一个 skill，")
    await asyncio.sleep(0.01)
    a.on_final(ME, "它是全量 TDD，而 Matt 的 skill 只在每个 seam 上测，所以轻很多。")
    await asyncio.sleep(0.02)
    await a.wait()


async def test_my_answer_corrects_a_misheard_question():
    slow = Rechecker(True)
    events = []
    a = Assistant(KB, FakeModel(), slow, events.append, silence_s=10, scene="候选人比较过 Superpowers 和 Matt Pocock 的 skill")
    await answer_long(a)
    plans = [e for e in events if e.kind == "slow_plan"]
    assert [p.data["mode"] for p in plans] == ["full", "correction"]
    first = next(e for e in events if e.kind == "question")
    fix = [e for e in events if e.kind == "question"][-1]
    assert fix.data["correction"] and fix.turn == first.turn  # 更正写在原卡上
    recheck = next(m for m in slow.prompts if "候选人已经开始回答" in m[1]["content"])
    assert "之前的理解：SDD 和 Serverless 的区别" in recheck[1]["content"] and "面试场景：" in recheck[1]["content"]
    write = slow.prompts[-1][1]["content"]
    assert "之前以为问的是「SDD 和 Serverless 的区别」" in write and "面试场景：" in write


async def test_understanding_that_holds_is_left_alone_and_checked_once():
    slow = Rechecker(False)
    events = []
    a = Assistant(KB, FakeModel(), slow, events.append, silence_s=10)
    await answer_long(a)
    a.on_final(ME, "再补充一句，Matt 的 skill 问得更多，执行更轻，适合长期迭代的项目。")
    await asyncio.sleep(0.02)
    await a.wait()
    assert [e.data["mode"] for e in events if e.kind == "slow_plan"] == ["full"]
    assert sum("候选人已经开始回答" in m[1]["content"] for m in slow.prompts) == 1
