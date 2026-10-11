import asyncio

from copilot.assistant import INTERVIEWER, ME, Assistant
from copilot.coding import CodingCoach, section, strip_fence, syntax_check
from copilot.kb import KnowledgeBase, chunk_markdown

ANSWER = "【思路判断】对\n【现在说】我用迭代\n【完整代码】\n```python\nprint(1)\n```\n【我的代码】目前没问题\n【复杂度与边界】O(n)"


class FakeModel:
    def __init__(self, answer=ANSWER, delay=0.0):
        self.answer, self.delay, self.prompts = answer, delay, []

    async def json(self, messages, max_tokens):
        return {"is_question": True, "kind": "八股", "question": "q", "queries": []}

    async def yes_prob(self, messages, yes="是", no="否"):
        return 0.9

    async def stream(self, messages, max_tokens):
        self.prompts.append(messages)
        for i in range(0, len(self.answer), 20):
            await asyncio.sleep(self.delay)
            yield self.answer[i : i + 20]


def page(code="def f(): pass", **kw):
    return {"type": "page", "url": "https://leetcode.cn/problems/reverse-linked-list/", "title": "206. 反转链表",
            "problem": "给你单链表的头节点 head，请你反转链表", "code": code, "lang": "python3", "output": "", "editor": "monaco"} | kw


def make(model=None, idle=0.02):
    out = []
    return CodingCoach(model or FakeModel(), out.append, code_idle_s=idle), out


def types(out):
    return [m["type"] for m in out if m["type"] != "coding_input"]


async def test_first_editor_page_enters_coding_mode_and_answers():
    coach, out = make()
    coach.on_page(page())
    await coach.wait()
    assert types(out)[:3] == ["coding_mode", "coding_page", "coding_start"] and out[0]["on"]
    assert next(m for m in out if m["type"] == "coding_done")["text"] == ANSWER
    assert coach.s.last_code == "print(1)"


async def test_page_without_an_editor_is_ignored():
    coach, out = make()
    coach.on_page(page(code="", editor=""))
    await asyncio.sleep(0.01)
    assert out == [] and not coach.active


async def test_typing_is_debounced_into_one_update():
    model = FakeModel()
    coach, out = make(model, idle=0.03)
    coach.on_page(page())
    await coach.wait()
    for i in range(5):  # 连续敲字
        coach.on_page(page(code=f"def f(): pass  # {i}"))
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.06)
    await coach.wait()
    assert types(out).count("coding_start") == 2
    assert "# 4" in model.prompts[-1][1]["content"]


async def test_run_output_and_speech_trigger_at_once_and_rerun_after_a_busy_generation():
    model = FakeModel(delay=0.01)
    coach, out = make(model, idle=10)
    coach.on_page(page())
    await asyncio.sleep(0.005)  # 第一次还在生成
    coach.on_final(INTERVIEWER, "尽量写迭代，不要递归")
    coach.on_final(ME, "我觉得可以用栈先存起来")
    coach.on_page(page(output="Traceback: NameError: name 'cur' is not defined"))
    await coach.wait()
    assert types(out).count("coding_start") == 2  # 忙的时候来的都并成一次重跑
    last = model.prompts[-1][1]["content"]
    assert "尽量写迭代" in last and "候选人：我觉得可以用栈" in last and "NameError" in last


async def test_short_backchannel_does_not_trigger():
    coach, out = make()
    coach.on_page(page())
    await coach.wait()
    coach.on_final(INTERVIEWER, "嗯，对")
    await asyncio.sleep(0.01)
    await coach.wait()
    assert types(out).count("coding_start") == 1


async def test_same_code_is_not_rewritten():
    model = FakeModel()
    coach, out = make(model)
    coach.on_page(page())
    await coach.wait()
    model.answer = ANSWER.replace("```python\nprint(1)\n```", "同上")
    coach.on_final(INTERVIEWER, "可以，你写吧，写完跑一下")
    await coach.wait()
    assert coach.s.last_code == "print(1)"
    assert "上一版完整代码：\n```\nprint(1)\n```" in model.prompts[-1][1]["content"]


async def test_stop_leaves_coding_mode():
    coach, out = make()
    coach.on_page(page())
    await coach.wait()
    coach.stop()
    coach.on_final(INTERVIEWER, "好，那我们聊聊别的项目")
    await asyncio.sleep(0.01)
    assert out[-1] == {"type": "coding_mode", "on": False} and not coach.active


async def test_assistant_question_cards_pause_while_coding():
    events = []
    kb = KnowledgeBase(chunk_markdown("## MQ\n### Kafka\n1. **消息积压**\n   - 加分区加消费者\n"))
    a = Assistant(kb, FakeModel(), FakeModel(), events.append, silence_s=0.01)
    coding = True
    a.paused = lambda: coding
    a.on_final(INTERVIEWER, "这里 cur 为什么要先存下来？")
    await asyncio.sleep(0.03)
    assert events == []


def test_sections_are_split_by_headers():
    assert section(ANSWER, "现在说") == "我用迭代"
    assert strip_fence(section(ANSWER, "完整代码")) == "print(1)"
    assert section(ANSWER, "复杂度与边界") == "O(n)"
    assert section("【思路判断】还在写", "完整代码") == ""


def test_python_syntax_is_checked_locally_and_given_to_the_model():
    assert syntax_check("x = 1\n32323\n", "python3") == "没有语法错误"  # 单独一个数字是合法语句
    err = syntax_check("x = 1\n3232332 尼手表复查 3\n", "python3")
    assert err.startswith("第 2 行语法错误")
    assert syntax_check("int main() {}", "cpp") == ""


async def test_syntax_error_reaches_the_prompt():
    model = FakeModel()
    coach, out = make(model)
    coach.on_page(page(code="def f():\n    return 1\n    3232332 尼手表复查 3\n"))
    await coach.wait()
    assert "本地语法检查（编译器结果，以它为准）：第 3 行语法错误" in model.prompts[-1][1]["content"]


async def test_partial_code_read_from_the_screen_is_flagged_and_not_syntax_checked():
    model = FakeModel()
    coach, out = make(model)
    coach.on_page(page(code="def f(:\n    pass", lang="（只读到看得见的第 1 到 2 行）", editor="monaco-dom"))
    await coach.wait()
    prompt = model.prompts[-1][1]["content"]
    assert "只读到看得见的第 1 到 2 行" in prompt and "看不到的部分不要当成没写" in prompt
    assert "本地语法检查" not in prompt


async def test_interviewer_asking_to_code_enters_before_any_page():
    model = FakeModel()
    coach, out = make(model)
    coach.start("好，那我们来写一道题吧")
    await coach.wait()
    assert out[0] == {"type": "coding_mode", "on": True}
    prompt = model.prompts[-1][1]["content"]
    assert "还没读到编程页面" in prompt and "面试官：好，那我们来写一道题吧" in prompt
    assert coach.s.last_page_at >= coach.s.started  # 不会被「120 秒没页面」立刻踢出


async def test_stage_finished_leaves_coding_mode():
    coach, out = make(FakeModel(answer="【阶段】结束\n" + ANSWER))
    coach.on_page(page())
    await coach.wait()
    assert out[-1] == {"type": "coding_mode", "on": False} and not coach.active


async def test_assistant_hands_a_coding_request_to_the_coach():
    class Plan(FakeModel):
        async def json(self, messages, max_tokens):
            return {"is_question": True, "kind": "手撕", "question": "写反转链表", "queries": []}
    events, handed = [], []
    kb = KnowledgeBase(chunk_markdown("## MQ\n### Kafka\n1. **消息积压**\n   - 加分区加消费者\n"))
    a = Assistant(kb, FakeModel(), Plan(), events.append, silence_s=0.01)
    a.on_coding = handed.append
    a.on_final(INTERVIEWER, "那我们来写一道题，反转链表")
    await asyncio.sleep(0.03)
    await a.wait()
    assert handed == ["那我们来写一道题，反转链表"]
    assert "slow_token" not in [e.kind for e in events]
    assert next(e for e in events if e.kind == "slow_skip").data["coding"]
