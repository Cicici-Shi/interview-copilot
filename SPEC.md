# 提词器行为约定

这份是问题识别、卡片和界面行为的唯一出处。改其中任何一条：先改这里，再改代码和对应测试（每条后面标了测试名，在 `tests/test_assistant.py`，标 `live` 的在 `tests/test_live.py`）。
数字和方案对比在 `bench/RESULTS.md`，用法在 README。

## 1. 名词

- **轮（round）**：面试官连着说的一段，到我开始回答为止。
- **版（Question，qid）**：一次触发。同一轮里面试官说一段、停一下又接着说，会触发好几版。
- **卡片（card，事件里的 `turn`）**：屏幕上的一张卡，对应一个主问题，加上之后并进来的补充和追问。一张卡可以跨好几轮。
- **锚点（anchor）**：我照着开口回答的那一版。它不会被取消，后面的问题都接着它写。
- **临时卡**：我答过之后面试官再开口时先开的一张卡，等慢答判出关系再决定并回原卡，还是自己留下来。

## 2. 一轮的开始和结束

1. 面试官的话定稿，且这一轮还没开始：开新的一轮。
2. **我这一路转写出实话**才算一轮结束，中间结果和定稿都算；「嗯」「对对对」「好的」「OK」「明白」这类附和不算（`BACKCHANNEL`：去掉附和字后不足 2 个字）。一轮结束时锁定当前这一版。
   `test_backchannel_or_noise_does_not_end_the_turn`、`test_question_i_started_answering_is_locked_and_anchors_the_next`
3. 本地 VAD 报「我开口」只用来**提前触发**，不结束这一轮、不锁定。VAD 只看能量，附和、咳嗽、杂音都会报。（10-10 那场 94 次锁定里有 45 次之后我一个实字没说。）
   `test_my_speech_start_fires_without_waiting_for_silence`
4. 面试官那一路还没出过声时（没共享会议、自己对着麦克风练），我说的话当作面试官的问题。
   `test_my_speech_is_the_question_until_the_interviewer_has_spoken`

## 3. 触发

1. 面试官最后一句定稿后静音 0.8 秒，或者 VAD 报我开口，就触发；一轮里所有已定稿的话拼起来作为原话。
   `test_silence_after_interviewer_fires_fast_and_slow`、`test_interviewer_pieces_in_one_turn_are_asked_as_one_question`
2. VAD 报面试官又开口：取消静音计时，还没说完就不触发。`test_interviewer_starting_the_next_sentence_holds_the_trigger`
3. 原话和上次触发时一样就不再问；少于 4 个字不调模型。
   `test_repeated_trigger_with_same_text_does_not_reask`、`test_short_backchannel_never_calls_the_model`

## 4. 每一版放到哪张卡

| 情况 | 放哪 | 写法（见第 6 节） |
|---|---|---|
| 这一轮的第一版，前面没有锚点（开场第一题） | 新卡 | 完整 |
| 同一轮里面试官接着说 | 接在本轮上一版后面，同一张卡，不拆卡 | 补充（他多半在把同一段话说完；按追问写会同一段答好几遍） |
| 我答过（有锚点）之后面试官再开口 | 先开**临时卡**，原卡不撤 | 等慢答判出关系 |
| 临时卡判为补充 / 追问 | 并回原卡：接到原卡最新那一版后面，之后的事件都发到原卡 | 补充 / 追问 |
| 临时卡判为新问题 | 留下来当新卡，撤掉原卡 | 完整 |
| 临时卡判为不是问题 | 撤掉临时卡，原卡恢复成当前卡 | 不写 |
| 临时卡整张都被判为不是问题，同一轮面试官又接着说 | 重新开一张临时卡，再和原卡比一次（不接在作废的临时卡后面） | 等慢答判出关系 |
| 同上，但接着说时临时卡还没判完 | 先接在临时卡后面；慢答先等临时卡判完，作废了就脱离出来当临时卡再比一次 | 等慢答判出关系 |

- 卡片编号单独计数（`next_card`），和面试官第几轮（`round`）分开。
- 卡片归属看 base 链：一版的卡片编号等于它所在补充链最前面那一版的编号（`card_root`）。并卡就是把临时这一版的 `base` 接到原卡最新一版上，同一轮之后再补的版本会自动跟着过去。
- 「原卡」指锚点所在的卡；锚点那张卡如果每一版都被判为不是问题，就往前找上一张真答过的（`anchor_of`）。
- `test_followup_after_my_answer_merges_into_the_card_and_is_written_in_full`、`test_new_topic_after_my_answer_keeps_its_own_card`、`test_first_question_has_no_card_to_compare_with`、`test_interviewer_continuing_after_pause_appends_instead_of_rewriting`、`test_reask_within_a_turn_keeps_the_turn`、`test_interviewer_filler_on_a_provisional_card_does_not_swallow_the_real_follow_up`

## 5. 慢答第一步：一次 JSON 判定

输入：面试场景描述（`kb/scene.md`，自然语言写候选人做过什么、这场面什么；不用词表）、最近 14 句对话、面试官这段话；有卡可比时（同一轮上一版，或临时卡对应的原卡），再给「当前卡片的主问题」和已并进来的各版。

字段：
- `relation`（只在有卡可比时要；放在**第一个字段**，放在最后模型常漏写）：
  - **补充**：不要新内容，还在帮我答当前这题。包括给提示、举例、补条件、给部分答案；回答或确认我的追问和复述；面试官自己插问背景前提，问完还会回到主问题；换个说法再问；教我按什么顺序答。
  - **追问**：话题没变，但要一块新内容，比如更细的机制、原因、对比、数据、怎么验证、还有别的办法吗。
  - **新问题**：换了话题。嘴上说「换个问题」不一定算，看实际问的是什么。
- `is_question`：附和、确认、过渡、收尾、问我有没有问题、回答我的反问，都是 false。
- `kind`（八股 / 项目 / 场景 / 其他）、`project_ref`、`question`（改写成的独立问题）、`queries`。

提示文本是 `copilot/assistant.py` 里的 `PLAN_SYSTEM`、`RELATION_FIELD`、`RELATION_RULE`，离线评测 `bench/relation/run.py --prompt v3` 直接引用它们，不另写一份。

### 回答时复核
我锁定一张卡后开始回答，说满 40 个字（只算实话）时，拿「面试场景 + 最近对话 + 面试官原话 + 之前的理解 + 我目前的回答」再判定一次，JSON 第一个字段 `changed` 判断之前是不是把问的对象或内容弄错了。弄错了：在原卡上加一版（`relation=更正`），按更正后的问题写；没弄错：不动。每张卡最多复核一次。
为什么：转写听错词、代词指代不清时，面试官说完那一刻理解不对，我的回答是最好的证据（10-11：Superpowers 被转成 Serverless，只看原话 3/3 错，加场景描述和我的前几句回答后 3/3 对）。
`test_my_answer_corrects_a_misheard_question`、`test_understanding_that_holds_is_left_alone_and_checked_once`

## 6. 慢答的写法（`slow_plan` 事件的 `mode`）

- **full**：完整口述稿，按题型模板（`SLOW_STYLE`），最多 12 行。
- **append（补充）**：只写原卡里没被直接回答的子问题、新条件或更正，第一行「补充：」，最多 6 行；都答过了就只输出「无」（`APPEND_RULE`）。
  `test_append_after_a_skipped_first_version_answers_in_full`
- **followup（追问）**：看着原卡写，不重复原卡写过的点，但按完整答案的规格写：第一行「追问：」，正常分小节、给数字，最多 12 行（`FOLLOWUP_RULE`）。

- **correction（更正）**：回答时复核判出之前理解错了。看着原卡写，第一行「更正：」说明实际问的是什么，再完整回答实际的问题，最多 12 行，接着我已经说的往下讲（`CORRECTION_RULE`）。

原卡还没出过字（全被判为不是问题）时，并进来的版本按 full 写。

## 7. 快答

1. 本地 BM25 取 3 条，一次流式调用出 3 行提纲；门控（是/否概率，阈值 0.25）和快答并行，门控说不用答就不上屏。
   `test_gate_saying_no_hides_the_fast_answer`、`test_tokens_wait_for_a_slow_gate_then_come_out_together`、`test_gate_failure_still_shows_the_fast_answer`
2. 同一张卡上已经有开口句上过屏，后面的版本不再出快答。临时卡在判定之前还没有 base，照常出快答。
   `test_append_skips_the_fast_answer_already_shown`、`test_append_still_gets_a_fast_answer_when_the_first_was_gated_out`
3. 要不要答以慢答为准：慢答判为不是问题，就把快答卡撤掉。`test_slow_stops_when_plan_says_not_a_question`

## 8. 取消和锁定

1. 开新卡（没有 base 的版本）时，取消还没锁定的旧任务；锁定的任务写完为止。
2. 接在后面的版本（补充、追问）不取消前一版。
3. 下一问的快慢答都会带上锚点：上一问，以及我照着说的那版骨架（去掉过渡句），要求接着我说过的往下讲（`ANCHOR_RULE`）。被判为不是问题的版本不能当锚点。
   `test_skipped_question_is_not_used_as_anchor`
4. 一路失败不影响另一路。`test_one_path_failing_leaves_the_other_running`

## 9. 界面（`copilot/web/index.html`）

- **实时**：只留当前卡。临时卡上屏时原卡变淡留在下面；判为补充或追问，撤掉临时卡、原卡回到最上面；判为新问题，撤掉原卡；判为不是问题，撤掉临时卡、原卡恢复。
- **回放 / 复看**：所有卡都留着，旧卡变小，判为不用答的变淡，并卡的临时卡也变淡并注明「并入」，方便核对。
- 骨架行：`过渡：` 是要直接说出口的整句；`补充：`、`追问：` 开头的段带标签显示。
- **回放**（`--replay-*` 两路录音，可加 `--segments`）：从录音中间开始放时，当作面试官已经说过话（不进自练模式）。
- **复看**（`--review` 日志 + `--replay-audio` 录音）：左栏一开始就列出整场转写，点哪句跳到哪句；右栏按录音时间重放当时的卡片；往回拖时清空重画。

## 10. 事件

`Event(qid, kind, elapsed, text, data, turn)`，`turn` 是卡片编号。

| kind | data | 说明 |
|---|---|---|
| question | `append`、`provisional` | 新一版；`provisional` 表示这是临时卡 |
| relation | `merge_into`（并卡时才有） | `text` 是补充 / 追问 / 新问题。并卡时这条事件本身仍发在临时卡上，之后的事件发到原卡 |
| slow_plan | `type`、`queries`、`hits`、`mode` | |
| kb_hit、gate、fast_token、fast_done、fast_skip、slow_skip、slow_token、slow_done、locked、cancelled、error | | |

## 11. 手撕代码模式（`copilot/coding.py`，设计见 LEETCODE.md）

1. 输入：Chrome 扩展（`extension/`）每两秒读一次编程页面（编辑器全文和语言、题面、运行输出），内容变了才经 `/page` 推来；30 秒一次心跳。
2. 进入手撕模式（问答卡暂停、对话照常记、右栏换成手撕卡）有两条路：慢答判定把面试官的话分成「手撕」（要写代码、做题，或只是宣布接下来要做题；这一条先于「是不是问题」判断），或收到有编辑器的页面。
   退出：手撕卡的【阶段】写「结束」（面试官结束这道题、问我有没有问题、转去聊别的），点「结束手撕」，或 120 秒没收到页面（从进入时起算）。
   `test_interviewer_asking_to_code_enters_before_any_page`、`test_assistant_hands_a_coding_request_to_the_coach`、`test_first_editor_page_enters_coding_mode_and_answers`、`test_stage_finished_leaves_coding_mode`、`test_stop_leaves_coding_mode`、`test_assistant_question_cards_pause_while_coding`
3. 什么时候更新：换题或运行输出变了，立刻更新；代码停改 3 秒；任一方说完一句 6 个字以上的话。生成途中又来新输入，就等这一次写完再来一次，不叠加。
   `test_typing_is_debounced_into_one_update`、`test_run_output_and_speech_trigger_at_once_and_rerun_after_a_busy_generation`、`test_short_backchannel_does_not_trigger`
4. 每次更新输出七块，按这个顺序：阶段（读题 / 讲思路 / 写代码 / 调试 / 讲解 / 结束，显示在卡片顶上）→ 题目（完整就不显示；不完整就提示缺什么、去页面哪里翻，不单开一条读题的线）→ 思路判断（不对要给反例和正确思路）→ 现在说（一句，40 字以内，只写能对面试官说出口的话）→ 我的代码（先解释报错，再指出错的行和改法）→ 复杂度与边界 → 完整代码（按讨论定下的语言和约束写，ACM 模式写全样板；没变就写「同上」，界面保留上一版）。
   `test_same_code_is_not_rewritten`、`test_sections_are_split_by_headers`
5. Python 代码在本地先编译一遍，语法检查结果（第几行、什么错，或「没有语法错误」）作为确定事实放进提示，模型以它为准。其他语言暂不检查。
   `test_python_syntax_is_checked_locally_and_given_to_the_model`、`test_syntax_error_reaches_the_prompt`

## 12. 已知限制

- 判定大约在触发后 1.5 到 2 秒回来，这段时间临时卡和原卡同时在屏幕上。
- 一张卡会越并越长（比如 RAG 连着追问好几层），旧内容还没做折叠。
- 「是从零到一吧？」「是的是的」这种是非确认后，紧接着的新问题会接在同一轮里（因为「是的」算附和）。
- 手撕卡的模型经常不听「同上」，代码整份重写（排在最后，不挡前面几块）；同一份代码，「我的代码」有时只抓到语法错、漏掉逻辑错。
- 更正出来时我已经开口，第一眼看到的仍是理解错的那版；它主要帮后半段回答和后面的追问。场景描述里的内容是 10-11 看过 Superpowers 这一例后写的，这一例验对不算数，要在别的误听上再验。
- 关系判定只在两场面试上评测过（`bench/relation/`），标签是 Claude 标的，用户只复核了有分歧的几处。
