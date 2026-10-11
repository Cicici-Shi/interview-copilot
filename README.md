# interview-copilot

面试实时提词：听到面试官的问题后，先秒出一个提纲（快答），几秒后跟上一段能直接念的口述稿（慢答）。
思路来自黑客松项目 personal_agent_long_context 的会议快答 / 深答（`src/meetings/quick-answer.ts`），换成面试场景和自己的知识库。

## 知识库

`python -m copilot.sync_kb` 用 lark-cli 把两篇飞书文档拉到 `kb/`（不进 git）：

| 文件 | 来源 | 切法 |
|---|---|---|
| `kb/interview.md` | 面试八股总文档 | 按顶格编号条目切，一条一张卡 |
| `kb/projects.md` | 项目口径 | 按 `##` 问题切，答案里的编号列表不拆 |

检索：BM25，标题和正文分字段打分（长答案的标题命中不被长度稀释），中文按字二元组。
慢答另加 bge-m3 向量，两路名次用 RRF 合并；向量按内容 hash 缓存在 `kb/.cache/`。

## 问题识别和快慢答

说话人按音频通道区分：系统音频是面试官，麦克风是我。听到面试官的问题后，快答秒出 3 行提纲，慢答几秒后跟上关键词骨架。
什么时候算一轮、放到哪张卡、补充和追问怎么写、界面怎么变，都以 [SPEC.md](SPEC.md) 为准。

模型走 NUS SoC LaaS（`~/Downloads/soclaas.env`）：快答、慢答、门控都用 `qwen3.8:27b`，都关思考模式；429 限流时出字前退避重试。

## 实时使用（网页）

```bash
set -a; . ~/Downloads/soclaas.env; set +a
export $(grep ^VOLC_ ~/Desktop/git/personal_agent_long_context/.env | xargs)   # 豆包 ASR key
python3 -m copilot.live          # 打开 http://localhost:8766
```

在 Chrome 打开页面点「开始」，共享面板里选会议所在的标签页（桌面版会议软件就选「整个屏幕」），**勾选共享音频**，再允许麦克风。
共享的声音是面试官那一路，麦克风是我那一路，各连一条豆包；本地能量 VAD（`copilot/vad.py`）给 `on_speech_start`。
要戴耳机：外放时面试官的声音会漏进麦克风，被当成我开口，提前结束这一轮。
左边是转写，右边是问题卡片（快答提纲 + 口述稿），每场的全部输出存 `reports/live-<时间>.jsonl`。

自己练习（没有会议）：点「开始」后在共享面板点取消，只开麦克风。面试官那一路没出过声之前，我对麦克风说的话都当面试官的问题；面试官一开口就回到正常逻辑。

手撕代码：在 `chrome://extensions` 打开开发者模式，「加载已解压的扩展程序」选 `extension/` 目录。之后在线编辑器里的代码、题面、运行输出会推给提词器（要先点「开始」），右栏自动换成手撕卡。只连 8766 端口。

不开会也能看效果：拿两路录音按真实时间喂，网页只看。

```bash
python3 -m copilot.live --replay-interviewer reports/.tmp/985-interviewer.wav --replay-me reports/.tmp/985-candidate.wav --duration 180
```

加 `--replay-audio 原始录音.m4a`，网页上点「播放声音」跟着回放进度放原音。

复看一场存下来的输出（不跑模型，带进度条，拖到哪儿就显示那一刻的转写和卡片）：

```bash
python3 -m copilot.live --port 8767 --review reports/live-20261010-105744.jsonl --offset 158 --replay-audio 录音.m4a
```

`--offset` 是日志时间减录音时间（提词器比录音早开多少秒）；回放生成的日志不用给，从里面读。

## 回放测试

```bash
set -a; . ~/Downloads/soclaas.env; set +a
python -m copilot.replay 逐字稿.md --concurrency 1            # 每轮单独跑，报告写到 reports/
python -m copilot.replay 逐字稿.md --no-punct                 # 去掉标点，模拟实时 ASR
python -m copilot.replay 逐字稿.md --live --from 00:00 --to 03:00   # 按真实时间流式喂，看触发和取消
python -m pytest -q
```

逐字稿格式：`**[mm:ss] 面试官**：……` / `**[mm:ss] 候选人**：……`。
