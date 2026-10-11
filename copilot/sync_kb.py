"""用 lark-cli 把飞书知识库文档拉成本地 markdown：python -m copilot.sync_kb"""

import json
import subprocess
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "kb"
DOCS = {
    "interview.md": "https://m0t4f8ok2kh.feishu.cn/wiki/BqKrwxevtiycv0k205Lc91egnvA",  # 八股，按编号条目切
    "projects.md": "https://m0t4f8ok2kh.feishu.cn/wiki/K9qawc84CiywxSkHr1uc9VSXnq9",  # 项目口径，按 ## 问题切
}


def fetch(name: str, url: str) -> None:
    raw = subprocess.run(
        ["lark-cli", "docs", "+fetch", "--as", "user", "--doc", url, "--doc-format", "markdown"],
        capture_output=True, text=True, check=True,
    ).stdout
    reply = json.loads(raw)
    if not reply.get("ok"):
        sys.exit(f"拉取失败：{raw[:300]}")
    content = reply["data"]["document"]["content"]
    out = KB_DIR / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"{len(content)} 字 → {out}")


def main() -> None:
    for name, url in DOCS.items():
        fetch(name, url)


if __name__ == "__main__":
    main()
