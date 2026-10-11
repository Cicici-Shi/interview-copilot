"""人工标注「新问题 / 补充 / 不用答」：网页逐处看前后转写、点按钮，结果写回 JSON。

  python -m bench.relation.label_server bench/relation/cases-1010.json --out bench/relation/labels-1010-user.json
  打开 http://localhost:8768
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make_handler(args, cases, extra):
    out = Path(args.out)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, headers=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, (HERE / "label.html").read_bytes(), "text/html; charset=utf-8", [("Cache-Control", "no-store")])
            elif self.path == "/data":
                labels = json.loads(out.read_text()) if out.exists() else {}
                self.send(200, json.dumps({"cases": cases, "labels": labels, "extra": extra}, ensure_ascii=False).encode(), "application/json")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/label":
                return self.send(404, b"not found", "text/plain")
            item = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            labels = json.loads(out.read_text()) if out.exists() else {}
            labels[item["id"]] = {"label": item["label"], "note": item.get("note", "")}
            out.write_text(json.dumps(labels, ensure_ascii=False, indent=1))
            self.send(200, b"ok", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--claude", help="Claude 的标签，页面上默认藏起来")
    ap.add_argument("--runs", nargs="*", default=[], help="模型评测结果，页面上默认藏起来")
    ap.add_argument("--only", help="只标这几处，逗号分隔的 id")
    ap.add_argument("--port", type=int, default=8768)
    args = ap.parse_args()
    cases = json.load(open(args.cases))
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    extra = {}
    if args.claude:
        for k, v in json.load(open(args.claude)).items():
            if not k.startswith("_"):
                extra.setdefault(k, {})["claude"] = v
    for path in args.runs:
        for r in json.load(open(path)):
            extra.setdefault(r["id"], {}).setdefault("model", []).append(r["got"])
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args, cases, extra)).serve_forever()


if __name__ == "__main__":
    main()
