// 在页面自己的环境（MAIN world）里读编辑器：编辑器只渲染看得见的几行，读 DOM 会漏，要调编辑器对象拿全文。
// 两秒看一次，内容变了才发；30 秒发一次心跳，本地服务据此判断还在手撕。读到的东西经 postMessage 交给 relay.js。
(() => {
  if (window.__copilotReader) return;
  window.__copilotReader = true;

  // 认不出编辑器时，只在这些在线编程网站发整页文字
  const PAD_SITES = /leetcode|nowcoder|coderpad|codeinterview|hackerrank|codesignal|codility|shuati|acwing|luogu/i;
  const visible = (el) => !!el && el.offsetParent !== null && el.getClientRects().length > 0;
  const text = (sel, cap) => [...document.querySelectorAll(sel)].filter(visible).map((e) => e.innerText.trim()).filter(Boolean).join("\n").slice(0, cap);

  // 没暴露 window.monaco 的网站（CoderPad 等 webpack 打包）：往 webpackChunk* 里塞一个假分包拿到内部加载器，
  // 在模块源码里找 Monaco 的 API 模块（同时有 getModels 和 createModel），取已加载的导出。找到一次就缓存；找不到 10 秒内不再扫。
  let cachedMonaco = null, lastScan = 0;
  function monacoApi() {
    if (window.monaco?.editor) return window.monaco;
    if (cachedMonaco) return cachedMonaco;
    if (Date.now() - lastScan < 10000) return null;
    lastScan = Date.now();
    for (const key of Object.keys(window).filter((k) => k.startsWith("webpackChunk"))) {
      let req;
      try { window[key].push([[Symbol("copilot")], {}, (r) => { req = r; }]); } catch { continue; }
      if (!req?.m) continue;
      for (const [id, fn] of Object.entries(req.m)) {
        const src = Function.prototype.toString.call(fn);
        if (!src.includes("getModels") || !src.includes("createModel")) continue;
        try {
          const ex = req(id);
          for (const v of [ex, ex?.default, ...Object.values(ex || {})]) {
            if (v?.editor && typeof v.editor.getModels === "function") return (cachedMonaco = v);
          }
        } catch {}
      }
    }
    return null;
  }

  // 最后一招：Monaco 只渲染看得见的行，读这些行并注明范围（代码比编辑器高时会缺）
  function monacoDom() {
    const box = [...document.querySelectorAll(".monaco-editor")].filter(visible).sort((a, b) => b.clientHeight - a.clientHeight)[0];
    if (!box) return null;
    const lines = [...box.querySelectorAll(".view-line")].map((l) => ({ top: parseFloat(l.style.top) || 0, text: l.innerText.replace(/\u00a0/g, " ") }))
      .sort((a, b) => a.top - b.top);
    const nums = [...box.querySelectorAll(".line-numbers")].map((n) => parseInt(n.innerText, 10)).filter((n) => n > 0).sort((a, b) => a - b);
    if (!lines.length) return null;
    const range = nums.length ? `（只读到看得见的第 ${nums[0]} 到 ${nums[nums.length - 1]} 行）` : "（只读到看得见的行）";
    return { editor: "monaco-dom", code: lines.map((l) => l.text).join("\n"), lang: range };
  }

  function editor() {
    const m = monacoApi();
    if (m?.editor) {
      const eds = m.editor.getEditors?.() || [];
      const shown = eds.filter((e) => visible(e.getDomNode?.()));
      const pick = (shown.length ? shown : eds).map((e) => e.getModel()).filter(Boolean)
        .sort((a, b) => b.getValue().length - a.getValue().length)[0]
        || m.editor.getModels().filter((x) => x.getLanguageId() !== "plaintext").sort((a, b) => b.getValue().length - a.getValue().length)[0];
      if (pick) return { editor: "monaco", code: pick.getValue(), lang: pick.getLanguageId() };
    }
    for (const el of document.querySelectorAll(".cm-content")) {
      const view = el.cmView?.view || el.cmView?.rootView?.view;
      if (view && visible(el)) return { editor: "codemirror6", code: view.state.doc.toString(), lang: el.dataset.language || "" };
    }
    for (const el of document.querySelectorAll(".CodeMirror")) {
      if (el.CodeMirror && visible(el)) {
        const mode = el.CodeMirror.getOption("mode");
        return { editor: "codemirror5", code: el.CodeMirror.getValue(), lang: typeof mode === "string" ? mode : mode?.name || "" };
      }
    }
    if (window.ace) {
      for (const el of document.querySelectorAll(".ace_editor")) {
        if (!visible(el)) continue;
        const ed = window.ace.edit(el);
        return { editor: "ace", code: ed.getValue(), lang: (ed.session.getMode().$id || "").replace("ace/mode/", "") };
      }
    }
    const md = monacoDom();
    if (md) return md;
    const ta = [...document.querySelectorAll("textarea")].filter(visible).sort((a, b) => b.value.length - a.value.length)[0];
    if (ta && ta.value.length > 20) return { editor: "textarea", code: ta.value, lang: "" };
    if (PAD_SITES.test(location.hostname)) return { editor: "text", code: document.body.innerText.slice(0, 8000), lang: "" };
    return null;
  }

  // 题面：力扣有固定位置；别的网站在「类名像题目」的元素里挑一个：只要最外层（嵌套的会重复），
  // 不要包着编辑器的大容器，优先有「描述/输入/输出/示例」字样、字最多的那个
  const PROBLEM_SEL = '[class*="question"],[class*="subject"],[class*="problem"],[class*="description"],[class*="describe"],[class*="prompt"],[class*="instruction"]';
  const PROBLEM_WORDS = /描述|给定|输入|输出|示例|样例|Example|Input|Output|Constraints/i;
  function problem() {
    const lc = text('[data-track-load="description_content"]', 4000);
    if (lc) return lc;
    const cands = [...document.querySelectorAll(PROBLEM_SEL)].filter(visible)
      .filter((e) => !e.querySelector(".monaco-editor,.cm-editor,.CodeMirror,.ace_editor,textarea"));
    const outer = cands.filter((e) => !cands.some((o) => o !== e && o.contains(e)));
    const best = outer.map((e) => ({ e, t: e.innerText.trim() }))
      .sort((a, b) => (PROBLEM_WORDS.test(b.t) - PROBLEM_WORDS.test(a.t)) || (b.t.length - a.t.length))[0];
    return best ? best.t.slice(0, 4000) : "";
  }

  function output() {
    return text('[data-e2e-locator="console-result"],[class*="console"],[class*="result"],[class*="output"],[class*="judge"],.xterm-rows', 2000);
  }

  let last = "", lastSent = 0;
  // 平时只读看得见的页面（后台开着的其他题不算）；切走的那一刻再补读一次，免得最后几下改动没发出去
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") tick(true); });
  setInterval(() => tick(false), 2000);
  function tick(leaving) {
    if (!leaving && document.visibilityState !== "visible") return;
    let ed;
    try { ed = editor(); } catch (e) { return; }
    if (!ed) return;
    const page = { type: "page", url: location.href, title: document.title, problem: problem(), output: output(), ...ed };
    const key = JSON.stringify(page);
    if (key === last && Date.now() - lastSent < 30000) return;
    last = key; lastSent = Date.now();
    window.postMessage({ __copilotPage: page }, "*");
  }
})();
