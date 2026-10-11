// 从后台连本地服务：页面自己去连 localhost 常被网站的 CSP（connect-src）拦。
const URL_ = "ws://localhost:8766/page";
let ws = null, queue = [];

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  ws = new WebSocket(URL_);
  ws.onopen = () => { for (const m of queue.splice(0)) ws.send(m); };
  ws.onclose = () => { ws = null; };
  ws.onerror = () => {};
}

chrome.runtime.onMessage.addListener((page) => {
  const msg = JSON.stringify(page);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(msg);
  else { queue = [msg]; connect(); } // 只留最新一份，连上再发
});

// 扩展安装、刷新、更新后，已经开着的页面里的旧脚本会失效（转发那份拿不到 chrome.runtime），
// 不刷新页面就收不到内容。这里给每个已开的页面补注入一遍；reader.js 自带防重复。
chrome.runtime.onInstalled.addListener(async () => {
  for (const tab of await chrome.tabs.query({})) {
    if (!tab.id || !/^https?:/.test(tab.url || "")) continue;
    const target = { tabId: tab.id, allFrames: true };
    chrome.scripting.executeScript({ target, files: ["relay.js"] }).catch(() => {});
    chrome.scripting.executeScript({ target, files: ["reader.js"], world: "MAIN" }).catch(() => {});
  }
});
