// 页面环境拿不到 chrome.runtime，这里转一手给后台。
// 扩展刷新或更新后，旧页面里的这份脚本就失效了（Extension context invalidated），sendMessage 会同步抛错，
// .catch 接不住；失效后就不再转发，刷新页面后换上新脚本。
function relay(e) {
  if (e.source !== window || !e.data || !e.data.__copilotPage) return;
  try {
    if (!chrome.runtime?.id) throw new Error("context invalidated");
    chrome.runtime.sendMessage(e.data.__copilotPage).catch(() => {});
  } catch {
    window.removeEventListener("message", relay);
  }
}
window.addEventListener("message", relay);
