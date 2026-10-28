/* 极简前端：没有构建工具，改完刷新浏览器即可 */
const $ = (id) => document.getElementById(id);
const messages = $("messages");

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function toParagraphs(text) {
  return escapeHtml(text)
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

/* ---- markdown renderer begin（无依赖，先转义再解析，防 XSS） ---- */
function renderInline(text) {
  return text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function renderTextBlock(raw) {
  const lines = escapeHtml(raw).split("\n");
  const out = [];
  let para = [];
  let list = null;

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${para.join("<br>")}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      out.push(`<${list.type}>${list.items.map((item) => `<li>${item}</li>`).join("")}</${list.type}>`);
      list = null;
    }
  };
  const flushAll = () => {
    flushPara();
    flushList();
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushAll();
      continue;
    }
    let match;

    if ((match = /^(#{1,6})\s+(.+)$/.exec(line))) {
      flushAll();
      const level = Math.min(match[1].length + 2, 6);
      out.push(`<h${level}>${renderInline(match[2])}</h${level}>`);
      continue;
    }
    if (/^([-*_])\1{2,}$/.test(line)) {
      flushAll();
      out.push("<hr>");
      continue;
    }
    if ((match = /^&gt;\s?(.*)$/.exec(line))) {
      flushAll();
      out.push(`<blockquote>${renderInline(match[1])}</blockquote>`);
      continue;
    }
    if ((match = /^\d+[.)、]\s+(.+)$/.exec(line))) {
      flushPara();
      if (!list || list.type !== "ol") {
        flushList();
        list = { type: "ol", items: [] };
      }
      list.items.push(renderInline(match[1]));
      continue;
    }
    if ((match = /^[-*+]\s+(.+)$/.exec(line))) {
      flushPara();
      if (!list || list.type !== "ul") {
        flushList();
        list = { type: "ul", items: [] };
      }
      list.items.push(renderInline(match[1]));
      continue;
    }
    flushList();
    para.push(renderInline(line));
  }
  flushAll();
  return out.join("\n");
}

function renderMarkdown(text) {
  const source = String(text ?? "");
  const parts = [];
  const codePattern = /```([^\n`]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;
  while ((match = codePattern.exec(source))) {
    parts.push({ type: "text", value: source.slice(lastIndex, match.index) });
    parts.push({ type: "code", value: match[2].replace(/\n$/, "") });
    lastIndex = match.index + match[0].length;
  }
  parts.push({ type: "text", value: source.slice(lastIndex) });
  return parts
    .map((part) =>
      part.type === "code"
        ? `<pre><code>${escapeHtml(part.value)}</code></pre>`
        : renderTextBlock(part.value)
    )
    .join("\n");
}
/* ---- markdown renderer end ---- */

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `请求失败（${response.status}）`);
  }
  return data;
}

function toast(message, isError = true) {
  const node = document.createElement("div");
  node.className = `toast${isError ? " error" : ""}`;
  node.textContent = message;
  document.body.appendChild(node);
  setTimeout(() => node.remove(), 4000);
}

/* ---------- 顶部状态 + 文档列表 ---------- */
const COLLECTION_KEY = "rag-collection";

function currentCollection() {
  const select = $("collection");
  return (select && select.value) || "default";
}

function saveCollection(name) {
  try {
    localStorage.setItem(COLLECTION_KEY, name);
  } catch (_) {
    /* 隐私模式下 localStorage 可能不可用，忽略即可 */
  }
}

function restoreCollection() {
  try {
    return localStorage.getItem(COLLECTION_KEY) || "default";
  } catch (_) {
    return "default";
  }
}

async function refreshCollections() {
  const collections = await api("/api/collections");
  const select = $("collection");
  const wanted = select.value || restoreCollection();
  select.innerHTML = "";
  for (const item of collections) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `${item.name}（${item.documents}）`;
    select.appendChild(option);
  }
  const hasWanted = collections.some((item) => item.name === wanted);
  select.value = hasWanted ? wanted : collections[0] ? collections[0].name : "default";
  saveCollection(select.value);
  return select.value;
}

async function refresh() {
  const collection = await refreshCollections();
  const query = `?collection=${encodeURIComponent(collection)}`;
  const [stats, docs] = await Promise.all([api(`/api/stats${query}`), api(`/api/documents${query}`)]);
  $("stats").textContent = `${stats.documents} 篇 / ${stats.chunks} 块`;
  $("meta").textContent = `模型 ${stats.model} · ${stats.has_key ? "Key 已配置" : "Key 未配置"}`;
  $("collection-hint").textContent = collection === "default" ? "默认知识库" : "当前知识库";

  const list = $("docs");
  list.innerHTML = "";
  if (!docs.length) {
    list.innerHTML = '<li class="empty">这个知识库还没有文档，点上面按钮上传</li>';
    return;
  }
  for (const doc of docs) {
    const item = document.createElement("li");
    item.innerHTML = `
      <div class="doc-main">
        <span class="doc-name">${escapeHtml(doc.name)}</span>
        <span class="doc-meta">${doc.chunks} 块 · ${(doc.chars / 1000).toFixed(1)}k 字</span>
      </div>
      <button class="doc-del" title="删除">✕</button>`;
    item.querySelector(".doc-del").addEventListener("click", async () => {
      if (!confirm(`删除《${doc.name}》？`)) return;
      try {
        await api(`/api/documents/${doc.id}`, { method: "DELETE" });
        toast(`已删除《${doc.name}》`, false);
        refresh();
      } catch (error) {
        toast(error.message);
      }
    });
    list.appendChild(item);
  }
}

/* ---------- 知识库切换 / 新建 ---------- */
$("collection").addEventListener("change", () => {
  saveCollection(currentCollection());
  refresh().catch((error) => toast(error.message));
});

$("new-collection").addEventListener("click", () => {
  const name = (prompt("新知识库名称（中文、字母、数字、下划线或短横线）") || "").trim();
  if (!name) return;
  if (!/^[\w\u4e00-\u9fff-]{1,64}$/.test(name)) {
    toast("名称不合法：只能用中文、字母、数字、下划线、短横线");
    return;
  }
  const select = $("collection");
  if (!Array.from(select.options).some((option) => option.value === name)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name}（0）`;
    select.appendChild(option);
  }
  select.value = name;
  saveCollection(name);
  toast(`已切换到「${name}」，上传文档后会出现在这里`, false);
  refresh().catch((error) => toast(error.message));
});

/* ---------- 上传 ---------- */
$("file").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("collection", currentCollection());
  try {
    const result = await api("/api/upload", { method: "POST", body: formData });
    toast(`已收录《${result.name}》到「${result.collection}」（${result.chunks} 块）`, false);
    refresh();
  } catch (error) {
    toast(error.message);
  } finally {
    event.target.value = "";
  }
});

/* ---------- 提问 ---------- */
let conversation = []; // 多轮对话历史（只发给后端最近 6 条）

function appendUser(text) {
  const node = document.createElement("div");
  node.className = "msg user";
  node.innerHTML = `<div class="bubble">${toParagraphs(text)}</div>`;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function paintBot(node, payload, loading = false) {
  const citations = (payload && payload.citations) || [];
  const badges = payload
    ? `<span class="badge">${payload.latency_ms} ms</span>${
        payload.tokens ? `<span class="badge">${payload.tokens} tokens</span>` : ""
      }`
    : "";
  node.querySelector(".bubble").innerHTML = loading
    ? '<span class="typing">知识库检索中…</span>'
    : renderMarkdown(payload.answer);
  node.querySelector(".citations").innerHTML = citations
    .map(
      (c) => `
      <div class="citation">
        <span class="cite-id">[${c.index + 1}]</span>
        <span class="cite-doc">${escapeHtml(c.doc_name)}</span>
        <span class="cite-score">score ${c.score}</span>
        <div class="cite-snippet">${escapeHtml(c.snippet)}…</div>
      </div>`
    )
    .join("");
  node.querySelector(".badges").innerHTML = badges;
  messages.scrollTop = messages.scrollHeight;
}

function appendBot(payload, loading = false) {
  const node = document.createElement("div");
  node.className = "msg bot";
  node.innerHTML = `
    <div class="bubble"></div>
    <div class="citations"></div>
    <div class="badges"></div>`;
  messages.appendChild(node);
  paintBot(node, payload, loading);
  return node;
}

/** 读取 SSE 流，每收到一个 data: {...} 事件就回调一次 */
async function askStream(question, onEvent) {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: conversation.slice(-6), collection: currentCollection() }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `请求失败（${response.status}）`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(6)));
    }
  }
}

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const textarea = $("question");
  const question = textarea.value.trim();
  if (!question) return;

  textarea.value = "";
  appendUser(question);
  const pending = appendBot(null, true);
  const bubble = pending.querySelector(".bubble");

  let answerText = "";
  let citations = [];
  let latency = null;
  let tokens = null;

  try {
    await askStream(question, (ev) => {
      if (ev.type === "trace") {
        bubble.innerHTML = '<span class="typing">已检索到资料，正在生成回答…</span>';
      } else if (ev.type === "delta") {
        answerText += ev.text;
        bubble.innerHTML = renderMarkdown(answerText);
        messages.scrollTop = messages.scrollHeight;
      } else if (ev.type === "done") {
        answerText = ev.answer || answerText;
        citations = ev.citations || [];
        latency = ev.latency_ms;
        tokens = ev.tokens;
      } else if (ev.type === "error") {
        throw new Error(ev.detail || "生成失败");
      }
    });
    paintBot(pending, { answer: answerText, citations, latency_ms: latency, tokens });
    conversation.push({ role: "user", content: question });
    conversation.push({ role: "assistant", content: answerText });
  } catch (error) {
    paintBot(pending, { answer: `❌ ${error.message}`, citations: [] });
  }
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("form").requestSubmit();
  }
});

refresh().catch((error) => toast(error.message));
