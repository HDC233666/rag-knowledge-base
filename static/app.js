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
async function refresh() {
  const [stats, docs] = await Promise.all([api("/api/stats"), api("/api/documents")]);
  $("stats").textContent = `${stats.documents} 篇 / ${stats.chunks} 块`;
  $("meta").textContent = `模型 ${stats.model} · ${stats.has_key ? "Key 已配置" : "Key 未配置"}`;

  const list = $("docs");
  list.innerHTML = "";
  if (!docs.length) {
    list.innerHTML = '<li class="empty">还没有文档，点上面按钮上传</li>';
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

/* ---------- 上传 ---------- */
$("file").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const result = await api("/api/upload", { method: "POST", body: formData });
    toast(`已收录《${result.name}》（${result.chunks} 块）`, false);
    refresh();
  } catch (error) {
    toast(error.message);
  } finally {
    event.target.value = "";
  }
});

/* ---------- 提问 ---------- */
function appendUser(text) {
  const node = document.createElement("div");
  node.className = "msg user";
  node.innerHTML = `<div class="bubble">${toParagraphs(text)}</div>`;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function appendBot(payload, loading = false) {
  const node = document.createElement("div");
  node.className = "msg bot";
  const citations = (payload && payload.citations) || [];
  const badges = payload
    ? `<span class="badge">${payload.latency_ms} ms</span>${
        payload.tokens ? `<span class="badge">${payload.tokens} tokens</span>` : ""
      }`
    : "";
  node.innerHTML = `
    <div class="bubble">
      ${loading ? '<span class="typing">知识库检索中…</span>' : renderMarkdown(payload.answer)}
      <div class="citations">
        ${citations
          .map(
            (c) => `
          <div class="citation">
            <span class="cite-id">[${c.index + 1}]</span>
            <span class="cite-doc">${escapeHtml(c.doc_name)}</span>
            <span class="cite-score">score ${c.score}</span>
            <div class="cite-snippet">${escapeHtml(c.snippet)}…</div>
          </div>`
          )
          .join("")}
      </div>
      <div class="badges">${badges}</div>
    </div>`;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const textarea = $("question");
  const question = textarea.value.trim();
  if (!question) return;

  textarea.value = "";
  appendUser(question);
  const pending = appendBot(null, true);

  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    pending.remove();
    appendBot(result);
  } catch (error) {
    pending.remove();
    appendBot({ answer: `❌ ${error.message}`, citations: [] });
  }
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("form").requestSubmit();
  }
});

refresh().catch((error) => toast(error.message));
