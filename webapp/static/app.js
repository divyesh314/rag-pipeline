// Frontend for the RAG web app. Talks to the FastAPI server in webapp/server.py.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const STOP = new Set("a an the and or but of to in on at for from by with as is are was were be been being it its this that these those what which who whom how why when where do does did can could should would will just than then so such into over about not no only also very more most other some any each both i you he she we they me my your our their there here up out if".split(" "));
  const stem = (w) => (w.length > 4 ? w.replace(/(ing|ed|ies|es|s)$/, "") : w);
  const keyTerms = (q) => new Set((q.toLowerCase().match(/[a-z0-9@]+/g) || []).filter((w) => w.length > 1 && !STOP.has(w)).map(stem));

  let busy = false;

  async function api(path, options) {
    const res = await fetch(path, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return body;
  }

  // ---------- topics ----------
  async function loadTopics() {
    try {
      const topics = await api("/api/topics");
      $("shelf-count").textContent = `${topics.length} documents${topics.some((t) => t.sample_question) ? " · tap one to try its question" : ""}`;
      $("shelf").innerHTML = "";
      for (const t of topics) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "card";
        b.dataset.id = t.id;
        b.title = t.id;
        b.innerHTML =
          `<span class="tag"><span>${esc(t.topic || "general")}</span><b>${t.chunks} passage${t.chunks === 1 ? "" : "s"}</b></span>` +
          `<span class="name">${esc(t.title)}</span>` +
          `<span class="ask">${t.sample_question ? "“" + esc(t.sample_question) + "”" : esc(t.preview)}</span>`;
        b.addEventListener("click", () => {
          $("question").value = t.sample_question || `Tell me about ${t.title}.`;
          $("question").focus();
        });
        $("shelf").append(b);
      }
    } catch (e) {
      $("shelf-count").textContent = "";
      $("shelf").innerHTML = `<p class="error">Couldn't load the topics: ${esc(e.message)}. Check that the server is running.</p>`;
    }
  }

  // ---------- status ----------
  async function loadStatus() {
    try {
      const s = await api("/api/status");
      $("mode").textContent = `Search: ${s.embedder} · Rerank: ${s.reranker} · Answers: ${s.llm}`;
      $("footer").textContent = `${s.documents} documents split into ${s.chunks} passages. Keyword search (BM25) and similarity search are merged with Reciprocal Rank Fusion, then reranked.`;
      if (!s.real_answers) {
        $("mode-banner").innerHTML =
          "<span>Demo mode: answers show the best-matching passage instead of a written reply. Set <code>ANTHROPIC_API_KEY</code> on the server and install <code>anthropic</code> to get real answers from Claude.</span>";
        $("mode-banner").hidden = false;
      }
    } catch {
      $("mode").textContent = "";
    }
  }

  // ---------- ask ----------
  function setSteps(state) {
    document.querySelectorAll("#steps li").forEach((li) => (li.className = state));
  }

  function highlight(text, terms) {
    return esc(text).replace(/[A-Za-z0-9@]+/g, (w) => (terms.has(stem(w.toLowerCase())) && !STOP.has(w.toLowerCase()) ? `<mark>${w}</mark>` : w));
  }

  function renderAnswer(text) {
    const el = $("answer");
    el.className = "answer-body";
    el.innerHTML = esc(text).replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
  }

  function renderSources(sources, terms) {
    const used = new Set(sources.map((s) => s.doc_id));
    document.querySelectorAll(".card").forEach((c) => c.classList.toggle("hit", used.has(c.dataset.id)));
    $("sources-wrap").hidden = sources.length === 0;
    $("sources").innerHTML = sources
      .map(
        (s) =>
          `<div class="src"><span class="n">[${s.n}]</span>` +
          `<div class="meta"><b>${esc(s.title)}</b><span>${esc(s.chunk_id)}</span>` +
          `<span>fusion ${s.fusion_score.toFixed(4)}</span>` +
          `<span>keyword #${s.keyword_rank ?? "–"}</span>` +
          `<span>semantic #${s.semantic_rank ?? "–"}</span></div>` +
          `<p>${highlight(s.text, terms)}</p></div>`
      )
      .join("");
  }

  async function ask(question) {
    question = question.trim();
    if (!question) {
      $("question").focus();
      return;
    }
    if (busy) return;
    busy = true;
    $("submit").disabled = true;
    $("submit").textContent = "Working…";
    $("asked").textContent = question;
    $("answer").className = "answer-body wait";
    $("answer").textContent = "Searching the documents and writing an answer…";
    setSteps("on");

    try {
      const res = await api("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 3 }),
      });
      renderAnswer(res.answer);
      renderSources(res.sources, keyTerms(question));
      $("ans-h").textContent = `Answer · ${res.elapsed_ms} ms`;
      setSteps("done");
    } catch (e) {
      $("answer").className = "answer-body error";
      $("answer").textContent = e.message;
      $("sources-wrap").hidden = true;
      setSteps("");
    } finally {
      busy = false;
      $("submit").disabled = false;
      $("submit").textContent = "Get answer";
    }
  }

  $("desk").addEventListener("submit", (e) => {
    e.preventDefault();
    ask($("question").value);
  });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask($("question").value);
    }
  });

  loadStatus();
  loadTopics();
})();
