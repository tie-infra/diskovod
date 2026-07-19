(() => {
  "use strict";

  const root = document.documentElement;
  const body = document.body;
  const theme = root.dataset.adminTheme || "system";
  const colorScheme = window.matchMedia("(prefers-color-scheme: dark)");

  const applySystemTheme = () => {
    if (theme === "system") root.dataset.bsTheme = colorScheme.matches ? "dark" : "light";
  };
  applySystemTheme();
  colorScheme.addEventListener("change", applySystemTheme);

  const sidebarButton = document.querySelector(".sidebar-toggle");
  sidebarButton?.addEventListener("click", () => {
    const open = body.classList.toggle("sidebar-open");
    sidebarButton.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && body.classList.contains("sidebar-open")) {
      body.classList.remove("sidebar-open");
      sidebarButton?.setAttribute("aria-expanded", "false");
      sidebarButton?.focus();
    }
    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const target = event.target;
      if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement)) {
        const search = document.querySelector('input[type="search"]');
        if (search) {
          event.preventDefault();
          search.focus();
        }
      }
    }
  });

  const locale = root.lang || "en";
  const dateFormatter = new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeStyle: "short" });
  const localizeTimes = (scope = document) => {
    scope.querySelectorAll("[data-local-time][data-timestamp]").forEach((element) => {
      const timestamp = Number(element.dataset.timestamp);
      if (Number.isFinite(timestamp)) {
        const date = new Date(timestamp * 1000);
        element.dateTime = date.toISOString();
        element.textContent = dateFormatter.format(date);
        element.title = date.toLocaleString(locale, { dateStyle: "full", timeStyle: "long" });
      }
    });
  };
  localizeTimes();

  const topic = body.dataset.liveTopic || "jobs";
  if (!("ReadableStream" in window) || !("TextDecoderStream" in window)) return;

  const terminalStates = new Set(["succeeded", "failed", "cancelled"]);
  const labels = {
    live: body.dataset.liveLabel || "Live",
    reconnecting: body.dataset.reconnectingLabel || "Reconnecting…",
    offline: body.dataset.offlineLabel || "Offline",
    deleted: body.dataset.deletedMessageLabel || "Message deleted",
    newMessages: body.dataset.newMessagesLabel || "New messages",
    edited: body.dataset.editedLabel || "Edited",
  };
  let stopped = false;
  let controller = null;

  const setLiveStatus = (state) => {
    const indicator = document.querySelector("[data-live-status]");
    if (!indicator) return;
    indicator.textContent = labels[state];
    indicator.classList.toggle("text-warning", state === "reconnecting");
    indicator.classList.toggle("text-danger", state === "offline");
  };

  const fetchJSON = async (url) => {
    const response = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return response.json();
  };

  const refreshJobIndicator = async () => {
    const payload = await fetchJSON("/api/jobs?limit=5");
    const badge = document.querySelector("[data-job-indicator] .badge");
    if (badge) badge.textContent = String(payload.active_count);
  };

  const refreshInboxIndicator = async () => {
    const payload = await fetchJSON("/api/inbox?limit=1");
    const badge = document.querySelector("[data-inbox-indicator]");
    if (!badge) return;
    badge.textContent = String(payload.total);
    badge.hidden = payload.total === 0;
  };

  const applyJob = (job) => {
    const status = document.querySelector("[data-job-status]");
    const stage = document.querySelector("[data-job-stage]");
    const error = document.querySelector("[data-job-error]");
    const result = document.querySelector("[data-job-result]");
    const cancel = document.querySelector("[data-job-cancel]");
    if (status) {
      status.textContent = job.status;
      status.className = `badge status-${job.status}`;
    }
    if (stage) stage.textContent = job.progress_stage || "—";
    if (error) {
      error.textContent = job.error_summary || "";
      error.hidden = !job.error_summary;
    }
    if (result) {
      result.replaceChildren();
      if (job.result_id) {
        const value = `${job.result_kind}:${job.result_id}`;
        if (job.result_url) {
          const link = document.createElement("a");
          link.href = job.result_url;
          link.textContent = value;
          result.append(link);
        } else {
          result.textContent = value;
        }
        result.hidden = false;
      } else {
        result.hidden = true;
      }
    }
    if (cancel && terminalStates.has(job.status)) cancel.remove();
  };

  const refreshJob = async (id) => applyJob(await fetchJSON(`/api/jobs/${encodeURIComponent(id)}`));

  const createMessage = (message) => {
    const article = document.createElement("article");
    article.id = `message-${message.id}`;
    article.dataset.messageId = message.id;
    article.className = `message message-${message.role}${message.direction === "out" ? " message-out" : ""}${message.deleted_at ? " is-deleted" : ""}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const author = document.createElement("strong");
    author.textContent = message.author_name;
    const timestamp = document.createElement("time");
    timestamp.dataset.timestamp = String(message.timestamp);
    timestamp.dataset.localTime = "";
    meta.append(author, timestamp);
    if (message.edited_at) {
      const edited = document.createElement("span");
      edited.textContent = labels.edited;
      meta.append(edited);
    }

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = message.deleted_at ? labels.deleted : message.content;
    article.append(meta, bubble);
    for (const attachment of message.attachments || []) {
      const chip = document.createElement("span");
      chip.className = "attachment-chip";
      chip.textContent = `📎 ${attachment.filename || attachment.name || "attachment"}`;
      article.append(chip);
    }
    localizeTimes(article);
    return article;
  };

  const refreshChat = async () => {
    const container = document.querySelector("[data-chat-messages]");
    if (!container) return;
    const channel = container.dataset.channelId;
    const payload = await fetchJSON(`/api/chats/${encodeURIComponent(channel)}/messages?limit=100`);
    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
    let added = 0;
    for (const message of payload.items) {
      const existing = Array.from(container.querySelectorAll("[data-message-id]")).find(
        (element) => element.dataset.messageId === message.id,
      );
      const rendered = createMessage(message);
      if (existing) existing.replaceWith(rendered);
      else {
        container.querySelector("[data-new-messages]")?.before(rendered);
        added += 1;
      }
    }
    container.querySelector(".empty-state")?.remove();
    if (added && nearBottom) container.scrollTop = container.scrollHeight;
    else if (added) {
      const button = container.querySelector("[data-new-messages]");
      if (button) {
        button.textContent = `${labels.newMessages} (${added})`;
        button.hidden = false;
      }
    }
  };

  const loadOlder = async (button) => {
    const container = button.closest("[data-chat-messages]");
    if (!container) return;
    button.disabled = true;
    try {
      const channel = container.dataset.channelId;
      const before = encodeURIComponent(button.dataset.before);
      const payload = await fetchJSON(
        `/api/chats/${encodeURIComponent(channel)}/messages?limit=50&before=${before}`,
      );
      const previousHeight = container.scrollHeight;
      const firstMessage = container.querySelector("[data-message-id]");
      for (const message of payload.items) container.insertBefore(createMessage(message), firstMessage);
      container.scrollTop += container.scrollHeight - previousHeight;
      if (payload.next_before === null) button.remove();
      else {
        button.dataset.before = String(payload.next_before);
        button.disabled = false;
      }
    } catch (error) {
      button.disabled = false;
      throw error;
    }
  };

  document.querySelector("[data-load-older]")?.addEventListener("click", (event) => {
    loadOlder(event.currentTarget).catch(() => setLiveStatus("offline"));
  });
  document.querySelector("[data-new-messages]")?.addEventListener("click", (event) => {
    const container = event.currentTarget.closest("[data-chat-messages]");
    if (container) container.scrollTop = container.scrollHeight;
    event.currentTarget.hidden = true;
  });

  const refreshRun = async (id) => {
    const payload = await fetchJSON(`/api/runs/${encodeURIComponent(id)}`);
    const status = document.querySelector("[data-run-status]");
    if (status) {
      status.textContent = payload.run.status;
      status.className = `badge status-${payload.run.status}`;
    }
    const timeline = document.querySelector("[data-run-timeline]");
    if (!timeline) return;
    for (const event of payload.timeline) {
      if (timeline.querySelector(`[data-event-sequence="${event.sequence}"]`)) continue;
      const article = document.createElement("article");
      article.className = "timeline-event";
      article.dataset.eventSequence = String(event.sequence);
      const marker = document.createElement("div");
      marker.className = "timeline-marker";
      const copy = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = event.kind;
      copy.append(name);
      article.append(marker, copy);
      timeline.append(article);
    }
  };

  const resynchronize = async () => {
    const tasks = [refreshJobIndicator()];
    if (topic.startsWith("job:")) tasks.push(refreshJob(topic.slice(4)));
    else if (topic.startsWith("chat:")) tasks.push(refreshChat());
    else if (topic.startsWith("run:")) tasks.push(refreshRun(topic.slice(4)));
    else if (topic === "inbox") tasks.push(refreshInboxIndicator());
    await Promise.all(tasks);
  };

  const handleRecord = async (payload) => {
    if (payload.type === "hello") {
      setLiveStatus("live");
      await resynchronize();
    } else if (payload.type === "heartbeat") {
      setLiveStatus("live");
    } else if (payload.type === "jobs.updated") {
      await refreshJobIndicator();
    } else if (payload.type === "inbox.updated") {
      await refreshInboxIndicator();
    } else if (payload.type === "job.updated" && topic === `job:${payload.id}`) {
      await refreshJob(payload.id);
    } else if (payload.type === "chat.updated" && topic === `chat:${payload.id}`) {
      await refreshChat();
    } else if (payload.type === "run.updated" && topic === `run:${payload.id}`) {
      await refreshRun(payload.id);
    }
  };

  const consume = async () => {
    let backoff = 750;
    while (!stopped) {
      controller = new AbortController();
      try {
        const topics = Array.from(new Set(["jobs", "inbox", topic])).join(",");
        const response = await fetch(`/api/events/stream?topics=${encodeURIComponent(topics)}`, {
          headers: { Accept: "application/x-ndjson" },
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok || !response.body) throw new Error(`stream ${response.status}`);
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffered = "";
        backoff = 750;
        while (!stopped) {
          const { value, done } = await reader.read();
          if (done) break;
          buffered += value;
          if (buffered.length > 65536) throw new Error("oversized stream record");
          const lines = buffered.split("\n");
          buffered = lines.pop() || "";
          for (const line of lines) {
            if (line) await handleRecord(JSON.parse(line));
          }
        }
      } catch (error) {
        if (stopped || error.name === "AbortError") return;
        setLiveStatus("reconnecting");
      }
      const delay = backoff + Math.random() * Math.min(backoff * 0.25, 1000);
      await new Promise((resolve) => window.setTimeout(resolve, delay));
      backoff = Math.min(backoff * 2, 15000);
    }
  };

  window.addEventListener("pagehide", () => {
    stopped = true;
    controller?.abort();
  });
  consume().catch(() => setLiveStatus("offline"));
})();
