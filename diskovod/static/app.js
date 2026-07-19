(() => {
  "use strict";

  const root = document.documentElement;
  const theme = root.dataset.adminTheme || "system";
  const colorScheme = window.matchMedia("(prefers-color-scheme: dark)");

  const applySystemTheme = () => {
    if (theme === "system") {
      root.dataset.bsTheme = colorScheme.matches ? "dark" : "light";
    }
  };
  applySystemTheme();
  colorScheme.addEventListener("change", applySystemTheme);

  const sidebarButton = document.querySelector(".sidebar-toggle");
  sidebarButton?.addEventListener("click", () => {
    const open = document.body.classList.toggle("sidebar-open");
    sidebarButton.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
      document.body.classList.remove("sidebar-open");
      sidebarButton?.setAttribute("aria-expanded", "false");
      sidebarButton?.focus();
    }
  });

  const terminalStates = new Set(["succeeded", "failed", "cancelled"]);
  const topic = document.body.dataset.liveTopic;
  if (!topic || !("ReadableStream" in window)) return;

  let stopped = false;
  const consume = async () => {
    while (!stopped) {
      try {
        const response = await fetch(`/api/events/stream?topics=jobs,${encodeURIComponent(topic)}`, {
          headers: { Accept: "application/x-ndjson" },
          cache: "no-store",
        });
        if (!response.ok || !response.body) throw new Error(`stream ${response.status}`);
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffered = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffered += value;
          const lines = buffered.split("\n");
          buffered = lines.pop() || "";
          for (const line of lines) {
            if (line) update(JSON.parse(line));
          }
        }
      } catch (_error) {
        document.querySelector("[data-live-status]")?.classList.add("text-warning");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
  };

  const update = (payload) => {
    if (!Array.isArray(payload.jobs)) return;
    const active = payload.active_job_count ?? payload.jobs.filter((job) => !terminalStates.has(job.status)).length;
    const badge = document.querySelector("[data-job-indicator] .badge");
    if (badge) badge.textContent = String(active);
    if (topic.startsWith("job:")) {
      const id = topic.slice(4);
      const job = payload.jobs.find((item) => item.id === id);
      if (!job) return;
      const status = document.querySelector("[data-job-status]");
      const stage = document.querySelector("[data-job-stage]");
      if (status) {
        status.textContent = job.status;
        status.className = `badge status-${job.status}`;
      }
      if (stage) stage.textContent = job.progress_stage || "—";
      if (terminalStates.has(job.status)) window.setTimeout(() => window.location.reload(), 250);
    }
  };

  window.addEventListener("pagehide", () => {
    stopped = true;
  });
  consume();
})();
