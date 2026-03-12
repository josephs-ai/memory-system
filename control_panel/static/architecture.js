import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

mermaid.initialize({
  startOnLoad: true,
  theme: "dark",
  securityLevel: "loose",
  flowchart: {
    htmlLabels: true,
    curve: "basis",
    useMaxWidth: false
  }
});

function bindGraphControls() {
  const surface = document.getElementById("graph-surface");
  const canvas = document.getElementById("graph-canvas");
  const zoomIn = document.getElementById("graph-zoom-in");
  const zoomOut = document.getElementById("graph-zoom-out");
  const zoomReset = document.getElementById("graph-zoom-reset");

  if (!surface || !canvas || surface.dataset.graphBound === "1") return;
  surface.dataset.graphBound = "1";

  let scale = 1;
  const MIN_SCALE = 0.55;
  const MAX_SCALE = 2.2;
  const STEP = 0.12;

  const updateZoomLabel = () => {
    if (zoomReset) zoomReset.textContent = `${Math.round(scale * 100)}%`;
  };

  const applyScale = (nextScale, anchorX = surface.clientWidth / 2, anchorY = surface.clientHeight / 2) => {
    const clamped = Math.max(MIN_SCALE, Math.min(MAX_SCALE, nextScale));
    const prevScale = scale;
    if (clamped === prevScale) return;

    const worldX = (surface.scrollLeft + anchorX) / prevScale;
    const worldY = (surface.scrollTop + anchorY) / prevScale;

    scale = clamped;
    canvas.style.transform = `scale(${scale})`;

    surface.scrollLeft = worldX * scale - anchorX;
    surface.scrollTop = worldY * scale - anchorY;
    updateZoomLabel();
  };

  zoomIn?.addEventListener("click", () => applyScale(scale + STEP));
  zoomOut?.addEventListener("click", () => applyScale(scale - STEP));
  zoomReset?.addEventListener("click", () => applyScale(1));

  surface.addEventListener(
    "wheel",
    (e) => {
      const zoomGesture = e.ctrlKey || e.metaKey;

      if (zoomGesture) {
        e.preventDefault();
        const rect = surface.getBoundingClientRect();
        const anchorX = e.clientX - rect.left;
        const anchorY = e.clientY - rect.top;
        const direction = e.deltaY > 0 ? -STEP : STEP;
        applyScale(scale + direction, anchorX, anchorY);
        return;
      }

      e.preventDefault();
      surface.scrollLeft += e.deltaX;
      surface.scrollTop += e.deltaY;
    },
    { passive: false }
  );

  updateZoomLabel();
}

function showArchitectureDetail(nodeId) {
  const ids = window.OPENCLAW_ARCH_NODE_IDS || [];
  ids.forEach((id) => {
    const panel = document.getElementById(`detail-${id}`);
    if (panel) {
      panel.style.display = id === nodeId ? "block" : "none";
    }
  });

  document.querySelectorAll(".arch-node-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.nodeId === nodeId);
  });
}

function bindDetailButtons() {
  document.querySelectorAll(".arch-node-btn").forEach((btn) => {
    if (btn.dataset.bound === "1") return;
    btn.dataset.bound = "1";
    btn.addEventListener("click", () => {
      showArchitectureDetail(btn.dataset.nodeId);
    });
  });
}

function bindGraphNodeLinks() {
  const surface = document.getElementById("graph-surface");
  if (!surface || surface.dataset.graphLinksBound === "1") return;
  surface.dataset.graphLinksBound = "1";

  surface.addEventListener("click", (event) => {
    const link = event.target.closest("a");
    if (!link) return;

    const href = link.getAttribute("href") || "";
    const prefix = "/architecture/node/";
    if (!href.includes(prefix)) return;

    event.preventDefault();

    const nodeId = href.split(prefix)[1];
    if (!nodeId) return;

    showArchitectureDetail(nodeId);
  });

  surface.addEventListener("mouseover", (event) => {
    const link = event.target.closest("a");
    if (link) {
      link.style.cursor = "pointer";
    }
  });
}

window.addEventListener("load", () => {
  setTimeout(() => {
    bindGraphControls();
    bindDetailButtons();
    bindGraphNodeLinks();

    if (window.OPENCLAW_ARCH_FIRST_NODE) {
      showArchitectureDetail(window.OPENCLAW_ARCH_FIRST_NODE);
    }
  }, 900);
});
