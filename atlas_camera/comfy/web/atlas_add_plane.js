/**
 * Atlas Draw Plane ✏️ — click an outline on the plate, get a plane clipped to it.
 *
 * Why this exists: every AtlasDerive* node fits planes from depth and emits
 * axis-extent RECTANGLES around each cluster. The rectangle covers image area
 * the real surface never occupied, and the viewport projects by WORLD POSITION
 * (see atlas_blockout.js:makeProjectionMaterial — geometry UVs are unused for
 * projection), so the overshoot receives paint from unrelated parts of the
 * plate. That is the smear on derived walls. Here the artist supplies the
 * outline, so the emitted mesh covers exactly the clicked region.
 *
 * This widget only edits the `polygons` STRING widget (0..1 plate-relative
 * points). All fitting lives in Python (core/polygon_planes.py) — nothing here
 * decides geometry, so a hand-edited JSON blob and a clicked one behave
 * identically.
 *
 * Conventions this file must keep (see docs/DESIGN_RULES.md):
 *  - NEVER assign lifecycle callbacks after addDOMWidget — always chain, or
 *    the DOM is orphaned on workflow switch.
 *  - No JS resize hooks for layout; sizing is CSS (`height:100%` chain,
 *    `min-width:0`). The canvas backing store is matched to its CSS box at
 *    draw time, which is not a layout hook.
 */
import { app } from "../../scripts/app.js";

const NODE_CLASS = "AtlasAddPlanePolygon";
const HANDLE_PX = 5;
const HIT_PX = 8;
const DEFAULT_NODE_SIZE = [420, 560];

// LiteGraph otherwise lays the DOM widget out at its own `width`, which lets
// the canvas escape the node border. Same fix, same reason, as
// atlas_blockout.js:pinDomWidgetFullWidth.
function pinDomWidgetFullWidth(domWidget) {
  try {
    Object.defineProperty(domWidget, "width", {
      configurable: true,
      get() { return undefined; },
      set() {},
    });
  } catch (e) {
    console.warn("[Atlas] could not pin polygon canvas width:", e);
  }
}

// Outline colour by last-run outcome, so a fallback is visible on the canvas
// rather than buried in the report string.
const OUTCOME_COLORS = {
  pending: "#8ab4f8",
  depth_ransac: "#7ddc86",
  rectangle_homography: "#f0b65a",
  failed: "#f06a6a",
};

function parsePolygons(text) {
  if (!text || !text.trim()) return [];
  try {
    const blob = JSON.parse(text);
    const list = Array.isArray(blob) ? blob : blob.polygons;
    if (!Array.isArray(list)) return [];
    return list.map((p, i) => ({
      id: p.id || `p${i + 1}`,
      label: p.label || `plane ${i + 1}`,
      points: (p.points || []).map((pt) => [Number(pt[0]), Number(pt[1])]),
      fit_mode: p.fit_mode || "inherit",
      enabled: p.enabled !== false,
    }));
  } catch (e) {
    console.warn("[Atlas] polygons widget is not valid JSON; leaving it alone", e);
    return null;
  }
}

function serializePolygons(polygons) {
  return JSON.stringify({
    version: 1,
    polygons: polygons.map((p) => ({
      id: p.id,
      label: p.label,
      points: p.points.map(([u, v]) => [Number(u.toFixed(5)), Number(v.toFixed(5))]),
      fit_mode: p.fit_mode,
      enabled: p.enabled,
    })),
  }, null, 1);
}

function nextId(polygons) {
  let n = polygons.length + 1;
  const taken = new Set(polygons.map((p) => p.id));
  while (taken.has(`p${n}`)) n += 1;
  return `p${n}`;
}

app.registerExtension({
  name: "atlas.add_plane_polygon",

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;
    await new Promise((r) => setTimeout(r, 0));

    const polygonsWidget = node.widgets?.find((w) => w.name === "polygons");
    if (!polygonsWidget) return;

    const state = {
      polygons: parsePolygons(polygonsWidget.value) || [],
      activeIndex: -1,
      draft: null,          // in-progress outline (array of [u, v])
      dragVertex: null,     // { polygon, index }
      statuses: {},         // polygon id -> { ok, method, note }
      plate: null,          // HTMLImageElement
      imageRect: null,      // where the plate is drawn inside the canvas
    };

    const container = document.createElement("div");
    container.style.cssText =
      "width:100%;height:100%;min-width:0;display:flex;flex-direction:column;gap:4px;" +
      "overflow:hidden;";

    const hint = document.createElement("div");
    hint.style.cssText = "font-size:10px;color:#9aa;line-height:1.4;padding:0 2px;";
    hint.textContent =
      "Click to place points · double-click or Enter closes · Esc discards · " +
      "drag a point to move · Delete removes the selected outline. Run once to load the plate.";

    const canvas = document.createElement("canvas");
    canvas.tabIndex = 0;
    canvas.style.cssText =
      "flex:1 1 auto;width:100%;min-height:160px;min-width:0;display:block;" +
      "background:#181818;border-radius:4px;outline:none;cursor:crosshair;";

    const list = document.createElement("div");
    list.style.cssText =
      "flex:0 0 auto;max-height:96px;overflow-y:auto;font-size:11px;" +
      "display:flex;flex-direction:column;gap:2px;";

    container.append(hint, canvas, list);

    // Sizing follows atlas_blockout.js's proven DOM-widget contract: pinned
    // width plus the sanctioned getMinHeight/getMaxHeight hooks. Never
    // `computeSize` — returning a width there is what let this canvas render
    // outside the node's own border.
    const domWidget = node.addDOMWidget("atlas_polygon_canvas", "div", container, {
      serialize: false,
      getValue() { return null; },
      setValue() {},
      getMinHeight() { return 220; },
      getMaxHeight() { return 4096; },
    });
    pinDomWidgetFullWidth(domWidget);

    // ---- geometry helpers -------------------------------------------------

    function layout() {
      const box = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(box.width));
      const h = Math.max(1, Math.round(box.height));
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      const aspect = state.plate ? state.plate.width / state.plate.height : 16 / 9;
      let rw = w;
      let rh = w / aspect;
      if (rh > h) { rh = h; rw = h * aspect; }
      state.imageRect = { x: (w - rw) / 2, y: (h - rh) / 2, w: rw, h: rh };
    }

    function toCanvas([u, v]) {
      const r = state.imageRect;
      return [r.x + u * r.w, r.y + v * r.h];
    }

    function toNormalized(px, py) {
      const r = state.imageRect;
      return [
        Math.min(1, Math.max(0, (px - r.x) / r.w)),
        Math.min(1, Math.max(0, (py - r.y) / r.h)),
      ];
    }

    function eventPoint(ev) {
      const box = canvas.getBoundingClientRect();
      return [
        (ev.clientX - box.left) * (canvas.width / box.width),
        (ev.clientY - box.top) * (canvas.height / box.height),
      ];
    }

    function outlineColor(polygon) {
      const status = state.statuses[polygon.id];
      if (!status) return OUTCOME_COLORS.pending;
      if (!status.ok) return OUTCOME_COLORS.failed;
      return OUTCOME_COLORS[status.method] || OUTCOME_COLORS.pending;
    }

    function hitVertex(px, py) {
      for (let i = state.polygons.length - 1; i >= 0; i -= 1) {
        const poly = state.polygons[i];
        for (let j = 0; j < poly.points.length; j += 1) {
          const [cx, cy] = toCanvas(poly.points[j]);
          if (Math.hypot(cx - px, cy - py) <= HIT_PX) return { polygon: i, index: j };
        }
      }
      return null;
    }

    function hitBody(px, py) {
      for (let i = state.polygons.length - 1; i >= 0; i -= 1) {
        const pts = state.polygons[i].points.map(toCanvas);
        let inside = false;
        for (let a = 0, b = pts.length - 1; a < pts.length; b = a, a += 1) {
          const [xa, ya] = pts[a];
          const [xb, yb] = pts[b];
          if ((ya > py) !== (yb > py)
              && px < ((xb - xa) * (py - ya)) / (yb - ya) + xa) inside = !inside;
        }
        if (inside) return i;
      }
      return -1;
    }

    // ---- rendering --------------------------------------------------------

    function drawOutline(ctx, points, color, { closed, active }) {
      if (!points.length) return;
      ctx.beginPath();
      points.map(toCanvas).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (closed) ctx.closePath();
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.stroke();
      if (closed) {
        // Fills stay faint: the plate underneath has to stay readable.
        ctx.fillStyle = color + "22";
        ctx.fill();
      }
      ctx.fillStyle = color;
      for (const [x, y] of points.map(toCanvas)) {
        ctx.beginPath();
        ctx.arc(x, y, HANDLE_PX, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function draw() {
      layout();
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const r = state.imageRect;
      if (state.plate) {
        ctx.drawImage(state.plate, r.x, r.y, r.w, r.h);
      } else {
        ctx.fillStyle = "#222";
        ctx.fillRect(r.x, r.y, r.w, r.h);
        ctx.fillStyle = "#666";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Wire an image and run once to load the plate",
                     r.x + r.w / 2, r.y + r.h / 2);
      }
      state.polygons.forEach((poly, i) => {
        ctx.globalAlpha = poly.enabled ? 1 : 0.35;
        drawOutline(ctx, poly.points, outlineColor(poly),
                    { closed: true, active: i === state.activeIndex });
        ctx.globalAlpha = 1;
      });
      if (state.draft) {
        drawOutline(ctx, state.draft, "#ffffff", { closed: false, active: true });
      }
    }

    function renderList() {
      list.replaceChildren();
      state.polygons.forEach((poly, i) => {
        const row = document.createElement("div");
        row.style.cssText =
          "display:flex;align-items:center;gap:6px;padding:2px 4px;border-radius:3px;" +
          (i === state.activeIndex ? "background:#2c3540;" : "");
        const swatch = document.createElement("span");
        swatch.style.cssText =
          `width:8px;height:8px;border-radius:2px;background:${outlineColor(poly)};flex:0 0 auto;`;
        const toggle = document.createElement("input");
        toggle.type = "checkbox";
        toggle.checked = poly.enabled;
        toggle.onchange = () => { poly.enabled = toggle.checked; commit(); };
        const name = document.createElement("span");
        name.textContent = `${poly.label} · ${poly.points.length} pts`;
        name.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;";
        const status = state.statuses[poly.id];
        const note = document.createElement("span");
        note.style.cssText = "color:#9aa;flex:0 0 auto;";
        note.textContent = status ? (status.ok ? status.method : `skipped: ${status.note}`) : "";
        row.onclick = (ev) => {
          if (ev.target === toggle) return;
          state.activeIndex = i;
          renderList();
          draw();
        };
        row.append(swatch, toggle, name, note);
        list.append(row);
      });
    }

    function commit() {
      polygonsWidget.value = serializePolygons(state.polygons);
      renderList();
      draw();
    }

    // ---- interaction ------------------------------------------------------

    function closeDraft() {
      if (!state.draft || state.draft.length < 3) { state.draft = null; draw(); return; }
      state.polygons.push({
        id: nextId(state.polygons),
        label: `plane ${state.polygons.length + 1}`,
        points: state.draft,
        fit_mode: "inherit",
        enabled: true,
      });
      state.activeIndex = state.polygons.length - 1;
      state.draft = null;
      commit();
    }

    canvas.addEventListener("pointerdown", (ev) => {
      canvas.focus({ preventScroll: true });
      const [px, py] = eventPoint(ev);
      const vertex = hitVertex(px, py);
      if (vertex && !state.draft) {
        state.dragVertex = vertex;
        state.activeIndex = vertex.polygon;
        canvas.setPointerCapture(ev.pointerId);
        renderList();
        return;
      }
      if (!state.draft) {
        const body = hitBody(px, py);
        if (body >= 0 && !ev.shiftKey) {
          state.activeIndex = body;
          renderList();
          draw();
          return;
        }
        state.draft = [];
      }
      state.draft.push(toNormalized(px, py));
      draw();
    });

    canvas.addEventListener("pointermove", (ev) => {
      if (!state.dragVertex) return;
      const [px, py] = eventPoint(ev);
      const poly = state.polygons[state.dragVertex.polygon];
      poly.points[state.dragVertex.index] = toNormalized(px, py);
      draw();
    });

    canvas.addEventListener("pointerup", (ev) => {
      if (!state.dragVertex) return;
      state.dragVertex = null;
      canvas.releasePointerCapture?.(ev.pointerId);
      commit();
    });

    canvas.addEventListener("dblclick", (ev) => {
      ev.preventDefault();
      // The dblclick's own pointerdown already appended a duplicate point.
      if (state.draft && state.draft.length > 3) state.draft.pop();
      closeDraft();
    });

    canvas.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { closeDraft(); ev.preventDefault(); }
      else if (ev.key === "Escape") { state.draft = null; draw(); ev.preventDefault(); }
      else if (ev.key === "Delete" || ev.key === "Backspace") {
        if (state.draft) { state.draft.pop(); draw(); }
        else if (state.activeIndex >= 0) {
          state.polygons.splice(state.activeIndex, 1);
          state.activeIndex = -1;
          commit();
        }
        ev.preventDefault();
      }
    });

    // ---- lifecycle (CHAINED, never assigned) ------------------------------

    const prevExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const out = prevExecuted?.apply(this, arguments);
      const plate = message?.atlas_plate?.[0];
      if (plate) {
        const img = new Image();
        img.onload = () => { state.plate = img; draw(); };
        img.src = plate;
      }
      const statuses = message?.atlas_polygon_status?.[0];
      if (statuses) {
        try {
          state.statuses = Object.fromEntries(
            JSON.parse(statuses).map((s) => [s.id, s]));
        } catch (e) {
          console.warn("[Atlas] could not read polygon statuses", e);
        }
        renderList();
        draw();
      }
      return out;
    };

    // A hand-edited or workflow-restored blob is the source of truth; re-read
    // it rather than assuming the canvas wrote whatever is in there.
    const prevConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const out = prevConfigure?.apply(this, args);
      const parsed = parsePolygons(polygonsWidget.value);
      if (parsed) {
        state.polygons = parsed;
        state.activeIndex = -1;
        renderList();
        draw();
      }
      return out;
    };

    // The canvas backing store has to follow its CSS box, or a node resize
    // leaves a stale (stretched, or blank grey) bitmap. This is a draw-time
    // sync, not a layout hook: CSS still decides the box, and no JS ever sets
    // the element's width/height style.
    let boxW = 0;
    let boxH = 0;
    let ticking = true;
    const tick = () => {
      if (!ticking) return;
      if (container.isConnected
          && (canvas.clientWidth !== boxW || canvas.clientHeight !== boxH)) {
        boxW = canvas.clientWidth;
        boxH = canvas.clientHeight;
        if (boxW > 0 && boxH > 0) draw();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    const prevRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
      ticking = false;
      state.plate = null;
      state.polygons = [];
      list.replaceChildren();
      return prevRemoved?.apply(this, args);
    };

    // Freshly added nodes get a usable canvas; a node restored from a save
    // keeps whatever size it stored (onConfigure has already run by now).
    node.setSize([
      Math.max(node.size[0], DEFAULT_NODE_SIZE[0]),
      Math.max(node.size[1], DEFAULT_NODE_SIZE[1]),
    ]);

    renderList();
    draw();
  },
});
