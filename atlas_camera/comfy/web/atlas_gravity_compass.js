/**
 * Atlas Gravity Compass — direct-manipulation gravity override.
 *
 * The hidden native widgets remain the serialized/API contract.  This panel
 * is deliberately only a view/controller over those values, so a workflow is
 * reproducible without the frontend and there is no second solve path.
 */
import { app } from "../../scripts/app.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap180 = (v) => ((v + 180) % 360 + 360) % 360 - 180;

// World-down expressed in camera coordinates for Atlas's absolute controls.
// Exported by name for the cross-language numeric contract test.
function gravityDirectionFromAngles(pitchDeg, rollDeg) {
  const p = pitchDeg * Math.PI / 180;
  const r = rollDeg * Math.PI / 180;
  return { x: Math.sin(r) * Math.cos(p), y: -Math.cos(r) * Math.cos(p), z: -Math.sin(p) };
}

function anglesFromGravityDirection(direction) {
  const n = Math.hypot(direction.x, direction.y, direction.z) || 1;
  const x = direction.x / n, y = direction.y / n, z = direction.z / n;
  return { pitch: Math.asin(clamp(-z, -1, 1)) * 180 / Math.PI,
    roll: Math.atan2(x, -y) * 180 / Math.PI };
}

function hideNativeWidget(widget) {
  if (!widget || widget._atlasCompassHidden) return;
  widget._atlasCompassHidden = true;
  widget._atlasCompassOriginalType = widget.type;
  widget._atlasCompassOriginalComputeSize = widget.computeSize;
  widget.type = "atlas_hidden";
  widget.computeSize = (width) => [width, -4];
  // LiteGraph and Vue consult different visibility fields.  Both must be set
  // or a zero-height native value can still paint across the compass/node.
  widget.hidden = true;
  if (widget.options) widget.options.hidden = true;
}

function setNative(widget, value) {
  if (!widget || widget.value === value) return;
  widget.value = value;
  widget.callback?.(value);
}

function button(label, title) {
  const el = document.createElement("button");
  el.textContent = label;
  el.title = title;
  Object.assign(el.style, {
    border: "1px solid #375568", borderRadius: "7px", padding: "7px 10px",
    color: "#d9f7ff", background: "#122430", font: "700 10px system-ui",
    letterSpacing: ".08em", cursor: "pointer",
  });
  return el;
}

app.registerExtension({
  name: "AtlasCamera.GravityCompass",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["AtlasGravityCompass", "AtlasSolveGate"].includes(nodeData.name)) return;
    const THREE = await import("./lib/atlas-three.bundle.js");
    const original = nodeType.prototype.onNodeCreated;
    const originalExecuted = nodeType.prototype.onExecuted;

    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      const source = Array.isArray(message?.source_image)
        ? message.source_image[0] : message?.source_image;
      if (source) {
        this._atlasGravityCompassImage = source;
        if (this._atlasGravityCompassPlate) this._atlasGravityCompassPlate.src = source;
      }
    };

    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      const node = this;
      const enabled = node.widgets?.find((w) => w.name === "apply_override");
      const pitch = node.widgets?.find((w) => w.name === "pitch_deg");
      const roll = node.widgets?.find((w) => w.name === "roll_deg");
      const headingEnabled = node.widgets?.find((w) => w.name === "heading_override");
      const headingAngle = node.widgets?.find((w) => w.name === "heading_deg");
      [enabled, pitch, roll, headingEnabled, headingAngle].forEach(hideNativeWidget);

      const root = document.createElement("div");
      Object.assign(root.style, {
        width: "100%", height: "430px", boxSizing: "border-box", overflow: "hidden",
        border: "1px solid #294a58", borderRadius: "11px", color: "#d9f7ff",
        background: "radial-gradient(circle at 50% 35%, #173b45 0%, #09161e 62%, #060d12 100%)",
        fontFamily: "Inter, system-ui, sans-serif", userSelect: "none", position: "relative",
      });

      const plate = document.createElement("img");
      plate.alt = "Connected source image";
      plate.draggable = false;
      Object.assign(plate.style, {
        position: "absolute", left: "0", right: "0", top: "50px", bottom: "92px",
        width: "100%", height: "288px", objectFit: "contain", zIndex: "0",
        opacity: ".72", filter: "saturate(.72) brightness(.52) contrast(1.12)",
        pointerEvents: "none", background: "#050a0e",
      });
      if (node._atlasGravityCompassImage) plate.src = node._atlasGravityCompassImage;
      node._atlasGravityCompassPlate = plate;
      root.appendChild(plate);

      const heading = document.createElement("div");
      heading.innerHTML = '<div style="font-size:10px;letter-spacing:.24em;color:#6cd8e5">ATLAS ORIENTATION</div>' +
        '<div style="font-size:19px;font-weight:800;letter-spacing:.05em;margin-top:2px">GRAVITY COMPASS</div>';
      Object.assign(heading.style, { position: "absolute", left: "15px", top: "13px", zIndex: 2 });
      root.appendChild(heading);

      const badge = document.createElement("div");
      Object.assign(badge.style, { position: "absolute", right: "13px", top: "14px", zIndex: 2,
        borderRadius: "99px", padding: "5px 8px", font: "800 9px system-ui", letterSpacing: ".11em" });
      root.appendChild(badge);

      const canvas = document.createElement("canvas");
      Object.assign(canvas.style, { position: "absolute", inset: "50px 0 92px 0", width: "100%",
        height: "288px", cursor: "grab", touchAction: "none", zIndex: "1" });
      root.appendChild(canvas);

      const readout = document.createElement("div");
      Object.assign(readout.style, { position: "absolute", left: "14px", right: "14px", bottom: "53px",
        display: "flex", justifyContent: "space-between", color: "#86dbe4",
        font: "600 11px ui-monospace, monospace", letterSpacing: ".05em" });
      root.appendChild(readout);

      const controls = document.createElement("div");
      Object.assign(controls.style, { position: "absolute", left: "12px", right: "12px", bottom: "10px",
        display: "flex", gap: "7px" });
      const solveBtn = button("USE SOLVE", "Disable the override and preserve solved gravity");
      const levelBtn = button("LEVEL", "Set absolute pitch and roll to zero");
      const downBtn = button("LOOK DOWN", "Set a clear 30 degree downward camera pitch");
      controls.append(solveBtn, levelBtn, downBtn);
      root.appendChild(controls);

      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x000000, 0);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(31, 1, 0.1, 100);
      camera.position.set(3.15, 2.15, 4.5);
      camera.lookAt(0, 0, 0);
      scene.add(new THREE.AmbientLight(0x9bd8e0, 1.4));
      const key = new THREE.DirectionalLight(0xffbd62, 2.5);
      key.position.set(2, 3, 4);
      scene.add(key);

      const ringMat = new THREE.LineBasicMaterial({ color: 0x397486, transparent: true, opacity: 0.72 });
      [0, Math.PI / 2].forEach((rot) => {
        const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
          Array.from({ length: 96 }, (_, i) => new THREE.Vector3(Math.cos(i * Math.PI * 2 / 96) * 1.25,
            Math.sin(i * Math.PI * 2 / 96) * 1.25, 0))), ringMat);
        ring.rotation.y = rot;
        scene.add(ring);
      });
      // Gravity resolves world up; this frame resolves the remaining yaw
      // ambiguity. The X/Y/Z triad and its floor rotate together so the artist
      // can line the Atlas world grid up directly against the plate.
      const worldFrame = new THREE.Group();
      scene.add(worldFrame);
      const floor = new THREE.GridHelper(3.25, 12, 0x438596, 0x193a47);
      floor.position.y = -1.28;
      worldFrame.add(floor);
      const axisOrigin = new THREE.Vector3(0, -0.42, 0);
      const axes = [
        [new THREE.Vector3(1, 0, 0), 0xef4a4a],
        [new THREE.Vector3(0, 1, 0), 0x63e17d],
        [new THREE.Vector3(0, 0, 1), 0x4a8eff],
      ];
      for (const [direction, color] of axes) {
        worldFrame.add(new THREE.ArrowHelper(direction, axisOrigin, .72, color, .16, .09));
      }
      const headingRing = new THREE.Mesh(
        new THREE.RingGeometry(.58, .7, 64),
        new THREE.MeshBasicMaterial({ color: 0xffb347, transparent: true, opacity: .82,
          side: THREE.DoubleSide, depthTest: false }));
      headingRing.rotation.x = -Math.PI / 2;
      headingRing.position.copy(axisOrigin);
      worldFrame.add(headingRing);
      const headingArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), axisOrigin,
        .92, 0xffb347, .2, .11);
      worldFrame.add(headingArrow);
      const headingPick = new THREE.Mesh(
        new THREE.CircleGeometry(.82, 48),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, side: THREE.DoubleSide,
          depthWrite: false }));
      headingPick.rotation.x = -Math.PI / 2;
      headingPick.position.copy(axisOrigin);
      worldFrame.add(headingPick);
      const core = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0x7ee8f1, emissive: 0x123e47, metalness: .45, roughness: .25 }));
      core.position.y = .88;
      scene.add(core);
      const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, .88, 0), 2.15,
        0xffa72f, .35, .19);
      scene.add(arrow);

      function approveGateDecision() {
        // Direct manipulation AT the solve gate is the artist's approval. Let
        // this corrected solve flow on the next queue, then atlas_solve_gate.js
        // scopes the approval to the fingerprint returned by that execution.
        if (nodeData.name === "AtlasSolveGate") {
          setNative(node.widgets?.find((w) => w.name === "proceed"), true);
          setNative(node.widgets?.find((w) => w.name === "approved_for"), "");
          node._atlasSolveGateFingerprint = null;
          node._atlasCompassPendingApproval = true;
        }
      }
      function activate(p, r, h = Number(headingAngle?.value || 0), useHeading = Boolean(headingEnabled?.value)) {
        approveGateDecision();
        setNative(enabled, true);
        setNative(pitch, Math.round(clamp(p, -89, 89) * 20) / 20);
        setNative(roll, Math.round(wrap180(r) * 20) / 20);
        setNative(headingEnabled, Boolean(useHeading));
        if (useHeading) setNative(headingAngle, Math.round(wrap180(h) * 20) / 20);
        node.setDirtyCanvas(true, true);
      }
      function queueDecision() {
        // Submit once at the interaction boundary, never on every pointermove:
        // the downstream depth/geometry branch may be expensive.
        app.queuePrompt(0, 1);
      }
      solveBtn.onclick = () => {
        approveGateDecision();
        setNative(enabled, false);
        setNative(headingEnabled, false);
        node.setDirtyCanvas(true, true);
        queueDecision();
      };
      levelBtn.onclick = () => { activate(0, 0); queueDecision(); };
      downBtn.onclick = () => { activate(30, 0); queueDecision(); };

      let drag = null;
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      canvas.addEventListener("pointerdown", (event) => {
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
        const rect = canvas.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const headingHit = raycaster.intersectObject(headingPick, false).length > 0;
        drag = { x: event.clientX, y: event.clientY,
          mode: headingHit ? "heading" : "gravity",
          pitch: Number(pitch?.value || 0), roll: Number(roll?.value || 0),
          heading: Number(headingAngle?.value || 0), moved: false };
      });
      canvas.addEventListener("pointermove", (event) => {
        if (!drag) return;
        if (Math.abs(event.clientX - drag.x) + Math.abs(event.clientY - drag.y) > 1) {
          drag.moved = true;
        }
        const fine = event.shiftKey ? .18 : 1;
        if (drag.mode === "heading") {
          activate(drag.pitch, drag.roll,
            drag.heading + (event.clientX - drag.x) * .62 * fine, true);
        } else {
          activate(drag.pitch + (event.clientY - drag.y) * .34 * fine,
            drag.roll + (event.clientX - drag.x) * .48 * fine);
        }
      });
      const endDrag = () => {
        const changed = Boolean(drag?.moved);
        drag = null;
        canvas.style.cursor = "grab";
        if (changed) queueDecision();
      };
      canvas.addEventListener("pointerup", endDrag);
      canvas.addEventListener("pointercancel", endDrag);

      let disposed = false;
      function frame() {
        if (disposed) return;
        const active = Boolean(enabled?.value);
        const p = Number(pitch?.value || 0), r = Number(roll?.value || 0);
        const h = Number(headingAngle?.value || 0);
        const headingActive = Boolean(headingEnabled?.value);
        const g = gravityDirectionFromAngles(p, r);
        arrow.setDirection(new THREE.Vector3(g.x, g.y, g.z).normalize());
        arrow.setColor(new THREE.Color(active ? 0xffa72f : 0x54727a));
        badge.textContent = active ? "OVERRIDE ACTIVE" : "SOLVED GRAVITY";
        Object.assign(badge.style, { color: active ? "#ffd28b" : "#8eb1b9",
          background: active ? "#4b2d0c" : "#142a31", border: `1px solid ${active ? "#b87721" : "#35545d"}` });
        worldFrame.rotation.y = headingActive ? -h * Math.PI / 180 : 0;
        headingRing.material.color.set(headingActive ? 0xffb347 : 0x54727a);
        headingArrow.setColor(new THREE.Color(headingActive ? 0xffb347 : 0x54727a));
        readout.innerHTML = `<span>PITCH ${p >= 0 ? "+" : ""}${p.toFixed(1)}°</span>` +
          `<span>HEADING ${headingActive ? `${h >= 0 ? "+" : ""}${h.toFixed(1)}°` : "SOLVED"}</span>` +
          `<span>ROLL ${r >= 0 ? "+" : ""}${r.toFixed(1)}°</span>`;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width)), height = Math.max(1, Math.round(rect.height));
        if (canvas.width !== width || canvas.height !== height) {
          renderer.setSize(width, height, false);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
        }
        renderer.render(scene, camera);
        requestAnimationFrame(frame);
      }
      frame();

      const domWidget = node.addDOMWidget("gravity_compass", "GRAVITY_COMPASS", root, { serialize: false });
      domWidget.serialize = false;
      node.setSize([Math.max(node.size?.[0] || 0, 430), 535]);

      const priorRemoved = node.onRemoved;
      node.onRemoved = function () {
        disposed = true;
        renderer.dispose();
        priorRemoved?.apply(this, arguments);
      };
    };
  },
});
