import type { Constraints, LlmGuidance, Viewport3DProxyObject, Viewport3DProxyType, Viewport3DState } from "./types";

export const defaultViewport3DState: Viewport3DState = {
  schema_version: 1,
  display: {
    active_mode: "image_match",
    show_image: true,
    show_grid: true,
    show_axes: true,
    show_frustum: true,
    show_guides: true,
    show_proxies: true,
    show_horizon: true,
    show_projection: false,
    image_opacity: 0.78,
    grid_scale: 1,
    lock_camera_to_view: false
  },
  proxy_objects: [],
  camera_overrides: {},
  selected_proxy_id: null
};

const proxyPresets: Record<Viewport3DProxyType, Omit<Viewport3DProxyObject, "id" | "source" | "locked">> = {
  person_card: {
    type: "person_card",
    label: "Person card 1.75m",
    position: [0, 0.875, -4],
    rotation: [0, 0, 0],
    scale: [0.55, 1.75, 0.02]
  },
  box: {
    type: "box",
    label: "Cardboard box",
    position: [1, 0.35, -3],
    rotation: [0, 0, 0],
    scale: [0.75, 0.7, 0.55]
  },
  floor_plane: {
    type: "floor_plane",
    label: "Floor plane",
    position: [0, 0, -3],
    rotation: [0, 0, 0],
    scale: [6, 0.02, 8]
  },
  wall_plane: {
    type: "wall_plane",
    label: "Wall plane",
    position: [-2, 1.5, -4],
    rotation: [0, 0, 0],
    scale: [0.04, 3, 8]
  },
  corridor: {
    type: "corridor",
    label: "Corridor volume",
    position: [0, 1.25, -5],
    rotation: [0, 0, 0],
    scale: [3, 2.5, 8]
  },
  unit_box: {
    type: "unit_box",
    label: "Unit cube 1m",
    position: [0, 0.5, -2],
    rotation: [0, 0, 0],
    scale: [1, 1, 1]
  },
  custom_box: {
    type: "custom_box",
    label: "Custom scale box",
    position: [0, 0.5, -2],
    rotation: [0, 0, 0],
    scale: [1, 1, 1]
  }
};

export function normalizeViewport3DState(value: unknown): Viewport3DState {
  if (!value || typeof value !== "object") return structuredClone(defaultViewport3DState);
  const incoming = value as Partial<Viewport3DState>;
  return {
    ...structuredClone(defaultViewport3DState),
    ...incoming,
    schema_version: 1,
    display: {
      ...defaultViewport3DState.display,
      ...(incoming.display ?? {})
    },
    proxy_objects: Array.isArray(incoming.proxy_objects) ? incoming.proxy_objects : [],
    camera_overrides: incoming.camera_overrides ?? {},
    selected_proxy_id: incoming.selected_proxy_id ?? null
  };
}

export function withViewport3DState(constraints: Constraints, viewport3d: Viewport3DState): Constraints {
  return {
    ...constraints,
    viewport3d
  };
}

export function addProxyObject(state: Viewport3DState, type: Viewport3DProxyType, source: Viewport3DProxyObject["source"] = "user"): Viewport3DState {
  const preset = proxyPresets[type];
  const id = `${type}_${Date.now().toString(36)}_${state.proxy_objects.length + 1}`;
  const proxy: Viewport3DProxyObject = {
    ...preset,
    id,
    source,
    locked: false
  };
  return {
    ...state,
    proxy_objects: [...state.proxy_objects, proxy],
    selected_proxy_id: id
  };
}

export function addLlmScaleCandidates(state: Viewport3DState, guidance: LlmGuidance | null): Viewport3DState {
  if (!guidance?.scale_candidates?.length) return state;
  const existingLabels = new Set(state.proxy_objects.map((proxy) => proxy.label.toLowerCase()));
  let next = state;
  for (const candidate of guidance.scale_candidates.slice(0, 6)) {
    const label = candidate.trim();
    if (!label || existingLabels.has(label.toLowerCase())) continue;
    const type = /person|human|silhouette/i.test(label)
      ? "person_card"
      : /corridor|tunnel|wall|ceiling/i.test(label)
        ? "corridor"
        : "box";
    next = addProxyObject(next, type, "llm_suggestion");
    const last = next.proxy_objects[next.proxy_objects.length - 1];
    next = updateProxyObject(next, last.id, { label });
    existingLabels.add(label.toLowerCase());
  }
  return next;
}

export function updateProxyObject(
  state: Viewport3DState,
  id: string,
  patch: Partial<Pick<Viewport3DProxyObject, "label" | "position" | "rotation" | "scale" | "locked">>
): Viewport3DState {
  return {
    ...state,
    proxy_objects: state.proxy_objects.map((proxy) => (proxy.id === id ? { ...proxy, ...patch } : proxy))
  };
}

export function selectProxyObject(state: Viewport3DState, id: string | null): Viewport3DState {
  return {
    ...state,
    selected_proxy_id: id
  };
}

export function selectedProxy(state: Viewport3DState): Viewport3DProxyObject | null {
  return state.proxy_objects.find((proxy) => proxy.id === state.selected_proxy_id) ?? null;
}
