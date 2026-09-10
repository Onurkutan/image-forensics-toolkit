/**
 * The workbench itself: what the buttons do, and the one object that remembers it.
 *
 * `api.js` knows how to ask the server, `viewer.js` knows how to draw, `ui.js`
 * knows how to build the controls; this file is the only place that decides.
 * All of its memory lives in the `state` object below -- there is no other
 * mutable module-level value -- so "what is the workbench showing" is one
 * thing to read, and every render is a function of it.
 *
 * The rules worth stating, because they are choices and not consequences:
 *
 * - **The original image never comes back from the server.** It is drawn from
 *   the file the user picked, so the upload travels once and panning costs
 *   nothing. That is also why the page's CSP allows `blob:` images.
 * - **A tool runs when it is opened and whenever a parameter settles**, 400 ms
 *   after the last change, because a slider drag is one intention and not
 *   forty. The run counter it bumps is what makes the previous run's tiles
 *   unreachable.
 * - **A failure is a toast, never a blank panel.** The only failure with a way
 *   out is an expired session, and that toast carries the way out.
 */

import { ApiError, createSession, downloadReport, getFusion, getHealth, getTools, runTool } from "./api.js";
import * as ui from "./ui.js";
import { Viewer } from "./viewer.js";

/** Panels drawn side by side; more than four of them stop being comparable. */
const GRID_MAX = 4;

/** How many of those the maps may have: the original always takes one. */
const GRID_MAPS = GRID_MAX - 1;

/** How long a parameter has to stay still before its tool re-runs. */
const RERUN_DELAY_MS = 400;

/**
 * The band drawn behind a tool's own score when no fuser offers a better one:
 * the thresholds `label_from_score` uses. It is a drawing hint -- every label
 * on this page is the server's -- but a score means little without it.
 */
const DEFAULT_BAND = { low: 0.35, high: 0.65 };

const state = {
  version: "",
  tools: [],
  byName: new Map(),
  session: null,
  selected: null,
  opened: [],
  order: [],
  results: new Map(),
  values: new Map(),
  runs: new Map(),
  display: new Map(),
  band: DEFAULT_BAND,
  noFuser: false,
  rerun: 0,
};

const dom = {
  version: document.getElementById("version"),
  facts: document.getElementById("session-facts"),
  file: document.getElementById("file"),
  tree: document.getElementById("tool-tree"),
  area: document.getElementById("canvas-area"),
  hint: document.getElementById("drop-hint"),
  grid: document.getElementById("panel-grid"),
  strip: document.getElementById("panel-strip"),
  title: document.getElementById("tool-title"),
  note: document.getElementById("tool-note"),
  parameters: document.getElementById("parameters"),
  run: document.getElementById("run"),
  runState: document.getElementById("run-state"),
  result: document.getElementById("result"),
  fusion: document.getElementById("fusion"),
  report: document.getElementById("report"),
  toasts: document.getElementById("toasts"),
};

const viewer = new Viewer(dom.grid, { onError: report });

/**
 * Run something that talks to the server, and turn any failure into a toast.
 *
 * @param {() => Promise<unknown>} action
 */
async function guard(action) {
  try {
    await action();
  } catch (error) {
    report(error);
  }
}

/**
 * Show one failure, with a way out when there is one.
 *
 * A 404 while a session is open can only be that session: every tool this
 * client runs came from the catalogue, so the only name it can have got wrong
 * is the one the server forgot.
 *
 * @param {Error} error
 */
function report(error) {
  if (error instanceof ApiError && error.status === 404 && state.session) {
    closeSession();
    ui.showToast(dom.toasts, "This session has expired. Open the image again to carry on.", {
      label: "Open image",
      onClick: () => dom.file.click(),
    });
    return;
  }
  ui.showToast(dom.toasts, error.message || String(error));
}

/** Forget the session, keeping the catalogue and the tools the user opened. */
function closeSession() {
  state.session = null;
  state.results.clear();
  state.runs.clear();
  state.noFuser = false;
  viewer.setBase(null);
  dom.facts.textContent = "";
  dom.fusion.hidden = true;
  dom.report.disabled = true;
  clearInterface();
}

/** Redraw the parts of the page that depend on the session or the selection. */
function clearInterface() {
  layout();
  renderTree();
  renderInspector();
}

// --------------------------------------------------------------------------
// The image
// --------------------------------------------------------------------------

/**
 * Upload one image, start a session over it, and show it.
 *
 * @param {File} file
 */
async function openImage(file) {
  const session = await createSession(file);
  const bitmap = await createImageBitmap(file);
  state.session = session;
  state.results.clear();
  state.runs.clear();
  state.display.clear();
  state.noFuser = false;
  dom.facts.textContent = `${session.name} -- ${session.width}x${session.height} ${session.format}`;
  dom.report.disabled = false;
  dom.fusion.hidden = true;
  layout();
  // After the layout, so that fitting the image measures a panel that exists.
  viewer.setBase(bitmap);
  renderTree();
  renderInspector();
  for (const tool of state.opened) await guard(() => run(tool));
}

// --------------------------------------------------------------------------
// Tools
// --------------------------------------------------------------------------

/** Draw the tool tree from the catalogue and what is open right now. */
function renderTree() {
  ui.buildToolTree(
    dom.tree,
    state.tools,
    { selected: state.selected, isOpen: (name) => state.opened.includes(name) },
    {
      onSelect: (name) => {
        state.selected = name;
        renderTree();
        renderInspector();
      },
      onToggle: (name, open) => guard(() => toggle(name, open)),
    },
  );
}

/**
 * Put a tool's maps on the canvas, or take them off.
 *
 * @param {string} name
 * @param {boolean} open
 */
async function toggle(name, open) {
  if (!open) {
    state.opened = state.opened.filter((tool) => tool !== name);
    layout();
    return;
  }
  if (!state.session) {
    ui.showToast(dom.toasts, "Open an image first, then tick the tools to run on it.");
    renderTree();
    return;
  }
  state.opened.push(name);
  state.selected = name;
  renderTree();
  renderInspector();
  if (!state.results.has(name)) await run(name);
  else layout();
  const result = state.results.get(name);
  if (result && !Object.keys(result.maps).length) {
    state.opened = state.opened.filter((tool) => tool !== name);
    ui.showToast(dom.toasts, `${label(name)} draws no map; its numbers are on the right.`);
    renderTree();
    layout();
  }
}

/**
 * Run one tool with what the user has set, and show what came back.
 *
 * @param {string} name
 */
async function run(name) {
  if (!state.session) return;
  dom.runState.textContent = "running...";
  try {
    const result = await runTool(state.session.id, name, state.values.get(name) || {});
    state.results.set(name, result);
    const count = (state.runs.get(name) || 0) + 1;
    state.runs.set(name, count);
    viewer.forgetTool(name, count);
    dom.runState.textContent = `${Math.round(result.elapsed_ms)} ms`;
    layout();
    if (state.selected === name) renderResult();
    await refreshFusion();
  } catch (error) {
    dom.runState.textContent = "";
    throw error;
  }
}

/** Re-run a tool once its parameters have stopped moving. */
function scheduleRun(name) {
  window.clearTimeout(state.rerun);
  state.rerun = window.setTimeout(() => guard(() => run(name)), RERUN_DELAY_MS);
}

/** @param {string} name @returns {string} The tool's display name. */
function label(name) {
  const spec = state.byName.get(name);
  return spec ? spec.display_name : name;
}

// --------------------------------------------------------------------------
// The canvas area
// --------------------------------------------------------------------------

/**
 * Every map panel that should exist, in the order the user opened them.
 *
 * @returns {object[]}
 */
function mapPanels() {
  const specs = [];
  for (const tool of state.opened) {
    const result = state.results.get(tool);
    if (!result) continue;
    for (const [map, info] of Object.entries(result.maps)) {
      specs.push({
        key: `${tool}/${map}`,
        kind: "map",
        tool,
        map,
        info,
        run: state.runs.get(tool),
        title: `${label(tool)} -- ${map}`,
      });
    }
  }
  return specs;
}

/**
 * Rebuild the grid and the strip.
 *
 * The panel elements are made afresh each time rather than moved about: the
 * tiles they draw are cached in the viewer, so a rebuilt canvas costs one
 * redraw and no requests, and the alternative -- keeping elements alive
 * across every open, close and swap -- is a second piece of bookkeeping to
 * get wrong.
 */
function layout() {
  const specs = mapPanels();
  const keys = specs.map((spec) => spec.key);
  state.order = state.order.filter((key) => keys.includes(key));
  for (const key of keys) if (!state.order.includes(key)) state.order.push(key);
  const ordered = state.order.map((key) => specs.find((spec) => spec.key === key));

  ui.clear(dom.grid);
  ui.clear(dom.strip);
  dom.hint.hidden = Boolean(state.session);
  if (!state.session) {
    viewer.setPanels([], []);
    return;
  }

  const all = [{ key: "original", kind: "image", title: state.session.name }, ...ordered];
  const panels = all.slice(0, GRID_MAX).map((spec) => {
    const panel = ui.createPanel({ ...spec, ...(state.display.get(spec.key) || {}) }, handlers);
    dom.grid.appendChild(panel.element);
    if (spec.kind === "map") {
      const result = state.results.get(spec.tool);
      const kind = state.byName.get(spec.tool);
      ui.setPanelVerdict(panel, result, state.band, kind ? kind.kind === "view" : false);
    }
    return panel;
  });
  const thumbs = all.slice(GRID_MAX).map((spec) => {
    const panel = { ...spec };
    dom.strip.appendChild(ui.createStripItem(panel, focusPanel));
    return panel;
  });
  dom.strip.hidden = thumbs.length === 0;
  dom.grid.className = `panel-grid count-${Math.max(1, panels.length)}`;
  viewer.setPanels(panels, thumbs);
}

/** What a panel's own controls do: remember the choice, and redraw. */
const handlers = {
  onClose: (key) => {
    const tool = key.split("/")[0];
    guard(() => toggle(tool, false));
  },
  onChange: () => {
    for (const panel of viewer.panels) {
      state.display.set(panel.key, { overlay: panel.overlay, alpha: panel.alpha });
    }
    viewer.schedule();
  },
};

/**
 * Swap a strip thumbnail into the last map slot of the grid.
 *
 * @param {string} key
 */
function focusPanel(key) {
  const index = state.order.indexOf(key);
  if (index < 0 || index < GRID_MAPS) return;
  const last = GRID_MAPS - 1;
  const swapped = state.order[last];
  state.order[last] = key;
  state.order[index] = swapped;
  layout();
}

// --------------------------------------------------------------------------
// The tool panel
// --------------------------------------------------------------------------

/** The selected tool's note, controls and last result. */
function renderInspector() {
  const spec = state.selected ? state.byName.get(state.selected) : null;
  dom.title.textContent = spec ? spec.display_name : "No tool selected";
  dom.note.textContent = spec ? spec.note : "Pick a tool on the left to see what it reads.";
  dom.run.disabled = !spec || !state.session;
  if (!spec) {
    ui.clear(dom.parameters);
    ui.clear(dom.result);
    return;
  }
  ui.buildParameters(
    dom.parameters,
    spec.parameters,
    state.values.get(spec.name) || {},
    (name, value) => {
      const values = { ...(state.values.get(spec.name) || {}) };
      values[name] = value;
      state.values.set(spec.name, values);
      if (state.session) scheduleRun(spec.name);
    },
  );
  renderResult();
}

/**
 * The result half of the tool panel, on its own.
 *
 * Kept apart from the controls above it because a re-run must not rebuild the
 * slider that caused it -- doing so would take the drag out of the user's
 * hand halfway through.
 */
function renderResult() {
  const spec = state.selected ? state.byName.get(state.selected) : null;
  if (!spec) return;
  ui.renderResult(dom.result, state.results.get(spec.name) || null, state.band, spec.kind === "view");
}

/** The fused verdict, when this server has a fuser to fuse with. */
async function refreshFusion() {
  if (!state.session || state.noFuser) return;
  try {
    const payload = await getFusion(state.session.id);
    state.band = payload.band;
    ui.renderFusion(dom.fusion, payload);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      // Either there is no fuser or the session is gone; both mean there is
      // nothing to show, and the next call the user makes tells them which.
      state.noFuser = true;
      dom.fusion.hidden = true;
      return;
    }
    throw error;
  }
}

// --------------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------------

/**
 * Pan and zoom from the keyboard, unless the user is typing into something.
 *
 * @param {KeyboardEvent} event
 */
function onKey(event) {
  const target = event.target;
  if (target instanceof HTMLElement && target.matches("input, select, textarea")) return;
  const step = 60;
  const moves = {
    ArrowLeft: [step, 0],
    ArrowRight: [-step, 0],
    ArrowUp: [0, step],
    ArrowDown: [0, -step],
  };
  if (event.key === "+" || event.key === "=") viewer.zoomBy(1.25);
  else if (event.key === "-" || event.key === "_") viewer.zoomBy(0.8);
  else if (event.key === "0") viewer.reset();
  else if (moves[event.key]) viewer.panBy(moves[event.key][0], moves[event.key][1]);
  else return;
  event.preventDefault();
}

function wire() {
  dom.file.addEventListener("change", () => {
    const file = dom.file.files[0];
    if (file) guard(() => openImage(file));
  });
  dom.area.addEventListener("dragover", (event) => {
    event.preventDefault();
    dom.area.classList.add("dropping");
  });
  dom.area.addEventListener("dragleave", () => dom.area.classList.remove("dropping"));
  dom.area.addEventListener("drop", (event) => {
    event.preventDefault();
    dom.area.classList.remove("dropping");
    const file = event.dataTransfer.files[0];
    if (file) guard(() => openImage(file));
  });
  dom.run.addEventListener("click", () => {
    if (state.selected) guard(() => run(state.selected));
  });
  dom.report.addEventListener("click", () => {
    if (state.session) guard(() => downloadReport(state.session.id));
  });
  window.addEventListener("resize", () => viewer.schedule());
  document.addEventListener("keydown", onKey);
}

/** Load the catalogue, draw the tree, and wait for an image. */
async function boot() {
  wire();
  await guard(async () => {
    const health = await getHealth();
    state.version = health.version;
    dom.version.textContent = `v${health.version}`;
  });
  await guard(async () => {
    state.tools = await getTools();
    state.byName = new Map(state.tools.map((tool) => [tool.name, tool]));
    renderTree();
  });
}

boot();
