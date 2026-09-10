/**
 * Everything that builds DOM: the tool tree, the panels, the controls, the toasts.
 *
 * Two rules hold throughout. Nothing is built from a string of markup --
 * every tool name, note, parameter description and detail value comes from
 * the server, and text nodes are how you show such a thing without asking
 * whether it could contain a tag. And nothing here fetches or decides: these
 * functions take data and callbacks and hand back elements, so that what the
 * workbench *does* stays in one file (`app.js`) and what it *looks like*
 * stays in this one.
 */

/**
 * An element, optionally with a class and some text.
 *
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {HTMLElement}
 */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** @param {HTMLElement} node */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/**
 * A number as a person reads it: three decimals, or a short exponent.
 *
 * @param {number} value
 * @returns {string}
 */
export function formatNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value !== 0 && Math.abs(value) < 1e-3) return value.toExponential(2);
  return String(Math.round(value * 1000) / 1000);
}

/**
 * One value of a result's `details`, as one line of text.
 *
 * @param {unknown} value
 * @returns {string}
 */
function formatValue(value) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean" || typeof value === "string") return String(value);
  const text = JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 157)}...` : text;
}

/**
 * A message that goes away: an API error, or a session that has expired.
 *
 * @param {HTMLElement} container
 * @param {string} message
 * @param {{label: string, onClick: () => void}} [action] An offer to fix it.
 */
export function showToast(container, message, action) {
  const toast = el("div", "toast");
  toast.appendChild(el("span", "toast-text", message));
  if (action) {
    const button = el("button", "button small", action.label);
    button.type = "button";
    button.addEventListener("click", () => {
      toast.remove();
      action.onClick();
    });
    toast.appendChild(button);
  }
  const dismiss = el("button", "button small ghost", "Dismiss");
  dismiss.type = "button";
  dismiss.addEventListener("click", () => toast.remove());
  toast.appendChild(dismiss);
  container.appendChild(toast);
  window.setTimeout(() => toast.remove(), 12000);
}

/**
 * A score with the abstain band drawn behind it.
 *
 * The band is a drawing hint -- the label under it always comes from the
 * server -- but it is the difference between "0.58" and "0.58, which is
 * inside the band where this verdict declines to commit".
 *
 * @param {number} score
 * @param {{low: number, high: number}} band
 * @returns {HTMLElement}
 */
export function scoreBar(score, band) {
  const bar = el("div", "score-bar");
  const inside = el("span", "band");
  inside.style.setProperty("--band-left", `${band.low * 100}%`);
  inside.style.setProperty("--band-width", `${(band.high - band.low) * 100}%`);
  const marker = el("span", "marker");
  marker.style.setProperty("--score", `${Math.min(1, Math.max(0, score)) * 100}%`);
  bar.appendChild(inside);
  bar.appendChild(marker);
  return bar;
}

/**
 * The tool tree: one section per category, in the catalogue's own order.
 *
 * @param {HTMLElement} container
 * @param {object[]} tools The catalogue, already sorted by category and name.
 * @param {{selected: string | null, isOpen: (name: string) => boolean}} view
 * @param {{onSelect: (name: string) => void, onToggle: (name: string, open: boolean) => void}} handlers
 */
export function buildToolTree(container, tools, view, handlers) {
  clear(container);
  let category = null;
  let group = null;
  for (const tool of tools) {
    if (tool.category !== category) {
      category = tool.category;
      group = el("div", "group");
      group.appendChild(el("h3", "group-title", category));
      container.appendChild(group);
    }
    group.appendChild(toolRow(tool, view, handlers));
  }
}

/**
 * One row of the tool tree.
 *
 * A tool whose weights are missing is greyed but neither hidden nor disabled:
 * it still runs and still abstains with a reason, and the tooltip says what
 * to fetch to make it do more than that.
 *
 * @param {object} tool
 * @param {{selected: string | null, isOpen: (name: string) => void}} view
 * @param {{onSelect: Function, onToggle: Function}} handlers
 * @returns {HTMLElement}
 */
function toolRow(tool, view, handlers) {
  const row = el("div", "tool");
  if (!tool.installed) {
    row.classList.add("missing");
    row.title = `${tool.note}\n\nWeights are not on this machine, so it will abstain. Fetch them with: imgforensics weights fetch ${tool.name} --accept-license`;
  } else {
    row.title = tool.note;
  }
  if (view.selected === tool.name) row.classList.add("selected");

  const toggle = el("input", "open");
  toggle.type = "checkbox";
  toggle.checked = view.isOpen(tool.name);
  toggle.setAttribute("aria-label", `Show ${tool.display_name} on the canvas`);
  toggle.addEventListener("change", () => handlers.onToggle(tool.name, toggle.checked));

  const name = el("button", "name", tool.display_name);
  name.type = "button";
  name.addEventListener("click", () => handlers.onSelect(tool.name));

  row.appendChild(toggle);
  row.appendChild(name);
  row.appendChild(el("span", "kind", tool.kind));
  return row;
}

/**
 * One canvas panel, with its header and its own overlay controls.
 *
 * The returned object is what the viewer draws: it carries the canvas and the
 * two display choices (overlay or map alone, and how strongly), which are the
 * panel's own rather than the workbench's -- comparing an overlay against a
 * bare map side by side is half of what this canvas is for.
 *
 * @param {{key: string, kind: string, title: string, tool?: string, map?: string, info?: object, run?: number}} spec
 * @param {{onClose: (key: string) => void, onChange: () => void}} handlers
 * @returns {object} The panel object, with `element` and `canvas` on it.
 */
export function createPanel(spec, handlers) {
  const panel = { overlay: true, alpha: 0.6, ...spec };
  const element = el("article", "panel");
  const header = el("header", "panel-head");
  header.appendChild(el("span", "panel-title", spec.title));
  const verdict = el("span", "verdict");
  header.appendChild(verdict);
  element.appendChild(header);

  if (spec.kind === "map") {
    const controls = el("div", "panel-controls");
    const overlay = el("label", "toggle");
    const box = el("input");
    box.type = "checkbox";
    box.checked = panel.overlay;
    box.addEventListener("change", () => {
      panel.overlay = box.checked;
      handlers.onChange();
    });
    overlay.appendChild(box);
    overlay.appendChild(el("span", undefined, "overlay"));
    overlay.title = "Draw the map over the image, or on its own";
    const alpha = el("input", "alpha");
    alpha.type = "range";
    alpha.min = "0.1";
    alpha.max = "1";
    alpha.step = "0.05";
    alpha.value = String(panel.alpha);
    alpha.title = "How strongly the map is painted over the image";
    alpha.addEventListener("input", () => {
      panel.alpha = Number(alpha.value);
      handlers.onChange();
    });
    const close = el("button", "button small ghost close", "×");
    close.type = "button";
    close.title = `Take ${spec.title} off the canvas`;
    close.setAttribute("aria-label", close.title);
    close.addEventListener("click", () => handlers.onClose(spec.key));
    controls.appendChild(overlay);
    controls.appendChild(alpha);
    controls.appendChild(close);
    header.appendChild(controls);
  }

  const canvas = el("canvas", "panel-canvas");
  element.appendChild(canvas);
  panel.element = element;
  panel.canvas = canvas;
  panel.verdict = verdict;
  return panel;
}

/**
 * Put a tool's verdict in its panel header, band and all.
 *
 * @param {object} panel
 * @param {{score: number, label: string} | null} result
 * @param {{low: number, high: number}} band
 * @param {boolean} isView
 */
export function setPanelVerdict(panel, result, band, isView) {
  clear(panel.verdict);
  if (!result) return;
  if (isView) {
    panel.verdict.appendChild(el("span", "muted", "view -- no verdict"));
    return;
  }
  panel.verdict.appendChild(el("span", `label ${result.label}`, result.label));
  panel.verdict.appendChild(el("span", "score", formatNumber(result.score)));
  panel.verdict.appendChild(scoreBar(result.score, band));
}

/**
 * A strip entry: a thumbnail of a map that is open but not in the grid.
 *
 * @param {object} panel
 * @param {(key: string) => void} onPick
 * @returns {HTMLElement}
 */
export function createStripItem(panel, onPick) {
  const item = el("button", "strip-item");
  item.type = "button";
  item.title = `Show ${panel.title} in the grid`;
  const canvas = el("canvas", "strip-canvas");
  item.appendChild(canvas);
  item.appendChild(el("span", "strip-label", panel.title));
  item.addEventListener("click", () => onPick(panel.key));
  panel.canvas = canvas;
  panel.element = item;
  return item;
}

/**
 * The controls for one tool's parameters, from what the catalogue declared.
 *
 * @param {HTMLElement} form
 * @param {object[]} specs `ParameterSpec` entries, in display order.
 * @param {Record<string, unknown>} values What the user has set so far.
 * @param {(name: string, value: unknown) => void} onChange
 */
export function buildParameters(form, specs, values, onChange) {
  clear(form);
  if (!specs.length) {
    form.appendChild(el("p", "muted", "This tool takes no parameters."));
    return;
  }
  for (const spec of specs) {
    const field = el("div", "field");
    const label = el("label", "field-label", spec.name.replace(/_/g, " "));
    label.htmlFor = `parameter-${spec.name}`;
    field.appendChild(label);
    field.appendChild(parameterControl(spec, values, onChange));
    if (spec.description) field.appendChild(el("p", "hint muted", spec.description));
    form.appendChild(field);
  }
}

/**
 * One parameter's control: a slider with a number box, a checkbox, or a select.
 *
 * The slider and the number box are two views of one value, which is why they
 * are built together: dragging is how a forensic parameter is explored, and
 * typing is how it is reproduced afterwards.
 *
 * @param {object} spec
 * @param {Record<string, unknown>} values
 * @param {(name: string, value: unknown) => void} onChange
 * @returns {HTMLElement}
 */
function parameterControl(spec, values, onChange) {
  const current = values[spec.name] !== undefined ? values[spec.name] : spec.default;
  if (spec.kind === "bool") {
    const box = el("input", "control");
    box.type = "checkbox";
    box.id = `parameter-${spec.name}`;
    box.checked = Boolean(current);
    box.addEventListener("change", () => onChange(spec.name, box.checked));
    return box;
  }
  if (spec.kind === "choice") {
    const select = el("select", "control");
    select.id = `parameter-${spec.name}`;
    for (const choice of spec.choices) {
      const option = el("option", undefined, choice);
      option.value = choice;
      select.appendChild(option);
    }
    select.value = String(current);
    select.addEventListener("change", () => onChange(spec.name, select.value));
    return select;
  }

  const step = spec.step || (spec.kind === "int" ? 1 : 0.01);
  const minimum = spec.minimum !== null && spec.minimum !== undefined ? spec.minimum : 0;
  const maximum = spec.maximum !== null && spec.maximum !== undefined ? spec.maximum : 100;
  const wrap = el("div", "number");
  const slider = el("input", "control slider");
  slider.type = "range";
  slider.id = `parameter-${spec.name}`;
  const box = el("input", "control box");
  box.type = "number";
  for (const control of [slider, box]) {
    control.min = String(minimum);
    control.max = String(maximum);
    control.step = String(step);
    control.value = String(current);
  }
  const push = (source, other) => {
    other.value = source.value;
    const value = spec.kind === "int" ? Math.round(Number(source.value)) : Number(source.value);
    if (Number.isFinite(value)) onChange(spec.name, value);
  };
  slider.addEventListener("input", () => push(slider, box));
  box.addEventListener("change", () => push(box, slider));
  wrap.appendChild(slider);
  wrap.appendChild(box);
  return wrap;
}

/**
 * A run's numbers: the verdict, how long it took, and its details.
 *
 * @param {HTMLElement} container
 * @param {object | null} result
 * @param {{low: number, high: number}} band
 * @param {boolean} isView Views claim no verdict, so theirs is not shown.
 */
export function renderResult(container, result, band, isView) {
  clear(container);
  if (!result) return;
  const card = el("div", "card");
  const head = el("div", "verdict big");
  if (isView) {
    head.appendChild(el("span", "muted", "view -- no verdict"));
  } else {
    head.appendChild(el("span", `label ${result.label}`, result.label));
    head.appendChild(el("span", "score", formatNumber(result.score)));
    head.appendChild(scoreBar(result.score, band));
  }
  card.appendChild(head);
  if (typeof result.elapsed_ms === "number") {
    card.appendChild(el("p", "muted", `${Math.round(result.elapsed_ms)} ms`));
  }
  card.appendChild(detailsTable(result.details || {}));
  container.appendChild(card);
}

/**
 * A result's `details` as a key/value table.
 *
 * @param {Record<string, unknown>} details
 * @returns {HTMLElement}
 */
function detailsTable(details) {
  const table = el("table", "details");
  const body = el("tbody");
  for (const [key, value] of Object.entries(details)) {
    const row = el("tr");
    row.appendChild(el("th", undefined, key));
    row.appendChild(el("td", undefined, formatValue(value)));
    body.appendChild(row);
  }
  table.appendChild(body);
  return table;
}

/**
 * The fused verdict card: the probability, its band, and who moved it.
 *
 * @param {HTMLElement} section
 * @param {object} payload The `/fusion` body.
 */
export function renderFusion(section, payload) {
  clear(section);
  section.hidden = false;
  section.appendChild(el("h3", undefined, "Fused verdict"));
  const head = el("div", "verdict big");
  head.appendChild(el("span", `label ${payload.label}`, payload.label));
  head.appendChild(el("span", "score", formatNumber(payload.probability)));
  head.appendChild(scoreBar(payload.probability, payload.band));
  section.appendChild(head);
  section.appendChild(
    el("p", "muted", `abstains between ${formatNumber(payload.band.low)} and ${formatNumber(payload.band.high)}`),
  );
  const list = el("ul", "contributions");
  for (const item of payload.contributions) {
    const entry = el("li", item.present ? "contribution" : "contribution absent");
    entry.title = item.note || "";
    entry.appendChild(el("span", "contribution-name", item.detector));
    entry.appendChild(
      el("span", "contribution-score", item.present ? formatNumber(item.score) : "did not run"),
    );
    const weight = el("span", "contribution-bar");
    weight.style.setProperty("--weight", `${Math.min(100, Math.abs(item.contribution) * 40)}%`);
    weight.classList.add(item.contribution >= 0 ? "towards-fake" : "towards-real");
    entry.appendChild(weight);
    list.appendChild(entry);
  }
  section.appendChild(list);
}
