/**
 * One transform, many panels: the shared canvas the workbench is built around.
 *
 * Comparing two forensic maps means looking at the same pixels in both, so
 * there is exactly one pan/zoom transform here and every panel draws from it.
 * A panel is a canvas plus a source -- the original image, or one map of one
 * tool -- and the Viewer owns three things the panels share: that transform,
 * the tile cache, and the scheduler that decides which tiles are worth
 * fetching right now.
 *
 * The tile rules, in one place because they are easy to get subtly wrong:
 *
 * - **The level follows the zoom.** A map is a pyramid whose level 0 is full
 *   resolution; the level drawn is the coarsest one whose own scale does not
 *   exceed the scale it will be drawn at, so zooming out never fetches
 *   hundreds of tiles no pixel of which survives the downscale.
 * - **A tile is coloured once.** The server sends grayscale PNG; the ramp is
 *   applied when the tile arrives and the cache holds the coloured bitmap, so
 *   panning is `drawImage` and nothing else.
 * - **Only visible tiles stay wanted.** Every draw records what it wanted;
 *   anything still in flight that the last draw did not want is aborted,
 *   which is what keeps a fast zoom from queueing a level the user has
 *   already left behind.
 */

import { fetchTile, tileUrl } from "./api.js";

/** Blue -> cyan -> green -> yellow -> red: the ramp the report builder writes its overlays with. */
const COLORMAP_STOPS = [
  [0.0, 0, 0, 255],
  [0.25, 0, 255, 255],
  [0.5, 0, 255, 0],
  [0.75, 255, 255, 0],
  [1.0, 255, 0, 0],
];

/** How far in and out the transform may go, as screen pixels per image pixel. */
const MIN_SCALE = 0.02;
const MAX_SCALE = 32;

/** Coloured tiles kept before the oldest are dropped; a few screens' worth. */
const MAX_CACHED_TILES = 320;

/**
 * A 256-entry RGBA lookup table over the ramp above.
 *
 * @returns {Uint8ClampedArray}
 */
function buildColormap() {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let index = 0; index < 256; index += 1) {
    const value = index / 255;
    let stop = 0;
    while (stop < COLORMAP_STOPS.length - 2 && value > COLORMAP_STOPS[stop + 1][0]) stop += 1;
    const [from, red, green, blue] = COLORMAP_STOPS[stop];
    const [to, red2, green2, blue2] = COLORMAP_STOPS[stop + 1];
    const weight = to === from ? 0 : (value - from) / (to - from);
    lut[index * 4] = red + (red2 - red) * weight;
    lut[index * 4 + 1] = green + (green2 - green) * weight;
    lut[index * 4 + 2] = blue + (blue2 - blue) * weight;
    lut[index * 4 + 3] = 255;
  }
  return lut;
}

const COLORMAP = buildColormap();

/**
 * Turn one grayscale tile into a coloured one, and give the grayscale back.
 *
 * @param {ImageBitmap} bitmap
 * @returns {Promise<ImageBitmap>}
 */
async function colorize(bitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = image.data;
  for (let index = 0; index < pixels.length; index += 4) {
    // A grayscale PNG decodes with all three channels equal, so one is the value.
    const entry = pixels[index] * 4;
    pixels[index] = COLORMAP[entry];
    pixels[index + 1] = COLORMAP[entry + 1];
    pixels[index + 2] = COLORMAP[entry + 2];
    pixels[index + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return createImageBitmap(canvas);
}

/**
 * Match a canvas's backing store to its box, and return the device ratio.
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {number}
 */
function sizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

/** @param {number} value @param {number} low @param {number} high */
function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * The cache key of one tile, run included so a re-run cannot be shown stale.
 *
 * @param {object} panel
 * @param {number} level
 * @param {number} tileX
 * @param {number} tileY
 * @returns {string}
 */
function tileKey(panel, level, tileX, tileY) {
  return `${panel.tool}|${panel.run}|${panel.map}|${level}|${tileX}|${tileY}`;
}

/**
 * The shared view over one image and the maps opened on it.
 *
 * Panels are plain objects the caller owns and this class only reads:
 * `{key, canvas, kind: "image" | "map", tool, map, info, run, overlay, alpha}`,
 * where `info` is the `maps` entry a tool run returned and `run` counts that
 * tool's runs, so a re-run invalidates its tiles without touching anyone
 * else's.
 */
export class Viewer {
  /**
   * @param {HTMLElement} root The element the panels live in; it carries the
   *   pointer and wheel listeners, so a panel added later needs no wiring.
   * @param {{onError?: (error: Error) => void}} [options]
   */
  constructor(root, options = {}) {
    this.root = root;
    this.onError = options.onError || (() => {});
    this.base = null;
    this.panels = [];
    this.thumbs = [];
    this.transform = { scale: 1, x: 0, y: 0 };
    this.cache = new Map();
    this.pending = new Map();
    this.wanted = new Set();
    this.frame = 0;
    this.drag = null;
    this.bindPointer();
  }

  /**
   * Show a new image: the base every panel draws, and what the transform fits.
   *
   * @param {ImageBitmap | null} bitmap
   */
  setBase(bitmap) {
    if (this.base) this.base.close();
    this.base = bitmap;
    this.dropTiles(() => true);
    if (bitmap) this.reset();
    else this.schedule();
  }

  /**
   * @param {object[]} panels The panels in the grid, in layout order.
   * @param {object[]} [thumbs] Map panels in the strip, drawn whole and small
   *   rather than through the shared transform: a strip is for finding a map,
   *   not for reading one.
   */
  setPanels(panels, thumbs = []) {
    this.panels = panels;
    this.thumbs = thumbs;
    this.schedule();
  }

  /**
   * Forget one tool's cached tiles from before its latest run.
   *
   * @param {string} tool
   * @param {number} run The run whose tiles are still good.
   */
  forgetTool(tool, run) {
    this.dropTiles((key) => key.startsWith(`${tool}|`) && !key.startsWith(`${tool}|${run}|`));
  }

  /** Fit the image in a panel, centred: the view every reset returns to. */
  reset() {
    const canvas = this.panels.length ? this.panels[0].canvas : null;
    if (!this.base || !canvas || !canvas.clientWidth) return;
    const scale = Math.min(
      canvas.clientWidth / this.base.width,
      canvas.clientHeight / this.base.height,
    );
    this.transform.scale = clamp(scale * 0.98, MIN_SCALE, MAX_SCALE);
    this.transform.x = (canvas.clientWidth - this.base.width * this.transform.scale) / 2;
    this.transform.y = (canvas.clientHeight - this.base.height * this.transform.scale) / 2;
    this.schedule();
  }

  /**
   * Zoom about a point on a panel, so what is under the cursor stays there.
   *
   * @param {number} factor
   * @param {HTMLCanvasElement} canvas
   * @param {number} clientX
   * @param {number} clientY
   */
  zoomAt(factor, canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const pointX = clientX - rect.left;
    const pointY = clientY - rect.top;
    const transform = this.transform;
    const scale = clamp(transform.scale * factor, MIN_SCALE, MAX_SCALE);
    transform.x = pointX - ((pointX - transform.x) / transform.scale) * scale;
    transform.y = pointY - ((pointY - transform.y) / transform.scale) * scale;
    transform.scale = scale;
    this.schedule();
  }

  /** Zoom about the middle of the first panel -- what the keyboard does. */
  zoomBy(factor) {
    const canvas = this.panels.length ? this.panels[0].canvas : null;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    this.zoomAt(factor, canvas, rect.left + rect.width / 2, rect.top + rect.height / 2);
  }

  /** @param {number} dx @param {number} dy */
  panBy(dx, dy) {
    this.transform.x += dx;
    this.transform.y += dy;
    this.schedule();
  }

  /** Ask for one redraw before the next frame, however many callers asked. */
  schedule() {
    if (this.frame) return;
    this.frame = window.requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  /** Draw every panel, then drop the fetches the drawing did not want. */
  draw() {
    this.wanted = new Set();
    for (const panel of this.panels) this.drawPanel(panel);
    for (const panel of this.thumbs) this.drawThumbnail(panel);
    for (const [key, controller] of this.pending) {
      if (!this.wanted.has(key)) {
        controller.abort();
        this.pending.delete(key);
      }
    }
  }

  /** @param {object} panel */
  drawPanel(panel) {
    const ratio = sizeCanvas(panel.canvas);
    const context = panel.canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = panel.canvas.width / ratio;
    const height = panel.canvas.height / ratio;
    context.clearRect(0, 0, width, height);
    if (!this.base) return;
    const { scale, x, y } = this.transform;
    // Smoothed while shrinking, blocky while enlarging: past 1:1 the user is
    // looking at individual pixels and interpolation would invent some.
    context.imageSmoothingEnabled = scale < 1;
    if (panel.kind === "image" || panel.overlay) {
      context.drawImage(this.base, x, y, this.base.width * scale, this.base.height * scale);
    }
    if (panel.kind === "map") this.drawMap(context, panel, width, height);
  }

  /**
   * The tiles of one map, at the level this zoom deserves.
   *
   * @param {CanvasRenderingContext2D} context
   * @param {object} panel
   * @param {number} width Panel width in CSS pixels.
   * @param {number} height Panel height in CSS pixels.
   */
  drawMap(context, panel, width, height) {
    const info = panel.info;
    const level = this.levelFor(panel);
    const [levelHeight, levelWidth] = info.shapes[level];
    const size = info.tile_size;
    const { x, y } = this.transform;
    const perTileX = this.tileStep(panel, level, 0);
    const perTileY = this.tileStep(panel, level, 1);
    context.save();
    // Clipped to the image: a tile is zero-padded past the edge of its level,
    // and zero is a colour on this ramp, not nothing.
    context.beginPath();
    context.rect(x, y, this.base.width * this.transform.scale, this.base.height * this.transform.scale);
    context.clip();
    context.globalAlpha = panel.overlay ? panel.alpha : 1;
    // The coarsest level is a single tile and is always asked for first, so it
    // stands in for the fine tiles still in flight instead of a hole.
    if (level !== info.levels - 1) this.drawTile(context, panel, info.levels - 1, 0, 0);
    const firstX = Math.max(0, Math.floor(-x / perTileX));
    const lastX = Math.min(Math.ceil(levelWidth / size) - 1, Math.floor((width - x) / perTileX));
    const firstY = Math.max(0, Math.floor(-y / perTileY));
    const lastY = Math.min(Math.ceil(levelHeight / size) - 1, Math.floor((height - y) / perTileY));
    for (let tileY = firstY; tileY <= lastY; tileY += 1) {
      for (let tileX = firstX; tileX <= lastX; tileX += 1) {
        this.drawTile(context, panel, level, tileX, tileY);
      }
    }
    context.restore();
  }

  /**
   * One whole map, coarsest level, fitted into a strip thumbnail.
   *
   * @param {object} panel
   */
  drawThumbnail(panel) {
    const ratio = sizeCanvas(panel.canvas);
    const context = panel.canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = panel.canvas.width / ratio;
    const height = panel.canvas.height / ratio;
    context.clearRect(0, 0, width, height);
    const level = panel.info.levels - 1;
    const key = tileKey(panel, level, 0, 0);
    this.wanted.add(key);
    const bitmap = this.cache.get(key);
    if (!bitmap) {
      this.request(panel, level, 0, 0, key);
      return;
    }
    const [mapHeight, mapWidth] = panel.info.shapes[level];
    const scale = Math.min(width / mapWidth, height / mapHeight);
    context.drawImage(
      bitmap,
      0,
      0,
      mapWidth,
      mapHeight,
      (width - mapWidth * scale) / 2,
      (height - mapHeight * scale) / 2,
      mapWidth * scale,
      mapHeight * scale,
    );
  }

  /**
   * One tile if it is cached, a fetch for it if it is not.
   *
   * @param {CanvasRenderingContext2D} context
   * @param {object} panel
   * @param {number} level
   * @param {number} tileX
   * @param {number} tileY
   */
  drawTile(context, panel, level, tileX, tileY) {
    const key = tileKey(panel, level, tileX, tileY);
    this.wanted.add(key);
    const bitmap = this.cache.get(key);
    if (!bitmap) {
      this.request(panel, level, tileX, tileY, key);
      return;
    }
    const [levelHeight, levelWidth] = panel.info.shapes[level];
    const size = panel.info.tile_size;
    const sourceWidth = Math.min(size, levelWidth - tileX * size);
    const sourceHeight = Math.min(size, levelHeight - tileY * size);
    if (sourceWidth <= 0 || sourceHeight <= 0) return;
    const perPixelX = this.tileStep(panel, level, 0) / size;
    const perPixelY = this.tileStep(panel, level, 1) / size;
    const left = this.transform.x + tileX * size * perPixelX;
    const top = this.transform.y + tileY * size * perPixelY;
    // Snapped outwards to whole pixels, or a fractional scale leaves hairline
    // gaps between neighbouring tiles.
    const drawX = Math.floor(left);
    const drawY = Math.floor(top);
    context.drawImage(
      bitmap,
      0,
      0,
      sourceWidth,
      sourceHeight,
      drawX,
      drawY,
      Math.ceil(left + sourceWidth * perPixelX) - drawX,
      Math.ceil(top + sourceHeight * perPixelY) - drawY,
    );
  }

  /**
   * How many screen pixels one tile of a level covers, across (0) or down (1).
   *
   * A map is shaped like the image it came from, so a level pixel is
   * `image size / level size` image pixels wide, and the transform turns that
   * into screen pixels. Doing it per axis costs nothing and keeps a map that
   * was resampled to a slightly different shape registered with the original.
   *
   * @param {object} panel
   * @param {number} level
   * @param {number} axis
   */
  tileStep(panel, level, axis) {
    const levelSize = panel.info.shapes[level][axis === 0 ? 1 : 0];
    const imageSize = axis === 0 ? this.base.width : this.base.height;
    return panel.info.tile_size * (imageSize / levelSize) * this.transform.scale;
  }

  /**
   * The pyramid level to draw at this zoom.
   *
   * The coarsest level whose own scale does not exceed the scale it would be
   * drawn at: one step finer would deliver four times the tiles for pixels
   * the downscale throws away.
   *
   * @param {object} panel
   * @returns {number}
   */
  levelFor(panel) {
    const full = panel.info.shapes[0][1];
    const screenScale = (this.base.width / full) * this.transform.scale;
    const level = Math.ceil(Math.log2(1 / Math.max(screenScale, 1e-6)));
    return clamp(level, 0, panel.info.levels - 1);
  }

  /**
   * Fetch and colour one tile, unless it is already on its way.
   *
   * @param {object} panel
   * @param {number} level
   * @param {number} tileX
   * @param {number} tileY
   * @param {string} key
   */
  request(panel, level, tileX, tileY, key) {
    if (this.pending.has(key)) return;
    const controller = new AbortController();
    this.pending.set(key, controller);
    fetchTile(tileUrl(panel.info.tile_url, level, tileX, tileY, panel.run), controller.signal)
      .then(colorize)
      .then((bitmap) => {
        this.pending.delete(key);
        this.store(key, bitmap);
        this.schedule();
      })
      .catch((error) => {
        this.pending.delete(key);
        if (error.name !== "AbortError") this.onError(error);
      });
  }

  /**
   * Keep one coloured tile, dropping the oldest once the cache is full.
   *
   * @param {string} key
   * @param {ImageBitmap} bitmap
   */
  store(key, bitmap) {
    this.cache.set(key, bitmap);
    while (this.cache.size > MAX_CACHED_TILES) {
      const oldest = this.cache.keys().next().value;
      this.cache.get(oldest).close();
      this.cache.delete(oldest);
    }
  }

  /**
   * Drop cached tiles and abort fetches whose key a predicate accepts.
   *
   * @param {(key: string) => boolean} matches
   */
  dropTiles(matches) {
    for (const [key, bitmap] of this.cache) {
      if (matches(key)) {
        bitmap.close();
        this.cache.delete(key);
      }
    }
    for (const [key, controller] of this.pending) {
      if (matches(key)) {
        controller.abort();
        this.pending.delete(key);
      }
    }
  }

  /** Wheel, drag and double-click, once, on the element the panels sit in. */
  bindPointer() {
    this.root.addEventListener(
      "wheel",
      (event) => {
        const canvas = event.target.closest("canvas");
        if (!canvas) return;
        event.preventDefault();
        this.zoomAt(Math.exp(-event.deltaY * 0.002), canvas, event.clientX, event.clientY);
      },
      { passive: false },
    );
    this.root.addEventListener("pointerdown", (event) => {
      const canvas = event.target.closest("canvas");
      if (!canvas || event.button !== 0) return;
      canvas.setPointerCapture(event.pointerId);
      this.drag = { x: event.clientX, y: event.clientY, canvas };
    });
    this.root.addEventListener("pointermove", (event) => {
      if (!this.drag) return;
      this.panBy(event.clientX - this.drag.x, event.clientY - this.drag.y);
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
    });
    const release = () => {
      this.drag = null;
    };
    this.root.addEventListener("pointerup", release);
    this.root.addEventListener("pointercancel", release);
    this.root.addEventListener("dblclick", (event) => {
      if (event.target.closest("canvas")) this.reset();
    });
  }
}
