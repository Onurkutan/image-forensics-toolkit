/**
 * Every call this client makes, and the one error type it throws.
 *
 * The point of gathering them here is that no other module has to remember
 * how the API says no: a failure arrives as `{"detail": "..."}` with a
 * message the service wrote to be read by whoever sent the request, so it is
 * parsed once and re-thrown as an ApiError whose `message` is that text and
 * whose `status` is the code. A toast can then show any error without knowing
 * which route produced it.
 *
 * Nothing here caches, retries or interprets: a 404 means the session, the
 * tool or the map is not there, and only the caller knows which of the three
 * it just asked for.
 */

/** An error the server described, or a network failure dressed as one. */
export class ApiError extends Error {
  /**
   * @param {number} status HTTP status, or 0 when the request never arrived.
   * @param {string} detail The message to show.
   */
  constructor(status, detail) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * The `detail` of a failed response, falling back to the status line.
 *
 * A body that is not JSON is not an emergency -- a proxy in front of this
 * server may answer 502 in HTML -- so it becomes the generic message rather
 * than a second error on top of the first.
 *
 * @param {Response} response
 * @returns {Promise<string>}
 */
async function detailOf(response) {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") return body.detail;
    if (body && Array.isArray(body.detail)) return JSON.stringify(body.detail);
  } catch (ignored) {
    // Fall through to the status line below.
  }
  return `The server answered ${response.status} ${response.statusText}`.trim();
}

/**
 * One request, with the failures already turned into ApiError.
 *
 * @param {string} path
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (cause) {
    throw new ApiError(0, "The server is not answering. Is it still running?");
  }
  if (!response.ok) throw new ApiError(response.status, await detailOf(response));
  return response.json();
}

/** @returns {Promise<{status: string, version: string}>} */
export function getHealth() {
  return request("/health");
}

/** The tool catalogue, already ordered by category and then by name. */
export function getTools() {
  return request("/tools");
}

/**
 * Upload one image and start a session over it.
 *
 * @param {File} file
 * @returns {Promise<{id: string, name: string, width: number, height: number, format: string}>}
 */
export function createSession(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  return request("/sessions", { method: "POST", body: form });
}

/**
 * Run one tool on a session's image.
 *
 * @param {string} sessionId
 * @param {string} tool
 * @param {Record<string, unknown>} parameters Only what the user set; the
 *   server fills in every default it does not hear about.
 */
export function runTool(sessionId, tool, parameters) {
  return request(`/sessions/${encodeURIComponent(sessionId)}/tools/${encodeURIComponent(tool)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parameters }),
  });
}

/** The fused verdict, or a 404 when this server was started without a fuser. */
export function getFusion(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}/fusion`);
}

/**
 * One tile's URL, with the run it belongs to appended.
 *
 * A tile URL names a tool and a map, never the parameters, so re-running a
 * tool changes what the same URL serves. The run counter rides along as a
 * query parameter the server ignores, which is exactly what stops the
 * browser's cache from painting the previous parameters' map over the new one.
 *
 * @param {string} template A `{z}/{x}/{y}` template from a run's `maps`.
 * @param {number} z Pyramid level, 0 being full resolution.
 * @param {number} x Tile column.
 * @param {number} y Tile row.
 * @param {number} run Which run of the tool this tile belongs to.
 */
export function tileUrl(template, z, x, y, run) {
  const path = template.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y));
  return `${path}?run=${run}`;
}

/**
 * One map tile, decoded.
 *
 * @param {string} url
 * @param {AbortSignal} signal
 * @returns {Promise<ImageBitmap>} The 8-bit grayscale tile, still grayscale.
 */
export async function fetchTile(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new ApiError(response.status, await detailOf(response));
  return createImageBitmap(await response.blob());
}

/**
 * Fetch the report archive and hand it to the browser as a download.
 *
 * Fetched rather than navigated to, so that a failure is a toast like every
 * other failure instead of a page of JSON where the workbench used to be.
 *
 * @param {string} sessionId
 */
export async function downloadReport(sessionId) {
  const url = `/sessions/${encodeURIComponent(sessionId)}/report.zip`;
  const response = await fetch(url);
  if (!response.ok) throw new ApiError(response.status, await detailOf(response));
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filenameFrom(response.headers.get("content-disposition"));
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

/**
 * The file name the server put in Content-Disposition, if it put one there.
 *
 * @param {string | null} header
 * @returns {string}
 */
function filenameFrom(header) {
  const match = header ? /filename="?([^";]+)"?/.exec(header) : null;
  return match ? match[1] : "imgforensics_report.zip";
}
