// Shared helpers for all pages: API access, navigation, live updates, small UI utilities.

export const API = "/api/v1";

export async function api(path, { method = "GET", json, form } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (form) {
    opts.body = form;
  }
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not json */ }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.status === 204 ? null : r.json();
}

const PAGES = [
  ["index.html", "Dashboard"],
  ["gallery.html", "Cats"],
  ["events.html", "Events"],
  ["stats.html", "Statistics"],
  ["benchmark.html", "Benchmark"],
  ["settings.html", "Settings"],
];

export function renderNav() {
  const current = location.pathname.split("/").pop() || "index.html";
  const header = document.createElement("header");
  header.className = "topbar";
  header.innerHTML = `
    <a class="brand" href="index.html">🐈 CatAlarm</a>
    <nav>${PAGES.map(([href, label]) =>
      `<a href="${href}" class="${current === href || (current === "camera.html" && href === "index.html") ? "active" : ""}">${label}</a>`
    ).join("")}</nav>
    <span class="live" id="live-indicator"><span class="dot"></span> connecting…</span>`;
  document.body.prepend(header);
  if (!document.getElementById("toast")) {
    const t = document.createElement("div");
    t.id = "toast";
    document.body.append(t);
  }
}

// Server-Sent Events with automatic reconnect; calls onMessage(obj) for every update.
export function live(onMessage) {
  const indicator = () => document.getElementById("live-indicator");
  const set = (cls, text) => {
    const el = indicator();
    if (el) el.innerHTML = `<span class="dot ${cls}"></span> ${text}`;
  };
  let es;
  const connect = () => {
    es = new EventSource(API + "/live");
    es.onopen = () => set("ok", "live");
    es.onmessage = (e) => onMessage(JSON.parse(e.data));
    es.onerror = () => {
      set("bad", "offline");
      es.close();
      setTimeout(connect, 3000);
    };
  };
  connect();
}

let toastTimer;
export function toast(message, error = false) {
  const t = document.getElementById("toast");
  t.textContent = message;
  t.className = "show" + (error ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = ""), error ? 5000 : 2500);
}

// wrap an async handler: errors become a toast instead of an unhandled rejection
export const guard = (fn) => async (...args) => {
  try {
    return await fn(...args);
  } catch (e) {
    console.error(e);
    toast(e.message, true);
  }
};

// Sequential color (one hue): t = 0 (little) .. 1 (a lot). "More" always means more contrast
// against the surface, so the ramp runs light->dark in light mode and dark->bright in dark mode.
const darkScheme = window.matchMedia("(prefers-color-scheme: dark)");
export function seqColor(t) {
  const v = Math.min(1, Math.max(0, t));
  const light = darkScheme.matches ? 22 + v * 46 : 93 - v * 55;
  return { bg: `hsl(25 75% ${light}%)`, fg: light > 55 ? "#000" : "#fff" };
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export function fmtTime(ts, withDate = true) {
  const d = new Date(ts * 1000);
  return withDate ? d.toLocaleString() : d.toLocaleTimeString();
}

export function fmtAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

// YYYY-MM-DD in the browser's local time (matches the server's day grouping)
export function localDay(d = new Date()) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function fmtDuration(seconds) {
  if (seconds == null) return "–";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

// cats are loaded once per page and used to show names/colors for ids
let catsCache;
export async function cats() {
  catsCache ??= api("/gallery/cats");
  return catsCache;
}
export function catName(list, id) {
  return list.find((c) => c.id === id)?.name ?? id;
}
export function catColor(list, id) {
  return list.find((c) => c.id === id)?.color ?? "#999";
}

const STATE_LABELS = { no_motion: "no motion", no_cat: "no cat", unconfirmed: "unconfirmed" };
export function stateLabel(list, state) {
  return STATE_LABELS[state] ?? catName(list, state);
}
export function stateDot(state) {
  if (state === "no_motion") return "";
  if (state === "no_cat") return "warn";
  if (state === "unconfirmed") return "warn";
  return "ok";
}

// ---- cropping dialog (Cropper.js), shared by gallery, events and camera pages ----
// Resolves to a Blob with the cropped image, or null when cancelled.
export function cropDialog(src, { title = "Crop reference image", catsList = [], defaultCat, defaultMode = "day" } = {}) {
  return new Promise((resolve) => {
    const dlg = document.createElement("dialog");
    dlg.innerHTML = `
      <h2>${esc(title)}</h2>
      <p class="small muted">Drag to select only the cat. Tight crops of the head and body work best.</p>
      <div class="crop-area"><img src="${esc(src)}" alt="image to crop"></div>
      <div class="row spread" style="margin-top:1rem">
        <div class="row">
          <select name="cat">${catsList.map((c) =>
            `<option value="${esc(c.id)}" ${c.id === defaultCat ? "selected" : ""}>${esc(c.name)}</option>`).join("")}</select>
          <select name="mode">
            <option value="day" ${defaultMode === "day" ? "selected" : ""}>day</option>
            <option value="night" ${defaultMode === "night" ? "selected" : ""}>night (IR)</option>
          </select>
        </div>
        <div class="row">
          <button type="button" data-act="cancel">Cancel</button>
          <button type="button" class="primary" data-act="save">Add to gallery</button>
        </div>
      </div>`;
    document.body.append(dlg);
    const img = dlg.querySelector("img");
    let cropper;
    img.onload = () => {
      cropper = new window.Cropper(img, { viewMode: 1, autoCropArea: 0.8, background: false });
    };
    const close = (value) => {
      cropper?.destroy();
      dlg.close();
      dlg.remove();
      resolve(value);
    };
    dlg.addEventListener("cancel", () => close(null));
    dlg.querySelector('[data-act="cancel"]').onclick = () => close(null);
    dlg.querySelector('[data-act="save"]').onclick = () => {
      if (!cropper) return;
      const cat = dlg.querySelector('[name="cat"]').value;
      const mode = dlg.querySelector('[name="mode"]').value;
      cropper.getCroppedCanvas({ maxWidth: 640, maxHeight: 640 }).toBlob(
        (blob) => close({ blob, cat, mode }), "image/jpeg", 0.92);
    };
    dlg.showModal();
  });
}

export async function uploadReference({ blob, cat, mode }) {
  const form = new FormData();
  form.append("file", blob, "reference.jpg");
  form.append("cat_id", cat);
  form.append("mode", mode);
  return api("/gallery/images", { method: "POST", form });
}

// opens the crop dialog for an image URL and uploads the result
export async function addToGallery(src, opts = {}) {
  const list = await cats();
  if (!list.length) {
    toast("Create a cat first (Cats page)", true);
    return null;
  }
  const result = await cropDialog(src, { catsList: list, ...opts });
  if (!result) return null;
  const ref = await uploadReference(result);
  toast(`Added to ${catName(list, result.cat)} (${result.mode})`);
  return ref;
}
