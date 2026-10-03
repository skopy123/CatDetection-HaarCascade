import { $, $$, addToGallery, api, cats, catName, esc, fmtTime, guard, live, renderNav, stateDot, stateLabel, toast }
  from "./api.js";

renderNav();
const id = new URLSearchParams(location.search).get("id");
const base = `/api/v1/cameras/${encodeURIComponent(id)}`;
const catList = await cats();
let camera = await api(`/cameras/${encodeURIComponent(id)}`);
document.title = `CatAlarm – ${camera.name || camera.id}`;
$("#title").textContent = camera.name || camera.id;

// ---- live stream with overlay toggles ----
function overlays() {
  return $$("#overlays input:checked").map((i) => i.value).join(",");
}
function restartStream() {
  $("#stream").src = `${base}/stream.mjpg?fps=5&overlay=${overlays()}&t=${Date.now()}`;
}
$$("#overlays input").forEach((i) => (i.onchange = restartStream));
restartStream();

setInterval(() => {
  $("#motion-img").src = `${base}/motion.png?t=${Date.now()}`;
}, 1000);

// ---- status ----
function renderStatus(s) {
  $("#dot").className = "dot " + (s.connected ? stateDot(s.state) : "bad");
  $("#state").textContent = s.connected ? stateLabel(catList, s.state) : "offline";
  $("#pause").textContent = s.paused ? "Resume" : "Pause";
  const rows = [
    ["connected", s.connected ? "yes" : `no ${s.error ? "– " + s.error : ""}`],
    ["stream fps", s.fps],
    ["frame age", s.frame_age != null ? `${s.frame_age} s` : "–"],
    ["mode", s.night ? "night (IR)" : "day"],
    ["motion", s.motion != null ? `${(s.motion * 100).toFixed(2)} % ${s.motion_detected ? "(moving)" : ""}` : "–"],
    ["confidence", s.confidence ? s.confidence.toFixed(2) : "–"],
    ["HA entity", `<code>${esc(s.entity_id)}</code>`],
  ];
  $("#status").innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
  $("#motion-live").textContent = s.motion != null ? `currently ${(s.motion * 100).toFixed(2)} % changed` : "";
  if (s.last_analysis) renderAnalysis(s.last_analysis);
}

function renderAnalysis(a) {
  const best = a.best;
  const dist = Object.entries(best.distances || {});
  const max = Math.max(1, ...dist.map(([, d]) => d));
  $("#analysis").classList.remove("muted");
  $("#analysis").innerHTML = `
    <div class="row spread"><span>${fmtTime(a.ts, false)}</span>
      <span>detect ${a.detect_ms} ms · embed ${a.embed_ms} ms</span></div>
    <p><strong>${esc(best.state)}</strong> ${best.cat_id ? `→ ${esc(catName(catList, best.cat_id))}` : ""}
      · ${a.detections.length} cat box(es) · margin ${best.margin}</p>
    <div class="bars">${dist.map(([cat, d]) => `
      <div class="bar"><span>${esc(catName(catList, cat))}</span>
        <div class="track"><div class="fill" style="width:${(1 - d / max) * 100}%"></div></div>
        <span class="num">${d.toFixed(3)}</span></div>`).join("") || "<span class='muted'>no references</span>"}
    </div>
    <p class="muted">Bar = similarity, number = cosine distance to the closest references (lower = more similar).</p>`;
}

const refresh = guard(async () => renderStatus(await api(`/cameras/${encodeURIComponent(id)}/status`)));
await refresh();
live((msg) => {
  if (msg.type === "camera" && msg.camera.id === id) renderStatus(msg.camera);
});

// ---- controls ----
$("#pause").onclick = guard(async () => {
  const paused = $("#pause").textContent === "Pause";
  await api(`/cameras/${encodeURIComponent(id)}/pause`, { method: "POST", json: { paused } });
  await refresh();
});
$("#step").onclick = guard(async () => {
  await api(`/cameras/${encodeURIComponent(id)}/step`, { method: "POST" });
  setTimeout(refresh, 600);
});
$("#capture").onclick = guard(async () => {
  const cap = await api(`/cameras/${encodeURIComponent(id)}/capture`, { method: "POST" });
  const status = await api(`/cameras/${encodeURIComponent(id)}/status`);
  await addToGallery(cap.url, { defaultMode: status.night ? "night" : "day", defaultCat: status.cat_id });
});

// ---- motion settings ----
function fillMotion() {
  const m = camera.motion;
  $("#min-area").value = +(m.min_area_ratio * 100).toFixed(3);
  $("#threshold").value = m.threshold;
  $("#motion-enabled").checked = m.enabled;
}
fillMotion();

async function saveMotion(mask = camera.motion.mask) {
  const body = {
    mask,
    min_area_ratio: Number($("#min-area").value) / 100,
    threshold: Number($("#threshold").value),
    enabled: $("#motion-enabled").checked,
  };
  camera.motion = await api(`/cameras/${encodeURIComponent(id)}/motion`, { method: "PUT", json: body });
  toast("Motion settings saved");
}
$("#motion-save").onclick = guard(() => saveMotion());

// ---- mask editor: polygons in normalized coordinates, drawn over the live image ----
const canvas = $("#mask-canvas");
let polygons = [];
let current = [];

function imageRect() {
  // the stream is letterboxed (object-fit: contain), find the area the frame occupies
  const img = $("#stream");
  const box = $("#video").getBoundingClientRect();
  const iw = img.naturalWidth || 16, ih = img.naturalHeight || 9;
  const scale = Math.min(box.width / iw, box.height / ih);
  const w = iw * scale, h = ih * scale;
  return { x: (box.width - w) / 2, y: (box.height - h) / 2, w, h, box };
}

function drawMask() {
  const { x, y, w, h, box } = imageRect();
  canvas.width = box.width;
  canvas.height = box.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const path = (pts, close) => {
    ctx.beginPath();
    pts.forEach(([px, py], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x + px * w, y + py * h));
    if (close) ctx.closePath();
  };
  ctx.lineWidth = 2;
  for (const poly of polygons) {
    path(poly, true);
    ctx.fillStyle = "rgba(40,40,40,.55)";
    ctx.fill();
    ctx.strokeStyle = "#f08a35";
    ctx.stroke();
  }
  if (current.length) {
    path(current, false);
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    for (const [px, py] of current) {
      ctx.fillStyle = "#f08a35";
      ctx.fillRect(x + px * w - 3, y + py * h - 3, 6, 6);
    }
  }
}

canvas.addEventListener("click", (e) => {
  const { x, y, w, h } = imageRect();
  const r = canvas.getBoundingClientRect();
  const px = (e.clientX - r.left - x) / w, py = (e.clientY - r.top - y) / h;
  current.push([Math.min(1, Math.max(0, +px.toFixed(4))), Math.min(1, Math.max(0, +py.toFixed(4)))]);
  drawMask();
});

function setEditing(on) {
  canvas.hidden = !on;
  ["#mask-close", "#mask-clear", "#mask-save", "#mask-help"].forEach((s) => ($(s).hidden = !on));
  $("#edit-mask").textContent = on ? "Cancel" : "Edit ignored areas";
  if (on) {
    polygons = structuredClone(camera.motion.mask);
    current = [];
    drawMask();
  }
}
$("#edit-mask").onclick = () => setEditing(canvas.hidden);
$("#mask-close").onclick = () => {
  if (current.length >= 3) polygons.push(current);
  current = [];
  drawMask();
};
$("#mask-clear").onclick = () => {
  polygons = [];
  current = [];
  drawMask();
};
$("#mask-save").onclick = guard(async () => {
  if (current.length >= 3) polygons.push(current);
  await saveMotion(polygons);
  setEditing(false);
});
window.addEventListener("resize", () => !canvas.hidden && drawMask());

// ---- thresholds (global) ----
const TH = [
  ["match", "Match distance", 0, 1, 0.01, "confirmed when the best cat is closer than this"],
  ["unconfirmed", "Unconfirmed distance", 0, 1, 0.01, "reported as “unconfirmed” below this"],
  ["margin", "Required margin", 0, 0.5, 0.01, "gap between best and second-best cat"],
  ["vote_k", "Votes needed", 1, 10, 1, "wins needed within the vote window"],
  ["vote_n", "Vote window", 1, 20, 1, "number of recent analyses that vote"],
  ["hold_s", "Hold time (s)", 0, 120, 1, "keep reporting a cat this long after it was last seen"],
];
const settings = await api("/settings");
$("#thresholds").innerHTML = TH.map(([key, label, min, max, step, help]) => `
  <label title="${esc(help)}">${label}: <strong data-out="${key}">${settings.thresholds[key]}</strong>
    <input type="range" data-th="${key}" min="${min}" max="${max}" step="${step}" value="${settings.thresholds[key]}">
  </label>`).join("");
$$("[data-th]").forEach((el) => (el.oninput = () => ($(`[data-out="${el.dataset.th}"]`).textContent = el.value)));
$("#th-save").onclick = guard(async () => {
  const thresholds = { ...settings.thresholds };
  $$("[data-th]").forEach((el) => (thresholds[el.dataset.th] = Number(el.value)));
  await api("/settings", { method: "PUT", json: { thresholds } });
  toast("Thresholds applied");
});
