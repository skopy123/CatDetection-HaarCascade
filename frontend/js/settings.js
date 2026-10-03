import { $, api, esc, fmtAgo, guard, renderNav, toast } from "./api.js";

renderNav();

// ---- cameras ----
let cameras = [];
const dlg = $("#camera-dialog");
const cform = $("#camera-form");
let editing = null;

function openCamera(cam) {
  editing = cam;
  $("#camera-dialog-title").textContent = cam ? `Edit ${cam.name || cam.id}` : "Add camera";
  cform.elements.id.value = cam?.id ?? "";
  cform.elements.id.readOnly = Boolean(cam);
  cform.elements.name.value = cam?.name ?? "";
  cform.elements.url.value = cam?.url ?? "";
  cform.elements.ha_entity.value = cam?.ha_entity ?? "";
  cform.elements.enabled.checked = cam?.enabled ?? true;
  dlg.showModal();
}

dlg.addEventListener("close", guard(async () => {
  if (dlg.returnValue !== "save") return;
  const body = {
    ...(editing ?? {}),
    id: cform.elements.id.value.trim(),
    name: cform.elements.name.value.trim(),
    url: cform.elements.url.value.trim(),
    ha_entity: cform.elements.ha_entity.value.trim() || null,
    enabled: cform.elements.enabled.checked,
  };
  delete body.status;
  delete body.entity_id;
  if (editing) await api(`/cameras/${encodeURIComponent(editing.id)}`, { method: "PUT", json: body });
  else await api("/cameras", { method: "POST", json: body });
  toast("Camera saved");
  await loadCameras();
}));

async function loadCameras() {
  cameras = await api("/cameras");
  $("#camera-rows").innerHTML = cameras.map((c) => `
    <tr data-id="${esc(c.id)}">
      <td><a href="camera.html?id=${encodeURIComponent(c.id)}">${esc(c.id)}</a></td>
      <td>${esc(c.name)}</td>
      <td class="small mono" style="word-break:break-all">${esc(c.url)}</td>
      <td class="small mono">${esc(c.entity_id)}</td>
      <td>${c.enabled ? (c.status?.connected ? '<span class="dot ok"></span> online' : '<span class="dot bad"></span> offline') : "disabled"}</td>
      <td class="row"><button data-act="edit">Edit</button><button data-act="delete" class="danger">Delete</button></td>
    </tr>`).join("") || `<tr><td colspan="6" class="muted">No cameras yet.</td></tr>`;
  for (const tr of document.querySelectorAll("#camera-rows tr[data-id]")) {
    const cam = cameras.find((c) => c.id === tr.dataset.id);
    tr.querySelector('[data-act="edit"]').onclick = () => openCamera(cam);
    tr.querySelector('[data-act="delete"]').onclick = guard(async () => {
      if (!confirm(`Delete camera ${cam.id}? Its statistics are kept.`)) return;
      await api(`/cameras/${encodeURIComponent(cam.id)}`, { method: "DELETE" });
      await loadCameras();
    });
  }
}
$("#add-camera").onclick = () => openCamera(null);

// ---- settings form: field names are dotted paths into the settings object ----
const form = $("#settings");
let settings;

const get = (obj, path) => path.split(".").reduce((o, k) => o?.[k], obj);
function set(obj, path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  keys.reduce((o, k) => (o[k] ??= {}), obj)[last] = value;
}

async function loadSettings() {
  const [s, embedders, health] = await Promise.all([api("/settings"), api("/settings/embedders"), api("/health")]);
  settings = s;
  form.embedder.innerHTML = embedders.map((e) =>
    `<option value="${esc(e.name)}" title="${esc(e.description)}" ${e.available ? "" : "disabled"}>${esc(e.name)}${e.available ? "" : " (not installed)"}</option>`).join("");
  for (const el of form.elements) {
    if (!el.name) continue;
    const v = get(s, el.name);
    if (el.type === "checkbox") el.checked = Boolean(v);
    else el.value = v ?? "";
  }
  const ha = health.ha;
  $("#ha-status").innerHTML = !ha.enabled ? "HA publishing is off (no URL)."
    : ha.last_error ? `<span style="color:var(--bad)">Last error: ${esc(ha.last_error)}</span>`
    : ha.last_ok ? `Last successful update ${fmtAgo(ha.last_ok)}.` : "No update sent yet.";
}

form.onsubmit = guard(async (e) => {
  e.preventDefault();
  const body = structuredClone(settings);
  for (const el of form.elements) {
    if (!el.name) continue;
    let v = el.type === "checkbox" ? el.checked : el.value;
    if (el.type === "number") v = Number(v);
    if (el.type !== "checkbox" && el.type !== "number" && v === "") v = null;
    set(body, el.name, v);
  }
  await api("/settings", { method: "PUT", json: body });
  toast("Settings saved");
  await loadSettings();
});

await Promise.all([loadCameras(), loadSettings()]);
if (location.hash === "#cameras" && !cameras.length) openCamera(null);
