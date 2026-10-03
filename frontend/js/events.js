import { $, addToGallery, api, cats, catName, esc, fmtTime, guard, live, renderNav, stateLabel, toast } from "./api.js";

renderNav();
const catList = await cats();
const cameras = await api("/cameras");
const PAGE = 50;
let oldest = null;

const form = $("#filters");
form.camera_id.innerHTML += cameras.map((c) => `<option value="${esc(c.id)}">${esc(c.name || c.id)}</option>`).join("");
form.cat_id.innerHTML += catList.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("");

function query(before) {
  const p = new URLSearchParams({ limit: PAGE });
  for (const k of ["camera_id", "cat_id"]) if (form[k].value) p.set(k, form[k].value);
  if (form.include_idle.checked) p.set("include_idle", "true");
  if (before) p.set("before", before);
  return `/events?${p}`;
}

function row(e) {
  const tr = document.createElement("tr");
  tr.dataset.event = e.id;
  const state = e.state === "unconfirmed" && e.cat_id
    ? `unconfirmed <span class="muted">(${esc(catName(catList, e.cat_id))}?)</span>` : esc(stateLabel(catList, e.state));
  tr.innerHTML = `
    <td>${e.crop_url ? `<img src="${e.crop_url}" alt="" style="width:80px;height:56px;object-fit:cover;border-radius:6px;cursor:zoom-in" data-act="view">` : ""}</td>
    <td class="small">${fmtTime(e.ts)}</td>
    <td>${esc(e.camera_id)}</td>
    <td>${state}${e.night ? ' <span class="pill">night</span>' : ""}</td>
    <td class="num">${e.confidence != null ? e.confidence.toFixed(2) : "–"}</td>
    <td><select data-act="relabel">
      <option value="">– none –</option>
      ${catList.map((c) => `<option value="${esc(c.id)}" ${c.id === e.cat_id ? "selected" : ""}>${esc(c.name)}</option>`).join("")}
    </select></td>
    <td>${e.snapshot_url ? `<button data-act="gallery">Add to gallery</button>` : ""}</td>`;
  tr.querySelector('[data-act="view"]')?.addEventListener("click", () => {
    $("#viewer-img").src = e.snapshot_url;
    $("#viewer").showModal();
  });
  tr.querySelector('[data-act="relabel"]').onchange = guard(async (ev) => {
    await api(`/events/${e.id}`, { method: "PATCH", json: { cat_id: ev.target.value || null } });
    toast("Relabelled");
  });
  const btn = tr.querySelector('[data-act="gallery"]');
  if (btn) {
    btn.onclick = guard(() => addToGallery(e.snapshot_url, {
      defaultCat: tr.querySelector('[data-act="relabel"]').value || e.cat_id,
      defaultMode: e.night ? "night" : "day",
    }));
  }
  return tr;
}

const load = guard(async (append = false) => {
  const events = await api(query(append ? oldest : null));
  if (!append) $("#rows").replaceChildren();
  $("#rows").append(...events.map(row));
  if (events.length) oldest = events[events.length - 1].ts;
  $("#more").hidden = events.length < PAGE;
  if (!append && !events.length) $("#rows").innerHTML = `<tr><td colspan="7" class="muted">No events.</td></tr>`;
});

form.onchange = () => load(false);
$("#more").onclick = () => load(true);
await load(false);

live((msg) => {
  if (msg.type !== "event") return;
  const e = msg.event;
  if (!form.include_idle.checked && ["no_motion", "no_cat"].includes(e.state)) return;
  if (form.camera_id.value && form.camera_id.value !== e.camera_id) return;
  if (form.cat_id.value && form.cat_id.value !== e.cat_id) return;
  const url = e.snapshot ? `/api/v1/events/${e.id}/snapshot.jpg` : null;
  $("#rows").querySelector("td[colspan]")?.parentElement.remove();
  $("#rows").prepend(row({ ...e, snapshot_url: url, crop_url: url && `${url}?crop=true` }));
});
