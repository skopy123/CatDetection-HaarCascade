import { $, api, cropDialog, esc, guard, renderNav, seqColor, toast, uploadReference } from "./api.js";

renderNav();
let catList = [];

function catSection(cat, images) {
  const el = document.createElement("section");
  el.className = "card";
  el.innerHTML = `
    <div class="row spread">
      <div class="row">
        <input type="color" value="${esc(cat.color)}" data-act="color" title="color">
        <h2 style="margin:0">${esc(cat.name)}</h2>
        <span class="pill">${cat.images.day} day</span><span class="pill">${cat.images.night} night</span>
        <code class="small muted">${esc(cat.id)}</code>
      </div>
      <div class="row">
        <button data-act="upload" class="primary">Upload picture</button>
        <button data-act="rename">Rename</button>
        <button data-act="delete" class="danger">Delete</button>
      </div>
    </div>
    <div class="thumbs" style="margin-top:.75rem">
      ${images.map((im) => `
        <div class="thumb" data-image="${esc(im.id)}">
          <img src="${im.url}" alt="" loading="lazy">
          <span class="pill tag">${im.mode}</span>
          <div class="actions">
            <select data-act="move" title="move to another cat">
              <option value="">move…</option>
              ${catList.filter((c) => c.id !== cat.id).map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("")}
            </select>
            <button data-act="toggle-mode" title="switch day/night">${im.mode === "day" ? "→ night" : "→ day"}</button>
            <button data-act="delete-image" class="danger" title="delete">✕</button>
          </div>
        </div>`).join("") || `<p class="muted small">No pictures yet.</p>`}
    </div>`;

  el.querySelector('[data-act="upload"]').onclick = () => pickAndUpload(cat);
  el.querySelector('[data-act="rename"]').onclick = guard(async () => {
    const name = prompt("New name", cat.name);
    if (!name) return;
    await api(`/gallery/cats/${cat.id}`, { method: "PATCH", json: { name } });
    await load();
  });
  el.querySelector('[data-act="color"]').onchange = guard(async (e) => {
    await api(`/gallery/cats/${cat.id}`, { method: "PATCH", json: { color: e.target.value } });
  });
  el.querySelector('[data-act="delete"]').onclick = guard(async () => {
    if (!confirm(`Delete ${cat.name} and all ${cat.images.day + cat.images.night} pictures?`)) return;
    await api(`/gallery/cats/${cat.id}`, { method: "DELETE" });
    await load();
  });
  el.querySelectorAll("[data-image]").forEach((thumb) => {
    const imageId = thumb.dataset.image;
    const image = images.find((i) => i.id === imageId);
    thumb.querySelector('[data-act="delete-image"]').onclick = guard(async () => {
      await api(`/gallery/images/${imageId}`, { method: "DELETE" });
      await load();
    });
    thumb.querySelector('[data-act="toggle-mode"]').onclick = guard(async () => {
      await api(`/gallery/images/${imageId}`, {
        method: "PATCH", json: { mode: image.mode === "day" ? "night" : "day" } });
      await load();
    });
    thumb.querySelector('[data-act="move"]').onchange = guard(async (e) => {
      if (!e.target.value) return;
      await api(`/gallery/images/${imageId}`, { method: "PATCH", json: { cat_id: e.target.value } });
      await load();
    });
  });
  return el;
}

function pickAndUpload(cat) {
  const input = $("#file");
  input.value = "";
  input.onchange = guard(async () => {
    const file = input.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const result = await cropDialog(url, { catsList: catList, defaultCat: cat.id });
    URL.revokeObjectURL(url);
    if (!result) return;
    await uploadReference(result);
    toast("Picture added");
    await load();
  });
  input.click();
}

async function renderMatrix() {
  const mode = $("#matrix-mode").value;
  const m = await api(`/gallery/matrix?mode=${mode}`);
  $("#matrix-embedder").textContent = m.embedder;
  const n = m.ids.length;
  if (!n) {
    $("#matrix").innerHTML = `<p class="muted">No ${mode} references.</p>`;
    return;
  }
  const colorOf = (id) => catList.find((c) => c.id === id)?.color ?? "#999";
  const cell = 26;
  const head = m.labels.map((l, i) =>
    `<div title="${esc(l)}" style="background:${colorOf(l)};height:6px;grid-column:${i + 2};grid-row:1"></div>`).join("");
  const side = m.labels.map((l, i) =>
    `<div title="${esc(l)}" style="background:${colorOf(l)};width:6px;grid-column:1;grid-row:${i + 2}"></div>`).join("");
  let cells = "";
  m.distance.forEach((row, i) => row.forEach((d, j) => {
    const c = seqColor(1 - d); // similar pictures (small distance) get the strongest color
    cells += `<div title="${esc(m.labels[i])} ↔ ${esc(m.labels[j])}: ${d.toFixed(3)}"
      style="grid-column:${j + 2};grid-row:${i + 2};background:${c.bg};color:${c.fg}">
      ${n <= 24 ? d.toFixed(1) : ""}</div>`;
  }));
  $("#matrix").innerHTML = `<div class="heat" style="grid-template-columns:6px repeat(${n}, ${cell}px);
    grid-template-rows:6px repeat(${n}, ${cell}px)">${head}${side}${cells}</div>`;
}

const load = guard(async () => {
  const mode = $("#mode-filter").value;
  const [list, images] = await Promise.all([
    api("/gallery/cats"), api(`/gallery/images${mode ? `?mode=${mode}` : ""}`),
  ]);
  catList = list;
  $("#cats").replaceChildren(...catList.map((c) => catSection(c, images.filter((i) => i.cat_id === c.id))));
  if (!catList.length) $("#cats").innerHTML = `<p class="muted">No cats yet – add one below.</p>`;
  await renderMatrix();
});

$("#mode-filter").onchange = load;
$("#matrix-mode").onchange = guard(renderMatrix);
$("#add-cat").onsubmit = guard(async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const name = f.get("name").trim();
  const id = (f.get("id") || name).toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 32);
  // no color: the server assigns the next slot of the validated palette
  await api("/gallery/cats", { method: "POST", json: { id, name } });
  e.target.reset();
  toast(`${name} added – now upload some pictures`);
  await load();
});

await load();
