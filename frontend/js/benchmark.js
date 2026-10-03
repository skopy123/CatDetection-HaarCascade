import { $, api, cats, catName, esc, guard, renderNav, toast } from "./api.js";

renderNav();
const catList = await cats();
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)} %`);

// ---- test an image ----
$("#test-file").onchange = guard(async () => {
  const file = $("#test-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("mode", $("#test-mode").value);
  const r = await api("/debug/identify", { method: "POST", form });

  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  const canvas = $("#test-canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = Math.max(2, img.naturalWidth / 300);
  ctx.font = `${Math.max(14, img.naturalWidth / 40)}px system-ui`;
  r.detections.forEach((d, i) => {
    const res = r.results[i];
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = res?.state === "match" ? "#2f9e44" : "#e0a100";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${res ? catName(catList, res.cat_id) : "cat"} ${res ? res.distance.toFixed(2) : ""}`, x1 + 6, y1 + 22);
  });
  URL.revokeObjectURL(img.src);

  const best = r.best;
  $("#test-details").innerHTML = `
    <p><strong>${esc(best.state)}</strong>${best.cat_id ? ` – ${esc(catName(catList, best.cat_id))}` : ""}
      · ${r.night ? "night" : "day"} references · ${r.detections.length} cat box(es)</p>
    <p class="muted">detector ${r.detect_ms} ms · embedder ${r.embed_ms} ms</p>
    <table><thead><tr><th>Cat</th><th class="num">Distance</th></tr></thead><tbody>
      ${Object.entries(best.distances).map(([c, d]) =>
        `<tr><td>${esc(catName(catList, c))}</td><td class="num">${d.toFixed(3)}</td></tr>`).join("")}
    </tbody></table>
    ${r.detections.length ? "" : `<p>No cat detected. Try a clearer picture, or lower the detector confidence in Settings.</p>`}`;
  $("#test-result").hidden = false;
});

// ---- benchmark ----
function confusion(m) {
  return `<table class="small"><thead><tr><th>actual \\ predicted</th>${m.cats.map((c) =>
    `<th class="num">${esc(catName(catList, c))}</th>`).join("")}</tr></thead><tbody>${m.cats.map((a) =>
    `<tr><th>${esc(catName(catList, a))}</th>${m.cats.map((p) => {
      const v = m.confusion[a][p];
      const style = v ? (a === p ? "font-weight:700" : "color:var(--bad);font-weight:700") : "color:var(--muted)";
      return `<td class="num" style="${style}">${v}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
}

$("#run").onclick = guard(async () => {
  $("#run").disabled = true;
  $("#results").innerHTML = `<p class="muted">Embedding all references with every available model…</p>`;
  try {
    const r = await api("/benchmark", { method: "POST", json: {} });
    const th = r.thresholds;
    const rows = r.results.flatMap((res) => Object.entries(res.modes).map(([mode, m]) => `
      <tr>
        <td><strong>${esc(res.embedder)}</strong>${res.embedder === r.current ? ' <span class="pill">active</span>' : ""}</td>
        <td>${mode}</td><td class="num">${m.images}</td>
        <td class="num">${pct(m.top1_accuracy)}</td><td class="num">${pct(m.accepted_rate)}</td>
        <td class="num">${pct(m.accepted_precision)}</td>
        <td class="num">${m.same_cat_distance?.mean.toFixed(2) ?? "–"}</td>
        <td class="num">${m.other_cat_distance?.mean.toFixed(2) ?? "–"}</td>
        <td class="num">${res.ms_per_image}</td>
      </tr>`)).join("");
    const details = r.results.map((res) => Object.entries(res.modes).map(([mode, m]) => `
      <details><summary>${esc(res.embedder)} · ${mode} – confusion matrix</summary>${confusion(m)}</details>`).join("")).join("");
    $("#results").innerHTML = `
      <p class="small muted">Thresholds used: match ≤ ${th.match}, margin ≥ ${th.margin}, k = ${th.knn_k}.
        Switch the active model in <a href="settings.html">Settings</a>.</p>
      <div style="overflow-x:auto"><table><thead><tr><th>Embedder</th><th>Mode</th><th class="num">Images</th>
        <th class="num">Top-1</th><th class="num">Accepted</th><th class="num">Precision</th>
        <th class="num">Same cat</th><th class="num">Other cat</th><th class="num">ms / image</th></tr></thead>
        <tbody>${rows}</tbody></table></div>${details}`;
    toast("Benchmark finished");
  } finally {
    $("#run").disabled = false;
  }
});
