import { $, api, esc, fmtDuration, guard, localDay, renderNav, seqColor } from "./api.js";

renderNav();
const Chart = window.Chart;
const css = getComputedStyle(document.documentElement);
const ink = css.getPropertyValue("--muted").trim();
const surface = css.getPropertyValue("--panel").trim();
const grid = css.getPropertyValue("--border").trim();
Chart.defaults.color = ink;
Chart.defaults.font.family = "system-ui, sans-serif";
Chart.defaults.plugins.tooltip.mode = "index";
Chart.defaults.plugins.tooltip.intersect = false;

let dailyChart, perfChart;

function days(n) {
  const out = [];
  for (let i = n - 1; i >= 0; i--) out.push(localDay(new Date(Date.now() - i * 86400000)));
  return out;
}

function renderDaily(summary) {
  const labels = days(summary.days);
  const byCat = Object.fromEntries(summary.cats.map((c) => [c.id, Object.fromEntries(labels.map((d) => [d, 0]))]));
  for (const r of summary.daily) if (byCat[r.cat_id] && r.day in byCat[r.cat_id]) byCat[r.cat_id][r.day] += r.visits;
  const datasets = summary.cats.map((c) => ({
    label: c.name,
    data: labels.map((d) => byCat[c.id][d]),
    backgroundColor: c.color,
    borderColor: surface, // 2px surface gap between stacked segments
    borderWidth: 2,
    borderRadius: 4,
    borderSkipped: false,
    maxBarThickness: 28,
  }));
  dailyChart?.destroy();
  dailyChart = new Chart($("#daily"), {
    type: "bar",
    data: { labels: labels.map((d) => d.slice(5)), datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } },
      },
      plugins: { legend: { position: "top", align: "start", labels: { boxWidth: 12, boxHeight: 12 } } },
    },
  });
  $("#daily-table").innerHTML = `<table><thead><tr><th>Day</th>${summary.cats.map((c) =>
    `<th class="num">${esc(c.name)}</th>`).join("")}</tr></thead><tbody>${labels.map((d) =>
    `<tr><td>${d}</td>${summary.cats.map((c) => `<td class="num">${byCat[c.id][d]}</td>`).join("")}</tr>`).join("")}
    </tbody></table>`;
}

function renderHourly(summary) {
  const counts = {};
  for (const r of summary.hourly) counts[`${r.cat_id}:${r.hour}`] = r.visits;
  const max = Math.max(1, ...Object.values(counts));
  const hours = [...Array(24).keys()];
  const shade = (v) => (v ? seqColor(0.15 + (0.85 * v) / max).bg : "var(--bg)");
  let html = `<div class="heat" style="grid-template-columns:6rem repeat(24, minmax(14px, 1fr))">
    <div class="label"></div>${hours.map((h) => `<div class="label muted">${h % 3 === 0 ? h : ""}</div>`).join("")}`;
  for (const c of summary.cats) {
    html += `<div class="label" style="justify-items:start">${esc(c.name)}</div>`;
    html += hours.map((h) => {
      const v = counts[`${c.id}:${h}`] ?? 0;
      return `<div title="${esc(c.name)}, ${h}:00–${h + 1}:00: ${v} visits" style="background:${shade(v)}"></div>`;
    }).join("");
  }
  $("#hourly").innerHTML = summary.cats.length ? html + "</div>" : `<p class="muted">No cats.</p>`;
}

function renderDurations(summary) {
  const total = {};
  for (const r of summary.daily) {
    total[r.cat_id] ??= { visits: 0, seconds: 0 };
    total[r.cat_id].visits += r.visits;
    total[r.cat_id].seconds += r.seconds ?? 0;
  }
  const max = Math.max(1, ...Object.values(total).map((t) => t.seconds));
  $("#durations").innerHTML = summary.cats.map((c) => {
    const t = total[c.id] ?? { visits: 0, seconds: 0 };
    const avg = t.visits ? t.seconds / t.visits : null;
    return `<div class="bar" style="grid-template-columns:6rem 1fr 9rem" title="${t.visits} visits, average ${fmtDuration(avg)}">
      <span>${esc(c.name)}</span>
      <div class="track"><div class="fill" style="width:${(t.seconds / max) * 100}%;background:${c.color}"></div></div>
      <span class="num small">${fmtDuration(t.seconds)} · ${t.visits}×</span></div>`;
  }).join("") || `<p class="muted">No data.</p>`;
}

function renderStates(summary) {
  const cams = {};
  for (const r of summary.states) {
    cams[r.camera_id] ??= { identified: 0, unconfirmed: 0, no_cat: 0 };
    const key = r.state === "unconfirmed" ? "unconfirmed" : ["no_cat", "no_motion"].includes(r.state) ? null : "identified";
    if (r.state === "no_cat") cams[r.camera_id].no_cat += r.n;
    if (key) cams[r.camera_id][key] += r.n;
  }
  const rows = Object.entries(cams);
  $("#states").innerHTML = rows.length ? `<table><thead><tr><th>Camera</th><th class="num">Identified</th>
    <th class="num">Unconfirmed</th><th class="num">Motion, no cat</th><th class="num">Identified %</th></tr></thead>
    <tbody>${rows.map(([cam, v]) => {
      const pct = v.identified + v.unconfirmed ? Math.round((100 * v.identified) / (v.identified + v.unconfirmed)) : null;
      return `<tr><td>${esc(cam)}</td><td class="num">${v.identified}</td><td class="num">${v.unconfirmed}</td>
        <td class="num">${v.no_cat}</td><td class="num">${pct ?? "–"}${pct != null ? " %" : ""}</td></tr>`;
    }).join("")}</tbody></table>` : `<p class="muted">No data.</p>`;
}

function renderPerf(rows) {
  // one unit (ms) on one axis; averaged over all cameras per 10 minutes
  const buckets = {};
  for (const r of rows) {
    const b = (buckets[r.ts] ??= { detect: [], embed: [], motion: [] });
    if (r.detect_ms != null) b.detect.push(r.detect_ms);
    if (r.embed_ms != null) b.embed.push(r.embed_ms);
    if (r.motion_ms != null) b.motion.push(r.motion_ms);
  }
  const ts = Object.keys(buckets).map(Number).sort((a, b) => a - b);
  const avg = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : null);
  const line = (label, key, color) => ({
    label, data: ts.map((t) => avg(buckets[t][key])), borderColor: color, backgroundColor: color,
    borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, spanGaps: true, tension: 0.2,
  });
  perfChart?.destroy();
  perfChart = new Chart($("#perf"), {
    type: "line",
    data: {
      labels: ts.map((t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })),
      datasets: [line("detector", "detect", "#2a78d6"), line("embedder", "embed", "#eb6834"),
        line("motion", "motion", "#1baf7a")],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { beginAtZero: true, grid: { color: grid }, title: { display: true, text: "ms" } },
      },
      plugins: { legend: { position: "top", align: "start", labels: { boxWidth: 12, boxHeight: 2 } } },
    },
  });
}

const load = guard(async () => {
  const d = $("#days").value;
  const [summary, perf] = await Promise.all([api(`/stats/summary?days=${d}`), api("/stats/perf?hours=24")]);
  renderDaily(summary);
  renderHourly(summary);
  renderDurations(summary);
  renderStates(summary);
  renderPerf(perf);
});

$("#days").onchange = load;
await load();
