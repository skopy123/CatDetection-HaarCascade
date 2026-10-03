import { $, api, cats, catColor, catName, esc, fmtAgo, fmtDuration, guard, live, localDay, renderNav, stateDot, stateLabel }
  from "./api.js";

renderNav();
const catList = await cats();

function cameraCard(cam) {
  const el = document.createElement("a");
  el.className = "card flush";
  el.href = `camera.html?id=${encodeURIComponent(cam.id)}`;
  el.style.color = "inherit";
  el.style.textDecoration = "none";
  el.dataset.camera = cam.id;
  el.innerHTML = `
    <div class="video">
      ${cam.enabled
        ? `<img alt="live view ${esc(cam.name)}" src="/api/v1/cameras/${encodeURIComponent(cam.id)}/stream.mjpg?overlay=boxes&fps=3">`
        : `<div class="overlay-msg">disabled</div>`}
    </div>
    <div class="body row spread">
      <div>
        <div><strong>${esc(cam.name || cam.id)}</strong></div>
        <div class="small muted" data-field="meta"></div>
      </div>
      <div class="row"><span class="dot" data-field="dot"></span><span class="state" data-field="state">–</span></div>
    </div>`;
  updateCard(el, cam.status);
  return el;
}

function updateCard(el, status) {
  if (!status) return;
  const dot = el.querySelector('[data-field="dot"]');
  dot.className = "dot " + (status.connected ? stateDot(status.state) : "bad");
  el.querySelector('[data-field="state"]').textContent = status.connected
    ? stateLabel(catList, status.state) : "offline";
  el.querySelector('[data-field="meta"]').textContent = [
    status.night ? "night" : "day",
    status.fps ? `${status.fps} fps` : null,
    status.error,
  ].filter(Boolean).join(" · ");
}

function eventRow(e) {
  return `<div class="event">
    ${e.crop_url ? `<img src="${e.crop_url}" alt="">` : `<img alt="">`}
    <div>
      <div><span class="dot" style="background:${catColor(catList, e.cat_id)}"></span>
        <strong>${esc(e.state === "unconfirmed" ? `unconfirmed (${catName(catList, e.cat_id)}?)` : stateLabel(catList, e.state))}</strong></div>
      <div class="small muted">${esc(e.camera_id)} · ${fmtAgo(e.ts)}${e.confidence ? ` · conf ${e.confidence.toFixed(2)}` : ""}</div>
    </div></div>`;
}

const load = guard(async () => {
  const [cams, events, summary] = await Promise.all([
    api("/cameras"), api("/events?limit=8"), api("/stats/summary?days=1"),
  ]);
  const box = $("#cameras");
  box.replaceChildren(...cams.map(cameraCard));
  $("#no-cameras").hidden = cams.length > 0;
  $("#events").innerHTML = events.map(eventRow).join("") || `<p class="muted">No events yet.</p>`;

  const today = localDay();
  const perCat = {};
  for (const r of summary.daily.filter((r) => r.day === today)) {
    perCat[r.cat_id] ??= { visits: 0, seconds: 0 };
    perCat[r.cat_id].visits += r.visits;
    perCat[r.cat_id].seconds += r.seconds;
  }
  const max = Math.max(1, ...Object.values(perCat).map((v) => v.visits));
  $("#today").innerHTML = catList.map((c) => {
    const v = perCat[c.id] ?? { visits: 0, seconds: 0 };
    return `<div class="bar" title="${fmtDuration(v.seconds)} total">
      <span>${esc(c.name)}</span>
      <div class="track"><div class="fill" style="width:${(v.visits / max) * 100}%;background:${c.color}"></div></div>
      <span class="num">${v.visits}×</span></div>`;
  }).join("") || `<p class="muted">No cats yet.</p>`;
});

await load();

live((msg) => {
  if (msg.type === "camera") {
    const el = document.querySelector(`[data-camera="${CSS.escape(msg.camera.id)}"]`);
    if (el) updateCard(el, msg.camera);
  } else if (msg.type === "hello") {
    for (const s of msg.cameras) {
      const el = document.querySelector(`[data-camera="${CSS.escape(s.id)}"]`);
      if (el) updateCard(el, s);
    }
  } else if (msg.type === "event" && !["no_motion", "no_cat"].includes(msg.event.state)) {
    const box = $("#events");
    box.querySelector("p")?.remove();
    const snap = msg.event.snapshot ? `/api/v1/events/${msg.event.id}/snapshot.jpg?crop=true` : null;
    box.insertAdjacentHTML("afterbegin", eventRow({ ...msg.event, crop_url: snap }));
    while (box.children.length > 8) box.lastElementChild.remove();
  }
});
