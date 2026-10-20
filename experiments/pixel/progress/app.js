// flock pixel progress page. No build step, no libraries.
// Core (parseLog, Replayer) is DOM-free so node can test it; the UI runs only in a browser.
const PX = {};

PX.parseLog = function (text) {
  const lines = text.split('\n').filter(Boolean);
  let total = 0;
  const entries = lines.map((l) => JSON.parse(l));
  for (const e of entries) total += e.px.length;
  const idx = new Int32Array(total), rgb = new Uint8Array(total * 3), aid = new Int32Array(total);
  const ends = new Int32Array(entries.length + 1);
  const meta = new Array(entries.length + 1);
  return { entries, idx, rgb, aid, names: [], ends, meta, total, filled: false };
};

PX.index = function (log, width) {
  // Fill idx/rgb/ends once; refused pixels were never applied, so they are skipped.
  let k = 0;
  const ids = new Map();
  log.entries.forEach((e, i) => {
    let a = ids.get(e.agent);
    if (a === undefined) { a = log.names.length; ids.set(e.agent, a); log.names.push(e.agent); }
    for (const p of e.px) {
      if (p[3] === 'refused') continue;
      log.aid[k] = a;
      log.idx[k] = p[1] * width + p[0];
      const h = p[2];
      log.rgb[3 * k] = parseInt(h.slice(0, 2), 16);
      log.rgb[3 * k + 1] = parseInt(h.slice(2, 4), 16);
      log.rgb[3 * k + 2] = parseInt(h.slice(4, 6), 16);
      k++;
    }
    log.ends[i + 1] = k;
  });
  log.filled = true;
  return log;
};

PX.Replayer = class {
  constructor(log, width, height, blank, every = 2000, cache = 6) {
    this.log = log; this.w = width; this.h = height; this.every = every; this.max = cache;
    this.blank = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      this.blank[4 * i] = blank[0]; this.blank[4 * i + 1] = blank[1];
      this.blank[4 * i + 2] = blank[2]; this.blank[4 * i + 3] = 255;
    }
    this.kf = new Map(); // keyframe entry -> RGBA, insertion order = LRU
  }
  get length() { return this.log.count ?? this.log.entries.length; }
  apply(buf, lo, hi) {
    const { idx, rgb } = this.log;
    for (let k = lo; k < hi; k++) {
      const o = 4 * idx[k];
      buf[o] = rgb[3 * k]; buf[o + 1] = rgb[3 * k + 1]; buf[o + 2] = rgb[3 * k + 2];
    }
  }
  keyframe(k) {
    if (k === 0) return this.blank;
    if (this.kf.has(k)) { const v = this.kf.get(k); this.kf.delete(k); this.kf.set(k, v); return v; }
    let prev = 0;
    for (const c of this.kf.keys()) if (c < k && c > prev) prev = c;
    const buf = new Uint8ClampedArray(prev === 0 ? this.blank : this.kf.get(prev));
    this.apply(buf, this.log.ends[prev], this.log.ends[k]);
    this.kf.set(k, buf);
    while (this.kf.size > this.max) this.kf.delete(this.kf.keys().next().value);
    return buf;
  }
  frame(t) {
    t = Math.max(0, Math.min(this.length, t | 0));
    const k = Math.floor(t / this.every) * this.every;
    const buf = new Uint8ClampedArray(this.keyframe(k));
    this.apply(buf, this.log.ends[k], this.log.ends[t]);
    return buf;
  }
};

// Painted mask at log position t (1 = touched at least once), same keyframe scheme.
PX.Replayer.prototype.maskKeyframe = function (k) {
  this.mkf = this.mkf || new Map();
  if (k === 0) return new Uint8Array(this.w * this.h);
  if (this.mkf.has(k)) { const v = this.mkf.get(k); this.mkf.delete(k); this.mkf.set(k, v); return v; }
  let prev = 0;
  for (const c of this.mkf.keys()) if (c < k && c > prev) prev = c;
  const m = prev === 0 ? new Uint8Array(this.w * this.h) : new Uint8Array(this.mkf.get(prev));
  const { idx, ends } = this.log;
  for (let j = ends[prev]; j < ends[k]; j++) m[idx[j]] = 1;
  this.mkf.set(k, m);
  while (this.mkf.size > this.max) this.mkf.delete(this.mkf.keys().next().value);
  return m;
};
PX.Replayer.prototype.maskFrame = function (t) {
  t = Math.max(0, Math.min(this.length, t | 0));
  const k = Math.floor(t / this.every) * this.every;
  const m = new Uint8Array(this.maskKeyframe(k));
  const { idx, ends } = this.log;
  for (let j = ends[k]; j < ends[t]; j++) m[idx[j]] = 1;
  return m;
};
// Stable agent colour; mirrors canvas/world.py agent_colour exactly (FNV-1a 32, golden ratio).
PX.agentColour = function (name) {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(name)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  const hue = (h * 0.6180339887498949) % 1.0;
  const s = [0.6, 0.8, 1.0][(h >>> 8) % 3], v = [0.75, 0.9, 1.0][(h >>> 16) % 3];
  const h6 = hue * 6.0, i = Math.floor(h6) % 6, f = h6 - Math.floor(h6);
  const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  const [r, g, b] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i];
  return [Math.floor(r * 255 + 0.5), Math.floor(g * 255 + 0.5), Math.floor(b * 255 + 0.5)];
};
// Last painter per pixel at log position t (-1 = unpainted), same keyframe scheme.
PX.Replayer.prototype.ownerKeyframe = function (k) {
  this.okf = this.okf || new Map();
  const blank = () => new Int32Array(this.w * this.h).fill(-1);
  if (k === 0) return blank();
  if (this.okf.has(k)) { const v = this.okf.get(k); this.okf.delete(k); this.okf.set(k, v); return v; }
  let prev = 0;
  for (const c of this.okf.keys()) if (c < k && c > prev) prev = c;
  const o = prev === 0 ? blank() : new Int32Array(this.okf.get(prev));
  const { idx, aid, ends } = this.log;
  for (let j = ends[prev]; j < ends[k]; j++) o[idx[j]] = aid[j];
  this.okf.set(k, o);
  while (this.okf.size > this.max) this.okf.delete(this.okf.keys().next().value);
  return o;
};
PX.Replayer.prototype.ownerFrame = function (t) {
  t = Math.max(0, Math.min(this.length, t | 0));
  const k = Math.floor(t / this.every) * this.every;
  const o = new Int32Array(this.ownerKeyframe(k));
  const { idx, aid, ends } = this.log;
  for (let j = ends[k]; j < ends[t]; j++) o[idx[j]] = aid[j];
  return o;
};
// Root ancestor of a flock name (`leader-0003-painter-a-0042` -> `leader-0003`); mirrors
// canvas/world.py root_of. Children of one leader share its colour (leader territories).
PX.rootOf = function (name) { const m = /^([a-z][a-z0-9-]*?-\d{4,})-/.exec(name); return m ? m[1] : name; };
PX.Replayer.prototype.agentFrame = function (t, by = 'agent') {
  const key = by === 'parent' ? 'ppalette' : 'palette';
  if (!this[key]) this[key] = this.log.names.map((n) => PX.agentColour(by === 'parent' ? PX.rootOf(n) : n));
  this.palette = this[key];
  const o = this.ownerFrame(t), buf = new Uint8ClampedArray(this.blank);
  for (let i = 0; i < o.length; i++) {
    if (o[i] < 0) continue;
    const c = this.palette[o[i]];
    buf[4 * i] = c[0]; buf[4 * i + 1] = c[1]; buf[4 * i + 2] = c[2];
  }
  return buf;
};

// Painted pixels in magenta over a dimmed frame (RGBA, in place).
PX.overlayMask = function (rgba, mask, dim = 0.35) {
  for (let i = 0; i < mask.length; i++) {
    const o = 4 * i;
    if (mask[i]) { rgba[o] = 255; rgba[o + 1] = 0; rgba[o + 2] = 255; }
    else { rgba[o] *= dim; rgba[o + 1] *= dim; rgba[o + 2] *= dim; }
  }
  return rgba;
};

PX.gunzipText = async function (buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x1f || u8[1] !== 0x8b) return new TextDecoder().decode(u8); // already decoded
  const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
};

PX.smooth = function (xs, k = 3) {
  return xs.map((_, i) => { const w = xs.slice(Math.max(0, i - k + 1), i + 1).filter((v) => v !== null);
    return w.length ? w.reduce((a, b) => a + b, 0) / w.length : null; });
}
PX.series = function (rows) {
  // Per-row derived metrics; px/s and calls/s from deltas between driver ticks.
  return rows.map((r, i) => {
    const p = rows[i - 1], dt = p ? r.elapsed_s - p.elapsed_s : 0;
    const st = r.agents_by_state || {};
    const idle = r.idle_pct ?? (r.agents ? 100 * (st.idle || 0) / r.agents : null);
    return {
      m: r.elapsed_s / 60,
      px: p && dt > 0 ? (r.work_done - p.work_done) / dt : null,
      calls: r.paint_calls_per_s ?? (p && dt > 0 ? (r.seq - p.seq) / dt : null),
      active: r.active_agents ?? st.working ?? null,
      idle,
      gap: r.step_gap_p50_s ?? null,
      model: r.model_p50_s ?? null,
    };
  });
};
PX.summary = function (run, rows) {
  const f = rows[0], l = rows[rows.length - 1];
  if (!f || !l) return null;
  const span = Math.max(1, l.elapsed_s - f.elapsed_s), ser = PX.series(rows);
  const px = PX.smooth(ser.map((s) => s.px)).filter((v) => v !== null);
  return {
    duration_min: ((run.finished_at && run.started_at) ? (run.finished_at - run.started_at) : span) / 60,
    px_mean: (l.work_done - f.work_done) / span, px_peak: px.length ? Math.max(...px) : null,
    calls_mean: (l.seq - f.seq) / span, agents: Math.max(...rows.map((r) => r.agents || 0)),
    gpus: run.gpus ?? null, cost: (l.gpu_usd_est || 0) + (l.flock_cost_usd || 0), done: l.done_pct,
  };
};


// ------------------------------------------------- side-by-side replay core (node-tested)
// Per-entry arrays for time / completion alignment, then the parsed entries are dropped so two
// ~62k-entry logs fit comfortably in memory (only typed arrays are kept).
PX.prepare = function (text, width, workTotal) {
  const log = PX.index(PX.parseLog(text), width);
  const n = log.entries.length;
  log.ts = new Float64Array(n);
  log.eagent = new Int32Array(n);
  log.cum = new Int32Array(n + 1); // running max of done pixels (useful - overwritten)
  const ids = new Map(log.names.map((nm, i) => [nm, i]));
  let done = 0, best = 0;
  log.entries.forEach((e, i) => {
    log.ts[i] = e.ts;
    log.eagent[i] = ids.get(e.agent);
    for (const p of e.px) { if (p[3] === 'useful') done++; else if (p[3] === 'overwritten') done--; }
    best = Math.max(best, done);
    log.cum[i + 1] = best;
  });
  log.workTotal = workTotal;
  log.count = n;
  log.entries = null;
  return log;
};
PX.upper = function (arr, v, n) { // number of elements <= v (arr ascending)
  let lo = 0, hi = n;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= v) lo = m + 1; else hi = m; }
  return lo;
};
PX.duration = function (log) { return log.count ? log.ts[log.count - 1] - log.ts[0] : 0; };
// (a) wall time: entries with ts <= first paint + s (the shorter run freezes at its end).
PX.indexAtTime = function (log, s) { return log.count ? PX.upper(log.ts, log.ts[0] + s, log.count) : 0; };
// (b) completion: the first position whose done count reaches frac of the work set.
PX.indexAtDone = function (log, frac) {
  const want = Math.ceil(frac * log.workTotal);
  let lo = 0, hi = log.count;
  while (lo < hi) { const m = (lo + hi) >> 1; if (log.cum[m] >= want) hi = m; else lo = m + 1; }
  return lo;
};
PX.readout = function (log, k) {
  if (!k) return { t: 0, done: 0, pxs: 0, active: 0 };
  const now = log.ts[k - 1], k10 = PX.upper(log.ts, now - 10, k);
  const seen = new Set();
  for (let j = k10; j < k; j++) seen.add(log.eagent[j]);
  return { t: now - log.ts[0], done: 100 * log.cum[k] / log.workTotal,
           pxs: (log.cum[k] - log.cum[k10]) / 10, active: seen.size };
};

if (typeof module !== 'undefined') module.exports = PX;

// ------------------------------------------------------------------- UI
if (typeof document !== 'undefined') (() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d);
  const bust = () => `ts=${Date.now()}`;
  let runs = [], run = null, liveTimer = null, progTimer = null, replayer = null, playing = false;
  let view = 'canvas'; // canvas | mask | agents
  function liveSrc(base) {
    if (view === 'agents') return `${base}/agents.png?${bust()}`;
    if (view === 'leaders') return `${base}/agents.png?by=parent&${bust()}`;
    return `${base}/canvas.png?${bust()}`;
  }
  function setView(v) {
    view = v;
    for (const id of ['livefig', 'replayfig']) $(id).classList.toggle('masked', v === 'mask');
    if (run && run.canvas_url) {
      $('liveimg').src = liveSrc(run.canvas_url);
      if (v === 'mask') $('livemask').src = `${run.canvas_url}/mask.png?style=overlay&${bust()}`;
    }
    if (replayer) draw(+$('slider').value);
  }

  async function getJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function chart(el, rows, series, yMax) {
    const dpr = window.devicePixelRatio || 1, W = el.clientWidth, H = el.clientHeight;
    el.width = W * dpr; el.height = H * dpr;
    const c = el.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, W, H);
    const pad = { l: 36, r: 8, t: 8, b: 18 };
    c.strokeStyle = css('--line'); c.fillStyle = css('--muted'); c.font = '11px system-ui'; c.lineWidth = 1;
    if (!rows.length) { c.fillText('no data yet', pad.l, H / 2); return; }
    const x0 = rows[0].elapsed_s, x1 = Math.max(x0 + 1, rows[rows.length - 1].elapsed_s);
    const top = yMax ?? Math.max(1, ...rows.flatMap((r) => series.map((s) => s.get(r) || 0))) * 1.1;
    const X = (v) => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
    const Y = (v) => H - pad.b - v / top * (H - pad.t - pad.b);
    for (const f of [0, 0.5, 1]) {
      const y = Y(top * f); c.beginPath(); c.moveTo(pad.l, y); c.lineTo(W - pad.r, y); c.stroke();
      c.fillText(fmt(top * f, top >= 10 ? 0 : 1), 2, y + 4);
    }
    c.fillText(`${Math.round((x1 - x0) / 60)} min`, W - pad.r - 40, H - 4);
    for (const s of series) {
      c.strokeStyle = css(s.color); c.lineWidth = 2; c.beginPath();
      rows.forEach((r, i) => { const v = s.get(r); if (v === null || v === undefined) return;
        i ? c.lineTo(X(r.elapsed_s), Y(v)) : c.moveTo(X(r.elapsed_s), Y(v)); });
      c.stroke();
    }
  }

  function tiles(score, status) {
    const t = [
      ['done (work set)', `${fmt(score.coverage_pct)}%`], ['exact match', `${fmt(score.exact_match_pct)}%`],
      ['pixels to go', (score.work_total - score.work_done).toLocaleString()],
      ['MSE', fmt(score.mse, 1)], ['PSNR', score.psnr === null ? 'perfect' : `${fmt(score.psnr, 1)} dB`],
      ['useful paints', score.paints.useful.toLocaleString()], ['wrong + overwr.', (score.paints.wrong + score.paints.overwritten).toLocaleString()],
    ];
    if (status) t.push(['req/s (10 s)', fmt(status.req_per_s_10s, 1)], ['claims out', status.claims_outstanding.toLocaleString()],
      ['agents active', `${status.agents_active_60s} / ${status.agents_seen}`]);
    $('tiles').innerHTML = t.map(([k, v]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
    const ag = Object.entries(score.agents || {}).sort((a, b) => b[1].useful - a[1].useful).slice(0, 10);
    $('agents').tBodies[0].innerHTML = ag.map(([n, s]) => `<tr><td>${n}</td><td>${s.useful}</td><td>${s.duplicate}</td><td>${s.wrong}</td><td>${s.overwritten}</td><td>${s.calls}</td></tr>`).join('')
      || '<tr><td colspan="6" class="muted">no paints yet</td></tr>';
  }

  async function refreshLive() {
    if (!run || !run.canvas_url) return;
    const base = run.canvas_url;
    try {
      const [score, status] = await Promise.all([getJSON(`${base}/score?${bust()}`), getJSON(`${base}/status?${bust()}`)]);
      tiles(score, status);
      $('liveimg').src = liveSrc(base);
      if (view === 'mask') $('livemask').src = `${base}/mask.png?style=overlay&${bust()}`;
      $('livecap').textContent = `live, every 10 s · ${new Date().toLocaleTimeString()} · seq ${status.seq}`;
      $('liveerr').textContent = '';
    } catch (e) {
      $('liveerr').textContent = `canvas not reachable (${e.message}); the job may have ended.`;
    }
  }

  async function refreshProgress() {
    if (!run) return;
    let rows = [];
    try {
      const r = await fetch(`runs/${run.id}/progress.jsonl?${bust()}`, { cache: 'no-store' });
      if (r.ok) rows = (await r.text()).split('\n').filter(Boolean).map((l) => JSON.parse(l));
    } catch (e) { /* not uploaded yet */ }
    chart($('chart'), rows, [{ get: (r) => r.done_pct, color: '--a' }, { get: (r) => r.exact_pct, color: '--b' }], 100);
    chart($('chart2'), rows, [{ get: (r) => r.req_s, color: '--a' }, { get: (r) => (r.agents_by_state || {}).working, color: '--b' }]);
    const last = rows[rows.length - 1];
    $('flock').textContent = last
      ? `flock (driver, ${new Date(last.ts * 1000).toLocaleTimeString()}): agents ${last.agents} ` +
        `(${Object.entries(last.agents_by_state || {}).map(([k, v]) => `${k} ${v}`).join(', ') || '-'}), ` +
        `turns ${last.turns ?? '-'}, pending prompts ${last.pending_prompts ?? '-'}, ` +
        `cost flock $${fmt(last.flock_cost_usd)} + GPU ~$${fmt(last.gpu_usd_est)}`
      : 'no driver progress uploaded yet';
  }

  async function loadReplay() {
    $('replaysec').hidden = false;
    const base = run.final;
    $('finalimg').src = `${base}/canvas.png`;
    try {
      const score = await getJSON(`${base}/score.json`);
      if (run.status !== 'running') tiles(score, null);
      const tgt = score.target || {};
      const w = score.width, h = score.height;
      const blank = (tgt.blank || '000000').match(/../g).map((x) => parseInt(x, 16));
      $('replaycap').textContent = 'downloading paint log...';
      const r = await fetch(`${base}/paint-log.jsonl.gz`);
      if (!r.ok) throw new Error(`paint log: HTTP ${r.status}`);
      const text = await PX.gunzipText(await r.arrayBuffer());
      const log = PX.index(PX.parseLog(text), w);
      replayer = new PX.Replayer(log, w, h, blank);
      const cv = $('replay'); cv.width = w; cv.height = h;
      $('slider').max = replayer.length; $('slider').value = replayer.length;
      draw(replayer.length);
    } catch (e) { $('replaycap').textContent = `replay unavailable: ${e.message}`; }
  }

  function draw(t) {
    if (!replayer) return;
    const buf = view === 'agents' ? replayer.agentFrame(t)
      : view === 'leaders' ? replayer.agentFrame(t, 'parent') : replayer.frame(t);
    if (view === 'mask') PX.overlayMask(buf, replayer.maskFrame(t));
    $('replay').getContext('2d').putImageData(new ImageData(buf, replayer.w, replayer.h), 0, 0);
    const e = replayer.log.entries[t - 1];
    $('replaycap').textContent = `t = ${t} / ${replayer.length}`;
    const counts = {};
    if (e) for (const p of e.px) counts[p[3]] = (counts[p[3]] || 0) + 1;
    $('entry').textContent = e ? `#${e.seq} ${e.agent} · ${new Date(e.ts * 1000).toISOString().slice(11, 19)} UTC · ` +
      Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ') : '';
  }

  function links() {
    const l = [];
    if (run.controller_url) l.push(['controller status', run.controller_url]);
    if (run.canvas_url) for (const p of ['canvas.png', 'score', 'status', 'replay', 'log']) l.push([`canvas /${p}`, `${run.canvas_url}/${p}`]);
    if (run.board_url) l.push(['board thread', run.board_url]);
    if (run.final) l.push(['final score.json', `${run.final}/score.json`], ['paint log (gz)', `${run.final}/paint-log.jsonl.gz`]);
    $('links').innerHTML = '<h2>Links</h2>' + (l.map(([k, u]) => `<a href="${u}" target="_blank" rel="noopener">${k}</a>`).join('') || '<span class="muted">none</span>');
  }

  function select(id) {
    run = runs.find((r) => r.id === id) || runs[runs.length - 1];
    clearInterval(liveTimer); clearInterval(progTimer); replayer = null; playing = false;
    $('pill').textContent = run.status; $('pill').className = `pill ${run.status === 'running' ? 'running' : ''}`;
    $('when').textContent = `started ${new Date(run.started_at * 1000).toLocaleString()}` +
      (run.finished_at ? ` · ended ${new Date(run.finished_at * 1000).toLocaleString()}` : '');
    $('live').hidden = !run.canvas_url;
    if (run.canvas_url) $('targetimg').src = `${run.canvas_url}/target.png`;
    $('replaysec').hidden = true;
    links();
    refreshProgress(); progTimer = setInterval(refreshProgress, 30000);
    if (run.status === 'running') { refreshLive(); liveTimer = setInterval(refreshLive, 10000); }
    else if (run.canvas_url) refreshLive(); // may still answer if the job is up
    if (run.final) loadReplay();
  }

  // ---------------------------------------------------------------- compare
  function chartXY(el, sets) {
    const dpr = window.devicePixelRatio || 1, W = el.clientWidth, H = el.clientHeight;
    el.width = W * dpr; el.height = H * dpr;
    const c = el.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, W, H);
    const pad = { l: 40, r: 8, t: 8, b: 18 };
    const pts = sets.flatMap((s) => s.xs.map((x, i) => [x, s.ys[i]]).filter((q) => q[1] !== null));
    c.fillStyle = css('--muted'); c.strokeStyle = css('--line'); c.font = '11px system-ui'; c.lineWidth = 1;
    if (!pts.length) { c.fillText('no data', pad.l, H / 2); return; }
    // Short runs (the 1k run paints for ~3 min): seconds on the axis below 10 minutes.
    const span = Math.max(...pts.map((q) => q[0])), secs = span < 10;
    const x1 = Math.max(secs ? 0.05 : 1, span), top = Math.max(1e-9, ...pts.map((q) => q[1])) * 1.1;
    const X = (v) => pad.l + v / x1 * (W - pad.l - pad.r), Y = (v) => H - pad.b - v / top * (H - pad.t - pad.b);
    for (const f of [0, 0.5, 1]) { const y = Y(top * f); c.beginPath(); c.moveTo(pad.l, y); c.lineTo(W - pad.r, y); c.stroke();
      c.fillText(fmt(top * f, top >= 10 ? 0 : 1), 2, y + 4); }
    c.fillText(secs ? `${fmt(x1 * 60, 0)} s` : `${fmt(x1, 0)} min`, W - pad.r - 44, H - 4);
    for (const s of sets) {
      c.strokeStyle = css(s.color); c.lineWidth = 2; c.setLineDash(s.dash ? [5, 4] : []); c.beginPath();
      let on = false;
      s.xs.forEach((x, i) => { const y = s.ys[i]; if (y === null) { on = false; return; }
        on ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)); on = true; });
      c.stroke();
    }
    c.setLineDash([]);
  }
  async function loadRows(r) {
    try {
      const res = await fetch(`runs/${r.id}/progress.jsonl?${bust()}`, { cache: 'no-store' });
      return res.ok ? (await res.text()).split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
    } catch (e) { return []; }
  }
  async function renderCompare() {
    const A = runs.find((r) => r.id === $('cmpA').value), B = runs.find((r) => r.id === $('cmpB').value);
    const [ra, rb] = await Promise.all([A ? loadRows(A) : [], B ? loadRows(B) : []]);
    const sa = PX.series(ra), sb = PX.series(rb);
    const sets = (key, sm) => [
      { xs: sa.map((s) => s.m), ys: sm ? PX.smooth(sa.map((s) => s[key])) : sa.map((s) => s[key]), color: '--a' },
      { xs: sb.map((s) => s.m), ys: sm ? PX.smooth(sb.map((s) => s[key])) : sb.map((s) => s[key]), color: '--b', dash: true },
    ];
    chartXY($('cPx'), sets('px', true)); chartXY($('cCalls'), sets('calls', true));
    chartXY($('cActive'), sets('active')); chartXY($('cIdle'), sets('idle'));
    chartXY($('cLat'), [...sets('gap'), ...sets('model').map((s) => ({ ...s, color: s.color === '--a' ? '--ok' : '--bad' }))]);
    const ua = A && PX.summary(A, ra), ub = B && PX.summary(B, rb);
    const rowsT = [['', A ? A.name || A.id : '-', B ? B.name || B.id : '-'],
      ['duration (min)', ua && fmt(ua.duration_min, 1), ub && fmt(ub.duration_min, 1)],
      ['done %', ua && fmt(ua.done, 1), ub && fmt(ub.done, 1)],
      ['px/s mean', ua && fmt(ua.px_mean, 0), ub && fmt(ub.px_mean, 0)],
      ['px/s peak (30 s avg)', ua && fmt(ua.px_peak, 0), ub && fmt(ub.px_peak, 0)],
      ['paint calls/s mean', ua && fmt(ua.calls_mean, 1), ub && fmt(ub.calls_mean, 1)],
      ['agents', ua && ua.agents, ub && ub.agents], ['GPUs', ua && (ua.gpus ?? '-'), ub && (ub.gpus ?? '-')],
      ['cost (GPU est + flock), $', ua && fmt(ua.cost), ub && fmt(ub.cost)]];
    $('cmpTable').innerHTML = rowsT.map((r, i) => `<tr>${r.map((v) => i ? `<td>${v ?? '-'}</td>` : `<th>${v}</th>`).join('')}</tr>`).join('');
    for (const [run, f, a, cap] of [[A, 'fA', 'aA', 'fAc'], [B, 'fB', 'aB', 'fBc']]) {
      const base = run && run.final;
      $(f).src = base ? `${base}/canvas.png` : (run && run.canvas_url ? `${run.canvas_url}/canvas.png?${bust()}` : '');
      $(a).onerror = () => { $(a).removeAttribute('src'); $(a).alt = 'no agents.png for this run (older run); the Side by side tab computes the agent map from its log'; };
      $(a).src = base ? `${base}/agents.png` : (run && run.canvas_url ? `${run.canvas_url}/agents.png?${bust()}` : '');
      $(cap).textContent = run ? `${run.name || run.id} (${run.status})` : '-';
    }
  }

  // ------------------------------------------------------------- side by side
  const SBS = { logs: [null, null], reps: [null, null], ids: [null, null], playing: false, loading: 0 };
  function sbsMax() {
    const [a, b] = SBS.logs;
    if ($('sbsAlign').value === 'pct') return 1000;
    return Math.ceil(Math.max(a ? PX.duration(a) : 0, b ? PX.duration(b) : 0));
  }
  function mmss(t) { const s = Math.max(0, Math.round(t)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
  function sbsDraw() {
    const v = +$('sbsSlider').value, pct = $('sbsAlign').value === 'pct';
    for (const i of [0, 1]) {
      const log = SBS.logs[i], rep = SBS.reps[i], side = i ? 'R' : 'L';
      if (!log || !rep) continue;
      const k = pct ? PX.indexAtDone(log, v / 1000) : PX.indexAtTime(log, v);
      const by = $('sbsAgents').value;
      const buf = by === 'canvas' ? rep.frame(k) : rep.agentFrame(k, by);
      $(`sbs${side}`).getContext('2d').putImageData(new ImageData(buf, rep.w, rep.h), 0, 0);
      const r = PX.readout(log, k);
      $(`sbs${side}r`).textContent = `t ${mmss(r.t)} · done ${fmt(r.done, 1)}% · ` +
        `${fmt(r.pxs, 0)} px/s (last 10 s) · ${r.active} active agents (last 10 s)` +
        (k >= log.count ? ' · finished' : '');
    }
    $('sbsPos').textContent = pct ? `${fmt(v / 10, 1)}% done` : `${mmss(v)} since first paint`;
  }
  async function sbsLoad(i, id) {
    const r = runs.find((x) => x.id === id), side = i ? 'R' : 'L';
    SBS.logs[i] = SBS.reps[i] = null; SBS.ids[i] = id;
    $(`sbs${side}c`).textContent = r ? `${r.name || r.id}` : '-';
    if (!r || !r.final) { $(`sbs${side}r`).textContent = 'no final/ paint log for this run'; return; }
    $(`sbs${side}r`).textContent = 'loading paint log...';
    try {
      const score = await getJSON(`${r.final}/score.json`);
      const blank = ((score.target || {}).blank || 'ffffff').match(/../g).map((x) => parseInt(x, 16));
      const res = await fetch(`${r.final}/paint-log.jsonl.gz`);
      if (!res.ok) throw new Error(`paint log: HTTP ${res.status}`);
      const text = await PX.gunzipText(await res.arrayBuffer());
      if (SBS.ids[i] !== id) return; // changed meanwhile
      const log = PX.prepare(text, score.width, score.work_total);
      SBS.logs[i] = log;
      SBS.reps[i] = new PX.Replayer(log, score.width, score.height, blank, 2000, 4);
      const cv = $(`sbs${side}`); cv.width = score.width; cv.height = score.height;
    } catch (e) { $(`sbs${side}r`).textContent = `unavailable: ${e.message}`; }
    $('sbsSlider').max = sbsMax();
    sbsDraw();
  }
  function sbsInit() {
    if ($('sbsA').options.length) return;
    const opts = runs.slice().reverse().filter((r) => r.final)
      .map((r) => `<option value="${r.id}">${r.name || r.id}</option>`).join('');
    $('sbsA').innerHTML = opts; $('sbsB').innerHTML = opts;
    const has = (id) => runs.some((r) => r.id === id && r.final);
    const fin = runs.filter((r) => r.final);
    $('sbsA').value = has('run1-20261019') ? 'run1-20261019' : fin[fin.length - 2]?.id ?? fin[0]?.id ?? '';
    $('sbsB').value = has('run1k-20261019') ? 'run1k-20261019' : fin[fin.length - 1]?.id ?? '';
    sbsLoad(0, $('sbsA').value); sbsLoad(1, $('sbsB').value);
  }
  $('sbsA').addEventListener('change', (e) => sbsLoad(0, e.target.value));
  $('sbsB').addEventListener('change', (e) => sbsLoad(1, e.target.value));
  $('sbsSlider').addEventListener('input', sbsDraw);
  $('sbsAgents').addEventListener('change', sbsDraw);
  $('sbsAlign').addEventListener('change', () => {
    const was = +$('sbsSlider').value / Math.max(1, +$('sbsSlider').max);
    $('sbsSlider').max = sbsMax(); $('sbsSlider').value = Math.round(was * sbsMax()); sbsDraw();
  });
  $('sbsPlay').addEventListener('click', () => {
    SBS.playing = !SBS.playing; $('sbsPlay').textContent = SBS.playing ? 'Pause' : 'Play';
    if (SBS.playing && +$('sbsSlider').value >= +$('sbsSlider').max) $('sbsSlider').value = 0;
    let last = performance.now();
    const step = (now) => {
      if (!SBS.playing) return;
      const dt = (now - last) / 1000; last = now;
      const speed = +$('sbsSpeed').value, max = +$('sbsSlider').max;
      // time mode: speed x real time; % mode: the longer run's real pace x speed
      const longest = Math.max(1, ...SBS.logs.filter(Boolean).map(PX.duration));
      const inc = $('sbsAlign').value === 'pct' ? speed * dt * 1000 / longest : speed * dt;
      SBS.acc = (SBS.acc || 0) + inc;
      if (SBS.acc >= 1) { $('sbsSlider').value = Math.min(max, +$('sbsSlider').value + Math.floor(SBS.acc)); SBS.acc %= 1; sbsDraw(); }
      if (+$('sbsSlider').value >= max) { SBS.playing = false; $('sbsPlay').textContent = 'Play'; return; }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
  function showTab(name) {
    const cmp = name === 'compare', sbs = name === 'sbs', other = cmp || sbs;
    $('tabRun').classList.toggle('on', !other); $('tabCompare').classList.toggle('on', cmp);
    $('tabSbs').classList.toggle('on', sbs);
    $('compare').hidden = !cmp; $('sbs').hidden = !sbs; $('main').hidden = other || !runs.length;
    for (const id of ['run', 'pill', 'when']) $(id).style.display = other ? 'none' : '';
    if (sbs) sbsInit();
    if (cmp) {
      const opts = runs.slice().reverse().map((r) => `<option value="${r.id}">${r.name || r.id} (${r.status})</option>`).join('');
      if (!$('cmpA').options.length) {
        $('cmpA').innerHTML = opts; $('cmpB').innerHTML = opts;
        $('cmpA').value = runs[runs.length - 1]?.id ?? ''; $('cmpB').value = runs[runs.length - 2]?.id ?? runs[0]?.id ?? '';
      }
      renderCompare();
    }
  }
  $('tabRun').addEventListener('click', () => showTab('run'));
  $('tabCompare').addEventListener('click', () => showTab('compare'));
  $('tabSbs').addEventListener('click', () => showTab('sbs'));
  $('cmpA').addEventListener('change', renderCompare);
  $('cmpB').addEventListener('change', renderCompare);

  $('run').addEventListener('change', (e) => select(e.target.value));
  $('view').addEventListener('change', (e) => setView(e.target.value));
  $('slider').addEventListener('input', (e) => draw(+e.target.value));
  $('play').addEventListener('click', () => {
    playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play';
    if (playing && +$('slider').value >= replayer?.length) $('slider').value = 0;
    const tick = () => {
      if (!playing || !replayer) return;
      const t = Math.min(replayer.length, +$('slider').value + +$('speed').value);
      $('slider').value = t; draw(t);
      if (t >= replayer.length) { playing = false; $('play').textContent = 'Play'; return; }
      requestAnimationFrame(tick);
    };
    tick();
  });

  (async () => {
    try { runs = await getJSON(`runs.json?${bust()}`); } catch (e) { runs = []; }
    if (!Array.isArray(runs) || !runs.length) { $('empty').hidden = false; return; }
    $('main').hidden = false;
    $('run').innerHTML = runs.slice().reverse().map((r) => `<option value="${r.id}">${r.name || r.id} (${r.status})</option>`).join('');
    const qp = new URLSearchParams(location.search);
    const want = qp.get('run');
    if (['compare', 'sbs'].includes(qp.get('tab'))) setTimeout(() => showTab(qp.get('tab')), 0);
    select(want || runs[runs.length - 1].id);
    $('run').value = run.id;
    setInterval(async () => { // pick up status changes (finished, new final files)
      try {
        const fresh = await getJSON(`runs.json?${bust()}`);
        const f = fresh.find((r) => r.id === run.id);
        if (fresh.length !== runs.length || (f && f.status !== run.status)) { runs = fresh; select(run.id); }
      } catch (e) { /* keep the old list */ }
    }, 60000);
  })();
})();
