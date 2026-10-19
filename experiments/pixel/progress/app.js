// flock pixel progress page. No build step, no libraries.
// Core (parseLog, Replayer) is DOM-free so node can test it; the UI runs only in a browser.
const PX = {};

PX.parseLog = function (text) {
  const lines = text.split('\n').filter(Boolean);
  let total = 0;
  const entries = lines.map((l) => JSON.parse(l));
  for (const e of entries) total += e.px.length;
  const idx = new Int32Array(total), rgb = new Uint8Array(total * 3);
  const ends = new Int32Array(entries.length + 1);
  const meta = new Array(entries.length + 1);
  return { entries, idx, rgb, ends, meta, total, filled: false };
};

PX.index = function (log, width) {
  // Fill idx/rgb/ends once; refused pixels were never applied, so they are skipped.
  let k = 0;
  log.entries.forEach((e, i) => {
    for (const p of e.px) {
      if (p[3] === 'refused') continue;
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
  get length() { return this.log.entries.length; }
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

PX.gunzipText = async function (buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x1f || u8[1] !== 0x8b) return new TextDecoder().decode(u8); // already decoded
  const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).text();
};

if (typeof module !== 'undefined') module.exports = PX;

// ------------------------------------------------------------------- UI
if (typeof document !== 'undefined') (() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d);
  const bust = () => `ts=${Date.now()}`;
  let runs = [], run = null, liveTimer = null, progTimer = null, replayer = null, playing = false;

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
      $('liveimg').src = `${base}/canvas.png?${bust()}`;
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
    const buf = replayer.frame(t);
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

  $('run').addEventListener('change', (e) => select(e.target.value));
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
    const want = new URLSearchParams(location.search).get('run');
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
