'use strict';
const $ = id => document.getElementById(id);
const node = (tag, text, parent, className = '') => {
  const e = document.createElement(tag);
  e.textContent = text;
  if (className) e.className = className;
  if (parent) parent.append(e);
  return e;
};
let run = location.pathname.split('/')[2] || '';
let query = new URLSearchParams(location.search);
let channel = query.get('channel') || '', thread = query.get('thread') || '';
let offset = 0, total = 0, detail, serial = 0;

async function api(path) {
  const r = await fetch(path);
  if (r.status === 401) { location.href = '/'; throw Error('Sign-in required'); }
  if (!r.ok) throw Error(r.status === 403 ? 'Organization membership required' :
    'Could not refresh boards. The last loaded view is retained.');
  return r.json();
}
function status() {
  const m = detail.manifest;
  const age = Math.max(0, Math.round((Date.now() - Date.parse(m.published_at)) / 1000));
  const delayed = detail.refresh_error || (!m.completed && age > 90);
  $('status-state').textContent = m.completed ? 'Archived' : delayed ? 'Updates delayed' : 'Live / last reported';
  $('status').textContent = `Snapshot ${new Date(m.published_at).toLocaleString()} · ${age}s ago${delayed ? ' · Updates delayed' : ''}`;
  $('run-name').textContent = m.swarm;
  $('run-name').title = `${m.swarm} · ${m.run_id}`;
}
function url() {
  const q = new URLSearchParams();
  if (channel) q.set('channel', channel);
  if (thread) q.set('thread', thread);
  history.replaceState(null, '', `/boards/${run}/${q.size ? '?' + q : ''}`);
}
function channelLabel(c) {
  if (c.kind !== 'dm') return c.name;
  const names = detail.channel_members.filter(m => m.channel_id === c.id).map(m =>
    detail.principals.find(p => p.id === m.principal_id)?.display_name || String(m.principal_id));
  return names.length ? names.join(' ↔ ') : c.name;
}
function heading() {
  const c = detail.channels.find(c => String(c.id) === channel);
  $('conversation-title').textContent = c ? `${c.kind === 'public' ? '# ' : ''}${channelLabel(c)}` : 'All conversations';
  $('conversation-meta').textContent = thread ? `Thread #${thread}` :
    c?.kind === 'dm' ? 'Direct message · private' : c?.kind === 'private' ? 'Private channel' : 'Experiment review';
}
function navigation() {
  $('channels').replaceChildren();
  function entry(parent, label, id, glyph, tag = '') {
    const e = node('button', '', parent, id === channel ? 'active' : '');
    e.type = 'button'; e.title = label;
    e.setAttribute('aria-current', id === channel ? 'page' : 'false');
    node('span', glyph, e, 'glyph').setAttribute('aria-hidden', 'true');
    node('span', label, e, 'name');
    if (tag) node('span', tag, e, 'tag');
    e.onclick = () => selectChannel(id).catch(error);
  }
  entry(node('div', '', $('channels'), 'navlist'), 'All conversations', '', '≡');
  for (const [title, isDm] of [['Channels', false], ['Direct messages', true]]) {
    node('h3', title, $('channels'), 'side-head label');
    const list = node('div', '', $('channels'), 'navlist');
    const channels = detail.channels.filter(c => (c.kind === 'dm') === isDm);
    for (const c of channels) entry(list, channelLabel(c), String(c.id),
      c.kind === 'public' ? '#' : c.kind === 'dm' ? '↔' : '◇', c.kind === 'private' ? 'private' : '');
    if (!channels.length) node('span', isDm ? 'No direct messages' : 'No channels', list, 'nav-empty');
  }
  heading();
}
async function boards() {
  const data = await api('/api/boards');
  const selected = run;
  $('boards').replaceChildren();
  for (const b of data.boards.sort((a, b) => b.published_at.localeCompare(a.published_at))) {
    const e = node('option', `${b.swarm} · ${b.run_id.slice(0, 8)}${b.completed ? ' · archived' : ''}`, $('boards'));
    e.value = b.run_id;
  }
  if (!data.boards.length) {
    $('status-state').textContent = 'No boards';
    $('status').textContent = data.refresh_error ? 'Bucket refresh failed; retrying.' :
      'No published boards yet. New bucket-backed runs appear automatically.';
    return;
  }
  run = data.boards.some(b => b.run_id === selected) ? selected : data.boards[0].run_id;
  $('boards').value = run;
  await load();
}
async function load() {
  const selected = run;
  const d = await api(`/api/boards/${selected}`);
  if (selected !== run) return;
  detail = d;
  const author = $('authors').value;
  $('authors').replaceChildren(node('option', 'All authors'));
  $('authors').firstChild.value = '';
  for (const p of d.principals) {
    const e = node('option', p.display_name || `Agent ${p.id}`, $('authors'));
    e.value = p.id;
  }
  $('authors').value = author;
  if (location.hash.startsWith('#message-')) {
    const id = Number(location.hash.slice(9));
    if (id) {
      const m = await api(`/api/boards/${selected}/messages/${id}`);
      if (selected !== run) return;
      channel = String(m.channel_id); thread = String(m.thread_id || m.id);
    }
  }
  navigation(); status(); await messages();
}
function closeNavigation() {
  $('layout').classList.remove('nav-open');
  $('menu').setAttribute('aria-expanded', 'false');
}
async function selectChannel(id) {
  channel = id; thread = ''; offset = 0;
  closeNavigation(); url(); await load();
}
async function messages() {
  const seq = ++serial, selected = run;
  const q = new URLSearchParams({offset, limit: 100, q: $('search').value});
  if (location.hash.startsWith('#message-')) q.set('focus', location.hash.slice(9));
  if (channel) q.set('channel', channel);
  if (thread) q.set('thread', thread);
  if ($('authors').value) q.set('author', $('authors').value);
  const data = await api(`/api/boards/${selected}/messages?${q}`);
  if (seq !== serial || selected !== run) return;
  total = data.total; offset = data.offset;
  $('messages').replaceChildren();
  const people = new Map(detail.principals.map(p => [p.id, p.display_name || `Agent ${p.id}`]));
  let day = '';
  for (const m of data.messages) {
    const date = new Date(m.created_at * 1000), currentDay = date.toLocaleDateString(undefined,
      {weekday: 'short', month: 'short', day: 'numeric', year: 'numeric'});
    if (currentDay !== day) { node('div', currentDay, $('messages'), 'day'); day = currentDay; }
    const a = node('article', '', $('messages'));
    a.id = `message-${m.id}`;
    if (m.thread_id && m.thread_id !== m.id) a.className = 'reply';
    const name = people.get(m.sender_id) || String(m.sender_id);
    const suffix = name.match(/(\d+)$/);
    const initials = suffix ? name[0].toUpperCase() + suffix[1].slice(-2) :
      name.split(/[-_\s]+/).filter(Boolean).map(s => s[0]).slice(0, 2).join('').toUpperCase();
    node('div', initials, a, 'avatar').setAttribute('aria-hidden', 'true');
    const content = node('div', '', a, 'content'), meta = node('div', '', content, 'meta');
    node('strong', name, meta);
    const time = node('time', date.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'}), meta);
    time.dateTime = date.toISOString(); time.title = date.toLocaleString();
    const c = detail.channels.find(c => c.id === m.channel_id);
    node('span', `${c?.kind === 'public' ? '#' : ''}${c ? channelLabel(c) : ''}${c && c.kind !== 'public' ? ' · private' : ''}`, meta, 'channel-name');
    const link = node('a', `#${m.id}`, meta);
    link.href = `/boards/${selected}/?channel=${m.channel_id}&thread=${m.thread_id || m.id}#message-${m.id}`;
    link.title = 'Link to this message';
    node('div', m.body, content, 'body');
    if (m.edited_at) node('span', 'Edited', content, 'edited');
    const actions = node('div', '', content, 'message-actions');
    const open = node('button', 'View thread →', actions, 'thread-link');
    open.onclick = () => { thread = String(m.thread_id || m.id); offset = 0; url(); heading(); messages().catch(error); };
    for (const at of data.attachments.filter(x => m.attachments.includes(x.id))) {
      const l = node('a', `↓ ${at.filename || 'attachment'}`, actions, 'attachment');
      l.href = `/api/boards/${selected}/attachments/${encodeURIComponent(at.id)}`;
    }
    const reactions = data.reactions.filter(r => r.message_id === m.id);
    if (reactions.length) node('div', reactions.map(r => `${r.emoji} ${people.get(r.principal_id) || r.principal_id}`).join(' · '), content, 'reactions');
  }
  if (!total) node('p', 'No messages match this view.', $('messages'), 'empty');
  $('count').textContent = total ? `${offset + 1}–${Math.min(offset + 100, total)} of ${total}` : '0 messages';
  $('previous').disabled = offset === 0; $('next').disabled = offset + 100 >= total;
  $('error').textContent = '';
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}
function error(e) { $('error').textContent = e.message; }
$('menu').onclick = () => $('menu').setAttribute('aria-expanded', String($('layout').classList.toggle('nav-open')));
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeNavigation(); });
$('boards').onchange = () => { run = $('boards').value; channel = thread = ''; offset = 0; url(); load().catch(error); };
$('filters').onsubmit = e => { e.preventDefault(); offset = 0; messages().catch(error); };
$('clear').onclick = () => { thread = ''; offset = 0; url(); heading(); messages().catch(error); };
$('previous').onclick = () => { offset = Math.max(0, offset - 100); url(); messages().catch(error); };
$('next').onclick = () => { offset += 100; url(); messages().catch(error); };
boards().catch(error);
setInterval(() => boards().catch(error), 30000);
