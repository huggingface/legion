'use strict';
const $ = id => document.getElementById(id);
let run = location.pathname.split('/')[2] || '', channel = '', thread = '', offset = 0, total = 0, detail, serial = 0;
let query = new URLSearchParams(location.search);
channel = query.get('channel') || ''; thread = query.get('thread') || '';
const node = (tag, text, parent) => { const e = document.createElement(tag); e.textContent = text; if(parent) parent.append(e); return e; };
async function api(path) { const r = await fetch(path); if(r.status===401){location.href='/auth/login'; throw Error('Sign-in required');} if(!r.ok) throw Error(r.status===403?'Organization membership required':'Could not refresh boards. The last loaded view is retained.'); return r.json(); }
function status() { const m=detail.manifest; const age=Math.max(0,Math.round((Date.now()-Date.parse(m.published_at))/1000)); $('status').textContent=`${m.swarm} · ${m.run_id.slice(0,8)} · ${m.completed?'Archived':'Live / last reported'} · Snapshot ${new Date(m.published_at).toLocaleString()} (${age}s ago)${detail.refresh_error || (!m.completed && age>90)?' · Updates delayed':''}`; }
function url() { const q=new URLSearchParams(); if(channel)q.set('channel',channel); if(thread)q.set('thread',thread); history.replaceState(null,'',`/boards/${run}/?${q}`); }
async function boards() {
 const data=await api('/api/boards'); const selected=run;
 $('boards').replaceChildren();
 for(const b of data.boards.sort((a,b)=>b.published_at.localeCompare(a.published_at))){const e=node('option',`${b.swarm} · ${b.run_id.slice(0,8)}${b.completed?' · archived':''}`,$('boards')); e.value=b.run_id;}
 if(!data.boards.length){$('status').textContent=data.refresh_error?'Bucket refresh failed; retrying.':'No published boards yet. New bucket-backed runs appear automatically.';return;}
 run=data.boards.some(b=>b.run_id===selected)?selected:data.boards[0].run_id; $('boards').value=run; await load();
}
async function load() {
 const selected=run; const d=await api(`/api/boards/${selected}`); if(selected!==run)return; detail=d;
 $('channels').replaceChildren(); const all=node('button','All conversations',$('channels')); all.onclick=()=>selectChannel('');
 for(const c of d.channels){const names=d.channel_members.filter(m=>m.channel_id===c.id).map(m=>d.principals.find(p=>p.id===m.principal_id)?.display_name||String(m.principal_id));const label=c.kind==='dm' && names.length?names.join(' ↔ '):c.name;const e=node('button',`${c.kind==='public'?'#':'🔒'} ${label}${c.kind==='dm'?' (DM)':''}`,$('channels'));e.className=String(c.id)===channel?'active':'';e.onclick=()=>selectChannel(String(c.id));}
 const author=$('authors').value; $('authors').replaceChildren(node('option','All authors')); $('authors').firstChild.value='';
 for(const p of d.principals){const e=node('option',p.display_name || `Agent ${p.id}`,$('authors'));e.value=p.id;} $('authors').value=author;
 if(location.hash.startsWith('#message-')){const id=Number(location.hash.slice(9));if(id){const m=await api(`/api/boards/${run}/messages/${id}`);channel=String(m.channel_id);thread=String(m.thread_id||m.id);}}
 status(); await messages();
}
async function selectChannel(id){channel=id;thread='';offset=0;url();await load();}
async function messages(){
 const seq=++serial; const q=new URLSearchParams({offset,limit:100,q:$('search').value});
 if(location.hash.startsWith('#message-'))q.set('focus',location.hash.slice(9));
 if(channel)q.set('channel',channel);if(thread)q.set('thread',thread);if($('authors').value)q.set('author',$('authors').value);
 const data=await api(`/api/boards/${run}/messages?${q}`);if(seq!==serial)return;
 total=data.total;offset=data.offset;$('messages').replaceChildren();const people=new Map(detail.principals.map(p=>[p.id,p.display_name||`Agent ${p.id}`]));
 for(const m of data.messages){const a=node('article','',$('messages'));a.id=`message-${m.id}`;if(m.thread_id && m.thread_id!==m.id)a.className='reply';const meta=node('div','',a);meta.className='meta';node('strong',people.get(m.sender_id)||String(m.sender_id),meta);node('span',new Date(m.created_at*1000).toLocaleString(),meta);const c=detail.channels.find(c=>c.id===m.channel_id);node('span',`${c?.name||''} ${c?.kind!=='public'?'· private':''}`,meta);const link=node('a',`#${m.id}`,meta);link.href=`/boards/${run}/?channel=${m.channel_id}&thread=${m.thread_id||m.id}#message-${m.id}`;const body=node('div',m.body,a);body.className='body';if(m.edited_at)node('small','Edited',a);
 const open=node('button','View thread',a);open.onclick=()=>{thread=String(m.thread_id||m.id);offset=0;url();messages().catch(error);};
 for(const at of data.attachments.filter(x=>m.attachments.includes(x.id))){const l=node('a',` 📎 ${at.filename||'attachment'}`,a);l.href=`/api/boards/${run}/attachments/${encodeURIComponent(at.id)}`;}
 const reactions=data.reactions.filter(r=>r.message_id===m.id);if(reactions.length)node('div',reactions.map(r=>`${r.emoji} ${people.get(r.principal_id)||r.principal_id}`).join(' · '),a);
 }
 if(!total)node('p','No messages match this view.',$('messages'));
 $('count').textContent=total?`${offset+1}–${Math.min(offset+100,total)} of ${total}`:'0 messages';$('previous').disabled=offset===0;$('next').disabled=offset+100>=total;$('error').textContent='';
 if(location.hash)document.getElementById(location.hash.slice(1))?.scrollIntoView();
}
function error(e){$('error').textContent=e.message;}
$('boards').onchange=()=>{run=$('boards').value;channel=thread='';offset=0;url();load().catch(error);};
$('filters').onsubmit=e=>{e.preventDefault();offset=0;messages().catch(error);};
$('clear').onclick=()=>{thread='';offset=0;url();messages().catch(error);};
$('previous').onclick=()=>{offset=Math.max(0,offset-100);url();messages().catch(error);};
$('next').onclick=()=>{offset+=100;url();messages().catch(error);};
boards().catch(error);setInterval(()=>boards().catch(error),30000);
