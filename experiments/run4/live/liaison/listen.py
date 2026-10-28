#!/usr/bin/env python3 -I
"""Long-poll the board; print new relevant messages and exit when any arrive. Cursor persisted in cursor.txt."""
import json, os, sys, time, urllib.request, urllib.error, datetime
D=os.path.dirname(os.path.abspath(__file__))
TOK=open(os.path.join(D,"token")).read().strip()
B="https://lvwerra-msgboard-demo.hf.space/v1"
CUR=os.path.join(D,"cursor.txt")
ME=133
max_wait=float(sys.argv[1]) if len(sys.argv)>1 else 3300  # seconds before giving up
def get(url):
    req=urllib.request.Request(url,headers={"Authorization":f"Bearer {TOK}"})
    return json.load(urllib.request.urlopen(req,timeout=60))
cursor=open(CUR).read().strip() if os.path.exists(CUR) else ""
if not cursor:
    d=get(f"{B}/activity?channels=joined"); cursor=str(d["next_cursor"]); open(CUR,"w").write(cursor)
t0=time.time(); hits=[]
while time.time()-t0<max_wait:
    try: d=get(f"{B}/activity?after={cursor}&wait=25&channels=joined")
    except urllib.error.HTTPError as e:
        wait=int(e.headers.get("Retry-After") or 30) if e.code==429 else 15
        print("poll error",e.code,"sleep",wait); time.sleep(wait); continue
    except Exception as e: print("poll error",e); time.sleep(15); continue
    for m in d.get("messages",[]):
        s=m.get("sender",{})
        if s.get("id")==ME: continue
        t=datetime.datetime.utcfromtimestamp(m.get("created_at",0)).strftime('%H:%M')
        hits.append(f"[{m.get('reason')}|{m.get('channel',{}).get('name')}|id{m['id']}|th{m.get('thread_id')}] {t} {s.get('display_name') or s.get('external_id')}: {m.get('body','')}")
    cursor=str(d["next_cursor"]); open(CUR,"w").write(cursor)
    if hits:
        print("\n----\n".join(hits)); sys.exit(0)
    if d.get("has_more"): continue
print("no messages in",int(time.time()-t0),"s"); sys.exit(3)
