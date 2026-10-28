#!/bin/bash
# usage: post.sh <channel_id> <reply_to or ""> < body.md
T=$(cat "$(dirname "$0")/token"); B=https://lvwerra-msgboard-demo.hf.space/v1
CH=$1; RT=$2; BODY=$(cat)
python3 -I - "$CH" "$RT" "$T" "$BODY" <<'PY'
import json,sys,urllib.request,uuid
ch,rt,tok,body=sys.argv[1:5]
payload={"body":body}
if rt: payload["reply_to"]=int(rt)
req=urllib.request.Request(f"https://lvwerra-msgboard-demo.hf.space/v1/channels/{ch}/messages",
    data=json.dumps(payload).encode(),headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json","Idempotency-Key":str(uuid.uuid4())},method="POST")
import time
for attempt in range(8):
    try:
        r=json.load(urllib.request.urlopen(req, timeout=60)); print("posted id",r.get("id"),"seq",r.get("seq")); break
    except urllib.error.HTTPError as e:
        if e.code in (429,500,502,503,504):
            w=int(e.headers.get("Retry-After") or 30); print("retry",e.code,"in",w,"s"); time.sleep(w); continue
        print("ERR",e.code,e.read().decode()[:300]); break
    except Exception as e: print("retry",e); time.sleep(20)
else: print("GAVE UP")
PY
