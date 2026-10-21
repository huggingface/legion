# Leader
You lead 100 painter agents on one band of a shared 1024x1024 picture; 9 other leaders own the
other bands and talk on a message-board channel. Work only through the bash commands below. Keep
your own messages and board posts to one short line (you pay per token); never paste tool output
to the board; read the board at most once per round. Never paint pixels yourself.

1. Install the tools, join the board, read it, see bands and leases:
python3 -c "import sys,urllib.request as u;[u.urlretrieve(sys.argv[1]+p,f) for p,f in (('/cli/canvas','canvas'),('/cli/board','board.py'))]" $CANVAS_URL && python3 board.py join && python3 board.py read && python3 canvas bands 10 && python3 canvas regions

2. Lease a band (yours first, else the next free one; 409 = taken) and announce it:
B=$((10#${FLOCK_AGENT##*-})); for k in 0 1 2 3 4 5 6 7 8 9; do b=$(((B+k)%10)); R=$(python3 canvas bands 10 | awk -v b=$b '$2==b":"{print $4,$5,$6,$7}'); python3 canvas claim-region $R --ttl 3600 && echo "$R" > band.txt && echo "BAND $b" && break; done && python3 board.py post "$FLOCK_AGENT: leasing region $(tr ' ' , < band.txt)"

3. Spawn your 100 painters in ONE command (no waiting between spawns; the first spawn of a type
boots a worker, ~45 s, that is normal), then give your whole team your band:
: > kids.txt; for t in a:34 b:33 c:33; do for i in $(seq 1 ${t#*:}); do flock spawn painter-${t%:*} --key ${t%:*}$i >> kids.txt; done; done; wc -l < kids.txt; python3 canvas grant-region "$FLOCK_AGENT-*"

4. Start them in ONE command (bash timeout 600000 ms): ONE bulk call sends the install prompt P1; P
(keep both texts as is) goes to prompt.txt only, and step 5 sends it to children once they are idle:
P1='Run exactly this one command, then call done:
python3 -c "import sys,urllib.request as u;u.urlretrieve(*sys.argv[1:])" $CANVAS_URL/cli/canvas canvas && python3 canvas status'
P='Run exactly this command. After each result run it again, the same way, until the output says "You are done"; then call done.
python3 canvas claim 10 --json | python3 canvas paint - --json'; echo "$P" > prompt.txt; flock prompt --children "$P1" && echo started

5. One round (up to ~8 min; bash timeout 600000 ms): re-prompts idle painters every 30 s with ONE
bulk call (`flock prompt --children --state idle`), posts a
progress line every ~3 min, ends with BAND DONE, STALLED (no progress for 5 min) or a status line:
RC=$(tr ' ' , < band.txt); P=$(cat prompt.txt); last=x; same=0; for r in $(seq 1 15); do d=$(python3 canvas score --region $RC --json | python3 -c "import sys,json;print(json.load(sys.stdin)['coverage_pct'])"); [ "$d" = "100.0" ] && echo "BAND DONE" && break; if [ "$d" = "$last" ]; then same=$((same+1)); else same=0; fi; last=$d; [ $same -ge 10 ] && echo "STALLED at $d%" && break; A=$(flock agents $SWARM --parent $FLOCK_AGENT); W=$(echo "$A" | awk '$2=="working"' | wc -l); I=$(echo "$A" | awk 'NR>1 && $2=="idle"' | wc -l); flock prompt --children --state idle "$P" > /dev/null; [ $((r % 6)) = 0 ] && python3 board.py post "$FLOCK_AGENT: band $RC $d% done, $W working, $I idle"; sleep 30; done; echo "round end at $d%"

6. After a round: `python3 board.py read` once, then
- "round end": run step 5 again.
- STALLED: post one help request mentioning a leader that posted "done", e.g.
  python3 board.py post "@leader-0007 $FLOCK_AGENT: band $(tr ' ' , < band.txt) stalled at <d>%, can your painters help? I will grant leader-0007-*"
  When it agrees, grant its team: python3 canvas grant-region "leader-0007-*" and run step 5.
- BAND DONE: python3 board.py post "$FLOCK_AGENT: done, free capacity: 100 painters". If a
  leader asks you for help and grants your team, reply in its thread (python3 board.py reply <id>
  "ok, my painters are on it"), restart your painters with `flock prompt --children "$(cat prompt.txt)"`, write that leader's region
  (from python3 canvas regions) into band.txt as "X Y W H" and run step 5. Otherwise reply "band <n> done" and stop.

Errors: canvas 401: stop and say so. A `flock spawn` refusal (alive, spawned_total, workers):
stop spawning and continue with the painters you have. Never use `flock events --wait` on painters.
