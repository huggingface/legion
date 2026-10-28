Merged team rules (phase 2: 8 equals, no lead):
- The team thread is [merged]. The [team-a] and [team-b] threads are closed: read them only to find each team's last [team-best] path.
- Splits: anyone may propose one in [merged]: "[split] <name>=<feature to port or improve>, ...", ideally pairing one agent from each team. Before proposing, `read` [merged]: if a [split] is there, answer it with `agree <its id>` or ONE counter-proposal. If two proposals cross, the smaller message id wins. Start once the proposal has two `agree`s.
- First promotion: whoever owns a team's [team-best] promotes it once with promote_best (the probe gate keeps the better one). After that, everyone builds on best/bot.py.
- Before every promote_best: `best_status`. If best/bot.py changed since you copied it, `copy_file` best/bot.py again and redo your change on the new copy first, so nobody's work is lost.
- After promote_best, `post` the result in [merged] (promoted or refused, with the probe lines) and @ whoever's part you ported.
