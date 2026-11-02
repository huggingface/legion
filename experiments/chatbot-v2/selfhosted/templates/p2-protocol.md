Merged team rules (phase 2: <<n_agents>> equals, no lead):
- The team thread is [merged]. The team threads (<<threads>>) are closed: read them only to find each team's last [team-best] path.
- Compare round first (while best_status shows `compare_round`): everyone runs the next team's [team-best] bot (team a runs b's, ..., the last team runs a's) and posts "[keep] <path>: <what to keep>; <what to drop>". No promotion until it ends.
- Splits: anyone may propose one in [merged]: "[split] <name>=<feature module to port or improve>, ...", mixing agents from different teams. Before proposing, `read` [merged]: if a [split] is there, answer it with `agree <its id>` or ONE counter-proposal. If two proposals cross, the smaller message id wins. Start once the proposal has two `agree`s.
- First promotion: after the compare round, whoever owns a team's [team-best] promotes it once with promote_best (the probe gate keeps the better one). After that, everyone builds on best/.
- The probes are the gate: a RED becomes a probe (add_probe, then post its line) BEFORE its fix. A promotion must beat best on the probes, so it must pass a probe best fails; a tie does not promote.
- Before every promote_best: `best_status`. If best changed since you copied it, `copy_file` best again and redo your change on the new copy first, so nobody's work is lost.
- After promote_best, `post` the result in [merged] (promoted or refused, with the probes it now passes) and @ whoever's part you ported.
