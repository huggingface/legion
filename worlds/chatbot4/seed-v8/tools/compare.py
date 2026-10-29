"""Compare two checkouts of the bot on a random sample of the deck, blind.

    git worktree add ../base origin/main            # the checkout to compare against
    python tools/compare.py ../base . --seed 41     # base, then head (this clone)
    python tools/compare.py --score /tmp/compare-41.key "1 2 = 2 1"

The first command runs both bots on the same deck conversations (deck/*.jsonl in head) and
prints every turn whose reply changed: single turns as pairs, multi-turn conversations as two
whole transcripts. Which side is which is random and written to a key file, not printed. Judge
each item by STYLE.md, in order, with 1, 2 or = (no better), then score the verdicts with the
key: the second command prints how often head was better and worse. --sample 0 runs the
whole deck. The summary also counts which modules answer more or fewer turns in head.
"""
import argparse
import collections
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

DRIVER = r'''
import json, os, sys
sys.path.insert(0, os.getcwd())
sys.dont_write_bytecode = True
import bot.runtime as runtime

def run(turns):
    memory, out = {}, []
    for text in turns:
        if hasattr(runtime, "turn"):
            t = runtime.turn(text, memory)
            out.append({"reply": t["reply"], "module": t["module"], "candidates": len(t["candidates"])})
        else:
            out.append({"reply": runtime.reply(text, memory), "module": None, "candidates": None})
    return out

for line in sys.stdin:
    turns = json.loads(line)
    read, write = os.pipe()
    pid = os.fork()  # every conversation starts from the freshly loaded bot, as in production
    if pid == 0:
        os.close(read)
        try:
            data = json.dumps(run(turns))
        except BaseException as exc:
            data = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        with os.fdopen(write, "w") as f:
            f.write(data)
        os._exit(0)
    os.close(write)
    with os.fdopen(read) as f:
        data = f.read()
    os.waitpid(pid, 0)
    print(data or json.dumps({"error": "the conversation crashed"}), flush=True)
'''


def load_deck(root):
    items = []
    for path in sorted((Path(root) / "deck").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def sample(deck, n, seed, multi_share):
    rng = random.Random(seed)
    if n <= 0 or n >= len(deck):
        return list(deck)
    multi = [c for c in deck if len(c["turns"]) > 1]
    single = [c for c in deck if len(c["turns"]) == 1]
    k = min(len(multi), round(n * multi_share))
    picked = rng.sample(multi, k) + rng.sample(single, min(len(single), n - k))
    return sorted(picked, key=lambda c: c["id"])


def run_bot(root, conversations, timeout):
    lines = "".join(json.dumps(c["turns"]) + "\n" for c in conversations)
    p = subprocess.run([sys.executable, "-c", DRIVER], input=lines, text=True, capture_output=True,
                       cwd=root, timeout=timeout, check=False)
    results = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    if p.returncode or len(results) != len(conversations):
        raise SystemExit(f"the bot in {root} failed:\n{p.stderr[-2000:]}")
    return results


def revision(root):
    p = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                       text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                           text=True, check=False).stdout.strip()
    return (p.stdout.strip() or "?") + (" + local changes" if dirty else "")


def transcript(turns, replies):
    return "\n".join(f"  User: {text}\n  Bot:  {r['reply']}" for text, r in zip(turns, replies, strict=True))


def compare(args):
    deck = load_deck(args.head)
    if not deck:
        raise SystemExit(f"no deck in {args.head}/deck")
    picked = sample(deck, args.sample, args.seed, args.multi_share)
    base = run_bot(args.base, picked, args.timeout)
    head = run_bot(args.head, picked, args.timeout)
    rng = random.Random(f"{args.seed}-order")
    items, key, crashes = [], [], []
    moved, answers = collections.Counter(), collections.Counter()
    turns = changed = 0
    for conv, b, h in zip(picked, base, head, strict=True):
        if "error" in b or "error" in h:
            crashes.append(f"{conv['id']}: base {b.get('error', 'ok')}; head {h.get('error', 'ok')}")
            continue
        turns += len(conv["turns"])
        diff = [i for i, (x, y) in enumerate(zip(b, h, strict=True)) if x["reply"] != y["reply"]]
        changed += len(diff)
        for x, y in zip(b, h, strict=True):
            answers[x["module"]] -= 1
            answers[y["module"]] += 1
            if x["module"] != y["module"]:
                moved[(x["module"], y["module"])] += 1
        if not diff:
            continue
        head_first = rng.random() < 0.5
        key.append("head=1" if head_first else "head=2")
        first, second = (h, b) if head_first else (b, h)
        if len(conv["turns"]) == 1:
            text = f"  User: {conv['turns'][0]}\n  1: {first[0]['reply']}\n  2: {second[0]['reply']}"
        else:
            text = (f"  Conversation 1:\n{transcript(conv['turns'], first)}\n"
                    f"  Conversation 2:\n{transcript(conv['turns'], second)}")
        items.append((conv, diff, text))
    singles = sum(1 for c in picked if len(c["turns"]) == 1)
    print(f"base {args.base} ({revision(args.base)}) vs head {args.head} ({revision(args.head)})")
    print(f"deck sample: {len(picked)} conversations (seed {args.seed}), {singles} single-turn, "
          f"{len(picked) - singles} multi-turn; {changed} of {turns} replies changed")
    gained = [f"{m or '(none)'} {n:+d}" for m, n in answers.most_common() if n > 0]
    lost = [f"{m or '(none)'} {n:+d}" for m, n in reversed(answers.most_common()) if n < 0]
    if gained or lost:
        print(f"turns answered, head vs base: {', '.join(gained + lost)}")
    for (was, now), n in moved.most_common(8):
        print(f"  {n} turn(s) moved from {was or '(none)'} to {now or '(none)'}")
    for crash in crashes:
        print(f"CRASHED {crash}")
    if not items:
        print("no reply changed")
        return
    print("\nJudge each item by STYLE.md: which would you rather get? Answer 1, 2 or = per item.\n")
    for i, (conv, diff, text) in enumerate(items, 1):
        where = "" if len(conv["turns"]) == 1 else f", turns {', '.join(str(d + 1) for d in diff)} changed"
        print(f"[{i}] {conv['id']} ({conv['area']}){where}\n{text}\n")
    path = Path(args.key or Path(tempfile.gettempdir()) / f"compare-{args.seed}.key")
    path.write_text(json.dumps({"seed": args.seed, "items": [c["id"] for c, _, _ in items], "key": key}))
    print(f"key: {path} (score with: python tools/compare.py --score {path} \"<verdicts>\")")


def score(path, verdicts):
    data = json.loads(Path(path).read_text())
    marks = verdicts.replace(",", " ").split()
    if len(marks) != len(data["key"]):
        raise SystemExit(f"give one verdict per item ({len(data['key'])}), got {len(marks)}")
    better, worse, same = [], [], []
    for k, item, m in zip(data["key"], data["items"], marks, strict=True):
        if m == "=":
            same.append(item)
        elif m in ("1", "2"):
            (better if k == f"head={m}" else worse).append(item)
        else:
            raise SystemExit(f"verdicts are 1, 2 or =, got {m!r}")
    print(f"seed {data['seed']}: head better {len(better)}, base better {len(worse)}, same {len(same)}")
    if worse:
        print(f"head worse on: {', '.join(worse)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", nargs="?", help="checkout to compare against (e.g. a worktree of origin/main)")
    parser.add_argument("head", nargs="?", default=".", help="checkout to test (default: this clone)")
    parser.add_argument("--sample", type=int, default=40, help="conversations to run (0: the whole deck)")
    parser.add_argument("--seed", type=int, default=None, help="sample seed (default: random)")
    parser.add_argument("--multi-share", type=float, default=0.4, help="share of multi-turn conversations")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per bot")
    parser.add_argument("--key", help="where to write the key (default: the temp directory)")
    parser.add_argument("--score", nargs=2, metavar=("KEY", "VERDICTS"), help="score verdicts with a key")
    args = parser.parse_args()
    if args.score:
        return score(*args.score)
    if not args.base:
        parser.error("give the base checkout (and optionally head), or --score KEY VERDICTS")
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(1, 10000)
    args.base, args.head = os.path.abspath(args.base), os.path.abspath(args.head)
    compare(args)


if __name__ == "__main__":
    main()
