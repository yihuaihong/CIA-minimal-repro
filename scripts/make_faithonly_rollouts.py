"""Faith-only twin of a scored rollout file (RS-B / DPO-B input): acc := 1 for every rollout, reward := faith.
Mirrors the `_faithonly.jsonl` branch of scripts/score_hint_rollouts_v2.py for scorers that lack it (TwoHop v3).
Usage: python scripts/make_faithonly_rollouts.py IN_scored.jsonl [OUT.jsonl]  (default OUT = IN with _faithonly suffix)
"""
import json, sys
src = sys.argv[1]; dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".jsonl", "_faithonly.jsonl")
n = 0
with open(dst, "w") as f:
    for l in open(src):
        r = json.loads(l); K = len(r["completions"])
        r["acc_orig"] = r["acc"]; r["acc"] = [1.0] * K; r["reward"] = [float(x) for x in r["faith"]]; n += 1
        f.write(json.dumps(r) + "\n")
print("saved", dst, "rows", n)
