#!/usr/bin/env python3
"""
Honest walk-forward backtest for the Romanian 6/49 data.
Pure standard library — no deps.

For each test draw t, it uses ONLY draws [0, t) to rank the 49 numbers by
several strategies (the same ideas Loto_RO.py uses), picks the top 6, and
counts how many it got right in the actual draw t. It then compares each
strategy's average hit-rate against the exact random expectation.

Random expectation for a 6/49 ticket:  mean matches = 6 * 6/49 = 0.7347
If a strategy can't beat that by a statistically significant margin, it has
no predictive power — it's reacting to noise.
"""
import csv, math, random, sys
from collections import Counter

CSV = sys.argv[1] if len(sys.argv) > 1 else "drawn_RO.csv"
MAXN, K = 49, 6

# ---- load ----
draws = []
with open(CSV) as f:
    for row in csv.DictReader(f):
        nums = []
        for c in row:
            if c.strip().upper().startswith("N"):
                try: nums.append(int(row[c]))
                except: pass
        nums = [n for n in nums if 1 <= n <= MAXN]
        if len(set(nums)) == K:
            draws.append(sorted(nums))
print(f"Loaded {len(draws)} valid draws from {CSV}")

# ---- strategies: each returns a ranking score per number, from history only ----
def s_freq_hot(hist):           # most frequent all-time ("hot")
    c = Counter(n for d in hist for n in d)
    return [c.get(n, 0) for n in range(1, MAXN+1)]

def s_freq_cold(hist):          # least frequent all-time ("cold")
    c = Counter(n for d in hist for n in d)
    return [-c.get(n, 0) for n in range(1, MAXN+1)]

def s_overdue(hist):            # largest gap since last seen ("overdue")
    last = {n: -1 for n in range(1, MAXN+1)}
    for i, d in enumerate(hist):
        for n in d: last[n] = i
    now = len(hist)
    return [now - last[n] for n in range(1, MAXN+1)]

def s_hot20(hist):              # hottest in last 20 draws
    c = Counter(n for d in hist[-20:] for n in d)
    return [c.get(n, 0) for n in range(1, MAXN+1)]

def s_recency(hist):            # recency-weighted (exp decay, half-life 12)
    score = [0.0]*MAXN
    H = len(hist)
    for i, d in enumerate(hist):
        w = math.exp(-math.log(2) * (H-1-i) / 12)
        for n in d: score[n-1] += w
    return score

def s_recent_repeat(hist):      # numbers from the most recent draw
    score = [0.0]*MAXN
    if hist:
        for n in hist[-1]: score[n-1] = 1.0
    return score

STRATS = {
    "freq_hot": s_freq_hot, "freq_cold": s_freq_cold, "overdue": s_overdue,
    "hot_last20": s_hot20, "recency_hl12": s_recency, "repeat_last": s_recent_repeat,
}

def top6(score):
    return set(sorted(range(1, MAXN+1), key=lambda n: score[n-1], reverse=True)[:K])

# ---- walk-forward over the last TEST draws ----
TEST = min(800, len(draws) - 200)
start = len(draws) - TEST
results = {name: [] for name in STRATS}
results["random"] = []
rng = random.Random(12345)

for t in range(start, len(draws)):
    hist = draws[:t]
    actual = set(draws[t])
    for name, fn in STRATS.items():
        results[name].append(len(top6(fn(hist)) & actual))
    results["random"].append(len(set(rng.sample(range(1, MAXN+1), K)) & actual))

# ---- exact random baseline (hypergeometric) ----
mean_rand = K * K / MAXN                       # 0.7347
var_rand = K * (K/MAXN) * ((MAXN-K)/MAXN) * ((MAXN-K)/(MAXN-1))

print(f"\nWalk-forward test draws: {TEST}  (each predicted from prior history only)")
print(f"Exact random expectation: mean matches = {mean_rand:.4f} per ticket\n")
print(f"{'strategy':14s} {'mean':>7s} {'≥3 hits':>8s} {'best':>5s} {'z vs random':>12s}")
print("-"*52)
def stats(vals):
    n = len(vals); m = sum(vals)/n
    sd = (sum((v-m)**2 for v in vals)/n) ** 0.5
    z = (m - mean_rand) / (math.sqrt(var_rand)/math.sqrt(n))  # vs theoretical random
    ge3 = sum(1 for v in vals if v >= 3)/n*100
    return m, sd, ge3, max(vals), z
for name in list(STRATS) + ["random"]:
    m, sd, ge3, bst, z = stats(results[name])
    flag = "  <-- noise" if abs(z) < 2 else "  <-- SIGNAL?!"
    print(f"{name:14s} {m:7.4f} {ge3:7.1f}% {bst:5d} {z:12.2f}{flag if name!='random' else ''}")

print(f"\n|z| < 2  => indistinguishable from random guessing (no predictive power).")
print(f"A 6/49 ticket wins something (3+) about {100*0.0186:.1f}% of the time by pure chance.")
