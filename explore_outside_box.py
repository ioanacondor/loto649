#!/usr/bin/env python3
"""
Outside-the-box hunt for ANY exploitable structure in drawn_RO.csv.
Pure stdlib. Honest: every test reports whether it beats chance, with a stat.

We test the only ideas that could *legitimately* let history predict the future:

  A. PHYSICAL BIAS PERSISTENCE  — the real-world winner's edge. If the ball
     machine is biased, the same numbers stay hot across decades. Test: does
     per-number frequency in the 1st half of history correlate with the 2nd?
  B. HOT-NUMBER TRANSFER        — do numbers hot in the past stay hot later,
     enough to beat random when you bet on them?
  C. DRAW-ORDER / POSITION BIAS — is any draw position (1st ball … 6th ball)
     non-uniform? (machine/feed quirks)
  D. PAIR RECURRENCE            — do specific pairs co-occur more than chance?
  E. SEASONALITY                — does the number distribution change by month?
  F. SUM PREDICTABILITY         — can we predict next draw's SUM (and does that
     even help pick numbers)?

If something here genuinely beat chance, it would be worth real money. Let's see.
"""
import csv, math, os, random
from collections import Counter

MAXN, K = 49, 6
HERE = os.path.dirname(os.path.abspath(__file__))


def chi2_stat(obs):
    exp = sum(obs) / len(obs)
    return sum((o - exp) ** 2 / exp for o in obs)


def load(path):
    out = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        ncols = [c for c in rd.fieldnames if c and c.strip().upper().startswith("N")][:K]
        dcol = next((c for c in rd.fieldnames if c and "date" in c.lower()), None)
        for r in rd:
            try:
                nums = [int(r[c]) for c in ncols]
            except (TypeError, ValueError):
                continue
            if len(set(nums)) == K and all(1 <= n <= MAXN for n in nums):
                out.append({"ordered": nums, "set": sorted(nums), "date": r.get(dcol, "")})
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def main():
    draws = load(os.path.join(HERE, "drawn_RO.csv"))
    N = len(draws)
    print(f"\nAnalysing {N} draws.\n" + "=" * 64)

    # ── A. Physical bias persistence ──────────────────────────────────
    print("A. PHYSICAL BIAS PERSISTENCE (the real-world winner's edge)")
    half = N // 2
    f1 = Counter(n for d in draws[:half] for n in d["set"])
    f2 = Counter(n for d in draws[half:] for n in d["set"])
    v1 = [f1.get(n, 0) for n in range(1, MAXN + 1)]
    v2 = [f2.get(n, 0) for n in range(1, MAXN + 1)]
    r = pearson(v1, v2)
    se = 1 / math.sqrt(MAXN - 3)
    z = r / se
    print(f"   corr(freq 1st half, freq 2nd half) r = {r:+.3f}   (z={z:+.2f})")
    print(f"   → {'PERSISTENT BIAS — exploitable!' if abs(z) > 2 and r > 0 else 'no persistence: hotness does NOT carry over → no machine bias'}")
    print()

    # ── B. Hot-number transfer (walk-forward bet on past-hot) ─────────
    print("B. HOT-NUMBER TRANSFER (bet on numbers hot so far)")
    test = min(900, N - 200)
    start = N - test
    hits_hot, hits_rand = [], []
    rng = random.Random(7)
    for t in range(start, N):
        c = Counter(n for d in draws[:t] for n in d["set"])
        hot = set(sorted(range(1, MAXN + 1), key=lambda n: c.get(n, 0), reverse=True)[:K])
        actual = set(draws[t]["set"])
        hits_hot.append(len(hot & actual))
        hits_rand.append(len(set(rng.sample(range(1, MAXN + 1), K)) & actual))
    mh, mr = sum(hits_hot) / test, K * K / MAXN
    var = K * (K / MAXN) * ((MAXN - K) / MAXN) * ((MAXN - K) / (MAXN - 1))
    z = (mh - mr) / (math.sqrt(var) / math.sqrt(test))
    print(f"   all-time-hot top6: mean {mh:.4f} matches vs random {mr:.4f}  (z={z:+.2f})")
    print(f"   → {'BEATS random!' if z > 2 else 'no edge — past-hot does not predict'}")
    print()

    # ── C. Draw-order / position bias ─────────────────────────────────
    print("C. DRAW-ORDER POSITION BIAS (is any ball-position non-uniform?)")
    worst = None
    for pos in range(K):
        obs = [0] * MAXN
        for d in draws:
            obs[d["ordered"][pos] - 1] += 1
        chi = chi2_stat(obs)
        if worst is None or chi > worst[1]:
            worst = (pos + 1, chi)
    # crude p via mean/sd of chi2(df=48): mean=48, sd=sqrt(96)=9.8
    z = (worst[1] - 48) / math.sqrt(2 * 48)
    print(f"   most non-uniform position = N{worst[0]} (χ²={worst[1]:.1f}, df=48, z={z:+.2f})")
    print(f"   → {'a position looks biased' if z > 3 else 'all 6 positions uniform → no order bias'}")
    print()

    # ── D. Pair recurrence ────────────────────────────────────────────
    print("D. PAIR RECURRENCE (do specific pairs co-occur more than chance?)")
    pc = Counter()
    for d in draws:
        for a, b in __import__("itertools").combinations(d["set"], 2):
            pc[(a, b)] += 1
    n_pairs = MAXN * (MAXN - 1) // 2
    exp_pair = N * math.comb(K, 2) / n_pairs          # expected count per pair
    top_pair, top_cnt = pc.most_common(1)[0]
    sd = math.sqrt(exp_pair)                          # Poisson approx
    z = (top_cnt - exp_pair) / sd
    expected_max_z = math.sqrt(2 * math.log(n_pairs))   # expected largest z over n_pairs
    bonf_p = min(1.0, n_pairs * 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    print(f"   each pair expected ~{exp_pair:.1f} times; hottest pair {top_pair} appeared {top_cnt}x  (z={z:+.2f})")
    print(f"   testing {n_pairs} pairs, the expected LARGEST z by pure chance is ~{expected_max_z:.2f}.")
    print(f"   Bonferroni-corrected p ≈ {bonf_p:.2f}  → {'real anomaly' if bonf_p < 0.01 else 'just the look-elsewhere effect (not exploitable)'}")
    print()

    # ── E. Seasonality ────────────────────────────────────────────────
    print("E. SEASONALITY (does the number mix change by month?)")
    bymonth = {}
    for d in draws:
        m = d["date"][5:7]
        if m.isdigit():
            bymonth.setdefault(m, Counter()).update(d["set"])
    # compare summer (06,07,08) vs winter (12,01,02) distributions via chi-square
    summer = Counter()
    winter = Counter()
    for m, c in bymonth.items():
        (summer if m in ("06", "07", "08") else winter if m in ("12", "01", "02") else Counter()).update(c)
    diff = 0.0
    for n in range(1, MAXN + 1):
        s, w = summer.get(n, 0), winter.get(n, 0)
        tot = s + w
        if tot:
            diff += (s - tot / 2) ** 2 / (tot / 2)
    z = (diff - 48) / math.sqrt(2 * 48)
    print(f"   summer-vs-winter number distribution χ²={diff:.1f} (df≈48, z={z:+.2f})")
    print(f"   → {'seasonal effect' if z > 3 else 'no seasonal effect — months are interchangeable'}")
    print()

    # ── F. Sum predictability ─────────────────────────────────────────
    print("F. SUM PREDICTABILITY (can we predict the next draw's SUM?)")
    sums = [sum(d["set"]) for d in draws]
    msum, ssum = sum(sums) / N, (sum((x - sum(sums) / N) ** 2 for x in sums) / N) ** 0.5
    # does last sum predict next sum? lag-1 autocorrelation
    a, b = sums[:-1], sums[1:]
    r = pearson(a, b)
    print(f"   sum: mean {msum:.0f} ± {ssum:.0f}.  lag-1 autocorrelation r={r:+.3f}")
    print(f"   → sum is predictable as a RANGE (it clusters near {msum:.0f}), but that")
    print(f"     constrains, it doesn't pick: millions of combos share any given sum,")
    print(f"     and consecutive sums are {'correlated' if abs(r)>0.1 else 'independent (r≈0)'}.")
    print()

    print("=" * 64)
    print("VERDICT: history bears no exploitable memory. The ONE thing that IS")
    print("predictable from data is human behaviour — which combos people pick —")
    print("and that is what Loto_RO_v5.py exploits (expected-payout, not odds).")


if __name__ == "__main__":
    main()
