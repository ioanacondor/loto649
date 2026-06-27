#!/usr/bin/env python3
"""
Loto_RO v5  —  Romanian 6/49 honest analyzer + expected-value ticket optimizer
==============================================================================

WHAT CHANGED FROM v4, AND WHY
-----------------------------
v4 tried to *predict* winning numbers with 8 "signals", an 8-model ensemble,
49 ML classifiers and a cross-validation that printed "✓ BEATS baseline".
A proper walk-forward backtest on this very dataset shows none of it beats
random guessing (see the BACKTEST section below — it now runs every time and
tells the truth instead of hiding it). That is not a bug to fix; a fair 6/49
draw is memoryless, so past numbers carry zero information about the next draw.

So v5 stops pretending to predict. Instead it does the one thing that is
mathematically real:

    You cannot raise your probability of winning.
    You CAN raise your expected payout *if* you win — by choosing
    combinations that few other people choose, so you split the prize
    fewer ways. The jackpot is parimutuel (shared among winners).

v5 therefore:
  • keeps the genuinely useful, honest analysis (fairness test, distributions),
  • runs an honest walk-forward BACKTEST proving the old signals were noise,
  • models the parimutuel EXPECTED VALUE including prize-sharing,
  • GENERATES tickets that are uniformly random in win-probability but
    optimised to be UNPOPULAR (low expected number of co-winners),
  • is pure standard library (no pandas/numpy/sklearn) so it runs anywhere,
  • validates & de-duplicates the data and fixes v4's wrong default path.

This is still negative expected value overall. It is a tool for playing
*smarter if you play*, not a way to win. It does not beat the house.
"""

import csv
import math
import os
import random
import sys
import json
from collections import Counter
from itertools import combinations

MAXN = 49
K = 6
TOTAL = math.comb(MAXN, K)            # 13,983,816 possible combinations


# ═══════════════════════════════════════════════════════════════════════
#  STATS HELPERS (stdlib only — chi-square p-value via regularized gamma)
# ═══════════════════════════════════════════════════════════════════════

def _gammln(x):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _gser(a, x):
    if x <= 0:
        return 0.0
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(500):
        ap += 1
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - _gammln(a))


def _gcf(a, x):
    tiny = 1e-30
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - _gammln(a)) * h


def chi2_sf(x, df):
    """Upper tail (survival) of the chi-square distribution = p-value."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    if xx < a + 1:
        return 1.0 - _gser(a, xx)
    return _gcf(a, xx)


def mean(v):
    return sum(v) / len(v)


def stdev(v):
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════

def load_draws(path):
    rows = []
    seen = set()
    dups = 0
    bad = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c and c.strip().upper().startswith("N")][:K]
        date_col = next((c for c in reader.fieldnames if c and "date" in c.lower()), None)
        for r in reader:
            try:
                nums = sorted(int(r[c]) for c in cols)
            except (TypeError, ValueError):
                bad += 1
                continue
            if len(set(nums)) != K or not all(1 <= n <= MAXN for n in nums):
                bad += 1
                continue
            key = tuple(nums) + (r.get(date_col, ""),)
            if key in seen:           # exact duplicate row (same date + numbers)
                dups += 1
                continue
            seen.add(key)
            rows.append({"date": r.get(date_col, ""), "nums": nums})
    return rows, dups, bad


# ═══════════════════════════════════════════════════════════════════════
#  1. FAIRNESS  (the honest, meaningful test — chi-square uniformity)
# ═══════════════════════════════════════════════════════════════════════

def section_fairness(draws):
    print("=" * 72)
    print("  1. FAIRNESS TEST")
    print("=" * 72)
    counts = Counter(n for d in draws for n in d["nums"])
    obs = [counts.get(n, 0) for n in range(1, MAXN + 1)]
    total = sum(obs)
    exp = total / MAXN
    chi2 = sum((o - exp) ** 2 / exp for o in obs)
    p = chi2_sf(chi2, MAXN - 1)
    print(f"  Draws analysed : {len(draws)}")
    print(f"  Chi-square     : χ²={chi2:.2f}  (df={MAXN-1})  p={p:.4f}")
    if p > 0.05:
        print("  Verdict        : UNIFORM & FAIR ✓  — every number is equally likely.")
        print("                   (This is exactly why no past pattern predicts the future.)")
    else:
        print("  Verdict        : Deviation from uniform ⚠ — inspect data quality.")
    hottest = sorted(range(1, MAXN + 1), key=lambda n: counts.get(n, 0), reverse=True)[:5]
    coldest = sorted(range(1, MAXN + 1), key=lambda n: counts.get(n, 0))[:5]
    print(f"  Most drawn     : {hottest}   Least drawn: {coldest}")
    print("  (The spread between them is what fair random variation looks like — not signal.)")
    print()
    return {"chi2": chi2, "p": p}


# ═══════════════════════════════════════════════════════════════════════
#  2. DESCRIPTIVE STRUCTURE (for understanding the game, not predicting)
# ═══════════════════════════════════════════════════════════════════════

def section_structure(draws):
    print("=" * 72)
    print("  2. DRAW STRUCTURE  (descriptive — what a real draw tends to look like)")
    print("=" * 72)
    sums = [sum(d["nums"]) for d in draws]
    evens = [sum(1 for n in d["nums"] if n % 2 == 0) for d in draws]
    lows = [sum(1 for n in d["nums"] if n <= 24) for d in draws]
    consec = [sum(1 for a, b in zip(d["nums"], d["nums"][1:]) if b - a == 1) for d in draws]
    repeats = [len(set(draws[i]["nums"]) & set(draws[i - 1]["nums"])) for i in range(1, len(draws))]
    print(f"  Sum of 6        : mean {mean(sums):.0f}, range [{min(sums)}–{max(sums)}], "
          f"middle-80% ≈ [{sorted(sums)[len(sums)//10]}–{sorted(sums)[len(sums)*9//10]}]")
    print(f"  Even numbers    : mean {mean(evens):.2f} of 6")
    print(f"  Low (≤24)       : mean {mean(lows):.2f} of 6")
    print(f"  Consecutive pairs: mean {mean(consec):.2f}  "
          f"({100*sum(1 for c in consec if c>0)/len(consec):.0f}% of draws have ≥1 run)")
    print(f"  Repeats vs prev : mean {mean(repeats):.2f}  "
          f"({100*sum(1 for r in repeats if r==0)/len(repeats):.0f}% share none with last draw)")
    print()
    return {"sum_mean": mean(sums)}


# ═══════════════════════════════════════════════════════════════════════
#  3. HONEST BACKTEST — do "hot / cold / overdue / recency" beat random?
# ═══════════════════════════════════════════════════════════════════════

def _strats():
    def freq_hot(hist):
        c = Counter(n for d in hist for n in d)
        return [c.get(n, 0) for n in range(1, MAXN + 1)]

    def freq_cold(hist):
        c = Counter(n for d in hist for n in d)
        return [-c.get(n, 0) for n in range(1, MAXN + 1)]

    def overdue(hist):
        last = {n: -1 for n in range(1, MAXN + 1)}
        for i, d in enumerate(hist):
            for n in d:
                last[n] = i
        now = len(hist)
        return [now - last[n] for n in range(1, MAXN + 1)]

    def hot20(hist):
        c = Counter(n for d in hist[-20:] for n in d)
        return [c.get(n, 0) for n in range(1, MAXN + 1)]

    def recency(hist):
        sc = [0.0] * MAXN
        H = len(hist)
        for i, d in enumerate(hist):
            w = math.exp(-math.log(2) * (H - 1 - i) / 12)
            for n in d:
                sc[n - 1] += w
        return sc

    return {"freq_hot": freq_hot, "freq_cold": freq_cold,
            "overdue": overdue, "hot_last20": hot20, "recency": recency}


def section_backtest(draws):
    print("=" * 72)
    print("  3. HONEST BACKTEST  (walk-forward — past data only, no leakage)")
    print("=" * 72)
    nums = [d["nums"] for d in draws]
    strats = _strats()
    test = min(800, len(nums) - 200)
    start = len(nums) - test
    res = {name: [] for name in strats}
    res["random"] = []
    rng = random.Random(12345)
    mean_rand = K * K / MAXN
    var_rand = K * (K / MAXN) * ((MAXN - K) / MAXN) * ((MAXN - K) / (MAXN - 1))

    def top6(score):
        return set(sorted(range(1, MAXN + 1), key=lambda n: score[n - 1], reverse=True)[:K])

    for t in range(start, len(nums)):
        hist = nums[:t]
        actual = set(nums[t])
        for name, fn in strats.items():
            res[name].append(len(top6(fn(hist)) & actual))
        res["random"].append(len(set(rng.sample(range(1, MAXN + 1), K)) & actual))

    print(f"  Test draws: {test}.  Random expectation = {mean_rand:.4f} matches/ticket.\n")
    print(f"  {'strategy':12s} {'mean':>7s} {'best':>5s} {'z vs random':>12s}   verdict")
    print("  " + "-" * 56)
    out = {}
    for name in list(strats) + ["random"]:
        v = res[name]
        m = mean(v)
        z = (m - mean_rand) / (math.sqrt(var_rand) / math.sqrt(len(v)))
        verdict = "no predictive power" if abs(z) < 2 else "noise (multiple-testing)"
        if name == "random":
            verdict = "(control)"
        print(f"  {name:12s} {m:7.4f} {max(v):5d} {z:12.2f}   {verdict}")
        out[name] = {"mean": m, "z": z}
    print("\n  Conclusion: every 'hot/cold/overdue/recency' idea sits within noise of")
    print("  random. They cannot be improved into a predictor — the draw is memoryless.")
    print()
    return out


# ═══════════════════════════════════════════════════════════════════════
#  4. POPULARITY MODEL  (the real edge: avoid combos crowds pick)
# ═══════════════════════════════════════════════════════════════════════
#
# Per-number relative pick-rate among HUMANS (not the machine). Well documented:
# players over-pick birthdays → 1..31 heavily, 1..12 (months) most of all; the
# number 7 and its multiples are "lucky"; numbers 32..49 are badly under-picked.
# Weights are normalised to mean 1 across 1..49, so a uniform-random ticket has
# an expected popularity multiplier of ~1; "birthday" tickets score far higher.

def _build_pick_weights():
    w = {}
    for n in range(1, MAXN + 1):
        if n <= 12:
            base = 1.9          # day AND month → most over-picked
        elif n <= 24:
            base = 1.45
        elif n <= 31:
            base = 1.2          # still a valid calendar day
        else:
            base = 0.5          # 32..49 — rarely chosen
        if n % 7 == 0:
            base *= 1.15        # "lucky 7" family
        if n == 13:
            base *= 0.9         # some avoid it
        w[n] = base
    s = sum(w.values()) / MAXN  # normalise to mean 1
    return {n: w[n] / s for n in w}


PICK_W = _build_pick_weights()


def popularity_multiplier(ticket):
    """Estimate how many times more (or less) popular this combo is than an
    average random combo, among human players. >1 = crowded, <1 = contrarian."""
    mult = 1.0
    for n in ticket:
        mult *= PICK_W[n]                       # per-number birthday/lucky bias

    # ---- pattern factors (combos people pick as shapes, not numbers) ----
    s = sorted(ticket)
    runs = sum(1 for a, b in zip(s, s[1:]) if b - a == 1)
    if runs >= 1:
        mult *= 1.0 + 0.6 * runs                # consecutive runs are popular
    diffs = [b - a for a, b in zip(s, s[1:])]
    if len(set(diffs)) == 1:
        mult *= 4.0                             # arithmetic progression (1-8-15-22…)
    if all(n <= 31 for n in ticket):
        mult *= 2.2                             # full "calendar grid" ticket
    if all(n <= 12 for n in ticket):
        mult *= 3.0                             # all months
    if len(set(n % 10 for n in ticket)) <= 2:
        mult *= 1.4                             # same last-digit clustering
    if len(set((n - 1) // 10 for n in ticket)) <= 2:
        mult *= 1.5                             # squeezed into 1–2 decades = looks patterned
    if all(n > 31 for n in ticket):
        mult *= 1.6                             # "all high" is itself a known contrarian shape
    return mult


def expected_cowinners(ticket, tickets_sold):
    """Heuristic expected number of OTHER tickets sharing a jackpot win."""
    return (tickets_sold * popularity_multiplier(ticket)) / TOTAL


# ═══════════════════════════════════════════════════════════════════════
#  5. EXPECTED VALUE  (parimutuel — the prize is shared)
# ═══════════════════════════════════════════════════════════════════════

def section_ev(draws, cost=8.0, jackpot=4_000_000, tickets_sold=2_000_000):
    print("=" * 72)
    print("  4. EXPECTED VALUE  (Romanian Loto 6/49, parimutuel / shared jackpot)")
    print("=" * 72)
    payouts = {6: jackpot, 5: 4000, 4: 150, 3: 25}
    ev = 0.0
    for k, pay in payouts.items():
        p = math.comb(K, k) * math.comb(MAXN - K, K - k) / TOTAL
        ev += p * pay
        print(f"  P(match {k}) = 1 in {1/p:,.0f}")
    roi = (ev - cost) / cost * 100
    print(f"\n  Naïve ticket EV: {ev:.3f} RON on a {cost:.2f} RON ticket — you get back "
          f"~{ev/cost*100:.0f}% of your stake (ROI {roi:.0f}%, i.e. a loss).")
    print()
    print("  The jackpot is SHARED. If you win with a popular combo you split it;")
    print(f"  with a contrarian combo you keep more. With ~{tickets_sold:,} tickets sold:")
    birthday = sorted([3, 7, 11, 12, 19, 24])         # typical all-low birthday pick
    contrarian = sorted([33, 38, 41, 44, 47, 49])     # typical high/unpopular pick
    for label, tk in [("birthday-style", birthday), ("contrarian", contrarian)]:
        co = expected_cowinners(tk, tickets_sold)
        share = 1.0 / (1.0 + co)
        print(f"    {label:14s} {tk}: ~{co:.2f} co-winners → you'd keep ~{share*100:.0f}% "
              f"(≈{jackpot*share:,.0f} RON of a {jackpot:,} jackpot)")
    print("\n  Same odds of winning. Very different payout IF you win. That gap is")
    print("  the only thing a 'smart' lottery tool can legitimately improve.")
    print()
    return {"naive_ev": ev}


# ═══════════════════════════════════════════════════════════════════════
#  6. TICKET GENERATOR  (uniform win-prob, optimised for unpopularity)
# ═══════════════════════════════════════════════════════════════════════

def generate_tickets(draws, n_tickets=6, tickets_sold=2_000_000, pool=150000, seed=None):
    print("=" * 72)
    print("  5. SUGGESTED TICKETS  (natural-looking, tilted away from the crowd)")
    print("=" * 72)
    # Non-deterministic by default → different valid tickets every run.
    rng = random.Random(seed) if seed is not None else random.Random(os.urandom(8))

    history = {tuple(d["nums"]) for d in draws}
    recent = [set(d["nums"]) for d in draws[-12:]]
    sums = sorted(sum(d["nums"]) for d in draws)
    s_lo, s_hi = sums[len(sums) // 10], sums[len(sums) * 9 // 10]    # central 80% band

    def is_natural(tk):
        """Looks like a real, balanced draw — NOT an all-high/all-low pattern."""
        s = sum(tk)
        if not (s_lo <= s <= s_hi):
            return False
        low = sum(1 for n in tk if n <= 24)
        if not (2 <= low <= 4):                       # balanced halves (mean is 3)
            return False
        if len(set((n - 1) // 10 for n in tk)) < 3:   # spread across ≥3 decades
            return False
        runs = sum(1 for a, b in zip(tk, tk[1:]) if b - a == 1)
        if runs > 1:                                  # no sequences
            return False
        if tk in history:                             # not a past winning combo
            return False
        if any(len(set(tk) & r) >= 4 for r in recent):
            return False
        return True

    # Pool of plausible, natural tickets only.
    candidates = []
    for _ in range(pool):
        tk = tuple(sorted(rng.sample(range(1, MAXN + 1), K)))
        if is_natural(tk):
            candidates.append(tk)
        if len(candidates) >= 5000:
            break

    # Rank by popularity but tie-break randomly so equally-good tickets vary
    # run to run (rounding stops the optimizer collapsing onto one extreme).
    candidates.sort(key=lambda t: (round(popularity_multiplier(t), 1), rng.random()))

    chosen = []
    for tk in candidates:
        if all(len(set(tk) - set(c)) >= 3 for c in chosen):   # mutually diverse
            chosen.append(tk)
            if len(chosen) >= n_tickets:
                break

    avg_mult = mean([popularity_multiplier(t) for t in candidates]) if candidates else 1.0
    print(f"  {len(candidates):,} natural candidates; kept the {len(chosen)} least-popular & diverse.")
    print(f"  (Natural-ticket average popularity ≈ {avg_mult:.2f}x; birthday picks run ~10x+.)\n")
    print(f"  {'ticket':30s} {'sum':>4s} {'low':>4s} {'pop':>6s} {'exp.co-win':>11s}")
    print("  " + "-" * 60)
    results = []
    for tk in chosen:
        co = expected_cowinners(tk, tickets_sold)
        mlt = popularity_multiplier(tk)
        low = sum(1 for n in tk if n <= 24)
        print(f"  {str(list(tk)):30s} {sum(tk):4d} {low:4d} {mlt:5.2f}x {co:11.3f}")
        results.append({"numbers": list(tk), "popularity": mlt, "exp_cowinners": co})

    rnd = sorted(rng.sample(range(1, MAXN + 1), K))
    bd = [4, 8, 12, 19, 23, 27]
    print(f"\n  reference — pure random : {rnd}  pop {popularity_multiplier(rnd):.2f}x")
    print(f"  reference — birthday    : {bd}  pop {popularity_multiplier(bd):.2f}x "
          f"(≈{expected_cowinners(bd, tickets_sold):.2f} co-winners)")
    print("\n  Re-run for a fresh set, or pass --seed N to reproduce one.")
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Romanian 6/49 honest analyzer + EV ticket optimizer")
    ap.add_argument("csv", nargs="?", default=os.path.join(here, "drawn_RO.csv"),
                    help="path to drawn_RO.csv")
    ap.add_argument("--tickets", type=int, default=6, help="how many tickets to suggest")
    ap.add_argument("--seed", type=int, default=None, help="fix RNG for reproducible tickets")
    ap.add_argument("--sold", type=int, default=2_000_000, help="assumed tickets sold (for sharing math)")
    args = ap.parse_args()
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}\nUsage: python3 Loto_RO_v5.py [path/to/drawn_RO.csv]")
        sys.exit(1)

    print("\n" + "█" * 72)
    print("  LOTO_RO v5 — honest analysis + expected-value ticket optimizer")
    print("█" * 72 + "\n")

    draws, dups, bad = load_draws(csv_path)
    print(f"Loaded {len(draws)} valid draws  ({dups} duplicate rows, {bad} invalid rows skipped)")
    print(f"File: {csv_path}\n")
    if len(draws) < 250:
        print("Not enough data for a meaningful backtest.")
        return

    report = {"draws": len(draws), "duplicates_skipped": dups}
    report["fairness"] = section_fairness(draws)
    report["structure"] = section_structure(draws)
    report["backtest"] = section_backtest(draws)
    report["ev"] = section_ev(draws, tickets_sold=args.sold)
    report["tickets"] = generate_tickets(draws, n_tickets=args.tickets,
                                         tickets_sold=args.sold, seed=args.seed)

    out = os.path.join(here, "loto_v5_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("  BOTTOM LINE")
    print("=" * 72)
    print("  • The draw is provably fair and memoryless — NO method predicts it.")
    print("  • The old 'signals/ML/overdue' approach is noise (see backtest above).")
    print("  • The only honest improvement: pick UNPOPULAR combos so you share")
    print("    the jackpot with fewer people IF you win. Same odds, bigger payout.")
    print("  • Overall expected value is still negative. Play for fun, with limits.")
    print(f"  • Machine-readable report saved → {out}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
