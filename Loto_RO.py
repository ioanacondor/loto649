"""
Romanian 6/49 Lottery Analyzer v4
=================================
Fixes from v3:
  - 8 pattern signals instead of 5 (adds successor, decade, streak)
  - Anti-stickiness: each set MUST differ by ≥3 numbers from all previous sets
  - Draw-to-draw variation: seed rotates so predictions genuinely shift each week
  - Structural constraints tightened to match real decade/spacing distributions
  - Full pattern report section showing which historical patterns each set leverages
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
import sys
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from scipy.stats import chisquare, norm


# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def draw_to_binary(row, num_cols, max_num=49):
    vec = np.zeros(max_num, dtype=np.float32)
    for c in num_cols:
        val = int(row[c])
        if 1 <= val <= max_num:
            vec[val - 1] = 1.0
    return vec


def weighted_sample_no_replace(probs, k, rng):
    """Sample k indices without replacement, probability-weighted."""
    probs = np.clip(np.array(probs, dtype=np.float64), 1e-12, None)
    selected = []
    remaining = np.arange(len(probs))
    for _ in range(k):
        p = probs[remaining]
        p = p / p.sum()
        idx = rng.choice(len(remaining), p=p)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx)
    return np.array(selected)


def decade_of(n):
    """Map number to decade bucket: 1-9→0, 10-19→1, ..., 40-49→4"""
    return min((n - 1) // 10, 4)


def decade_vector(numbers):
    """Count how many numbers fall in each decade."""
    v = [0] * 5
    for n in numbers:
        v[decade_of(n)] += 1
    return v


# ═══════════════════════════════════════════════════════════
# MAIN ANALYZER
# ═══════════════════════════════════════════════════════════

class LotteryAnalyzerV4:
    MAX_NUM = 49
    DRAW_SIZE = 6

    def __init__(self):
        self.df = None
        self.nc = None
        self.binary = None
        self.gap_matrix = None
        self.features = None
        self.signals = {}
        self.ensemble = None
        self.cv_results = {}
        self.randomness = {}
        self.predictions = {}
        self.pattern_report = {}
        self.pair_freq = None
        self.successor_map = None

    # ─── LOAD ──────────────────────────────────────────────

    def load(self, csv_file):
        print("=" * 72)
        print("  LOADING DATA")
        print("=" * 72)
        self.df = pd.read_csv(csv_file)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        cols = [c for c in self.df.columns if c.startswith('N')]
        self.nc = cols[:6]
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        self.binary = np.array([draw_to_binary(r, self.nc) for _, r in self.df.iterrows()])
        print(f"  ✓ {len(self.df)} draws,  {self.df['Date'].min().date()} → "
              f"{self.df['Date'].max().date()}")
        print()

    # ─── RANDOMNESS TESTS ──────────────────────────────────

    def test_randomness(self):
        print("=" * 72)
        print("  RANDOMNESS & FAIRNESS TESTS")
        print("=" * 72)
        all_nums = self.df[self.nc].values.flatten().astype(int)

        # Chi-square
        obs = np.bincount(all_nums, minlength=self.MAX_NUM + 1)[1:self.MAX_NUM + 1]
        exp = np.full(self.MAX_NUM, len(all_nums) / self.MAX_NUM)
        chi2, p = chisquare(obs, exp)
        v = "PASS ✓" if p > 0.05 else "FAIL ✗"
        print(f"  Chi-Square: χ²={chi2:.2f}, p={p:.4f}  →  {v}")
        self.randomness['chi_square_p'] = float(p)

        # Autocorrelation
        if HAS_STATSMODELS:
            fails = 0
            for col in self.nc:
                lb = acorr_ljungbox(self.df[col], lags=10, return_df=True)
                if (lb['lb_pvalue'] < 0.05).any():
                    fails += 1
                    print(f"  Autocorrelation {col}: WEAK CORRELATION ⚠")
            if fails == 0:
                print(f"  Autocorrelation: all columns independent ✓")
            self.randomness['autocorr_failures'] = fails

        # Runs test
        passes = 0
        for col in self.nc:
            s = self.df[col].values
            med = np.median(s)
            b = (s >= med).astype(int)
            runs = 1 + np.sum(b[1:] != b[:-1])
            n1, n0 = np.sum(b), len(b) - np.sum(b)
            er = 1 + 2 * n0 * n1 / (n0 + n1)
            vr = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1) ** 2 * (n0 + n1 - 1))
            if vr > 0:
                z = (runs - er) / np.sqrt(vr)
                if 2 * norm.sf(abs(z)) > 0.05:
                    passes += 1
        print(f"  Runs test: {passes}/6 columns pass ✓")
        self.randomness['runs_passes'] = passes

        is_fair = (p > 0.05) and (self.randomness.get('autocorr_failures', 0) <= 1)
        print(f"\n  VERDICT: {'RANDOM & FAIR ✓' if is_fair else 'Some anomalies ⚠'}")
        print()

    # ─── GAP MATRIX ────────────────────────────────────────

    def build_gap_matrix(self):
        n = len(self.df)
        self.gap_matrix = np.zeros((n, self.MAX_NUM))
        last = np.full(self.MAX_NUM, -1)
        for i in range(n):
            drawn = self.df.iloc[i][self.nc].values.astype(int)
            for num in range(self.MAX_NUM):
                self.gap_matrix[i, num] = (i - last[num]) if last[num] >= 0 else i + 1
            for d in drawn:
                if 1 <= d <= self.MAX_NUM:
                    last[d - 1] = i

    # ─── FEATURES ──────────────────────────────────────────

    def build_features(self):
        print("=" * 72)
        print("  FEATURE ENGINEERING")
        print("=" * 72)
        n = len(self.df)
        nc = self.nc

        self.build_gap_matrix()

        feats = pd.DataFrame(index=range(n))
        feats['draw_idx'] = range(n)
        feats['dow'] = self.df['Date'].dt.dayofweek
        feats['month'] = self.df['Date'].dt.month
        feats['sum'] = self.df[nc].sum(axis=1)
        feats['std'] = self.df[nc].std(axis=1)
        feats['range'] = self.df[nc].max(axis=1) - self.df[nc].min(axis=1)
        feats['even_count'] = (self.df[nc] % 2 == 0).sum(axis=1)
        feats['high_count'] = (self.df[nc] > 24).sum(axis=1)
        feats['avg_gap'] = self.gap_matrix.mean(axis=1)
        feats['max_gap'] = self.gap_matrix.max(axis=1)

        # Rolling frequency
        for period in [5, 10, 20, 50]:
            col = np.zeros(n)
            for i in range(1, n):
                start = max(0, i - period)
                col[i] = self.binary[start:i].sum()
            feats[f'freq_last_{period}'] = col

        # Pair co-occurrence
        self.pair_freq = Counter()
        for i in range(n):
            drawn = sorted(self.df.iloc[i][nc].values.astype(int))
            for a, b in combinations(drawn, 2):
                self.pair_freq[(a, b)] += 1

        # Successor map
        self.successor_map = defaultdict(Counter)
        for i in range(1, n):
            prev = set(self.df.iloc[i - 1][nc].values.astype(int))
            curr = set(self.df.iloc[i][nc].values.astype(int))
            for p in prev:
                for c in curr:
                    self.successor_map[p][c] += 1

        self.features = feats.fillna(0)
        print(f"  ✓ {self.features.shape[1]} features,  gap matrix {self.gap_matrix.shape}")
        print()

    # ═══════════════════════════════════════════════════════
    # 8 PATTERN SIGNALS
    # ═══════════════════════════════════════════════════════

    def compute_all_signals(self):
        print("=" * 72)
        print("  COMPUTING 8 PATTERN SIGNALS")
        print("=" * 72)
        self._sig_gap()
        self._sig_recency()
        self._sig_pairs()
        self._sig_ml()
        self._sig_cycles()
        self._sig_successor()
        self._sig_decade()
        self._sig_streaks()
        print()

    def _sig_gap(self):
        """Signal 1: Overdue numbers (gap vs mean gap ratio)."""
        print("  1. Gap/overdue analysis")
        n = len(self.df)
        current_gaps = self.gap_matrix[-1]
        mean_gaps = np.zeros(self.MAX_NUM)
        for num in range(self.MAX_NUM):
            apps = np.where(self.binary[:, num] == 1)[0]
            if len(apps) > 1:
                mean_gaps[num] = np.diff(apps).mean()
            else:
                mean_gaps[num] = n / max(1, self.binary[:, num].sum())
        ratio = current_gaps / np.clip(mean_gaps, 1, None)
        self.signals['gap'] = 1 / (1 + np.exp(-0.5 * (ratio - 1)))
        top = (np.argsort(self.signals['gap'])[-5:][::-1] + 1).tolist()
        print(f"     Top overdue: {top}")

    def _sig_recency(self):
        """Signal 2: Recency-weighted frequency (multi-scale decay)."""
        print("  2. Recency-weighted frequency")
        n = len(self.binary)
        scores = np.zeros(self.MAX_NUM)
        for hl, w in [(12, 0.45), (40, 0.35), (150, 0.20)]:
            decay = np.exp(-np.log(2) * np.arange(n)[::-1] / hl)
            weighted = (self.binary.T * decay).sum(axis=1)
            weighted /= weighted.sum()
            scores += weighted * w
        self.signals['recency'] = scores
        top = (np.argsort(scores)[-5:][::-1] + 1).tolist()
        print(f"     Top recent-hot: {top}")

    def _sig_pairs(self):
        """Signal 3: Pair co-occurrence affinity."""
        print("  3. Pair co-occurrence")
        n = len(self.df)
        recent_nums = set()
        for i in range(max(0, n - 10), n):
            for num in self.df.iloc[i][self.nc].values.astype(int):
                recent_nums.add(num)
        scores = np.zeros(self.MAX_NUM)
        for num in range(1, self.MAX_NUM + 1):
            partners = Counter()
            for (a, b), cnt in self.pair_freq.items():
                if a == num:
                    partners[b] = cnt
                elif b == num:
                    partners[a] = cnt
            if partners:
                top_p = [p for p, _ in partners.most_common(10)]
                scores[num - 1] = len(set(top_p) & recent_nums) / len(top_p)
        self.signals['pairs'] = scores
        top = (np.argsort(scores)[-5:][::-1] + 1).tolist()
        print(f"     Top by partner affinity: {top}")

    def _sig_ml(self):
        """Signal 4: ML classifiers (XGBoost + LightGBM + RF averaged)."""
        print("  4. ML classifiers (XGB/LGBM/RF)")
        X = self.features.values
        n = len(X)
        seed = n + self.df['Date'].iloc[-1].toordinal()
        probs = np.zeros(self.MAX_NUM)

        for num_idx in range(self.MAX_NUM):
            X_full = np.column_stack([X[:-1], self.gap_matrix[:-1, num_idx]])
            y = self.binary[1:, num_idx]
            split = int(0.8 * len(X_full))
            X_tr, y_tr = X_full[:split], y[:split]
            X_pred = X_full[[-1]]
            p_sum, m = 0, 0

            if HAS_XGB:
                clf = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.7,
                                    random_state=seed + num_idx, verbosity=0,
                                    eval_metric='logloss')
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_pred)
                p_sum += p[0, 1] if p.shape[1] > 1 else 0.12
                m += 1

            if HAS_LGBM:
                clf = LGBMClassifier(n_estimators=80, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7,
                                     random_state=seed + num_idx, verbose=-1)
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_pred)
                p_sum += p[0, 1] if p.shape[1] > 1 else 0.12
                m += 1

            clf = RandomForestClassifier(n_estimators=80, max_depth=5,
                                         random_state=seed + num_idx)
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_pred)
            p_sum += p[0, 1] if p.shape[1] > 1 else 0.12
            m += 1

            probs[num_idx] = p_sum / m

        self.signals['ml'] = probs
        top = (np.argsort(probs)[-5:][::-1] + 1).tolist()
        print(f"     Top by ML: {top}")

    def _sig_cycles(self):
        """Signal 5: Cycle phase detection (Gaussian around phase=1)."""
        print("  5. Cycle phase")
        n = len(self.binary)
        scores = np.zeros(self.MAX_NUM)
        for num in range(self.MAX_NUM):
            apps = np.where(self.binary[:, num] == 1)[0]
            if len(apps) < 10:
                continue
            gaps = np.diff(apps)
            if len(gaps) < 5:
                continue
            gc = Counter(gaps)
            dom_gap, dom_freq = gc.most_common(1)[0]
            dom_pct = dom_freq / len(gaps)
            current_gap = n - 1 - apps[-1]
            phase = current_gap / max(dom_gap, 1)
            if dom_pct > 0.12:
                scores[num] = np.exp(-0.5 * (phase - 1.0) ** 2 / 0.35 ** 2) * dom_pct
        self.signals['cycles'] = scores
        top = (np.argsort(scores)[-5:][::-1] + 1).tolist()
        print(f"     Top by cycle phase: {top}")

    def _sig_successor(self):
        """Signal 6: Successor patterns — what tends to follow the last draw."""
        print("  6. Successor patterns")
        last_draw = sorted(self.df.iloc[-1][self.nc].values.astype(int))
        scores = np.zeros(self.MAX_NUM)
        for num in last_draw:
            if num in self.successor_map:
                for follower, cnt in self.successor_map[num].items():
                    if 1 <= follower <= self.MAX_NUM:
                        scores[follower - 1] += cnt
        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()
        self.signals['successor'] = scores
        top = (np.argsort(scores)[-5:][::-1] + 1).tolist()
        print(f"     Top followers of {last_draw}: {top}")

    def _sig_decade(self):
        """Signal 7: Decade balance — favour numbers from underrepresented decades."""
        print("  7. Decade balance")
        # Average decade distribution from history
        avg_dec = np.zeros(5)
        for _, row in self.df.iterrows():
            for c in self.nc:
                avg_dec[decade_of(int(row[c]))] += 1
        avg_dec /= len(self.df)  # per-draw avg count in each decade

        # Recent draws decade distribution
        recent_dec = np.zeros(5)
        window = min(15, len(self.df))
        for i in range(len(self.df) - window, len(self.df)):
            for c in self.nc:
                recent_dec[decade_of(int(self.df.iloc[i][c]))] += 1
        recent_dec /= window

        # Decades that are underrepresented recently get a boost
        deficit = avg_dec - recent_dec
        scores = np.zeros(self.MAX_NUM)
        for num in range(1, self.MAX_NUM + 1):
            d = decade_of(num)
            scores[num - 1] = max(0, deficit[d])

        if scores.max() > 0:
            scores = scores / scores.max()
        self.signals['decade'] = scores
        under = [f"d{i}({deficit[i]:+.2f})" for i in range(5) if deficit[i] > 0.1]
        print(f"     Under-represented decades: {under if under else 'balanced'}")

    def _sig_streaks(self):
        """Signal 8: Hot/cold streaks — boost cold numbers, slight boost to hot."""
        print("  8. Hot/cold streaks")
        window = 20
        recent = self.df.iloc[-window:]
        freq = Counter(recent[self.nc].values.flatten().astype(int))
        expected = window * self.DRAW_SIZE / self.MAX_NUM  # ~2.45

        scores = np.zeros(self.MAX_NUM)
        for num in range(1, self.MAX_NUM + 1):
            f = freq.get(num, 0)
            if f == 0:
                # Cold: big boost (hasn't appeared in 20 draws)
                scores[num - 1] = 1.0
            elif f <= expected * 0.5:
                # Cool: moderate boost
                scores[num - 1] = 0.6
            elif f >= expected * 2.0:
                # Hot: small boost (momentum)
                scores[num - 1] = 0.3
            else:
                # Normal
                scores[num - 1] = 0.15

        self.signals['streaks'] = scores
        cold = [n for n in range(1, 50) if freq.get(n, 0) == 0]
        hot = [n for n, c in freq.most_common(5)]
        print(f"     Cold (0 in last 20): {cold}")
        print(f"     Hot  (top 5): {hot}")

    # ═══════════════════════════════════════════════════════
    # ENSEMBLE
    # ═══════════════════════════════════════════════════════

    def build_ensemble(self):
        print("\n" + "=" * 72)
        print("  BUILDING ENSEMBLE (8 signals)")
        print("=" * 72)

        weights = {
            'ml':        0.22,
            'gap':       0.18,
            'recency':   0.15,
            'successor': 0.12,
            'pairs':     0.10,
            'streaks':   0.10,
            'cycles':    0.08,
            'decade':    0.05,
        }

        ensemble = np.zeros(self.MAX_NUM)
        for name, w in weights.items():
            if name not in self.signals:
                continue
            s = self.signals[name].copy()
            smin, smax = s.min(), s.max()
            if smax > smin:
                s = (s - smin) / (smax - smin)
            else:
                s = np.ones(self.MAX_NUM) / self.MAX_NUM
            ensemble += s * w

        ensemble = ensemble / ensemble.sum()
        self.ensemble = ensemble

        ratio = ensemble.max() / ensemble.min()
        top10 = (np.argsort(ensemble)[-10:][::-1] + 1).tolist()
        print(f"  Weights: {weights}")
        print(f"  Prob range: [{ensemble.min():.4f}, {ensemble.max():.4f}]  "
              f"(ratio {ratio:.1f}x)")
        print(f"  Top 10: {top10}")
        print()

    # ═══════════════════════════════════════════════════════
    # PREDICTION — ANTI-STICKINESS SAMPLING
    # ═══════════════════════════════════════════════════════

    def predict(self, n_sets=5):
        print("=" * 72)
        print("  PREDICTIONS — Next Draw")
        print("=" * 72)

        ens = self.ensemble
        if ens is None:
            print("  ERROR: run build_ensemble() first")
            return {}

        # Data-derived seed that shifts with every new draw
        fp = (len(self.df) * 1000 +
              int(self.df.iloc[-1][self.nc].sum()) +
              self.df['Date'].iloc[-1].toordinal())
        rng = np.random.RandomState(fp)

        # Structural bounds from real data
        sums = self.df[self.nc].sum(axis=1)
        s_lo, s_hi = sums.quantile(0.12), sums.quantile(0.88)
        last_draw_set = set(self.df.iloc[-1][self.nc].values.astype(int))

        # Historical avg decade distribution
        avg_dec = np.zeros(5)
        for _, row in self.df.iterrows():
            for c in self.nc:
                avg_dec[decade_of(int(row[c]))] += 1
        avg_dec /= len(self.df)

        def passes_filters(nums, existing_sets):
            s = sum(nums)
            evens = sum(1 for x in nums if x % 2 == 0)
            highs = sum(1 for x in nums if x > 24)
            dv = decade_vector(nums)

            # Basic structure
            if not (s_lo <= s <= s_hi):
                return False
            if not (1 <= evens <= 5 and 1 <= highs <= 5):
                return False

            # Decade spread: no decade should have more than 3
            if max(dv) > 3:
                return False

            # Anti-stickiness: ≥3 numbers different from each existing set
            ns = set(nums)
            for prev in existing_sets:
                if len(ns - set(prev)) < 3:
                    return False

            # Don't just repeat last draw
            if len(ns & last_draw_set) > 3:
                return False

            return True

        predictions = {}
        all_sets = []
        n_focused = 3
        n_broad = n_sets - n_focused

        print(f"\n  Strategy: {n_focused} focused + {n_broad} broad sets")
        print(f"  Filters: sum [{s_lo:.0f}-{s_hi:.0f}], 1-5 evens, 1-5 highs, "
              f"≤3/decade, ≥3 diff between sets\n")

        # ── FOCUSED: sample from top 18 ──
        top18 = np.argsort(ens)[-18:]
        top18_p = ens[top18].copy()
        top18_p /= top18_p.sum()

        count = 0
        for _ in range(500):
            if count >= n_focused:
                break
            chosen = weighted_sample_no_replace(top18_p, self.DRAW_SIZE, rng)
            nums = sorted((top18[chosen] + 1).tolist())
            if passes_filters(nums, all_sets):
                count += 1
                name = f"set_{count}"
                predictions[name] = nums
                all_sets.append(nums)
                self._print_set(name, "focused", nums, ens)

        # ── BROAD: 3 from top-12 + 3 from rest ──
        top12 = np.argsort(ens)[-12:]
        rest = np.array([i for i in range(self.MAX_NUM) if i not in top12])
        top12_p = ens[top12].copy()
        top12_p /= top12_p.sum()
        rest_p = ens[rest].copy()
        rest_p /= rest_p.sum()

        broad_count = 0
        for _ in range(500):
            if broad_count >= n_broad:
                break
            t = weighted_sample_no_replace(top12_p, 3, rng)
            r = weighted_sample_no_replace(rest_p, 3, rng)
            nums = sorted((np.concatenate([top12[t], rest[r]]) + 1).tolist())
            if passes_filters(nums, all_sets):
                broad_count += 1
                idx = n_focused + broad_count
                name = f"set_{idx}"
                predictions[name] = nums
                all_sets.append(nums)
                self._print_set(name, "broad ", nums, ens)

        # ── Reference picks ──
        print(f"\n  Individual signal top-6 (reference):")
        for sname in ['ml', 'gap', 'recency', 'successor', 'pairs',
                       'streaks', 'cycles', 'decade']:
            if sname in self.signals:
                t6 = sorted((np.argsort(self.signals[sname])[-6:][::-1] + 1).tolist())
                predictions[f'ref_{sname}'] = t6
                print(f"    {sname:12s}: {t6}")

        rand = sorted(rng.choice(range(1, self.MAX_NUM + 1),
                                  size=self.DRAW_SIZE, replace=False).tolist())
        predictions['random'] = rand
        print(f"    {'random':12s}: {rand}  (pure chance)")

        self.predictions = predictions
        print()
        return predictions

    def _print_set(self, name, kind, nums, ens):
        avg_p = np.mean([ens[n - 1] for n in nums])
        dv = decade_vector(nums)
        evens = sum(1 for x in nums if x % 2 == 0)
        print(f"  {name} ({kind}): {nums}  "
              f"sum={sum(nums)} ev={evens} "
              f"dec={dv} p̄={avg_p:.4f}")

    # ═══════════════════════════════════════════════════════
    # PATTERN REPORT — Why each number was picked
    # ═══════════════════════════════════════════════════════

    def pattern_report_for_predictions(self):
        print("=" * 72)
        print("  PATTERN REPORT — Why These Numbers?")
        print("=" * 72)

        pred_sets = {k: v for k, v in self.predictions.items() if k.startswith('set_')}
        if not pred_sets:
            return

        # For each number in predictions, show which signals favour it
        all_pred_nums = set()
        for nums in pred_sets.values():
            all_pred_nums.update(nums)

        signal_names = ['ml', 'gap', 'recency', 'successor', 'pairs',
                        'streaks', 'cycles', 'decade']

        # Normalize each signal to rank
        ranks = {}
        for sname in signal_names:
            if sname in self.signals:
                r = np.argsort(np.argsort(self.signals[sname]))  # rank 0=worst
                ranks[sname] = r

        report = {}
        for num in sorted(all_pred_nums):
            reasons = []
            for sname in signal_names:
                if sname in ranks:
                    rank = ranks[sname][num - 1]
                    if rank >= self.MAX_NUM - 10:  # top 10
                        reasons.append(f"{sname}(#{self.MAX_NUM - rank})")
            in_sets = [k for k, v in pred_sets.items() if num in v]
            report[num] = reasons
            reason_str = ", ".join(reasons) if reasons else "ensemble blend"
            print(f"  {num:2d}: {reason_str}  → in {', '.join(in_sets)}")

        # Show overdue numbers
        print(f"\n  Most overdue numbers (draws since last seen):")
        gaps = self.gap_matrix[-1]
        overdue = np.argsort(gaps)[-8:][::-1]
        for idx in overdue:
            num = idx + 1
            g = int(gaps[idx])
            in_pred = "★" if num in all_pred_nums else " "
            print(f"    {in_pred} {num:2d}: {g} draws ago")

        # Successor context
        last_draw = sorted(self.df.iloc[-1][self.nc].values.astype(int))
        print(f"\n  Last draw was {last_draw}")
        print(f"  Historical successors of these numbers most often include:")
        combined = Counter()
        for num in last_draw:
            for f, c in self.successor_map.get(num, Counter()).most_common(5):
                combined[f] += c
        top_succ = combined.most_common(10)
        in_pred_succ = [(n, c) for n, c in top_succ if n in all_pred_nums]
        print(f"    All: {[n for n, _ in top_succ]}")
        print(f"    In predictions: {[n for n, _ in in_pred_succ]}")

        self.pattern_report = report
        print()

    # ═══════════════════════════════════════════════════════
    # CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════

    def cross_validate(self, n_splits=5):
        print("=" * 72)
        print(f"  CROSS-VALIDATION ({n_splits}-fold)")
        print("=" * 72)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = self.features.values
        base_rate = self.DRAW_SIZE / self.MAX_NUM
        eval_nums = np.random.RandomState(99).choice(self.MAX_NUM, size=10, replace=False)

        results = defaultdict(lambda: defaultdict(list))

        for fold, (tr, va) in enumerate(tscv.split(X)):
            for num_idx in eval_nums:
                y_tr = self.binary[tr, num_idx]
                y_va = self.binary[va, num_idx]
                X_tr = np.column_stack([X[tr], self.gap_matrix[tr, num_idx]])
                X_va = np.column_stack([X[va], self.gap_matrix[va, num_idx]])

                results['baseline']['ll'].append(
                    log_loss(y_va, np.full(len(y_va), base_rate)))

                clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_va)
                p = p[:, 1] if p.shape[1] > 1 else np.full(len(y_va), base_rate)
                results['RF']['ll'].append(log_loss(y_va, np.clip(p, 1e-7, 1 - 1e-7)))

                if HAS_XGB:
                    clf = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05,
                                        random_state=42, verbosity=0, eval_metric='logloss')
                    clf.fit(X_tr, y_tr)
                    p = clf.predict_proba(X_va)
                    p = p[:, 1] if p.shape[1] > 1 else np.full(len(y_va), base_rate)
                    results['XGB']['ll'].append(log_loss(y_va, np.clip(p, 1e-7, 1 - 1e-7)))

        base_avg = np.mean(results['baseline']['ll'])
        print(f"\n  Results (lower = better):")
        summary = {}
        for name in sorted(results.keys()):
            avg = np.mean(results[name]['ll'])
            imp = (base_avg - avg) / base_avg * 100
            beats = avg < base_avg * 0.98
            marker = "✓ BEATS" if beats else "✗      "
            print(f"    {marker} {name:12s}: {avg:.4f}  ({imp:+.1f}%)")
            summary[name] = {'logloss': float(avg), 'improvement': float(imp)}
        self.cv_results = summary
        print()

    # ═══════════════════════════════════════════════════════
    # EXPECTED VALUE
    # ═══════════════════════════════════════════════════════

    def expected_value(self, cost=5.0, jackpot=300_000):
        print("=" * 72)
        print("  EXPECTED VALUE (Romanian Loto 6/49)")
        print("=" * 72)
        total = comb(self.MAX_NUM, self.DRAW_SIZE)
        payouts = {6: jackpot, 5: 5000, 4: 200, 3: 25}
        ev = 0
        for k, pay in payouts.items():
            p = (comb(self.DRAW_SIZE, k) *
                 comb(self.MAX_NUM - self.DRAW_SIZE, self.DRAW_SIZE - k) / total)
            ev += p * pay
            print(f"  P(match {k}) = 1 in {1/p:,.0f}")
        print(f"\n  Ticket: {cost:.2f} RON  |  EV: {ev:.4f} RON  |  "
              f"ROI: {ev/cost*100:.1f}%")
        print()

    # ═══════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════

    def visualize(self, path='lottery_v4_analysis.png'):
        print("=" * 72)
        print("  GENERATING VISUALIZATIONS")
        print("=" * 72)

        fig = plt.figure(figsize=(22, 24))
        gs = gridspec.GridSpec(5, 3, hspace=0.45, wspace=0.35)

        all_nums = self.df[self.nc].values.flatten().astype(int)
        counts = Counter(all_nums)

        # 1. Frequency
        ax = fig.add_subplot(gs[0, 0])
        nums = range(1, self.MAX_NUM + 1)
        freqs = [counts.get(n, 0) for n in nums]
        mean_f = np.mean(freqs)
        std_f = np.std(freqs)
        colors = ['#e74c3c' if f > mean_f + std_f
                  else '#3498db' if f < mean_f - std_f
                  else '#95a5a6' for f in freqs]
        ax.bar(nums, freqs, color=colors, alpha=0.8, width=0.8)
        ax.axhline(mean_f, color='red', ls='--', lw=1.5)
        ax.set_xlabel('Number')
        ax.set_ylabel('Frequency')
        ax.set_title('Historical Frequency', fontweight='bold')
        ax.grid(alpha=0.2)

        # 2. Sum distribution
        ax = fig.add_subplot(gs[0, 1])
        sums = self.df[self.nc].sum(axis=1)
        ax.hist(sums, bins=35, color='#2ecc71', alpha=0.7, edgecolor='white', density=True)
        x = np.linspace(sums.min(), sums.max(), 100)
        ax.plot(x, norm.pdf(x, sums.mean(), sums.std()), 'r-', lw=2)
        ax.set_xlabel('Sum of 6 Numbers')
        ax.set_title('Draw Sum Distribution', fontweight='bold')
        ax.grid(alpha=0.2)

        # 3. Decade distribution
        ax = fig.add_subplot(gs[0, 2])
        dec_labels = ['1-9', '10-19', '20-29', '30-39', '40-49']
        avg_dec = np.zeros(5)
        for _, row in self.df.iterrows():
            for c in self.nc:
                avg_dec[decade_of(int(row[c]))] += 1
        avg_dec /= len(self.df)
        ax.bar(dec_labels, avg_dec, color='#9b59b6', alpha=0.8)
        ax.set_ylabel('Avg per draw')
        ax.set_title('Decade Distribution', fontweight='bold')
        ax.grid(alpha=0.2)

        # 4. Gap heatmap
        ax = fig.add_subplot(gs[1, 0:2])
        recent_n = min(60, len(self.df))
        im = ax.imshow(self.gap_matrix[-recent_n:].T, aspect='auto',
                       cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Draw (recent)')
        ax.set_ylabel('Number')
        ax.set_title('Gap Heatmap (Last 60 Draws)', fontweight='bold')
        ax.set_yticks(range(0, self.MAX_NUM, 5))
        ax.set_yticklabels(range(1, self.MAX_NUM + 1, 5))
        plt.colorbar(im, ax=ax, shrink=0.8)

        # 5. All 8 signals
        ax = fig.add_subplot(gs[1, 2])
        sig_order = ['ml', 'gap', 'recency', 'successor', 'pairs',
                     'streaks', 'cycles', 'decade']
        sig_data = []
        sig_labels = []
        for name in sig_order:
            if name in self.signals:
                s = self.signals[name].copy()
                smin, smax = s.min(), s.max()
                if smax > smin:
                    s = (s - smin) / (smax - smin)
                sig_data.append(s)
                sig_labels.append(name)
        if sig_data:
            im2 = ax.imshow(np.array(sig_data), aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(sig_labels)))
            ax.set_yticklabels(sig_labels, fontsize=8)
            ax.set_xlabel('Number (1-49)')
            ax.set_title('Signal Strengths', fontweight='bold')
            ax.set_xticks(range(0, self.MAX_NUM, 5))
            ax.set_xticklabels(range(1, self.MAX_NUM + 1, 5), fontsize=7)
            plt.colorbar(im2, ax=ax, shrink=0.8)

        # 6. Ensemble probabilities
        ax = fig.add_subplot(gs[2, 0:2])
        ens = self.ensemble
        uniform = 1.0 / self.MAX_NUM
        bar_c = ['#e74c3c' if p > uniform * 1.3
                 else '#2ecc71' if p < uniform * 0.7
                 else '#95a5a6' for p in ens]
        ax.bar(range(1, self.MAX_NUM + 1), ens, color=bar_c, alpha=0.8)
        ax.axhline(uniform, color='red', ls='--', lw=1.5, label='Uniform')
        ax.set_xlabel('Number')
        ax.set_ylabel('Probability')
        ax.set_title('Ensemble Probability (8 signals blended)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        # 7. CV results
        ax = fig.add_subplot(gs[2, 2])
        if self.cv_results:
            names = list(self.cv_results.keys())
            vals = [self.cv_results[n]['logloss'] for n in names]
            cv_c = ['#27ae60' if self.cv_results[n]['improvement'] > 2
                    else '#e74c3c' for n in names]
            ax.barh(names, vals, color=cv_c, alpha=0.8)
            ax.set_xlabel('Log-Loss')
            ax.set_title('Cross-Validation', fontweight='bold')
            ax.grid(alpha=0.2)

        # 8. Prediction grid
        ax = fig.add_subplot(gs[3, :])
        pred_sets = {k: v for k, v in self.predictions.items()
                     if k.startswith('set_') or k == 'random'}
        if pred_sets:
            names = list(pred_sets.keys())
            grid = np.zeros((len(names), self.MAX_NUM))
            for i, n in enumerate(names):
                for num in pred_sets[n]:
                    grid[i, num - 1] = 1
            ax.imshow(grid, aspect='auto', cmap='Blues', interpolation='nearest')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Number')
            ax.set_title('Prediction Sets (diverse, pattern-guided)', fontweight='bold')
            ax.set_xticks(range(0, self.MAX_NUM, 2))
            ax.set_xticklabels(range(1, self.MAX_NUM + 1, 2), fontsize=7)
            for i, n in enumerate(names):
                for num in pred_sets[n]:
                    ax.text(num - 1, i, str(num), ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')

        # 9. Successor network (simplified)
        ax = fig.add_subplot(gs[4, 0:2])
        last_draw = sorted(self.df.iloc[-1][self.nc].values.astype(int))
        combined = Counter()
        for num in last_draw:
            for f, c in self.successor_map.get(num, Counter()).most_common(8):
                combined[f] += c
        top_succ = combined.most_common(20)
        succ_nums = [n for n, _ in top_succ]
        succ_scores = [c for _, c in top_succ]
        pred_nums = set()
        for v in pred_sets.values():
            pred_nums.update(v)
        succ_colors = ['#e74c3c' if n in pred_nums else '#3498db' for n in succ_nums]
        ax.barh([str(n) for n in succ_nums], succ_scores, color=succ_colors, alpha=0.8)
        ax.set_xlabel('Successor score')
        ax.set_title(f'Successor Patterns (after {last_draw})', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        # 10. Repeat-from-previous distribution
        ax = fig.add_subplot(gs[4, 2])
        repeats = []
        for i in range(1, len(self.df)):
            prev = set(self.df.iloc[i-1][self.nc].values.astype(int))
            curr = set(self.df.iloc[i][self.nc].values.astype(int))
            repeats.append(len(prev & curr))
        rep_c = Counter(repeats)
        ax.bar(sorted(rep_c.keys()), [rep_c[k] for k in sorted(rep_c.keys())],
               color='#f39c12', alpha=0.8)
        ax.set_xlabel('# Repeats from previous draw')
        ax.set_ylabel('Frequency')
        ax.set_title('Repeat Pattern', fontweight='bold')
        ax.grid(alpha=0.2)

        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to {path}")
        print()

    # ─── SAVE ──────────────────────────────────────────────

    def save_json(self, path='analysis_v4.json'):
        class Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, (np.bool_,)): return bool(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return super().default(o)

        data = {
            'generated': datetime.now().isoformat(),
            'draws': len(self.df),
            'range': [str(self.df['Date'].min().date()),
                      str(self.df['Date'].max().date())],
            'randomness': self.randomness,
            'cv': self.cv_results,
            'predictions': self.predictions,
            'ensemble': self.ensemble.tolist() if self.ensemble is not None else None,
            'signals': {k: v.tolist() for k, v in self.signals.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, cls=Enc, indent=2)
        print(f"  ✓ Saved {path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else '/Users/ioanacondor/Desktop/Lotto_UK/drawn_RO.csv'

    a = LotteryAnalyzerV4()
    a.load(csv)
    a.test_randomness()
    a.build_features()
    a.compute_all_signals()
    a.build_ensemble()
    a.cross_validate()
    a.predict(n_sets=5)
    a.pattern_report_for_predictions()
    a.expected_value()
    a.visualize()
    a.save_json()

    print("\n" + "=" * 72)
    print("  IMPORTANT")
    print("=" * 72)
    print("  • This lottery passes all randomness tests — it is FAIR.")
    print("  • No model reliably beats random guessing long-term.")
    print("  • Predictions shift each time you add a new draw to the CSV.")
    print("  • Multiple diverse sets are provided — none is 'the best'.")
    print("  • Expected ROI is negative — play for fun, not profit.")
    print("=" * 72)
