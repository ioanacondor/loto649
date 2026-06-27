# Loto_RO — Romanian 6/49 honest analyzer

An honest toolkit for the Romanian 6/49 lottery. It does **not** predict winning
numbers — a fair draw is memoryless and provably unpredictable (the tools show
this on 33 years of real data). Instead it does the one thing that's
mathematically real: helps you play *smarter if you play*, by choosing
combinations few other people choose, so you'd split a jackpot fewer ways.

## Files

| File | What it does |
|---|---|
| **`Loto_RO_v5.py`** | Main tool (pure stdlib). Fairness test, honest walk-forward backtest, parimutuel expected-value with prize-sharing, and an **unpopularity-optimized ticket generator**. |
| **`backtest_honest.py`** | Standalone walk-forward backtest: proves "hot/cold/overdue/recency" strategies don't beat random. |
| **`explore_outside_box.py`** | Tests every legitimate way history *could* predict the future (machine bias, position bias, recurring pairs, seasonality, sum). All come back negative. |
| **`drawn_RO.csv`** | Draw history (Date, N1…N6). |
| `Loto_RO.py` | Original v4 (kept for reference — superseded by v5). |

## Run

```bash
python3 Loto_RO_v5.py            # full report
python3 explore_outside_box.py   # the "can history predict?" hunt
python3 backtest_honest.py       # the proof, standalone
```

No dependencies — standard library only.

## The honest bottom line

- The draw is **fair and memoryless** → no method predicts it. The backtest and
  the outside-the-box hunt confirm this empirically.
- You **can't** change your odds of winning. You **can** raise your expected
  payout *if* you win, by picking unpopular combinations (the jackpot is shared).
- Overall expected value is still negative. Play for fun, with limits.
