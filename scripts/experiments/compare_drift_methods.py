"""One-off experiment behind decisions.md D4/D5: which reference strategy should the
retrain trigger use, and which per-sensor check should the sensor-fault alert use?

Compared on real FD001 (5 folds of 20 engines; thresholds set at the 99th percentile of
normal windows from folds 0-2, rates reported on folds 3-4 only):
- reference: full run-to-failure trajectories ("full"), NASA-style truncated mid-life
  trajectories ("mid"), or life-stage matched by predicted RUL ("matched");
- per-column statistic: marginal normed Wasserstein, or the same on residuals of each
  sensor regressed on the others ("resid");
- decision rule: more than half the columns ("share>0.5", the retrain rule) or any
  single column ("any_column", the sensor-alert rule).

Kept for reproducibility; production code is pdm.drift.life_stage / sensor_check and
the thresholds come from scripts/calibrate_drift.py. Takes ~25 minutes.

Usage (from the repo root):
    python scripts/experiments/compare_drift_methods.py
"""

# ruff: noqa: E402
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from pdm.common.config import load_yaml
from pdm.data.cmapss import load_test, load_train
from pdm.data.features import build_feature_matrix, feature_columns

cfg = load_yaml("training.yaml")
fcfg = cfg["features"]
cols = load_yaml("drift.yaml")["drift"]["columns"]
mcols = feature_columns(fcfg["sensor_columns"], 20)
params = {**cfg["model"]["params"], "verbose": -1}
N = 500
SHARE = 0.5
WPF = 100


def feats(df):
    f = build_feature_matrix(df, fcfg["sensor_columns"], fcfg["rolling_windows"], 20)
    d = df.sort_values(["unit_number", "time_in_cycles"])
    if "rul" in d:
        f["rul"] = d["rul"].clip(upper=125).values
    f["life_frac"] = f["time_in_cycles"] / f.groupby("unit_number")["time_in_cycles"].transform(
        "max"
    )
    return f


fd1, fd2, fd3 = (
    feats(load_train("data/raw", "FD001")),
    feats(load_train("data/raw", "FD002")),
    feats(load_train("data/raw", "FD003")),
)
t1, _ = load_test("data/raw", "FD001")
t1 = feats(t1)
units = np.random.default_rng(42).permutation(fd1.unit_number.unique())
folds = np.array_split(units, 5)


def nw(r, c):
    return wasserstein_distance(r, c) / max(np.std(r), 1e-3)


def truncate(df, rng, lo=0.1, hi=0.9):
    cut = {u: rng.uniform(lo, hi) for u in df.unit_number.unique()}
    return df[df.life_frac <= df.unit_number.map(cut)]


class Fold:
    def __init__(s, k):
        s.ref = fd1[~fd1.unit_number.isin(folds[k])].copy()
        s.held = fd1[fd1.unit_number.isin(folds[k])].copy()
        oof = np.zeros(len(s.ref))
        for tr, va in GroupKFold(5).split(s.ref, groups=s.ref.unit_number):
            m = LGBMRegressor(**params).fit(s.ref.iloc[tr][mcols], s.ref.iloc[tr].rul)
            oof[va] = m.predict(s.ref.iloc[va][mcols])
        s.ref["pred"] = oof
        s.model = LGBMRegressor(**params).fit(s.ref[mcols], s.ref.rul)
        rng = np.random.default_rng(k)
        s.ref_mid = truncate(s.ref, rng)  # user's option: NASA-style mid-life reference
        s.res = {
            c: LinearRegression().fit(s.ref[[o for o in cols if o != c]], s.ref[c]) for c in cols
        }
        s.ref_res = s.residuals(s.ref)

    def residuals(s, w):
        return pd.DataFrame(
            {c: w[c] - s.res[c].predict(w[[o for o in cols if o != c]]) for c in cols}
        )

    def matched_ref(s, w, rng, bins=np.arange(0, 140, 10)):
        cb = np.clip(np.digitize(w.pred, bins), 0, len(bins))
        rb = np.clip(np.digitize(s.ref.pred, bins), 0, len(bins))
        want = pd.Series(cb).value_counts(normalize=True)
        have = pd.Series(rb).value_counts(normalize=True)
        wts = pd.Series(rb).map(lambda b: want.get(b, 0) / have[b]).values
        idx = rng.choice(len(s.ref), size=5000, replace=True, p=wts / wts.sum())
        return s.ref.iloc[idx]

    def stats(s, w, rng):
        w = w.copy()
        w["pred"] = s.model.predict(w[mcols])
        full = np.array([nw(s.ref[c].values, w[c].values) for c in cols])
        mid = np.array([nw(s.ref_mid[c].values, w[c].values) for c in cols])
        mr = s.matched_ref(w, rng)
        matched = np.array([nw(mr[c].values, w[c].values) for c in cols])
        wr = s.residuals(w)
        resid = np.array([nw(s.ref_res[c].values, wr[c].values) for c in cols])
        return {"full": full, "mid": mid, "matched": matched, "resid": resid}


def fleet(pool, rng):
    return pool.sample(n=min(N, len(pool)), random_state=int(rng.integers(1 << 31)))


def small(pool, rng):
    u = rng.choice(pool.unit_number.unique(), 5, replace=False)
    sub = pool[pool.unit_number.isin(u)]
    return sub.sample(n=min(N, len(sub)), random_state=int(rng.integers(1 << 31)))


def scen(F, rng):
    yield "N_steady_fleet", fleet(F.held, rng)
    yield "N_small_fleet_5_engines", small(F.held, rng)
    yield "A_young_fleet_truncated", fleet(truncate(F.held, rng), rng)
    yield "A_fd001_official_test", fleet(t1, rng)
    yield "A_old_fleet_rul_le_30", fleet(F.held[F.held.rul <= 30], rng)
    yield "D_fd002", fleet(fd2, rng)
    yield "D_fd003", fleet(fd3, rng)
    base = fleet(F.held, rng)
    c = cols[int(rng.integers(len(cols)))]
    for k in (0.5, 1, 2, 3):
        w = base.copy()
        w[c] = w[c] + k * F.ref[c].std()
        yield f"S_one_sensor_offset_{k}std", w
    w = base.copy()
    w[c] = F.ref[c].median()
    yield "S_one_sensor_stuck", w


out = {}
for k in range(5):
    F = Fold(k)
    rng = np.random.default_rng(100 + k)
    for _ in range(WPF):
        for name, w in scen(F, rng):
            out.setdefault((k, name), []).append(F.stats(w, rng))
    print("fold", k, "done", flush=True)


def kth(d):
    return np.sort(d)[::-1][int(np.floor(len(d) * SHARE))]  # value that must be exceeded to fire


res = {}
cal = [0, 1, 2]
ev = [3, 4]
for method in ("full", "mid", "matched", "resid"):
    for rule, stat in (("share>0.5", kth), ("any_column", np.max)):
        calib = [
            stat(s[method])
            for k in cal
            for n in ("N_steady_fleet", "N_small_fleet_5_engines")
            for s in out[(k, n)]
        ]
        t = float(np.quantile(calib, 0.99))
        rates = {}
        for name in sorted({n for (_, n) in out}):
            v = [stat(s[method]) > t for k in ev for s in out[(k, name)]]
            rates[name] = round(float(np.mean(v)), 3)
        res[f"{method}|{rule}"] = {"threshold": round(t, 3), "fire_rate_heldout": rates}
json.dump(res, open("reports/drift_method_comparison.json", "w"), indent=1)
names = sorted({n for (_, n) in out})
print(f"{'scenario':32s}" + "".join(f"{m[:18]:>19s}" for m in res))
for n in names:
    print(f"{n:32s}" + "".join(f"{res[m]['fire_rate_heldout'][n]:>19.2f}" for m in res))
print("thresholds", {m: res[m]["threshold"] for m in res})
