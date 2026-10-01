"""The C-MAPSS model bundle: a point RUL model, a calibrated prediction interval, and the
maintenance threshold chosen for it, served together as one MLflow pyfunc.

Why an interval: a maintenance planner acts on "how sure are we", not just a number.
The interval is conformalized quantile regression (CQR): two quantile models give a raw
band, then a correction measured on engines the models never saw widens (or narrows) it
so that about `coverage` of true RULs fall inside. Calibration is done per row, but rows
from one engine are correlated, so the guarantee is approximate; the achieved coverage
on held-out engines is logged and checked by the champion/challenger gate.

Why the threshold travels with the model: the maintenance rule ("maintain when the lower
bound of predicted RUL drops to H") is tuned for each model's own errors
(pdm.evaluation.decision.choose_threshold). A new model with an old model's H would be
judged on a rule it was never tuned for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow.pyfunc
import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ["rul", "rul_lower", "rul_upper", "maintenance_recommended"]


def _make_regressor(algorithm: str, params: dict, quantile: float | None = None):
    if quantile is not None:
        # Not LightGBM's quantile objective: with the RUL cap, ~half the training
        # targets equal 125, LightGBM starts the 95% model at 125 and never moves (it
        # predicted exactly 125 for every row on FD001). scikit-learn's quantile loss
        # learns a proper bound (upper bound ~30 near failure, ~110 far from it).
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth"),
            random_state=params.get("random_state", 42),
        )
    if algorithm == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params)
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


class SeedEnsemble:
    """Average of the same regressor trained with different random seeds. Seed-to-seed
    variance moved holdout cost by ~8% (decisions.md D-14), more than the gate's
    tolerance; averaging seeds is the standard way to shrink it."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def _fit_point(algorithm: str, params: dict, X, y, ensemble_seeds: list[int] | None):
    if not ensemble_seeds:
        return _make_regressor(algorithm, params).fit(X, y)
    return SeedEnsemble(
        [
            _make_regressor(algorithm, {**params, "random_state": seed}).fit(X, y)
            for seed in ensemble_seeds
        ]
    )


@dataclass
class RULIntervalModel:
    feature_columns: list[str]
    point: object
    lower: object | None = None
    upper: object | None = None
    conformal_correction: float = 0.0
    coverage_target: float | None = None
    maintenance_threshold: float | None = None
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        algorithm: str,
        params: dict,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_cal: pd.DataFrame | None = None,
        y_cal: pd.Series | None = None,
        coverage: float | None = None,
        ensemble_seeds: list[int] | None = None,
    ) -> RULIntervalModel:
        """Fits the point model on (X_fit, y_fit) - one model, or the average of one per
        seed in `ensemble_seeds`. If `coverage` is given, also fits the quantile models
        on the same rows and calibrates the interval on (X_cal, y_cal), which must come
        from different engines."""
        point = _fit_point(algorithm, params, X_fit, y_fit, ensemble_seeds)
        model = cls(feature_columns=list(X_fit.columns), point=point)
        if coverage is None:
            return model
        alpha = 1.0 - coverage
        model.lower = _make_regressor(algorithm, params, alpha / 2).fit(X_fit, y_fit)
        model.upper = _make_regressor(algorithm, params, 1 - alpha / 2).fit(X_fit, y_fit)
        lo, hi = model.lower.predict(X_cal), model.upper.predict(X_cal)
        y = np.asarray(y_cal, dtype=float)
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        model.conformal_correction = float(np.quantile(scores, level, method="higher"))
        model.coverage_target = coverage
        return model

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_columns]
        out = pd.DataFrame({"rul": self.point.predict(X)}, index=X.index)
        if self.lower is not None:
            q = self.conformal_correction
            lo = np.minimum(self.lower.predict(X), out["rul"]) - q
            hi = np.maximum(self.upper.predict(X), out["rul"]) + q
            out["rul_lower"] = np.maximum(lo, 0.0)
            out["rul_upper"] = hi
        else:
            out["rul_lower"] = np.nan
            out["rul_upper"] = np.nan
        signal = out["rul_lower"].where(out["rul_lower"].notna(), out["rul"])
        if self.maintenance_threshold is None:
            out["maintenance_recommended"] = np.nan
        else:
            out["maintenance_recommended"] = (signal <= self.maintenance_threshold).astype(float)
        return out[OUTPUT_COLUMNS]

    def decision_signal(self, X: pd.DataFrame) -> np.ndarray:
        """What the maintenance rule compares with its threshold: the interval's lower
        bound when intervals exist (act on the pessimistic case), else the point RUL."""
        frame = self.predict_frame(X)
        signal = frame["rul_lower"].where(frame["rul_lower"].notna(), frame["rul"])
        return signal.to_numpy()


class RULPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper. predict() returns a DataFrame with OUTPUT_COLUMNS."""

    def __init__(self, bundle: RULIntervalModel):
        self.bundle = bundle

    def predict(self, context, model_input, params=None):
        return self.bundle.predict_frame(pd.DataFrame(model_input))


def point_predictions(output) -> np.ndarray:
    """Point RUL from any model output: a RULPyfunc DataFrame or a plain array."""
    if isinstance(output, pd.DataFrame):
        return output["rul"].to_numpy(dtype=float)
    return np.asarray(output, dtype=float).reshape(-1)
