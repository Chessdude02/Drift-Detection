"""Prometheus metrics for the serving app: request counts/latency, prediction distribution,
and rolling per-feature stats (mean/stddev) that double as cheap drift signals in Grafana.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter("pdm_predictions_total", "Total prediction requests", ["status", "shadow"])
REQUEST_LATENCY = Histogram(
    "pdm_request_latency_seconds", "Prediction request latency in seconds", ["shadow"]
)
PREDICTION_VALUE = Histogram(
    "pdm_prediction_value",
    "Distribution of predicted RUL values",
    buckets=(0, 10, 25, 50, 75, 100, 125, 150, float("inf")),
)
FEATURE_MEAN = Gauge("pdm_feature_mean", "Rolling mean of an input feature", ["feature"])
FEATURE_STDDEV = Gauge("pdm_feature_stddev", "Rolling stddev of an input feature", ["feature"])
MODEL_INFO = Gauge(
    "pdm_model_version_info", "Currently loaded model version", ["model_name", "version"]
)


class RollingFeatureStats:
    """Maintains a bounded per-feature window and pushes mean/stddev into Gauges on update."""

    def __init__(self, window: int = 200):
        self.window = window
        self._lock = threading.Lock()
        self._data: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, features: dict[str, float]) -> None:
        with self._lock:
            for name, value in features.items():
                buf = self._data[name]
                buf.append(value)
                FEATURE_MEAN.labels(feature=name).set(statistics.fmean(buf))
                FEATURE_STDDEV.labels(feature=name).set(
                    statistics.pstdev(buf) if len(buf) > 1 else 0.0
                )


feature_stats = RollingFeatureStats()
