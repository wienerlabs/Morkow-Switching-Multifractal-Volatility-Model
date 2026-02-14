"""Custom Prometheus metrics for the Cortex Risk Engine."""

from prometheus_client import Counter, Gauge, Histogram

model_calibration_duration_seconds = Histogram(
    "model_calibration_duration_seconds",
    "Time spent calibrating risk models",
    ["model_type"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

var_computation_duration_seconds = Histogram(
    "var_computation_duration_seconds",
    "Time spent computing Value-at-Risk",
    ["method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

guardian_veto_total = Counter(
    "guardian_veto_total",
    "Total Guardian veto decisions",
    ["decision"],
)

active_websocket_connections = Gauge(
    "active_websocket_connections",
    "Number of active WebSocket connections",
)

