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

# --- DX Research metrics ---

narrator_request_duration_seconds = Histogram(
    "narrator_request_duration_seconds",
    "Time spent generating narrator responses",
    ["function"],
    buckets=(1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

narrator_request_total = Counter(
    "narrator_request_total",
    "Total narrator requests",
    ["function", "status"],
)

stigmergy_pheromone_deposits = Counter(
    "stigmergy_pheromone_deposits",
    "Total stigmergy pheromone deposits",
    ["signal_type"],
)

ising_cascade_alerts = Counter(
    "ising_cascade_alerts",
    "Total Ising cascade alerts detected",
    ["severity"],
)

human_override_actions = Counter(
    "human_override_actions",
    "Total human override actions",
    ["action"],
)

dx_module_status = Gauge(
    "dx_module_status",
    "DX module enabled status (1=on, 0=off)",
    ["module"],
)

