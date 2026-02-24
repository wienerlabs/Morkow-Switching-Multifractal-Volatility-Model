"""Guardian backtesting engine for the Cortex risk system."""

from cortex.backtest.walk_forward import run_walk_forward, WalkForwardConfig, WalkForwardResult
from cortex.backtest.report import generate_report
from cortex.backtest.export import export_historical_data
from cortex.backtest.execution import ExecutionSimulator, ExecutionResult
from cortex.backtest.data_feed import HistoricalDataFeed
from cortex.backtest.guardian_backtester import GuardianBacktester, BacktestConfig, BacktestResult

# Optional: analytics (may not be installed yet during parallel build)
try:
    from cortex.backtest.analytics import PerformanceAnalyzer
except ImportError:
    PerformanceAnalyzer = None

__all__ = [
    "run_walk_forward",
    "WalkForwardConfig",
    "WalkForwardResult",
    "generate_report",
    "export_historical_data",
    "ExecutionSimulator",
    "ExecutionResult",
    "HistoricalDataFeed",
    "GuardianBacktester",
    "BacktestConfig",
    "BacktestResult",
    "PerformanceAnalyzer",
]
