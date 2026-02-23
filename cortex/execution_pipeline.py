"""Three-Stage Execution Pipeline — stage → validate → execute.

Prevents partial/accidental trade execution by requiring explicit progression
through three phases. Each stage has independent validation and rollback.
Wires into TradeLedger for audit trail.
"""
from __future__ import annotations

__all__ = ["ExecutionPipeline", "PipelineResult", "PipelineStage"]

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cortex.config import (
    EXECUTION_PIPELINE_ENABLED,
    EXECUTION_PIPELINE_VALIDATE_TIMEOUT,
)

logger = logging.getLogger(__name__)


class PipelineStage(str, enum.Enum):
    STAGED = "staged"
    VALIDATED = "validated"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class PipelineResult:
    stage: PipelineStage
    entry_hash: str | None
    token: str
    direction: str
    trade_size_usd: float
    guardian_approved: bool
    guardian_score: float
    recommended_size: float
    validation_errors: list[str]
    execution_result: dict[str, Any] | None
    execution_error: str | None
    elapsed_ms: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "entry_hash": self.entry_hash,
            "token": self.token,
            "direction": self.direction,
            "trade_size_usd": self.trade_size_usd,
            "guardian_approved": self.guardian_approved,
            "guardian_score": self.guardian_score,
            "recommended_size": self.recommended_size,
            "validation_errors": self.validation_errors,
            "execution_result": self.execution_result,
            "execution_error": self.execution_error,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "timestamp": self.timestamp,
        }


class ExecutionPipeline:
    """Three-stage execution: stage → validate → execute.

    Stage 1 (stage): Record intent in trade ledger, gather Guardian assessment.
    Stage 2 (validate): Check Guardian approval, portfolio limits, circuit breakers.
    Stage 3 (execute): Call the actual execution function, record result.

    If any stage fails, the pipeline stops and records the failure.
    """

    def __init__(
        self,
        guardian_fn: Callable[..., dict] | None = None,
        execute_fn: Callable[..., dict] | None = None,
    ) -> None:
        self._guardian_fn = guardian_fn
        self._execute_fn = execute_fn
        self._history: list[PipelineResult] = []

    def run(
        self,
        token: str,
        direction: str,
        trade_size_usd: float,
        strategy: str = "spot",
        intent: dict[str, Any] | None = None,
        guardian_kwargs: dict[str, Any] | None = None,
        execute_kwargs: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Run the full three-stage pipeline."""
        t0 = time.time()
        validation_errors: list[str] = []
        entry_hash: str | None = None

        # ── Stage 1: STAGE ──
        try:
            from cortex.trade_ledger import get_trade_ledger
            ledger = get_trade_ledger()
            entry_hash = ledger.stage(
                token=token,
                direction=direction,
                trade_size_usd=trade_size_usd,
                strategy=strategy,
                intent=intent,
            )
        except Exception as e:
            logger.warning("Pipeline stage failed: %s", e)

        # ── Stage 2: VALIDATE ──
        guardian_result: dict[str, Any] = {}
        guardian_approved = False
        guardian_score = 0.0
        recommended_size = trade_size_usd

        if self._guardian_fn:
            try:
                gkw = guardian_kwargs or {}
                guardian_result = self._guardian_fn(
                    token=token,
                    trade_size_usd=trade_size_usd,
                    direction=direction,
                    **gkw,
                )
                guardian_approved = guardian_result.get("approved", False)
                guardian_score = guardian_result.get("risk_score", 0.0)
                recommended_size = guardian_result.get("recommended_size", trade_size_usd)

                if not guardian_approved:
                    veto = guardian_result.get("veto_reasons", [])
                    validation_errors.append(f"guardian_rejected: {veto}")
            except Exception as e:
                validation_errors.append(f"guardian_error: {e}")
                logger.warning("Pipeline validation failed: %s", e)
        else:
            validation_errors.append("no_guardian_fn: bypassing validation")
            guardian_approved = True

        # Commit to ledger
        if entry_hash:
            try:
                from cortex.trade_ledger import get_trade_ledger
                ledger = get_trade_ledger()
                msg = f"score={guardian_score:.1f} approved={guardian_approved} size={recommended_size}"
                ledger.commit(entry_hash, msg)
            except Exception as e:
                logger.warning("Pipeline ledger commit failed: %s", e)

        if validation_errors and not guardian_approved:
            # Rollback in ledger
            if entry_hash:
                try:
                    from cortex.trade_ledger import get_trade_ledger
                    get_trade_ledger().rollback(entry_hash, "; ".join(validation_errors))
                except Exception:
                    pass

            result = PipelineResult(
                stage=PipelineStage.REJECTED,
                entry_hash=entry_hash,
                token=token,
                direction=direction,
                trade_size_usd=trade_size_usd,
                guardian_approved=False,
                guardian_score=guardian_score,
                recommended_size=recommended_size,
                validation_errors=validation_errors,
                execution_result=None,
                execution_error=None,
                elapsed_ms=(time.time() - t0) * 1000,
                timestamp=time.time(),
            )
            self._record(result)
            return result

        # ── Stage 3: EXECUTE ──
        if dry_run:
            if entry_hash:
                try:
                    from cortex.trade_ledger import get_trade_ledger
                    get_trade_ledger().push(entry_hash, result={"dry_run": True})
                except Exception:
                    pass

            result = PipelineResult(
                stage=PipelineStage.VALIDATED,
                entry_hash=entry_hash,
                token=token,
                direction=direction,
                trade_size_usd=recommended_size,
                guardian_approved=guardian_approved,
                guardian_score=guardian_score,
                recommended_size=recommended_size,
                validation_errors=[],
                execution_result={"dry_run": True},
                execution_error=None,
                elapsed_ms=(time.time() - t0) * 1000,
                timestamp=time.time(),
            )
            self._record(result)
            return result

        execution_result: dict[str, Any] | None = None
        execution_error: str | None = None

        if self._execute_fn:
            try:
                ekw = execute_kwargs or {}
                execution_result = self._execute_fn(
                    token=token,
                    direction=direction,
                    size_usd=recommended_size,
                    strategy=strategy,
                    **ekw,
                )
            except Exception as e:
                execution_error = str(e)
                logger.warning("Pipeline execution failed: %s", e)

        # Push to ledger
        if entry_hash:
            try:
                from cortex.trade_ledger import get_trade_ledger
                get_trade_ledger().push(entry_hash, result=execution_result, error=execution_error)
            except Exception:
                pass

        stage = PipelineStage.EXECUTED if not execution_error else PipelineStage.FAILED

        result = PipelineResult(
            stage=stage,
            entry_hash=entry_hash,
            token=token,
            direction=direction,
            trade_size_usd=recommended_size,
            guardian_approved=guardian_approved,
            guardian_score=guardian_score,
            recommended_size=recommended_size,
            validation_errors=validation_errors,
            execution_result=execution_result,
            execution_error=execution_error,
            elapsed_ms=(time.time() - t0) * 1000,
            timestamp=time.time(),
        )
        self._record(result)
        return result

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._history[-limit:]]

    def _record(self, result: PipelineResult) -> None:
        self._history.append(result)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        logger.info(
            "pipeline %s token=%s dir=%s size=%.2f score=%.1f hash=%s",
            result.stage.value, result.token, result.direction,
            result.trade_size_usd, result.guardian_score,
            result.entry_hash or "none",
        )
