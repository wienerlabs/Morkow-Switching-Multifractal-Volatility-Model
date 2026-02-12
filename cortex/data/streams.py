"""Helius Enhanced WebSocket transaction stream monitor.

Subscribes to Solana transactions via Helius ``transactionSubscribe`` and
classifies events (large swaps, pool creation, liquidations) into risk
signals consumable by the Guardian layer.
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

from cortex.config import (
    HELIUS_ACCOUNTS,
    HELIUS_API_KEY,
    HELIUS_EVENT_BUFFER,
    HELIUS_PING_INTERVAL,
    HELIUS_WS_URL,
    LARGE_SWAP_THRESHOLD_USD,
)

logger = logging.getLogger(__name__)

_event_buffer: deque[dict[str, Any]] = deque(maxlen=HELIUS_EVENT_BUFFER)
_stream_status: dict[str, Any] = {
    "connected": False,
    "last_event_time": None,
    "events_received": 0,
    "started_at": None,
}


def classify_event(tx_data: dict) -> dict[str, Any]:
    """Classify a Helius transaction notification into a risk event."""
    meta = tx_data.get("transaction", {}).get("meta", {})
    logs = meta.get("logMessages", [])
    signature = tx_data.get("signature", "")
    slot = tx_data.get("slot", 0)

    log_text = " ".join(logs).lower()

    event_type = "unknown"
    severity = "info"
    details: dict[str, Any] = {}

    if "swap" in log_text or "raydium" in log_text or "jupiter" in log_text:
        pre = sum(meta.get("preBalances", []))
        post = sum(meta.get("postBalances", []))
        lamport_delta = abs(post - pre)
        sol_delta = lamport_delta / 1e9
        if sol_delta * 150 > LARGE_SWAP_THRESHOLD_USD:
            event_type = "large_swap"
            severity = "warning"
            details["estimated_usd"] = sol_delta * 150
        else:
            event_type = "swap"

    elif "initializemint" in log_text or "createpool" in log_text:
        event_type = "pool_creation"
        severity = "info"

    elif "liquidat" in log_text:
        event_type = "liquidation"
        severity = "critical"

    elif meta.get("err") is not None:
        event_type = "failed_tx"
        severity = "info"

    return {
        "event_type": event_type,
        "severity": severity,
        "signature": signature,
        "slot": slot,
        "timestamp": time.time(),
        "details": details,
    }


async def _run_helius_stream() -> None:
    """Connect to Helius WebSocket and stream transaction events."""
    if not HELIUS_API_KEY:
        logger.warning("HELIUS_API_KEY not set — stream disabled")
        return

    try:
        import websockets
    except ImportError:
        logger.error("websockets package required for Helius streaming")
        return

    ws_url = f"{HELIUS_WS_URL}/?api-key={HELIUS_API_KEY}"
    accounts = HELIUS_ACCOUNTS or []

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                _stream_status["connected"] = True
                _stream_status["started_at"] = time.time()
                logger.info("Helius WebSocket connected")

                subscribe_msg = json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "transactionSubscribe",
                    "params": [
                        {"accountInclude": accounts, "failed": False, "vote": False},
                        {
                            "commitment": "confirmed",
                            "encoding": "jsonParsed",
                            "transactionDetails": "full",
                            "maxSupportedTransactionVersion": 0,
                        },
                    ],
                })
                await ws.send(subscribe_msg)

                async def ping_loop():
                    while True:
                        await asyncio.sleep(HELIUS_PING_INTERVAL)
                        await ws.ping()

                ping_task = asyncio.create_task(ping_loop())
                try:
                    async for raw in ws:
                        payload = json.loads(raw)
                        result = payload.get("params", {}).get("result")
                        if not result:
                            continue
                        event = classify_event(result)
                        _event_buffer.append(event)
                        _stream_status["events_received"] += 1
                        _stream_status["last_event_time"] = time.time()
                finally:
                    ping_task.cancel()

        except Exception:
            _stream_status["connected"] = False
            logger.exception("Helius WebSocket error, reconnecting in 5s")
            await asyncio.sleep(5)



def get_recent_events(limit: int = 50, severity: str | None = None) -> list[dict[str, Any]]:
    """Return recent classified events, optionally filtered by severity."""
    events = list(_event_buffer)
    if severity:
        events = [e for e in events if e["severity"] == severity]
    return events[-limit:]


def get_stream_status() -> dict[str, Any]:
    """Return current stream connection status."""
    return dict(_stream_status)


def get_risk_signals() -> list[dict[str, Any]]:
    """Extract Guardian-consumable risk signals from recent events."""
    signals: list[dict[str, Any]] = []
    cutoff = time.time() - 300  # last 5 minutes

    for event in _event_buffer:
        if event["timestamp"] < cutoff:
            continue
        if event["severity"] in ("warning", "critical"):
            signals.append({
                "source": "helius_stream",
                "event_type": event["event_type"],
                "severity": event["severity"],
                "signature": event["signature"],
                "timestamp": event["timestamp"],
                "details": event.get("details", {}),
            })

    return signals
