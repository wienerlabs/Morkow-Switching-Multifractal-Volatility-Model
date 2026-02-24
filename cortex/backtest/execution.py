"""Realistic Solana DEX trade execution simulator.

Models Jupiter/Raydium execution costs including slippage (square-root
market impact model), DEX fees, and Solana transaction costs.
"""

import math
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of a simulated trade execution."""

    execution_price: float
    slippage_bps: float
    fee_usd: float
    price_impact_pct: float
    net_cost_usd: float
    filled: bool


class ExecutionSimulator:
    """Models realistic Solana DEX trade execution costs.

    Uses the standard square-root market impact model for TCA:
        slippage = impact_coefficient * sqrt(size / volume)
    """

    def __init__(
        self,
        dex_fee_bps: float = 25.0,
        solana_base_fee: float = 0.000005,
        solana_priority_fee: float = 0.00002,
        sol_price_usd: float = 150.0,
        impact_coefficient: float = 0.1,
        max_slippage_bps: float = 500.0,
        min_volume_ratio: float = 0.001,
    ):
        self.dex_fee_bps = dex_fee_bps
        self.solana_base_fee = solana_base_fee
        self.solana_priority_fee = solana_priority_fee
        self.sol_price_usd = sol_price_usd
        self.impact_coefficient = impact_coefficient
        self.max_slippage_bps = max_slippage_bps
        self.min_volume_ratio = min_volume_ratio

    def simulate_execution(
        self,
        price: float,
        size_usd: float,
        volume_24h: float,
        direction: str = "long",
    ) -> ExecutionResult:
        """Simulate trade execution with realistic costs.

        Args:
            price: Current market price.
            size_usd: Trade size in USD.
            volume_24h: 24-hour trading volume in USD.
            direction: "long" (buy) or "short" (sell).
        """
        if volume_24h <= 0 or size_usd / volume_24h > self.min_volume_ratio:
            return ExecutionResult(
                execution_price=price,
                slippage_bps=0,
                fee_usd=0,
                price_impact_pct=0,
                net_cost_usd=0,
                filled=False,
            )

        volume_ratio = size_usd / volume_24h
        price_impact_pct = self.impact_coefficient * math.sqrt(volume_ratio) * 100
        slippage_bps = min(price_impact_pct * 100, self.max_slippage_bps)

        if direction == "long":
            execution_price = price * (1 + slippage_bps / 10_000)
        else:
            execution_price = price * (1 - slippage_bps / 10_000)

        dex_fee = size_usd * (self.dex_fee_bps / 10_000)
        gas_fee = (self.solana_base_fee + self.solana_priority_fee) * self.sol_price_usd
        total_fee = dex_fee + gas_fee

        slippage_cost = abs(execution_price - price) / price * size_usd
        net_cost = total_fee + slippage_cost

        return ExecutionResult(
            execution_price=execution_price,
            slippage_bps=slippage_bps,
            fee_usd=total_fee,
            price_impact_pct=price_impact_pct,
            net_cost_usd=net_cost,
            filled=True,
        )

    def get_transaction_cost(self, size_usd: float) -> float:
        """Quick estimate of total transaction cost (fees only, no slippage)."""
        dex_fee = size_usd * (self.dex_fee_bps / 10_000)
        gas_fee = (self.solana_base_fee + self.solana_priority_fee) * self.sol_price_usd
        return dex_fee + gas_fee
