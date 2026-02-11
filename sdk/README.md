# cortex-risk-sdk

TypeScript SDK for the [CortexAgent Risk Engine](https://www.cortex-agent.xyz) — 45 typed endpoints covering MSM regime detection, EVT, SVJ, Hawkes, rough volatility, copula VaR, and Guardian risk veto.

[![npm](https://img.shields.io/npm/v/cortex-risk-sdk)](https://www.npmjs.com/package/cortex-risk-sdk)

## Install

```bash
npm install cortex-risk-sdk
```

Requires **Node 18+** (native `fetch`).

## Quick Start

```typescript
import { RiskEngineClient } from "cortex-risk-sdk";

const risk = new RiskEngineClient({
  baseUrl: "http://localhost:8000",
  timeout: 15_000,
  retries: 3,
  validateResponses: true,
});

// Calibrate MSM model
await risk.calibrate({ token: "SOL-USD", num_states: 5 });

// Check regime
const regime = await risk.regime("SOL-USD");
console.log(regime.regime_state, regime.regime_name);

// Guardian risk veto
const assessment = await risk.guardianAssess({
  token: "SOL-USD",
  trade_size_usd: 50_000,
  direction: "long",
});

if (assessment.approved) {
  console.log(`Approved — size $${assessment.recommended_size}`);
} else {
  console.log(`Vetoed — ${assessment.veto_reasons.join(", ")}`);
}
```

## Modules (45 endpoints)

| Module | Methods | Description |
|--------|---------|-------------|
| **Core MSM** | `calibrate`, `regime`, `var`, `volatilityForecast`, `backtestSummary`, `tailProbs` | Regime detection, VaR |
| **Regime Analytics** | `regimeDurations`, `regimeHistory`, `regimeStatistics`, `transitionAlert` | Temporal regime analysis |
| **Model Comparison** | `compare`, `comparisonReport` | 9-model benchmark |
| **Portfolio VaR** | `portfolioCalibrate`, `portfolioVar`, `marginalVar`, `stressVar` | Multi-asset risk |
| **Copula VaR** | `copulaVar`, `copulaCompare`, `copulaDiagnostics`, `regimeDependentCopulaVar` | Dependence modeling |
| **EVT** | `evtCalibrate`, `evtVar`, `evtDiagnostics` | Tail risk (GPD) |
| **Hawkes** | `hawkesCalibrate`, `hawkesIntensity`, `hawkesClusters`, `hawkesVar`, `hawkesSimulate` | Crash contagion |
| **Multifractal** | `hurst`, `spectrum`, `regimeHurst`, `fractalDiagnostics` | Hurst exponent |
| **Rough Vol** | `roughCalibrate`, `roughForecast`, `roughDiagnostics`, `roughCompareMsm` | Rough Bergomi |
| **SVJ** | `svjCalibrate`, `svjVar`, `svjJumpRisk`, `svjDiagnostics` | Jump risk |
| **News** | `newsFeed`, `newsSentiment`, `newsSignal` | Sentiment signals |
| **Guardian** | `guardianAssess` | Unified risk veto |

## WebSocket Streaming

```typescript
import { RegimeStreamClient } from "cortex-risk-sdk";

const stream = new RegimeStreamClient({
  baseUrl: "http://localhost:8000",
  token: "SOL-USD",
  onRegime: (msg) => console.log(`Regime ${msg.regime_state}`),
  onError: (err) => console.error(err),
});
stream.connect();
```

## Resilience

Built-in via [cockatiel](https://github.com/connor4312/cockatiel):

- **Retry** — exponential backoff (configurable `retries`)
- **Circuit breaker** — opens after consecutive failures, half-opens after cooldown
- **Timeout** — per-request timeout with `AbortSignal`

## Validation

Optional [zod](https://github.com/colinhacks/zod) runtime validation for critical responses (Guardian, VaR, Regime). Enable with `validateResponses: true`.

## Configuration

```typescript
const risk = new RiskEngineClient({
  baseUrl: "http://localhost:8000",  // Risk Engine URL
  timeout: 10_000,                   // Request timeout (ms)
  retries: 3,                        // Max retry attempts
  cbThreshold: 5,                    // Circuit breaker threshold
  cbResetMs: 30_000,                 // Circuit breaker reset (ms)
  validateResponses: false,          // Zod validation
});
```

## License

MIT — [Cortex AI](https://www.cortex-agent.xyz)

