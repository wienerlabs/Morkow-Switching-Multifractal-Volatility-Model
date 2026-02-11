/**
 * Example: Cortex trading agent using cortex-risk-sdk
 *
 * Flow: calibrate → check regime → Guardian assess → execute or abort
 */
import { RiskEngineClient, RegimeStreamClient } from "../src";

const risk = new RiskEngineClient({
  baseUrl: "http://localhost:8000",
  timeout: 15_000,
  retries: 2,
});

async function tradingLoop(token: string): Promise<void> {
  // 1. Calibrate all models
  const cal = await risk.calibrate({ token, num_states: 5 });
  console.log(`[MSM] ${token} calibrated — ${cal.num_states} states`);

  await risk.evtCalibrate({ token });
  console.log(`[EVT] ${token} calibrated`);

  await risk.hawkesCalibrate({ token });
  console.log(`[Hawkes] ${token} calibrated`);

  await risk.svjCalibrate({ token });
  console.log(`[SVJ] ${token} calibrated`);

  // 2. Check current regime
  const regime = await risk.regime(token);
  console.log(`[Regime] State ${regime.regime_state} (${regime.regime_name}) — prob ${(regime.probability * 100).toFixed(1)}%`);

  // 3. Guardian risk assessment
  const assessment = await risk.guardianAssess({
    token,
    trade_size_usd: 50_000,
    direction: "long",
  });

  if (!assessment.approved) {
    console.log(`[Guardian] VETOED — score ${assessment.risk_score.toFixed(1)}`);
    console.log(`  Reasons: ${assessment.veto_reasons.join(", ")}`);
    return;
  }

  console.log(`[Guardian] APPROVED — score ${assessment.risk_score.toFixed(1)}`);
  console.log(`  Recommended size: $${assessment.recommended_size.toLocaleString()}`);

  // 4. Get VaR for position sizing
  const var99 = await risk.var(token, 99);
  console.log(`[VaR] 99% = ${var99.var_value.toFixed(2)}%`);

  // 5. Execute trade (placeholder — your DEX integration here)
  console.log(`[Trade] Executing $${assessment.recommended_size.toLocaleString()} ${token} long`);
}

// Real-time regime monitoring via WebSocket
function startRegimeStream(token: string): void {
  const stream = new RegimeStreamClient({
    baseUrl: "http://localhost:8000",
    token,
    onRegime: (msg) => {
      console.log(`[WS] Regime ${msg.regime_state} (${msg.regime_name}) — VaR95 ${msg.var_95.toFixed(2)}%`);
    },
    onError: (err) => console.error("[WS] Error:", err.message),
    onClose: () => console.log("[WS] Disconnected"),
  });
  stream.connect();
}

// Run
tradingLoop("SOL-USD").catch(console.error);
startRegimeStream("SOL-USD");

