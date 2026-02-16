/**
 * A-LAMS Model Calibration Service
 *
 * Automatically calibrates (fits) the A-LAMS-VaR model on startup and
 * periodically refits every ALAMS_REFIT_INTERVAL_MS (default 24h).
 *
 * Flow: Birdeye OHLCV → log returns → POST /risk/var/fit
 */

import { BirdeyeProvider, TOKENS, type OHLCVData } from '../providers/birdeye.js';
import { ALAMSVaRClient } from './risk/alamsVarClient.js';
import { logger } from './logger.js';

const CANDLE_COUNT = 500;
const SOL_MINT = TOKENS.SOL;

// ── helpers ──────────────────────────────────────────────────────────

export function computeLogReturns(candles: { close: number }[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const prev = candles[i - 1].close;
    const curr = candles[i].close;
    if (prev > 0 && curr > 0) {
      returns.push(Math.log(curr / prev));
    }
  }
  return returns;
}

// ── calibration runner ───────────────────────────────────────────────

let refitTimer: ReturnType<typeof setInterval> | null = null;
let isCalibrating = false;

async function runCalibration(birdeye: BirdeyeProvider, alams: ALAMSVaRClient): Promise<boolean> {
  if (isCalibrating) {
    logger.warn('[Calibration] Skipping — calibration already in progress');
    return false;
  }
  isCalibrating = true;

  try {
    const now = Math.floor(Date.now() / 1000);
    const from = now - CANDLE_COUNT * 3600; // 500 hours back
    const candles = await birdeye.getOHLCV(SOL_MINT, '1h', from, now);

    if (candles.length < 50) {
      logger.warn(`[Calibration] Only ${candles.length} candles returned — skipping fit`);
      return false;
    }

    const returns = computeLogReturns(candles);
    if (returns.length < 50) {
      logger.warn(`[Calibration] Only ${returns.length} returns computed — skipping fit`);
      return false;
    }

    logger.info(`[Calibration] Fitting A-LAMS model with ${returns.length} log returns…`);
    const result = await alams.fitModel(returns, { token: 'SOL' });

    logger.info(
      `[Calibration] Model fitted — n_obs=${result.n_obs}, regimes=${result.n_regimes}, ` +
      `LL=${result.log_likelihood.toFixed(2)}, AIC=${result.aic.toFixed(2)}`
    );
    return true;
  } catch (error) {
    logger.error('[Calibration] Failed to calibrate A-LAMS model', {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  } finally {
    isCalibrating = false;
  }
}

// ── public API ───────────────────────────────────────────────────────

export function startCalibrationSchedule(
  birdeye: BirdeyeProvider,
  alams: ALAMSVaRClient,
  intervalMs: number,
): void {
  if (refitTimer) {
    clearInterval(refitTimer);
  }

  refitTimer = setInterval(() => {
    runCalibration(birdeye, alams).catch((err) => {
      logger.error('[Calibration] Periodic refit error', { error: String(err) });
    });
  }, intervalMs);

  logger.info(`[Calibration] Periodic refit scheduled every ${(intervalMs / 3600000).toFixed(1)}h`);
}

export function stopCalibrationSchedule(): void {
  if (refitTimer) {
    clearInterval(refitTimer);
    refitTimer = null;
    logger.info('[Calibration] Periodic refit stopped');
  }
}

/**
 * Run initial calibration (fire-and-forget) and start periodic refit.
 * Call this from start.ts after services are initialized.
 */
export async function initCalibration(): Promise<void> {
  const enabled = (process.env.ALAMS_CALIBRATION_ENABLED ?? 'true') !== 'false';
  if (!enabled) {
    logger.info('[Calibration] Disabled via ALAMS_CALIBRATION_ENABLED=false');
    return;
  }

  const apiKey = process.env.BIRDEYE_API_KEY;
  if (!apiKey) {
    logger.warn('[Calibration] BIRDEYE_API_KEY not set — skipping calibration');
    return;
  }

  const birdeye = new BirdeyeProvider(apiKey);
  const alams = new ALAMSVaRClient();
  const intervalMs = parseInt(process.env.ALAMS_REFIT_INTERVAL_MS || '86400000', 10);

  // Initial calibration (non-blocking)
  const success = await runCalibration(birdeye, alams);
  if (success) {
    logger.info('[Calibration] Initial calibration complete');
  } else {
    logger.warn('[Calibration] Initial calibration failed — agent continues without fitted model');
  }

  // Schedule periodic refit
  startCalibrationSchedule(birdeye, alams, intervalMs);
}
