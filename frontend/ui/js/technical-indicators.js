/**
 * technical-indicators.js — Pure-JS technical indicator library
 *
 * All functions accept arrays of candle objects: { time, open, high, low, close, volume }
 * All return arrays of { time, value } (or multi-value objects) for Lightweight Charts overlay.
 *
 * Investing.com parity: RSI, MACD, Bollinger Bands, SMA, EMA, WMA, Stochastic,
 * ADX, CCI, Williams %R, ATR, ROC, OBV, VWAP, Bull/Bear Power, Ultimate Oscillator,
 * Ichimoku Cloud, Pivot Points, + Technical Summary signal generator.
 */

(function (global) {
    'use strict';

    // =========================================================================
    // Helpers
    // =========================================================================

    function closes(candles) {
        return candles.map(function (c) { return c.close; });
    }

    function highs(candles) {
        return candles.map(function (c) { return c.high; });
    }

    function lows(candles) {
        return candles.map(function (c) { return c.low; });
    }

    function volumes(candles) {
        return candles.map(function (c) { return c.volume || 0; });
    }

    function times(candles) {
        return candles.map(function (c) { return c.time; });
    }

    // =========================================================================
    // Moving Averages
    // =========================================================================

    function sma(candles, period) {
        var c = closes(candles);
        var t = times(candles);
        var result = [];
        for (var i = period - 1; i < c.length; i++) {
            var sum = 0;
            for (var j = i - period + 1; j <= i; j++) sum += c[j];
            result.push({ time: t[i], value: sum / period });
        }
        return result;
    }

    function ema(candles, period) {
        var c = closes(candles);
        var t = times(candles);
        if (c.length < period) return [];
        var k = 2 / (period + 1);
        var sum = 0;
        for (var i = 0; i < period; i++) sum += c[i];
        var prev = sum / period;
        var result = [{ time: t[period - 1], value: prev }];
        for (var j = period; j < c.length; j++) {
            prev = c[j] * k + prev * (1 - k);
            result.push({ time: t[j], value: prev });
        }
        return result;
    }

    function wma(candles, period) {
        var c = closes(candles);
        var t = times(candles);
        var result = [];
        var denom = period * (period + 1) / 2;
        for (var i = period - 1; i < c.length; i++) {
            var sum = 0;
            for (var j = 0; j < period; j++) {
                sum += c[i - period + 1 + j] * (j + 1);
            }
            result.push({ time: t[i], value: sum / denom });
        }
        return result;
    }

    // Internal: EMA over raw values array (no candle wrapper)
    function _emaRaw(values, period) {
        if (values.length < period) return [];
        var k = 2 / (period + 1);
        var sum = 0;
        for (var i = 0; i < period; i++) sum += values[i];
        var prev = sum / period;
        var result = new Array(period - 1).fill(null);
        result.push(prev);
        for (var j = period; j < values.length; j++) {
            prev = values[j] * k + prev * (1 - k);
            result.push(prev);
        }
        return result;
    }

    function _smaRaw(values, period) {
        var result = new Array(period - 1).fill(null);
        for (var i = period - 1; i < values.length; i++) {
            var sum = 0;
            for (var j = i - period + 1; j <= i; j++) sum += values[j];
            result.push(sum / period);
        }
        return result;
    }

    // =========================================================================
    // RSI — Relative Strength Index
    // =========================================================================

    function rsi(candles, period) {
        period = period || 14;
        var c = closes(candles);
        var t = times(candles);
        if (c.length < period + 1) return [];

        var gains = [];
        var losses = [];
        for (var i = 1; i < c.length; i++) {
            var diff = c[i] - c[i - 1];
            gains.push(diff > 0 ? diff : 0);
            losses.push(diff < 0 ? -diff : 0);
        }

        var avgGain = 0, avgLoss = 0;
        for (var j = 0; j < period; j++) {
            avgGain += gains[j];
            avgLoss += losses[j];
        }
        avgGain /= period;
        avgLoss /= period;

        var result = [];
        var rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        result.push({ time: t[period], value: 100 - 100 / (1 + rs) });

        for (var k = period; k < gains.length; k++) {
            avgGain = (avgGain * (period - 1) + gains[k]) / period;
            avgLoss = (avgLoss * (period - 1) + losses[k]) / period;
            rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
            result.push({ time: t[k + 1], value: 100 - 100 / (1 + rs) });
        }
        return result;
    }

    // =========================================================================
    // MACD — Moving Average Convergence Divergence
    // =========================================================================

    function macd(candles, fastPeriod, slowPeriod, signalPeriod) {
        fastPeriod = fastPeriod || 12;
        slowPeriod = slowPeriod || 26;
        signalPeriod = signalPeriod || 9;

        var c = closes(candles);
        var t = times(candles);

        var emaFast = _emaRaw(c, fastPeriod);
        var emaSlow = _emaRaw(c, slowPeriod);

        var macdLine = [];
        for (var i = 0; i < c.length; i++) {
            if (emaFast[i] !== null && emaSlow[i] !== null) {
                macdLine.push(emaFast[i] - emaSlow[i]);
            } else {
                macdLine.push(null);
            }
        }

        var validMacd = macdLine.filter(function (v) { return v !== null; });
        var signalRaw = _emaRaw(validMacd, signalPeriod);

        var result = { macd: [], signal: [], histogram: [] };
        var validStart = macdLine.indexOf(validMacd[0]);
        var si = 0;

        for (var j = validStart; j < c.length; j++) {
            var mVal = macdLine[j];
            if (mVal === null) continue;
            var sVal = signalRaw[si] !== null && signalRaw[si] !== undefined ? signalRaw[si] : null;
            result.macd.push({ time: t[j], value: mVal });
            if (sVal !== null) {
                result.signal.push({ time: t[j], value: sVal });
                result.histogram.push({ time: t[j], value: mVal - sVal, color: mVal - sVal >= 0 ? 'rgba(0,170,0,0.5)' : 'rgba(204,0,0,0.5)' });
            }
            si++;
        }
        return result;
    }

    // =========================================================================
    // Bollinger Bands
    // =========================================================================

    function bollingerBands(candles, period, stdDev) {
        period = period || 20;
        stdDev = stdDev || 2;
        var c = closes(candles);
        var t = times(candles);
        var result = { upper: [], middle: [], lower: [] };

        for (var i = period - 1; i < c.length; i++) {
            var sum = 0;
            for (var j = i - period + 1; j <= i; j++) sum += c[j];
            var mean = sum / period;
            var sqSum = 0;
            for (var k = i - period + 1; k <= i; k++) {
                sqSum += (c[k] - mean) * (c[k] - mean);
            }
            var sd = Math.sqrt(sqSum / period);
            result.upper.push({ time: t[i], value: mean + stdDev * sd });
            result.middle.push({ time: t[i], value: mean });
            result.lower.push({ time: t[i], value: mean - stdDev * sd });
        }
        return result;
    }

    // =========================================================================
    // Stochastic Oscillator
    // =========================================================================

    function stochastic(candles, kPeriod, dPeriod) {
        kPeriod = kPeriod || 14;
        dPeriod = dPeriod || 3;
        var h = highs(candles);
        var l = lows(candles);
        var c = closes(candles);
        var t = times(candles);

        var kValues = [];
        for (var i = kPeriod - 1; i < c.length; i++) {
            var hh = -Infinity, ll = Infinity;
            for (var j = i - kPeriod + 1; j <= i; j++) {
                if (h[j] > hh) hh = h[j];
                if (l[j] < ll) ll = l[j];
            }
            var kVal = hh === ll ? 50 : ((c[i] - ll) / (hh - ll)) * 100;
            kValues.push(kVal);
        }

        var result = { k: [], d: [] };
        for (var m = 0; m < kValues.length; m++) {
            result.k.push({ time: t[m + kPeriod - 1], value: kValues[m] });
        }

        for (var n = dPeriod - 1; n < kValues.length; n++) {
            var sum = 0;
            for (var p = n - dPeriod + 1; p <= n; p++) sum += kValues[p];
            result.d.push({ time: t[n + kPeriod - 1], value: sum / dPeriod });
        }
        return result;
    }

    // =========================================================================
    // ATR — Average True Range
    // =========================================================================

    function atr(candles, period) {
        period = period || 14;
        var t = times(candles);
        if (candles.length < period + 1) return [];

        var trValues = [];
        for (var i = 1; i < candles.length; i++) {
            var tr = Math.max(
                candles[i].high - candles[i].low,
                Math.abs(candles[i].high - candles[i - 1].close),
                Math.abs(candles[i].low - candles[i - 1].close)
            );
            trValues.push(tr);
        }

        var sum = 0;
        for (var j = 0; j < period; j++) sum += trValues[j];
        var atrVal = sum / period;
        var result = [{ time: t[period], value: atrVal }];

        for (var k = period; k < trValues.length; k++) {
            atrVal = (atrVal * (period - 1) + trValues[k]) / period;
            result.push({ time: t[k + 1], value: atrVal });
        }
        return result;
    }

    // =========================================================================
    // ADX — Average Directional Index
    // =========================================================================

    function adx(candles, period) {
        period = period || 14;
        var t = times(candles);
        if (candles.length < period * 2 + 1) return { adx: [], pdi: [], ndi: [] };

        var pdm = [], ndm = [], tr = [];
        for (var i = 1; i < candles.length; i++) {
            var upMove = candles[i].high - candles[i - 1].high;
            var downMove = candles[i - 1].low - candles[i].low;
            pdm.push(upMove > downMove && upMove > 0 ? upMove : 0);
            ndm.push(downMove > upMove && downMove > 0 ? downMove : 0);
            tr.push(Math.max(
                candles[i].high - candles[i].low,
                Math.abs(candles[i].high - candles[i - 1].close),
                Math.abs(candles[i].low - candles[i - 1].close)
            ));
        }

        var smoothTr = 0, smoothPdm = 0, smoothNdm = 0;
        for (var j = 0; j < period; j++) {
            smoothTr += tr[j];
            smoothPdm += pdm[j];
            smoothNdm += ndm[j];
        }

        var dxValues = [];
        var result = { adx: [], pdi: [], ndi: [] };

        for (var k = period; k <= tr.length; k++) {
            if (k > period) {
                smoothTr = smoothTr - smoothTr / period + tr[k - 1];
                smoothPdm = smoothPdm - smoothPdm / period + pdm[k - 1];
                smoothNdm = smoothNdm - smoothNdm / period + ndm[k - 1];
            }
            var pdi = smoothTr > 0 ? (smoothPdm / smoothTr) * 100 : 0;
            var ndi = smoothTr > 0 ? (smoothNdm / smoothTr) * 100 : 0;
            var diSum = pdi + ndi;
            var dx = diSum > 0 ? Math.abs(pdi - ndi) / diSum * 100 : 0;
            dxValues.push(dx);

            result.pdi.push({ time: t[k], value: pdi });
            result.ndi.push({ time: t[k], value: ndi });
        }

        // Smooth DX to get ADX
        if (dxValues.length >= period) {
            var adxSum = 0;
            for (var m = 0; m < period; m++) adxSum += dxValues[m];
            var adxVal = adxSum / period;
            result.adx.push({ time: result.pdi[period - 1].time, value: adxVal });

            for (var n = period; n < dxValues.length; n++) {
                adxVal = (adxVal * (period - 1) + dxValues[n]) / period;
                result.adx.push({ time: result.pdi[n].time, value: adxVal });
            }
        }
        return result;
    }

    // =========================================================================
    // CCI — Commodity Channel Index
    // =========================================================================

    function cci(candles, period) {
        period = period || 20;
        var t = times(candles);
        var result = [];

        var tp = candles.map(function (c) {
            return (c.high + c.low + c.close) / 3;
        });

        for (var i = period - 1; i < tp.length; i++) {
            var sum = 0;
            for (var j = i - period + 1; j <= i; j++) sum += tp[j];
            var mean = sum / period;
            var madSum = 0;
            for (var k = i - period + 1; k <= i; k++) madSum += Math.abs(tp[k] - mean);
            var mad = madSum / period;
            var cciVal = mad === 0 ? 0 : (tp[i] - mean) / (0.015 * mad);
            result.push({ time: t[i], value: cciVal });
        }
        return result;
    }

    // =========================================================================
    // Williams %R
    // =========================================================================

    function williamsR(candles, period) {
        period = period || 14;
        var h = highs(candles);
        var l = lows(candles);
        var c = closes(candles);
        var t = times(candles);
        var result = [];

        for (var i = period - 1; i < c.length; i++) {
            var hh = -Infinity, ll = Infinity;
            for (var j = i - period + 1; j <= i; j++) {
                if (h[j] > hh) hh = h[j];
                if (l[j] < ll) ll = l[j];
            }
            var wr = hh === ll ? -50 : ((hh - c[i]) / (hh - ll)) * -100;
            result.push({ time: t[i], value: wr });
        }
        return result;
    }

    // =========================================================================
    // ROC — Rate of Change
    // =========================================================================

    function roc(candles, period) {
        period = period || 12;
        var c = closes(candles);
        var t = times(candles);
        var result = [];
        for (var i = period; i < c.length; i++) {
            var prev = c[i - period];
            result.push({ time: t[i], value: prev === 0 ? 0 : ((c[i] - prev) / prev) * 100 });
        }
        return result;
    }

    // =========================================================================
    // OBV — On Balance Volume
    // =========================================================================

    function obv(candles) {
        var c = closes(candles);
        var v = volumes(candles);
        var t = times(candles);
        var result = [{ time: t[0], value: v[0] }];
        var cumObv = v[0];

        for (var i = 1; i < c.length; i++) {
            if (c[i] > c[i - 1]) cumObv += v[i];
            else if (c[i] < c[i - 1]) cumObv -= v[i];
            result.push({ time: t[i], value: cumObv });
        }
        return result;
    }

    // =========================================================================
    // VWAP — Volume Weighted Average Price
    // =========================================================================

    function vwap(candles) {
        var t = times(candles);
        var cumTPV = 0, cumVol = 0;
        var result = [];
        for (var i = 0; i < candles.length; i++) {
            var tp = (candles[i].high + candles[i].low + candles[i].close) / 3;
            var vol = candles[i].volume || 0;
            cumTPV += tp * vol;
            cumVol += vol;
            result.push({ time: t[i], value: cumVol > 0 ? cumTPV / cumVol : tp });
        }
        return result;
    }

    // =========================================================================
    // Bull/Bear Power (Elder)
    // =========================================================================

    function bullPower(candles, period) {
        period = period || 13;
        var emaValues = _emaRaw(closes(candles), period);
        var t = times(candles);
        var result = [];
        for (var i = 0; i < candles.length; i++) {
            if (emaValues[i] === null) continue;
            result.push({ time: t[i], value: candles[i].high - emaValues[i] });
        }
        return result;
    }

    function bearPower(candles, period) {
        period = period || 13;
        var emaValues = _emaRaw(closes(candles), period);
        var t = times(candles);
        var result = [];
        for (var i = 0; i < candles.length; i++) {
            if (emaValues[i] === null) continue;
            result.push({ time: t[i], value: candles[i].low - emaValues[i] });
        }
        return result;
    }

    // =========================================================================
    // Ultimate Oscillator
    // =========================================================================

    function ultimateOscillator(candles, p1, p2, p3) {
        p1 = p1 || 7;
        p2 = p2 || 14;
        p3 = p3 || 28;
        var t = times(candles);
        if (candles.length < p3 + 1) return [];

        var bp = [], tr = [];
        for (var i = 1; i < candles.length; i++) {
            var prevClose = candles[i - 1].close;
            var trueLow = Math.min(candles[i].low, prevClose);
            bp.push(candles[i].close - trueLow);
            tr.push(Math.max(candles[i].high, prevClose) - trueLow);
        }

        var result = [];
        for (var j = p3 - 1; j < bp.length; j++) {
            var bpSum1 = 0, trSum1 = 0;
            var bpSum2 = 0, trSum2 = 0;
            var bpSum3 = 0, trSum3 = 0;
            for (var k = j - p1 + 1; k <= j; k++) { bpSum1 += bp[k]; trSum1 += tr[k]; }
            for (var m = j - p2 + 1; m <= j; m++) { bpSum2 += bp[m]; trSum2 += tr[m]; }
            for (var n = j - p3 + 1; n <= j; n++) { bpSum3 += bp[n]; trSum3 += tr[n]; }

            var avg1 = trSum1 > 0 ? bpSum1 / trSum1 : 0;
            var avg2 = trSum2 > 0 ? bpSum2 / trSum2 : 0;
            var avg3 = trSum3 > 0 ? bpSum3 / trSum3 : 0;

            var uo = ((avg1 * 4) + (avg2 * 2) + avg3) / 7 * 100;
            result.push({ time: t[j + 1], value: uo });
        }
        return result;
    }

    // =========================================================================
    // Ichimoku Cloud
    // =========================================================================

    function ichimoku(candles, tenkanPeriod, kijunPeriod, senkouBPeriod, displacement) {
        tenkanPeriod = tenkanPeriod || 9;
        kijunPeriod = kijunPeriod || 26;
        senkouBPeriod = senkouBPeriod || 52;
        displacement = displacement || 26;

        var h = highs(candles);
        var l = lows(candles);
        var c = closes(candles);
        var t = times(candles);

        function midHL(start, len) {
            var hh = -Infinity, ll = Infinity;
            for (var i = start; i < start + len && i < h.length; i++) {
                if (h[i] > hh) hh = h[i];
                if (l[i] < ll) ll = l[i];
            }
            return (hh + ll) / 2;
        }

        var tenkan = [], kijun = [], senkouA = [], senkouB = [], chikou = [];

        for (var i = 0; i < candles.length; i++) {
            if (i >= tenkanPeriod - 1) {
                tenkan.push({ time: t[i], value: midHL(i - tenkanPeriod + 1, tenkanPeriod) });
            }
            if (i >= kijunPeriod - 1) {
                kijun.push({ time: t[i], value: midHL(i - kijunPeriod + 1, kijunPeriod) });
            }
            if (i >= tenkanPeriod - 1 && i >= kijunPeriod - 1) {
                var tenkanVal = midHL(i - tenkanPeriod + 1, tenkanPeriod);
                var kijunVal = midHL(i - kijunPeriod + 1, kijunPeriod);
                senkouA.push({ time: t[Math.min(i + displacement, candles.length - 1)], value: (tenkanVal + kijunVal) / 2 });
            }
            if (i >= senkouBPeriod - 1) {
                senkouB.push({ time: t[Math.min(i + displacement, candles.length - 1)], value: midHL(i - senkouBPeriod + 1, senkouBPeriod) });
            }
            if (i >= displacement) {
                chikou.push({ time: t[i - displacement], value: c[i] });
            }
        }

        return { tenkan: tenkan, kijun: kijun, senkouA: senkouA, senkouB: senkouB, chikou: chikou };
    }

    // =========================================================================
    // Pivot Points (Standard)
    // =========================================================================

    function pivotPoints(candles) {
        if (candles.length < 2) return null;
        var prev = candles[candles.length - 2];
        var pp = (prev.high + prev.low + prev.close) / 3;
        return {
            pp: pp,
            r1: 2 * pp - prev.low,
            r2: pp + (prev.high - prev.low),
            r3: prev.high + 2 * (pp - prev.low),
            s1: 2 * pp - prev.high,
            s2: pp - (prev.high - prev.low),
            s3: prev.low - 2 * (prev.high - pp)
        };
    }

    // =========================================================================
    // StochRSI
    // =========================================================================

    function stochRSI(candles, rsiPeriod, stochPeriod, kSmooth, dSmooth) {
        rsiPeriod = rsiPeriod || 14;
        stochPeriod = stochPeriod || 14;
        kSmooth = kSmooth || 3;
        dSmooth = dSmooth || 3;

        var rsiData = rsi(candles, rsiPeriod);
        if (rsiData.length < stochPeriod) return { k: [], d: [] };

        var rsiValues = rsiData.map(function (r) { return r.value; });
        var rsiTimes = rsiData.map(function (r) { return r.time; });

        var rawK = [];
        for (var i = stochPeriod - 1; i < rsiValues.length; i++) {
            var hh = -Infinity, ll = Infinity;
            for (var j = i - stochPeriod + 1; j <= i; j++) {
                if (rsiValues[j] > hh) hh = rsiValues[j];
                if (rsiValues[j] < ll) ll = rsiValues[j];
            }
            rawK.push({ time: rsiTimes[i], value: hh === ll ? 50 : ((rsiValues[i] - ll) / (hh - ll)) * 100 });
        }

        // Smooth K
        var kArr = rawK.map(function (r) { return r.value; });
        var smoothedK = _smaRaw(kArr, kSmooth);
        var resultK = [];
        for (var m = 0; m < rawK.length; m++) {
            if (smoothedK[m] !== null) {
                resultK.push({ time: rawK[m].time, value: smoothedK[m] });
            }
        }

        // D line
        var kVals = resultK.map(function (r) { return r.value; });
        var dRaw = _smaRaw(kVals, dSmooth);
        var resultD = [];
        for (var n = 0; n < resultK.length; n++) {
            if (dRaw[n] !== null) {
                resultD.push({ time: resultK[n].time, value: dRaw[n] });
            }
        }

        return { k: resultK, d: resultD };
    }

    // =========================================================================
    // Technical Summary — Signal Generator (mirrors investing.com logic)
    // =========================================================================

    function technicalSummary(candles) {
        if (candles.length < 52) return null;

        var signals = { buy: 0, sell: 0, neutral: 0 };

        // Moving Averages analysis
        var maPeriods = [5, 10, 20, 50, 100, 200];
        var maSignals = { buy: 0, sell: 0, neutral: 0 };
        var lastClose = candles[candles.length - 1].close;

        maPeriods.forEach(function (p) {
            if (candles.length < p) return;
            var smaResult = sma(candles, p);
            var emaResult = ema(candles, p);
            if (smaResult.length > 0) {
                var smaVal = smaResult[smaResult.length - 1].value;
                if (lastClose > smaVal) maSignals.buy++;
                else if (lastClose < smaVal) maSignals.sell++;
                else maSignals.neutral++;
            }
            if (emaResult.length > 0) {
                var emaVal = emaResult[emaResult.length - 1].value;
                if (lastClose > emaVal) maSignals.buy++;
                else if (lastClose < emaVal) maSignals.sell++;
                else maSignals.neutral++;
            }
        });

        // Oscillator analysis
        var oscSignals = { buy: 0, sell: 0, neutral: 0 };

        // RSI
        var rsiData = rsi(candles, 14);
        if (rsiData.length > 0) {
            var rsiVal = rsiData[rsiData.length - 1].value;
            if (rsiVal < 30) oscSignals.buy++;
            else if (rsiVal > 70) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // Stochastic
        var stochData = stochastic(candles, 14, 3);
        if (stochData.k.length > 0) {
            var stochK = stochData.k[stochData.k.length - 1].value;
            if (stochK < 20) oscSignals.buy++;
            else if (stochK > 80) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // CCI
        var cciData = cci(candles, 14);
        if (cciData.length > 0) {
            var cciVal = cciData[cciData.length - 1].value;
            if (cciVal < -100) oscSignals.buy++;
            else if (cciVal > 100) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // Williams %R
        var wrData = williamsR(candles, 14);
        if (wrData.length > 0) {
            var wrVal = wrData[wrData.length - 1].value;
            if (wrVal < -80) oscSignals.buy++;
            else if (wrVal > -20) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // MACD
        var macdData = macd(candles, 12, 26, 9);
        if (macdData.histogram.length > 0) {
            var histVal = macdData.histogram[macdData.histogram.length - 1].value;
            if (histVal > 0) oscSignals.buy++;
            else if (histVal < 0) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // ADX
        var adxData = adx(candles, 14);
        if (adxData.pdi.length > 0 && adxData.ndi.length > 0) {
            var lastPdi = adxData.pdi[adxData.pdi.length - 1].value;
            var lastNdi = adxData.ndi[adxData.ndi.length - 1].value;
            if (lastPdi > lastNdi) oscSignals.buy++;
            else if (lastNdi > lastPdi) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // Ultimate Oscillator
        var uoData = ultimateOscillator(candles, 7, 14, 28);
        if (uoData.length > 0) {
            var uoVal = uoData[uoData.length - 1].value;
            if (uoVal < 30) oscSignals.buy++;
            else if (uoVal > 70) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        // Bull/Bear Power
        var bpData = bullPower(candles, 13);
        var brData = bearPower(candles, 13);
        if (bpData.length > 0 && brData.length > 0) {
            var bpVal = bpData[bpData.length - 1].value;
            var brVal = brData[brData.length - 1].value;
            if (bpVal > 0 && brVal > 0) oscSignals.buy++;
            else if (bpVal < 0 && brVal < 0) oscSignals.sell++;
            else oscSignals.neutral++;
        }

        signals.buy = maSignals.buy + oscSignals.buy;
        signals.sell = maSignals.sell + oscSignals.sell;
        signals.neutral = maSignals.neutral + oscSignals.neutral;
        var total = signals.buy + signals.sell + signals.neutral;

        var verdict;
        var buyPct = total > 0 ? signals.buy / total : 0;
        var sellPct = total > 0 ? signals.sell / total : 0;

        if (buyPct > 0.6) verdict = 'STRONG_BUY';
        else if (buyPct > 0.4) verdict = 'BUY';
        else if (sellPct > 0.6) verdict = 'STRONG_SELL';
        else if (sellPct > 0.4) verdict = 'SELL';
        else verdict = 'NEUTRAL';

        return {
            verdict: verdict,
            signals: signals,
            maSignals: maSignals,
            oscSignals: oscSignals,
            details: {
                rsi: rsiData.length > 0 ? rsiData[rsiData.length - 1].value : null,
                stochK: stochData.k.length > 0 ? stochData.k[stochData.k.length - 1].value : null,
                cci: cciData.length > 0 ? cciData[cciData.length - 1].value : null,
                williamsR: wrData.length > 0 ? wrData[wrData.length - 1].value : null,
                macdHistogram: macdData.histogram.length > 0 ? macdData.histogram[macdData.histogram.length - 1].value : null,
                adxPdi: adxData.pdi.length > 0 ? adxData.pdi[adxData.pdi.length - 1].value : null,
                adxNdi: adxData.ndi.length > 0 ? adxData.ndi[adxData.ndi.length - 1].value : null,
                ultimateOsc: uoData.length > 0 ? uoData[uoData.length - 1].value : null
            }
        };
    }

    // =========================================================================
    // Exposed namespace
    // =========================================================================

    global.TechnicalIndicators = {
        // Moving Averages
        sma: sma,
        ema: ema,
        wma: wma,

        // Oscillators
        rsi: rsi,
        macd: macd,
        stochastic: stochastic,
        stochRSI: stochRSI,
        cci: cci,
        williamsR: williamsR,
        roc: roc,
        ultimateOscillator: ultimateOscillator,
        bullPower: bullPower,
        bearPower: bearPower,

        // Volatility
        bollingerBands: bollingerBands,
        atr: atr,

        // Trend
        adx: adx,
        ichimoku: ichimoku,

        // Volume
        obv: obv,
        vwap: vwap,

        // Levels
        pivotPoints: pivotPoints,

        // Composite
        technicalSummary: technicalSummary,

        // Helpers (for custom indicators)
        _emaRaw: _emaRaw,
        _smaRaw: _smaRaw,
    };

})(window);
