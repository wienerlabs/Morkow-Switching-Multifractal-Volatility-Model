/**
 * market-ta.js — Fetches OHLCV data and populates the Technical Analysis panel on market.html
 *
 * Uses GeckoTerminal API (same as dashboard chart) to get SOL/USDC candles,
 * then runs all indicators and fills the TA summary + mini cards.
 */
(function () {
    'use strict';

    var SOL_POOL = '8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj';
    var TA_TIMEFRAME = 'hour';
    var TA_AGGREGATE = 4;
    var TA_LIMIT = 300;

    async function fetchOHLCV() {
        var url = 'https://api.geckoterminal.com/api/v2/networks/solana/pools/' +
            SOL_POOL + '/ohlcv/' + TA_TIMEFRAME + '?aggregate=' + TA_AGGREGATE + '&limit=' + TA_LIMIT;
        var res = await fetch(url);
        if (!res.ok) throw new Error('OHLCV fetch failed: ' + res.status);
        var data = await res.json();
        var ohlcv = data.data.attributes.ohlcv_list;
        return ohlcv.map(function (d) {
            return {
                time: d[0],
                open: parseFloat(d[1]),
                high: parseFloat(d[2]),
                low: parseFloat(d[3]),
                close: parseFloat(d[4]),
                volume: parseFloat(d[5])
            };
        }).sort(function (a, b) { return a.time - b.time; });
    }

    function setVal(id, value, signal) {
        var el = document.getElementById(id);
        if (!el) return;
        el.textContent = value;
        if (signal === 'Buy') el.style.color = '#00aa00';
        else if (signal === 'Sell') el.style.color = '#cc0000';
        else el.style.color = '#ff9800';
    }

    function fmt(v) {
        if (v === null || v === undefined) return '--';
        return v.toFixed(2);
    }

    function fmtPrice(v) {
        if (v === null || v === undefined) return '--';
        return '$' + v.toFixed(2);
    }

    function rsiSignal(v) { return v < 30 ? 'Buy' : v > 70 ? 'Sell' : 'Neutral'; }
    function stochSignal(v) { return v < 20 ? 'Buy' : v > 80 ? 'Sell' : 'Neutral'; }
    function cciSignal(v) { return v < -100 ? 'Buy' : v > 100 ? 'Sell' : 'Neutral'; }
    function wrSignal(v) { return v < -80 ? 'Buy' : v > -20 ? 'Sell' : 'Neutral'; }
    function uoSignal(v) { return v < 30 ? 'Buy' : v > 70 ? 'Sell' : 'Neutral'; }

    async function loadTA() {
        if (typeof TechnicalIndicators === 'undefined') return;

        try {
            var candles = await fetchOHLCV();
            if (!candles || candles.length < 52) return;

            var TI = TechnicalIndicators;
            var lastClose = candles[candles.length - 1].close;

            // Oscillators
            var rsiData = TI.rsi(candles, 14);
            var rsiVal = rsiData.length > 0 ? rsiData[rsiData.length - 1].value : null;
            setVal('taRSIVal', fmt(rsiVal), rsiVal !== null ? rsiSignal(rsiVal) : 'Neutral');

            var macdData = TI.macd(candles, 12, 26, 9);
            var macdHist = macdData.histogram.length > 0 ? macdData.histogram[macdData.histogram.length - 1].value : null;
            setVal('taMACDVal', fmt(macdHist), macdHist !== null ? (macdHist > 0 ? 'Buy' : 'Sell') : 'Neutral');

            var stochData = TI.stochastic(candles, 14, 3);
            var stochK = stochData.k.length > 0 ? stochData.k[stochData.k.length - 1].value : null;
            setVal('taStochVal', fmt(stochK), stochK !== null ? stochSignal(stochK) : 'Neutral');

            var cciData = TI.cci(candles, 20);
            var cciVal = cciData.length > 0 ? cciData[cciData.length - 1].value : null;
            setVal('taCCIVal', fmt(cciVal), cciVal !== null ? cciSignal(cciVal) : 'Neutral');

            var wrData = TI.williamsR(candles, 14);
            var wrVal = wrData.length > 0 ? wrData[wrData.length - 1].value : null;
            setVal('taWRVal', fmt(wrVal), wrVal !== null ? wrSignal(wrVal) : 'Neutral');

            var adxData = TI.adx(candles, 14);
            var adxVal = adxData.adx.length > 0 ? adxData.adx[adxData.adx.length - 1].value : null;
            setVal('taADXVal', fmt(adxVal), adxData.pdi.length > 0 ? (adxData.pdi[adxData.pdi.length - 1].value > adxData.ndi[adxData.ndi.length - 1].value ? 'Buy' : 'Sell') : 'Neutral');

            var atrData = TI.atr(candles, 14);
            var atrVal = atrData.length > 0 ? atrData[atrData.length - 1].value : null;
            setVal('taATRVal', fmtPrice(atrVal), 'Neutral');

            var uoData = TI.ultimateOscillator(candles, 7, 14, 28);
            var uoVal = uoData.length > 0 ? uoData[uoData.length - 1].value : null;
            setVal('taUOVal', fmt(uoVal), uoVal !== null ? uoSignal(uoVal) : 'Neutral');

            // Moving Averages
            function maVal(data) { return data.length > 0 ? data[data.length - 1].value : null; }
            function maSignal(val) {
                if (val === null) return 'Neutral';
                return lastClose > val ? 'Buy' : lastClose < val ? 'Sell' : 'Neutral';
            }

            var sma20 = maVal(TI.sma(candles, 20));
            var sma50 = maVal(TI.sma(candles, 50));
            var sma200 = maVal(TI.sma(candles, 200));
            var ema12 = maVal(TI.ema(candles, 12));
            var ema26 = maVal(TI.ema(candles, 26));
            var vwapVal = maVal(TI.vwap(candles));

            setVal('taSMA20', fmtPrice(sma20), maSignal(sma20));
            setVal('taSMA50', fmtPrice(sma50), maSignal(sma50));
            setVal('taSMA200', fmtPrice(sma200), maSignal(sma200));
            setVal('taEMA12', fmtPrice(ema12), maSignal(ema12));
            setVal('taEMA26', fmtPrice(ema26), maSignal(ema26));
            setVal('taVWAP', fmtPrice(vwapVal), maSignal(vwapVal));

            // Technical Summary panel
            if (typeof IndicatorsUI !== 'undefined') {
                IndicatorsUI.renderTechnicalSummaryPanel(candles);
            }

        } catch (err) {
            console.error('[market-ta] Failed to load TA data:', err);
        }
    }

    // Load on page ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { setTimeout(loadTA, 1000); });
    } else {
        setTimeout(loadTA, 1000);
    }

    // Refresh every 5 minutes
    setInterval(loadTA, 300000);

})();
