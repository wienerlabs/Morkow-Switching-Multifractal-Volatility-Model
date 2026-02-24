/**
 * indicators-ui.js — Wires TechnicalIndicators SDK to Lightweight Charts + Market page panel
 *
 * Dashboard (index.html): Adds overlay series to existing tvChart
 * Market (market.html): Renders Technical Summary gauge + indicator table
 */
(function () {
    'use strict';

    // =========================================================================
    // DOM helpers (safe — no innerHTML)
    // =========================================================================

    function el(tag, styles, text) {
        var e = document.createElement(tag);
        if (styles) e.style.cssText = styles;
        if (text) e.textContent = text;
        return e;
    }

    function appendText(parent, tag, styles, text) {
        var e = el(tag, styles, text);
        parent.appendChild(e);
        return e;
    }

    // =========================================================================
    // Dashboard: Indicator overlays on Lightweight Charts
    // =========================================================================

    var _indicatorSeries = {};
    var _activeIndicators = {};

    function getChart() { return window.tvChart || null; }
    function getCandles() { return window._tvCandleData || null; }

    function removeIndicatorSeries(name) {
        if (_indicatorSeries[name]) {
            var chart = getChart();
            if (chart) {
                var items = _indicatorSeries[name];
                if (!Array.isArray(items)) items = [items];
                items.forEach(function (s) {
                    try { chart.removeSeries(s); } catch (e) {}
                });
            }
            delete _indicatorSeries[name];
            delete _activeIndicators[name];
        }
    }

    function addLineSeries(chart, data, color, lineWidth, priceScaleId) {
        var opts = {
            color: color || '#2196F3',
            lineWidth: lineWidth || 1,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        };
        if (priceScaleId) opts.priceScaleId = priceScaleId;
        var series = chart.addSeries(LightweightCharts.LineSeries, opts);
        series.setData(data);
        return series;
    }

    // --- Individual indicator toggles ---

    function toggleSMA(candles, chart, period, color) {
        var key = 'sma_' + period;
        if (_indicatorSeries[key]) { removeIndicatorSeries(key); return; }
        var data = TechnicalIndicators.sma(candles, period);
        _indicatorSeries[key] = addLineSeries(chart, data, color || '#FF9800', 1);
        _activeIndicators[key] = true;
    }

    function toggleEMA(candles, chart, period, color) {
        var key = 'ema_' + period;
        if (_indicatorSeries[key]) { removeIndicatorSeries(key); return; }
        var data = TechnicalIndicators.ema(candles, period);
        _indicatorSeries[key] = addLineSeries(chart, data, color || '#2196F3', 1);
        _activeIndicators[key] = true;
    }

    function toggleBollingerBands(candles, chart) {
        var key = 'bb';
        if (_indicatorSeries[key]) { removeIndicatorSeries(key); return; }
        var bb = TechnicalIndicators.bollingerBands(candles, 20, 2);
        var upper = addLineSeries(chart, bb.upper, '#9C27B0', 1);
        var middle = addLineSeries(chart, bb.middle, '#9C27B0', 1);
        var lower = addLineSeries(chart, bb.lower, '#9C27B0', 1);
        _indicatorSeries[key] = [upper, middle, lower];
        _activeIndicators[key] = true;
    }

    function toggleVWAP(candles, chart) {
        var key = 'vwap';
        if (_indicatorSeries[key]) { removeIndicatorSeries(key); return; }
        var data = TechnicalIndicators.vwap(candles);
        _indicatorSeries[key] = addLineSeries(chart, data, '#E91E63', 2);
        _activeIndicators[key] = true;
    }

    function toggleIchimoku(candles, chart) {
        var key = 'ichimoku';
        if (_indicatorSeries[key]) { removeIndicatorSeries(key); return; }
        var ichi = TechnicalIndicators.ichimoku(candles);
        var tenkan = addLineSeries(chart, ichi.tenkan, '#0000FF', 1);
        var kijun = addLineSeries(chart, ichi.kijun, '#FF0000', 1);
        var senkouA = addLineSeries(chart, ichi.senkouA, '#00AA00', 1);
        var senkouB = addLineSeries(chart, ichi.senkouB, '#CC0000', 1);
        var chikou = addLineSeries(chart, ichi.chikou, '#888888', 1);
        _indicatorSeries[key] = [tenkan, kijun, senkouA, senkouB, chikou];
        _activeIndicators[key] = true;
    }

    function reapplyIndicators() {
        var chart = getChart();
        var candles = getCandles();
        if (!chart || !candles || !candles.length) return;

        var active = Object.keys(_activeIndicators);
        active.forEach(function (key) { removeIndicatorSeries(key); });

        active.forEach(function (key) {
            if (key.startsWith('sma_')) toggleSMA(candles, chart, parseInt(key.split('_')[1]));
            else if (key.startsWith('ema_')) toggleEMA(candles, chart, parseInt(key.split('_')[1]));
            else if (key === 'bb') toggleBollingerBands(candles, chart);
            else if (key === 'vwap') toggleVWAP(candles, chart);
            else if (key === 'ichimoku') toggleIchimoku(candles, chart);
        });
    }

    // =========================================================================
    // Market page: Technical Analysis Summary Panel (safe DOM construction)
    // =========================================================================

    function renderTechnicalSummaryPanel(candles) {
        var container = document.getElementById('taSummaryPanel');
        if (!container || !candles || candles.length < 52) return;

        var summary = TechnicalIndicators.technicalSummary(candles);
        if (!summary) return;

        var verdictColors = {
            STRONG_BUY: '#00aa00', BUY: '#66bb6a',
            NEUTRAL: '#ff9800',
            SELL: '#ef5350', STRONG_SELL: '#cc0000'
        };
        var verdictLabels = {
            STRONG_BUY: 'Strong Buy', BUY: 'Buy',
            NEUTRAL: 'Neutral',
            SELL: 'Sell', STRONG_SELL: 'Strong Sell'
        };

        var color = verdictColors[summary.verdict] || '#888';
        var label = verdictLabels[summary.verdict] || summary.verdict;
        var total = summary.signals.buy + summary.signals.sell + summary.signals.neutral;

        container.textContent = '';

        // Gauge header
        var gaugeDiv = el('div', 'text-align:center;padding:1.25rem 0;');
        appendText(gaugeDiv, 'div', 'font-size:0.55rem;color:var(--dim);text-transform:uppercase;margin-bottom:0.5rem;', 'Technical Summary');
        appendText(gaugeDiv, 'div', 'font-size:1.8rem;font-weight:600;color:' + color + ';margin-bottom:0.25rem;', label);

        var signalRow = el('div', 'font-size:0.6rem;color:var(--dim);');
        var buySpan = el('span', 'color:#00aa00;', 'Buy: ' + summary.signals.buy);
        var neutSpan = el('span', 'color:#ff9800;', 'Neutral: ' + summary.signals.neutral);
        var sellSpan = el('span', 'color:#cc0000;', 'Sell: ' + summary.signals.sell);
        signalRow.appendChild(buySpan);
        signalRow.appendChild(document.createTextNode(' \u00B7 '));
        signalRow.appendChild(neutSpan);
        signalRow.appendChild(document.createTextNode(' \u00B7 '));
        signalRow.appendChild(sellSpan);
        gaugeDiv.appendChild(signalRow);
        container.appendChild(gaugeDiv);

        // Signal bar
        var barDiv = el('div', 'display:flex;height:6px;margin:0 1rem 1rem;overflow:hidden;');
        var buyW = total > 0 ? (summary.signals.buy / total * 100) : 33;
        var neutW = total > 0 ? (summary.signals.neutral / total * 100) : 34;
        var sellW = total > 0 ? (summary.signals.sell / total * 100) : 33;
        barDiv.appendChild(el('div', 'width:' + buyW + '%;background:#00aa00;'));
        barDiv.appendChild(el('div', 'width:' + neutW + '%;background:#ff9800;'));
        barDiv.appendChild(el('div', 'width:' + sellW + '%;background:#cc0000;'));
        container.appendChild(barDiv);

        // Sub-panels: MA + Oscillators
        var subGrid = el('div', 'display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--border);');

        function makeSubPanel(title, signals) {
            var t2 = signals.buy + signals.sell + signals.neutral;
            var subVerdict;
            if (t2 > 0) {
                var bPct = signals.buy / t2;
                var sPct = signals.sell / t2;
                if (bPct > 0.6) subVerdict = 'Strong Buy';
                else if (bPct > 0.4) subVerdict = 'Buy';
                else if (sPct > 0.6) subVerdict = 'Strong Sell';
                else if (sPct > 0.4) subVerdict = 'Sell';
                else subVerdict = 'Neutral';
            } else {
                subVerdict = 'N/A';
            }
            var col = subVerdict.includes('Buy') ? '#00aa00' : subVerdict.includes('Sell') ? '#cc0000' : '#ff9800';

            var div = el('div', 'padding:0.75rem 1rem;text-align:center;');
            appendText(div, 'div', 'font-size:0.5rem;color:var(--dim);text-transform:uppercase;margin-bottom:0.3rem;', title);
            appendText(div, 'div', 'font-size:0.85rem;color:' + col + ';font-weight:600;', subVerdict);
            appendText(div, 'div', 'font-size:0.5rem;color:var(--dim);margin-top:0.2rem;', 'B:' + signals.buy + ' N:' + signals.neutral + ' S:' + signals.sell);
            return div;
        }

        subGrid.appendChild(makeSubPanel('Moving Averages', summary.maSignals));
        subGrid.appendChild(makeSubPanel('Oscillators', summary.oscSignals));
        container.appendChild(subGrid);

        // Oscillator detail table
        var d = summary.details;
        var rows = [
            ['RSI (14)', d.rsi, d.rsi < 30 ? 'Buy' : d.rsi > 70 ? 'Sell' : 'Neutral'],
            ['Stochastic %K', d.stochK, d.stochK < 20 ? 'Buy' : d.stochK > 80 ? 'Sell' : 'Neutral'],
            ['CCI (14)', d.cci, d.cci < -100 ? 'Buy' : d.cci > 100 ? 'Sell' : 'Neutral'],
            ['Williams %R', d.williamsR, d.williamsR < -80 ? 'Buy' : d.williamsR > -20 ? 'Sell' : 'Neutral'],
            ['MACD Histogram', d.macdHistogram, d.macdHistogram > 0 ? 'Buy' : d.macdHistogram < 0 ? 'Sell' : 'Neutral'],
            ['ADX +DI / -DI', (d.adxPdi !== null ? d.adxPdi.toFixed(1) + ' / ' + d.adxNdi.toFixed(1) : null), d.adxPdi > d.adxNdi ? 'Buy' : d.adxNdi > d.adxPdi ? 'Sell' : 'Neutral'],
            ['Ultimate Osc', d.ultimateOsc, d.ultimateOsc < 30 ? 'Buy' : d.ultimateOsc > 70 ? 'Sell' : 'Neutral'],
        ];

        var table = el('div', 'border-top:1px solid var(--border);');

        // Table header
        var header = el('div', 'display:grid;grid-template-columns:1fr 1fr 1fr;padding:0.4rem 1rem;background:#f5f5f5;font-size:0.5rem;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--border);');
        appendText(header, 'div', '', 'Indicator');
        appendText(header, 'div', 'text-align:right;', 'Value');
        appendText(header, 'div', 'text-align:right;', 'Signal');
        table.appendChild(header);

        rows.forEach(function (row) {
            if (row[1] === null) return;
            var signalColor = row[2] === 'Buy' ? '#00aa00' : row[2] === 'Sell' ? '#cc0000' : '#ff9800';
            var valStr = typeof row[1] === 'number' ? row[1].toFixed(2) : row[1];

            var r = el('div', 'display:grid;grid-template-columns:1fr 1fr 1fr;padding:0.35rem 1rem;font-size:0.6rem;border-bottom:1px solid #f0f0f0;');
            appendText(r, 'div', '', row[0]);
            appendText(r, 'div', 'text-align:right;font-family:var(--font-mono);', valStr);
            appendText(r, 'div', 'text-align:right;color:' + signalColor + ';font-weight:600;', row[2]);
            table.appendChild(r);
        });

        container.appendChild(table);
    }

    // =========================================================================
    // Indicator toggle toolbar (dashboard)
    // =========================================================================

    function handleIndicatorToggle(btn) {
        var indicator = btn.dataset.indicator;
        var chart = getChart();
        var candles = getCandles();
        if (!chart || !candles || !candles.length) return;

        btn.classList.toggle('active');

        switch (indicator) {
            case 'sma20': toggleSMA(candles, chart, 20, '#FF9800'); break;
            case 'sma50': toggleSMA(candles, chart, 50, '#4CAF50'); break;
            case 'sma200': toggleSMA(candles, chart, 200, '#F44336'); break;
            case 'ema12': toggleEMA(candles, chart, 12, '#2196F3'); break;
            case 'ema26': toggleEMA(candles, chart, 26, '#00BCD4'); break;
            case 'bb': toggleBollingerBands(candles, chart); break;
            case 'vwap': toggleVWAP(candles, chart); break;
            case 'ichimoku': toggleIchimoku(candles, chart); break;
        }
    }

    function initIndicatorToolbar() {
        var toolbar = document.getElementById('indicatorToolbar');
        if (!toolbar) return;
        toolbar.addEventListener('click', function (e) {
            var btn = e.target.closest('[data-indicator]');
            if (btn) handleIndicatorToggle(btn);
        });
    }

    // =========================================================================
    // Exported API
    // =========================================================================

    window.IndicatorsUI = {
        reapplyIndicators: reapplyIndicators,
        renderTechnicalSummaryPanel: renderTechnicalSummaryPanel,
        handleIndicatorToggle: handleIndicatorToggle,
        removeAll: function () {
            Object.keys(_indicatorSeries).forEach(removeIndicatorSeries);
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initIndicatorToolbar);
    } else {
        initIndicatorToolbar();
    }

})();
