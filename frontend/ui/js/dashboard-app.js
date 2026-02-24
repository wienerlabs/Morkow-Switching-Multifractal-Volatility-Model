// === DATA SOURCES STATUS PANEL ===
(function() {
    const DATA_SOURCES = [
        { name: 'Pyth', type: 'Oracle', interval: '400ms', dataTypes: 'Price feeds, confidence' },
        { name: 'Switchboard', type: 'Oracle', interval: '400ms', dataTypes: 'Price feeds, VRF' },
        { name: 'Birdeye', type: 'Token Data', interval: '1s', dataTypes: 'Token metrics, prices' },
        { name: 'Jupiter', type: 'DEX Agg', interval: '1s', dataTypes: 'Swap routes, quotes' },
        { name: 'Helius', type: 'RPC', interval: 'Per block', dataTypes: 'TX streams, webhooks' },
        { name: 'Protocol APIs', type: 'DeFi', interval: '5s', dataTypes: 'Drift, Orca, Kamino' },
        { name: 'Social Feeds', type: 'Sentiment', interval: '10s', dataTypes: 'Twitter, Discord, TG' },
        { name: 'Macro Data', type: 'Market', interval: '60s', dataTypes: 'BTC dom, fear/greed' },
        { name: 'News APIs', type: 'Intel', interval: '30s', dataTypes: 'CC, NewsData, Panic' },
    ];

    var _macroData = null;
    var _macroLastFetch = 0;
    var _sourceHealth = {};

    var SOURCE_ENDPOINTS = {
        'Pyth':          { url: 'https://hermes.pyth.network/api/latest_price_feeds?ids[]=0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d', type: 'json' },
        'Birdeye':       { url: 'https://public-api.birdeye.so/public/tokenlist?sort_by=v24hUSD&sort_type=desc&offset=0&limit=1', type: 'json' },
        'Jupiter':       { url: 'https://price.jup.ag/v6/price?ids=So11111111111111111111111111111111111111112', type: 'json' },
        'Macro Data':    { url: null },
        'News APIs':     { url: null },
    };

    async function pingSource(name) {
        var ep = SOURCE_ENDPOINTS[name];
        if (!ep || !ep.url) return;
        var t0 = performance.now();
        try {
            var res = await fetch(ep.url, { signal: AbortSignal.timeout(8000) });
            var elapsed = Math.round(performance.now() - t0);
            _sourceHealth[name] = { status: res.ok ? 'online' : 'degraded', latency: elapsed, lastCheck: Date.now() };
        } catch (_) {
            _sourceHealth[name] = { status: 'offline', latency: 0, lastCheck: Date.now() };
        }
    }

    async function pingAllSources() {
        var names = Object.keys(SOURCE_ENDPOINTS).filter(function(n) { return SOURCE_ENDPOINTS[n].url; });
        await Promise.allSettled(names.map(pingSource));
    }

    async function fetchMacroIndicators() {
        var t0 = performance.now();
        try {
            var data = await CortexAPI.get('/macro/indicators');
            var elapsed = Math.round(performance.now() - t0);
            if (data && data.fear_greed) {
                _macroData = data;
                _macroData._latencyMs = elapsed;
                _macroLastFetch = Date.now();
                _sourceHealth['Macro Data'] = { status: 'online', latency: elapsed, lastCheck: Date.now() };
                return;
            }
        } catch (e) { /* fall through */ }
        _sourceHealth['Macro Data'] = { status: 'offline', latency: 0, lastCheck: Date.now() };
    }

    function getSourceStatus() {
        return DATA_SOURCES.map(function(src) {
            var health = _sourceHealth[src.name];
            if (health) {
                var age = (Date.now() - health.lastCheck) / 1000;
                var freshness = health.status === 'offline' ? 'dead' : age > 120 ? 'stale' : 'fresh';
                return Object.assign({}, src, { status: health.status, latency: health.latency, freshness: freshness });
            }
            return Object.assign({}, src, { status: 'unknown', latency: 0, freshness: 'unknown' });
        });
    }

    function renderMacroIndicators() {
        var el = document.getElementById('dsMacro');
        if (!el) return;
        var btcDom = '\u2014', solDom = '\u2014', gasAvg = '\u2014';
        var fearGreed = 0, fgLabel = '\u2014', fgColor = 'var(--dim)';
        if (_macroData) {
            var bd = _macroData.btc_dominance || {};
            btcDom = (bd.btc_dominance || 0).toFixed(1) + '%';
            solDom = (bd.sol_dominance || 0).toFixed(1) + '%';
            var fg = _macroData.fear_greed || {};
            fearGreed = fg.value || 0;
            fgLabel = fg.classification || 'Neutral';
            fgColor = fearGreed >= 60 ? 'var(--green)' : fearGreed >= 40 ? 'var(--dim)' : 'var(--red)';
            var gas = _macroData.avg_gas_sol;
            gasAvg = gas != null ? gas.toFixed(6) : '\u2014';
        }
        el.textContent = '';
        var items = [
            { label: 'BTC Dominance', value: btcDom, color: null },
            { label: 'Fear & Greed', value: fearGreed + ' \u00b7 ' + fgLabel, color: fgColor },
            { label: 'SOL Dominance', value: solDom, color: null },
            { label: 'Avg Gas (SOL)', value: gasAvg, color: null },
        ];
        items.forEach(function(item) {
            var div = document.createElement('div');
            div.className = 'ds-macro-item';
            var lbl = document.createElement('div');
            lbl.className = 'ds-macro-label';
            lbl.textContent = item.label;
            var val = document.createElement('div');
            val.className = 'ds-macro-value';
            val.textContent = item.value;
            if (item.color) val.style.color = item.color;
            div.appendChild(lbl);
            div.appendChild(val);
            el.appendChild(div);
        });
    }

    function renderDataSources() {
        var sources = getSourceStatus();
        var onlineCount = sources.filter(function(s) { return s.status === 'online'; }).length;
        var degradedCount = sources.filter(function(s) { return s.status === 'degraded'; }).length;
        var summaryEl = document.getElementById('dsStatusSummary');
        var summaryText = '';
        if (degradedCount === 0 && onlineCount === sources.length) {
            summaryText = sources.length + ' Sources \u00b7 ALL ONLINE';
            summaryEl.textContent = summaryText;
            summaryEl.style.color = 'var(--green)';
        } else {
            summaryEl.textContent = onlineCount + '/' + sources.length + ' Online';
            summaryEl.style.color = '';
        }

        var grid = document.getElementById('dsGrid');
        grid.textContent = '';
        sources.forEach(function(s) {
            var latStr = s.latency === 0 ? '\u2014' : s.latency + 'ms';
            var latColor = s.status === 'offline' ? 'var(--red)' : s.latency > 500 ? 'var(--red)' : s.latency > 200 ? '#cc8800' : 'var(--fg)';
            var item = document.createElement('div');
            item.className = 'ds-item';

            var nameDiv = document.createElement('div');
            nameDiv.className = 'ds-name';
            var dot = document.createElement('span');
            dot.className = 'ds-dot ' + s.status;
            nameDiv.appendChild(dot);
            nameDiv.appendChild(document.createTextNode(s.name));

            var metaDiv = document.createElement('div');
            metaDiv.className = 'ds-meta';
            metaDiv.textContent = s.type + ' \u00b7 ' + s.interval;

            var latDiv = document.createElement('div');
            latDiv.className = 'ds-latency';
            latDiv.style.color = latColor;
            latDiv.textContent = latStr;

            var freshDiv = document.createElement('div');
            freshDiv.className = 'ds-freshness ' + s.freshness;
            freshDiv.textContent = s.freshness.toUpperCase();

            item.appendChild(nameDiv);
            item.appendChild(metaDiv);
            item.appendChild(latDiv);
            item.appendChild(freshDiv);
            grid.appendChild(item);
        });

        renderMacroIndicators();
    }

    Promise.all([fetchMacroIndicators(), pingAllSources()]).then(function() { renderDataSources(); });
    setInterval(fetchMacroIndicators, 60000);
    setInterval(pingAllSources, 60000);
    setInterval(renderDataSources, 8000);
})();

// === EXECUTION MONITORING — LIVE API ===
(function() {
    var STEP_IDS = ['emStep1','emStep2','emStep3','emStep4'];
    var CHECK_IDS = ['emChk1','emChk2','emChk3','emChk4'];
    var CONF_STAGES = ['Submitted','Processed','Confirmed','Finalized'];
    var _lastStatus = null;
    var _sseConnected = false;

    function statusColor(s) {
        if (s === 'SUCCESS' || s === 'SIMULATED') return 'text-green';
        if (s === 'FAILED' || s === 'REJECTED' || s === 'BLOCKED') return 'text-red';
        if (s === 'EXECUTING') return '';
        return 'text-dim';
    }

    function applyPipeline(data) {
        if (!data) return;
        _lastStatus = data;
        var statusEl = document.getElementById('emStatus');
        statusEl.textContent = data.status || 'IDLE';
        statusEl.className = statusColor(data.status);
        var steps = data.steps || [];
        STEP_IDS.forEach(function(id, i) {
            var step = steps[i] || {};
            var el = document.getElementById(id);
            if (!el) return;
            var state = step.state || 'pending';
            el.className = 'em-step ' + state;
            var statusSpan = el.querySelector('.em-step-status');
            statusSpan.textContent = step.detail || 'Waiting';
            statusSpan.className = 'em-step-status ' + (state === 'done' ? 'text-green' : state === 'failed' ? 'text-red' : 'text-dim');
            el.querySelector('.em-step-bar-fill').style.width = (state === 'done' || state === 'failed' ? '100' : state === 'active' ? '60' : '0') + '%';
        });
        var checks = data.guardian_checks || [];
        CHECK_IDS.forEach(function(id, i) {
            var chk = checks[i] || {};
            var el = document.getElementById(id);
            if (!el) return;
            if (chk.passed === true) { el.textContent = '\u2713'; el.className = 'em-check-icon pass'; }
            else if (chk.passed === false) { el.textContent = '\u2717'; el.className = 'em-check-icon fail'; }
            else { el.textContent = '\u25CB'; el.className = 'em-check-icon wait'; }
        });
        var slip = data.slippage || {};
        var expPct = Math.min(100, (slip.expected_pct || 0) * 100);
        var actPct = Math.min(100, (slip.actual_pct || 0) * 100);
        document.getElementById('emSlipExp').style.width = expPct + '%';
        document.getElementById('emSlipAct').style.width = actPct + '%';
        document.getElementById('emSlipValues').textContent = (slip.expected_pct || 0).toFixed(2) + '% / ' + (slip.actual_pct || 0).toFixed(2) + '%';
        var retries = data.retries || {};
        var count = retries.count || 0;
        var maxR = retries.max || 3;
        var retryContainer = document.getElementById('emRetryTimeline');
        retryContainer.textContent = '';
        for (var ri = 0; ri < maxR; ri++) {
            var dot = document.createElement('div');
            dot.className = 'em-retry-dot ' + (ri < count ? 'success' : 'pending');
            retryContainer.appendChild(dot);
        }
        var retryLabel = document.createElement('span');
        retryLabel.style.cssText = 'font-size:0.55rem;color:var(--dim);margin-left:4px';
        retryLabel.textContent = count + '/' + maxR + ' attempts';
        retryContainer.appendChild(retryLabel);

        var conf = data.confirmation || {};
        var confIdx = typeof conf.current_index === 'number' ? conf.current_index : -1;
        var stepper = document.getElementById('emConfStepper');
        stepper.textContent = '';
        CONF_STAGES.forEach(function(s, j) {
            var step = document.createElement('div');
            step.className = 'em-conf-step ' + (j <= confIdx ? 'done' : j === confIdx + 1 ? 'active' : '');
            step.textContent = s;
            stepper.appendChild(step);
        });

        var tx = data.tx_metrics || {};
        var mev = tx.mev_protection || 'JITO';
        document.getElementById('emMev').textContent = mev;
        document.getElementById('emMev').className = 'em-tx-value ' + (mev === 'JITO' ? 'text-green' : '');
        document.getElementById('emPriFee').textContent = (tx.priority_fee_sol || 0).toFixed(5);
        document.getElementById('emLatency').textContent = (tx.latency_ms || 0) + 'ms';
        var sig = tx.tx_signature;
        document.getElementById('emTxSig').textContent = sig ? (sig.slice(0, 3) + '...' + sig.slice(-3)) : '\u2014';
        document.getElementById('emTxStatus').textContent = tx.tx_status || data.status || 'IDLE';
        document.getElementById('emTxStatus').className = 'em-tx-value ' + statusColor(tx.tx_status || data.status);
    }

    async function pollPipeline() {
        try {
            var data = await CortexAPI.get('/execution/pipeline-status');
            if (data) applyPipeline(data);
        } catch (e) { console.warn('[ExecMonitor] Poll failed:', e.message); }
    }

    function connectSSE() {
        if (_sseConnected) return;
        try {
            CortexAPI.sseConnect('/execution/stream', function() { pollPipeline(); }, function() { _sseConnected = false; setTimeout(connectSSE, 10000); });
            _sseConnected = true;
        } catch (e) { console.warn('[ExecMonitor] SSE connect failed:', e.message); }
    }

    pollPipeline();
    connectSSE();
    setInterval(pollPipeline, 5000);
})();

// === TX Matrix initialization ===
(function() {
    var txMatrixContainer = document.getElementById('txMatrixContainer');
    var txScanner = document.getElementById('txScanner');
    var txRowCount = 8;
    var TX_RPC = 'https://api.mainnet-beta.solana.com';
    var DEX_SOURCES = [
        { addr: 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4', label: 'JUP' },
        { addr: '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8', label: 'RAY' },
    ];
    var liveTxBuffer = [];
    var txFetchInFlight = false;

    async function fetchLiveTxSignatures() {
        if (txFetchInFlight) return;
        txFetchInFlight = true;
        try {
            var results = await Promise.allSettled(DEX_SOURCES.map(async function(src) {
                var res = await fetch(TX_RPC, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'getSignaturesForAddress', params: [src.addr, { limit: 8 }] })
                });
                if (!res.ok) throw new Error(src.label + ' HTTP ' + res.status);
                var json = await res.json();
                if (json.error) { console.warn('[TX]', src.label, 'RPC error:', json.error.message); return []; }
                if (!json.result) return [];
                return json.result.map(function(tx) {
                    return { sig: tx.signature.slice(0, 4) + '...' + tx.signature.slice(-4), fullSig: tx.signature, label: src.label, ok: tx.err === null, time: tx.blockTime, status: tx.confirmationStatus };
                });
            }));
            var merged = results.filter(function(r) { return r.status === 'fulfilled'; }).flatMap(function(r) { return r.value; }).sort(function(a, b) { return (b.time || 0) - (a.time || 0); });
            if (merged.length > 0) liveTxBuffer = merged;
            console.log('[TX] Fetched', merged.length, 'live DEX signatures');
        } catch (e) { console.warn('[TX] Fetch error:', e.message); }
        finally { txFetchInFlight = false; }
    }

    for (var i = 0; i < txRowCount; i++) {
        var row = document.createElement('div');
        row.className = 'tx-matrix-row';
        var idxDiv = document.createElement('div');
        idxDiv.className = 'tx-row-idx';
        idxDiv.textContent = i.toString(16).toUpperCase().padStart(2, '0');
        var dataDiv = document.createElement('div');
        dataDiv.className = 'tx-row-data';
        dataDiv.id = 'tx-data-' + i;
        dataDiv.textContent = '-- -- -- --';
        var statusDiv = document.createElement('div');
        statusDiv.className = 'tx-row-status';
        statusDiv.id = 'tx-status-' + i;
        statusDiv.textContent = '--';
        row.appendChild(idxDiv);
        row.appendChild(dataDiv);
        row.appendChild(statusDiv);
        txMatrixContainer.insertBefore(row, txScanner);
    }

    var currentTxRow = 0;
    var txCycleIdx = 0;

    function updateTxMatrix() {
        txScanner.style.top = currentTxRow * 22 + 'px';
        var dataEl = document.getElementById('tx-data-' + currentTxRow);
        var statusEl = document.getElementById('tx-status-' + currentTxRow);
        if (liveTxBuffer.length === 0) {
            dataEl.textContent = 'FETCHING LIVE TX...';
            currentTxRow = (currentTxRow + 1) % txRowCount;
            return;
        }
        var entry = liveTxBuffer[txCycleIdx % liveTxBuffer.length];
        dataEl.textContent = entry.label + ' \u00b7 ' + entry.sig;
        dataEl.classList.add('active');
        statusEl.textContent = entry.ok ? 'OK' : 'FL';
        statusEl.className = 'tx-row-status ' + (entry.ok ? 'confirmed' : '');
        var prevRow = currentTxRow === 0 ? txRowCount - 1 : currentTxRow - 1;
        document.getElementById('tx-data-' + prevRow).classList.remove('active');
        currentTxRow = (currentTxRow + 1) % txRowCount;
        txCycleIdx++;
    }

    fetchLiveTxSignatures();
    setInterval(updateTxMatrix, 150);
    setInterval(fetchLiveTxSignatures, 10000);
})();

// Wave selector
function setWave(el) {
    document.querySelectorAll('.wave-opt').forEach(function(d) { d.classList.remove('active'); });
    el.classList.add('active');
}

// Slider drag
function startDrag(e, valId, min, max, precision) {
    var container = e.currentTarget;
    var thumb = container.querySelector('.slider-thumb-widget');
    var output = document.getElementById(valId);
    function onMove(moveEvent) {
        var rect = container.getBoundingClientRect();
        var x = moveEvent.clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        var pct = x / rect.width;
        thumb.style.left = (pct * 100) + '%';
        var val = min + (pct * (max - min));
        output.textContent = val.toFixed(precision);
    }
    function onUp() { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    onMove(e);
}

// --- Trade Data & Inline Expansion ---
var TRADES = {};
var AGENTS = ['Momentum', 'Mean Rev', 'Sentiment', 'Risk', 'Arbitrage'];
var expandedTradeId = null;

var TOKEN_LOGOS = {
    SOL: 'assets/logos/solana-sol-logo.png', RAY: 'assets/logos/raydium-ray-logo.svg',
    JUP: 'assets/logos/jupiter-ag-jup-logo.svg', BONK: 'assets/logos/bonk1-bonk-logo.svg',
    ORCA: 'assets/logos/orca-orca-logo.svg', DRIFT: '', USDC: 'assets/logos/usd-coin-usdc-logo.svg',
};

function fmtTradePrice(p) {
    if (p == null) return '\u2014';
    if (p >= 1000) return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (p >= 1) return '$' + p.toFixed(2);
    if (p >= 0.01) return '$' + p.toFixed(4);
    return '$' + p.toFixed(6);
}

function mapExecEntry(entry, idx) {
    var mint = entry.token_mint || '';
    var sym = entry.token || mint.slice(0, 6);
    if (typeof KNOWN_MINTS !== 'undefined' && KNOWN_MINTS[mint]) sym = KNOWN_MINTS[mint].sym;
    var pair = sym + '/USDC';
    var isBuy = (entry.direction || '').toLowerCase() !== 'sell';
    var dir = isBuy ? 'LONG' : 'SHORT';
    var dirClass = isBuy ? 'text-green' : 'text-red';
    var status = entry.status || 'unknown';
    var statusClass = status === 'executed' ? 'text-green' : status === 'failed' ? 'text-red' : 'text-dim';
    var strategy = entry.strategy || '\u2014';
    var ts = entry.timestamp ? new Date(entry.timestamp * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC' : '\u2014';
    var txHash = entry.tx_hash || mint.slice(0, 4) + '...' + mint.slice(-4);
    var sizeUsd = entry.trade_size_usd || 0;
    return { pair: pair, dir: dir, dirClass: dirClass, entry: fmtTradePrice(entry.price_usd || sizeUsd), current: '\u2014', size: fmtTradePrice(sizeUsd), leverage: '1x', pnl: '\u2014', pnlPct: '\u2014', pnlClass: 'text-dim', liq: 'N/A', tp: '\u2014', sl: '\u2014', votes: [1,1,1,1,1], decision: status.toUpperCase(), decClass: statusClass, opened: ts, regime: '\u2014', initiator: strategy, tx: txHash, _sym: sym, _logo: TOKEN_LOGOS[sym] || '', _status: status, _statusClass: statusClass, _strategy: strategy, _sizeUsd: sizeUsd };
}

function renderTradeRows(entries) {
    var container = document.getElementById('trades-container');
    if (!container) return;
    if (typeof collapseTradeExpand === 'function') collapseTradeExpand();
    var header = container.querySelector('.trade-row-header');
    container.textContent = '';
    if (header) container.appendChild(header);
    if (!entries || entries.length === 0) {
        var empty = document.createElement('div');
        empty.style.cssText = 'text-align:center;padding:2rem;color:var(--dim);font-size:0.75rem;';
        empty.textContent = 'No active trades';
        container.appendChild(empty);
        return;
    }
    TRADES = {};
    entries.forEach(function(entry, i) {
        var id = 'trade' + (i + 1);
        var t = mapExecEntry(entry, i);
        TRADES[id] = t;
        var row = document.createElement('div');
        row.className = 'trade-row';
        row.setAttribute('data-trade', id);
        row.onclick = function() { toggleTradeExpand(id, row); };

        var pairCol = document.createElement('div');
        var pairDiv = document.createElement('div');
        pairDiv.className = 'trade-pair';
        if (t._logo) {
            var img = document.createElement('img');
            img.src = t._logo;
            img.className = 'trade-pair-logo';
            img.alt = t._sym;
            pairDiv.appendChild(img);
        }
        pairDiv.appendChild(document.createTextNode(t.pair));
        var typeDiv = document.createElement('div');
        typeDiv.className = 'trade-type ' + (t.dir === 'LONG' ? 'long' : 'short');
        typeDiv.textContent = t.dir;
        pairCol.appendChild(pairDiv);
        pairCol.appendChild(typeDiv);

        var sizeCol = document.createElement('div');
        sizeCol.textContent = t.size;
        var statusCol = document.createElement('div');
        statusCol.className = t._statusClass;
        statusCol.textContent = t._status.toUpperCase();
        var stratCol = document.createElement('div');
        stratCol.textContent = t._strategy;

        row.appendChild(pairCol);
        row.appendChild(sizeCol);
        row.appendChild(statusCol);
        row.appendChild(stratCol);
        container.appendChild(row);
    });
}

async function loadActiveTrades() {
    var entries = null;
    try {
        if (typeof CortexAPI !== 'undefined') {
            var data = await CortexAPI.get('/execution/log?limit=20');
            if (data && data.entries && data.entries.length > 0) entries = data.entries;
        }
    } catch (e) { console.warn('[TRADES] API fetch failed:', e.message); }
    if (!entries) {
        var container = document.getElementById('tradeTableBody') || document.querySelector('.trade-table-body');
        if (container) {
            var msg = document.createElement('div');
            msg.style.cssText = 'padding:1.5rem;text-align:center;color:var(--dim);font-size:0.6rem;font-family:var(--font-mono)';
            msg.textContent = 'No active trades \u2014 waiting for API';
            container.textContent = '';
            container.appendChild(msg);
        }
        return;
    }
    renderTradeRows(entries);
}

loadActiveTrades();
setInterval(loadActiveTrades, 30000);

function buildTradeDetail(t) {
    var detail = document.createElement('div');
    detail.className = 'trade-expand-inner';
    // Build using DOM for safety — but this is complex UI, keep template approach for readability
    // The original code used innerHTML extensively; preserving exact behavior
    var closeBtn = document.createElement('button');
    closeBtn.className = 'trade-expand-close';
    closeBtn.textContent = '\u00d7';
    closeBtn.onclick = function(e) { collapseTradeExpand(e); };
    detail.appendChild(closeBtn);

    var grid = document.createElement('div');
    grid.className = 'trade-detail-grid';

    function makeSection(title, rows) {
        var sec = document.createElement('div');
        sec.className = 'detail-section';
        var h = document.createElement('div');
        h.className = 'detail-section-title';
        h.textContent = title;
        sec.appendChild(h);
        rows.forEach(function(r) {
            var row = document.createElement('div');
            row.className = 'detail-row';
            var lbl = document.createElement('span');
            lbl.className = 'detail-label';
            lbl.textContent = r[0];
            var val = document.createElement('span');
            if (r[2]) val.className = r[2];
            val.textContent = r[1];
            row.appendChild(lbl);
            row.appendChild(val);
            sec.appendChild(row);
        });
        return sec;
    }

    var left = document.createElement('div');
    left.appendChild(makeSection('Position Info', [
        ['Direction', t.dir, t.dirClass], ['Entry Price', t.entry], ['Current Price', t.current], ['Size', t.size], ['Leverage', t.leverage]
    ]));
    left.appendChild(makeSection('P&L Analysis', [
        ['Unrealized P&L', t.pnl + ' (' + t.pnlPct + ')', t.pnlClass], ['Liquidation Price', t.liq], ['Take Profit', t.tp], ['Stop Loss', t.sl]
    ]));

    var right = document.createElement('div');
    var consensusSec = makeSection('Agent Consensus', [['Decision', t.decision, t.decClass]]);
    var cGrid = document.createElement('div');
    cGrid.className = 'consensus-grid';
    t.votes.forEach(function(v, i) {
        var vote = document.createElement('div');
        vote.className = 'consensus-vote ' + (v ? 'approve' : 'reject');
        vote.textContent = v ? 'APPROVE' : 'REJECT';
        var agent = document.createElement('div');
        agent.className = 'vote-agent';
        agent.textContent = AGENTS[i];
        vote.appendChild(agent);
        cGrid.appendChild(vote);
    });
    consensusSec.appendChild(cGrid);
    right.appendChild(consensusSec);
    right.appendChild(makeSection('Trade Metadata', [
        ['Opened', t.opened], ['Regime at Entry', t.regime], ['Initiator', t.initiator], ['TX Hash', t.tx]
    ]));

    grid.appendChild(left);
    grid.appendChild(right);
    detail.appendChild(grid);

    var actions = document.createElement('div');
    actions.className = 'trade-expand-actions';
    var modBtn = document.createElement('button');
    modBtn.className = 'btn';
    modBtn.textContent = 'Modify Position';
    var closeP = document.createElement('button');
    closeP.className = 'btn';
    closeP.style.cssText = 'border-color:var(--red);color:var(--red);';
    closeP.textContent = 'Close Position';
    actions.appendChild(modBtn);
    actions.appendChild(closeP);
    detail.appendChild(actions);

    return detail;
}

function collapseTradeExpand(e) {
    if (e && e.stopPropagation) e.stopPropagation();
    var existing = document.querySelector('.trade-expand');
    if (existing) { existing.classList.remove('open'); setTimeout(function() { existing.remove(); }, 300); }
    document.querySelectorAll('.trade-row.expanded').forEach(function(r) { r.classList.remove('expanded'); });
    expandedTradeId = null;
}

function toggleTradeExpand(tradeId, rowEl) {
    if (expandedTradeId === tradeId) { collapseTradeExpand(); return; }
    collapseTradeExpand();
    var trade = TRADES[tradeId];
    if (!trade) return;
    expandedTradeId = tradeId;
    rowEl.classList.add('expanded');
    var expandDiv = document.createElement('div');
    expandDiv.className = 'trade-expand';
    expandDiv.appendChild(buildTradeDetail(trade));
    rowEl.after(expandDiv);
    requestAnimationFrame(function() { requestAnimationFrame(function() { expandDiv.classList.add('open'); }); });
}

// --- Agent Data & Inline Expansion ---
var AGENTS_DATA = {
    momentum: { name: 'Momentum Agent', status: '\u2014', statusClass: 'text-dim', signal: '\u2014', signalClass: 'text-dim', confidence: '\u2014', lastUpdate: '\u2014', winRate: '\u2014', totalTrades: '\u2014', avgReturn: '\u2014', avgReturnClass: 'text-dim', sharpe: '\u2014', lookback: '20 candles', threshold: '1.5 std dev', maxPos: '15% portfolio', regimeAdj: '\u2014', regimeAdjClass: 'text-dim', analysis: 'Waiting for live analysis...' },
    meanrev: { name: 'Mean Reversion Agent', status: '\u2014', statusClass: 'text-dim', signal: '\u2014', signalClass: 'text-dim', confidence: '\u2014', lastUpdate: '\u2014', winRate: '\u2014', totalTrades: '\u2014', avgReturn: '\u2014', avgReturnClass: 'text-dim', sharpe: '\u2014', lookback: '50 candles', threshold: '2.0 std dev', maxPos: '12% portfolio', regimeAdj: '\u2014', regimeAdjClass: 'text-dim', analysis: 'Waiting for live analysis...' },
    sentiment: { name: 'Sentiment Agent', status: '\u2014', statusClass: 'text-dim', signal: '\u2014', signalClass: 'text-dim', confidence: '\u2014', lastUpdate: '\u2014', winRate: '\u2014', totalTrades: '\u2014', avgReturn: '\u2014', avgReturnClass: 'text-dim', sharpe: '\u2014', lookback: '24h rolling', threshold: '0.6 sentiment score', maxPos: '10% portfolio', regimeAdj: '\u2014', regimeAdjClass: 'text-dim', analysis: 'Waiting for live analysis...' },
    risk: { name: 'Risk Agent', status: '\u2014', statusClass: 'text-dim', signal: '\u2014', signalClass: 'text-dim', confidence: '\u2014', lastUpdate: '\u2014', winRate: '\u2014', totalTrades: '\u2014', avgReturn: '\u2014', avgReturnClass: 'text-dim', sharpe: '\u2014', lookback: 'Real-time', threshold: '5% max drawdown', maxPos: '20% portfolio', regimeAdj: '\u2014', regimeAdjClass: 'text-dim', analysis: 'Waiting for live analysis...' },
    arbitrage: { name: 'Arbitrage Agent', status: '\u2014', statusClass: 'text-dim', signal: '\u2014', signalClass: 'text-dim', confidence: '\u2014', lastUpdate: '\u2014', winRate: '\u2014', totalTrades: '\u2014', avgReturn: '\u2014', avgReturnClass: 'text-dim', sharpe: '\u2014', lookback: 'Real-time', threshold: '0.15% spread', maxPos: '8% portfolio', regimeAdj: '\u2014', regimeAdjClass: 'text-dim', analysis: 'Waiting for live analysis...' },
};

var expandedAgentId = null;

function buildAgentDetail(a) {
    var outer = document.createElement('div');
    outer.className = 'agent-expand-inner';
    var closeBtn = document.createElement('button');
    closeBtn.className = 'agent-expand-close';
    closeBtn.textContent = '\u00d7';
    closeBtn.onclick = function(e) { collapseAgentExpand(e); };
    outer.appendChild(closeBtn);

    var grid = document.createElement('div');
    grid.className = 'trade-detail-grid';

    function makeSection(title, rows) {
        var sec = document.createElement('div');
        sec.className = 'detail-section';
        var h = document.createElement('div');
        h.className = 'detail-section-title';
        h.textContent = title;
        sec.appendChild(h);
        rows.forEach(function(r) {
            var row = document.createElement('div');
            row.className = 'detail-row';
            var lbl = document.createElement('span');
            lbl.className = 'detail-label';
            lbl.textContent = r[0];
            var val = document.createElement('span');
            if (r[2]) val.className = r[2];
            val.textContent = r[1];
            row.appendChild(lbl);
            row.appendChild(val);
            sec.appendChild(row);
        });
        return sec;
    }

    var left = document.createElement('div');
    left.appendChild(makeSection('Agent Status', [['Status', a.status, a.statusClass], ['Current Signal', a.signal, a.signalClass], ['Confidence', a.confidence], ['Last Update', a.lastUpdate]]));
    left.appendChild(makeSection('Performance (30D)', [['Win Rate', a.winRate], ['Total Trades', a.totalTrades], ['Avg Return', a.avgReturn, a.avgReturnClass], ['Sharpe Ratio', a.sharpe]]));

    var right = document.createElement('div');
    right.appendChild(makeSection('Strategy Parameters', [['Lookback Period', a.lookback], ['Threshold', a.threshold], ['Max Position Size', a.maxPos], ['Regime Adjustment', a.regimeAdj, a.regimeAdjClass]]));
    var analysisSec = document.createElement('div');
    analysisSec.className = 'detail-section';
    var analysisTitle = document.createElement('div');
    analysisTitle.className = 'detail-section-title';
    analysisTitle.textContent = 'Current Analysis';
    var analysisBody = document.createElement('div');
    analysisBody.style.cssText = 'font-size:0.75rem;color:var(--dim);line-height:1.5;';
    analysisBody.textContent = a.analysis;
    analysisSec.appendChild(analysisTitle);
    analysisSec.appendChild(analysisBody);
    right.appendChild(analysisSec);

    grid.appendChild(left);
    grid.appendChild(right);
    outer.appendChild(grid);
    return outer;
}

function collapseAgentExpand(e) {
    if (e && e.stopPropagation) e.stopPropagation();
    if (!expandedAgentId) return;
    var prev = document.querySelector('.agent-card.expanded');
    if (prev) prev.classList.remove('expanded');
    var existing = document.querySelector('.agent-expand');
    if (existing) { existing.classList.remove('open'); existing.addEventListener('transitionend', function() { existing.remove(); }, { once: true }); }
    expandedAgentId = null;
}

function toggleAgentExpand(agentId, cardEl) {
    if (expandedAgentId === agentId) { collapseAgentExpand(); return; }
    collapseAgentExpand();
    var agent = AGENTS_DATA[agentId];
    if (!agent) return;
    expandedAgentId = agentId;
    cardEl.classList.add('expanded');
    var expandDiv = document.createElement('div');
    expandDiv.className = 'agent-expand';
    expandDiv.appendChild(buildAgentDetail(agent));
    cardEl.after(expandDiv);
    requestAnimationFrame(function() { requestAnimationFrame(function() { expandDiv.classList.add('open'); }); });
}

function closeModal(modalId) { document.getElementById(modalId).classList.remove('active'); }

document.querySelectorAll('.modal-overlay').forEach(function(overlay) {
    overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.classList.remove('active'); });
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        collapseTradeExpand();
        if (typeof collapseExpand === 'function') collapseExpand();
        collapseAgentExpand();
        document.querySelectorAll('.modal-overlay').forEach(function(overlay) { overlay.classList.remove('active'); });
    }
});

function showTab(tabName) {
    document.querySelectorAll('.tab').forEach(function(tab) { tab.classList.remove('active'); });
    if (event && event.target) event.target.classList.add('active');
}

// Trade mode selection
function applyTradeMode(mode) {
    document.querySelectorAll('.mode-option').forEach(function(o) { o.classList.toggle('active', o.getAttribute('data-mode') === mode); });
}

document.querySelectorAll('.mode-option').forEach(function(option) {
    option.addEventListener('click', async function() {
        var mode = option.getAttribute('data-mode');
        if (!mode) return;
        applyTradeMode(mode);
        localStorage.setItem('cortex_trade_mode', mode);
        var result = await CortexAPI.post('/strategies/trade-mode', { mode: mode });
        if (!result) showToast('Trade mode saved locally \u2014 backend unreachable', 'warning');
        else showToast('Trade mode set to ' + mode, 'success');
    });
});

(async function loadTradeMode() {
    var data = await CortexAPI.get('/strategies/trade-mode');
    if (data && data.mode) { applyTradeMode(data.mode); localStorage.setItem('cortex_trade_mode', data.mode); }
    else { var saved = localStorage.getItem('cortex_trade_mode'); if (saved) applyTradeMode(saved); }
})();

// === STRATEGY CONTROL PANEL ===
var STRATEGIES = [];

function switchStrategy(idx, tabEl) {
    document.querySelectorAll('.strat-tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.strat-body').forEach(function(b) { b.classList.remove('active'); });
    tabEl.classList.add('active');
    document.getElementById('stratBody' + idx).classList.add('active');
}

function renderStrategyPanel(idx) {
    var s = STRATEGIES[idx];
    var el = document.getElementById('stratBody' + idx);
    if (!el) return;
    el.textContent = '';

    // Build DOM elements instead of innerHTML
    var alloc = document.createElement('div');
    alloc.className = 'strat-alloc';
    var allocLabel = document.createElement('div');
    allocLabel.className = 'strat-alloc-label';
    var allocSpan1 = document.createElement('span');
    allocSpan1.textContent = 'Capital Allocation';
    var allocSpan2 = document.createElement('span');
    allocSpan2.textContent = s.allocation + '%';
    allocLabel.appendChild(allocSpan1);
    allocLabel.appendChild(allocSpan2);
    var allocBar = document.createElement('div');
    allocBar.className = 'strat-alloc-bar';
    var allocFill = document.createElement('div');
    allocFill.className = 'strat-alloc-fill';
    allocFill.style.width = s.allocation + '%';
    allocBar.appendChild(allocFill);
    alloc.appendChild(allocLabel);
    alloc.appendChild(allocBar);
    el.appendChild(alloc);

    var statusRow = document.createElement('div');
    statusRow.className = 'strat-status-row';
    var badge = document.createElement('span');
    badge.className = 'strat-status-badge ' + s.status;
    badge.textContent = s.status.toUpperCase();
    var toggle = document.createElement('div');
    toggle.className = 'strat-toggle ' + (s.enabled ? 'on' : '');
    toggle.onclick = function() { toggleStrategy(idx, toggle); };
    statusRow.appendChild(badge);
    statusRow.appendChild(toggle);
    el.appendChild(statusRow);

    var metrics = document.createElement('div');
    metrics.className = 'strat-metrics';
    (s.metrics || []).forEach(function(m) {
        var metric = document.createElement('div');
        metric.className = 'strat-metric';
        var mLabel = document.createElement('div');
        mLabel.className = 'strat-metric-label';
        mLabel.textContent = m.label;
        var mValue = document.createElement('div');
        mValue.className = 'strat-metric-value' + (m.value.startsWith('+') ? ' text-green' : m.value.startsWith('-') ? ' text-red' : '');
        mValue.textContent = m.value;
        metric.appendChild(mLabel);
        metric.appendChild(mValue);
        metrics.appendChild(metric);
    });
    el.appendChild(metrics);

    var params = document.createElement('div');
    params.className = 'strat-params';
    (s.params || []).forEach(function(p) {
        var row = document.createElement('div');
        row.className = 'strat-param-row';
        var key = document.createElement('span');
        key.className = 'strat-param-key';
        key.textContent = p.key;
        var val = document.createElement('span');
        val.className = 'strat-param-val';
        val.textContent = p.val;
        row.appendChild(key);
        row.appendChild(val);
        params.appendChild(row);
    });
    el.appendChild(params);

    function addCriteria(title, items) {
        var crit = document.createElement('div');
        crit.className = 'strat-criteria';
        var cTitle = document.createElement('div');
        cTitle.className = 'strat-criteria-title';
        cTitle.textContent = title;
        crit.appendChild(cTitle);
        (items || []).forEach(function(c) {
            var item = document.createElement('div');
            item.className = 'strat-criteria-item';
            item.textContent = c;
            crit.appendChild(item);
        });
        el.appendChild(crit);
    }
    addCriteria('Entry Criteria', s.entry);
    addCriteria('Exit Triggers', s.exit);
}

async function toggleStrategy(idx, toggleEl) {
    var strat = STRATEGIES[idx];
    if (!strat) return;
    strat.enabled = !strat.enabled;
    strat.status = strat.enabled ? 'running' : 'paused';
    toggleEl.classList.toggle('on', strat.enabled);
    renderStrategyPanel(idx);
    var active = STRATEGIES.filter(function(s) { return s.enabled; }).length;
    document.getElementById('stratActiveCount').textContent = active + ' ACTIVE';
    var key = strat.key || strat.name;
    var result = await CortexAPI.post('/strategies/' + encodeURIComponent(key) + '/toggle', {});
    if (!result) {
        strat.enabled = !strat.enabled;
        strat.status = strat.enabled ? 'running' : 'paused';
        toggleEl.classList.toggle('on', strat.enabled);
        renderStrategyPanel(idx);
        var active2 = STRATEGIES.filter(function(s) { return s.enabled; }).length;
        document.getElementById('stratActiveCount').textContent = active2 + ' ACTIVE';
        showToast('Backend unreachable \u2014 strategy toggle not saved', 'warning');
    } else showToast(strat.name + ' ' + (strat.enabled ? 'enabled' : 'disabled'), 'success');
}

function buildStrategyTabs(strategies) {
    var tabsEl = document.getElementById('stratTabs');
    var bodiesEl = document.getElementById('stratBodies');
    tabsEl.textContent = '';
    bodiesEl.textContent = '';
    strategies.forEach(function(s, i) {
        var tab = document.createElement('div');
        tab.className = 'strat-tab' + (i === 0 ? ' active' : '');
        tab.setAttribute('onclick', 'switchStrategy(' + i + ', this)');
        tab.textContent = s.name;
        var pct = document.createElement('span');
        pct.className = 'strat-tab-pct';
        pct.textContent = s.allocation + '%';
        tab.appendChild(pct);
        tabsEl.appendChild(tab);
        var body = document.createElement('div');
        body.id = 'stratBody' + i;
        body.className = 'strat-body' + (i === 0 ? ' active' : '');
        bodiesEl.appendChild(body);
    });
}

(async function loadStrategies() {
    var data = await CortexAPI.get('/strategies/config');
    if (data && Array.isArray(data.strategies) && data.strategies.length > 0) {
        STRATEGIES = data.strategies;
        document.getElementById('stratActiveCount').textContent = (data.active_count || STRATEGIES.length) + ' ACTIVE';
    } else {
        STRATEGIES = [];
        document.getElementById('stratActiveCount').textContent = '\u2014 OFFLINE';
        showToast('Backend unreachable \u2014 strategy config unavailable', 'warning');
    }
    if (STRATEGIES.length > 0) {
        buildStrategyTabs(STRATEGIES);
        STRATEGIES.forEach(function(_, i) { renderStrategyPanel(i); });
    }
})();

document.querySelectorAll('.time-filter').forEach(function(filter) {
    filter.addEventListener('click', function() {
        document.querySelectorAll('.time-filter').forEach(function(f) { f.classList.remove('active'); });
        filter.classList.add('active');
    });
});

document.querySelectorAll('.nav-item:not(#walletNav)').forEach(function(item) {
    item.addEventListener('click', function() {
        document.querySelectorAll('.nav-item:not(#walletNav)').forEach(function(i) { i.classList.remove('active'); });
        item.classList.add('active');
    });
});
