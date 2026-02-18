// Guardian API Service — connects admin dashboard to live Cortex Guardian API
// Uses CortexAPI client (api.js) for all HTTP and SSE communication.
// Data rendered here comes exclusively from the trusted Cortex backend API.

const GUARDIAN_POLL_MS = 15000;
const GUARDIAN_RETRY_MS = 5000;

let guardianState = {
    connected: false,
    lastAssess: null,
    kellyStats: null,
    circuitBreakers: null,
    recentDebates: null,
    debateStats: null,
    sseConnected: false,
    sseEvents: [],
    error: null,
};

let guardianTimers = { kelly: null, breakers: null, debates: null };
let guardianSSE = null;

// --- Data fetchers (via CortexAPI) ---

async function fetchKellyStats() {
    var data = await CortexAPI.get('/guardian/kelly-stats');
    if (data) {
        guardianState.kellyStats = data;
        guardianState.error = null;
        renderKellyStats();
    }
}

async function fetchCircuitBreakers() {
    var data = await CortexAPI.get('/guardian/circuit-breakers');
    if (data) {
        guardianState.circuitBreakers = data;
        guardianState.error = null;
        renderCircuitBreakers();
    }
}

async function fetchRecentDebates() {
    var data = await CortexAPI.get('/guardian/debates/recent?limit=5');
    if (data) {
        guardianState.recentDebates = data;
        guardianState.error = null;
        renderRecentDebates();
    }
}

async function fetchDebateStats() {
    var data = await CortexAPI.get('/guardian/debates/stats?hours=24');
    if (data) {
        guardianState.debateStats = data;
        guardianState.error = null;
        renderDebateStats();
    }
}

async function runGuardianAssess(token, tradeSize, direction, strategy) {
    var body = {
        token: token || 'SOL',
        trade_size_usd: tradeSize || 10000,
        direction: direction || 'long',
        strategy: strategy || '',
        run_debate: false,
    };
    var data = await CortexAPI.post('/guardian/assess', body);
    if (data) {
        guardianState.lastAssess = data;
        guardianState.error = null;
        renderGuardianAssess();
        return data;
    }
    var msg = 'Guardian assess request failed';
    guardianState.error = msg;
    renderGuardianError(msg);
    return null;
}

// --- SSE Stream (via CortexAPI.sseConnect) ---

function connectGuardianSSE() {
    if (guardianSSE) {
        guardianSSE.close();
        guardianSSE = null;
    }

    guardianSSE = CortexAPI.sseConnect(
        '/guardian/stream',
        function (data) {
            guardianState.sseEvents.unshift(data);
            if (guardianState.sseEvents.length > 50) guardianState.sseEvents.length = 50;
            guardianState.lastAssess = data;
            onGuardianSSEEvent(data);
        },
        function () {
            guardianState.sseConnected = false;
            updateGuardianSSEStatus(false);
            if (guardianSSE) { guardianSSE.close(); guardianSSE = null; }
            setTimeout(connectGuardianSSE, GUARDIAN_RETRY_MS);
        }
    );

    guardianSSE.onopen = function () {
        guardianState.sseConnected = true;
        guardianState.error = null;
        updateGuardianSSEStatus(true);
        console.log('[GUARDIAN] SSE connected');
    };
}

function disconnectGuardianSSE() {
    if (guardianSSE) {
        guardianSSE.close();
        guardianSSE = null;
    }
    guardianState.sseConnected = false;
    updateGuardianSSEStatus(false);
}

// --- Renderers (safe DOM methods) ---

function updateGuardianSSEStatus(connected) {
    var el = document.getElementById('guardianSSEStatus');
    if (!el) return;
    el.textContent = connected ? 'LIVE' : 'DISCONNECTED';
    el.className = connected ? 'text-green' : 'text-red';
}

function onGuardianSSEEvent(data) {
    renderGuardianLiveScore(data);
    updateExecutionMonitorFromGuardian(data);
}

function guardianSetText(id, text, color) {
    var el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (color) el.style.color = color;
}

function guardianScoreColor(score) {
    if (score >= 75) return 'var(--red)';
    if (score >= 50) return '#cc8800';
    return 'var(--green)';
}

function renderGuardianLiveScore(data) {
    var score = data.risk_score != null ? data.risk_score : 0;
    guardianSetText('guardianRiskScore', score.toFixed(1), guardianScoreColor(score));

    var approved = data.approved;
    var el = document.getElementById('guardianApproved');
    if (el) {
        el.textContent = approved ? 'APPROVED' : 'VETOED';
        el.className = approved ? 'text-green' : 'text-red';
    }

    guardianSetText('guardianToken', data.token || '\u2014');
    guardianSetText('guardianRegime', data.regime_state || '\u2014');

    var vetosEl = document.getElementById('guardianVetos');
    if (vetosEl) {
        var reasons = data.veto_reasons || [];
        vetosEl.textContent = reasons.length ? reasons.join(', ') : 'None';
        vetosEl.style.color = reasons.length ? 'var(--red)' : 'var(--dim)';
    }
}

function renderGuardianAssess() {
    var d = guardianState.lastAssess;
    if (!d) return;

    renderGuardianLiveScore(d);

    // Component scores — build DOM safely
    var compEl = document.getElementById('guardianComponents');
    if (compEl && d.component_scores) {
        compEl.textContent = '';
        d.component_scores.forEach(function (c) {
            var color = guardianScoreColor(c.score);
            var row = document.createElement('div');
            row.className = 'guardian-comp';

            var name = document.createElement('div');
            name.className = 'guardian-comp-name';
            name.textContent = c.component;
            row.appendChild(name);

            var bar = document.createElement('div');
            bar.className = 'guardian-comp-bar';
            var fill = document.createElement('div');
            fill.className = 'guardian-comp-fill';
            fill.style.width = c.score + '%';
            fill.style.background = color;
            bar.appendChild(fill);
            row.appendChild(bar);

            var val = document.createElement('div');
            val.className = 'guardian-comp-val';
            val.style.color = color;
            val.textContent = c.score.toFixed(1);
            row.appendChild(val);

            compEl.appendChild(row);
        });
    }

    guardianSetText('guardianSize', d.recommended_size != null
        ? '$' + d.recommended_size.toLocaleString(undefined, { maximumFractionDigits: 0 })
        : '\u2014');

    guardianSetText('guardianConfidence', d.confidence != null
        ? (d.confidence * 100).toFixed(1) + '%'
        : '\u2014');

    var cacheEl = document.getElementById('guardianCache');
    if (cacheEl) {
        cacheEl.textContent = d.from_cache ? 'CACHED' : 'FRESH';
        cacheEl.className = d.from_cache ? 'text-dim' : 'text-green';
    }
}

function guardianStatCell(label, value, className) {
    var cell = document.createElement('div');
    cell.className = 'guardian-stat';
    var lbl = document.createElement('div');
    lbl.className = 'guardian-stat-label';
    lbl.textContent = label;
    cell.appendChild(lbl);
    var val = document.createElement('div');
    val.className = 'guardian-stat-value' + (className ? ' ' + className : '');
    val.textContent = value;
    cell.appendChild(val);
    return cell;
}

function renderKellyStats() {
    var d = guardianState.kellyStats;
    if (!d) return;
    var el = document.getElementById('guardianKelly');
    if (!el) return;
    el.textContent = '';

    el.appendChild(guardianStatCell('Win Rate', ((d.win_rate || 0) * 100).toFixed(1) + '%'));
    el.appendChild(guardianStatCell('Kelly Fraction', ((d.kelly_fraction || 0) * 100).toFixed(2) + '%'));
    el.appendChild(guardianStatCell('Avg Win', (d.avg_win || 0).toFixed(2)));
    el.appendChild(guardianStatCell('Avg Loss', (d.avg_loss || 0).toFixed(2)));
    el.appendChild(guardianStatCell('Total Trades', String(d.total_trades || 0)));
    el.appendChild(guardianStatCell('Recommended Bet', ((d.recommended_bet_fraction || 0) * 100).toFixed(2) + '%'));
}

function renderCircuitBreakers() {
    var d = guardianState.circuitBreakers;
    if (!d || !d.breakers) return;
    var el = document.getElementById('guardianBreakers');
    if (!el) return;
    el.textContent = '';

    var breakers = d.breakers;
    if (!breakers.length) {
        var empty = document.createElement('div');
        empty.style.cssText = 'padding:0.5rem;color:var(--dim);font-size:0.65rem';
        empty.textContent = 'No circuit breakers configured';
        el.appendChild(empty);
        return;
    }

    breakers.forEach(function (b) {
        var isTripped = b.tripped || b.blocked;
        var color = isTripped ? 'var(--red)' : 'var(--green)';
        var status = isTripped ? 'TRIPPED' : 'OK';

        var row = document.createElement('div');
        row.className = 'guardian-breaker';

        var dot = document.createElement('div');
        dot.className = 'guardian-breaker-dot';
        dot.style.background = color;
        row.appendChild(dot);

        var nameEl = document.createElement('div');
        nameEl.className = 'guardian-breaker-name';
        nameEl.textContent = b.name || b.strategy || 'global';
        row.appendChild(nameEl);

        var statusEl = document.createElement('div');
        statusEl.className = 'guardian-breaker-status';
        statusEl.style.color = color;
        statusEl.textContent = status;
        row.appendChild(statusEl);

        el.appendChild(row);
    });
}

function renderRecentDebates() {
    var d = guardianState.recentDebates;
    if (!d || !d.transcripts) return;
    var el = document.getElementById('guardianDebates');
    if (!el) return;
    el.textContent = '';

    if (!d.transcripts.length) {
        var empty = document.createElement('div');
        empty.style.cssText = 'padding:0.5rem;color:var(--dim);font-size:0.65rem';
        empty.textContent = 'No recent debates';
        el.appendChild(empty);
        return;
    }

    d.transcripts.slice(0, 5).forEach(function (t) {
        var decision = t.final_decision || t.decision || '\u2014';
        var decColor = (decision === 'approve' || decision === 'APPROVE') ? 'var(--green)' : 'var(--red)';
        var strategy = t.strategy || '\u2014';
        var score = t.risk_score != null ? t.risk_score.toFixed(1) : '\u2014';

        var item = document.createElement('div');
        item.className = 'guardian-debate-item';

        var decEl = document.createElement('div');
        decEl.className = 'guardian-debate-decision';
        decEl.style.color = decColor;
        decEl.textContent = decision.toUpperCase();
        item.appendChild(decEl);

        var meta = document.createElement('div');
        meta.className = 'guardian-debate-meta';
        meta.textContent = strategy + ' \u00b7 Risk: ' + score;
        item.appendChild(meta);

        el.appendChild(item);
    });
}

function renderDebateStats() {
    var d = guardianState.debateStats;
    if (!d) return;
    var el = document.getElementById('guardianDebateStats');
    if (!el) return;
    el.textContent = '';

    var total = d.total_decisions || 0;
    var approved = d.approved || 0;
    var vetoed = d.vetoed || 0;
    var rate = total > 0 ? ((approved / total) * 100).toFixed(0) : '\u2014';

    el.appendChild(guardianStatCell('24h Decisions', String(total)));
    el.appendChild(guardianStatCell('Approved', String(approved), 'text-green'));
    el.appendChild(guardianStatCell('Vetoed', String(vetoed), 'text-red'));
    el.appendChild(guardianStatCell('Approval Rate', rate + '%'));
}

function renderGuardianError(msg) {
    var el = document.getElementById('guardianError');
    if (!el) return;
    el.textContent = msg || '';
    el.style.display = msg ? 'block' : 'none';
}

// Wire into the existing Execution Monitor
function updateExecutionMonitorFromGuardian(data) {
    if (!data) return;

    var approved = data.approved;
    var checks = ['emChk1', 'emChk2', 'emChk3', 'emChk4'];
    var checkResults = [
        true,
        approved,
        true,
        !(data.veto_reasons && data.veto_reasons.length > 0),
    ];

    checks.forEach(function (id, i) {
        var el = document.getElementById(id);
        if (!el) return;
        if (checkResults[i]) {
            el.textContent = '\u2713';
            el.className = 'em-check-icon pass';
        } else {
            el.textContent = '\u2717';
            el.className = 'em-check-icon fail';
        }
    });

    var step2Status = document.querySelector('#emStep2 .em-step-status');
    if (step2Status) {
        var passCount = checkResults.filter(function (v) { return v; }).length;
        if (approved) {
            step2Status.textContent = '\u2713 ' + passCount + '/' + checks.length + ' Checks';
            step2Status.className = 'em-step-status text-green';
        } else {
            step2Status.textContent = '\u2717 VETOED';
            step2Status.className = 'em-step-status text-red';
        }
    }

    var statusEl = document.getElementById('emStatus');
    if (statusEl) {
        statusEl.textContent = approved ? 'APPROVED' : 'VETOED';
        statusEl.className = approved ? 'text-green' : 'text-red';
    }
}

// --- Lifecycle ---

function startGuardianService() {
    console.log('[GUARDIAN] Starting Guardian API service');

    fetchKellyStats();
    fetchCircuitBreakers();
    fetchRecentDebates();
    fetchDebateStats();
    connectGuardianSSE();

    guardianTimers.kelly = setInterval(fetchKellyStats, GUARDIAN_POLL_MS);
    guardianTimers.breakers = setInterval(fetchCircuitBreakers, GUARDIAN_POLL_MS);
    guardianTimers.debates = setInterval(fetchRecentDebates, GUARDIAN_POLL_MS * 2);
}

function stopGuardianService() {
    Object.keys(guardianTimers).forEach(function (k) {
        if (guardianTimers[k]) clearInterval(guardianTimers[k]);
        guardianTimers[k] = null;
    });
    disconnectGuardianSSE();
}
