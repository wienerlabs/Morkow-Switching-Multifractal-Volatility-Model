// Cerebral page — live API connections with graceful fallback to mock data
// Uses cortexFetch from api.js + CORTEX_CONFIG from config.js

const CerebralAPI = (function() {
    const DEFAULT_TOKEN = 'SOL';
    let _token = DEFAULT_TOKEN;
    let _apiAvailable = null; // null = unknown, true/false after first check

    function setToken(t) { _token = t || DEFAULT_TOKEN; }
    function getToken() { return _token; }

    async function _fetch(path) {
        if (typeof cortexFetch !== 'function') return null;
        try {
            const data = await cortexFetch(path);
            if (data) _apiAvailable = true;
            return data;
        } catch (e) {
            console.warn('[CerebralAPI]', path, e.message);
            return null;
        }
    }

    async function _post(path, body) {
        if (typeof cortexFetch !== 'function') return null;
        try {
            const data = await cortexFetch(path, {
                method: 'POST',
                body: JSON.stringify(body),
            });
            if (data) _apiAvailable = true;
            return data;
        } catch (e) {
            console.warn('[CerebralAPI]', path, e.message);
            return null;
        }
    }

    // Check if API is reachable by hitting health endpoint
    async function checkHealth() {
        const base = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.API_BASE)
            ? CORTEX_CONFIG.API_BASE.replace('/api/v1', '')
            : '';
        try {
            const resp = await fetch(base + '/health', { signal: AbortSignal.timeout(3000) });
            _apiAvailable = resp.ok;
        } catch {
            _apiAvailable = false;
        }
        console.log('[CerebralAPI] API available:', _apiAvailable);
        return _apiAvailable;
    }

    function isAvailable() { return _apiAvailable === true; }

    // --- Regime ---
    async function fetchRegime() {
        return _fetch('/regime/transition-alert?token=' + encodeURIComponent(_token));
    }

    // --- VaR ---
    async function fetchVaR(confidence) {
        // Portfolio VaR is a POST endpoint; use _post
        return _post('/portfolio/var', {
            token: _token,
            alpha: confidence === 99.5 ? 0.005 : confidence === 99 ? 0.01 : 0.05,
        });
    }

    // --- Volatility (via regime statistics) ---
    async function fetchVolatility() {
        return _fetch('/regime/statistics?token=' + encodeURIComponent(_token));
    }

    // --- Regime LVaR ---
    async function fetchRegimeLVaR(confidence, positionValue) {
        const params = new URLSearchParams({
            token: _token,
            confidence: confidence || 95,
            position_value: positionValue || 100000,
        });
        return _fetch('/lvar/regime-var?' + params);
    }

    // --- Guardian Debate ---
    async function runDebate(tradeSize, direction, urgency) {
        return _post('/guardian/debate', {
            token: _token,
            trade_size_usd: tradeSize || 10000,
            direction: direction || 'long',
            urgency: urgency || false,
        });
    }

    async function fetchRecentDebates(limit) {
        return _fetch('/guardian/debates/recent?limit=' + (limit || 10));
    }

    async function fetchDebateStats(hours) {
        return _fetch('/guardian/debates/stats?hours=' + (hours || 24));
    }

    // --- Circuit Breakers ---
    async function fetchCircuitBreakers() {
        return _fetch('/guardian/circuit-breakers');
    }

    // --- Guardian Assess (full) ---
    async function fetchGuardianAssess(tradeSize, direction) {
        return _post('/guardian/assess', {
            token: _token,
            trade_size_usd: tradeSize || 10000,
            direction: direction || 'long',
            run_debate: false,
        });
    }

    // --- Regime History & Durations ---
    async function fetchRegimeHistory() {
        return _fetch('/regime/history?token=' + encodeURIComponent(_token));
    }

    async function fetchRegimeDurations() {
        return _fetch('/regime/durations?token=' + encodeURIComponent(_token));
    }

    async function fetchRegimeStatistics() {
        return _fetch('/regime/statistics?token=' + encodeURIComponent(_token));
    }

    async function fetchTransitionAlert(threshold) {
        return _fetch('/regime/transition-alert?token=' + encodeURIComponent(_token) +
            '&threshold=' + (threshold || 0.3));
    }

    return {
        setToken, getToken,
        checkHealth, isAvailable,
        fetchRegime, fetchVaR, fetchVolatility,
        fetchRegimeLVaR, runDebate, fetchRecentDebates, fetchDebateStats,
        fetchCircuitBreakers, fetchGuardianAssess,
        fetchRegimeHistory, fetchRegimeDurations,
        fetchRegimeStatistics, fetchTransitionAlert,
    };
})();
