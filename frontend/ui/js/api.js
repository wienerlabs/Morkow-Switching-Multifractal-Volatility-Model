/**
 * Cortex API Client — shared fetch wrapper with auth, timeout, and graceful fallback.
 * Tracks API call timestamps in localStorage for the usage heatmap.
 */
const CortexAPI = (() => {
    const USAGE_KEY = 'cortex_api_usage';
    const USAGE_MAX_AGE_MS = 31 * 24 * 3600 * 1000; // 31 days

    function baseUrl() {
        return (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.API_BASE) || 'http://localhost:8000/api/v1';
    }

    function apiKey() {
        return (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.API_KEY) || '';
    }

    function recordCall() {
        try {
            const now = Date.now();
            const cutoff = now - USAGE_MAX_AGE_MS;
            let calls = JSON.parse(localStorage.getItem(USAGE_KEY) || '[]');
            calls = calls.filter(function(t) { return t > cutoff; });
            calls.push(now);
            localStorage.setItem(USAGE_KEY, JSON.stringify(calls));
        } catch (_) { /* quota exceeded or private mode */ }
    }

    function getUsageData() {
        try {
            const now = Date.now();
            const cutoff = now - USAGE_MAX_AGE_MS;
            const calls = JSON.parse(localStorage.getItem(USAGE_KEY) || '[]');
            return calls.filter(function(t) { return t > cutoff; });
        } catch (_) { return []; }
    }

    async function request(path, opts = {}) {
        const url = baseUrl() + path;
        const headers = { 'Content-Type': 'application/json' };
        const key = apiKey();
        if (key) headers['X-API-Key'] = key;

        const controller = new AbortController();
        const timeout = opts.timeout || 10000;
        const timer = setTimeout(() => controller.abort(), timeout);

        try {
            const res = await fetch(url, {
                method: opts.method || 'GET',
                headers: { ...headers, ...(opts.headers || {}) },
                body: opts.body ? JSON.stringify(opts.body) : undefined,
                signal: controller.signal,
            });
            clearTimeout(timer);
            recordCall();
            if (!res.ok) throw new Error('HTTP ' + res.status + ': ' + res.statusText);
            return await res.json();
        } catch (err) {
            clearTimeout(timer);
            console.warn('[CortexAPI] ' + path + ' failed:', err.message);
            return null;
        }
    }

    function get(path, opts) { return request(path, Object.assign({}, opts, { method: 'GET' })); }
    function post(path, body, opts) { return request(path, Object.assign({}, opts, { method: 'POST', body: body })); }

    function sseConnect(path, onEvent, onError) {
        var url = baseUrl() + path;
        var key = apiKey();
        var fullUrl = key ? url + (url.includes('?') ? '&' : '?') + 'api_key=' + encodeURIComponent(key) : url;
        var es = new EventSource(fullUrl);
        es.onmessage = function(e) {
            try { onEvent(JSON.parse(e.data)); } catch (err) { console.warn('[SSE] Parse error:', err); }
        };
        es.onerror = function(e) {
            if (onError) onError(e);
            else console.warn('[SSE] Connection error on', path);
        };
        return es;
    }

    return { get: get, post: post, request: request, sseConnect: sseConnect, baseUrl: baseUrl, getUsageData: getUsageData };
})();

// Global convenience alias used by ticker.js, news-service.js, regime-service.js
async function cortexFetch(path, options) {
    return CortexAPI.request(path, options);
}
