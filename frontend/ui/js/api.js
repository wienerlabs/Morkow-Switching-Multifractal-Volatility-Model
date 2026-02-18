/**
 * Cortex API Client — shared fetch wrapper with auth, timeout, and graceful fallback.
 */
const CortexAPI = (() => {
    function baseUrl() {
        return (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.API_BASE) || 'http://localhost:8000/api/v1';
    }

    function apiKey() {
        return (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.API_KEY) || '';
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

    return { get: get, post: post, request: request, sseConnect: sseConnect, baseUrl: baseUrl };
})();

// Global convenience alias used by ticker.js, news-service.js, regime-service.js
async function cortexFetch(path, options) {
    return CortexAPI.request(path, options);
}
