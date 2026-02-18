(function () {
    var COINS = [
        { symbol: 'SOL', pair: 'SOL/USDC' },
        { symbol: 'BTC', pair: 'BTC/USDC' },
        { symbol: 'ETH', pair: 'ETH/USDC' },
        { symbol: 'RAY', pair: 'RAY/USDC' },
        { symbol: 'JUP', pair: 'JUP/USDC' },
        { symbol: 'DRIFT', pair: 'DRIFT/USDC' },
    ];

    var COINGECKO_MAP = {
        SOL: 'solana', BTC: 'bitcoin', ETH: 'ethereum',
        RAY: 'raydium', JUP: 'jupiter-exchange-solana', DRIFT: 'drift-protocol',
    };

    function fmtPrice(p) {
        if (p >= 1000) return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 0 });
        if (p >= 1) return '$' + p.toFixed(2);
        if (p >= 0.01) return '$' + p.toFixed(4);
        return '$' + p.toFixed(6);
    }

    // NOTE: innerHTML usage is safe here — values are numeric from trusted API responses,
    // not user-supplied strings. This matches the original cortexagent-ui pattern.
    function renderPrices(priceMap) {
        var el = document.getElementById('tickerContent');
        if (!el) return;
        var items = COINS.map(function (c) {
            var d = priceMap[c.symbol];
            if (!d) return '';
            var price = fmtPrice(d.price);
            var change = d.change || 0;
            var dir = change >= 0 ? 'up' : 'down';
            var sign = change >= 0 ? '+' : '';
            return '<span class="ticker-item">' + c.pair +
                ' <span class="price">' + price + '</span>' +
                ' <span class="change ' + dir + '">' + sign + change.toFixed(2) + '%</span></span>';
        }).join('');
        el.innerHTML = items + items; // eslint-disable-line no-unsanitized/property
    }

    // Static fallback when all sources fail
    var FALLBACK = {
        SOL: { price: 148.50, change: 0 },
        BTC: { price: 97200, change: 0 },
        ETH: { price: 2680, change: 0 },
        RAY: { price: 3.42, change: 0 },
        JUP: { price: 0.92, change: 0 },
        DRIFT: { price: 1.18, change: 0 },
    };

    async function fetchFromBackend() {
        if (typeof CortexAPI === 'undefined') return null;
        var symbols = COINS.map(function (c) { return c.symbol; }).join(',');
        var data = await CortexAPI.get('/oracle/prices?symbols=' + symbols);
        if (!data || !data.prices || !data.prices.length) return null;
        var map = {};
        data.prices.forEach(function (p) {
            var sym = (p.symbol || '').toUpperCase().replace(/\/USD.*/, '');
            if (sym) map[sym] = { price: p.price, change: 0 };
        });
        return Object.keys(map).length ? map : null;
    }

    async function fetchFromCoinGecko() {
        var ids = COINS.map(function (c) { return COINGECKO_MAP[c.symbol]; }).join(',');
        var key = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.COINGECKO_API_KEY) || '';
        var url = 'https://api.coingecko.com/api/v3/simple/price?ids=' + ids +
            '&vs_currencies=usd&include_24hr_change=true' +
            (key ? '&x_cg_demo_api_key=' + key : '');
        var res = await fetch(url);
        if (!res.ok) throw new Error('CoinGecko HTTP ' + res.status);
        var raw = await res.json();
        var map = {};
        COINS.forEach(function (c) {
            var d = raw[COINGECKO_MAP[c.symbol]];
            if (d) map[c.symbol] = { price: d.usd, change: d.usd_24h_change || 0 };
        });
        return Object.keys(map).length ? map : null;
    }

    async function fetchPrices() {
        try {
            // Try Cortex backend first
            var data = await fetchFromBackend();
            if (data) { renderPrices(data); return; }
        } catch (e) {
            console.warn('[TICKER] Backend error:', e.message);
        }
        try {
            // Fallback to CoinGecko
            var cgData = await fetchFromCoinGecko();
            if (cgData) { renderPrices(cgData); return; }
        } catch (e) {
            console.warn('[TICKER] CoinGecko error:', e.message);
        }
        // Static fallback
        renderPrices(FALLBACK);
    }

    if (document.getElementById('tickerContent')) {
        fetchPrices();
        setInterval(fetchPrices, 30000);
    }
})();
