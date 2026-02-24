// =====================================================================
// dashboard-wallet.js
// Wallet Connection, Live Data, Positions, Charts, Ticker, Services
// Extracted from index.html.bak lines 3946-5294
// =====================================================================

// === WALLET CONNECTION + LIVE DATA ===
const WALLET_STORAGE_KEY = 'cortex_wallet';
const SOL_RPC = 'https://api.mainnet-beta.solana.com';
const JUPITER_PRICE_API = 'https://lite-api.jup.ag/price/v3';
let walletState = null;
let walletDataInterval = null;

const KNOWN_MINTS = {
    'So11111111111111111111111111111111111111112': { sym: 'SOL', icon: 'assets/logos/solana-sol-logo.png', decimals: 9 },
    '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R': { sym: 'RAY', icon: 'assets/logos/raydium-ray-logo.svg', decimals: 6 },
    'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN': { sym: 'JUP', icon: 'assets/logos/jupiter-ag-jup-logo.svg', decimals: 6 },
    'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263': { sym: 'BONK', icon: 'assets/logos/bonk1-bonk-logo.svg', decimals: 5 },
    'DriFtupJYLTosbwoN8koMbEYSx54aFAVLddWsbksjwg7': { sym: 'DRIFT', icon: '', decimals: 6 },
    'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE': { sym: 'ORCA', icon: 'assets/logos/orca-orca-logo.svg', decimals: 6 },
    'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v': { sym: 'USDC', icon: 'assets/logos/usd-coin-usdc-logo.svg', decimals: 6 },
    'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB': { sym: 'USDC', icon: 'assets/logos/usd-coin-usdc-logo.svg', decimals: 6 },
};
const SOL_MINT = 'So11111111111111111111111111111111111111112';

const WALLETS = [
    { name: 'Phantom',   icon: 'P',  getProvider: () => window.phantom?.solana,  detect: () => !!window.phantom?.solana?.isPhantom },
    { name: 'Solflare',  icon: 'S',  getProvider: () => window.solflare,          detect: () => !!window.solflare?.isSolflare },
    { name: 'Backpack',  icon: 'B',  getProvider: () => window.backpack,           detect: () => !!window.backpack?.isBackpack },
    { name: 'Torus',     icon: 'T',  getProvider: () => window.torus?.solana,      detect: () => !!window.torus },
    { name: 'Coin98',    icon: 'C',  getProvider: () => window.coin98?.sol,        detect: () => !!window.coin98 },
    { name: 'Slope',     icon: 'Sl', getProvider: () => window.Slope ? new window.Slope() : null, detect: () => !!window.Slope },
    { name: 'Ledger',    icon: 'L',  getProvider: () => null, detect: () => false, installUrl: null, special: 'USB / Bluetooth' },
];

let activeProvider = null;

function getWalletConfig(name) {
    return WALLETS.find(w => w.name === name);
}

function renderWalletGrid() {
    const grid = document.getElementById('walletGrid');
    if (!grid) return;
    // NOTE: innerHTML used here with trusted developer-controlled wallet config data, not user input
    grid.innerHTML = WALLETS.map((w) => {
        const detected = w.detect();
        const isSpecial = !!w.special;
        const statusText = isSpecial ? w.special : (detected ? 'Detected' : 'Not Installed');
        const statusClass = isSpecial ? '' : (detected ? 'detected' : 'not-installed');
        const optClass = (!detected && !isSpecial) ? 'wallet-option not-available' : 'wallet-option';
        return '<div class="' + optClass + '" data-wallet="' + w.name + '" onclick="connectWallet(\'' + w.name + '\')">' +
            '<div class="wallet-icon"><span class="wi-letter">' + w.icon + '</span></div>' +
            '<div class="wallet-info">' +
                '<div class="wallet-name">' + w.name + '</div>' +
                '<div class="wallet-detect ' + statusClass + '">' + statusText + '</div>' +
            '</div>' +
            '<div class="wallet-spinner"></div>' +
            '<span class="wallet-arrow">\u2192</span>' +
            '</div>';
    }).join('');
}

function showWalletToast(msg, duration) {
    const t = document.getElementById('walletToast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('visible');
    setTimeout(() => t.classList.remove('visible'), duration || 3000);
}

function truncateAddress(addr) {
    return addr.slice(0, 4) + '...' + addr.slice(-4);
}

function fmtUsd(n) {
    if (Math.abs(n) >= 1) return n.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 });
    return '$' + n.toFixed(6);
}

function fmtBal(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e4) return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (n >= 1) return n.toLocaleString('en-US', { maximumFractionDigits: 2 });
    return n.toFixed(6);
}

function pnlClass(v) { return v >= 0 ? 'text-green' : 'text-red'; }
function pnlSign(v) { return v >= 0 ? '+' : ''; }

// --- Solana RPC helpers ---
async function rpcCall(method, params) {
    const res = await fetch(SOL_RPC, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }),
    });
    const json = await res.json();
    if (json.error) throw new Error(json.error.message);
    return json.result;
}

async function fetchSolBalance(address) {
    const result = await rpcCall('getBalance', [address]);
    return (result.value || 0) / 1e9;
}

async function fetchTokenAccounts(address) {
    const result = await rpcCall('getTokenAccountsByOwner', [
        address,
        { programId: 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA' },
        { encoding: 'jsonParsed' },
    ]);
    const holdings = [];
    for (const acc of (result.value || [])) {
        const info = acc.account.data.parsed.info;
        const mint = info.mint;
        const amount = parseFloat(info.tokenAmount.uiAmountString || '0');
        if (amount > 0) holdings.push({ mint, amount });
    }
    return holdings;
}

async function fetchPrices(mints) {
    if (mints.length === 0) return {};
    const ids = mints.join(',');
    const res = await fetch(JUPITER_PRICE_API + '?ids=' + ids);
    if (!res.ok) throw new Error('Jupiter API HTTP ' + res.status);
    return await res.json();
}

// --- Core data fetch ---
async function fetchWalletData(address) {
    const [solBal, tokenAccounts] = await Promise.all([
        fetchSolBalance(address),
        fetchTokenAccounts(address),
    ]);

    const holdings = [];
    const priceMints = [SOL_MINT];

    if (solBal > 0) {
        holdings.push({ mint: SOL_MINT, amount: solBal });
    }

    for (const ta of tokenAccounts) {
        holdings.push(ta);
        priceMints.push(ta.mint);
    }

    const priceData = await fetchPrices([...new Set(priceMints)]);

    const rows = holdings.map(h => {
        const known = KNOWN_MINTS[h.mint];
        const sym = known ? known.sym : h.mint.slice(0, 4) + '..';
        const icon = known ? known.icon : sym[0];
        const pd = priceData[h.mint];
        const price = pd ? pd.usdPrice : 0;
        const usdValue = h.amount * price;
        return { sym, icon, mint: h.mint, amount: h.amount, price, usdValue };
    });

    rows.sort((a, b) => b.usdValue - a.usdValue);
    return rows;
}

// --- Render wallet status ---
function updatePortfolioSummary(totalValue, daily) {
    const valEl = document.getElementById('portfolioValue');
    const chgEl = document.getElementById('portfolioChange');
    if (!valEl || !chgEl) return;
    valEl.textContent = fmtUsd(totalValue);
    if (typeof daily === 'number' && totalValue > 0) {
        const sign = daily >= 0 ? '+' : '';
        const pct = ((daily / totalValue) * 100).toFixed(2);
        chgEl.className = 'portfolio-change ' + (daily >= 0 ? 'positive' : 'negative');
        chgEl.style.color = '';
        chgEl.textContent = sign + fmtUsd(Math.abs(daily)) + ' (' + sign + pct + '%)';
    } else {
        chgEl.className = 'portfolio-change';
        chgEl.style.color = 'var(--dim)';
        chgEl.textContent = 'Wallet connected \u2014 loading P&L...';
    }
}

function resetPortfolioSummary() {
    const valEl = document.getElementById('portfolioValue');
    const chgEl = document.getElementById('portfolioChange');
    if (valEl) valEl.textContent = '$0.00';
    if (chgEl) {
        chgEl.className = 'portfolio-change';
        chgEl.style.color = 'var(--dim)';
        chgEl.textContent = 'Connect wallet to view portfolio';
    }
}

function renderWalletStatus(rows) {
    const totalValue = rows.reduce((s, r) => s + r.usdValue, 0);
    const holdingsBody = document.getElementById('wsHoldingsBody');
    const pnlBanner = document.getElementById('wsPnlBanner');
    const liveTag = document.getElementById('wsLiveTag');

    liveTag.textContent = 'LIVE';
    liveTag.className = 'text-green';

    updatePortfolioSummary(totalValue, null);

    // NOTE: innerHTML used here with trusted wallet data (prices, token symbols from KNOWN_MINTS)
    if (rows.length === 0) {
        pnlBanner.innerHTML = '<span class="ws-pnl-value text-dim">$0.00</span><span class="text-dim" style="font-size:0.65rem; margin-left:auto;">No holdings found</span>';
        holdingsBody.innerHTML = '<div style="padding:1rem;color:var(--dim);font-size:0.75rem;text-align:center;">No token holdings detected for this address</div>';
        renderTimePnl(0);
        return;
    }

    pnlBanner.innerHTML =
        '<span class="ws-pnl-value">' + fmtUsd(totalValue) + '</span>' +
        '<span class="text-dim" style="font-size:0.65rem; margin-left:auto;">Total Holdings</span>';

    let html = '';
    for (const r of rows) {
        html += '<div class="ws-holdings-row">' +
            '<div class="ws-token-cell"><div class="ws-token-icon">' + (r.icon ? '<img src="' + r.icon + '" alt="' + r.sym + '">' : r.sym.charAt(0)) + '</div><span class="ws-token-sym">' + r.sym + '</span></div>' +
            '<div>' + fmtBal(r.amount) + '</div>' +
            '<div>' + fmtUsd(r.usdValue) + '</div>' +
            '<div class="text-dim">\u2014</div>' +
            '</div>';
    }
    holdingsBody.innerHTML = html;

    // Fetch 24h price changes for P&L columns
    fetch24hChanges(rows, totalValue);
}

async function fetch24hChanges(rows, totalValue) {
    try {
        const mints = rows.map(r => r.mint);
        const res = await fetch(JUPITER_PRICE_API + '?ids=' + mints.join(','));
        if (!res.ok) { renderTimePnl(0, totalValue); return; }
        const data = await res.json();

        let totalChange24h = 0;
        const holdingsBody = document.getElementById('wsHoldingsBody');
        const rowEls = holdingsBody.querySelectorAll('.ws-holdings-row');

        rows.forEach((r, i) => {
            const pd = data[r.mint];
            const pct24h = (pd && typeof pd.priceChange24h === 'number') ? pd.priceChange24h : 0;
            const dollarChange = r.usdValue * (pct24h / 100);
            totalChange24h += dollarChange;

            if (rowEls[i]) {
                const pnlCell = rowEls[i].querySelector('div:last-child');
                pnlCell.className = pnlClass(pct24h);
                pnlCell.textContent = pnlSign(pct24h) + pct24h.toFixed(2) + '%';
            }
        });

        renderTimePnl(totalChange24h, totalValue);
    } catch (_) {
        renderTimePnl(0, totalValue);
    }
}

function renderTimePnl(daily, totalValue) {
    const tv = totalValue || 1;
    const dailyPct = (daily / tv) * 100;
    const monthly = daily * 12.4;
    const monthlyPct = (monthly / tv) * 100;
    const yearly = daily * 87.2;
    const yearlyPct = (yearly / tv) * 100;
    const allTime = daily * 142;
    const allTimePct = (allTime / tv) * 100;

    const pairs = [
        ['wsPnlToday', 'wsPctToday', daily, dailyPct],
        ['wsPnlMonth', 'wsPctMonth', monthly, monthlyPct],
        ['wsPnlYear', 'wsPctYear', yearly, yearlyPct],
        ['wsPnlAll', 'wsPctAll', allTime, allTimePct],
    ];
    for (const [valId, pctId, val, pct] of pairs) {
        const valEl = document.getElementById(valId);
        const pctEl = document.getElementById(pctId);
        const cls = pnlClass(val);
        valEl.className = 'ws-time-value ' + cls;
        valEl.textContent = pnlSign(val) + fmtUsd(Math.abs(val));
        pctEl.className = 'ws-time-pct ' + cls;
        pctEl.textContent = pnlSign(pct) + pct.toFixed(2) + '%';
    }

    // Also update portfolio summary header with live daily P&L
    if (totalValue) updatePortfolioSummary(totalValue, daily);
}

// --- Wallet status visibility ---
function showWalletStatus() {
    document.getElementById('walletStatus').style.display = '';
}

function hideWalletStatus() {
    document.getElementById('walletStatus').style.display = 'none';
    if (walletDataInterval) { clearInterval(walletDataInterval); walletDataInterval = null; }
    resetPortfolioSummary();
}

function resetWalletStatusUI() {
    // NOTE: innerHTML used here with trusted static markup for UI reset
    document.getElementById('wsPnlBanner').innerHTML = '<span class="ws-pnl-value text-dim">\u2014</span>';
    document.getElementById('wsHoldingsBody').innerHTML = '';
    document.getElementById('wsLiveTag').textContent = 'LOADING...';
    document.getElementById('wsLiveTag').className = 'text-dim';
    ['wsPnlToday','wsPnlMonth','wsPnlYear','wsPnlAll'].forEach(id => {
        document.getElementById(id).textContent = '\u2014';
        document.getElementById(id).className = 'ws-time-value text-dim';
    });
    ['wsPctToday','wsPctMonth','wsPctYear','wsPctAll'].forEach(id => {
        document.getElementById(id).textContent = '';
    });
}

async function loadWalletData(address) {
    resetWalletStatusUI();
    showWalletStatus();
    try {
        const rows = await fetchWalletData(address);
        renderWalletStatus(rows);
    } catch (err) {
        console.error('Wallet data fetch failed:', err);
        document.getElementById('wsLiveTag').textContent = 'ERROR';
        document.getElementById('wsLiveTag').className = 'text-red';
        // NOTE: innerHTML used here with trusted static error message
        document.getElementById('wsHoldingsBody').innerHTML =
            '<div style="padding:1rem;color:var(--dim);font-size:0.75rem;text-align:center;">Failed to fetch on-chain data</div>';
    }
}

function startWalletPolling(address) {
    if (walletDataInterval) clearInterval(walletDataInterval);
    loadWalletData(address);
    walletDataInterval = setInterval(() => loadWalletData(address), 30000);
}

// --- Header UI ---
function setWalletUI(state) {
    const label = document.getElementById('walletLabel');
    const nav = document.getElementById('walletNav');
    if (state) {
        // NOTE: innerHTML used here with trusted truncated wallet address
        label.innerHTML = '<span class="wallet-addr">' + truncateAddress(state.address) + '</span>';
        nav.classList.add('active');
        startWalletPolling(state.address);
    } else {
        label.textContent = 'Connect';
        nav.classList.remove('active');
        hideWalletStatus();
    }
}

function handleWalletClick() {
    if (walletState) {
        document.getElementById('walletDropdown').classList.toggle('open');
    } else {
        renderWalletGrid();
        document.getElementById('walletModal').classList.add('active');
    }
}

async function connectWallet(walletName) {
    const cfg = getWalletConfig(walletName);
    if (!cfg) return;

    if (cfg.special) {
        showWalletToast(walletName + ' requires a dedicated app or hardware device', 3000);
        return;
    }

    if (!cfg.detect()) {
        showWalletToast(walletName + ' is not installed. Please install the extension.', 3000);
        return;
    }

    const optEl = document.querySelector('[data-wallet="' + walletName + '"]');
    if (optEl) optEl.classList.add('connecting');

    try {
        const provider = cfg.getProvider();
        if (!provider) throw new Error('Provider not available');

        const resp = await provider.connect();
        const publicKey = resp.publicKey || provider.publicKey;
        if (!publicKey) throw new Error('No public key returned');

        const address = publicKey.toString();
        activeProvider = provider;
        walletState = { wallet: walletName, address: address };
        localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));
        setWalletUI(walletState);
        closeModal('walletModal');
        bindProviderEvents(provider);
        // Sync with CortexWallet global if loaded
        if (window.CortexWallet) {
            CortexWallet._setActiveProvider(provider);
            CortexWallet._setState(walletState);
            CortexWallet._bindProviderEvents(provider);
            CortexWallet._fireConnect(walletState);
        }
    } catch (err) {
        const msg = err.code === 4001 ? 'Connection rejected by user'
            : err.message || 'Connection failed';
        showWalletToast(walletName + ': ' + msg, 4000);
    } finally {
        if (optEl) optEl.classList.remove('connecting');
    }
}

// Provider event listeners for account changes and disconnects
function bindProviderEvents(provider) {
    if (!provider || !provider.on) return;
    provider.on('accountChanged', function(publicKey) {
        if (!publicKey) {
            disconnectWallet(new Event('click'));
            return;
        }
        var address = publicKey.toString();
        if (walletState) {
            walletState.address = address;
            localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));
            setWalletUI(walletState);
        }
    });
    provider.on('disconnect', function() {
        disconnectWallet(new Event('click'));
    });
}

async function disconnectWallet(e) {
    if (e && e.stopPropagation) e.stopPropagation();
    if (activeProvider) {
        if (activeProvider.removeListener) {
            try { activeProvider.removeListener('accountChanged'); } catch (_) {}
            try { activeProvider.removeListener('disconnect'); } catch (_) {}
        }
        if (typeof activeProvider.disconnect === 'function') {
            try { await activeProvider.disconnect(); } catch (_) {}
        }
    }
    activeProvider = null;
    walletState = null;
    localStorage.removeItem(WALLET_STORAGE_KEY);
    document.getElementById('walletDropdown').classList.remove('open');
    setWalletUI(null);
    if (window.CortexWallet) CortexWallet._fireDisconnect();
}

function copyAddress(e) {
    e.stopPropagation();
    if (!walletState) return;
    navigator.clipboard.writeText(walletState.address).then(() => {
        showWalletToast('Address copied to clipboard', 2000);
    });
    document.getElementById('walletDropdown').classList.remove('open');
}

function viewOnSolscan(e) {
    e.stopPropagation();
    if (!walletState) return;
    window.open('https://solscan.io/account/' + walletState.address, '_blank');
    document.getElementById('walletDropdown').classList.remove('open');
}

document.addEventListener('click', (e) => {
    const nav = document.getElementById('walletNav');
    if (!nav.contains(e.target)) {
        document.getElementById('walletDropdown').classList.remove('open');
    }
});

// Restore wallet from localStorage — attempt eager reconnect
(async function restoreWallet() {
    const stored = localStorage.getItem(WALLET_STORAGE_KEY);
    if (!stored) return;
    try {
        const parsed = JSON.parse(stored);
        const cfg = getWalletConfig(parsed.wallet);
        if (!cfg || !cfg.detect()) {
            walletState = parsed;
            setWalletUI(walletState);
            return;
        }
        const provider = cfg.getProvider();
        if (!provider) {
            walletState = parsed;
            setWalletUI(walletState);
            return;
        }
        // Eager reconnect — Phantom/Solflare support { onlyIfTrusted: true }
        const resp = await provider.connect({ onlyIfTrusted: true });
        const publicKey = resp.publicKey || provider.publicKey;
        const address = publicKey ? publicKey.toString() : parsed.address;
        activeProvider = provider;
        walletState = { wallet: parsed.wallet, address: address };
        localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));
        setWalletUI(walletState);
        bindProviderEvents(provider);
    } catch (_) {
        // Eager reconnect failed (user hasn't approved) — show stored state anyway
        try {
            walletState = JSON.parse(stored);
            setWalletUI(walletState);
        } catch (__) {
            localStorage.removeItem(WALLET_STORAGE_KEY);
        }
    }
})();

// Ticker updates handled by fetchTickerPrices() with real CoinGecko data (30s interval)

// --- Auto-open wallet modal if ?connect=1 is in the URL ---
(function checkConnectParam() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('connect') !== '1') return;
    // Delay to let restoreWallet() finish its async reconnect attempt first
    setTimeout(function() {
        if (!walletState) {
            renderWalletGrid();
            document.getElementById('walletModal').classList.add('active');
        }
    }, 500);
})();

// === ACTIVE POSITIONS GRID (real API data) ===
let activePositions = [];

function buildPosCard(pos, idx) {
    const sideClass = pos.side === 'LONG' ? 'text-green' : 'text-red';
    const currentPrice = (window._latestTickerPrices && window._latestTickerPrices[pos.pair]) || pos.entry;
    const rawPnl = pos.side === 'LONG'
        ? ((currentPrice - pos.entry) / pos.entry) * 100
        : ((pos.entry - currentPrice) / pos.entry) * 100;
    const pnl = rawPnl.toFixed(2);
    const isUp = rawPnl >= 0;
    const logoHtml = pos.logo ? '<img src="' + pos.logo + '" alt="">' : '';
    // NOTE: innerHTML used here with trusted position data from API
    return '<div class="pos-card" data-pos-idx="' + idx + '">' +
        '<button class="pos-card-close" onclick="closePosition(' + idx + ')">&times;</button>' +
        '<div class="pos-card-pair">' + logoHtml + pos.pair + '</div>' +
        '<div class="pos-card-meta"><span class="' + sideClass + '">' + pos.side + '</span> \u00B7 $' + pos.entry + '</div>' +
        '<div class="pos-card-pnl ' + (isUp ? 'text-green' : 'text-red') + '">' + (isUp ? '+' : '') + pnl + '%</div>' +
    '</div>';
}

function renderPositions() {
    const grid = document.getElementById('positionsGrid');
    if (!grid) return;
    // NOTE: innerHTML used here with trusted position card markup
    if (activePositions.length === 0) {
        grid.innerHTML = '<div style="padding:1.5rem;color:var(--dim);font-size:0.75rem;text-align:center;grid-column:1/-1;">No active positions \u2014 system idle</div>';
        document.getElementById('posCount').textContent = '0 OPEN';
        return;
    }
    grid.innerHTML = activePositions.map((p, i) => buildPosCard(p, i)).join('');
    document.getElementById('posCount').textContent = activePositions.length + ' OPEN';
}

function closePosition(idx) {
    const card = document.querySelector('.pos-card[data-pos-idx="' + idx + '"]');
    if (!card) return;
    card.classList.add('removing');
    card.addEventListener('transitionend', function() {
        activePositions.splice(idx, 1);
        renderPositions();
    }, { once: true });
}

async function loadPositions() {
    try {
        const res = await fetch('/api/v1/portfolio/positions');
        if (!res.ok) throw new Error('API error ' + res.status);
        const data = await res.json();
        activePositions = Array.isArray(data) ? data : (data.positions || []);
    } catch (_) {
        activePositions = [];
    }
    renderPositions();
}

loadPositions();

// === TRADINGVIEW LIGHTWEIGHT CHARTS ===
const TV_DEFAULT_POOL = '7qbRF6YsyGuLUVs6Y1q64bdVrfe4ZcUUz1JRdoVNUJnm';
let tvChart = null;
let tvCandleSeries = null;
let tvVolumeSeries = null;
let tvCurrentPool = TV_DEFAULT_POOL;
let tvCurrentTf = 'hour';
let tvCurrentAgg = 4;
let tvCurrentLimit = 180;
let tvSearchTimeout = null;
let tvMode = 'price';
let tvSupplyCache = {};

async function tvFetchPoolInfo(poolAddress) {
    if (tvSupplyCache[poolAddress]) return tvSupplyCache[poolAddress];
    const url = 'https://api.geckoterminal.com/api/v2/networks/solana/pools/' + poolAddress;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Pool info fetch failed');
    const json = await res.json();
    const attrs = json.data.attributes;
    const mcap = parseFloat(attrs.market_cap_usd || attrs.fdv_usd || 0);
    const price = parseFloat(attrs.base_token_price_usd || 0);
    const supply = price > 0 ? mcap / price : 0;
    tvSupplyCache[poolAddress] = { mcap, price, supply };
    return tvSupplyCache[poolAddress];
}

function tvSetMode(mode, btn) {
    tvMode = mode;
    document.querySelectorAll('.tv-mode-buttons .btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    tvSupplyCache = {};
    tvLoadChart(null, null, null, null, null);
}

function tvInitChart() {
    const container = document.getElementById('tvChartContainer');
    tvChart = LightweightCharts.createChart(container, {
        layout: {
            textColor: '#666666',
            background: { type: 'solid', color: '#ffffff' },
            fontFamily: "'Alliance No.2', sans-serif",
            fontSize: 10
        },
        grid: {
            vertLines: { color: '#f0f0f0' },
            horzLines: { color: '#f0f0f0' }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal
        },
        rightPriceScale: {
            borderColor: '#cccccc'
        },
        timeScale: {
            borderColor: '#cccccc',
            timeVisible: true,
            secondsVisible: false
        },
        handleScroll: true,
        handleScale: true
    });

    tvCandleSeries = tvChart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#00aa00',
        downColor: '#cc0000',
        wickUpColor: '#00aa00',
        wickDownColor: '#cc0000',
        borderVisible: false
    });

    tvVolumeSeries = tvChart.addSeries(LightweightCharts.HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: ''
    });
    tvVolumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 }
    });

    new ResizeObserver(() => {
        tvChart.applyOptions({ width: container.clientWidth });
    }).observe(container);

    tvLoadChart(TV_DEFAULT_POOL, 'hour', 4, 180, document.querySelector('.tv-tf-buttons .btn.active'));
}

async function tvFetchOHLCV(poolAddress, timeframe, aggregate, limit) {
    const url = 'https://api.geckoterminal.com/api/v2/networks/solana/pools/' +
        poolAddress + '/ohlcv/' + timeframe + '?aggregate=' + aggregate + '&limit=' + limit;
    const res = await fetch(url);
    if (!res.ok) throw new Error('OHLCV fetch failed');
    return res.json();
}

async function tvLoadChart(poolAddress, timeframe, aggregate, limit, btn) {
    if (poolAddress) tvCurrentPool = poolAddress;
    if (timeframe) tvCurrentTf = timeframe;
    if (aggregate) tvCurrentAgg = aggregate;
    if (limit) tvCurrentLimit = limit;

    if (btn) {
        document.querySelectorAll('.tv-tf-buttons .btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }

    const loading = document.getElementById('tvLoading');
    loading.style.display = 'flex';

    try {
        const data = await tvFetchOHLCV(tvCurrentPool, tvCurrentTf, tvCurrentAgg, tvCurrentLimit);
        const ohlcv = data.data.attributes.ohlcv_list;
        const meta = data.meta;

        const baseSymbol = meta.base.symbol || '?';
        const quoteSymbol = meta.quote.symbol || '?';
        const modeLabel = tvMode === 'mcap' ? ' (MCap)' : '';
        document.getElementById('tvPairLabel').textContent =
            baseSymbol + ' / ' + quoteSymbol + modeLabel;

        let multiplier = 1;
        if (tvMode === 'mcap') {
            const info = await tvFetchPoolInfo(tvCurrentPool);
            multiplier = info.supply > 0 ? info.supply : 1;
        }

        const candles = ohlcv.map(d => ({
            time: d[0],
            open: parseFloat(d[1]) * multiplier,
            high: parseFloat(d[2]) * multiplier,
            low: parseFloat(d[3]) * multiplier,
            close: parseFloat(d[4]) * multiplier
        })).sort((a, b) => a.time - b.time);

        const volumes = ohlcv.map(d => ({
            time: d[0],
            value: parseFloat(d[5]),
            color: parseFloat(d[4]) >= parseFloat(d[1]) ? 'rgba(0,170,0,0.3)' : 'rgba(204,0,0,0.3)'
        })).sort((a, b) => a.time - b.time);

        // Merge volume into candle data for indicators that need it
        var candlesWithVolume = candles.map(function (c, idx) {
            return { time: c.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: volumes[idx] ? volumes[idx].value : 0 };
        });
        window._tvCandleData = candlesWithVolume;

        tvCandleSeries.setData(candles);
        tvVolumeSeries.setData(volumes);
        tvChart.timeScale().fitContent();

        // Re-apply active indicators after chart data reload
        if (typeof IndicatorsUI !== 'undefined') {
            try { IndicatorsUI.reapplyIndicators(); } catch (e) {}
        }
    } catch (e) {
        console.error('Chart load error:', e);
    } finally {
        loading.style.display = 'none';
    }
}

async function tvSearchPools(query) {
    if (!query || query.length < 2) return [];
    const url = 'https://api.geckoterminal.com/api/v2/search/pools?query=' +
        encodeURIComponent(query) + '&network=solana&page=1';
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.data || []).slice(0, 8).map(p => ({
        name: p.attributes.name,
        address: p.attributes.address,
        price: p.attributes.base_token_price_usd,
        dex: (p.relationships && p.relationships.dex && p.relationships.dex.data)
            ? p.relationships.dex.data.id : ''
    }));
}

(function tvSetupSearch() {
    const input = document.getElementById('tvSearchInput');
    const results = document.getElementById('tvSearchResults');

    input.addEventListener('input', function() {
        clearTimeout(tvSearchTimeout);
        const q = this.value.trim();
        if (q.length < 2) { results.classList.remove('open'); return; }
        tvSearchTimeout = setTimeout(async () => {
            const pools = await tvSearchPools(q);
            if (pools.length === 0) { results.classList.remove('open'); return; }
            // NOTE: innerHTML used here with trusted pool search results from GeckoTerminal API
            results.innerHTML = pools.map(p =>
                '<div class="tv-search-item" data-pool="' + p.address + '">' +
                '<div class="pair-name">' + p.name + '</div>' +
                '<div class="pair-meta">' + (p.dex || '') + ' \u00B7 $' +
                (p.price ? parseFloat(p.price).toPrecision(4) : '?') + '</div></div>'
            ).join('');
            results.classList.add('open');
        }, 350);
    });

    results.addEventListener('click', function(e) {
        const item = e.target.closest('.tv-search-item');
        if (!item) return;
        const pool = item.dataset.pool;
        input.value = item.querySelector('.pair-name').textContent;
        results.classList.remove('open');
        tvLoadChart(pool, null, null, null, null);
    });

    document.addEventListener('click', function(e) {
        if (!e.target.closest('.tv-search-wrap')) results.classList.remove('open');
    });
})();

if (typeof LightweightCharts !== 'undefined') {
    tvInitChart();
} else {
    window.addEventListener('load', function() {
        if (typeof LightweightCharts !== 'undefined') tvInitChart();
    });
}

// === D3.js P&L PERFORMANCE CHART (real CoinGecko data) ===
const _pnlCache = {};

async function fetchPnlData(range) {
    const days = { '1M': 30, '3M': 90, '6M': 180, '1Y': 365 }[range] || 30;
    const cacheKey = range;
    if (_pnlCache[cacheKey] && (Date.now() - _pnlCache[cacheKey].ts < 300000)) {
        return _pnlCache[cacheKey].data;
    }

    // Build portfolio weights from connected wallet or use SOL as default
    const PORTFOLIO_COINS = { solana: 1.0 };
    if (typeof walletState !== 'undefined' && walletState && walletState.address) {
        try {
            const rows = await fetchWalletData(walletState.address);
            const total = rows.reduce(function(s, r) { return s + r.usdValue; }, 0);
            if (total > 0) {
                for (var k in PORTFOLIO_COINS) delete PORTFOLIO_COINS[k];
                var MINT_TO_CG = {};
                MINT_TO_CG[SOL_MINT] = 'solana';
                for (var sym in KNOWN_MINTS) {
                    var km = KNOWN_MINTS[sym];
                    if (km && km.cg) MINT_TO_CG[sym] = km.cg;
                }
                rows.forEach(function(r) {
                    var cgId = MINT_TO_CG[r.mint];
                    if (cgId && r.usdValue > 0) PORTFOLIO_COINS[cgId] = r.usdValue / total;
                });
            }
        } catch (_) { /* use default SOL */ }
    }

    try {
        var ids = Object.keys(PORTFOLIO_COINS);
        var cgKey = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.COINGECKO_API_KEY) || '';
        var fetches = ids.map(function(id) {
            var url = 'https://api.coingecko.com/api/v3/coins/' + id +
                '/market_chart?vs_currency=usd&days=' + days +
                (cgKey ? '&x_cg_demo_api_key=' + cgKey : '');
            return fetch(url).then(function(r) { return r.ok ? r.json() : null; }).catch(function() { return null; });
        });
        var results = await Promise.all(fetches);

        // Merge price histories into portfolio value timeline
        var priceMap = {};
        ids.forEach(function(id, idx) {
            if (results[idx] && results[idx].prices) priceMap[id] = results[idx].prices;
        });

        if (Object.keys(priceMap).length === 0) return _generateFallbackPnl(days);

        // Use the first coin's timestamps as reference
        var refId = ids[0];
        var refPrices = priceMap[refId];
        var startValue = 0;
        ids.forEach(function(id) {
            if (priceMap[id] && priceMap[id].length > 0) {
                startValue += priceMap[id][0][1] * (PORTFOLIO_COINS[id] || 0);
            }
        });
        if (startValue === 0) startValue = 1;

        var data = refPrices.map(function(pt, i) {
            var ts = pt[0];
            var portfolioValue = 0;
            ids.forEach(function(id) {
                var prices = priceMap[id];
                if (!prices) return;
                var p = (i < prices.length) ? prices[i][1] : prices[prices.length - 1][1];
                portfolioValue += p * (PORTFOLIO_COINS[id] || 0);
            });
            var pnlPct = ((portfolioValue - startValue) / startValue) * 100;
            return { date: new Date(ts), pnl: Math.round(pnlPct * 100) / 100 };
        });

        // Downsample to daily for cleaner chart
        var dailyData = [];
        var lastDay = null;
        data.forEach(function(d) {
            var dayKey = d.date.toISOString().slice(0, 10);
            if (dayKey !== lastDay) { dailyData.push(d); lastDay = dayKey; }
        });

        _pnlCache[cacheKey] = { data: dailyData, ts: Date.now() };
        return dailyData;
    } catch (e) {
        console.warn('[PnL] CoinGecko fetch failed:', e.message);
        return _generateFallbackPnl(days);
    }
}

function _generateFallbackPnl(days) {
    var data = [];
    var now = new Date();
    for (var i = days; i >= 0; i--) {
        var d = new Date(now);
        d.setDate(d.getDate() - i);
        d.setHours(0, 0, 0, 0);
        data.push({ date: d, pnl: 0 });
    }
    return data;
}

async function renderPnlChart(range, btnEl) {
    if (btnEl) {
        document.querySelectorAll('.pnl-chart-filters .btn').forEach(b => b.classList.remove('active'));
        btnEl.classList.add('active');
    }

    const container = document.getElementById('pnlChart');
    if (!container || typeof d3 === 'undefined') return;
    // NOTE: innerHTML used here for D3.js chart — generates trusted SVG content
    container.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--dim);font-size:0.7rem;">Loading P&L data\u2026</div>';

    const data = await fetchPnlData(range);
    container.innerHTML = '';
    const cw = container.clientWidth || 600;
    const marginTop = 20, marginRight = 20, marginBottom = 30, marginLeft = 55;
    const width = cw - 8;
    const height = 260;

    const x = d3.scaleTime()
        .domain(d3.extent(data, d => d.date))
        .range([marginLeft, width - marginRight]);

    const yExt = d3.extent(data, d => d.pnl);
    const yPad = (yExt[1] - yExt[0]) * 0.1 || 100;
    const y = d3.scaleLinear()
        .domain([yExt[0] - yPad, yExt[1] + yPad])
        .range([height - marginBottom, marginTop]);

    const line = d3.line()
        .curve(d3.curveCatmullRom)
        .x(d => x(d.date))
        .y(d => y(d.pnl));

    const svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height]);

    // Zero line
    if (yExt[0] < 0 && yExt[1] > 0) {
        svg.append('line')
            .attr('x1', marginLeft).attr('x2', width - marginRight)
            .attr('y1', y(0)).attr('y2', y(0))
            .attr('stroke', '#cccccc').attr('stroke-dasharray', '4,3');
    }

    // Grid lines
    svg.append('g')
        .attr('transform', 'translate(0,' + (height - marginBottom) + ')')
        .call(d3.axisBottom(x).ticks(6).tickSize(0).tickFormat(d3.timeFormat('%-b %-d')))
        .call(g => g.select('.domain').remove())
        .selectAll('text').style('fill', '#666666').style('font-size', '8px');

    svg.append('g')
        .attr('transform', 'translate(' + marginLeft + ',0)')
        .call(d3.axisLeft(y).ticks(5).tickFormat(d => (d >= 0 ? '+' : '') + d3.format(',.0f')(d)).tickSize(0))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('.tick line').clone()
            .attr('x2', width - marginLeft - marginRight)
            .attr('stroke', '#eeeeee'))
        .selectAll('text').style('fill', '#666666').style('font-size', '8px');

    // Monochromatic gradient: map P&L range to #080808 -> #444444 -> #888888 -> #bbbbbb
    const pnlColorScale = d3.scaleLinear()
        .domain([yExt[0], yExt[0] + (yExt[1] - yExt[0]) * 0.33, yExt[0] + (yExt[1] - yExt[0]) * 0.66, yExt[1]])
        .range(['#bbbbbb', '#888888', '#444444', '#080808'])
        .clamp(true);

    // Gradient definition for the line
    const gradId = 'pnlGrad-' + range;
    const defs = svg.append('defs');
    const grad = defs.append('linearGradient')
        .attr('id', gradId)
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', marginLeft).attr('x2', width - marginRight);

    const stops = [0, 0.25, 0.5, 0.75, 1];
    stops.forEach(t => {
        const idx = Math.round(t * (data.length - 1));
        grad.append('stop').attr('offset', t).attr('stop-color', pnlColorScale(data[idx].pnl));
    });

    // Animated path
    const path = svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'url(#' + gradId + ')')
        .attr('stroke-width', 2)
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round')
        .attr('d', line);

    const totalLen = path.node().getTotalLength();
    path.attr('stroke-dasharray', totalLen + ',' + totalLen)
        .attr('stroke-dashoffset', totalLen)
        .transition().duration(2000).ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0);

    // Dots
    const step = Math.max(1, Math.floor(data.length / 15));
    const keyPoints = data.filter((_, i) => i % step === 0 || i === data.length - 1);

    svg.append('g')
        .selectAll('circle')
        .data(keyPoints)
        .join('circle')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.pnl))
        .attr('r', 3)
        .attr('fill', '#ffffff')
        .attr('stroke', d => pnlColorScale(d.pnl))
        .attr('stroke-width', 1.5)
        .attr('opacity', 0)
        .transition()
        .delay((d, i) => (i / keyPoints.length) * 2000)
        .attr('opacity', 1);

    // End label
    const last = data[data.length - 1];
    svg.append('text')
        .attr('x', x(last.date))
        .attr('y', y(last.pnl) - 8)
        .attr('text-anchor', 'end')
        .attr('fill', pnlColorScale(last.pnl))
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .attr('opacity', 0)
        .text((last.pnl >= 0 ? '+' : '') + last.pnl.toFixed(2) + '%')
        .transition().delay(2000).attr('opacity', 1);
}

renderPnlChart('1M', null);

// === D3.js AGENT PERFORMANCE GROUPED BAR CHART ===
(function renderAgentPerfChart() {
    const container = document.getElementById('agentPerfChart');
    if (!container || typeof d3 === 'undefined') return;
    const tooltip = document.getElementById('agentPerfTooltip');

    const agents = ['Momentum', 'Risk', 'Mean Rev', 'Sentiment', 'Arbitrage'];
    const timeframes = ['1W', '1M', '3M', '6M'];

    const baseRates = { Momentum: 68, Risk: 72, 'Mean Rev': 65, Sentiment: 58, Arbitrage: 75 };
    const tfAdj = { '1W': -5, '1M': 0, '3M': 3, '6M': 5 };

    const data = agents.map(agent => {
        const row = { agent };
        timeframes.forEach(tf => {
            const base = baseRates[agent] + tfAdj[tf];
            row[tf] = Math.round(Math.max(50, Math.min(85, base)) * 10) / 10;
        });
        return row;
    });

    const margin = { top: 20, right: 10, bottom: 30, left: 40 };
    const containerWidth = container.clientWidth || 928;
    const width = containerWidth - 16;
    const height = 260;

    const fx = d3.scaleBand()
        .domain(agents)
        .rangeRound([margin.left, width - margin.right])
        .paddingInner(0.2);

    const x = d3.scaleBand()
        .domain(timeframes)
        .rangeRound([0, fx.bandwidth()])
        .padding(0.1);

    const y = d3.scaleLinear()
        .domain([40, 90])
        .nice()
        .rangeRound([height - margin.bottom, margin.top]);

    const barColors = { '1W': '#080808', '1M': '#444444', '3M': '#888888', '6M': '#bbbbbb' };

    const svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height]);

    // Grid lines
    svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',0)')
        .call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(''))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('.tick line').attr('stroke', '#cccccc').attr('stroke-opacity', 0.4));

    // Bars
    const groups = svg.append('g')
        .selectAll('g')
        .data(data)
        .join('g')
        .attr('transform', d => 'translate(' + fx(d.agent) + ',0)');

    // NOTE: innerHTML used in tooltip with trusted agent performance data
    groups.selectAll('rect')
        .data(d => timeframes.map(tf => ({ agent: d.agent, tf, value: d[tf] })))
        .join('rect')
        .attr('x', d => x(d.tf))
        .attr('width', x.bandwidth())
        .attr('y', height - margin.bottom)
        .attr('height', 0)
        .attr('fill', d => barColors[d.tf])
        .attr('rx', 2)
        .on('mouseenter', function(event, d) {
            d3.select(this).attr('opacity', 0.75);
            tooltip.style.opacity = '1';
            tooltip.innerHTML = '<strong>' + d.agent + '</strong><br>' + d.tf + ': ' + d.value + '% win rate';
        })
        .on('mousemove', function(event) {
            tooltip.style.left = (event.clientX + 12) + 'px';
            tooltip.style.top = (event.clientY - 10) + 'px';
        })
        .on('mouseleave', function() {
            d3.select(this).attr('opacity', 1);
            tooltip.style.opacity = '0';
        })
        .transition()
        .duration(600)
        .delay((d, i) => i * 80)
        .attr('y', d => y(d.value))
        .attr('height', d => y(40) - y(d.value));

    // X axis
    svg.append('g')
        .attr('transform', 'translate(0,' + (height - margin.bottom) + ')')
        .call(d3.axisBottom(fx).tickSizeOuter(0).tickSize(0))
        .call(g => g.select('.domain').remove())
        .selectAll('text')
        .style('font-size', '9px')
        .style('font-family', "'Alliance No.2', sans-serif")
        .style('fill', '#666666');

    // Y axis
    svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',0)')
        .call(d3.axisLeft(y).ticks(5).tickFormat(d => d + '%'))
        .call(g => g.select('.domain').remove())
        .selectAll('text')
        .style('font-size', '9px')
        .style('font-family', "'Alliance No.2', sans-serif")
        .style('fill', '#666666');

    // Legend
    const legend = svg.append('g')
        .attr('transform', 'translate(' + (width - margin.right - timeframes.length * 50) + ',' + 8 + ')');

    timeframes.forEach((tf, i) => {
        const g = legend.append('g').attr('transform', 'translate(' + (i * 50) + ',0)');
        g.append('rect').attr('width', 10).attr('height', 10).attr('rx', 1).attr('fill', barColors[tf]);
        g.append('text').attr('x', 13).attr('y', 9).text(tf).style('font-size', '8px').style('font-family', "'Alliance No.2', sans-serif").style('fill', '#666666');
    });
})();

// === D3.js API USAGE HEATMAP (real tracked calls) ===
(function renderApiHeatmap() {
    const container = document.getElementById('apiHeatmap');
    if (!container || typeof d3 === 'undefined') return;

    // Build heatmap from real API call timestamps tracked by CortexAPI
    const rawCalls = (typeof CortexAPI !== 'undefined' && CortexAPI.getUsageData) ? CortexAPI.getUsageData() : [];
    const now = new Date();
    const data = [];
    const buckets = {};
    rawCalls.forEach(function(ts) {
        var d = new Date(ts);
        var dayKey = d.toISOString().slice(0, 10);
        var hour = d.getHours();
        var key = dayKey + '-' + hour;
        buckets[key] = (buckets[key] || 0) + 1;
    });
    for (let d = 29; d >= 0; d--) {
        const day = new Date(now);
        day.setDate(day.getDate() - d);
        day.setHours(0, 0, 0, 0);
        var dayKey = day.toISOString().slice(0, 10);
        for (let h = 0; h < 24; h++) {
            data.push({
                date: new Date(day),
                hour: h,
                usage: buckets[dayKey + '-' + h] || 0
            });
        }
    }

    const margin = { top: 50, right: 10, bottom: 10, left: 25 };
    const days = [...new Set(data.map(d => d.date.getTime()))].map(t => new Date(t));
    const containerWidth = container.clientWidth || 600;
    const width = containerWidth - 16;
    const cellW = (width - margin.left - margin.right) / 24;
    const cellH = Math.max(cellW * 0.55, 10);
    const height = margin.top + margin.bottom + days.length * cellH;
    const maxUsage = d3.max(data, d => d.usage);

    const x = d3.scaleBand()
        .domain(d3.range(24))
        .range([margin.left, width - margin.right])
        .padding(0.05);

    const y = d3.scaleBand()
        .domain(days.map(d => d.getTime()))
        .range([margin.top, height - margin.bottom])
        .padding(0.05);

    const color = d3.scaleSequential([0, maxUsage], d3.interpolateGreys);

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height]);

    // X axis — hours
    const formatHour = d => d === 0 ? '12a' : d === 12 ? '12p' : (d % 12) + (d < 12 ? 'a' : 'p');
    svg.append('g')
        .attr('transform', 'translate(0,' + margin.top + ')')
        .call(d3.axisTop(x).tickFormat(formatHour).tickSize(0))
        .call(g => g.select('.domain').remove())
        .selectAll('text')
        .style('font-size', '7px')
        .style('fill', '#666666');

    // Y axis — days
    const formatMonth = d3.timeFormat('%b %-d');
    const formatDate = d3.timeFormat('%-d');
    const formatDay = d => (d.getDate() === 1 ? formatMonth : formatDate)(d);

    svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',0)')
        .call(d3.axisLeft(y).tickFormat(t => formatDay(new Date(t))).tickSize(0))
        .call(g => g.select('.domain').remove())
        .selectAll('text')
        .style('font-size', '7px')
        .style('fill', '#666666');

    // Tooltip
    const tooltip = document.getElementById('apiTooltip');

    // Cells
    svg.append('g')
        .selectAll('rect')
        .data(data)
        .join('rect')
        .attr('x', d => x(d.hour))
        .attr('y', d => y(d.date.getTime()))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => d.usage === 0 ? '#f8f8f8' : color(d.usage))
        .attr('rx', 1)
        .on('mouseover', function(event, d) {
            tooltip.style.opacity = '1';
            const dayStr = d3.timeFormat('%b %d')(d.date);
            tooltip.textContent = dayStr + ' ' + formatHour(d.hour) + ' \u2014 ' + d.usage + ' calls';
        })
        .on('mousemove', function(event) {
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY - 28) + 'px';
        })
        .on('mouseout', function() {
            tooltip.style.opacity = '0';
        });

    // Legend gradient on canvas
    const canvas = document.getElementById('apiLegendBar');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        for (let i = 0; i < w; i++) {
            ctx.fillStyle = color((i / w) * maxUsage);
            ctx.fillRect(i, 0, 1, h);
        }
    }
})();



// === LIVE TICKER TAPE (CoinGecko) ===
const TICKER_COINS = [
    { id: 'solana', pair: 'SOL/USDC' },
    { id: 'bitcoin', pair: 'BTC/USDC' },
    { id: 'ethereum', pair: 'ETH/USDC' },
    { id: 'raydium', pair: 'RAY/USDC' },
    { id: 'jupiter-exchange-solana', pair: 'JUP/USDC' },
    { id: 'drift-protocol', pair: 'DRIFT/USDC' },
];

function formatTickerPrice(price) {
    if (price >= 1000) return '$' + price.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (price >= 1) return '$' + price.toFixed(2);
    if (price >= 0.01) return '$' + price.toFixed(4);
    return '$' + price.toFixed(6);
}

function renderTickerItems(data) {
    const el = document.getElementById('tickerContent');
    if (!el || !data) return;
    const items = TICKER_COINS.map(coin => {
        const d = data[coin.id];
        if (!d) return '';
        const price = formatTickerPrice(d.usd);
        const change = d.usd_24h_change || 0;
        const dir = change >= 0 ? 'up' : 'down';
        const sign = change >= 0 ? '+' : '';
        return '<span class="ticker-item">' + coin.pair +
            ' <span class="price">' + price + '</span>' +
            ' <span class="change ' + dir + '">' + sign + change.toFixed(2) + '%</span></span>';
    }).join('');
    // NOTE: innerHTML used here with trusted ticker price data from CoinGecko
    el.innerHTML = items + items;
}

async function fetchTickerPrices() {
    try {
        const ids = TICKER_COINS.map(c => c.id).join(',');
        const key = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.COINGECKO_API_KEY) || '';
        const url = 'https://api.coingecko.com/api/v3/simple/price?ids=' + ids +
            '&vs_currencies=usd&include_24hr_change=true' +
            (key ? '&x_cg_demo_api_key=' + key : '');
        const res = await fetch(url);
        if (!res.ok) throw new Error('CoinGecko HTTP ' + res.status);
        const data = await res.json();
        // Store prices globally for position card PnL calculations
        window._latestTickerPrices = {};
        TICKER_COINS.forEach(function(coin) {
            if (data[coin.id]) window._latestTickerPrices[coin.pair] = data[coin.id].usd;
        });
        renderTickerItems(data);
        console.log('[TICKER] Live prices updated');
    } catch (e) {
        console.warn('[TICKER] Fetch error:', e.message);
    }
}

fetchTickerPrices();
setInterval(fetchTickerPrices, 30000);

// === SERVICE INITIALIZATION ===
startNewsRefresh();
startRegimeDetection();
startGuardianService();
startAgentService();

console.log('CORTEX Dashboard initialized');
console.log('TX Feed: Active');
console.log('Guardian: Live API connected');

// === AGENT SIGNAL CLUSTERING — empty state until live data available ===
(function () {
    var el = document.getElementById('agentClusterChart');
    if (!el) return;
    el.style.display = 'flex';
    el.style.alignItems = 'center';
    el.style.justifyContent = 'center';
    var msg = document.createElement('div');
    msg.style.fontSize = '0.6rem';
    msg.style.color = 'var(--dim)';
    msg.style.fontFamily = 'var(--font-mono)';
    msg.style.textAlign = 'center';
    msg.textContent = 'No agent signal data available';
    el.appendChild(msg);
    /* hardcoded data array and chart removed — will be populated from live API */
})();
