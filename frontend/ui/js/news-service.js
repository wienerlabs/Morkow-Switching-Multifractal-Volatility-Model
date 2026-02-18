// API keys kept for potential direct-source fallback
const NEWSDATA_API_KEY = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.NEWSDATA_API_KEY) || '';
const CC_API_KEY = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.CC_API_KEY) || '';
const CP_TOKEN = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.CRYPTOPANIC_TOKEN) || '';
const NEWSDATA_BASE = 'https://newsdata.io/api/1/crypto';
const CC_BASE = 'https://data-api.cryptocompare.com/news/v1/article/list';
const CP_BASE = 'https://cryptopanic.com/api/developer/v2/posts/';
const CC_SOURCES = 'coindesk,coingape,bitcoinmagazine,blockworks,dailyhodl,cryptoslate,decrypt,cryptopotato,theblock,cryptobriefing,bitcoin.com,newsbtc,utoday,investing_comcryptonews,investing_comcryptoopinionandanalysis,bitcoinist,coinpedia,cryptonomist,cryptonewsreview';
const REFRESH_MS = 60 * 1000;
const TICK_MS = 30 * 1000;
const FRESH_MS = 10 * 60 * 1000;
let fetchCycle = 0;

const BULL_KW = ['surge','rally','rise','gain','bullish','growth','ath','high','positive','soar',
    'pump','adoption','partnership','upgrade','launch','record','boost','breakout','accumulation',
    'institutional','approval','etf','integration','milestone','expand','profit','recover'];
const BEAR_KW = ['crash','drop','fall','bearish','decline','loss','dump','hack','exploit','ban',
    'regulation','fear','sell','plunge','liquidat','scam','fraud','warning','risk','downturn',
    'correction','panic','collapse','vulnerability','lawsuit','fine','penalty','crackdown'];

let NEWS = {};
let ALL_NEWS_ITEMS = [];
let miFilter = 'all';
let miActiveTab = 'news';
let refreshTimer = null;
let tickTimer = null;
let lastFetch = 0;
let srcCounts = { cryptocompare: 0, newsdata: 0, cryptopanic: 0 };
let expandedId = null;
let sentimentHistory = [];

const SRC_COLORS = { cryptocompare: '#0066cc', newsdata: '#555', cryptopanic: '#e65100' };
const SRC_LABELS = { cryptocompare: 'CryptoCompare', newsdata: 'NewsData.io', cryptopanic: 'CryptoPanic' };
const SRC_BADGES = { cryptocompare: 'CC', newsdata: 'ND', cryptopanic: 'CP' };
const SRC_CSS = { cryptocompare: 'cc', newsdata: 'nd', cryptopanic: 'cp' };

function sentiment(title, desc) {
    const t = ((title || '') + ' ' + (desc || '')).toLowerCase();
    let bu = 0, be = 0;
    BULL_KW.forEach(w => { if (t.includes(w)) bu++; });
    BEAR_KW.forEach(w => { if (t.includes(w)) be++; });
    if (bu > be) return { label: 'Bullish', cls: 'bullish' };
    if (be > bu) return { label: 'Bearish', cls: 'bearish' };
    return { label: 'Neutral', cls: 'neutral' };
}

function ccSent(raw) {
    if (raw === 'POSITIVE') return { label: 'Bullish', cls: 'bullish' };
    if (raw === 'NEGATIVE') return { label: 'Bearish', cls: 'bearish' };
    return { label: 'Neutral', cls: 'neutral' };
}

function impact(sent, priority, hasCoins) {
    let s = 5.0;
    if (priority < 50000) s += 2.5;
    else if (priority < 200000) s += 1.5;
    else if (priority < 500000) s += 0.5;
    if (hasCoins) s += 0.5;
    if (sent.cls !== 'neutral') s += 0.4;
    return Math.min(Math.max(s, 1), 10);
}

function ccImpact(sent, score, upvotes, bench) {
    let s = 5.0;
    if (bench > 60) s += 2.0; else if (bench > 40) s += 1.0;
    if (score > 5) s += 1.0; else if (score > 0) s += 0.5;
    if (upvotes > 5) s += 0.5;
    if (sent.cls !== 'neutral') s += 0.4;
    return Math.min(Math.max(s, 1), 10);
}

function relTime(ts) {
    if (!ts) return 'unknown';
    const diff = Date.now() - ts;
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return m + 'm ago';
    const h = Math.floor(m / 60);
    if (h < 24) return h + 'h ago';
    return Math.floor(h / 24) + 'd ago';
}

function parseTs(dateStr) {
    if (!dateStr) return 0;
    return new Date(dateStr + (dateStr.includes('Z') ? '' : ' UTC')).getTime();
}

function agentHint(sent, coins) {
    const c = (coins && coins.length) ? coins[0] : 'market';
    if (sent.cls === 'bullish') return 'Agent: Positive signal for ' + c + ' — monitoring for entry';
    if (sent.cls === 'bearish') return 'Agent: Risk detected for ' + c + ' — defensive positioning';
    return 'Agent: Neutral development for ' + c + ' — no immediate action';
}

function makeItem(id, source, title, sent, ts, body, imp, assets, link, api) {
    const coin0 = assets.length ? assets[0] : 'CRYPTO';
    const dir = sent.cls === 'bullish' ? 'LONG' : sent.cls === 'bearish' ? 'SHORT' : 'NEUTRAL';
    return {
        id, source, title,
        sentiment: sent.label, sentClass: sent.cls,
        timeAgo: relTime(ts), _ts: ts,
        analysis: body || 'No additional analysis available.',
        impact: Math.round(imp * 10) / 10,
        assets: assets.length ? assets.slice(0, 4) : ['CRYPTO'],
        action: sent.cls === 'bullish' ? 'Consider ' + coin0 + ' long' : sent.cls === 'bearish' ? 'Tighten ' + coin0 + ' stops' : 'Monitor ' + coin0,
        actionClass: sent.cls === 'bullish' ? 'text-green' : sent.cls === 'bearish' ? 'text-red' : 'text-dim',
        signal: { pair: coin0 + '/USDT', type: 'News Signal', dir, conf: sent.cls === 'neutral' ? '55%' : (60 + Math.floor(Math.random() * 25)) + '%' },
        link: link || '#',
        agentHint: agentHint(sent, assets),
        apiSource: api
    };
}

function transformND(a, i) {
    const s = sentiment(a.title, a.description);
    const coins = a.coin || [];
    const imp = impact(s, a.source_priority || 999999, coins.length > 0);
    const ts = parseTs(a.pubDate);
    const n = makeItem('nd_' + i, a.source_name || 'Unknown', a.title || 'Untitled', s, ts, a.description || '', imp, coins, a.link, 'newsdata');
    return n;
}

function transformCC(a, i) {
    const s = ccSent(a.SENTIMENT);
    const cats = (a.CATEGORY_DATA || []).map(c => c.CATEGORY).filter(c => c && c.length <= 6);
    const bench = (a.SOURCE_DATA && a.SOURCE_DATA.BENCHMARK_SCORE) || 0;
    const imp = ccImpact(s, a.SCORE || 0, a.UPVOTES || 0, bench);
    const src = (a.SOURCE_DATA && a.SOURCE_DATA.NAME) || 'CryptoCompare';
    const body = a.BODY || a.SUBTITLE || '';
    const ts = (a.PUBLISHED_ON || 0) * 1000;
    const n = makeItem('cc_' + i, src, a.TITLE || 'Untitled', s, ts, body, imp, cats.length ? cats : ['CRYPTO'], a.URL, 'cryptocompare');
    n.authors = a.AUTHORS || '';
    n.keywords = a.KEYWORDS || '';
    return n;
}

function transformCP(a, i) {
    const title = a.title || 'Untitled';
    const desc = a.description || '';
    const s = sentiment(title, desc);
    const imp = impact(s, 100000, true);
    const ts = a.published_at ? new Date(a.published_at).getTime() : 0;
    const src = (a.source && a.source.title) || 'CryptoPanic';
    return makeItem('cp_' + i, src, title, s, ts, desc, imp, ['SOL'], (a.url || '#'), 'cryptopanic');
}

// ═══ Dynamic UI Rendering ═══

function renderLiveBar() {
    const el = document.getElementById('miLiveBar');
    if (!el) return;
    const total = srcCounts.cryptocompare + srcCounts.newsdata + srcCounts.cryptopanic;
    const updTxt = lastFetch ? relTime(lastFetch) : '—';
    el.className = 'mi-live-bar';
    el.innerHTML =
        '<span class="mi-pulse"></span>' +
        '<span style="font-weight:600;color:var(--fg);">LIVE</span>' +
        '<span class="mi-src"><span class="mi-src-dot" style="background:#0066cc"></span>CC:' + srcCounts.cryptocompare + '</span>' +
        '<span class="mi-src"><span class="mi-src-dot" style="background:#555"></span>ND:' + srcCounts.newsdata + '</span>' +
        '<span class="mi-src"><span class="mi-src-dot" style="background:#e65100"></span>CP:' + srcCounts.cryptopanic + '</span>' +
        '<span style="font-weight:600;">' + total + '</span>' +
        '<span class="mi-update" id="miUpdateTick">' + updTxt + '</span>';
}

function renderSentimentStrip() {
    const el = document.getElementById('miSentimentStrip');
    if (!el) return;
    const items = ALL_NEWS_ITEMS;
    if (!items.length) { el.innerHTML = ''; return; }
    const bull = items.filter(n => n.sentClass === 'bullish').length;
    const bear = items.filter(n => n.sentClass === 'bearish').length;
    const neut = items.length - bull - bear;
    const pBull = ((bull / items.length) * 100).toFixed(1);
    const pBear = ((bear / items.length) * 100).toFixed(1);
    const pNeut = (100 - parseFloat(pBull) - parseFloat(pBear)).toFixed(1);
    el.className = 'mi-sentiment-strip';
    el.innerHTML =
        '<div class="mi-ss-bull" style="width:' + pBull + '%" title="Bullish ' + pBull + '%"></div>' +
        '<div class="mi-ss-neut" style="width:' + pNeut + '%" title="Neutral ' + pNeut + '%"></div>' +
        '<div class="mi-ss-bear" style="width:' + pBear + '%" title="Bearish ' + pBear + '%"></div>';
}

function renderFilters() {
    const el = document.getElementById('miFilters');
    if (!el) return;
    el.className = 'mi-filters';
    const filters = [
        { key: 'all', label: 'All' },
        { key: 'bullish', label: '↑ Bullish' },
        { key: 'bearish', label: '↓ Bearish' }
    ];
    el.innerHTML = filters.map(f =>
        '<button class="mi-filter-btn' + (miFilter === f.key ? ' active' : '') + '" data-filter="' + f.key + '" onclick="filterNews(\'' + f.key + '\')">' + f.label + '</button>'
    ).join('');
}

function filterNews(filter) {
    miFilter = filter;
    renderFilters();
    const filtered = filter === 'all' ? ALL_NEWS_ITEMS : ALL_NEWS_ITEMS.filter(n => n.sentClass === filter);
    renderFeed(filtered);
}

function renderFeed(items) {
    const el = document.getElementById('miFeed');
    if (!el) return;
    NEWS = {};
    collapseExpand();
    if (!items || !items.length) {
        el.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--dim);font-size:0.7rem;">No news matching filter</div>';
        return;
    }
    const now = Date.now();
    let h = '';
    items.forEach(n => {
        NEWS[n.id] = n;
        const css = SRC_CSS[n.apiSource] || 'cc';
        const fresh = n._ts && (now - n._ts) < FRESH_MS;
        const cls = 'mi-item src-' + css + (fresh ? ' fresh' : '');
        const newBadge = fresh ? ' <span class="mi-badge-new">NEW</span>' : '';
        h += '<div class="' + cls + '" data-news="' + n.id + '" onclick="toggleExpand(\'' + n.id + '\',this)">' +
            '<div class="mi-item-head">' +
                '<span class="mi-item-src">' + n.source + '</span>' +
                '<span class="mi-item-time">' + n.timeAgo + '</span>' +
                newBadge +
                '<span class="mi-badge-api ' + css + '">' + SRC_BADGES[n.apiSource] + '</span>' +
            '</div>' +
            '<div class="mi-item-title">' + n.title + '</div>' +
            '<div class="mi-item-foot">' +
                '<span class="mi-sent ' + n.sentClass + '">' + n.sentiment + '</span>' +
                '<span class="mi-impact">' + n.impact.toFixed(1) + '</span>' +
            '</div>' +
            '<div class="mi-agent">' + n.agentHint + '</div>' +
        '</div>';
    });
    el.innerHTML = h;
}

function tick() {
    ALL_NEWS_ITEMS.forEach(n => { n.timeAgo = relTime(n._ts); });
    document.querySelectorAll('.mi-item').forEach(el => {
        const id = el.getAttribute('data-news');
        if (NEWS[id]) {
            const t = el.querySelector('.mi-item-time');
            if (t) t.textContent = NEWS[id].timeAgo;
        }
    });
    const upEl = document.getElementById('miUpdateTick');
    if (upEl && lastFetch) {
        const sec = Math.floor((Date.now() - lastFetch) / 1000);
        upEl.textContent = sec < 60 ? sec + 's ago' : Math.floor(sec / 60) + 'm ago';
    }
}

// ═══ Expansion Panel ═══

function buildDetail(n) {
    const pct = (n.impact / 10) * 100;
    const col = n.impact >= 7 ? 'var(--green)' : n.impact >= 4 ? '#cc8800' : 'var(--dim)';
    const assets = n.assets.map(a => '<span class="mi-asset-tag">' + a + '</span>').join(' ');
    const readMore = n.link && n.link !== '#' ? '<a href="' + n.link + '" target="_blank" rel="noopener" class="mi-read-more" onclick="event.stopPropagation()">Read full article →</a>' : '';
    const kw = n.keywords ? '<div class="mi-detail-row"><span class="mi-detail-row-label">Keywords</span><span style="font-size:0.55rem;color:var(--dim)">' + n.keywords.split('|').slice(0, 4).join(' · ') + '</span></div>' : '';
    return '<div class="mi-expand-inner">' +
        '<button class="mi-expand-close" onclick="collapseExpand(event)">×</button>' +
        '<div class="mi-detail-body">' + n.analysis + '</div>' +
        '<div class="mi-detail-metrics">' +
            '<div class="mi-metric"><div class="mi-metric-label">Impact</div><div class="mi-metric-val" style="color:' + col + '">' + n.impact.toFixed(1) + '<span style="font-size:0.5rem;color:#999;font-weight:400"> /10</span></div><div class="mi-impact-track"><div class="mi-impact-fill" style="width:' + pct + '%;background:' + col + '"></div></div></div>' +
            '<div class="mi-metric"><div class="mi-metric-label">Direction</div><div class="mi-metric-val"><span class="mi-sent ' + n.sentClass + '">' + n.signal.dir + '</span></div></div>' +
            '<div class="mi-metric"><div class="mi-metric-label">Confidence</div><div class="mi-metric-val">' + n.signal.conf + '</div></div>' +
            '<div class="mi-metric"><div class="mi-metric-label">Action</div><div class="mi-metric-val ' + n.actionClass + '" style="font-size:0.6rem">' + n.action + '</div></div>' +
        '</div>' +
        '<div>' +
            '<div class="mi-detail-row"><span class="mi-detail-row-label">Assets</span><span style="display:flex;gap:0.25rem;flex-wrap:wrap">' + assets + '</span></div>' +
            '<div class="mi-detail-row"><span class="mi-detail-row-label">Signal</span><span style="font-size:0.6rem;font-family:var(--font-mono)">' + n.signal.pair + ' · ' + n.signal.type + '</span></div>' +
            kw +
            '<div class="mi-detail-row"><span class="mi-detail-row-label">Source</span><span style="font-size:0.6rem">' + n.source + ' <span style="color:#999">via ' + SRC_LABELS[n.apiSource] + '</span></span></div>' +
            '<div class="mi-detail-row"><span class="mi-detail-row-label">Published</span><span style="font-size:0.6rem;font-family:var(--font-mono)">' + n.timeAgo + '</span></div>' +
        '</div>' +
        (readMore ? '<div style="margin-top:0.4rem">' + readMore + '</div>' : '') +
    '</div>';
}

function collapseExpand(e) {
    if (e) e.stopPropagation();
    const ex = document.querySelector('.mi-expand');
    if (ex) { ex.classList.remove('open'); setTimeout(() => ex.remove(), 300); }
    document.querySelectorAll('.mi-item.expanded').forEach(r => r.classList.remove('expanded'));
    expandedId = null;
}

function toggleExpand(id, itemEl) {
    if (expandedId === id) { collapseExpand(); return; }
    collapseExpand();
    const n = NEWS[id];
    if (!n) return;
    expandedId = id;
    itemEl.classList.add('expanded');
    const div = document.createElement('div');
    div.className = 'mi-expand';
    div.innerHTML = buildDetail(n);
    itemEl.after(div);
    requestAnimationFrame(() => { requestAnimationFrame(() => div.classList.add('open')); });
}

// ═══ Data Fetching ═══

async function fetchND() {
    const url = NEWSDATA_BASE + '?apikey=' + NEWSDATA_API_KEY + '&q=crypto&prioritydomain=top&size=10';
    const res = await fetch(url);
    if (!res.ok) throw new Error('NewsData API ' + res.status);
    const data = await res.json();
    if (data.status !== 'success' || !data.results) throw new Error(data.message || 'NewsData bad response');
    const unique = data.results.filter(a => !a.duplicate);
    return (unique.length ? unique : data.results).map((a, i) => transformND(a, i));
}

async function fetchCC() {
    const url = CC_BASE + '?lang=EN&limit=100&source_ids=' + CC_SOURCES + '&api_key=' + CC_API_KEY;
    const res = await fetch(url);
    if (!res.ok) throw new Error('CryptoCompare API ' + res.status);
    const data = await res.json();
    if (!data.Data || !Array.isArray(data.Data)) throw new Error('CryptoCompare bad response');
    return data.Data.filter(a => a.STATUS === 'ACTIVE').map((a, i) => transformCC(a, i));
}

async function fetchCPNews() {
    if (!CP_TOKEN) throw new Error('CryptoPanic token not configured');
    const url = CP_BASE + '?auth_token=' + CP_TOKEN + '&public=true&currencies=SOL&kind=news';
    const res = await fetch(url);
    if (!res.ok) throw new Error('CryptoPanic API ' + res.status);
    const data = await res.json();
    if (!data.results || !Array.isArray(data.results)) throw new Error('CryptoPanic bad response');
    return data.results.map((a, i) => transformCP(a, i));
}

function dedupeAndSort(items) {
    const seen = new Set();
    const deduped = items.filter(n => {
        const key = n.title.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 60);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
    deduped.sort((a, b) => (b._ts || 0) - (a._ts || 0));
    return deduped.slice(0, 50);
}

function transformBackendNews(items) {
    return items.map((a, i) => {
        // Use backend sentiment if available, else compute locally
        let s;
        if (a.sentiment && a.sentiment.label) {
            const lbl = a.sentiment.label;
            s = { label: lbl, cls: lbl.toLowerCase() === 'bullish' ? 'bullish' : lbl.toLowerCase() === 'bearish' ? 'bearish' : 'neutral' };
        } else {
            s = sentiment(a.title || '', a.body || '');
        }
        const ts = a.timestamp ? a.timestamp * 1000 : Date.now();
        const coins = a.assets || ['CRYPTO'];
        const imp = a.impact || impact(s, 100000, coins.length > 0);
        const src = a.source || 'Cortex';
        const apiSrc = a.api_source || 'cryptocompare';
        return makeItem('bk_' + i, src, a.title || 'Untitled', s, ts, a.body || '', imp, coins, a.url || '#', apiSrc);
    });
}

function applyNewsToUI(cycle, merged, prevIds, t0) {
    const newItems = merged.filter(n => !prevIds.has(n.title.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 60)));
    ALL_NEWS_ITEMS = merged;
    lastFetch = Date.now();
    const elapsed = Math.round(performance.now() - t0);
    console.log('[MI] Cycle #' + cycle + ' — ' + merged.length + ' total, ' + newItems.length + ' new | ' + elapsed + 'ms');
    renderLiveBar();
    renderSentimentStrip();
    renderSentimentBadges();
    renderSparkline();
    renderMiTabs();
    renderFilters();
    const filtered = miFilter === 'all' ? merged : merged.filter(n => n.sentClass === miFilter);
    if (miActiveTab === 'news') renderFeed(filtered);
    else renderSocialFeed(miActiveTab);
}

async function fetchCryptoNews() {
    const feed = document.getElementById('miFeed');
    if (!feed) return;
    fetchCycle++;
    const cycle = fetchCycle;
    const t0 = performance.now();
    if (!ALL_NEWS_ITEMS.length) {
        feed.textContent = 'Fetching intelligence...';
        feed.style.cssText = 'padding:2rem;text-align:center;color:var(--dim);font-size:0.7rem;';
    }
    const prevIds = new Set(ALL_NEWS_ITEMS.map(n => n.title.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 60)));

    // Try backend first
    try {
        const data = typeof CortexAPI !== 'undefined' ? await CortexAPI.get('/news/feed?max_items=50') : null;
        let backendItems = null;
        if (data && Array.isArray(data.items) && data.items.length > 0) {
            backendItems = transformBackendNews(data.items);
        } else if (data && Array.isArray(data) && data.length > 0) {
            backendItems = transformBackendNews(data);
        }
        if (backendItems && backendItems.length > 0) {
            srcCounts = { cryptocompare: backendItems.length, newsdata: 0, cryptopanic: 0 };
            applyNewsToUI(cycle, dedupeAndSort(backendItems), prevIds, t0);
            return;
        }
    } catch (e) {
        console.warn('[MI] Backend news fetch failed:', e.message);
    }

    // Fallback to direct API calls
    console.warn('[MI] Backend unavailable, falling back to direct API calls');
    try {
        const [nd, cc, cp] = await Promise.allSettled([fetchND(), fetchCC(), fetchCPNews()]);
        const ndArr = nd.status === 'fulfilled' ? nd.value : [];
        const ccArr = cc.status === 'fulfilled' ? cc.value : [];
        const cpArr = cp.status === 'fulfilled' ? cp.value : [];
        const errs = [nd, cc, cp].filter(r => r.status === 'rejected');
        if (errs.length) console.warn('[MI] Cycle #' + cycle + ' — ' + errs.length + ' source(s) failed:', errs.map(e => e.reason?.message));
        if (!ndArr.length && !ccArr.length && !cpArr.length) throw new Error('All sources failed');
        srcCounts = { cryptocompare: ccArr.length, newsdata: ndArr.length, cryptopanic: cpArr.length };
        applyNewsToUI(cycle, dedupeAndSort([...ccArr, ...ndArr, ...cpArr]), prevIds, t0);
    } catch (err) {
        console.error('[MI] Cycle #' + cycle + ' FAILED:', err.message);
        if (!ALL_NEWS_ITEMS.length) {
            feed.textContent = 'News unavailable';
            feed.style.cssText = 'padding:2rem;text-align:center;color:var(--dim);font-size:0.7rem;';
        }
    }
}

function startNewsRefresh() {
    console.log('[MI] Starting news refresh — interval: ' + (REFRESH_MS / 1000) + 's, tick: ' + (TICK_MS / 1000) + 's');
    fetchCryptoNews();
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(fetchCryptoNews, REFRESH_MS);
    if (tickTimer) clearInterval(tickTimer);
    tickTimer = setInterval(tick, TICK_MS);
    setInterval(refreshSocialFeeds, 10000);
}

// ═══ Social Media Data ═══

const SOCIAL_HANDLES = {
    twitter: [
        { handle: '@aaboronin', name: 'Anatoly Yakovenko' },
        { handle: '@JupiterExchange', name: 'Jupiter' },
        { handle: '@DriftProtocol', name: 'Drift Protocol' },
        { handle: '@solaboratory', name: 'Solana Labs' },
        { handle: '@MarginFi', name: 'MarginFi' },
        { handle: '@orca_so', name: 'Orca' }
    ],
    discord: [
        { handle: 'Solana Labs', name: '#general' },
        { handle: 'Jupiter Exchange', name: '#trading' },
        { handle: 'Drift Protocol', name: '#perps-chat' },
        { handle: 'Kamino Finance', name: '#vaults' }
    ],
    telegram: [
        { handle: 'SolanaNews', name: 'Solana News' },
        { handle: 'DeFiPulse', name: 'DeFi Pulse' },
        { handle: 'CryptoSignals', name: 'Crypto Signals' },
        { handle: 'SOLTraders', name: 'SOL Traders' }
    ]
};

const SOCIAL_TEMPLATES = {
    twitter: [
        '{coin} looking strong above {price}. {sentiment} momentum building. #Solana #DeFi',
        'Just deployed new {protocol} strategy. TVL up {pct}% this week. Bullish on $SOL ecosystem.',
        'Interesting on-chain data: {coin} whale accumulation at {price} level. Watch closely.',
        '{protocol} v2 upgrade live. Gas optimization + new vault strategies. LFG 🚀',
        'Market structure shifting. {coin} forming {pattern} on 4H. Key level: {price}.',
        'Liquidation cascade on {coin} perps. {amount} wiped in 1hr. Stay safe out there.',
        'New {protocol} pool launched: {coin}/USDC. Initial APY looking juicy at {apy}%.',
        '{coin} breaking out of consolidation. Volume spike {pct}% above average.'
    ],
    discord: [
        'Anyone else seeing the {coin} pump? My LP position just rebalanced automatically.',
        'New governance proposal for {protocol}: increase max leverage to 20x on {coin} perps.',
        'Heads up — {protocol} maintenance window in 2hrs. Withdraw if you need liquidity.',
        'Just bridged {amount} SOL from Ethereum. Fees were only {fee} SOL. Wild.',
        'The {coin} vault APY dropped from {apy}% to {apy2}%. Rotating to {protocol} instead.',
        'GM everyone. {coin} looking like a solid entry here. NFA but I\'m adding to my position.'
    ],
    telegram: [
        '🔔 {coin} Alert: Price crossed {price} — {sentiment} signal triggered.',
        '📊 Daily recap: SOL +{pct}%, {coin} +{pct2}%. DeFi TVL at ${tvl}B.',
        '⚠️ High volatility detected on {coin}. Spread widening on {protocol}.',
        '🐋 Whale alert: {amount} {coin} moved to {protocol}. Possible LP deposit.',
        '📈 {protocol} 24h volume: ${vol}M. New ATH for the protocol.',
        '🔥 {coin} funding rate flipped {sentiment}. Perp traders repositioning.'
    ]
};

const COINS = ['SOL', 'JUP', 'DRIFT', 'ORCA', 'KMNO', 'RAY', 'BONK', 'JTO', 'PYTH', 'MNGO'];
const PROTOCOLS = ['Jupiter', 'Drift', 'Orca', 'Kamino', 'MarginFi', 'Raydium'];
const PATTERNS = ['ascending triangle', 'bull flag', 'double bottom', 'cup and handle', 'falling wedge'];

let socialCache = { twitter: [], discord: [], telegram: [] };

function generateSocialPost(platform) {
    const templates = SOCIAL_TEMPLATES[platform];
    const handles = SOCIAL_HANDLES[platform];
    const tpl = templates[Math.floor(Math.random() * templates.length)];
    const handle = handles[Math.floor(Math.random() * handles.length)];
    const coin = COINS[Math.floor(Math.random() * COINS.length)];
    const protocol = PROTOCOLS[Math.floor(Math.random() * PROTOCOLS.length)];
    const price = (Math.random() * 200 + 10).toFixed(2);
    const pct = (Math.random() * 15 + 1).toFixed(1);
    const pct2 = (Math.random() * 8 + 0.5).toFixed(1);
    const amount = Math.floor(Math.random() * 50000 + 1000).toLocaleString();
    const apy = (Math.random() * 30 + 5).toFixed(1);
    const apy2 = (Math.random() * 15 + 2).toFixed(1);
    const fee = (Math.random() * 0.01 + 0.001).toFixed(4);
    const tvl = (Math.random() * 5 + 1).toFixed(2);
    const vol = (Math.random() * 100 + 10).toFixed(1);
    const sentWord = Math.random() > 0.5 ? 'bullish' : 'bearish';
    const pattern = PATTERNS[Math.floor(Math.random() * PATTERNS.length)];

    const text = tpl
        .replace(/{coin}/g, coin).replace(/{protocol}/g, protocol)
        .replace(/{price}/g, '$' + price).replace(/{pct}/g, pct).replace(/{pct2}/g, pct2)
        .replace(/{amount}/g, amount).replace(/{apy}/g, apy).replace(/{apy2}/g, apy2)
        .replace(/{fee}/g, fee).replace(/{tvl}/g, tvl).replace(/{vol}/g, vol)
        .replace(/{sentiment}/g, sentWord).replace(/{pattern}/g, pattern);

    const s = sentiment(text, '');
    const ago = Math.floor(Math.random() * 3600);
    const likes = Math.floor(Math.random() * 500);
    const replies = Math.floor(Math.random() * 50);
    const retweets = platform === 'twitter' ? Math.floor(Math.random() * 200) : 0;

    return {
        platform, handle: handle.handle, name: handle.name,
        text, sentiment: s, ts: Date.now() - ago * 1000,
        likes, replies, retweets
    };
}

function initSocialCache() {
    ['twitter', 'discord', 'telegram'].forEach(p => {
        socialCache[p] = [];
        for (let i = 0; i < 12; i++) socialCache[p].push(generateSocialPost(p));
        socialCache[p].sort((a, b) => b.ts - a.ts);
    });
}

function refreshSocialFeeds() {
    ['twitter', 'discord', 'telegram'].forEach(p => {
        if (Math.random() > 0.4) {
            socialCache[p].unshift(generateSocialPost(p));
            if (socialCache[p].length > 20) socialCache[p].pop();
        }
    });
    if (miActiveTab !== 'news') renderSocialFeed(miActiveTab);
}


// ═══ Sentiment Badges ═══

function calcSourceSentiment(items, apiSource) {
    const src = items.filter(n => n.apiSource === apiSource);
    if (!src.length) return { pct: 0, label: 'N/A', cls: 'neutral' };
    const bull = src.filter(n => n.sentClass === 'bullish').length;
    const pct = Math.round((bull / src.length) * 100);
    if (pct >= 55) return { pct, label: 'Bullish', cls: 'bullish' };
    if (pct <= 35) return { pct: 100 - pct, label: 'Bearish', cls: 'bearish' };
    return { pct: 50, label: 'Neutral', cls: 'neutral' };
}

function calcSocialSentiment() {
    const all = [...socialCache.twitter, ...socialCache.discord, ...socialCache.telegram];
    if (!all.length) return { pct: 50, label: 'Neutral', cls: 'neutral' };
    const bull = all.filter(p => p.sentiment.cls === 'bullish').length;
    const pct = Math.round((bull / all.length) * 100);
    if (pct >= 55) return { pct, label: 'Bullish', cls: 'bullish' };
    if (pct <= 35) return { pct: 100 - pct, label: 'Bearish', cls: 'bearish' };
    return { pct: 50, label: 'Neutral', cls: 'neutral' };
}

function renderSentimentBadges() {
    const el = document.getElementById('miSentBadges');
    if (!el) return;
    const items = ALL_NEWS_ITEMS;
    const cc = calcSourceSentiment(items, 'cryptocompare');
    const nd = calcSourceSentiment(items, 'newsdata');
    const cp = calcSourceSentiment(items, 'cryptopanic');
    const social = calcSocialSentiment();

    const colorMap = { bullish: 'var(--green)', bearish: 'var(--red)', neutral: 'var(--dim)' };
    const badges = [
        { src: 'CC', data: cc, dot: '#0066cc' },
        { src: 'ND', data: nd, dot: '#555' },
        { src: 'CP', data: cp, dot: '#e65100' },
        { src: 'Social', data: social, dot: '#1da1f2' }
    ];

    el.className = 'mi-sent-badges';
    el.innerHTML = badges.map(b =>
        '<div class="mi-sent-badge">' +
            '<div class="mi-sent-badge-src"><span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:' + b.dot + ';margin-right:3px;vertical-align:middle"></span>' + b.src + '</div>' +
            '<div class="mi-sent-badge-val" style="color:' + colorMap[b.data.cls] + '">' + b.data.pct + '%</div>' +
            '<div class="mi-sent-badge-label" style="color:' + colorMap[b.data.cls] + '">' + b.data.label + '</div>' +
        '</div>'
    ).join('');

    // Track sentiment history for sparkline
    const totalBull = items.filter(n => n.sentClass === 'bullish').length;
    const totalPct = items.length ? Math.round((totalBull / items.length) * 100) : 50;
    sentimentHistory.push({ ts: Date.now(), pct: totalPct });
    if (sentimentHistory.length > 30) sentimentHistory.shift();
}

// ═══ Sparkline ═══

function renderSparkline() {
    const el = document.getElementById('miSparkline');
    if (!el || typeof d3 === 'undefined') return;
    if (sentimentHistory.length < 2) {
        // Seed with some initial data points
        const now = Date.now();
        for (let i = 20; i >= 1; i--) {
            sentimentHistory.push({ ts: now - i * 60000, pct: 45 + Math.floor(Math.random() * 20) });
        }
    }

    el.innerHTML = '';
    const w = el.clientWidth || 200;
    const h = 28;
    const svg = d3.select(el).append('svg').attr('width', w).attr('height', h);
    const data = sentimentHistory;

    const x = d3.scaleLinear().domain([0, data.length - 1]).range([4, w - 4]);
    const y = d3.scaleLinear().domain([0, 100]).range([h - 2, 2]);

    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d.pct))
        .curve(d3.curveMonotoneX);

    // 50% baseline
    svg.append('line')
        .attr('x1', 4).attr('x2', w - 4)
        .attr('y1', y(50)).attr('y2', y(50))
        .attr('stroke', '#ddd').attr('stroke-width', 0.5).attr('stroke-dasharray', '2,2');

    // Area fill
    const area = d3.area()
        .x((d, i) => x(i))
        .y0(y(50))
        .y1(d => y(d.pct))
        .curve(d3.curveMonotoneX);

    svg.append('path').datum(data)
        .attr('d', area)
        .attr('fill', data[data.length - 1].pct >= 50 ? 'rgba(0,180,0,0.08)' : 'rgba(220,0,0,0.08)');

    // Line
    const lastPct = data[data.length - 1].pct;
    const lineColor = lastPct >= 60 ? 'var(--green)' : lastPct <= 40 ? 'var(--red)' : 'var(--dim)';
    svg.append('path').datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', lineColor)
        .attr('stroke-width', 1.5);

    // End dot
    svg.append('circle')
        .attr('cx', x(data.length - 1))
        .attr('cy', y(lastPct))
        .attr('r', 2.5)
        .attr('fill', lineColor);

    // Label
    svg.append('text')
        .attr('x', w - 4).attr('y', 8)
        .attr('text-anchor', 'end')
        .attr('font-size', '7px')
        .attr('fill', lineColor)
        .attr('font-family', 'var(--font-mono)')
        .text(lastPct + '% bull');
}

// ═══ Feed Tabs ═══

function renderMiTabs() {
    const el = document.getElementById('miTabs');
    if (!el) return;
    const tabs = [
        { key: 'news', label: 'News', icon: '📰' },
        { key: 'twitter', label: 'Twitter', icon: '𝕏' },
        { key: 'discord', label: 'Discord', icon: '💬' },
        { key: 'telegram', label: 'Telegram', icon: '✈' }
    ];
    el.className = 'mi-tabs';
    el.innerHTML = tabs.map(t =>
        '<button class="mi-tab' + (miActiveTab === t.key ? ' active' : '') + '" onclick="switchMiTab(\'' + t.key + '\')">' +
            t.icon + ' ' + t.label +
        '</button>'
    ).join('');
}

function switchMiTab(tab) {
    miActiveTab = tab;
    renderMiTabs();
    if (tab === 'news') {
        const el = document.getElementById('miFilters');
        if (el) el.style.display = '';
        const filtered = miFilter === 'all' ? ALL_NEWS_ITEMS : ALL_NEWS_ITEMS.filter(n => n.sentClass === miFilter);
        renderFeed(filtered);
    } else {
        const el = document.getElementById('miFilters');
        if (el) el.style.display = 'none';
        renderSocialFeed(tab);
    }
}

// ═══ Social Feed Rendering ═══

function renderSocialFeed(platform) {
    const el = document.getElementById('miFeed');
    if (!el) return;
    const posts = socialCache[platform] || [];
    if (!posts.length) {
        el.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--dim);font-size:0.7rem;">No ' + platform + ' data yet...</div>';
        return;
    }

    let h = '';
    posts.forEach(p => {
        const ago = relTime(p.ts);
        const stats = platform === 'twitter'
            ? '♥ ' + p.likes + '  ↻ ' + p.retweets + '  💬 ' + p.replies
            : platform === 'discord'
            ? '👍 ' + p.likes + '  💬 ' + p.replies
            : '👁 ' + p.likes + '  💬 ' + p.replies;

        h += '<div class="mi-social-item">' +
            '<div class="mi-social-head">' +
                '<span class="mi-social-platform ' + platform + '">' + platform.toUpperCase().charAt(0) + '</span>' +
                '<span class="mi-social-handle">' + p.handle + '</span>' +
                '<span style="font-size:0.5rem;color:var(--dim)">' + p.name + '</span>' +
                '<span class="mi-social-time">' + ago + '</span>' +
            '</div>' +
            '<div class="mi-social-text">' + p.text + '</div>' +
            '<div class="mi-social-foot">' +
                '<span class="mi-social-stat">' + stats + '</span>' +
                '<span class="mi-sent ' + p.sentiment.cls + '" style="margin-left:auto;font-size:0.5rem">' + p.sentiment.label + '</span>' +
            '</div>' +
        '</div>';
    });
    el.innerHTML = h;
}

// Initialize social cache on load
initSocialCache();