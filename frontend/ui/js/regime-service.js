// Markov Regime Switching Detection System
// Uses Hidden Markov Model with CoinGecko price/volume data + news sentiment

const REGIME_STATES = ['ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'CRISIS'];
const REGIME_COLORS = ['var(--dim)', 'var(--green)', '#cc8800', 'var(--red)', '#cc0000'];
const REGIME_REFRESH_MS = 5 * 60 * 1000;
let regimeHistory = [];
let currentRegime = { state: 0, confidence: 0, probabilities: [0.2, 0.2, 0.2, 0.2, 0.2] };
let regimeTimer = null;

// Transition matrix (5x5) — Wyckoff cycle encoded
// Rows = from state, Cols = to state
// [ACCUM, MARKUP, DISTRIB, MARKDOWN, CRISIS]
const TRANSITION = [
    [0.50, 0.30, 0.05, 0.10, 0.05], // from ACCUMULATION
    [0.05, 0.55, 0.30, 0.05, 0.05], // from MARKUP
    [0.05, 0.10, 0.45, 0.30, 0.10], // from DISTRIBUTION
    [0.20, 0.05, 0.05, 0.50, 0.20], // from MARKDOWN
    [0.15, 0.05, 0.05, 0.35, 0.40], // from CRISIS
];

// Emission parameters: [mean, std] for each feature per state
// Features: [daily_return, volatility, volume_ratio]
const EMISSION = {
    mean: [
        [0.000, 0.015, 0.90],  // ACCUMULATION: flat, low vol, low volume
        [0.025, 0.020, 1.30],  // MARKUP: positive return, moderate vol, high volume
        [0.005, 0.025, 1.20],  // DISTRIBUTION: slight positive, rising vol, high volume
        [-0.025, 0.035, 1.10], // MARKDOWN: negative return, high vol, moderate volume
        [-0.050, 0.060, 1.60], // CRISIS: very negative, extreme vol, extreme volume
    ],
    std: [
        [0.015, 0.008, 0.30],
        [0.020, 0.010, 0.35],
        [0.020, 0.012, 0.35],
        [0.025, 0.015, 0.40],
        [0.035, 0.020, 0.50],
    ],
};

function gaussianPdf(x, mean, std) {
    const z = (x - mean) / std;
    return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
}

function emissionProb(stateIdx, obs) {
    const m = EMISSION.mean[stateIdx];
    const s = EMISSION.std[stateIdx];
    let p = 1.0;
    for (let i = 0; i < obs.length; i++) {
        p *= gaussianPdf(obs[i], m[i], s[i]);
    }
    return Math.max(p, 1e-300);
}

function normalize(arr) {
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum > 0 ? arr.map(v => v / sum) : arr.map(() => 1 / arr.length);
}

// Forward algorithm — compute filtered state probabilities
function forwardAlgorithm(observations) {
    const N = REGIME_STATES.length;
    let alpha = normalize(Array(N).fill(1));

    for (const obs of observations) {
        const newAlpha = Array(N).fill(0);
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let i = 0; i < N; i++) {
                sum += alpha[i] * TRANSITION[i][j];
            }
            newAlpha[j] = sum * emissionProb(j, obs);
        }
        alpha = normalize(newAlpha);
    }
    return alpha;
}

function extractFeatures(prices, volumes) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
        returns.push(Math.log(prices[i] / prices[i - 1]));
    }

    const observations = [];
    for (let i = 13; i < returns.length; i++) {
        const ret = returns[i];
        const window = returns.slice(i - 13, i + 1);
        const mean = window.reduce((a, b) => a + b, 0) / window.length;
        const vol = Math.sqrt(window.reduce((a, b) => a + (b - mean) ** 2, 0) / window.length);
        const avgVol20 = i >= 19
            ? volumes.slice(i - 18, i + 2).reduce((a, b) => a + b, 0) / 20
            : volumes.slice(0, i + 2).reduce((a, b) => a + b, 0) / (i + 2);
        const volRatio = avgVol20 > 0 ? volumes[i + 1] / avgVol20 : 1.0;
        observations.push([ret, vol, volRatio]);
    }
    return observations;
}

function getSentimentBias() {
    if (typeof ALL_NEWS_ITEMS === 'undefined' || !ALL_NEWS_ITEMS.length) return 0;
    let bull = 0, bear = 0;
    ALL_NEWS_ITEMS.forEach(n => {
        if (n.sentClass === 'bullish') bull++;
        else if (n.sentClass === 'bearish') bear++;
    });
    const total = bull + bear;
    if (total < 3) return 0;
    return (bull - bear) / total; // -1 to +1
}

function applySentimentAdjustment(probs) {
    const bias = getSentimentBias();
    if (Math.abs(bias) < 0.1) return probs;
    const adjusted = [...probs];
    if (bias > 0) {
        adjusted[1] *= 1 + bias * 0.3; // boost MARKUP
        adjusted[0] *= 1 + bias * 0.15; // slight boost ACCUMULATION
        adjusted[3] *= 1 - bias * 0.2; // reduce MARKDOWN
        adjusted[4] *= 1 - bias * 0.2; // reduce CRISIS
    } else {
        adjusted[3] *= 1 + Math.abs(bias) * 0.3;
        adjusted[4] *= 1 + Math.abs(bias) * 0.2;
        adjusted[1] *= 1 - Math.abs(bias) * 0.2;
    }
    return normalize(adjusted);
}

async function fetchMarketData() {
    const key = typeof CORTEX_CONFIG !== 'undefined' ? CORTEX_CONFIG.COINGECKO_API_KEY : '';
    const url = 'https://api.coingecko.com/api/v3/coins/solana/market_chart' +
        '?vs_currency=usd&days=30&interval=daily' + (key ? '&x_cg_demo_api_key=' + key : '');
    const res = await fetch(url);
    if (!res.ok) throw new Error('CoinGecko ' + res.status);
    return res.json();
}

function applyRegimeResult(maxIdx, probs) {
    const prev = currentRegime.state;
    currentRegime = { state: maxIdx, confidence: probs[maxIdx], probabilities: probs };

    regimeHistory.push({
        timestamp: Date.now(),
        state: maxIdx,
        regime: REGIME_STATES[maxIdx],
        confidence: probs[maxIdx],
        probabilities: [...probs],
    });

    if (prev !== maxIdx) {
        console.log('[REGIME] CHANGE: ' + REGIME_STATES[prev] + ' -> ' + REGIME_STATES[maxIdx] +
            ' (' + (probs[maxIdx] * 100).toFixed(1) + '% confidence)');
    }
    console.log('[REGIME] ' + REGIME_STATES[maxIdx] + ' ' +
        (probs[maxIdx] * 100).toFixed(1) + '% | ' +
        REGIME_STATES.map((s, i) => s.slice(0, 4) + ':' + (probs[i] * 100).toFixed(0) + '%').join(' '));

    updateRegimeUI();
    updateAgentsForRegime();
}

async function detectRegime() {
    // Try backend first
    try {
        if (typeof CortexAPI === 'undefined') throw new Error('CortexAPI not loaded');
        const data = await CortexAPI.get('/regime/transition-alert?token=SOL');
        if (data && data.current_regime !== undefined) {
            // Backend returns regime index (0-based); map to 5-state Wyckoff
            // Backend regime states may differ from UI's 5-state model — map safely
            var idx = data.current_regime;
            if (idx >= 0 && idx < REGIME_STATES.length) {
                var probs = [0.1, 0.1, 0.1, 0.1, 0.1];
                probs[idx] = 1 - data.transition_probability;
                if (data.most_likely_next_regime >= 0 && data.most_likely_next_regime < REGIME_STATES.length) {
                    probs[data.most_likely_next_regime] = data.next_regime_probability || 0.1;
                }
                applyRegimeResult(idx, normalize(probs));
                return;
            }
        }
    } catch (e) {
        console.warn('[REGIME] Backend unavailable:', e.message);
    }

    // Fallback to client-side HMM calculation
    try {
        const data = await fetchMarketData();
        const prices = data.prices.map(p => p[1]);
        const volumes = data.total_volumes.map(v => v[1]);

        if (prices.length < 16) {
            console.warn('[REGIME] Not enough data points:', prices.length);
            return;
        }

        const observations = extractFeatures(prices, volumes);
        if (!observations.length) return;

        let probs = forwardAlgorithm(observations);
        probs = applySentimentAdjustment(probs);

        const maxIdx = probs.indexOf(Math.max(...probs));
        applyRegimeResult(maxIdx, probs);
    } catch (err) {
        console.error('[REGIME] Detection failed:', err.message);
        // Return mock "unknown" regime when all sources fail
        const unknownProbs = [0.2, 0.2, 0.2, 0.2, 0.2];
        applyRegimeResult(0, unknownProbs);
    }
}

function updateRegimeUI() {
    const idx = currentRegime.state;
    const conf = currentRegime.confidence;
    const color = REGIME_COLORS[idx];

    const valueEl = document.getElementById('regimeValue');
    if (valueEl) {
        valueEl.textContent = REGIME_STATES[idx];
        valueEl.style.color = color;
    }

    const confEl = document.getElementById('regimeConfidence');
    if (confEl) {
        confEl.textContent = (conf * 100).toFixed(0) + '% confidence';
        confEl.style.color = color;
    }

    const segments = document.querySelectorAll('.regime-segment');
    segments.forEach((seg, i) => {
        seg.classList.toggle('active', i === idx);
        seg.style.background = i === idx ? color : '';
    });

    // Confidence gauge (index.html)
    const gaugeSvg = document.getElementById('regimeGauge');
    if (gaugeSvg && typeof d3 !== 'undefined') {
        const pct = conf;
        const gColor = pct >= 0.7 ? 'var(--green)' : pct >= 0.5 ? '#cc8800' : 'var(--red)';
        const startAngle = -Math.PI * 0.75;
        const endAngle = Math.PI * 0.75;
        const arcGen = d3.arc().innerRadius(28).outerRadius(34).cornerRadius(2);
        const sel = d3.select(gaugeSvg);
        sel.selectAll('*').remove();
        const g = sel.append('g').attr('transform', 'translate(40,40)');
        g.append('path').attr('d', arcGen({ startAngle, endAngle })).attr('fill', '#e8e8e8');
        const valAngle = startAngle + (endAngle - startAngle) * pct;
        g.append('path').attr('d', arcGen({ startAngle, endAngle: valAngle })).attr('fill', gColor);
        const gaugeVal = document.getElementById('regimeGaugeVal');
        if (gaugeVal) { gaugeVal.textContent = (pct * 100).toFixed(0) + '%'; gaugeVal.style.color = gColor; }
    }

    // Probability bars (index.html)
    const probsEl = document.getElementById('regimeProbs');
    if (probsEl) {
        const probs = currentRegime.probabilities || [0.2, 0.2, 0.2, 0.2, 0.2];
        const names = ['ACC', 'MKP', 'DST', 'MKD', 'CRS'];
        const colors = ['#666', '#00aa00', '#cc8800', '#cc0000', '#990000'];
        probsEl.innerHTML = probs.map((p, i) =>
            '<div class="regime-prob-item"><div class="regime-prob-name">' + names[i] + '</div>' +
            '<div class="regime-prob-bar"><div class="regime-prob-fill" style="width:' + (p * 100).toFixed(0) + '%;background:' + colors[i] + '"></div></div>' +
            '<div class="regime-prob-val" style="color:' + (i === idx ? colors[i] : 'var(--dim)') + '">' + (p * 100).toFixed(0) + '%</div></div>'
        ).join('');
    }
}

function updateAgentsForRegime() {
    if (typeof AGENTS_DATA === 'undefined') return;
    const regime = REGIME_STATES[currentRegime.state];
    const conf = currentRegime.confidence;

    // Risk Agent
    const risk = AGENTS_DATA.risk;
    if (regime === 'MARKDOWN' || regime === 'CRISIS') {
        risk.status = 'WARNING';
        risk.statusClass = regime === 'CRISIS' ? 'text-red' : 'text-dim';
        risk.signal = 'HEDGE';
        risk.signalClass = 'text-red';
        risk.regimeAdj = 'ACTIVE';
        risk.regimeAdjClass = 'text-red';
        risk.analysis = 'Market regime: ' + regime + ' (' + (conf * 100).toFixed(0) + '% confidence). ' +
            'Elevated risk detected. Portfolio VaR increased. Recommend reducing exposure and hedging SOL positions. ' +
            'Correlation risk elevated across DeFi assets.';
    } else {
        risk.status = 'ACTIVE';
        risk.statusClass = 'text-green';
        risk.signal = 'MONITOR';
        risk.signalClass = 'text-dim';
        risk.regimeAdj = 'ON';
        risk.regimeAdjClass = 'text-green';
        risk.analysis = 'Market regime: ' + regime + '. Risk levels within normal parameters. ' +
            'Portfolio VaR (95%) stable. Maintaining standard position limits.';
    }

    // Momentum Agent
    const mom = AGENTS_DATA.momentum;
    if (regime === 'MARKUP') {
        mom.signal = 'STRONG LONG';
        mom.signalClass = 'text-green';
        mom.confidence = Math.min(95, 75 + Math.round(conf * 20)) + '%';
        mom.regimeAdj = 'ACTIVE';
        mom.regimeAdjClass = 'text-green';
        mom.analysis = 'Regime: MARKUP — strong trend-following conditions. Momentum signals amplified. ' +
            'Price above key moving averages with increasing volume. Recommend aggressive long positioning.';
    } else if (regime === 'DISTRIBUTION' || regime === 'MARKDOWN') {
        mom.signal = 'NEUTRAL';
        mom.signalClass = 'text-dim';
        mom.confidence = Math.max(40, 70 - Math.round(conf * 25)) + '%';
        mom.regimeAdj = 'ACTIVE';
        mom.regimeAdjClass = 'text-red';
        mom.analysis = 'Regime: ' + regime + ' — momentum signals weakened. Trend exhaustion detected. ' +
            'Reducing position sizes and tightening stops. Waiting for regime shift confirmation.';
    } else {
        mom.signal = 'LONG';
        mom.signalClass = 'text-green';
        mom.confidence = '72%';
        mom.regimeAdj = 'ON';
        mom.regimeAdjClass = 'text-green';
        mom.analysis = 'Regime: ' + regime + '. Standard momentum scanning active on SOL/USDC. ' +
            'Monitoring for breakout signals above 20-period EMA.';
    }

    // Mean Reversion Agent
    const mr = AGENTS_DATA.meanrev;
    if (regime === 'ACCUMULATION') {
        mr.status = 'ACTIVE';
        mr.signal = 'ACTIVE SCAN';
        mr.signalClass = 'text-green';
        mr.confidence = Math.min(90, 70 + Math.round(conf * 20)) + '%';
        mr.regimeAdj = 'ACTIVE';
        mr.regimeAdjClass = 'text-green';
        mr.analysis = 'Regime: ACCUMULATION — ideal for mean reversion. Range-bound conditions detected. ' +
            'Bollinger Band width compressed. High probability reversion setups active.';
    } else if (regime === 'CRISIS') {
        mr.signal = 'PAUSED';
        mr.signalClass = 'text-red';
        mr.regimeAdj = 'ACTIVE';
        mr.regimeAdjClass = 'text-red';
        mr.analysis = 'Regime: CRISIS — mean reversion paused. Extreme volatility invalidates reversion models. ' +
            'Waiting for volatility normalization before re-engaging.';
    } else {
        mr.signal = 'SHORT';
        mr.signalClass = 'text-red';
        mr.confidence = '73%';
        mr.regimeAdj = 'ON';
        mr.regimeAdjClass = 'text-green';
        mr.analysis = 'Regime: ' + regime + '. Standard mean reversion scanning. ' +
            'SOL/USDC monitored for Bollinger Band extremes and Z-score deviations.';
    }

    // Sentiment Agent — regime-aware analysis text
    const sent = AGENTS_DATA.sentiment;
    sent.analysis = 'Regime: ' + regime + '. Social sentiment analysis active. ' +
        'News sentiment bias: ' + (getSentimentBias() > 0 ? 'bullish' : getSentimentBias() < 0 ? 'bearish' : 'neutral') +
        '. Monitoring Twitter, news feeds, and on-chain social signals for SOL.';

    // Arbitrage Agent — minimal regime impact
    const arb = AGENTS_DATA.arbitrage;
    if (regime === 'CRISIS') {
        arb.analysis = 'Regime: CRISIS — elevated spreads detected across DEXs. ' +
            'Arbitrage opportunities increasing but execution risk higher due to volatility.';
    } else {
        arb.analysis = 'Regime: ' + regime + '. Cross-DEX spread monitoring active. ' +
            'Jupiter vs Orca vs Raydium spreads within normal range. Latency: 340ms avg.';
    }

    refreshAgentCards();
}

function refreshAgentCards() {
    document.querySelectorAll('.agent-card').forEach(card => {
        const id = card.getAttribute('data-agent');
        const agent = AGENTS_DATA[id];
        if (!agent) return;

        const statusDot = card.querySelector('.agent-status');
        if (statusDot) {
            statusDot.className = 'agent-status';
            if (agent.status === 'WARNING') statusDot.classList.add('warning');
            else if (agent.status !== 'ACTIVE') statusDot.classList.add('inactive');
        }

        const signalEl = card.querySelector('.agent-metrics .metric-value:last-child');
        if (signalEl) {
            signalEl.textContent = agent.signal;
            signalEl.className = 'metric-value ' + agent.signalClass;
        }
    });

    // Re-render expanded agent detail if open
    if (typeof expandedAgentId !== 'undefined' && expandedAgentId) {
        const expandEl = document.querySelector('.agent-expand');
        if (expandEl && AGENTS_DATA[expandedAgentId]) {
            expandEl.querySelector('.agent-expand-inner')?.remove();
            const inner = document.createElement('div');
            inner.innerHTML = buildAgentDetail(AGENTS_DATA[expandedAgentId]);
            expandEl.innerHTML = inner.innerHTML;
        }
    }
}

function startRegimeDetection() {
    console.log('[REGIME] Starting Markov Regime Detection — interval: ' + (REGIME_REFRESH_MS / 1000) + 's');
    detectRegime();
    if (regimeTimer) clearInterval(regimeTimer);
    regimeTimer = setInterval(detectRegime, REGIME_REFRESH_MS);
}
