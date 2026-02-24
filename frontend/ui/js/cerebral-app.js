// Seeded PRNG (mulberry32) — deterministic random for visualizations
var _cxSeed = 42;
function cxRandom() {
    _cxSeed = (_cxSeed + 0x6D2B79F5) | 0;
    var t = Math.imul(_cxSeed ^ (_cxSeed >>> 15), 1 | _cxSeed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function cxSeed(s) { _cxSeed = s | 0; }
// Deterministic hash for index-based seeding
function cxHash(i) { return (i * 2654435761) >>> 0; }
// === 1. DECISION FLOW PIPELINE — Sankey Diagram ===
(function() {
    const container = document.getElementById('sankeyChart');
    const width = container.clientWidth || 928;
    const height = 420;

    // Pipeline flow data: nodes = processing stages, links = signal/data volume
    const SANKEY_NODES = [
        { name: 'Market Data' },
        { name: 'On-Chain RPC' },
        { name: 'News Feeds' },
        { name: 'ML Regime Detection' },
        { name: 'Analyst Reports' },
        { name: 'Sentiment Engine' },
        { name: 'Adversarial Debate' },
        { name: 'Trade Proposal' },
        { name: 'Risk Assessment' },
        { name: 'Final Approval' },
        { name: 'Guardian Exec' },
        { name: 'Live Execution' },
        { name: 'Rejected' },
    ];

    // Static architectural flow values (schematic, not real-time data)
    const BASE_LINKS = [
        { source: 'Market Data', target: 'ML Regime Detection', value: 80 },
        { source: 'Market Data', target: 'Analyst Reports', value: 45 },
        { source: 'On-Chain RPC', target: 'ML Regime Detection', value: 55 },
        { source: 'On-Chain RPC', target: 'Analyst Reports', value: 30 },
        { source: 'News Feeds', target: 'Sentiment Engine', value: 60 },
        { source: 'News Feeds', target: 'Analyst Reports', value: 20 },
        { source: 'ML Regime Detection', target: 'Adversarial Debate', value: 65 },
        { source: 'ML Regime Detection', target: 'Risk Assessment', value: 40 },
        { source: 'Analyst Reports', target: 'Adversarial Debate', value: 70 },
        { source: 'Sentiment Engine', target: 'Adversarial Debate', value: 40 },
        { source: 'Adversarial Debate', target: 'Trade Proposal', value: 130 },
        { source: 'Trade Proposal', target: 'Risk Assessment', value: 80 },
        { source: 'Trade Proposal', target: 'Final Approval', value: 35 },
        { source: 'Risk Assessment', target: 'Final Approval', value: 100 },
        { source: 'Final Approval', target: 'Guardian Exec', value: 105 },
        { source: 'Final Approval', target: 'Rejected', value: 18 },
        { source: 'Guardian Exec', target: 'Live Execution', value: 95 },
    ];

    // Monochromatic node colors: darker for later stages
    const NODE_COLORS = {
        'Market Data': '#aaaaaa', 'On-Chain RPC': '#aaaaaa', 'News Feeds': '#aaaaaa',
        'ML Regime Detection': '#888888', 'Analyst Reports': '#888888', 'Sentiment Engine': '#888888',
        'Adversarial Debate': '#666666', 'Trade Proposal': '#555555',
        'Risk Assessment': '#444444', 'Final Approval': '#333333',
        'Guardian Exec': '#222222', 'Live Execution': '#080808', 'Rejected': '#cc0000',
    };

    function renderSankey() {
        container.textContent = '';

        const nodes = SANKEY_NODES.map(d => Object.assign({}, d));
        const links = BASE_LINKS.map(d => ({ source: d.source, target: d.target, value: d.value }));

        const sankey = d3.sankey()
            .nodeId(d => d.name)
            .nodeAlign(d3.sankeyLeft)
            .nodeWidth(14)
            .nodePadding(16)
            .extent([[1, 8], [width - 1, height - 8]]);

        const graph = sankey({ nodes, links });

        const svg = d3.select(container).append('svg')
            .attr('viewBox', [0, 0, width, height])
            .attr('width', width)
            .attr('height', height)
            .attr('style', 'max-width: 100%; height: auto;');

        // Unique gradient IDs
        const uid = (name) => 'sk-' + name.replace(/\s+/g, '-') + '-' + Date.now();

        // Links with source→target gradient
        const link = svg.append('g')
            .attr('fill', 'none')
            .attr('stroke-opacity', 0.25)
          .selectAll('g')
          .data(graph.links)
          .join('g')
            .attr('class', 'sankey-link');

        const gradient = link.append('linearGradient')
            .attr('id', (d, i) => (d.uid = 'skgrad' + i))
            .attr('gradientUnits', 'userSpaceOnUse')
            .attr('x1', d => d.source.x1)
            .attr('x2', d => d.target.x0);
        gradient.append('stop').attr('offset', '0%').attr('stop-color', d => NODE_COLORS[d.source.name] || '#888');
        gradient.append('stop').attr('offset', '100%').attr('stop-color', d => NODE_COLORS[d.target.name] || '#888');

        link.append('path')
            .attr('d', d3.sankeyLinkHorizontal())
            .attr('stroke', d => 'url(#' + d.uid + ')')
            .attr('stroke-width', d => Math.max(1, d.width));

        link.append('title')
            .text(d => d.source.name + ' → ' + d.target.name + '\n' + d.value + ' signals');

        // Nodes
        svg.append('g')
          .selectAll('rect')
          .data(graph.nodes)
          .join('rect')
            .attr('class', 'sankey-node')
            .attr('x', d => d.x0)
            .attr('y', d => d.y0)
            .attr('height', d => Math.max(1, d.y1 - d.y0))
            .attr('width', d => d.x1 - d.x0)
            .attr('fill', d => NODE_COLORS[d.name] || '#666')
            .attr('stroke', 'none')
          .append('title')
            .text(d => d.name + '\n' + d.value + ' signals');

        // Labels
        svg.append('g')
          .selectAll('text')
          .data(graph.nodes)
          .join('text')
            .attr('class', 'sankey-label')
            .attr('x', d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
            .attr('y', d => (d.y1 + d.y0) / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', d => d.x0 < width / 2 ? 'start' : 'end')
            .text(d => d.name)
          .clone(true).lower()
            .attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', 3).attr('stroke-linejoin', 'round');

        // Update cycle time display
        const totalSignals = graph.links.reduce((a, l) => a + l.value, 0);
        document.getElementById('dpCycleTime').textContent = 'Total Cycle: 3,740ms · ' + totalSignals + ' signals';
    }

    renderSankey();
})();

// === 2. ADVERSARIAL DEBATE THEATER ===
var debateArchive = [];

function renderDebate(archiveIdx) {
    var isArchive = typeof archiveIdx === 'number';
    var scenario = isArchive ? debateArchive[archiveIdx] : (debateArchive.length ? debateArchive[0] : null);
    var container = document.getElementById('dtContainer');
    if (!scenario) {
        container.innerHTML = '<div style="color:var(--dim);padding:2rem;text-align:center">No debate data available</div>';
        document.getElementById('dtTimestamp').textContent = '--';
        document.getElementById('dtSummary').innerHTML = '';
        return;
    }
    document.getElementById('dtTimestamp').textContent = scenario.trigger;

    // Rounds with score progress bars
    var html = '<div class="dt-rounds">';
    scenario.rounds.forEach(function(r, i) {
        var speakerColor = r.role === 'Challenge' ? 'var(--red)' : r.role === 'Decision' ? 'var(--fg)' : 'var(--dim)';
        var barColor = r.score >= 75 ? 'var(--green)' : r.score >= 50 ? '#cc8800' : 'var(--red)';
        var scoreColor = r.score >= 75 ? 'var(--green)' : r.score >= 50 ? 'var(--dim)' : 'var(--red)';
        html += '<div class="dt-round">' +
            '<div><div class="dt-speaker" style="color:' + speakerColor + '">' + r.speaker + '</div><div class="dt-speaker-role">Round ' + (i + 1) + ' · ' + r.role + '</div></div>' +
            '<div class="dt-text">' + r.text + '</div>' +
            '<div><div class="dt-score" style="color:' + scoreColor + '">' + r.score + '%<br><span style="font-weight:400;font-size:0.55rem">' + r.scoreLabel + '</span></div>' +
            '<div class="dt-score-bar"><div class="dt-score-track"><div class="dt-score-fill" style="width:' + r.score + '%;background:' + barColor + '"></div></div></div></div>' +
            '</div>';
    });
    html += '</div>';

    // Verdict
    html += '<div class="dt-verdict"><span>Verdict: <strong class="' + scenario.verdictClass + '">' + scenario.verdict + '</strong></span><span>Confidence: ' + scenario.confidence + '%</span></div>';
    container.innerHTML = html;

    // Summary cards
    var avgScore = Math.round(scenario.rounds.reduce(function(s, r) { return s + r.score; }, 0) / scenario.rounds.length);
    var riskRound = scenario.rounds.find(function(r) { return r.role === 'Challenge'; });
    var riskScore = riskRound ? riskRound.score : 0;
    var finalRound = scenario.rounds.find(function(r) { return r.role === 'Decision'; });
    var finalScore = finalRound ? finalRound.score : 0;
    var execStatus = scenario.verdict.indexOf('APPROVED') >= 0 ? 'EXECUTING' : 'BLOCKED';
    var execColor = execStatus === 'EXECUTING' ? 'var(--green)' : 'var(--red)';

    var summaryEl = document.getElementById('dtSummary');
    summaryEl.innerHTML = '<div class="dt-summary">' +
        '<div class="dt-summary-card"><div class="dt-summary-label">Rounds</div><div class="dt-summary-val">' + scenario.rounds.length + '</div></div>' +
        '<div class="dt-summary-card"><div class="dt-summary-label">Avg Score</div><div class="dt-summary-val" style="color:' + (avgScore >= 75 ? 'var(--green)' : avgScore >= 50 ? '#cc8800' : 'var(--red)') + '">' + avgScore + '%</div></div>' +
        '<div class="dt-summary-card"><div class="dt-summary-label">Risk Assessment</div><div class="dt-summary-val" style="color:' + (riskScore >= 75 ? 'var(--red)' : riskScore >= 50 ? '#cc8800' : 'var(--green)') + '">' + riskScore + '%</div></div>' +
        '<div class="dt-summary-card"><div class="dt-summary-label">Execution</div><div class="dt-summary-val" style="color:' + execColor + '">' + execStatus + '</div></div>' +
        '</div>';
}



function renderArchiveList(activeIdx) {
    var el = document.getElementById('dtArchive');
    if (!debateArchive.length) { el.innerHTML = ''; return; }
    var html = '<div class="dt-archive"><div class="dt-archive-header">Debate Archive (' + debateArchive.length + ')</div>';
    debateArchive.forEach(function(d, i) {
        var verdictColor = d.verdictClass === 'text-green' ? 'var(--green)' : d.verdictClass === 'text-red' ? 'var(--red)' : 'var(--dim)';
        html += '<div class="dt-archive-item' + (i === activeIdx ? ' active' : '') + '" onclick="loadArchiveDebate(' + i + ')">' +
            '<div class="dt-archive-time">' + d.timestamp + '</div>' +
            '<div class="dt-archive-trigger">' + d.trigger + '</div>' +
            '<div class="dt-archive-verdict" style="color:' + verdictColor + '">' + d.verdict + '</div>' +
            '</div>';
    });
    html += '</div>';
    el.innerHTML = html;
}

function loadArchiveDebate(idx) {
    renderDebate(idx);
    renderArchiveList(idx);
}

// --- Live API integration for debates ---
async function fetchLiveDebate() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var data = await CerebralAPI.fetchRecentDebates(1);
        if (!data || !data.transcripts || !data.transcripts.length) return false;
        var t = data.transcripts[0];
        var rounds = (t.rounds || []).map(function(r, i) {
            return {
                speaker: r.trader ? 'Trader Agent' : 'Agent',
                role: i === 0 ? 'Proposal' : i === (t.rounds.length - 1) ? 'Decision' : 'Challenge',
                text: [
                    r.trader ? (r.trader.arguments || []).join(' ') : '',
                    r.risk_manager ? (r.risk_manager.arguments || []).join(' ') : '',
                    r.arbitrator ? (r.arbitrator.reasoning || []).join(' ') : '',
                ].filter(Boolean).join(' | ') || 'No transcript available',
                score: Math.round((r.arbitrator ? r.arbitrator.confidence : (r.trader ? r.trader.confidence : 0.5)) * 100),
                scoreLabel: i === 0 ? 'Conviction' : i === (t.rounds.length - 1) ? 'Final' : 'Risk Score',
            };
        });
        if (!rounds.length) return false;
        var scenario = {
            trigger: (t.token || 'SOL') + ' — ' + (t.direction || 'trade') + ' $' + (t.trade_size_usd || '?'),
            rounds: rounds,
            verdict: t.final_decision === 'approve' ? 'APPROVED' : 'REJECTED',
            verdictClass: t.final_decision === 'approve' ? 'text-green' : 'text-red',
            confidence: Math.round((t.final_confidence || 0.5) * 100),
        };
        // Push to archive and render
        var now = new Date();
        var ts = ('0' + now.getHours()).slice(-2) + ':' + ('0' + now.getMinutes()).slice(-2) + ':' + ('0' + now.getSeconds()).slice(-2);
        debateArchive.unshift(Object.assign({ timestamp: ts }, scenario));
        if (debateArchive.length > 10) debateArchive.pop();
        renderDebate();
        renderArchiveList(-1);
        console.log('[Cerebral] Live debate loaded:', scenario.trigger);
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live debate fetch failed:', e.message);
        return false;
    }
}

var _debateLiveActive = false;

// Initial render (empty until API responds)
renderDebate();

// Try live debates, show empty state if unavailable
(async function() {
    var live = await fetchLiveDebate();
    _debateLiveActive = live;
    var statusEl = document.getElementById('dtLiveStatus');
    if (statusEl) statusEl.textContent = live ? 'LIVE' : 'Offline';
    if (!live) {
        console.log('[Cerebral] No live debates available');
        renderDebate();
    }
})();

setInterval(async function() {
    var live = await fetchLiveDebate();
    _debateLiveActive = live;
    var statusEl = document.getElementById('dtLiveStatus');
    if (statusEl) statusEl.textContent = live ? 'LIVE' : 'Offline';
    if (!live) renderDebate();
}, 10000);

// === 3. MARKOV REGIME ENGINE — Chord Diagram ===
const RG_STATES = ['ACCUM', 'MARKUP', 'DISTRIB', 'MARKDN', 'CRISIS'];
const RG_FULL = ['ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'CRISIS'];
const RG_COLORS = ['var(--dim)', 'var(--green)', '#cc8800', 'var(--red)', '#cc0000'];
const RG_HEX = ['#666666', '#00aa00', '#cc8800', '#cc0000', '#990000'];

function renderRegimeChord() {
    const container = document.getElementById('rgChord');
    container.innerHTML = '';
    const w = Math.min(container.clientWidth || 500, 600);
    const h = w;
    const outerRadius = w * 0.5 - 36;
    const innerRadius = outerRadius - 16;

    // Scale transition probabilities to integers for chord layout
    const matrix = TRANSITION.map(row => row.map(v => Math.round(v * 1000)));

    const chord = d3.chord()
        .padAngle(18 / innerRadius)
        .sortSubgroups(d3.descending);

    const arc = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);
    const ribbon = d3.ribbon().radius(innerRadius);

    const svg = d3.select(container).append('svg')
        .attr('viewBox', [-w / 2, -h / 2, w, h])
        .attr('width', w).attr('height', h)
        .attr('style', 'max-width: 100%; height: auto; font: 9px var(--font-mono);');

    const chords = chord(matrix);

    // Arcs for each regime state
    const group = svg.append('g').selectAll('g').data(chords.groups).join('g');

    group.append('path')
        .attr('fill', d => RG_HEX[d.index])
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.5)
        .attr('d', arc)
      .append('title')
        .text(d => RG_FULL[d.index] + ': ' + (d.value / 10).toFixed(1) + '%');

    // Ticks on arcs
    const tickStep = 50;
    function groupTicks(d, step) {
        const k = (d.endAngle - d.startAngle) / d.value;
        return d3.range(0, d.value, step).map(v => ({ value: v, angle: v * k + d.startAngle }));
    }

    const tick = group.append('g').selectAll('g')
        .data(d => groupTicks(d, tickStep)).join('g')
        .attr('transform', d => 'rotate(' + (d.angle * 180 / Math.PI - 90) + ') translate(' + outerRadius + ',0)');
    tick.append('line').attr('stroke', 'var(--dim)').attr('x2', 4);
    tick.filter(d => d.value % 200 === 0).append('text')
        .attr('x', 6).attr('dy', '.35em')
        .attr('transform', d => d.angle > Math.PI ? 'rotate(180) translate(-12)' : null)
        .attr('text-anchor', d => d.angle > Math.PI ? 'end' : null)
        .attr('fill', 'var(--dim)').attr('font-size', '8px')
        .text(d => (d.value / 10) + '%');

    // Regime name labels on arcs
    group.append('text')
        .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
        .attr('dy', '.35em')
        .attr('transform', d => 'rotate(' + (d.angle * 180 / Math.PI - 90) + ') translate(' + (outerRadius + 20) + ')' + (d.angle > Math.PI ? ' rotate(180)' : ''))
        .attr('text-anchor', d => d.angle > Math.PI ? 'end' : 'start')
        .attr('font-size', '9px').attr('font-weight', '700').attr('fill', 'var(--fg)')
        .text(d => RG_STATES[d.index]);

    // Ribbons showing transitions
    svg.append('g')
        .attr('fill-opacity', 0.55)
      .selectAll('path')
      .data(chords)
      .join('path')
        .attr('d', ribbon)
        .attr('fill', d => RG_HEX[d.target.index])
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.3)
      .append('title')
        .text(d => RG_FULL[d.source.index] + ' → ' + RG_FULL[d.target.index] + ': ' + (d.source.value / 10).toFixed(1) + '%'
            + (d.source.index !== d.target.index ? '\n' + RG_FULL[d.target.index] + ' → ' + RG_FULL[d.source.index] + ': ' + (d.target.value / 10).toFixed(1) + '%' : ''));

    // Highlight current regime arc
    const idx = currentRegime.state;
    group.selectAll('path').attr('opacity', (d, i) => i === idx ? 1 : 0.7);
}

// Render KaTeX formula after KaTeX loads
function renderHMMFormula() {
    const el = document.getElementById('rgFormula');
    if (typeof katex !== 'undefined') {
        katex.render(
            '\\alpha_t(j) = \\sum_{i=1}^{N} \\alpha_{t-1}(i) \\cdot a_{ij} \\cdot b_j(o_t)',
            el, { displayMode: true, throwOnError: false }
        );
    } else {
        setTimeout(renderHMMFormula, 200);
    }
}
renderHMMFormula();

// Render contextual KaTeX formulas across all sections
function renderAllFormulas() {
    if (typeof katex === 'undefined') { setTimeout(renderAllFormulas, 200); return; }
    const formulas = {
        fSankey:  'T_{\\text{cycle}} = \\sum_{i=1}^{L} \\tau_i + \\delta_{\\text{queue}}',
        fDebate:  'S_{\\text{final}} = \\frac{\\sum_{r=1}^{R} w_r \\cdot s_r}{\\sum_{r=1}^{R} w_r}',
        fLoad:    'L_i = \\frac{Q_i}{\\mu_i} + \\lambda_i \\cdot \\rho_i',
        fBreaker: 'CB_i = \\mathbb{1}\\!\\left[\\sum_{t=1}^{T} f_t \\geq \\theta_i\\right]',
        fCorr:    '\\rho_{xy} = \\frac{\\operatorname{Cov}(X,Y)}{\\sigma_X \\, \\sigma_Y}',
        fDrift:   '\\theta_{t+1} = \\theta_t + \\eta \\cdot \\nabla_\\theta J(\\theta_t) + \\epsilon_t'
    };
    for (const [id, tex] of Object.entries(formulas)) {
        const el = document.getElementById(id);
        if (el) katex.render(tex, el, { displayMode: true, throwOnError: false });
    }
}
renderAllFormulas();

function updateRegimeDisplay() {
    const idx = currentRegime.state;
    const conf = currentRegime.confidence;
    const badge = document.getElementById('rgStateBadge');
    badge.textContent = RG_FULL[idx];
    badge.style.color = RG_COLORS[idx];
    document.getElementById('rgConf').textContent = (conf * 100).toFixed(1) + '% confidence';
    document.getElementById('rgLastUpdate').textContent = new Date().toLocaleTimeString();

    const REGIME_PARAMS = {
        0: { posSize: '5%', stopLoss: '5%', entryConf: '60%', riskLimit: '100%' },
        1: { posSize: '7%', stopLoss: '4%', entryConf: '55%', riskLimit: '120%' },
        2: { posSize: '3%', stopLoss: '6%', entryConf: '70%', riskLimit: '80%' },
        3: { posSize: '2%', stopLoss: '7%', entryConf: '75%', riskLimit: '60%' },
        4: { posSize: '1%', stopLoss: '10%', entryConf: '85%', riskLimit: '30%' },
    };
    const p = REGIME_PARAMS[idx];
    const params = document.getElementById('rgParams');
    params.innerHTML = [
        { label: 'Position Size', value: p.posSize },
        { label: 'Stop-Loss', value: p.stopLoss },
        { label: 'Entry Confidence', value: p.entryConf },
        { label: 'Risk Limit', value: p.riskLimit },
    ].map(x => '<div class="rg-param"><div class="rg-param-label">' + x.label + '</div><div class="rg-param-value">' + x.value + '</div></div>').join('');

    renderRegimeTimeline();
    renderRegimeChord();
}

function renderRegimeTimeline() {
    const tl = document.getElementById('rgTimeline');
    if (regimeHistory.length < 2) {
        tl.innerHTML = '<div style="padding:8px 1rem;font-size:0.6rem;color:var(--dim)">Collecting regime data...</div>';
        return;
    }
    tl.innerHTML = '';
    regimeHistory.forEach(h => {
        const seg = document.createElement('div');
        seg.className = 'rg-tl-seg';
        seg.style.background = RG_COLORS[h.state];
        seg.style.opacity = '0.6';
        seg.title = h.regime + ' (' + (h.confidence * 100).toFixed(0) + '%)';
        tl.appendChild(seg);
    });
}

// === 3b. REGIME ACCURACY CHART ===
var rgAccHistory = [];
(function seedAccuracy() {
    // Deterministic seed using sine-based pseudo-variation
    for (var i = 0; i < 20; i++) {
        var t = i / 20;
        rgAccHistory.push({
            cycle: i + 1,
            accuracy: 65 + Math.sin(i * 0.7) * 15 + i * 0.8,
            falsePos: 7 + Math.sin(i * 1.1 + 2) * 4,
            stability: 55 + Math.sin(i * 0.5 + 1) * 18 + i * 0.5
        });
    }
})();

function renderAccuracyChart() {
    var container = document.getElementById('rgAccuracyChart');
    if (!container || typeof d3 === 'undefined') return;
    container.innerHTML = '';
    var W = Math.min(container.clientWidth || 400, 500), H = 120;
    var m = { t: 10, r: 10, b: 22, l: 30 };
    var svg = d3.select(container).append('svg').attr('width', W).attr('height', H);
    var x = d3.scaleLinear().domain([1, rgAccHistory.length]).range([m.l, W - m.r]);
    var y = d3.scaleLinear().domain([0, 100]).range([H - m.b, m.t]);

    svg.append('g').attr('transform', 'translate(0,' + (H - m.b) + ')')
        .call(d3.axisBottom(x).ticks(5).tickFormat(d => '#' + d)).selectAll('text').attr('font-size', '7px');
    svg.append('g').attr('transform', 'translate(' + m.l + ',0)')
        .call(d3.axisLeft(y).ticks(4).tickFormat(d => d + '%')).selectAll('text').attr('font-size', '7px');

    var lineGen = function(key) { return d3.line().x(d => x(d.cycle)).y(d => y(d[key])).curve(d3.curveMonotoneX); };

    var series = [
        { key: 'accuracy', color: 'var(--green)', label: 'Accuracy' },
        { key: 'falsePos', color: 'var(--red)', label: 'False Pos' },
        { key: 'stability', color: '#cc8800', label: 'Stability' }
    ];

    series.forEach(function(s) {
        svg.append('path').datum(rgAccHistory).attr('d', lineGen(s.key))
            .attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 1.5);
    });

    var legend = svg.append('g').attr('transform', 'translate(' + (m.l + 5) + ',' + (m.t + 2) + ')');
    series.forEach(function(s, i) {
        legend.append('line').attr('x1', 0).attr('x2', 12).attr('y1', i * 10).attr('y2', i * 10)
            .attr('stroke', s.color).attr('stroke-width', 1.5);
        legend.append('text').attr('x', 15).attr('y', i * 10 + 3).text(s.label)
            .attr('font-size', '6px').attr('fill', 'var(--dim)');
    });
}

// === 3c. REGIME TRANSITION HEATMAP ===
var observedTransitions = TRANSITION.map(function(row) { return row.map(function(v) { return v; }); });

function renderTransitionHeatmap() {
    var container = document.getElementById('rgHeatmap');
    if (!container || typeof d3 === 'undefined') return;
    container.innerHTML = '';
    var cellSize = 42, pad = 50, labelW = 48;
    var W = labelW + cellSize * 5 + pad, H = labelW + cellSize * 5 + 10;
    var svg = d3.select(container).append('svg').attr('width', W).attr('height', H);
    var g = svg.append('g').attr('transform', 'translate(' + labelW + ',' + labelW + ')');

    var maxVal = 0;
    observedTransitions.forEach(function(row) { row.forEach(function(v) { if (v > maxVal) maxVal = v; }); });
    var colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxVal]);

    for (var i = 0; i < 5; i++) {
        for (var j = 0; j < 5; j++) {
            var val = observedTransitions[i][j];
            g.append('rect').attr('x', j * cellSize).attr('y', i * cellSize)
                .attr('width', cellSize - 1).attr('height', cellSize - 1)
                .attr('fill', colorScale(val)).attr('stroke', '#fff').attr('stroke-width', 0.5)
                .attr('opacity', (i === currentRegime.state) ? 1 : 0.8);
            g.append('text').attr('x', j * cellSize + cellSize / 2).attr('y', i * cellSize + cellSize / 2 + 3)
                .attr('text-anchor', 'middle').attr('font-size', '8px').attr('font-weight', '600')
                .attr('fill', val > maxVal * 0.6 ? '#fff' : '#222')
                .text((val * 100).toFixed(0) + '%');
        }
    }

    // Row labels (From)
    RG_STATES.forEach(function(s, i) {
        svg.append('text').attr('x', labelW - 4).attr('y', labelW + i * cellSize + cellSize / 2 + 3)
            .attr('text-anchor', 'end').attr('font-size', '7px').attr('font-weight', i === currentRegime.state ? '700' : '400')
            .attr('fill', i === currentRegime.state ? RG_HEX[i] : 'var(--dim)').text(s);
    });

    // Column labels (To)
    RG_STATES.forEach(function(s, j) {
        svg.append('text').attr('x', labelW + j * cellSize + cellSize / 2).attr('y', labelW - 6)
            .attr('text-anchor', 'middle').attr('font-size', '7px').attr('font-weight', j === currentRegime.state ? '700' : '400')
            .attr('fill', j === currentRegime.state ? RG_HEX[j] : 'var(--dim)').text(s);
    });

    svg.append('text').attr('x', 2).attr('y', labelW + cellSize * 2.5).attr('font-size', '6px')
        .attr('fill', 'var(--dim)').attr('transform', 'rotate(-90,6,' + (labelW + cellSize * 2.5) + ')').text('FROM');
    svg.append('text').attr('x', labelW + cellSize * 2.5).attr('y', labelW - 16).attr('text-anchor', 'middle')
        .attr('font-size', '6px').attr('fill', 'var(--dim)').text('TO');
}

// Evolve observed transitions when regime changes
var prevRegimeForHeatmap = currentRegime.state;
function evolveTransitions() {
    var curr = currentRegime.state;
    if (curr !== prevRegimeForHeatmap) {
        var from = prevRegimeForHeatmap, to = curr;
        observedTransitions[from][to] = Math.min(0.99, observedTransitions[from][to] + 0.01);
        // Re-normalize row
        var rowSum = observedTransitions[from].reduce(function(a, b) { return a + b; }, 0);
        observedTransitions[from] = observedTransitions[from].map(function(v) { return v / rowSum; });
        prevRegimeForHeatmap = curr;
    }
    // Add accuracy data point — use live regime statistics when available
    if (rgAccHistory.length >= 30) rgAccHistory.shift();
    var liveStats = window._liveRegimeStats;
    var prev = rgAccHistory[rgAccHistory.length - 1] || { cycle: 0, accuracy: 75, falsePos: 8, stability: 70 };
    if (liveStats && liveStats.accuracy !== undefined) {
        rgAccHistory.push({
            cycle: prev.cycle + 1,
            accuracy: Math.max(40, Math.min(98, liveStats.accuracy * 100 || prev.accuracy)),
            falsePos: Math.max(0, Math.min(25, liveStats.false_positive_rate * 100 || prev.falsePos)),
            stability: Math.max(30, Math.min(95, liveStats.stability * 100 || prev.stability)),
        });
    } else {
        // Fallback: deterministic drift based on regime confidence
        var conf = currentRegime.confidence || 0.5;
        var drift = (conf - 0.5) * 2; // -1 to 1
        rgAccHistory.push({
            cycle: prev.cycle + 1,
            accuracy: Math.max(40, Math.min(98, prev.accuracy + drift * 2)),
            falsePos: Math.max(0, Math.min(25, prev.falsePos - drift * 1.5)),
            stability: Math.max(30, Math.min(95, prev.stability + drift * 1.8)),
        });
    }
    renderAccuracyChart();
    renderTransitionHeatmap();
}

renderRegimeChord();
renderAccuracyChart();
renderTransitionHeatmap();
setInterval(evolveTransitions, 8000);

// === 4. ML SIGNAL DASHBOARD ===
const ML_SIGNALS = [
    { name: 'Momentum', key: 'momentum' },
    { name: 'Mean Reversion', key: 'meanrev' },
    { name: 'Volatility', key: 'volatility' },
    { name: 'Sentiment', key: 'sentiment' },
];

function renderMLSignals() {
    const grid = document.getElementById('mlGrid');
    const regime = currentRegime.state;
    const signals = generateMLSignals(regime);
    var mlStatus = document.getElementById('mlSignalStatus');
    if (mlStatus) mlStatus.textContent = (window._liveAgentData && window._liveAgentData.agents) ? 'LIVE' : 'Offline';
    grid.innerHTML = signals.map(s => {
        const color = s.value >= 70 ? 'var(--green)' : s.value >= 40 ? 'var(--dim)' : 'var(--red)';
        return '<div class="ml-signal">' +
            '<div class="ml-signal-name">' + s.name + '</div>' +
            '<div class="ml-bar-track"><div class="ml-bar-fill" style="width:' + s.value + '%;background:' + color + '"></div></div>' +
            '<div class="ml-signal-val" style="color:' + color + '">' + s.value + '% <span style="font-weight:400;font-size:0.6rem">' + s.direction + '</span></div>' +
            '</div>';
    }).join('');
}

function generateMLSignals(regime) {
    // Use live agent status data when available
    var liveAgents = window._liveAgentData;
    if (liveAgents && liveAgents.agents) {
        var a = liveAgents.agents;
        var mom = a.momentum || {};
        var mr = a.meanrev || {};
        var risk = a.risk || {};
        var sent = a.sentiment || {};
        return [
            { name: 'Momentum', value: parseInt(mom.confidence) || 50, direction: mom.signal || 'Neutral' },
            { name: 'Mean Reversion', value: parseInt(mr.confidence) || 50, direction: mr.signal || 'Neutral' },
            { name: 'Volatility', value: parseInt(risk.confidence) || 50, direction: risk.signal || 'Monitor' },
            { name: 'Sentiment', value: parseInt(sent.confidence) || 50, direction: sent.signal || 'Neutral' },
        ];
    }
    // Fallback: regime-based deterministic values
    const base = {
        0: [55, 65, 30, 50],
        1: [82, 35, 45, 72],
        2: [48, 58, 62, 42],
        3: [30, 72, 78, 28],
        4: [15, 45, 95, 18],
    };
    const dirs = {
        0: ['Neutral', 'Active Scan', 'Low', 'Neutral'],
        1: ['Strong Long', 'Weak', 'Moderate', 'Bullish'],
        2: ['Fading', 'Active', 'Rising', 'Bearish'],
        3: ['Short', 'Strong', 'High', 'Bearish'],
        4: ['Extreme Short', 'Paused', 'Extreme', 'Fear'],
    };
    const b = base[regime] || base[0];
    const d = dirs[regime] || dirs[0];
    return ML_SIGNALS.map((s, i) => ({
        name: s.name,
        value: b[i],
        direction: d[i],
    }));
}
renderMLSignals();

// === 5. AGENT COGNITIVE LOAD MONITOR ===
const AGENTS = [
    { name: 'Technical', fullName: 'Technical Analyst' },
    { name: 'On-Chain', fullName: 'On-Chain Analyst' },
    { name: 'Sentiment', fullName: 'Sentiment Analyst' },
    { name: 'Macro', fullName: 'Macro Analyst' },
    { name: 'Protocol', fullName: 'Protocol Researcher' },
    { name: 'Risk Res.', fullName: 'Risk Researcher' },
    { name: 'Trader', fullName: 'Trader Agent' },
    { name: 'Risk Mgr', fullName: 'Risk Manager' },
    { name: 'Portfolio', fullName: 'Portfolio Manager' },
    { name: 'Guardian', fullName: 'Guardian Validator' },
];

// Map live agent status to load percentage
function _agentLoadFromLive(agentKey) {
    var liveAgents = window._liveAgentData;
    if (!liveAgents || !liveAgents.agents) return null;
    var a = liveAgents.agents[agentKey];
    if (!a) return null;
    var conf = parseInt(a.confidence) || 50;
    // ACTIVE agents with high confidence = high load, PAUSED = low
    if (a.status === 'PAUSED') return 15;
    if (a.status === 'WARNING') return 70 + Math.round(conf * 0.2);
    return Math.min(95, 40 + Math.round(conf * 0.5));
}

// Map AGENTS array indices to backend agent keys
var AGENT_KEY_MAP = {
    'Technical': 'momentum',
    'On-Chain': 'momentum',
    'Sentiment': 'sentiment',
    'Macro': 'risk',
    'Protocol': 'arbitrage',
    'Risk Res.': 'risk',
    'Trader': 'momentum',
    'Risk Mgr': 'risk',
    'Portfolio': 'risk',
    'Guardian': 'risk',
};

// Deterministic fallback loads per agent (regime-indexed)
var AGENT_FALLBACK_LOADS = {
    0: [42, 55, 38, 30, 48, 62, 70, 58, 45, 50],
    1: [65, 48, 52, 35, 55, 72, 82, 68, 58, 60],
    2: [55, 60, 45, 42, 50, 68, 75, 65, 52, 55],
    3: [72, 68, 58, 55, 62, 85, 88, 78, 65, 75],
    4: [85, 78, 70, 65, 75, 92, 95, 88, 78, 90],
};
var _agLoadTick = 0;

function renderAgentLoad() {
    var isLive = !!(window._liveAgentData && window._liveAgentData.agents);
    var statusEl = document.getElementById('agLoadStatus');
    if (statusEl) statusEl.textContent = isLive ? 'LIVE' : 'Offline';
    _agLoadTick++;

    const grid = document.getElementById('agGrid');
    grid.innerHTML = AGENTS.map(function(a, idx) {
        var liveLoad = _agentLoadFromLive(AGENT_KEY_MAP[a.name]);
        var ri = (typeof currentRegime !== 'undefined') ? currentRegime.state : 0;
        var baseLoads = AGENT_FALLBACK_LOADS[ri] || AGENT_FALLBACK_LOADS[0];
        var drift = Math.round(Math.sin(_agLoadTick * 0.3 + idx * 1.1) * 5);
        const load = liveLoad !== null ? liveLoad : Math.max(15, Math.min(95, baseLoads[idx] + drift));
        const color = load >= 80 ? 'var(--red)' : load >= 50 ? 'var(--dim)' : 'var(--green)';
        const status = load >= 80 ? 'HIGH LOAD' : load >= 50 ? 'PROCESSING' : 'IDLE';
        const pct = load / 100;
        const r = 16, cx = 20, cy = 20;
        const circ = 2 * Math.PI * r;
        const dash = circ * pct;
        return '<div class="ag-card" title="' + a.fullName + '">' +
            '<div class="ag-name">' + a.name + '</div>' +
            '<svg class="ag-load-ring" viewBox="0 0 40 40">' +
            '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="#eee" stroke-width="3"/>' +
            '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="3" ' +
            'stroke-dasharray="' + dash.toFixed(1) + ' ' + circ.toFixed(1) + '" transform="rotate(-90 ' + cx + ' ' + cy + ')"/>' +
            '<text x="' + cx + '" y="' + (cy + 4) + '" text-anchor="middle" font-size="10" fill="' + color + '" font-family="var(--font-mono)">' + load + '%</text>' +
            '</svg>' +
            '<div class="ag-status">' + status + '</div></div>';
    }).join('');
}
renderAgentLoad();
setInterval(renderAgentLoad, 4000);

// === 6. CIRCUIT BREAKERS (EXPANDED) ===
(function() {
    var BREAKERS = [
        { name: 'LP Strategy', triggers: ['3 consecutive IL exits', 'Pool TVL drop >20%', 'Impermanent loss >5%'], threshold: 3, current: 0 },
        { name: 'Arbitrage', triggers: ['5 failed executions', 'Slippage >2%', 'Latency >500ms'], threshold: 5, current: 1 },
        { name: 'Perpetuals', triggers: ['2 stop-losses hit', 'Funding rate spike', 'Liquidation proximity'], threshold: 2, current: 0 },
    ];
    var dailyDD = 1.2, weeklyDD = 3.4, oracleStale = 4;

    // Overlay live circuit breaker data when available
    function applyLiveCBData() {
        var liveCB = window._liveCBData;
        if (!liveCB || !liveCB.breakers) return;
        var breakers = liveCB.breakers;
        // Map live breaker states to our display breakers by strategy name
        var strategyMap = { 'lp': 0, 'arb': 1, 'perp': 2 };
        if (Array.isArray(breakers)) {
            breakers.forEach(function(b) {
                var key = (b.name || b.strategy || '').toLowerCase();
                var idx = strategyMap[key];
                if (idx !== undefined) {
                    BREAKERS[idx].current = b.consecutive_failures || b.current || 0;
                    BREAKERS[idx].threshold = b.threshold || BREAKERS[idx].threshold;
                    if (b.tripped || b.is_tripped) BREAKERS[idx].current = BREAKERS[idx].threshold;
                }
            });
        }
    }
    var cbAlertLog = [
        { time: '14:32:08', type: 'Drawdown', sev: 'info', msg: 'Daily DD 1.2% — normal', action: 'Monitor' },
        { time: '14:28:15', type: 'Oracle', sev: 'info', msg: 'Pyth latency 4s — normal', action: 'None' },
        { time: '14:15:42', type: 'Breaker', sev: 'warning', msg: 'Arb: 1/5 failed executions', action: 'Monitor' },
        { time: '13:58:30', type: 'Drawdown', sev: 'info', msg: 'Weekly DD 3.4% — within range', action: 'None' },
        { time: '13:42:11', type: 'System', sev: 'info', msg: 'All circuit breakers armed', action: 'None' },
    ];

    function now() { return new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }); }

    function renderBreakers() {
        applyLiveCBData();
        document.getElementById('cbBreakers').innerHTML = BREAKERS.map(function(b) {
            var ratio = b.current / b.threshold;
            var color = ratio >= 1 ? 'var(--red)' : ratio >= 0.66 ? '#cc8800' : 'var(--green)';
            var text = ratio >= 1 ? 'TRIPPED' : ratio >= 0.66 ? 'WARNING' : 'ARMED';
            return '<div class="cb-breaker"><div class="cb-breaker-name">' + b.name + '</div>' +
                '<div class="cb-breaker-status" style="color:' + color + '">' + text + ' (' + b.current + '/' + b.threshold + ')</div>' +
                '<div class="cb-breaker-detail">' + b.triggers.map(function(t) { return '• ' + t; }).join('<br>') + '</div></div>';
        }).join('');

        var dailyLim = 5;
        var weeklyLim = 10;
        var oracleLim = 30;

        var GLOBALS = [
            { label: 'Daily Drawdown', limit: dailyLim + '%', current: dailyDD.toFixed(1) + '%', status: dailyDD >= dailyLim ? 'bad' : dailyDD >= dailyLim * 0.6 ? 'warn' : 'ok' },
            { label: 'Weekly Drawdown', limit: weeklyLim + '%', current: weeklyDD.toFixed(1) + '%', status: weeklyDD >= weeklyLim ? 'bad' : weeklyDD >= weeklyLim * 0.6 ? 'warn' : 'ok' },
            { label: 'Oracle Staleness', limit: oracleLim + 's', current: oracleStale + 's', status: oracleStale >= oracleLim ? 'bad' : oracleStale >= oracleLim * 0.6 ? 'warn' : 'ok' },
        ];

        document.getElementById('cbGlobal').innerHTML = GLOBALS.map(function(g) {
            var c = g.status === 'ok' ? 'var(--green)' : g.status === 'warn' ? '#cc8800' : 'var(--red)';
            return '<div class="cb-global-item"><div class="cb-global-label">' + g.label + ' (limit: ' + g.limit + ')</div>' +
                '<div class="cb-global-value" style="color:' + c + '">' + g.current + '</div></div>';
        }).join('');

        // Drawdown bars
        var dPct = Math.min(100, (dailyDD / dailyLim) * 100);
        var wPct = Math.min(100, (weeklyDD / weeklyLim) * 100);
        var dColor = dPct >= 80 ? 'var(--red)' : dPct >= 50 ? '#cc8800' : 'var(--green)';
        var wColor = wPct >= 80 ? 'var(--red)' : wPct >= 50 ? '#cc8800' : 'var(--green)';
        document.getElementById('cbDdDaily').style.width = dPct + '%';
        document.getElementById('cbDdDaily').style.background = dColor;
        document.getElementById('cbDdDailyVal').textContent = dailyDD.toFixed(1) + '% / ' + dailyLim + '%';
        document.getElementById('cbDdWeekly').style.width = wPct + '%';
        document.getElementById('cbDdWeekly').style.background = wColor;
        document.getElementById('cbDdWeeklyVal').textContent = weeklyDD.toFixed(1) + '% / ' + weeklyLim + '%';

        // Status
        var allClear = BREAKERS.every(function(b) { return b.current < b.threshold; }) && GLOBALS.every(function(g) { return g.status === 'ok'; });
        document.getElementById('cbStatus').textContent = allClear ? 'ALL CLEAR' : 'ALERT';
        document.getElementById('cbStatus').className = allClear ? 'text-green' : 'text-red';
    }

    function renderAlertLog() {
        var log = document.getElementById('cbAlertLog');
        var header = '<div class="cb-alert-row header"><div class="cb-alert-cell">Time</div><div class="cb-alert-cell">Type</div><div class="cb-alert-cell">Severity</div><div class="cb-alert-cell">Message</div><div class="cb-alert-cell">Action</div></div>';
        log.innerHTML = header + cbAlertLog.slice(0, 10).map(function(a) {
            return '<div class="cb-alert-row"><div class="cb-alert-cell">' + a.time + '</div>' +
                '<div class="cb-alert-cell">' + a.type + '</div>' +
                '<div class="cb-alert-cell"><span class="cb-sev ' + a.sev + '">' + a.sev.toUpperCase() + '</span></div>' +
                '<div class="cb-alert-cell">' + a.msg + '</div>' +
                '<div class="cb-alert-cell">' + a.action + '</div></div>';
        }).join('');
    }

    function addAlert(type, sev, msg, action) {
        cbAlertLog.unshift({ time: now(), type: type, sev: sev, msg: msg, action: action });
        if (cbAlertLog.length > 50) cbAlertLog.length = 50;
        renderAlertLog();
    }

    var _cbLiveActive = false;

    function applyLiveDrawdown() {
        var dd = window._liveDrawdownData;
        if (!dd) return false;
        var prevDaily = dailyDD, prevWeekly = weeklyDD;
        if (dd.daily_drawdown !== undefined) dailyDD = Math.abs(dd.daily_drawdown * 100);
        if (dd.weekly_drawdown !== undefined) weeklyDD = Math.abs(dd.weekly_drawdown * 100);
        if (dd.oracle_staleness !== undefined) oracleStale = dd.oracle_staleness;

        // Generate alerts on significant changes
        var dailyLim = 5;
        if (dailyDD >= dailyLim && prevDaily < dailyLim) {
            addAlert('Drawdown', 'critical', 'Daily DD ' + dailyDD.toFixed(1) + '% ≥ ' + dailyLim + '% — PAUSED', 'Pause All');
        } else if (dailyDD >= dailyLim * 0.6 && prevDaily < dailyLim * 0.6) {
            addAlert('Drawdown', 'warning', 'Daily DD ' + dailyDD.toFixed(1) + '% approaching limit', 'Monitor');
        }
        return true;
    }

    renderBreakers();
    renderAlertLog();
    setInterval(async function() {
        var live = await fetchLiveCircuitBreakers();
        if (live) {
            _cbLiveActive = true;
            applyLiveDrawdown();
            applyLiveCBData();
            renderBreakers();
        } else {
            _cbLiveActive = false;
            var statusEl = document.getElementById('cbStatus');
            if (statusEl) {
                statusEl.textContent = 'Waiting for circuit breaker data...';
                statusEl.className = 'text-dim';
            }
        }
    }, 6000);
})();

// === 2. A-LAMS-VaR RISK DASHBOARD ===
(function() {
    const RG_NAMES = ['ACCUM', 'MARKUP', 'DISTRIB', 'MARKDN', 'CRISIS'];
    const RG_COLORS_VAR = ['#666666', '#00aa00', '#cc8800', '#cc0000', '#990000'];

    function getRegimeIndex() {
        if (typeof currentRegime !== 'undefined' && currentRegime.state !== undefined) return currentRegime.state;
        return 2;
    }

    function renderVarMetrics() {
        const ri = getRegimeIndex();
        const live = window._liveVaRData || {};
        const liveReg = window._liveRegimeData || {};

        // Use live data if available, else deterministic fallback
        let varVal, sigma, delta, liqContrib;
        let distribution = 'normal';

        if (live.var95 && live.var95.var_value !== undefined) {
            varVal = Math.abs(live.var95.var_value);
            sigma = live.var95.sigma_forecast ? Math.abs(live.var95.sigma_forecast) / 100 : 0.031;
            distribution = live.var95.distribution || 'normal';
            // Derive delta from live sigma (higher vol = higher asymmetry)
            delta = Math.min(0.189, 0.127 + sigma * 0.8);
            // Derive liquidity impact from regime index (higher regime = more liq stress)
            liqContrib = Math.min(47, 23 + ri * 6);
            document.getElementById('varLastUpdate').textContent = 'Live \u00b7 ' + distribution + ' \u00b7 ' + CerebralAPI.getToken();
        } else {
            const sigmas = [0.012, 0.018, 0.031, 0.052, 0.089];
            sigma = sigmas[ri];
            delta = Math.min(0.189, 0.127 + sigma * 0.8);
            liqContrib = 23 + ri * 6;
            varVal = (sigma * 1.645 + (liqContrib / 100) * sigma * 0.5) * 100;
            document.getElementById('varLastUpdate').textContent = 'Estimated · Risk Researcher';
        }

        const volForecast = liveReg.volatility_forecast
            ? Math.abs(liveReg.volatility_forecast)
            : sigma * 100;

        const metrics = [
            { label: 'Current VaR (95%)', value: varVal.toFixed(2) + '%', sub: 'Regime: ' + RG_NAMES[ri] + (live.var95 ? ' \u00b7 LIVE' : ''), color: varVal > 5 ? 'var(--red)' : varVal > 3 ? '#cc8800' : 'var(--green)' },
            { label: 'Volatility (\u03c3)', value: volForecast.toFixed(1) + '%', sub: (live.var95 ? 'MSM forecast' : 'K=' + (ri + 1) + ' of 5 states'), color: 'var(--fg)' },
            { label: 'Asymmetry (\u03b4)', value: delta.toFixed(3), sub: 'Range: [0.127, 0.189]', color: delta > 0.16 ? '#cc8800' : 'var(--fg)' },
            { label: 'Liquidity Impact', value: liqContrib.toFixed(0) + '%', sub: 'of stress-period losses', color: liqContrib > 35 ? 'var(--red)' : 'var(--fg)' },
        ];

        const varMetricsEl = document.getElementById('varMetrics');
        varMetricsEl.textContent = '';
        metrics.forEach(function(m) {
            var card = document.createElement('div');
            card.className = 'var-metric';
            var lbl = document.createElement('div');
            lbl.className = 'var-metric-label';
            lbl.textContent = m.label;
            var val = document.createElement('div');
            val.className = 'var-metric-value';
            val.style.color = m.color;
            val.textContent = m.value;
            var sub = document.createElement('div');
            sub.className = 'var-metric-sub';
            sub.textContent = m.sub;
            card.appendChild(lbl);
            card.appendChild(val);
            card.appendChild(sub);
            varMetricsEl.appendChild(card);
        });

        // Asymmetry gauge — derive from delta and regime
        const deltaBear = Math.min(0.189, delta * 1.1 + ri * 0.005);
        const deltaBull = Math.max(0.085, delta * 0.7 - ri * 0.003);
        const gaugeData = [
            { label: 'Bear δ (negative returns)', value: deltaBear, max: 0.25, color: 'var(--red)' },
            { label: 'Bull δ (positive returns)', value: deltaBull, max: 0.25, color: 'var(--green)' },
            { label: 'Net Asymmetry', value: deltaBear - deltaBull, max: 0.15, color: '#cc8800' },
        ];
        document.getElementById('varAsymGauge').innerHTML = gaugeData.map(g => {
            const pct = Math.min((g.value / g.max) * 100, 100);
            return '<div class="var-gauge-cell">' +
                '<div class="var-gauge-label">' + g.label + '</div>' +
                '<div class="var-bar-track"><div class="var-bar-fill" style="width:' + pct.toFixed(0) + '%;background:' + g.color + '"></div></div>' +
                '<div style="font-size:0.75rem;font-weight:600">' + g.value.toFixed(3) + '</div></div>';
        }).join('');

        // Confidence intervals — use live data if available
        let ci95 = varVal, ci99, ci995;
        if (live.var99 && live.var99.var_value !== undefined) {
            ci99 = Math.abs(live.var99.var_value);
        } else {
            ci99 = varVal * 1.42;
        }
        if (live.var995 && live.var995.var_value !== undefined) {
            ci995 = Math.abs(live.var995.var_value);
        } else {
            ci995 = varVal * 1.68;
        }

        const ciData = [
            { level: '95% (\u03b1=0.05)', value: ci95, range: '\u00b1' + (ci95 * 0.12).toFixed(2) + '%' },
            { level: '99% (\u03b1=0.01)', value: ci99, range: '\u00b1' + (ci99 * 0.15).toFixed(2) + '%' },
            { level: '99.5% (\u03b1=0.005)', value: ci995, range: '\u00b1' + (ci995 * 0.18).toFixed(2) + '%' },
        ];
        const ciGrid = document.getElementById('varCIGrid');
        ciGrid.textContent = '';
        ciData.forEach(function(c) {
            const col = c.value > 6 ? 'var(--red)' : c.value > 4 ? '#cc8800' : 'var(--green)';
            var item = document.createElement('div');
            item.className = 'var-ci-item';
            var lvl = document.createElement('div');
            lvl.className = 'var-ci-level';
            lvl.textContent = c.level;
            var v = document.createElement('div');
            v.className = 'var-ci-value';
            v.style.color = col;
            v.textContent = c.value.toFixed(2) + '%';
            var rng = document.createElement('div');
            rng.className = 'var-ci-range';
            rng.textContent = c.range;
            item.appendChild(lvl);
            item.appendChild(v);
            item.appendChild(rng);
            ciGrid.appendChild(item);
        });
    }

    // Live regime-profile data cache
    var _liveRegimeProfile = null;

    async function fetchLiveRegimeProfile() {
        if (typeof CerebralAPI === 'undefined') return;
        try {
            var data = await CerebralAPI.fetchRegimeProfile();
            if (data && data.profiles) _liveRegimeProfile = data;
        } catch (e) { /* graceful fallback */ }
    }

    function renderLiquidityScoring() {
        const ri = getRegimeIndex();
        var pools;

        if (_liveRegimeProfile && _liveRegimeProfile.profiles) {
            // Map live regime profiles to pool-like display items
            var REGIME_NAMES = ['ACCUM', 'MARKUP', 'DISTRIB', 'MARKDN', 'CRISIS'];
            pools = _liveRegimeProfile.profiles.map(function(p) {
                var score = p.liquidity_score != null ? p.liquidity_score : 50;
                var spread = p.spread_pct != null ? p.spread_pct : 0.1;
                var rName = REGIME_NAMES[p.regime - 1] || ('R' + p.regime);
                return {
                    name: rName + ' Regime',
                    protocol: 'n=' + p.n_obs,
                    depth: Math.round(score),
                    slippage: spread,
                    tvl: p.insufficient_data ? 'N/A' : (p.n_obs + ' obs'),
                };
            });
        } else {
            // Deterministic fallback
            pools = [
                { name: 'SOL/USDC', protocol: 'Orca', depth: 85 - ri * 12, slippage: 0.05 + ri * 0.08, tvl: '$42.1M' },
                { name: 'SOL/USDT', protocol: 'Raydium', depth: 78 - ri * 10, slippage: 0.08 + ri * 0.12, tvl: '$28.7M' },
                { name: 'RAY/USDC', protocol: 'Raydium', depth: 52 - ri * 8, slippage: 0.15 + ri * 0.22, tvl: '$8.3M' },
                { name: 'JUP/SOL', protocol: 'Jupiter', depth: 65 - ri * 11, slippage: 0.12 + ri * 0.15, tvl: '$15.6M' },
                { name: 'mSOL/SOL', protocol: 'Marinade', depth: 90 - ri * 7, slippage: 0.03 + ri * 0.05, tvl: '$67.2M' },
                { name: 'BONK/SOL', protocol: 'Orca', depth: 35 - ri * 6, slippage: 0.25 + ri * 0.35, tvl: '$4.1M' },
            ];
        }

        document.getElementById('varLiqGrid').innerHTML = pools.map(function(p) {
            const d = Math.max(p.depth, 5);
            const col = d >= 60 ? 'var(--green)' : d >= 35 ? '#cc8800' : 'var(--red)';
            return '<div class="var-liq-item">' +
                '<div class="var-liq-name">' + p.name + ' <span style="font-weight:400;color:var(--dim)">' + p.protocol + '</span></div>' +
                '<div class="var-liq-bar"><div class="var-liq-bar-fill" style="width:' + d + '%;background:' + col + '"></div></div>' +
                '<div class="var-liq-detail">Depth: ' + d + '% · Slippage: ' + p.slippage.toFixed(2) + '% · TVL: ' + p.tvl + '</div></div>';
        }).join('');
    }

    // Fetch live data on init
    fetchLiveRegimeProfile();

    function renderSlippageChart() {
        const container = document.getElementById('varSlippageChart');
        container.innerHTML = '';
        const w = container.clientWidth || 400, h = 180;
        const margin = { top: 20, right: 20, bottom: 30, left: 50 };

        const svg = d3.select(container).append('svg')
            .attr('width', w).attr('height', h)
            .attr('viewBox', [0, 0, w, h])
            .attr('style', 'max-width:100%;height:auto;');

        const tradeSizes = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000];
        const regimes = [
            { name: 'ACCUM', color: '#666666', mult: 1.0 },
            { name: 'MARKUP', color: '#00aa00', mult: 0.8 },
            { name: 'DISTRIB', color: '#cc8800', mult: 1.5 },
            { name: 'MARKDN', color: '#cc0000', mult: 2.2 },
            { name: 'CRISIS', color: '#990000', mult: 4.0 },
        ];

        const x = d3.scaleLog().domain([1000, 500000]).range([margin.left, w - margin.right]);
        const y = d3.scaleLinear().domain([0, 5]).range([h - margin.bottom, margin.top]);

        svg.append('g').attr('transform', 'translate(0,' + (h - margin.bottom) + ')')
            .call(d3.axisBottom(x).tickValues([1000, 10000, 100000, 500000]).tickFormat(d => d >= 1000 ? (d / 1000) + 'K' : d))
            .selectAll('text').attr('fill', 'var(--dim)').attr('font-size', '8px');

        svg.append('g').attr('transform', 'translate(' + margin.left + ',0)')
            .call(d3.axisLeft(y).ticks(5).tickFormat(d => d + '%'))
            .call(g => g.select('.domain').remove())
            .selectAll('text').attr('fill', 'var(--dim)').attr('font-size', '8px');

        svg.append('text').attr('x', w / 2).attr('y', h - 4)
            .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', 'var(--dim)').text('Trade Size (USD)');

        regimes.forEach(r => {
            const line = d3.line()
                .x(d => x(d.size))
                .y(d => y(d.slip))
                .curve(d3.curveMonotoneX);

            const data = tradeSizes.map(s => ({
                size: s,
                slip: Math.min((Math.pow(s / 1000, 0.65) * 0.02 * r.mult), 5)
            }));

            svg.append('path').datum(data)
                .attr('fill', 'none').attr('stroke', r.color).attr('stroke-width', 1.5)
                .attr('stroke-dasharray', r.name === 'CRISIS' ? '4,2' : 'none')
                .attr('d', line);

            const last = data[data.length - 1];
            svg.append('text').attr('x', x(last.size) + 3).attr('y', y(last.slip))
                .attr('font-size', '7px').attr('fill', r.color).attr('dy', '0.35em').text(r.name);
        });
    }

    function renderTransitionMatrix() {
        const ri = getRegimeIndex();
        const liveReg = window._liveRegimeData || {};

        // Use live transition matrix if available, pad/truncate to 5x5
        let matrix;
        if (liveReg.transition_matrix && liveReg.transition_matrix.length > 0) {
            const raw = liveReg.transition_matrix;
            const n = raw.length;
            if (n === 5) {
                matrix = raw;
            } else if (n < 5) {
                matrix = [];
                for (let i = 0; i < 5; i++) {
                    matrix[i] = [];
                    for (let j = 0; j < 5; j++) {
                        matrix[i][j] = (i < n && j < n) ? raw[i][j] : (i === j ? 0.5 : 0.1);
                    }
                }
            } else {
                // Downsample to 5x5 by bucketing
                matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]];
                for (let i = 0; i < n; i++) {
                    const bi = Math.min(4, Math.floor(i / n * 5));
                    for (let j = 0; j < n; j++) {
                        const bj = Math.min(4, Math.floor(j / n * 5));
                        matrix[bi][bj] += raw[i][j];
                    }
                }
                // Normalize rows
                matrix = matrix.map(function(row) {
                    const s = row.reduce(function(a,b){return a+b;}, 0);
                    return s > 0 ? row.map(function(v){return v/s;}) : row;
                });
            }
        } else {
            // Default asymmetric transition probabilities
            matrix = [
                [0.82, 0.12, 0.04, 0.01, 0.01],
                [0.08, 0.78, 0.10, 0.03, 0.01],
                [0.03, 0.07, 0.74, 0.12, 0.04],
                [0.01, 0.02, 0.08, 0.76, 0.13],
                [0.01, 0.01, 0.03, 0.10, 0.85],
            ];
        }

        let html = '<table><tr><th>From \\ To</th>';
        RG_NAMES.forEach(n => { html += '<th>' + n + '</th>'; });
        html += '</tr>';
        matrix.forEach((row, i) => {
            html += '<tr><th style="font-weight:600;color:' + RG_COLORS_VAR[i] + '">' + RG_NAMES[i] + '</th>';
            row.forEach((val, j) => {
                const bg = i === ri ? (j === ri ? '#e8f5e9' : '#fff8e1') : (val > 0.5 ? '#f5f5f5' : 'transparent');
                const fw = i === j ? '700' : '400';
                const col = val >= 0.7 ? 'var(--fg)' : val >= 0.1 ? 'var(--dim)' : '#ccc';
                html += '<td style="background:' + bg + ';font-weight:' + fw + ';color:' + col + '">' + (val * 100).toFixed(0) + '%</td>';
            });
            html += '</tr>';
        });
        html += '</table>';
        document.getElementById('varTransMatrix').innerHTML = html;
    }

    function renderPipeline() {
        const steps = [
            { num: '01', name: 'Regime Detection', desc: 'Hamilton filter classifies 5 volatility regimes in real-time' },
            { num: '02', name: 'Asymmetric Adj.', desc: 'δ shifts transition probabilities after negative returns' },
            { num: '03', name: 'Liquidity Scoring', desc: 'AMM pool depth + slippage factored into position sizing' },
            { num: '04', name: 'Dynamic VaR', desc: 'Regime-aware VaR feeds into position limits & circuit breakers' },
        ];
        document.getElementById('varPipeline').innerHTML = steps.map((s, i) =>
            '<div class="var-pipe-step">' +
            '<div class="var-pipe-num">STEP ' + s.num + '</div>' +
            '<div class="var-pipe-name">' + s.name + '</div>' +
            '<div class="var-pipe-desc">' + s.desc + '</div>' +
            (i < steps.length - 1 ? '<div class="var-pipe-arrow">→</div>' : '') +
            '</div>'
        ).join('');
    }

    function renderVarFormula() {
        const el = document.getElementById('fVaR');
        if (el && typeof katex !== 'undefined') {
            katex.render(
                'VaR_{\\alpha}(t) = \\sum_{k} P(S_t=k \\mid \\Omega_{t-1}) \\cdot [\\mu_k + \\sigma_k \\cdot \\Phi^{-1}(\\alpha)] + \\lambda(k, V_t) \\cdot Slippage(k)',
                el, { displayMode: true, throwOnError: false }
            );
        }
    }

    function renderAllVaR() {
        renderVarMetrics();
        renderLiquidityScoring();
        renderSlippageChart();
        renderTransitionMatrix();
        renderPipeline();
        renderVarFormula();
    }

    renderAllVaR();
    setInterval(function() {
        fetchLiveRegimeProfile();
        renderVarMetrics();
        renderLiquidityScoring();
    }, 5000);
    window.addEventListener('resize', renderSlippageChart);
})();

// === 5b. AGENT MEMORY TIMELINE ===
(function() {
    const FALLBACK_MEM_OPS = [
        { action: 'store', agent: 'Technical Analyst', desc: 'Stored RSI divergence pattern for SOL/USDC — bearish signal at 4H timeframe', ago: '2m' },
        { action: 'recall', agent: 'Trader Agent', desc: 'Retrieved last 5 trade outcomes for LP Rebalancing strategy — 4/5 profitable', ago: '3m' },
        { action: 'store', agent: 'On-Chain Analyst', desc: 'Stored whale wallet movement: 45K SOL transferred to Raydium LP', ago: '5m' },
        { action: 'update', agent: 'Risk Researcher', desc: 'Updated A-LAMS-VaR parameters — regime shifted from MARKUP to DISTRIBUTION', ago: '8m' },
        { action: 'recall', agent: 'Portfolio Manager', desc: 'Retrieved portfolio allocation state — LP: 42%, Arb: 28%, Perp: 30%', ago: '10m' },
        { action: 'store', agent: 'Sentiment Analyst', desc: 'Stored social sentiment snapshot — Twitter volume +45%, bullish bias 72/100', ago: '12m' },
        { action: 'prune', agent: 'System', desc: 'Pruned 23 stale memory entries older than 7 days — hot → warm tier migration', ago: '15m' },
        { action: 'recall', agent: 'Risk Manager', desc: 'Retrieved daily drawdown history — current: 1.2%, limit: 5%', ago: '18m' },
        { action: 'update', agent: 'Macro Analyst', desc: 'Updated BTC dominance correlation matrix — SOL decorrelation confirmed (0.48)', ago: '22m' },
        { action: 'store', agent: 'Protocol Researcher', desc: 'Stored Orca pool analysis — CLMM fee tier optimal at 0.3%, TVL $12.4M', ago: '25m' },
        { action: 'recall', agent: 'Trader Agent', desc: 'Retrieved arbitrage execution history — last 10 trades, 8 successful, avg spread 0.62%', ago: '30m' },
        { action: 'prune', agent: 'System', desc: 'Compressed 156 warm-tier entries to cold storage — indefinite retention', ago: '1h' },
    ];

    function renderMemoryTimeline(memOps) {
        const timeline = document.getElementById('memTimeline');
        timeline.innerHTML = '';
        memOps.forEach(function(m) {
            timeline.innerHTML += '<div class="mem-entry">' +
                '<div class="mem-dot ' + m.action + '"></div>' +
                '<div class="mem-content">' +
                    '<div class="mem-action">' + m.action + ' — ' + m.agent + '</div>' +
                    '<div class="mem-desc">' + m.desc + '</div>' +
                '</div>' +
                '<div class="mem-time">' + m.ago + '</div>' +
            '</div>';
        });
    }

    async function fetchAndRenderMemory() {
        if (typeof CerebralAPI !== 'undefined') {
            try {
                var status = await CerebralAPI.fetchNarratorStatus();
                if (status && status.enabled) {
                    // Build live status entry + keep fallback entries for history
                    var liveOps = [{
                        action: 'update',
                        agent: 'Narrator',
                        desc: 'Model: ' + (status.model || 'unknown') +
                              ' · Calls: ' + (status.call_count || 0) +
                              ' · Errors: ' + (status.error_count || 0) +
                              ' · Avg latency: ' + (status.avg_latency_ms || 0).toFixed(0) + 'ms',
                        ago: 'now',
                    }];
                    renderMemoryTimeline(liveOps.concat(FALLBACK_MEM_OPS.slice(0, 11)));
                    return;
                }
            } catch (e) { /* fallback */ }
        }
        renderMemoryTimeline(FALLBACK_MEM_OPS);
    }

    fetchAndRenderMemory();
    setInterval(fetchAndRenderMemory, 15000);

    // Goal Tracking — context.md: ElizaOS goal tracking
    const goals = [
        { name: 'Maximize LP Fee Yield', progress: 78, status: 'active', target: 'APY > 45%', current: 'APY: 38.2%' },
        { name: 'Reduce Portfolio Drawdown', progress: 92, status: 'active', target: '< 3% daily', current: '1.2% daily' },
        { name: 'Arbitrage Execution Rate', progress: 81, status: 'active', target: '> 80% success', current: '81.3% success' },
        { name: 'Perp Position Optimization', progress: 43, status: 'pending', target: 'Win rate > 55%', current: '43.5% win rate' },
        { name: 'Data Source Uptime', progress: 100, status: 'done', target: '> 99.5%', current: '99.8% uptime' },
        { name: 'Regime Detection Accuracy', progress: 67, status: 'active', target: '> 75%', current: '67% accuracy' },
    ];

    const goalList = document.getElementById('goalList');
    goals.forEach(g => {
        goalList.innerHTML += '<div class="goal-item">' +
            '<div class="goal-head">' +
                '<span class="goal-name">' + g.name + '</span>' +
                '<span class="goal-status ' + g.status + '">' + g.status.toUpperCase() + '</span>' +
            '</div>' +
            '<div class="goal-bar"><div class="goal-bar-fill" style="width:' + g.progress + '%"></div></div>' +
            '<div class="goal-meta"><span>' + g.current + '</span><span>Target: ' + g.target + '</span></div>' +
        '</div>';
    });
})();

// === 5c. TOOL ORCHESTRATION ===
(function() {
    // Tool Flow Diagram (D3) — context.md: Solana Agent Kit protocols
    const flowEl = document.getElementById('toolFlow');
    const fw = flowEl.clientWidth || 600;
    const fh = 160;
    const svg = d3.select(flowEl).append('svg')
        .attr('width', fw).attr('height', fh)
        .attr('viewBox', [0, 0, fw, fh])
        .attr('style', 'max-width:100%;height:auto;');

    const protocols = [
        { name: 'Drift', x: 0, caps: 'Positions · Funding' },
        { name: 'Jupiter', x: 1, caps: 'Swap · Aggregation' },
        { name: 'Orca', x: 2, caps: 'LP · CLMM' },
        { name: 'Kamino', x: 3, caps: 'Vaults · Yield' },
        { name: 'MarginFi', x: 4, caps: 'Lend · Borrow' },
    ];

    const boxW = 90, boxH = 50, gap = (fw - protocols.length * boxW) / (protocols.length + 1);
    const agentY = 10, protoY = 95;

    // Agent Kit hub
    svg.append('rect').attr('x', fw/2 - 70).attr('y', agentY).attr('width', 140).attr('height', 30)
        .attr('fill', 'none').attr('stroke', 'var(--border)').attr('stroke-width', 1);
    svg.append('text').attr('x', fw/2).attr('y', agentY + 19).attr('text-anchor', 'middle')
        .attr('font-size', '9px').attr('fill', 'var(--fg)').text('SOLANA AGENT KIT');

    protocols.forEach((p, i) => {
        const px = gap + i * (boxW + gap);
        // Connection line
        svg.append('line').attr('x1', fw/2).attr('y1', agentY + 30).attr('x2', px + boxW/2).attr('y2', protoY)
            .attr('stroke', '#ccc').attr('stroke-width', 1);
        // Protocol box
        svg.append('rect').attr('x', px).attr('y', protoY).attr('width', boxW).attr('height', boxH)
            .attr('fill', '#fafafa').attr('stroke', 'var(--border)').attr('stroke-width', 1);
        svg.append('text').attr('x', px + boxW/2).attr('y', protoY + 18).attr('text-anchor', 'middle')
            .attr('font-size', '9px').attr('font-weight', '600').attr('fill', 'var(--fg)').text(p.name);
        svg.append('text').attr('x', px + boxW/2).attr('y', protoY + 32).attr('text-anchor', 'middle')
            .attr('font-size', '7px').attr('fill', 'var(--dim)').text(p.caps);
    });

    // Protocol Action Log
    const actions = [
        { tool: 'jupiter.swap', protocol: 'Jupiter', result: 'success', latency: '340ms', desc: 'SOL → USDC · 12.5 SOL · Slippage: 0.12%' },
        { tool: 'orca.addLiquidity', protocol: 'Orca', result: 'success', latency: '520ms', desc: 'SOL/USDC CLMM · Range: $148-$165 · $2,400' },
        { tool: 'drift.openPosition', protocol: 'Drift', result: 'success', latency: '280ms', desc: 'SOL-PERP LONG · 3x leverage · $1,500 margin' },
        { tool: 'kamino.deposit', protocol: 'Kamino', result: 'success', latency: '410ms', desc: 'USDC vault deposit · $5,000 · Est. APY: 8.2%' },
        { tool: 'marginfi.borrow', protocol: 'MarginFi', result: 'fail', latency: '890ms', desc: 'USDC borrow rejected — utilization rate 94%' },
        { tool: 'jupiter.swap', protocol: 'Jupiter', result: 'success', latency: '310ms', desc: 'JUP → SOL · 500 JUP · Route: JUP→USDC→SOL' },
        { tool: 'drift.closePosition', protocol: 'Drift', result: 'success', latency: '260ms', desc: 'SOL-PERP closed · P&L: +$142.30 · Hold: 4.2h' },
        { tool: 'orca.removeLiquidity', protocol: 'Orca', result: 'pending', latency: '—', desc: 'SOL/USDC CLMM removal pending — IL check in progress' },
    ];

    const toolLog = document.getElementById('toolLog');
    actions.forEach(a => {
        toolLog.innerHTML += '<div class="tool-entry">' +
            '<span class="tool-name">' + a.tool + '</span>' +
            '<span class="tool-protocol"><span class="proto-badge">' + a.protocol + '</span></span>' +
            '<span class="tool-result ' + a.result + '">' + a.result.toUpperCase() + '</span>' +
            '<span class="tool-latency">' + a.latency + '</span>' +
        '</div>';
    });
})();

// === 5d. AGENT DECISION TREE ===
(function() {
    const el = document.getElementById('dtreeContainer');
    const w = el.clientWidth || 900;
    const h = 320;

    const svg = d3.select(el).append('svg')
        .attr('width', w).attr('height', h)
        .attr('viewBox', [0, 0, w, h])
        .attr('style', 'max-width:100%;height:auto;');

    // context.md: 4 Analyst → 2 Researcher → 3 Decision (Trader → Risk Mgr → Portfolio Mgr)
    const nodes = [
        { id: 'data', label: 'DATA SOURCES', x: w/2, y: 20, w: 100, h: 24, tier: 'input' },
        { id: 'tech', label: 'Technical', x: w*0.1, y: 80, w: 80, h: 24, tier: 'analyst' },
        { id: 'onchain', label: 'On-Chain', x: w*0.3, y: 80, w: 80, h: 24, tier: 'analyst' },
        { id: 'sentiment', label: 'Sentiment', x: w*0.5, y: 80, w: 80, h: 24, tier: 'analyst' },
        { id: 'macro', label: 'Macro', x: w*0.7, y: 80, w: 80, h: 24, tier: 'analyst' },
        { id: 'proto_res', label: 'Protocol Res.', x: w*0.3, y: 150, w: 90, h: 24, tier: 'researcher' },
        { id: 'risk_res', label: 'Risk Res.', x: w*0.6, y: 150, w: 90, h: 24, tier: 'researcher' },
        { id: 'trader', label: 'Trader Agent', x: w*0.25, y: 220, w: 90, h: 24, tier: 'decision' },
        { id: 'risk_mgr', label: 'Risk Manager', x: w*0.5, y: 220, w: 90, h: 24, tier: 'decision' },
        { id: 'pm', label: 'Portfolio Mgr', x: w*0.75, y: 220, w: 90, h: 24, tier: 'decision' },
        { id: 'exec', label: 'EXECUTION', x: w/2, y: 290, w: 100, h: 24, tier: 'output' },
    ];

    const edges = [
        ['data', 'tech'], ['data', 'onchain'], ['data', 'sentiment'], ['data', 'macro'],
        ['tech', 'proto_res'], ['onchain', 'proto_res'], ['tech', 'risk_res'], ['macro', 'risk_res'],
        ['sentiment', 'risk_res'], ['onchain', 'risk_res'],
        ['proto_res', 'trader'], ['risk_res', 'trader'], ['risk_res', 'risk_mgr'],
        ['trader', 'risk_mgr'], ['trader', 'pm'], ['risk_mgr', 'pm'],
        ['pm', 'exec'],
    ];

    const tierColors = {
        input: '#f5f5f5', analyst: '#e8f5e9', researcher: '#e3f2fd', decision: '#fff3e0', output: '#f5f5f5'
    };

    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    // Draw edges
    edges.forEach(([from, to]) => {
        const f = nodeMap[from], t = nodeMap[to];
        svg.append('line')
            .attr('x1', f.x).attr('y1', f.y + f.h)
            .attr('x2', t.x).attr('y2', t.y)
            .attr('stroke', '#ccc').attr('stroke-width', 1);
    });

    // Draw nodes
    nodes.forEach(n => {
        svg.append('rect')
            .attr('x', n.x - n.w/2).attr('y', n.y)
            .attr('width', n.w).attr('height', n.h)
            .attr('fill', tierColors[n.tier]).attr('stroke', 'var(--border)').attr('stroke-width', 1).attr('rx', 2);
        svg.append('text')
            .attr('x', n.x).attr('y', n.y + n.h/2 + 3)
            .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', 'var(--fg)')
            .text(n.label);
    });

    // Tier labels
    const tiers = [
        { label: 'INPUT', y: 32, color: 'var(--dim)' },
        { label: 'ANALYSTS (4)', y: 92, color: 'var(--green)' },
        { label: 'RESEARCHERS (2)', y: 162, color: '#0066cc' },
        { label: 'DECISION (3)', y: 232, color: '#cc8800' },
        { label: 'OUTPUT', y: 302, color: 'var(--dim)' },
    ];
    tiers.forEach(t => {
        svg.append('text').attr('x', 8).attr('y', t.y)
            .attr('font-size', '7px').attr('fill', t.color).attr('font-weight', '600').text(t.label);
    });
})();

// === 7. ML SIGNAL CORRELATION MATRIX — Brushable Scatterplot ===
(function() {
    const container = document.getElementById('scatterMatrix');
    container.innerHTML = '<div style="padding:2rem 1rem;text-align:center;color:var(--dim);font-size:0.7rem;line-height:1.8">' +
        '<div style="font-size:1.2rem;margin-bottom:0.5rem">◇</div>' +
        'Insufficient data — collecting signal correlation samples<br>' +
        '<span style="font-size:0.6rem">Requires ≥ 50 signal observations across 4 dimensions to render matrix</span></div>';
})();

// === 8. INTER-AGENT COMMUNICATION NETWORK — Ridgeline Plot ===
(function() {
    const container = document.getElementById('ridgelineChart');
    container.innerHTML = '<div style="padding:2rem 1rem;text-align:center;color:var(--dim);font-size:0.7rem;line-height:1.8">' +
        '<div style="font-size:1.2rem;margin-bottom:0.5rem">◇</div>' +
        'Insufficient data — collecting inter-agent message logs<br>' +
        '<span style="font-size:0.6rem">Requires 24h of agent communication data (96 intervals × 10 agents) to render ridgeline</span></div>';
})();

// === 9. PARAMETER ADAPTATION RIDGELINE ===
(function() {
    const container = document.getElementById('ptChart');
    const paramNames = [
        'Stop-Loss %', 'Position Size %', 'Leverage',
        'Slippage Tolerance', 'Rebalance Freq',
        'Max Drawdown %', 'Hedge Ratio'
    ];
    const POINTS = 120;
    const overlap = 7;

    // Deterministic phase offsets per parameter (golden-ratio spaced)
    const phases = paramNames.map(function(_, i) { return (i * 2.399) % (Math.PI * 2); });
    const freqs  = paramNames.map(function(_, i) { return 0.08 + ((i * 0.618) % 1) * 0.12; });
    const amps   = paramNames.map(function(_, i) { return 0.4 + ((i * 0.382) % 1) * 0.5; });

    let tick = 0;

    function renderParamRidgeline() {
        container.innerHTML = '';
        const now = Date.now();
        const dates = Array.from({ length: POINTS }, (_, i) =>
            new Date(now - (POINTS - 1 - i) * 6 * 3600000)
        );

        const series = paramNames.map((name, pi) => {
            const ph = phases[pi] + tick * 0.15;
            const fr = freqs[pi];
            const am = amps[pi];
            const values = dates.map((_, i) => {
                const t = i / POINTS;
                const wave1 = Math.sin(t * Math.PI * 6 * fr + ph) * am;
                const wave2 = Math.sin(t * Math.PI * 10 * fr + ph * 1.7) * am * 0.5;
                const wave3 = Math.sin(t * Math.PI * 16 * fr + ph * 2.3) * am * 0.25;
                var noise = Math.sin(i * 1.1 + pi * 3.7 + tick * 0.2) * 0.075;
                return Math.max(0, wave1 + wave2 + wave3 + noise + am + 0.3);
            });
            return { name, values };
        });

        tick++;

        const cw = container.clientWidth || 928;
        const width = cw;
        const height = series.length * 42;
        const marginTop = 30;
        const marginRight = 20;
        const marginBottom = 30;
        const marginLeft = 140;

        const x = d3.scaleTime()
            .domain(d3.extent(dates))
            .range([marginLeft, width - marginRight]);

        const y = d3.scalePoint()
            .domain(series.map(d => d.name))
            .range([marginTop, height - marginBottom]);

        const z = d3.scaleLinear()
            .domain([0, d3.max(series, d => d3.max(d.values))]).nice()
            .range([0, -overlap * y.step()]);

        const area = d3.area()
            .curve(d3.curveBasis)
            .defined(d => !isNaN(d))
            .x((d, i) => x(dates[i]))
            .y0(0)
            .y1(d => z(d));

        const line = area.lineY1();

        const svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [0, 0, width, height])
            .attr('style', 'max-width: 100%; height: auto;');

        svg.append('g')
            .attr('transform', 'translate(0,' + (height - marginBottom) + ')')
            .call(d3.axisBottom(x).ticks(width / 100).tickSizeOuter(0))
            .selectAll('text').attr('font-size', '8px').attr('fill', 'var(--dim)');

        svg.append('g')
            .attr('transform', 'translate(' + marginLeft + ',0)')
            .call(d3.axisLeft(y).tickSize(0).tickPadding(4))
            .call(g => g.select('.domain').remove())
            .selectAll('text').attr('font-size', '9px').attr('font-weight', '600').attr('fill', 'var(--fg)');

        const group = svg.append('g')
          .selectAll('g')
          .data(series)
          .join('g')
            .attr('transform', d => 'translate(0,' + (y(d.name) + 1) + ')');

        group.append('path')
            .attr('fill', '#ddd')
            .attr('d', d => area(d.values));

        group.append('path')
            .attr('fill', 'none')
            .attr('stroke', '#080808')
            .attr('stroke-width', 1)
            .attr('d', d => line(d.values));
    }

    renderParamRidgeline();
    setInterval(renderParamRidgeline, 4000);
})();

// === 10. CONFIDENCE THRESHOLDS SCATTER ===
(function() {
    const container = document.getElementById('ctChart');
    container.innerHTML = '<div style="padding:2rem 1rem;text-align:center;color:var(--dim);font-size:0.7rem;line-height:1.8">' +
        '<div style="font-size:1.2rem;margin-bottom:0.5rem">◇</div>' +
        'Insufficient data — collecting confidence threshold observations<br>' +
        '<span style="font-size:0.6rem">Requires ≥ 1,000 agent signal events to render threshold scatter (10 agents × signal strength × magnitude)</span></div>';
})();

// === AGENT SIGNAL HORIZON CHART ===
(function() {
    const container = document.getElementById('hzChart');
    if (!container) return;
    container.innerHTML = '<div style="padding:2rem 1rem;text-align:center;color:var(--dim);font-size:0.7rem;line-height:1.8">' +
        '<div style="font-size:1.2rem;margin-bottom:0.5rem">◇</div>' +
        'Insufficient data — collecting agent signal horizon data<br>' +
        '<span style="font-size:0.6rem">Requires continuous signal stream from 10 agents to render horizon bands</span></div>';
})();

// === 11. POSITION DISTRIBUTION — ECharts Scatter ↔ Bar ===
(function () {
    const el = document.getElementById('posDistChart');
    if (!el || typeof echarts === 'undefined') return;
    const chart = echarts.init(el);

    const longData = [[161.2,51.6],[167.5,59.0],[159.5,49.2],[157.0,63.0],[155.8,53.6],
        [170.0,59.0],[159.1,47.6],[166.0,69.8],[176.2,66.8],[160.2,75.2],
        [172.5,55.2],[170.9,54.2],[172.9,62.5],[153.4,42.0],[160.0,50.0],
        [147.2,49.8],[168.2,49.2],[175.0,73.2],[157.0,47.8],[167.6,68.8],
        [159.5,50.6],[175.0,82.5],[166.8,57.2],[176.5,87.8],[170.2,72.8],
        [174.0,54.5],[173.0,59.8],[179.9,67.3],[170.5,67.8],[160.0,47.0],
        [154.4,46.2],[162.0,55.0],[176.5,83.0],[160.0,54.4],[152.0,45.8],
        [162.1,53.6],[170.0,73.2],[160.2,52.1],[161.3,67.9],[166.4,56.6],
        [168.9,62.3],[163.8,58.5],[167.6,54.5],[160.0,50.2],[161.3,60.3],
        [167.6,58.3],[165.1,56.2],[160.0,50.2],[170.0,72.9],[157.5,59.8],
        [167.6,61.0],[160.7,69.1],[163.2,55.9],[152.4,46.5],[157.5,54.3],
        [168.3,54.8],[180.3,60.7],[165.5,60.0],[165.0,62.0],[164.5,60.3],
        [156.0,52.7],[160.0,74.3],[163.0,62.0],[165.7,73.1],[161.0,80.0],
        [162.0,54.7],[166.0,53.2],[174.0,75.7],[172.7,61.1],[167.6,55.7],
        [151.1,48.7],[164.5,52.3],[163.5,50.0],[152.0,59.3],[169.0,62.5],
        [164.0,55.7],[161.2,54.8],[155.0,45.9],[170.0,70.6],[176.2,67.2],
        [170.0,69.4],[162.5,58.2],[170.3,64.8],[164.1,71.6],[169.5,52.8],
        [163.2,59.8],[154.5,49.0],[159.8,50.0],[173.2,69.2],[170.0,55.9],
        [161.4,63.4],[169.0,58.2],[166.2,58.6],[159.4,45.7],[162.5,52.2],
        [159.0,48.6],[162.8,57.8],[159.0,55.6],[179.8,66.8],[162.9,59.4],
        [161.0,53.6],[151.1,73.2],[168.2,53.4],[168.9,69.0],[173.2,58.4],
        [171.8,56.2],[178.0,70.6],[164.3,59.8],[163.0,72.0],[168.5,65.2],
        [166.8,56.6],[172.7,105.2],[163.5,51.8],[169.4,63.4],[167.8,59.0],
        [159.5,47.6],[167.6,63.0],[161.2,55.2],[160.0,45.0],[163.2,54.0],
        [162.2,50.2],[161.3,60.2],[149.5,44.8],[157.5,58.8],[163.2,56.4],
        [172.7,62.0],[155.0,49.2],[156.5,67.2],[164.0,53.8],[160.9,54.4],
        [162.8,58.0],[167.0,59.8],[160.0,54.8],[160.0,43.2],[168.9,60.5],
        [158.2,46.4],[156.0,64.4],[160.0,48.8],[167.1,62.2],[158.0,55.5],
        [167.6,57.8],[156.0,54.6],[162.1,59.2],[173.4,52.7],[159.8,53.2],
        [170.5,64.5],[159.2,51.8],[157.5,56.0],[161.3,63.6],[162.6,63.2],
        [160.0,59.5],[168.9,56.8],[165.1,64.1],[162.6,50.0],[165.1,72.3],
        [166.4,55.0],[160.0,55.9],[152.4,60.4],[170.2,69.1],[162.6,84.5],
        [170.2,55.9],[158.8,55.5],[172.7,69.5],[167.6,76.4],[162.6,61.4],
        [167.6,65.9],[156.2,58.6],[175.2,66.8],[172.1,56.6],[162.6,58.6],
        [160.0,55.9],[165.1,59.1],[182.9,81.8],[166.4,70.7],[165.1,56.8],
        [177.8,60.0],[165.1,58.2],[175.3,72.7],[154.9,54.1],[158.8,49.1],
        [172.7,75.9],[168.9,55.0],[161.3,57.3],[167.6,55.0],[165.1,65.5],
        [175.3,65.5],[157.5,48.6],[163.8,58.6],[167.6,63.6],[165.1,55.2],
        [165.1,62.7],[168.9,56.6],[162.6,53.9],[164.5,63.2],[176.5,73.6],
        [168.9,62.0],[175.3,63.6],[159.4,53.2],[160.0,53.4],[170.2,55.0],
        [162.6,70.5],[167.6,54.5],[162.6,54.5],[160.7,55.9],[160.0,59.0],
        [157.5,63.6],[162.6,54.5],[152.4,47.3],[170.2,67.7],[165.1,80.9],
        [172.7,70.5],[165.1,60.9],[170.2,63.6],[170.2,54.5],[170.2,59.1],
        [161.3,70.5],[167.6,52.7],[167.6,62.7],[165.1,86.3],[162.6,66.4],
        [152.4,67.3],[168.9,63.0],[170.2,73.6],[175.2,62.3],[175.2,57.7],
        [160.0,55.4],[165.1,104.1],[174.0,55.5],[170.2,77.3],[160.0,80.5],
        [167.6,64.5],[167.6,72.3],[167.6,61.4],[154.9,58.2],[162.6,81.8],
        [175.3,63.6],[171.4,53.4],[157.5,54.5],[165.1,53.6],[160.0,60.0],
        [174.0,73.6],[162.6,61.4],[174.0,55.5],[162.6,63.6],[161.3,60.9],
        [156.2,60.0],[149.9,46.8],[169.5,57.3],[160.0,64.1],[175.3,63.6],
        [169.5,67.3],[160.0,75.5],[172.7,68.2],[162.6,61.4],[157.5,76.8],
        [176.5,71.8],[164.4,55.5],[160.7,48.6],[174.0,66.4],[163.8,67.3]];

    const shortData = [[174.0,65.6],[175.3,71.8],[193.5,80.7],[186.5,72.6],[187.2,78.8],
        [181.5,74.8],[184.0,86.4],[184.5,78.4],[175.0,62.0],[184.0,81.6],
        [180.0,76.6],[177.8,83.6],[192.0,90.0],[176.0,74.6],[174.0,71.0],
        [184.0,79.6],[192.7,93.8],[171.5,70.0],[173.0,72.4],[176.0,85.9],
        [176.0,78.8],[180.5,77.8],[172.7,66.2],[176.0,86.4],[173.5,81.8],
        [178.0,89.6],[180.3,82.8],[180.3,76.4],[164.5,63.2],[173.0,60.9],
        [183.5,74.8],[175.5,70.0],[188.0,72.4],[189.2,84.1],[172.8,69.1],
        [170.0,59.5],[182.0,67.2],[170.0,61.3],[177.8,68.6],[184.2,80.1],
        [186.7,87.8],[171.4,84.7],[172.7,73.4],[175.3,72.1],[180.3,82.6],
        [182.9,88.7],[188.0,84.1],[177.2,94.1],[172.1,74.9],[167.0,59.1],
        [169.5,75.6],[174.0,86.2],[172.7,75.3],[182.2,87.1],[164.1,55.2],
        [163.0,57.0],[171.5,61.4],[184.2,76.8],[174.0,86.8],[174.0,72.2],
        [177.0,71.6],[186.0,84.8],[167.0,68.2],[171.8,66.1],[182.0,72.0],
        [167.0,64.6],[177.8,74.8],[164.5,70.0],[192.0,101.6],[175.5,63.2],
        [171.2,79.1],[181.6,78.9],[167.4,67.7],[181.1,66.0],[177.0,68.2],
        [174.5,63.9],[177.5,72.0],[170.5,56.8],[182.4,74.5],[197.1,90.9],
        [180.1,93.0],[175.5,80.9],[180.6,72.7],[184.4,68.0],[175.5,70.9],
        [180.6,72.5],[177.0,72.5],[177.1,83.4],[181.6,75.5],[176.5,73.0],
        [175.0,70.2],[174.0,73.4],[165.1,70.5],[177.0,68.9],[192.0,102.3],
        [176.5,68.4],[169.4,65.9],[182.1,75.7],[179.8,84.5],[175.3,87.7],
        [184.9,86.4],[177.3,73.2],[167.4,53.9],[178.1,72.0],[168.9,55.5],
        [157.2,58.4],[180.3,83.2],[170.2,72.7],[177.8,64.1],[172.7,72.3],
        [165.1,65.0],[186.7,86.4],[165.1,65.0],[174.0,88.6],[175.3,84.1],
        [185.4,66.8],[177.8,75.5],[180.3,93.2],[180.3,82.7],[177.8,58.0],
        [177.8,79.5],[177.8,78.6],[177.8,71.8],[177.8,116.4],[163.8,72.2],
        [188.0,83.6],[198.1,85.5],[175.3,90.9],[166.4,85.9],[190.5,89.1],
        [166.4,75.0],[177.8,77.7],[179.7,86.4],[172.7,90.9],[190.5,73.6],
        [185.4,76.4],[168.9,69.1],[167.6,84.5],[175.3,64.5],[170.2,69.1],
        [190.5,108.6],[177.8,86.4],[190.5,80.9],[177.8,87.7],[184.2,94.5],
        [176.5,80.2],[177.8,72.0],[180.3,71.4],[171.4,72.7],[172.7,84.1],
        [172.7,76.8],[177.8,63.6],[177.8,80.9],[182.9,80.9],[170.2,85.5],
        [167.6,68.6],[175.3,67.7],[165.1,66.4],[185.4,102.3],[181.6,70.5],
        [172.7,95.9],[190.5,84.1],[179.1,87.3],[175.3,71.8],[170.2,65.9],
        [193.0,95.9],[171.4,91.4],[177.8,81.8],[177.8,96.8],[167.6,69.1],
        [167.6,82.7],[180.3,75.5],[182.9,79.5],[176.5,73.6],[186.7,91.8],
        [188.0,84.1],[188.0,85.9],[177.8,81.8],[174.0,82.5],[177.8,80.5],
        [171.4,70.0],[185.4,81.8],[185.4,84.1],[188.0,90.5],[188.0,91.4],
        [182.9,89.1],[176.5,85.0],[175.3,69.1],[175.3,73.6],[188.0,80.5],
        [188.0,82.7],[175.3,86.4],[170.5,67.7],[179.1,92.7],[177.8,93.6],
        [175.3,70.9],[182.9,75.0],[170.8,93.2],[188.0,93.2],[180.3,77.7],
        [177.8,61.4],[185.4,94.1],[168.9,75.0],[185.4,83.6],[180.3,85.5],
        [174.0,73.9],[167.6,66.8],[182.9,87.3],[160.0,72.3],[180.3,88.6],
        [167.6,75.5],[186.7,101.4],[175.3,91.1],[175.3,67.3],[175.9,77.7],
        [175.3,81.8],[179.1,75.5],[181.6,84.5],[177.8,76.6],[182.9,85.0],
        [177.8,102.5],[184.2,77.3],[179.1,71.8],[176.5,87.9],[188.0,94.3],
        [174.0,70.9],[167.6,64.5],[170.2,77.3],[167.6,72.3],[188.0,87.3],
        [174.0,80.0],[176.5,82.3],[180.3,73.6],[167.6,74.1],[188.0,85.9],
        [180.3,73.2],[167.6,76.3],[183.0,65.9],[183.0,90.9],[179.1,89.1],
        [170.2,62.3],[177.8,82.7],[179.1,79.1],[190.5,98.2],[177.8,84.1],
        [180.3,83.2],[180.3,83.2]];

    function avg(data, dim) {
        return data.reduce((s, d) => s + d[dim], 0) / data.length;
    }

    const fontFamily = getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim() || 'sans-serif';
    const green = '#d0f7da';
    const fg = '#080808';

    const scatterOpt = {
        backgroundColor: 'transparent',
        textStyle: { fontFamily },
        tooltip: {
            trigger: 'item',
            backgroundColor: fg,
            borderColor: fg,
            textStyle: { color: '#fff', fontFamily, fontSize: 11 },
            formatter: function (p) {
                return p.seriesName + '<br/>Entry Score: ' + p.value[0].toFixed(1) + '<br/>Size ($K): ' + p.value[1].toFixed(1);
            }
        },
        legend: {
            data: ['Long', 'Short'],
            textStyle: { color: fg, fontFamily, fontSize: 11 },
            top: 10
        },
        xAxis: { scale: true, name: 'Entry Score', nameTextStyle: { color: fg, fontFamily, fontSize: 11 }, axisLine: { lineStyle: { color: fg } }, splitLine: { lineStyle: { color: '#e0e0e0' } } },
        yAxis: { scale: true, name: 'Position Size ($K)', nameTextStyle: { color: fg, fontFamily, fontSize: 11 }, axisLine: { lineStyle: { color: fg } }, splitLine: { lineStyle: { color: '#e0e0e0' } } },
        series: [
            {
                name: 'Long', type: 'scatter', id: 'long', dataGroupId: 'long',
                itemStyle: { color: green, opacity: 0.7 },
                symbolSize: 6,
                universalTransition: { enabled: true, delay: function (idx) { return (idx * 23) % 400; } },
                data: longData
            },
            {
                name: 'Short', type: 'scatter', id: 'short', dataGroupId: 'short',
                itemStyle: { color: fg, opacity: 0.55 },
                symbolSize: 6,
                universalTransition: { enabled: true, delay: function (idx) { return (idx * 29) % 400; } },
                data: shortData
            }
        ]
    };

    const barOpt = {
        backgroundColor: 'transparent',
        textStyle: { fontFamily },
        tooltip: { trigger: 'item', backgroundColor: fg, borderColor: fg, textStyle: { color: '#fff', fontFamily, fontSize: 11 } },
        xAxis: { type: 'category', data: ['Short', 'Long'], axisLine: { lineStyle: { color: fg } }, axisLabel: { color: fg, fontFamily, fontSize: 12 } },
        yAxis: { name: 'Avg Entry Score', nameTextStyle: { color: fg, fontFamily, fontSize: 11 }, axisLine: { lineStyle: { color: fg } }, splitLine: { lineStyle: { color: '#e0e0e0' } } },
        series: [{
            type: 'bar', id: 'total',
            itemStyle: { color: function (p) { return p.dataIndex === 0 ? fg : green; } },
            barWidth: '40%',
            label: { show: true, position: 'top', color: fg, fontFamily, fontSize: 11, formatter: function (p) { return p.value.toFixed(1); } },
            data: [
                { value: avg(shortData, 0), groupId: 'short' },
                { value: avg(longData, 0), groupId: 'long' }
            ],
            universalTransition: { enabled: true, seriesKey: ['long', 'short'], delay: function (idx) { return (idx * 37) % 400; } }
        }]
    };

    let currentOpt = scatterOpt;
    chart.setOption(currentOpt);
    setInterval(function () {
        currentOpt = currentOpt === scatterOpt ? barOpt : scatterOpt;
        chart.setOption(currentOpt, true);
    }, 3000);

    window.addEventListener('resize', function () { chart.resize(); });
})();

// === 12. AGENT SIGNAL MATRIX — Scatter Matrix + Parallel Coordinates ===
(function () {
    const el = document.getElementById('agentSignalMatrix');
    if (!el || typeof echarts === 'undefined') return;

    const schema = [
        { name: 'volatility', index: 1, text: 'Volatility' },
        { name: 'volume', index: 2, text: 'Volume' },
        { name: 'momentum', index: 3, text: 'Momentum' },
        { name: 'correlation', index: 4, text: 'Correlation' },
        { name: 'sharpe', index: 5, text: 'Sharpe' },
        { name: 'drawdown', index: 6, text: 'Drawdown' },
        { name: 'grade', index: 7, text: 'Grade' }
    ];

    // DeFi agent signal data: [volatility, volume, momentum, correlation, sharpe, drawdown, grade, group]
    const rawData = [
        [55,9,56,0.46,1.8,6,'Strong','Momentum'],[25,11,21,0.65,3.4,9,'Excellent','Momentum'],
        [56,7,63,0.30,1.4,5,'Strong','Momentum'],[33,7,29,0.33,1.6,6,'Excellent','Momentum'],
        [42,24,44,0.76,4.0,16,'Excellent','Momentum'],[82,58,90,1.77,6.8,33,'Strong','Momentum'],
        [74,49,77,1.46,4.8,27,'Strong','Momentum'],[78,55,80,1.29,5.9,29,'Strong','Momentum'],
        [267,216,280,4.80,10.8,64,'Critical','Momentum'],[185,127,216,2.52,6.1,27,'Moderate','Momentum'],
        [39,19,38,0.57,3.1,15,'Excellent','Momentum'],[41,11,40,0.43,2.1,7,'Excellent','Momentum'],
        [64,38,74,1.04,4.6,22,'Strong','Momentum'],[108,79,120,1.70,7.5,41,'Weak','Momentum'],
        [108,63,116,1.48,4.4,26,'Weak','Momentum'],[33,6,29,0.34,1.3,5,'Excellent','Momentum'],
        [94,66,110,1.54,6.2,31,'Strong','Momentum'],[186,142,192,3.88,9.3,79,'Moderate','Momentum'],
        [57,31,54,0.96,3.2,14,'Strong','Momentum'],[22,8,17,0.48,2.3,10,'Excellent','Momentum'],
        [39,15,36,0.61,2.9,13,'Excellent','Momentum'],[94,69,114,2.08,7.3,39,'Strong','Momentum'],
        [99,73,110,2.43,7.6,48,'Strong','Momentum'],[31,12,30,0.50,3.2,16,'Excellent','Momentum'],
        [42,27,43,1.00,5.3,22,'Excellent','Momentum'],[154,117,157,3.05,9.2,58,'Moderate','Momentum'],
        [234,185,230,4.09,12.3,69,'Critical','Momentum'],[160,120,186,2.77,9.1,50,'Moderate','Momentum'],
        [134,96,165,2.76,8.3,41,'Weak','Momentum'],[52,24,60,1.03,5.0,21,'Strong','Momentum'],
        [46,5,49,0.28,1.0,6,'Excellent','Momentum'],
        [26,37,27,1.16,2.7,13,'Excellent','Sentiment'],[85,62,71,1.20,6.0,8,'Strong','Sentiment'],
        [78,38,74,1.36,3.7,7,'Strong','Sentiment'],[21,21,36,0.63,4.0,9,'Excellent','Sentiment'],
        [41,42,46,0.92,8.1,13,'Excellent','Sentiment'],[56,52,69,1.07,9.2,16,'Strong','Sentiment'],
        [64,30,28,0.92,5.1,2,'Strong','Sentiment'],[55,48,74,1.24,7.5,26,'Strong','Sentiment'],
        [76,85,113,1.24,11.4,27,'Strong','Sentiment'],[91,81,104,1.04,5.6,40,'Strong','Sentiment'],
        [84,39,60,0.96,2.5,11,'Strong','Sentiment'],[64,51,101,0.86,5.8,23,'Strong','Sentiment'],
        [70,69,120,1.20,6.5,36,'Strong','Sentiment'],[77,105,178,2.55,6.4,16,'Strong','Sentiment'],
        [109,68,87,1.00,7.4,29,'Weak','Sentiment'],[73,68,97,0.91,5.1,34,'Strong','Sentiment'],
        [54,27,47,0.59,5.3,12,'Strong','Sentiment'],[51,61,97,0.81,6.5,19,'Strong','Sentiment'],
        [91,71,121,1.37,4.3,18,'Strong','Sentiment'],[73,102,182,2.79,4.4,19,'Strong','Sentiment'],
        [73,50,76,0.72,3.1,20,'Strong','Sentiment'],[84,94,140,2.24,6.8,18,'Strong','Sentiment'],
        [93,77,104,1.17,5.3,7,'Strong','Sentiment'],[99,130,227,3.97,5.5,15,'Strong','Sentiment'],
        [146,84,139,1.09,4.0,17,'Weak','Sentiment'],[113,108,137,1.48,4.8,15,'Weak','Sentiment'],
        [81,48,62,1.62,2.6,3,'Strong','Sentiment'],[56,48,68,1.34,3.7,9,'Strong','Sentiment'],
        [82,92,174,3.29,0.0,13,'Strong','Sentiment'],[106,116,188,3.63,10.1,16,'Weak','Sentiment'],
        [118,50,0,1.38,7.6,11,'Weak','Sentiment'],
        [91,45,125,0.82,3.4,23,'Strong','Risk'],[65,27,78,0.86,4.5,29,'Strong','Risk'],
        [83,60,84,1.09,7.3,27,'Strong','Risk'],[109,81,121,1.28,6.8,51,'Weak','Risk'],
        [106,77,114,1.07,5.5,51,'Weak','Risk'],[109,81,121,1.28,6.8,51,'Weak','Risk'],
        [106,77,114,1.07,5.5,51,'Weak','Risk'],[89,65,78,0.86,5.1,26,'Strong','Risk'],
        [53,33,47,0.64,5.0,17,'Strong','Risk'],[80,55,80,1.01,7.5,24,'Strong','Risk'],
        [117,81,124,1.03,4.5,24,'Weak','Risk'],[99,71,142,1.10,6.2,42,'Strong','Risk'],
        [95,69,130,1.28,7.4,50,'Strong','Risk'],[116,87,131,1.47,8.4,40,'Weak','Risk'],
        [108,80,121,1.30,8.5,37,'Weak','Risk'],[134,83,167,1.16,5.7,43,'Weak','Risk'],
        [79,43,107,1.05,5.9,37,'Strong','Risk'],[71,46,89,0.86,6.4,25,'Strong','Risk'],
        [97,71,113,1.17,8.8,31,'Strong','Risk'],[84,57,91,0.85,5.5,31,'Strong','Risk'],
        [87,63,101,0.90,5.6,41,'Strong','Risk'],[104,77,119,1.09,7.3,48,'Weak','Risk'],
        [87,62,100,1.00,7.2,28,'Strong','Risk'],[168,128,172,1.49,9.7,56,'Moderate','Risk'],
        [65,45,51,0.74,3.9,17,'Strong','Risk'],[39,24,38,0.61,4.7,17,'Excellent','Risk'],
        [39,24,39,0.59,5.0,19,'Excellent','Risk'],[93,68,96,1.05,7.9,29,'Strong','Risk'],
        [188,143,197,1.66,9.9,51,'Moderate','Risk'],[174,131,174,1.55,10.8,50,'Moderate','Risk'],
        [187,143,201,1.39,8.9,53,'Moderate','Risk']
    ];

    const CAT_DIM_COUNT = 6;
    const GAP = 2;
    const BASE_LEFT = 5;
    const BASE_TOP = 10;
    const GRID_W = (100 - BASE_LEFT - GAP) / CAT_DIM_COUNT - GAP;
    const GRID_H = (100 - BASE_TOP - GAP) / CAT_DIM_COUNT - GAP;
    const CAT_DIM = 7;
    const SYM_SIZE = 4;

    function scatterData(data, dx, dy) {
        return data.map(function (d) { var r = [d[dx], d[dy]]; r[CAT_DIM] = d[CAT_DIM]; return r; });
    }

    function makeGrids() {
        var idx = 0, grid = [], xA = [], yA = [], ser = [];
        for (var i = 0; i < CAT_DIM_COUNT; i++) {
            for (var j = 0; j < CAT_DIM_COUNT; j++) {
                if (CAT_DIM_COUNT - i + j >= CAT_DIM_COUNT) continue;
                grid.push({ left: BASE_LEFT + i * (GRID_W + GAP) + '%', top: BASE_TOP + j * (GRID_H + GAP) + '%', width: GRID_W + '%', height: GRID_H + '%' });
                xA.push({ splitNumber: 3, position: 'top', axisLine: { show: j === 0, onZero: false }, axisTick: { show: j === 0, inside: true }, axisLabel: { show: j === 0, fontSize: 9 }, type: 'value', gridIndex: idx, scale: true });
                yA.push({ splitNumber: 3, position: 'right', axisLine: { show: i === CAT_DIM_COUNT - 1, onZero: false }, axisTick: { show: i === CAT_DIM_COUNT - 1, inside: true }, axisLabel: { show: i === CAT_DIM_COUNT - 1, fontSize: 9 }, type: 'value', gridIndex: idx, scale: true });
                ser.push({ type: 'scatter', symbolSize: SYM_SIZE, xAxisIndex: idx, yAxisIndex: idx, data: scatterData(rawData, i, j) });
                idx++;
            }
        }
        return { grid: grid, xAxis: xA, yAxis: yA, series: ser };
    }

    var g = makeGrids();
    var fontFamily = getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim() || 'sans-serif';

    var chart = echarts.init(el);
    chart.setOption({
        animation: false,
        backgroundColor: 'transparent',
        textStyle: { fontFamily, fontSize: 10 },
        brush: {
            brushLink: 'all',
            xAxisIndex: g.xAxis.map(function (_, i) { return i; }),
            yAxisIndex: g.yAxis.map(function (_, i) { return i; }),
            inBrush: { opacity: 1 }
        },
        visualMap: {
            type: 'piecewise',
            categories: ['Momentum', 'Sentiment', 'Risk'],
            dimension: CAT_DIM,
            orient: 'horizontal',
            top: 0,
            left: 'center',
            inRange: { color: ['#a8d8ea', '#c3b1e1', '#f6c6a8'] },
            outOfRange: { color: '#ddd' },
            textStyle: { fontFamily, fontSize: 10, color: '#222' },
            seriesIndex: g.series.map(function (_, i) { return i + 1; })
        },
        tooltip: { trigger: 'item', backgroundColor: '#080808', borderColor: '#080808', textStyle: { color: '#fff', fontFamily, fontSize: 11 } },
        parallelAxis: [
            { dim: 0, name: schema[0].text },
            { dim: 1, name: schema[1].text },
            { dim: 2, name: schema[2].text },
            { dim: 3, name: schema[3].text },
            { dim: 4, name: schema[4].text },
            { dim: 5, name: schema[5].text },
            { dim: 6, name: schema[6].text, type: 'category', data: ['Excellent', 'Strong', 'Weak', 'Moderate', 'Critical'] }
        ],
        parallel: {
            bottom: '5%', left: '2%', height: '30%', width: '55%',
            parallelAxisDefault: {
                type: 'value', nameLocation: 'end', nameGap: 20,
                splitNumber: 3,
                nameTextStyle: { fontSize: 11, fontFamily, color: '#222' },
                axisLine: { lineStyle: { color: '#000' } },
                axisTick: { lineStyle: { color: '#000' } },
                splitLine: { show: false },
                axisLabel: { color: '#222', fontSize: 9 }
            }
        },
        xAxis: g.xAxis,
        yAxis: g.yAxis,
        grid: g.grid,
        series: [
            { name: 'parallel', type: 'parallel', smooth: true, lineStyle: { width: 1, opacity: 0.3 }, data: rawData },
            ...g.series
        ]
    });

    window.addEventListener('resize', function () { chart.resize(); });
})();

// === WIRE UP REGIME + LIVE API ===
// Map backend regime_state (1-indexed, variable num_states) to cerebral 0-4 index
function mapBackendRegime(regimeState, numStates) {
    // Backend: 1=low-vol ... N=high-vol. Map to [ACCUM=0, MARKUP=1, DISTRIB=2, MARKDN=3, CRISIS=4]
    if (numStates <= 5) return Math.max(0, Math.min(4, regimeState - 1));
    // For >5 states, bucket into 5 zones
    return Math.max(0, Math.min(4, Math.floor((regimeState - 1) / numStates * 5)));
}

async function fetchLiveRegime() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var data = await CerebralAPI.fetchRegime();
        if (!data || data.regime_state === undefined) return false;

        var numStates = (data.regime_probabilities || []).length || 5;
        var idx = mapBackendRegime(data.regime_state, numStates);
        var probs = data.regime_probabilities || [];

        // Map variable-length probs to 5 slots
        var mapped = [0.2, 0.2, 0.2, 0.2, 0.2];
        if (probs.length === 5) {
            mapped = probs;
        } else if (probs.length > 0) {
            mapped = [0, 0, 0, 0, 0];
            probs.forEach(function(p, i) {
                var bucket = Math.min(4, Math.floor(i / probs.length * 5));
                mapped[bucket] += p;
            });
        }
        var maxProb = Math.max.apply(null, mapped);

        currentRegime = { state: idx, confidence: maxProb, probabilities: mapped };
        regimeHistory.push({
            timestamp: Date.now(),
            state: idx,
            regime: RG_FULL[idx],
            confidence: maxProb,
            probabilities: mapped.slice(),
        });

        // Store additional data for VaR section
        window._liveRegimeData = {
            var_95: data.var_95,
            volatility_forecast: data.volatility_forecast,
            volatility_filtered: data.volatility_filtered,
            transition_matrix: data.transition_matrix,
            regime_name: data.regime_name,
        };

        console.log('[Cerebral] Live regime:', RG_FULL[idx], '(' + (maxProb * 100).toFixed(1) + '%)');
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live regime fetch failed:', e.message);
        return false;
    }
}

async function fetchLiveVaR() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var results = await Promise.all([
            CerebralAPI.fetchVaR(95),
            CerebralAPI.fetchVaR(99),
            CerebralAPI.fetchVaR(99.5),
            CerebralAPI.fetchVolatility(),
        ]);
        if (!results[0]) return false;
        window._liveVaRData = {
            var95: results[0],
            var99: results[1],
            var995: results[2],
            volatility: results[3],
        };
        console.log('[Cerebral] Live VaR loaded — 95%:', results[0].var_value?.toFixed(2) + '%');
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live VaR fetch failed:', e.message);
        return false;
    }
}

async function fetchLiveCircuitBreakers() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var data = await CerebralAPI.fetchCircuitBreakers();
        if (!data) return false;
        window._liveCBData = data;
        console.log('[Cerebral] Live circuit breakers loaded');
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live CB fetch failed:', e.message);
        return false;
    }
}

async function fetchLiveAgentStatus() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var data = await CerebralAPI.fetchAgentStatus();
        if (!data || !data.agents) return false;
        window._liveAgentData = data;
        var statusEl = document.getElementById('mlSignalStatus');
        if (statusEl) statusEl.textContent = 'LIVE';
        var agStatusEl = document.getElementById('agLoadStatus');
        if (agStatusEl) agStatusEl.textContent = 'LIVE';
        console.log('[Cerebral] Live agent status loaded');
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live agent status fetch failed:', e.message);
        return false;
    }
}

async function fetchLiveDrawdown() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var data = await CerebralAPI.fetchDrawdown();
        if (!data) return false;
        window._liveDrawdownData = data;
        console.log('[Cerebral] Live drawdown loaded');
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live drawdown fetch failed:', e.message);
        return false;
    }
}

async function fetchLiveRegimeStats() {
    if (!CerebralAPI.isAvailable()) return false;
    try {
        var data = await CerebralAPI.fetchRegimeStatistics();
        if (!data) return false;
        window._liveRegimeStats = data;
        console.log('[Cerebral] Live regime statistics loaded');
        return true;
    } catch (e) {
        console.warn('[Cerebral] Live regime stats fetch failed:', e.message);
        return false;
    }
}

// Master refresh: try live API, then update displays
async function cerebralRefresh() {
    var liveRegime = await fetchLiveRegime();
    if (!liveRegime) {
        await detectRegime();
    }
    updateRegimeDisplay();
    renderMLSignals();

    // Parallel fetch: VaR, circuit breakers, agents, drawdown, regime stats
    await Promise.all([
        fetchLiveVaR(),
        fetchLiveCircuitBreakers(),
        fetchLiveAgentStatus(),
        fetchLiveDrawdown(),
        fetchLiveRegimeStats(),
    ]);
}

// Initialize
(async function() {
    await CerebralAPI.checkHealth();
    if (CerebralAPI.isAvailable()) {
        console.log('[Cerebral] API connected — using live data');
        await cerebralRefresh();
    } else {
        console.log('[Cerebral] API unavailable — using deterministic fallbacks');
        startRegimeDetection();
    }
})();

// Periodic refresh
setInterval(async function() {
    if (CerebralAPI.isAvailable()) {
        await cerebralRefresh();
    } else {
        // Re-check health periodically
        await CerebralAPI.checkHealth();
        if (!CerebralAPI.isAvailable()) {
            updateRegimeDisplay();
            renderMLSignals();
        }
    }
}, 5000);

console.log('CORTEX Cerebral initialized');
console.log('Regime Detection: Active (live API preferred, CoinGecko HMM fallback)');
console.log('Decision Pipeline: Cycling');
console.log('Debate Theater: Live API only');
