// history-service.js — Fetches real data from Cortex API for the History page.
// Falls back to mock data when API is unavailable.

const HistoryService = (function () {
    const POLL_MS = (typeof CORTEX_CONFIG !== 'undefined' && CORTEX_CONFIG.POLL_INTERVAL) || 30000;
    let _intervals = [];

    // ── Execution Log + Stats ─────────────────────────────────────────

    async function fetchExecutionLog(limit = 50) {
        const data = await CortexAPI.get(`/execution/log?limit=${limit}`);
        return data ? data.entries : null;
    }

    async function fetchExecutionStats() {
        return await CortexAPI.get('/execution/stats');
    }

    // ── Guardian / Kelly ──────────────────────────────────────────────

    async function fetchKellyStats() {
        return await CortexAPI.get('/guardian/kelly-stats');
    }

    // ── Debates ───────────────────────────────────────────────────────

    async function fetchRecentDebates(limit = 20) {
        const data = await CortexAPI.get(`/guardian/debates/recent?limit=${limit}`);
        return data ? data.transcripts : null;
    }

    async function fetchDebateStats(hours = 24) {
        return await CortexAPI.get(`/guardian/debates/stats?hours=${hours}`);
    }

    async function fetchDebateStorageStats() {
        return await CortexAPI.get('/guardian/debates/storage/stats');
    }

    async function fetchDebatesByStrategy(strategy, limit = 50) {
        const data = await CortexAPI.get(`/guardian/debates/by-strategy/${strategy}?limit=${limit}`);
        return data ? data.transcripts : null;
    }

    // ── Circuit Breakers ──────────────────────────────────────────────

    async function fetchCircuitBreakers() {
        return await CortexAPI.get('/guardian/circuit-breakers');
    }

    // ── Narrator Briefing (analyst reports) ───────────────────────────

    async function fetchBriefing() {
        return await CortexAPI.get('/narrator/briefing');
    }

    // ── Strategy Config ───────────────────────────────────────────────

    async function fetchStrategyConfig() {
        return await CortexAPI.get('/strategies/config');
    }

    // ── Agent Status ──────────────────────────────────────────────────

    async function fetchAgentStatus() {
        return await CortexAPI.get('/agents/status');
    }

    // ── Derived data: daily PnL from execution log ────────────────────

    function computeDailyPnl(entries) {
        if (!entries || !entries.length) return null;
        var byDate = {};
        entries.forEach(function (e) {
            if (!e.timestamp) return;
            var d = new Date(typeof e.timestamp === 'number' ? e.timestamp * 1000 : e.timestamp);
            var key = d.getUTCFullYear() + '-' + String(d.getUTCMonth() + 1).padStart(2, '0') + '-' + String(d.getUTCDate()).padStart(2, '0');
            if (!byDate[key]) byDate[key] = { pnl: 0, count: 0 };
            byDate[key].pnl += (e.pnl || 0);
            byDate[key].count += 1;
        });
        var keys = Object.keys(byDate).sort();
        if (!keys.length) return null;
        var cumPnl = 10000;
        return keys.map(function (k) {
            var change = byDate[k].pnl / cumPnl;
            cumPnl += byDate[k].pnl;
            return { Date: k, Close: cumPnl, Change: change };
        });
    }

    // ── Derived data: strategy breakdown from execution log ───────────

    function computeStrategyBreakdown(entries, strategyConfig) {
        if (!entries || !entries.length) return null;
        var stratMap = { lp: 'LP Rebalancing', arb: 'Arbitrage', perp: 'Perpetuals' };
        var nameMap = {};
        Object.keys(stratMap).forEach(function (k) { nameMap[stratMap[k]] = k; nameMap[k] = k; });

        var stats = {};
        entries.forEach(function (e) {
            var sKey = nameMap[e.strategy] || e.strategy || 'unknown';
            if (!stats[sKey]) stats[sKey] = { trades: 0, wins: 0, pnl: 0 };
            stats[sKey].trades += 1;
            var pnl = e.pnl || 0;
            stats[sKey].pnl += pnl;
            if (pnl > 0) stats[sKey].wins += 1;
        });

        var configs = {};
        if (strategyConfig && strategyConfig.strategies) {
            strategyConfig.strategies.forEach(function (s) { configs[s.key] = s; });
        }

        return ['lp', 'arb', 'perp'].map(function (key) {
            var s = stats[key] || { trades: 0, wins: 0, pnl: 0 };
            var cfg = configs[key] || {};
            var alloc = cfg.allocation ? cfg.allocation + '%' : (key === 'lp' ? '40%' : '30%');
            var params = (cfg.params || []).map(function (p) { return p.key + ': ' + p.val; }).join(' · ');
            return {
                name: stratMap[key] || key,
                alloc: alloc,
                pnl: +s.pnl.toFixed(2),
                trades: s.trades,
                winRate: s.trades > 0 ? +(s.wins / s.trades * 100).toFixed(1) : 0,
                maxPnl: 12000,
                color: s.pnl >= 0 ? 'var(--green)' : 'var(--red)',
                params: params || 'No config available',
            };
        });
    }

    // ── Derived data: monthly strategy chart from execution log ───────

    function computeMonthlyStrategy(entries) {
        if (!entries || !entries.length) return null;
        var stratMap = { 'LP Rebalancing': 'lp', 'Arbitrage': 'arb', 'Perpetuals': 'perp', lp: 'lp', arb: 'arb', perp: 'perp' };
        var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        var byMonth = {};

        entries.forEach(function (e) {
            if (!e.timestamp) return;
            var d = new Date(typeof e.timestamp === 'number' ? e.timestamp * 1000 : e.timestamp);
            var mKey = months[d.getUTCMonth()];
            var sKey = stratMap[e.strategy] || 'lp';
            if (!byMonth[mKey]) byMonth[mKey] = { lp: 0, arb: 0, perp: 0 };
            byMonth[mKey][sKey] += (e.pnl || 0);
        });

        var now = new Date();
        var activeMonths = months.slice(0, now.getMonth() + 1);
        return activeMonths.map(function (m) {
            var d = byMonth[m] || { lp: 0, arb: 0, perp: 0 };
            return { month: m, lp: +d.lp.toFixed(0), arb: +d.arb.toFixed(0), perp: +d.perp.toFixed(0) };
        });
    }

    // ── Derived data: agent attribution from agent status ─────────────

    function buildAgentAttribution(agentStatus) {
        if (!agentStatus || !agentStatus.agents) return null;
        var agents = agentStatus.agents;
        var result = [];
        var roleMap = {
            momentum: { name: 'Momentum Agent', role: 'Analyst' },
            meanrev: { name: 'Mean Reversion Agent', role: 'Analyst' },
            sentiment: { name: 'Sentiment Agent', role: 'Analyst' },
            risk: { name: 'Risk Manager', role: 'Decision' },
            arbitrage: { name: 'Arbitrage Agent', role: 'Analyst' },
        };

        Object.keys(agents).forEach(function (key) {
            var a = agents[key];
            var meta = roleMap[key] || { name: a.name || key, role: 'Agent' };
            var conf = parseInt(a.confidence || '0', 10);
            result.push({
                name: meta.name,
                role: meta.role,
                trades: 0,
                winRate: conf || 0,
                pnl: 0,
                status: a.status || 'UNKNOWN',
                signal: a.signal || '—',
                analysis: a.analysis || '',
            });
        });
        return result.length > 0 ? result : null;
    }

    // ── Aggregated fetch: one call for all history sections ───────────

    async function fetchAll() {
        const [execLog, execStats, kelly, debates, debateStats, circuitBreakers, stratConfig, agentSt] =
            await Promise.allSettled([
                fetchExecutionLog(500),
                fetchExecutionStats(),
                fetchKellyStats(),
                fetchRecentDebates(20),
                fetchDebateStats(168),
                fetchCircuitBreakers(),
                fetchStrategyConfig(),
                fetchAgentStatus(),
            ]);

        var entries = execLog.status === 'fulfilled' ? execLog.value : null;
        var sCfg = stratConfig.status === 'fulfilled' ? stratConfig.value : null;
        var aSt = agentSt.status === 'fulfilled' ? agentSt.value : null;

        return {
            executionLog: entries,
            executionStats: execStats.status === 'fulfilled' ? execStats.value : null,
            kellyStats: kelly.status === 'fulfilled' ? kelly.value : null,
            debates: debates.status === 'fulfilled' ? debates.value : null,
            debateStats: debateStats.status === 'fulfilled' ? debateStats.value : null,
            circuitBreakers: circuitBreakers.status === 'fulfilled' ? circuitBreakers.value : null,
            strategyConfig: sCfg,
            agentStatus: aSt,
            dailyPnl: computeDailyPnl(entries),
            strategyBreakdown: computeStrategyBreakdown(entries, sCfg),
            monthlyStrategy: computeMonthlyStrategy(entries),
            agentAttribution: buildAgentAttribution(aSt),
        };
    }

    // ── Mock fallbacks (internal only, used when API returns null) ─────

    function _mockExecutionLog(count) {
        count = count || 15;
        var pairs = ['SOL/USDC', 'JUP/SOL', 'RAY/USDC', 'ORCA/SOL', 'BONK/SOL', 'WIF/USDC', 'JTO/SOL', 'PYTH/USDC'];
        var strats = ['LP Rebalancing', 'Arbitrage', 'Perpetuals'];
        var entries = [];
        for (var i = 0; i < count; i++) {
            var daysAgo = Math.floor(Math.random() * 120);
            var ts = new Date(Date.now() - daysAgo * 86400000);
            var pnl = Math.random() * 800 - 300;
            entries.push({
                token: pairs[Math.floor(Math.random() * pairs.length)],
                direction: Math.random() > 0.5 ? 'buy' : 'sell',
                amount: +(100 + Math.random() * 900).toFixed(2),
                price_usd: +(10 + Math.random() * 200).toFixed(2),
                pnl: +pnl.toFixed(2),
                strategy: strats[Math.floor(Math.random() * strats.length)],
                confidence: +(50 + Math.random() * 45).toFixed(1),
                timestamp: ts.toISOString(),
            });
        }
        entries.sort(function (a, b) { return new Date(b.timestamp) - new Date(a.timestamp); });
        return entries;
    }

    function _mockDebates(count) {
        count = count || 6;
        var topics = [
            { pair: 'SOL/USDC', action: 'LONG LP Position', verdict: 'approved' },
            { pair: 'JUP/SOL', action: 'Arbitrage Execution', verdict: 'approved' },
            { pair: 'RAY/USDC', action: 'SHORT Perpetual', verdict: 'rejected' },
            { pair: 'ORCA/SOL', action: 'LP Rebalance', verdict: 'modified' },
            { pair: 'BONK/SOL', action: 'LONG Perpetual', verdict: 'rejected' },
            { pair: 'WIF/USDC', action: 'Arbitrage Execution', verdict: 'approved' },
        ];
        return topics.slice(0, count).map(function (d, i) {
            var daysAgo = Math.floor(Math.random() * 30);
            var confidence = +(55 + Math.random() * 40).toFixed(1);
            return {
                id: 'mock_debate_' + i,
                token: d.pair.split('/')[0],
                strategy: d.action,
                direction: d.action.includes('SHORT') ? 'sell' : 'buy',
                verdict: d.verdict,
                risk_score: +(20 + Math.random() * 60).toFixed(1),
                confidence: confidence,
                timestamp: new Date(Date.now() - daysAgo * 86400000).toISOString(),
                rounds: [
                    { speaker: 'Trader', text: 'Opportunity identified — ' + confidence + '% confidence, favorable risk/reward ratio.' },
                    { speaker: 'Risk Mgr', text: d.verdict === 'rejected' ? 'Position size exceeds risk limits.' : 'Acceptable risk profile. VaR within bounds.' },
                    { speaker: 'Trader', text: d.verdict === 'rejected' ? 'Acknowledged. Risk remains elevated.' : 'Confirmed entry parameters.' },
                    { speaker: 'PM', text: d.verdict === 'approved' ? 'Approved. Execute.' : d.verdict === 'rejected' ? 'Vetoed.' : 'Approved with modifications.' },
                ],
            };
        });
    }

    function _mockAnalystReports(count) {
        count = count || 12;
        var types = [
            { type: 'Technical', outputs: ['Support/resistance levels identified', 'Trend analysis: bullish continuation pattern', 'RSI divergence detected on 4H timeframe'] },
            { type: 'On-Chain', outputs: ['Whale accumulation detected', 'Protocol TVL increased 12% in 24h', 'Smart money flow: net positive $2.3M'] },
            { type: 'Sentiment', outputs: ['Social sentiment score: 72/100 (bullish)', 'Community sentiment shifting positive', 'Twitter mention volume up 45%'] },
            { type: 'Macro', outputs: ['BTC dominance declining — alt season signal', 'Risk-on environment: DXY weakening', 'Fear/Greed Index: 68 (Greed)'] },
        ];
        var reports = [];
        for (var i = 0; i < count; i++) {
            var analyst = types[Math.floor(Math.random() * types.length)];
            var hoursAgo = Math.floor(Math.random() * 168);
            reports.push({
                type: analyst.type,
                output: analyst.outputs[Math.floor(Math.random() * analyst.outputs.length)],
                hoursAgo: hoursAgo,
                timeStr: hoursAgo < 1 ? 'just now' : hoursAgo < 24 ? hoursAgo + 'h ago' : Math.floor(hoursAgo / 24) + 'd ago',
            });
        }
        reports.sort(function (a, b) { return a.hoursAgo - b.hoursAgo; });
        return reports;
    }

    // ── Polling ───────────────────────────────────────────────────────

    function startPolling(callback) {
        callback();
        var id = setInterval(callback, POLL_MS);
        _intervals.push(id);
        return id;
    }

    function stopPolling() {
        _intervals.forEach(clearInterval);
        _intervals = [];
    }

    return {
        fetchAll: fetchAll,
        fetchExecutionLog: fetchExecutionLog,
        fetchExecutionStats: fetchExecutionStats,
        fetchKellyStats: fetchKellyStats,
        fetchRecentDebates: fetchRecentDebates,
        fetchDebateStats: fetchDebateStats,
        fetchDebateStorageStats: fetchDebateStorageStats,
        fetchDebatesByStrategy: fetchDebatesByStrategy,
        fetchCircuitBreakers: fetchCircuitBreakers,
        fetchBriefing: fetchBriefing,
        fetchStrategyConfig: fetchStrategyConfig,
        fetchAgentStatus: fetchAgentStatus,
        computeDailyPnl: computeDailyPnl,
        computeStrategyBreakdown: computeStrategyBreakdown,
        computeMonthlyStrategy: computeMonthlyStrategy,
        buildAgentAttribution: buildAgentAttribution,
        _mockExecutionLog: _mockExecutionLog,
        _mockDebates: _mockDebates,
        _mockAnalystReports: _mockAnalystReports,
        startPolling: startPolling,
        stopPolling: stopPolling,
    };
})();
