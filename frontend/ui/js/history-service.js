// history-service.js — Fetches real data from Cortex API for the History page.

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

    function computeDailyPnl(entries, startingBalance) {
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
        var cumPnl = (typeof startingBalance === 'number' && startingBalance > 0) ? startingBalance : 0;
        return keys.map(function (k) {
            var change = cumPnl > 0 ? byDate[k].pnl / cumPnl : 0;
            cumPnl += byDate[k].pnl;
            return { Date: k, Close: cumPnl, Change: change };
        });
    }

    // ── Derived data: strategy breakdown from execution log ───────────

    function computeStrategyBreakdown(entries, strategyConfig, dailyPnl) {
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

        var computedMaxPnl = 0;
        if (dailyPnl && dailyPnl.length) {
            computedMaxPnl = Math.max.apply(null, dailyPnl.map(function (d) { return d.Close; }));
        }

        return ['lp', 'arb', 'perp'].map(function (key) {
            var s = stats[key] || { trades: 0, wins: 0, pnl: 0 };
            var cfg = configs[key] || {};
            var alloc = cfg.allocation ? cfg.allocation + '%' : '—';
            var params = (cfg.params || []).map(function (p) { return p.key + ': ' + p.val; }).join(' · ');
            return {
                name: stratMap[key] || key,
                alloc: alloc,
                pnl: +s.pnl.toFixed(2),
                trades: s.trades,
                winRate: s.trades > 0 ? +(s.wins / s.trades * 100).toFixed(1) : 0,
                maxPnl: computedMaxPnl,
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
        var eStats = execStats.status === 'fulfilled' ? execStats.value : null;
        var sCfg = stratConfig.status === 'fulfilled' ? stratConfig.value : null;
        var aSt = agentSt.status === 'fulfilled' ? agentSt.value : null;

        // Extract starting balance: API field > first entry portfolio_value > 0
        var startingBalance = 0;
        if (eStats && typeof eStats.initial_balance === 'number') {
            startingBalance = eStats.initial_balance;
        } else if (eStats && typeof eStats.starting_balance === 'number') {
            startingBalance = eStats.starting_balance;
        } else if (eStats && typeof eStats.total_volume_usd === 'number' && eStats.total_volume_usd > 0) {
            startingBalance = 0;
        }
        if (startingBalance === 0 && entries && entries.length) {
            var firstEntry = entries[0];
            if (typeof firstEntry.portfolio_value === 'number' && firstEntry.portfolio_value > 0) {
                startingBalance = firstEntry.portfolio_value;
            }
        }
        if (startingBalance === 0) {
            console.warn('[HistoryService] No starting balance available — using 0');
        }

        var pnl = computeDailyPnl(entries, startingBalance);

        return {
            executionLog: entries,
            executionStats: eStats,
            kellyStats: kelly.status === 'fulfilled' ? kelly.value : null,
            debates: debates.status === 'fulfilled' ? debates.value : null,
            debateStats: debateStats.status === 'fulfilled' ? debateStats.value : null,
            circuitBreakers: circuitBreakers.status === 'fulfilled' ? circuitBreakers.value : null,
            strategyConfig: sCfg,
            agentStatus: aSt,
            dailyPnl: pnl,
            strategyBreakdown: computeStrategyBreakdown(entries, sCfg, pnl),
            monthlyStrategy: computeMonthlyStrategy(entries),
            agentAttribution: buildAgentAttribution(aSt),
        };
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
        startPolling: startPolling,
        stopPolling: stopPolling,
    };
})();
