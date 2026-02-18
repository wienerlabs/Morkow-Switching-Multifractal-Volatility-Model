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

    // ── Aggregated fetch: one call for all history sections ───────────

    async function fetchAll() {
        const [execLog, execStats, kelly, debates, debateStats, circuitBreakers] =
            await Promise.allSettled([
                fetchExecutionLog(100),
                fetchExecutionStats(),
                fetchKellyStats(),
                fetchRecentDebates(20),
                fetchDebateStats(168),   // 7 days
                fetchCircuitBreakers(),
            ]);

        return {
            executionLog: execLog.status === 'fulfilled' ? execLog.value : null,
            executionStats: execStats.status === 'fulfilled' ? execStats.value : null,
            kellyStats: kelly.status === 'fulfilled' ? kelly.value : null,
            debates: debates.status === 'fulfilled' ? debates.value : null,
            debateStats: debateStats.status === 'fulfilled' ? debateStats.value : null,
            circuitBreakers: circuitBreakers.status === 'fulfilled' ? circuitBreakers.value : null,
        };
    }

    // ── Mock data generators (fallback when API unavailable) ──────────

    function mockExecutionLog(count = 15) {
        const pairs = ['SOL/USDC', 'JUP/SOL', 'RAY/USDC', 'ORCA/SOL', 'BONK/SOL', 'WIF/USDC', 'JTO/SOL', 'PYTH/USDC'];
        const strats = ['LP Rebalancing', 'Arbitrage', 'Perpetuals'];
        const entries = [];
        for (let i = 0; i < count; i++) {
            const daysAgo = Math.floor(Math.random() * 120);
            const ts = new Date(Date.now() - daysAgo * 86400000);
            const pnl = Math.random() * 800 - 300;
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
        entries.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        return entries;
    }

    function mockExecutionStats() {
        return {
            total_trades: 204,
            successful: 147,
            failed: 57,
            avg_slippage_bps: 12.4,
            total_volume_usd: 284350.0,
        };
    }

    function mockDebates(count = 6) {
        const topics = [
            { pair: 'SOL/USDC', action: 'LONG LP Position', verdict: 'approved' },
            { pair: 'JUP/SOL', action: 'Arbitrage Execution', verdict: 'approved' },
            { pair: 'RAY/USDC', action: 'SHORT Perpetual', verdict: 'rejected' },
            { pair: 'ORCA/SOL', action: 'LP Rebalance', verdict: 'modified' },
            { pair: 'BONK/SOL', action: 'LONG Perpetual', verdict: 'rejected' },
            { pair: 'WIF/USDC', action: 'Arbitrage Execution', verdict: 'approved' },
        ];
        return topics.slice(0, count).map((d, i) => {
            const daysAgo = Math.floor(Math.random() * 30);
            const confidence = +(55 + Math.random() * 40).toFixed(1);
            return {
                id: `mock_debate_${i}`,
                token: d.pair.split('/')[0],
                strategy: d.action,
                direction: d.action.includes('SHORT') ? 'sell' : 'buy',
                verdict: d.verdict,
                risk_score: +(20 + Math.random() * 60).toFixed(1),
                confidence: confidence,
                timestamp: new Date(Date.now() - daysAgo * 86400000).toISOString(),
                rounds: [
                    { speaker: 'Trader', text: `Opportunity identified — ${confidence}% confidence, favorable risk/reward ratio.` },
                    { speaker: 'Risk Mgr', text: d.verdict === 'rejected' ? 'Position size exceeds risk limits. Correlated exposure at 14.2%, approaching 15% hard limit.' : 'Acceptable risk profile. VaR within bounds. Slippage estimate: 0.3%.' },
                    { speaker: 'Trader', text: d.verdict === 'rejected' ? 'Acknowledged. Proposed reduced size but risk remains elevated.' : 'Confirmed entry parameters. Stop-loss and take-profit levels set.' },
                    { speaker: 'PM', text: d.verdict === 'approved' ? 'Approved. All criteria met. Execute with standard parameters.' : d.verdict === 'rejected' ? 'Vetoed. Risk/reward insufficient given current portfolio exposure.' : 'Approved with modifications. Reduced position size by 30%.' },
                ],
            };
        });
    }

    function mockAnalystReports(count = 12) {
        const types = [
            { type: 'Technical', outputs: ['Support/resistance levels identified', 'Trend analysis: bullish continuation pattern', 'RSI divergence detected on 4H timeframe', 'Bollinger Band squeeze indicating breakout imminent', 'MACD histogram turning positive'] },
            { type: 'On-Chain', outputs: ['Whale accumulation detected: 3 wallets added 50K+ SOL', 'Protocol TVL increased 12% in 24h', 'Smart money flow: net positive $2.3M', 'Distribution pattern from early investors detected', 'DEX volume spike: 3x average on Raydium'] },
            { type: 'Sentiment', outputs: ['Social sentiment score: 72/100 (bullish)', 'FUD detected: exchange withdrawal rumors — low credibility', 'Community sentiment shifting positive after protocol upgrade', 'Narrative tracking: DeFi rotation gaining momentum', 'Twitter mention volume up 45% — organic growth pattern'] },
            { type: 'Macro', outputs: ['BTC dominance declining — alt season signal', 'Risk-on environment: DXY weakening', 'Correlation matrix: SOL decorrelating from BTC (0.62 → 0.48)', 'Fear/Greed Index: 68 (Greed) — caution advised', 'Fed rate decision neutral — no immediate impact expected'] },
        ];
        const reports = [];
        for (let i = 0; i < count; i++) {
            const analyst = types[Math.floor(Math.random() * types.length)];
            const hoursAgo = Math.floor(Math.random() * 168);
            reports.push({
                type: analyst.type,
                output: analyst.outputs[Math.floor(Math.random() * analyst.outputs.length)],
                hoursAgo,
                timeStr: hoursAgo < 1 ? 'just now' : hoursAgo < 24 ? hoursAgo + 'h ago' : Math.floor(hoursAgo / 24) + 'd ago',
            });
        }
        reports.sort((a, b) => a.hoursAgo - b.hoursAgo);
        return reports;
    }

    // ── Polling ───────────────────────────────────────────────────────

    function startPolling(callback) {
        callback();
        const id = setInterval(callback, POLL_MS);
        _intervals.push(id);
        return id;
    }

    function stopPolling() {
        _intervals.forEach(clearInterval);
        _intervals = [];
    }

    return {
        fetchAll,
        fetchExecutionLog,
        fetchExecutionStats,
        fetchKellyStats,
        fetchRecentDebates,
        fetchDebateStats,
        fetchDebateStorageStats,
        fetchDebatesByStrategy,
        fetchCircuitBreakers,
        fetchBriefing,
        mockExecutionLog,
        mockExecutionStats,
        mockDebates,
        mockAnalystReports,
        startPolling,
        stopPolling,
    };
})();
