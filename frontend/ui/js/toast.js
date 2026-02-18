(function () {
    const style = document.createElement('style');
    style.textContent = `
        .toast-container { position: fixed; top: 1rem; right: 1rem; z-index: 99999; display: flex; flex-direction: column; gap: 0.5rem; max-width: 380px; pointer-events: none; }
        .toast { pointer-events: auto; padding: 0.75rem 1rem; border: 1px solid var(--border); background: var(--bg); font-family: var(--font-mono); font-size: 0.65rem; display: flex; align-items: flex-start; gap: 0.6rem; opacity: 0; transform: translateX(100%); animation: toast-in 0.3s ease forwards; position: relative; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .toast.removing { animation: toast-out 0.3s ease forwards; }
        .toast-icon { font-size: 0.8rem; flex-shrink: 0; margin-top: 1px; }
        .toast-body { flex: 1; min-width: 0; }
        .toast-title { font-weight: 600; text-transform: uppercase; font-size: 0.55rem; letter-spacing: 0.5px; margin-bottom: 0.2rem; }
        .toast-msg { line-height: 1.5; color: var(--dim); }
        .toast-time { font-size: 0.5rem; color: var(--dim); margin-top: 0.25rem; }
        .toast-close { position: absolute; top: 0.4rem; right: 0.5rem; cursor: pointer; font-size: 0.7rem; color: var(--dim); background: none; border: none; padding: 0; line-height: 1; }
        .toast-close:hover { color: var(--fg); }
        .toast-bar { position: absolute; bottom: 0; left: 0; height: 2px; background: var(--fg); }
        .toast.info { border-left: 3px solid #0066cc; }
        .toast.info .toast-icon { color: #0066cc; }
        .toast.info .toast-title { color: #0066cc; }
        .toast.info .toast-bar { background: #0066cc; }
        .toast.warning { border-left: 3px solid #cc8800; }
        .toast.warning .toast-icon { color: #cc8800; }
        .toast.warning .toast-title { color: #cc8800; }
        .toast.warning .toast-bar { background: #cc8800; }
        .toast.critical { border-left: 3px solid var(--red); }
        .toast.critical .toast-icon { color: var(--red); }
        .toast.critical .toast-title { color: var(--red); }
        .toast.critical .toast-bar { background: var(--red); }
        .toast.success { border-left: 3px solid var(--green); }
        .toast.success .toast-icon { color: var(--green); }
        .toast.success .toast-title { color: var(--green); }
        .toast.success .toast-bar { background: var(--green); }
        @keyframes toast-in { from { opacity: 0; transform: translateX(100%); } to { opacity: 1; transform: translateX(0); } }
        @keyframes toast-out { from { opacity: 1; transform: translateX(0); } to { opacity: 0; transform: translateX(100%); } }
        @keyframes toast-bar-shrink { from { width: 100%; } to { width: 0%; } }
    `;
    document.head.appendChild(style);

    const container = document.createElement('div');
    container.className = 'toast-container';
    container.id = 'toastContainer';
    document.body.appendChild(container);

    const ICONS = { info: 'ℹ', warning: '⚠', critical: '✖', success: '✓' };
    const TITLES = { info: 'Info', warning: 'Warning', critical: 'Critical', success: 'Success' };
    const MAX_TOASTS = 5;
    const alertHistory = [];

    function now() {
        return new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    window.showToast = function (message, type, duration) {
        type = type || 'info';
        duration = duration || (type === 'critical' ? 8000 : type === 'warning' ? 6000 : 4000);

        alertHistory.unshift({ message: message, type: type, time: now(), timestamp: Date.now() });
        if (alertHistory.length > 50) alertHistory.length = 50;

        const el = document.createElement('div');
        el.className = 'toast ' + type;
        el.innerHTML =
            '<span class="toast-icon">' + ICONS[type] + '</span>' +
            '<div class="toast-body">' +
            '<div class="toast-title">' + TITLES[type] + '</div>' +
            '<div class="toast-msg">' + message + '</div>' +
            '<div class="toast-time">' + now() + '</div>' +
            '</div>' +
            '<button class="toast-close" onclick="this.parentElement.classList.add(\'removing\');setTimeout(()=>this.parentElement.remove(),300)">&times;</button>' +
            '<div class="toast-bar" style="animation:toast-bar-shrink ' + duration + 'ms linear forwards"></div>';

        container.appendChild(el);

        while (container.children.length > MAX_TOASTS) {
            container.removeChild(container.firstChild);
        }

        setTimeout(function () {
            if (el.parentElement) {
                el.classList.add('removing');
                setTimeout(function () { if (el.parentElement) el.remove(); }, 300);
            }
        }, duration);

        return el;
    };

    window.showDrawdownAlert = function (daily, weekly) {
        if (daily >= 5) {
            showToast('Daily drawdown ' + daily.toFixed(1) + '% — ALL STRATEGIES PAUSED', 'critical', 10000);
        } else if (daily >= 3) {
            showToast('Daily drawdown ' + daily.toFixed(1) + '% — approaching 5% limit', 'warning', 6000);
        }
        if (weekly >= 10) {
            showToast('Weekly drawdown ' + weekly.toFixed(1) + '% — FULL STOP, manual review required', 'critical', 12000);
        } else if (weekly >= 7) {
            showToast('Weekly drawdown ' + weekly.toFixed(1) + '% — approaching 10% limit', 'warning', 6000);
        }
    };

    window.showCircuitBreakerAlert = function (strategy, trigger, action) {
        var type = action === 'TRIPPED' ? 'critical' : action === 'WARNING' ? 'warning' : 'info';
        showToast(strategy + ' circuit breaker: ' + trigger + ' — ' + action, type);
    };

    window.showOracleAlert = function (source, staleness) {
        if (staleness >= 30) {
            showToast(source + ' oracle stale (' + staleness + 's) — new trades rejected', 'critical');
        } else if (staleness >= 20) {
            showToast(source + ' oracle latency elevated (' + staleness + 's)', 'warning');
        }
    };

    window.showExecutionAlert = function (failRate) {
        if (failRate >= 10) {
            showToast('Execution failure rate ' + failRate.toFixed(1) + '% — investigation required', 'critical');
        } else if (failRate >= 7) {
            showToast('Execution failure rate ' + failRate.toFixed(1) + '% — elevated', 'warning');
        }
    };

    window.getAlertHistory = function () { return alertHistory; };

    // Demo alerts on page load
    setTimeout(function () { showToast('System initialized — all monitors active', 'success'); }, 2000);
    setTimeout(function () { showToast('Daily drawdown at 1.2% — within normal range', 'info'); }, 5000);

    // Periodic simulated alerts
    setInterval(function () {
        var r = Math.random();
        if (r < 0.03) {
            showDrawdownAlert(3 + Math.random() * 3, 3 + Math.random() * 4);
        } else if (r < 0.06) {
            var strats = ['LP Rebalancing', 'Arbitrage', 'Perpetuals'];
            var triggers = ['IL exit detected', 'Failed execution', 'Stop-loss hit'];
            var idx = Math.floor(Math.random() * 3);
            showCircuitBreakerAlert(strats[idx], triggers[idx], Math.random() < 0.3 ? 'TRIPPED' : 'WARNING');
        } else if (r < 0.08) {
            var sources = ['Pyth', 'Switchboard', 'Birdeye'];
            showOracleAlert(sources[Math.floor(Math.random() * 3)], 20 + Math.floor(Math.random() * 20));
        } else if (r < 0.10) {
            showExecutionAlert(7 + Math.random() * 6);
        }
    }, 15000);
})();

