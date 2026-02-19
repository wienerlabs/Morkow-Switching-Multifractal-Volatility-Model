// Agent Status Service — polls /agents/status and merges live data into AGENTS_DATA.

const AGENT_POLL_MS = 15000;
const AGENT_RETRY_MS = 5000;

let agentServiceTimer = null;
let agentServiceConnected = false;

async function fetchAgentStatus() {
    const data = await CortexAPI.get('/agents/status');
    if (!data || !data.agents) {
        if (agentServiceConnected) {
            console.warn('[AGENTS] Lost connection to agent status API');
            agentServiceConnected = false;
        }
        return;
    }

    if (!agentServiceConnected) {
        console.log('[AGENTS] Connected to agent status API');
        agentServiceConnected = true;
    }

    // Merge live data into AGENTS_DATA, preserving fields the API doesn't provide
    const live = data.agents;
    for (const key of Object.keys(live)) {
        if (!AGENTS_DATA[key]) continue;
        const src = live[key];
        const dst = AGENTS_DATA[key];

        // Only overwrite fields that the API actually returned
        for (const field of Object.keys(src)) {
            if (src[field] !== undefined && src[field] !== null) {
                dst[field] = src[field];
            }
        }
    }

    // Update DOM
    refreshAgentCardsLive();
}

function refreshAgentCardsLive() {
    document.querySelectorAll('.agent-card').forEach(function (card) {
        const id = card.getAttribute('data-agent');
        const agent = AGENTS_DATA[id];
        if (!agent) return;

        // Status dot
        const statusDot = card.querySelector('.agent-status');
        if (statusDot) {
            statusDot.className = 'agent-status';
            if (agent.status === 'WARNING') statusDot.classList.add('warning');
            else if (agent.status !== 'ACTIVE') statusDot.classList.add('inactive');
        }

        // Signal text + class
        const metricValues = card.querySelectorAll('.agent-metrics .metric-value');
        if (metricValues.length >= 2) {
            // First metric-value = Win Rate, Second = Signal
            metricValues[0].textContent = agent.winRate || metricValues[0].textContent;
            metricValues[1].textContent = agent.signal;
            metricValues[1].className = 'metric-value ' + agent.signalClass;
        }
    });

    // Re-render expanded detail panel if open
    if (typeof expandedAgentId !== 'undefined' && expandedAgentId) {
        const expandEl = document.querySelector('.agent-expand');
        if (expandEl && AGENTS_DATA[expandedAgentId]) {
            expandEl.innerHTML = buildAgentDetail(AGENTS_DATA[expandedAgentId]);
        }
    }
}

function startAgentService() {
    console.log('[AGENTS] Starting Agent Status Service — interval: ' + (AGENT_POLL_MS / 1000) + 's');
    fetchAgentStatus();
    if (agentServiceTimer) clearInterval(agentServiceTimer);
    agentServiceTimer = setInterval(fetchAgentStatus, AGENT_POLL_MS);
}

function stopAgentService() {
    if (agentServiceTimer) {
        clearInterval(agentServiceTimer);
        agentServiceTimer = null;
    }
    agentServiceConnected = false;
    console.log('[AGENTS] Agent Status Service stopped');
}

