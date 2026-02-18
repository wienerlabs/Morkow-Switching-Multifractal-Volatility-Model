// Cross-page wallet state manager
// Reads wallet state from localStorage (set by index.html) and renders wallet nav on secondary pages
(function () {
    const WALLET_STORAGE_KEY = 'cortex_wallet';

    const WALLETS = [
        { name: 'Phantom',  getProvider: () => window.phantom?.solana,  detect: () => !!window.phantom?.solana?.isPhantom },
        { name: 'Solflare', getProvider: () => window.solflare,          detect: () => !!window.solflare?.isSolflare },
        { name: 'Backpack', getProvider: () => window.backpack,           detect: () => !!window.backpack?.isBackpack },
        { name: 'Torus',    getProvider: () => window.torus?.solana,      detect: () => !!window.torus },
        { name: 'Coin98',   getProvider: () => window.coin98?.sol,        detect: () => !!window.coin98 },
        { name: 'Slope',    getProvider: () => window.Slope ? new window.Slope() : null, detect: () => !!window.Slope },
    ];

    let activeProvider = null;
    let walletState = null;

    function truncAddr(addr) {
        return addr.slice(0, 4) + '...' + addr.slice(-4);
    }

    function getWalletConfig(name) {
        return WALLETS.find(w => w.name === name);
    }

    function injectWalletNav() {
        const header = document.querySelector('header');
        if (!header) return;

        const nav = document.createElement('div');
        nav.className = 'nav-item wallet-connected';
        nav.id = 'walletNav';
        nav.onclick = handleClick;
        nav.innerHTML =
            '<span id="walletLabel">Connect</span>' +
            '<div class="wallet-dropdown" id="walletDropdown">' +
                '<div class="wallet-dropdown-item" id="wdWalletName"><span class="dd-icon">◉</span> <span id="wdName">—</span></div>' +
                '<div class="wallet-dropdown-item" id="wdCopy"><span class="dd-icon">⎘</span> Copy Address</div>' +
                '<div class="wallet-dropdown-item" id="wdSolscan"><span class="dd-icon">↗</span> View on Solscan</div>' +
                '<div class="wallet-dropdown-item disconnect" id="wdDisconnect"><span class="dd-icon">⏻</span> Disconnect</div>' +
            '</div>';
        header.appendChild(nav);

        document.getElementById('wdCopy').onclick = function (e) {
            e.stopPropagation();
            if (!walletState) return;
            navigator.clipboard.writeText(walletState.address);
            document.getElementById('walletDropdown').classList.remove('open');
        };
        document.getElementById('wdSolscan').onclick = function (e) {
            e.stopPropagation();
            if (!walletState) return;
            window.open('https://solscan.io/account/' + walletState.address, '_blank');
            document.getElementById('walletDropdown').classList.remove('open');
        };
        document.getElementById('wdDisconnect').onclick = async function (e) {
            e.stopPropagation();
            if (activeProvider && typeof activeProvider.disconnect === 'function') {
                try { await activeProvider.disconnect(); } catch (_) {}
            }
            activeProvider = null;
            walletState = null;
            localStorage.removeItem(WALLET_STORAGE_KEY);
            document.getElementById('walletDropdown').classList.remove('open');
            updateUI(null);
        };

        document.addEventListener('click', function (e) {
            const n = document.getElementById('walletNav');
            if (n && !n.contains(e.target)) {
                document.getElementById('walletDropdown').classList.remove('open');
            }
        });
    }

    function updateUI(state) {
        const label = document.getElementById('walletLabel');
        const nav = document.getElementById('walletNav');
        const nameEl = document.getElementById('wdName');
        if (!label || !nav) return;
        if (state) {
            label.innerHTML = '<span class="wallet-addr">' + truncAddr(state.address) + '</span>';
            nav.classList.add('active');
            if (nameEl) nameEl.textContent = state.wallet;
        } else {
            label.textContent = 'Connect';
            nav.classList.remove('active');
            if (nameEl) nameEl.textContent = '—';
        }
    }

    function handleClick() {
        if (walletState) {
            document.getElementById('walletDropdown').classList.toggle('open');
        } else {
            window.location.href = 'index.html?connect=1';
        }
    }

    async function restore() {
        const stored = localStorage.getItem(WALLET_STORAGE_KEY);
        if (!stored) return;
        try {
            const parsed = JSON.parse(stored);
            walletState = parsed;
            updateUI(walletState);

            const cfg = getWalletConfig(parsed.wallet);
            if (cfg && cfg.detect()) {
                const provider = cfg.getProvider();
                if (provider) {
                    try {
                        const resp = await provider.connect({ onlyIfTrusted: true });
                        const pk = resp.publicKey || provider.publicKey;
                        if (pk) {
                            activeProvider = provider;
                            walletState.address = pk.toString();
                            localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));
                            updateUI(walletState);
                        }
                    } catch (_) { /* eager reconnect failed, keep stored state */ }
                }
            }
        } catch (_) {
            localStorage.removeItem(WALLET_STORAGE_KEY);
        }
    }

    // Init on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { injectWalletNav(); restore(); });
    } else {
        injectWalletNav();
        restore();
    }
})();

