// === VESTING CLAIM UI CONTROLLER ===
(function () {
    'use strict';

    var VESTING_RPC = 'https://api.mainnet-beta.solana.com';
    var _connection = null;
    var _vestingInfo = null;

    function getConnection() {
        if (!_connection) _connection = new solanaWeb3.Connection(VESTING_RPC, 'confirmed');
        return _connection;
    }

    function fmtCrtx(raw) {
        var val = raw / VestingSDK.CRTX_MULTIPLIER;
        return val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function setVestingStatus(msg, isError) {
        var el = document.getElementById('vestingStatusMsg');
        if (!el) return;
        el.style.display = msg ? 'block' : 'none';
        el.textContent = msg || '';
        el.style.borderColor = isError ? 'var(--red)' : 'var(--green)';
        el.style.color = isError ? 'var(--red)' : 'var(--green)';
    }

    // ---------------------------------------------------------------
    // Data loading
    // ---------------------------------------------------------------

    async function loadVestingData(walletAddress) {
        if (!window.VestingSDK || typeof solanaWeb3 === 'undefined') return;

        var section = document.getElementById('vestingClaimSection');
        var noVesting = document.getElementById('vestingNoSchedule');

        if (!walletAddress) {
            if (section) section.style.display = 'none';
            return;
        }

        if (section) section.style.display = '';

        var conn = getConnection();
        try {
            var info = await VestingSDK.getVestingInfo(conn, walletAddress);
            _vestingInfo = info;

            if (!info) {
                if (noVesting) noVesting.style.display = 'block';
                updateVestingDisplay(null);
                return;
            }

            if (noVesting) noVesting.style.display = 'none';
            updateVestingDisplay(info.schedule);
        } catch (e) {
            console.warn('[VestingUI] loadVestingData error:', e.message);
        }
    }

    function updateVestingDisplay(schedule) {
        var categoryEl = document.getElementById('vestingCategory');
        var totalEl = document.getElementById('vestingTotal');
        var claimedEl = document.getElementById('vestingClaimed');
        var claimableEl = document.getElementById('vestingClaimable');
        var progressEl = document.getElementById('vestingProgressBar');
        var progressPctEl = document.getElementById('vestingProgressPct');
        var statusEl = document.getElementById('vestingStatusText');
        var claimBtn = document.getElementById('vestingClaimBtn');
        var rewardAmountEl = document.getElementById('vestingRewardAmount');

        if (!schedule) {
            if (categoryEl) categoryEl.textContent = '--';
            if (totalEl) totalEl.textContent = '--';
            if (claimedEl) claimedEl.textContent = '--';
            if (claimableEl) claimableEl.textContent = '--';
            if (progressEl) progressEl.style.width = '0%';
            if (progressPctEl) progressPctEl.textContent = '0%';
            if (statusEl) statusEl.textContent = 'No vesting schedule found';
            if (claimBtn) claimBtn.style.display = 'none';
            return;
        }

        if (categoryEl) categoryEl.textContent = schedule.categoryName;
        if (totalEl) totalEl.textContent = fmtCrtx(schedule.totalAmount) + ' CRTX';
        if (claimedEl) claimedEl.textContent = fmtCrtx(schedule.claimedAmount) + ' CRTX';

        var claimable = VestingSDK.calculateClaimable(schedule);
        if (claimableEl) claimableEl.textContent = fmtCrtx(claimable) + ' CRTX';
        if (rewardAmountEl) rewardAmountEl.textContent = fmtCrtx(claimable) + ' CRTX';

        var progress = VestingSDK.getVestingProgress(schedule);
        if (progressEl) progressEl.style.width = progress.percent + '%';
        if (progressPctEl) progressPctEl.textContent = progress.percent + '%';

        var STATUS_LABELS = {
            not_started: 'Not started',
            cliff: 'Cliff period',
            vesting: 'Vesting in progress',
            fully_vested: 'Fully vested',
        };
        if (statusEl) statusEl.textContent = STATUS_LABELS[progress.status] || '--';

        if (claimBtn) claimBtn.style.display = claimable > 0 ? '' : 'none';
    }

    // ---------------------------------------------------------------
    // Execute claim
    // ---------------------------------------------------------------

    window.executeVestingClaim = async function () {
        setVestingStatus('', false);

        if (typeof walletState === 'undefined' || !walletState || !walletState.address) {
            setVestingStatus('Wallet not connected.', true);
            return;
        }
        if (typeof activeProvider === 'undefined' || !activeProvider) {
            setVestingStatus('Wallet provider not available.', true);
            return;
        }

        var claimBtn = document.getElementById('vestingClaimBtn');
        if (claimBtn) { claimBtn.disabled = true; claimBtn.textContent = 'Processing\u2026'; }

        setVestingStatus('Building transaction\u2026', false);

        try {
            var conn = getConnection();
            var tx = await VestingSDK.buildClaimTokensTx(conn, walletState.address);

            setVestingStatus('Requesting wallet signature\u2026', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setVestingStatus('Sending transaction\u2026', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false, preflightCommitment: 'confirmed',
            });

            setVestingStatus('Confirming\u2026', false);
            await conn.confirmTransaction(sig, 'confirmed');

            setVestingStatus('Tokens claimed! TX: ' + sig.slice(0, 8) + '\u2026', false);
            if (typeof showToast === 'function') showToast('Vesting tokens claimed', 'success');

            await loadVestingData(walletState.address);
        } catch (err) {
            var msg = err.message || 'Claim failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) msg = 'Transaction rejected by user.';
            setVestingStatus('Error: ' + msg, true);
            if (typeof showToast === 'function') showToast('Vesting claim failed: ' + msg, 'critical');
        } finally {
            if (claimBtn) { claimBtn.disabled = false; claimBtn.textContent = 'Claim Tokens'; }
        }
    };

    // ---------------------------------------------------------------
    // Hook: load on wallet connect (for tokenomics page)
    // ---------------------------------------------------------------

    // wallet.js sets walletState — we use a load event to pick it up
    window.addEventListener('load', function () {
        setTimeout(function () {
            if (typeof VestingSDK === 'undefined' || typeof solanaWeb3 === 'undefined') return;
            if (typeof walletState !== 'undefined' && walletState && walletState.address) {
                loadVestingData(walletState.address);
            }
        }, 2500);
    });

    // Also hook into setWalletUI if it exists on this page
    var _origSetWalletUI = window.setWalletUI;
    window.setWalletUI = function (state) {
        if (typeof _origSetWalletUI === 'function') _origSetWalletUI(state);
        if (state && state.address) {
            setTimeout(function () {
                if (typeof VestingSDK !== 'undefined') loadVestingData(state.address);
            }, 500);
        }
    };

    window._vestingLoadData = loadVestingData;

})();
