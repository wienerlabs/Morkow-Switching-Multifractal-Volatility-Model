// === STAKING UI CONTROLLER ===
(function () {
    'use strict';

    var STAKING_RPC = 'https://api.mainnet-beta.solana.com';
    var _connection = null;
    var _poolInfo = null;
    var _userStake = null;
    var _userCrtxBalance = 0;
    var _currentTab = 'stake';

    function getConnection() {
        if (!_connection) _connection = new solanaWeb3.Connection(STAKING_RPC, 'confirmed');
        return _connection;
    }

    function fmtCrtx(raw) {
        var val = raw / StakingSDK.CRTX_MULTIPLIER;
        return val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function fmtCrtxFull(raw) {
        var val = raw / StakingSDK.CRTX_MULTIPLIER;
        return val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 9 });
    }

    function setSmStatus(msg, isError) {
        var el = document.getElementById('smStatusMsg');
        if (!el) return;
        el.style.display = msg ? 'block' : 'none';
        el.textContent = msg || '';
        el.style.borderColor = isError ? 'var(--red)' : 'var(--green)';
        el.style.color = isError ? 'var(--red)' : 'var(--green)';
    }

    function setSmBtnState(btnId, loading, label) {
        var btn = document.getElementById(btnId);
        if (!btn) return;
        btn.disabled = loading;
        btn.textContent = loading ? 'Processing\u2026' : label;
    }

    // ---------------------------------------------------------------
    // Data loading
    // ---------------------------------------------------------------

    async function loadStakingData(walletAddress) {
        if (!window.StakingSDK || typeof solanaWeb3 === 'undefined') return;
        var conn = getConnection();

        try {
            var poolRes = await StakingSDK.getPoolInfo(conn);
            _poolInfo = poolRes;

            if (!poolRes) {
                updateStakingCard(null, null);
                return;
            }

            _userStake = null;
            _userCrtxBalance = 0;

            if (walletAddress) {
                try {
                    var stakeRes = await StakingSDK.getUserStakeInfo(conn, walletAddress);
                    _userStake = stakeRes ? stakeRes.stakeInfo : null;
                } catch (e) {
                    console.warn('[StakingUI] Failed to fetch stake info:', e.message);
                }

                try {
                    var balRes = await StakingSDK.getUserCrtxBalance(conn, walletAddress);
                    _userCrtxBalance = balRes.balance;
                } catch (e) {
                    console.warn('[StakingUI] Failed to fetch CRTX balance:', e.message);
                }
            }

            updateStakingCard(poolRes.pool, _userStake);
            updateModalInfo(poolRes.pool, _userStake, walletAddress);

        } catch (e) {
            console.warn('[StakingUI] loadStakingData error:', e.message);
        }
    }

    // ---------------------------------------------------------------
    // Dashboard card update
    // ---------------------------------------------------------------

    function updateStakingCard(pool, stakeInfo) {
        var card = document.getElementById('stakingCard');
        if (!card) return;

        var stakedEl = document.getElementById('stakingUserStaked');
        var tierEl = document.getElementById('stakingUserTier');
        var rewardsEl = document.getElementById('stakingUserRewards');
        var statusEl = document.getElementById('stakingStatus');
        var poolTotalEl = document.getElementById('stakingPoolTotal');

        if (!pool) {
            if (stakedEl) stakedEl.textContent = '--';
            if (tierEl) tierEl.textContent = '--';
            if (rewardsEl) rewardsEl.textContent = '--';
            if (statusEl) statusEl.textContent = 'Not initialized';
            if (poolTotalEl) poolTotalEl.textContent = '--';
            return;
        }

        if (poolTotalEl) poolTotalEl.textContent = fmtCrtx(pool.totalStaked) + ' CRTX';

        if (pool.paused) {
            if (statusEl) { statusEl.textContent = 'PAUSED'; statusEl.style.color = 'var(--red)'; }
        } else {
            if (statusEl) { statusEl.textContent = 'Active'; statusEl.style.color = 'var(--green)'; }
        }

        if (stakeInfo && stakeInfo.amount > 0) {
            if (stakedEl) stakedEl.textContent = fmtCrtx(stakeInfo.amount) + ' CRTX';
            var tier = StakingSDK.getUserTier(stakeInfo.amount, pool.tierThresholds);
            if (tierEl) tierEl.textContent = StakingSDK.TIER_NAMES[tier];
            var rewards = StakingSDK.calculatePendingRewards(stakeInfo, pool);
            if (rewardsEl) rewardsEl.textContent = fmtCrtx(rewards) + ' CRTX';
        } else {
            if (stakedEl) stakedEl.textContent = '0.00 CRTX';
            if (tierEl) tierEl.textContent = StakingSDK.TIER_NAMES[0];
            if (rewardsEl) rewardsEl.textContent = '0.00 CRTX';
        }
    }

    // ---------------------------------------------------------------
    // Modal info update
    // ---------------------------------------------------------------

    function updateModalInfo(pool, stakeInfo, walletAddress) {
        var smPoolTotal = document.getElementById('smPoolTotal');
        var smUserStaked = document.getElementById('smUserStaked');
        var smUserRewards = document.getElementById('smUserRewards');

        if (smPoolTotal) smPoolTotal.textContent = pool ? fmtCrtx(pool.totalStaked) + ' CRTX' : '--';

        var smRewardAmount = document.getElementById('smRewardAmount');

        if (walletAddress && stakeInfo && stakeInfo.amount > 0) {
            if (smUserStaked) smUserStaked.textContent = fmtCrtx(stakeInfo.amount) + ' CRTX';
            var rewards = StakingSDK.calculatePendingRewards(stakeInfo, pool);
            if (smUserRewards) smUserRewards.textContent = fmtCrtx(rewards) + ' CRTX';
            if (smRewardAmount) smRewardAmount.textContent = fmtCrtx(rewards) + ' CRTX';
        } else {
            if (smUserStaked) smUserStaked.textContent = walletAddress ? '0.00 CRTX' : '--';
            if (smUserRewards) smUserRewards.textContent = walletAddress ? '0.00 CRTX' : '--';
            if (smRewardAmount) smRewardAmount.textContent = '0.00 CRTX';
        }

        // Update balance display on stake tab
        var stakeBal = document.getElementById('smStakeBalance');
        if (stakeBal) stakeBal.textContent = fmtCrtx(_userCrtxBalance);

        // Update unstake tab info
        var unstakeBal = document.getElementById('smUnstakeBalance');
        if (unstakeBal && stakeInfo) unstakeBal.textContent = fmtCrtx(stakeInfo.amount);
        else if (unstakeBal) unstakeBal.textContent = '0.00';

        // Cooldown / lock status display
        updateUnstakeStatus(stakeInfo);

        // Pool paused warning
        var warnEl = document.getElementById('smStateWarning');
        if (warnEl) {
            if (pool && pool.paused) {
                warnEl.style.display = 'block';
                warnEl.textContent = 'Staking pool is PAUSED \u2014 staking is disabled.';
            } else {
                warnEl.style.display = 'none';
            }
        }
    }

    function updateUnstakeStatus(stakeInfo) {
        var statusEl = document.getElementById('smUnstakeStatusMsg');
        var cooldownBtn = document.getElementById('smCooldownBtn');
        var unstakeBtn = document.getElementById('smUnstakeBtn');
        var claimBtn = document.getElementById('smClaimBtn');

        if (!statusEl) return;

        if (!stakeInfo || stakeInfo.amount === 0) {
            statusEl.textContent = 'No active stake.';
            statusEl.style.display = 'block';
            if (cooldownBtn) cooldownBtn.style.display = 'none';
            if (unstakeBtn) unstakeBtn.style.display = 'none';
            return;
        }

        var status = StakingSDK.getStakeStatus(stakeInfo);
        var now = Math.floor(Date.now() / 1000);

        switch (status) {
            case 'locked':
                var lockRemaining = stakeInfo.lockEnd - now;
                var days = Math.ceil(lockRemaining / 86400);
                statusEl.textContent = 'Stake locked \u2014 ' + days + ' day' + (days !== 1 ? 's' : '') + ' remaining.';
                statusEl.style.display = 'block';
                if (cooldownBtn) cooldownBtn.style.display = 'none';
                if (unstakeBtn) unstakeBtn.style.display = 'none';
                break;
            case 'unlocked':
                statusEl.textContent = 'Lock period ended. Start cooldown to unstake.';
                statusEl.style.display = 'block';
                if (cooldownBtn) cooldownBtn.style.display = '';
                if (unstakeBtn) unstakeBtn.style.display = 'none';
                break;
            case 'cooldown':
                var cdEnd = stakeInfo.cooldownStart + StakingSDK.COOLDOWN_SECONDS;
                var cdRemaining = cdEnd - now;
                var cdHours = Math.ceil(cdRemaining / 3600);
                statusEl.textContent = 'Cooldown active \u2014 ' + cdHours + ' hour' + (cdHours !== 1 ? 's' : '') + ' remaining.';
                statusEl.style.display = 'block';
                if (cooldownBtn) cooldownBtn.style.display = 'none';
                if (unstakeBtn) unstakeBtn.style.display = 'none';
                break;
            case 'ready_to_unstake':
                statusEl.textContent = 'Cooldown complete! You can unstake now.';
                statusEl.style.display = 'block';
                statusEl.style.color = 'var(--green)';
                if (cooldownBtn) cooldownBtn.style.display = 'none';
                if (unstakeBtn) unstakeBtn.style.display = '';
                break;
            default:
                statusEl.style.display = 'none';
                if (cooldownBtn) cooldownBtn.style.display = 'none';
                if (unstakeBtn) unstakeBtn.style.display = 'none';
        }

        // Claim button visibility
        if (claimBtn) {
            var rewards = _poolInfo && stakeInfo ? StakingSDK.calculatePendingRewards(stakeInfo, _poolInfo.pool) : 0;
            claimBtn.style.display = rewards > 0 ? '' : 'none';
        }
    }

    // ---------------------------------------------------------------
    // Modal open
    // ---------------------------------------------------------------

    window.openStakingModal = async function () {
        if (typeof walletState === 'undefined' || !walletState || !walletState.address) {
            if (typeof renderWalletGrid === 'function') renderWalletGrid();
            document.getElementById('walletModal').classList.add('active');
            return;
        }

        setSmStatus('', false);
        stakingSwitchTab('stake');
        document.getElementById('stakingModal').classList.add('active');
        await loadStakingData(walletState.address);
    };

    // ---------------------------------------------------------------
    // Tab switching
    // ---------------------------------------------------------------

    window.stakingSwitchTab = function (tab) {
        _currentTab = tab;
        var panels = ['smStakePanel', 'smUnstakePanel', 'smRewardsPanel'];
        var tabs = ['smTabStake', 'smTabUnstake', 'smTabRewards'];
        var tabNames = ['stake', 'unstake', 'rewards'];

        for (var i = 0; i < panels.length; i++) {
            var panel = document.getElementById(panels[i]);
            var tabEl = document.getElementById(tabs[i]);
            if (panel) panel.style.display = tabNames[i] === tab ? 'block' : 'none';
            if (tabEl) {
                tabEl.style.borderBottomColor = tabNames[i] === tab ? 'var(--fg)' : 'transparent';
                tabEl.style.color = tabNames[i] === tab ? 'var(--fg)' : 'var(--dim)';
            }
        }
        setSmStatus('', false);
    };

    // ---------------------------------------------------------------
    // MAX buttons
    // ---------------------------------------------------------------

    window.stakingSetMaxStake = function () {
        var input = document.getElementById('smStakeAmount');
        if (!input) return;
        input.value = (_userCrtxBalance / StakingSDK.CRTX_MULTIPLIER).toFixed(2);
    };

    window.stakingSetMaxUnstake = function () {
        var input = document.getElementById('smUnstakeAmount');
        if (!input || !_userStake) return;
        input.value = (_userStake.amount / StakingSDK.CRTX_MULTIPLIER).toFixed(2);
    };

    // ---------------------------------------------------------------
    // Execute stake
    // ---------------------------------------------------------------

    window.executeStake = async function () {
        setSmStatus('', false);

        if (!walletState || !walletState.address) { setSmStatus('Wallet not connected.', true); return; }
        if (!activeProvider) { setSmStatus('Wallet provider not available.', true); return; }

        var input = document.getElementById('smStakeAmount');
        var amt = parseFloat(input ? input.value : '0');
        if (!amt || amt <= 0) { setSmStatus('Enter a valid stake amount.', true); return; }

        var lockSelect = document.getElementById('smLockType');
        var lockType = lockSelect ? parseInt(lockSelect.value) : 0;
        var amtRaw = Math.floor(amt * StakingSDK.CRTX_MULTIPLIER);

        if (amtRaw < 1000000000) { setSmStatus('Minimum stake is 1 CRTX.', true); return; }
        if (amtRaw > _userCrtxBalance) { setSmStatus('Insufficient CRTX balance.', true); return; }

        setSmBtnState('smStakeBtn', true, 'Stake');
        setSmStatus('Building transaction\u2026', false);

        try {
            var conn = getConnection();
            var tx = await StakingSDK.buildStakeTx(conn, walletState.address, amtRaw, lockType);

            setSmStatus('Requesting wallet signature\u2026', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setSmStatus('Sending transaction\u2026', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false, preflightCommitment: 'confirmed',
            });

            setSmStatus('Confirming\u2026', false);
            await conn.confirmTransaction(sig, 'confirmed');

            var lockLabel = StakingSDK.LOCK_TYPES[lockType].label;
            setSmStatus('Staked ' + amt.toFixed(2) + ' CRTX (' + lockLabel + ')! TX: ' + sig.slice(0, 8) + '\u2026', false);
            if (typeof showToast === 'function') showToast('Staked ' + amt.toFixed(2) + ' CRTX', 'success');
            if (input) input.value = '';

            await loadStakingData(walletState.address);
        } catch (err) {
            var msg = err.message || 'Stake failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) msg = 'Transaction rejected by user.';
            setSmStatus('Error: ' + msg, true);
            if (typeof showToast === 'function') showToast('Stake failed: ' + msg, 'critical');
        } finally {
            setSmBtnState('smStakeBtn', false, 'Stake CRTX');
        }
    };

    // ---------------------------------------------------------------
    // Execute initiate cooldown
    // ---------------------------------------------------------------

    window.executeInitiateCooldown = async function () {
        setSmStatus('', false);

        if (!walletState || !walletState.address) { setSmStatus('Wallet not connected.', true); return; }
        if (!activeProvider) { setSmStatus('Wallet provider not available.', true); return; }

        setSmBtnState('smCooldownBtn', true, 'Start Cooldown');
        setSmStatus('Building transaction\u2026', false);

        try {
            var conn = getConnection();
            var tx = await StakingSDK.buildInitiateCooldownTx(conn, walletState.address);

            setSmStatus('Requesting wallet signature\u2026', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setSmStatus('Sending transaction\u2026', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false, preflightCommitment: 'confirmed',
            });

            setSmStatus('Confirming\u2026', false);
            await conn.confirmTransaction(sig, 'confirmed');

            setSmStatus('Cooldown started! 3-day period begins now. TX: ' + sig.slice(0, 8) + '\u2026', false);
            if (typeof showToast === 'function') showToast('Cooldown started \u2014 3 days until unstake', 'success');

            await loadStakingData(walletState.address);
        } catch (err) {
            var msg = err.message || 'Cooldown initiation failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) msg = 'Transaction rejected by user.';
            setSmStatus('Error: ' + msg, true);
        } finally {
            setSmBtnState('smCooldownBtn', false, 'Start Cooldown');
        }
    };

    // ---------------------------------------------------------------
    // Execute unstake
    // ---------------------------------------------------------------

    window.executeUnstake = async function () {
        setSmStatus('', false);

        if (!walletState || !walletState.address) { setSmStatus('Wallet not connected.', true); return; }
        if (!activeProvider) { setSmStatus('Wallet provider not available.', true); return; }

        var input = document.getElementById('smUnstakeAmount');
        var amt = parseFloat(input ? input.value : '0');
        if (!amt || amt <= 0) { setSmStatus('Enter a valid unstake amount.', true); return; }

        var amtRaw = Math.floor(amt * StakingSDK.CRTX_MULTIPLIER);
        if (_userStake && amtRaw > _userStake.amount) {
            setSmStatus('Amount exceeds staked balance.', true);
            return;
        }

        setSmBtnState('smUnstakeBtn', true, 'Unstake');
        setSmStatus('Building transaction\u2026', false);

        try {
            var conn = getConnection();
            var tx = await StakingSDK.buildUnstakeTx(conn, walletState.address, amtRaw);

            setSmStatus('Requesting wallet signature\u2026', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setSmStatus('Sending transaction\u2026', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false, preflightCommitment: 'confirmed',
            });

            setSmStatus('Confirming\u2026', false);
            await conn.confirmTransaction(sig, 'confirmed');

            setSmStatus('Unstaked ' + amt.toFixed(2) + ' CRTX! TX: ' + sig.slice(0, 8) + '\u2026', false);
            if (typeof showToast === 'function') showToast('Unstaked ' + amt.toFixed(2) + ' CRTX', 'success');
            if (input) input.value = '';

            await loadStakingData(walletState.address);
        } catch (err) {
            var msg = err.message || 'Unstake failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) msg = 'Transaction rejected by user.';
            setSmStatus('Error: ' + msg, true);
            if (typeof showToast === 'function') showToast('Unstake failed: ' + msg, 'critical');
        } finally {
            setSmBtnState('smUnstakeBtn', false, 'Unstake CRTX');
        }
    };

    // ---------------------------------------------------------------
    // Execute claim rewards
    // ---------------------------------------------------------------

    window.executeClaimRewards = async function () {
        setSmStatus('', false);

        if (!walletState || !walletState.address) { setSmStatus('Wallet not connected.', true); return; }
        if (!activeProvider) { setSmStatus('Wallet provider not available.', true); return; }

        setSmBtnState('smClaimBtn', true, 'Claim Rewards');
        setSmStatus('Building transaction\u2026', false);

        try {
            var conn = getConnection();
            var tx = await StakingSDK.buildClaimRewardsTx(conn, walletState.address);

            setSmStatus('Requesting wallet signature\u2026', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setSmStatus('Sending transaction\u2026', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false, preflightCommitment: 'confirmed',
            });

            setSmStatus('Confirming\u2026', false);
            await conn.confirmTransaction(sig, 'confirmed');

            setSmStatus('Rewards claimed! TX: ' + sig.slice(0, 8) + '\u2026', false);
            if (typeof showToast === 'function') showToast('Rewards claimed successfully', 'success');

            await loadStakingData(walletState.address);
        } catch (err) {
            var msg = err.message || 'Claim failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) msg = 'Transaction rejected by user.';
            setSmStatus('Error: ' + msg, true);
            if (typeof showToast === 'function') showToast('Claim failed: ' + msg, 'critical');
        } finally {
            setSmBtnState('smClaimBtn', false, 'Claim Rewards');
        }
    };

    // ---------------------------------------------------------------
    // Hook into wallet connection (chain after vault-ui.js)
    // ---------------------------------------------------------------

    var _origSetWalletUI = window.setWalletUI;
    window.setWalletUI = function (state) {
        if (typeof _origSetWalletUI === 'function') _origSetWalletUI(state);

        var stakingBtn = document.getElementById('stakingActionBtn');
        if (state && state.address) {
            if (stakingBtn) stakingBtn.style.display = '';
            setTimeout(function () {
                if (typeof StakingSDK !== 'undefined' && typeof solanaWeb3 !== 'undefined') {
                    loadStakingData(state.address);
                }
            }, 800);
        } else {
            if (stakingBtn) stakingBtn.style.display = 'none';
            updateStakingCard(null, null);
        }
    };

    // ---------------------------------------------------------------
    // Initial load
    // ---------------------------------------------------------------

    window.addEventListener('load', function () {
        setTimeout(function () {
            if (typeof StakingSDK === 'undefined' || typeof solanaWeb3 === 'undefined') return;
            if (typeof walletState !== 'undefined' && walletState && walletState.address) {
                var stakingBtn = document.getElementById('stakingActionBtn');
                if (stakingBtn) stakingBtn.style.display = '';
                loadStakingData(walletState.address);
            }
        }, 2000);
    });

    window._stakingLoadData = loadStakingData;

})();
