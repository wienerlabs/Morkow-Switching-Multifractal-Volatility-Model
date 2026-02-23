/**
 * vesting-sdk.js — Browser-compatible SDK for Cortex Vesting interactions
 *
 * Vesting Program ID: 5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns
 * Framework: Anchor
 *
 * PDA Seeds:
 *   vesting_schedule: ["vesting", beneficiary_pubkey]
 */

(function (global) {
    'use strict';

    var VESTING_PROGRAM_ID = '5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns';
    var CRTX_MINT = 'HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg';
    var TOKEN_PROGRAM_ID = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA';
    var ASSOCIATED_TOKEN_PROGRAM_ID = 'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS';

    var CRTX_DECIMALS = 9;
    var CRTX_MULTIPLIER = Math.pow(10, CRTX_DECIMALS);

    var CATEGORY_NAMES = ['PrivateSale', 'PublicSale', 'Team', 'Treasury', 'Marketing'];

    var DISCRIMINATORS = {
        claim_tokens: new Uint8Array(8),
    };

    var _discriminatorsReady = false;

    async function ensureDiscriminators() {
        if (_discriminatorsReady) return;
        try {
            var encoder = new TextEncoder();
            var data = encoder.encode('global:claim_tokens');
            var hashBuffer = await crypto.subtle.digest('SHA-256', data);
            DISCRIMINATORS.claim_tokens = new Uint8Array(hashBuffer).slice(0, 8);
            _discriminatorsReady = true;
        } catch (e) {
            console.warn('[VestingSDK] SubtleCrypto unavailable');
            _discriminatorsReady = true;
        }
    }

    // -------------------------------------------------------------------------
    // PDA derivation
    // -------------------------------------------------------------------------

    function getProgramPubkey() {
        return new solanaWeb3.PublicKey(VESTING_PROGRAM_ID);
    }

    async function getVestingSchedulePDA(beneficiaryPubkey) {
        return solanaWeb3.PublicKey.findProgramAddress(
            [
                Buffer.from('vesting'),
                new solanaWeb3.PublicKey(beneficiaryPubkey).toBuffer(),
            ],
            getProgramPubkey()
        );
    }

    async function getAssociatedTokenAddress(walletPubkey, mintPubkey) {
        var [ata] = await solanaWeb3.PublicKey.findProgramAddress(
            [
                new solanaWeb3.PublicKey(walletPubkey).toBuffer(),
                new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID).toBuffer(),
                new solanaWeb3.PublicKey(mintPubkey).toBuffer(),
            ],
            new solanaWeb3.PublicKey(ASSOCIATED_TOKEN_PROGRAM_ID)
        );
        return ata;
    }

    // -------------------------------------------------------------------------
    // Account deserialization
    // -------------------------------------------------------------------------

    /**
     * VestingSchedule layout (Anchor serialization):
     *   8  bytes - discriminator
     *   32 bytes - beneficiary (Pubkey)
     *   1  byte  - category (enum u8)
     *   8  bytes - total_amount (u64 LE)
     *   8  bytes - claimed_amount (u64 LE)
     *   8  bytes - start_time (i64 LE)
     *   8  bytes - cliff_duration (i64 LE)
     *   8  bytes - vesting_duration (i64 LE)
     *   1  byte  - tge_unlock_percent (u8)
     *   1  byte  - tge_claimed (bool)
     *   1  byte  - bump (u8)
     */
    function deserializeVestingSchedule(data) {
        var buf = data instanceof Buffer ? data : Buffer.from(data);
        var offset = 8;

        function readPubkey() {
            var pk = new solanaWeb3.PublicKey(buf.slice(offset, offset + 32));
            offset += 32;
            return pk;
        }
        function readU64() {
            var lo = buf.readUInt32LE(offset);
            var hi = buf.readUInt32LE(offset + 4);
            offset += 8;
            if (typeof BigInt !== 'undefined') return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo));
            return hi * 4294967296 + lo;
        }
        function readI64() {
            var lo = buf.readUInt32LE(offset);
            var hi = buf.readInt32LE(offset + 4);
            offset += 8;
            if (typeof BigInt !== 'undefined') return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo >>> 0));
            return hi * 4294967296 + lo;
        }
        function readU8() {
            var v = buf.readUInt8(offset);
            offset += 1;
            return v;
        }
        function readBool() {
            var v = buf.readUInt8(offset);
            offset += 1;
            return v !== 0;
        }

        var beneficiary = readPubkey();
        var categoryRaw = readU8();
        var totalAmount = readU64();
        var claimedAmount = readU64();
        var startTime = readI64();
        var cliffDuration = readI64();
        var vestingDuration = readI64();
        var tgeUnlockPercent = readU8();
        var tgeClaimed = readBool();
        var bump = readU8();

        return {
            beneficiary: beneficiary,
            category: categoryRaw,
            categoryName: CATEGORY_NAMES[categoryRaw] || 'Unknown',
            totalAmount: totalAmount,
            claimedAmount: claimedAmount,
            startTime: startTime,
            cliffDuration: cliffDuration,
            vestingDuration: vestingDuration,
            tgeUnlockPercent: tgeUnlockPercent,
            tgeClaimed: tgeClaimed,
            bump: bump,
        };
    }

    // -------------------------------------------------------------------------
    // Client-side claimable calculation (mirrors on-chain logic)
    // -------------------------------------------------------------------------

    function calculateClaimable(schedule, currentTime) {
        if (!schedule) return 0;
        currentTime = currentTime || Math.floor(Date.now() / 1000);

        if (currentTime < schedule.startTime) return 0;

        var totalUnlocked = 0;

        // TGE unlock
        if (schedule.tgeUnlockPercent > 0 && !schedule.tgeClaimed) {
            totalUnlocked = Math.floor(schedule.totalAmount * schedule.tgeUnlockPercent / 100);
        }

        // Check cliff
        var cliffEnd = schedule.startTime + schedule.cliffDuration;
        if (currentTime < cliffEnd) {
            return Math.max(0, totalUnlocked - schedule.claimedAmount);
        }

        // Linear vesting
        var vestingEnd = schedule.startTime + schedule.cliffDuration + schedule.vestingDuration;
        var tgeAmount = Math.floor(schedule.totalAmount * schedule.tgeUnlockPercent / 100);
        var vestingAmount = schedule.totalAmount - tgeAmount;

        if (currentTime >= vestingEnd) {
            totalUnlocked = schedule.totalAmount;
        } else {
            var timeSinceCliff = currentTime - cliffEnd;
            var vested = Math.floor(vestingAmount * timeSinceCliff / schedule.vestingDuration);
            totalUnlocked += vested;
        }

        return Math.max(0, totalUnlocked - schedule.claimedAmount);
    }

    function getVestingProgress(schedule) {
        if (!schedule) return { percent: 0, status: 'none' };
        var now = Math.floor(Date.now() / 1000);

        if (now < schedule.startTime) return { percent: 0, status: 'not_started' };

        var cliffEnd = schedule.startTime + schedule.cliffDuration;
        var vestingEnd = schedule.startTime + schedule.cliffDuration + schedule.vestingDuration;

        if (now < cliffEnd) return { percent: 0, status: 'cliff' };
        if (now >= vestingEnd) return { percent: 100, status: 'fully_vested' };

        var elapsed = now - cliffEnd;
        var total = schedule.vestingDuration;
        var percent = Math.min(100, Math.floor(elapsed / total * 100));
        return { percent: percent, status: 'vesting' };
    }

    // -------------------------------------------------------------------------
    // Data fetching
    // -------------------------------------------------------------------------

    async function getVestingInfo(connection, beneficiaryPubkey) {
        var [schedulePubkey] = await getVestingSchedulePDA(beneficiaryPubkey);
        var accountInfo = await connection.getAccountInfo(schedulePubkey);
        if (!accountInfo) return null;
        return {
            pubkey: schedulePubkey,
            schedule: deserializeVestingSchedule(accountInfo.data),
        };
    }

    // -------------------------------------------------------------------------
    // TX builder
    // -------------------------------------------------------------------------

    /**
     * Build claim_tokens TX
     * Accounts: vesting_schedule, vesting_vault, beneficiary_token_account, beneficiary, token_program
     */
    async function buildClaimTokensTx(connection, beneficiaryPubkey) {
        await ensureDiscriminators();

        var beneficiary = new solanaWeb3.PublicKey(beneficiaryPubkey);
        var program = getProgramPubkey();
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);

        var [schedulePubkey] = await getVestingSchedulePDA(beneficiaryPubkey);

        // Vesting vault: the schedule PDA's ATA for CRTX
        var vestingVault = await getAssociatedTokenAddress(schedulePubkey, CRTX_MINT);
        var beneficiaryAta = await getAssociatedTokenAddress(beneficiaryPubkey, CRTX_MINT);

        var ix = new solanaWeb3.TransactionInstruction({
            programId: program,
            keys: [
                { pubkey: schedulePubkey, isSigner: false, isWritable: true },
                { pubkey: vestingVault, isSigner: false, isWritable: true },
                { pubkey: beneficiaryAta, isSigner: false, isWritable: true },
                { pubkey: beneficiary, isSigner: true, isWritable: false },
                { pubkey: tokenProgram, isSigner: false, isWritable: false },
            ],
            data: Buffer.from(DISCRIMINATORS.claim_tokens),
        });

        var tx = new solanaWeb3.Transaction();
        tx.add(ix);
        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = beneficiary;
        return tx;
    }

    // -------------------------------------------------------------------------
    // Exposed namespace
    // -------------------------------------------------------------------------

    global.VestingSDK = {
        VESTING_PROGRAM_ID: VESTING_PROGRAM_ID,
        CRTX_MINT: CRTX_MINT,
        CRTX_DECIMALS: CRTX_DECIMALS,
        CRTX_MULTIPLIER: CRTX_MULTIPLIER,
        CATEGORY_NAMES: CATEGORY_NAMES,

        getVestingSchedulePDA: getVestingSchedulePDA,
        getAssociatedTokenAddress: getAssociatedTokenAddress,

        getVestingInfo: getVestingInfo,
        deserializeVestingSchedule: deserializeVestingSchedule,

        calculateClaimable: calculateClaimable,
        getVestingProgress: getVestingProgress,

        buildClaimTokensTx: buildClaimTokensTx,
    };

})(window);
