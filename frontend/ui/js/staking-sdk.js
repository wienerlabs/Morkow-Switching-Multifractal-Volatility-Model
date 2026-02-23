/**
 * staking-sdk.js — Browser-compatible SDK for Cortex Staking interactions
 *
 * Staking Program ID: rYantWFyB4PsL36r9XB7nUb8TQ1pAhn9A87S6TbpMsr
 * Framework: Anchor
 *
 * PDA Seeds (from lib.rs):
 *   staking_pool: ["staking_pool"]
 *   stake_info:   ["stake_info", user_pubkey]
 *   stake_vault:  ["stake_vault"]
 *   reward_vault: ["reward_vault"]
 *
 * Instructions use Anchor 8-byte discriminators: sha256("global:<ix_name>")[..8]
 */

(function (global) {
    'use strict';

    var STAKING_PROGRAM_ID = 'rYantWFyB4PsL36r9XB7nUb8TQ1pAhn9A87S6TbpMsr';
    var CRTX_MINT = 'HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg';
    var TOKEN_PROGRAM_ID = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA';
    var ASSOCIATED_TOKEN_PROGRAM_ID = 'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS';
    var SYSTEM_PROGRAM_ID = '11111111111111111111111111111111';

    var CRTX_DECIMALS = 9;
    var CRTX_MULTIPLIER = Math.pow(10, CRTX_DECIMALS);

    var COOLDOWN_SECONDS = 3 * 24 * 60 * 60; // 3 days

    var LOCK_TYPES = [
        { id: 0, label: 'Flexible',  duration: 0,                  multiplier: 0.5 },
        { id: 1, label: '14 Days',   duration: 14 * 24 * 60 * 60,  multiplier: 1.0 },
        { id: 2, label: '30 Days',   duration: 30 * 24 * 60 * 60,  multiplier: 1.5 },
        { id: 3, label: '90 Days',   duration: 90 * 24 * 60 * 60,  multiplier: 2.5 },
        { id: 4, label: '180 Days',  duration: 180 * 24 * 60 * 60, multiplier: 4.0 },
        { id: 5, label: '365 Days',  duration: 365 * 24 * 60 * 60, multiplier: 6.0 },
    ];

    // Pre-computed Anchor discriminators: sha256("global:<fn_name>")[..8]
    var DISCRIMINATORS = {
        stake:              new Uint8Array([0x06, 0x44, 0xe0, 0xa2, 0xb1, 0x89, 0x05, 0xde]),
        initiate_cooldown:  new Uint8Array([0xd4, 0x3c, 0x80, 0x2e, 0xb1, 0x01, 0x9f, 0xf2]),
        unstake:            new Uint8Array([0x90, 0x95, 0xee, 0xa1, 0x20, 0xe1, 0xeb, 0xc7]),
        claim_rewards:      new Uint8Array([0x04, 0x92, 0x8a, 0x97, 0x67, 0x88, 0x01, 0x5b]),
    };

    // We'll compute the real discriminators at init time using SubtleCrypto
    var _discriminatorsReady = false;

    async function computeDiscriminatorAsync(name) {
        var encoder = new TextEncoder();
        var data = encoder.encode(name);
        var hashBuffer = await crypto.subtle.digest('SHA-256', data);
        return new Uint8Array(hashBuffer).slice(0, 8);
    }

    async function ensureDiscriminators() {
        if (_discriminatorsReady) return;
        try {
            DISCRIMINATORS.stake = await computeDiscriminatorAsync('global:stake');
            DISCRIMINATORS.initiate_cooldown = await computeDiscriminatorAsync('global:initiate_cooldown');
            DISCRIMINATORS.unstake = await computeDiscriminatorAsync('global:unstake');
            DISCRIMINATORS.claim_rewards = await computeDiscriminatorAsync('global:claim_rewards');
            _discriminatorsReady = true;
        } catch (e) {
            // Fallback to pre-computed values (already set above)
            console.warn('[StakingSDK] SubtleCrypto unavailable, using hardcoded discriminators');
            _discriminatorsReady = true;
        }
    }

    // -------------------------------------------------------------------------
    // PDA derivation
    // -------------------------------------------------------------------------

    function getProgramPubkey() {
        return new solanaWeb3.PublicKey(STAKING_PROGRAM_ID);
    }

    async function getStakingPoolPDA() {
        return solanaWeb3.PublicKey.findProgramAddress(
            [Buffer.from('staking_pool')],
            getProgramPubkey()
        );
    }

    async function getStakeInfoPDA(userPubkey) {
        return solanaWeb3.PublicKey.findProgramAddress(
            [Buffer.from('stake_info'), new solanaWeb3.PublicKey(userPubkey).toBuffer()],
            getProgramPubkey()
        );
    }

    async function getStakeVaultPDA() {
        return solanaWeb3.PublicKey.findProgramAddress(
            [Buffer.from('stake_vault')],
            getProgramPubkey()
        );
    }

    async function getRewardVaultPDA() {
        return solanaWeb3.PublicKey.findProgramAddress(
            [Buffer.from('reward_vault')],
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
     * StakingPool layout (Anchor serialization):
     *   8  bytes - discriminator
     *   32 bytes - authority (Pubkey)
     *   32 bytes - stake_mint (Pubkey)
     *   32 bytes - stake_vault (Pubkey)
     *   8  bytes - total_staked (u64 LE)
     *   8  bytes - total_weight (u64 LE)
     *   24 bytes - tier_thresholds ([u64; 3] LE)
     *   8  bytes - reward_rate (u64 LE)
     *   8  bytes - last_update_time (i64 LE)
     *   16 bytes - acc_reward_per_weight (u128 LE)
     *   1  byte  - paused (bool)
     *   1  byte  - bump (u8)
     */
    function deserializeStakingPool(data) {
        var buf = data instanceof Buffer ? data : Buffer.from(data);
        var offset = 8; // skip discriminator

        function readPubkey() {
            var pk = new solanaWeb3.PublicKey(buf.slice(offset, offset + 32));
            offset += 32;
            return pk;
        }
        function readU64() {
            var lo = buf.readUInt32LE(offset);
            var hi = buf.readUInt32LE(offset + 4);
            offset += 8;
            if (typeof BigInt !== 'undefined') {
                return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo));
            }
            return hi * 4294967296 + lo;
        }
        function readI64() {
            var lo = buf.readUInt32LE(offset);
            var hi = buf.readInt32LE(offset + 4);
            offset += 8;
            if (typeof BigInt !== 'undefined') {
                return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo >>> 0));
            }
            return hi * 4294967296 + lo;
        }
        function readU128() {
            // Read as BigInt for precision
            if (typeof BigInt !== 'undefined') {
                var lo = buf.readBigUInt64LE(offset);
                var hi = buf.readBigUInt64LE(offset + 8);
                offset += 16;
                return hi * BigInt(0x10000000000000000) + lo;
            }
            // Fallback: read lower 64 bits only
            var val = readU64();
            offset += 8;
            return val;
        }
        function readBool() {
            var v = buf.readUInt8(offset);
            offset += 1;
            return v !== 0;
        }
        function readU8() {
            var v = buf.readUInt8(offset);
            offset += 1;
            return v;
        }

        var authority = readPubkey();
        var stakeMint = readPubkey();
        var stakeVault = readPubkey();
        var totalStaked = readU64();
        var totalWeight = readU64();
        var tier0 = readU64();
        var tier1 = readU64();
        var tier2 = readU64();
        var rewardRate = readU64();
        var lastUpdateTime = readI64();
        var accRewardPerWeight = readU128();
        var paused = readBool();
        var bump = readU8();

        return {
            authority: authority,
            stakeMint: stakeMint,
            stakeVault: stakeVault,
            totalStaked: totalStaked,
            totalWeight: totalWeight,
            tierThresholds: [tier0, tier1, tier2],
            rewardRate: rewardRate,
            lastUpdateTime: lastUpdateTime,
            accRewardPerWeight: accRewardPerWeight,
            paused: paused,
            bump: bump,
        };
    }

    /**
     * StakeInfo layout (Anchor serialization):
     *   8  bytes - discriminator
     *   32 bytes - owner (Pubkey)
     *   8  bytes - amount (u64 LE)
     *   8  bytes - lock_end (i64 LE)
     *   1  byte  - lock_type (u8)
     *   8  bytes - weight (u64 LE)
     *   8  bytes - cooldown_start (i64 LE)
     *   8  bytes - reward_debt (u64 LE)
     *   8  bytes - pending_rewards (u64 LE)
     *   1  byte  - bump (u8)
     */
    function deserializeStakeInfo(data) {
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
            if (typeof BigInt !== 'undefined') {
                return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo));
            }
            return hi * 4294967296 + lo;
        }
        function readI64() {
            var lo = buf.readUInt32LE(offset);
            var hi = buf.readInt32LE(offset + 4);
            offset += 8;
            if (typeof BigInt !== 'undefined') {
                return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo >>> 0));
            }
            return hi * 4294967296 + lo;
        }
        function readU8() {
            var v = buf.readUInt8(offset);
            offset += 1;
            return v;
        }

        var owner = readPubkey();
        var amount = readU64();
        var lockEnd = readI64();
        var lockType = readU8();
        var weight = readU64();
        var cooldownStart = readI64();
        var rewardDebt = readU64();
        var pendingRewards = readU64();
        var bump = readU8();

        return {
            owner: owner,
            amount: amount,
            lockEnd: lockEnd,
            lockType: lockType,
            weight: weight,
            cooldownStart: cooldownStart,
            rewardDebt: rewardDebt,
            pendingRewards: pendingRewards,
            bump: bump,
        };
    }

    // -------------------------------------------------------------------------
    // Data fetching
    // -------------------------------------------------------------------------

    async function getPoolInfo(connection) {
        var [poolPubkey] = await getStakingPoolPDA();
        var accountInfo = await connection.getAccountInfo(poolPubkey);
        if (!accountInfo) return null;
        return { pubkey: poolPubkey, pool: deserializeStakingPool(accountInfo.data) };
    }

    async function getUserStakeInfo(connection, userPubkey) {
        var [stakeInfoPubkey] = await getStakeInfoPDA(userPubkey);
        var accountInfo = await connection.getAccountInfo(stakeInfoPubkey);
        if (!accountInfo) return null;
        return { pubkey: stakeInfoPubkey, stakeInfo: deserializeStakeInfo(accountInfo.data) };
    }

    async function getUserCrtxBalance(connection, userPubkey) {
        var ata = await getAssociatedTokenAddress(userPubkey, CRTX_MINT);
        var accountInfo = await connection.getTokenAccountBalance(ata).catch(function () { return null; });
        if (!accountInfo) return { balance: 0, ata: ata };
        var balance = accountInfo.value ? Number(accountInfo.value.amount) : 0;
        return { balance: balance, ata: ata };
    }

    // -------------------------------------------------------------------------
    // Client-side reward calculation (mirrors on-chain logic)
    // -------------------------------------------------------------------------

    function calculatePendingRewards(stakeInfo, pool) {
        if (!stakeInfo || stakeInfo.weight === 0) return 0;
        var PRECISION = typeof BigInt !== 'undefined' ? BigInt(1000000000000) : 1e12;

        if (typeof BigInt !== 'undefined') {
            var acc = BigInt(stakeInfo.weight) * BigInt(pool.accRewardPerWeight) / PRECISION;
            var pending = Number(acc - BigInt(stakeInfo.rewardDebt));
            return Math.max(0, pending + stakeInfo.pendingRewards);
        }
        // Fallback for no BigInt
        return stakeInfo.pendingRewards;
    }

    function getUserTier(stakeAmount, tierThresholds) {
        if (stakeAmount >= tierThresholds[2]) return 3;
        if (stakeAmount >= tierThresholds[1]) return 2;
        if (stakeAmount >= tierThresholds[0]) return 1;
        return 0;
    }

    var TIER_NAMES = ['Bronze', 'Silver', 'Gold', 'Diamond'];

    function getStakeStatus(stakeInfo) {
        if (!stakeInfo || stakeInfo.amount === 0) return 'none';
        var now = Math.floor(Date.now() / 1000);
        if (stakeInfo.cooldownStart > 0) {
            if (now >= stakeInfo.cooldownStart + COOLDOWN_SECONDS) return 'ready_to_unstake';
            return 'cooldown';
        }
        if (stakeInfo.lockEnd > 0 && now < stakeInfo.lockEnd) return 'locked';
        return 'unlocked';
    }

    // -------------------------------------------------------------------------
    // Instruction builders
    // -------------------------------------------------------------------------

    function encodeU64(value) {
        var buf = Buffer.alloc(8);
        if (typeof BigInt !== 'undefined') {
            buf.writeBigUInt64LE(BigInt(Math.floor(value)));
        } else {
            var lo = value >>> 0;
            var hi = Math.floor(value / 4294967296) >>> 0;
            buf.writeUInt32LE(lo, 0);
            buf.writeUInt32LE(hi, 4);
        }
        return buf;
    }

    function encodeU8(value) {
        var buf = Buffer.alloc(1);
        buf.writeUInt8(value, 0);
        return buf;
    }

    /**
     * Build stake TX
     * Accounts: staking_pool, stake_info, stake_vault, user_token_account, user, token_program, system_program
     */
    async function buildStakeTx(connection, walletPubkey, amount, lockType) {
        await ensureDiscriminators();

        var user = new solanaWeb3.PublicKey(walletPubkey);
        var program = getProgramPubkey();
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);
        var systemProgram = new solanaWeb3.PublicKey(SYSTEM_PROGRAM_ID);

        var [poolPubkey] = await getStakingPoolPDA();
        var [stakeInfoPubkey] = await getStakeInfoPDA(walletPubkey);
        var [stakeVaultPubkey] = await getStakeVaultPDA();
        var userTokenAta = await getAssociatedTokenAddress(walletPubkey, CRTX_MINT);

        var data = Buffer.concat([
            DISCRIMINATORS.stake,
            encodeU64(amount),
            encodeU8(lockType),
        ]);

        var ix = new solanaWeb3.TransactionInstruction({
            programId: program,
            keys: [
                { pubkey: poolPubkey, isSigner: false, isWritable: true },
                { pubkey: stakeInfoPubkey, isSigner: false, isWritable: true },
                { pubkey: stakeVaultPubkey, isSigner: false, isWritable: true },
                { pubkey: userTokenAta, isSigner: false, isWritable: true },
                { pubkey: user, isSigner: true, isWritable: true },
                { pubkey: tokenProgram, isSigner: false, isWritable: false },
                { pubkey: systemProgram, isSigner: false, isWritable: false },
            ],
            data: data,
        });

        var tx = new solanaWeb3.Transaction();
        tx.add(ix);
        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = user;
        return tx;
    }

    /**
     * Build initiate_cooldown TX
     * Accounts: staking_pool, stake_info, user
     */
    async function buildInitiateCooldownTx(connection, walletPubkey) {
        await ensureDiscriminators();

        var user = new solanaWeb3.PublicKey(walletPubkey);
        var program = getProgramPubkey();

        var [poolPubkey] = await getStakingPoolPDA();
        var [stakeInfoPubkey] = await getStakeInfoPDA(walletPubkey);

        var ix = new solanaWeb3.TransactionInstruction({
            programId: program,
            keys: [
                { pubkey: poolPubkey, isSigner: false, isWritable: true },
                { pubkey: stakeInfoPubkey, isSigner: false, isWritable: true },
                { pubkey: user, isSigner: true, isWritable: false },
            ],
            data: Buffer.from(DISCRIMINATORS.initiate_cooldown),
        });

        var tx = new solanaWeb3.Transaction();
        tx.add(ix);
        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = user;
        return tx;
    }

    /**
     * Build unstake TX
     * Accounts: staking_pool, stake_info, stake_vault, user_token_account, user, token_program
     */
    async function buildUnstakeTx(connection, walletPubkey, amount) {
        await ensureDiscriminators();

        var user = new solanaWeb3.PublicKey(walletPubkey);
        var program = getProgramPubkey();
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);

        var [poolPubkey] = await getStakingPoolPDA();
        var [stakeInfoPubkey] = await getStakeInfoPDA(walletPubkey);
        var [stakeVaultPubkey] = await getStakeVaultPDA();
        var userTokenAta = await getAssociatedTokenAddress(walletPubkey, CRTX_MINT);

        var data = Buffer.concat([
            DISCRIMINATORS.unstake,
            encodeU64(amount),
        ]);

        var ix = new solanaWeb3.TransactionInstruction({
            programId: program,
            keys: [
                { pubkey: poolPubkey, isSigner: false, isWritable: true },
                { pubkey: stakeInfoPubkey, isSigner: false, isWritable: true },
                { pubkey: stakeVaultPubkey, isSigner: false, isWritable: true },
                { pubkey: userTokenAta, isSigner: false, isWritable: true },
                { pubkey: user, isSigner: true, isWritable: true },
                { pubkey: tokenProgram, isSigner: false, isWritable: false },
            ],
            data: data,
        });

        var tx = new solanaWeb3.Transaction();
        tx.add(ix);
        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = user;
        return tx;
    }

    /**
     * Build claim_rewards TX
     * Accounts: staking_pool, stake_info, reward_vault, user_token_account, user, token_program
     */
    async function buildClaimRewardsTx(connection, walletPubkey) {
        await ensureDiscriminators();

        var user = new solanaWeb3.PublicKey(walletPubkey);
        var program = getProgramPubkey();
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);

        var [poolPubkey] = await getStakingPoolPDA();
        var [stakeInfoPubkey] = await getStakeInfoPDA(walletPubkey);
        var [rewardVaultPubkey] = await getRewardVaultPDA();
        var userTokenAta = await getAssociatedTokenAddress(walletPubkey, CRTX_MINT);

        var ix = new solanaWeb3.TransactionInstruction({
            programId: program,
            keys: [
                { pubkey: poolPubkey, isSigner: false, isWritable: true },
                { pubkey: stakeInfoPubkey, isSigner: false, isWritable: true },
                { pubkey: rewardVaultPubkey, isSigner: false, isWritable: true },
                { pubkey: userTokenAta, isSigner: false, isWritable: true },
                { pubkey: user, isSigner: true, isWritable: true },
                { pubkey: tokenProgram, isSigner: false, isWritable: false },
            ],
            data: Buffer.from(DISCRIMINATORS.claim_rewards),
        });

        var tx = new solanaWeb3.Transaction();
        tx.add(ix);
        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = user;
        return tx;
    }

    // -------------------------------------------------------------------------
    // Exposed namespace
    // -------------------------------------------------------------------------

    global.StakingSDK = {
        STAKING_PROGRAM_ID: STAKING_PROGRAM_ID,
        CRTX_MINT: CRTX_MINT,
        CRTX_DECIMALS: CRTX_DECIMALS,
        CRTX_MULTIPLIER: CRTX_MULTIPLIER,
        COOLDOWN_SECONDS: COOLDOWN_SECONDS,
        LOCK_TYPES: LOCK_TYPES,
        TIER_NAMES: TIER_NAMES,

        getStakingPoolPDA: getStakingPoolPDA,
        getStakeInfoPDA: getStakeInfoPDA,
        getStakeVaultPDA: getStakeVaultPDA,
        getRewardVaultPDA: getRewardVaultPDA,
        getAssociatedTokenAddress: getAssociatedTokenAddress,

        getPoolInfo: getPoolInfo,
        getUserStakeInfo: getUserStakeInfo,
        getUserCrtxBalance: getUserCrtxBalance,

        deserializeStakingPool: deserializeStakingPool,
        deserializeStakeInfo: deserializeStakeInfo,

        calculatePendingRewards: calculatePendingRewards,
        getUserTier: getUserTier,
        getStakeStatus: getStakeStatus,

        buildStakeTx: buildStakeTx,
        buildInitiateCooldownTx: buildInitiateCooldownTx,
        buildUnstakeTx: buildUnstakeTx,
        buildClaimRewardsTx: buildClaimRewardsTx,

        encodeU64: encodeU64,
    };

})(window);
