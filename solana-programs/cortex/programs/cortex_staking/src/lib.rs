use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer, Mint};

declare_id!("rYantWFyB4PsL36r9XB7nUb8TQ1pAhn9A87S6TbpMsr");

#[cfg(not(feature = "testing"))]
pub const COOLDOWN_PERIOD: i64 = 3 * 24 * 60 * 60; // 3 days (259200 seconds)

#[cfg(feature = "testing")]
pub const COOLDOWN_PERIOD: i64 = 60; // 1 minute — test-only override
pub const MAX_STAKE_PER_USER: u64 = 100_000_000_000_000_000; // 100M tokens max (9 decimals)
pub const MIN_STAKE: u64 = 1_000_000_000; // 1 token minimum (9 decimals)
pub const PRECISION: u128 = 1_000_000_000_000; // 1e12 for reward calculations

// Lock durations in seconds
pub const LOCK_FLEXIBLE: i64 = 0;
pub const LOCK_14_DAYS: i64 = 14 * 24 * 60 * 60;
pub const LOCK_30_DAYS: i64 = 30 * 24 * 60 * 60;
pub const LOCK_90_DAYS: i64 = 90 * 24 * 60 * 60;
pub const LOCK_180_DAYS: i64 = 180 * 24 * 60 * 60;
pub const LOCK_365_DAYS: i64 = 365 * 24 * 60 * 60;

// Weight multipliers (in basis points, 10000 = 1x)
pub const MULTIPLIER_FLEXIBLE: u64 = 5_000;   // 0.5x
pub const MULTIPLIER_14_DAYS: u64 = 10_000;   // 1x
pub const MULTIPLIER_30_DAYS: u64 = 15_000;   // 1.5x
pub const MULTIPLIER_90_DAYS: u64 = 25_000;   // 2.5x
pub const MULTIPLIER_180_DAYS: u64 = 40_000;  // 4x
pub const MULTIPLIER_365_DAYS: u64 = 60_000;  // 6x

#[program]
pub mod cortex_staking {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>, tier_thresholds: [u64; 3]) -> Result<()> {
        let pool = &mut ctx.accounts.staking_pool;
        pool.authority = ctx.accounts.authority.key();
        pool.stake_mint = ctx.accounts.stake_mint.key();
        pool.stake_vault = ctx.accounts.stake_vault.key();
        pool.total_staked = 0;
        pool.total_weight = 0;
        pool.tier_thresholds = tier_thresholds;
        pool.reward_rate = 0;
        pool.last_update_time = Clock::get()?.unix_timestamp;
        pool.acc_reward_per_weight = 0;
        pool.bump = *ctx.bumps.get("staking_pool").unwrap();
        Ok(())
    }

    pub fn stake(ctx: Context<Stake>, amount: u64, lock_type: u8) -> Result<()> {
        require!(amount >= MIN_STAKE, StakingError::AmountTooSmall);
        require!(lock_type <= 5, StakingError::InvalidDuration);

        let clock = Clock::get()?;
        let pool = &mut ctx.accounts.staking_pool;
        let stake_info = &mut ctx.accounts.stake_info;

        // Validate new total won't exceed max
        let new_total = stake_info.amount.checked_add(amount).ok_or(StakingError::MathOverflow)?;
        require!(new_total <= MAX_STAKE_PER_USER, StakingError::ExceedsMaxStake);

        update_rewards(pool, clock.unix_timestamp)?;

        // Handle existing stake - claim pending rewards first
        let old_weight = stake_info.weight;
        if stake_info.amount > 0 {
            let pending = calculate_pending_reward(stake_info, pool)?;
            stake_info.pending_rewards = stake_info.pending_rewards
                .checked_add(pending)
                .ok_or(StakingError::MathOverflow)?;
        }

        let (lock_duration, multiplier) = get_lock_params(lock_type);
        let new_weight = calculate_weight(new_total, multiplier)?;

        // CEI: Update state before transfer
        stake_info.owner = ctx.accounts.user.key();
        stake_info.amount = new_total;
        stake_info.lock_end = clock.unix_timestamp.checked_add(lock_duration).ok_or(StakingError::MathOverflow)?;
        stake_info.lock_type = lock_type;
        stake_info.weight = new_weight;
        stake_info.cooldown_start = 0;
        stake_info.reward_debt = (new_weight as u128)
            .checked_mul(pool.acc_reward_per_weight)
            .ok_or(StakingError::MathOverflow)?
            .checked_div(PRECISION)
            .ok_or(StakingError::MathOverflow)? as u64;
        stake_info.bump = *ctx.bumps.get("stake_info").unwrap();

        // Update pool totals with checked math
        pool.total_staked = pool.total_staked.checked_add(amount).ok_or(StakingError::MathOverflow)?;
        pool.total_weight = pool.total_weight
            .checked_sub(old_weight)
            .ok_or(StakingError::MathOverflow)?
            .checked_add(new_weight)
            .ok_or(StakingError::MathOverflow)?;

        // Transfer tokens
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.user_token_account.to_account_info(),
                    to: ctx.accounts.stake_vault.to_account_info(),
                    authority: ctx.accounts.user.to_account_info(),
                },
            ),
            amount,
        )?;

        emit!(Staked { user: ctx.accounts.user.key(), amount, lock_type, weight: new_weight });
        Ok(())
    }

    pub fn initiate_cooldown(ctx: Context<InitiateCooldown>) -> Result<()> {
        let clock = Clock::get()?;
        let pool = &mut ctx.accounts.staking_pool;
        let stake_info = &mut ctx.accounts.stake_info;

        require!(stake_info.amount > 0, StakingError::NoStake);
        require!(clock.unix_timestamp >= stake_info.lock_end, StakingError::StakeLocked);

        update_rewards(pool, clock.unix_timestamp)?;
        let pending = calculate_pending_reward(stake_info, pool)?;
        stake_info.pending_rewards = stake_info.pending_rewards.checked_add(pending).ok_or(StakingError::MathOverflow)?;

        pool.total_weight = pool.total_weight.saturating_sub(stake_info.weight);
        stake_info.weight = 0;
        stake_info.reward_debt = 0;
        stake_info.cooldown_start = clock.unix_timestamp;

        emit!(CooldownInitiated { user: ctx.accounts.user.key(), cooldown_end: clock.unix_timestamp + COOLDOWN_PERIOD });
        Ok(())
    }

    pub fn unstake(ctx: Context<Unstake>, amount: u64) -> Result<()> {
        let clock = Clock::get()?;
        let stake_info = &mut ctx.accounts.stake_info;

        require!(stake_info.amount > 0, StakingError::NoStake);
        require!(amount > 0, StakingError::ZeroAmount);
        require!(stake_info.cooldown_start > 0, StakingError::NoCooldownActive);
        require!(clock.unix_timestamp >= stake_info.cooldown_start + COOLDOWN_PERIOD, StakingError::CooldownNotComplete);

        // Determine actual unstake amount (cap at staked amount)
        let unstake_amount = if amount >= stake_info.amount { stake_info.amount } else { amount };
        let remaining_amount = stake_info.amount.checked_sub(unstake_amount).ok_or(StakingError::MathOverflow)?;
        let old_weight = stake_info.weight;
        let lock_type = stake_info.lock_type;
        let stored_pending = stake_info.pending_rewards;

        // Get pool bump before mutable borrow
        let pool_bump = ctx.accounts.staking_pool.bump;

        // Calculate new weight
        let (_, multiplier) = get_lock_params(lock_type);
        let new_weight = if remaining_amount > 0 {
            calculate_weight(remaining_amount, multiplier)?
        } else {
            0
        };

        // Update pool state
        {
            let pool = &mut ctx.accounts.staking_pool;
            update_rewards(pool, clock.unix_timestamp)?;
        }

        // Calculate pending rewards
        let pending = calculate_pending_reward(stake_info, &ctx.accounts.staking_pool)?
            .checked_add(stored_pending).ok_or(StakingError::MathOverflow)?;
        let acc_reward = ctx.accounts.staking_pool.acc_reward_per_weight;

        // Update pool totals
        {
            let pool = &mut ctx.accounts.staking_pool;
            pool.total_staked = pool.total_staked.checked_sub(unstake_amount).ok_or(StakingError::MathOverflow)?;
            pool.total_weight = pool.total_weight.checked_sub(old_weight).ok_or(StakingError::MathOverflow)?
                .checked_add(new_weight).ok_or(StakingError::MathOverflow)?;
        }

        // Transfer tokens
        let seeds = &[b"staking_pool".as_ref(), &[pool_bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.stake_vault.to_account_info(),
                    to: ctx.accounts.user_token_account.to_account_info(),
                    authority: ctx.accounts.staking_pool.to_account_info(),
                },
                signer,
            ),
            unstake_amount,
        )?;

        // Update stake info
        stake_info.amount = remaining_amount;
        stake_info.weight = new_weight;
        stake_info.pending_rewards = pending;

        // Reset cooldown only if fully unstaked
        if remaining_amount == 0 {
            stake_info.cooldown_start = 0;
        }

        // Update reward debt for remaining stake
        stake_info.reward_debt = (new_weight as u128)
            .checked_mul(acc_reward)
            .ok_or(StakingError::MathOverflow)?
            .checked_div(PRECISION)
            .ok_or(StakingError::MathOverflow)? as u64;

        emit!(Unstaked { user: ctx.accounts.user.key(), amount: unstake_amount });
        Ok(())
    }

    pub fn claim_rewards(ctx: Context<ClaimRewards>) -> Result<()> {
        let clock = Clock::get()?;
        let pool = &mut ctx.accounts.staking_pool;
        let stake_info = &mut ctx.accounts.stake_info;

        // Validate ownership
        require!(stake_info.owner == ctx.accounts.user.key(), StakingError::Unauthorized);

        update_rewards(pool, clock.unix_timestamp)?;

        let calculated = calculate_pending_reward(stake_info, pool)?;
        let pending = calculated.checked_add(stake_info.pending_rewards).ok_or(StakingError::MathOverflow)?;
        require!(pending > 0, StakingError::NoRewards);

        // CEI: Update state before transfer
        stake_info.reward_debt = (stake_info.weight as u128)
            .checked_mul(pool.acc_reward_per_weight)
            .ok_or(StakingError::MathOverflow)?
            .checked_div(PRECISION)
            .ok_or(StakingError::MathOverflow)? as u64;
        stake_info.pending_rewards = 0;

        // Transfer reward tokens from reward vault
        let seeds = &[b"staking_pool".as_ref(), &[pool.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.reward_vault.to_account_info(),
                    to: ctx.accounts.user_token_account.to_account_info(),
                    authority: pool.to_account_info(),
                },
                signer,
            ),
            pending,
        )?;

        emit!(RewardsClaimed { user: ctx.accounts.user.key(), amount: pending });
        Ok(())
    }

    pub fn get_user_tier(ctx: Context<GetUserTier>) -> Result<u8> {
        let pool = &ctx.accounts.staking_pool;
        let stake_info = &ctx.accounts.stake_info;

        let tier = if stake_info.amount >= pool.tier_thresholds[2] { 3 }
            else if stake_info.amount >= pool.tier_thresholds[1] { 2 }
            else if stake_info.amount >= pool.tier_thresholds[0] { 1 }
            else { 0 };

        Ok(tier)
    }

    pub fn set_reward_rate(ctx: Context<SetRewardRate>, reward_rate: u64) -> Result<()> {
        let clock = Clock::get()?;
        let pool = &mut ctx.accounts.staking_pool;

        require!(ctx.accounts.authority.key() == pool.authority, StakingError::Unauthorized);

        update_rewards(pool, clock.unix_timestamp)?;
        pool.reward_rate = reward_rate;

        emit!(RewardRateUpdated { new_rate: reward_rate });
        Ok(())
    }

    pub fn init_reward_vault(_ctx: Context<InitRewardVault>) -> Result<()> {
        Ok(())
    }
}

fn update_rewards(pool: &mut Account<StakingPool>, current_time: i64) -> Result<()> {
    if pool.total_weight > 0 && pool.reward_rate > 0 {
        let time_elapsed = current_time.saturating_sub(pool.last_update_time) as u128;
        let reward = time_elapsed.checked_mul(pool.reward_rate as u128).unwrap_or(0);
        let reward_per_weight = reward
            .checked_mul(PRECISION)
            .unwrap_or(0)
            .checked_div(pool.total_weight as u128)
            .unwrap_or(0);
        pool.acc_reward_per_weight = pool.acc_reward_per_weight.saturating_add(reward_per_weight);
    }
    pool.last_update_time = current_time;
    Ok(())
}

fn calculate_pending_reward(stake_info: &Account<StakeInfo>, pool: &Account<StakingPool>) -> Result<u64> {
    if stake_info.weight == 0 { return Ok(0); }
    let acc = (stake_info.weight as u128)
        .checked_mul(pool.acc_reward_per_weight)
        .unwrap_or(0)
        .checked_div(PRECISION)
        .unwrap_or(0) as u64;
    Ok(acc.saturating_sub(stake_info.reward_debt))
}

fn get_lock_params(lock_type: u8) -> (i64, u64) {
    match lock_type {
        0 => (LOCK_FLEXIBLE, MULTIPLIER_FLEXIBLE),
        1 => (LOCK_14_DAYS, MULTIPLIER_14_DAYS),
        2 => (LOCK_30_DAYS, MULTIPLIER_30_DAYS),
        3 => (LOCK_90_DAYS, MULTIPLIER_90_DAYS),
        4 => (LOCK_180_DAYS, MULTIPLIER_180_DAYS),
        5 => (LOCK_365_DAYS, MULTIPLIER_365_DAYS),
        _ => (LOCK_FLEXIBLE, MULTIPLIER_FLEXIBLE),
    }
}

fn calculate_weight(amount: u64, multiplier: u64) -> Result<u64> {
    let weight = (amount as u128)
        .checked_mul(multiplier as u128)
        .ok_or(StakingError::MathOverflow)?
        .checked_div(10_000)
        .ok_or(StakingError::MathOverflow)?;
    Ok(weight as u64)
}


#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + StakingPool::INIT_SPACE,
        seeds = [b"staking_pool"],
        bump
    )]
    pub staking_pool: Account<'info, StakingPool>,
    pub stake_mint: Account<'info, Mint>,
    #[account(
        init,
        payer = authority,
        token::mint = stake_mint,
        token::authority = staking_pool,
        seeds = [b"stake_vault"],
        bump
    )]
    pub stake_vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct Stake<'info> {
    #[account(mut, seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + StakeInfo::INIT_SPACE,
        seeds = [b"stake_info", user.key().as_ref()],
        bump
    )]
    pub stake_info: Account<'info, StakeInfo>,
    #[account(mut, token::authority = staking_pool)]
    pub stake_vault: Account<'info, TokenAccount>,
    #[account(mut, token::authority = user)]
    pub user_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitiateCooldown<'info> {
    #[account(mut, seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    #[account(mut, seeds = [b"stake_info", user.key().as_ref()], bump = stake_info.bump,
        constraint = stake_info.owner == user.key() @ StakingError::Unauthorized)]
    pub stake_info: Account<'info, StakeInfo>,
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct Unstake<'info> {
    #[account(mut, seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    #[account(mut, seeds = [b"stake_info", user.key().as_ref()], bump = stake_info.bump)]
    pub stake_info: Account<'info, StakeInfo>,
    #[account(mut, token::authority = staking_pool)]
    pub stake_vault: Account<'info, TokenAccount>,
    #[account(mut, token::authority = user)]
    pub user_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct ClaimRewards<'info> {
    #[account(mut, seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    #[account(mut, seeds = [b"stake_info", user.key().as_ref()], bump = stake_info.bump,
        constraint = stake_info.owner == user.key() @ StakingError::Unauthorized)]
    pub stake_info: Account<'info, StakeInfo>,
    #[account(mut, seeds = [b"reward_vault"], bump, token::authority = staking_pool)]
    pub reward_vault: Account<'info, TokenAccount>,
    #[account(mut, token::authority = user)]
    pub user_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct GetUserTier<'info> {
    #[account(seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    #[account(seeds = [b"stake_info", user.key().as_ref()], bump = stake_info.bump)]
    pub stake_info: Account<'info, StakeInfo>,
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct SetRewardRate<'info> {
    #[account(mut, seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct InitRewardVault<'info> {
    #[account(seeds = [b"staking_pool"], bump = staking_pool.bump)]
    pub staking_pool: Account<'info, StakingPool>,
    pub stake_mint: Account<'info, Mint>,
    #[account(
        init,
        payer = authority,
        token::mint = stake_mint,
        token::authority = staking_pool,
        seeds = [b"reward_vault"],
        bump
    )]
    pub reward_vault: Account<'info, TokenAccount>,
    #[account(mut, constraint = authority.key() == staking_pool.authority @ StakingError::Unauthorized)]
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[account]
#[derive(InitSpace)]
pub struct StakingPool {
    pub authority: Pubkey,
    pub stake_mint: Pubkey,
    pub stake_vault: Pubkey,
    pub total_staked: u64,
    pub total_weight: u64,
    pub tier_thresholds: [u64; 3],
    pub reward_rate: u64,
    pub last_update_time: i64,
    pub acc_reward_per_weight: u128,
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct StakeInfo {
    pub owner: Pubkey,
    pub amount: u64,
    pub lock_end: i64,
    pub lock_type: u8,  // 0=flex, 1=14d, 2=30d, 3=90d, 4=180d, 5=365d
    pub weight: u64,
    pub cooldown_start: i64,
    pub reward_debt: u64,
    pub pending_rewards: u64,
    pub bump: u8,
}

#[event]
pub struct Staked { pub user: Pubkey, pub amount: u64, pub lock_type: u8, pub weight: u64 }
#[event]
pub struct Unstaked { pub user: Pubkey, pub amount: u64 }
#[event]
pub struct CooldownInitiated { pub user: Pubkey, pub cooldown_end: i64 }
#[event]
pub struct RewardsClaimed { pub user: Pubkey, pub amount: u64 }
#[event]
pub struct RewardRateUpdated { pub new_rate: u64 }

#[error_code]
pub enum StakingError {
    #[msg("Amount must be greater than zero")] ZeroAmount,
    #[msg("Amount below minimum stake")] AmountTooSmall,
    #[msg("Exceeds maximum stake per user")] ExceedsMaxStake,
    #[msg("Invalid lock duration")] InvalidDuration,
    #[msg("Stake is still locked")] StakeLocked,
    #[msg("Cooldown not complete")] CooldownNotComplete,
    #[msg("No cooldown active")] NoCooldownActive,
    #[msg("No stake found")] NoStake,
    #[msg("No rewards to claim")] NoRewards,
    #[msg("Unauthorized")] Unauthorized,
    #[msg("Math overflow")] MathOverflow,
}
