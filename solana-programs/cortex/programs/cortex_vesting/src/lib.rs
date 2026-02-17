use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns");

#[program]
pub mod cortex_vesting {
    use super::*;

    pub fn create_vesting(
        ctx: Context<CreateVesting>,
        total_amount: u64,
        category: VestingCategory,
        start_time: i64,
    ) -> Result<()> {
        let schedule = &mut ctx.accounts.vesting_schedule;
        
        let (cliff_duration, vesting_duration, tge_unlock_percent) = match category {
            VestingCategory::PrivateSale => (0, 180 * 24 * 60 * 60, 10), // 10% TGE, 6 months
            VestingCategory::PublicSale => (0, 90 * 24 * 60 * 60, 25),   // 25% TGE, 3 months
            VestingCategory::Team => (180 * 24 * 60 * 60, 540 * 24 * 60 * 60, 0), // 6mo cliff, 18mo vest
            VestingCategory::Treasury => (0, 730 * 24 * 60 * 60, 5),     // 5% TGE, 24 months
            VestingCategory::Marketing => (0, 540 * 24 * 60 * 60, 10),   // 10% TGE, 18 months
        };

        schedule.beneficiary = ctx.accounts.beneficiary.key();
        schedule.category = category;
        schedule.total_amount = total_amount;
        schedule.claimed_amount = 0;
        schedule.start_time = start_time;
        schedule.cliff_duration = cliff_duration;
        schedule.vesting_duration = vesting_duration;
        schedule.tge_unlock_percent = tge_unlock_percent;
        schedule.tge_claimed = false;
        schedule.bump = *ctx.bumps.get("vesting_schedule").unwrap();

        Ok(())
    }

    pub fn claim_tokens(ctx: Context<ClaimTokens>) -> Result<()> {
        let clock = Clock::get()?;

        let claimable = calculate_claimable(&ctx.accounts.vesting_schedule, clock.unix_timestamp)?;
        require!(claimable > 0, VestingError::NoTokensAvailable);

        let bump = ctx.accounts.vesting_schedule.bump;
        let beneficiary = ctx.accounts.vesting_schedule.beneficiary;
        let seeds = &[
            b"vesting",
            beneficiary.as_ref(),
            &[bump]
        ];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.vesting_vault.to_account_info(),
                    to: ctx.accounts.beneficiary_token_account.to_account_info(),
                    authority: ctx.accounts.vesting_schedule.to_account_info(),
                },
                signer,
            ),
            claimable,
        )?;

        ctx.accounts.vesting_schedule.claimed_amount += claimable;

        if ctx.accounts.vesting_schedule.tge_unlock_percent > 0
            && !ctx.accounts.vesting_schedule.tge_claimed
        {
            ctx.accounts.vesting_schedule.tge_claimed = true;
        }

        emit!(TokensClaimed {
            beneficiary,
            amount: claimable,
            total_claimed: ctx.accounts.vesting_schedule.claimed_amount,
        });

        Ok(())
    }

    pub fn get_claimable_amount(ctx: Context<GetClaimable>) -> Result<u64> {
        let schedule = &ctx.accounts.vesting_schedule;
        let clock = Clock::get()?;
        calculate_claimable(schedule, clock.unix_timestamp)
    }
}

fn calculate_claimable(schedule: &VestingSchedule, current_time: i64) -> Result<u64> {
    if current_time < schedule.start_time {
        return Ok(0);
    }

    let mut total_unlocked = 0u64;

    // TGE unlock
    if schedule.tge_unlock_percent > 0 && !schedule.tge_claimed {
        total_unlocked = (schedule.total_amount as u128)
            .checked_mul(schedule.tge_unlock_percent as u128)
            .ok_or(VestingError::MathOverflow)?
            .checked_div(100)
            .ok_or(VestingError::MathOverflow)? as u64;
    }

    // Check cliff
    let cliff_end = schedule.start_time + schedule.cliff_duration;
    if current_time < cliff_end {
        return Ok(total_unlocked.saturating_sub(schedule.claimed_amount));
    }

    // Linear vesting
    let vesting_end = schedule.start_time + schedule.cliff_duration + schedule.vesting_duration;
    let vesting_amount = schedule.total_amount - (schedule.total_amount * schedule.tge_unlock_percent as u64 / 100);
    
    if current_time >= vesting_end {
        total_unlocked = schedule.total_amount;
    } else {
        let time_since_cliff = current_time - cliff_end;
        let vested = (vesting_amount as u128)
            .checked_mul(time_since_cliff as u128)
            .ok_or(VestingError::MathOverflow)?
            .checked_div(schedule.vesting_duration as u128)
            .ok_or(VestingError::MathOverflow)? as u64;
        
        total_unlocked += vested;
    }

    Ok(total_unlocked.saturating_sub(schedule.claimed_amount))
}

#[derive(Accounts)]
pub struct CreateVesting<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + VestingSchedule::INIT_SPACE,
        seeds = [b"vesting", beneficiary.key().as_ref()],
        bump
    )]
    pub vesting_schedule: Account<'info, VestingSchedule>,
    
    /// CHECK: Beneficiary
    pub beneficiary: AccountInfo<'info>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ClaimTokens<'info> {
    #[account(
        mut,
        seeds = [b"vesting", beneficiary.key().as_ref()],
        bump = vesting_schedule.bump,
        has_one = beneficiary
    )]
    pub vesting_schedule: Account<'info, VestingSchedule>,
    
    #[account(mut)]
    pub vesting_vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub beneficiary_token_account: Account<'info, TokenAccount>,
    
    pub beneficiary: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct GetClaimable<'info> {
    #[account(
        seeds = [b"vesting", beneficiary.key().as_ref()],
        bump = vesting_schedule.bump
    )]
    pub vesting_schedule: Account<'info, VestingSchedule>,
    pub beneficiary: Signer<'info>,
}

#[account]
#[derive(InitSpace)]
pub struct VestingSchedule {
    pub beneficiary: Pubkey,
    pub category: VestingCategory,
    pub total_amount: u64,
    pub claimed_amount: u64,
    pub start_time: i64,
    pub cliff_duration: i64,
    pub vesting_duration: i64,
    pub tge_unlock_percent: u8,
    pub tge_claimed: bool,
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, InitSpace)]
pub enum VestingCategory {
    PrivateSale,
    PublicSale,
    Team,
    Treasury,
    Marketing,
}

#[event]
pub struct TokensClaimed {
    pub beneficiary: Pubkey,
    pub amount: u64,
    pub total_claimed: u64,
}

#[error_code]
pub enum VestingError {
    #[msg("No tokens available to claim")]
    NoTokensAvailable,
    #[msg("Math overflow")]
    MathOverflow,
}

