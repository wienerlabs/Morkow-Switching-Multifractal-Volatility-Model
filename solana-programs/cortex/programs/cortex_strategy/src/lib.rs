use anchor_lang::prelude::*;
use anchor_lang::solana_program::hash::hash;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("FvLigiv3jhcrwZH6hnDiZreawuKsRkJKaqZPdhCb8EsD");

const MAX_NAME_LEN: usize = 64;
const MAX_PROTOCOL_LEN: usize = 32;
const SEED_HASH_LEN: usize = 16;

fn name_seed(name: &str) -> [u8; SEED_HASH_LEN] {
    let h = hash(name.as_bytes());
    let mut out = [0u8; SEED_HASH_LEN];
    out.copy_from_slice(&h.to_bytes()[..SEED_HASH_LEN]);
    out
}

#[program]
pub mod cortex_strategy {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>, name: String, protocol: String) -> Result<()> {
        require!(name.len() <= MAX_NAME_LEN, StrategyError::NameTooLong);
        require!(!name.is_empty(), StrategyError::NameEmpty);
        require!(protocol.len() <= MAX_PROTOCOL_LEN, StrategyError::NameTooLong);

        let strategy = &mut ctx.accounts.strategy;
        strategy.vault = ctx.accounts.vault.key();
        strategy.authority = ctx.accounts.authority.key();
        strategy.asset_mint = ctx.accounts.asset_mint.key();
        strategy.strategy_vault = ctx.accounts.strategy_vault.key();
        strategy.name = name.clone();
        strategy.protocol = protocol;
        strategy.name_hash = name_seed(&name);
        strategy.deposited = 0;
        strategy.is_active = true;
        strategy.last_harvest = Clock::get()?.unix_timestamp;
        strategy.total_profit = 0;
        strategy.bump = ctx.bumps.strategy;
        Ok(())
    }

    pub fn deposit(ctx: Context<StrategyDeposit>, amount: u64) -> Result<()> {
        require!(amount > 0, StrategyError::ZeroAmount);
        require!(ctx.accounts.strategy.is_active, StrategyError::StrategyInactive);

        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.vault_asset_account.to_account_info(),
                    to: ctx.accounts.strategy_vault.to_account_info(),
                    authority: ctx.accounts.agent.to_account_info(),
                },
            ),
            amount,
        )?;

        ctx.accounts.strategy.deposited += amount;
        emit!(StrategyDeposited { strategy: ctx.accounts.strategy.key(), amount });
        Ok(())
    }

    pub fn withdraw(ctx: Context<StrategyWithdraw>, amount: u64) -> Result<()> {
        require!(amount > 0, StrategyError::ZeroAmount);
        require!(ctx.accounts.strategy.deposited >= amount, StrategyError::InsufficientFunds);

        let strategy = &mut ctx.accounts.strategy;
        let name_hash = strategy.name_hash;
        let seeds = &[b"strategy".as_ref(), strategy.vault.as_ref(), name_hash.as_ref(), &[strategy.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.strategy_vault.to_account_info(),
                    to: ctx.accounts.vault_asset_account.to_account_info(),
                    authority: strategy.to_account_info(),
                },
                signer,
            ),
            amount,
        )?;

        strategy.deposited -= amount;
        emit!(StrategyWithdrawn { strategy: strategy.key(), amount });
        Ok(())
    }

    pub fn harvest(ctx: Context<Harvest>) -> Result<u64> {
        let strategy = &mut ctx.accounts.strategy;
        let current_balance = ctx.accounts.strategy_vault.amount;
        let profit = current_balance.saturating_sub(strategy.deposited);

        if profit > 0 {
            strategy.total_profit += profit;
            strategy.deposited = current_balance;
        }
        strategy.last_harvest = Clock::get()?.unix_timestamp;

        emit!(StrategyHarvested { strategy: strategy.key(), profit });
        Ok(profit)
    }

    pub fn set_active(ctx: Context<SetActive>, is_active: bool) -> Result<()> {
        ctx.accounts.strategy.is_active = is_active;
        emit!(StrategyStatusChanged { strategy: ctx.accounts.strategy.key(), is_active });
        Ok(())
    }

    pub fn emergency_exit(ctx: Context<EmergencyExit>) -> Result<()> {
        let strategy = &mut ctx.accounts.strategy;
        let amount = ctx.accounts.strategy_vault.amount;

        if amount > 0 {
            let name_hash = strategy.name_hash;
            let seeds = &[b"strategy".as_ref(), strategy.vault.as_ref(), name_hash.as_ref(), &[strategy.bump]];
            let signer = &[&seeds[..]];

            token::transfer(
                CpiContext::new_with_signer(
                    ctx.accounts.token_program.to_account_info(),
                    Transfer {
                        from: ctx.accounts.strategy_vault.to_account_info(),
                        to: ctx.accounts.vault_asset_account.to_account_info(),
                        authority: strategy.to_account_info(),
                    },
                    signer,
                ),
                amount,
            )?;
        }

        strategy.deposited = 0;
        strategy.is_active = false;
        emit!(StrategyEmergencyExit { strategy: strategy.key(), amount });
        Ok(())
    }
}

#[derive(Accounts)]
#[instruction(name: String, protocol: String)]
pub struct Initialize<'info> {
    #[account(
        init, payer = authority, space = 8 + Strategy::INIT_SPACE,
        seeds = [b"strategy", vault.key().as_ref(), &name_seed(&name)], bump
    )]
    pub strategy: Account<'info, Strategy>,
    /// CHECK: Vault program account
    pub vault: UncheckedAccount<'info>,
    pub asset_mint: Account<'info, anchor_spl::token::Mint>,
    #[account(
        init, payer = authority,
        token::mint = asset_mint, token::authority = strategy,
        seeds = [b"strategy_vault", strategy.key().as_ref()], bump
    )]
    pub strategy_vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct StrategyDeposit<'info> {
    #[account(mut, seeds = [b"strategy", vault.key().as_ref(), strategy.name_hash.as_ref()], bump = strategy.bump)]
    pub strategy: Account<'info, Strategy>,
    /// CHECK: Vault key reference
    pub vault: UncheckedAccount<'info>,
    #[account(mut)]
    pub vault_asset_account: Account<'info, TokenAccount>,
    #[account(mut, address = strategy.strategy_vault)]
    pub strategy_vault: Account<'info, TokenAccount>,
    #[account(constraint = agent.key() == strategy.authority @ StrategyError::Unauthorized)]
    pub agent: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct StrategyWithdraw<'info> {
    #[account(mut, seeds = [b"strategy", vault.key().as_ref(), strategy.name_hash.as_ref()], bump = strategy.bump)]
    pub strategy: Account<'info, Strategy>,
    /// CHECK: Vault key reference
    pub vault: UncheckedAccount<'info>,
    #[account(mut)]
    pub vault_asset_account: Account<'info, TokenAccount>,
    #[account(mut, address = strategy.strategy_vault)]
    pub strategy_vault: Account<'info, TokenAccount>,
    #[account(constraint = agent.key() == strategy.authority @ StrategyError::Unauthorized)]
    pub agent: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct Harvest<'info> {
    #[account(mut, seeds = [b"strategy", strategy.vault.as_ref(), strategy.name_hash.as_ref()], bump = strategy.bump)]
    pub strategy: Account<'info, Strategy>,
    #[account(address = strategy.strategy_vault)]
    pub strategy_vault: Account<'info, TokenAccount>,
    pub agent: Signer<'info>,
}

#[derive(Accounts)]
pub struct SetActive<'info> {
    #[account(mut, seeds = [b"strategy", strategy.vault.as_ref(), strategy.name_hash.as_ref()], bump = strategy.bump, has_one = authority)]
    pub strategy: Account<'info, Strategy>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyExit<'info> {
    #[account(mut, seeds = [b"strategy", strategy.vault.as_ref(), strategy.name_hash.as_ref()], bump = strategy.bump)]
    pub strategy: Account<'info, Strategy>,
    #[account(mut, address = strategy.strategy_vault)]
    pub strategy_vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub vault_asset_account: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
#[derive(InitSpace)]
pub struct Strategy {
    pub vault: Pubkey,
    pub authority: Pubkey,
    pub asset_mint: Pubkey,
    pub strategy_vault: Pubkey,
    #[max_len(64)]
    pub name: String,
    #[max_len(32)]
    pub protocol: String,
    pub name_hash: [u8; 16],
    pub deposited: u64,
    pub is_active: bool,
    pub last_harvest: i64,
    pub total_profit: u64,
    pub bump: u8,
}

#[event]
pub struct StrategyDeposited { pub strategy: Pubkey, pub amount: u64 }
#[event]
pub struct StrategyWithdrawn { pub strategy: Pubkey, pub amount: u64 }
#[event]
pub struct StrategyHarvested { pub strategy: Pubkey, pub profit: u64 }
#[event]
pub struct StrategyStatusChanged { pub strategy: Pubkey, pub is_active: bool }
#[event]
pub struct StrategyEmergencyExit { pub strategy: Pubkey, pub amount: u64 }

#[error_code]
pub enum StrategyError {
    #[msg("Name too long (max 64 chars)")] NameTooLong,
    #[msg("Name cannot be empty")] NameEmpty,
    #[msg("Zero amount")] ZeroAmount,
    #[msg("Strategy inactive")] StrategyInactive,
    #[msg("Insufficient funds")] InsufficientFunds,
    #[msg("Unauthorized")] Unauthorized,
}
