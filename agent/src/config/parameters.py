"""
ML Model Parameters and Training Configuration.
These are STARTING VALUES - Optuna will optimize them.

Updated for Solana blockchain:
- Lower fees (~0.00025 SOL vs $5-50 on Ethereum)
- Faster block times (400ms vs 12s)
- Different DEX structure (Raydium, Orca, Phoenix vs Uniswap, SushiSwap)
"""

# =============================================================================
# SOLANA CHAIN PARAMETERS
# =============================================================================

SOLANA_CHAIN_PARAMS = {
    # Transaction costs
    'base_tx_fee_lamports': 5000,           # 5000 lamports = 0.000005 SOL base fee
    'priority_fee_lamports': 50000,         # Priority fee for faster inclusion
    'compute_unit_price': 1000,             # Micro-lamports per compute unit
    'compute_units_limit': 200000,          # Max compute units per tx

    # Block timing
    'slot_time_ms': 400,                    # ~400ms per slot
    'slots_per_epoch': 432000,              # ~2-3 days per epoch

    # DEX fees
    'raydium_fee_pct': 0.25,                # Raydium AMM: 0.25%
    'orca_fee_pct': 0.30,                   # Orca Whirlpool: 0.3% (varies by pool)
    'phoenix_fee_pct': 0.10,                # Phoenix: ~0.1% (order book)
    'jupiter_fee_pct': 0.00,                # Jupiter: no additional fee (uses underlying DEX fees)

    # Slippage defaults
    'default_slippage_bps': 50,             # 0.5% default slippage tolerance
    'min_slippage_bps': 10,                 # 0.1% minimum
    'max_slippage_bps': 300,                # 3% maximum
}

# =============================================================================
# XGBoost Model Parameters (Starting Values)
# =============================================================================

ARBITRAGE_PARAMS = {
    'n_estimators': 150,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42
}

LENDING_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.9,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.8,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'random_state': 42
}

LP_PARAMS = {
    'n_estimators': 180,
    'max_depth': 9,
    'learning_rate': 0.04,
    'subsample': 0.75,
    'colsample_bytree': 0.85,
    'gamma': 0.12,
    'reg_alpha': 0.15,
    'reg_lambda': 1.2,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'tree_method': 'hist',
    'random_state': 42
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_CONFIG = {
    'train_size': 0.80,
    'validation_size': 0.15,
    'test_size': 0.05,
    'split_method': 'time_based',
    'cv_folds': 5,
    'cv_strategy': 'TimeSeriesSplit',
    'early_stopping_rounds': 50,
    'verbose_eval': 10
}

# =============================================================================
# Feature Engineering Parameters
# =============================================================================

FEATURE_PARAMS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'price_lag': [1, 3, 5, 10],
    'volume_rolling_window': [5, 15, 60],
    'spread_threshold': 0.001,              # Lower threshold for Solana (cheaper fees)
    'fee_threshold_lamports': 100000,       # ~0.0001 SOL max fee acceptable
    'min_liquidity': 50000                  # Lower min liquidity for Solana
}

# =============================================================================
# Labeling Parameters (Solana-optimized)
# =============================================================================

LABELING_PARAMS = {
    'min_profit_pct': 0.001,                # 0.1% min profit (lower due to cheap fees)
    'tx_cost_estimate': True,               # Include Solana tx cost
    'slippage_estimate': 0.001,             # 0.1% slippage estimate
    'holding_period': 1,                    # 1 slot (~400ms)
    # Solana-specific cost estimates
    'sol_tx_fee': 0.00025,                  # ~0.00025 SOL per tx (priority included)
    'raydium_fee_pct': 0.0025,              # 0.25%
    'orca_fee_pct': 0.003,                  # 0.3%
}

# =============================================================================
# Optuna Hyperparameter Tuning
# =============================================================================

OPTUNA_CONFIG = {
    'n_trials': 100,
    'timeout': 7200,  # 2 hours
    'n_jobs': -1,
    'direction': 'maximize',
    'metric': 'sharpe_ratio',
    'pruner': 'MedianPruner',
    'search_space': {
        'n_estimators': (100, 300),
        'max_depth': (5, 12),
        'learning_rate': (0.01, 0.15),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 0.5),
        'reg_alpha': (0, 2),
        'reg_lambda': (0, 2)
    }
}

# =============================================================================
# Risk Management Parameters (Solana-optimized)
# =============================================================================

RISK_PARAMS = {
    'max_position_size': 0.15,              # Max 15% per position
    'max_open_positions': 5,                # More positions allowed (lower fees)
    'min_liquidity': 50_000,                # Minimum liquidity USD
    'stop_loss': 0.02,                      # 2% stop loss
    'take_profit': 0.03,                    # 3% take profit (faster on Solana)
    'max_daily_loss': 0.05,                 # 5% daily loss limit
    'max_priority_fee_lamports': 500000,    # Max priority fee: 0.0005 SOL
    'min_profit_after_fees': 0.001,         # 0.1% min profit after fees
    'consecutive_losses': 5,                # More tolerance (cheaper to test)
    'weekly_loss_limit': 0.10               # 10% weekly loss limit
}

