/**
 * CRTX Agent (Orchestrator)
 *
 * Lightweight orchestrator that coordinates independent analyst agents:
 * - ArbitrageAnalyst: CEX/DEX arbitrage opportunities
 * - LPAnalyst: Liquidity pool rebalancing with ML
 * - MomentumAnalyst: Funding rate arbitrage
 * - SpeculationAnalyst: Multi-source sentiment analysis (Twitter, CryptoPanic, Telegram)
 * - FundamentalAnalyst: On-chain token health analysis (Helius API)
 *
 * Responsibilities:
 * - Coordinate parallel analyst execution
 * - Risk management and position sizing
 * - Portfolio management
 * - Trade execution
 *
 * ARCHITECTURE: Analysts run independently in parallel via Promise.all()
 */

import { Keypair, Connection, PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { getSolanaConnection, getActiveRpcUrl, recordRpcFailure, recordRpcSuccess } from '../services/solana/connection.js';
import { TOKEN_PROGRAM_ID, getMint } from '@solana/spl-token';
import bs58 from 'bs58';
import { RiskManager } from '../services/riskManager.js';
import { scanMarkets, type MarketSnapshot, type ArbitrageOpportunity, type LPPool, type PerpsOpportunity, type FundingArbitrageOpportunity } from '../services/marketScanner/index.js';
import { createArbitrageExecutor, type ArbitrageResult } from '../services/arbitrageExecutor.js';
import { logger } from '../services/logger.js';
import { logArbitrageEvaluation, logLPEvaluation } from '../services/agentThoughts.js';
import { getPerpsService, getPerpsScanner, type PerpsVenue, type PositionSide } from '../services/perps/index.js';
import { getVolatility24h, calculateVolatilityFromPrices } from '../services/marketData.js';
import { featureBuilder } from '../services/featureBuilder.js';
import { lpRebalancerModel, type PredictionResult } from '../inference/model.js';
import { getPortfolioManager, type PortfolioManager } from '../services/portfolioManager.js';
import { LPExecutor, type LPPoolInfo, type SupportedDex, TOKEN_MINTS } from '../services/lpExecutor/index.js';
import { getSentimentIntegration, type SentimentAdjustedScore } from '../services/sentiment/sentimentIntegration.js';
import { ArbitrageAnalyst, MomentumAnalyst, LPAnalyst, SpeculationAnalyst, FundamentalAnalyst, LendingAnalyst, SpotAnalyst, PumpFunAnalyst, NewsAnalyst } from './analysts/index.js';
import { getTradingMode, type ModeConfig, TradingMode } from '../config/tradingModes.js';
import { promptTradingMode, displayModeConfig } from '../config/modeSelector.js';
import { SimpleLendingExecutor, type LendingDepositParams } from '../services/lending/simpleLendingExecutor.js';
import { SpotExecutor, type SpotBuyParams } from '../services/spot/spotExecutor.js';
import { LPPositionMonitor } from '../services/lpExecutor/lpPositionMonitor.js';
import { ExitManager } from '../services/trading/exitManager.js';
import type { SpotPosition, ApprovedToken, ExitLevels } from '../services/trading/types.js';
import { LendingPositionMonitor, type TrackedLendingPosition } from '../services/lending/lendingPositionMonitor.js';

// Consensus system imports
import {
  getVotingEngine,
  getResearchDebateManager,
  getPerformanceTracker,
  logVotes,
  logConsensusResult,
  logResearchDebate,
  type Vote,
  type VoteDecision,
  type ConsensusResult,
  DEFAULT_CONSENSUS_CONFIG,
} from '../services/consensus/index.js';
import { getBullishResearcher, type ResearchInput } from '../services/researchers/BullishResearcher.js';
import { getBearishResearcher } from '../services/researchers/BearishResearcher.js';

// Evaluated opportunity with risk assessment
export interface EvaluatedOpportunity {
  type: 'arbitrage' | 'lp' | 'perps' | 'funding_arb' | 'speculation' | 'fundamental' | 'lending' | 'spot' | 'pumpfun' | 'news';
  name: string;
  expectedReturn: number;  // % return
  riskScore: number;       // 1-10
  confidence: number;      // 0-1 model confidence (may be sentiment-adjusted)
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  raw: ArbitrageOpportunity | LPPool | PerpsOpportunity | FundingArbitrageOpportunity | any;

  // Sentiment adjustment details (if available)
  sentimentAdjustment?: SentimentAdjustedScore;

  // Perps-specific fields
  perpsDetails?: {
    venue: PerpsVenue;
    market: string;
    side: PositionSide;
    leverage: number;
    fundingRate: number;
    liquidationDistance?: number;
  };

  // PumpFun-specific fields
  pumpfunDetails?: {
    tokenMint: string;
    amountSol: number;
    slippageBps: number;
    riskCheck: any;
    execute: () => Promise<any>;
  };
}

export interface AgentConfig {
  portfolioValueUsd: number;
  minConfidence: number;       // Min model confidence to trade
  minRiskAdjustedReturn: number; // Min return after risk adjustment
  dryRun: boolean;             // Log only, don't execute
  volatility24h?: number;      // Current market volatility
  solanaPrivateKey?: string;   // Wallet private key for live execution
  solanaRpcUrl?: string;       // Solana RPC URL
}

const DEFAULT_CONFIG: AgentConfig = {
  portfolioValueUsd: parseFloat(process.env.PORTFOLIO_VALUE_USD || '100'),
  minConfidence: 0.60,
  minRiskAdjustedReturn: 1.0,
  dryRun: false,
  volatility24h: undefined,
  solanaPrivateKey: process.env.SOLANA_PRIVATE_KEY,
  solanaRpcUrl: process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com',
};

/**
 * CRTX Agent - Orchestrator for independent analyst agents
 *
 * Coordinates parallel execution of:
 * - ArbitrageAnalyst (CEX/DEX arbitrage)
 * - LPAnalyst (LP pool rebalancing with ML)
 * - MomentumAnalyst (funding rate arbitrage)
 * - SpeculationAnalyst (multi-source sentiment analysis)
 */
// Wallet token balance interface
export interface WalletTokenBalance {
  mint: string;
  symbol: string;
  balance: number;
  decimals: number;
  valueUsd: number;
}

export interface WalletInfo {
  address: string;
  solBalance: number;
  solValueUsd: number;
  tokens: WalletTokenBalance[];
  totalValueUsd: number;
}

// Strategy allocation interface
export interface StrategyAllocation {
  strategy: string;
  percentage: number;
  allocatedUsd: number;
  description: string;
}

export class CRTXAgent {
  private riskManager: RiskManager;
  private portfolioManager: PortfolioManager;
  private config: AgentConfig;
  private modeConfig: ModeConfig;
  private lastEvaluatedOpportunities: EvaluatedOpportunity[] = [];
  private mlModelInitialized = false;
  private perpsServiceInitialized = false;
  private featureCache: Map<string, PredictionResult> = new Map();
  private lpExecutor: LPExecutor | null = null;
  private lendingExecutor: SimpleLendingExecutor | null = null;
  private spotExecutor: SpotExecutor | null = null;
  private wallet: Keypair | null = null;
  private connection: Connection | null = null;
  private lpPositionMonitor: LPPositionMonitor;
  private monitoringIntervalId: NodeJS.Timeout | null = null;

  // Spot position monitoring
  private exitManager: ExitManager;
  private spotPositions: Map<string, SpotPosition> = new Map();
  private spotMonitoringIntervalId: NodeJS.Timeout | null = null;

  // Lending position monitoring
  private lendingPositionMonitor: LendingPositionMonitor;
  private lendingPositions: Map<string, TrackedLendingPosition> = new Map();
  private lendingMonitoringIntervalId: NodeJS.Timeout | null = null;

  // Independent analyst agents (run in parallel)
  private arbitrageAnalyst: ArbitrageAnalyst;
  private momentumAnalyst: MomentumAnalyst;
  private lpAnalyst: LPAnalyst;
  private speculationAnalyst: SpeculationAnalyst;
  private fundamentalAnalyst: FundamentalAnalyst;
  private lendingAnalyst: LendingAnalyst;
  private spotAnalyst: SpotAnalyst;
  private pumpfunAnalyst: PumpFunAnalyst | null = null; // Only in AGGRESSIVE mode
  private newsAnalyst: NewsAnalyst;

  // Consensus system components
  private votingEngine = getVotingEngine();
  private researchDebateManager = getResearchDebateManager();
  private performanceTracker = getPerformanceTracker();
  private bullishResearcher = getBullishResearcher();
  private bearishResearcher = getBearishResearcher();
  private consensusEnabled = true;

  constructor(config: Partial<AgentConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.riskManager = new RiskManager();
    this.portfolioManager = getPortfolioManager({
      initialCapitalUsd: this.config.portfolioValueUsd,
    });

    // Get trading mode configuration
    this.modeConfig = getTradingMode();

    // Initialize ArbitrageAnalyst (independent, runs in parallel)
    this.arbitrageAnalyst = new ArbitrageAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
    });

    // Initialize MomentumAnalyst (independent, runs in parallel)
    this.momentumAnalyst = new MomentumAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
    });

    // Initialize LPAnalyst (independent, runs in parallel)
    this.lpAnalyst = new LPAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
      minTvl: this.modeConfig.minTVL,           // Mode-aware TVL threshold
    });

    // Initialize SpeculationAnalyst (independent, runs in parallel)
    this.speculationAnalyst = new SpeculationAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
    });

    // Initialize FundamentalAnalyst (independent, runs in parallel)
    this.fundamentalAnalyst = new FundamentalAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
      tradingMode: this.modeConfig, // Pass mode config for health score thresholds
    });

    // Initialize LendingAnalyst (independent, runs in parallel)
    this.lendingAnalyst = new LendingAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
    });

    // Initialize SpotAnalyst (independent, runs in parallel)
    this.spotAnalyst = new SpotAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
      minLiquidity: this.modeConfig.minLiquidity, // Mode-aware liquidity threshold
    });

    // Initialize PumpFunAnalyst ONLY in AGGRESSIVE mode
    if (this.modeConfig.enablePumpFun) {
      this.pumpfunAnalyst = new PumpFunAnalyst(this.modeConfig, {
        minConfidence: this.config.minConfidence,
        portfolioValueUsd: this.config.portfolioValueUsd,
        volatility24h: this.config.volatility24h ?? 0.05,
        verbose: true,
      });
      logger.info('[CRTX] PumpFunAnalyst initialized (AGGRESSIVE mode)');
    }

    // Initialize NewsAnalyst (independent, runs in parallel)
    this.newsAnalyst = new NewsAnalyst({
      minConfidence: this.config.minConfidence,
      portfolioValueUsd: this.config.portfolioValueUsd,
      volatility24h: this.config.volatility24h ?? 0.05,
      verbose: true,
    });

    // Initialize LP Position Monitor
    this.lpPositionMonitor = new LPPositionMonitor({
      stopLossPct: 0.05,           // 5% stop loss
      takeProfit1Pct: 0.10,        // 10% take profit
      takeProfit2Pct: 0.20,        // 20% take profit
      maxHoldDaysLosing: 7,        // 7 days max for losing positions
      maxHoldDaysFlat: 14,         // 14 days max for flat positions
      maxImpermanentLossPct: 0.08, // 8% max impermanent loss
      checkIntervalMs: 5 * 60 * 1000, // Check every 5 minutes
    });

    // Initialize Spot Exit Manager
    this.exitManager = new ExitManager();

    // Initialize Lending Position Monitor
    this.lendingPositionMonitor = new LendingPositionMonitor({
      minApyPct: 0.02,             // 2% minimum APY
      apyDropThresholdPct: 0.5,    // 50% APY drop from entry triggers exit
      maxHoldDays: 30,             // 30 days max hold
      minHealthFactor: 1.2,        // Exit if health factor drops below 1.2
      checkIntervalMs: 10 * 60 * 1000, // Check every 10 minutes
    });

    // Initialize wallet if private key provided
    this.initializeWallet();

    // Initialize LP executor
    this.initializeLPExecutor();

    // Initialize lending and spot executors (async)
    this.initializeLendingExecutor().catch(error => {
      logger.error('[CRTX] Failed to initialize lending executor in constructor', { error });
    });
    this.initializeSpotExecutor();

    // Get trading mode and log startup info
    const tradingMode = getTradingMode();

    logger.info('[CRTX] CRTXAgent initialized (Orchestrator)', {
      portfolio: this.config.portfolioValueUsd,
      portfolioState: this.portfolioManager.getSummary(),
      dryRun: this.config.dryRun,
      hasWallet: !!this.wallet,
      hasLPExecutor: !!this.lpExecutor,
      tradingMode: {
        mode: tradingMode.mode,
        minHealthScore: tradingMode.minHealthScore,
        enablePumpFun: tradingMode.enablePumpFun,
        riskMultiplier: tradingMode.riskMultiplier,
        maxPositionSize: tradingMode.maxPositionSize,
        minTVL: tradingMode.minTVL,
        minHolders: tradingMode.minHolders,
        description: tradingMode.description,
      },
      analysts: {
        arbitrage: true,
        momentum: true,
        lp: true,
        speculation: true,
        fundamental: true,
        lending: true,
      },
    });

    // Display mode banner
    if (tradingMode.mode === TradingMode.AGGRESSIVE) {
      console.log('\n╔═══════════════════════════════════════════════════════════╗');
      console.log('║  🚀 AGGRESSIVE TRADING MODE ACTIVE                        ║');
      console.log('╠═══════════════════════════════════════════════════════════╣');
      console.log('║  Min Health Score:  40 (vs 60 in NORMAL mode)            ║');
      console.log('║  Pump.fun:          ENABLED (memecoin trading)           ║');
      console.log('║  Risk Multiplier:   1.5x                                 ║');
      console.log('║  Max Position:      10% (vs 5% in NORMAL)                ║');
      console.log('║  Min TVL:           $10K (vs $100K in NORMAL)            ║');
      console.log('║                                                           ║');
      console.log('║  ⚠️  WARNING: Higher risk of loss with low-health tokens  ║');
      console.log('╚═══════════════════════════════════════════════════════════╝\n');
    } else {
      console.log('\n╔═══════════════════════════════════════════════════════════╗');
      console.log('║  ✅ NORMAL TRADING MODE ACTIVE                            ║');
      console.log('╠═══════════════════════════════════════════════════════════╣');
      console.log('║  Min Health Score:  60 (conservative)                    ║');
      console.log('║  Pump.fun:          DISABLED                             ║');
      console.log('║  Risk Multiplier:   1.0x                                 ║');
      console.log('║  Max Position:      5%                                   ║');
      console.log('║  Min TVL:           $100K                                ║');
      console.log('╚═══════════════════════════════════════════════════════════╝\n');
    }
  }

  /**
   * Create CRTXAgent with interactive mode selection
   * Prompts user to choose NORMAL or AGGRESSIVE mode
   */
  static async createInteractive(config: Partial<AgentConfig> = {}): Promise<CRTXAgent> {
    // Prompt user for trading mode
    const selectedMode = await promptTradingMode();

    // Set environment variable for this session
    process.env.TRADING_MODE = selectedMode;

    // Display mode configuration
    displayModeConfig(selectedMode);

    // Create agent with selected mode
    return new CRTXAgent(config);
  }

  /**
   * Initialize wallet from private key
   */
  private initializeWallet(): void {
    // Initialize Solana connection — use failover unless explicit rpcUrl in config
    if (this.config.solanaRpcUrl) {
      this.connection = new Connection(this.config.solanaRpcUrl, 'confirmed');
      logger.info('[CRTX] Solana connection initialized (config override)', { rpcUrl: this.config.solanaRpcUrl.slice(0, 30) + '...' });
    } else {
      this.connection = getSolanaConnection();
      logger.info('[CRTX] Solana connection initialized (failover)', { rpcUrl: getActiveRpcUrl().slice(0, 30) + '...' });
    }

    if (!this.config.solanaPrivateKey) {
      logger.info('[CRTX] No wallet configured (dry-run mode only)');
      return;
    }

    try {
      // Try bs58 format first
      const secretKey = bs58.decode(this.config.solanaPrivateKey);
      this.wallet = Keypair.fromSecretKey(secretKey);
      logger.info('[CRTX] Wallet initialized', {
        address: this.wallet.publicKey.toBase58().slice(0, 8) + '...',
      });
    } catch (error1) {
      try {
        // Try base64 format
        const secretKey = Uint8Array.from(Buffer.from(this.config.solanaPrivateKey, 'base64'));
        this.wallet = Keypair.fromSecretKey(secretKey);
        logger.info('[CRTX] Wallet initialized (base64)', {
          address: this.wallet.publicKey.toBase58().slice(0, 8) + '...',
        });
      } catch (error2) {
        logger.warn('[CRTX] Invalid wallet private key format', {
          bs58Error: error1 instanceof Error ? error1.message : String(error1),
          base64Error: error2 instanceof Error ? error2.message : String(error2),
          keyLength: this.config.solanaPrivateKey?.length,
          keyPrefix: this.config.solanaPrivateKey?.substring(0, 10)
        });
      }
    }
  }

  /**
   * Initialize LP executor
   */
  private initializeLPExecutor(): void {
    try {
      this.lpExecutor = new LPExecutor({
        rpcUrl: this.config.solanaRpcUrl || 'https://api.mainnet-beta.solana.com',
        maxPriceImpactPct: 1.0,  // Max 1% price impact
        defaultSlippageBps: 300, // 3% slippage
        priorityFeeLamports: 50000,
      });
      logger.info('[CRTX] LP Executor initialized');
    } catch (error) {
      logger.error('[CRTX] Failed to initialize LP executor', { error });
    }
  }

  /**
   * Initialize Lending executor
   */
  private async initializeLendingExecutor(): Promise<void> {
    if (!this.wallet) {
      logger.warn('[CRTX] Cannot initialize lending executor without wallet');
      return;
    }

    try {
      this.lendingExecutor = new SimpleLendingExecutor({
        rpcUrl: this.config.solanaRpcUrl || 'https://api.mainnet-beta.solana.com',
        wallet: this.wallet,
      });
      await this.lendingExecutor.initialize();
      logger.info('[CRTX] Lending Executor initialized');
    } catch (error) {
      logger.error('[CRTX] Failed to initialize lending executor', { error });
    }
  }

  /**
   * Initialize Spot executor
   */
  private initializeSpotExecutor(): void {
    if (!this.wallet) {
      logger.warn('[CRTX] Cannot initialize spot executor without wallet');
      return;
    }

    try {
      this.spotExecutor = new SpotExecutor({
        rpcUrl: this.config.solanaRpcUrl || 'https://api.mainnet-beta.solana.com',
        wallet: this.wallet,
        slippageBps: 100, // 1% slippage
      });
      logger.info('[CRTX] Spot Executor initialized');
    } catch (error) {
      logger.error('[CRTX] Failed to initialize spot executor', { error });
    }
  }

  /**
   * Initialize Perps Service (Drift, Jupiter Perps, Flash, Adrena)
   * This is needed for funding rate scanning and perps execution
   */
  private async initializePerpsService(): Promise<void> {
    if (!this.wallet) {
      logger.warn('[CRTX] Cannot initialize perps service without wallet');
      return;
    }

    try {
      const rpcUrl = this.config.solanaRpcUrl || 'https://api.mainnet-beta.solana.com';

      // Create and initialize perps service
      // NOTE: Only Flash Trade is enabled - Drift/Jupiter/Adrena have SDK/WebSocket issues
      const perpsService = getPerpsService({
        rpcUrl,
        privateKey: this.config.solanaPrivateKey,
        env: 'mainnet-beta',
        useProduction: true,
        enableDrift: true,     // ✅ Using DLOB HTTP API for funding rates
        enableJupiter: false,  // Production client initialization fails
        enableFlash: true,     // ✅ Works - has native SL/TP support
        enableAdrena: false,   // adrena-sdk-ts module not found
        defaultVenue: 'drift', // Use Drift as default (better funding rates)
        enableSOR: false,      // Disabled since only two venues enabled
        useJitoMev: true,
      });

      // Initialize venue clients (this connects funding aggregator)
      const success = await perpsService.initialize();

      if (success) {
        logger.info('[CRTX] PerpsService initialized', {
          venues: perpsService.getEnabledVenues(),
          production: perpsService.isProductionMode(),
        });

        // PerpsScanner uses the same singleton FundingRateAggregator
        // which was already connected by PerpsService.initialize()
        // Just call getPerpsScanner() to ensure it's initialized
        getPerpsScanner();
        logger.info('[CRTX] PerpsScanner connected to venue clients');
      } else {
        logger.warn('[CRTX] PerpsService initialization partial/failed');
      }
    } catch (error) {
      logger.error('[CRTX] Failed to initialize perps service', { error });
    }
  }

  /**
   * Get wallet for execution (or null if dry-run)
   */
  getWallet(): Keypair | null {
    return this.wallet;
  }

  /**
   * Fetch wallet assets dynamically from Solana blockchain
   * Returns SOL balance and all SPL token balances with USD values
   */
  async getWalletInfo(): Promise<WalletInfo | null> {
    if (!this.wallet || !this.connection) {
      logger.warn('[CRTX] Cannot fetch wallet info - no wallet or connection');
      return null;
    }

    try {
      const walletAddress = this.wallet.publicKey;

      // Fetch SOL balance
      const solBalance = await this.connection.getBalance(walletAddress);
      recordRpcSuccess();
      const solAmount = solBalance / LAMPORTS_PER_SOL;

      // Fetch SOL price from multiple sources (fallback chain)
      let solPrice = 0;
      try {
        // Try CoinGecko first (most reliable)
        const cgResp = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd');
        const cgData = await cgResp.json() as any;
        solPrice = cgData?.solana?.usd || 0;
      } catch (e1) {
        try {
          // Fallback to DexScreener
          const dsResp = await fetch('https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112');
          const dsData = await dsResp.json() as any;
          solPrice = parseFloat(dsData?.pairs?.[0]?.priceUsd || '0');
        } catch (e2) {
          logger.warn('[CRTX] Failed to fetch SOL price from all sources');
        }
      }

      const solValueUsd = solAmount * solPrice;

      // Fetch all SPL token accounts
      const tokenAccounts = await this.connection.getTokenAccountsByOwner(
        walletAddress,
        { programId: TOKEN_PROGRAM_ID }
      );

      const tokens: WalletTokenBalance[] = [];
      let totalTokensValueUsd = 0;

      // Parse each token account
      for (const { account } of tokenAccounts.value) {
        try {
          // Parse account data (SPL Token layout)
          const data = account.data;
          const mint = new PublicKey(data.slice(0, 32));
          const amountBuffer = data.slice(64, 72);
          const amount = Number(amountBuffer.readBigUInt64LE(0));

          if (amount === 0) continue; // Skip zero balances

          // Fetch mint info for decimals
          const mintInfo = await this.connection!.getAccountInfo(mint);
          const decimals = mintInfo?.data?.[44] || 9; // Default to 9 decimals

          const tokenBalance = amount / Math.pow(10, decimals);

          // Fetch token price from Jupiter
          let tokenPrice = 0;
          let symbol = mint.toBase58().slice(0, 4) + '...';
          try {
            const priceResp = await fetch(`https://api.jup.ag/price/v2?ids=${mint.toBase58()}`);
            const priceData = await priceResp.json() as any;
            const tokenData = priceData?.data?.[mint.toBase58()];
            tokenPrice = tokenData?.price || 0;
          } catch (e) {
            // Price fetch failed, continue without price
          }

          // Try to get symbol from Jupiter token list
          try {
            const tokenListResp = await fetch(`https://tokens.jup.ag/token/${mint.toBase58()}`);
            if (tokenListResp.ok) {
              const tokenListData = await tokenListResp.json() as any;
              symbol = tokenListData?.symbol || symbol;
            }
          } catch (e) {
            // Symbol fetch failed, use shortened mint
          }

          const tokenValueUsd = tokenBalance * tokenPrice;
          totalTokensValueUsd += tokenValueUsd;

          tokens.push({
            mint: mint.toBase58(),
            symbol,
            balance: tokenBalance,
            decimals,
            valueUsd: tokenValueUsd,
          });
        } catch (e) {
          // Skip tokens we can't parse
          continue;
        }
      }

      // Sort tokens by USD value
      tokens.sort((a, b) => b.valueUsd - a.valueUsd);

      const walletInfo: WalletInfo = {
        address: walletAddress.toBase58(),
        solBalance: solAmount,
        solValueUsd,
        tokens,
        totalValueUsd: solValueUsd + totalTokensValueUsd,
      };

      return walletInfo;
    } catch (error) {
      recordRpcFailure();
      logger.error('[CRTX] Failed to fetch wallet info', { error });
      return null;
    }
  }

  /**
   * Display wallet assets in a formatted box
   */
  async displayWalletAssets(): Promise<void> {
    const walletInfo = await this.getWalletInfo();

    if (!walletInfo) {
      console.log('\n╔═══════════════════════════════════════════════════════════╗');
      console.log('║  ❌ WALLET NOT AVAILABLE                                   ║');
      console.log('╚═══════════════════════════════════════════════════════════╝\n');
      return;
    }

    console.log('\n╔═══════════════════════════════════════════════════════════╗');
    console.log('║  💰 WALLET ASSETS (DYNAMIC)                                ║');
    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log(`║  Address: ${walletInfo.address.slice(0, 8)}...${walletInfo.address.slice(-4)}                                  ║`);
    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log(`║  SOL:     ${walletInfo.solBalance.toFixed(4)} SOL ($${walletInfo.solValueUsd.toFixed(2)})`.padEnd(62) + '║');

    // Show top tokens (max 5)
    const topTokens = walletInfo.tokens.slice(0, 5);
    for (const token of topTokens) {
      const line = `║  ${token.symbol.padEnd(8)}: ${token.balance.toFixed(4)} ($${token.valueUsd.toFixed(2)})`;
      console.log(line.padEnd(62) + '║');
    }

    if (walletInfo.tokens.length > 5) {
      console.log(`║  ... and ${walletInfo.tokens.length - 5} more tokens`.padEnd(62) + '║');
    }

    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log(`║  📊 TOTAL VALUE: $${walletInfo.totalValueUsd.toFixed(2)}`.padEnd(62) + '║');
    console.log('╚═══════════════════════════════════════════════════════════╝\n');
  }

  /**
   * Calculate strategy allocation based on total wallet value
   * Returns how portfolio should be distributed across strategies
   */
  getStrategyAllocation(totalValueUsd: number): StrategyAllocation[] {
    // Dynamic strategy allocation percentages based on trading mode
    const tradingMode = getTradingMode();

    let allocations: { strategy: string; percentage: number; description: string }[];

    if (tradingMode.mode === 'AGGRESSIVE') {
      allocations = [
        { strategy: 'SPOT', percentage: 30, description: 'Token spot trading via Jupiter' },
        { strategy: 'LP', percentage: 25, description: 'Liquidity pool rebalancing' },
        { strategy: 'ARBITRAGE', percentage: 20, description: 'CEX/DEX arbitrage opportunities' },
        { strategy: 'PERPS', percentage: 15, description: 'Perpetual futures trading' },
        { strategy: 'PUMPFUN', percentage: 10, description: 'High-risk meme token trading' },
      ]; // WE DETERMINED DISTRIBUTIONS ABOVE!
    } else {
      // NORMAL mode - more conservative
      allocations = [
        { strategy: 'SPOT', percentage: 35, description: 'Token spot trading via Jupiter' },
        { strategy: 'LP', percentage: 30, description: 'Liquidity pool rebalancing' },
        { strategy: 'ARBITRAGE', percentage: 20, description: 'CEX/DEX arbitrage opportunities' },
        { strategy: 'LENDING', percentage: 10, description: 'DeFi lending protocols' },
        { strategy: 'PERPS', percentage: 5, description: 'Perpetual futures trading' },
      ];
    }

    return allocations.map(a => ({
      ...a,
      allocatedUsd: (totalValueUsd * a.percentage) / 100,
    }));
  }

  /**
   * Display strategy allocation in a formatted box
   */
  async displayStrategyAllocation(): Promise<void> {
    const walletInfo = await this.getWalletInfo();
    const totalValue = walletInfo?.totalValueUsd || this.config.portfolioValueUsd;

    const allocations = this.getStrategyAllocation(totalValue);
    const tradingMode = getTradingMode();

    console.log('\n╔═══════════════════════════════════════════════════════════╗');
    console.log(`║  📊 STRATEGY ALLOCATION (${tradingMode.mode} MODE)`.padEnd(62) + '║');
    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log(`║  Total Portfolio: $${totalValue.toFixed(2)}`.padEnd(62) + '║');
    console.log('╠═══════════════════════════════════════════════════════════╣');

    for (const alloc of allocations) {
      const line = `║  ${alloc.strategy.padEnd(10)}: ${alloc.percentage}% → $${alloc.allocatedUsd.toFixed(2)}`;
      console.log(line.padEnd(62) + '║');
    }

    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log(`║  💡 Allocation based on ${tradingMode.mode} mode settings`.padEnd(62) + '║');
    console.log('╚═══════════════════════════════════════════════════════════╝\n');
  }

  /**
   * Initialize ML model and feature builder
   * Call this before running the agent
   */
  async initializeML(birdeyeApiKey?: string): Promise<void> {
    if (this.mlModelInitialized) return;

    try {
      // Initialize ONNX model
      await lpRebalancerModel.initialize();
      console.log('[AGENT] 🧠 ML model loaded (61 features, 84% accuracy)');

      // Initialize feature builder if API key provided
      if (birdeyeApiKey) {
        featureBuilder.initialize(birdeyeApiKey);
        console.log('[AGENT] 📊 Feature builder initialized with Birdeye API');
      } else {
        console.log('[AGENT] ⚠️ No Birdeye API key - using fallback features');
      }

      this.mlModelInitialized = true;
    } catch (error) {
      console.error('[AGENT] ❌ ML initialization failed:', error);
      // Continue without ML - will use heuristic fallback
    }
  }

  /**
   * Fetch and update current market volatility
   * Uses live price data from market scanner
   */
  private async updateVolatility(snapshot?: MarketSnapshot): Promise<number> {
    // Try to calculate from CEX price history if available
    if (snapshot?.cexPrices && snapshot.cexPrices.length > 0) {
      const solPrices = snapshot.cexPrices
        .filter(p => p.symbol === 'SOL')
        .map(p => p.price);

      if (solPrices.length >= 2) {
        const volatility = calculateVolatilityFromPrices(solPrices);
        this.config.volatility24h = volatility;
        return volatility;
      }
    }

    // Fallback to cached volatility from marketData service
    const volatility = await getVolatility24h();
    this.config.volatility24h = volatility;
    return volatility;
  }

  /**
   * Get current volatility, with sensible default
   */
  private getVolatility(): number {
    return this.config.volatility24h ?? 0.05; // 5% default if not yet fetched
  }

  /**
   * Get last evaluated opportunities for dashboard
   */
  getLastEvaluatedOpportunities(): EvaluatedOpportunity[] {
    return this.lastEvaluatedOpportunities;
  }

  /**
   * Get portfolio summary for dashboard
   */
  getPortfolioSummary(): ReturnType<PortfolioManager['getSummary']> {
    return this.portfolioManager.getSummary();
  }

  /**
   * Get portfolio manager instance for direct access
   */
  getPortfolioManager(): PortfolioManager {
    return this.portfolioManager;
  }

  /**
   * Main entry: Scan → Evaluate → Select → Execute
   *
   * Orchestrates parallel execution of independent analysts:
   * - ArbitrageAnalyst, LPAnalyst, MomentumAnalyst run concurrently
   * - Results aggregated and best opportunity selected
   * - Risk management and execution handled by orchestrator
   */
  async run(): Promise<EvaluatedOpportunity | null> {
    console.log('\n[CRTX] 🤖 Starting trading cycle (Orchestrator)...');

    // 0. Initialize ML model if not already done
    if (!this.mlModelInitialized) {
      await this.initializeML(process.env.BIRDEYE_API_KEY);
    }

    // 0.5. Initialize Perps Service (for funding rate scanning and perps execution)
    if (!this.perpsServiceInitialized) {
      try {
        await this.initializePerpsService();
        this.perpsServiceInitialized = true;
        console.log('[CRTX] ✅ Perps service initialized');
      } catch (error: any) {
        console.warn('[CRTX] ⚠️  Perps init failed, skipping perps strategies:', error.message);
      }
    }

    // 1. Scan markets
    const snapshot = await scanMarkets({ minArbitrageSpread: 0.1 });

    // 1.5. Update volatility from live market data
    const volatility = await this.updateVolatility(snapshot);
    console.log(`[AGENT] 📊 Volatility (24h): ${(volatility * 100).toFixed(2)}%`);

    // Log price summary
    if (snapshot.cexPrices.length > 0) {
      const solCex = snapshot.cexPrices.find(p => p.symbol === 'SOL');
      const solDex = snapshot.dexPrices.find(p => p.symbol === 'SOL');
      if (solCex && solDex) {
        const spread = ((solDex.price - solCex.price) / solCex.price * 100);
        console.log(`[AGENT] SOL: CEX $${solCex.price.toFixed(2)} | DEX $${solDex.price.toFixed(2)} | Spread: ${spread >= 0 ? '+' : ''}${spread.toFixed(3)}%`);
      }
    }

    console.log(`[AGENT] Scanner returned: ${snapshot.arbitrage.length} arbitrage, ${snapshot.lpPools.length} LP pools`);

    // 2. Evaluate opportunities with EARLY EXECUTION
    // NEW: Execute first approved opportunity immediately instead of waiting for all analysts
    const best = await this.evaluateOpportunitiesWithEarlyExecution(snapshot);

    if (!best) {
      console.log('[AGENT] No suitable opportunity found');
      return null;
    }

    // 3. Log portfolio summary after execution
    this.logPortfolioSummary();

    return best;
  }

  /**
   * Log portfolio summary to console
   */
  private logPortfolioSummary(): void {
    const summary = this.portfolioManager.getSummary();
    console.log('\n[AGENT] 💼 Portfolio Summary:');
    console.log(`  Total Value: $${summary.totalValueUsd.toFixed(2)}`);
    console.log(`  Realized PnL: $${summary.realizedPnlUsd.toFixed(2)}`);
    console.log(`  Unrealized PnL: $${summary.unrealizedPnlUsd.toFixed(2)}`);
    console.log(`  Open LP Positions: ${summary.openLpPositions}`);
    console.log(`  Today's PnL: $${summary.dailyPnlUsd.toFixed(2)}`);
    console.log(`  Total Trades: ${summary.totalTrades}`);
  }

  /**
   * Helper: Calculate position size for an opportunity
   * Applies trading mode risk multiplier and max position size
   */
  private calculatePositionSize(opp: EvaluatedOpportunity): number {
    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: opp.confidence,
      currentVolatility24h: this.getVolatility(),
      portfolioValueUsd: this.config.portfolioValueUsd,
    });

    // Apply mode risk multiplier
    let adjustedPositionPct = positionCalc.positionPct * this.modeConfig.riskMultiplier;

    // Cap at mode max position size
    adjustedPositionPct = Math.min(adjustedPositionPct, this.modeConfig.maxPositionSize);

    // Convert back to USD
    const adjustedPositionUsd = (adjustedPositionPct * this.config.portfolioValueUsd);

    return adjustedPositionUsd;
  }
  
  /**
   * Evaluate all opportunities from snapshot with detailed thought logging
   *
   * ORCHESTRATION: Delegates to independent analysts running in parallel:
   * - ArbitrageAnalyst: CEX/DEX arbitrage opportunities
   * - LPAnalyst: LP pool rebalancing with ML (84% accuracy)
   * - MomentumAnalyst: Funding rate arbitrage (delta-neutral)
   * - SpeculationAnalyst: Multi-source sentiment analysis (Twitter, CryptoPanic, Telegram)
   *
   * Each analyst is self-contained and runs independently.
   */
  async evaluateOpportunities(snapshot: MarketSnapshot, verbose = true): Promise<EvaluatedOpportunity[]> {
    const opportunities: EvaluatedOpportunity[] = [];
    const volatility = this.getVolatility();

    const totalOpps = snapshot.arbitrage.length + snapshot.lpPools.length +
                      (snapshot.perpsOpportunities?.length || 0) +
                      (snapshot.fundingArbitrage?.length || 0);

    if (verbose && totalOpps > 0) {
      console.log('\n[CRTX] 🧠 Orchestrating analyst evaluation (parallel execution)...');
    }

    // Evaluate arbitrage opportunities using standalone ArbitrageAnalyst (runs independently)
    if (snapshot.arbitrage.length > 0) {
      // Update analyst config with current volatility
      this.arbitrageAnalyst.updateConfig({ volatility24h: volatility });

      // Call ArbitrageAnalyst.analyze() with market data
      const arbResults = await this.arbitrageAnalyst.analyze({
        opportunities: snapshot.arbitrage,
        volatility24h: volatility,
      });

      // Convert ArbitrageOpportunityResult to EvaluatedOpportunity and log
      for (const arbResult of arbResults) {
        const opp: EvaluatedOpportunity = {
          type: arbResult.type,
          name: arbResult.name,
          expectedReturn: arbResult.expectedReturn,
          riskScore: arbResult.riskScore,
          confidence: arbResult.confidence,
          riskAdjustedReturn: arbResult.riskAdjustedReturn,
          approved: arbResult.approved,
          rejectReason: arbResult.rejectReason,
          warnings: arbResult.warnings,
          raw: arbResult.raw,
          sentimentAdjustment: arbResult.sentimentAdjustment,
        };

        // Log detailed thought process
        if (verbose) {
          const posSize = this.calculatePositionSize(opp);
          logArbitrageEvaluation(arbResult.raw, opp.approved, opp.rejectReason, opp.riskAdjustedReturn, posSize);
        }

        opportunities.push(opp);
      }
    }

    // ===== ML-ENHANCED LP POOL EVALUATION =====
    // Pre-filter pools with scam filters BEFORE fetching ML features (saves API calls)
    const ALLOWED_TOKENS = ['SOL', 'USDC', 'USDT', 'JUP', 'BONK', 'mSOL', 'stSOL', 'jitoSOL', 'RAY', 'ORCA'];
    const MAX_APY = 500;        // Production threshold - filters unrealistic APY pools
    const MIN_TVL = 300_000;    // Lowered to $300k - smaller pools can be legitimate on Solana
    const MIN_VOLUME_TVL = 0.3;

    const eligiblePools = snapshot.lpPools.filter(pool => {
      if (pool.apy > MAX_APY) return false;
      if (pool.tvl < MIN_TVL) return false;
      const volumeTvlRatio = pool.volume24h / pool.tvl;
      if (volumeTvlRatio < MIN_VOLUME_TVL) return false;
      const tokens = pool.name.split('/').map(t => t.trim().toUpperCase());
      if (tokens.some(t => !ALLOWED_TOKENS.includes(t))) return false;
      return true;
    });

    if (verbose && eligiblePools.length < snapshot.lpPools.length) {
      console.log(`[AGENT] 🛡️ Pre-filter: ${snapshot.lpPools.length - eligiblePools.length} pools rejected by scam filters`);
    }

    if (verbose && eligiblePools.length > 0) {
      console.log(`[AGENT] 🎯 ${eligiblePools.length} pools eligible for ML: ${eligiblePools.map(p => p.name).join(', ')}`);
    }

    // Fetch ML features only for eligible pools (saves API calls!)
    let mlPredictions: Map<string, PredictionResult> = new Map();

    if (this.mlModelInitialized && eligiblePools.length > 0) {
      if (verbose) {
        console.log(`[AGENT] 🧠 Fetching ML features for ${eligiblePools.length} eligible pools...`);
      }

      try {
        // Get features only for pools that pass scam filters
        const featuresMap = await featureBuilder.batchGetFeatures(eligiblePools);

        // Run ML inference for each pool
        for (const [address, features] of featuresMap) {
          const prediction = await lpRebalancerModel.predict(features);
          mlPredictions.set(address, prediction);
          this.featureCache.set(address, prediction); // Cache for later use
        }

        if (verbose) {
          console.log(`[AGENT] 🧠 ML predictions: ${mlPredictions.size}/${eligiblePools.length} eligible pools`);
        }
      } catch (error) {
        console.error('[AGENT] ML feature fetch failed, using heuristics:', error);
      }
    } else if (this.mlModelInitialized && eligiblePools.length === 0 && verbose) {
      console.log(`[AGENT] 🧠 No eligible pools for ML evaluation (all rejected by scam filters)`);
    } else if (!this.mlModelInitialized && verbose) {
      console.log(`[AGENT] ⚠️ ML model not initialized - using heuristics`);
    }

    // NEW: Evaluate LP pools with LPAnalyst
    if (verbose) {
      console.log(`\n[AGENT] 💧 Evaluating ${snapshot.lpPools.length} LP pools...`);
    }

    const lpResults = await this.lpAnalyst.analyze({
      pools: snapshot.lpPools,
      volatility24h: volatility,
      portfolioValueUsd: this.config.portfolioValueUsd,
      mlPredictions,
    });

    // Log: ALL approved LP pools + first 3 rejected
    let rejectedCount = 0;
    for (let i = 0; i < lpResults.length; i++) {
      const result = lpResults[i];
      const pool = snapshot.lpPools[i];

      if (verbose) {
        if (result.approved) {
          const riskLabel = result.riskScore <= 3 ? 'Low' : result.riskScore <= 6 ? 'Medium' : 'High';
          logLPEvaluation(pool, result.approved, result.rejectReason, result.riskAdjustedReturn, riskLabel);
        } else if (rejectedCount < 3) {
          const riskLabel = result.riskScore <= 3 ? 'Low' : result.riskScore <= 6 ? 'Medium' : 'High';
          logLPEvaluation(pool, result.approved, result.rejectReason, result.riskAdjustedReturn, riskLabel);
          rejectedCount++;
        }
      }

      opportunities.push(result);
    }

    // NEW: Evaluate perps funding opportunities with sentiment
    if (snapshot.perpsOpportunities && snapshot.perpsOpportunities.length > 0) {
      if (verbose) {
        console.log(`\n[AGENT] 📈 Evaluating ${snapshot.perpsOpportunities.length} perps opportunities...`);
      }

      for (const perp of snapshot.perpsOpportunities) {
        const opp = await this.evaluatePerpsOpportunity(perp, volatility);

        if (verbose && (opp.approved || opportunities.filter(o => o.type === 'perps' && !o.approved).length < 2)) {
          this.logPerpsEvaluation(perp, opp);
        }

        opportunities.push(opp);
      }
    }

    // NEW: Evaluate funding arbitrage opportunities with MomentumAnalyst
    if (snapshot.fundingArbitrage && snapshot.fundingArbitrage.length > 0) {
      if (verbose) {
        console.log(`\n[AGENT] 🔄 Evaluating ${snapshot.fundingArbitrage.length} funding arb opportunities...`);
      }

      const momentumResults = await this.momentumAnalyst.analyze({
        opportunities: snapshot.fundingArbitrage,
        volatility24h: volatility,
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      for (const result of momentumResults) {
        if (verbose && result.approved) {
          this.logFundingArbEvaluation(result.raw, result);
        }

        opportunities.push(result);
      }
    }

    // NEW: Evaluate sentiment-based opportunities with SpeculationAnalyst
    // Extract unique tokens from all opportunities
    const tokensSet = new Set<string>();
    snapshot.arbitrage.forEach(arb => tokensSet.add(arb.symbol));
    snapshot.lpPools.forEach(pool => {
      const tokens = pool.name.split('/').map(t => t.trim().toUpperCase());
      tokens.forEach(t => tokensSet.add(t));
    });
    snapshot.perpsOpportunities?.forEach(perp => {
      const baseAsset = perp.market.split('-')[0];
      tokensSet.add(baseAsset);
    });

    const tokens = Array.from(tokensSet);

    if (tokens.length > 0) {
      if (verbose) {
        console.log(`\n[AGENT] 💭 Evaluating sentiment for ${tokens.length} tokens: ${tokens.join(', ')}...`);
      }

      const speculationResults = await this.speculationAnalyst.analyze({
        tokens,
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      for (const result of speculationResults) {
        if (verbose && (result.approved || opportunities.filter(o => o.type === 'speculation' && !o.approved).length < 3)) {
          this.logSpeculationEvaluation(result);
        }

        opportunities.push(result);
      }

      // NEW: Evaluate news impact with NewsAnalyst
      if (verbose) {
        console.log(`\n[AGENT] 📰 Evaluating news for ${tokens.length} tokens: ${tokens.join(', ')}...`);
      }

      const newsResults = await this.newsAnalyst.analyze({
        assets: tokens,
        lookbackHours: 24,
      });

      for (const result of newsResults) {
        if (verbose && (result.approved || opportunities.filter(o => o.type === 'news' && !o.approved).length < 3)) {
          this.logNewsEvaluation(result);
        }

        opportunities.push(result);
      }

      // NEW: Evaluate token fundamentals with FundamentalAnalyst
      if (verbose) {
        console.log(`\n[AGENT] 🔬 Evaluating fundamentals for ${tokens.length} tokens: ${tokens.join(', ')}...`);
      }

      const fundamentalResults = await this.fundamentalAnalyst.analyze({
        tokens,
        pools: snapshot.lpPools.map(p => ({ name: p.name, tvl: p.tvl })),
      });

      for (const result of fundamentalResults) {
        if (verbose && (result.approved || opportunities.filter(o => o.type === 'fundamental' && !o.approved).length < 3)) {
          this.logFundamentalEvaluation(result);
        }

        opportunities.push(result);
      }
    }

    // ============= LENDING EVALUATION =============
    // Evaluate lending opportunities using LendingAnalyst
    if (snapshot.lendingMarkets && snapshot.lendingMarkets.length > 0) {
      // Initialize lending analyst ML model
      await this.lendingAnalyst.initialize();

      // Update analyst config with current volatility
      this.lendingAnalyst.updateConfig({ volatility24h: volatility });

      // Call LendingAnalyst.analyze() with market data
      const lendingResults = await this.lendingAnalyst.analyze({
        markets: snapshot.lendingMarkets,
        volatility24h: volatility,
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      // Convert to EvaluatedOpportunity
      for (const lendingResult of lendingResults) {
        const opp: EvaluatedOpportunity = {
          type: lendingResult.type,
          name: lendingResult.name,
          expectedReturn: lendingResult.expectedReturn,
          riskScore: lendingResult.riskScore,
          confidence: lendingResult.confidence,
          riskAdjustedReturn: lendingResult.riskAdjustedReturn,
          approved: lendingResult.approved,
          rejectReason: lendingResult.rejectReason,
          warnings: lendingResult.warnings,
          raw: lendingResult.raw,
        };

        // Log if approved or if we have few rejected ones
        if (verbose && (opp.approved || opportunities.filter(o => o.type === 'lending' && !o.approved).length < 3)) {
          console.log(`\n[LendingAnalyst] ${opp.approved ? '✅' : '❌'} ${opp.name}`);
          console.log(`  Expected Return: ${opp.expectedReturn.toFixed(2)}%`);
          console.log(`  Risk Score: ${opp.riskScore}/10`);
          console.log(`  Confidence: ${(opp.confidence * 100).toFixed(1)}%`);
          console.log(`  Risk-Adjusted Return: ${opp.riskAdjustedReturn.toFixed(2)}`);
          if (!opp.approved && opp.rejectReason) {
            console.log(`  Reject Reason: ${opp.rejectReason}`);
          }
          if (opp.warnings.length > 0) {
            console.log(`  Warnings: ${opp.warnings.join(', ')}`);
          }
        }

        opportunities.push(opp);
      }
    }

    // ============= SPOT TRADING EVALUATION =============
    // Evaluate spot trading opportunities using SpotAnalyst
    if (snapshot.spotTokens && snapshot.spotTokens.length > 0) {
      // Initialize spot analyst ML model
      await this.spotAnalyst.initialize();

      // Update analyst config with current volatility
      this.spotAnalyst.updateConfig({ volatility24h: volatility });

      // Call SpotAnalyst.analyze() with market data
      const spotResults = await this.spotAnalyst.analyze({
        tokens: snapshot.spotTokens,
        marketData: snapshot.spotTokens.reduce((map, token) => {
          if (token.marketData) {
            map.set(token.address, token.marketData);
          }
          return map;
        }, new Map()),
        volatility24h: volatility,
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      // Get list of tokens we already have positions in
      const openPositions = this.getOpenSpotPositions();
      const tokensWithPositions = new Set(openPositions.map(p => p.token.address));

      // Convert to EvaluatedOpportunity
      for (const spotResult of spotResults) {
        // SKIP if we already have a position in this token
        if (spotResult.approved && tokensWithPositions.has(spotResult.token.address)) {
          console.log(`\n[SpotAnalyst] ⏭️ ${spotResult.name} - SKIPPED (already have position)`);
          continue;
        }

        // Check max positions limit
        if (spotResult.approved && openPositions.length >= 4) {
          console.log(`\n[SpotAnalyst] ⏭️ ${spotResult.name} - SKIPPED (max 4 positions reached)`);
          continue;
        }

        // IMPORTANT: Include token info in raw object for executeSpot to use
        const opp: EvaluatedOpportunity = {
          type: spotResult.type,
          name: spotResult.name,
          expectedReturn: spotResult.expectedReturn,
          riskScore: spotResult.riskScore,
          confidence: spotResult.confidence,
          riskAdjustedReturn: spotResult.riskAdjustedReturn,
          approved: spotResult.approved,
          rejectReason: spotResult.rejectReason,
          warnings: spotResult.warnings,
          raw: {
            ...spotResult.raw,
            token: spotResult.token,  // Include token info for execution
            trading: spotResult.trading,  // Include trading params (entry, sl, tp)
          },
        };

        // Log if approved or if we have few rejected ones
        if (verbose && (opp.approved || opportunities.filter(o => o.type === 'spot' && !o.approved).length < 3)) {
          console.log(`\n[SpotAnalyst] ${opp.approved ? '✅' : '❌'} ${opp.name}`);
          console.log(`  Expected Return: ${opp.expectedReturn.toFixed(2)}%`);
          console.log(`  Risk Score: ${opp.riskScore}/10`);
          console.log(`  Confidence: ${(Number(opp.confidence) * 100).toFixed(1)}%`);
          console.log(`  Risk-Adjusted Return: ${opp.riskAdjustedReturn.toFixed(2)}`);
          if (!opp.approved && opp.rejectReason) {
            console.log(`  Reject Reason: ${opp.rejectReason}`);
          }
          if (opp.warnings.length > 0) {
            console.log(`  Warnings: ${opp.warnings.join(', ')}`);
          }
        }

        opportunities.push(opp);
      }
    }

    // ============= PUMP.FUN ANALYST (AGGRESSIVE MODE ONLY) =============
    if (this.pumpfunAnalyst) {
      if (verbose) {
        console.log('\n[CRTX] 🚀 Evaluating Pump.fun memecoins (AGGRESSIVE mode)...');
      }

      try {
        // Call PumpFunAnalyst.analyze()
        const pumpfunResults = await this.pumpfunAnalyst.analyze({ limit: 50, offset: 0 });

        // Convert to EvaluatedOpportunity
        for (const pumpfunResult of pumpfunResults) {
          const opp: EvaluatedOpportunity = {
            type: 'pumpfun',
            name: pumpfunResult.name,
            expectedReturn: pumpfunResult.expectedReturn,
            riskScore: pumpfunResult.riskScore,
            confidence: pumpfunResult.confidence,
            riskAdjustedReturn: pumpfunResult.riskAdjustedReturn,
            approved: pumpfunResult.approved,
            rejectReason: pumpfunResult.rejectReason,
            warnings: pumpfunResult.warnings,
            raw: pumpfunResult.raw,
            // Include PumpFun-specific execution details
            pumpfunDetails: {
              tokenMint: pumpfunResult.token.mint,
              amountSol: pumpfunResult.trading.amountSol,
              slippageBps: pumpfunResult.trading.slippageBps,
              riskCheck: pumpfunResult.riskCheck,
              execute: pumpfunResult.execute,
            },
          };

          // Log if approved
          if (verbose && opp.approved) {
            console.log(`\n[PumpFunAnalyst] ✅ ${opp.name}`);
            console.log(`  Expected Return: ${opp.expectedReturn.toFixed(2)}%`);
            console.log(`  Risk Score: ${opp.riskScore}/10`);
            console.log(`  Confidence: ${(Number(opp.confidence) * 100).toFixed(1)}%`);
            console.log(`  Risk-Adjusted Return: ${opp.riskAdjustedReturn.toFixed(2)}`);
            console.log(`  Amount: ${pumpfunResult.trading.amountSol.toFixed(3)} SOL`);
            console.log(`  Risk Check: ${pumpfunResult.riskCheck.riskScore}/100 (${pumpfunResult.riskCheck.isRugPull ? '⚠️ RUG RISK' : '✅ SAFE'})`);
            if (opp.warnings.length > 0) {
              console.log(`  Warnings: ${opp.warnings.join(', ')}`);
            }
          }

          opportunities.push(opp);
        }

        if (verbose) {
          const approvedCount = pumpfunResults.filter(r => r.approved).length;
          console.log(`\n[PumpFunAnalyst] ${approvedCount}/${pumpfunResults.length} Pump.fun tokens approved`);
        }
      } catch (error: any) {
        logger.error('[PumpFunAnalyst] Analysis failed', { error: error.message });
      }
    }

    // ============= FUNDAMENTAL HEALTH CROSS-FILTERING =============
    // Filter all opportunities (arbitrage, LP, perps, momentum, speculation) by fundamental health
    // Fundamental opportunities are already self-filtered (approved field)

    const fundamentalOpps = opportunities.filter(o => o.type === 'fundamental');
    const otherOpps = opportunities.filter(o => o.type !== 'fundamental');

    if (fundamentalOpps.length > 0 && otherOpps.length > 0) {
      logger.info('[CRTXAgent] Cross-filtering opportunities by fundamental health', {
        fundamentalCount: fundamentalOpps.length,
        otherOppsCount: otherOpps.length,
      });
    }

    const healthyOpps = otherOpps.filter(opp => {
      // Extract token symbols from opportunity
      const tokens = this.extractTokensFromOpportunity(opp);

      if (tokens.length === 0) {
        // No tokens extracted - allow (might be a non-token opportunity)
        return true;
      }

      // Check if any token has poor fundamental health
      for (const token of tokens) {
        const fundamental = fundamentalOpps.find(f =>
          f.name.includes(token) || f.raw?.token === token
        );

        if (fundamental && !fundamental.approved) {
          logger.info('[CRTXAgent] Filtered opportunity due to poor fundamental health', {
            type: opp.type,
            name: opp.name,
            token,
            healthScore: fundamental.raw?.healthScore || 0,
            rating: fundamental.raw?.rating || 'UNKNOWN',
            reason: 'Token failed fundamental health check',
          });

          // Update opportunity to mark as rejected
          opp.approved = false;
          opp.rejectReason = `Token ${token} failed fundamental health check (score: ${fundamental.raw?.healthScore || 0}/100)`;
          opp.warnings.push(`Unhealthy token: ${token} (${fundamental.raw?.rating || 'POOR'})`);

          return false;
        }
      }

      return true;
    });

    const filteredCount = otherOpps.length - healthyOpps.length;

    if (filteredCount > 0) {
      logger.info('[CRTXAgent] Fundamental cross-filtering complete', {
        totalOpps: otherOpps.length,
        healthyOpps: healthyOpps.length,
        filteredOut: filteredCount,
      });

      console.log(`\n╔═══════════════════════════════════════════════════════════╗`);
      console.log(`║        🏥 FUNDAMENTAL HEALTH CROSS-FILTERING             ║`);
      console.log(`╠═══════════════════════════════════════════════════════════╣`);
      console.log(`║  Total Opportunities:        ${otherOpps.length.toString().padEnd(28)}║`);
      console.log(`║  Healthy (Passed):           ${healthyOpps.length.toString().padEnd(28)}║`);
      console.log(`║  ⚠️  Filtered (Unhealthy):    ${filteredCount.toString().padEnd(28)}║`);
      console.log(`╚═══════════════════════════════════════════════════════════╝\n`);
    }

    // Combine healthy opportunities with fundamental opportunities
    const finalOpportunities = [...healthyOpps, ...fundamentalOpps];
    // ==================================================================

    // ============= SENTIMENT IMPACT SUMMARY =============
    const withSentiment = finalOpportunities.filter(o => o.sentimentAdjustment?.sentimentAvailable);
    const sentimentPrevented = finalOpportunities.filter(o =>
      o.sentimentAdjustment?.sentimentAvailable &&
      o.sentimentAdjustment.mlConfidence >= this.config.minConfidence &&
      o.sentimentAdjustment.finalScore < this.config.minConfidence
    );
    const sentimentEnabled = finalOpportunities.filter(o =>
      o.sentimentAdjustment?.sentimentAvailable &&
      o.sentimentAdjustment.mlConfidence < this.config.minConfidence &&
      o.sentimentAdjustment.finalScore >= this.config.minConfidence
    );

    console.log(`\n╔═══════════════════════════════════════════════════════════╗`);
    console.log(`║           📊 SENTIMENT INTEGRATION SUMMARY                ║`);
    console.log(`╠═══════════════════════════════════════════════════════════╣`);
    console.log(`║  Total Opportunities:        ${finalOpportunities.length.toString().padEnd(28)}║`);
    console.log(`║  With Sentiment Data:        ${withSentiment.length.toString().padEnd(28)}║`);
    console.log(`║  Without Sentiment (ML-only): ${(finalOpportunities.length - withSentiment.length).toString().padEnd(27)}║`);
    console.log(`╠═══════════════════════════════════════════════════════════╣`);
    console.log(`║  ⚠️  Trades Prevented by Sentiment: ${sentimentPrevented.length.toString().padEnd(21)}║`);
    console.log(`║  ✨ Trades Enabled by Sentiment:   ${sentimentEnabled.length.toString().padEnd(21)}║`);
    console.log(`╠═══════════════════════════════════════════════════════════╣`);
    if (withSentiment.length > 0) {
      const avgDelta = withSentiment.reduce((sum, o) =>
        sum + (o.sentimentAdjustment!.finalScore - o.sentimentAdjustment!.mlConfidence), 0
      ) / withSentiment.length;
      console.log(`║  Avg Confidence Delta:       ${(avgDelta >= 0 ? '+' : '') + (avgDelta * 100).toFixed(2) + '%'.padEnd(25)}║`);
    }
    console.log(`╚═══════════════════════════════════════════════════════════╝\n`);
    // ==================================================================

    return finalOpportunities;
  }

  /**
   * NEW: Evaluate opportunities with EARLY EXECUTION
   *
   * Instead of waiting for ALL analysts to finish, this function:
   * 1. Evaluates analysts in priority order (LP > Arbitrage > Perps > Lending > Spot)
   * 2. Executes the FIRST approved opportunity immediately
   * 3. Skips remaining analysts to save time and API calls
   *
   * This is much faster when free API keys are rate-limited.
   */
  private async evaluateOpportunitiesWithEarlyExecution(snapshot: MarketSnapshot): Promise<EvaluatedOpportunity | null> {
    const volatility = this.getVolatility();
    console.log('\n[CRTX] 🧠 Evaluating opportunities with EARLY EXECUTION mode...');

    // ============================================================
    // PRIORITY 1: SPOT TRADING (Jupiter swap)
    // ============================================================
    if (snapshot.spotTokens && snapshot.spotTokens.length > 0) {
      console.log(`\n[AGENT] 💱 PRIORITY 1: Evaluating ${snapshot.spotTokens.length} spot trading opportunities...`);

      await this.spotAnalyst.initialize();
      this.spotAnalyst.updateConfig({ volatility24h: volatility });

      const spotResults = await this.spotAnalyst.analyze({
        tokens: snapshot.spotTokens,
        marketData: snapshot.spotTokens.reduce((map, token) => {
          if (token.marketData) {
            map.set(token.address, token.marketData);
          }
          return map;
        }, new Map()),
        portfolioValueUsd: this.config.portfolioValueUsd,
        volatility24h: volatility,
      });

      // Get list of tokens we already have positions in
      const openPositions = this.getOpenSpotPositions();
      const tokensWithPositions = new Set(openPositions.map(p => p.token.address));

      for (const spotResult of spotResults) {
        if (spotResult.approved) {
          // SKIP if we already have a position in this token
          if (tokensWithPositions.has(spotResult.token.address)) {
            console.log(`\n[SpotAnalyst] ⏭️ ${spotResult.name} - SKIPPED (already have position)`);
            continue;
          }

          // Check max positions limit
          if (openPositions.length >= 4) {
            console.log(`\n[SpotAnalyst] ⏭️ ${spotResult.name} - SKIPPED (max 4 positions reached)`);
            continue;
          }

          console.log(`\n[SpotAnalyst] ✅ ${spotResult.name}`);
          console.log(`  Expected Return: ${spotResult.expectedReturn.toFixed(2)}%`);
          console.log(`  Risk Score: ${spotResult.riskScore}/10`);
          console.log(`  Confidence: ${(spotResult.confidence * 100).toFixed(1)}%`);

          // Convert to EvaluatedOpportunity format
          const opp: EvaluatedOpportunity = {
            type: 'spot',
            name: spotResult.name,
            approved: true,
            confidence: spotResult.confidence,
            expectedReturn: spotResult.expectedReturn,
            riskScore: spotResult.riskScore,
            riskAdjustedReturn: spotResult.riskAdjustedReturn,
            raw: {
              ...spotResult.raw,
              token: spotResult.token,
              trading: spotResult.trading,
            },
            warnings: [],
          };

          console.log(`\n[CRTX] ✅ Opportunity found: SPOT ${spotResult.name}`);
          console.log(`  💭 "First approved opportunity - executing immediately!"`);
          console.log(`  🚀 Executing via Jupiter swap...`);

          await this.execute(opp);
          return opp;
        }
      }

      console.log(`[AGENT] No approved spot trading opportunities found`);
    }

    // ============================================================
    // PRIORITY 2: LENDING
    // ============================================================
    if (snapshot.lendingMarkets && snapshot.lendingMarkets.length > 0) {
      console.log(`\n[AGENT] 🏦 PRIORITY 1: Evaluating ${snapshot.lendingMarkets.length} lending opportunities...`);

      await this.lendingAnalyst.initialize();
      this.lendingAnalyst.updateConfig({ volatility24h: volatility });

      const lendingResults = await this.lendingAnalyst.analyze({
        markets: snapshot.lendingMarkets,
        volatility24h: volatility,
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      for (const lendingResult of lendingResults) {
        if (lendingResult.approved) {
          console.log(`\n[LendingAnalyst] ✅ ${lendingResult.name}`);
          console.log(`  Expected Return: ${lendingResult.expectedReturn.toFixed(2)}%`);
          console.log(`  Risk Score: ${lendingResult.riskScore}/10`);
          console.log(`  Confidence: ${(lendingResult.confidence * 100).toFixed(1)}%`);

          const opp: EvaluatedOpportunity = {
            type: 'lending',
            name: lendingResult.name,
            approved: true,
            confidence: lendingResult.confidence,
            expectedReturn: lendingResult.expectedReturn,
            riskScore: lendingResult.riskScore,
            riskAdjustedReturn: lendingResult.riskAdjustedReturn,
            raw: lendingResult.raw,
            warnings: lendingResult.warnings,
          };

          console.log(`\n[CRTX] ✅ Opportunity found: LENDING ${lendingResult.name}`);
          console.log(`  💭 "First approved opportunity - executing immediately!"`);
          console.log(`  🚀 Executing...`);

          await this.execute(opp);
          return opp;
        }
      }

      console.log(`[AGENT] No approved lending opportunities found`);
    }

    // ============================================================
    // PRIORITY 3: LP POOLS
    // ============================================================
    if (snapshot.lpPools.length > 0) {
      console.log(`\n[AGENT] 💧 Evaluating ${snapshot.lpPools.length} LP pools...`);

      // Pre-filter pools with scam filters
      const ALLOWED_TOKENS = ['SOL', 'USDC', 'USDT', 'JUP', 'BONK', 'mSOL', 'stSOL', 'jitoSOL', 'RAY', 'ORCA'];
      const MAX_APY = 500;
      const MIN_TVL = 100_000;
      const MIN_VOLUME_TVL = 0.3;

      const eligiblePools = snapshot.lpPools.filter(pool => {
        if (pool.apy > MAX_APY) return false;
        if (pool.tvl < MIN_TVL) return false;
        const volumeTvlRatio = pool.volume24h / pool.tvl;
        if (volumeTvlRatio < MIN_VOLUME_TVL) return false;
        const tokens = pool.name.split('/').map(t => t.trim().toUpperCase());
        if (tokens.some(t => !ALLOWED_TOKENS.includes(t))) return false;

        // Reject Raydium pools (CLMM pools too complex - tick ranges, position NFTs, CreateAccountWithSeed errors)
        const dex = pool.dex.toLowerCase();
        if (dex.includes('raydium') || dex.includes('ray')) {
          console.log(`[AGENT] ❌ Rejecting ${pool.dex} pool ${pool.name} (CLMM too complex)`);
          return false;
        }

        // Reject Orca pools from DexScreener (address mismatch - DexScreener returns pair addresses, not Whirlpool addresses)
        // TODO: Re-enable when we can fetch actual Whirlpool addresses from Orca API
        if (dex.includes('orca') && pool.address) {
          console.log(`[AGENT] ❌ Rejecting ${pool.dex} pool ${pool.name} (DexScreener address incompatible with Orca SDK)`);
          return false;
        }

        console.log(`[AGENT] ✅ Accepting ${pool.dex} pool ${pool.name} for evaluation`);
        return true;
      });

      if (eligiblePools.length > 0) {
        console.log(`[AGENT] 🎯 ${eligiblePools.length} pools eligible for ML`);

        // Fetch ML features for eligible pools
        let mlPredictions: Map<string, PredictionResult> = new Map();
        if (this.mlModelInitialized) {
          try {
            const featuresMap = await featureBuilder.batchGetFeatures(eligiblePools);
            for (const [address, features] of featuresMap) {
              const prediction = await lpRebalancerModel.predict(features);
              mlPredictions.set(address, prediction);
            }
            console.log(`[AGENT] 🧠 ML predictions: ${mlPredictions.size}/${eligiblePools.length}`);
          } catch (error) {
            console.error('[AGENT] ML feature fetch failed:', error);
          }
        }

        // Evaluate LP pools - use eligiblePools that passed pre-filter
        const lpResults = await this.lpAnalyst.analyze({
          pools: eligiblePools,
          volatility24h: volatility,
          portfolioValueUsd: this.config.portfolioValueUsd,
          mlPredictions,
        });

        // Check for approved LP opportunities
        for (let i = 0; i < lpResults.length; i++) {
          const result = lpResults[i];
          const pool = eligiblePools[i];  // Use eligiblePools to match index

          if (result.approved) {
            const riskLabel = result.riskScore <= 3 ? 'Low' : result.riskScore <= 6 ? 'Medium' : 'High';
            logLPEvaluation(pool, result.approved, result.rejectReason, result.riskAdjustedReturn, riskLabel);

            // Check if it meets minimum thresholds
            if (result.expectedReturn >= 5) {
              console.log(`\n[CRTX] ✅ Opportunity found: ${result.type.toUpperCase()} ${result.name}`);
              console.log(`  💭 "First approved opportunity - executing immediately!"`);
              console.log(`  📊 APY: ${result.expectedReturn.toFixed(2)}% | Risk: ${riskLabel}`);
              console.log(`  🚀 Executing...`);

              // Execute immediately
              await this.execute(result);
              return result;
            }
          }
        }

        console.log(`[AGENT] No approved LP pools found (checked ${lpResults.length} eligible pools)`);
      } else {
        console.log(`[AGENT] No eligible LP pools (all filtered by scam filters)`);
      }
    }

    // PRIORITY 2: ARBITRAGE (fast execution, low risk)
    if (snapshot.arbitrage.length > 0) {
      console.log(`\n[AGENT] 🔄 Evaluating ${snapshot.arbitrage.length} arbitrage opportunities...`);

      this.arbitrageAnalyst.updateConfig({ volatility24h: volatility });
      const arbResults = await this.arbitrageAnalyst.analyze({
        opportunities: snapshot.arbitrage,
        volatility24h: volatility,
      });

      for (const arbResult of arbResults) {
        if (arbResult.approved) {
          const posSize = this.calculatePositionSize(arbResult);
          logArbitrageEvaluation(arbResult.raw, arbResult.approved, arbResult.rejectReason, arbResult.riskAdjustedReturn, posSize);

          console.log(`\n[CRTX] ✅ Opportunity found: ${arbResult.type.toUpperCase()} ${arbResult.name}`);
          console.log(`  💭 "First approved opportunity - executing immediately!"`);
          console.log(`  🚀 Executing...`);

          await this.execute(arbResult);
          return arbResult;
        }
      }

      console.log(`[AGENT] No approved arbitrage opportunities found`);
    }

    console.log(`\n[AGENT] ❌ No approved opportunities found across all analysts`);
    return null;
  }

  // NOTE: evaluateArbitrage() has been extracted to ArbitrageAnalyst
  // See: agent/eliza/src/agents/analysts/ArbitrageAnalyst.ts

  // NOTE: evaluateLPPool() has been extracted to LPAnalyst
  // See: agent/eliza/src/agents/analysts/LPAnalyst.ts

  // NOTE: evaluateLPPoolWithML() and rejectPool() have been extracted to LPAnalyst
  // See: agent/eliza/src/agents/analysts/LPAnalyst.ts

  // ============= PERPS EVALUATION (NEW) =============

  /**
   * Evaluate single perps funding opportunity with sentiment integration
   * Uses funding rate + OI + liquidity + sentiment (25% weight) for decision
   */
  private async evaluatePerpsOpportunity(perp: PerpsOpportunity, volatility: number): Promise<EvaluatedOpportunity> {
    const name = `${perp.side.toUpperCase()} ${perp.market} [${perp.venue}]`;
    const defaultLeverage = 3; // Conservative leverage

    // Confidence based on funding rate strength and market conditions
    const fundingBps = Math.abs(perp.fundingRate) * 10000;
    let baseConfidence = 0.5;

    // Higher funding = higher confidence
    if (fundingBps >= 20) baseConfidence += 0.2;      // >0.2% hourly
    else if (fundingBps >= 10) baseConfidence += 0.15; // >0.1% hourly
    else if (fundingBps >= 5) baseConfidence += 0.1;   // >0.05% hourly

    // High OI = more reliable signal
    if (perp.openInterest > 50_000_000) baseConfidence += 0.1;
    else if (perp.openInterest > 10_000_000) baseConfidence += 0.05;

    // Low OI change = less crowded
    if (Math.abs(perp.oiChangePct24h) < 20) baseConfidence += 0.05;

    // Cap base confidence at 0.95
    baseConfidence = Math.min(0.95, baseConfidence);

    // ============= SENTIMENT INTEGRATION (25% weight for perps) =============
    // Extract token from market (e.g., "SOL-PERP" -> "SOL")
    const token = perp.market.split('-')[0] || 'SOL';
    const sentimentIntegration = getSentimentIntegration();
    const sentimentAdjustment = await sentimentIntegration.getAdjustedScore(
      token,
      baseConfidence,
      'perps'
    );

    // Use sentiment-adjusted confidence
    const confidence = sentimentAdjustment.finalScore;
    const warnings: string[] = [];

    if (sentimentAdjustment.sentimentAvailable) {
      warnings.push(`Sentiment: ${sentimentAdjustment.signal} (${(sentimentAdjustment.rawSentiment * 100).toFixed(0)}%)`);
    }

    // ============= DETAILED SENTIMENT LOGGING =============
    const mlPassesThreshold = baseConfidence >= this.config.minConfidence;
    const finalPassesThreshold = confidence >= this.config.minConfidence;
    const delta = confidence - baseConfidence;

    console.log(`\n┌─────────────────────────────────────────────────────────┐`);
    console.log(`│  🎯 PERPS OPPORTUNITY EVALUATION WITH SENTIMENT         │`);
    console.log(`├─────────────────────────────────────────────────────────┤`);
    console.log(`│  Symbol:              ${token.padEnd(35)}│`);
    console.log(`│  Market:              ${perp.market.padEnd(35)}│`);
    console.log(`│  Side:                ${perp.side.toUpperCase().padEnd(35)}│`);
    console.log(`├─────────────────────────────────────────────────────────┤`);
    console.log(`│  ML Confidence:       ${baseConfidence.toFixed(3).padEnd(35)}│`);
    if (sentimentAdjustment.sentimentAvailable) {
      console.log(`│  Sentiment Score:     ${sentimentAdjustment.rawSentiment.toFixed(3)} (${sentimentAdjustment.signal})`.padEnd(60) + `│`);
      console.log(`│  Normalized:          ${sentimentAdjustment.normalizedSentiment.toFixed(3).padEnd(35)}│`);
      console.log(`│  Sentiment Weight:    ${'25%'.padEnd(35)}│`);
      console.log(`│  Final Confidence:    ${confidence.toFixed(3).padEnd(35)}│`);
      console.log(`│  Delta:               ${(delta >= 0 ? '+' : '') + delta.toFixed(3)}`.padEnd(60) + `│`);
    } else {
      console.log(`│  Sentiment:           ${'unavailable (using ML only)'.padEnd(35)}│`);
    }
    console.log(`├─────────────────────────────────────────────────────────┤`);
    console.log(`│  Threshold:           ${this.config.minConfidence.toFixed(3).padEnd(35)}│`);
    console.log(`│  Decision:            ${(finalPassesThreshold ? 'TRADE ✅' : 'SKIP ❌').padEnd(35)}│`);
    if (mlPassesThreshold && !finalPassesThreshold) {
      console.log(`│  ⚠️  SENTIMENT PREVENTED TRADE                          │`);
    }
    if (!mlPassesThreshold && finalPassesThreshold) {
      console.log(`│  ✨ SENTIMENT ENABLED TRADE                             │`);
    }
    console.log(`└─────────────────────────────────────────────────────────┘`);
    // ==================================================================

    // Risk check
    const riskCheck = this.riskManager.checkTradeAllowed({
      proposedPositionPct: 5, // 5% for perps
      currentVolatility24h: volatility,
      expectedReturnPct: perp.expectedReturnPct,
    });

    // Calculate risk-adjusted return
    const riskAdjustedReturn = perp.expectedReturnPct * (1 - perp.riskScore / 20);

    let rejectReason: string | undefined;
    if (!riskCheck.allowed) {
      rejectReason = riskCheck.reason;
    } else if (fundingBps < 3) {
      rejectReason = `Funding too low: ${fundingBps.toFixed(1)}bps (min 3bps)`;
    } else if (perp.openInterest < 500_000) {
      rejectReason = `OI too low: $${(perp.openInterest / 1_000).toFixed(0)}K (min $500K)`;
    } else if (confidence < this.config.minConfidence) {
      rejectReason = `Confidence too low: ${(confidence * 100).toFixed(0)}% (min ${(this.config.minConfidence * 100).toFixed(0)}%)`;
    }

    const approved = !rejectReason && riskCheck.allowed && confidence >= this.config.minConfidence;

    return {
      type: 'perps',
      name,
      expectedReturn: perp.expectedReturnPct,
      riskScore: perp.riskScore,
      confidence,
      riskAdjustedReturn,
      approved,
      rejectReason,
      warnings: [...warnings, ...riskCheck.warnings],
      raw: perp,
      sentimentAdjustment,
      perpsDetails: {
        venue: perp.venue as PerpsVenue,
        market: perp.market,
        side: perp.side as PositionSide,
        leverage: defaultLeverage,
        fundingRate: perp.fundingRate,
      },
    };
  }

  // NOTE: evaluateFundingArbitrage() has been extracted to MomentumAnalyst
  // See: agent/eliza/src/agents/analysts/MomentumAnalyst.ts

  /**
   * Log perps evaluation thought process
   */
  private logPerpsEvaluation(perp: PerpsOpportunity, opp: EvaluatedOpportunity): void {
    const emoji = opp.approved ? '✅' : '❌';
    const fundingPct = (perp.fundingRate * 100).toFixed(4);

    console.log(`\n${emoji} PERPS: ${opp.name}`);
    console.log(`   💰 Funding: ${fundingPct}% hourly (${perp.annualizedRate.toFixed(1)}% APR)`);
    console.log(`   📊 OI: $${(perp.openInterest / 1_000_000).toFixed(1)}M | Change: ${perp.oiChangePct24h > 0 ? '+' : ''}${perp.oiChangePct24h.toFixed(1)}%`);
    console.log(`   🎯 Confidence: ${(opp.confidence * 100).toFixed(0)}% | Risk: ${opp.riskScore}/10`);
    console.log(`   📈 Expected: +${opp.expectedReturn.toFixed(2)}% | Risk-adj: +${opp.riskAdjustedReturn.toFixed(2)}%`);
    if (!opp.approved && opp.rejectReason) {
      console.log(`   ⚠️ Rejected: ${opp.rejectReason}`);
    }
  }

  /**
   * Log funding arb evaluation
   */
  private logFundingArbEvaluation(arb: FundingArbitrageOpportunity, opp: EvaluatedOpportunity): void {
    console.log(`\n✅ FUNDING ARB: ${opp.name}`);
    console.log(`   🔄 Long @${arb.longVenue}: ${(arb.longRate * 100).toFixed(4)}%`);
    console.log(`   🔄 Short @${arb.shortVenue}: ${(arb.shortRate * 100).toFixed(4)}%`);
    console.log(`   💰 Spread: ${arb.estimatedProfitBps.toFixed(1)}bps (${opp.expectedReturn.toFixed(1)}% APR)`);
    console.log(`   🎯 Confidence: ${(opp.confidence * 100).toFixed(0)}% | Risk: ${opp.riskScore}/10`);
  }

  /**
   * Log speculation (sentiment) evaluation
   */
  private logSpeculationEvaluation(result: any): void {
    const emoji = result.approved ? '✅' : '❌';
    const signalEmoji = result.signal === 'BULLISH' ? '📈' : result.signal === 'BEARISH' ? '📉' : '➡️';

    console.log(`\n${emoji} SENTIMENT: ${result.name}`);
    console.log(`   ${signalEmoji} Signal: ${result.signal} | Score: ${result.sentimentScore.toFixed(3)}`);
    console.log(`   📊 Sources: Twitter=${result.sources.twitter.available ? '✓' : '✗'} CryptoPanic=${result.sources.cryptopanic.available ? '✓' : '✗'} Telegram=${result.sources.telegram.available ? '✓' : '✗'}`);
    console.log(`   🎯 Confidence: ${(result.confidence * 100).toFixed(0)}% | Risk: ${result.riskScore.toFixed(1)}/10`);
    console.log(`   📈 Expected: ${result.expectedReturn > 0 ? '+' : ''}${result.expectedReturn.toFixed(2)}% | Risk-adj: ${result.riskAdjustedReturn > 0 ? '+' : ''}${result.riskAdjustedReturn.toFixed(2)}%`);

    // Display related news if available
    if (result.news && result.news.length > 0) {
      console.log(`\n   📰 Related News:`);
      for (const newsItem of result.news) {
        console.log(`   → ${newsItem.title}`);
        if (newsItem.description) {
          // Truncate description to 120 chars for readability
          const desc = newsItem.description.length > 120
            ? newsItem.description.substring(0, 120) + '...'
            : newsItem.description;
          console.log(`      ${desc}`);
        }
        console.log(`      Link: ${newsItem.url}`);
      }
    }

    if (!result.approved && result.rejectReason) {
      console.log(`   ⚠️ Rejected: ${result.rejectReason}`);
    }

    if (result.warnings.length > 0) {
      console.log(`   ⚠️ Warnings: ${result.warnings.join(', ')}`);
    }
  }

  /**
   * Log news impact evaluation
   */
  private logNewsEvaluation(result: any): void {
    const emoji = result.approved ? '✅' : '❌';
    const impactEmoji = result.impact === 'POSITIVE' ? '📈' : result.impact === 'NEGATIVE' ? '📉' : result.impact === 'MIXED' ? '🔀' : '➡️';
    const severityEmoji = result.severity === 'CRITICAL' ? '🚨' : result.severity === 'HIGH' ? '⚠️' : result.severity === 'MEDIUM' ? '⚡' : 'ℹ️';

    console.log(`\n${emoji} NEWS: ${result.name}`);
    console.log(`   ${impactEmoji} Impact: ${result.impact} (${result.immediateImpact > 0 ? '+' : ''}${result.immediateImpact}) | ${severityEmoji} Severity: ${result.severity}`);
    console.log(`   📊 Action: ${result.tradingAction} | News Count: ${result.newsCount}`);
    console.log(`   🎯 Confidence: ${(result.confidence * 100).toFixed(0)}% | Risk: ${result.riskScore.toFixed(1)}/10`);
    console.log(`   📈 Expected: ${result.expectedReturn > 0 ? '+' : ''}${result.expectedReturn.toFixed(2)}% | Risk-adj: ${result.riskAdjustedReturn > 0 ? '+' : ''}${result.riskAdjustedReturn.toFixed(2)}%`);

    // Display top news items if available
    if (result.topNews && result.topNews.length > 0) {
      console.log(`\n   📰 Top News:`);
      for (const newsItem of result.topNews.slice(0, 3)) {
        console.log(`   → ${newsItem.title}`);
        console.log(`      Impact: ${newsItem.score.immediateImpact} | Action: ${newsItem.score.tradingAction}`);
      }
    }

    if (!result.approved && result.rejectReason) {
      console.log(`   ⚠️ Rejected: ${result.rejectReason}`);
    }

    if (result.warnings && result.warnings.length > 0) {
      console.log(`   ⚠️ Warnings: ${result.warnings.join(', ')}`);
    }
  }

  /**
   * Extract token symbols from an opportunity
   * Used for cross-filtering by fundamental health
   */
  private extractTokensFromOpportunity(opp: EvaluatedOpportunity): string[] {
    const tokens: string[] = [];

    switch (opp.type) {
      case 'arbitrage':
        // ArbitrageOpportunity has symbol field
        if (opp.raw?.symbol) {
          tokens.push(opp.raw.symbol);
        }
        break;

      case 'lp':
        // LPPool has token0 and token1 fields
        if (opp.raw?.token0) tokens.push(opp.raw.token0);
        if (opp.raw?.token1) tokens.push(opp.raw.token1);
        break;

      case 'perps':
      case 'funding_arb':
        // PerpsOpportunity and FundingArbitrageOpportunity have market field (e.g., 'SOL-PERP')
        if (opp.raw?.market) {
          const token = opp.raw.market.split('-')[0];
          if (token) tokens.push(token);
        }
        break;

      case 'speculation':
        // SpeculationOpportunity has token field
        if (opp.raw?.token) {
          tokens.push(opp.raw.token);
        }
        break;

      case 'fundamental':
        // FundamentalOpportunity has token field
        if (opp.raw?.token) {
          tokens.push(opp.raw.token);
        }
        break;

      case 'news':
        // NewsOpportunity has asset field
        if (opp.raw?.asset) {
          tokens.push(opp.raw.asset);
        }
        break;
    }

    return tokens;
  }

  /**
   * Log fundamental (on-chain) evaluation
   */
  private logFundamentalEvaluation(result: any): void {
    const emoji = result.approved ? '✅' : '❌';
    const ratingEmoji: Record<string, string> = {
      EXCELLENT: '🌟',
      GOOD: '👍',
      FAIR: '⚠️',
      POOR: '❌',
    };
    const ratingIcon = ratingEmoji[result.rating] || '❓';

    // Format holder count display
    const holderCountDisplay = result.details.holderCount === -1
      ? '>1M (highly distributed)'
      : result.details.holderCount.toLocaleString();

    console.log(`\n${emoji} FUNDAMENTAL: ${result.name}`);
    console.log(`   ${ratingIcon} Rating: ${result.rating} | Score: ${result.healthScore}/100`);
    console.log(`   📊 Metrics: Holders=${result.metrics.holderConcentration}/40 Liquidity=${result.metrics.liquidityDepth}/30 Age=${result.metrics.tokenAge}/20 Whale=${result.metrics.whaleActivity}/10`);
    console.log(`   🎯 Details: Holders=${holderCountDisplay} Top10=${result.details.topHoldersPercentage.toFixed(1)}% TVL=$${(result.details.tvlUsd / 1_000_000).toFixed(2)}M Age=${result.details.ageInDays}d Whale=${result.details.whaleTransferCount}`);
    console.log(`   🎲 Risk: ${result.riskScore.toFixed(1)}/10 | Confidence: ${(result.confidence * 100).toFixed(0)}%`);

    if (!result.approved && result.rejectReason) {
      console.log(`   ⚠️ Rejected: ${result.rejectReason}`);
    }

    if (result.warnings.length > 0) {
      console.log(`   ⚠️ Warnings: ${result.warnings.join(', ')}`);
    }
  }

  /**
   * Select best opportunity by risk-adjusted return
   */
  private selectBest(opportunities: EvaluatedOpportunity[]): EvaluatedOpportunity | null {
    const approved = opportunities
      .filter(o => o.approved)
      // For LP: expectedReturn is APY (annual), for arbitrage it's spread %
      // Use different thresholds: LP needs 10%+ APY, arb needs 0.5%+ spread
      .filter(o => {
        if (o.type === 'lp') {
          return o.expectedReturn >= 10; // 10% APY minimum
        }
        return o.riskAdjustedReturn >= this.config.minRiskAdjustedReturn;
      })
      .sort((a, b) => b.riskAdjustedReturn - a.riskAdjustedReturn);

    return approved[0] || null;
  }

  /**
   * Validate opportunity with multi-agent consensus voting
   *
   * Collects votes from all analysts, calculates weighted consensus,
   * and optionally triggers research debate for high-value trades.
   *
   * @returns true if consensus reached, false if rejected
   */
  private async validateWithConsensus(
    opp: EvaluatedOpportunity,
    positionUsd: number
  ): Promise<{ passed: boolean; consensusResult?: ConsensusResult }> {
    if (!this.consensusEnabled) {
      console.log('[Consensus] ⏭️ Consensus voting disabled');
      return { passed: true };
    }

    console.log('\n[Consensus] 🗳️ Starting multi-agent consensus voting...');

    // Collect votes from analysts based on opportunity type
    const analystVotes: Vote[] = [];
    const asset = this.extractAssetFromOpportunity(opp);

    // Each analyst casts a vote based on their analysis
    // Convert analyst confidence to votes
    const opportunityVote = this.opportunityToVote(opp);
    analystVotes.push(opportunityVote);

    // Get researcher votes for high-value trades
    if (positionUsd >= DEFAULT_CONSENSUS_CONFIG.debateRequiredAboveUsd) {
      console.log(`[Consensus] 🎭 High-value trade ($${positionUsd.toFixed(0)}) - adding researcher votes...`);

      // Build research input from opportunity
      const researchInput = this.buildResearchInput(opp);

      // Get researcher votes
      const bullishVote = await this.bullishResearcher.vote(asset, researchInput);
      const bearishVote = await this.bearishResearcher.vote(asset, researchInput);

      analystVotes.push(bullishVote, bearishVote);
    }

    // Collect votes with performance-adjusted weights
    const votes = await this.votingEngine.collectVotes(opp, analystVotes);
    logVotes(votes, { asset, opportunityType: opp.type });

    // Calculate consensus
    const isPerps = opp.type === 'perps' || opp.type === 'funding_arb';
    const positionPct = (positionUsd / this.config.portfolioValueUsd) * 100;

    const consensusResult = await this.votingEngine.calculateConsensus(votes, {
      opportunity: opp,
      positionSizePct: positionPct,
      isPerps,
    });

    logConsensusResult(consensusResult, { asset });

    if (!consensusResult.passed) {
      console.log(`[Consensus] ❌ Consensus not reached: ${consensusResult.failReason}`);
      return { passed: false, consensusResult };
    }

    // Trigger research debate for high-value trades
    if (positionUsd >= DEFAULT_CONSENSUS_CONFIG.debateRequiredAboveUsd &&
        consensusResult.decision === 'BUY') {
      console.log('[Consensus] 🎭 Conducting research debate...');
      const researchInput = this.buildResearchInput(opp);
      const debate = await this.researchDebateManager.conductDebate(researchInput);
      logResearchDebate(debate);

      if (debate.consensus.recommendation === 'REJECT' ||
          debate.consensus.recommendation.includes('SELL')) {
        console.log(`[Consensus] ❌ Research debate rejected: ${debate.consensus.reasoning}`);
        return { passed: false, consensusResult };
      }
    }

    console.log(`[Consensus] ✅ Consensus reached: ${consensusResult.decision} with ${consensusResult.agreementPct.toFixed(0)}% agreement`);
    return { passed: true, consensusResult };
  }

  /**
   * Convert opportunity to a vote
   */
  private opportunityToVote(opp: EvaluatedOpportunity): Vote {
    let decision: VoteDecision = 'HOLD';
    if (opp.approved && opp.confidence >= 0.5) {
      decision = 'BUY';
    } else if (!opp.approved && opp.confidence < 0.3) {
      decision = 'SELL';
    } else if (opp.confidence < 0.4) {
      decision = 'ABSTAIN';
    }

    return {
      agentId: `${opp.type}-analyst`,
      agentType: 'analyst',
      decision,
      confidence: opp.confidence,
      weight: 1.0,
      reasoning: opp.approved
        ? `Approved: ${opp.name} with ${(opp.confidence * 100).toFixed(0)}% confidence`
        : `Rejected: ${opp.rejectReason || 'Low confidence'}`,
      timestamp: new Date(),
      metrics: {
        expectedReturn: opp.expectedReturn,
        riskScore: opp.riskScore,
      },
    };
  }

  /**
   * Extract asset symbol from opportunity
   */
  private extractAssetFromOpportunity(opp: EvaluatedOpportunity): string {
    if (opp.type === 'perps' || opp.type === 'funding_arb') {
      return opp.perpsDetails?.market?.replace('-PERP', '') || 'UNKNOWN';
    }
    if (opp.type === 'spot') {
      return opp.raw?.token?.symbol || 'UNKNOWN';
    }
    if (opp.type === 'lp') {
      return opp.name.split('/')[0] || 'UNKNOWN';
    }
    if (opp.type === 'arbitrage') {
      return opp.raw?.symbol || 'UNKNOWN';
    }
    return opp.name.split(' ')[0] || 'UNKNOWN';
  }

  /**
   * Build research input from opportunity for researchers
   */
  private buildResearchInput(opp: EvaluatedOpportunity): ResearchInput {
    const asset = this.extractAssetFromOpportunity(opp);

    // Extract real price from opportunity raw data
    let currentPrice = 0;
    if (opp.raw) {
      if (typeof opp.raw.price === 'number') currentPrice = opp.raw.price;
      else if (typeof opp.raw.buyPrice === 'number') currentPrice = opp.raw.buyPrice;
      else if (typeof opp.raw.sellPrice === 'number') currentPrice = opp.raw.sellPrice;
      else if (opp.raw.token?.marketData?.price) currentPrice = opp.raw.token.marketData.price;
    }

    return {
      asset,
      fundamentals: undefined,
      sentiment: undefined,
      news: undefined,
      priceData: {
        currentPrice,
        change24h: opp.expectedReturn / 10,
        change7d: 0,
        volume24h: opp.raw?.volume24h || 0,
      },
    };
  }

  /**
   * Execute the selected opportunity
   */
  private async execute(opp: EvaluatedOpportunity): Promise<void> {
    console.log(`[AGENT] 🚀 Executing ${opp.type}: ${opp.name}`);

    // CRITICAL: Verify opportunity is approved before execution
    if (!opp.approved) {
      console.log(`[AGENT] ❌ BLOCKED: Opportunity not approved. Reason: ${opp.rejectReason || 'Unknown'}`);
      return;
    }

    // Double-check risk manager approval (defense in depth)
    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: opp.confidence,
      currentVolatility24h: this.getVolatility(),
      portfolioValueUsd: this.portfolioManager.getTotalValueUsd(),
    });

    const riskRecheck = this.riskManager.checkTradeAllowed({
      proposedPositionPct: positionCalc.positionPct,
      currentVolatility24h: this.getVolatility(),
      expectedReturnPct: opp.expectedReturn / 365, // Daily return
    });

    if (!riskRecheck.allowed) {
      console.log(`[AGENT] ❌ BLOCKED by Risk Manager: ${riskRecheck.reason}`);
      return;
    }

    const positionUsd = positionCalc.positionPct * this.portfolioManager.getTotalValueUsd();
    console.log(`[AGENT] 🗳️  Running consensus validation (position: $${positionUsd.toFixed(2)})...`);
    const consensusCheck = await this.validateWithConsensus(opp, positionUsd);
    if (!consensusCheck.passed) {
      console.log(`[AGENT] ❌ BLOCKED by Consensus: Trade rejected by multi-agent voting`);
      return;
    }
    console.log(`[AGENT] ✅ Consensus passed`);

    // Check if we have sufficient capital
    const portfolioValue = this.portfolioManager.getTotalValueUsd();
    const minCapital = 10; // $10 minimum to trade
    if (portfolioValue < minCapital) {
      console.log(`[AGENT] ❌ Insufficient capital: $${portfolioValue.toFixed(2)} < $${minCapital}`);
      return;
    }

    if (opp.type === 'lp') {
      await this.executeLp(opp);
    } else if (opp.type === 'perps') {
      await this.executePerps(opp);
    } else if (opp.type === 'funding_arb') {
      await this.executeFundingArb(opp);
    } else if (opp.type === 'lending') {
      await this.executeLending(opp);
    } else if (opp.type === 'spot') {
      await this.executeSpot(opp);
    } else if (opp.type === 'pumpfun') {
      await this.executePumpfun(opp);
    } else {
      await this.executeArbitrage(opp);
    }
  }

  /**
   * Execute LP opportunity
   * Uses LPExecutor for real DEX deposits (Orca, Raydium, Meteora)
   */
  private async executeLp(opp: EvaluatedOpportunity): Promise<void> {
    const pool = opp.raw as LPPool;

    // Calculate position size using risk manager
    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: opp.confidence,
      currentVolatility24h: this.getVolatility(),
      portfolioValueUsd: this.portfolioManager.getTotalValueUsd(),
    });

    const capitalUsd = positionCalc.positionUsd;
    const tokens = pool.name.split('/');

    if (this.config.dryRun) {
      console.log('[AGENT] 📝 DRY RUN - No actual execution');
      console.log(`[AGENT]    Would provide liquidity to ${opp.name}`);
      console.log(`[AGENT]    Capital: $${capitalUsd.toFixed(2)} (${positionCalc.positionPct.toFixed(1)}%)`);

      // Track position in portfolio even in dry-run (for paper trading)
      const positionId = this.portfolioManager.openLPPosition({
        poolAddress: pool.address,
        poolName: pool.name,
        dex: pool.dex,
        token0: tokens[0] || 'TOKEN0',
        token1: tokens[1] || 'TOKEN1',
        capitalUsd,
        entryApy: pool.apy,
      });

      console.log(`[AGENT]    Position ID: ${positionId}`);
      console.log(`[AGENT]    Portfolio: $${this.portfolioManager.getTotalValueUsd().toFixed(2)}`);
      console.log('[AGENT] ✅ Done (simulated, position tracked)');
      return;
    }

    // ============= LIVE LP EXECUTION =============
    console.log('[AGENT] 💰 Executing REAL LP deposit...');
    console.log(`[AGENT]    Pool: ${pool.name} [${pool.dex}]`);
    console.log(`[AGENT]    Address: ${pool.address}`);
    console.log(`[AGENT]    Capital: $${capitalUsd.toFixed(2)} (${positionCalc.positionPct.toFixed(1)}%)`);

    // Check prerequisites
    if (!this.wallet) {
      console.log('[AGENT] ❌ No wallet configured - cannot execute live trades');
      console.log('[AGENT]    Set SOLANA_PRIVATE_KEY environment variable');
      return;
    }

    if (!this.lpExecutor) {
      console.log('[AGENT] ❌ LP Executor not initialized');
      return;
    }

    // Convert LPPool to LPPoolInfo for executor
    // Look up token mints from known tokens, default to empty if unknown
    const token0Symbol = tokens[0]?.toUpperCase() || 'TOKEN0';
    const token1Symbol = tokens[1]?.toUpperCase() || 'TOKEN1';
    const token0Mint = (TOKEN_MINTS as Record<string, string>)[token0Symbol] || '';
    const token1Mint = (TOKEN_MINTS as Record<string, string>)[token1Symbol] || '';

    // Determine decimals based on token type (stables=6, others=9)
    const stableTokens = ['USDC', 'USDT'];
    const token0Decimals = stableTokens.includes(token0Symbol) ? 6 : 9;
    const token1Decimals = stableTokens.includes(token1Symbol) ? 6 : 9;

    const lpPoolInfo: LPPoolInfo = {
      address: pool.address,
      name: pool.name,
      dex: pool.dex as SupportedDex,
      token0: {
        symbol: token0Symbol,
        mint: token0Mint,
        decimals: token0Decimals,
      },
      token1: {
        symbol: token1Symbol,
        mint: token1Mint,
        decimals: token1Decimals,
      },
      fee: Math.round((pool.feeRate || 0.3) * 100), // Convert % to basis points
      tvlUsd: pool.tvl,
      apy: pool.apy,
    };

    try {
      // Execute deposit via LP Executor
      const result = await this.lpExecutor.deposit({
        pool: lpPoolInfo,
        amountUsd: capitalUsd,
        wallet: this.wallet,
        slippageBps: 100, // 1%
      });

      if (result.success) {
        console.log('\n╔═══════════════════════════════════════════════════════════╗');
        console.log('║  ✅ LP DEPOSIT SUCCESSFUL!                                ║');
        console.log('╠═══════════════════════════════════════════════════════════╣');
        console.log(`║  Pool:        ${pool.name.padEnd(44)} ║`);
        console.log(`║  DEX:         ${pool.dex.padEnd(44)} ║`);
        console.log(`║  Capital:     $${capitalUsd.toFixed(2).padEnd(42)} ║`);
        console.log('╠═══════════════════════════════════════════════════════════╣');
        console.log(`║  🔗 TX SIGNATURE:                                         ║`);
        console.log(`║  ${result.txSignature?.padEnd(56) || 'N/A'.padEnd(56)} ║`);
        console.log('╠═══════════════════════════════════════════════════════════╣');
        console.log(`║  🌐 View on Solscan:                                      ║`);
        console.log(`║  https://solscan.io/tx/${result.txSignature || 'N/A'}${' '.repeat(Math.max(0, 24 - (result.txSignature?.length || 0)))} ║`);
        console.log('╚═══════════════════════════════════════════════════════════╝\n');

        if (result.positionId) {
          console.log(`[AGENT]    Position ID: ${result.positionId}`);
        }
        if (result.lpTokenMint) {
          console.log(`[AGENT]    LP Token: ${result.lpTokenMint}`);
        }

        // Track position in portfolio with real data
        // Note: LP token details logged above, portfolio tracks USD value
        const positionId = this.portfolioManager.openLPPosition({
          poolAddress: pool.address,
          poolName: pool.name,
          dex: pool.dex,
          token0: tokens[0] || 'TOKEN0',
          token1: tokens[1] || 'TOKEN1',
          capitalUsd,
          entryApy: pool.apy,
        });

        this.riskManager.recordTrade(0);
        console.log(`[AGENT]    Portfolio Position: ${positionId}`);
        console.log(`[AGENT]    Portfolio Value: $${this.portfolioManager.getTotalValueUsd().toFixed(2)}`);
      } else {
        console.log(`[AGENT] ❌ LP deposit failed: ${result.error}`);
        if (result.priceImpactPct) {
          console.log(`[AGENT]    Price impact: ${result.priceImpactPct.toFixed(2)}%`);
        }
        this.riskManager.recordTrade(-0.1); // Small loss for failed trade
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.log(`[AGENT] ❌ LP execution error: ${errorMsg}`);
      this.riskManager.recordTrade(-0.1);
    }
  }

  /**
   * Execute Arbitrage opportunity
   */
  private async executeArbitrage(opp: EvaluatedOpportunity): Promise<void> {
    const arb = opp.raw as ArbitrageOpportunity;

    // Create executor with dryRun from config
    const executor = createArbitrageExecutor(this.config.dryRun);

    // Calculate position size (use Half-Kelly)
    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: opp.confidence,
      currentVolatility24h: this.getVolatility(),
      portfolioValueUsd: this.config.portfolioValueUsd,
    });

    const amountUsd = Math.min(positionCalc.positionUsd, 5000); // Cap at $5K for safety

    console.log(`[AGENT]    Position size: $${amountUsd.toFixed(0)} (${positionCalc.positionPct.toFixed(1)}%)`);

    // Execute!
    const result: ArbitrageResult = await executor.execute(arb, amountUsd);

    if (result.success) {
      console.log(`[AGENT] ✅ Arbitrage complete!`);
      console.log(`[AGENT]    Net profit: $${result.netProfit.toFixed(2)} (+${result.profitPct.toFixed(2)}%)`);
      console.log(`[AGENT]    Fees: CEX $${result.fees.cexTrade.toFixed(2)}, Withdraw $${result.fees.withdrawal.toFixed(2)}, DEX $${result.fees.dexSwap.toFixed(2)}`);

      // Record profit
      const pnlPct = (result.netProfit / amountUsd) * 100;
      this.riskManager.recordTrade(pnlPct);
    } else {
      console.log(`[AGENT] ❌ Arbitrage failed: ${result.error}`);
      this.riskManager.recordTrade(-0.1); // Small loss for failed trade (gas)
    }
  }

  /**
   * Execute perps funding opportunity
   */
  private async executePerps(opp: EvaluatedOpportunity): Promise<void> {
    const perp = opp.raw as PerpsOpportunity;
    const details = opp.perpsDetails!;

    if (this.config.dryRun) {
      console.log('[AGENT] 📝 DRY RUN - No actual execution');
      console.log(`[AGENT]    Would open ${details.side.toUpperCase()} ${perp.market} on ${details.venue}`);
      console.log(`[AGENT]    Leverage: ${details.leverage}x`);
      console.log(`[AGENT]    Funding: ${(details.fundingRate * 100).toFixed(4)}% hourly`);
      console.log('[AGENT] ✅ Done (simulated)');
      return;
    }

    // Live perps execution
    try {
      const perpsService = getPerpsService();

      // Calculate position size
      const positionCalc = this.riskManager.calculatePositionSize({
        modelConfidence: opp.confidence,
        currentVolatility24h: this.getVolatility(),
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      const sizeUsd = Math.min(positionCalc.positionUsd, 2000); // Cap at $2K for perps

      console.log(`[AGENT]    Opening ${details.side} ${perp.market} on ${details.venue}...`);
      console.log(`[AGENT]    Size: $${sizeUsd.toFixed(0)} @ ${details.leverage}x leverage`);

      const result = await perpsService.openPosition({
        venue: details.venue,
        market: perp.market,
        side: details.side,
        size: sizeUsd,
        leverage: details.leverage,
      });

      if (result.success) {
        console.log(`[AGENT] ✅ Perps position opened!`);
        console.log(`[AGENT]    Entry: $${result.entryPrice?.toFixed(2)}`);
        const totalFees = result.fees.trading + result.fees.funding + result.fees.gas;
        console.log(`[AGENT]    Fees: $${totalFees.toFixed(2)}`);
        this.riskManager.recordTrade(0); // No immediate P&L
      } else {
        console.log(`[AGENT] ❌ Perps failed: ${result.error}`);
        this.riskManager.recordTrade(-0.1);
      }
    } catch (error) {
      console.log(`[AGENT] ❌ Perps error: ${error}`);
      this.riskManager.recordTrade(-0.1);
    }
  }

  /**
   * Execute funding rate arbitrage (delta-neutral)
   */
  private async executeFundingArb(opp: EvaluatedOpportunity): Promise<void> {
    const arb = opp.raw as FundingArbitrageOpportunity;

    if (this.config.dryRun) {
      console.log('[AGENT] 📝 DRY RUN - No actual execution');
      console.log(`[AGENT]    Would open LONG ${arb.market} on ${arb.longVenue}`);
      console.log(`[AGENT]    Would open SHORT ${arb.market} on ${arb.shortVenue}`);
      console.log(`[AGENT]    Expected spread: ${arb.estimatedProfitBps.toFixed(1)}bps`);
      console.log('[AGENT] ✅ Done (simulated)');
      return;
    }

    // Live funding arb execution
    try {
      const perpsService = getPerpsService();

      // Calculate position size (larger for delta-neutral)
      const positionCalc = this.riskManager.calculatePositionSize({
        modelConfidence: opp.confidence,
        currentVolatility24h: this.getVolatility(),
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      const sizeUsd = Math.min(positionCalc.positionUsd * 2, 5000); // Larger for delta-neutral

      console.log(`[AGENT]    Opening delta-neutral position...`);
      console.log(`[AGENT]    LONG ${arb.market} on ${arb.longVenue}: $${sizeUsd.toFixed(0)}`);
      console.log(`[AGENT]    SHORT ${arb.market} on ${arb.shortVenue}: $${sizeUsd.toFixed(0)}`);

      // Open both legs
      const [longResult, shortResult] = await Promise.all([
        perpsService.openPosition({
          venue: arb.longVenue as PerpsVenue,
          market: arb.market,
          side: 'long',
          size: sizeUsd,
          leverage: 2,
        }),
        perpsService.openPosition({
          venue: arb.shortVenue as PerpsVenue,
          market: arb.market,
          side: 'short',
          size: sizeUsd,
          leverage: 2,
        }),
      ]);

      if (longResult.success && shortResult.success) {
        console.log(`[AGENT] ✅ Funding arb opened!`);
        console.log(`[AGENT]    Long entry: $${longResult.entryPrice?.toFixed(2)}`);
        console.log(`[AGENT]    Short entry: $${shortResult.entryPrice?.toFixed(2)}`);
        this.riskManager.recordTrade(0);
      } else {
        console.log(`[AGENT] ⚠️ Partial execution - manual intervention needed`);
        if (!longResult.success) console.log(`[AGENT]    Long failed: ${longResult.error}`);
        if (!shortResult.success) console.log(`[AGENT]    Short failed: ${shortResult.error}`);
        this.riskManager.recordTrade(-0.2);
      }
    } catch (error) {
      console.log(`[AGENT] ❌ Funding arb error: ${error}`);
      this.riskManager.recordTrade(-0.2);
    }
  }

  /**
   * Execute lending opportunity
   */
  private async executeLending(opp: EvaluatedOpportunity): Promise<void> {
    const market = opp.raw as import('../services/lending/types.js').LendingMarketData;

    if (this.config.dryRun) {
      console.log('[AGENT] 📝 DRY RUN - No actual execution');
      console.log(`[AGENT]    Would deposit to ${market.asset} on ${market.protocol}`);
      console.log(`[AGENT]    Expected APY: ${(market.supplyApy * 100).toFixed(2)}%`);
      console.log(`[AGENT]    TVL: $${(market.tvlUsd / 1_000_000).toFixed(2)}M`);
      console.log('[AGENT] ✅ Done (simulated)');
      return;
    }

    // Live lending execution
    try {
      // Calculate position size
      const positionCalc = this.riskManager.calculatePositionSize({
        modelConfidence: opp.confidence,
        currentVolatility24h: this.getVolatility(),
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      const depositUsd = Math.min(positionCalc.positionUsd, 5000); // Cap at $5K for lending

      console.log(`[AGENT]    Depositing $${depositUsd.toFixed(0)} to ${market.asset} on ${market.protocol}...`);
      console.log(`[AGENT]    Expected APY: ${(market.supplyApy * 100).toFixed(2)}%`);

      // Execute lending deposit
      if (!this.lendingExecutor) {
        console.log(`[AGENT] ⚠️  Lending executor not initialized`);
        return;
      }

      const result = await this.lendingExecutor.deposit({
        protocol: market.protocol as 'kamino' | 'marginfi' | 'solend',
        asset: market.asset,
        amountUsd: depositUsd,
      });

      if (result.success && result.signature) {
        console.log(`[AGENT] ✅ Deposited via ${market.protocol}: ${result.signature}`);

        // Track position for monitoring
        this.openLendingPosition({
          protocol: market.protocol,
          asset: market.asset,
          depositedUsd: depositUsd,
          entryApy: market.supplyApy,
        });

        this.riskManager.recordTrade(0.5); // Positive outcome for successful deposit

      } else {
        console.log(`[AGENT] ❌ Deposit failed: ${result.error}`);
        this.riskManager.recordTrade(-0.3); // Negative outcome for failed deposit
      }
    } catch (error) {
      console.log(`[AGENT] ❌ Lending error: ${error}`);
      this.riskManager.recordTrade(-0.1);
    }
  }

  /**
   * Execute spot trading opportunity
   */
  private async executeSpot(opp: EvaluatedOpportunity): Promise<void> {
    const spotData = opp.raw as any;
    const token = spotData.token;

    if (this.config.dryRun) {
      console.log('[AGENT] 📝 DRY RUN - No actual execution');
      console.log(`[AGENT]    Would buy ${token.symbol} spot`);
      console.log(`[AGENT]    Expected Return: ${opp.expectedReturn.toFixed(2)}%`);
      console.log(`[AGENT]    ML Probability: ${(Number(spotData.mlProbability) * 100).toFixed(1)}%`);
      console.log(`[AGENT]    Rule Score: ${Number(spotData.ruleScore).toFixed(1)}`);
      console.log('[AGENT] ✅ Done (simulated)');
      return;
    }

    // Live spot trading execution
    try {
      // Calculate position size
      const positionCalc = this.riskManager.calculatePositionSize({
        modelConfidence: opp.confidence,
        currentVolatility24h: this.getVolatility(),
        portfolioValueUsd: this.config.portfolioValueUsd,
      });

      const buyAmountUsd = Math.min(positionCalc.positionUsd, 2000); // Cap at $2K for spot

      console.log(`[AGENT]    Buying $${buyAmountUsd.toFixed(0)} of ${token.symbol}...`);
      console.log(`[AGENT]    Entry Price: $${token.currentPrice || 'N/A'}`);
      console.log(`[AGENT]    Target: +${opp.expectedReturn.toFixed(1)}% (TP1: +12%)`);
      console.log(`[AGENT]    Stop Loss: -8%`);

      // Execute spot buy via Jupiter
      if (!this.spotExecutor) {
        console.log(`[AGENT] ⚠️  Spot executor not initialized`);
        return;
      }

      // Use SOL as input since wallet has SOL (not USDC)
      const SOL_MINT = 'So11111111111111111111111111111111111111112';

      const result = await this.spotExecutor.buy({
        inputMint: SOL_MINT,
        outputMint: token.address,
        amountUsd: buyAmountUsd,
      });

      if (result.success && result.signature) {
        console.log(`[AGENT] ✅ Bought ${token.symbol}: ${result.signature}`);
        console.log(`[AGENT]    Received: ${result.outputAmount} tokens (raw lamports)`);

        // Get token decimals to convert raw lamports to actual token amount
        let tokenDecimals = 6; // Default to 6 (common for SPL tokens)
        try {
          if (this.connection) {
            const tokenMint = new PublicKey(token.address);
            const mintInfo = await getMint(this.connection, tokenMint);
            tokenDecimals = mintInfo.decimals;
            logger.info('[CRTX] Token decimals fetched', { symbol: token.symbol, decimals: tokenDecimals });
          }
        } catch (mintError) {
          logger.warn('[CRTX] Could not fetch token decimals, using default', { symbol: token.symbol, default: tokenDecimals });
        }

        // Convert raw lamports to actual token amount
        const actualTokenAmount = result.outputAmount ? result.outputAmount / Math.pow(10, tokenDecimals) : 0;

        // Calculate correct entry price: USD spent / actual tokens received
        const entryPrice = actualTokenAmount > 0 ? buyAmountUsd / actualTokenAmount : (token.currentPrice || 0);

        console.log(`[AGENT]    Actual tokens: ${actualTokenAmount.toFixed(6)} ${token.symbol}`);
        console.log(`[AGENT]    Entry price: $${entryPrice.toFixed(6)}`);

        // Track position for monitoring
        const approvedToken: ApprovedToken = {
          symbol: token.symbol,
          address: token.address,
          marketCap: token.marketCap || 0,
          liquidity: token.liquidity || 0,
          volume24h: token.volume24h || 0,
          holders: token.holders || 0,
          age: token.age || 90,
          tier: token.tier || 2,
          dexes: token.dexes || ['jupiter'],
          verified: token.verified !== false,
          approvedAt: Date.now(),
        };

        this.openSpotPosition({
          token: approvedToken,
          entryPrice,
          entrySize: actualTokenAmount, // Store actual token amount, not raw lamports
        });

        this.riskManager.recordTrade(0.5); // Positive outcome for successful buy
      } else {
        console.log(`[AGENT] ❌ Buy failed: ${result.error}`);
        this.riskManager.recordTrade(-0.3); // Negative outcome for failed buy
      }
    } catch (error) {
      console.log(`[AGENT] ❌ Spot trading error: ${error}`);
      this.riskManager.recordTrade(-0.1);
    }
  }

  /**
   * Execute PumpFun token opportunity
   * Uses PumpFunClient with Guardian validation and Jito MEV protection
   */
  private async executePumpfun(opp: EvaluatedOpportunity): Promise<void> {
    const pumpfunData = opp.raw as any;
    const token = pumpfunData;
    const details = opp.pumpfunDetails;

    if (this.config.dryRun) {
      console.log('[AGENT] 📝 DRY RUN - No actual execution');
      console.log(`[AGENT]    Would buy PumpFun token: ${token.name || token.symbol}`);
      console.log(`[AGENT]    Mint: ${details?.tokenMint || token.mint}`);
      console.log(`[AGENT]    Amount: ${details?.amountSol?.toFixed(3) || '?'} SOL`);
      console.log(`[AGENT]    Slippage: ${(details?.slippageBps || 500) / 100}%`);
      console.log(`[AGENT]    Risk Score: ${details?.riskCheck?.riskScore || '?'}/100`);
      console.log(`[AGENT]    Expected Return: ${opp.expectedReturn.toFixed(2)}%`);
      console.log(`[AGENT]    Confidence: ${(opp.confidence * 100).toFixed(1)}%`);
      console.log('[AGENT] ✅ Done (simulated)');
      return;
    }

    // Live PumpFun trading via PumpFunClient (with Guardian + Jito)
    try {
      // Check if we have the execute function from PumpFunAnalyst
      if (details?.execute) {
        console.log(`[AGENT]    Buying PumpFun token ${token.name || token.symbol}...`);
        console.log(`[AGENT]    Mint: ${details.tokenMint}`);
        console.log(`[AGENT]    Amount: ${details.amountSol.toFixed(3)} SOL`);
        console.log(`[AGENT]    Slippage: ${details.slippageBps / 100}%`);
        console.log(`[AGENT]    Risk Score: ${details.riskCheck?.riskScore || 0}/100`);
        console.log(`[AGENT]    Target: +${opp.expectedReturn.toFixed(1)}%`);

        // Execute via PumpFunClient (includes Guardian validation + Jito MEV protection)
        const result = await details.execute();

        if (result.success && result.signature) {
          console.log(`[AGENT] ✅ Bought PumpFun token ${token.name || token.symbol}: ${result.signature}`);
          console.log(`[AGENT]    Tokens Received: ${result.tokensReceived || 'unknown'}`);
          console.log(`[AGENT]    SOL Spent: ${result.solSpent?.toFixed(4) || details.amountSol.toFixed(4)} SOL`);

          // Track position for monitoring (using SPOT monitoring with higher risk tier)
          const approvedToken: ApprovedToken = {
            symbol: token.symbol || token.name || 'PUMP',
            address: details.tokenMint,
            marketCap: token.marketCap || 0,
            liquidity: token.marketCap || 0,
            volume24h: token.volume || 0,
            holders: token.holderCount || 0,
            age: token.ageHours || 1,
            tier: 3, // Highest risk tier for PumpFun
            dexes: ['pumpfun'],
            verified: false,
            approvedAt: Date.now(),
          };

          // Calculate entry price from result
          const entryPrice = result.tokensReceived && result.solSpent
            ? result.solSpent / result.tokensReceived
            : 0;

          this.openSpotPosition({
            token: approvedToken,
            entryPrice,
            entrySize: result.tokensReceived || 0,
          });

          this.riskManager.recordTrade(0.5);
        } else {
          console.log(`[AGENT] ❌ PumpFun buy failed: ${result.error}`);
          this.riskManager.recordTrade(-0.3);
        }
      } else {
        // Fallback to SpotExecutor (Jupiter) if no execute function
        console.log(`[AGENT] ⚠️  No PumpFun execute function, falling back to Jupiter...`);
        await this.executePumpfunViaJupiter(opp);
      }
    } catch (error) {
      console.log(`[AGENT] ❌ PumpFun trading error: ${error}`);
      this.riskManager.recordTrade(-0.1);
    }
  }

  /**
   * Fallback: Execute PumpFun trade via Jupiter (SpotExecutor)
   */
  private async executePumpfunViaJupiter(opp: EvaluatedOpportunity): Promise<void> {
    const token = opp.raw as any;

    // Calculate position size - smaller for high-risk PumpFun tokens
    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: opp.confidence,
      currentVolatility24h: this.getVolatility(),
      portfolioValueUsd: this.config.portfolioValueUsd,
    });

    // Cap PumpFun trades at $500 (high risk)
    const buyAmountUsd = Math.min(positionCalc.positionUsd * 0.5, 500);

    if (!this.spotExecutor) {
      console.log(`[AGENT] ⚠️  Spot executor not initialized for PumpFun trade`);
      return;
    }

    const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';

    const result = await this.spotExecutor.buy({
      inputMint: USDC_MINT,
      outputMint: token.mint,
      amountUsd: buyAmountUsd,
    });

    if (result.success && result.signature) {
      console.log(`[AGENT] ✅ Bought PumpFun token via Jupiter: ${result.signature}`);
      this.riskManager.recordTrade(0.5);
    } else {
      console.log(`[AGENT] ❌ PumpFun Jupiter buy failed: ${result.error}`);
      this.riskManager.recordTrade(-0.3);
    }
  }

  /**
   * Start LP position monitoring loop
   * Checks all open LP positions periodically and exits when needed
   */
  startLPMonitoring(): void {
    if (this.monitoringIntervalId) {
      logger.warn('[CRTX] LP monitoring already running');
      return;
    }

    const intervalMs = this.lpPositionMonitor.getCheckInterval();

    logger.info('[CRTX] Starting LP position monitoring', {
      intervalMs,
      intervalMinutes: intervalMs / (60 * 1000),
    });

    // Run immediately
    this.checkLPPositions().catch(error => {
      logger.error('[CRTX] LP monitoring check failed', { error });
    });

    // Then run periodically
    this.monitoringIntervalId = setInterval(() => {
      this.checkLPPositions().catch(error => {
        logger.error('[CRTX] LP monitoring check failed', { error });
      });
    }, intervalMs);
  }

  /**
   * Stop LP position monitoring loop
   */
  stopLPMonitoring(): void {
    if (this.monitoringIntervalId) {
      clearInterval(this.monitoringIntervalId);
      this.monitoringIntervalId = null;
      logger.info('[CRTX] LP position monitoring stopped');
    }
  }

  /**
   * Check all open LP positions and exit if needed
   */
  private async checkLPPositions(): Promise<void> {
    const openPositions = this.portfolioManager.getOpenLPPositions();

    if (openPositions.length === 0) {
      return;
    }

    logger.info('[CRTX] Checking LP positions', {
      count: openPositions.length,
    });

    for (const position of openPositions) {
      try {
        const decision = this.lpPositionMonitor.shouldExit(position);

        if (decision.shouldExit) {
          logger.info('[CRTX] LP position exit triggered', {
            positionId: position.id,
            poolName: position.poolName,
            reason: decision.reason,
            exitType: decision.exitType,
            exitPercentage: decision.exitPercentage,
          });

          console.log(`\n[AGENT] 🚪 Exiting LP position: ${position.poolName}`);
          console.log(`[AGENT]    Reason: ${decision.reason}`);
          console.log(`[AGENT]    Exit: ${decision.exitPercentage}%`);

          // Execute withdrawal
          await this.exitLPPosition(position, decision.exitPercentage || 100);
        }
      } catch (error) {
        logger.error('[CRTX] Failed to check LP position', {
          positionId: position.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  /**
   * Exit an LP position
   */
  private async exitLPPosition(position: any, percentage: number): Promise<void> {
    if (!this.lpExecutor || !this.wallet) {
      logger.error('[CRTX] Cannot exit LP position - no executor or wallet');
      return;
    }

    try {
      const result = await this.lpExecutor.withdraw({
        pool: {
          address: position.poolAddress,
          name: position.poolName,
          dex: position.dex,
          token0: { symbol: position.token0 },
          token1: { symbol: position.token1 },
        } as any,
        positionId: position.id,
        portfolioPositionId: position.id,
        percentage,
        wallet: this.wallet,
      });

      if (result.success) {
        console.log(`\n╔═══════════════════════════════════════════════════════════╗`);
        console.log(`║  ✅ LP WITHDRAWAL SUCCESSFUL!                             ║`);
        console.log(`╠═══════════════════════════════════════════════════════════╣`);
        console.log(`║  Pool:        ${position.poolName.padEnd(43)} ║`);
        console.log(`║  DEX:         ${position.dex.padEnd(43)} ║`);
        console.log(`║  Withdrawn:   ${percentage}%${' '.repeat(41 - String(percentage).length)} ║`);
        console.log(`║  Value:       $${(result.amountUsd || 0).toFixed(2).padEnd(41)} ║`);
        console.log(`╠═══════════════════════════════════════════════════════════╣`);
        if (result.txSignature) {
          console.log(`║  🔗 TX SIGNATURE:                                         ║`);
          console.log(`║  ${result.txSignature.slice(0, 55).padEnd(55)} ║`);
          console.log(`╠═══════════════════════════════════════════════════════════╣`);
          console.log(`║  🌐 View on Solscan:                                      ║`);
          console.log(`║  https://solscan.io/tx/${result.txSignature.slice(0, 30)}... ║`);
        }
        console.log(`╚═══════════════════════════════════════════════════════════╝\n`);

        logger.info('[CRTX] LP position exited successfully', {
          positionId: position.id,
          poolName: position.poolName,
          percentage,
          amountUsd: result.amountUsd,
          txSignature: result.txSignature,
        });
      } else {
        console.log(`[AGENT] ❌ LP withdrawal failed: ${result.error}`);
        logger.error('[CRTX] LP withdrawal failed', {
          positionId: position.id,
          error: result.error,
        });
      }
    } catch (error) {
      console.log(`[AGENT] ❌ LP withdrawal error: ${error}`);
      logger.error('[CRTX] LP withdrawal error', {
        positionId: position.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Get current risk state
   */
  getRiskState() {
    return this.riskManager.getState();
  }

  // ============= SPOT POSITION MONITORING =============

  /**
   * Start spot position monitoring loop
   * Checks all open spot positions periodically and exits when needed
   */
  startSpotMonitoring(): void {
    if (this.spotMonitoringIntervalId) {
      logger.warn('[CRTX] Spot monitoring already running');
      return;
    }

    const intervalMs = 60 * 1000; // Check every 1 minute

    logger.info('[CRTX] Starting spot position monitoring', {
      intervalMs,
      intervalMinutes: intervalMs / (60 * 1000),
    });

    // Run immediately
    this.checkSpotPositions().catch(error => {
      logger.error('[CRTX] Spot monitoring check failed', { error });
    });

    // Then run periodically
    this.spotMonitoringIntervalId = setInterval(() => {
      this.checkSpotPositions().catch(error => {
        logger.error('[CRTX] Spot monitoring check failed', { error });
      });
    }, intervalMs);
  }

  /**
   * Stop spot position monitoring loop
   */
  stopSpotMonitoring(): void {
    if (this.spotMonitoringIntervalId) {
      clearInterval(this.spotMonitoringIntervalId);
      this.spotMonitoringIntervalId = null;
      logger.info('[CRTX] Spot position monitoring stopped');
    }
  }

  /**
   * Check all open spot positions and exit if conditions are met
   */
  private async checkSpotPositions(): Promise<void> {
    const openPositions = Array.from(this.spotPositions.values()).filter(p => p.status !== 'closed');

    if (openPositions.length === 0) {
      return;
    }

    logger.info('[CRTX] Checking spot positions', {
      count: openPositions.length,
    });

    for (const position of openPositions) {
      try {
        // Get current price from market data
        const currentPrice = await this.getTokenPrice(position.token.address);
        if (!currentPrice) {
          logger.warn('[CRTX] Could not get price for spot position', {
            symbol: position.token.symbol,
          });
          continue;
        }

        // Update position with current price
        position.currentPrice = currentPrice;
        position.currentValue = position.remainingSize * currentPrice;
        position.pnl = position.currentValue - (position.remainingSize * position.entryPrice);
        position.pnlPercent = (currentPrice - position.entryPrice) / position.entryPrice;
        position.daysHeld = (Date.now() - position.entryTimestamp) / (24 * 60 * 60 * 1000);

        // Display position status
        const pnlEmoji = position.pnlPercent >= 0 ? '📈' : '📉';
        const pnlColor = position.pnlPercent >= 0 ? '+' : '';
        console.log(`\n${pnlEmoji} [SPOT MONITOR] ${position.token.symbol}`);
        console.log(`   Entry: $${position.entryPrice.toFixed(4)} | Current: $${currentPrice.toFixed(4)}`);
        console.log(`   PnL: ${pnlColor}${(position.pnlPercent * 100).toFixed(2)}% ($${position.pnl.toFixed(2)})`);
        console.log(`   Size: ${position.remainingSize.toFixed(4)} tokens ($${position.currentValue.toFixed(2)})`);
        console.log(`   TP1: $${position.exitLevels.tp1.price.toFixed(4)} (+12%) | Stop: $${position.exitLevels.stopLoss.price.toFixed(4)} (-8%)`);
        console.log(`   Held: ${position.daysHeld.toFixed(2)} days`);

        // Update trailing stop if TP1 was hit
        if (position.tp1Hit) {
          position.exitLevels = this.exitManager.updateTrailingStop(position, currentPrice);
        }

        // Move stop to breakeven at +7%
        if (position.pnlPercent >= 0.07 && position.exitLevels.stopLoss.price < position.entryPrice) {
          position.exitLevels = this.exitManager.moveStopToBreakeven(position);
        }

        // Check if should exit
        const decision = this.exitManager.shouldExit(position, currentPrice);

        if (decision.shouldExit) {
          // HYBRID APPROACH: Verify price with Jupiter before executing
          // DexScreener says exit, but let's confirm with Jupiter's real execution price
          let jupiterPrice: number | null = null;
          if (this.spotExecutor) {
            jupiterPrice = await this.spotExecutor.getJupiterPrice(position.token.address);
          }

          if (jupiterPrice) {
            // Recalculate PnL with Jupiter's real price
            const jupiterPnlPercent = (jupiterPrice - position.entryPrice) / position.entryPrice;

            logger.info('[CRTX] Hybrid price verification', {
              symbol: position.token.symbol,
              dexScreenerPrice: currentPrice.toFixed(6),
              jupiterPrice: jupiterPrice.toFixed(6),
              priceDiff: ((jupiterPrice - currentPrice) / currentPrice * 100).toFixed(2) + '%',
              dexScreenerPnl: (position.pnlPercent * 100).toFixed(2) + '%',
              jupiterPnl: (jupiterPnlPercent * 100).toFixed(2) + '%',
            });

            console.log(`\n[AGENT] 🔍 Price verification for ${position.token.symbol}:`);
            console.log(`[AGENT]    DexScreener: $${currentPrice.toFixed(4)} (PnL: ${(position.pnlPercent * 100).toFixed(2)}%)`);
            console.log(`[AGENT]    Jupiter:     $${jupiterPrice.toFixed(4)} (PnL: ${(jupiterPnlPercent * 100).toFixed(2)}%)`);

            // Re-check exit conditions with Jupiter price
            const jupiterDecision = this.exitManager.shouldExit(position, jupiterPrice);

            if (!jupiterDecision.shouldExit) {
              console.log(`[AGENT] ⏸️  Exit cancelled - Jupiter price doesn't confirm exit condition`);
              logger.warn('[CRTX] Exit cancelled after Jupiter verification', {
                symbol: position.token.symbol,
                reason: 'Jupiter price does not confirm exit',
                dexScreenerPrice: currentPrice,
                jupiterPrice: jupiterPrice,
              });
              continue; // Skip this exit, wait for next check
            }

            // Use Jupiter price for final decision
            position.currentPrice = jupiterPrice;
            position.pnlPercent = jupiterPnlPercent;
          }

          logger.info('[CRTX] Spot position exit confirmed', {
            positionId: position.id,
            symbol: position.token.symbol,
            reason: decision.reason,
            exitType: decision.exitType,
            exitSize: decision.exitSize,
            pnlPercent: (position.pnlPercent * 100).toFixed(2) + '%',
            priceSource: jupiterPrice ? 'Jupiter' : 'DexScreener',
          });

          console.log(`\n[AGENT] 🚪 Exiting spot position: ${position.token.symbol}`);
          console.log(`[AGENT]    Reason: ${decision.reason}`);
          console.log(`[AGENT]    PnL: ${(position.pnlPercent * 100).toFixed(2)}%`);
          console.log(`[AGENT]    Exit Size: ${decision.exitSize} tokens`);
          console.log(`[AGENT]    Price Source: ${jupiterPrice ? 'Jupiter (verified)' : 'DexScreener'}`);

          // Execute sell
          await this.exitSpotPosition(position, decision.exitSize || position.remainingSize, decision.exitType!);
        }
      } catch (error) {
        logger.error('[CRTX] Failed to check spot position', {
          positionId: position.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  /**
   * Exit a spot position (sell tokens)
   */
  private async exitSpotPosition(position: SpotPosition, amount: number, exitType: string): Promise<void> {
    if (!this.spotExecutor) {
      logger.error('[CRTX] Cannot exit spot position - no executor');
      return;
    }

    const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';

    try {
      const result = await this.spotExecutor.sell({
        inputMint: position.token.address,
        outputMint: USDC_MINT,
        amountTokens: amount,
      });

      if (result.success && result.signature) {
        // Update position state
        position.remainingSize -= amount;

        // Mark TP levels as hit
        if (exitType === 'tp1') {
          position.tp1Hit = true;
          position.status = 'partial';
        } else if (exitType === 'tp2') {
          position.tp2Hit = true;
          position.status = 'partial';
        } else if (exitType === 'tp3') {
          position.tp3Hit = true;
          position.status = 'closed';
        } else {
          // stop_loss, trailing_stop, time_based - exit all
          position.status = 'closed';
        }

        // If no remaining size, mark as closed
        if (position.remainingSize <= 0) {
          position.status = 'closed';
        }

        console.log(`\n╔═══════════════════════════════════════════════════════════╗`);
        console.log(`║  ✅ SPOT SELL SUCCESSFUL!                                 ║`);
        console.log(`╠═══════════════════════════════════════════════════════════╣`);
        console.log(`║  Token:       ${position.token.symbol.padEnd(43)} ║`);
        console.log(`║  Exit Type:   ${exitType.toUpperCase().padEnd(43)} ║`);
        console.log(`║  Sold:        ${amount.toFixed(4).padEnd(43)} ║`);
        console.log(`║  USD Received: $${(result.usdReceived || 0).toFixed(2).padEnd(40)} ║`);
        console.log(`║  PnL:         ${(position.pnlPercent * 100).toFixed(2)}%${' '.repeat(39 - ((position.pnlPercent * 100).toFixed(2)).length)} ║`);
        console.log(`╠═══════════════════════════════════════════════════════════╣`);
        console.log(`║  🔗 TX: ${result.signature.slice(0, 48).padEnd(48)} ║`);
        console.log(`║  🌐 https://solscan.io/tx/${result.signature.slice(0, 28)}... ║`);
        console.log(`╚═══════════════════════════════════════════════════════════╝\n`);

        // Record PnL
        const pnlUsd = (result.usdReceived || 0) - (amount * position.entryPrice);
        const pnlPct = pnlUsd / (amount * position.entryPrice);
        this.riskManager.recordTrade(pnlPct);

        logger.info('[CRTX] Spot position exited', {
          positionId: position.id,
          symbol: position.token.symbol,
          exitType,
          soldAmount: amount,
          usdReceived: result.usdReceived,
          pnlUsd,
          pnlPct,
          remaining: position.remainingSize,
          status: position.status,
        });
      } else {
        console.log(`[AGENT] ❌ Spot sell failed: ${result.error}`);
        logger.error('[CRTX] Spot sell failed', {
          positionId: position.id,
          error: result.error,
        });
      }
    } catch (error) {
      console.log(`[AGENT] ❌ Spot sell error: ${error}`);
      logger.error('[CRTX] Spot sell error', {
        positionId: position.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Get token price from multiple sources (DexScreener, CoinGecko fallback)
   */
  private async getTokenPrice(tokenMint: string): Promise<number | null> {
    // Try DexScreener first (most reliable for Solana tokens)
    try {
      const dexResponse = await fetch(`https://api.dexscreener.com/latest/dex/tokens/${tokenMint}`);
      const dexData = await dexResponse.json() as { pairs?: Array<{ priceUsd: string }> };
      if (dexData.pairs && dexData.pairs.length > 0 && dexData.pairs[0].priceUsd) {
        const price = parseFloat(dexData.pairs[0].priceUsd);
        logger.info('[CRTX] Got token price from DexScreener', { tokenMint: tokenMint.slice(0, 8), price });
        return price;
      }
    } catch (error) {
      logger.warn('[CRTX] DexScreener price fetch failed', {
        tokenMint: tokenMint.slice(0, 8),
        error: error instanceof Error ? error.message : String(error),
      });
    }

    // Fallback to CoinGecko for known tokens
    try {
      // Map known token mints to CoinGecko IDs
      const mintToCoinGecko: Record<string, string> = {
        '27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4': 'jupiter-perpetuals-liquidity-provider-token', // JLP
        'So11111111111111111111111111111111111111112': 'solana', // SOL
        'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN': 'jupiter-exchange-solana', // JUP
        'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263': 'bonk', // BONK
        'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm': 'dogwifcoin', // WIF
      };

      const coinGeckoId = mintToCoinGecko[tokenMint];
      if (coinGeckoId) {
        const cgResponse = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${coinGeckoId}&vs_currencies=usd`);
        const cgData = await cgResponse.json() as Record<string, { usd: number }>;
        if (cgData[coinGeckoId]?.usd) {
          const price = cgData[coinGeckoId].usd;
          logger.info('[CRTX] Got token price from CoinGecko', { tokenMint: tokenMint.slice(0, 8), coinGeckoId, price });
          return price;
        }
      }
    } catch (error) {
      logger.warn('[CRTX] CoinGecko price fetch failed', {
        tokenMint: tokenMint.slice(0, 8),
        error: error instanceof Error ? error.message : String(error),
      });
    }

    logger.error('[CRTX] Failed to get token price from all sources', { tokenMint: tokenMint.slice(0, 8) });
    return null;
  }

  /**
   * Open a tracked spot position
   */
  openSpotPosition(params: {
    token: ApprovedToken;
    entryPrice: number;
    entrySize: number;
  }): string {
    const id = `spot_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    const exitLevels = this.exitManager.calculateExitLevels(
      params.entryPrice,
      params.entrySize,
      params.token.tier
    );

    const position: SpotPosition = {
      id,
      token: params.token,
      entryPrice: params.entryPrice,
      entrySize: params.entrySize,
      entryTimestamp: Date.now(),
      currentPrice: params.entryPrice,
      currentValue: params.entrySize * params.entryPrice,
      pnl: 0,
      pnlPercent: 0,
      exitLevels,
      tp1Hit: false,
      tp2Hit: false,
      tp3Hit: false,
      remainingSize: params.entrySize,
      daysHeld: 0,
      status: 'open',
    };

    this.spotPositions.set(id, position);

    logger.info('[CRTX] Spot position opened', {
      id,
      symbol: params.token.symbol,
      entryPrice: params.entryPrice,
      entrySize: params.entrySize,
      tp1Price: exitLevels.tp1.price,
      stopLossPrice: exitLevels.stopLoss.price,
    });

    console.log(`[AGENT] 📊 Spot position tracked: ${params.token.symbol}`);
    console.log(`[AGENT]    Entry: $${params.entryPrice.toFixed(4)}`);
    console.log(`[AGENT]    Size: ${params.entrySize} tokens`);
    console.log(`[AGENT]    TP1: $${exitLevels.tp1.price.toFixed(4)} (+12%)`);
    console.log(`[AGENT]    Stop: $${exitLevels.stopLoss.price.toFixed(4)} (-${(exitLevels.stopLoss.percentage * 100).toFixed(0)}%)`);

    return id;
  }

  /**
   * Get all open spot positions
   */
  getOpenSpotPositions(): SpotPosition[] {
    return Array.from(this.spotPositions.values()).filter(p => p.status !== 'closed');
  }

  // ============= LENDING POSITION MONITORING =============

  /**
   * Start lending position monitoring loop
   */
  startLendingMonitoring(): void {
    if (this.lendingMonitoringIntervalId) {
      logger.warn('[CRTX] Lending monitoring already running');
      return;
    }

    const intervalMs = this.lendingPositionMonitor.getCheckInterval();

    logger.info('[CRTX] Starting lending position monitoring', {
      intervalMs,
      intervalMinutes: intervalMs / (60 * 1000),
    });

    // Run immediately
    this.checkLendingPositions().catch(error => {
      logger.error('[CRTX] Lending monitoring check failed', { error });
    });

    // Then run periodically
    this.lendingMonitoringIntervalId = setInterval(() => {
      this.checkLendingPositions().catch(error => {
        logger.error('[CRTX] Lending monitoring check failed', { error });
      });
    }, intervalMs);
  }

  /**
   * Stop lending position monitoring loop
   */
  stopLendingMonitoring(): void {
    if (this.lendingMonitoringIntervalId) {
      clearInterval(this.lendingMonitoringIntervalId);
      this.lendingMonitoringIntervalId = null;
      logger.info('[CRTX] Lending position monitoring stopped');
    }
  }

  /**
   * Check all open lending positions and exit if conditions are met
   */
  private async checkLendingPositions(): Promise<void> {
    const openPositions = Array.from(this.lendingPositions.values()).filter(p => p.status !== 'closed');

    if (openPositions.length === 0) {
      return;
    }

    logger.info('[CRTX] Checking lending positions', {
      count: openPositions.length,
    });

    for (const position of openPositions) {
      try {
        // Get current APY and health factor from protocol
        const currentData = await this.getLendingPositionData(position.protocol, position.asset);
        if (currentData) {
          position.currentApy = currentData.apy;
          position.healthFactor = currentData.healthFactor;
          position.currentValueUsd = currentData.valueUsd || position.depositedUsd;
          position.earnedUsd = position.currentValueUsd - position.depositedUsd;
        }

        // Check if should exit
        const decision = this.lendingPositionMonitor.shouldExit(position);

        if (decision.shouldExit) {
          logger.info('[CRTX] Lending position exit triggered', {
            positionId: position.id,
            protocol: position.protocol,
            asset: position.asset,
            reason: decision.reason,
            exitType: decision.exitType,
          });

          console.log(`\n[AGENT] 🚪 Exiting lending position: ${position.asset} on ${position.protocol}`);
          console.log(`[AGENT]    Reason: ${decision.reason}`);
          console.log(`[AGENT]    Exit: ${decision.exitPercentage}%`);

          // Execute withdrawal
          await this.exitLendingPosition(position, decision.exitPercentage || 100);
        }
      } catch (error) {
        logger.error('[CRTX] Failed to check lending position', {
          positionId: position.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  /**
   * Get current lending position data from DeFiLlama
   */
  private async getLendingPositionData(protocol: string, asset: string): Promise<{ apy: number; healthFactor: number; valueUsd: number } | null> {
    try {
      // Fetch APY from DeFiLlama
      const response = await fetch('https://yields.llama.fi/pools');
      const data = await response.json() as { data: Array<{ project: string; symbol: string; apy: number }> };

      // Find matching pool
      const pool = data.data?.find(p =>
        p.project.toLowerCase().includes(protocol.toLowerCase()) &&
        p.symbol.toUpperCase().includes(asset.toUpperCase())
      );

      return {
        apy: pool ? pool.apy / 100 : 0, // Convert from percentage to decimal
        healthFactor: 1.5, // Default healthy (would need protocol-specific query)
        valueUsd: 0, // Would need position query
      };
    } catch (error) {
      logger.error('[CRTX] Failed to get lending position data', { protocol, asset, error });
      return null;
    }
  }

  /**
   * Exit a lending position (withdraw)
   */
  private async exitLendingPosition(position: TrackedLendingPosition, percentage: number): Promise<void> {
    if (!this.lendingExecutor) {
      logger.error('[CRTX] Cannot exit lending position - no executor');
      return;
    }

    try {
      const withdrawAmount = (position.depositedUsd * percentage) / 100;

      const result = await this.lendingExecutor.withdraw({
        protocol: position.protocol as 'kamino' | 'marginfi' | 'solend',
        asset: position.asset,
        amountUsd: withdrawAmount,
      });

      if (result.success && result.signature) {
        // Update position state
        if (percentage >= 100) {
          position.status = 'closed';
        } else {
          position.depositedUsd -= withdrawAmount;
        }

        const pnlUsd = position.earnedUsd * (percentage / 100);
        const pnlPct = position.depositedUsd > 0 ? pnlUsd / position.depositedUsd : 0;

        console.log(`\n╔═══════════════════════════════════════════════════════════╗`);
        console.log(`║  ✅ LENDING WITHDRAWAL SUCCESSFUL!                        ║`);
        console.log(`╠═══════════════════════════════════════════════════════════╣`);
        console.log(`║  Asset:       ${position.asset.padEnd(43)} ║`);
        console.log(`║  Protocol:    ${position.protocol.padEnd(43)} ║`);
        console.log(`║  Withdrawn:   $${withdrawAmount.toFixed(2).padEnd(41)} ║`);
        console.log(`║  PnL:         $${pnlUsd.toFixed(2).padEnd(41)} ║`);
        console.log(`╠═══════════════════════════════════════════════════════════╣`);
        console.log(`║  🔗 TX: ${result.signature.slice(0, 48).padEnd(48)} ║`);
        console.log(`╚═══════════════════════════════════════════════════════════╝\n`);

        this.riskManager.recordTrade(pnlPct);

        logger.info('[CRTX] Lending position exited', {
          positionId: position.id,
          protocol: position.protocol,
          asset: position.asset,
          withdrawnUsd: withdrawAmount,
          pnlUsd,
          status: position.status,
        });
      } else {
        console.log(`[AGENT] ❌ Lending withdrawal failed: ${result.error}`);
        logger.error('[CRTX] Lending withdrawal failed', {
          positionId: position.id,
          error: result.error,
        });
      }
    } catch (error) {
      console.log(`[AGENT] ❌ Lending withdrawal error: ${error}`);
      logger.error('[CRTX] Lending withdrawal error', {
        positionId: position.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Open a tracked lending position
   */
  openLendingPosition(params: {
    protocol: string;
    asset: string;
    depositedUsd: number;
    entryApy: number;
  }): string {
    const id = `lending_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    const position: TrackedLendingPosition = {
      id,
      protocol: params.protocol,
      asset: params.asset,
      depositedUsd: params.depositedUsd,
      entryApy: params.entryApy,
      currentApy: params.entryApy,
      entryTime: Date.now(),
      currentValueUsd: params.depositedUsd,
      earnedUsd: 0,
      healthFactor: 1.5, // Default healthy
      status: 'open',
    };

    this.lendingPositions.set(id, position);

    logger.info('[CRTX] Lending position opened', {
      id,
      protocol: params.protocol,
      asset: params.asset,
      depositedUsd: params.depositedUsd,
      entryApy: (params.entryApy * 100).toFixed(2) + '%',
    });

    console.log(`[AGENT] 📊 Lending position tracked: ${params.asset} on ${params.protocol}`);
    console.log(`[AGENT]    Deposited: $${params.depositedUsd.toFixed(2)}`);
    console.log(`[AGENT]    Entry APY: ${(params.entryApy * 100).toFixed(2)}%`);
    console.log(`[AGENT]    Min APY Exit: 2%`);
    console.log(`[AGENT]    Max Hold: 30 days`);

    return id;
  }

  /**
   * Get all open lending positions
   */
  getOpenLendingPositions(): TrackedLendingPosition[] {
    return Array.from(this.lendingPositions.values()).filter(p => p.status !== 'closed');
  }
}