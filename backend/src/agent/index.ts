import { CortexSolanaAgent } from "./solana-agent.js";
import { config } from "../config/index.js";
import { createChildLogger } from "../lib/logger.js";
import type { AgentConfig, AgentLimits } from "./types.js";
import { DEFAULT_AGENT_LIMITS } from "./types.js";

const log = createChildLogger({ module: "agent" });

let agentInstance: CortexSolanaAgent | null = null;

export function initializeAgent(customLimits?: Partial<AgentLimits>): CortexSolanaAgent {
  const privateKey = process.env.AGENT_WALLET_PRIVATE_KEY;
  
  if (!privateKey) {
    throw new Error("AGENT_WALLET_PRIVATE_KEY environment variable is required");
  }

  const agentConfig: AgentConfig = {
    rpcUrl: process.env.SOLANA_RPC_URL ?? "https://api.mainnet-beta.solana.com",
    openaiApiKey: process.env.OPENAI_API_KEY,
    heliusApiKey: process.env.HELIUS_API_KEY,
  };

  const limits: AgentLimits = {
    ...DEFAULT_AGENT_LIMITS,
    ...customLimits,
  };

  agentInstance = new CortexSolanaAgent(privateKey, agentConfig, limits);
  
  log.info({ publicKey: agentInstance.getPublicKey(), maxTradeUsd: limits.maxTradeAmountUsd, dailyLimitUsd: limits.dailyTradeLimitUsd }, "Agent initialized");

  return agentInstance;
}

export function getAgent(): CortexSolanaAgent {
  if (!agentInstance) {
    throw new Error("Agent not initialized. Call initializeAgent() first.");
  }
  return agentInstance;
}

export function isAgentInitialized(): boolean {
  return agentInstance !== null;
}

export { CortexSolanaAgent } from "./solana-agent.js";
export { AgentWalletManager } from "./wallet-manager.js";
export * from "./types.js";

