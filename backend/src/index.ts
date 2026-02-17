import express from "express";
import cors from "cors";
import { config } from "./config/index.js";
import { logger } from "./lib/logger.js";
import { requestIdMiddleware } from "./middleware/requestId.js";
import healthRouter from "./api/health.js";
import vaultsRouter from "./api/vaults.js";
import relayRouter from "./api/relay.js";
import solanaRouter from "./api/solana.js";
import agentRouter from "./api/agent.js";
import orchestratorRouter from "./routes/orchestrator.js";
import { initializeAgent } from "./agent/index.js";
import { getOrchestrator } from "./agent/orchestrator/index.js";
import { solanaAuth } from "./middleware/solanaAuth.js";

const app = express();

app.use(cors({ origin: config.CORS_ORIGIN }));
app.use(express.json());
app.use(requestIdMiddleware);

app.use("/api/health", healthRouter);
app.use("/api/vaults", vaultsRouter);
app.use("/api/relay", solanaAuth, relayRouter);
app.use("/api/solana", solanaRouter);
app.use("/api/agent", solanaAuth, agentRouter);
app.use("/api/orchestrator", orchestratorRouter);

app.use((_req, res) => {
  res.status(404).json({ success: false, error: "Not found" });
});

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  logger.error({ err }, "Unhandled error");
  res.status(500).json({ success: false, error: "Internal server error" });
});

app.listen(config.PORT, async () => {
  logger.info({ port: config.PORT, env: config.NODE_ENV, chainId: config.CHAIN_ID }, "Cortex Backend started");

  if (process.env.AGENT_WALLET_PRIVATE_KEY) {
    try {
      initializeAgent();
      logger.info("AI Agent initialized");
    } catch (error) {
      logger.warn({ err: error }, "Failed to initialize agent");
    }
  } else {
    logger.info("Agent not initialized (AGENT_WALLET_PRIVATE_KEY not set)");
  }

  try {
    const orchestrator = getOrchestrator();
    await orchestrator.initialize();
    const status = await orchestrator.getStatus();
    logger.info({ mlAgentAvailable: status.mlAgentAvailable }, "Orchestrator initialized");
  } catch (error) {
    logger.warn({ err: error }, "Orchestrator init warning");
  }
});

export default app;

