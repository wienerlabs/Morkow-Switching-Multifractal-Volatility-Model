import "dotenv/config";
import { EventIndexer } from "./indexer.js";
import type { Address } from "viem";
import { prisma } from "../lib/prisma.js";
import { createChildLogger } from "../lib/logger.js";

const log = createChildLogger({ module: "indexer-cli" });

async function main(): Promise<void> {
  log.info("Indexer CLI starting");

  const vaults = await prisma.vault.findMany({
    select: { address: true },
  });

  if (vaults.length === 0) {
    log.info("No vaults found in database. Add vaults first.");
    process.exit(0);
  }

  const vaultAddresses = vaults.map((v) => v.address as Address);
  log.info({ vaultCount: vaultAddresses.length }, "Found vaults");

  const startBlock = process.env.INDEXER_START_BLOCK
    ? BigInt(process.env.INDEXER_START_BLOCK)
    : undefined;

  const indexer = new EventIndexer({ vaultAddresses, startBlock });

  process.on("SIGINT", () => {
    log.info("Shutting down");
    indexer.stop();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    indexer.stop();
    process.exit(0);
  });

  await indexer.start();
}

main().catch((err) => {
  log.fatal({ err }, "Fatal error");
  process.exit(1);
});

