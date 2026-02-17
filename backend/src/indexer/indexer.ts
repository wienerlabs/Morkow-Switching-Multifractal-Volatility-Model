import { type Address, type Log, encodeEventTopics } from "viem";
import { getPublicClient } from "../services/blockchain.js";
import { VAULT_ABI, decodeVaultEvent, type DecodedEvent } from "./events.js";
import { prisma } from "../lib/prisma.js";
import { createChildLogger } from "../lib/logger.js";

const indexerLog = createChildLogger({ module: "indexer" });
const BATCH_SIZE = 1000;
const POLL_INTERVAL_MS = 12_000;

interface IndexerConfig {
  vaultAddresses: Address[];
  startBlock?: bigint;
}

export class EventIndexer {
  private vaultAddresses: Address[];
  private isRunning = false;
  private startBlock: bigint;

  constructor(config: IndexerConfig) {
    this.vaultAddresses = config.vaultAddresses;
    this.startBlock = config.startBlock ?? 0n;
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    this.isRunning = true;
    indexerLog.info({ vaultCount: this.vaultAddresses.length }, "Indexer starting");
    this.poll();
  }

  stop(): void {
    this.isRunning = false;
    indexerLog.info("Indexer stopped");
  }

  private async poll(): Promise<void> {
    while (this.isRunning) {
      try {
        await this.processBlocks();
      } catch (err) {
        indexerLog.error({ err }, "Indexer poll error");
      }
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    }
  }

  private async processBlocks(): Promise<void> {
    const client = getPublicClient();
    const state = await prisma.indexerState.upsert({
      where: { id: "singleton" },
      create: { id: "singleton", lastBlockNumber: Number(this.startBlock) },
      update: {},
    });

    const fromBlock = BigInt(state.lastBlockNumber) + 1n;
    const latestBlock = await client.getBlockNumber();

    if (fromBlock > latestBlock) return;

    const toBlock = fromBlock + BigInt(BATCH_SIZE) > latestBlock
      ? latestBlock
      : fromBlock + BigInt(BATCH_SIZE);

    indexerLog.debug({ fromBlock: fromBlock.toString(), toBlock: toBlock.toString() }, "Processing blocks");

    const logs = await client.getLogs({
      address: this.vaultAddresses,
      fromBlock,
      toBlock,
      events: VAULT_ABI,
    });

    for (const log of logs) {
      await this.processLog(log);
    }

    await prisma.indexerState.update({
      where: { id: "singleton" },
      data: { lastBlockNumber: Number(toBlock) },
    });

    indexerLog.info({ eventCount: logs.length, toBlock: toBlock.toString() }, "Processed events");
  }

  private async processLog(log: Log): Promise<void> {
    const decoded = decodeVaultEvent(log);
    if (!decoded) return;

    const vaultAddress = log.address.toLowerCase();
    const vault = await prisma.vault.findUnique({
      where: { address: vaultAddress },
    });

    if (!vault) {
      indexerLog.warn({ vaultAddress }, "Unknown vault");
      return;
    }

    const block = await getPublicClient().getBlock({ blockNumber: log.blockNumber! });
    const timestamp = new Date(Number(block.timestamp) * 1000);

    switch (decoded.type) {
      case "deposit":
        await this.handleDeposit(vault.id, log, decoded, timestamp);
        break;
      case "withdraw":
        await this.handleWithdraw(vault.id, log, decoded, timestamp);
        break;
      case "harvest":
        await this.handleHarvest(vault.id, log, decoded, timestamp);
        break;
    }
  }

  private async handleDeposit(
    vaultId: string,
    log: Log,
    event: { owner: Address; assets: bigint; shares: bigint },
    timestamp: Date
  ): Promise<void> {
    await prisma.deposit.upsert({
      where: { txHash: log.transactionHash! },
      create: {
        vaultId,
        user: event.owner.toLowerCase(),
        assets: event.assets.toString(),
        shares: event.shares.toString(),
        txHash: log.transactionHash!,
        blockNumber: Number(log.blockNumber!),
        timestamp,
      },
      update: {},
    });
  }

  private async handleWithdraw(
    vaultId: string,
    log: Log,
    event: { owner: Address; assets: bigint; shares: bigint },
    timestamp: Date
  ): Promise<void> {
    await prisma.withdrawal.upsert({
      where: { txHash: log.transactionHash! },
      create: {
        vaultId,
        user: event.owner.toLowerCase(),
        assets: event.assets.toString(),
        shares: event.shares.toString(),
        txHash: log.transactionHash!,
        blockNumber: Number(log.blockNumber!),
        timestamp,
      },
      update: {},
    });
  }

  private async handleHarvest(
    vaultId: string,
    log: Log,
    event: { strategy: Address; profit: bigint; fee: bigint },
    timestamp: Date
  ): Promise<void> {
    const strategy = await prisma.strategy.findUnique({
      where: { address: event.strategy.toLowerCase() },
    });
    if (!strategy) return;

    await prisma.harvest.upsert({
      where: { txHash: log.transactionHash! },
      create: {
        vaultId,
        strategyId: strategy.id,
        profit: event.profit.toString(),
        fee: event.fee.toString(),
        txHash: log.transactionHash!,
        blockNumber: Number(log.blockNumber!),
        timestamp,
      },
      update: {},
    });
  }
}

