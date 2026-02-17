import { Router, Request, Response } from "express";
import { PublicKey } from "@solana/web3.js";
import { solanaService } from "../services/solana.js";
import { createChildLogger } from "../lib/logger.js";

const log = createChildLogger({ module: "api:solana" });

const router = Router();

router.get("/vault/:address", async (req: Request, res: Response) => {
  try {
    const { address } = req.params;
    const pubkey = new PublicKey(address);
    const vaultData = await solanaService.getVaultData(pubkey);
    
    if (!vaultData) {
      res.status(404).json({ error: "Vault not found" });
      return;
    }

    res.json({
      authority: vaultData.authority.toBase58(),
      guardian: vaultData.guardian.toBase58(),
      agent: vaultData.agent.toBase58(),
      assetMint: vaultData.assetMint.toBase58(),
      shareMint: vaultData.shareMint.toBase58(),
      assetVault: vaultData.assetVault.toBase58(),
      treasury: vaultData.treasury.toBase58(),
      totalAssets: vaultData.totalAssets.toString(),
      totalShares: vaultData.totalShares.toString(),
      performanceFee: vaultData.performanceFee,
      state: vaultData.state,
    });
  } catch (error) {
    log.error({ err: error }, "Error fetching vault");
    res.status(400).json({ error: "Invalid vault address" });
  }
});

router.get("/staking/pool", async (_req: Request, res: Response) => {
  try {
    const poolData = await solanaService.getStakingPoolData();
    
    if (!poolData) {
      res.status(404).json({ error: "Staking pool not found" });
      return;
    }

    res.json({
      authority: poolData.authority.toBase58(),
      stakeMint: poolData.stakeMint.toBase58(),
      stakeVault: poolData.stakeVault.toBase58(),
      totalStaked: poolData.totalStaked.toString(),
      totalWeight: poolData.totalWeight.toString(),
      tierThresholds: poolData.tierThresholds.map((t) => t.toString()),
      rewardRate: poolData.rewardRate.toString(),
      lastUpdateTime: poolData.lastUpdateTime.toString(),
    });
  } catch (error) {
    log.error({ err: error }, "Error fetching staking pool");
    res.status(500).json({ error: "Failed to fetch staking pool" });
  }
});

router.get("/staking/user/:address", async (req: Request, res: Response) => {
  try {
    const { address } = req.params;
    const pubkey = new PublicKey(address);
    const stakeInfo = await solanaService.getStakeInfo(pubkey);
    
    if (!stakeInfo) {
      res.json({
        staked: false,
        amount: "0",
        lockEnd: "0",
        weight: "0",
        pendingRewards: "0",
      });
      return;
    }

    res.json({
      staked: true,
      owner: stakeInfo.owner.toBase58(),
      amount: stakeInfo.amount.toString(),
      lockEnd: stakeInfo.lockEnd.toString(),
      weight: stakeInfo.weight.toString(),
      cooldownStart: stakeInfo.cooldownStart.toString(),
      rewardDebt: stakeInfo.rewardDebt.toString(),
      pendingRewards: stakeInfo.pendingRewards.toString(),
    });
  } catch (error) {
    log.error({ err: error }, "Error fetching stake info");
    res.status(400).json({ error: "Invalid address" });
  }
});

router.get("/programs", (_req: Request, res: Response) => {
  const programIds = solanaService.getProgramIds();
  res.json({
    cortex: programIds.cortex.toBase58(),
    staking: programIds.staking.toBase58(),
    vault: programIds.vault.toBase58(),
    strategy: programIds.strategy.toBase58(),
    treasury: programIds.treasury.toBase58(),
  });
});

export default router;

