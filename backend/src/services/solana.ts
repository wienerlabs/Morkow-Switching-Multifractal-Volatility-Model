import { Connection, PublicKey, Keypair, clusterApiUrl } from "@solana/web3.js";
import { Program, AnchorProvider, Wallet, Idl, BN } from "@coral-xyz/anchor";
import { createChildLogger } from "../lib/logger.js";

const log = createChildLogger({ module: "solana" });

const PROGRAM_IDS = {
  cortex: new PublicKey("9qeww9s48ByyxzPuoiwhvYfB9dfG4Q3a1VnJrYcahwo6"),
  staking: new PublicKey("rYantWFyB4PsL36r9XB7nUb8TQ1pAhn9A87S6TbpMsr"),
  vault: new PublicKey("7nVZjbjEiJVPzaKBPDWJ5dPVMmcEBduDtiWi9o67nosS"),
  strategy: new PublicKey("WpEYz9eHtu3jhZkLsfgyyFpWR5C7bjswktpso5BWeQS"),
  treasury: new PublicKey("GsMtBFGq3DWFGDqtJ6Fb4LjdB6sBADKL2Lix9xaYV7GS"),
};

export class SolanaService {
  private connection: Connection;
  private provider: AnchorProvider | null = null;

  constructor(rpcUrl?: string) {
    this.connection = new Connection(
      rpcUrl ?? process.env.SOLANA_RPC_URL ?? clusterApiUrl("devnet"),
      "confirmed"
    );
  }

  initializeProvider(keypair: Keypair): void {
    const wallet = new Wallet(keypair);
    this.provider = new AnchorProvider(this.connection, wallet, {
      commitment: "confirmed",
    });
  }

  async getVaultData(vaultPubkey: PublicKey): Promise<VaultData | null> {
    try {
      const accountInfo = await this.connection.getAccountInfo(vaultPubkey);
      if (!accountInfo) return null;

      // Decode vault account data (skip 8-byte discriminator)
      const data = accountInfo.data.slice(8);
      return {
        authority: new PublicKey(data.slice(0, 32)),
        guardian: new PublicKey(data.slice(32, 64)),
        agent: new PublicKey(data.slice(64, 96)),
        assetMint: new PublicKey(data.slice(96, 128)),
        shareMint: new PublicKey(data.slice(128, 160)),
        assetVault: new PublicKey(data.slice(160, 192)),
        treasury: new PublicKey(data.slice(192, 224)),
        totalAssets: readU64(data, 256),
        totalShares: readU64(data, 264),
        performanceFee: data.readUInt16LE(272),
        state: data[274],
      };
    } catch (error) {
      log.error({ err: error, vault: vaultPubkey.toBase58() }, "Failed to fetch vault data");
      return null;
    }
  }

  async getStakeInfo(userPubkey: PublicKey): Promise<StakeInfo | null> {
    const [stakeInfoPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("stake_info"), userPubkey.toBuffer()],
      PROGRAM_IDS.staking
    );
    
    try {
      const accountInfo = await this.connection.getAccountInfo(stakeInfoPDA);
      if (!accountInfo) return null;

      const data = accountInfo.data.slice(8);
      return {
        owner: new PublicKey(data.slice(0, 32)),
        amount: readU64(data, 32),
        lockEnd: readI64(data, 40),
        weight: readU64(data, 48),
        cooldownStart: readI64(data, 56),
        rewardDebt: readU64(data, 64),
        pendingRewards: readU64(data, 72),
      };
    } catch {
      return null;
    }
  }

  async getStakingPoolData(): Promise<StakingPoolData | null> {
    const [poolPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("staking_pool")],
      PROGRAM_IDS.staking
    );
    
    try {
      const accountInfo = await this.connection.getAccountInfo(poolPDA);
      if (!accountInfo) return null;

      const data = accountInfo.data.slice(8);
      return {
        authority: new PublicKey(data.slice(0, 32)),
        stakeMint: new PublicKey(data.slice(32, 64)),
        stakeVault: new PublicKey(data.slice(64, 96)),
        totalStaked: readU64(data, 96),
        totalWeight: readU64(data, 104),
        tierThresholds: [
          readU64(data, 112),
          readU64(data, 120),
          readU64(data, 128),
        ],
        rewardRate: readU64(data, 136),
        lastUpdateTime: readI64(data, 144),
      };
    } catch {
      return null;
    }
  }

  getConnection(): Connection {
    return this.connection;
  }

  getProgramIds() {
    return PROGRAM_IDS;
  }
}

function readU64(data: Buffer, offset: number): bigint {
  return data.readBigUInt64LE(offset);
}

function readI64(data: Buffer, offset: number): bigint {
  return data.readBigInt64LE(offset);
}

interface VaultData {
  authority: PublicKey;
  guardian: PublicKey;
  agent: PublicKey;
  assetMint: PublicKey;
  shareMint: PublicKey;
  assetVault: PublicKey;
  treasury: PublicKey;
  totalAssets: bigint;
  totalShares: bigint;
  performanceFee: number;
  state: number;
}

interface StakeInfo {
  owner: PublicKey;
  amount: bigint;
  lockEnd: bigint;
  weight: bigint;
  cooldownStart: bigint;
  rewardDebt: bigint;
  pendingRewards: bigint;
}

interface StakingPoolData {
  authority: PublicKey;
  stakeMint: PublicKey;
  stakeVault: PublicKey;
  totalStaked: bigint;
  totalWeight: bigint;
  tierThresholds: [bigint, bigint, bigint];
  rewardRate: bigint;
  lastUpdateTime: bigint;
}

export const solanaService = new SolanaService();

