import { z } from "zod";
import "dotenv/config";

const envSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.string().default("3001").transform(Number),
  DATABASE_URL: z.string(),

  // EVM
  RPC_URL: z.string().url(),
  CHAIN_ID: z.string().default("1").transform(Number),

  // Solana
  SOLANA_RPC_URL: z.string().url().optional(),
  SOLANA_NETWORK: z.enum(["mainnet-beta", "devnet", "testnet"]).default("devnet"),

  VAULT_ADDRESS: z.string().optional(),
  TREASURY_ADDRESS: z.string().optional(),

  AGENT_PRIVATE_KEY: z.string().optional(),
  GUARDIAN_ADDRESS: z.string().optional(),

  CORS_ORIGIN: z.string().default("*"),

  // Auth — Solana public key authorized to call mutating endpoints
  AUTHORIZED_PUBLIC_KEY: z.string().optional(),

  LOG_LEVEL: z.enum(["fatal", "error", "warn", "info", "debug", "trace"]).default("info"),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  process.stderr.write(`Invalid environment variables: ${JSON.stringify(parsed.error.flatten().fieldErrors)}\n`);
  process.exit(1);
}

export const config = parsed.data;
export type Config = typeof config;

