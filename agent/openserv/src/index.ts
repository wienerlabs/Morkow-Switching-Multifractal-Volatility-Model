import "dotenv/config";
import { Agent, run } from "@openserv-labs/sdk";
import { z } from "zod";

const CORTEX_API_URL =
  process.env.CORTEX_API_URL || "http://localhost:8000";

const agent = new Agent({
  systemPrompt:
    "You are Cortex Narrator — the voice of the Cortex-A LAMS risk engine. " +
    "You translate quantitative risk assessments, market signals, news events, " +
    "and DX Research intelligence (agent coordination, herd detection, vault state, " +
    "human overrides) into clear, actionable narratives for human operators.",
  apiKey: process.env.OPENSERV_API_KEY!,
});

// ── 1. explain-trade-decision ──────────────────────────────────────

agent.addCapability({
  name: "explain-trade-decision",
  description:
    "Generate an LLM narrative explaining a guardian trade assessment. " +
    "Provide token, direction, trade_size_usd. Optionally provide a full assessment dict.",
  inputSchema: z.object({
    token: z.string().describe("Token symbol, e.g. SOL"),
    direction: z
      .string()
      .default("long")
      .describe("Trade direction: long or short"),
    trade_size_usd: z
      .number()
      .default(0)
      .describe("Proposed trade size in USD"),
  }),
  async run({ args }) {
    try {
      const resp = await fetch(`${CORTEX_API_URL}/api/v1/narrator/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token: args.token,
          direction: args.direction,
          trade_size_usd: args.trade_size_usd,
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        return `Error from Cortex API (${resp.status}): ${text}`;
      }

      const data = await resp.json();
      return data.narrative ?? "No narrative returned.";
    } catch (err) {
      return `Failed to reach Cortex API: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
});

// ── 2. interpret-news ──────────────────────────────────────────────

agent.addCapability({
  name: "interpret-news",
  description:
    "Interpret recent news items through Cortex's LLM narrator. " +
    "Optionally provide news_items array and news_signal object.",
  inputSchema: z.object({
    news_items: z
      .array(z.record(z.unknown()))
      .optional()
      .describe("Array of news item dicts. If omitted, reads from buffer."),
    news_signal: z
      .record(z.unknown())
      .optional()
      .describe("Aggregate signal dict. If omitted, reads from buffer."),
  }),
  async run({ args }) {
    try {
      const resp = await fetch(`${CORTEX_API_URL}/api/v1/narrator/news`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          news_items: args.news_items ?? null,
          news_signal: args.news_signal ?? null,
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        return `Error from Cortex API (${resp.status}): ${text}`;
      }

      const data = await resp.json();
      return data.interpretation ?? "No interpretation returned.";
    } catch (err) {
      return `Failed to reach Cortex API: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
});

// ── 3. market-briefing ─────────────────────────────────────────────

agent.addCapability({
  name: "market-briefing",
  description:
    "Generate a comprehensive market briefing from all risk model outputs. No input required.",
  inputSchema: z.object({}),
  async run() {
    try {
      const resp = await fetch(
        `${CORTEX_API_URL}/api/v1/narrator/briefing`,
        { method: "GET" },
      );

      if (!resp.ok) {
        const text = await resp.text();
        return `Error from Cortex API (${resp.status}): ${text}`;
      }

      const data = await resp.json();
      return data.briefing ?? "No briefing returned.";
    } catch (err) {
      return `Failed to reach Cortex API: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
});

// ── 4. ask-cortex ──────────────────────────────────────────────────

agent.addCapability({
  name: "ask-cortex",
  description:
    "Ask the Cortex narrator a free-form question about the system's current state.",
  inputSchema: z.object({
    question: z.string().min(1).describe("Your question about the system"),
  }),
  async run({ args }) {
    try {
      const resp = await fetch(`${CORTEX_API_URL}/api/v1/narrator/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: args.question }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        return `Error from Cortex API (${resp.status}): ${text}`;
      }

      const data = await resp.json();
      return data.answer ?? "No answer returned.";
    } catch (err) {
      return `Failed to reach Cortex API: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
});

// ── 5. dx-intelligence ────────────────────────────────────────────

agent.addCapability({
  name: "dx-intelligence",
  description:
    "Get DX Research module state: stigmergy consensus, Ising cascade risk, " +
    "vault deltas, active overrides, and feature flags. Provide a token for " +
    "token-specific data, or omit for a system-wide overview.",
  inputSchema: z.object({
    token: z
      .string()
      .optional()
      .describe("Token symbol for token-specific data (e.g. SOL)"),
    vault_id: z
      .string()
      .optional()
      .describe("Vault ID for vault delta data"),
  }),
  async run({ args }) {
    try {
      const results: Record<string, unknown> = {};

      // DX status (feature flags)
      const statusResp = await fetch(`${CORTEX_API_URL}/api/v1/dx/status`);
      if (statusResp.ok) results.status = await statusResp.json();

      // Stigmergy
      if (args.token) {
        const stigResp = await fetch(
          `${CORTEX_API_URL}/api/v1/dx/stigmergy/${args.token}`,
        );
        if (stigResp.ok) results.stigmergy = await stigResp.json();
      } else {
        const stigResp = await fetch(
          `${CORTEX_API_URL}/api/v1/dx/stigmergy`,
        );
        if (stigResp.ok) results.stigmergy = await stigResp.json();
      }

      // Ising cascade (requires token)
      if (args.token) {
        const cascResp = await fetch(
          `${CORTEX_API_URL}/api/v1/dx/cascade/${args.token}`,
        );
        if (cascResp.ok) results.cascade = await cascResp.json();
      }

      // Vault delta
      if (args.vault_id) {
        const vaultResp = await fetch(
          `${CORTEX_API_URL}/api/v1/dx/vault/${args.vault_id}`,
        );
        if (vaultResp.ok) results.vault = await vaultResp.json();
      }

      // Active overrides
      const ovrResp = await fetch(`${CORTEX_API_URL}/api/v1/dx/overrides`);
      if (ovrResp.ok) results.overrides = await ovrResp.json();

      return JSON.stringify(results, null, 2);
    } catch (err) {
      return `Failed to reach Cortex API: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
});

// ── 6. manage-override ───────────────────────────────────────────

agent.addCapability({
  name: "manage-override",
  description:
    "Create or revoke a human override. Actions: force_approve, force_reject, " +
    "size_cap, cooldown. Use revoke_id to revoke an existing override.",
  inputSchema: z.object({
    action: z
      .enum(["force_approve", "force_reject", "size_cap", "cooldown"])
      .optional()
      .describe("Override action (required for create)"),
    token: z
      .string()
      .default("*")
      .describe("Token symbol or * for global"),
    reason: z.string().default("").describe("Reason for override"),
    created_by: z.string().default("openserv-agent").describe("Who created it"),
    ttl: z.number().optional().describe("TTL in seconds"),
    size_cap_usd: z
      .number()
      .optional()
      .describe("Size cap in USD (for size_cap action)"),
    revoke_id: z
      .string()
      .optional()
      .describe("Override ID to revoke (mutually exclusive with action)"),
  }),
  async run({ args }) {
    try {
      // Revoke path
      if (args.revoke_id) {
        const resp = await fetch(
          `${CORTEX_API_URL}/api/v1/dx/overrides/${args.revoke_id}?revoked_by=${encodeURIComponent(args.created_by)}`,
          { method: "DELETE" },
        );
        if (!resp.ok) {
          const text = await resp.text();
          return `Revoke failed (${resp.status}): ${text}`;
        }
        return JSON.stringify(await resp.json(), null, 2);
      }

      // Create path
      if (!args.action) {
        return "Error: either 'action' (for create) or 'revoke_id' (for revoke) is required.";
      }

      const resp = await fetch(`${CORTEX_API_URL}/api/v1/dx/overrides`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: args.action,
          token: args.token,
          reason: args.reason,
          created_by: args.created_by,
          ttl: args.ttl ?? null,
          size_cap_usd: args.size_cap_usd ?? null,
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        return `Create failed (${resp.status}): ${text}`;
      }

      return JSON.stringify(await resp.json(), null, 2);
    } catch (err) {
      return `Failed to reach Cortex API: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
});

// ── Start agent in tunnel mode ─────────────────────────────────────

run(agent).then(({ stop }) => {
  console.log(`Cortex Narrator agent running on port ${process.env.PORT ?? 7378}`);

  process.on("SIGINT", async () => {
    await stop();
    process.exit(0);
  });
});
