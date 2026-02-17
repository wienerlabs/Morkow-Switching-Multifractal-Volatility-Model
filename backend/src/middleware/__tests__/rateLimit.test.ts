import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Request, Response, NextFunction } from "express";
import { rateLimitMiddleware } from "../rateLimit.js";

function makeReq(overrides: Partial<Request> = {}): Request {
  return {
    path: "/api/vaults",
    method: "GET",
    headers: {},
    ip: "127.0.0.1",
    ...overrides,
  } as unknown as Request;
}

function makeRes(): Response {
  const headers: Record<string, string> = {};
  const res = {
    setHeader: vi.fn((k: string, v: string) => { headers[k] = v; }),
    status: vi.fn().mockReturnThis(),
    json: vi.fn(),
    _headers: headers,
  };
  return res as unknown as Response;
}

describe("rateLimitMiddleware", () => {
  beforeEach(() => {
    // Reset module-level buckets by re-importing wouldn't work easily,
    // so we set NODE_ENV to test to skip rate limiting for most tests.
    // For rate-limit-specific tests, we override NODE_ENV.
    vi.unstubAllEnvs();
  });

  it("skips rate limiting in test environment", () => {
    vi.stubEnv("NODE_ENV", "test");
    const next = vi.fn();
    rateLimitMiddleware(makeReq(), makeRes(), next);
    expect(next).toHaveBeenCalled();
  });

  it("skips exempt paths", () => {
    vi.stubEnv("NODE_ENV", "production");
    const next = vi.fn();
    rateLimitMiddleware(makeReq({ path: "/api/health" }), makeRes(), next);
    expect(next).toHaveBeenCalled();
  });

  it("extracts wallet from x-solana-pubkey header", () => {
    vi.stubEnv("NODE_ENV", "production");
    const next = vi.fn();
    const res = makeRes();

    rateLimitMiddleware(
      makeReq({ headers: { "x-solana-pubkey": "WalletABC" } as any }),
      res,
      next,
    );

    expect(next).toHaveBeenCalled();
    expect(res.setHeader).toHaveBeenCalledWith("X-RateLimit-Limit", expect.any(String));
    expect(res.setHeader).toHaveBeenCalledWith("X-RateLimit-Remaining", expect.any(String));
  });

  it("different wallets get independent limits", () => {
    vi.stubEnv("NODE_ENV", "production");
    const nextA = vi.fn();
    const nextB = vi.fn();

    // Exhaust wallet A's write limit (10 writes)
    for (let i = 0; i < 10; i++) {
      const n = vi.fn();
      rateLimitMiddleware(
        makeReq({ method: "POST", headers: { "x-solana-pubkey": "WalletExhaust" } as any }),
        makeRes(),
        n,
      );
    }

    // Wallet A should be rate limited now
    const resA = makeRes();
    rateLimitMiddleware(
      makeReq({ method: "POST", headers: { "x-solana-pubkey": "WalletExhaust" } as any }),
      resA,
      nextA,
    );
    expect(nextA).not.toHaveBeenCalled();
    expect(resA.status).toHaveBeenCalledWith(429);

    // Wallet B should still work
    rateLimitMiddleware(
      makeReq({ method: "POST", headers: { "x-solana-pubkey": "WalletFresh" } as any }),
      makeRes(),
      nextB,
    );
    expect(nextB).toHaveBeenCalled();
  });

  it("returns 429 with retry-after when limit exceeded", () => {
    vi.stubEnv("NODE_ENV", "production");
    const wallet = "Wallet429Test";

    for (let i = 0; i < 10; i++) {
      rateLimitMiddleware(
        makeReq({ method: "POST", headers: { "x-solana-pubkey": wallet } as any }),
        makeRes(),
        vi.fn(),
      );
    }

    const res = makeRes();
    const next = vi.fn();
    rateLimitMiddleware(
      makeReq({ method: "POST", headers: { "x-solana-pubkey": wallet } as any }),
      res,
      next,
    );

    expect(next).not.toHaveBeenCalled();
    expect(res.status).toHaveBeenCalledWith(429);
    expect(res.json).toHaveBeenCalledWith(
      expect.objectContaining({ error: "Rate limit exceeded" }),
    );
    expect(res.setHeader).toHaveBeenCalledWith("Retry-After", expect.any(String));
  });

  it("falls back to IP when no wallet header", () => {
    vi.stubEnv("NODE_ENV", "production");
    const next = vi.fn();
    const res = makeRes();

    rateLimitMiddleware(
      makeReq({ ip: "10.0.0.99" }),
      res,
      next,
    );

    expect(next).toHaveBeenCalled();
  });
});
