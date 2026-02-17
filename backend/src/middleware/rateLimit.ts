import type { Request, Response, NextFunction } from "express";

const WINDOW_MS = 60_000;
const DEFAULT_READ_LIMIT = 60;
const DEFAULT_WRITE_LIMIT = 10;
const EXEMPT_PATHS = new Set(["/api/health"]);

const buckets = new Map<string, number[]>();

function getReadLimit(): number {
  return parseInt(process.env.RATE_LIMIT_READ ?? String(DEFAULT_READ_LIMIT), 10);
}

function getWriteLimit(): number {
  return parseInt(process.env.RATE_LIMIT_WRITE ?? String(DEFAULT_WRITE_LIMIT), 10);
}

function clientKey(req: Request): string {
  const pubkey = req.headers["x-solana-pubkey"] as string | undefined;
  if (pubkey) return `wallet:${pubkey}`;

  const forwarded = req.headers["x-forwarded-for"];
  if (forwarded) {
    const first = (Array.isArray(forwarded) ? forwarded[0] : forwarded).split(",")[0].trim();
    return `ip:${first}`;
  }

  return `ip:${req.ip ?? "unknown"}`;
}

function prune(timestamps: number[], now: number): number[] {
  const cutoff = now - WINDOW_MS;
  return timestamps.filter((t) => t > cutoff);
}

export function rateLimitMiddleware(req: Request, res: Response, next: NextFunction): void {
  if (EXEMPT_PATHS.has(req.path)) {
    next();
    return;
  }

  if (process.env.NODE_ENV === "test") {
    next();
    return;
  }

  const isWrite = req.method !== "GET" && req.method !== "HEAD" && req.method !== "OPTIONS";
  const limit = isWrite ? getWriteLimit() : getReadLimit();
  const key = clientKey(req);
  const now = Date.now();

  const timestamps = prune(buckets.get(key) ?? [], now);

  if (timestamps.length >= limit) {
    const retryAfter = Math.ceil((WINDOW_MS - (now - timestamps[0])) / 1000) + 1;
    res.setHeader("Retry-After", String(retryAfter));
    res.setHeader("X-RateLimit-Limit", String(limit));
    res.setHeader("X-RateLimit-Remaining", "0");
    res.status(429).json({ success: false, error: "Rate limit exceeded", retry_after: retryAfter });
    return;
  }

  timestamps.push(now);
  buckets.set(key, timestamps);

  res.setHeader("X-RateLimit-Limit", String(limit));
  res.setHeader("X-RateLimit-Remaining", String(limit - timestamps.length));
  next();
}
