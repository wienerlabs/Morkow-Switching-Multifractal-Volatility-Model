import { randomUUID } from "node:crypto";
import type { Request, Response, NextFunction } from "express";
import { logger, requestContext } from "../lib/logger.js";

export function requestIdMiddleware(req: Request, res: Response, next: NextFunction): void {
  const requestId = (req.headers["x-request-id"] as string) || randomUUID();

  res.setHeader("x-request-id", requestId);

  requestContext.run({ requestId }, () => {
    const start = Date.now();

    res.on("finish", () => {
      logger.info({
        method: req.method,
        url: req.originalUrl,
        statusCode: res.statusCode,
        durationMs: Date.now() - start,
      }, "request completed");
    });

    next();
  });
}
