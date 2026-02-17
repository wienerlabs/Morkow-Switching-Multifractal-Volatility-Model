import pino from "pino";
import { config } from "../config/index.js";
import { AsyncLocalStorage } from "node:async_hooks";

const requestContext = new AsyncLocalStorage<{ requestId: string }>();

export const logger = pino({
  level: config.LOG_LEVEL,
  ...(config.NODE_ENV === "development"
    ? { transport: { target: "pino-pretty", options: { colorize: true } } }
    : {}),
  formatters: {
    level: (label) => ({ level: label }),
  },
  mixin() {
    const store = requestContext.getStore();
    return store ? { requestId: store.requestId } : {};
  },
});

export { requestContext };

export function createChildLogger(bindings: Record<string, unknown>) {
  return logger.child(bindings);
}
