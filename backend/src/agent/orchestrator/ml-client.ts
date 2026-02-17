import type {
  MLPredictionRequest,
  MLPredictionResponse,
  RecommendationsResponse,
  ExecutionRequest,
  ExecutionResponse,
  StrategyType,
} from "./types.js";
import { requestContext } from "../../lib/logger.js";

export class MLAgentClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = "http://localhost:8000", timeout: number = 10000) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const res = await this.fetch("/health");
    return res.json() as Promise<{ status: string; timestamp: string }>;
  }

  async predict(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    const res = await this.fetch("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify(request),
    });
    return res.json() as Promise<MLPredictionResponse>;
  }

  async getRecommendations(): Promise<RecommendationsResponse> {
    const res = await this.fetch("/api/v1/recommendations");
    return res.json() as Promise<RecommendationsResponse>;
  }

  async getStrategies(): Promise<{
    strategies: Array<{
      name: StrategyType;
      description: string;
      model_loaded: boolean;
      model_version: string;
    }>;
    timestamp: string;
  }> {
    const res = await this.fetch("/api/v1/strategies");
    return res.json() as Promise<{
      strategies: Array<{
        name: StrategyType;
        description: string;
        model_loaded: boolean;
        model_version: string;
      }>;
      timestamp: string;
    }>;
  }

  async simulateExecution(request: ExecutionRequest): Promise<ExecutionResponse> {
    const res = await this.fetch("/api/v1/execute", {
      method: "POST",
      body: JSON.stringify({ ...request, simulate_only: true }),
    });
    return res.json() as Promise<ExecutionResponse>;
  }

  async executeStrategy(request: ExecutionRequest): Promise<ExecutionResponse> {
    const res = await this.fetch("/api/v1/execute", {
      method: "POST",
      body: JSON.stringify({ ...request, simulate_only: false }),
    });
    return res.json() as Promise<ExecutionResponse>;
  }

  async approveExecution(executionId: string, approved: boolean): Promise<ExecutionResponse> {
    const res = await this.fetch("/api/v1/approve", {
      method: "POST",
      body: JSON.stringify({ execution_id: executionId, approved }),
    });
    return res.json() as Promise<ExecutionResponse>;
  }

  async getExecutionStatus(executionId: string): Promise<ExecutionResponse> {
    const res = await this.fetch(`/api/v1/status/${executionId}`);
    return res.json() as Promise<ExecutionResponse>;
  }

  async getExecutionHistory(limit: number = 10): Promise<{
    executions: ExecutionResponse[];
    total: number;
  }> {
    const res = await this.fetch(`/api/v1/history?limit=${limit}`);
    return res.json() as Promise<{ executions: ExecutionResponse[]; total: number }>;
  }

  private async fetch(path: string, options: RequestInit = {}): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    const store = requestContext.getStore();
    const correlationHeaders: Record<string, string> = store?.requestId
      ? { "x-request-id": store.requestId }
      : {};

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...correlationHeaders,
          ...options.headers,
        },
        signal: controller.signal,
      });

      if (!res.ok) {
        const error = await res.text();
        throw new Error(`ML Agent error (${res.status}): ${error}`);
      }

      return res;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new Error(`ML Agent request timeout after ${this.timeout}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  isAvailable(): Promise<boolean> {
    return this.healthCheck()
      .then(() => true)
      .catch(() => false);
  }
}

export const createMLClient = (url?: string): MLAgentClient => {
  return new MLAgentClient(url ?? process.env.ML_AGENT_URL ?? "http://localhost:8000");
};

