/**
 * Model Registry Service
 *
 * Tracks ML model versions, their performance metrics, and deployment status.
 * Provides version comparison, rollback capability, and inference logging.
 *
 * All data is dynamically loaded from model metadata files.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';

// ============= TYPES =============

export interface ModelMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  sharpe?: number;
  ece?: number; // Expected Calibration Error
}

export interface ModelVersion {
  modelName: string;
  version: string;
  onnxPath: string;
  metadataPath: string;
  calibrationPath?: string;
  trainedAt: Date;
  deployedAt?: Date;
  metrics: ModelMetrics;
  status: 'active' | 'deprecated' | 'experimental' | 'archived';
  gitCommit?: string;
  mlflowRunId?: string;
  fileHash?: string;
}

export interface PredictionLog {
  id: string;
  tradeId: string;
  modelName: string;
  modelVersion: string;
  prediction: number;
  confidence: number;
  calibratedConfidence?: number;
  features?: Record<string, number>;
  timestamp: Date;
  latencyMs?: number;
}

export interface VersionComparison {
  versions: string[];
  metrics: Record<string, ModelMetrics>;
  winner: string;
  improvements: Record<string, number>;
}

// ============= CONSTANTS =============

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MODELS_DIR = path.resolve(__dirname, '../../../models');
const REGISTRY_FILE = path.resolve(__dirname, '../../../models/registry.json');
const PREDICTION_LOG_DIR = path.resolve(__dirname, '../../../logs/predictions');

// ============= MODEL REGISTRY CLASS =============

class ModelRegistry {
  private registry: Map<string, ModelVersion[]> = new Map();
  private activeModels: Map<string, ModelVersion> = new Map();
  private predictionLogs: PredictionLog[] = [];
  private initialized = false;

  constructor() {
    this.loadRegistry();
  }

  /**
   * Load registry from disk
   */
  private loadRegistry(): void {
    try {
      if (fs.existsSync(REGISTRY_FILE)) {
        const data = JSON.parse(fs.readFileSync(REGISTRY_FILE, 'utf-8'));
        
        for (const [modelName, versions] of Object.entries(data.models || {})) {
          const versionList = (versions as any[]).map(v => ({
            ...v,
            trainedAt: new Date(v.trainedAt),
            deployedAt: v.deployedAt ? new Date(v.deployedAt) : undefined,
          }));
          this.registry.set(modelName, versionList);
          
          // Set active model
          const active = versionList.find(v => v.status === 'active');
          if (active) {
            this.activeModels.set(modelName, active);
          }
        }
        this.initialized = true;
      } else {
        // Initialize from model metadata files
        this.initializeFromMetadata();
      }
    } catch (error) {
      logger.error('[ModelRegistry] Failed to load registry:', { error: String(error) });
      this.initializeFromMetadata();
    }
  }

  /**
   * Initialize registry from existing model metadata files
   */
  private initializeFromMetadata(): void {
    const metadataDir = path.join(MODELS_DIR, 'metadata');
    
    if (!fs.existsSync(metadataDir)) {
      logger.info('[ModelRegistry] No metadata directory found');
      return;
    }

    // Scan for metadata files
    const metadataFiles = fs.readdirSync(metadataDir)
      .filter(f => f.endsWith('_metadata.json') || f === 'model_config.json');

    for (const file of metadataFiles) {
      try {
        const filePath = path.join(metadataDir, file);
        const metadata = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        
        const modelName = this.extractModelName(file, metadata);
        const version = this.createVersionFromMetadata(modelName, metadata, filePath);
        
        if (version) {
          this.addVersion(version);
        }
      } catch (error) {
        logger.error(`[ModelRegistry] Failed to parse ${file}:`, { error: String(error) });
      }
    }

    this.saveRegistry();
    this.initialized = true;
  }

  private extractModelName(filename: string, metadata: any): string {
    if (metadata.model_info?.name) return metadata.model_info.name;
    if (metadata.model_name) return metadata.model_name;
    return filename.replace('_metadata.json', '').replace('.json', '');
  }

  private createVersionFromMetadata(
    modelName: string,
    metadata: any,
    metadataPath: string
  ): ModelVersion | null {
    const version = metadata.model_info?.version || metadata.version || '1.0.0';
    const onnxFile = `${modelName.replace('_predictor', '')}_predictor.onnx`;
    const onnxPath = path.join(MODELS_DIR, onnxFile);

    const metrics: ModelMetrics = {
      precision: metadata.metrics?.precision || metadata.validation?.precision || 0,
      recall: metadata.metrics?.recall || metadata.validation?.recall || 0,
      f1Score: metadata.metrics?.f1_score || metadata.metrics?.f1 || 0,
      rocAuc: metadata.metrics?.roc_auc || metadata.validation?.roc_auc || 0,
      sharpe: metadata.metrics?.sharpe_ratio || metadata.backtest?.sharpe_ratio,
      ece: metadata.calibration?.ece || metadata.metrics?.ece,
    };

    return {
      modelName,
      version,
      onnxPath,
      metadataPath,
      calibrationPath: this.findCalibrationPath(modelName),
      trainedAt: new Date(metadata.model_info?.timestamp || metadata.trained_at || Date.now()),
      metrics,
      status: 'active',
      gitCommit: metadata.git_commit,
      mlflowRunId: metadata.mlflow_run_id,
    };
  }

  private findCalibrationPath(modelName: string): string | undefined {
    const calibDir = path.join(MODELS_DIR, 'calibration');
    const calibFile = `${modelName}_calibration.json`;
    const calibPath = path.join(calibDir, calibFile);
    return fs.existsSync(calibPath) ? calibPath : undefined;
  }

  /**
   * Add a new model version
   */
  addVersion(version: ModelVersion): void {
    const versions = this.registry.get(version.modelName) || [];

    // Deprecate previous active version
    if (version.status === 'active') {
      versions.forEach(v => {
        if (v.status === 'active') v.status = 'deprecated';
      });
      this.activeModels.set(version.modelName, version);
    }

    versions.push(version);
    this.registry.set(version.modelName, versions);
    this.saveRegistry();
  }

  /**
   * Get active model version
   */
  getActiveVersion(modelName: string): ModelVersion | undefined {
    return this.activeModels.get(modelName);
  }

  /**
   * Get all versions for a model
   */
  getVersions(modelName: string): ModelVersion[] {
    return this.registry.get(modelName) || [];
  }

  /**
   * Get specific version
   */
  getVersion(modelName: string, version: string): ModelVersion | undefined {
    const versions = this.registry.get(modelName) || [];
    return versions.find(v => v.version === version);
  }

  /**
   * Rollback to a previous version
   */
  rollback(modelName: string, targetVersion: string): boolean {
    const versions = this.registry.get(modelName);
    if (!versions) return false;

    const target = versions.find(v => v.version === targetVersion);
    if (!target) return false;

    // Deprecate current active
    versions.forEach(v => {
      if (v.status === 'active') v.status = 'deprecated';
    });

    // Activate target
    target.status = 'active';
    target.deployedAt = new Date();
    this.activeModels.set(modelName, target);
    this.saveRegistry();

    logger.info(`[ModelRegistry] Rolled back ${modelName} to ${targetVersion}`);
    return true;
  }

  /**
   * Compare two model versions
   */
  compareVersions(modelName: string, v1: string, v2: string): VersionComparison | null {
    const version1 = this.getVersion(modelName, v1);
    const version2 = this.getVersion(modelName, v2);

    if (!version1 || !version2) return null;

    const improvements: Record<string, number> = {};
    const m1 = version1.metrics;
    const m2 = version2.metrics;

    improvements.precision = ((m2.precision - m1.precision) / m1.precision) * 100;
    improvements.recall = ((m2.recall - m1.recall) / m1.recall) * 100;
    improvements.rocAuc = ((m2.rocAuc - m1.rocAuc) / m1.rocAuc) * 100;

    // Determine winner based on F1 score
    const winner = m2.f1Score >= m1.f1Score ? v2 : v1;

    return {
      versions: [v1, v2],
      metrics: { [v1]: m1, [v2]: m2 },
      winner,
      improvements,
    };
  }

  /**
   * Log a prediction for tracking
   */
  logPrediction(log: Omit<PredictionLog, 'id' | 'timestamp'>): string {
    const id = `pred_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
    const entry: PredictionLog = {
      ...log,
      id,
      timestamp: new Date(),
    };

    this.predictionLogs.push(entry);

    // Persist to disk periodically
    if (this.predictionLogs.length % 100 === 0) {
      this.flushPredictionLogs();
    }

    return id;
  }

  /**
   * Flush prediction logs to disk
   */
  private flushPredictionLogs(): void {
    if (this.predictionLogs.length === 0) return;

    try {
      if (!fs.existsSync(PREDICTION_LOG_DIR)) {
        fs.mkdirSync(PREDICTION_LOG_DIR, { recursive: true });
      }

      const date = new Date().toISOString().split('T')[0];
      const logFile = path.join(PREDICTION_LOG_DIR, `predictions_${date}.jsonl`);

      const lines = this.predictionLogs.map(log => JSON.stringify(log)).join('\n') + '\n';
      fs.appendFileSync(logFile, lines);

      this.predictionLogs = [];
    } catch (error) {
      logger.error('[ModelRegistry] Failed to flush prediction logs:', { error: String(error) });
    }
  }

  /**
   * Save registry to disk
   */
  private saveRegistry(): void {
    try {
      const data = {
        lastUpdated: new Date().toISOString(),
        models: Object.fromEntries(this.registry),
      };

      fs.writeFileSync(REGISTRY_FILE, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('[ModelRegistry] Failed to save registry:', { error: String(error) });
    }
  }

  /**
   * Get all registered models
   */
  getAllModels(): string[] {
    return Array.from(this.registry.keys());
  }

  /**
   * Check if registry is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get registry summary
   */
  getSummary(): Record<string, any> {
    const summary: Record<string, any> = {};

    for (const [modelName, versions] of this.registry) {
      const active = versions.find(v => v.status === 'active');
      summary[modelName] = {
        totalVersions: versions.length,
        activeVersion: active?.version || 'none',
        latestMetrics: active?.metrics || null,
        trainedAt: active?.trainedAt || null,
      };
    }

    return summary;
  }

  /**
   * Get prediction logs for a specific model and version
   * Reads from in-memory buffer and disk logs
   */
  getPredictionLogs(modelName: string, version?: string): PredictionLog[] {
    // Get in-memory logs
    let logs = this.predictionLogs.filter(log => {
      const matchesModel = log.modelName === modelName;
      const matchesVersion = !version || log.modelVersion === version;
      return matchesModel && matchesVersion;
    });

    // Also read from disk logs (last 7 days)
    try {
      const now = new Date();
      for (let i = 0; i < 7; i++) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        const dateStr = date.toISOString().split('T')[0];
        const logFile = path.join(PREDICTION_LOG_DIR, `predictions_${dateStr}.jsonl`);

        if (fs.existsSync(logFile)) {
          const content = fs.readFileSync(logFile, 'utf-8');
          const lines = content.trim().split('\n').filter(Boolean);

          for (const line of lines) {
            try {
              const log = JSON.parse(line) as PredictionLog;
              if (log.modelName === modelName && (!version || log.modelVersion === version)) {
                logs.push(log);
              }
            } catch {
              // Skip malformed lines
            }
          }
        }
      }
    } catch (error) {
      logger.error('[ModelRegistry] Failed to read prediction logs from disk:', { error: String(error) });
    }

    return logs;
  }
}

// Singleton instance
export const modelRegistry = new ModelRegistry();
export { ModelRegistry };

