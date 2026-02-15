import Database from 'better-sqlite3';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_DB_PATH = path.join(__dirname, '../../../data/cortex.db');

let db: Database.Database | null = null;

export function getDb(dbPath?: string): Database.Database {
  if (db) return db;

  const resolvedPath = dbPath || process.env.CORTEX_DB_PATH || DEFAULT_DB_PATH;
  const dir = path.dirname(resolvedPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  db = new Database(resolvedPath);
  db.pragma('journal_mode = WAL');
  db.pragma('synchronous = NORMAL');
  db.pragma('foreign_keys = ON');

  initSchema(db);

  logger.info('[DB] SQLite initialized', { path: resolvedPath });
  return db;
}

function initSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS portfolio_state (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      version INTEGER NOT NULL DEFAULT 1,
      last_updated INTEGER NOT NULL,
      total_value_usd REAL NOT NULL DEFAULT 0,
      realized_pnl_usd REAL NOT NULL DEFAULT 0,
      unrealized_pnl_usd REAL NOT NULL DEFAULT 0,
      total_fees_usd REAL NOT NULL DEFAULT 0,
      daily_pnl_usd REAL NOT NULL DEFAULT 0,
      daily_trade_count INTEGER NOT NULL DEFAULT 0,
      day_start_timestamp INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS balances (
      symbol TEXT PRIMARY KEY,
      mint TEXT NOT NULL,
      balance REAL NOT NULL,
      value_usd REAL NOT NULL,
      last_updated INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS positions (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL CHECK (type IN ('lp', 'perps')),
      data TEXT NOT NULL,
      is_open INTEGER NOT NULL DEFAULT 1,
      created_at INTEGER NOT NULL,
      closed_at INTEGER
    );

    CREATE TABLE IF NOT EXISTS trades (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      asset TEXT NOT NULL,
      side TEXT NOT NULL,
      amount_usd REAL NOT NULL,
      venue TEXT NOT NULL,
      tx_signature TEXT,
      fees REAL NOT NULL DEFAULT 0,
      slippage REAL,
      pnl_usd REAL NOT NULL DEFAULT 0,
      ml_confidence REAL,
      notes TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
    CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(type);

    CREATE TABLE IF NOT EXISTS risk_state (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      circuit_breaker_state TEXT NOT NULL DEFAULT 'ACTIVE',
      circuit_breaker_reason TEXT,
      last_circuit_breaker_trigger INTEGER,
      peak_value_usd REAL NOT NULL DEFAULT 0,
      day_start_value_usd REAL NOT NULL DEFAULT 0,
      week_start_value_usd REAL NOT NULL DEFAULT 0,
      month_start_value_usd REAL NOT NULL DEFAULT 0,
      day_start_timestamp INTEGER NOT NULL DEFAULT 0,
      week_start_timestamp INTEGER NOT NULL DEFAULT 0,
      month_start_timestamp INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS risk_alerts (
      id TEXT PRIMARY KEY,
      timestamp INTEGER NOT NULL,
      severity TEXT NOT NULL,
      category TEXT NOT NULL,
      message TEXT NOT NULL,
      data TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_risk_alerts_timestamp ON risk_alerts(timestamp);
  `);
}

export function closeDb(): void {
  if (db) {
    db.close();
    db = null;
    logger.info('[DB] SQLite connection closed');
  }
}

export function resetDb(): void {
  db = null;
}
