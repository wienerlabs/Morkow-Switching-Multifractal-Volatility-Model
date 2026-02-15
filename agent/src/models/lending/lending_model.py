"""
Solana Lending Strategy XGBoost Model.

Optimized for lending protocol selection on Solana:
- MarginFi, Kamino, Solend protocol comparison
- APY-based switching decisions
- Health factor management
- Risk-adjusted yield optimization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import structlog
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from ..base import BaseModel
from ...config import LENDING_PARAMS, SOLANA_CHAIN_PARAMS

logger = structlog.get_logger()


# Solana lending-optimized parameters
SOLANA_LENDING_PARAMS = {
    **LENDING_PARAMS,
    # Lending-specific hyperparameters
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.9,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.8,
    # Binary classification for LEND vs NO_LEND
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'min_child_weight': 3,
    'scale_pos_weight': 1.0,  # Balanced classes expected
}


class LendingModel(BaseModel):
    """
    XGBoost classifier for Solana lending strategy.
    
    Predicts whether to lend on a specific protocol based on:
    - Net APY (after fees and costs)
    - Protocol health (TVL, utilization, age)
    - Health factor safety
    - Asset quality
    - Market conditions
    
    Key features:
    - Multi-protocol comparison (MarginFi, Kamino, Solend)
    - APY sustainability scoring
    - Risk-adjusted returns
    - Emergency exit detection
    """
    
    def __init__(self, params: dict | None = None):
        super().__init__(
            name="lending",
            params=params or SOLANA_LENDING_PARAMS
        )
        self.chain_params = SOLANA_CHAIN_PARAMS
        
        # Lending-specific thresholds (from config)
        self.min_apy = 0.02  # 2% minimum APY
        self.max_apy = 0.50  # 50% maximum APY (suspicious)
        self.min_health_factor = 2.0  # Minimum health factor
        self.min_confidence = 0.65  # 65% minimum confidence to lend
    
    def create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier optimized for lending decisions."""
        return xgb.XGBClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", 8),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.7),
            colsample_bytree=self.params.get("colsample_bytree", 0.9),
            gamma=self.params.get("gamma", 0.05),
            reg_alpha=self.params.get("reg_alpha", 0.05),
            reg_lambda=self.params.get("reg_lambda", 0.8),
            min_child_weight=self.params.get("min_child_weight", 3),
            scale_pos_weight=self.params.get("scale_pos_weight", 1.0),
            objective="binary:logistic",
            eval_metric=["auc", "logloss"],
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=self.training_config.get("early_stopping_rounds", 50),
            use_label_encoder=False,
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict lending opportunity probability.
        
        Returns:
            Array of probabilities (0-1) for profitable lending.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        threshold: float = 0.65
    ) -> pd.DataFrame:
        """
        Predict with confidence levels.
        
        Returns DataFrame with:
        - probability: Raw prediction probability
        - prediction: Binary prediction (LEND or NO_LEND)
        - confidence: Confidence level (low/medium/high/very_high)
        - recommended_action: Action to take
        """
        probs = self.predict(X)
        
        result = pd.DataFrame({
            'probability': probs,
            'prediction': (probs >= threshold).astype(int),
        })
        
        # Confidence levels
        result['confidence'] = pd.cut(
            probs,
            bins=[0, 0.4, 0.6, 0.75, 0.85, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # Recommended action based on probability
        result['recommended_action'] = result['probability'].apply(
            lambda p: 'full_position' if p >= 0.80 else
                     'partial_position' if p >= 0.65 else
                     'no_position'
        )

        return result

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, float]:
        """
        Evaluate model with lending-specific metrics.

        Includes risk-adjusted metrics since lending prioritizes
        capital preservation over maximum returns.
        """
        if self.model is None:
            raise ValueError("Model not trained")

        y_prob = self.predict(X)
        y_pred = (y_prob >= 0.65).astype(int)  # Use 65% threshold

        # Standard classification metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0,
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)

        # Lending-specific metrics
        # False positives = lending when shouldn't (capital at risk)
        # False negatives = missed yield opportunities
        metrics["fp_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["fn_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Conservative bias: FP should be lower than FN for lending
        metrics["conservative_ratio"] = (fp / (fp + 1)) / (fn / (fn + 1))

        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    def should_lend(
        self,
        features: pd.DataFrame,
        min_confidence: float | None = None,
        protocol: str | None = None
    ) -> dict:
        """
        Determine if lending should be executed.

        Args:
            features: Current protocol/market features
            min_confidence: Minimum prediction probability (default: 0.65)
            protocol: Protocol name (marginfi, kamino, solend)

        Returns:
            Dict with decision, reasoning, and recommended parameters.
        """
        if self.model is None:
            return {"lend": False, "reason": "Model not loaded"}

        min_conf = min_confidence or self.min_confidence

        # Get prediction
        prob = self.predict(features)[0]

        # Extract key features for decision logic
        net_apy = features.get("net_apy", pd.Series([0])).iloc[0]
        health_factor = features.get("health_factor", pd.Series([2.5])).iloc[0]
        utilization = features.get("utilization_rate", pd.Series([0])).iloc[0]
        protocol_tvl = features.get("protocol_tvl_usd", pd.Series([0])).iloc[0]

        # Safety checks
        if net_apy < self.min_apy:
            return {
                "lend": False,
                "reason": f"APY too low: {net_apy:.2%} < {self.min_apy:.2%}",
                "probability": prob,
                "net_apy": net_apy,
                "protocol": protocol
            }

        if net_apy > self.max_apy:
            return {
                "lend": False,
                "reason": f"APY suspiciously high: {net_apy:.2%} > {self.max_apy:.2%}",
                "probability": prob,
                "net_apy": net_apy,
                "protocol": protocol
            }

        if health_factor < self.min_health_factor:
            return {
                "lend": False,
                "reason": f"Health factor too low: {health_factor:.2f} < {self.min_health_factor:.2f}",
                "probability": prob,
                "health_factor": health_factor,
                "protocol": protocol
            }

        if utilization > 0.90:
            return {
                "lend": False,
                "reason": f"Utilization too high: {utilization:.1%} > 90%",
                "probability": prob,
                "utilization": utilization,
                "protocol": protocol
            }

        if protocol_tvl < 50_000_000:  # $50M minimum
            return {
                "lend": False,
                "reason": f"Protocol TVL too low: ${protocol_tvl:,.0f} < $50M",
                "probability": prob,
                "protocol_tvl": protocol_tvl,
                "protocol": protocol
            }

        # Confidence check
        if prob < min_conf:
            return {
                "lend": False,
                "reason": f"Low confidence: {prob:.2%} < {min_conf:.2%}",
                "probability": prob,
                "net_apy": net_apy,
                "protocol": protocol
            }

        # Determine position size based on confidence
        if prob >= 0.80:
            position_size = "full"
            leverage = 1.0  # No leverage by default
        elif prob >= 0.70:
            position_size = "75_percent"
            leverage = 1.0
        else:
            position_size = "50_percent"
            leverage = 1.0

        return {
            "lend": True,
            "reason": "High confidence lending opportunity",
            "probability": prob,
            "net_apy": net_apy,
            "health_factor": health_factor,
            "utilization": utilization,
            "protocol_tvl": protocol_tvl,
            "protocol": protocol,
            "position_size": position_size,
            "recommended_leverage": leverage,
            "confidence_level": "high" if prob >= 0.80 else "medium"
        }

    def compare_protocols(
        self,
        features_dict: dict[str, pd.DataFrame]
    ) -> dict:
        """
        Compare multiple protocols and recommend the best one.

        Args:
            features_dict: Dict mapping protocol names to their features
                          e.g., {"marginfi": df1, "kamino": df2, "solend": df3}

        Returns:
            Dict with best protocol, comparison scores, and reasoning.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        results = {}

        for protocol, features in features_dict.items():
            decision = self.should_lend(features, protocol=protocol)
            results[protocol] = {
                "probability": decision.get("probability", 0),
                "lend": decision.get("lend", False),
                "net_apy": decision.get("net_apy", 0),
                "reason": decision.get("reason", ""),
                "position_size": decision.get("position_size", "none")
            }

        # Find best protocol
        lendable = {k: v for k, v in results.items() if v["lend"]}

        if not lendable:
            return {
                "recommended_protocol": None,
                "reason": "No suitable protocols found",
                "all_results": results
            }

        # Sort by probability (confidence)
        best_protocol = max(lendable.items(), key=lambda x: x[1]["probability"])

        return {
            "recommended_protocol": best_protocol[0],
            "probability": best_protocol[1]["probability"],
            "net_apy": best_protocol[1]["net_apy"],
            "position_size": best_protocol[1]["position_size"],
            "reason": f"Highest confidence: {best_protocol[1]['probability']:.2%}",
            "all_results": results,
            "alternatives": [k for k in lendable.keys() if k != best_protocol[0]]
        }

    def should_switch_protocol(
        self,
        current_protocol: str,
        current_features: pd.DataFrame,
        alternative_features: dict[str, pd.DataFrame],
        min_apy_difference: float = 0.02,  # 2% minimum
        min_time_since_last_switch_hours: float = 12
    ) -> dict:
        """
        Determine if we should switch from current protocol to another.

        Args:
            current_protocol: Current protocol name
            current_features: Features for current protocol
            alternative_features: Dict of alternative protocol features
            min_apy_difference: Minimum APY difference to justify switch
            min_time_since_last_switch_hours: Minimum time since last switch

        Returns:
            Dict with switch decision and reasoning.
        """
        # Get current protocol score
        current_decision = self.should_lend(current_features, protocol=current_protocol)
        current_apy = current_decision.get("net_apy", 0)
        current_prob = current_decision.get("probability", 0)

        # Compare with alternatives
        comparison = self.compare_protocols(alternative_features)

        if not comparison.get("recommended_protocol"):
            return {
                "switch": False,
                "reason": "No better alternatives found",
                "current_protocol": current_protocol,
                "current_apy": current_apy
            }

        best_alternative = comparison["recommended_protocol"]
        best_apy = comparison["net_apy"]
        best_prob = comparison["probability"]

        # Check if switch is worthwhile
        apy_improvement = best_apy - current_apy

        if apy_improvement < min_apy_difference:
            return {
                "switch": False,
                "reason": f"APY improvement too small: {apy_improvement:.2%} < {min_apy_difference:.2%}",
                "current_protocol": current_protocol,
                "current_apy": current_apy,
                "best_alternative": best_alternative,
                "best_apy": best_apy,
                "apy_improvement": apy_improvement
            }

        # Check gas profitability
        # Estimate: 2 transactions (withdraw + deposit) at ~0.0005 SOL each
        gas_cost_sol = 0.001
        sol_price = current_features.get("sol_price_usd", pd.Series([100])).iloc[0]
        position_size = current_features.get("position_size_usd", pd.Series([10000])).iloc[0]

        gas_cost_usd = gas_cost_sol * sol_price
        gas_cost_percent = gas_cost_usd / position_size

        # Annualized gas cost
        # Assume we hold for at least 30 days, so annualized = gas_cost * 12
        annualized_gas_cost = gas_cost_percent * 12

        net_apy_improvement = apy_improvement - annualized_gas_cost

        if net_apy_improvement < 0.005:  # 0.5% minimum after gas
            return {
                "switch": False,
                "reason": f"Not profitable after gas: {net_apy_improvement:.2%} < 0.5%",
                "current_protocol": current_protocol,
                "current_apy": current_apy,
                "best_alternative": best_alternative,
                "best_apy": best_apy,
                "apy_improvement": apy_improvement,
                "gas_cost_percent": gas_cost_percent,
                "net_improvement": net_apy_improvement
            }

        return {
            "switch": True,
            "reason": f"Profitable switch: {net_apy_improvement:.2%} net improvement",
            "current_protocol": current_protocol,
            "current_apy": current_apy,
            "new_protocol": best_alternative,
            "new_apy": best_apy,
            "apy_improvement": apy_improvement,
            "gas_cost_percent": gas_cost_percent,
            "net_improvement": net_apy_improvement,
            "confidence": best_prob
        }

    def get_feature_importance_by_category(self) -> dict[str, list]:
        """
        Get feature importance grouped by category.

        Categories:
        - apy: APY-related features
        - protocol: Protocol health features
        - risk: Risk and health factor features
        - utilization: Utilization features
        - tvl: TVL and liquidity features
        - asset: Asset quality features
        - time: Time-based features
        """
        categories = {
            'apy': [],
            'protocol': [],
            'risk': [],
            'utilization': [],
            'tvl': [],
            'asset': [],
            'time': [],
            'other': []
        }

        for feature, importance in self.feature_importance.items():
            if 'apy' in feature.lower():
                categories['apy'].append((feature, importance))
            elif any(x in feature.lower() for x in ['protocol', 'audit', 'age']):
                categories['protocol'].append((feature, importance))
            elif any(x in feature.lower() for x in ['health', 'leverage', 'liquidation']):
                categories['risk'].append((feature, importance))
            elif 'utilization' in feature.lower():
                categories['utilization'].append((feature, importance))
            elif 'tvl' in feature.lower():
                categories['tvl'].append((feature, importance))
            elif any(x in feature.lower() for x in ['asset', 'tier', 'quality']):
                categories['asset'].append((feature, importance))
            elif any(x in feature.lower() for x in ['hour', 'day', 'time', 'duration']):
                categories['time'].append((feature, importance))
            else:
                categories['other'].append((feature, importance))

        # Sort each category by importance
        for cat in categories:
            categories[cat].sort(key=lambda x: x[1], reverse=True)

        return categories

