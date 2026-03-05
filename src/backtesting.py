"""Walk-forward backtesting engine and trading simulator."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .models import BaseForecaster


# ======================================================================
# Metrics
# ======================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                         y_prev: np.ndarray) -> float:
    """Fraction of non-zero moves where the predicted direction matches actual."""
    true_change = y_true - y_prev
    pred_change = y_pred - y_prev
    # Only evaluate on steps where the price actually moved
    moved = np.abs(true_change) > 1e-8
    if moved.sum() == 0:
        return 0.5  # no moves → undefined, return baseline
    true_dir = np.sign(true_change[moved])
    pred_dir = np.sign(pred_change[moved])
    return float(np.mean(true_dir == pred_dir))


def calibration_score(y_true: np.ndarray, lower: np.ndarray,
                      upper: np.ndarray) -> float:
    """Empirical coverage of prediction intervals."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prev: np.ndarray | None = None,
                    lower: np.ndarray | None = None,
                    upper: np.ndarray | None = None) -> dict[str, float]:
    m: dict[str, float] = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }
    if y_prev is not None and len(y_prev) == len(y_true):
        m["Directional Accuracy"] = directional_accuracy(y_true, y_pred, y_prev)
    if lower is not None and upper is not None:
        m["Coverage (95%)"] = calibration_score(y_true, lower, upper)
        m["Avg Interval Width"] = float(np.mean(upper - lower))
    return m


# ======================================================================
# Walk-forward backtesting
# ======================================================================

@dataclass
class BacktestResult:
    predictions: np.ndarray
    actuals: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    fold_metrics: list = field(default_factory=list)


class WalkForwardBacktest:
    """Expanding-window walk-forward validation."""

    def __init__(self, min_train_size: int = 100, step_size: int = 20,
                 retrain_every: int = 1):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.retrain_every = retrain_every

    def run(self, model_factory, X: np.ndarray, y: np.ndarray,
            seq_len: int = 0) -> BacktestResult:
        """
        model_factory: callable that returns a fresh BaseForecaster.
        seq_len: if > 0, model needs sequence input (handled automatically).
        """
        all_preds = []
        all_actuals = []
        all_lower = []
        all_upper = []
        fold_metrics_list = []

        n = len(X)
        fold = 0
        i = self.min_train_size

        while i < n:
            end = min(i + self.step_size, n)
            X_train, y_train = X[:i], y[:i]
            X_test, y_test = X[i:end], y[i:end]

            if fold % self.retrain_every == 0:
                model = model_factory()
                model.fit(X_train, y_train)

            preds = model.predict(X_test)

            # Handle length mismatch from sequence models
            actual_len = min(len(preds), len(y_test))
            if actual_len == 0:
                i = end
                fold += 1
                continue

            preds = preds[:actual_len]
            y_test_use = y_test[:actual_len]

            try:
                _, lo, hi = model.predict_interval(X_test)
                lo = lo[:actual_len]
                hi = hi[:actual_len]
            except Exception:
                lo = preds - 0.05
                hi = preds + 0.05

            all_preds.append(preds)
            all_actuals.append(y_test_use)
            all_lower.append(lo)
            all_upper.append(hi)

            fold_m = compute_metrics(y_test_use, preds)
            fold_m["fold"] = fold
            fold_metrics_list.append(fold_m)

            i = end
            fold += 1

        if not all_preds:
            return BacktestResult(np.array([]), np.array([]))

        predictions = np.concatenate(all_preds)
        actuals = np.concatenate(all_actuals)
        lower = np.concatenate(all_lower)
        upper = np.concatenate(all_upper)

        y_prev = np.concatenate([np.array([actuals[0]]), actuals[:-1]])
        overall = compute_metrics(actuals, predictions, y_prev, lower, upper)

        return BacktestResult(
            predictions=predictions, actuals=actuals,
            lower=lower, upper=upper,
            metrics=overall, fold_metrics=fold_metrics_list,
        )


# ======================================================================
# Trading simulator
# ======================================================================

@dataclass
class TradeRecord:
    step: int
    action: str  # "buy", "sell", "hold"
    price: float
    position: float
    pnl: float
    cumulative_pnl: float


class TradingSimulator:
    """Simple signal-based trading on prediction market contracts."""

    def __init__(self, threshold: float = 0.02, position_size: float = 100.0,
                 transaction_cost: float = 0.001):
        self.threshold = threshold
        self.position_size = position_size
        self.transaction_cost = transaction_cost

    def run(self, prices: np.ndarray,
            predicted_prices: np.ndarray) -> pd.DataFrame:
        """
        Simulate trading: buy when model predicts price increase > threshold,
        sell when model predicts decrease > threshold.
        """
        n = min(len(prices), len(predicted_prices))
        prices = prices[:n]
        predicted_prices = predicted_prices[:n]

        records: list[dict] = []
        position = 0.0  # +1 = long, -1 = short, 0 = flat
        cumulative_pnl = 0.0

        for i in range(1, n):
            expected_return = (predicted_prices[i] - prices[i - 1]) / max(prices[i - 1], 0.01)
            actual_return = (prices[i] - prices[i - 1]) / max(prices[i - 1], 0.01)

            # Decide action
            if expected_return > self.threshold:
                action = "buy"
                new_position = 1.0
            elif expected_return < -self.threshold:
                action = "sell"
                new_position = -1.0
            else:
                action = "hold"
                new_position = position

            # P&L from held position
            step_pnl = position * actual_return * self.position_size

            # Transaction cost on position change
            if new_position != position:
                step_pnl -= self.transaction_cost * self.position_size

            cumulative_pnl += step_pnl
            position = new_position

            records.append({
                "step": i,
                "action": action,
                "price": prices[i],
                "predicted_price": predicted_prices[i],
                "position": position,
                "step_pnl": step_pnl,
                "cumulative_pnl": cumulative_pnl,
            })

        return pd.DataFrame(records)

    @staticmethod
    def compute_trading_metrics(results: pd.DataFrame) -> dict[str, float]:
        if results.empty:
            return {}
        pnl = results["step_pnl"]
        cum = results["cumulative_pnl"]
        total_return = cum.iloc[-1]

        # Sharpe (annualized assuming hourly bars, ~8760 hours/year)
        sharpe = (pnl.mean() / pnl.std() * np.sqrt(252 * 24)) if pnl.std() > 0 else 0.0

        # Max drawdown
        peak = cum.cummax()
        drawdown = (cum - peak)
        max_dd = float(drawdown.min())

        # Win rate
        trades = results[results["action"] != "hold"]
        win_rate = float((trades["step_pnl"] > 0).mean()) if len(trades) > 0 else 0.0

        return {
            "Total Return ($)": round(total_return, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown ($)": round(max_dd, 2),
            "Win Rate": round(win_rate, 3),
            "Num Trades": int(len(trades)),
        }
