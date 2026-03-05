"""Plotting utilities for prediction market analysis."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def save(fig, name: str):
    fig.savefig(RESULTS_DIR / f"{name}.png")


# ------------------------------------------------------------------
# EDA plots
# ------------------------------------------------------------------

def plot_price_series(df: pd.DataFrame, col: str = "close",
                      title: str = "Price Series", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[col], linewidth=1)
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    return ax


def plot_multi_market(datasets: dict[str, pd.DataFrame], col: str = "close"):
    n = len(datasets)
    fig, axes = plt.subplots(min(n, 5), 1, figsize=(12, 3 * min(n, 5)), sharex=True)
    if min(n, 5) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, list(datasets.items())[:5]):
        ax.plot(df.index, df[col], linewidth=1)
        ax.set_ylabel("Price")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    save(fig, "multi_market_prices")
    return fig


def plot_distribution(df: pd.DataFrame, col: str = "close"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(df[col].dropna(), bins=50, edgecolor="white")
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price")

    ret = df[col].pct_change().dropna()
    axes[1].hist(ret, bins=50, edgecolor="white")
    axes[1].set_title("Return Distribution")
    axes[1].set_xlabel("Return")

    if "volume" in df.columns:
        axes[2].hist(df["volume"].dropna(), bins=50, edgecolor="white")
        axes[2].set_title("Volume Distribution")
    plt.tight_layout()
    save(fig, "distributions")
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 40):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    save(fig, "acf_pacf")
    return fig


def plot_decomposition(series: pd.Series, period: int = 24):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(series.dropna(), period=period, model="additive",
                                extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    save(fig, "decomposition")
    return fig


# ------------------------------------------------------------------
# Feature plots
# ------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, max_features: int = 20):
    cols = df.select_dtypes(include=[np.number]).columns[:max_features]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    save(fig, "correlation_heatmap")
    return fig


def plot_feature_importance(names: list[str], importances: np.ndarray,
                            top_n: int = 15):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    save(fig, "feature_importance")
    return fig


# ------------------------------------------------------------------
# Model result plots
# ------------------------------------------------------------------

def plot_predictions(actuals: np.ndarray, predictions: np.ndarray,
                     lower: np.ndarray | None = None,
                     upper: np.ndarray | None = None,
                     title: str = "Model Predictions",
                     index: pd.DatetimeIndex | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = index if index is not None else np.arange(len(actuals))
    ax.plot(x, actuals, label="Actual", linewidth=1.2, alpha=0.8)
    ax.plot(x, predictions, label="Predicted", linewidth=1.2, alpha=0.8)
    if lower is not None and upper is not None:
        ax.fill_between(x, lower, upper, alpha=0.2, label="95% CI")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save(fig, title.lower().replace(" ", "_"))
    return fig


def plot_residuals(actuals: np.ndarray, predictions: np.ndarray,
                   title: str = "Residual Diagnostics"):
    resid = actuals - predictions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(resid, linewidth=0.8)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("Residuals over Time")

    axes[1].hist(resid, bins=40, edgecolor="white")
    axes[1].set_title("Residual Distribution")

    from scipy import stats
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    save(fig, "residuals")
    return fig


def plot_model_comparison(metrics: dict[str, dict[str, float]]):
    """Bar chart comparing models across metrics."""
    df = pd.DataFrame(metrics).T
    fig, axes = plt.subplots(1, len(df.columns), figsize=(4 * len(df.columns), 5))
    if len(df.columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        df[col].plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_title(col)
        ax.set_ylabel(col)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save(fig, "model_comparison")
    return fig


def plot_training_loss(train_loss: list[float], val_loss: list[float] | None = None,
                       title: str = "Training Loss"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label="Train")
    if val_loss:
        ax.plot(val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    save(fig, title.lower().replace(" ", "_"))
    return fig


# ------------------------------------------------------------------
# Trading plots
# ------------------------------------------------------------------

def plot_trading_pnl(results: pd.DataFrame, title: str = "Trading P&L"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(results["step"], results["cumulative_pnl"], linewidth=1.2)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Cumulative P&L ($)")
    axes[0].set_title(title)

    colors = {"buy": "green", "sell": "red", "hold": "gray"}
    for action in ["buy", "sell"]:
        mask = results["action"] == action
        axes[1].scatter(results.loc[mask, "step"], results.loc[mask, "price"],
                        c=colors[action], label=action, s=10, alpha=0.6)
    axes[1].plot(results["step"], results["price"], linewidth=0.8, alpha=0.5, color="black")
    axes[1].set_ylabel("Price")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    plt.tight_layout()
    save(fig, "trading_pnl")
    return fig
