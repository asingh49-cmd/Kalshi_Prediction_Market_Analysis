"""Forecasting models: LSTM, GRU, Transformer, BSTS, ARIMA, XGBoost, Ensemble.

All models expose fit(X_train, y_train), predict(X), predict_interval(X, alpha).
Deep-learning models use PyTorch; statistical models use statsmodels/pmdarima.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Base class
# ======================================================================

class BaseForecaster(ABC):
    """Unified interface for all forecasters."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_interval(self, X: np.ndarray,
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point, lower, upper). Default: point ± 2*residual_std."""
        point = self.predict(X)
        std = getattr(self, "residual_std_", 0.05)
        z = 1.96 if alpha == 0.05 else 2.576
        return point, point - z * std, point + z * std


# ======================================================================
# PyTorch sequence models
# ======================================================================

def _make_sequences(X: np.ndarray, y: np.ndarray,
                    seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window transform: (N, F) → (N-seq_len, seq_len, F)."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


class _GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2)]


class _TransformerNet(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class SequenceModelWrapper(BaseForecaster):
    """Training loop wrapper for LSTM / GRU / Transformer."""

    def __init__(self, model_type: str = "lstm", seq_len: int = 30,
                 hidden_size: int = 64, num_layers: int = 2,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 64,
                 patience: int = 8, device: str | None = None):
        self.model_type = model_type
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model_: nn.Module | None = None
        self.residual_std_ = 0.05
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def _build_net(self, input_size: int) -> nn.Module:
        if self.model_type == "lstm":
            return _LSTMNet(input_size, self.hidden_size, self.num_layers)
        elif self.model_type == "gru":
            return _GRUNet(input_size, self.hidden_size, self.num_layers)
        elif self.model_type == "transformer":
            return _TransformerNet(input_size, d_model=self.hidden_size)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            **kw) -> "SequenceModelWrapper":
        Xs, ys = _make_sequences(X_train, y_train, self.seq_len)
        net = self._build_net(Xs.shape[2]).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            Xv, yv = _make_sequences(X_val, y_val, self.seq_len)
            if len(Xv) > 0:
                val_ds = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
                val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = net(xb).squeeze(-1)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            self.train_loss_history_.append(epoch_loss / len(Xs))

            # Validation
            if val_loader is not None:
                net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        pred = net(xb).squeeze(-1)
                        val_loss += criterion(pred, yb).item() * len(xb)
                val_loss /= len(Xv)
                self.val_loss_history_.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model_ = net

        # Residual std for intervals — use validation if available
        if val_loader is not None and len(Xv) > 0:
            with torch.no_grad():
                pred_val = self._predict_raw(Xv)
            self.residual_std_ = float(np.std(yv - pred_val))
        else:
            with torch.no_grad():
                pred_train = self._predict_raw(Xs)
            self.residual_std_ = float(np.std(ys - pred_train))
        return self

    def _predict_raw(self, X_seq: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(X_seq).to(self.device)
        with torch.no_grad():
            return self.model_(t).squeeze(-1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X can be (N, F) – will create sequences – or (N, seq_len, F)."""
        if X.ndim == 2:
            Xs, _ = _make_sequences(X, np.zeros(len(X)), self.seq_len)
        else:
            Xs = X.astype(np.float32)
        if len(Xs) == 0:
            return np.array([])
        return self._predict_raw(Xs)


# Convenience aliases
class LSTMForecaster(SequenceModelWrapper):
    def __init__(self, **kw):
        super().__init__(model_type="lstm", **kw)


class GRUForecaster(SequenceModelWrapper):
    def __init__(self, **kw):
        super().__init__(model_type="gru", **kw)


class TransformerForecaster(SequenceModelWrapper):
    def __init__(self, **kw):
        super().__init__(model_type="transformer", **kw)


# ======================================================================
# BSTS (Bayesian Structural Time Series via UnobservedComponents)
# ======================================================================

class BSTSForecaster(BaseForecaster):
    """Local-linear-trend + seasonal model on logit-transformed prices."""

    def __init__(self, seasonal_period: int = 24, use_logit: bool = True):
        self.seasonal_period = seasonal_period
        self.use_logit = use_logit
        self.model_ = None
        self.result_ = None
        self.residual_std_ = 0.05

    def _to_logit(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 0.001, 0.999)
        return np.log(p / (1 - p))

    def _from_logit(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw) -> "BSTSForecaster":
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        series = y_train.copy()
        if self.use_logit:
            series = self._to_logit(series)

        try:
            model = UnobservedComponents(
                series,
                level="local linear trend",
                seasonal=self.seasonal_period,
                stochastic_seasonal=True,
            )
            self.result_ = model.fit(disp=False, maxiter=200)
        except Exception:
            # Fallback: no seasonal
            model = UnobservedComponents(series, level="local linear trend")
            self.result_ = model.fit(disp=False, maxiter=200)

        resid = self.result_.resid
        self.residual_std_ = float(np.nanstd(resid))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_steps = len(X)
        forecast = self.result_.forecast(steps=n_steps)
        forecast = np.asarray(forecast)
        if self.use_logit:
            forecast = self._from_logit(forecast)
        return np.clip(forecast, 0, 1)

    def predict_interval(self, X: np.ndarray,
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_steps = len(X)
        fc = self.result_.get_forecast(steps=n_steps)
        ci = fc.conf_int(alpha=alpha)
        point = np.asarray(fc.predicted_mean)
        lower = np.asarray(ci.iloc[:, 0]) if hasattr(ci, "iloc") else ci[:, 0]
        upper = np.asarray(ci.iloc[:, 1]) if hasattr(ci, "iloc") else ci[:, 1]
        if self.use_logit:
            point = self._from_logit(point)
            lower = self._from_logit(lower)
            upper = self._from_logit(upper)
        return np.clip(point, 0, 1), np.clip(lower, 0, 1), np.clip(upper, 0, 1)


# ======================================================================
# ARIMA
# ======================================================================

class ARIMAForecaster(BaseForecaster):
    """SARIMAX with automatic order selection via pmdarima."""

    def __init__(self, seasonal: bool = False, m: int = 24):
        self.seasonal = seasonal
        self.m = m
        self.model_ = None
        self.residual_std_ = 0.05

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw) -> "ARIMAForecaster":
        import pmdarima as pm
        self.model_ = pm.auto_arima(
            y_train, seasonal=self.seasonal, m=self.m,
            suppress_warnings=True, error_action="ignore",
            stepwise=True, max_p=3, max_q=3, max_d=2,
            max_order=6,
        )
        self.residual_std_ = float(np.std(self.model_.resid()))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        fc = self.model_.predict(n_periods=len(X))
        return np.clip(np.asarray(fc), 0, 1)

    def predict_interval(self, X: np.ndarray,
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fc, ci = self.model_.predict(n_periods=len(X), return_conf_int=True, alpha=alpha)
        return (np.clip(np.asarray(fc), 0, 1),
                np.clip(ci[:, 0], 0, 1),
                np.clip(ci[:, 1], 0, 1))


# ======================================================================
# XGBoost
# ======================================================================

class XGBoostForecaster(BaseForecaster):
    """XGBRegressor with early stopping."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, early_stopping: int = 20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.model_: xgb.XGBRegressor | None = None
        self.residual_std_ = 0.05

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            **kw) -> "XGBoostForecaster":
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:squarederror",
            verbosity=0,
        )
        fit_params: dict = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model_.fit(X_train, y_train, **fit_params)
        # Use validation residuals for interval estimation (avoids overfitting bias)
        if X_val is not None and y_val is not None:
            resid = y_val - self.model_.predict(X_val)
        else:
            resid = y_train - self.model_.predict(X_train)
        self.residual_std_ = float(np.std(resid))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model_.predict(X), 0, 1)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_


# ======================================================================
# Conformal Predictor
# ======================================================================

class ConformalPredictor:
    """Conformal prediction intervals adapted for time series.

    Standard conformal prediction assumes exchangeability, which is
    violated in time series (distribution shift). We compensate by:
    1. Using the max calibration residual (not 95th percentile) as base
    2. Adding a nonstationarity buffer proportional to calibration volatility
    This trades interval width for more reliable coverage.
    """

    def __init__(self, base_model: BaseForecaster):
        self.base = base_model
        self.residuals_: np.ndarray | None = None
        self.cal_prices_: np.ndarray | None = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        preds = self.base.predict(X_cal)
        n = min(len(preds), len(y_cal))
        self.residuals_ = np.abs(y_cal[-n:] - preds[-n:])
        self.cal_prices_ = y_cal[-n:]

    def predict_interval(self, X: np.ndarray,
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = self.base.predict(X)

        # Base: use a high quantile of calibration residuals
        q_base = np.quantile(self.residuals_, min(1 - alpha, 0.99))

        # Nonstationarity buffer: accounts for distribution shift
        # Uses the overall range of calibration price changes as proxy
        price_changes = np.abs(np.diff(self.cal_prices_))
        drift_buffer = np.quantile(price_changes, 0.95) * 4

        half_width = q_base + drift_buffer

        lower = np.clip(point - half_width, 0, 1)
        upper = np.clip(point + half_width, 0, 1)

        return point, lower, upper


# ======================================================================
# Stacked Ensemble
# ======================================================================

class StackedEnsemble(BaseForecaster):
    """Combines multiple forecasters via Ridge meta-learner or weighted average."""

    def __init__(self, forecasters: dict[str, BaseForecaster],
                 method: str = "ridge"):
        self.forecasters = forecasters
        self.method = method
        self.meta_: Ridge | None = None
        self.weights_: dict[str, float] = {}
        self.residual_std_ = 0.05

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            **kw) -> "StackedEnsemble":
        if X_val is None or y_val is None:
            # Use last 20% of training data for meta-learning
            split = int(0.8 * len(X_train))
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]

        # Get base predictions on validation set
        meta_features = {}
        for name, model in self.forecasters.items():
            preds = model.predict(X_val)
            # Align lengths (sequence models may produce fewer outputs)
            if len(preds) < len(y_val):
                meta_features[name] = np.full(len(y_val), np.nan)
                meta_features[name][-len(preds):] = preds
            else:
                meta_features[name] = preds[:len(y_val)]

        meta_X = np.column_stack(list(meta_features.values()))
        # Drop rows with NaN
        mask = ~np.isnan(meta_X).any(axis=1)
        meta_X = meta_X[mask]
        meta_y = y_val[mask]

        if self.method == "ridge":
            self.meta_ = Ridge(alpha=1.0)
            self.meta_.fit(meta_X, meta_y)
            for i, name in enumerate(self.forecasters):
                self.weights_[name] = float(self.meta_.coef_[i])
        else:
            # Simple inverse-error weighting
            for name in self.forecasters:
                preds = meta_features[name][mask]
                err = np.mean((meta_y - preds) ** 2) + 1e-8
                self.weights_[name] = 1.0 / err
            total = sum(self.weights_.values())
            self.weights_ = {k: v / total for k, v in self.weights_.items()}

        resid = meta_y - self._predict_meta(meta_X)
        self.residual_std_ = float(np.std(resid))
        return self

    def _predict_meta(self, meta_X: np.ndarray) -> np.ndarray:
        if self.meta_ is not None:
            return np.clip(self.meta_.predict(meta_X), 0, 1)
        # Weighted average fallback
        return np.clip(meta_X @ np.array(list(self.weights_.values())), 0, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = {}
        min_len = len(X)
        for name, model in self.forecasters.items():
            p = model.predict(X)
            preds[name] = p
            min_len = min(min_len, len(p))

        meta_X = np.column_stack([p[:min_len] for p in preds.values()])
        return self._predict_meta(meta_X)
