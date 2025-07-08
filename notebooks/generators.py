import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from notebooks.transformations import apply_all_transformations


# decomposition utility
def decompose_series(series: pd.Series, model: str, period: int):
    """
    Performs seasonal decomposition on a time series.

    Parameters:
        series: Input time series.
        model: 'additive' or 'multiplicative'.
        period: Seasonal period.

    Returns:
        DecompositionResult
    """
    if series.empty:
        raise ValueError("Input series is empty.")
    if model not in ["additive", "multiplicative"]:
        raise ValueError("Model must be 'additive' or 'multiplicative'.")
    if model == "multiplicative" and (series <= 0).any():
        raise ValueError("Multiplicative decomposition requires all values to be positive.")

    return seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')


# method1 - decompose and apply shift and scale transformation
def generate_synthetic_method1(
    series: pd.Series,
    model: str,
    period: int,
    **kwargs
) -> pd.Series:
    """
    Generate synthetic series using full recomposition of trend, seasonal, and residual components.

    Parameters:
        series: Input time series.
        model: 'additive' or 'multiplicative'.
        period: Seasonal period.
        **kwargs: transformation options like method, scale_factor, shift_value

    Returns:
        pd.Series: Synthetic time series.
    """
    decomp = decompose_series(series, model, period)

    trend = decomp.trend.reindex(series.index).ffill().bfill()
    seasonal = decomp.seasonal.reindex(series.index).ffill().bfill()
    residual = decomp.resid.reindex(series.index).ffill().bfill()

    if model == "additive":
        synthetic = trend + seasonal + residual
    else:
        synthetic = trend * seasonal * residual

    return apply_all_transformations(synthetic, **kwargs)


# method2 - Bootstrap Residuals
def generate_synthetic_method2_bootstrap(
    series: pd.Series,
    model: str,
    period: int,
    **kwargs
) -> pd.Series:
    """
    Generate synthetic series using decomposed trend & seasonality + bootstrapped residuals.

    Parameters:
        series: Input time series.
        model: 'additive' or 'multiplicative'.
        period: Seasonal period.
        **kwargs: transformation options like method, scale_factor, shift_value

    Returns:
        pd.Series: Synthetic time series.
    """
    decomp = decompose_series(series, model, period)

    trend = decomp.trend.reindex(series.index).ffill().bfill()
    seasonal = decomp.seasonal.reindex(series.index).ffill().bfill()
    residual = decomp.resid.dropna()

    # Bootstrap residuals
    bootstrapped_resid = np.random.choice(residual, size=len(series), replace=True)
    bootstrapped_resid = pd.Series(bootstrapped_resid, index=series.index)

    if model == "additive":
        synthetic = trend + seasonal + bootstrapped_resid
    else:
        synthetic = trend * seasonal * bootstrapped_resid.fillna(1.0)

    return apply_all_transformations(synthetic, **kwargs)
