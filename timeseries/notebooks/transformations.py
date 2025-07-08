import pandas as pd


def shift_and_scale(
    series: pd.Series,
    scale_factor: float = None,
    shift_value: float = None
) -> pd.Series:
    """
    Applies scaling and shifting to a time series if parameters are provided.

    Parameters:
        series: The time series to transform.
        scale_factor: If provided, multiplies the series by this factor.
        shift_value: If provided, adds this value to the series.

    Returns:
        pd.Series: Transformed time series.
    """
    transformed = series.copy()

    if scale_factor is not None:
        transformed *= scale_factor
    if shift_value is not None:
        transformed += shift_value

    return transformed


TRANSFORMATION_REGISTRY = {
    "shift_scale": shift_and_scale,
    # Future methods can be added like:
    # "log_transform": log_transform,
    # "moving_average": moving_average_transform,
}


def apply_all_transformations(series: pd.Series, method: str = None, **kwargs) -> pd.Series:
    """
    Applies the specified transformation method using additional kwargs.

    Parameters:
        series (pd.Series): Input time series.
        method (str): Name of transformation method (must be in registry).
        kwargs: Arguments required by the chosen method.

    Returns:
        pd.Series: Transformed time series, or original if method is None.
    """
    if method is None:
        return series

    if method not in TRANSFORMATION_REGISTRY:
        raise ValueError(f"Unknown transformation method: '{method}'")

    transform_fn = TRANSFORMATION_REGISTRY[method]
    return transform_fn(series, **kwargs)
