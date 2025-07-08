import pandas as pd
from timeseries.notebooks.generators import generate_synthetic_method1, generate_synthetic_method2_bootstrap
from timeseries.notebooks.transformations import apply_all_transformations


# Generator Method Registry
GENERATOR_REGISTRY = {
    "decomposition": generate_synthetic_method1,
    "bootstrapping": generate_synthetic_method2_bootstrap,
}


def get_generator(name: str):
    """
    Retrieves the generator function based on its name.
    """
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator method: '{name}'")
    return GENERATOR_REGISTRY[name]


def generate_synthetic_series(
    real_series: pd.Series,
    generator_name: str,
    generator_params: dict,
    transformation_params: dict = None
) -> pd.Series:
    """
    Main interface to generate synthetic time series.

    Parameters:
        real_series (pd.Series): Input time series.
        generator_name (str): Method name from the registry.
        generator_params (dict): Parameters for the generator function (e.g., model, period).
        transformation_params (dict): Optional transformation config:
            {
                "method": "shift_scale",
                "scale_factor": value,
                "shift_value": value
            }

    Returns:
        pd.Series: Generated synthetic series.
    """
    generator_fn = get_generator(generator_name)
    all_params = {**generator_params}

    # Merge transformation params if provided
    if transformation_params:
        all_params.update(transformation_params)

    return generator_fn(real_series, **all_params)
