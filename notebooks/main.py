import pandas as pd
import matplotlib.pyplot as plt
from notebooks.pipeline import generate_synthetic_series

# CONFIG
CSV_PATH = "./data/AirPassengers.csv"
DATE_COL = "Month"
VALUE_COL = "#Passengers"
GENERATOR_NAME = "bootstrapping"
GENERATOR_PARAMS = {
    "model": "multiplicative",
    "period": 12
}

# Optional transformations
TRANSFORMATION_PARAMS = None
# = {
#     "method": "shift_scale",       
#     "scale_factor": 1.2,
#     "shift_value": 10
# }
# Use: TRANSFORMATION_PARAMS = None to skip transformation

# load csv
df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL], index_col=DATE_COL)
series = df[VALUE_COL].astype(float).asfreq('MS')

# generate synthetic data
synthetic_series = generate_synthetic_series(
    real_series=series,
    generator_name=GENERATOR_NAME,
    generator_params=GENERATOR_PARAMS,
    transformation_params=TRANSFORMATION_PARAMS
)

# save output
output_path = "./data/synthetic_output.csv"
synthetic_series.to_csv(output_path, header=[f"{VALUE_COL}_synthetic"])

# plot results
def plot_series(original, synthetic, title):
    plt.figure(figsize=(14, 6))
    plt.plot(original.index, original, label="Original", color="blue")
    plt.plot(synthetic.index, synthetic, label="Synthetic", color="red", linestyle="--")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_series(series, synthetic_series, f"Synthetic vs Original ({GENERATOR_NAME})")
