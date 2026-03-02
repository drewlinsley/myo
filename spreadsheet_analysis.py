import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype


# Define data
f = "mean_cell_analysis.csv"
df = pd.read_csv(f)

# Rename experiments
experimental_condition = df.experimental_condition
clean_experimental_condition = []
for ec in experimental_condition:
    if "control" in ec.lower():
        clean_experimental_condition.append("control")
    else:
        clean_experimental_condition.append(ec.lower().split("-")[0])
df["experimental_condition"] = clean_experimental_condition
numeric_cols = [is_numeric_dtype(df[k]) for k in df.columns]
numeric_col_names = df.columns[numeric_cols]

# Group data
grouped = df.groupby(
    [
        "experimental_condition",
        "experiment_name",
        "perturbation"]).mean(
            numeric_only=True).reset_index()

# Process
diffs_means, diffs_stds = {}, {}
exp_means, exp_stds = {}, {}
unique_exps = grouped.experiment_name.unique()
for exp in unique_exps:
    exp_df = grouped[grouped.experiment_name == exp]
    control_idx = exp_df.experimental_condition == "control"
    exp_idx = exp_df.experimental_condition != "control"
    diffs_means[exp] = (exp_df[control_idx][numeric_col_names].mean() - \
        exp_df[exp_idx][numeric_col_names].mean())  # Take mean diff. Could be z/t/whatever
    diffs_stds[exp] = np.sqrt((exp_df[control_idx][numeric_col_names].std() + \
        exp_df[exp_idx][numeric_col_names].std()) * 0.5)  # Get bootrstrapped CI
    exp_means[exp] = exp_df[exp_idx][numeric_col_names].mean()
    exp_stds[exp] = exp_df[exp_idx][numeric_col_names].std()

diffs_means = pd.DataFrame.from_dict(diffs_means, orient="index")
diffs_stds = pd.DataFrame.from_dict(diffs_stds, orient="index")
exp_means = pd.DataFrame.from_dict(exp_means, orient="index")
exp_stds = pd.DataFrame.from_dict(exp_stds, orient="index")

# Columns:
# Index(['Unnamed: 0', 'cell_number', 'spark_score', 'cell_row', 'cell_col',
#        'num_sparks', 'mass', 'mean', 'magnitude', 'fwhm', 'duration', 'radius',
#        'abs_spark_row', 'abs_spark_col', 'rel_spark_row', 'rel_spark_col',
#        'onset_frame'],
#       dtype='object')

# Plot
which_condition = "num_sparks"  # "magnitude"
f, ax = plt.subplots(1, 1)
plt.errorbar(
    x=np.arange(len(exp_means[which_condition])),
    y=exp_means[which_condition],
    yerr=exp_stds[which_condition],
    fmt="o",
    capsize=5)
plt.ylabel(which_condition.replace("_", " "))
plt.xlabel("Experiment")
ax.set_xticks(np.arange(len(exp_means[which_condition])), exp_means.index)
ax.set_xticklabels(exp_means.index, rotation=30)
plt.title("Spark analysis")
plt.show()
