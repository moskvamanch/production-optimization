import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/Users/aram/Downloads/Graz_Uni/main/DS_sem3/project in ds/production-optimization/src/simulation/single_buffer_tuning_with_cost.csv"
)

for tuned_buffer in sorted(df["Tuned_Buffer"].unique()):

    subset = df[df["Tuned_Buffer"] == tuned_buffer]

    grouped = subset.groupby("Capacity")["Objective"]

    mean_values = grouped.mean()
    std_values = grouped.std()

    capacities = mean_values.index

    upper = mean_values + std_values
    lower = mean_values - std_values

    plt.figure(figsize=(12, 6))

    plt.plot(
        capacities,
        mean_values,
        marker='o',
        label='Mean objective'
    )

    plt.fill_between(
        capacities,
        lower,
        upper,
        alpha=0.3,
        label='±1 std'
    )

    plt.xlabel(f"Buffer {tuned_buffer} Capacity")
    plt.ylabel("Objective")

    plt.title(
        f"Objective vs Buffer {tuned_buffer} Capacity"
    )

    plt.grid(True)
    plt.legend()

    plt.show()