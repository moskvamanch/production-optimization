import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/aram/Downloads/Graz_Uni/main/DS_sem3/project in ds/production-optimization/src/simulation/single_buffer_tuning_raw.csv")

for tuned_buffer in sorted(df["Tuned_Buffer"].unique()):

    subset = df[df["Tuned_Buffer"] == tuned_buffer]

    capacities = sorted(subset["Capacity"].unique())

    data = [
        subset[subset["Capacity"] == cap]["Throughput"]
        for cap in capacities
    ]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data)

    plt.xticks(
        range(1, len(capacities) + 1),
        capacities
    )

    plt.xlabel(f"Buffer {tuned_buffer} Capacity")
    plt.ylabel("Throughput, chairs/hour")
    plt.title(f"Boxplot: Throughput vs Buffer {tuned_buffer} Capacity")
    plt.grid(True)

    plt.show()