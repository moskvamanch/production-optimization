import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/aram/Downloads/Graz_Uni/main/DS_sem3/project in ds/production-optimization/src/simulation/test.csv")

for param_name in sorted(df["Parameter"].unique()):

    subset = df[df["Parameter"] == param_name]
    values = sorted(subset["Value"].unique())

    data = [
        subset[subset["Value"] == value]["Throughput"]
        for value in values
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data)

    plt.xticks(
        range(1, len(values) + 1),
        values
    )

    plt.xlabel(param_name)
    plt.ylabel("Throughput, chairs/hour")
    plt.title(f"Throughput vs {param_name}")
    plt.grid(True)

    plt.show()