import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/aram/Downloads/Graz_Uni/main/DS_sem3/project in ds/production-optimization/src/simulation/buffer4_objective_experiment.csv")
alpha = df["Alpha"].iloc[0]


summary = (
    df.groupby("Buffer_4_Capacity")
    .agg(
        Mean_Throughput=("Throughput", "mean"),
        Std_Throughput=("Throughput", "std"),
        Mean_Negative_Cost=("Negative_Cost", "mean"),
        Mean_Objective=("Objective", "mean"),
        Std_Objective=("Objective", "std")
    )
    .reset_index()
)

x = summary["Buffer_4_Capacity"]

# -----------------------------
# Plot 1: Throughput
# -----------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    summary["Mean_Throughput"],
    marker="o",
    label="Mean Throughput"
)

plt.fill_between(
    x,
    summary["Mean_Throughput"] - summary["Std_Throughput"],
    summary["Mean_Throughput"] + summary["Std_Throughput"],
    alpha=0.3,
    label="±1 Std"
)

plt.xlabel("Buffer 4 Capacity")
plt.ylabel("Throughput, chairs/hour")
plt.title("Throughput vs Buffer 4 Capacity")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot 2: Negative Cost
# -----------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    summary["Mean_Negative_Cost"],
    marker="o",
    label="-Cost"
)

plt.xlabel("Buffer 4 Capacity")
plt.ylabel("-Cost")
plt.title("-Cost vs Buffer 4 Capacity")
plt.grid(True)
plt.legend()
plt.show()








# -----------------------------
# Plot 3: Objective Function
# -----------------------------

summary = (
    df.groupby(["Alpha", "Buffer_4_Capacity"])
    .agg(
        Mean_Objective=("Objective", "mean"),
        Std_Objective=("Objective", "std")
    )
    .reset_index()
)

plt.figure(figsize=(10,6))

for alpha in sorted(summary["Alpha"].unique()):

    alpha_df = summary[summary["Alpha"] == alpha]

    plt.plot(
        alpha_df["Buffer_4_Capacity"],
        alpha_df["Mean_Objective"],
        marker="o",
        label=f"α = {alpha}"
    )

plt.xlabel("Buffer 4 Capacity")
plt.ylabel("Objective")
plt.title("Objective Function for Different Alpha Values")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# best_row = summary.loc[summary["Mean_Objective"].idxmax()]
# best_capacity = best_row["Buffer_4_Capacity"]
# best_objective = best_row["Mean_Objective"]
#
# plt.figure(figsize=(10, 6))
#
# plt.plot(
#     x,
#     summary["Mean_Objective"],
#     marker="o",
#     label="Mean Objective"
# )
#
# plt.fill_between(
#     x,
#     summary["Mean_Objective"] - summary["Std_Objective"],
#     summary["Mean_Objective"] + summary["Std_Objective"],
#     alpha=0.3,
#     label="±1 Std"
# )
#
# plt.axvline(
#     best_capacity,
#     linestyle="--",
#     label=f"Best capacity = {int(best_capacity)}"
# )
#
# plt.scatter(
#     best_capacity,
#     best_objective,
#     s=100,
#     label=f"Maximum J = {best_objective:.2f}"
# )
#
# plt.xlabel("Buffer 4 Capacity")
# plt.ylabel("Objective")
# plt.title(f"Objective Function: J = {alpha} * Throughput - Cost")
# plt.grid(True)
# plt.legend()
# plt.show()
#
# print("Best Buffer 4 capacity:", int(best_capacity))
# print("Best objective:", round(best_objective, 3))



