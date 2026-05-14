from src.simulation.production_line import run_simulation
import pandas as pd

results = []

for capacity in range(1, 21):

    throughput = run_simulation(
        buffer_capacity=capacity
    )

    results.append({
        "buffer_capacity": capacity,
        "throughput": throughput
    })

df = pd.DataFrame(results)

df.to_csv(
    "results/buffer_experiment.csv",
    index=False
)

print(df)