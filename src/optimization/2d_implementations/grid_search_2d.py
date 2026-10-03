import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.production_line import BASELINE, run_simulation


class GridSearch2D:
    """Exhaustive 2D reference search over Buffer 3 and Buffer 4."""

    def __init__(
        self,
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        simulation_seed=47,
        output_dir="results/grid_search_2d",
    ):
        self.alpha = alpha
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.fixed_buffers = tuple(fixed_buffers)
        self.simulation_seed = simulation_seed

        if self.buffer_min > self.buffer_max:
            raise ValueError("buffer_min must not exceed buffer_max.")
        if len(self.fixed_buffers) != 5:
            raise ValueError("fixed_buffers must contain five capacities.")
        if self.n_replications < 1:
            raise ValueError("n_replications must be positive.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.function_evaluations = 0
        self.simulation_runs = 0

    def full_buffer_vector(self, pair):
        """Map (b3, b4) to the full simulator input vector."""
        b3, b4 = (int(value) for value in pair)
        buffers = list(self.fixed_buffers)
        buffers[2] = b3
        buffers[3] = b4
        return buffers

    def evaluate_pair(self, pair):
        buffers = self.full_buffer_vector(pair)
        throughputs = []

        for run in range(self.n_replications):
            throughputs.append(
                run_simulation(
                    buffer_capacities=buffers,
                    params=BASELINE.copy(),
                    simulation_time=self.simulation_time,
                    seed=self.simulation_seed + run,
                )
            )

        mean_throughput = float(np.mean(throughputs))
        std_throughput = float(np.std(throughputs))
        cost = int(sum(buffers))
        objective = float(self.alpha * mean_throughput - cost)

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications
        return objective, mean_throughput, std_throughput, cost, buffers

    def plot_heatmaps(self, df):
        """Create objective and throughput maps for the full 2D grid."""
        objective = df.pivot(
            index="Buffer_4_Capacity",
            columns="Buffer_3_Capacity",
            values="Objective",
        ).sort_index().sort_index(axis=1)
        throughput = df.pivot(
            index="Buffer_4_Capacity",
            columns="Buffer_3_Capacity",
            values="Mean_Throughput",
        ).sort_index().sort_index(axis=1)

        best = df.loc[df["Objective"].idxmax()]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

        objective_image = axes[0].imshow(
            objective.values,
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        axes[0].scatter(
            best["Buffer_3_Capacity"] - self.buffer_min,
            best["Buffer_4_Capacity"] - self.buffer_min,
            marker="*",
            s=250,
            color="red",
            edgecolor="white",
            linewidth=0.8,
            label="Best grid point",
        )
        axes[0].set_title("Objective over the Full 2D Grid")
        axes[0].set_xlabel("Buffer 3 capacity")
        axes[0].set_ylabel("Buffer 4 capacity")
        axes[0].set_xticks(np.arange(len(objective.columns)), objective.columns)
        axes[0].set_yticks(np.arange(len(objective.index)), objective.index)
        axes[0].legend(loc="lower right")
        figure.colorbar(objective_image, ax=axes[0], label="Objective")

        throughput_image = axes[1].imshow(
            throughput.values,
            origin="lower",
            aspect="auto",
            cmap="cividis",
        )
        axes[1].set_title("Mean Throughput over the Full 2D Grid")
        axes[1].set_xlabel("Buffer 3 capacity")
        axes[1].set_ylabel("Buffer 4 capacity")
        axes[1].set_xticks(np.arange(len(throughput.columns)), throughput.columns)
        axes[1].set_yticks(np.arange(len(throughput.index)), throughput.index)
        figure.colorbar(throughput_image, ax=axes[1], label="Chairs per hour")

        output_path = self.output_dir / "grid_search_2d_heatmaps.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.show()
        return output_path

    def run(self):
        values = range(self.buffer_min, self.buffer_max + 1)
        pairs = list(itertools.product(values, repeat=2))
        total_pairs = len(pairs)
        results = []

        print("\n===== 2D Grid Search Started =====")
        print("Optimised buffers: 3 and 4")
        print(f"Fixed configuration: {self.fixed_buffers}")
        print(f"Configurations: {total_pairs}")
        print(f"N replications per configuration: {self.n_replications}")

        for index, pair in enumerate(pairs, start=1):
            objective, mean_throughput, std_throughput, cost, buffers = (
                self.evaluate_pair(pair)
            )
            results.append({
                "Buffer_1_Capacity": buffers[0],
                "Buffer_2_Capacity": buffers[1],
                "Buffer_3_Capacity": buffers[2],
                "Buffer_4_Capacity": buffers[3],
                "Buffer_5_Capacity": buffers[4],
                "Objective": objective,
                "Mean_Throughput": mean_throughput,
                "Std_Throughput": std_throughput,
                "Cost": cost,
                "Alpha": self.alpha,
                "N_Replications": self.n_replications,
                "Function_Evaluations": self.function_evaluations,
                "Simulation_Runs": self.simulation_runs,
                "Type": "full_grid_search",
            })

            if index % 10 == 0 or index == total_pairs:
                print(
                    f"Completed {index}/{total_pairs} | "
                    f"latest b3,b4={pair} | objective={objective:.3f}"
                )

        df = pd.DataFrame(results)
        best_row = df.loc[df["Objective"].idxmax()]
        csv_path = self.output_dir / "grid_search_2d.csv"
        df.to_csv(csv_path, index=False)
        heatmap_path = self.plot_heatmaps(df)

        print("\n===== Best 2D Grid Search Result =====")
        print(best_row)
        print("\n===== Evaluation Budget =====")
        print(f"Function evaluations: {self.function_evaluations}")
        print(f"Simulation runs: {self.simulation_runs}")
        print(f"\nSaved results to: {csv_path}")
        print(f"Saved heatmaps to: {heatmap_path}")
        return df, best_row


if __name__ == "__main__":
    search = GridSearch2D(
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        simulation_seed=47,
    )
    search.run()
