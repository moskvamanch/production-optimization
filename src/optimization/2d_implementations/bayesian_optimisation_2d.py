import itertools
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.simulation.production_line import BASELINE, run_simulation


class BayesianOptimizer2D:
    """Bayesian optimisation of Buffer 3 and Buffer 4 capacities only.

    The other capacities remain fixed at their baseline values.  The simulator
    still receives all five capacities, so the 2D experiment uses exactly the
    same production-line model and objective function as the 5D experiment.
    """

    def __init__(
        self,
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=20,
        fixed_buffers=(5, 5, 5, 5, 5),
        n_initial_points=6,
        n_iterations=60,
        random_seed=42,
        output_dir="results/bayesian_optimization_2d",
    ):
        self.alpha = alpha
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.fixed_buffers = tuple(fixed_buffers)
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.random_seed = random_seed

        if len(self.fixed_buffers) != 5:
            raise ValueError("fixed_buffers must contain five capacities.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        values = np.arange(buffer_min, buffer_max + 1)
        self.all_pairs = list(itertools.product(values, repeat=2))
        self.evaluated_pairs = []
        self.results = []
        self.function_evaluations = 0
        self.simulation_runs = 0

        random.seed(random_seed)
        np.random.seed(random_seed)

    def full_buffer_vector(self, pair):
        """Map a 2D point (b3, b4) to a complete 5-buffer configuration."""
        b3, b4 = (int(value) for value in pair)
        buffers = list(self.fixed_buffers)
        buffers[2] = b3  # Sanding -> Assembly
        buffers[3] = b4  # Assembly -> Painting
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
                    seed=47 + run,
                )
            )

        mean_throughput = float(np.mean(throughputs))
        std_throughput = float(np.std(throughputs))
        # The full capacity cost is retained.  The fixed part is a constant,
        # but this keeps the objective numerically identical to the 5D study.
        cost = sum(buffers)
        objective = self.alpha * mean_throughput - cost

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications
        return objective, mean_throughput, std_throughput, cost, buffers

    def build_model(self):
        X_train = np.asarray(self.evaluated_pairs)
        y_train = np.asarray([row["Objective"] for row in self.results])

        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * Matern(length_scale=2.0, nu=2.5)
            + WhiteKernel(noise_level=1.0)
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.random_seed,
            n_restarts_optimizer=2,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        return model

    def expected_improvement(self, X_candidates, model, xi=0.01):
        best_y = max(row["Objective"] for row in self.results)
        mean, std = model.predict(X_candidates, return_std=True)
        improvement = mean - best_y - xi

        with np.errstate(divide="ignore", invalid="ignore"):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei[std == 0.0] = 0.0
        return ei

    def choose_next_pair(self, model):
        unevaluated = [pair for pair in self.all_pairs if pair not in self.evaluated_pairs]
        if not unevaluated:
            return None
        candidates = np.asarray(unevaluated)
        return unevaluated[int(np.argmax(self.expected_improvement(candidates, model)))]

    def add_result(self, iteration, pair, result_type):
        pair = tuple(int(value) for value in pair)
        objective, mean_throughput, std_throughput, cost, buffers = self.evaluate_pair(pair)
        self.evaluated_pairs.append(pair)

        self.results.append({
            "Iteration": iteration,
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
            "Type": result_type,
        })

        print(
            f"{result_type.upper()} | iteration={iteration} | b3,b4={pair} | "
            f"objective={objective:.3f} | throughput={mean_throughput:.3f} | "
            f"function evals={self.function_evaluations}"
        )

    def plot_search_path(self, df):
        plt.figure(figsize=(8, 5))
        plt.plot(df["Function_Evaluations"], df["Buffer_3_Capacity"], marker="o", label="Buffer 3")
        plt.plot(df["Function_Evaluations"], df["Buffer_4_Capacity"], marker="o", label="Buffer 4")
        plt.xlabel("Function evaluations")
        plt.ylabel("Capacity")
        plt.title("Bayesian Optimisation: 2D Search Path")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "bo_2d_search_path.png", dpi=150)
        plt.show()

    def plot_evaluated_points(self, df):
        plt.figure(figsize=(7, 6))
        points = plt.scatter(
            df["Buffer_3_Capacity"],
            df["Buffer_4_Capacity"],
            c=df["Objective"],
            cmap="viridis",
            s=70,
        )
        best = df.loc[df["Objective"].idxmax()]
        plt.scatter(best["Buffer_3_Capacity"], best["Buffer_4_Capacity"],
                    marker="*", s=240, color="red", label="Best observed point")
        plt.xlabel("Buffer 3 capacity")
        plt.ylabel("Buffer 4 capacity")
        plt.title("Evaluated Points in the 2D BO Search")
        plt.colorbar(points, label="Objective")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "bo_2d_evaluated_points.png", dpi=150)
        plt.show()

    def run(self):
        print("\n===== 2D Bayesian Optimisation Started =====")
        print("Optimised buffers: 3 and 4")
        print(f"Fixed configuration: {self.fixed_buffers}")
        print(f"Search space: {self.buffer_min}...{self.buffer_max} for each variable")

        initial_points = random.sample(self.all_pairs, self.n_initial_points)
        for pair in initial_points:
            self.add_result(iteration=0, pair=pair, result_type="initial")

        for iteration in range(1, self.n_iterations + 1):
            next_pair = self.choose_next_pair(self.build_model())
            if next_pair is None:
                break
            self.add_result(iteration=iteration, pair=next_pair, result_type="bayesian")

        df = pd.DataFrame(self.results)
        best_row = df.loc[df["Objective"].idxmax()]
        output_csv = self.output_dir / "bayesian_optimization_2d.csv"
        df.to_csv(output_csv, index=False)

        self.plot_search_path(df)
        self.plot_evaluated_points(df)
        print("\n===== Best 2D BO Result =====")
        print(best_row)
        print(f"\nSaved results to: {output_csv}")
        return df, best_row


if __name__ == "__main__":
    optimizer = BayesianOptimizer2D(
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        # 66 evaluations in total: matches PSO with 6 particles and 10 iterations.
        n_initial_points=6,
        n_iterations=60,
        random_seed=42,
    )
    optimizer.run()
