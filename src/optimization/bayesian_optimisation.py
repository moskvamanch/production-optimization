import random
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from src.simulation.production_line import run_simulation, BASELINE


class BayesianOptimizer:
    def __init__(
        self,
        alpha=20,
        n_replications=20,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=20,
        n_initial_points=4,
        n_iterations=10,
        random_seed=42,
        output_dir="results/bayesian_optimization",
    ):
        self.alpha = alpha
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.random_seed = random_seed

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        values = np.arange(buffer_min, buffer_max + 1)

        self.all_capacities = list(
            itertools.product(values, repeat=5)
        )


        self.evaluated_capacities = []
        self.results = []

        self.function_evaluations = 0
        self.simulation_runs = 0

        random.seed(random_seed)
        np.random.seed(random_seed)

    def evaluate_buffers(self, buffer_capacities):
        """
        Evaluate all buffer capacities using Sample Average Approximation.
        """
        buffers = [int(b) for b in buffer_capacities]

        throughputs = []

        for run in range(self.n_replications):
            throughput = run_simulation(
                buffer_capacities=buffers,
                params=BASELINE.copy(),
                simulation_time=self.simulation_time,
                seed=47 + run,
            )
            throughputs.append(throughput)

        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)

        cost = sum(buffers)
        objective = self.alpha * mean_throughput - cost

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications

        return objective, mean_throughput, std_throughput, cost

    def build_model(self):
        """
        Fit Gaussian Process model on evaluated points.
        """
        X_train = np.array(self.evaluated_capacities)
        y_train = np.array([r["Objective"] for r in self.results])

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
        """
        Expected Improvement acquisition function.
        """
        y_train = np.array([r["Objective"] for r in self.results])
        best_y = np.max(y_train)

        mu, sigma = model.predict(X_candidates, return_std=True)

        sigma = sigma.reshape(-1)
        mu = mu.reshape(-1)

        improvement = mu - best_y - xi

        with np.errstate(divide="ignore"):
            z = improvement / sigma

        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] = 0.0

        return ei

    def choose_next_capacity(self, model):
        """
        Choose next unevaluated (b3, b4) pair by maximizing acquisition.
        """
        unevaluated = [
            pair for pair in self.all_capacities
            if pair not in self.evaluated_capacities
        ]

        if len(unevaluated) == 0:
            return None

        X_candidates = np.array(unevaluated)
        acquisition_values = self.expected_improvement(X_candidates, model)

        next_pair = unevaluated[int(np.argmax(acquisition_values))]

        return next_pair

    def add_result(self, iteration, capacity_tuple, result_type):
        buffers = tuple(int(b) for b in capacity_tuple)

        objective, mean_throughput, std_throughput, cost = self.evaluate_buffers(
            buffers
        )

        self.evaluated_capacities.append(buffers)

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
            f"{result_type.upper()} | "
            f"iteration={iteration} | "
            f"buffers={buffers} | "
            f"objective={objective:.3f} | "
            f"throughput={mean_throughput:.3f} | "
            f"function evals={self.function_evaluations} | "
            f"sim runs={self.simulation_runs}"
        )

    def plot_gp(self, model, iteration):
        """
        Plot GP mean, uncertainty interval, measured points, and current best.
        """
        X_plot = self.all_capacities.reshape(-1, 1)

        mu, sigma = model.predict(X_plot, return_std=True)

        evaluated_x = np.array(self.evaluated_capacities)
        evaluated_y = np.array([r["Objective"] for r in self.results])

        best_idx = np.argmax(evaluated_y)
        best_x = evaluated_x[best_idx]
        best_y = evaluated_y[best_idx]

        plt.figure(figsize=(10, 6))

        plt.plot(
            self.all_capacities,
            mu,
            marker="o",
            label="GP mean prediction",
        )

        plt.fill_between(
            self.all_capacities,
            mu - 2 * sigma,
            mu + 2 * sigma,
            alpha=0.2,
            label="GP uncertainty ±2 std",
        )

        plt.scatter(
            evaluated_x,
            evaluated_y,
            s=80,
            label="Evaluated capacities",
        )

        plt.axvline(
            best_x,
            linestyle="--",
            label=f"Current best capacity = {best_x}",
        )

        plt.xlabel("Buffer 4 Capacity")
        plt.ylabel("Objective")
        plt.title(f"Bayesian Optimization GP after iteration {iteration}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # path = self.output_dir / f"bo_gp_iteration_{iteration}.png"
        # plt.savefig(path, dpi=150)
        plt.show()

    def run(self):
        """
        Run Bayesian Optimization.
        """
        print("\n===== Bayesian Optimization Started =====")
        print(f"Alpha: {self.alpha}")
        print(f"N replications per evaluation: {self.n_replications}")
        print(f"Search space: {self.buffer_min}...{self.buffer_max}")

        initial_points = random.sample(
            list(self.all_capacities),
            self.n_initial_points,
        )

        for capacity_tuple in initial_points:
            self.add_result(
                iteration=0,
                capacity_tuple=capacity_tuple,
                result_type="initial",
            )

        model = self.build_model()
        # self.plot_gp(model, iteration=0)

        for iteration in range(1, self.n_iterations + 1):
            model = self.build_model()

            next_capacity_pair = self.choose_next_capacity(model)

            if next_capacity_pair is None:
                break

            self.add_result(
                iteration=iteration,
                capacity_tuple=next_capacity_pair,
                result_type="bayesian",
            )

            model = self.build_model()
            # self.plot_gp(model, iteration=iteration)

        df = pd.DataFrame(self.results)

        best_row = df.loc[df["Objective"].idxmax()]

        output_csv = self.output_dir / "bayesian_optimization_all_buffers.csv"
        df.to_csv(output_csv, index=False)

        print("\n===== Best Bayesian Optimization Result =====")
        print(best_row)

        print("\n===== Evaluation Budget =====")
        print(f"Function evaluations: {self.function_evaluations}")
        print(f"Simulation runs: {self.simulation_runs}")

        print(f"\nSaved results to: {output_csv}")

        return df, best_row


if __name__ == "__main__":
    optimizer = BayesianOptimizer(
        alpha=20,
        n_replications=20,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        n_initial_points=12,
        n_iterations=25,
        random_seed=42,
    )

    optimizer.run()