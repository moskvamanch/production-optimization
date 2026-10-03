from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.production_line import BASELINE, run_simulation


class FiniteDifferenceOptimizer2D:
    """Finite-difference optimisation of Buffer 3 and Buffer 4 capacities."""

    def __init__(
        self,
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        difference_step=1,
        learning_rate=1.0,
        evaluation_budget=66,
        initial_pair=(5, 5),
        random_seed=42,
        simulation_seed=47,
        output_dir="results/finite_difference_2d",
        show_plots=True,
    ):
        self.alpha = alpha
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.fixed_buffers = tuple(fixed_buffers)
        self.difference_step = difference_step
        self.learning_rate = learning_rate
        self.evaluation_budget = evaluation_budget
        self.initial_pair = initial_pair
        self.random_seed = random_seed
        self.simulation_seed = simulation_seed

        if self.buffer_min >= self.buffer_max:
            raise ValueError("buffer_min must be smaller than buffer_max.")
        if len(self.fixed_buffers) != 5:
            raise ValueError("fixed_buffers must contain five capacities.")
        if not isinstance(self.difference_step, int) or self.difference_step < 1:
            raise ValueError("difference_step must be a positive integer.")
        if self.n_replications < 1:
            raise ValueError("n_replications must be positive.")
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be positive.")

        self.rng = np.random.default_rng(random_seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots

        self.cache = {}
        self.results = []
        self.function_evaluations = 0
        self.simulation_runs = 0
        self.best_result = None

    def save_figure(self, figure, filename):
        """Save a plot and optionally display it without blocking the run."""
        output_path = self.output_dir / filename
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        if self.show_plots:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close(figure)
        return output_path

    def discretize(self, position):
        """Round a two-dimensional position and project it to the domain."""
        return np.clip(
            np.rint(position),
            self.buffer_min,
            self.buffer_max,
        ).astype(int)

    def full_buffer_vector(self, pair):
        """Map a pair (b3, b4) to the five inputs required by the simulator."""
        b3, b4 = (int(value) for value in pair)
        buffers = list(self.fixed_buffers)
        buffers[2] = b3
        buffers[3] = b4
        return np.asarray(buffers, dtype=int)

    def evaluate_pair(self, pair, iteration, evaluation_type):
        """Evaluate one unique integer pair using sample-average approximation."""
        pair = self.discretize(pair)
        key = tuple(int(value) for value in pair)

        if key in self.cache:
            return self.cache[key], True
        if self.function_evaluations >= self.evaluation_budget:
            return None, False

        buffers = self.full_buffer_vector(key)
        throughputs = []
        for run in range(self.n_replications):
            throughput = run_simulation(
                buffer_capacities=buffers.tolist(),
                params=BASELINE.copy(),
                simulation_time=self.simulation_time,
                seed=self.simulation_seed + run,
            )
            throughputs.append(throughput)

        mean_throughput = float(np.mean(throughputs))
        std_throughput = float(np.std(throughputs))
        # Retaining all five capacities keeps the 2D and 5D objectives equal.
        cost = int(np.sum(buffers))
        objective = float(self.alpha * mean_throughput - cost)

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications
        result = {
            "Iteration": iteration,
            "Buffer_1_Capacity": int(buffers[0]),
            "Buffer_2_Capacity": int(buffers[1]),
            "Buffer_3_Capacity": int(buffers[2]),
            "Buffer_4_Capacity": int(buffers[3]),
            "Buffer_5_Capacity": int(buffers[4]),
            "Objective": objective,
            "Mean_Throughput": mean_throughput,
            "Std_Throughput": std_throughput,
            "Cost": cost,
            "Alpha": self.alpha,
            "N_Replications": self.n_replications,
            "Function_Evaluations": self.function_evaluations,
            "Simulation_Runs": self.simulation_runs,
            "Type": evaluation_type,
        }

        if self.best_result is None or objective > self.best_result["Objective"]:
            self.best_result = result.copy()
        result["Best_Objective"] = self.best_result["Objective"]

        self.cache[key] = result
        self.results.append(result)
        print(
            f"{evaluation_type.upper()} | iteration={iteration} | "
            f"b3,b4={key} | objective={objective:.3f} | "
            f"function evals={self.function_evaluations} | "
            f"sim runs={self.simulation_runs}"
        )
        return result, False

    def estimate_gradient(self, position, current_result, iteration):
        """Use central differences, or one-sided differences at boundaries."""
        gradient = np.zeros(2, dtype=float)
        current_objective = current_result["Objective"]

        for dimension, buffer_number in enumerate((3, 4)):
            lower = position.copy()
            upper = position.copy()
            lower[dimension] = max(
                self.buffer_min, position[dimension] - self.difference_step
            )
            upper[dimension] = min(
                self.buffer_max, position[dimension] + self.difference_step
            )

            has_lower = lower[dimension] != position[dimension]
            has_upper = upper[dimension] != position[dimension]
            lower_result = None
            upper_result = None

            if has_lower:
                lower_result, _ = self.evaluate_pair(
                    lower, iteration, f"difference_minus_b{buffer_number}"
                )
                if lower_result is None:
                    return None
            if has_upper:
                upper_result, _ = self.evaluate_pair(
                    upper, iteration, f"difference_plus_b{buffer_number}"
                )
                if upper_result is None:
                    return None

            if has_lower and has_upper:
                gradient[dimension] = (
                    upper_result["Objective"] - lower_result["Objective"]
                ) / (upper[dimension] - lower[dimension])
            elif has_upper:
                gradient[dimension] = (
                    upper_result["Objective"] - current_objective
                ) / (upper[dimension] - position[dimension])
            elif has_lower:
                gradient[dimension] = (
                    current_objective - lower_result["Objective"]
                ) / (position[dimension] - lower[dimension])

        return gradient

    def propose_position(self, position, gradient):
        """Apply the same scaled projected gradient-ascent rule as in 5D."""
        max_absolute_gradient = float(np.max(np.abs(gradient)))
        if max_absolute_gradient == 0.0:
            return position.copy()

        scaled_gradient = gradient / max_absolute_gradient
        candidate = self.discretize(
            position.astype(float) + self.learning_rate * scaled_gradient
        )

        if np.array_equal(candidate, position) and np.any(gradient != 0.0):
            # At a boundary, the strongest gradient component may point
            # outside the feasible domain.  Select the strongest component
            # that can still change an integer coordinate.
            feasible_dimensions = []
            for dimension, value in enumerate(gradient):
                direction = int(np.sign(value))
                if direction == 0:
                    continue
                shifted_value = np.clip(
                    position[dimension] + direction,
                    self.buffer_min,
                    self.buffer_max,
                )
                if shifted_value != position[dimension]:
                    feasible_dimensions.append(dimension)

            if feasible_dimensions:
                dimension = max(
                    feasible_dimensions,
                    key=lambda index: abs(gradient[index]),
                )
                direction = int(np.sign(gradient[dimension]))
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction,
                    self.buffer_min,
                    self.buffer_max,
                )
        return candidate

    def random_unevaluated_pair(self):
        """Return an unseen pair for a restart."""
        search_space_size = (self.buffer_max - self.buffer_min + 1) ** 2
        if len(self.cache) >= search_space_size:
            return None

        while True:
            candidate = self.rng.integers(
                self.buffer_min, self.buffer_max + 1, size=2
            )
            if tuple(int(value) for value in candidate) not in self.cache:
                return candidate

    def plot_search_path(self, df):
        figure = plt.figure(figsize=(8, 5))
        plt.plot(df["Function_Evaluations"], df["Buffer_3_Capacity"], marker="o", label="Buffer 3")
        plt.plot(df["Function_Evaluations"], df["Buffer_4_Capacity"], marker="o", label="Buffer 4")
        plt.xlabel("Function evaluations")
        plt.ylabel("Capacity")
        plt.title("Finite Difference: 2D Search Path")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return self.save_figure(figure, "finite_difference_2d_search_path.png")

    def plot_evaluated_points(self, df):
        figure = plt.figure(figsize=(7, 6))
        points = plt.scatter(
            df["Buffer_3_Capacity"],
            df["Buffer_4_Capacity"],
            c=df["Objective"],
            cmap="viridis",
            s=70,
        )
        best = df.loc[df["Objective"].idxmax()]
        plt.scatter(
            best["Buffer_3_Capacity"], best["Buffer_4_Capacity"],
            marker="*", s=250, color="red", label="Best observed point",
        )
        plt.xlabel("Buffer 3 capacity")
        plt.ylabel("Buffer 4 capacity")
        plt.title("Finite Difference: Evaluated Points in 2D")
        plt.colorbar(points, label="Objective")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return self.save_figure(figure, "finite_difference_2d_evaluated_points.png")

    def run(self):
        print("\n===== 2D Finite-Difference Optimization Started =====")
        print("Optimised buffers: 3 and 4")
        print(f"Fixed configuration: {self.fixed_buffers}")
        print(f"Evaluation budget: {self.evaluation_budget}")

        if self.initial_pair is None:
            position = np.asarray([self.fixed_buffers[2], self.fixed_buffers[3]])
        else:
            if len(self.initial_pair) != 2:
                raise ValueError("initial_pair must contain Buffer 3 and Buffer 4.")
            position = np.asarray(self.initial_pair)
        position = self.discretize(position)

        current_result, _ = self.evaluate_pair(position, 0, "initial")
        iteration = 1
        consecutive_non_improving_steps = 0

        while self.function_evaluations < self.evaluation_budget:
            gradient = self.estimate_gradient(position, current_result, iteration)
            if gradient is None:
                break

            candidate = self.propose_position(position, gradient)
            if np.array_equal(candidate, position) or np.allclose(gradient, 0.0):
                candidate = self.random_unevaluated_pair()
                evaluation_type = "restart"
            else:
                evaluation_type = "gradient_update"
            if candidate is None:
                break

            candidate_result, was_cached = self.evaluate_pair(
                candidate, iteration, evaluation_type
            )
            if candidate_result is None:
                break

            candidate_improves = (
                candidate_result["Objective"] > current_result["Objective"]
            )

            if candidate_improves:
                consecutive_non_improving_steps = 0
            else:
                consecutive_non_improving_steps += 1

            # A cached point can be used to continue an improving trajectory.
            # Moving from the current point back to a cached worse point would
            # create a two-point cycle without consuming evaluations, so in
            # that case keep the current point and let the stall rule restart.
            if candidate_improves or not was_cached:
                position = self.discretize(candidate)
                current_result = candidate_result

            # A cached candidate is a valid already-known neighbour, not a
            # reason to abandon the local finite-difference trajectory.  A
            # restart is used only after actual repeated non-improvement.
            if (
                consecutive_non_improving_steps >= 2
            ) and self.function_evaluations < self.evaluation_budget:
                restart_position = self.random_unevaluated_pair()
                if restart_position is None:
                    break
                restart_result, _ = self.evaluate_pair(
                    restart_position, iteration, "restart"
                )
                if restart_result is None:
                    break
                position = restart_position
                current_result = restart_result
                consecutive_non_improving_steps = 0

            iteration += 1

        df = pd.DataFrame(self.results)
        best_row = df.loc[df["Objective"].idxmax()]
        output_csv = self.output_dir / "finite_difference_2d.csv"
        df.to_csv(output_csv, index=False)

        print("\n===== Best 2D Finite-Difference Result =====")
        print(best_row)
        print("\n===== Evaluation Budget =====")
        print(f"Function evaluations: {self.function_evaluations}")
        print(f"Simulation runs: {self.simulation_runs}")
        print(f"\nSaved results to: {output_csv}")
        path_search = self.plot_search_path(df)
        path_points = self.plot_evaluated_points(df)
        print(f"Saved plots to: {path_search} and {path_points}")
        return df, best_row


if __name__ == "__main__":
    optimizer = FiniteDifferenceOptimizer2D(
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        difference_step=1,
        learning_rate=1.0,
        evaluation_budget=66,
        initial_pair=(5, 5),
        random_seed=42,
        simulation_seed=47,
    )
    optimizer.run()
