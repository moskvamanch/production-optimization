from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.production_line import BASELINE, run_simulation


class FiniteDifferenceOptimizer:
    """Finite-difference optimization for integer buffer capacities."""

    def __init__(
        self,
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        difference_step=1,
        learning_rate=1.0,
        evaluation_budget=100,
        initial_buffers=None,
        random_seed=42,
        simulation_seed=47,
        output_dir="results/finite_difference",
    ):
        self.alpha = alpha
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.difference_step = difference_step
        self.learning_rate = learning_rate
        self.evaluation_budget = evaluation_budget
        self.initial_buffers = initial_buffers
        self.random_seed = random_seed
        self.simulation_seed = simulation_seed

        if self.buffer_min >= self.buffer_max:
            raise ValueError("buffer_min must be smaller than buffer_max.")
        if not isinstance(self.difference_step, int) or self.difference_step < 1:
            raise ValueError("difference_step must be a positive integer.")
        if self.n_replications < 1:
            raise ValueError("n_replications must be positive.")
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be positive.")

        self.rng = np.random.default_rng(random_seed)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache = {}
        self.results = []
        self.function_evaluations = 0
        self.simulation_runs = 0
        self.best_result = None

    def discretize(self, position):
        """Round a position and project it onto the feasible integer domain."""
        return np.clip(
            np.rint(position),
            self.buffer_min,
            self.buffer_max,
        ).astype(int)

    def evaluate_buffers(self, buffer_capacities, iteration, evaluation_type):
        """Evaluate one unique integer configuration using fixed replications."""
        buffers = self.discretize(buffer_capacities)
        key = tuple(int(value) for value in buffers)

        if key in self.cache:
            return self.cache[key], True

        if self.function_evaluations >= self.evaluation_budget:
            return None, False

        throughputs = []

        for run in range(self.n_replications):
            throughput = run_simulation(
                buffer_capacities=list(key),
                params=BASELINE.copy(),
                simulation_time=self.simulation_time,
                seed=self.simulation_seed + run,
            )
            throughputs.append(throughput)

        mean_throughput = float(np.mean(throughputs))
        std_throughput = float(np.std(throughputs))
        cost = int(sum(key))
        objective = float(self.alpha * mean_throughput - cost)

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications

        result = {
            "Iteration": iteration,
            "Buffer_1_Capacity": key[0],
            "Buffer_2_Capacity": key[1],
            "Buffer_3_Capacity": key[2],
            "Buffer_4_Capacity": key[3],
            "Buffer_5_Capacity": key[4],
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

        if (
            self.best_result is None
            or objective > self.best_result["Objective"]
        ):
            self.best_result = result.copy()

        result["Best_Objective"] = self.best_result["Objective"]

        self.cache[key] = result
        self.results.append(result)

        print(
            f"{evaluation_type.upper()} | "
            f"iteration={iteration} | "
            f"buffers={key} | "
            f"objective={objective:.3f} | "
            f"function evals={self.function_evaluations} | "
            f"sim runs={self.simulation_runs}"
        )

        return result, False

    def estimate_gradient(self, position, current_result, iteration):
        """
        Estimate the objective gradient coordinate by coordinate.

        A central difference is used inside the domain. At a boundary, the
        corresponding one-sided difference is used instead.
        """
        gradient = np.zeros(5, dtype=float)
        current_objective = current_result["Objective"]

        for dimension in range(5):
            lower = position.copy()
            upper = position.copy()

            lower[dimension] = max(
                self.buffer_min,
                position[dimension] - self.difference_step,
            )
            upper[dimension] = min(
                self.buffer_max,
                position[dimension] + self.difference_step,
            )

            has_lower = lower[dimension] != position[dimension]
            has_upper = upper[dimension] != position[dimension]

            lower_result = None
            upper_result = None

            if has_lower:
                lower_result, _ = self.evaluate_buffers(
                    lower,
                    iteration,
                    f"difference_minus_b{dimension + 1}",
                )
                if lower_result is None:
                    return None

            if has_upper:
                upper_result, _ = self.evaluate_buffers(
                    upper,
                    iteration,
                    f"difference_plus_b{dimension + 1}",
                )
                if upper_result is None:
                    return None

            if has_lower and has_upper:
                denominator = upper[dimension] - lower[dimension]
                gradient[dimension] = (
                    upper_result["Objective"] - lower_result["Objective"]
                ) / denominator
            elif has_upper:
                denominator = upper[dimension] - position[dimension]
                gradient[dimension] = (
                    upper_result["Objective"] - current_objective
                ) / denominator
            elif has_lower:
                denominator = position[dimension] - lower[dimension]
                gradient[dimension] = (
                    current_objective - lower_result["Objective"]
                ) / denominator

        return gradient

    def propose_position(self, position, gradient):
        """Apply a projected gradient-ascent update and return integer buffers."""
        max_absolute_gradient = float(np.max(np.abs(gradient)))

        if max_absolute_gradient == 0.0:
            return position.copy()

        # Scaling preserves the estimated ascent direction while preventing
        # the objective's numerical scale from causing very large jumps.
        scaled_gradient = gradient / max_absolute_gradient
        continuous_position = (
            position.astype(float) + self.learning_rate * scaled_gradient
        )
        candidate = self.discretize(continuous_position)

        # Rounding can leave an integer position unchanged. In that case,
        # move one unit along the strongest estimated gradient component.
        if np.array_equal(candidate, position) and np.any(gradient != 0.0):
            dimension = int(np.argmax(np.abs(gradient)))
            direction = int(np.sign(gradient[dimension]))
            candidate[dimension] = np.clip(
                candidate[dimension] + direction,
                self.buffer_min,
                self.buffer_max,
            )

        return candidate

    def random_unevaluated_position(self):
        """Return an unseen integer configuration for a new local start."""
        search_space_size = (
            self.buffer_max - self.buffer_min + 1
        ) ** 5

        if len(self.cache) >= search_space_size:
            return None

        while True:
            candidate = self.rng.integers(
                self.buffer_min,
                self.buffer_max + 1,
                size=5,
            )
            if tuple(int(value) for value in candidate) not in self.cache:
                return candidate

    def run(self):
        """Run finite-difference optimization until the budget is exhausted."""
        print("\n===== Finite-Difference Optimization Started =====")
        print(f"Alpha: {self.alpha}")
        print(f"N replications per evaluation: {self.n_replications}")
        print(f"Evaluation budget: {self.evaluation_budget}")
        print(f"Search space: {self.buffer_min}...{self.buffer_max}")

        if self.initial_buffers is None:
            position = np.full(5, 5, dtype=int)
            position = self.discretize(position)
        else:
            if len(self.initial_buffers) != 5:
                raise ValueError("initial_buffers must contain five values.")
            position = self.discretize(self.initial_buffers)

        current_result, _ = self.evaluate_buffers(
            position,
            iteration=0,
            evaluation_type="initial",
        )

        iteration = 1
        consecutive_non_improving_steps = 0

        while self.function_evaluations < self.evaluation_budget:
            gradient = self.estimate_gradient(
                position,
                current_result,
                iteration,
            )

            if gradient is None:
                break

            candidate = self.propose_position(position, gradient)

            if np.array_equal(candidate, position) or np.allclose(gradient, 0.0):
                candidate = self.random_unevaluated_position()
                evaluation_type = "restart"
            else:
                evaluation_type = "gradient_update"

            if candidate is None:
                break

            candidate_result, was_cached = self.evaluate_buffers(
                candidate,
                iteration,
                evaluation_type,
            )

            if candidate_result is None:
                break

            if candidate_result["Objective"] > current_result["Objective"]:
                consecutive_non_improving_steps = 0
            else:
                consecutive_non_improving_steps += 1

            position = self.discretize(candidate)
            current_result = candidate_result

            # A cached or repeatedly non-improving update indicates that the
            # current discrete local search has stalled. Start from an unseen
            # configuration, while preserving the best result found so far.
            if (
                was_cached
                or consecutive_non_improving_steps >= 2
            ) and self.function_evaluations < self.evaluation_budget:
                restart_position = self.random_unevaluated_position()
                if restart_position is None:
                    break

                restart_result, _ = self.evaluate_buffers(
                    restart_position,
                    iteration,
                    "restart",
                )
                if restart_result is None:
                    break

                position = restart_position
                current_result = restart_result
                consecutive_non_improving_steps = 0

            iteration += 1

        df = pd.DataFrame(self.results)
        best_row = df.loc[df["Objective"].idxmax()]

        output_csv = self.output_dir / "finite_difference_all_buffers.csv"
        df.to_csv(output_csv, index=False)

        print("\n===== Best Finite-Difference Result =====")
        print(best_row)

        print("\n===== Evaluation Budget =====")
        print(f"Function evaluations: {self.function_evaluations}")
        print(f"Simulation runs: {self.simulation_runs}")
        print(f"\nSaved results to: {output_csv}")

        return df, best_row


if __name__ == "__main__":
    optimizer = FiniteDifferenceOptimizer(
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        difference_step=1,
        learning_rate=1.0,
        evaluation_budget=100,
        initial_buffers=[5, 5, 5, 5, 5],
        random_seed=42,
        simulation_seed=47,
    )

    optimizer.run()
