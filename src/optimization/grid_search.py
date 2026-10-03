import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.simulation.production_line import BASELINE, run_simulation


def _evaluate_configuration(task):
    """Evaluate one configuration. Kept at module level for multiprocessing."""
    (
        buffers,
        alpha,
        n_replications,
        simulation_time,
        simulation_seed,
    ) = task

    throughputs = []

    for run in range(n_replications):
        throughput = run_simulation(
            buffer_capacities=list(buffers),
            params=BASELINE.copy(),
            simulation_time=simulation_time,
            seed=simulation_seed + run,
        )
        throughputs.append(throughput)

    mean_throughput = float(np.mean(throughputs))
    std_throughput = float(np.std(throughputs))
    cost = int(sum(buffers))
    objective = float(alpha * mean_throughput - cost)

    return {
        "Buffer_1_Capacity": int(buffers[0]),
        "Buffer_2_Capacity": int(buffers[1]),
        "Buffer_3_Capacity": int(buffers[2]),
        "Buffer_4_Capacity": int(buffers[3]),
        "Buffer_5_Capacity": int(buffers[4]),
        "Objective": objective,
        "Mean_Throughput": mean_throughput,
        "Std_Throughput": std_throughput,
        "Cost": cost,
        "Alpha": alpha,
        "N_Replications": n_replications,
    }


class GridSearchOptimizer:
    """Exhaustive reference search over five integer buffer capacities."""

    BUFFER_COLUMNS = [
        "Buffer_1_Capacity",
        "Buffer_2_Capacity",
        "Buffer_3_Capacity",
        "Buffer_4_Capacity",
        "Buffer_5_Capacity",
    ]

    def __init__(
        self,
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        simulation_seed=47,
        n_workers=1,
        checkpoint_every=250,
        resume=True,
        validation_top_k=10,
        validation_replications=50,
        validation_seed=10_000,
        output_dir="results/grid_search",
    ):
        self.alpha = alpha
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.simulation_seed = simulation_seed
        self.n_workers = n_workers
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.validation_top_k = validation_top_k
        self.validation_replications = validation_replications
        self.validation_seed = validation_seed

        if self.buffer_min > self.buffer_max:
            raise ValueError("buffer_min must not exceed buffer_max.")
        if self.n_replications < 1:
            raise ValueError("n_replications must be positive.")
        if self.simulation_time <= 0:
            raise ValueError("simulation_time must be positive.")
        if self.n_workers < 1:
            raise ValueError("n_workers must be positive.")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive.")
        if self.validation_top_k < 1:
            raise ValueError("validation_top_k must be positive.")
        if self.validation_replications < 1:
            raise ValueError("validation_replications must be positive.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_path = self.output_dir / "grid_search_all_buffers.csv"
        self.validation_path = self.output_dir / "grid_search_validation.csv"
        self.metadata_path = self.output_dir / "grid_search_metadata.json"

        values_per_buffer = self.buffer_max - self.buffer_min + 1
        self.total_configurations = values_per_buffer ** 5

    def _metadata(self):
        return {
            "alpha": self.alpha,
            "n_replications": self.n_replications,
            "simulation_time": self.simulation_time,
            "buffer_min": self.buffer_min,
            "buffer_max": self.buffer_max,
            "simulation_seed": self.simulation_seed,
            "dimensions": 5,
        }

    def _prepare_output(self):
        current_metadata = self._metadata()

        if not self.resume:
            self.results_path.unlink(missing_ok=True)
            self.validation_path.unlink(missing_ok=True)
            self.metadata_path.unlink(missing_ok=True)

        if self.results_path.exists():
            if not self.metadata_path.exists():
                raise RuntimeError(
                    "A checkpoint exists without metadata. Use a new output_dir "
                    "or set resume=False to start again."
                )

            saved_metadata = json.loads(self.metadata_path.read_text())
            if saved_metadata != current_metadata:
                raise RuntimeError(
                    "The existing checkpoint was created with different "
                    "parameters. Use a new output_dir or set resume=False."
                )
        else:
            self.metadata_path.write_text(
                json.dumps(current_metadata, indent=2),
                encoding="utf-8",
            )

    def _all_configurations(self):
        values = range(self.buffer_min, self.buffer_max + 1)
        return itertools.product(values, repeat=5)

    def _load_completed_configurations(self):
        if not self.results_path.exists():
            return set(), 0

        previous_results = pd.read_csv(self.results_path)

        if previous_results.duplicated(self.BUFFER_COLUMNS).any():
            raise RuntimeError("The checkpoint contains duplicate configurations.")

        completed = {
            tuple(int(row[column]) for column in self.BUFFER_COLUMNS)
            for _, row in previous_results.iterrows()
        }
        return completed, len(previous_results)

    def _task(self, buffers, n_replications=None, simulation_seed=None):
        return (
            tuple(int(value) for value in buffers),
            self.alpha,
            n_replications or self.n_replications,
            self.simulation_time,
            self.simulation_seed if simulation_seed is None else simulation_seed,
        )

    def _evaluate_tasks(self, tasks, executor=None):
        if executor is None:
            return [_evaluate_configuration(task) for task in tasks]

        return list(executor.map(_evaluate_configuration, tasks, chunksize=10))

    def _append_checkpoint(self, rows):
        frame = pd.DataFrame(rows)
        write_header = not self.results_path.exists()
        frame.to_csv(
            self.results_path,
            mode="a",
            header=write_header,
            index=False,
        )

    @staticmethod
    def _format_duration(seconds):
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        if seconds < 3600:
            return f"{seconds / 60:.1f} minutes"
        return f"{seconds / 3600:.2f} hours"

    def benchmark_runtime(self, n_configurations=100):
        """Time a small sample without writing it to the search checkpoint."""
        n_configurations = min(n_configurations, self.total_configurations)
        configurations = list(
            itertools.islice(self._all_configurations(), n_configurations)
        )
        tasks = [self._task(configuration) for configuration in configurations]

        start = perf_counter()
        if self.n_workers == 1:
            self._evaluate_tasks(tasks)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                self._evaluate_tasks(tasks, executor)
        elapsed = perf_counter() - start

        estimated_seconds = elapsed / n_configurations * self.total_configurations

        print("\n===== Grid Search Runtime Benchmark =====")
        print(f"Workers: {self.n_workers}")
        print(f"Measured configurations: {n_configurations}")
        print(f"Elapsed: {self._format_duration(elapsed)}")
        print(
            "Estimated full search time: "
            f"{self._format_duration(estimated_seconds)}"
        )

        return estimated_seconds

    def run(self):
        """Evaluate every configuration, with resumable CSV checkpoints."""
        self._prepare_output()
        completed, completed_count = self._load_completed_configurations()

        print("\n===== Exhaustive Grid Search Started =====")
        print(f"Search space: {self.buffer_min}...{self.buffer_max}")
        print(f"Total configurations: {self.total_configurations}")
        print(f"Already completed: {completed_count}")
        print(f"Replications per configuration: {self.n_replications}")
        print(f"Workers: {self.n_workers}")

        remaining = (
            configuration
            for configuration in self._all_configurations()
            if configuration not in completed
        )

        start = perf_counter()
        processed_this_run = 0
        executor = None

        try:
            if self.n_workers > 1:
                executor = ProcessPoolExecutor(max_workers=self.n_workers)

            while True:
                batch = list(itertools.islice(remaining, self.checkpoint_every))
                if not batch:
                    break

                tasks = [self._task(configuration) for configuration in batch]
                rows = self._evaluate_tasks(tasks, executor)

                for offset, row in enumerate(rows, start=1):
                    evaluation = completed_count + processed_this_run + offset
                    row["Function_Evaluations"] = evaluation
                    row["Simulation_Runs"] = evaluation * self.n_replications
                    row["Type"] = "grid_search"

                self._append_checkpoint(rows)
                processed_this_run += len(rows)

                total_completed = completed_count + processed_this_run
                elapsed = perf_counter() - start
                rate = processed_this_run / elapsed
                remaining_count = self.total_configurations - total_completed
                eta = remaining_count / rate if rate > 0 else float("inf")

                print(
                    f"Completed {total_completed}/{self.total_configurations} "
                    f"({100 * total_completed / self.total_configurations:.2f}%) | "
                    f"ETA: {self._format_duration(eta)}"
                )
        finally:
            if executor is not None:
                executor.shutdown()

        results = pd.read_csv(self.results_path)
        best_sample_row = results.loc[results["Objective"].idxmax()]

        print("\n===== Best Sample-Average Grid Result =====")
        print(best_sample_row)
        print(f"Saved results to: {self.results_path}")

        validation = self.validate_top_configurations(results)
        best_validated_row = validation.loc[
            validation["Validation_Objective"].idxmax()
        ]

        print("\n===== Best Independently Validated Grid Result =====")
        print(best_validated_row)
        print(f"Saved validation to: {self.validation_path}")

        return results, best_sample_row, validation, best_validated_row

    def validate_top_configurations(self, results):
        """Re-evaluate the top sample-average configurations on new seeds."""
        top_k = min(self.validation_top_k, len(results))
        candidates = results.nlargest(top_k, "Objective")
        configurations = [
            tuple(int(row[column]) for column in self.BUFFER_COLUMNS)
            for _, row in candidates.iterrows()
        ]

        tasks = [
            self._task(
                configuration,
                n_replications=self.validation_replications,
                simulation_seed=self.validation_seed,
            )
            for configuration in configurations
        ]

        if self.n_workers == 1:
            validated_rows = self._evaluate_tasks(tasks)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                validated_rows = self._evaluate_tasks(tasks, executor)

        original_by_configuration = {
            tuple(int(row[column]) for column in self.BUFFER_COLUMNS): row
            for _, row in candidates.iterrows()
        }

        output_rows = []
        for row in validated_rows:
            key = tuple(row[column] for column in self.BUFFER_COLUMNS)
            original = original_by_configuration[key]
            output_rows.append({
                **{column: row[column] for column in self.BUFFER_COLUMNS},
                "Search_Objective": float(original["Objective"]),
                "Search_Mean_Throughput": float(original["Mean_Throughput"]),
                "Validation_Objective": row["Objective"],
                "Validation_Mean_Throughput": row["Mean_Throughput"],
                "Validation_Std_Throughput": row["Std_Throughput"],
                "Cost": row["Cost"],
                "Validation_Replications": self.validation_replications,
                "Validation_Seed_Start": self.validation_seed,
            })

        validation = pd.DataFrame(output_rows).sort_values(
            "Validation_Objective",
            ascending=False,
        )
        validation.to_csv(self.validation_path, index=False)
        return validation


if __name__ == "__main__":
    # Leave False for the first run. This only benchmarks a small sample and
    # estimates the duration of the complete exhaustive search.
    RUN_FULL_SEARCH = True

    optimizer = GridSearchOptimizer(
        alpha=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        simulation_seed=47,
        n_workers=max(1, min(4, (os.cpu_count() or 2) - 1)),
        checkpoint_every=250,
        resume=True,
        validation_top_k=10,
        validation_replications=50,
        validation_seed=10_000,
        output_dir="results/grid_search",
    )

    if RUN_FULL_SEARCH:
        optimizer.run()
    else:
        optimizer.benchmark_runtime(n_configurations=100)
        print("\nSet RUN_FULL_SEARCH = True to start the complete search.")