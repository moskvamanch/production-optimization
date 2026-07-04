import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.production_line import run_simulation, BASELINE


class ParticleSwarmOptimizer:
    def __init__(
        self,
        alpha=20,
        n_particles=6,
        n_iterations=10,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=20,
        inertia=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
        random_seed=42,
        output_dir="results/pso",
    ):
        self.alpha = alpha
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.function_evaluations = 0
        self.simulation_runs = 0

    def evaluate_buffer4(self, capacity):
        buffers = [5, 5, 5, 5, 5]
        buffers[3] = int(capacity)

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

    def run(self):
        print("\n===== PSO Started =====")
        print(f"Alpha: {self.alpha}")
        print(f"Particles: {self.n_particles}")
        print(f"Iterations: {self.n_iterations}")

        positions = np.random.uniform(
            self.buffer_min,
            self.buffer_max,
            self.n_particles,
        )

        velocities = np.random.uniform(
            -1,
            1,
            self.n_particles,
        )

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.n_particles, -np.inf)

        global_best_position = None
        global_best_score = -np.inf

        for iteration in range(self.n_iterations + 1):
            for i in range(self.n_particles):
                capacity = int(round(positions[i]))
                capacity = int(np.clip(capacity, self.buffer_min, self.buffer_max))

                objective, mean_throughput, std_throughput, cost = self.evaluate_buffer4(capacity)

                if objective > personal_best_scores[i]:
                    personal_best_scores[i] = objective
                    personal_best_positions[i] = positions[i]

                if objective > global_best_score:
                    global_best_score = objective
                    global_best_position = positions[i]

                self.results.append({
                    "Iteration": iteration,
                    "Particle": i,
                    "Buffer_4_Capacity": capacity,
                    "Position": positions[i],
                    "Velocity": velocities[i],
                    "Objective": objective,
                    "Mean_Throughput": mean_throughput,
                    "Std_Throughput": std_throughput,
                    "Cost": cost,
                    "Alpha": self.alpha,
                    "N_Replications": self.n_replications,
                    "Function_Evaluations": self.function_evaluations,
                    "Simulation_Runs": self.simulation_runs,
                    "Personal_Best_Position": personal_best_positions[i],
                    "Personal_Best_Objective": personal_best_scores[i],
                    "Global_Best_Position": global_best_position,
                    "Global_Best_Objective": global_best_score,
                })

                print(
                    f"iteration={iteration} | "
                    f"particle={i} | "
                    f"capacity={capacity} | "
                    f"objective={objective:.3f} | "
                    f"gbest={global_best_score:.3f}"
                )

            for i in range(self.n_particles):
                r1 = random.random()
                r2 = random.random()

                cognitive = (
                    self.cognitive_weight
                    * r1
                    * (personal_best_positions[i] - positions[i])
                )

                social = (
                    self.social_weight
                    * r2
                    * (global_best_position - positions[i])
                )

                velocities[i] = (
                    self.inertia * velocities[i]
                    + cognitive
                    + social
                )

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.buffer_min, self.buffer_max)

        df = pd.DataFrame(self.results)

        best_row = df.loc[df["Objective"].idxmax()]

        output_csv = self.output_dir / "pso_buffer4.csv"
        df.to_csv(output_csv, index=False)

        print("\n===== Best PSO Result =====")
        print(best_row)

        print("\n===== Evaluation Budget =====")
        print(f"Function evaluations: {self.function_evaluations}")
        print(f"Simulation runs: {self.simulation_runs}")

        print(f"\nSaved results to: {output_csv}")

        return df, best_row


if __name__ == "__main__":
    optimizer = ParticleSwarmOptimizer(
        alpha=20,
        n_particles=6,
        n_iterations=10,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=20,
        random_seed=42,
    )

    optimizer.run()