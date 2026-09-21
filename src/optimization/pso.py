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
        inertia=0.5,
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

    def discretize_position(self, position):
        return np.clip(
            np.rint(position),
            self.buffer_min,
            self.buffer_max,
        ).astype(int)

    def evaluate_buffers(self, position):
        buffers = self.discretize_position(position)

        throughputs = []

        for run in range(self.n_replications):
            throughput = run_simulation(
                buffer_capacities=buffers.tolist(),
                params=BASELINE.copy(),
                simulation_time=self.simulation_time,
                seed=47 + run,
            )
            throughputs.append(throughput)

        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)

        cost = int(np.sum(buffers))
        objective = self.alpha * mean_throughput - cost

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications

        return objective, mean_throughput, std_throughput, cost, buffers

    def run(self):
        print("\n===== PSO Started =====")
        print(f"Alpha: {self.alpha}")
        print(f"Particles: {self.n_particles}")
        print(f"Iterations: {self.n_iterations}")
        n_dimensions = 5

        positions = np.random.uniform(
            self.buffer_min,
            self.buffer_max,
            size=(self.n_particles, n_dimensions),
        )

        velocities = np.random.uniform(
            -1,
            1,
            size=(self.n_particles, n_dimensions),
        )

        personal_best_positions = positions.copy()
        personal_best_buffers = np.zeros(
            (self.n_particles, n_dimensions),
            dtype=int,
        )
        personal_best_scores = np.full(self.n_particles, -np.inf)

        global_best_position = None
        global_best_buffers = None
        global_best_score = -np.inf

        for iteration in range(self.n_iterations + 1):
            for i in range(self.n_particles):

                objective, mean_throughput, std_throughput, cost, evaluated_position = (
                    self.evaluate_buffers(positions[i])
                )

                b1, b2, b3, b4, b5 = evaluated_position

                if objective > personal_best_scores[i]:
                    personal_best_scores[i] = objective
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_buffers[i] = evaluated_position.copy()

                if objective > global_best_score:
                    global_best_score = objective
                    global_best_position = positions[i].copy()
                    global_best_buffers = evaluated_position.copy()

                self.results.append({
                    "Iteration": iteration,
                    "Particle": i,

                    "Buffer_1_Capacity": int(b1),
                    "Buffer_2_Capacity": int(b2),
                    "Buffer_3_Capacity": int(b3),
                    "Buffer_4_Capacity": int(b4),
                    "Buffer_5_Capacity": int(b5),

                    "Position_Buffer_1": positions[i][0],
                    "Position_Buffer_2": positions[i][1],
                    "Position_Buffer_3": positions[i][2],
                    "Position_Buffer_4": positions[i][3],
                    "Position_Buffer_5": positions[i][4],

                    "Velocity_Buffer_1": velocities[i][0],
                    "Velocity_Buffer_2": velocities[i][1],
                    "Velocity_Buffer_3": velocities[i][2],
                    "Velocity_Buffer_4": velocities[i][3],
                    "Velocity_Buffer_5": velocities[i][4],

                    "Objective": objective,
                    "Mean_Throughput": mean_throughput,
                    "Std_Throughput": std_throughput,
                    "Cost": cost,
                    "Alpha": self.alpha,
                    "N_Replications": self.n_replications,
                    "Function_Evaluations": self.function_evaluations,
                    "Simulation_Runs": self.simulation_runs,

                    "Personal_Best_Buffer_1": int(personal_best_buffers[i][0]),
                    "Personal_Best_Buffer_2": int(personal_best_buffers[i][1]),
                    "Personal_Best_Buffer_3": int(personal_best_buffers[i][2]),
                    "Personal_Best_Buffer_4": int(personal_best_buffers[i][3]),
                    "Personal_Best_Buffer_5": int(personal_best_buffers[i][4]),

                    "Personal_Best_Objective": personal_best_scores[i],

                    "Global_Best_Buffer_1": int(global_best_buffers[0]),
                    "Global_Best_Buffer_2": int(global_best_buffers[1]),
                    "Global_Best_Buffer_3": int(global_best_buffers[2]),
                    "Global_Best_Buffer_4": int(global_best_buffers[3]),
                    "Global_Best_Buffer_5": int(global_best_buffers[4]),

                    "Global_Best_Objective": global_best_score,
                })

                print(
                    f"iteration={iteration} | "
                    f"particle={i} | "
                    f"buffers=({int(b1)}, {int(b2)}, {int(b3)}, {int(b4)}, {int(b5)}) | "
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

                mutation_probability = 0.15

                if random.random() < mutation_probability:
                    positions[i] = np.random.uniform(
                        self.buffer_min,
                        self.buffer_max,
                        size=n_dimensions,
                    )

        df = pd.DataFrame(self.results)

        best_row = df.loc[df["Objective"].idxmax()]

        output_csv = self.output_dir / "pso_all_buffers.csv"
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
        n_particles=15,
        n_iterations=20,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        random_seed=42,
        inertia=0.8,
        cognitive_weight=1.2,
        social_weight=1.2,
    )

    optimizer.run()