from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.production_line import BASELINE, run_simulation


class ParticleSwarmOptimizer2D:
    """PSO for Buffer 3 and Buffer 4 capacities.

    Particle positions remain continuous, as in the 5D implementation.  They
    are rounded and clipped only when a production-line simulation is run.
    """

    def __init__(
        self,
        alpha=20,
        n_particles=6,
        n_iterations=10,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        inertia=0.5,
        cognitive_weight=1.5,
        social_weight=1.5,
        mutation_probability=0.15,
        random_seed=42,
        output_dir="results/pso_2d",
        show_plots=True,
    ):
        self.alpha = alpha
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.n_replications = n_replications
        self.simulation_time = simulation_time
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max
        self.fixed_buffers = tuple(fixed_buffers)

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_probability = mutation_probability

        if len(self.fixed_buffers) != 5:
            raise ValueError("fixed_buffers must contain five capacities.")

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots

        self.results = []
        self.function_evaluations = 0
        self.simulation_runs = 0

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

    def wait_for_plots(self):
        """Keep displayed figures open after the final result is printed."""
        if self.show_plots:
            print("\nClose the plot window(s) to finish the program.")
            plt.show(block=True)

    def discretize_position(self, position):
        """Round a continuous 2D particle position to feasible capacities."""
        return np.clip(
            np.rint(position),
            self.buffer_min,
            self.buffer_max,
        ).astype(int)

    def full_buffer_vector(self, pair):
        """Map (b3, b4) back to the simulator's five-buffer input."""
        b3, b4 = (int(value) for value in pair)
        buffers = list(self.fixed_buffers)
        buffers[2] = b3
        buffers[3] = b4
        return np.asarray(buffers, dtype=int)

    def evaluate_position(self, position):
        pair = self.discretize_position(position)
        buffers = self.full_buffer_vector(pair)
        throughputs = []

        for run in range(self.n_replications):
            throughput = run_simulation(
                buffer_capacities=buffers.tolist(),
                params=BASELINE.copy(),
                simulation_time=self.simulation_time,
                seed=47 + run,
            )
            throughputs.append(throughput)

        mean_throughput = float(np.mean(throughputs))
        std_throughput = float(np.std(throughputs))
        # Full cost keeps the numerical objective identical to the 5D study.
        cost = int(np.sum(buffers))
        objective = self.alpha * mean_throughput - cost

        self.function_evaluations += 1
        self.simulation_runs += self.n_replications
        return objective, mean_throughput, std_throughput, cost, pair, buffers

    def plot_swarm_trajectories(self, df):
        """Visualise the movement of all particles in the (b3, b4) plane."""
        figure = plt.figure(figsize=(7, 6))
        for particle, trajectory in df.groupby("Particle"):
            plt.plot(
                trajectory["Position_Buffer_3"],
                trajectory["Position_Buffer_4"],
                marker="o",
                alpha=0.65,
                label=f"Particle {particle}",
            )

        best = df.loc[df["Objective"].idxmax()]
        plt.scatter(
            best["Buffer_3_Capacity"],
            best["Buffer_4_Capacity"],
            marker="*",
            s=250,
            color="red",
            label="Best observed point",
            zorder=3,
        )
        plt.xlim(self.buffer_min - 0.5, self.buffer_max + 0.5)
        plt.ylim(self.buffer_min - 0.5, self.buffer_max + 0.5)
        plt.xlabel("Buffer 3 position")
        plt.ylabel("Buffer 4 position")
        plt.title("PSO: Particle Trajectories in the 2D Search Space")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        return self.save_figure(figure, "pso_2d_particle_trajectories.png")

    def plot_global_best_progress(self, df):
        progress = df.groupby("Iteration", as_index=False)["Global_Best_Objective"].max()
        figure = plt.figure(figsize=(8, 5))
        plt.plot(progress["Iteration"], progress["Global_Best_Objective"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Best objective found so far")
        plt.title("PSO: Global-Best Objective over Iterations")
        plt.grid(True)
        plt.tight_layout()
        return self.save_figure(figure, "pso_2d_global_best_progress.png")

    def run(self):
        print("\n===== 2D PSO Started =====")
        print("Optimised buffers: 3 and 4")
        print(f"Particles: {self.n_particles}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Fixed configuration: {self.fixed_buffers}")

        n_dimensions = 2
        positions = self.rng.uniform(
            self.buffer_min,
            self.buffer_max,
            size=(self.n_particles, n_dimensions),
        )
        velocities = self.rng.uniform(-1, 1, size=(self.n_particles, n_dimensions))

        personal_best_positions = positions.copy()
        personal_best_pairs = np.zeros((self.n_particles, n_dimensions), dtype=int)
        personal_best_scores = np.full(self.n_particles, -np.inf)

        global_best_position = None
        global_best_pair = None
        global_best_score = -np.inf

        # Iteration 0 evaluates the initial swarm, just as in the 5D code.
        for iteration in range(self.n_iterations + 1):
            for particle in range(self.n_particles):
                objective, mean_throughput, std_throughput, cost, pair, buffers = (
                    self.evaluate_position(positions[particle])
                )

                if objective > personal_best_scores[particle]:
                    personal_best_scores[particle] = objective
                    personal_best_positions[particle] = positions[particle].copy()
                    personal_best_pairs[particle] = pair.copy()

                if objective > global_best_score:
                    global_best_score = objective
                    global_best_position = positions[particle].copy()
                    global_best_pair = pair.copy()

                personal_best_buffers = self.full_buffer_vector(personal_best_pairs[particle])
                global_best_buffers = self.full_buffer_vector(global_best_pair)

                self.results.append({
                    "Iteration": iteration,
                    "Particle": particle,
                    "Buffer_1_Capacity": int(buffers[0]),
                    "Buffer_2_Capacity": int(buffers[1]),
                    "Buffer_3_Capacity": int(buffers[2]),
                    "Buffer_4_Capacity": int(buffers[3]),
                    "Buffer_5_Capacity": int(buffers[4]),
                    "Position_Buffer_3": positions[particle][0],
                    "Position_Buffer_4": positions[particle][1],
                    "Velocity_Buffer_3": velocities[particle][0],
                    "Velocity_Buffer_4": velocities[particle][1],
                    "Objective": objective,
                    "Mean_Throughput": mean_throughput,
                    "Std_Throughput": std_throughput,
                    "Cost": cost,
                    "Alpha": self.alpha,
                    "N_Replications": self.n_replications,
                    "Function_Evaluations": self.function_evaluations,
                    "Simulation_Runs": self.simulation_runs,
                    "Personal_Best_Buffer_1": int(personal_best_buffers[0]),
                    "Personal_Best_Buffer_2": int(personal_best_buffers[1]),
                    "Personal_Best_Buffer_3": int(personal_best_buffers[2]),
                    "Personal_Best_Buffer_4": int(personal_best_buffers[3]),
                    "Personal_Best_Buffer_5": int(personal_best_buffers[4]),
                    "Personal_Best_Objective": personal_best_scores[particle],
                    "Global_Best_Buffer_1": int(global_best_buffers[0]),
                    "Global_Best_Buffer_2": int(global_best_buffers[1]),
                    "Global_Best_Buffer_3": int(global_best_buffers[2]),
                    "Global_Best_Buffer_4": int(global_best_buffers[3]),
                    "Global_Best_Buffer_5": int(global_best_buffers[4]),
                    "Global_Best_Objective": global_best_score,
                })

                print(
                    f"iteration={iteration} | particle={particle} | "
                    f"b3,b4=({int(pair[0])}, {int(pair[1])}) | "
                    f"objective={objective:.3f} | gbest={global_best_score:.3f}"
                )

            # This is the same velocity and position update as the 5D PSO.
            for particle in range(self.n_particles):
                r1 = self.rng.random(n_dimensions)
                r2 = self.rng.random(n_dimensions)
                cognitive = (
                    self.cognitive_weight * r1
                    * (personal_best_positions[particle] - positions[particle])
                )
                social = (
                    self.social_weight * r2
                    * (global_best_position - positions[particle])
                )
                velocities[particle] = (
                    self.inertia * velocities[particle] + cognitive + social
                )
                positions[particle] = np.clip(
                    positions[particle] + velocities[particle],
                    self.buffer_min,
                    self.buffer_max,
                )

                if self.rng.random() < self.mutation_probability:
                    positions[particle] = self.rng.uniform(
                        self.buffer_min,
                        self.buffer_max,
                        size=n_dimensions,
                    )

        df = pd.DataFrame(self.results)
        best_row = df.loc[df["Objective"].idxmax()]
        output_csv = self.output_dir / "pso_2d.csv"
        df.to_csv(output_csv, index=False)

        print("\n===== Best 2D PSO Result =====")
        print(best_row)
        print("\n===== Evaluation Budget =====")
        print(f"Function evaluations: {self.function_evaluations}")
        print(f"Simulation runs: {self.simulation_runs}")
        print(f"\nSaved results to: {output_csv}")
        trajectory_path = self.plot_swarm_trajectories(df)
        progress_path = self.plot_global_best_progress(df)
        print(f"Saved plots to: {trajectory_path} and {progress_path}")
        self.wait_for_plots()
        return df, best_row


if __name__ == "__main__":
    optimizer = ParticleSwarmOptimizer2D(
        alpha=20,
        n_particles=6,
        n_iterations=10,
        n_replications=5,
        simulation_time=5 * 8 * 60,
        buffer_min=1,
        buffer_max=10,
        fixed_buffers=(5, 5, 5, 5, 5),
        inertia=0.5,
        cognitive_weight=1.5,
        social_weight=1.5,
        mutation_probability=0.15,
        random_seed=42,
    )
    optimizer.run()
