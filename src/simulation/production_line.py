# pip install simpy numpy pandas matplotlib

import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Fixed baseline parameters
# -----------------------------

BASELINE = {
    "arrival_rate_per_hour": 12,

    "cutting_machines": 2,
    "cutting_mean_time": 4,

    "drilling_machines": 2,
    "drilling_mean_time": 3,

    "sanders": 3,
    "sanding_mean_time": 10,

    "assembly_workers": 2,
    "assembly_mean_time": 8,

    "painting_booths": 1,
    "painting_batch_size": 5,
    "painting_batch_time": 20,

    "drying_slots": 10,
    "drying_time": 40,

    "qc_inspectors": 2,
    "qc_mean_time": 4,

    "packaging_mean_time": 3,

    "qc_failure_probability": 0.10,
    "max_rework_attempts": 1,
}

# -----------------------------
# Helper function
# -----------------------------

def exp_time(mean):
    return random.expovariate(1 / mean)


# -----------------------------
# Simulation model
# -----------------------------

class ChairProductionLine:
    def __init__(self, env, params, buffer_capacities):
        self.env = env
        self.params = params

        self.finished_chairs = 0
        self.chair_id = 0

        self.qc_failed_chairs = 0
        self.reworked_chairs = 0
        self.scrapped_chairs = 0

        # Decision variables: buffer capacities
        self.buffer1 = simpy.Store(env, capacity=buffer_capacities[0])  # Cutting -> Drilling
        self.buffer2 = simpy.Store(env, capacity=buffer_capacities[1])  # Drilling -> Sanding
        self.buffer3 = simpy.Store(env, capacity=buffer_capacities[2])  # Sanding -> Assembly
        self.buffer4 = simpy.Store(env, capacity=buffer_capacities[3])  # Assembly -> Painting
        self.buffer5 = simpy.Store(env, capacity=buffer_capacities[4])  # Painting -> Drying

        # Input queue before cutting
        self.raw_input = simpy.Store(env, capacity=float("inf"))

        # Resources / stations
        self.cutting = simpy.Resource(env, capacity=params["cutting_machines"])
        self.drilling = simpy.Resource(env, capacity=params["drilling_machines"])
        self.sanding = simpy.Resource(env, capacity=params["sanders"])
        self.assembly = simpy.Resource(env, capacity=params["assembly_workers"])
        self.painting = simpy.Resource(env, capacity=params["painting_booths"])
        self.drying = simpy.Resource(env, capacity=params["drying_slots"])
        self.qc = simpy.Resource(env, capacity=params["qc_inspectors"])

        # Statistics
        self.finished_chairs = 0
        self.chair_id = 0

    # def generate_chairs(self):
    #     """Raw material / chair jobs arrive into the system."""
    #     arrival_mean = 60 / self.params["arrival_rate_per_hour"]
    #
    #     while True:
    #         yield self.env.timeout(arrival_mean)
    #
    #         self.chair_id += 1
    #         chair = {
    #             "id": self.chair_id,
    #             "arrival_time": self.env.now
    #         }
    #
    #         yield self.raw_input.put(chair)

    def generate_chairs(self):
        """Raw material / chair jobs arrive into the system."""

        arrival_mean = 60 / self.params["arrival_rate_per_hour"]

        # Uniform interval around the mean arrival time
        arrival_min = arrival_mean * 0.5
        arrival_max = arrival_mean * 1.5

        chair = {
            "id": self.chair_id,
            "arrival_time": self.env.now,
            "rework_count": 0
        }

        while True:
            interarrival_time = random.uniform(arrival_min, arrival_max)

            yield self.env.timeout(interarrival_time)

            self.chair_id += 1

            chair = {
                "id": self.chair_id,
                "arrival_time": self.env.now
            }

            yield self.raw_input.put(chair)

    def cutting_process(self):
        while True:
            chair = yield self.raw_input.get()

            with self.cutting.request() as req:
                yield req
                yield self.env.timeout(exp_time(self.params["cutting_mean_time"]))

            yield self.buffer1.put(chair)

    def drilling_process(self):
        while True:
            chair = yield self.buffer1.get()

            with self.drilling.request() as req:
                yield req
                yield self.env.timeout(exp_time(self.params["drilling_mean_time"]))

            yield self.buffer2.put(chair)

    def sanding_process(self):
        while True:
            chair = yield self.buffer2.get()

            with self.sanding.request() as req:
                yield req
                yield self.env.timeout(exp_time(self.params["sanding_mean_time"]))

            yield self.buffer3.put(chair)

    def assembly_process(self):
        while True:
            chair = yield self.buffer3.get()

            with self.assembly.request() as req:
                yield req
                yield self.env.timeout(exp_time(self.params["assembly_mean_time"]))

            yield self.buffer4.put(chair)

    def painting_process(self):
        """Batch station."""
        batch_size = self.params["painting_batch_size"]

        while True:
            batch = []

            for _ in range(batch_size):
                chair = yield self.buffer4.get()
                batch.append(chair)

            with self.painting.request() as req:
                yield req
                yield self.env.timeout(self.params["painting_batch_time"])

            for chair in batch:
                yield self.buffer5.put(chair)

    def drying_and_qc_process(self):
        while True:
            chair = yield self.buffer5.get()

            with self.drying.request() as req:
                yield req
                yield self.env.timeout(self.params["drying_time"])

            with self.qc.request() as req:
                yield req
                yield self.env.timeout(exp_time(self.params["qc_mean_time"]))

            if "rework_count" not in chair:
                chair["rework_count"] = 0
            failed = random.random() < self.params["qc_failure_probability"]

            if failed:
                self.qc_failed_chairs += 1

            if failed and chair["rework_count"] < self.params["max_rework_attempts"]:
                chair["rework_count"] += 1
                self.reworked_chairs += 1

                yield self.buffer3.put(chair)

            elif failed:
                self.scrapped_chairs += 1

            else:
                yield self.env.timeout(exp_time(self.params["packaging_mean_time"]))
                self.finished_chairs += 1




def calculate_objective(throughput, buffer_capacities, alpha=1.0, beta=0.02):
    total_buffer_capacity = sum(buffer_capacities)
    cost = beta * total_buffer_capacity
    objective = alpha * throughput - cost
    return objective


# -----------------------------
# Run one simulation
# -----------------------------

def run_simulation(
    buffer_capacities,
    params=None,
    simulation_time=8 * 60,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    if params is None:
        params = BASELINE.copy()

    env = simpy.Environment()
    line = ChairProductionLine(env, params, buffer_capacities)

    env.process(line.generate_chairs())

    for _ in range(params["cutting_machines"]):
        env.process(line.cutting_process())

    for _ in range(params["drilling_machines"]):
        env.process(line.drilling_process())

    for _ in range(params["sanders"]):
        env.process(line.sanding_process())

    for _ in range(params["assembly_workers"]):
        env.process(line.assembly_process())

    for _ in range(params["painting_booths"]):
        env.process(line.painting_process())

    for _ in range(params["drying_slots"]):
        env.process(line.drying_and_qc_process())

    env.run(until=simulation_time)

    throughput_per_hour = line.finished_chairs / (simulation_time / 60)

    return throughput_per_hour



results = []

alphas = [0.5, 1, 2, 5, 10, 15, 20]

for alpha in alphas:

    for capacity in range(1, 21):

        buffers = [5, 5, 5, 5, 5]
        buffers[3] = capacity   # Buffer 4: Assembly -> Painting

        for sim_run in range(20):

            throughput = run_simulation(
                buffer_capacities=buffers,
                params=BASELINE.copy(),
                simulation_time=5 * 8 * 60,
                seed=47 + sim_run
            )

            cost = sum(buffers)
            negative_cost = -cost
            objective = alpha * throughput - cost

            results.append({
                "Buffer_4_Capacity": capacity,
                "Simulation_run": sim_run + 1,
                "Throughput": throughput,
                "Cost": cost,
                "Negative_Cost": negative_cost,
                "Objective": objective,
                "Alpha": alpha
            })

            print(f'Buffer_4_Capacity: {capacity}___Throughput: {throughput}___Negative_Cost: {negative_cost}')



df = pd.DataFrame(results)
df.to_csv("buffer4_objective_experiment.csv", index=False)

summary = df.groupby(["Alpha", "Buffer_4_Capacity"]).agg(
    Mean_Throughput=("Throughput", "mean"),
    Std_Throughput=("Throughput", "std"),
    Mean_Objective=("Objective", "mean"),
    Std_Objective=("Objective", "std"),
    Cost=("Cost", "mean")
).reset_index()

print(summary)

best_by_alpha = summary.loc[
    summary.groupby("Alpha")["Mean_Objective"].idxmax()
]

print(best_by_alpha)


