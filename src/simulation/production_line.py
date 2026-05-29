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




def calculate_objective(throughput, buffer_capacities, alpha=0.02):
    total_buffer_capacity = sum(buffer_capacities)
    objective = throughput - alpha * total_buffer_capacity
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


# # -----------------------------
# # Example: one configuration
# # -----------------------------
#
# buffers = [5, 5, 5, 5, 5]
#
# throughput = run_simulation(buffers)
#
# print("Buffer capacities:", buffers)
# print("Throughput:", round(throughput, 2), "chairs/hour")

# results = []
#
# for buffer4_capacity in range(1, 21):
#     buffers = [5, buffer4_capacity,5 ,5 , 5]
#
#     throughput = 0
#     for _ in range(100):
#         throughput += run_simulation(
#         buffer_capacities=buffers,
#         simulation_time=5 * 8 * 60,
#         seed=47
#     )
#
#     results.append({
#         "Buffer 4 capacity": buffer4_capacity,
#         "Throughput": throughput
#     })
#
# df = pd.DataFrame(results)
# print(df)
#
# plt.figure(figsize=(8, 5))
# plt.plot(df["Buffer 4 capacity"], df["Throughput"], marker="o")
# plt.xlabel("Buffer 4 capacity: Assembly → Painting")
# plt.ylabel("Throughput, chairs/hour")
# plt.title("Throughput vs Buffer 2 Capacity")
# plt.grid(True)
# plt.show()

results = []

print("\n===== TUNING PROCESSING MEAN TIMES =====")

parameter_ranges = {
    "cutting_mean_time": [2, 3, 4, 5, 6, 7, 8],
    "drilling_mean_time": [1, 2, 3, 4, 5, 6],
    "sanding_mean_time": [5, 7, 10, 12, 15, 18, 20],
}

for param_name, values in parameter_ranges.items():

    print(f"\n===== TUNING {param_name} =====")

    for value in values:

        throughputs = []

        for sim_run in range(10):

            params = BASELINE.copy()
            params[param_name] = value

            throughput = run_simulation(
                buffer_capacities=[5, 5, 5, 5, 5],
                params=params,
                simulation_time=5 * 8 * 60,
                seed=47 + sim_run
            )

            throughputs.append(throughput)

            results.append({
                "Parameter": param_name,
                "Value": value,
                "Simulation_run": sim_run + 1,
                "Throughput": throughput
            })

        print(
            f"{param_name}={value} | "
            f"Mean={np.mean(throughputs):.2f} | "
            f"Std={np.std(throughputs):.2f}"
        )

df = pd.DataFrame(results)
df.to_csv("test.csv", index=False)

print("Saved: test.csv")