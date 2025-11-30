import traci
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv


# -----------------------------
# Q-LEARNING HYPERPARAMETERS
# -----------------------------
ALPHA = 0.1          # Learning rate
GAMMA = 0.95         # Discount factor
EPSILON = 0.1        # Exploration rate
EPISODES = 50        # Number of SUMO runs
STEP_LENGTH = 1      # SUMO step (seconds)

# -----------------------------
# ACTION SPACE
# -----------------------------
# 0 = keep current phase
# 1 = switch to next phase
ACTIONS = [0, 1]

# -----------------------------
# Q-TABLE
# -----------------------------
Q = defaultdict(lambda: {a: 0 for a in ACTIONS})

# -----------------------------
# INTERSECTION LANE GROUPS
# (EDIT THESE FOR YOUR NETWORK!)
# -----------------------------
# Inbound lanes used for delay/queue metrics + RL state
LANES_NS_IN = ["-E10_0", "-E10_1","-E10_2", "-E9_0", "-E9_1", "-E9_2"]     # LANES_NS_IN = ["N_in", "S_in"]
LANES_EW_IN = ["E8_0", "E8_1", "E8_2", "-E11_0", "-E11_1", "-E11_2"]      # LANES_EW_IN = ["E_in", "W_in"]

# Outbound lanes used for throughput counts
LANES_OUT = ["E9_0", "E9_1", "E9_2", "E10_0", "E10_1", "E10_2", "E11_0", "E11_1", "E11_2", "-E8_0", "-E8_1", "-E8_2"]  # LANES_OUT = ["N_out", "S_out", "E_out", "W_out"]

# -----------------------------
# STATE = bucketed queue (NS, EW)
# -----------------------------
def get_state():
    queue_ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES_NS_IN)
    queue_ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES_EW_IN)
    bucket_ns = min(queue_ns // 5, 5)
    bucket_ew = min(queue_ew // 5, 5)
    return (bucket_ns, bucket_ew)

# -----------------------------
# EPSILON-GREEDY ACTION SELECTION
# -----------------------------
def choose_action(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

# -----------------------------
# APPLY ACTION TO TLS
# -----------------------------
def apply_action(action, tls_id="TL1"):
    current_phase = traci.trafficlight.getPhase(tls_id)
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    num_phases = len(logic.phases)

    if action == 0:
        traci.trafficlight.setPhase(tls_id, current_phase)
    else:
        next_phase = (current_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)

# -----------------------------
# REWARD FUNCTION = negative total delay
# -----------------------------
def compute_reward():
    delay_ns = sum(traci.lane.getWaitingTime(l) for l in LANES_NS_IN)
    delay_ew = sum(traci.lane.getWaitingTime(l) for l in LANES_EW_IN)
    total_delay = delay_ns + delay_ew
    return -total_delay

# -----------------------------
# METRICS COLLECTION
# -----------------------------
def compute_metrics():
    inbound_lanes = LANES_NS_IN + LANES_EW_IN
    total_delay = sum(traci.lane.getWaitingTime(l) for l in inbound_lanes)
    avg_delay = total_delay / len(inbound_lanes)
    queue_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in inbound_lanes]
    avg_queue = np.mean(queue_lengths)
    max_queue = np.max(queue_lengths)
    throughput = traci.simulation.getArrivedNumber()
    return avg_delay, avg_queue, max_queue, throughput

# -----------------------------
# MAIN Q-LEARNING TRAINING LOOP
# -----------------------------
def run_qlearning():
    results = []
    # Prepare CSV file
    csv_file = open("qlearning_metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "avg_delay", "avg_queue", "max_queue", "throughput"])


    for ep in range(EPISODES):
        print(f"\n===== EPISODE {ep+1} START =====")
        sumo_cmd = ['sumo', '-c', 'test2.sumocfg', '--step-length', '0.1', "--no-step-log", "true"]
        traci.start(sumo_cmd)

        state = get_state()

        # Per-episode metric storage
        metrics = {"delay": [], "avg_queue": [], "max_queue": [], "throughput": []}

        step = 0
        MAX_STEPS = 2000

        while step < MAX_STEPS:
            traci.simulationStep()
            action = choose_action(state)
            apply_action(action)
            reward = compute_reward()
            new_state = get_state()

            # Q-Learning update
            best_next_action = max(Q[new_state], key=Q[new_state].get)
            Q[state][action] += ALPHA * (reward + GAMMA * Q[new_state][best_next_action] - Q[state][action])

            # Metrics
            avg_delay, avg_queue, max_queue, throughput = compute_metrics()
            metrics["delay"].append(avg_delay)
            metrics["avg_queue"].append(avg_queue)
            metrics["max_queue"].append(max_queue)
            metrics["throughput"].append(throughput)

            state = new_state
            step += 1

        traci.close()

        # Store episode summary (convert to plain Python types)
        episode_summary = {
            "episode": ep + 1,
            "avg_delay": float(np.mean(metrics["delay"])),
            "avg_queue": float(np.mean(metrics["avg_queue"])),
            "max_queue": int(np.max(metrics["max_queue"])),
            "throughput": int(np.sum(metrics["throughput"]))
        }
        # Write to CSV
        csv_writer.writerow([
            episode_summary["episode"],
            episode_summary["avg_delay"],
            episode_summary["avg_queue"],
            episode_summary["max_queue"],
            episode_summary["throughput"]
        ])

        results.append(episode_summary)
        print(episode_summary)

    return results
    csv_file.close()

# -----------------------------
# VISUALIZATION
# -----------------------------
def plot_avg_delay(results):
    episodes = [r["episode"] for r in results]
    avg_delays = [r["avg_delay"] for r in results]

    plt.figure(figsize=(8,5))
    plt.plot(episodes, avg_delays, marker='o', color='blue')
    plt.title("Q-Learning: Average Delay vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay (s)")
    plt.grid(True)
    plt.show()

def plot_avg_queue(results):
    episodes = [r["episode"] for r in results]
    avg_queues = [r["avg_queue"] for r in results]

    plt.figure(figsize=(8,5))
    plt.plot(episodes, avg_queues, marker='o', color='green')
    plt.title("Q-Learning: Average Queue Length vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length (vehicles)")
    plt.grid(True)
    plt.show()

def plot_throughput(results):
    episodes = [r["episode"] for r in results]
    throughputs = [r["throughput"] for r in results]

    plt.figure(figsize=(8,5))
    plt.plot(episodes, throughputs, marker='o', color='red')
    plt.title("Q-Learning: Throughput vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Throughput (vehicles)")
    plt.grid(True)
    plt.show()

def plot_max_queue(results):
    episodes = [r["episode"] for r in results]
    max_queues = [r["max_queue"] for r in results]

    plt.figure(figsize=(8,5))
    plt.plot(episodes, max_queues, marker='o', color='purple')
    plt.title("Q-Learning: Max Queue Length vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Max Queue Length (vehicles)")
    plt.grid(True)
    plt.show()

# -----------------------------
# EXECUTE
# -----------------------------
if __name__ == "__main__":
    final_results = run_qlearning()
    print("\nFINAL RESULTS:", final_results)
    plot_avg_delay(final_results)
