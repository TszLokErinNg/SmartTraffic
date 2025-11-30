#!/usr/bin/env python3
"""
dqn_4dir_intersection.py

DQN agent for a 4-direction intersection (3 lanes per direction; leftmost lane is left-turn).
Computes average delay, queue length (max & avg), and throughput (discharge rate).
Requires SUMO + TraCI, TensorFlow/Keras, numpy.

Before running:
 - Set SUMO_HOME in your environment.
 - Ensure your sumo network has lane-area detectors with IDs listed in LANE_DETECTORS below
   or change them to match your network.
"""

import os
import sys
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import csv

# ---------------------------
#  SUMO (TraCI) setup
# ---------------------------
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME' pointing to your SUMO installation")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import traci  # now importable

# Command used to start SUMO; can be 'sumo' or 'sumo-gui'
SUMO_BINARY = "sumo"  # or "sumo-gui" for debug visualization
SUMO_CONFIG = "test2.sumocfg"  # change to your sumo config
SUMO_CMD = [SUMO_BINARY, "-c", SUMO_CONFIG, "--step-length", "0.5"]

# ---------------------------
#  Environment specifics
# ---------------------------

# Traffic light id (change to match your network)
TLS_ID = "TL1"

# Lane detectors: 12 lane-area detectors (N,E,S,W each has lane 0:left,1:mid,2:right)
# Change to match detector IDs used in your SUMO network
LANE_DETECTORS = [
    "det_N_0", "det_N_1", "det_N_2",
    "det_E_0", "det_E_1", "det_E_2",
    "det_S_0", "det_S_1", "det_S_2",
    "det_W_0", "det_W_1", "det_W_2"
]


NUM_LANES = len(LANE_DETECTORS)  # 12

# ---------------------------
#  DQN hyperparameters
# ---------------------------
STATE_SIZE = NUM_LANES + 1  # queue counts per lane + current_phase index
ACTION_SIZE = 2             # 0=keep phase, 1=switch phase
GAMMA = 0.95
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000   # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30000
MIN_GREEN_STEPS = 10        # minimum number of sim steps to keep a phase
MAX_TRAINING_STEPS = 2000  # total sim steps

# ---------------------------
#  Utilities: Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

# ---------------------------
#  Build Q-network & target network
# ---------------------------
def build_q_network(input_shape, action_size):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# ---------------------------
#  State, reward, metric functions
# ---------------------------

def read_queue_lengths(detectors):
    """Return list of halting vehicle counts for each lane detector (ints)."""
    q = []
    for det in detectors:
        try:
            # getLastStepHaltingNumber gives vehicles that are halting (i.e., speed < 0.1)
            q_val = traci.lanearea.getLastStepHaltingNumber(det)
        except Exception:
            # If detector missing or error, fallback to 0
            q_val = 0
        q.append(int(q_val))
    return q

def get_current_phase(tls_id):
    try:
        return traci.trafficlight.getPhase(tls_id)
    except Exception:
        return 0

def compute_average_delay():
    """Average accumulated waiting time across vehicles currently in the simulation."""
    vehs = traci.vehicle.getIDList()
    if not vehs:
        return 0.0
    total_wait = 0.0
    for v in vehs:
        try:
            total_wait += traci.vehicle.getAccumulatedWaitingTime(v)
        except Exception:
            # if vehicle not found or error, skip
            continue
    return float(total_wait) / len(vehs)

# Throughput: we'll use the number of vehicles that arrived at their destination this step
def get_arrived_count():
    try:
        arrived = traci.simulation.getArrivedIDList()
        return len(arrived)
    except Exception:
        return 0

# Reward function: combine avg delay, max queue, and throughput
def compute_reward(max_queue, avg_queue, avg_delay, throughput):
    # The coefficients can be tuned depending on priority.
    # Penalize delay & queues, reward throughput (discharge)
    return - (2.0 * avg_delay + 1.0 * avg_queue + 1.5 * max_queue) + (3.0 * throughput)

# Build state vector: [q0,q1,...,q11, current_phase_normalized]
def build_state(queue_list, current_phase, num_phases=8):
    # Normalize phase to [0,1] by dividing by num_phases-1 (avoid zero div)
    phase_norm = current_phase / max(1, (num_phases - 1))
    state = np.array(queue_list + [phase_norm], dtype=np.float32)
    return state

# ---------------------------
#  Action application
# ---------------------------
def apply_action(action, tls_id, current_step, last_switch_step, min_green=MIN_GREEN_STEPS):
    """
    action: 0 = keep current phase, 1 = request switch to next phase
    Returns updated last_switch_step
    """
    if action == 0:
        return last_switch_step

    # action == 1 -> attempt to switch
    # Check if min green time has passed
    if (current_step - last_switch_step) < min_green:
        return last_switch_step

    # perform phase switch
    try:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        curr_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (curr_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        last_switch_step = current_step
    except Exception:
        # if trafficlight id or phases unavailable, ignore
        pass
    return last_switch_step

# ---------------------------
#  Main training routine
# ---------------------------
def train():
    # Start SUMO
    print("Starting SUMO...")
    traci.start(SUMO_CMD)
    print("SUMO started.")

    # build networks
    q_net = build_q_network(STATE_SIZE, ACTION_SIZE)
    target_net = build_q_network(STATE_SIZE, ACTION_SIZE)
    target_net.set_weights(q_net.get_weights())

    replay = ReplayBuffer(REPLAY_MEMORY_SIZE)

    # metrics recording
    steps = []
    avg_delay_hist = []
    avg_queue_hist = []
    max_queue_hist = []
    throughput_hist = []
    cumulative_reward_hist = []

    # Prepare CSV file
    csv_file = open("dqn_metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "avg_delay", "avg_queue", "max_queue", "throughput", "cumulative_reward"])

    # epsilon schedule
    eps = EPS_START
    eps_decay = (EPS_START - EPS_END) / EPS_DECAY_STEPS

    last_switch_step = -MIN_GREEN_STEPS
    cumulative_reward = 0.0

    # Pre-populate replay with random actions for MIN_REPLAY_SIZE steps (exploration)
    print("Warming up replay buffer with random actions...")
    sim_step = 0
    while len(replay) < MIN_REPLAY_SIZE and sim_step < MIN_REPLAY_SIZE * 2:
        # read state
        q = read_queue_lengths(LANE_DETECTORS)
        curr_phase = get_current_phase(TLS_ID)
        state = build_state(q, curr_phase, num_phases=8)

        # random action
        action = random.choice([0, 1])
        last_switch_step = apply_action(action, TLS_ID, sim_step, last_switch_step)

        traci.simulationStep()

        q2 = read_queue_lengths(LANE_DETECTORS)
        curr_phase2 = get_current_phase(TLS_ID)
        next_state = build_state(q2, curr_phase2, num_phases=8)
        avg_delay = compute_average_delay()
        max_q = max(q2) if q2 else 0
        avg_q = (sum(q2) / len(q2)) if q2 else 0
        throughput = get_arrived_count()
        reward = compute_reward(max_q, avg_q, avg_delay, throughput)
        done = False  # we use continuous episodes; you could define boundaries

        replay.add(state, action, reward, next_state, done)

        sim_step += 1

    print(f"Replay buffer filled to {len(replay)} entries. Starting training loop...")

    # Main training loop
    for sim_step in range(MIN_REPLAY_SIZE, MAX_TRAINING_STEPS):
        # Read current sensors
        q = read_queue_lengths(LANE_DETECTORS)
        curr_phase = get_current_phase(TLS_ID)
        state = build_state(q, curr_phase, num_phases=8)

        # epsilon-greedy selection
        if random.random() < eps:
            action = random.choice([0, 1])
        else:
            q_vals = q_net.predict(state.reshape(1, -1), verbose=0)[0]
            action = int(np.argmax(q_vals))

        # apply action respecting min green time
        last_switch_step = apply_action(action, TLS_ID, sim_step, last_switch_step)

        # advance SUMO one step
        traci.simulationStep()

        # read next state + metrics
        q2 = read_queue_lengths(LANE_DETECTORS)
        curr_phase2 = get_current_phase(TLS_ID)
        next_state = build_state(q2, curr_phase2, num_phases=8)

        avg_delay = compute_average_delay()
        max_q = max(q2) if q2 else 0
        avg_q = (sum(q2) / len(q2)) if q2 else 0
        throughput = get_arrived_count()

        reward = compute_reward(max_q, avg_q, avg_delay, throughput)
        cumulative_reward += reward

        # store transition into replay
        replay.add(state, action, reward, next_state, False)

        # Sample mini-batch and learn
        if len(replay) >= BATCH_SIZE:
            s_batch, a_batch, r_batch, ns_batch, d_batch = replay.sample(BATCH_SIZE)

            # predict Q(s,a) and Q(ns,*)
            q_s = q_net.predict(s_batch, verbose=0)
            q_ns = q_net.predict(ns_batch, verbose=0)
            q_ns_target = target_net.predict(ns_batch, verbose=0)

            # compute target Q values (Double DQN style)
            for i in range(BATCH_SIZE):
                if d_batch[i]:
                    q_s[i, a_batch[i]] = r_batch[i]
                else:
                    # Double DQN target: use q_net to select best action, target_net to evaluate
                    best_next_action = np.argmax(q_ns[i])
                    q_s[i, a_batch[i]] = r_batch[i] + GAMMA * q_ns_target[i, best_next_action]

            # train q_net on updated q_s targets
            q_net.fit(s_batch, q_s, epochs=1, verbose=0)

        # periodically update the target network
        if sim_step % TARGET_UPDATE_FREQ == 0 and sim_step > 0:
            target_net.set_weights(q_net.get_weights())

        # decay epsilon
        if eps > EPS_END:
            eps -= eps_decay
            eps = max(eps, EPS_END)

        # record metrics
        steps.append(sim_step)
        avg_delay_hist.append(avg_delay)
        avg_queue_hist.append(avg_q)
        max_queue_hist.append(max_q)
        throughput_hist.append(throughput)
        cumulative_reward_hist.append(cumulative_reward)
        # Write to CSV
        csv_writer.writerow([eps, avg_delay, avg_q, max_q, throughput, cumulative_reward])

        # Print status periodically
        if sim_step % 100 == 0:
            print(f"Step {sim_step:06d} | eps {eps:.3f} | reward {reward:.2f} | cum_reward {cumulative_reward:.1f} | avg_delay {avg_delay:.2f} | avg_q {avg_q:.2f} | max_q {max_q} | throughput {throughput}")

    # End of training
    print("Training finished. Closing SUMO.")
    traci.close()
    csv_file.close()

    # Save model
    q_net.save("dqn_tls_model.h5")
    print("Model saved to dqn_tls_model.h5")

    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cumulative_reward_hist, label="Cumulative Reward")
    plt.xlabel("Simulation step")
    plt.ylabel("Cumulative reward")
    plt.legend(); plt.grid(True); plt.title("Cumulative Reward")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, avg_delay_hist, label="Average Delay (s)")
    plt.plot(steps, avg_queue_hist, label="Average Queue Length")
    plt.plot(steps, max_queue_hist, label="Max Queue Length")
    plt.xlabel("Simulation step"); plt.ylabel("Metric value")
    plt.legend(); plt.grid(True); plt.title("Delay & Queue Metrics")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, throughput_hist, label="Throughput (arrivals per step)")
    plt.xlabel("Simulation step"); plt.ylabel("Throughput")
    plt.legend(); plt.grid(True); plt.title("Throughput over Time")
    plt.show()


if __name__ == "__main__":
    train()
