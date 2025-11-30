#!/usr/bin/env python3

"""
multi_intersection_dqn.py

Four independent DQN agents (one per intersection) controlling 4 traffic lights.
Structured to mimic the behavior and outputs of the original multi_intersection_a2c.py
as closely as possible (same state, actions, reward, logging).
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import csv

# ---------------------------
# SUMO setup
# ---------------------------
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME' pointing to your SUMO installation")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
import traci  

SUMO_BINARY = "sumo"
SUMO_CONFIG = "Dundurn King St W.sumocfg"
SUMO_CMD = [SUMO_BINARY, "-c", SUMO_CONFIG, "--step-length", "0.5"]

# ---------------------------
# Environment
# ---------------------------
TLS_IDS = ["TL1", "TL2", "TL3", "TL4"]

LANE_DETECTORS = [
    # TL1
    ["N1_0", "N1_1", "N1_2", "S1_0", "S1_1", "W1_0", "W1_1", "W1_2", "W1_3", "W1_4"],
    # TL2
    ["E2_0", "E2_1", "E2_2", "E2_3", "E2_4", "N2_0", "N2_1", "N2_2", "S2_0", "S2_1", "S2_2"],
    # TL3
    ["E3_0", "E3_1", "E3_2", "E3_3", "E3_4", "S3_0"],
    # TL4
    ["W4_0", "W4_1", "W4_2", "W4_3", "W4_4", "S4_1"]
]

MIN_GREEN_STEPS = 10
EPISODES = 10
STEPS_PER_EPISODE = 1000
GAMMA = 0.99

# DQN hyperparameters
DQN_LR = 1e-3
REPLAY_CAPACITY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_STEPS = 500
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20000
PRINT_EVERY = 500

# ---------------------------
# Utilities 
# ---------------------------
def read_queue_lengths(detectors):
    q = []
    for det in detectors:
        try:
            q_val = traci.lanearea.getLastStepHaltingNumber(det)
        except Exception:
            q_val = 0
        q.append(int(q_val))
    return q

def get_current_phase(tls_id):
    try:
        return traci.trafficlight.getPhase(tls_id)
    except Exception:
        return 0

def get_num_phases(tls_id):
    try:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        return len(program.phases)
    except Exception:
        return 1

def compute_average_delay():
    vehs = traci.vehicle.getIDList()
    if not vehs:
        return 0.0
    total_wait = sum([traci.vehicle.getAccumulatedWaitingTime(v) for v in vehs])
    return total_wait / len(vehs)

def get_arrived_count():
    try:
        return len(traci.simulation.getArrivedIDList())
    except Exception:
        return 0

def compute_reward(max_queue, avg_queue, avg_delay, throughput):
    # Same reward definition as original A2C code
    return - (2.0 * avg_delay + 1.0 * avg_queue + 1.5 * max_queue) + (3.0 * throughput)

def build_state_multi():
    # Same global state as in A2C: queues + normalized phase per intersection
    state = []
    max_lanes = max(len(det_list) for det_list in LANE_DETECTORS)
    for tls_id, det_list in zip(TLS_IDS, LANE_DETECTORS):
        q = read_queue_lengths(det_list)
        q += [0] * (max_lanes - len(q))
        phase = get_current_phase(tls_id)
        num_phases = get_num_phases(tls_id)
        phase_norm = phase / max(1, num_phases - 1)
        state.extend(q + [phase_norm])
    return np.array(state, dtype=np.float32)

def apply_actions_multi(actions, last_switch_steps, current_step):
    new_last_switch = []
    for action, tls_id, last_switch in zip(actions, TLS_IDS, last_switch_steps):
        num_phases = get_num_phases(tls_id)
        if (current_step - last_switch) < MIN_GREEN_STEPS:
            new_last_switch.append(last_switch)
            continue
        next_phase = action % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        new_last_switch.append(current_step)
    return new_last_switch

# ---------------------------
# DQN network and replay
# ---------------------------
def build_q_network(input_shape, num_actions):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    q_values = layers.Dense(num_actions, activation=None)(x)
    model = models.Model(inputs=inputs, outputs=q_values)
    return model

class ReplayBuffer:
    def __init__(self, capacity, state_dim, num_agents):
        self.capacity = capacity
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, actions, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = actions
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.dones[idxs])

# ---------------------------
# Training loop
# ---------------------------
def train_multi_dqn():
    traci.start(SUMO_CMD)
    print("SUMO started for DQN training.")

    max_lanes = max(len(det_list) for det_list in LANE_DETECTORS)
    num_phases_per_tls = [get_num_phases(tls) for tls in TLS_IDS]
    max_phases = max(num_phases_per_tls)
    num_agents = len(TLS_IDS)
    STATE_SIZE = num_agents * (max_lanes + 1)

    # One DQN per TLS: each sees the same global state, outputs Q-values for its own max_phases
    q_nets = []
    target_q_nets = []
    optimizers_list = []

    for _ in TLS_IDS:
        q = build_q_network(STATE_SIZE, max_phases)
        tq = build_q_network(STATE_SIZE, max_phases)
        tq.set_weights(q.get_weights())
        q_nets.append(q)
        target_q_nets.append(tq)
        optimizers_list.append(optimizers.Adam(DQN_LR))

    replay_buffer = ReplayBuffer(REPLAY_CAPACITY, STATE_SIZE, num_agents)

    # Metrics storage & CSV logging (similar to A2C file)
    csv_file = open("multi_dqn_metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Episode",
        "Cumulative Reward",
        "Mean Delay",
        "Mean Avg Queue",
        "Max Queue",
        "Throughput"
    ])

    global_step = 0

    def epsilon_by_step(step):
        frac = min(1.0, step / EPSILON_DECAY_STEPS)
        return EPSILON_START + frac * (EPSILON_END - EPSILON_START)

    for episode in range(EPISODES):
        print(f"\n===== DQN EPISODE {episode + 1} START =====")

        # Reload SUMO network for a fresh episode
        try:
            traci.load(["-c", SUMO_CONFIG])
        except Exception:
            try:
                traci.close()
            except Exception:
                pass
            traci.start(SUMO_CMD)

        last_switch_steps = [-MIN_GREEN_STEPS] * num_agents
        cumulative_reward = 0.0
        episode_delays = []
        episode_avg_queues = []
        episode_max_queues = []
        episode_throughput = 0

        state = build_state_multi()

        for sim_step in range(STEPS_PER_EPISODE):
            epsilon = epsilon_by_step(global_step)

            # Select action per agent
            actions = []
            for i in range(num_agents):
                num_phases = num_phases_per_tls[i]
                if np.random.rand() < epsilon:
                    # exploration
                    act = np.random.randint(num_phases)
                else:
                    q_vals = q_nets[i](state.reshape(1, -1), training=False).numpy().reshape(-1)
                    q_vals_agent = q_vals[:num_phases]
                    act = int(np.argmax(q_vals_agent))
                actions.append(act)

            # Apply actions with same MIN_GREEN_STEPS rule
            last_switch_steps = apply_actions_multi(actions, last_switch_steps, sim_step)

            # Step SUMO
            traci.simulationStep()

            # Build next state
            next_state = build_state_multi()

            # Metrics (same as A2C)
            q_per_tls = [read_queue_lengths(det) for det in LANE_DETECTORS]
            max_qs = [max(q) if q else 0 for q in q_per_tls]
            avg_qs = [np.mean(q) if q else 0.0 for q in q_per_tls]
            avg_delay_val = compute_average_delay()
            avg_delays = [avg_delay_val] * num_agents
            throughput = get_arrived_count()

            reward = sum(compute_reward(mq, aq, ad, throughput)
                         for mq, aq, ad in zip(max_qs, avg_qs, avg_delays))
            cumulative_reward += reward

            episode_delays.append(avg_delay_val)
            episode_avg_queues.append(np.mean(avg_qs))
            episode_max_queues.append(np.max(max_qs))
            episode_throughput += throughput

            done = (sim_step == STEPS_PER_EPISODE - 1)
            replay_buffer.add(state, np.array(actions, dtype=np.int32),
                              reward, next_state, done)

            state = next_state
            global_step += 1

            # DQN updates
            if replay_buffer.size >= BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(BATCH_SIZE)

                for i in range(num_agents):
                    num_phases = num_phases_per_tls[i]
                    with tf.GradientTape() as tape:
                        q_values = q_nets[i](states_b, training=True)  # [B, max_phases]
                        q_values = q_values[:, :num_phases]             # [B, num_phases]

                        # gather chosen actions for this agent
                        acts_i = actions_b[:, i]                        # [B]
                        acts_one_hot = tf.one_hot(acts_i, num_phases, dtype=tf.float32)
                        q_sa = tf.reduce_sum(q_values * acts_one_hot, axis=1)  # [B]

                        # target Q
                        target_q_values = target_q_nets[i](next_states_b, training=False)
                        target_q_values = target_q_values[:, :num_phases]
                        max_next_q = tf.reduce_max(target_q_values, axis=1)    # [B]

                        target = rewards_b + GAMMA * (1.0 - dones_b) * max_next_q
                        loss = tf.reduce_mean(tf.square(target - q_sa))

                    grads = tape.gradient(loss, q_nets[i].trainable_variables)
                    optimizers_list[i].apply_gradients(zip(grads, q_nets[i].trainable_variables))

            # Periodically update target networks
            if global_step % TARGET_UPDATE_STEPS == 0:
                for i in range(num_agents):
                    target_q_nets[i].set_weights(q_nets[i].get_weights())

            if sim_step % PRINT_EVERY == 0:
                print(f"DQN Ep{episode+1} Step{sim_step} | r={reward:.2f} "
                      f"cum_r={cumulative_reward:.1f} avg_delay={avg_delay_val:.2f} eps={epsilon:.3f}")

            if done:
                break

        mean_delay = np.mean(episode_delays) if episode_delays else 0.0
        mean_avg_queue = np.mean(episode_avg_queues) if episode_avg_queues else 0.0
        max_queue_ep = np.max(episode_max_queues) if episode_max_queues else 0

        csv_writer.writerow([
            episode,
            cumulative_reward,
            mean_delay,
            mean_avg_queue,
            max_queue_ep,
            episode_throughput
        ])

        print(f"DQN Episode {episode} finished | Reward={cumulative_reward:.2f} "
              f"| Mean delay={mean_delay:.2f} | Mean avg queue={mean_avg_queue:.2f} "
              f"| Max queue={max_queue_ep} | Throughput={episode_throughput:.2f}")

    traci.close()
    csv_file.close()
    print("DQN training finished.")

if __name__ == "__main__":
    train_multi_dqn()
