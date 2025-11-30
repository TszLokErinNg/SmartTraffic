#!/usr/bin/env python3
"""
multi_intersection_a2c.py

A2C agent for multiple intersections with unequal lanes & traffic phases.
Requires SUMO + TraCI, TensorFlow/Keras, numpy.

- TLS_IDS: list of traffic light IDs
- LANE_DETECTORS: list of lists (one per intersection)
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
#  SUMO setup
# ---------------------------
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME' pointing to your SUMO installation")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
import traci

SUMO_BINARY = "sumo"
SUMO_CONFIG = "Dundurn King St W.sumocfg"
SUMO_CMD = [SUMO_BINARY, "-c", SUMO_CONFIG, "--step-length", "0.5", "--start"]

# ---------------------------
#  Environment
# ---------------------------
TLS_IDS = ["TL1", "TL2", "TL3", "TL4"]

LANE_DETECTORS = [
    # TL1
    ["N1_0","N1_1","N1_2","S1_0","S1_1","W1_0","W1_1","W1_2","W1_3","W1_4"],
    # TL2
    ["E2_0","E2_1","E2_2","E2_3","E2_4","N2_0","N2_1","N2_2","S2_0","S2_1","S2_2"],
    # TL3
    ["E3_0","E3_1","E3_2","E3_3","E3_4","S3_0"],
    # TL4
    ["W4_0","W4_1","W4_2","W4_3","W4_4","S4_1"]
]

MIN_GREEN_STEPS = 10
MAX_TRAINING_STEPS = 2000
GAMMA = 0.99
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
ENTROPY_COEF = 1e-3
VALUE_LOSS_COEF = 0.5
PRINT_EVERY = 100
PRINT_DELAY = 0.0
EPISODES = 10
STEPS_PER_EPISODE = 1000


# ---------------------------
#  Utilities
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
    return - (2.0*avg_delay + 1.0*avg_queue + 1.5*max_queue) + (3.0*throughput)

def build_state_multi():
    state = []
    max_lanes = max(len(det_list) for det_list in LANE_DETECTORS)
    for tls_id, det_list in zip(TLS_IDS, LANE_DETECTORS):
        q = read_queue_lengths(det_list)
        # Pad queues to max_lanes
        q += [0]*(max_lanes - len(q))
        phase = get_current_phase(tls_id)
        num_phases = get_num_phases(tls_id)
        phase_norm = phase / max(1, num_phases-1)
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
#  Actor & Critic
# ---------------------------
def build_actor(input_shape, max_phases_per_intersection):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    logits = layers.Dense(len(TLS_IDS)*max_phases_per_intersection, activation=None)(x)
    probs = layers.Softmax()(logits)
    model = models.Model(inputs=inputs, outputs=[logits, probs])
    return model

def build_critic(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    value = layers.Dense(1, activation=None)(x)
    model = models.Model(inputs=inputs, outputs=value)
    return model

# ---------------------------
#  Training loop
# ---------------------------
def train_multi():
    traci.start(SUMO_CMD)
    print("SUMO started.")

    max_lanes = max(len(det_list) for det_list in LANE_DETECTORS)
    max_phases = max([get_num_phases(tls) for tls in TLS_IDS])
    STATE_SIZE = len(TLS_IDS)*(max_lanes + 1)

    actor = build_actor(STATE_SIZE, max_phases)
    critic = build_critic(STATE_SIZE)
    actor_optimizer = optimizers.Adam(ACTOR_LR)
    critic_optimizer = optimizers.Adam(CRITIC_LR)


    last_switch_steps = [-MIN_GREEN_STEPS]*len(TLS_IDS)
    cumulative_reward = 0.0

    # CSV logging
    csv_file = open("multi_a2c_metrics.csv","w",newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Episode",
        "Cumulative Reward",
        "Mean Delay",
        "Mean Avg Queue",
        "Max Queue",
        "Throughput"
    ])

    # --- Precompute / cache static values once ---
    max_lanes = max(len(det_list) for det_list in LANE_DETECTORS)
    num_phases_per_tls = [get_num_phases(tls) for tls in TLS_IDS]
    max_phases = max(num_phases_per_tls)

    # If you started SUMO earlier, you can reload per episode with traci.load (faster than close/start).
    # Ensure SUMO is running at least once before calling traci.load.
    # traci.start(SUMO_CMD)  # already done earlier in your code

    for episode in range(EPISODES):
        print(f"\n===== EPISODE {episode+1} START =====")
        # Reset episode-level metrics
        cumulative_reward = 0.0
        episode_delays = []
        episode_avg_queues = []
        episode_max_queues = []
        episode_throughput = 0

        # Reload SUMO network for a fresh episode (faster than close/start).
        # If traci is not connected, fall back to start/close.
        try:
            traci.load(["-c", SUMO_CONFIG])
        except Exception:
            # fallback if traci wasn't previously started or load not allowed
            try:
                traci.close()
            except Exception:
                pass
            traci.start(SUMO_CMD)

        # Reset last switch times
        last_switch_steps = [-MIN_GREEN_STEPS] * len(TLS_IDS)

        # --- STEP LOOP ---
        for sim_step in range(STEPS_PER_EPISODE):
            # Build state once
            state = build_state_multi()

            # Actor inference (single batched call) -> probs shaped [num_tls, max_phases]
            logits, probs_tf = actor(state.reshape(1, -1), training=False)
            probs = probs_tf.numpy().reshape(len(TLS_IDS), max_phases)

            # Sample actions using cached num_phases_per_tls
            actions = []
            for i, num_phases in enumerate(num_phases_per_tls):
                p = probs[i, :num_phases]
                # numeric safety: if sum is 0 (rare), use uniform
                s = np.sum(p)
                if s <= 0.0 or not np.isfinite(s):
                    p = np.ones(num_phases) / float(num_phases)
                else:
                    p = p / s
                actions.append(np.random.choice(num_phases, p=p))

            # Apply actions with MIN_GREEN_STEPS enforcement
            last_switch_steps = apply_actions_multi(actions, last_switch_steps, sim_step)

            # Advance SUMO one step
            traci.simulationStep()

            # Read detectors/queues once, reuse for metrics
            q_per_tls = [read_queue_lengths(det) for det in LANE_DETECTORS]  # list of lists
            max_qs = [max(q) if q else 0 for q in q_per_tls]
            avg_qs = [np.mean(q) if q else 0.0 for q in q_per_tls]

            # compute average delay once
            avg_delay_val = compute_average_delay()  # scalar
            avg_delays = [avg_delay_val] * len(TLS_IDS)

            # throughput: number of newly arrived vehicles this step
            throughput = get_arrived_count()

            # Reward aggregated over intersections (your original formula)
            reward = sum(compute_reward(mq, aq, ad, throughput)
                        for mq, aq, ad in zip(max_qs, avg_qs, avg_delays))
            cumulative_reward += reward

            # accumulate episode stats
            episode_delays.append(avg_delay_val)
            episode_avg_queues.append(np.mean(avg_qs))
            episode_max_queues.append(np.max(max_qs))
            episode_throughput += throughput

            # --- A2C training (same math as you had) ---
            # Convert to tensors
            s_tf = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            ns_tf = tf.convert_to_tensor(build_state_multi().reshape(1, -1), dtype=tf.float32)  # next state
            actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
            r_tf = tf.convert_to_tensor([reward], dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                v_s = tf.squeeze(critic(s_tf, training=True))
                v_ns = tf.squeeze(critic(ns_tf, training=True))

                # Advantage and expand per TLS
                advantage = r_tf + GAMMA * v_ns - v_s
                advantage_rep = tf.repeat(advantage, repeats=len(TLS_IDS))

                # Actor forward (training=True)
                logits_train, _ = actor(s_tf, training=True)
                logits_reshaped = tf.reshape(logits_train, (len(TLS_IDS), -1))
                probs_train = tf.nn.softmax(logits_reshaped, axis=1)
                log_probs = tf.math.log(tf.clip_by_value(probs_train, 1e-8, 1.0))

                # Gather log probs of chosen actions
                indices = tf.stack([tf.range(len(TLS_IDS)), actions_tf], axis=1)
                logp_taken = tf.gather_nd(log_probs, indices)

                policy_loss = -tf.reduce_mean(logp_taken * advantage_rep)

                entropy = -tf.reduce_mean(tf.reduce_sum(probs_train * log_probs, axis=1))
                actor_loss = policy_loss - ENTROPY_COEF * entropy

                critic_loss = VALUE_LOSS_COEF * tf.reduce_mean(tf.square(advantage_rep))

                total_loss = actor_loss + critic_loss

            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
            del tape

            # Optional per-step logging (keep minimal to avoid I/O overhead)
            if sim_step % PRINT_EVERY == 0:
                print(f"Ep{episode+1} Step{sim_step} | r={reward:.2f} cum_r={cumulative_reward:.1f} avg_delay={avg_delay_val:.2f}")

        # --- END OF EPISODE: compute aggregated metrics and write CSV ---
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
        
        print(f"Episode {episode} finished | Reward={cumulative_reward:.2f} | Mean delay={mean_delay:.2f} | Mean avg queue={mean_avg_queue:.2f} | Max queue={max_queue_ep} | Throughput={episode_throughput:.2f}")

    # After all episodes
    traci.close()
    csv_file.close()
    print("Training finished.")


        
    

if __name__ == "__main__":
    train_multi()
