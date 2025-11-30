#!/usr/bin/env python3
"""
a2c_4dir_intersection.py

A2C agent for a 4-direction intersection (3 lanes per direction; leftmost = left-turn,
middle = straight, rightmost = right-turn). One lane-area detector per lane.
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
import time
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
TLS_ID = "TL1"

LANE_DETECTORS = [
    "det_N_0", "det_N_1", "det_N_2",
    "det_E_0", "det_E_1", "det_E_2",
    "det_S_0", "det_S_1", "det_S_2",
    "det_W_0", "det_W_1", "det_W_2"
]

NUM_LANES = len(LANE_DETECTORS)  # 12

# ---------------------------
#  A2C hyperparameters
# ---------------------------
STATE_SIZE = NUM_LANES + 1  # queue counts per lane + current_phase normalized
ACTION_SIZE = 2             # 0 = keep phase, 1 = switch to next phase

GAMMA = 0.99
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
ENTROPY_COEF = 1e-3
VALUE_LOSS_COEF = 0.5
MAX_TRAINING_STEPS = 3600 # total sim steps
MIN_GREEN_STEPS = 10 # minimum number of sim steps to keep a phase

PRINT_EVERY = 100 
PRINT_DELAY = 0.2  # seconds to delay after each status print (set to 0 to disable)

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

def compute_average_delay():
    vehs = traci.vehicle.getIDList()
    if not vehs:
        return 0.0
    total_wait = 0.0
    for v in vehs:
        try:
            total_wait += traci.vehicle.getAccumulatedWaitingTime(v)
        except Exception:
            continue
    return float(total_wait) / len(vehs)

def get_arrived_count():
    try:
        arrived = traci.simulation.getArrivedIDList()
        return len(arrived)
    except Exception:
        return 0

def compute_reward(max_queue, avg_queue, avg_delay, throughput):
    # Similar to your DQN reward: penalize delay & queues, reward throughput
    return - (2.0 * avg_delay + 1.0 * avg_queue + 1.5 * max_queue) + (3.0 * throughput)

def build_state(queue_list, current_phase, num_phases=8):
    phase_norm = current_phase / max(1, (num_phases - 1))
    state = np.array(queue_list + [phase_norm], dtype=np.float32)
    return state

def apply_action(action, tls_id, current_step, last_switch_step, min_green=MIN_GREEN_STEPS):
    """
    action: 0 = keep current phase, 1 = request switch to next phase
    Returns updated last_switch_step
    """
    if action == 0:
        return last_switch_step

    if (current_step - last_switch_step) < min_green:
        return last_switch_step

    try:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        curr_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (curr_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        last_switch_step = current_step
    except Exception:
        pass
    return last_switch_step

# ---------------------------
#  Build Actor & Critic models
# ---------------------------
def build_actor(input_shape, action_size):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    logits = layers.Dense(action_size, activation=None)(x)
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
#  Main training routine
# ---------------------------
def train():
    # Start SUMO
    print("Starting SUMO...")
    traci.start(SUMO_CMD)
    print("SUMO started.")

    actor = build_actor(STATE_SIZE, ACTION_SIZE)
    critic = build_critic(STATE_SIZE)

    actor_optimizer = optimizers.Adam(learning_rate=ACTOR_LR)
    critic_optimizer = optimizers.Adam(learning_rate=CRITIC_LR)

    # Metrics
    steps = []
    avg_delay_hist = []
    avg_queue_hist = []
    max_queue_hist = []
    throughput_hist = []
    cumulative_reward_hist = []

    cumulative_reward = 0.0
    last_switch_step = -MIN_GREEN_STEPS

    # Prepare CSV file
    csv_file = open("a2c_metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "avg_delay", "avg_queue", "max_queue", "throughput", "cumulative_reward"])

    # Training loop
    for sim_step in range(0, MAX_TRAINING_STEPS):
        # Read current state
        q = read_queue_lengths(LANE_DETECTORS)
        curr_phase = get_current_phase(TLS_ID)
        state = build_state(q, curr_phase, num_phases=8)

        # Actor: get action probabilities
        logits, probs = actor(state.reshape(1, -1), training=False)
        probs = probs.numpy().flatten()
        # Sample from policy (stochastic policy)
        action = np.random.choice(ACTION_SIZE, p=probs)

        # Apply action (respect min green)
        last_switch_step = apply_action(action, TLS_ID, sim_step, last_switch_step)

        # Step SUMO
        traci.simulationStep()

        # Read next state and compute reward
        q2 = read_queue_lengths(LANE_DETECTORS)
        curr_phase2 = get_current_phase(TLS_ID)
        next_state = build_state(q2, curr_phase2, num_phases=8)

        avg_delay = compute_average_delay()
        max_q = max(q2) if q2 else 0
        avg_q = (sum(q2) / len(q2)) if q2 else 0
        throughput = get_arrived_count()
        reward = compute_reward(max_q, avg_q, avg_delay, throughput)
        cumulative_reward += reward

        # Convert to tensors for training
        s_tf = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        ns_tf = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)
        a_tf = tf.convert_to_tensor([action], dtype=tf.int32)
        r_tf = tf.convert_to_tensor([reward], dtype=tf.float32)

        # Compute values and advantage: A = r + gamma * V(ns) - V(s)
        with tf.GradientTape(persistent=True) as tape:
            v_s = tf.squeeze(critic(s_tf, training=True), axis=1)       # shape (1,)
            v_ns = tf.squeeze(critic(ns_tf, training=True), axis=1)     # shape (1,)

            # Advantage
            advantage = r_tf + GAMMA * v_ns - v_s                       # shape (1,)

            # Actor loss: -log pi(a|s) * advantage - entropy * coef
            logits, probs = actor(s_tf, training=True)
            probs = tf.clip_by_value(probs, 1e-8, 1.0)
            log_probs = tf.math.log(probs)
            # gather log prob of taken action
            idx = tf.stack([tf.range(tf.shape(a_tf)[0]), a_tf], axis=1)
            logp_taken = tf.gather_nd(log_probs, idx)
            policy_loss = -tf.reduce_mean(logp_taken * tf.stop_gradient(advantage))

            # entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1))
            actor_loss = policy_loss - ENTROPY_COEF * entropy

            # Critic loss: MSE of advantage
            critic_loss = VALUE_LOSS_COEF * tf.reduce_mean(tf.square(advantage))

            total_loss = actor_loss + critic_loss

        # Compute gradients and apply
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
        del tape

        # Record metrics
        steps.append(sim_step)
        avg_delay_hist.append(avg_delay)
        avg_queue_hist.append(avg_q)
        max_queue_hist.append(max_q)
        throughput_hist.append(throughput)
        cumulative_reward_hist.append(cumulative_reward)

        # Write to CSV
        csv_writer.writerow([sim_step, avg_delay, avg_q, max_q, throughput, cumulative_reward])

        # Periodic printing (mirrors your DQN print but without eps)
        if sim_step % PRINT_EVERY == 0:
            # Keep same print format but omit eps (A2C is policy-based)
            print(f"Step {sim_step:06d} | reward {reward:.2f} | cum_reward {cumulative_reward:.1f} | avg_delay {avg_delay:.2f} | avg_q {avg_q:.2f} | max_q {max_q} | throughput {throughput}")
            if PRINT_DELAY > 0:
                time.sleep(PRINT_DELAY)

    # End of training
    print("Training finished. Closing SUMO.")
    traci.close()
    csv_file.close()

    # Save models
    actor.save("a2c_actor_model.h5")
    critic.save("a2c_critic_model.h5")
    print("Saved actor -> a2c_actor_model.h5 and critic -> a2c_critic_model.h5")

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
