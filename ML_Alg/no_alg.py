import traci
import csv

# SUMO command
SUMO_CMD = [
    "sumo", 
    "-c", "test2.sumocfg", 
    "--no-step-log", "true"
]

# Lanes to read
LANES = ["-E10_0", "-E10_1", "-E10_2",
        "-E11_0", "-E11_1", "-E11_2",
        "E8_0", "E8_1", "E8_2",
        "-E9_0", "-E9_1", "-E9_2"]

EPISODES = 1
STEPS_PER_EPISODE = 2000

# Open CSV file once
with open("traffic_data_traci.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "episode", "step", "lane", "queue", "speed", "waiting", 
    ])

    for episode in range(1, EPISODES + 1):
        print(f"Starting Episode {episode}")
        
        # Start SUMO
        traci.start(SUMO_CMD)

        # Episode-level metrics
        total_waiting_time = 0
        total_vehicles = 0
        total_queue = 0
        max_queue = 0
        throughput = 0  # Number of vehicles that left the network

        for step in range(STEPS_PER_EPISODE):
            traci.simulationStep()

            for lane in LANES:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                
                # Vehicles in this lane
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in veh_ids)
                
                # Update episode-level metrics
                total_waiting_time += waiting_time
                total_vehicles += len(veh_ids)
                total_queue += queue
                if queue > max_queue:
                    max_queue = queue
            
            # Throughput: vehicles that have left the network so far
            throughput = traci.simulation.getDepartedNumber() - traci.simulation.getArrivedNumber()

            # Compute means
            mean_delay = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
            mean_queue = total_queue / (step + 1) / len(LANES)  # average per lane per step

            # Print per step
            #print(f"Episode {episode} | Step {step} | queue={queue}, speed={speed:.2f}, waiting={waiting_time:.2f} | mean_delay={mean_delay:.2f}, mean_queue={mean_queue:.2f}, max_queue={max_queue}, throughput={throughput}")

            # Write to CSV
            writer.writerow([
                episode, step, lane, queue, speed, waiting_time,
            ])  

        print(f"Ending Episode {episode} | Total Throughput: {throughput} | Max Queue: {max_queue} | Mean Delay: {mean_delay:.2f} | Mean Queue: {mean_queue:.2f}")
        
        traci.close()
        print(f"Episode {episode} finished\n")
