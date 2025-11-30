import traci
import csv
import os

# SUMO command
SUMO_CMD = [
    "sumo", 
    "-c", "Dundurn King St W.sumocfg", 
    "--no-step-log", "true",
    "--gui-settings-file", "my_gui_settings.xml"
]

# Lanes to read
LANES = [
    "E0.116_0", "E0.116_1", "E0.116_2", "E0.116_3", "E0.116_4", # Eastbound Dundurn St
    "-E1.6.165_0", "-E1.6.165_1", "-E1.6.165_2", # Westbound Dundurn St
"E12.10_0", "E12.10_1" # Southbound King St W
]

EPISODES = 1
STEPS_PER_EPISODE = 2000

# Open CSV file once
with open("traffic_data_traci.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "episode", "step", "lane", "queue", "speed", "waiting", 
        #"mean_delay", "mean_queue", "max_queue", "throughput"
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
            #print(f"Episode {episode} | Step {step} | lane={lane}, queue={queue}, speed={speed:.2f}, waiting={waiting_time:.2f}")

        
        print(f"Episode {episode} Summary: mean_delay={mean_delay:.2f}, mean_queue={mean_queue:.2f}, max_queue={max_queue}, throughput={throughput}") 


        
        
        traci.close()
        print(f"Episode {episode} finished\n")
