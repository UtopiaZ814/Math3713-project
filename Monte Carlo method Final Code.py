import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

df = pd.read_csv('PNS.csv') # Read file from piling drive noise SEL and frequencies
frequencies = df['Frequency (Hz)'].to_list()
source_sel_values = df['SEL (dB re 1 µPa)'].to_list()
min_r = 1  # Min range
max_r = 10000  # Max range
points = 1000  # Number of points
full_range = np.linspace(min_r, max_r, points)  # list of evenly spaced points along transect

# function to include frequency in transmission loss
def cTL(frequency, r):
    freq = frequency / 1000  # convert frequency from Hz to kHz
    alpha = (0.27 * freq**2) / (2.7 + freq**2) + (106 * freq**2) / (17400 + freq**2) + 2.2e-4 * freq #Equation from Zimmer passive acoustic monitoring of cetaceans
    TL = 10 * np.log10(r**2) + alpha * r 
    return TL

# function to calculate logarithmic sum of SEL values
def log10_sum(levels):
    argument = 0
    for level in levels:
        pressure = (10**(level/10))
        argument += pressure
    total = (10 * np.log10(argument))
    return total
#%%
def MMW(freq, group=None):
    # Parameters for each marine mammal group given in Southall paper 2019 page 149
    # f1: lower frequency transition value(kHz)
    # f2: Upper frequency transition value(kHz)
    # a: Low frequency exponent, the rate of decline of weighting function amplitude 
    # b: high frequency exponent, the rate of decline of weighting function amplitude
    # C: Constant, max amplitude of weighting function is equal to 0dB
    if group == "LF": #Low-frequency cetaceans
        f1, f2, a, b, C = 0.20, 19, 1, 2, 0.13
    elif group == "HF":# High-frequency cetaceans
        f1, f2, a, b, C = 8.8, 110, 1.6, 2, 1.20
    elif group == "VHF":# Very high-frequency cetaceans
        f1, f2, a, b, C = 12, 140, 1.8, 2, 1.36
    elif group == "SI":# Sirenians
        f1, f2, a, b, C = 4.3, 25, 1.8, 2, 2.62
    elif group == "PCW":# Phocid Carnivores in water
        f1, f2, a, b, C = 1.9, 30, 1, 2, 0.75
    elif group == "OCW":# Phocid Carnivores in air
        f1, f2, a, b, C = 0.94, 25, 2, 2, 0.64
    elif group == "PCA":# Other marine carnivores in water
        f1, f2, a, b, C = 0.75, 8.3, 2, 2, 1.50
    elif group == "OCA":# Other marine carnivores in air
        f1, f2, a, b, C = 2.0, 20, 1.4, 2, 1.39
    else:
        return np.zeros_like(freq)

    # Weight Equation given in Southall paper 2019 page 146
    weights = C + 10 * np.log10(((freq / f1) ** (2 * a)) / ((1 + (freq / f1) ** 2) ** a * (1 + (freq / f2) ** 2) ** b))
    return weights

group_to_threshold = { #TTS values from Southall
    "None": 200,  # Default value for groups not specified
    "LF": 168,
    "HF": 170,
    "VHF": 140,
    "SI": 175,
    "PCW": 170,
    "OCW": 188,
    "PCA": 123,# not required
    "OCA": 146 # Not required
}

speed_ranges = { #Speed range in m/s
    "None":(1,3)
    "LF": (2, 4),
    "HF": (8, 11),
    "VHF": (3.5, 4.5),
    "SI": (1,2.5), 
    "PCW": (5, 8),
    "OCW": (2, 4), 
    "PCA":(1, 3),# dont need cuz  its in air
    "OCA": (1, 3),# dont need cuz its in air
    
}

#%%

#Monte Carlo simulations
num_simulations = 100
# Simulation parameters
interval = 2  # Time interval between sounds in seconds
total_duration = 3600  # Total simulation duration in seconds (1 hour)
all_results = {}
injured_animals =[]
non_injured_animals = []


for _ in range(num_simulations):
    group = random.choice(["SI"]) # Choosing which marine mammal group to use

    initial_position = random.randint(10, 150) # choosing a random initial position
    speed_range = speed_ranges[group]  # Select speed range based on the group
    speed = round(random.uniform(speed_range[0], speed_range[1]), 2)  # Randomly select speed within the range
    
    injury_threshold = group_to_threshold[group] # Retrieves TTS threshold for the current group
    MMW_weights = MMW(full_range, group=group)
     
    sound_schedule = np.arange(0, total_duration, interval)
    total_sound = 0
    times = []
    total_SL = []
    current_position = initial_position
    exceeded_flag = 0
    flee_probability = 0.7 # chance of animal fleeing at different angle and speed
    new_speeds = []
    
 
    for time in range(total_duration):
        RLs = []

        if time in sound_schedule:
            # Calculate received levels for the current distance and frequencies
            for frequency, sel, MMW_weight in zip(frequencies, source_sel_values, MMW_weights):
                TLr = cTL(frequency, np.linalg.norm(current_position))
                f_received_level = sel - TLr + MMW_weight 
                RLs.append(f_received_level)

            total_RLs = log10_sum(RLs)

            # Accumulate total sound over time/ unlog it
            total_sound += 10 ** (total_RLs / 10)
            # Keep a total in dB and append time/total data
            total_db = 10 * np.log10(total_sound)
            times.append(time)
            total_SL.append(total_db)

            # Check if the total sound exceeds the injury threshold
            if (total_db >= injury_threshold) and (exceeded_flag == 0):
                distance_from_source = np.linalg.norm(current_position)
                #print(f'{group}: Total sound exceeds injury threshold:{injury_threshold:.2f} dB at {time} seconds.')
                #print(f'The animal started at {initial_position:.2f} metres with speed of {speed} m/s and would be injured within {distance_from_source:.2f} metres of the source.')
                injured_animals.append({
                    "group": group,
                    "injury_threshold": injury_threshold,
                    "initial_position": initial_position,
                    "speed": speed,
                    "time_of_injury": time,
                    "distance_from_source": distance_from_source
                })
                exceeded_flag = 1

        # Updating current position based on animal speed and time interval
        if random.uniform(0, 1) < flee_probability:
            speed_changer = random.uniform(0.9, 1.1 )# speed and change between -10% to 10% of the orginal speed
            new_speed = round(speed * speed_changer ,2)
            angle = random.uniform(-math.pi / 3, math.pi / 3 ) # angular change of direction of speed
            x_component = new_speed * interval * math.cos(angle)# x coordinates of animal
            y_component = new_speed * interval * math.sin(angle) # y coordinates of animal
            current_position += np.array([x_component, y_component], dtype=float)
        # Updating current position based on animal speed and time interval
        else:
            current_position += np.array([speed * interval, 0.0], dtype=float)

    # Calculate distance using Pythagorean theorem using x and y coordinates of animal
    distance_from_source = np.linalg.norm(current_position)

    if exceeded_flag == 0:
        non_injured_animals.append({
    "group": group,
    "injury_threshold": injury_threshold,
    "initial_position": initial_position,
    "speed": speed
})
        print(f'{group} The animal with injury threshold {injury_threshold} dB started at {initial_position} metres would not be injured')
        print(f' at a speed of { new_speed} m/s ')

    all_results[(group, injury_threshold, initial_position, speed)] = {"times": times, "total_SL": total_SL}
    
# Plot results for each simulation
for vel, results in all_results.items():
    group, injury_threshold, initial_position, speed = vel
    label_text = f'Group: {group}, Threshold: {injury_threshold}, Initial Position: {initial_position} m, Speed: {speed:} m/s'
    plt.plot(results["times"], results["total_SL"], label=label_text)

plt.axhline(y=injury_threshold, color='red', linestyle='--', label='Threshold Level')
plt.xlabel('Time (seconds)')
plt.ylabel('Total Sound Exposure Level (dB)')
plt.title(f'{group}:Monte Carlo Simulation: Total Sound Exposure Level vs. Time')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.show()

total_sound_levels = []
initial_positions = []
speeds=[]
new_speeds = []
injury_thresholds = []
exceeded_thresholds = 0

for vel, results in all_results.items():
    group, injury_threshold, initial_position, speed = vel
    total_sound_levels.extend(results["total_SL"])
    initial_positions.append(initial_position)
    speeds.append(speed)
    new_speeds.append(new_speed)
    injury_thresholds.append(injury_threshold)

    #number of simulations where the injury threshold is exceeded
    if max(results["total_SL"]) >= injury_threshold:
        exceeded_thresholds += 1

print(f'{group}: {exceeded_thresholds} out of {num_simulations} simulations where animals are hurt')

# Injured animals
if injured_animals:
    injury_distances = [animal['distance_from_source'] for animal in injured_animals]
    injury_times = [animal['time_of_injury'] for animal in injured_animals]
    injury_speeds = [animal['speed'] for animal in injured_animals]
    injury_positions = [animal['initial_position'] for animal in injured_animals]

    print(f"\nStatistics for Injured {group} group Animals:")
    print(f'Mean Distance from Source: {np.mean(injury_distances):.2f} metres')
    print(f'Standard Deviation Distance from source: {np.std(injury_distances):.2f} metres')
    print(f'Minimum Distance from Source: {min(injury_distances):.2f} metres')
    print(f'Maximum Distance from Source: {max(injury_distances):.2f} metres')

    print(f'Mean Time of Injury: {np.mean(injury_times):.2f} seconds')
    print(f'Standard Deviation Time of Injury: {np.std(injury_times):.2f} seconds')
    print(f'Minimum Time of Injury: {min(injury_times):.2f} seconds')
    print(f'Maximum Time of Injury: {max(injury_times):.2f} seconds')
    
    print(f'Mean Speed of Injured Animals: {np.mean(injury_speeds):.2f} m/s')
    print(f'Speed Standard Deviation: {np.std(injury_speeds):.2f} m/s')
    print(f'Minimum: {min(injury_speeds):.2f} m/s')
    print(f'Maximum: {max(injury_speeds):.2f} m/s')
    
    print(f'Mean Initial Position of Injured Animals: {np.mean(injury_positions):.2f} metres')
    print(f'Initial position Standard Deviation: {np.std(injury_positions):.2f} metres')
    print(f'Minimum: {min(injury_positions):.2f} metres')
    print(f'Maximum: {max(injury_positions):.2f} metres')

# Non-injured animals
if non_injured_animals:
    non_injured_distances = [animal['initial_position'] for animal in non_injured_animals]
    non_injured_speeds = [animal['speed'] for animal in non_injured_animals]
    
    print(f"\nStatistics for Non-Injured {group} group Animals:")
    print(f'Mean Initial Position: {np.mean(non_injured_distances):.2f} metres')
    print(f'Standard Deviation Initial Position: {np.std(non_injured_distances):.2f} metres')
    print(f'Minimum: {min(non_injured_distances):.2f} metres')
    print(f'Maximum: {max(non_injured_distances):.2f} metres')
    print(f'Mean Speed: {np.mean(non_injured_speeds):.2f} m/s')
    print(f'Standard Deviation Speed: {np.std(non_injured_speeds):.2f} m/s')
    print(f'Minimum: {min(non_injured_speeds):.2f} m/s')
    print(f'Maximum: {max(non_injured_speeds):.2f} m/s')
else:
    print("\nAll animals were injured in the simulations.")

# Total SEL
print(f'\n{group}: Total Sound Exposure Level:')
print(f'Mean: {np.mean(total_sound_levels):.2f} dB')
print(f'Standard Deviation: {np.std(total_sound_levels):.2f} dB')
print(f'Minimum: {min(total_sound_levels):.2f} dB')
print(f'Maximum: {max(total_sound_levels):.2f} dB')

# Initial Position
print(f'\n{group}: Total Initial Position:')
print(f'Mean: {np.mean(initial_positions):.2f} m')
print(f'Standard Deviation: {np.std(initial_positions):.2f} m')
print(f'Minimum: {min(initial_positions):.2f} m')
print(f'Maximum: {max(initial_positions):.2f} m')

# Speed 
print(f'\n{group}: Total initial speed:')
print(f'Mean: {np.mean(speeds):.2f} m/s')
print(f'Standard Deviation: {np.std(speeds):.2f} m/s')
print(f'Minimum: {min(speeds):.2f} m/s')
print(f'Maximum: {max(speeds):.2f} m/s')


#Scatter plot of time of injury vs distance
if injured_animals:
    # Extracting data from injured_animals
    injury_distances = [animal['distance_from_source'] for animal in injured_animals]
    injury_times = [animal['time_of_injury'] for animal in injured_animals]

    # Creating a scatter plot
    plt.scatter(injury_distances, injury_times, c='red', marker='o', label='Injured Animals')
    plt.xlabel('Distance from Source (metres)')
    plt.ylabel('Time of Injury (seconds)')
    plt.title(f'{group}:Scatter Plot of Time of Injury vs Distance')
    coefficients = np.polyfit(injury_distances, injury_times, 1)
    poly = np.poly1d(coefficients)
    plt.plot(injury_distances, poly(injury_distances), 'b--', label='Best Fit Line')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

# Scatter plot for Initial Speeds vs initial Positions
if injured_animals:
    injury_speeds = [animal['speed'] for animal in injured_animals]
    injury_positions = [animal['initial_position'] for animal in injured_animals]
    plt.scatter(injury_positions, injury_speeds, c='red', marker='o', label='Injured Animals')

if non_injured_animals:
    non_injured_speeds = [animal['speed'] for animal in non_injured_animals]
    non_injured_positions = [animal['initial_position'] for animal in non_injured_animals]
    plt.scatter(non_injured_positions, non_injured_speeds, c='blue', marker='o', label='Non-Injured Animals')
plt.xlabel('Initial Position (metres)')
plt.ylabel('Initial Speed (m/s)')
plt.title(f'Scatter Plot of Initial Speeds vs Initial Positions for {group} mammals')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

