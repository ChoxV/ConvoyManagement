import pandas as pd
import random

locations = ['A', 'B', 'C', 'D', 'E']
modes = ['road', 'rail', 'air']
weather_options = ['sunny', 'rainy', 'foggy']
traffic_levels = ['low', 'medium', 'high']
time_of_day = [0, 6, 12, 18]  

synthetic_data = []

for _ in range(1000): 
    src, dest = random.sample(locations, 2)
    mode = random.choice(modes)
    distance = random.randint(5, 50) 
    tod = random.choice(time_of_day)
    weather = random.choice(weather_options)
    traffic = random.choice(traffic_levels)

    base_speed = {'road': 40, 'rail': 80, 'air': 600}[mode]
    speed_modifier = 1.0

    if weather == 'rainy':
        speed_modifier *= 0.8
    elif weather == 'foggy':
        speed_modifier *= 0.6

    if traffic == 'medium':
        speed_modifier *= 0.9
    elif traffic == 'high':
        speed_modifier *= 0.7

    time = distance / (base_speed * speed_modifier)
    travel_time = round(time * 60, 2)  

    synthetic_data.append({
        'source': src,
        'destination': dest,
        'mode': mode,
        'distance': distance,
        'time_of_day': tod,
        'weather': weather,
        'traffic_level': traffic,
        'travel_time': travel_time
    })

# Save as CSV
df = pd.DataFrame(synthetic_data)
df.to_csv("synthetic_convoy_travel_data.csv", index=False)
print("Saved synthetic data to 'synthetic_convoy_travel_data.csv'")
