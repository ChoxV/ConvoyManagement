import random
import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.preprocessing import LabelEncoder
from collections import deque
from functools import lru_cache
import math

# Load ML model and data
try:
    model = joblib.load('convoy_rf_model.pkl')
    label_encoders = {
        'mode': joblib.load('label_encoder_mode.pkl'),
        'weather': joblib.load('label_encoder_weather.pkl'),
        'traffic_level': joblib.load('label_encoder_traffic_level.pkl')
    }
    synthetic_data = pd.read_csv('synthetic_convoy_travel_data.csv')
except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}. Ensure 'convoy_rf_model.pkl', encoders, and 'synthetic_convoy_travel_data.csv' are in the directory.")
    exit(1)
except Exception as e:
    print(f"Error loading model or data: {e}")
    exit(1)

def generate_random_map(num_nodes):
    """Generate a random graph with ML-predicted times and node coordinates."""
    nodes = [chr(65 + i) for i in range(num_nodes)]
    modes = ['road', 'rail', 'air']
    graph = {node: {} for node in nodes}
    
    weather_options = ['sunny', 'rainy', 'foggy']
    traffic_levels = ['low', 'medium', 'high']
    time_of_day = [0, 6, 12, 18]
    
    for i, node in enumerate(nodes):
        possible_neighbors = [n for n in nodes if n != node]
        num_edges = random.randint(1, min(3, len(possible_neighbors)))
        neighbors = random.sample(possible_neighbors, num_edges)
        for neighbor in neighbors:
            edge_modes = random.sample(modes, random.randint(1, len(modes)))
            mode_times = {}
            for mode in edge_modes:
                distance = random.randint(5, 50)
                tod = random.choice(time_of_day)
                weather = random.choice(weather_options)
                traffic = random.choice(traffic_levels)
                
                try:
                    input_data = pd.DataFrame([{
                        'distance': distance,
                        'time_of_day': tod,
                        'mode': label_encoders['mode'].transform([mode])[0],
                        'weather': label_encoders['weather'].transform([weather])[0],
                        'traffic_level': label_encoders['traffic_level'].transform([traffic])[0]
                    }])
                    travel_time = model.predict(input_data)[0] / 60
                    mode_times[mode] = {
                        'time': max(0.1, travel_time),
                        'weather': weather,
                        'traffic': traffic
                    }
                except ValueError as e:
                    print(f"Warning: ML prediction error for mode {mode}: {e}. Using default time.")
                    mode_times[mode] = {'time': 0.1, 'weather': weather, 'traffic': traffic}
                
            graph[node][neighbor] = mode_times
            graph[neighbor][node] = mode_times
    
    start_node = nodes[0]
    reachable = {start_node}
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    if len(reachable) < num_nodes:
        for node in nodes:
            if node not in reachable:
                neighbor = random.choice(list(reachable))
                mode = random.choice(modes)
                graph[node][neighbor] = {mode: {'time': 0.1, 'weather': 'sunny', 'traffic': 'low'}}
                graph[neighbor][node] = graph[node][neighbor]
                reachable.add(node)
    
    # Generate node coordinates
    node_coords = {}
    margin = 50
    width, height = 600, 500
    
    if num_nodes <= 8:
        radius = min(width, height) / 2.5
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / num_nodes
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_coords[node] = (x, y)
    else:
        grid_size = int(math.ceil(math.sqrt(num_nodes)))
        for i, node in enumerate(nodes):
            row = i // grid_size
            col = i % grid_size
            x = (col - grid_size/2) * width/grid_size
            y = (row - grid_size/2) * height/grid_size
            node_coords[node] = (x, y)
    
    return graph, node_coords, nodes, modes

MAX_TIME = 10

@lru_cache(maxsize=1024)
def calculate_route_time(route_tuple, mode_sequence_tuple):
    """Calculate total time for a route."""
    route = list(route_tuple)
    mode_sequence = list(mode_sequence_tuple)
    graph = generate_random_map.cache.get('graph', {})
    total_time = 0
    for i in range(len(route) - 1):
        if route[i+1] in graph.get(route[i], {}) and mode_sequence[i] in graph[route[i]][route[i+1]]:
            total_time += graph[route[i]][route[i+1]][mode_sequence[i]]['time']
        else:
            return float('inf')
    return total_time

def bfs_route(start, end, graph, modes):
    """Find a valid route using BFS."""
    queue = deque([(start, [start], [])])
    visited = {(start, tuple())}
    while queue:
        current, route, mode_seq = queue.popleft()
        if current == end:
            return route, mode_seq
        for next_node in graph[current]:
            for mode in graph[current][next_node]:
                new_route = route + [next_node]
                new_modes = mode_seq + [mode]
                state = (next_node, tuple(new_modes[-min(len(new_modes), 5):]))
                if state not in visited:
                    visited.add(state)
                    queue.append((next_node, new_route, new_modes))
    return None, None

def generate_random_route(start, end, graph, modes, max_steps=10):
    """Generate a route using random walks, falling back to BFS."""
    steps = 0
    while steps < max_steps:
        route = [start]
        mode_sequence = []
        current = start
        visited = {start}
        step_count = 0
        
        while current != end and step_count < max_steps:
            next_nodes = list(graph[current].keys())
            unvisited = [node for node in next_nodes if node not in visited]
            if not unvisited:
                break
            next_node = random.choice(unvisited)
            available_modes = list(graph[current][next_node].keys())
            mode = random.choice(available_modes)
            route.append(next_node)
            mode_sequence.append(mode)
            visited.add(next_node)
            current = next_node
            step_count += 1
        
        if current == end:
            return route, mode_sequence
        steps += 1
    
    return bfs_route(start, end, graph, modes)

def is_valid_route(route, graph):
    """Check if a route is valid."""
    for i in range(len(route) - 1):
        if route[i+1] not in graph[route[i]]:
            return False
    return True

def abc_cmp(start, end, graph, modes, num_bees=15, max_iterations=15):
    """Artificial Bee Colony algorithm."""
    bees = []
    for _ in range(num_bees):
        route, mode_seq = generate_random_route(start, end, graph, modes)
        if route:
            bees.append((route, mode_seq))
    if not bees:
        route, mode_seq = bfs_route(start, end, graph, modes)
        if route:
            bees.append((route, mode_seq))
        else:
            return None, None, float('inf')

    best_route, best_modes, best_time = None, None, float('inf')

    for _ in range(max_iterations):
        for i in range(len(bees)):
            current_route, current_modes = bees[i]
            current_time = calculate_route_time(tuple(current_route), tuple(current_modes))

            new_route, new_modes = current_route.copy(), current_modes.copy()
            if len(new_route) > 3:
                idx1, idx2 = random.sample(range(1, len(new_route)-1), 2)
                new_route[idx1], idx2 = new_route[idx2], new_route[idx1]
                if not is_valid_route(new_route, graph):
                    new_route, new_modes = generate_random_route(start, end, graph, modes)
                    if not new_route:
                        continue
                else:
                    new_modes = []
                    for j in range(len(new_route) - 1):
                        available = list(graph[new_route[j]][new_route[j+1]].keys())
                        new_modes.append(random.choice(available) if available else current_modes[j % len(current_modes)])
            
            new_time = calculate_route_time(tuple(new_route), tuple(new_modes))
            if new_time < current_time and new_time <= MAX_TIME:
                bees[i] = (new_route, new_modes)
                if new_time < best_time:
                    best_time = new_time
                    best_route, best_modes = new_route, new_modes

        for i in range(len(bees)):
            if random.random() < 0.5:
                current_route, current_modes = bees[i]
                current_time = calculate_route_time(tuple(current_route), tuple(current_modes))
                if current_time < best_time and current_time <= MAX_TIME:
                    best_time = current_time
                    best_route, best_modes = current_route, current_modes

        for i in range(len(bees)):
            if calculate_route_time(tuple(bees[i][0]), tuple(bees[i][1])) > MAX_TIME:
                new_route, new_modes = generate_random_route(start, end, graph, modes)
                if new_route:
                    bees[i] = (new_route, new_modes)

    return best_route, best_modes, best_time

def levy_flight_step():
    """Generate a Lévy flight step for Cuckoo Search."""
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = random.gauss(0, sigma)
    v = random.gauss(0, 1)
    return u / abs(v) ** (1 / beta)

def cuckoo_cmp(start, end, graph, modes, num_nests=15, max_iterations=15, pa=0.25):
    """Cuckoo Search algorithm."""
    nests = []
    for _ in range(num_nests):
        route, mode_seq = generate_random_route(start, end, graph, modes)
        if route:
            nests.append((route, mode_seq))
    if not nests:
        route, mode_seq = bfs_route(start, end, graph, modes)
        if route:
            nests.append((route, mode_seq))
        else:
            return None, None, float('inf')

    best_route, best_modes, best_time = None, None, float('inf')

    for _ in range(max_iterations):
        for i in range(len(nests)):
            current_route, current_modes = nests[i]
            current_time = calculate_route_time(tuple(current_route), tuple(current_modes))
            
            new_route, new_modes = current_route.copy(), current_modes.copy()
            if len(new_route) > 3:
                step = int(abs(levy_flight_step())) % (len(new_route) - 2) + 1
                idx = random.randint(1, len(new_route) - 2)
                if idx + step < len(new_route) - 1:
                    new_route[idx], new_route[idx + step] = new_route[idx + step], new_route[idx]
                    if not is_valid_route(new_route, graph):
                        new_route, new_modes = generate_random_route(start, end, graph, modes)
                        if not new_route:
                            continue
                    else:
                        new_modes = []
                        for j in range(len(new_route) - 1):
                            available = list(graph[new_route[j]][new_route[j+1]].keys())
                            new_modes.append(random.choice(available) if available else current_modes[j % len(current_modes)])
            
            new_time = calculate_route_time(tuple(new_route), tuple(new_modes))
            if new_time <= MAX_TIME and new_time < current_time:
                nests[i] = (new_route, new_modes)
                if new_time < best_time:
                    best_time = new_time
                    best_route, best_modes = new_route, new_modes

        nests.sort(key=lambda x: calculate_route_time(tuple(x[0]), tuple(x[1])))
        num_abandon = int(pa * len(nests))
        for i in range(num_abandon):
            new_route, new_modes = generate_random_route(start, end, graph, modes)
            if new_route:
                nests[i] = (new_route, new_modes)

        for nest_route, nest_modes in nests:
            nest_time = calculate_route_time(tuple(nest_route), tuple(nest_modes))
            if nest_time < best_time and nest_time <= MAX_TIME:
                best_time = nest_time
                best_route, best_modes = nest_route, best_modes

    return best_route, best_modes, best_time

def schedule_convoys(convoy_routes_modes, graph):
    """Schedule convoys with start time adjustments."""
    convoy_schedules = []
    edge_usage = {}
    
    sorted_convoys = []
    for route, route_modes in convoy_routes_modes:
        if route is None:
            continue
        total_time = calculate_route_time(tuple(route), tuple(route_modes))
        if total_time != float('inf'):
            sorted_convoys.append((route, route_modes, total_time))
    sorted_convoys.sort(key=lambda x: x[2])
    
    max_attempts = 10
    start_time_step = 0.5
    
    for route, route_modes, _ in sorted_convoys:
        if route is None:
            continue
        attempts = 0
        current_modes = route_modes.copy()
        scheduled = False
        
        while attempts < max_attempts and not scheduled:
            for start_time in [i * start_time_step for i in range(5)]:
                total_time = calculate_route_time(tuple(route), tuple(current_modes))
                if total_time == float('inf'):
                    break
                
                conflict = False
                route_times = []
                current_time = start_time
                for i in range(len(route) - 1):
                    edge = tuple(sorted([route[i], route[i+1]]))
                    travel_time = graph[route[i]][route[i+1]][current_modes[i]]['time']
                    time_interval = (current_time, current_time + travel_time)
                    route_times.append((edge, time_interval))
                    current_time += travel_time
                
                for edge, (t_start, t_end) in route_times:
                    if edge in edge_usage:
                        for used_start, used_end in edge_usage[edge]:
                            if not (t_end <= used_start or t_start >= used_end):
                                conflict = True
                                break
                    if conflict:
                        break
                
                if not conflict:
                    convoy_schedules.append((route, current_modes, start_time, route[0], route[-1], total_time))
                    for edge, (t_start, t_end) in route_times:
                        if edge not in edge_usage:
                            edge_usage[edge] = []
                        edge_usage[edge].append((t_start, t_end))
                    scheduled = True
                    break
            
            if not scheduled:
                current_modes = [random.choice(list(graph[route[i]][route[i+1]].keys())) for i in range(len(route)-1)]
                attempts += 1
        
        if not scheduled:
            print(f"Warning: Could not schedule convoy {route} without conflict after retries.")

    return convoy_schedules

class ConvoyGUI:
    def __init__(self, root, num_nodes):
        self.root = root
        self.root.title("Convoy Route Planner")
        self.root.configure(bg="#f4f6f8")
        self.num_nodes = num_nodes
        self.pairs = []
        self.convoy_schedules = []
        
        # Generate graph
        self.graph, self.node_coords, self.nodes, self.modes = generate_random_map(num_nodes)
        generate_random_map.cache = {'graph': self.graph}
        
        # GUI Layout
        self.root.geometry("1200x700")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 11, "bold"), padding=8, background="#007bff", foreground="#ffffff")
        self.style.configure("TLabel", font=("Helvetica", 12), background="#ffffff")
        self.style.map("TButton", 
                       background=[('active', '#0056b3'), ('!disabled', '#007bff')],
                       foreground=[('active', '#ffffff')])
        self.style.configure("TCombobox", font=("Helvetica", 10))
        
        # Left Panel: Controls
        self.left_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="flat", highlightthickness=1, highlightbackground="#dee2e6")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20, ipadx=15, ipady=15)
        
        tk.Label(self.left_frame, text="Route Planner", font=("Helvetica", 18, "bold"), bg="#ffffff", fg="#343a40").pack(anchor="w", pady=(0, 20))
        
        ttk.Label(self.left_frame, text="Source Node:", style="TLabel").pack(anchor="w", padx=10)
        self.source_var = tk.StringVar(value=self.nodes[0])
        self.source_menu = ttk.Combobox(self.left_frame, textvariable=self.source_var, values=self.nodes, state="readonly", font=("Helvetica", 10))
        self.source_menu.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(self.left_frame, text="Destination Node:", style="TLabel").pack(anchor="w", padx=10)
        self.dest_var = tk.StringVar(value=self.nodes[1] if len(self.nodes) > 1 else self.nodes[0])
        self.dest_menu = ttk.Combobox(self.left_frame, textvariable=self.dest_var, values=self.nodes, state="readonly", font=("Helvetica", 10))
        self.dest_menu.pack(fill=tk.X, padx=10, pady=5)
        
        self.add_button = ttk.Button(self.left_frame, text="Add Pair", command=self.add_pair, style="TButton")
        self.add_button.pack(fill=tk.X, padx=10, pady=15)
        
        ttk.Label(self.left_frame, text="Selected Pairs:", style="TLabel").pack(anchor="w", padx=10)
        self.pair_listbox = tk.Listbox(self.left_frame, height=10, font=("Helvetica", 10), bg="#f8f9fa", selectbackground="#a3cfbb", relief="flat", bd=1, highlightthickness=1, highlightbackground="#dee2e6")
        self.pair_listbox.pack(fill=tk.X, padx=10, pady=5)
        self.pair_scroll = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.pair_listbox.yview)
        self.pair_listbox.configure(yscrollcommand=self.pair_scroll.set)
        self.pair_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10), pady=(0,5))
        
        self.run_button = ttk.Button(self.left_frame, text="Run Routes", command=self.run_routes, style="TButton")
        self.run_button.pack(fill=tk.X, padx=10, pady=15)
        
        # Center Panel: Graph Canvas
        self.canvas_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="flat", highlightthickness=1, highlightbackground="#dee2e6")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=600, height=500, bg="white", highlightthickness=0)
        self.canvas.pack(padx=15, pady=15)
        
        # Right Panel: Results
        self.right_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="flat", highlightthickness=1, highlightbackground="#dee2e6")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20, ipadx=15, ipady=15)
        
        tk.Label(self.right_frame, text="Route Results", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#343a40").pack(anchor="w", pady=(0, 15))
        self.result_text = tk.Text(self.right_frame, height=22, width=45, font=("Helvetica", 10), bg="#f8f9fa", wrap="word", relief="flat", bd=1, highlightthickness=1, highlightbackground="#dee2e6")
        self.result_text.pack(fill=tk.BOTH, padx=10, pady=5)
        self.result_scroll = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=self.result_scroll.set)
        self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10), pady=(5,0))
        
        self.save_button = ttk.Button(self.right_frame, text="Save Results", command=self.save_results, style="TButton")
        self.save_button.pack(fill=tk.X, padx=10, pady=15)
        
        # Draw initial graph
        self.draw_graph()
    
    def draw_graph(self):
        """Draw the static graph with enhanced visuals."""
        self.canvas.delete("all")
        
        # Gradient background
        for i in range(500):
            shade = int(255 - (i / 500) * 30)
            color = f"#{shade:02x}{shade:02x}{255:02x}"
            self.canvas.create_line(0, i, 600, i, fill=color)
        
        # Scale coordinates
        margin = 60
        max_x = max(abs(x) for x, y in self.node_coords.values()) or 1
        max_y = max(abs(y) for x, y in self.node_coords.values()) or 1
        scale_factor = min((600 - 2*margin) / (2*max_x), (500 - 2*margin) / (2*max_y))
        self.scaled_coords = {node: (x*scale_factor + 300, y*scale_factor + 250) for node, (x, y) in self.node_coords.items()}
        
        mode_colors = {'road': '#007bff', 'rail': '#28a745', 'air': '#dc3545'}
        drawn_edges = set()
        
        # Draw edges
        for start in self.graph:
            for end, mode_times in self.graph[start].items():
                edge = tuple(sorted([start, end]))
                if edge not in drawn_edges:
                    start_x, start_y = self.scaled_coords[start]
                    end_x, end_y = self.scaled_coords[end]
                    drawn_edges.add(edge)
                    
                    offset = 0
                    num_modes = len(mode_times)
                    for mode, info in mode_times.items():
                        dx = end_x - start_x
                        dy = end_y - start_y
                        length = math.sqrt(dx*dx + dy*dy) if dx != 0 or dy != 0 else 1
                        if offset != 0:
                            nx, ny = -dy/length, dx/length
                            curve = 12 * offset / max(1, num_modes-1)
                            mid_x = (start_x + end_x)/2 + nx * curve
                            mid_y = (start_y + end_y)/2 + ny * curve
                            self.canvas.create_line(start_x, start_y, mid_x, mid_y, end_x, end_y,
                                                  fill=mode_colors[mode], width=4, smooth=True, tags="edge")
                        else:
                            self.canvas.create_line(start_x, start_y, end_x, end_y,
                                                  fill=mode_colors[mode], width=4, tags="edge")
                        
                        label_x = (start_x + end_x)/2
                        label_y = (start_y + end_y)/2
                        if offset != 0:
                            label_x += nx * 10 * offset
                            label_y += ny * 10 * offset
                        self.canvas.create_text(label_x, label_y, text=f"{mode}\n{info['time']:.1f}h",
                                              fill=mode_colors[mode], font=("Helvetica", 10, "bold"), tags="edge")
                        offset += 1
        
        # Draw nodes with gradient effect
        for node, (x, y) in self.scaled_coords.items():
            # Shadow
            self.canvas.create_oval(x-22, y-22, x+22, y+22, fill="#adb5bd", outline="", tags="node")
            # Node with gradient (simulated with two ovals)
            self.canvas.create_oval(x-20, y-20, x+20, y+20, fill="#17a2b8", outline="#495057", tags="node")
            self.canvas.create_oval(x-18, y-18, x+18, y+18, fill="#48c9b0", outline="", tags="node")
            self.canvas.create_text(x, y, text=node, font=("Helvetica", 14, "bold"), fill="#ffffff", tags="node")
        
        # Legend
        legend_x, legend_y = 20, 20
        self.canvas.create_rectangle(legend_x-5, legend_y-5, legend_x+100, legend_y+65, fill="#ffffff", outline="#dee2e6", tags="legend")
        for i, (mode, color) in enumerate(mode_colors.items()):
            y = legend_y + 20 * i + 10
            self.canvas.create_line(legend_x+10, y, legend_x+40, y, fill=color, width=4, tags="legend")
            self.canvas.create_text(legend_x+50, y, text=mode.capitalize(), anchor="w", font=("Helvetica", 10, "bold"), fill="#212529", tags="legend")
    
    def add_pair(self):
        """Add source-destination pair to the list."""
        source = self.source_var.get()
        dest = self.dest_var.get()
        if source == dest:
            messagebox.showwarning("Invalid Pair", "Source and destination cannot be the same.")
            return
        pair = (source, dest)
        if pair in self.pairs:
            messagebox.showwarning("Duplicate Pair", "This pair is already added.")
            return
        self.pairs.append(pair)
        self.pair_listbox.insert(tk.END, f"{source} -> {dest}")
    
    def run_routes(self):
        """Run route finding and animate convoys."""
        if not self.pairs:
            messagebox.showwarning("No Pairs", "Please add at least one source-destination pair.")
            return
        
        self.result_text.delete(1.0, tk.END)
        self.convoy_schedules = []
        algorithm = cuckoo_cmp if self.num_nodes > 10 else abc_cmp
        print(f"Using algorithm: {'Cuckoo Search' if self.num_nodes > 10 else 'Artificial Bee Colony'}")
        convoy_routes_modes = []
        
        for source, dest in self.pairs:
            route, route_modes, time = algorithm(source, dest, self.graph, self.modes)
            if route:
                convoy_routes_modes.append((route, route_modes))
                print(f"Route found: {source} -> {dest}, Time: {time:.1f}h")
            else:
                self.result_text.insert(tk.END, f"No valid route from {source} to {dest}\n")
                print(f"No valid route from {source} to {dest}")
        
        # Schedule convoys
        self.convoy_schedules = schedule_convoys(convoy_routes_modes, self.graph)
        
        # Display results
        self.results = []
        for route, route_modes, start_time, source, dest, total_time in self.convoy_schedules:
            segments = []
            for i in range(len(route) - 1):
                travel_time = self.graph[route[i]][route[i+1]][route_modes[i]]['time']
                segments.append(f"{route[i]}-{travel_time:.1f}H->{route[i+1]}")
            route_str = " + ".join(segments)
            result = f"Route ({source} -> {dest}): {route_str}\nTotal travel time: {total_time:.1f}H"
            self.result_text.insert(tk.END, result + "\n")
            self.results.append(result)
        
        if not self.convoy_schedules:
            self.result_text.insert(tk.END, "No routes could be scheduled due to conflicts.\n")
        else:
            # Start animation
            self.animate_convoys()
    
    def animate_convoys(self):
        """Animate convoys moving along their routes sequentially."""
        if not self.convoy_schedules:
            return
        
        self.canvas.delete("convoy")
        self.current_convoy = 0
        self.convoy_objects = []
        self.convoy_positions = []
        
        # Initialize convoys
        convoy_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f4a261', '#e76f51', '#2a9d8f']
        for i, (route, route_modes, start_time, source, dest, total_time) in enumerate(self.convoy_schedules):
            self.convoy_positions.append({
                'segment': 0,
                't': 0,
                'route': route,
                'modes': route_modes,
                'start_time': start_time,
                'color': convoy_colors[i % len(convoy_colors)],
                'label': f"{source}->{dest}"
            })
        
        # Start animating the first convoy
        self.animate_single_convoy()
    
    def animate_single_convoy(self):
        """Animate one convoy at a time."""
        if self.current_convoy >= len(self.convoy_positions):
            return  # All convoys done
        
        pos = self.convoy_positions[self.current_convoy]
        
        # Create convoy if not yet created
        if not self.convoy_objects or len(self.convoy_objects) <= self.current_convoy:
            x, y = self.scaled_coords[pos['route'][0]]
            convoy_id = self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=pos['color'], outline="#212529", width=2, tags="convoy")
            label_id = self.canvas.create_text(x, y-25, text=pos['label'], font=("Helvetica", 9, "bold"), fill="#212529", tags="convoy")
            self.convoy_objects.append((convoy_id, label_id))
        
        # Check start time
        animation_speed = 600  # ms per hour
        current_time = getattr(self.canvas, 'current_time', 0)
        if pos['t'] == 0 and pos['segment'] == 0 and current_time < pos['start_time'] * animation_speed:
            self.canvas.current_time = current_time + 50
            self.canvas.after(50, self.animate_single_convoy)
            return
        
        # Animate current segment
        step = 0.05
        if pos['segment'] < len(pos['route']) - 1:
            current_node = pos['route'][pos['segment']]
            next_node = pos['route'][pos['segment'] + 1]
            x0, y0 = self.scaled_coords[current_node]
            x1, y1 = self.scaled_coords[next_node]
            
            pos['t'] += step
            if pos['t'] >= 1:
                pos['t'] = 0
                pos['segment'] += 1
                if pos['segment'] >= len(pos['route']) - 1:
                    x, y = self.scaled_coords[next_node]
                else:
                    x, y = self.scaled_coords[pos['route'][pos['segment']]]
            else:
                x = x0 + (x1 - x0) * pos['t']
                y = y0 + (y1 - y0) * pos['t']
            
            # Update convoy
            convoy_id, label_id = self.convoy_objects[self.current_convoy]
            self.canvas.coords(convoy_id, x-12, y-12, x+12, y+12)
            self.canvas.coords(label_id, x, y-25)
        
        # Check if convoy is done
        if pos['segment'] >= len(pos['route']) - 1:
            self.current_convoy += 1
            self.canvas.current_time = 0
            self.canvas.after(500, self.animate_single_convoy)  # Brief pause before next convoy
        else:
            self.canvas.current_time = current_time + 50
            self.canvas.after(50, self.animate_single_convoy)
    
    def save_results(self):
        """Save route details to a text file."""
        if not hasattr(self, 'results') or not self.results:
            messagebox.showwarning("No Results", "Run the route planner first to generate results.")
            return
        try:
            with open("routes_output.txt", "w") as f:
                for result in self.results:
                    f.write(result + "\n")
            messagebox.showinfo("Success", "Results saved to routes_output.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

def main():
    """Main function to start the program."""
    while True:
        try:
            num_nodes = int(input("Enter the number of nodes (minimum 2): "))
            if num_nodes >= 2:
                break
            print("Please enter a number >= 2.")
        except ValueError:
            print("Please enter a valid integer.")
    
    root = tk.Tk()
    app = ConvoyGUI(root, num_nodes)
    root.mainloop()

if __name__ == "__main__":
    main()