import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_distances(cities):
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i][j] = distances[j][i] = np.linalg.norm(cities[i] - cities[j])
    return distances

def calculate_probabilities(current_city, unvisited, pheromones, distances, alpha, beta):
    pheromone = pheromones[current_city, unvisited] ** alpha
    heuristic = (1 / distances[current_city, unvisited]) ** beta
    probabilities = pheromone * heuristic
    return probabilities / np.sum(probabilities)

def construct_route(start_city, n_cities, pheromones, distances, alpha, beta):
    route = [start_city]
    while len(route) < n_cities:
        current_city = route[-1]
        unvisited = [city for city in range(n_cities) if city not in route]
        probabilities = calculate_probabilities(current_city, unvisited, pheromones, distances, alpha, beta)
        next_city = np.random.choice(unvisited, p=probabilities)
        route.append(next_city)
    route.append(route[0]) # návrat do výchozího města
    return route

def update_pheromones(pheromones, all_routes, all_distances, evaporation_rate, Q):
    pheromones *= (1 - evaporation_rate)
    for route, distance in zip(all_routes, all_distances):
        pheromone_contribution = Q / distance
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            pheromones[from_city][to_city] += pheromone_contribution

def ant_colony_optimization(cities, n_ants, n_iterations, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=1):
    n_cities = len(cities)
    distances = compute_distances(cities)
    pheromones = np.ones((n_cities, n_cities)) * 0.1
    best_distance = float('inf')
    best_route = None
    routes_history, distances_history = [], []

    for _ in range(n_iterations):
        all_routes = []
        all_distances = []

        for ant in range(n_ants):
            start_city = ant
            route = construct_route(start_city, n_cities, pheromones, distances, alpha, beta)
            all_routes.append(route)
            distance = np.sum(np.fromiter((distances[route[i], route[i + 1]] for i in range(n_cities)), dtype=float))
            all_distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_route = route

        update_pheromones(pheromones, all_routes, all_distances, evaporation_rate, Q)

        min_idx = np.argmin(all_distances)
        routes_history.append(all_routes[min_idx])
        distances_history.append(all_distances[min_idx])

    routes_history.append(best_route)
    distances_history.append(best_distance)
    return routes_history, distances_history

def show_route(routes, distances, cities):
    fig, ax = plt.subplots()
    def update(num):
        route = routes[num]
        distance = distances[num]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Traveling Salesman Path\nDistance: {distance:.2f}')        
        ax.scatter(cities[:, 0], cities[:, 1], c='red', s=50)
        for i, (x, y) in enumerate(cities):
            ax.text(x, y, str(i), fontsize=12, ha='right')
        
        path = route
        path_points = cities[path]
        ax.plot(path_points[:, 0], path_points[:, 1], 'b-', marker='o', markersize=5)

    ani = animation.FuncAnimation(fig, update, frames=len(routes), interval=100, repeat=False)
    return ani

def plot_distance_evolution(distances):
    plt.figure(figsize=(10, 5))
    plt.plot(distances, color='b', linestyle='-', label="Distance Evolution")
    
    min_distance = min(distances)
    min_index = np.argmin(distances)
    
    plt.scatter(min_index, min_distance, color='red', label=f"Min: {min_distance:.2f} (Gen {min_index})")
    plt.annotate(f"Min: {min_distance:.2f}\nGen: {min_index}", 
             xy=(min_index, min_distance), 
             xytext=(min_index + len(distances) * 0.05, min_distance + min_distance * 0.05),
             arrowprops=dict(facecolor='black', arrowstyle="->", linewidth=2),
             fontsize=10)
    
    plt.legend()    
    plt.title("Evolution of Distance Over Generations")
    plt.xlabel("Generations")
    plt.ylabel("Distance")
    plt.grid(True)

n_cities = 20
cities = np.random.rand(n_cities, 2)
routes_history, distances_history = ant_colony_optimization(cities, n_ants=20, n_iterations=100)

anim = show_route(routes_history, distances_history, cities)
plot_distance_evolution(distances_history[:-1])
plt.show()
