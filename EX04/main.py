import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

NUM_CITIES = 20
POP_SIZE = 20
NUM_GENERATIONS = 5000
MUTATION_RATE = 0.5

cities = np.random.rand(NUM_CITIES, 2)

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def route_length(route):
    return sum(distance(cities[route[i]], cities[route[(i + 1) % NUM_CITIES]]) for i in range(NUM_CITIES))

def create_population():
    population = []
    for _ in range(POP_SIZE):
        route = list(range(NUM_CITIES))
        random.shuffle(route)
        population.append(route)
    return population

def tournament_selection(population, parent1 ,fitness):
    tournament_size = 5
    selected = random.sample(range(POP_SIZE), tournament_size)
    selected = sorted(selected, key=lambda x: fitness[x])
    if population[selected[0]] == parent1:
        return population[selected[1]]
    return population[selected[0]]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [-1] * NUM_CITIES
    child[start:end] = parent1[start:end]
    for i in range(start, end):
        if parent2[i] not in child:
            pos = i
            while start <= pos < end:
                pos = parent2.index(parent1[pos])
            child[pos] = parent2[i]
    for i in range(NUM_CITIES):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        route[i], route[j] = route[j], route[i]

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
        
        path = route + [route[0]]
        path_points = cities[path]
        ax.plot(path_points[:, 0], path_points[:, 1], 'b-', marker='o', markersize=5)

    ani = animation.FuncAnimation(fig, update, frames=len(routes), interval=100, repeat=False)
    plt.show()

def plot_distance_evolution(distances):
    plt.figure(figsize=(10, 5))
    plt.plot(distances, marker='o', color='b', linestyle='-')
    plt.title("Evolution of Distance Over Generations")
    plt.xlabel("Change of length over generations")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

def genetic_algorithm():
    population = create_population()
    best_routes = []
    best_length = float('inf')
    best_lengths = []

    for generation in range(NUM_GENERATIONS):
        fitness = [1 / route_length(route) for route in population]
        
        new_population = population.copy()
        for j in range(POP_SIZE):
            parent1 = population[j]
            parent2 = tournament_selection(population, parent1 ,fitness)
            child1 = crossover(parent1, parent2)
            mutate(child1)
            if route_length(child1) < route_length(parent1):
                new_population[j] = child1
        population = new_population

        generation_best_length = min(route_length(route) for route in population)
        if generation_best_length < best_length:
            best_length = generation_best_length
            best_routes.append(min(population, key=lambda x: route_length(x)))
            best_lengths.append(best_length)

    show_route(best_routes, best_lengths, cities)
    plot_distance_evolution(best_lengths)

genetic_algorithm()