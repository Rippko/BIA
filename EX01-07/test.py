import numpy as np
from colorama import Fore
import time
from animations_students import (
    render_graph,
    render_anim,
    make_surface,
    SPHERE_BOUNDS,
    sphere,
    SCHWEFEL_BOUNDS,
    schwefel,
    ACKLEY_BOUNDS,
    ackley,
    RASTRIGIN_BOUNDS,
    rastrigin,
    ROSENBROCK_BOUNDS,
    rosenbrock,
    GRIEWANGK_BOUNDS,
    griewangk,
    LEVY_BOUNDS,
    levy,
    MICHALEWICZ_BOUNDS,
    michalewicz,
    ZAKHAROV_BOUNDS,
    zakharov,
    blind_search,
    hill_climbing,
    simulated_annealing,
    plot_heatmap,
    differential_evolution,
    particle_swarm_optimization,
    soma,
    firefly_algorithm,
    tlbo_algorithm,
    run_experiments,
)

def generate_random_data(function, bounds, num_points=50, num_frames=10):
    xy_data = [np.random.rand(num_points, 2) * (bounds[1] - bounds[0]) + bounds[0] for _ in range(num_frames)]
    print(f"Shape of xy_data: {[xy.shape for xy in xy_data]}")
    z_data = [function(xy.T) for xy in xy_data]
    return xy_data, z_data

def visualize_function(function, bounds, xy_data, z_data, title="Function Visualization"):
    range_size = bounds[1] - bounds[0]
    relative_step = range_size * 0.01
    X_surf, Y_surf, Z_surf = make_surface(min=bounds[0], max=bounds[1], function=function, step=relative_step)
    render_anim(X_surf, Y_surf, Z_surf, xy_data, z_data, title)

def launch(function, bounds, algorithm="blind_search", animate=True):
    dimension = 2
    if algorithm == "blind_search":
        best_position, best_value, xy_data, z_data = blind_search(function, bounds, 100000)
    elif algorithm == "hill_climbing":
        best_position, best_value, xy_data, z_data = hill_climbing(function, bounds, num_iterations=100000, num_neighbors=5)
    elif algorithm == "simulated_annealing":
        best_position, best_value, xy_data, z_data = simulated_annealing(function, bounds, min_temperature = 50, initial_temperature=200, cooling_rate=0.99)
        plot_heatmap(function, bounds, xy_data, z_data)
    elif algorithm == "differential_evolution":
        best_position, best_value, xy_data, z_data = differential_evolution(function, bounds, dimension, 100, 50, 0.5, 0.5)
    elif algorithm == "pso":
        v_min = (bounds[1] * -0.3) / 5
        v_max = (bounds[1] * 0.3) / 5
        best_position, best_value, xy_data, z_data = particle_swarm_optimization(function, bounds, dimension, pop_size=15, M_max=50, v_min = v_min, v_max = v_max, c1=2.0, c2=2.0)
    elif algorithm == "soma":
        best_position, best_value, xy_data, z_data = soma(function, bounds, dimension, pop_size=20, M_max=100, step=0.11, path_length=3.0)
    elif algorithm == "firefly_algorithm":
        best_position, best_value, xy_data, z_data = firefly_algorithm(function, bounds, dimension, pop_size=50, M_max=300)
    elif algorithm == "tlbo_algorithm":
        best_position, best_value, xy_data, z_data = tlbo_algorithm(function, bounds, dimension, pop_size=50, M_max=500)
    elif algorithm == "run_experiments":
        run_experiments(function=function, dimension=30, population_size=30, max_evaluations=3000, num_experiments=30, bounds=bounds, output_file="results.xlsx")
        return
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if animate:
        visualize_function(function, bounds, xy_data, z_data, title=f"{function.__name__.capitalize()} Function ({algorithm.capitalize()})")
    else:
        visualize_function(function, bounds, best_position, best_value, title=f"{function.__name__.capitalize()} Function ({algorithm.capitalize()})")    

functions = {
    "1": sphere,
    "2": schwefel,
    "3": ackley,
    "4": rastrigin,
    "5": rosenbrock,
    "6": griewangk,
    "7": levy,
    "8": michalewicz,
    "9": zakharov,
}

algorithms = {
    "1": "blind_search",
    "2": "hill_climbing",
    "3": "simulated_annealing",
    "4": "differential_evolution",
    "5": "pso",
    "6": "soma",
    "7": "firefly_algorithm",
    "8": "tlbo_algorithm",
}

while True:
    chosen_function = input(f"{Fore.MAGENTA}Choose function:\n{Fore.CYAN} (1) Sphere\n (2) Schwefel\n (3) Ackley\n (4) Rastrigin\n (5) Rosenbrock\n (6) Griewangk\n (7) Levy\n (8) Michalewicz\n (9) Zakharov\n (0) Exit\n {Fore.RESET}")
    if chosen_function == "0":
        print(f"{Fore.YELLOW}Exiting...See you next time! {Fore.RESET}")
        break
    elif chosen_function not in functions:
        print(f"{Fore.RED}!!! Invalid input. Please enter a number between 1 and 9 !!!{Fore.RESET}\n")
        continue

    wanna_run_experimets = input(f"{Fore.MAGENTA}Would you like to run experiments? (y/n){Fore.RESET}")
    if wanna_run_experimets == "y":
        launch(functions[chosen_function], globals()[f"{functions[chosen_function].__name__.upper()}_BOUNDS"], "run_experiments", False)
        break

    wanna_animate = input(f"{Fore.MAGENTA}Would you like to animate the function? (y/n){Fore.RESET}")
    
    chosen_algorithm = input(f"{Fore.CYAN}Choose algorithm: (1) Blind Search, (2) Hill Climbing,\n (3)Simulated Annealing, (4) Differential evolution, (5) PSO, (6) SOMA,\n (7) Firefly alg., (8) TLBO alg.{Fore.RESET}\n")

    algorithm_to_launch = algorithms.get(chosen_algorithm)
    
    launch(functions[chosen_function], globals()[f"{functions[chosen_function].__name__.upper()}_BOUNDS"], algorithm_to_launch, True if wanna_animate == "y" else False)
