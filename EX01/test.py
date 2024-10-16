import numpy as np
from colorama import Fore
from animation_students import (
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
)

def generate_random_data(function, bounds, num_points=50, num_frames=10):
    xy_data = [np.random.rand(num_points, 2) * (bounds[1] - bounds[0]) + bounds[0] for _ in range(num_frames)]
    print(f"Shape of xy_data: {[xy.shape for xy in xy_data]}")
    z_data = [function(xy.T) for xy in xy_data]
    return xy_data, z_data

def visualize_function(function, bounds, xy_data, z_data ,step=0.1, title="Function Visualization"):
    X_surf, Y_surf, Z_surf = make_surface(min=bounds[0], max=bounds[1], function=function, step=step)
    # render_graph(X_surf, Y_surf, Z_surf, title)
    render_anim(X_surf, Y_surf, Z_surf, xy_data, z_data, title)

def launch(function, bounds, algorithm="blind_search", animate=True):
    if algorithm == "blind_search":
        best_position, best_value, xy_data, z_data = blind_search(function, bounds, 100000)
    elif algorithm == "hill_climbing":
        best_position, best_value, xy_data, z_data = hill_climbing(function, bounds, num_iterations=100000, num_neighbors=5)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    if animate:
        visualize_function(function, bounds, xy_data, z_data, title=f"{function.__name__} Function ({algorithm})")
    else:
        visualize_function(function, bounds, best_position, best_value, title=f"{function.__name__} Function ({algorithm})")    

functions = {
    "1": sphere,
    "2": schwefel,
    "3": ackley,
    "4": rastrigin,
    "5": rosenbrock,
    "6": griewangk,
    "7": levy,
    "8": michalewicz,
    "9": zakharov
}

while True:
    chosen_function = input(f"{Fore.MAGENTA}Enter which function you would like to display\n {Fore.GREEN}(1) Sphere\n (2) Schwefel\n (3) Ackley\n (4) Rastrigin\n (5) Rosenbrock\n (6) Griewangk\n (7) Levy\n (8) Michalewicz\n (9) Zakharov\n (0) Exit\n {Fore.RESET}")
    if chosen_function == "0":
        print(f"{Fore.YELLOW}Exiting...See you next time! {Fore.RESET}")
        break

    wanna_animate = input("Would you like to animate the function? (y/n)")
    
    chosen_algorithm = input("Choose algorithm: (1) Blind Search, (2) Hill Climbing\n")
    algorithm_to_launch = "blind_search" if chosen_algorithm == "1" else "hill_climbing"
    
    if chosen_function in functions:
        function_to_launch = functions[chosen_function]
        launch(function_to_launch, globals()[f"{function_to_launch.__name__.upper()}_BOUNDS"], algorithm_to_launch, True if wanna_animate == "y" else False)
    else:
        print("Invalid input. Please enter a number between 1 and 9.")