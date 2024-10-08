import numpy as np
from animation_students import (
    render_graph,
    render_anim,
    make_surface,
    SPHERE_BONDS,
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
    blind_search
)

def generate_random_data(function, bounds, num_points=50, num_frames=10):
    xy_data = [np.random.rand(num_points, 2) * (bounds[1] - bounds[0]) + bounds[0] for _ in range(num_frames)]
    print(f"Shape of xy_data: {[xy.shape for xy in xy_data]}")
    z_data = [function(xy.T) for xy in xy_data]
    return xy_data, z_data

def visualize_function(function, bounds, xy_data, z_data ,step=0.1, title="Function Visualization"):
    X_surf, Y_surf, Z_surf = make_surface(min=bounds[0], max=bounds[1], function=function, step=step)
    render_graph(X_surf, Y_surf, Z_surf, title)
    render_anim(X_surf, Y_surf, Z_surf, xy_data, z_data, title)

def find_minima(function, bounds):
    best_position, best_value, xy_data, z_data = blind_search(function, bounds, 100000)
    visualize_function(function, bounds, xy_data, z_data, title=f"{function.__name__} Function")
    # visualize_function(function, bounds, best_position, best_value, title=f"{function.__name__} Function")
    

find_minima(sphere, SPHERE_BONDS)
find_minima(schwefel, SCHWEFEL_BOUNDS)
find_minima(ackley, ACKLEY_BOUNDS)
find_minima(rastrigin, RASTRIGIN_BOUNDS)
find_minima(rosenbrock, ROSENBROCK_BOUNDS)
find_minima(griewangk, GRIEWANGK_BOUNDS)
find_minima(levy, LEVY_BOUNDS)
find_minima(michalewicz, MICHALEWICZ_BOUNDS)
find_minima(zakharov, ZAKHAROV_BOUNDS)