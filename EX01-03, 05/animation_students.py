import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

from copy import deepcopy


SPHERE_BOUNDS = (-5.12, 5.12)
SCHWEFEL_BOUNDS = (-500, 500)
ACKLEY_BOUNDS = (-32.768, 32.768)
RASTRIGIN_BOUNDS = (-5.12, 5.12)
ROSENBROCK_BOUNDS = (-2, 2)
GRIEWANGK_BOUNDS = (-600, 600)
LEVY_BOUNDS = (-10, 10)
MICHALEWICZ_BOUNDS = (0, np.pi)
ZAKHAROV_BOUNDS = (-10, 10)


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=0)

def schwefel(x: np.ndarray) -> np.ndarray:
    return 418.9829 * x.shape[0] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=0)

def rosenbrock(x: np.ndarray) -> np.ndarray:
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, axis=0)

def rastrigin(x: np.ndarray) -> np.ndarray:
    return 10 * x.shape[0] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)

def griewangk(x: np.ndarray) -> np.ndarray:
    d = x.shape[0]
    sum_term = 0.0
    prod_term = 1.0

    for i in range(d):
        sum_term += (x[i] ** 2) / 4000
        prod_term *= np.cos(x[i] / np.sqrt(i + 1)) 

    result = 1 + sum_term - prod_term
    
    return result

def levy(x: np.ndarray) -> np.ndarray:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2), axis=0)
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def michalewicz(x: np.ndarray, m=10) -> np.ndarray:
    return -sum(np.sin(xi) * (np.sin(i * xi**2 / np.pi))**(2*m) for i, xi in enumerate(x, 1))

def zakharov(x: np.ndarray) -> np.ndarray:
    sum1 = np.sum(x**2, axis=0)
    sum2 = 0.0
    for i in range(x.shape[0]):
        sum2 += 0.5 * (i + 1) * x[i]
    return sum1 + sum2**2 + sum2**4

def ackley(x: np.ndarray) -> np.ndarray:
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=0) / x.shape[0]))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=0) / x.shape[0])
    return term1 + term2 + 20 + np.e

def blind_search(
    function: callable,
    bounds: tuple[float, float],
    num_samples: int,
):
    best_position = None
    best_value = np.inf
    xy_data = []
    z_data = []
    for _ in range(num_samples):
        x = np.random.uniform(bounds[0], bounds[1], 2)
        value = function(x)
        if value < best_value:
            best_value = value
            best_position = x
            xy_data.append(np.array([x]))
            z_data.append(np.array([value]))
    best_position = [np.array([best_position])]
    best_value = [np.array([best_value])]
    return best_position, best_value, xy_data, z_data

def hill_climbing(
    function: callable,
    bounds: tuple[float, float],
    num_iterations: int,
    num_neighbors: int,
    sigma: float = 0.5,
):
    current_position = np.random.uniform(bounds[0], bounds[1], 2)
    current_value = function(current_position)
    
    best_position = current_position
    best_value = current_value
    
    xy_data = [np.array([current_position])]
    z_data = [np.array([current_value])]
    
    for _ in range(num_iterations):
        neighbors = current_position + np.random.normal(0, sigma, size=(num_neighbors, 2))        
        neighbors = np.clip(neighbors, bounds[0], bounds[1])
        
        for neighbor in neighbors:
            value = function(neighbor)
            if value < current_value:
                current_position = neighbor
                current_value = value
                
        if current_value < best_value:
            best_position = current_position
            best_value = current_value
            xy_data.append(np.array([current_position]))
            z_data.append(np.array([current_value]))
                    
    best_position = [np.array([best_position])]
    best_value = [np.array([best_value])]
    return best_position, best_value, xy_data, z_data

def simulated_annealing(
    function: callable,
    bounds: tuple[float, float],
    min_temperature: int,
    initial_temperature: float,
    cooling_rate: float,
    sigma: float = 0.5,
):
    current_position = np.random.uniform(bounds[0], bounds[1], 2)
    current_value = function(current_position)
    
    best_position = current_position
    best_value = current_value

    xy_data = [np.array([current_position])]
    z_data = [np.array([current_value])]
    
    temperature = initial_temperature

    while temperature > min_temperature:
        neighbor = current_position + np.random.normal(0, sigma, 2)
        neighbor = np.clip(neighbor, bounds[0], bounds[1])
        
        neighbor_value = function(neighbor)
        
        delta_value = neighbor_value - current_value

        if delta_value < 0 or np.random.rand() < np.exp(-delta_value / temperature):
            current_position = neighbor
            current_value = neighbor_value
        
        if current_value < best_value:
            best_position = current_position
            best_value = current_value
            
            xy_data.append(np.array([current_position]))
            z_data.append(np.array([current_value]))
        
        temperature *= cooling_rate
        
    best_position = [np.array([best_position])]
    best_value = [np.array([best_value])]
    
    return best_position, best_value, xy_data, z_data

def differential_evolution(
    function, bounds, dimension, NP, g_max, F, CR
):
    lower_bound, upper_bound = bounds
    pop = np.random.uniform(lower_bound, upper_bound, (NP, dimension))
    fitness = np.array([function(ind) for ind in pop])
    xy_data, z_data = [], []

    g = 0
    while g < g_max:
        new_pop = deepcopy(pop)
        new_fitness = deepcopy(fitness)

        for i in range(NP):
            indices = [idx for idx in range(NP) if idx != i]
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = np.clip(v, lower_bound, upper_bound)
            u = np.copy(pop[i])
            j_rand = np.random.randint(0, dimension)

            for j in range(dimension):
                if np.random.uniform(0, 1) < CR or j == j_rand:
                    u[j] = v[j]

            f_u = function(u)
            
            if f_u <= fitness[i]:
                new_pop[i] = u
                new_fitness[i] = f_u

        pop = new_pop
        fitness = new_fitness

        xy_data.append(np.copy(pop[:, :2]))
        z_data.append(np.copy(fitness))

        g += 1

    best_position = [np.array(pop[np.argmin(fitness)])]
    best_value = [np.array(np.min(fitness))] 

    return best_position, best_value, xy_data, z_data


def update_frame(
    i: int,
    xy_data: list[np.array],
    z_data: list[np.array],
    scat,
    ax,
):
    scat[0].remove()
    scat[0] = ax[0].scatter(
        xy_data[i][:, 0], xy_data[i][:, 1], z_data[i], c="red"
    )


def render_anim(
    surface_X: np.array,
    surface_Y: np.array,
    surface_Z: np.array,
    xy_data: list[np.array],
    z_data: list[np.array],
    title: str
):
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        surface_X,
        surface_Y,
        surface_Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.6,
    )
    if len(xy_data) > 0 and len(z_data) > 0:
        if xy_data[0].ndim == 1:
            xy_data[0] = xy_data[0].reshape(1, -1)
        if z_data[0].ndim == 0:
            z_data[0] = np.array([z_data[0]])
        scat = ax.scatter(xy_data[0][:, 0], xy_data[0][:, 1], z_data[0], c="red")

        animation = FuncAnimation(
            fig,
            update_frame,
            len(xy_data),
            fargs=(xy_data, z_data, [scat], [ax]),
            interval=300,
            repeat=False,
        )
    else:
        print("No data provided for animation.")
    plt.title(title)
    plt.show()


def render_graph(
    surface_X: np.array,
    surface_Y: np.array,
    surface_Z: np.array,
    title: str
):
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        surface_X,
        surface_Y,
        surface_Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.6,
    )
    plt.title(title)
    plt.show()


def make_surface(
    min: float,
    max: float,
    function: callable,
    step: float,
):
    X = np.arange(min, max, step)
    Y = np.arange(min, max, step)
    X, Y = np.meshgrid(X, Y)
    Z = function(np.array([X, Y]))
    return X, Y, Z


def plot_heatmap(function, bounds, xy_data, z_data, step=0.1, title="Heatmap"):
    x_points = [pos[0][0] for pos in xy_data]
    y_points = [pos[0][1] for pos in xy_data]
    z_points = [val[0] for val in z_data]
    X, Y = np.meshgrid(
        np.arange(bounds[0], bounds[1], step),
        np.arange(bounds[0], bounds[1], step)
    )
    Z = function(np.array([X, Y]))
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, cmap="viridis", levels=50)
    plt.colorbar(label="Function Value")

    # Plot the path over the heatmap
    plt.plot(x_points, y_points, marker='o', color='red', markersize=4)
    plt.scatter(x_points[-1], y_points[-1], color='blue', label='Best Position', zorder=5)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

# def render_static(
#     function: callable,
#     bounds: tuple[float, float],
#     best_position: np.array,
#     best_value: np.array,
#     title: str
# ):
#     fig = plt.figure()
#     fig.canvas.manager.set_window_title(title)
#     ax = plt.axes(projection="3d")
#     X_surf, Y_surf, Z_surf = make_surface(min=bounds[0], max=bounds[1], function=function, step=0.1)
#     ax.plot_surface(X_surf, Y_surf, Z_surf, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.6)
#     ax.scatter(
#         best_position[0],  # X coordinate of the best position
#         best_position[1],  # Y coordinate of the best position
#         best_value,        # Z value (height) at the best position
#         c="red", s=100, label="Best Position"
#     )
#     plt.title(title)
#     plt.show()