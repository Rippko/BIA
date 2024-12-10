import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import pandas as pd
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

def particle_swarm_optimization(function, bounds, dimension, pop_size, M_max, v_min, v_max, ws=0.9, we=0.4, c1=2.0, c2=2.0):
    lower_bound, upper_bound = bounds
    swarm = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    velocities = np.random.uniform(v_min, v_max, (pop_size, dimension))
    fitness = np.array([function(particle) for particle in swarm])
    p_best = np.copy(swarm)
    p_best_fitness = deepcopy(fitness)
    global_best_position = swarm[np.argmin(fitness)]
    global_best_fitness = np.min(fitness)

    xy_data, z_data = [], []

    m = 0
    while m < M_max:
        for i in range(pop_size):
            r1, r2 = np.random.uniform(0, 1, 2)
            w = ws - ((ws - we) * m) / M_max
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (p_best[i] - swarm[i])
                + c2 * r2 * (global_best_position - swarm[i])
            )
            velocities[i] = np.clip(velocities[i], v_min, v_max)

            swarm[i] += velocities[i]
            swarm[i] = np.clip(swarm[i], lower_bound, upper_bound)

            current_fitness = function(swarm[i])
            fitness[i] = current_fitness

            if current_fitness < p_best_fitness[i]:
                p_best[i] = swarm[i]
                p_best_fitness[i] = current_fitness

                if current_fitness < global_best_fitness:
                    global_best_position = swarm[i]
                    global_best_fitness = current_fitness

        xy_data.append(np.copy(swarm[:, :2]))
        z_data.append(np.copy(fitness))

        m += 1

    best_position = [np.array(global_best_position)]
    best_value = [np.array(global_best_fitness)]

    return best_position, best_value, xy_data, z_data

def soma(function, bounds, dimension, pop_size, M_max, step, path_length, prt=0.4):
    lower_bound, upper_bound = bounds
    swarm = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    fitness = np.array([function(individual) for individual in swarm])
    
    leader_index = np.argmin(fitness)
    leader_position = np.copy(swarm[leader_index])
    leader_fitness = fitness[leader_index]

    xy_data, z_data = [], []

    for _ in range(M_max):
        new_population = np.copy(swarm)

        for i in range(pop_size):
            if i == leader_index:
                continue
            
            individual = swarm[i]
            current_fitness = fitness[i]
            migration_vector = leader_position - individual

            for t in np.arange(0, path_length + step, step):
                perturbation = np.random.uniform(0, 1, dimension) < prt
                trial_position = individual + t * migration_vector * perturbation
                trial_position = np.clip(trial_position, lower_bound, upper_bound)

                trial_fitness = function(trial_position)

                if trial_fitness < current_fitness:
                    new_population[i] = trial_position
                    current_fitness = trial_fitness

        swarm = np.copy(new_population)
        fitness = np.array([function(ind) for ind in swarm])

        leader_index = np.argmin(fitness)
        leader_position = np.copy(swarm[leader_index])
        leader_fitness = fitness[leader_index]

        xy_data.append(np.copy(swarm[:, :2]))
        z_data.append(np.copy(fitness))

    best_position = [np.array(leader_position)]
    best_value = [np.array(leader_fitness)]

    return best_position, best_value, xy_data, z_data

def firefly_algorithm(function, bounds, dimension, pop_size, M_max, alpha=0.3, beta_0=1.0, gamma=2.0):
    lower_bound, upper_bound = bounds
    fireflies = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    light_intensity = np.array([function(firefly) for firefly in fireflies])
    xy_data, z_data = [], []

    for _ in range(M_max):
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    distance = np.linalg.norm(fireflies[i] - fireflies[j])

                    light_intensity_i = light_intensity[i] * np.exp(-gamma * distance)
                    light_intensity_j = light_intensity[j] * np.exp(-gamma * distance)

                    if light_intensity_j < light_intensity_i:
                        #moving firefly i towards j
                        beta = beta_0 / (1 + distance)
                        epsilon = np.random.normal(0, 1, dimension)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * epsilon
                        fireflies[i] = np.clip(fireflies[i], lower_bound, upper_bound)

                        current_light_intensity = function(fireflies[i])
                        light_intensity[i] = current_light_intensity

        xy_data.append(np.copy(fireflies[:, :2]))
        z_data.append(np.copy(light_intensity))
    # Find the best firefly
    best_index = np.argmin(light_intensity)
    best_position = [np.array(fireflies[best_index])]
    best_value = [np.array(light_intensity[best_index])]

    return best_position, best_value, xy_data, z_data

def tlbo_algorithm(function, bounds, dimension, pop_size, M_max):
    lower_bound, upper_bound = bounds
    students = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    fitness_values = np.array([function(student) for student in students])
    
    xy_data, z_data = [], []

    for _ in range(M_max):
        teacher_index = np.argmin(fitness_values)
        teacher = students[teacher_index]
        mean = np.mean(students, axis=0)
        teaching_factor = np.random.choice([1, 2])
        
        for i in range(pop_size):
            r = np.random.uniform(0, 1, dimension)
            new_position = students[i] + r * (teacher - teaching_factor * mean)
            new_position = np.clip(new_position, lower_bound, upper_bound)
            new_fitness = function(new_position)
            
            if new_fitness < fitness_values[i]:
                students[i] = new_position
                fitness_values[i] = new_fitness

        for i in range(pop_size):
            partner_index = np.random.choice([x for x in range(pop_size) if x != i])
            partner = students[partner_index]
            r = np.random.uniform(0, 1, dimension)
            
            if fitness_values[i] < fitness_values[partner_index]:
                new_position = students[i] + r * (students[i] - partner)
            else:
                new_position = students[i] - r * (students[i] - partner)
            
            new_position = np.clip(new_position, lower_bound, upper_bound)
            new_fitness = function(new_position)
            
            if new_fitness < fitness_values[i]:
                students[i] = new_position
                fitness_values[i] = new_fitness

        xy_data.append(np.copy(students[:, :2]))
        z_data.append(np.copy(fitness_values))
    
    best_index = np.argmin(fitness_values)
    best_position = [np.array(students[best_index])]
    best_value = [np.array(fitness_values[best_index])]

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
            interval=10,
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
