import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from scipy.special import erfinv
from scipy.stats import norm



def run_full_filter(Depth, particles = 500, iterations = 500, min_init_depth = -1):

    #draw initial true position
    while True:
        x1_idx_true = np.random.randint(0, Depth.shape[0])
        x2_idx_true = np.random.randint(0, Depth.shape[1])
        if Depth[x1_idx_true, x2_idx_true] < min_init_depth:
            break

    y_meas_array = np.zeros(iterations+1)
    x_samples_array = np.zeros((iterations+1, particles, 2))
    x_true_array = np.zeros((iterations+1, 2))
    x_samples = np.zeros((particles, 2), dtype=int)
    wj_samples = np.ones(particles) / particles

    #y_meas = np.random.uniform(Depth[x1_idx_true, x2_idx_true]*0.85, Depth[x1_idx_true, x2_idx_true]*1.15)
    sigma = 0.15*Depth[x1_idx_true, x2_idx_true] / (np.sqrt(2)*erfinv(-0.95))
    print(sigma)
    y_meas = np.random.normal(Depth[x1_idx_true, x2_idx_true], sigma)
    y_belief = np.array([y_meas*1.2, y_meas*0.8]) #TODO: can be optimized
    
    for j in range(particles):
        while True:
            x1_idx = np.random.randint(0, Depth.shape[0], dtype=int)
            x2_idx = np.random.randint(0, Depth.shape[1], dtype=int)
            if Depth[x1_idx, x2_idx] < y_belief[1] and Depth[x1_idx, x2_idx] > y_belief[0]:
                break
        x_samples[j, 0] = x1_idx
        x_samples[j, 1] = x2_idx

    #store initial values
    x_true_array[0, 0] = x1_idx_true
    x_true_array[0, 1] = x2_idx_true
    x_samples_array[0, :, :] = x_samples
    y_meas_array[0] = y_meas

    for iter in tqdm(range(iterations), desc="Iterations"):

        sigma = 0.15*Depth[x1_idx_true, x2_idx_true] / (np.sqrt(2)*erfinv(-0.95))
        y_meas = np.random.normal(Depth[x1_idx_true, x2_idx_true], sigma)

        for j in range(particles):
            sigma = 0.15*Depth[x_samples[j, 0], x_samples[j, 1]] / (np.sqrt(2)*erfinv(-0.95))
            wj_samples[j] *= norm.pdf(y_meas, Depth[x_samples[j, 0], x_samples[j, 1]], sigma)
            
        wj_samples /= np.sum(wj_samples) #normalize weights
        N_eff = 1 / np.sum(wj_samples**2)

        #resample from the particles with positive probability
        if N_eff < 0.8*particles:
            idx = np.random.choice(range(0,particles), size=particles, p=wj_samples/np.sum(wj_samples))
            x_samples = x_samples[idx, :]
            wj_samples = np.ones(particles) / particles

    
         #true position diffusion step
        if x1_idx_true == 0:
            x1_idx_true = x1_idx_true + np.random.choice([0, 1])
        elif x1_idx_true == Depth.shape[0]-1:
            x1_idx_true = x1_idx_true + np.random.choice([-1, 0])
        else:
            x1_idx_true = x1_idx_true + np.random.choice([-1, 0, 1])

        if x2_idx_true == 0:
            x2_idx_true = x2_idx_true + np.random.choice([0, 1])
        elif x2_idx_true == Depth.shape[1]-1:
            x2_idx_true = x2_idx_true + np.random.choice([-1, 0])
        else:
            x2_idx_true = x2_idx_true + np.random.choice([-1, 0, 1])

        #take diffusion step for each particle
        for j in range(particles):
            #x1
            if x_samples[j, 0] == 0:
                x_samples[j, 0] = x_samples[j, 0] + np.random.choice([0, 1])
            elif x_samples[j, 0] == Depth.shape[0]-1:
                x_samples[j, 0] = x_samples[j, 0] + np.random.choice([-1, 0])
            else:
                x_samples[j, 0] = x_samples[j, 0] + np.random.choice([-1, 0, 1])
            #x2
            if x_samples[j, 1] == 0:
                x_samples[j, 1] = x_samples[j, 1] + np.random.choice([0, 1])
            elif x_samples[j, 1] == Depth.shape[1]-1:
                x_samples[j, 1] = x_samples[j, 1] + np.random.choice([-1, 0])
            else:
                x_samples[j, 1] = x_samples[j, 1] + np.random.choice([-1, 0, 1])

        #store values
        x_true_array[iter+1, 0] = x1_idx_true
        x_true_array[iter+1, 1] = x2_idx_true
        x_samples_array[iter+1, :, :] = x_samples
        y_meas_array[iter+1] = y_meas

    return x_true_array, x_samples_array, y_meas_array



def run_filter_iteration(Depth, x1_idx_true, x2_idx_true, x_samples):

    y_meas = np.random.uniform(Depth[x1_idx_true, x2_idx_true]*1.15, Depth[x1_idx_true, x2_idx_true]*0.85)
    particles = x_samples.shape[0]

    wj_samples = np.zeros(particles)
    for j in range(particles):
        #check if depth at particle could have given the measurement
        if y_meas > Depth[x_samples[j, 0], x_samples[j, 1]]*1.15 and y_meas < Depth[x_samples[j, 0], x_samples[j, 1]]*0.85:
            wj_samples[j] = 1
        else:
            wj_samples[j] = 0

    #resample from the particles with positive probability
    for j in range(particles):
        if wj_samples[j] == 0:
            idx = np.random.choice(range(0,particles), p=wj_samples/np.sum(wj_samples))
            x_samples[j, :] = x_samples[idx, :]

    #true position diffusion step
    if x1_idx_true == 0:
        x1_idx_true = x1_idx_true + np.random.choice([0, 1])
    elif x1_idx_true == Depth.shape[0]-1:
        x1_idx_true = x1_idx_true + np.random.choice([-1, 0])
    else:
        x1_idx_true = x1_idx_true + np.random.choice([-1, 0, 1])

    if x2_idx_true == 0:
        x2_idx_true = x2_idx_true + np.random.choice([0, 1])
    elif x2_idx_true == Depth.shape[1]-1:
        x2_idx_true = x2_idx_true + np.random.choice([-1, 0])
    else:
        x2_idx_true = x2_idx_true + np.random.choice([-1, 0, 1])

    #take diffusion step for each particle
    for j in range(particles):
        if x_samples[j, 0] == 0:
            x_samples[j, 0] = x_samples[j, 0] + np.random.choice([0, 1])
        elif x_samples[j, 0] == Depth.shape[0]-1:
            x_samples[j, 0] = x_samples[j, 0] + np.random.choice([-1, 0])
        else:
            x_samples[j, 0] = x_samples[j, 0] + np.random.choice([-1, 0, 1])

        if x_samples[j, 1] == 0:
            x_samples[j, 1] = x_samples[j, 1] + np.random.choice([0, 1])
        elif x_samples[j, 1] == Depth.shape[1]-1:
            x_samples[j, 1] = x_samples[j, 1] + np.random.choice([-1, 0])
        else:
            x_samples[j, 1] = x_samples[j, 1] + np.random.choice([-1, 0, 1])

    return x1_idx_true, x2_idx_true, x_samples, y_meas


def initialize_filter(Depth, particles = 500, min_init_depth = -1):

    #draw initial true position
    while True:
        x1_idx_true = np.random.randint(0, Depth.shape[0])
        x2_idx_true = np.random.randint(0, Depth.shape[1])
        if Depth[x1_idx_true, x2_idx_true] < min_init_depth:
            break

    y_meas = np.random.uniform(Depth[x1_idx_true, x2_idx_true]*0.85, Depth[x1_idx_true, x2_idx_true]*1.15)
    y_belief = np.array([y_meas*1.2, y_meas*0.8])

    x_samples = np.zeros((particles, 2), dtype=int)
    for j in range(particles):
        while True:
            x1_idx = np.random.randint(0, Depth.shape[0], dtype=int)
            x2_idx = np.random.randint(0, Depth.shape[1], dtype=int)
            if Depth[x1_idx, x2_idx] < y_belief[1] and Depth[x1_idx, x2_idx] > y_belief[0]:
                break
        x_samples[j, 0] = x1_idx
        x_samples[j, 1] = x2_idx

    return x1_idx_true, x2_idx_true, x_samples


if __name__ == "__main__":
    import numpy as np
    import scipy.io
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import time

    dataset = "Vombsjon" #choose from "Vombsjon", "Oresund", "Ringsjon", "Skagerack", "Bolmen"
    particles = 1000
    iterations = 500
    min_init_depth = -2

    mat = scipy.io.loadmat(f"data/{dataset}.mat")
    Depth = mat["Depth"].T
    x = np.arange(0, Depth.shape[0])
    y = np.arange(0, Depth.shape[1])
    X, Y = np.meshgrid(x, y)

    x_true_array, x_samples_array, y_meas_array = run_full_filter(Depth, particles, iterations, min_init_depth)


