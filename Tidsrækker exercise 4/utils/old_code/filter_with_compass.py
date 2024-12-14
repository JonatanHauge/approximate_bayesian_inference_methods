import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

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

    y_meas = np.random.uniform(Depth[x1_idx_true, x2_idx_true]*0.85, Depth[x1_idx_true, x2_idx_true]*1.15)
    y_belief = np.array([y_meas*1.2, y_meas*0.8])

    x_samples = np.zeros((particles, 2), dtype=int)
    wj_samples = np.zeros(particles)
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

        y_meas = np.random.uniform(Depth[x1_idx_true, x2_idx_true]*1.15, Depth[x1_idx_true, x2_idx_true]*0.85)

        for j in range(particles):
            #check if depth at particle could have given the measurement
            if y_meas > Depth[x_samples[j, 0], x_samples[j, 1]]*1.15 and y_meas < Depth[x_samples[j, 0], x_samples[j, 1]]*0.85:
                wj_samples[j] = 1
            else:
                wj_samples[j] = 0

        #resample from the particles with positive probability and add extra diffusion
        for j in range(particles):
            if wj_samples[j] == 0:
                idx = np.random.choice(range(0,particles), p=wj_samples/np.sum(wj_samples))
                if x_samples[idx, 0] == 0:
                    x1_diff_dir = np.random.choice([0, 1])
                elif x_samples[idx, 0] == Depth.shape[0]-1:
                    x1_diff_dir = np.random.choice([-1, 0])
                else:
                    x1_diff_dir = np.random.choice([-1, 0, 1])
                
                if x_samples[idx, 1] == 0:
                    x2_diff_dir = np.random.choice([0, 1])
                elif x_samples[idx, 1] == Depth.shape[1]-1:
                    x2_diff_dir = np.random.choice([-1, 0])
                else:
                    x2_diff_dir = np.random.choice([-1, 0, 1])
                    
                x_samples[j, 0] = x_samples[idx, 0] + x1_diff_dir
                x_samples[j, 1] = x_samples[idx, 1] + x2_diff_dir

    
        #true position diffusion step
        if x1_idx_true == 0:
            x1_diff_dir = np.random.choice([0, 1])
            x1_idx_true = x1_idx_true + x1_diff_dir
        elif x1_idx_true == Depth.shape[0]-1:
            x1_diff_dir = np.random.choice([-1, 0])
            x1_idx_true = x1_idx_true + x1_diff_dir
        else:
            x1_diff_dir = np.random.choice([-1, 0, 1])
            x1_idx_true = x1_idx_true + x1_diff_dir

        if x2_idx_true == 0:
            x2_diff_dir = np.random.choice([0, 1])
            x2_idx_true = x2_idx_true + x2_diff_dir
        elif x2_idx_true == Depth.shape[1]-1:
            x2_diff_dir = np.random.choice([-1, 0])
            x2_idx_true = x2_idx_true + x2_diff_dir
        else:
            x2_diff_dir = np.random.choice([-1, 0, 1])
            x2_idx_true = x2_idx_true + x2_diff_dir

        #take diffusion step for each particle
        for j in range(particles):
            if x_samples[j, 0] == 0 and x1_diff_dir == -1:
                x_samples[j, 0] = x_samples[j, 0] + np.random.choice([0, 1])
            elif x_samples[j, 0] == Depth.shape[0]-1 and x1_diff_dir == 1:
                x_samples[j, 0] = x_samples[j, 0] + np.random.choice([-1, 0])
            else:
                x_samples[j, 0] = x_samples[j, 0] + x1_diff_dir

            if x_samples[j, 1] == 0 and x2_diff_dir == -1:
                x_samples[j, 1] = x_samples[j, 1] + np.random.choice([0, 1])
            elif x_samples[j, 1] == Depth.shape[1]-1 and x2_diff_dir == 1:
                x_samples[j, 1] = x_samples[j, 1] + np.random.choice([-1, 0])
            else:
                x_samples[j, 1] = x_samples[j, 1] + x2_diff_dir

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

    #resample from the particles with positive probability and add extra diffusion
    for j in range(particles):
        if wj_samples[j] == 0:
            idx = np.random.choice(range(0,particles), p=wj_samples/np.sum(wj_samples))
            if x_samples[idx, 0] == 0:
                x1_diff_dir = np.random.choice([0, 1])
            elif x_samples[idx, 0] == Depth.shape[0]-1:
                x1_diff_dir = np.random.choice([-1, 0])
            else:
                x1_diff_dir = np.random.choice([-1, 0, 1])
            
            if x_samples[idx, 1] == 0:
                x2_diff_dir = np.random.choice([0, 1])
            elif x_samples[idx, 1] == Depth.shape[1]-1:
                x2_diff_dir = np.random.choice([-1, 0])
            else:
                x2_diff_dir = np.random.choice([-1, 0, 1])

            x_samples[j, 0] = x_samples[idx, 0] + x1_diff_dir
            x_samples[j, 1] = x_samples[idx, 1] + x2_diff_dir

    #true position diffusion step
    if x1_idx_true == 0:
        x1_diff_dir = np.random.choice([0, 1])
        x1_idx_true = x1_idx_true + x1_diff_dir
    elif x1_idx_true == Depth.shape[0]-1:
        x1_diff_dir = np.random.choice([-1, 0])
        x1_idx_true = x1_idx_true + x1_diff_dir
    else:
        x1_diff_dir = np.random.choice([-1, 0, 1])
        x1_idx_true = x1_idx_true + x1_diff_dir

    if x2_idx_true == 0:
        x2_diff_dir = np.random.choice([0, 1])
        x2_idx_true = x2_idx_true + x2_diff_dir
    elif x2_idx_true == Depth.shape[1]-1:
        x2_diff_dir = np.random.choice([-1, 0])
        x2_idx_true = x2_idx_true + x2_diff_dir
    else:
        x2_diff_dir = np.random.choice([-1, 0, 1])
        x2_idx_true = x2_idx_true + x2_diff_dir

    #take diffusion step for each particle
    for j in range(particles):
        if x_samples[j, 0] == 0 and x1_diff_dir == -1:
            x_samples[j, 0] = x_samples[j, 0] + np.random.choice([0, 1])
        elif x_samples[j, 0] == Depth.shape[0]-1 and x1_diff_dir == 1:
            x_samples[j, 0] = x_samples[j, 0] + np.random.choice([-1, 0])
        else:
            x_samples[j, 0] = x_samples[j, 0] + x1_diff_dir

        if x_samples[j, 1] == 0 and x2_diff_dir == -1:
            x_samples[j, 1] = x_samples[j, 1] + np.random.choice([0, 1])
        elif x_samples[j, 1] == Depth.shape[1]-1 and x2_diff_dir == 1:
            x_samples[j, 1] = x_samples[j, 1] + np.random.choice([-1, 0])
        else:
            x_samples[j, 1] = x_samples[j, 1] + x2_diff_dir

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


