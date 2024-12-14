import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from scipy.special import erfinv
from scipy.stats import norm
from helpers import check_convergence

def run_full_filter(Depth, particles = 500, iterations = 500, min_depth = -1, time_step = 1, radius = 10, ratio = 0.9, threshold_ratio=0.95):

    #draw initial true position
    while True:
        x1_idx_true = np.random.randint(0, Depth.shape[0])
        x2_idx_true = np.random.randint(0, Depth.shape[1])
        if Depth[x1_idx_true, x2_idx_true] < min_depth:
            break

    y_meas_array = np.zeros(iterations+1)
    x_samples_array = np.zeros((iterations+1, particles, 2), dtype=int)
    x_true_array = np.zeros((iterations+1, 2))
    x_samples = np.zeros((particles, 2), dtype=int)
    wj_samples = np.ones(particles) / particles
    x_mean = np.zeros((iterations+1, 2))

    sigma = 0.15*Depth[x1_idx_true, x2_idx_true] / (np.sqrt(2)*erfinv(-0.95))
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
    x_mean[0, 0] = np.average(x_samples[:, 0], weights=wj_samples)
    x_mean[0, 1] = np.average(x_samples[:, 1], weights=wj_samples)
    

    for iter in tqdm(range(iterations), desc="Iterations"):

        #check convergence
        if check_convergence(x_samples, radius, ratio):
            print(f"Converged after {iter} iterations.")
            #cut off the rest of the arrays
            x_true_array = x_true_array[:iter+1, :]
            x_samples_array = x_samples_array[:iter+1, :, :]
            y_meas_array = y_meas_array[:iter+1]

            break

        sigma = 0.15*Depth[x1_idx_true, x2_idx_true] / (np.sqrt(2)*erfinv(-0.95))
        if sigma <= 0:
            sigma = 1e-10
        y_meas = np.random.normal(Depth[x1_idx_true, x2_idx_true], sigma)

        for j in range(particles):
            sigma = 0.15*Depth[x_samples[j, 0], x_samples[j, 1]] / (np.sqrt(2)*erfinv(-0.95))
            if sigma <= 0:
                sigma = 1e-10
            wj_samples[j] *= norm.pdf(y_meas, Depth[x_samples[j, 0], x_samples[j, 1]], sigma)
            
        wj_samples /= np.sum(wj_samples) #normalize weights
        N_eff = 1 / np.sum(wj_samples**2)

        #resample from the particles with positive probability
        if N_eff < threshold_ratio*particles:
            idx = np.random.choice(range(0,particles), size=particles, p=wj_samples/np.sum(wj_samples))
            x_samples = x_samples[idx, :]
            wj_samples = np.ones(particles) / particles

    
        #true position diffusion step
        while True:
            x1_diff_dir = np.random.normal(0, np.sqrt(time_step))
            x2_diff_dir = np.random.normal(0, np.sqrt(time_step))
            if Depth[int(min(max(x1_idx_true + x1_diff_dir, 0), Depth.shape[0]-1)+0.5), int(min(max(x2_idx_true + x2_diff_dir, 0), Depth.shape[1]-1)+0.5)] < min_depth:
                break

        x1_idx_true = int(min(max(x1_idx_true + x1_diff_dir, 0), Depth.shape[0]-1)+0.5)
        x2_idx_true = int(min(max(x2_idx_true + x2_diff_dir, 0), Depth.shape[1]-1)+0.5)

        #take diffusion step for each particle
        for j in range(particles):
            #x1
            x1_diff_dir = np.random.normal(0, np.sqrt(time_step))
            x_samples[j, 0] = int(min(max(x_samples[j, 0] + x1_diff_dir, 0), Depth.shape[0]-1)+0.5)
            
            x2_diff_dir = np.random.normal(0, np.sqrt(time_step))
            x_samples[j, 1] = int(min(max(x_samples[j, 1] + x2_diff_dir, 0), Depth.shape[1]-1)+0.5)
            
        #store values
        x_true_array[iter+1, 0] = x1_idx_true
        x_true_array[iter+1, 1] = x2_idx_true
        x_samples_array[iter+1, :, :] = x_samples
        y_meas_array[iter+1] = y_meas
        x_mean[iter+1, 0] = np.average(x_samples[:, 0], weights=wj_samples)
        x_mean[iter+1, 1] = np.average(x_samples[:, 1], weights=wj_samples)

    return x_true_array, x_samples_array, y_meas_array, x_mean, wj_samples


def sail_home_w_compass(Depth, habour_idx, particles, wj_samples, x1_idx_true, 
                        x2_idx_true, threshold_ratio = 0.95, speed = 1):

    iterations = 10000
    num_particles = particles.shape[0]
    y_meas_array = np.zeros(iterations+1)
    x_samples_array = np.zeros((iterations+1, num_particles, 2), dtype=int)
    x_true_array = np.zeros((iterations+1, 2), dtype=int)
    x_mean = np.zeros((iterations+1, 2))
    x_samples = particles

    #store initial values
    x_true_array[0, 0] = x1_idx_true
    x_true_array[0, 1] = x2_idx_true
    x_samples_array[0, :, :] = particles
    x_mean[0, 0] = np.average(particles[:, 0], weights=wj_samples)
    x_mean[0, 1] = np.average(particles[:, 1], weights=wj_samples)

    for iter in tqdm(range(iterations), desc="Iterations in sail home"):
        
        #check if habour is within 5 m
        if (x1_idx_true - habour_idx[0])**2 + (x2_idx_true - habour_idx[1])**2 < 5**2:
            print(f"Sailed home after {iter} iterations.")
            #cut off the rest of the arrays
            x_true_array = x_true_array[:iter+1, :]
            x_samples_array = x_samples_array[:iter+1, :, :]
            y_meas_array = y_meas_array[:iter+1]
            break

        sigma = 0.15*Depth[x1_idx_true, x2_idx_true] / (np.sqrt(2)*erfinv(-0.95))
        if sigma <= 0:
            sigma = 1e-10
        y_meas = np.random.normal(Depth[x1_idx_true, x2_idx_true], sigma)

        for j in range(num_particles):
            sigma = 0.15*Depth[x_samples[j, 0], x_samples[j, 1]] / (np.sqrt(2)*erfinv(-0.95))
            if sigma <= 0:
                sigma = 1e-10
            wj_samples[j] *= norm.pdf(y_meas, Depth[x_samples[j, 0], x_samples[j, 1]], sigma)
            
        wj_samples /= np.sum(wj_samples) #normalize weights
        N_eff = 1 / np.sum(wj_samples**2)

        #resample from the particles with positive probability
        if N_eff < threshold_ratio*num_particles:
            idx = np.random.choice(range(0,num_particles), size=num_particles, p=wj_samples/np.sum(wj_samples))
            #int(min(max(x_samples[j, 0] + x1_diff_dir, 0), Depth.shape[0]-1)+0.5)
            x_samples = x_samples[idx, :]
            wj_samples = np.ones(num_particles) / num_particles

        #find direction to habour from mean particle position
        x1_mean = np.average(x_samples[:, 0], weights=wj_samples)
        x2_mean = np.average(x_samples[:, 1], weights=wj_samples)

        x1_diff_dir = habour_idx[0] - x1_mean
        x2_diff_dir = habour_idx[1] - x2_mean
        l = (x1_diff_dir**2 + x2_diff_dir**2)**0.5
        x1_diff_dir /= l
        x2_diff_dir /= l

        #take diffusion step for true position
        x1_idx_true = int(min(max(x1_idx_true + speed*x1_diff_dir, 0), Depth.shape[0]-1)+0.5)
        x2_idx_true = int(min(max(x2_idx_true + speed*x2_diff_dir, 0), Depth.shape[1]-1)+0.5)

        #take diffusion step for each particle
        for j in range(num_particles):
            x_samples[j, 0] = int(min(max(x_samples[j, 0] + speed*x1_diff_dir, 0), Depth.shape[0]-1)+0.5)
            x_samples[j, 1] = int(min(max(x_samples[j, 1] + speed*x2_diff_dir, 0), Depth.shape[1]-1)+0.5)

        #store values
        x_true_array[iter+1, 0] = x1_idx_true
        x_true_array[iter+1, 1] = x2_idx_true
        x_samples_array[iter+1, :, :] = x_samples
        y_meas_array[iter+1] = y_meas
        x_mean[iter+1, 0] = x1_mean
        x_mean[iter+1, 1] = x2_mean

    return x_true_array, x_samples_array, y_meas_array, x_mean





if __name__ == "__main__":
    import numpy as np
    import scipy.io
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import time

    dataset = "Vombsjon" #choose from "Vombsjon", "Oresund", "Ringsjon", "Skagerack", "Bolmen"
    particles = 1000
    iterations = 500
    min_depth = -2

    mat = scipy.io.loadmat(f"data/{dataset}.mat")
    Depth = mat["Depth"].T
    x = np.arange(0, Depth.shape[0])
    y = np.arange(0, Depth.shape[1])
    X, Y = np.meshgrid(x, y)

    x_true_array, x_samples_array, y_meas_array = run_full_filter(Depth, particles, iterations, min_depth)


