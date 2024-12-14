import numpy as np
from tqdm import tqdm
from scipy.stats import norm, uniform
from scipy.stats import t as t_dist
from helpers_w_compass import (check_convergence, 
                     initialize_true_pos, 
                     initialize_particles, 
                     get_new_normalized_weights,
                     true_pos_diffusion_step,
                     resample_with_compass,
                     true_pos_harbour_step,
                     particle_diffusion_from_vec, 
                     get_harbour_unit_direction,
                     check_within_radius,
                     get_measurement)

def run_full_filter(Depth, habour_idx, num_particles = 500, num_iterations = 500, min_depth = -1, 
                    time_step = 1, convergence_radius = 10, convergence_ratio = 0.9, 
                    resample_threshold_ratio=0.95, meas_error = 0.15, measurement_dist_func = "norm", harbour_radius = 10):

    #arrays to store the results
    max_iter = 10000
    global_counter = 0
    particles_converged = False
    particles_array = np.zeros((max_iter, num_particles, 2))
    x_true_array = np.zeros((max_iter+1, 2))
    particles_mean_array = np.zeros((max_iter+1, 2))
    particle_dist_to_true_array = np.zeros(max_iter+1)
    particle_dist_to_mean_array = np.zeros(max_iter+1)
    particle_dist_to_mean_std_array = np.zeros(max_iter+1)
    N_eff_array = np.zeros(max_iter)
    dist_to_harbour_array = np.zeros(max_iter)
    conv_idx_list = []
    sail_home_idx_list = []

    true_pos = initialize_true_pos(Depth, min_depth)
    
    #95% confidence that the measurement is within 15% of the true value
    y_meas = get_measurement(measurement_dist_func, true_pos, meas_error, Depth)
    
    particles = initialize_particles(num_particles, Depth, y_meas, meas_error)
    wj_samples = np.ones(num_particles) / num_particles
    particles_mean = np.average(particles, weights=wj_samples, axis=0)

    #store initial values
    x_true_array[0, :] = true_pos
    particles_array[0, :, :] = particles
    particles_mean_array[0, :] = particles_mean
    particle_dist_to_true_array[0] = np.average(np.linalg.norm((particles - true_pos), axis=1), weights = wj_samples)
    particle_dist_to_mean_array[0] = np.average(np.linalg.norm((particles - particles_mean), axis=1), weights = wj_samples)
    particle_dist_to_mean_std_array[0] = np.sqrt(np.cov(np.linalg.norm((particles - particles_mean), axis = 1), aweights=wj_samples))
    dist_to_harbour_array[0] = np.linalg.norm(true_pos - habour_idx)


    while not particles_converged:
        conv_idx_list.append([global_counter, 0])
        for _ in tqdm(range(num_iterations), desc="Iterations"):
            if (global_counter+1) % 100 == 0:
                print(f"iteration: {global_counter+1}")
                print(f"true position: {true_pos}")
                print(f"mean position: {particles_mean}")
            #check convergence
            global_counter += 1
            if check_convergence(particles_mean, particles, wj_samples, convergence_radius, convergence_ratio):
                print(f"Converged after {global_counter} iterations.")
                print(f"true position: {true_pos}")
                print(f"mean position: {particles_mean}")
                particles_converged = True
                global_counter -= 1
                conv_idx_list[-1][1] = global_counter
                break

            y_meas = get_measurement(measurement_dist_func, true_pos, meas_error, Depth)

            wj_samples, N_eff = get_new_normalized_weights(wj_samples, measurement_dist_func, meas_error, 
                                                    y_meas, Depth, particles)

            #resample from the particles with positive probability
            if N_eff < resample_threshold_ratio*num_particles:
                particles, wj_samples = resample_with_compass(particles, wj_samples)
        
            #true position diffusion step
            true_pos = true_pos_diffusion_step(true_pos, Depth, min_depth, time_step)
            true_dir = true_pos - x_true_array[global_counter-1]

            #take diffusion step for each particle
            particles = particle_diffusion_from_vec(particles, true_dir, Depth, time_step)
            particles_mean = np.average(particles, weights=wj_samples, axis=0)
                
            #store values
            x_true_array[global_counter, :] = true_pos
            particles_array[global_counter, :, :] = particles
            particles_mean_array[global_counter, :] = particles_mean
            particle_dist_to_true_array[global_counter] = np.average(np.linalg.norm((particles - true_pos), axis=1), weights = wj_samples)
            particle_dist_to_mean_array[global_counter] = np.average(np.linalg.norm((particles - particles_mean), axis=1), weights = wj_samples)
            particle_dist_to_mean_std_array[global_counter] = np.sqrt(np.cov(np.linalg.norm((particles - particles_mean), axis = 1), aweights=wj_samples))
            N_eff_array[global_counter] = N_eff
            dist_to_harbour_array[global_counter] = np.linalg.norm(true_pos - habour_idx)

        
        harbour_iterations = 200
        sail_straight_to_harbour = True
        N_eff = num_particles
        sail_home_idx_list.append([global_counter, 0])
        mean_at_harbour_counter = 0
        for i in tqdm(range(harbour_iterations), desc="Iterations in sail home"):
            global_counter += 1
            if (global_counter+1) % 50 == 0:
                print(f"iteration: {global_counter+1}")
                print(f"true position: {true_pos}")
                print(f"mean position: {particles_mean}")

            #check if habour is within radius m
            if check_within_radius(true_pos, habour_idx, harbour_radius):
                print(f"Sailed home after {global_counter} iterations.")
                particles_converged = True
                global_counter -= 1
                sail_home_idx_list[-1][1] = global_counter
                break
            if check_within_radius(particles_mean, habour_idx, harbour_radius):
                mean_at_harbour_counter += 1

            if N_eff < 1.1: #assume that the mean is not close enough to the true position
                particles_converged = False
                global_counter -= 1
                particles = initialize_particles(num_particles, Depth, y_meas, meas_error)
                wj_samples = np.ones(num_particles) / num_particles
                particles_mean = np.average(particles, weights=wj_samples, axis=0)
                sail_home_idx_list[-1][1] = global_counter
                break

            y_meas = get_measurement(measurement_dist_func, true_pos, meas_error, Depth)

            #find direction to habour from mean particle position
            x1_diff_dir, x2_diff_dir, sail_straight_to_harbour, aim_point = get_harbour_unit_direction(habour_idx, particles_mean, Depth)
            step_vec = time_step*np.array([x1_diff_dir, x2_diff_dir])
            
            true_pos = true_pos_harbour_step(true_pos, step_vec, Depth) # add noise
            
            #take diffusion step for each particle in true direction
            particles = particle_diffusion_from_vec(particles, step_vec, Depth, 1e-6)
            wj_samples, N_eff = get_new_normalized_weights(wj_samples, measurement_dist_func, meas_error, 
                                                    y_meas, Depth, particles)
            
            if N_eff < resample_threshold_ratio*num_particles:
                particles, wj_samples = resample_with_compass(particles, wj_samples) #maybe add some noise to this resampling

            particles_mean = np.average(particles, weights=wj_samples, axis=0)
            if not sail_straight_to_harbour:
                if check_within_radius(particles_mean, aim_point, 10):
                    sail_straight_to_harbour = True
                    

            #store values
            x_true_array[global_counter, :] = true_pos
            particles_array[global_counter, :, :] = particles
            particles_mean_array[global_counter, :] = particles_mean
            particle_dist_to_true_array[global_counter] = np.average(np.linalg.norm((particles - true_pos), axis=1), weights = wj_samples)
            particle_dist_to_mean_array[global_counter] = np.average(np.linalg.norm((particles - particles_mean), axis=1), weights = wj_samples)
            particle_dist_to_mean_std_array[global_counter] = np.sqrt(np.cov(np.linalg.norm((particles - particles_mean), axis = 1), aweights=wj_samples))
            N_eff_array[global_counter] = N_eff
            dist_to_harbour_array[global_counter] = np.linalg.norm(true_pos - habour_idx)

            if i == harbour_iterations-1 or mean_at_harbour_counter > 10:
                sail_home_idx_list[-1][1] = global_counter
                particles_converged = False
                particles = initialize_particles(num_particles, Depth, y_meas, meas_error)
                wj_samples = np.ones(num_particles) / num_particles
                particles_mean = np.average(particles, weights=wj_samples, axis=0)
                break

    x_true_array = x_true_array[:global_counter+1, :]
    particles_array = particles_array[:global_counter+1, :, :]
    particles_mean_array = particles_mean_array[:global_counter+1, :]
    particle_dist_to_true_array = particle_dist_to_true_array[:global_counter+1]
    particle_dist_to_mean_array = particle_dist_to_mean_array[:global_counter+1]
    particle_dist_to_mean_std_array = particle_dist_to_mean_std_array[:global_counter+1]
    N_eff_array = N_eff_array[:global_counter+1]
    dist_to_harbour_array = dist_to_harbour_array[:global_counter+1]

    return (x_true_array, particles_array, particles_mean_array, particle_dist_to_true_array, 
            particle_dist_to_mean_array, N_eff_array, conv_idx_list, sail_home_idx_list, dist_to_harbour_array,
            particle_dist_to_mean_std_array)





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


