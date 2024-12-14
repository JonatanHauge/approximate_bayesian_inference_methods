import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from scipy.special import erfinv
from scipy.stats import norm, uniform
from scipy.stats import t as t_dist
from helpers import (check_convergence, 
                     pos2_depth, initialize_true_pos, 
                     initialize_particles, 
                     get_new_normalized_weights,
                     get_sigma,
                     particle_diffusion_wo_compass,
                     true_pos_diffusion_step,
                     resample_wo_compass,
                     true_pos_harbour_step,
                     get_particles_degrees, 
                     degrees_to_unit_vector,
                     particle_diffusion_from_vec,
                     dir2_degrees,
                     true_pos_search_step,
                     particle_search_diffusion_from_vec,
                     get_search_normalized_weights)

def run_full_filter(Depth, habour_idx, num_particles = 500, num_iterations = 500, min_depth = -1, 
                    time_step = 1, convergence_radius = 10, convergence_ratio = 0.9, 
                    resample_threshold_ratio=0.95, meas_error = 0.15, measurement_dist_func = norm):

    #arrays to store the results
    particles_array = np.zeros((num_iterations+1, num_particles, 2))
    x_true_array = np.zeros((num_iterations+1, 2))
    particles_mean_array = np.zeros((num_iterations+1, 2))
    particle_dist_to_true_array = np.zeros(num_iterations+1)
    particle_dist_to_mean_array = np.zeros(num_iterations+1)

    x1_true_pos, x2_true_pos = initialize_true_pos(Depth, min_depth)
    #x1_true_pos, x2_true_pos = 38, 50
    #95% confidence that the measurement is within 15% of the true value
    true_depth = pos2_depth([x1_true_pos, x2_true_pos], Depth)
    sigma = get_sigma(meas_error, x1_true_pos, x2_true_pos, Depth)
    y_meas = measurement_dist_func.rvs(loc=true_depth, scale = sigma, size = 1)
    y_belief = np.array([y_meas*1.2, y_meas*0.8]) #TODO: can be optimized
    
    particles = initialize_particles(num_particles, Depth, y_belief)
    wj_samples = np.ones(num_particles) / num_particles
    particles_mean = np.average(particles, weights=wj_samples, axis=0)

    #store initial values
    x_true_array[0, :] = [x1_true_pos, x2_true_pos]
    particles_array[0, :, :] = particles
    particles_mean_array[0, :] = particles_mean
    particle_dist_to_true_array[0] = np.linalg.norm(particles - [x1_true_pos, x2_true_pos], axis=1).mean()
    particle_dist_to_mean_array[0] = np.linalg.norm(particles - particles_mean, axis=1).mean()
    
    for iter in tqdm(range(num_iterations), desc="Iterations"):
        #check convergence
        if check_convergence(particles_mean, particles, convergence_radius, convergence_ratio):
            print(f"Converged after {iter} iterations.")
            print(f"true position: {x1_true_pos}, {x2_true_pos}")
            print(f"mean position: {particles_mean}")
            #cut off the rest of the arrays
            x_true_array = x_true_array[:iter+1, :]
            particles_array = particles_array[:iter+1, :, :]
            particles_mean_array = particles_mean_array[:iter+1, :]
            particle_dist_to_true_array = particle_dist_to_true_array[:iter+1]
            particle_dist_to_mean_array = particle_dist_to_mean_array[:iter+1]
            break

        true_depth = pos2_depth([x1_true_pos, x2_true_pos], Depth)
        sigma = get_sigma(meas_error, x1_true_pos, x2_true_pos, Depth)
        if sigma <= 0:
            sigma = 1e-10
        y_meas = measurement_dist_func.rvs(loc=true_depth, scale = sigma, size = 1)

        wj_samples, N_eff = get_new_normalized_weights(wj_samples, measurement_dist_func, meas_error, 
                                                y_meas, Depth, particles)

        #resample from the particles with positive probability
        if N_eff < resample_threshold_ratio*num_particles:
            particles, wj_samples = resample_wo_compass(particles, wj_samples)
    
        #true position diffusion step
        x1_true_pos, x2_true_pos = true_pos_diffusion_step(x1_true_pos, x2_true_pos, Depth, min_depth, time_step)

        #take diffusion step for each particle
        particles = particle_diffusion_wo_compass(particles, Depth, time_step)
        particles_mean = np.average(particles, weights=wj_samples, axis=0)
            
        #store values
        x_true_array[iter+1, :] = [x1_true_pos, x2_true_pos]
        particles_array[iter+1, :, :] = particles
        particles_mean_array[iter+1, :] = particles_mean
        particle_dist_to_true_array[iter+1] = np.linalg.norm(particles - [x1_true_pos, x2_true_pos], axis=1).mean()
        particle_dist_to_mean_array[iter+1] = np.linalg.norm(particles - particles_mean, axis=1).mean()


    #sail home to harbour
    harbour_iterations = 200 + iter
    #arrays to store the results
    particles_degrees = np.zeros(num_particles)
    particles_degree_array = np.zeros((harbour_iterations, num_particles))
    particles_array_2 = np.zeros((harbour_iterations, num_particles, 2))
    x_true_array_2 = np.zeros((harbour_iterations, 2))
    particles_mean_array_2 = np.zeros((harbour_iterations, 2))
    particle_dist_to_true_array_2 = np.zeros(harbour_iterations)
    particle_dist_to_mean_array_2 = np.zeros(harbour_iterations)
    particles_array_2[:iter+1] = particles_array
    x_true_array_2[:iter+1] = x_true_array
    particles_mean_array_2[:iter+1] = particles_mean_array
    particle_dist_to_true_array_2[:iter+1] = particle_dist_to_true_array
    particle_dist_to_mean_array_2[:iter+1] = particle_dist_to_mean_array
    particles_degrees = get_particles_degrees(particles_array[-1], particles_array[-2])
    particles_degree_array[0] = particles_degrees

    mean_dir = particles_mean_array[iter] - particles_mean_array[iter-1]
    mean_dir_degrees = dir2_degrees(mean_dir)
    true_dir = x_true_array[iter] - x_true_array[iter-1]
    true_dir_degrees = dir2_degrees(true_dir)

    search_step = 20
    num_dir = 360
    step_dir = 360/num_dir
    N_eff_list = np.ones(num_dir)
    y_search_meas = np.zeros(4)

    for j in range(4): #true boat direction
        search_vec_true = degrees_to_unit_vector((true_dir_degrees + j*90) % 360) * search_step
        x1_true_pos, x2_true_pos = true_pos_search_step(x1_true_pos, x2_true_pos, search_vec_true, Depth)
        sigma = get_sigma(meas_error, x1_true_pos, x2_true_pos, Depth)
        y_meas = measurement_dist_func.rvs(loc=pos2_depth([x1_true_pos, x2_true_pos], Depth), scale = sigma, size = 1)
        y_search_meas[j] = y_meas


    #for i in range(num_dir):
    ##    w_tmp_samples = np.ones(particles.shape[0]) / particles.shape[0]
    ##    for j in range(4): 
     #       search_vec_mean = degrees_to_unit_vector((mean_dir_degrees+i*step_dir + j*90)%360) * search_step
     #       particles = particle_search_diffusion_from_vec(particles, search_vec_mean, Depth)
     #       w_tmp_samples, N_eff_tmp = get_new_normalized_weights(w_tmp_samples, measurement_dist_func, meas_error,
     #                                                  y_search_meas[j], Depth, particles)
     #   N_eff_list[i] = N_eff_tmp

    likelihood_list = np.zeros(num_dir)
    for i in range(num_dir):
        for j in range(4):
            search_vec_mean = degrees_to_unit_vector((mean_dir_degrees+i*step_dir + j*90)%360) * search_step
            particles_mean += search_vec_mean
            sigma = get_sigma(meas_error, particles_mean[0], particles_mean[1], Depth)
            if sigma <= 0:
                sigma = 1e-6
            likelihood_list[i] += np.log(measurement_dist_func.pdf(y_search_meas[j], pos2_depth(particles_mean, Depth), sigma))

    print("10 largest likelihoods", np.sort(likelihood_list)[-10:])
    print("Arguments of 10 largest likelihoods", np.argsort(likelihood_list)[-10:])

    #argmax_idx = np.argmax(N_eff_list)
    argmax_idx = np.argmax(likelihood_list)
    particles_degrees = (mean_dir_degrees+argmax_idx*step_dir)%360

    x_true_array_2[iter+1] = [x1_true_pos, x2_true_pos]
    particles_array_2[iter+1, :, :] = particles
    particles_mean_array_2[iter+1, :] = particles_mean
    particle_dist_to_true_array_2[iter+1] = np.linalg.norm(particles - [x1_true_pos, x2_true_pos], axis=1).mean()
    particle_dist_to_mean_array_2[iter+1] = np.linalg.norm(particles - particles_mean, axis=1).mean()
    particles_degree_array[iter+1] = particles_degrees

    mean_dir = degrees_to_unit_vector(particles_degrees)
    mean_dir_degrees = dir2_degrees(mean_dir)
    true_dir = x_true_array_2[iter] - x_true_array_2[iter-1]
    true_dir_degrees = dir2_degrees(true_dir)

    print("mean dir: ", mean_dir_degrees)
    print("true dir: ", true_dir_degrees)
    

    for h_iter in tqdm(range(iter+2, harbour_iterations), desc="Iterations in sail home"):

        #check if habour is within 5 m
        if (x1_true_pos - habour_idx[0])**2 + (x2_true_pos - habour_idx[1])**2 < 10**2:
            print(f"Sailed home after {h_iter} iterations.")
            #cut off the rest of the arrays
            x_true_array_2 = x_true_array_2[:h_iter+1, :]
            particles_array_2 = particles_array_2[:h_iter+1, :, :]
            particles_mean_array_2 = particles_mean_array_2[:h_iter+1, :]
            particle_dist_to_true_array_2 = particle_dist_to_true_array_2[:h_iter+1]
            particle_dist_to_mean_array_2 = particle_dist_to_mean_array_2[:h_iter+1]
            break

        true_depth = pos2_depth([x1_true_pos, x2_true_pos], Depth)
        sigma = get_sigma(meas_error, x1_true_pos, x2_true_pos, Depth)
        if sigma <= 0:
            sigma = 1e-10
        y_meas = measurement_dist_func.rvs(loc=true_depth, scale = sigma, size = 1)

        wj_samples, N_eff = get_new_normalized_weights(wj_samples, measurement_dist_func, meas_error, 
                                                y_meas, Depth, particles)
        if np.sum(np.isnan(wj_samples)) > 0:
            print("N_eff is nan")
            x_true_array_2 = x_true_array_2[:h_iter+1, :]
            particles_array_2 = particles_array_2[:h_iter+1, :, :]
            particles_mean_array_2 = particles_mean_array_2[:h_iter+1, :]
            particle_dist_to_true_array_2 = particle_dist_to_true_array_2[:h_iter+1]
            particle_dist_to_mean_array_2 = particle_dist_to_mean_array_2[:h_iter+1]
            break
        
        if N_eff < resample_threshold_ratio*num_particles:
            particles, wj_samples = resample_wo_compass(particles, wj_samples) #maybe add some noise to this resampling
        

        #find direction to habour from mean particle position
        x1_diff_dir = habour_idx[0] - particles_mean[0]
        x2_diff_dir = habour_idx[1] - particles_mean[1]
        l = (x1_diff_dir**2 + x2_diff_dir**2)**0.5
        x1_diff_dir /= l
        x2_diff_dir /= l

        #find out how many degrees to turn to get to the right direction to the habour
        diff_dir_degrees = np.arctan2(x2_diff_dir, x1_diff_dir) * 180 / np.pi
        diff_degrees = (diff_dir_degrees - mean_dir_degrees) % 360

        #take diffusion step for true position, by turning true_dir_degrees and sailing timestep steps in that direction
        true_dir_degrees = (true_dir_degrees + diff_degrees) % 360
        true_dir_vec = time_step * degrees_to_unit_vector(true_dir_degrees)
        print("mean dir: ", diff_dir_degrees%360)
        print("true dir: ", true_dir_degrees)
        try:
            x1_true_pos, x2_true_pos = true_pos_harbour_step(x1_true_pos, x2_true_pos, true_dir_vec, Depth) # add noise
        except Exception as e:
            print(f"Error: {e}")

        #take diffusion step for each particle in true direction \pm 45 degrees (uniformly distributed)
        particles = particle_diffusion_from_vec(particles, time_step*np.array([x1_diff_dir, x2_diff_dir]), Depth, time_step)
        particles_degrees = get_particles_degrees(particles, particles_array_2[h_iter-1]) #(particles_degrees + diff_degrees) % 360
        particles_mean = np.average(particles, weights=wj_samples, axis=0)
        mean_dir_degrees = np.arctan2(particles_mean[1] - particles_mean_array_2[h_iter-1, 1], particles_mean[0] - particles_mean_array_2[h_iter-1, 0]) * 180 / np.pi % 360
        

        #store values
        x_true_array_2[h_iter, :] = [x1_true_pos, x2_true_pos]
        particles_array_2[h_iter, :, :] = particles
        particles_mean_array_2[h_iter, :] = particles_mean
        particle_dist_to_true_array_2[h_iter] = np.linalg.norm(particles - [x1_true_pos, x2_true_pos], axis=1).mean()
        particle_dist_to_mean_array_2[h_iter] = np.linalg.norm(particles - particles_mean, axis=1).mean()
        particles_degree_array[h_iter] = particles_degrees


    return x_true_array_2, particles_array_2, particles_mean_array_2, particle_dist_to_true_array_2, particle_dist_to_mean_array_2









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


