import numpy as np
from scipy.special import erfinv
from scipy.stats import norm, uniform

def check_convergence(particles_mean, particles, wj_samples, radius=10, ratio=0.9):
    """
    Check if the particles have converged to a small area weighted by wj_samples
    """
    total_weight = np.dot(wj_samples, np.sum((particles - particles_mean)**2, axis=1))
    weight_within_radius = 0
    for j in range(particles.shape[0]):
        if (np.sum((particles_mean - particles[j,:])**2) < radius**2):
            weight_within_radius += wj_samples[j] * np.sum((particles_mean - particles[j,:])**2)

    return weight_within_radius / total_weight > ratio
    
    #return (np.sum((particles_mean[0] - particles[:,0])**2 + (particles_mean[1] - particles[:,1])**2 < radius**2) / particles.shape[0]) > ratio


def pos2_depth(pos, Depth):
    """
    Convert position to depth.
    """
    pos1_idx = int(min(max(pos[0], 0), Depth.shape[0]-1)+0.5)
    pos2_idx = int(min(max(pos[1], 0), Depth.shape[1]-1)+0.5)
    return Depth[pos1_idx, pos2_idx]

def check_position(pos, Depth):
    """
    Check if the position is within the bounds of the Depth map.
    """
    return 0 <= pos[0] < Depth.shape[0] and 0 <= pos[1] < Depth.shape[1]


def initialize_true_pos(Depth, min_depth=-1):
    """
    Draw initial true position.
    """
    while True:
        x1_true = np.random.uniform(0, Depth.shape[0])
        x2_true = np.random.uniform(0, Depth.shape[1])

        if pos2_depth([x1_true, x2_true], Depth) < min_depth:
            break
    true_pos = np.array([x1_true, x2_true])
    return true_pos


def initialize_particles(num_particles, Depth, y_meas, meas_error):

    y_belief = np.array([y_meas*(1+meas_error*1.3), y_meas*(1-meas_error*1.3)]) 
    particles = np.zeros((num_particles, 2))
    for j in range(num_particles):
        while True:
            x1_pos = np.random.uniform(0, Depth.shape[0])
            x2_pos = np.random.uniform(0, Depth.shape[1])
            if pos2_depth([x1_pos, x2_pos], Depth) < y_belief[1] and pos2_depth([x1_pos, x2_pos], Depth) > y_belief[0]:
                break
        particles[j, 0] = x1_pos
        particles[j, 1] = x2_pos

    return particles


def get_new_normalized_weights(wj_samples, measurement_dist_func, meas_error, y_meas, Depth, particles):
    for j in range(len(wj_samples)):
        wj_samples[j] *= get_pdf(y_meas, measurement_dist_func, particles[j,:], meas_error, Depth)
        
    wj_samples /= np.sum(wj_samples) #normalize weights
    N_eff = 1 / np.sum(wj_samples**2)
    if np.sum(np.isnan(wj_samples)) > 0 or np.sum(np.isnan(N_eff)) > 0:
        print("NAN in wj_samples or N_eff")
        wj_samples = np.ones(particles.shape[0]) / particles.shape[0]
        N_eff = 1 / np.sum(wj_samples**2)
    return wj_samples, N_eff

def add_within_boundary(pos1, pos2, diff_dir_1, diff_dir2, Depth):
    pos1 = min(max(pos1 + diff_dir_1, 0), Depth.shape[0]-1)
    pos2 = min(max(pos2 + diff_dir2, 0), Depth.shape[1]-1)
    return pos1, pos2

def get_sigma(meas_error, true_pos, Depth):
    sigma = meas_error*pos2_depth(true_pos, Depth) / (np.sqrt(2)*erfinv(-0.99))
    if sigma <= 0:
        sigma = 1e-6
    return sigma

def particle_diffusion_wo_compass(particles, Depth, time_step):
    #take diffusion step for each particle
    for j in range(len(particles)):
        x1_diff_dir, x2_diff_dir = np.random.normal(0, np.sqrt(time_step), 2)
        particles[j, :] = add_within_boundary(particles[j, 0], particles[j, 1], x1_diff_dir, x2_diff_dir, Depth)

    return particles

def true_pos_diffusion_step(true_pos, Depth, min_depth, time_step):
    x1_true_pos, x2_true_pos = true_pos[0], true_pos[1]
    while True:
        x1_diff_dir, x2_diff_dir = np.random.normal(0, np.sqrt(time_step), 2)
        if (pos2_depth([x1_true_pos + x1_diff_dir, x2_true_pos + x2_diff_dir], Depth) < min_depth):
            x1_true_pos, x2_true_pos = add_within_boundary(x1_true_pos, x2_true_pos, x1_diff_dir, x2_diff_dir, Depth)
            break
    true_pos = np.array([x1_true_pos, x2_true_pos])
    return true_pos

def resample_wo_compass(particles, wj_samples):
    num_particles = particles.shape[0]
    idx = np.random.choice(range(0, num_particles), size=num_particles, p=wj_samples/np.sum(wj_samples))
    particles = particles[idx, :]
    wj_samples = np.ones(num_particles) / num_particles
    return particles, wj_samples


def true_pos_harbour_step(true_pos, true_dir_vec, Depth):
    x1_true_pos, x2_true_pos = true_pos[0], true_pos[1]
    x1_diff_dir, x2_diff_dir = true_dir_vec[0], true_dir_vec[1]
    if (pos2_depth([x1_true_pos + x1_diff_dir, x2_true_pos + x2_diff_dir], Depth) < 0):
        x1_true_pos, x2_true_pos = add_within_boundary(x1_true_pos, x2_true_pos, x1_diff_dir, x2_diff_dir, Depth)
        return np.array([x1_true_pos, x2_true_pos])
    
    x1_diff_dir, x2_diff_dir = -true_dir_vec[1], true_dir_vec[0]
    if (pos2_depth([x1_true_pos + x1_diff_dir, x2_true_pos + x2_diff_dir], Depth) < 0):
        #take step perpendicular to the gradient
        x1_true_pos, x2_true_pos = add_within_boundary(x1_true_pos, x2_true_pos, x1_diff_dir, x2_diff_dir, Depth)
        return np.array([x1_true_pos, x2_true_pos])
    
    x1_diff_dir, x2_diff_dir = true_dir_vec[1], -true_dir_vec[0]
    if (pos2_depth([x1_true_pos + x1_diff_dir, x2_true_pos + x2_diff_dir], Depth) < 0):
        #take step perpendicular to the gradient
        x1_true_pos, x2_true_pos = add_within_boundary(x1_true_pos, x2_true_pos, x1_diff_dir, x2_diff_dir, Depth)
        return np.array([x1_true_pos, x2_true_pos])
    
    x1_diff_dir, x2_diff_dir = -true_dir_vec[0], -true_dir_vec[1]
    if (pos2_depth([x1_true_pos + x1_diff_dir, x2_true_pos + x2_diff_dir], Depth) < 0):
        #take step backwards
        x1_true_pos, x2_true_pos = add_within_boundary(x1_true_pos, x2_true_pos, x1_diff_dir, x2_diff_dir, Depth)
        return np.array([x1_true_pos, x2_true_pos])
    
def particle_diffusion_from_vec(particles, vec, Depth, timestep):
    #take diffusion step for each particle
    for j in range(len(particles)):
        #randomly sample a direction from the vector plus some noise
        sample_vec = vec + np.random.normal(0, np.sqrt(timestep), 2)
        if pos2_depth([particles[j, 0] + sample_vec[0], particles[j, 1] + sample_vec[1]], Depth) < 0:
            particles[j, :] = add_within_boundary(particles[j, 0], particles[j, 1], sample_vec[0], sample_vec[1], Depth)
        elif pos2_depth([particles[j, 0] - sample_vec[1], particles[j, 1] + sample_vec[0]], Depth) < 0:
            particles[j, :] = add_within_boundary(particles[j, 0], particles[j, 1], -sample_vec[1], sample_vec[0], Depth)
        elif pos2_depth([particles[j, 0] + sample_vec[1], particles[j, 1] - sample_vec[0]], Depth) < 0:
            particles[j, :] = add_within_boundary(particles[j, 0], particles[j, 1], sample_vec[1], -sample_vec[0], Depth)
        elif pos2_depth([particles[j, 0] - sample_vec[0], particles[j, 1] - sample_vec[1]], Depth) < 0:
            particles[j, :] = add_within_boundary(particles[j, 0], particles[j, 1], -sample_vec[0], -sample_vec[1], Depth)
            

    return particles

def degrees_to_unit_vector(degrees):
    return np.array([np.cos(degrees*np.pi/180), np.sin(degrees*np.pi/180)]) / np.linalg.norm(np.array([np.cos(degrees*np.pi/180), np.sin(degrees*np.pi/180)]))

def get_particles_degrees(particles, previous_particles):
    degrees = np.zeros(particles.shape[0])
    for j in range(particles.shape[0]):
        degrees[j] = (np.arctan2(particles[j, 1] - previous_particles[j, 1], particles[j, 0] - previous_particles[j, 0]) * 180 / np.pi) % 360
    return degrees
    
def dir2_degrees(vec):
    return (np.arctan2(vec[1], vec[0]) * 180 / np.pi) % 360


def true_pos_search_step(true_pos, true_dir_vec, Depth):
    x1_true_pos, x2_true_pos = true_pos[0], true_pos[1]
    x1_diff_dir, x2_diff_dir = true_dir_vec[0], true_dir_vec[1]
    x1_true_pos, x2_true_pos = add_within_boundary(x1_true_pos, x2_true_pos, x1_diff_dir, x2_diff_dir, Depth)
    return np.array([x1_true_pos, x2_true_pos])


def particle_search_diffusion_from_vec(particles, search_vec_mean, Depth):
    #take diffusion step for each particle in serach direction for mean boat
    for j in range(len(particles)):
        particles[j, :] = add_within_boundary(particles[j, 0], particles[j, 1], search_vec_mean[0], search_vec_mean[1], Depth)

    return particles


def get_search_normalized_weights(measurement_dist_func, meas_error, y_meas, Depth, particles):
    wj_samples = np.ones(particles.shape[0]) / particles.shape[0]
    for j in range(len(wj_samples)):
        wj_samples[j] *= get_pdf(y_meas, measurement_dist_func, particles[j,:], meas_error, Depth)
        
    wj_samples /= np.sum(wj_samples) #normalize weights
    N_eff = 1 / np.sum(wj_samples**2)
    return wj_samples, N_eff


def estimate_angle_from_particle_mean(true_pos, particles_mean, true_dir_degrees, mean_dir_degrees, meas_error, measurement_dist_func, Depth):
    search_step = min(12, int(Depth.shape[0] - true_pos[0]), int(Depth.shape[1] - true_pos[0]))
    num_dir = 180
    step_dir = 360/num_dir
    y_search_meas = np.zeros(12)
    true_pos_list = np.zeros((12, 2))

    plus_angle = 0
    for j in range(12): #true boat direction
        if j == 0:
            plus_angle += 0
        elif j%3 != 0:
            plus_angle += 270
        else:
            plus_angle += 90
        search_vec_true = degrees_to_unit_vector((true_dir_degrees + plus_angle) % 360) * search_step
        true_pos = true_pos_search_step(true_pos, search_vec_true, Depth)
        y_meas = get_measurement(measurement_dist_func, true_pos, meas_error, Depth)
        y_search_meas[j] = y_meas
        true_pos_list[j, :] = true_pos


    likelihood_list = np.zeros(num_dir)
    for i in range(num_dir):
        plus_angle = 0
        for j in range(12):
            if j == 0:
                plus_angle += 0
            elif j%3 != 0:
                plus_angle += 270 #turn right
            else:
                plus_angle += 90 #turn left
            search_vec_mean = degrees_to_unit_vector((mean_dir_degrees+i*step_dir + plus_angle)%360) * search_step
            particles_mean += search_vec_mean
            likelihood_list[i] += get_logpdf(y_search_meas[j], measurement_dist_func, particles_mean, meas_error, Depth)
            

    argmax_idx = np.argmax(likelihood_list)
    angle_estimate = (mean_dir_degrees+argmax_idx*step_dir)%360
    return angle_estimate, true_pos_list

def get_harbour_unit_direction(habour_idx, particles_mean, Depth):
    #find direction to habour from mean particle position
    x1_diff_dir = habour_idx[0] - particles_mean[0]
    x2_diff_dir = habour_idx[1] - particles_mean[1]
    l = (x1_diff_dir**2 + x2_diff_dir**2)**0.5
    x1_diff_dir /= l
    x2_diff_dir /= l
    sail_straight_to_harbour = True
    aim_point = habour_idx
    #check if all grid points on the the line from particle mean to harbour is with Depth < 0
    grid_line_points = np.zeros((100, 2))
    perpendic_dist = 5
    for i in range(100):
        grid_line_points[i,:] = particles_mean + i*l/100 * np.array([x1_diff_dir, x2_diff_dir])
        if pos2_depth(grid_line_points[i,:], Depth) >= 0:
            sail_straight_to_harbour = False
            while True:
                if pos2_depth([grid_line_points[i,0] - perpendic_dist*x2_diff_dir, grid_line_points[i,1] + perpendic_dist*x1_diff_dir], Depth) < 0:
                    aim_point = np.array([grid_line_points[i,0] - perpendic_dist*x2_diff_dir, grid_line_points[i,1] + perpendic_dist*x1_diff_dir])
                    break
                elif pos2_depth([grid_line_points[i,0] + 10*x2_diff_dir, grid_line_points[i,1] - 10*x1_diff_dir], Depth) < 0:
                    aim_point = np.array([grid_line_points[i,0] + perpendic_dist*x2_diff_dir, grid_line_points[i,1] - perpendic_dist*x1_diff_dir])
                    break
                else:
                    perpendic_dist += 5
            break

    if not sail_straight_to_harbour:
        x1_diff_dir = aim_point[0] - particles_mean[0]
        x2_diff_dir = aim_point[1] - particles_mean[1]
        l = (x1_diff_dir**2 + x2_diff_dir**2)**0.5
        x1_diff_dir /= l
        x2_diff_dir /= l
    
    return x1_diff_dir, x2_diff_dir, sail_straight_to_harbour, aim_point


def check_within_radius(point, checkpoint, radius=5):
    return  (point[0] - checkpoint[0])**2 + (point[1] - checkpoint[1])**2 < radius**2
        

def get_measurement(dist_name, true_pos, meas_error, Depth):

    if dist_name == "norm":
        true_depth = pos2_depth(true_pos, Depth)
        sigma = get_sigma(meas_error, true_pos, Depth)
        return norm.rvs(loc=true_depth, scale = sigma, size = 1)
    elif dist_name == "uniform":
        true_depth = pos2_depth(true_pos, Depth)
        return uniform.rvs(loc = true_depth + meas_error*true_depth, scale = -2*meas_error*true_depth, size = 1)
    else:
        raise ValueError("Unknown distribution name")
    
def get_pdf(y_meas, dist_name, true_pos, meas_error, Depth):

    if dist_name == "norm":
        sigma = get_sigma(meas_error, true_pos, Depth)
        return norm.pdf(y_meas, loc=pos2_depth(true_pos, Depth), scale = sigma)
    elif dist_name == "uniform":
        return uniform.pdf(y_meas, loc = pos2_depth(true_pos, Depth) + meas_error*pos2_depth(true_pos, Depth), scale = -2*meas_error*pos2_depth(true_pos, Depth))
    else:
        raise ValueError("Unknown distribution name")
    
def get_logpdf(y_meas, dist_name, true_pos, meas_error, Depth):

    if dist_name == "norm":
        sigma = get_sigma(meas_error, true_pos, Depth)
        return norm.logpdf(y_meas, loc=pos2_depth(true_pos, Depth), scale = sigma)
    elif dist_name == "uniform":
        return uniform.logpdf(y_meas, loc = pos2_depth(true_pos, Depth) + meas_error*pos2_depth(true_pos, Depth), scale = -2*meas_error*pos2_depth(true_pos, Depth))
    else:
        raise ValueError("Unknown distribution name")









if __name__ == "__main__":
    true_dir_degrees = 0.123
    mean_dir_degrees = 180.123
    meas_error = 0.05
    from scipy.stats import norm
    import scipy.io
    dataset = "Vombsjon"
    measurement_dist_func = norm
    mat = scipy.io.loadmat(f"data/{dataset}.mat")
    Depth = mat["Depth"].T
    particle_mean = np.array([30., 30.])
    true_pos = [32., 33.]


    print(estimate_angle_from_particle_mean(true_pos, particle_mean, true_dir_degrees, mean_dir_degrees, 
                         meas_error, measurement_dist_func, Depth))