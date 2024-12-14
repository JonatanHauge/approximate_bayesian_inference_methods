import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from filter_w_compass import run_full_filter


dataset = "Skagerack" #choose from "Vombsjon", "Oresund", "Ringsjon", "Skagerack", "Bolmen"
num_particles = 1000
num_iterations = 1000
min_depth = -1
time_step = 4
if dataset in ["Skagerack", "Oresund"]:
    convergence_radius = 20
    harbour_radius = 10
    convergence_ratio = 0.95
else:
    convergence_radius = 10
    harbour_radius = 5
    convergence_ratio = 0.95
resample_threshold_ratio = 0.90
meas_error = 0.15
measurement_dist_func = "uniform" # choose from "norm", "uniform",

mat = scipy.io.loadmat(f"data/{dataset}.mat")
Depth = mat["Depth"].T
x = np.arange(0, Depth.shape[0])
y = np.arange(0, Depth.shape[1])
X, Y = np.meshgrid(x, y)
harbour_idx = mat["harbour"][0]

(x_true_array, 
particles_array, 
particles_mean_array, 
particle_dist_to_true_array, 
particle_dist_to_mean_array, 
N_eff_array, 
conv_idx_list, 
sail_home_idx_list, 
dist_to_harbour_array,
particle_dist_to_mean_std_array) = run_full_filter(Depth, harbour_idx, num_particles, num_iterations, min_depth, 
                                                time_step, convergence_radius, convergence_ratio, resample_threshold_ratio, 
                                                meas_error, measurement_dist_func, harbour_radius)

conv_iterations = len(x_true_array)-1

# Create the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
contour = ax.contour(X, Y, Depth.T, cmap='rainbow', levels=20)
particles_plot, = ax.plot(particles_array[0, :, 0], particles_array[0, :, 1], "b+", markersize=5, alpha=1, label="Particles")
mean_particles_plot, = ax.plot(particles_mean_array[0, 0], particles_mean_array[0, 1], "g>", markersize=10, alpha=1, label="Mean Particle Position")
true_position_plot, = ax.plot(x_true_array[0,0], x_true_array[0,1], "r>", markersize=10, alpha=1, label="True Position")
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
harbour_plot = ax.plot(harbour_idx[0], harbour_idx[1], "y*", markersize=15, alpha=1, label="Harbour")
# set legend below plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Depth (m)')

# Update function for the animation
def update(frame):
    global x_true_array, particles_array, particles_mean_array
    
    # Update true position and particles
    true_position_plot.set_data(x_true_array[frame+1, 0], x_true_array[frame+1, 1])
    particles_plot.set_data(particles_array[frame+1, :, 0], particles_array[frame+1, :, 1])
    mean_particles_plot.set_data(particles_mean_array[frame+1, 0], particles_mean_array[frame+1, 1])

    # Update the iteration text
    iteration_text.set_text(f'Iteration: {frame + 1}/{conv_iterations+1}')

    # Return updated plots
    return true_position_plot, particles_plot, mean_particles_plot

# Create the animation
ani = FuncAnimation(fig, update, frames=conv_iterations, blit=False, repeat=False)

# Save the animation as a GIF
ani.save(f"plots/with_compass_gaussian/{dataset}_{num_particles}particles_{conv_iterations}iterations.gif", writer=PillowWriter(fps=5))

print(f"Animation saved as 'with_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.gif'.")

# Plot the final state
fig.savefig(f"plots/last_iter/with_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.png")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

#Move plot a bit up to make room for legend below
fig.subplots_adjust(top=0.9)

ax.plot(particle_dist_to_true_array, label="Distance to True Position", color = "red")
ax.plot(particle_dist_to_mean_array, label="Distance to Mean Position", color = "green")
ax.fill_between(np.arange(len(particle_dist_to_mean_array)), particle_dist_to_mean_array + particle_dist_to_mean_std_array, particle_dist_to_mean_array - particle_dist_to_mean_std_array, 
                color = "green", alpha = 0.25)

ax.plot(dist_to_harbour_array, label="Distance to Harbour", color = "blue", alpha = 0.5)
#ax.plot(N_eff_array/10, label="Effective Number of Particles (divided by 10)")
for i, (start, end) in enumerate(conv_idx_list):
    ax.axvspan(start, end, color='lightgrey', alpha=0.2, label="Filtering" if i == 0 else None)
for i, (start, end) in enumerate(sail_home_idx_list):
    ax.axvspan(start, end, color='grey', alpha=0.8, label="Sailing Home" if i == 0 else None)
ax.set_xlabel("Iteration")
ax.set_ylabel("Distance")
#make legend under x axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), shadow=True, ncol=3)
#ax.legend()
ax.set_title("Distance to True and Mean Position Over Iterations")


fig.savefig(f"plots/metric_plots/with_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.png")

print()
