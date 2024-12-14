import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from filter3 import run_full_filter

dataset = "Ringsjon" #choose from "Vombsjon", "Oresund", "Ringsjon", "Skagerack", "Bolmen"
num_particles = 2500
num_iterations = 1000
min_depth = -1
time_step = 4
if dataset in ["Skagerack", "Oresund"]:
    convergence_radius = 40
    harbour_radius = 20
    convergence_ratio = 0.50
else:
    convergence_radius = 20
    harbour_radius = 10
    convergence_ratio = 0.50
resample_threshold_ratio = 0.90
meas_error = 0.05
measurement_dist_func = "norm" # choose from "norm", "uniform", 

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
 change_boat_dir_idx_list,
 mean_dir_list, 
 true_dir_list,
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
    if particles_mean_array[frame+1, 0] == particles_mean_array[frame, 0]:
        iteration_text.set_text(f'**Estimating direction**')
    else:
        iteration_text.set_text(f'Iteration: {frame + 1}/{conv_iterations+1}')

    # Return updated plots
    return true_position_plot, particles_plot, mean_particles_plot

# Create the animation
ani = FuncAnimation(fig, update, frames=conv_iterations, blit=False, repeat=False)

# Save the animation as a GIF
ani.save(f"plots/without_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.gif", writer=PillowWriter(fps=5))

print(f"Animation saved as 'without_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.gif'.")

# Plot the final state
fig.savefig(f"plots/last_iter/without_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.png")

fig, ax = plt.subplots(2, 1, figsize=(15, 8))

#add distance between the plots to make room for legend box
fig.subplots_adjust(hspace=0.5)

ax[0].plot(particle_dist_to_true_array, label="Distance to True Position", color = "red")
ax[0].plot(particle_dist_to_mean_array, label="Distance to Mean Position", color = "green")
ax[0].fill_between(np.arange(len(particle_dist_to_mean_array)), particle_dist_to_mean_array + particle_dist_to_mean_std_array, particle_dist_to_mean_array - particle_dist_to_mean_std_array, 
                   color = "green", alpha = 0.25, label="Std of Distance to Mean Position")

ax[0].plot(dist_to_harbour_array, label="Distance to Harbour", color = "blue", alpha = 0.5)
#ax.plot(N_eff_array/10, label="Effective Number of Particles (divided by 10)")
for i, (start, end) in enumerate(conv_idx_list):
    ax[0].axvspan(start, end, color='lightgrey', alpha=0.2, label="Filtering" if i == 0 else None)
for i, (start, end) in enumerate(sail_home_idx_list):
    ax[0].axvspan(start, end, color='grey', alpha=0.8, label="Sailing Home" if i == 0 else None)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Distance")
#make legend under x axis
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
ax[0].set_title("Distance to True, Mean Position and Harbour")




diff_dir_list = ((np.array(mean_dir_list) - np.array(true_dir_list)) % 360)
diff_dir_list = [min(diff, 360-diff) for diff in diff_dir_list]
ax[1].plot(diff_dir_list, color='blue', alpha = 1, label="Difference in Direction" if i == 0 else None)
for i, (start, end) in enumerate(conv_idx_list):
    ax[1].axvspan(start, end, color='lightgrey', alpha=0.2, label="Filtering" if i == 0 else None)
for i, (start, end) in enumerate(sail_home_idx_list):
    ax[1].axvspan(start, end, color='grey', alpha=0.8, label="Sailing Home" if i == 0 else None)


ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Difference in Direction")
#make legend under x axis
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
ax[1].set_title("Difference in Direction, Mean Particle Direction and True Direction")
ax[1].set_ylim(0, 180)

fig.savefig(f"plots/metric_plots/without_compass/{dataset}_{num_particles}particles_{conv_iterations}iterations.png")

print()



"""
ax[1].plot(mean_dir_list, color='green', label="Mean Particle Direction" if i == 0 else None)
ax[1].plot(true_dir_list, color='red', label="True Direction" if i == 0 else None)

for i in range(len(true_dir_list)):
    if i == 0:
        ax[1].axvline(x=i, ymin = 0, ymax = 1, color='black', linewidth = 0.5, linestyle='--', alpha = 0.8, label="Change Boat Direction" if i == 0 else None)

    elif i < len(true_dir_list)-1:
        if true_dir_list[i] != true_dir_list[i+1] and true_dir_list[i] != np.nan and true_dir_list[i+1] != np.nan:
            ax[1].axvline(x=i, ymin = 0, ymax = 1, color='black', linewidth = 0.5, linestyle='--', alpha = 0.8, label="Change Boat Direction" if i == 0 else None)
        elif true_dir_list[i-1] != true_dir_list[i] and true_dir_list[i-1] != np.nan and true_dir_list[i] != np.nan:
            ax[1].axvline(x=i, ymin = 0, ymax = 1, color='black', linewidth = 0.5, linestyle='--', alpha = 0.8, label="Change Boat Direction" if i == 0 else None)
"""
