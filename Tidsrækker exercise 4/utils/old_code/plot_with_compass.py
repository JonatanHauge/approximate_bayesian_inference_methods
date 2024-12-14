import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time
from utils.old_code.filter_with_compass import run_full_filter, initialize_filter

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

# Create the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
contour = ax.contour(X, Y, Depth.T, cmap='rainbow', levels=20)
particles_plot, = ax.plot(x_samples_array[0, :, 0], x_samples_array[0, :, 1], "b+", markersize=5, alpha=1, label="Particles")
mean_particles_plot, = ax.plot(np.mean(x_samples_array[0, :, 0]), np.mean(x_samples_array[0, :, 1]), "g>", markersize=10, alpha=1, label="Mean Particle Position")
true_position_plot, = ax.plot(x_true_array[0,0], x_true_array[0,1], "r>", markersize=10, alpha=1, label="True Position")
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
# set legend below plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Depth (m)')

# Update function for the animation
def update(frame):
    global x_true_array, x_samples_array, y_meas_array
    
    # Update true position and particles
    true_position_plot.set_data(x_true_array[frame+1, 0], x_true_array[frame+1, 1])
    particles_plot.set_data(x_samples_array[frame+1, :, 0], x_samples_array[frame+1, :, 1])
    mean_particles_plot.set_data(np.mean(x_samples_array[frame+1, :, 0]), np.mean(x_samples_array[frame+1, :, 1]))

    # Update the iteration text
    iteration_text.set_text(f'Iteration: {frame + 1}/{iterations}')

    # Return updated plots
    return true_position_plot, particles_plot, mean_particles_plot

# Create the animation
ani = FuncAnimation(fig, update, frames=iterations, blit=False, repeat=False)

# Save the animation as a GIF
ani.save(f"plots/with_compass/{dataset}_{particles}particles_{iterations}iterations.gif", writer=PillowWriter(fps=5))


print(f"Animation saved as 'with_compass/{dataset}_{particles}particles_{iterations}iterations.gif'.")
