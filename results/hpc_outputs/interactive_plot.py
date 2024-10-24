import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk



# Define the variables
prior_sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
low_ranks = [0, 1, 2, 5, 10]
K_list = [10, 100, 1000]
color_list = ["lightsteelblue", "royalblue", "darkblue"]

# Function to create the plot
def plot_graph(dataset, type_plot, metric_to_plot):
    # Load the data based on the selected dataset
    path = Path(f"{dataset}_SWA_{type_plot}.csv")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.replace(" ", "")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, K in enumerate(K_list):
        # Get column indices for the current K
        idxs = [index for index, name in enumerate(df.columns) if f"K={K}," in name]
        # Extract value of T
        T = re.search(r'T=(\d+)', df.columns[idxs[0]]).group(1)
        df_K = df.iloc[:, idxs]
        df_metric = df_K.loc[metric_to_plot, :]

        # Plot based on type_plot
        if type_plot[:5] == "prior":
            ax.plot(prior_sigmas, df_metric.values, marker='o', label=f"K={K}, T={T}", color=color_list[i])
            ax.set_xscale('log')
            ax.set_xlabel("Prior variance")
            ax.set_title(f"{dataset} \n {metric_to_plot} vs prior variance")
        elif type_plot[:7] == "lowrank":
            ax.plot(low_ranks, df_metric.values, marker='o', label=f"K={K}, T={T}", color=color_list[i])
            ax.set_xlabel("rank")
            ax.set_title(f"{dataset} \n {metric_to_plot} vs rank")

        ax.set_ylabel(metric_to_plot)
        ax.legend()

    plt.show()

# Define the Tkinter GUI for dropdowns
def run_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Interactive Plot")

    # Increase the size of dropdowns
    style = ttk.Style()
    style.configure('TCombobox', font=('Arial', 12), padding=5)  # Larger font and padding

    # Dropdown for dataset
    tk.Label(root, text="Dataset:", font=('Arial', 12)).grid(row=0, column=0)
    dataset_var = tk.StringVar()
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, style='TCombobox')
    dataset_dropdown['values'] = ["mnist", "fashion_mnist"]
    dataset_dropdown.grid(row=0, column=1, padx=10, pady=5)
    dataset_dropdown.current(1)  # Default to fashion_mnist

    # Dropdown for type_plot
    tk.Label(root, text="Type Plot:", font=('Arial', 12)).grid(row=1, column=0)
    type_plot_var = tk.StringVar()
    type_plot_dropdown = ttk.Combobox(root, textvariable=type_plot_var, style='TCombobox')
    type_plot_dropdown['values'] = ["prior_variance_T_1", "prior_variance_T_vary", "lowrank"]
    type_plot_dropdown.grid(row=1, column=1, padx=10, pady=5)
    type_plot_dropdown.current(0)

    # Dropdown for metric_to_plot
    tk.Label(root, text="Metric:", font=('Arial', 12)).grid(row=2, column=0)
    metric_to_plot_var = tk.StringVar()
    metric_dropdown = ttk.Combobox(root, textvariable=metric_to_plot_var, style='TCombobox')
    metric_dropdown['values'] = ["Accuracy", "Entropy", "LPD", "ECE", "MCE", "OOD"]
    metric_dropdown.grid(row=2, column=1, padx=10, pady=5)
    metric_dropdown.current(2)  # Default to LPD

    # Button to trigger plot
    def on_plot():
        dataset = dataset_var.get()
        type_plot = type_plot_var.get()
        metric_to_plot = metric_to_plot_var.get()
        plot_graph(dataset, type_plot, metric_to_plot)

    plot_button = tk.Button(root, text="Plot", font=('Arial', 12), command=on_plot)
    plot_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Run the GUI event loop
    root.mainloop()

# Run the GUI application
if __name__ == "__main__":
    run_gui()
