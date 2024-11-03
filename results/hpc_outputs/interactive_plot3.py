import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk, Checkbutton, BooleanVar


# Define the variables
prior_sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
low_ranks = [0, 1, 2, 5, 10]
K_list = [10, 100, 1000]
color_list = ["lightsteelblue", "royalblue", "darkblue"]
color_list2 = ["pink", "red", "darkred"]

# Function to create the plot
def plot_graph(ax, dataset, type_plot, metrics_to_plot):
    # Load the data based on the selected dataset
    path = Path(f"{dataset}_SWA_{type_plot}.csv")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.replace(" ", "")

    path2 = Path(f"{dataset}_SWA_prior_variance_T_vary.csv")
    df2 = pd.read_csv(path2, index_col=0)
    df2.index = df2.index.str.replace(" ", "")

    # Baselines for each dataset
    fm_baseline = {"Accuracy": 0.9055, "Entropy": 0.169, "LPD": -0.289, "ECE": 0.031, "MCE": 0.271, "OOD": 0.375}
    m_baseline = {"Accuracy": 0.989, "Entropy": 0.023, "LPD": -0.031, "ECE": 0.004, "MCE": 0.189, "OOD": 1.217}
    dataset_baseline = {"mnist": m_baseline, "fashion_mnist": fm_baseline}

    # Clear the current axis
    ax.clear()

    # Plot each metric selected
    for metric_to_plot in metrics_to_plot:
        for i, K in enumerate(K_list):
            # Get column indices for the current K
            idxs = [index for index, name in enumerate(df.columns) if f"K={K}," in name]
            idxs2 = [index for index, name in enumerate(df2.columns) if f"K={K}," in name]
            # Extract value of T
            T = re.search(r'T=(\d+)', df.columns[idxs[0]]).group(1)
            T2 = re.search(r'T=(\d+)', df2.columns[idxs2[0]]).group(1)
            df_K = df.iloc[:, idxs]
            df_K2 = df2.iloc[:, idxs2]
            df_metric = df_K.loc[metric_to_plot, :]
            df_metric2 = df_K2.loc[metric_to_plot, :]
            if type_plot == "prior_variance_T_1":
                ax.plot(prior_sigmas, df_metric.values, marker='o', label=f"{metric_to_plot} K={K}, T={T}", color=color_list[i])
                ax.plot(prior_sigmas, df_metric2.values, marker='o', label=f"{metric_to_plot} K={K}, T={T2}", color=color_list2[i])

                ax.set_xscale('log')
                ax.set_xlabel("Prior variance")
                ax.set_title(f"{dataset} \n vs prior variance")
            elif type_plot[:7] == "lowrank":
                ax.plot(low_ranks, df_metric.values, marker='o', label=f"{metric_to_plot} K={K}, T={T}", color=color_list[i])
                ax.set_xlabel("rank")
                ax.set_title(f"{dataset} \n vs rank")

            ax.set_ylabel("Metrics")
            ax.hlines(dataset_baseline[dataset][metric_to_plot], 1e-6, 10, color='black', linestyle='--', label=f"{dataset} baseline")
    
    ax.legend()
    plt.draw()  # Update the figure


# Define the Tkinter GUI for checkboxes
def run_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Interactive Plot")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()  # Turn on interactive mode
    fig.show()

    # Define a list to hold checkboxes
    metric_vars = []
    
    # Checkbox for datasets
    tk.Label(root, text="Dataset:", font=('Arial', 12)).grid(row=0, column=0)
    dataset_var = tk.StringVar(value="fashion_mnist")
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, values=["mnist", "fashion_mnist"], font=('Arial', 12))
    dataset_dropdown.grid(row=0, column=1, padx=10, pady=5)

    # Checkbox for plot type
    tk.Label(root, text="Type Plot:", font=('Arial', 12)).grid(row=1, column=0)
    type_plot_var = tk.StringVar(value="prior_variance_T_1")
    type_plot_dropdown = ttk.Combobox(root, textvariable=type_plot_var, values=["prior_variance_T_1", "prior_variance_T_vary", "lowrank"], font=('Arial', 12))
    type_plot_dropdown.grid(row=1, column=1, padx=10, pady=5)

    # Add checkboxes for metrics
    tk.Label(root, text="Metrics:", font=('Arial', 12)).grid(row=2, column=0)
    metrics = ["Accuracy", "Entropy", "LPD", "ECE", "MCE", "OOD"]
    for i, metric in enumerate(metrics):
        var = BooleanVar()
        checkbox = Checkbutton(root, text=metric, variable=var, font=('Arial', 12))
        checkbox.grid(row=3+i, column=1, sticky="w")
        metric_vars.append((metric, var))

    # Button to trigger plot update
    def on_plot():
        selected_metrics = [metric for metric, var in metric_vars if var.get()]
        dataset = dataset_var.get()
        type_plot = type_plot_var.get()
        if selected_metrics:
            plot_graph(ax, dataset, type_plot, selected_metrics)

    plot_button = tk.Button(root, text="Plot", font=('Arial', 12), command=on_plot)
    plot_button.grid(row=9, column=0, columnspan=2, padx=10, pady=10)

    # Run the GUI event loop
    root.mainloop()


# Run the GUI application
if __name__ == "__main__":
    run_gui()
