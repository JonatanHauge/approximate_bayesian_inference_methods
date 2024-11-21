import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk, Checkbutton, BooleanVar
import pickle


with open(Path(r"results\hpc_outputs2\nested_result_dict.pkl"), "rb") as f:
     nested_result_dict = pickle.load(f)


# Define the variables
prior_sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
low_ranks = [0, 1, 2, 5, 10]
K_list = [10, 100, 1000]
color_list = ["lightsteelblue", "royalblue", "darkblue"]
color_list2 = ["pink", "red", "darkred"]

def plot_graph(ax, result_dict, dataset = "mnist", methods = ["BBVI"], 
               T_types = {"BBVI": ["fixed"], "Langevin": ["fixed"]},  plot_by="prior_var", 
               filter_by = {"low_rank":[0], "K":[10], "prior_var": [0.01]}, metric = ["Accuracy"]):
     
     KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
                 "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}
     
     ax.clear()
     fm_baseline = {"Accuracy": 0.9055, "Entropy": 0.169, "LPD": -0.289, "ECE": 0.031, "MCE": 0.271, "OOD": 0.375}
     m_baseline = {"Accuracy": 0.989, "Entropy": 0.023, "LPD": -0.031, "ECE": 0.004, "MCE": 0.189, "OOD": 1.217}
     
     method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), "Langevin": plt.get_cmap('Reds', 20), "LLLA": plt.get_cmap('Greens', 20)}

     if dataset == "mnist":
          ax.axhline(y=m_baseline[metric[0]], color='black', linestyle='--', label="MNIST Baseline")
     elif dataset == "fashion_mnist":
          ax.axhline(y=fm_baseline[metric[0]], color='black', linestyle='--', label="Fashion MNIST Baseline")
     
     for method in methods:

          col_idx = 8
          if method == "LLLA":
               for m in metric:
                    y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K=0"][f"T=1"][f"low_rank={0}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                    ax.plot(prior_sigmas, y, 'o-', label=f"{method}", c = method_color_dict[method](col_idx))
                    col_idx += 2
               break
          for i, T_type in enumerate(T_types[method]):
               if plot_by == "prior_var":
                    for low_rank in filter_by["low_rank"]:
                         if len(filter_by["low_rank"]) == 1 and method == "Langevin":
                              low_rank = 0
                         elif method == "Langevin" and low_rank != 0:
                              break
                         for K in filter_by["K"]:
                              for m in metric:
                                   if T_type == "fixed":
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                                        ax.plot(prior_sigmas, y, 'o-', label=f"{method}, {T_type} T, low_rank={low_rank}, K={K}", c = method_color_dict[method](col_idx))
                                        col_idx += 2
                                   else:
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                                        ax.plot(prior_sigmas, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, low_rank={low_rank}, K={K}", c = method_color_dict[method](col_idx))
                                        col_idx += 2
                    ax.set_xscale('log')
                    ax.set_xlabel("Prior sigma")

               elif plot_by == "low_rank":
                    for sigma in filter_by["prior_var"]:
                         for K in filter_by["K"]:
                              for m in metric:
                                   if T_type == "fixed":
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for low_rank in low_ranks]
                                        ax.plot(low_ranks, y, 'o-', label=f"{method}, {T_type} T, sigma={sigma}, K={K}", c = method_color_dict[method](col_idx))
                                        col_idx += 2
                                   else:
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for low_rank in low_ranks]
                                        ax.plot(low_ranks, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, K={K}", c = method_color_dict[method](col_idx))
                                        col_idx += 2
                    ax.set_xlabel("Low rank")

               elif plot_by == "K":
                    for sigma in filter_by["prior_var"]:
                         for low_rank in filter_by["low_rank"]:
                              if len(filter_by["low_rank"]) == 1 and method == "Langevin":
                                   low_rank = 0
                              elif method == "Langevin" and low_rank != 0:
                                   break
                              for m in metric:
                                   if T_type == "fixed":
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for K in K_list]
                                        ax.plot(K_list, y, 'o-', label=f"{method}, {T_type} T, sigma={sigma}, low_rank={low_rank}, c = method_color_dict[method](col_idx)")
                                        col_idx += 2
                                   else:
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for K in K_list]
                                        ax.plot(K_list, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, low_rank={low_rank}", c = method_color_dict[method](col_idx))
                                        col_idx += 2
                    ax.set_xlabel("K")
                    ax.set_xscale('log')


     ax.set_ylabel(metric[0])
     ax.set_title(f"{dataset}\n{metric[0]} vs {plot_by}")
     ax.legend()
     plt.draw()


# Define the Tkinter GUI for checkboxes
def run_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Interactive Plot")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()  # Turn on interactive mode
    fig.show()
    
    # dropdown for datasets
    tk.Label(root, text="Dataset:", font=('Arial', 16)).grid(row=0, column=0)
    dataset_var = tk.StringVar(value="fashion_mnist")
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, values=["mnist", "fashion_mnist"], font=('Arial', 16))
    dataset_dropdown.grid(row=0, column=1, padx=10, pady=5)

    # dropdown for plot type
    tk.Label(root, text="Plot by:", font=('Arial', 16)).grid(row=1, column=0)
    type_plot_var = tk.StringVar(value="prior_var")
    type_plot_dropdown = ttk.Combobox(root, textvariable=type_plot_var, values=["prior_var", "K", "low_rank"], font=('Arial', 16))
    type_plot_dropdown.grid(row=1, column=1, padx=10, pady=5)

    #dropdown for the metric
    tk.Label(root, text="Metric:", font=('Arial', 16)).grid(row=2, column=0)
    metric_var = tk.StringVar(value="Accuracy")
    metric_dropdown = ttk.Combobox(root, textvariable=metric_var, values=["Accuracy", "Entropy", "LPD", "ECE", "MCE", "OOD"], font=('Arial', 16))
    metric_dropdown.grid(row=2, column=1, padx=10, pady=5)

    # Checkbox for methods
    method_vars = []
    tk.Label(root, text="VI Methods:", font=('Arial', 16)).grid(row=3, column=0)
    method_types = ["BBVI", "Langevin", "LLLA"]
    for i, method in enumerate(method_types):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=method, variable=var, font=('Arial', 16))
        checkbox.grid(row=3, column=1 + i, sticky="w")
        method_vars.append((method, var))

    #checkbox for T_type
    T_vars = []
    tk.Label(root, text="T types:", font=('Arial', 16)).grid(row=4, column=0)
    T_types = ["fixed (BBVI)", "fixed (Langevin)", "vary (BBVI)", "vary (Langevin)"]
    for i, T_type in enumerate(T_types):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=T_type, variable=var, font=('Arial', 16))
        checkbox.grid(row=4 + (i > 1), column=1 + i%2, sticky="w")
        T_vars.append((T_type, var))

    filter_K_vars = []
    filter_low_rank_vars = []
    filter_prior_var_vars = []

    tk.Label(root, text="Filter by:", font=('Arial', 16)).grid(row=7, column=0)
    tk.Label(root, text="K:", font=('Arial', 16)).grid(row=8, column=0)
    Ks = [10, 100, 1000]
    for i, K in enumerate(Ks):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=K, variable=var, font=('Arial', 16))
        checkbox.grid(row=8, column=1 + i, sticky="w")
        filter_K_vars.append((K, var))
    var = BooleanVar()
    checkbox = tk.Checkbutton(root, text="All", variable=var, font=('Arial', 16))
    checkbox.grid(row=8, column=5, sticky="w")
    filter_K_vars.append(("All", var))
    
    tk.Label(root, text="Low rank:", font=('Arial', 16)).grid(row=9, column=0)
    low_ranks = [0, 1, 2, 5, 10]
    for i, low_rank in enumerate(low_ranks):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=low_rank, variable=var, font=('Arial', 16))
        checkbox.grid(row=9, column=1 + i, sticky="w")
        filter_low_rank_vars.append((low_rank, var))
    var = BooleanVar()
    checkbox = tk.Checkbutton(root, text="All", variable=var, font=('Arial', 16))
    checkbox.grid(row=9, column=7, sticky="w")
    filter_low_rank_vars.append(("All", var))

    tk.Label(root, text="Prior sigma:", font=('Arial', 16)).grid(row=10, column=0)
    prior_sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    for i, prior_sigma in enumerate(prior_sigmas):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=prior_sigma, variable=var, font=('Arial', 16))
        checkbox.grid(row=10, column=1 + i, sticky="w")
        filter_prior_var_vars.append((prior_sigma, var))
    var = BooleanVar()
    checkbox = tk.Checkbutton(root, text="All", variable=var, font=('Arial', 16))
    checkbox.grid(row=10, column=10, sticky="w")
    filter_prior_var_vars.append(("All", var))


    # Button to trigger plot update
    def on_plot():
        #selected_metrics = [metric for metric, var in metric_vars if var.get()]
        selected_metrics = [metric_var.get()]
        dataset = dataset_var.get()
        type_plot = type_plot_var.get()
        method = [method for method, var in method_vars if var.get()]
        T = [T_type for T_type, var in T_vars if var.get()]
        T_type_dict = {"BBVI": [], "Langevin": []}
        for t in T:
            if "BBVI" in t:
                T_type_dict["BBVI"].append(t.split(" ")[0])
            else:
                T_type_dict["Langevin"].append(t.split(" ")[0])
        filter_K = [K for K, var in filter_K_vars if var.get()]
        filter_low_rank = [low_rank for low_rank, var in filter_low_rank_vars if var.get()]
        filter_prior_var = [prior_var for prior_var, var in filter_prior_var_vars if var.get()]
        if "All" in filter_prior_var:
            filter_prior_var = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        if "All" in filter_K:
             filter_K = [10, 100, 1000]
        if "All" in filter_low_rank:
             filter_low_rank = [0, 1, 2, 5, 10]
        #print("selected_metrics: ", selected_metrics, "dataset: ", dataset, "type_plot: ", type_plot, "method: ", method, "T type dict: ", T_type_dict, "filter_K: ", filter_K, "filter_low_rank: ", filter_low_rank, "filter_prior_var: ", filter_prior_var)
        filter_dict = {"K": filter_K, "low_rank": filter_low_rank, "prior_var": filter_prior_var}
        if selected_metrics:
            plot_graph(ax, result_dict=nested_result_dict, dataset=dataset, methods=method, T_types=T_type_dict, 
                       plot_by=type_plot, filter_by=filter_dict, metric=selected_metrics)

    plot_button = tk.Button(root, text="Plot", font=('Arial', 16), command=on_plot)
    plot_button.grid(row=12, column=0, columnspan=2, padx=10, pady=10)

    # Run the GUI event loop
    root.mainloop()


# Run the GUI application
if __name__ == "__main__":
    run_gui()
