import pickle
import tkinter as tk
from tkinter import ttk, BooleanVar, StringVar, IntVar
import matplotlib.pyplot as plt
from pathlib import Path

with open(Path(r"results\hpc_outputs2\nested_result_dict.pkl"), "rb") as f:
     nested_result_dict = pickle.load(f)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))


def plot_graph(ax, result_dict, dataset = "mnist", methods = ["BBVI"], 
               T_types = ["fixed"],  plot_by="prior_var", 
               filter_by = {"low_rank":[0], "K":[10], "prior_var": [0.01]}, metric = ["Accuracy"]):
     
     KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
                 "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}
     
     ax.clear()
     
     
     for method in methods:
          for i, T_type in enumerate(T_types):
               if plot_by == "prior_var":
                    x = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
                    for low_rank in filter_by["low_rank"]:
                         for K in filter_by["K"]:
                              for m in metric:
                                   if T_type == "fixed":
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in x]
                                        ax.plot(x, y, 'o-', label=f"{method}, {T_type} T, low_rank={low_rank}, K={K}")
                                   else:
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in x]
                                        ax.plot(x, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, low_rank={low_rank}, K={K}")
                    ax.set_xscale('log')
                    ax.set_xlabel("Prior sigma")

               elif plot_by == "low_rank":
                    x = [0, 1, 2, 5, 10]
                    for sigma in filter_by["prior_var"]:
                         for K in filter_by["K"]:
                              for m in metric:
                                   if T_type == "fixed":
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for low_rank in x]
                                        ax.plot(x, y, 'o-', label=f"{method}, {T_type} T, sigma={sigma}, K={K}")
                                   else:
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for low_rank in x]
                                        ax.plot(x, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, K={K}")
                    ax.set_xlabel("Low rank")

               elif plot_by == "K":
                    x = [10, 100, 1000]
                    for sigma in filter_by["prior_var"]:
                         for low_rank in filter_by["low_rank"]:
                              for m in metric:
                                   if T_type == "fixed":
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for K in x]
                                        ax.plot(x, y, 'o-', label=f"{method}, {T_type} T, sigma={sigma}, low_rank={low_rank}")
                                   else:
                                        y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for K in x]
                                        ax.plot(x, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, low_rank={low_rank}")
                    ax.set_xlabel("K")
                    ax.set_xscale('log')


     ax.set_ylabel(metric[0])
     ax.set_title(f"{dataset}\n{metric[0]} vs {plot_by}")
     ax.legend()
     plt.draw()


def main_gui():
     
    root = tk.Tk()
    root.title("Interactive Plot")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()  # Turn on interactive mode
    fig.show()

    # Define a list to hold checkboxes
    method_vars = []
    T_vars = []

    # Checkbox for datasets
    tk.Label(root, text="Dataset:", font=('Arial', 12)).grid(row=0, column=0)
    dataset_var = tk.StringVar(value="fashion_mnist")
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, values=["mnist", "fashion_mnist"], font=('Arial', 12))
    dataset_dropdown.grid(row=0, column=1, padx=10, pady=5)

    # Checkbox for plot type
    tk.Label(root, text="Plot by:", font=('Arial', 12)).grid(row=1, column=0)
    type_plot_var = tk.StringVar(value="prior_var")
    type_plot_dropdown = ttk.Combobox(root, textvariable=type_plot_var, values=["prior_var", "low_rank", "K"], font=('Arial', 12))
    type_plot_dropdown.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text = "Metric to plot:", font=('Arial', 12)).grid(row=2, column=0)
    metric_var = tk.StringVar(value="Accuracy")
    metric_dropdown = ttk.Combobox(root, textvariable=metric_var, values=["Accuracy", "Entropy", "LPD", "ECE", "MCE", "OOD"], font=('Arial', 12))
    metric_dropdown.grid(row=2, column=1, padx=10, pady=5)
    
    # Add checkboxes for methods
    tk.Label(root, text="Methods:", font=('Arial', 12)).grid(row=3, column=0)
    methods = ["BBVI", "Langevin"]
    for i, method in enumerate(methods):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=method, variable=var, font=('Arial', 12))
        checkbox.grid(row=4, column=1+i, sticky="w")
        method_vars.append((method, var))

    # Add checkboxes for T types
    tk.Label(root, text="T types:", font=('Arial', 12)).grid(row=5, column=0)
    T_types = ["fixed (BBVI)", "fixed (Langevin)", "vary (BBVI)", "vary (Langevin)"]
    for i, T_type in enumerate(T_types):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=T_type, variable=var, font=('Arial', 12))
        checkbox.grid(row=6 + (i > 1), column=1 + i%2, sticky="w")
        T_vars.append((T_type, var))

    filter_vars_K = []
    filter_vars_low_rank = []
    filter_vars_prior_var = []

    # Add checkboxes for K
    tk.Label(root, text="K:", font=('Arial', 12)).grid(row=8, column=0)
    Ks = [10, 100, 1000]
    for i, K in enumerate(Ks):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=K, variable=var, font=('Arial', 12))
        checkbox.grid(row=9, column=1+i, sticky="w")
        filter_vars_K.append((K, var))

    # Add checkboxes for low rank
    tk.Label(root, text="Low rank:", font=('Arial', 12)).grid(row=10, column=1)
    low_ranks = [0, 1, 2, 5, 10]
    for i, low_rank in enumerate(low_ranks):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=low_rank, variable=var, font=('Arial', 12))
        checkbox.grid(row=11+i, column=1, sticky="w")
        filter_vars_low_rank.append((low_rank, var))

    # Add checkboxes for prior variance
    tk.Label(root, text="Prior variance:", font=('Arial', 12)).grid(row=10, column=2)
    prior_vars = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    for i, prior_var in enumerate(prior_vars):
        var = BooleanVar()
        checkbox = tk.Checkbutton(root, text=prior_var, variable=var, font=('Arial', 12))
        checkbox.grid(row=11+i, column=2, sticky="w")
        filter_vars_prior_var.append((prior_var, var))

    # Button to trigger plot update
    def on_plot():
        selected_methods = [method for method, var in method_vars if var.get()]
        selected_T_vars = [T_type for T_type, var in T_vars if var.get()]
        selected_Ks = [K for K, var in filter_vars_K if var.get()]
        selected_low_ranks = [low_rank for low_rank, var in filter_vars_low_rank if var.get()]
        selected_prior_vars = [prior_var for prior_var, var in filter_vars_prior_var if var.get()]
        selected_metric = [metric_var.get()]
        selected_dataset = dataset_var.get()
        type_plot = type_plot_var.get()
        print("selected_methods", selected_methods, "selected_T_vars", selected_T_vars, "selected_Ks", selected_Ks, "selected_low_ranks", selected_low_ranks, "selected_prior_vars", selected_prior_vars, "selected_metric", selected_metric)

        if selected_methods:
            plot_graph(ax, result_dict=nested_result_dict, dataset=selected_dataset, methods=selected_methods, 
                       T_types=selected_T_vars, plot_by=type_plot, 
                       filter_by={"low_rank":selected_low_ranks, "K":selected_Ks, "prior_var":selected_prior_vars}, 
                       metric=selected_metric)
            
    plot_button = tk.Button(root, text="Plot", font=('Arial', 12), command=on_plot)
    plot_button.grid(row=20, column=0, columnspan=3, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_gui()

