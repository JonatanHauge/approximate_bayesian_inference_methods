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
               filter_by = {"low_rank":[0], "K":[10], "prior_var": []}, metric = ["Accuracy"]):
     
     KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
                 "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}
     
     
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

      # Initialize main window
     root = tk.Tk()
     root.title("Interactive Plotting GUI")
     # Create a figure and axis
     fig, ax = plt.subplots(figsize=(10, 6))
     plt.ion()  # Turn on interactive mode
     fig.show()

     # Variables for storing user selections
     selected_dataset = StringVar(value="mnist")
     selected_plot_by = StringVar(value="prior_var")
     selected_metric = StringVar(value="Accuracy")
     BBVI_method = BooleanVar()
     Langevin_method = BooleanVar()
     selected_T_types = {"BBVI_fixed": BooleanVar(value=False), "BBVI_vary": BooleanVar(value=False), "Langevin_fixed": BooleanVar(value=False), "Langevin_vary": BooleanVar(value=False)}
     filter_low_rank = BooleanVar(value=True)
     filter_K = BooleanVar(value=True)
     filter_prior_var = BooleanVar(value=False)

     def update_plot():
          # Gather selected values
          dataset = selected_dataset.get()
          print(dataset)
          plot_by = selected_plot_by.get()
          metric = [selected_metric.get()]
          #methods = [m for m, v in selected_methods.items() if v.get()]
          #print([v.get() for m, v in selected_methods.items()])
          methods = []
          if BBVI_method.get():
               methods.append("BBVI")
          if Langevin_method.get():
               methods.append("Langevin")
          if not methods:
               print("No methods selected!")
               return
          T_types = []
          if selected_T_types["BBVI_fixed"].get() and "BBVI" in methods:
               T_types.append("fixed")
          if selected_T_types["BBVI_vary"].get() and "BBVI" in methods:
               T_types.append("vary")
          if selected_T_types["Langevin_fixed"].get() and "Langevin" in methods:
               T_types.append("fixed")
          if selected_T_types["Langevin_vary"].get() and "Langevin" in methods:
               T_types.append("vary")

          print(f"Dataset: {dataset}, Plot By: {plot_by}, Metric: {metric}, Methods: {methods}, T_types: {T_types}")

          
          if not T_types:
               print("No T_types selected!")
               return
          
          # Adjust filter options based on plot_by selection
          filter_by = {"low_rank": [0] if filter_low_rank.get() else [],
                         "K": [10, 100] if filter_K.get() else [],
                         "prior_var": [1e-5] if filter_prior_var.get() else []}

          # Debug: print selected parameters
          print(f"Dataset: {dataset}, Plot By: {plot_by}, Metric: {metric}, Methods: {methods}, T_types: {T_types}, Filters: {filter_by}")

          # Check if there are methods and T_types selected to plot
          if not methods:
               print("No methods selected!")
               return
          if not T_types:
               print("No T_types selected!")
               return
          
          # Clear previous plot
          ax.clear()
          
          # Re-plot with updated settings
          try:
               plot_graph(ax, nested_result_dict, dataset=dataset, methods=methods, T_types=T_types, 
                         plot_by=plot_by, filter_by=filter_by, metric=metric)
               print("Plotting successful!")
          except KeyError as e:
               print(f"Data not found for selection: {e}")
          
          # Update the plot
          fig.canvas.draw()

     # Helper function to clear other checkboxes in a single-choice set
     def clear_other_checks(var, others):
          for other in others:
               if other != var:
                    other.set(False)

     # Dataset selection (single choice)
     tk.Label(root, text="Choose Dataset:").grid(row=0, column=0, sticky="w")
     tk.Radiobutton(root, text="MNIST", variable=selected_dataset, value="mnist").grid(row=0, column=1, sticky="w")
     tk.Radiobutton(root, text="Fashion MNIST", variable=selected_dataset, value="fashion_mnist").grid(row=0, column=2, sticky="w")

     # Plot by selection (single choice)
     tk.Label(root, text="Plot By:").grid(row=1, column=0, sticky="w")
     tk.Radiobutton(root, text="Prior Variance", variable=selected_plot_by, value="prior_var").grid(row=1, column=1, sticky="w")
     tk.Radiobutton(root, text="Low Rank", variable=selected_plot_by, value="low_rank").grid(row=1, column=2, sticky="w")
     tk.Radiobutton(root, text="K", variable=selected_plot_by, value="K").grid(row=1, column=3, sticky="w")

     # Metric selection (single choice)
     tk.Label(root, text="Metric:").grid(row=2, column=0, sticky="w")
     tk.Radiobutton(root, text="Accuracy", variable=selected_metric, value="Accuracy").grid(row=2, column=1, sticky="w")
     tk.Radiobutton(root, text="Entropy", variable=selected_metric, value="Entropy").grid(row=2, column=2, sticky="w")
     tk.Radiobutton(root, text="LPD", variable=selected_metric, value="LPD").grid(row=2, column=3, sticky="w")

     # Method selection (multiple choices)
     tk.Label(root, text="Choose Method(s):").grid(row=3, column=0, sticky="w")
     tk.Checkbutton(root, text="BBVI", variable=BBVI_method).grid(row=3, column=1, sticky="w")
     tk.Checkbutton(root, text="Langevin", variable=Langevin_method).grid(row=3, column=2, sticky="w")

     # T_type selection based on methods chosen
     tk.Label(root, text="Choose T_type for each Method:").grid(row=4, column=0, sticky="w")
     tk.Checkbutton(root, text="Fixed (BBVI)", variable=selected_T_types["BBVI_fixed"]).grid(row=4, column=1, sticky="w")
     tk.Checkbutton(root, text="Vary (BBVI)", variable=selected_T_types["BBVI_vary"]).grid(row=4, column=2, sticky="w")
     tk.Checkbutton(root, text="Fixed (Langevin)", variable=selected_T_types["Langevin_fixed"]).grid(row=5, column=1, sticky="w")
     tk.Checkbutton(root, text="Vary (Langevin)", variable=selected_T_types["Langevin_vary"]).grid(row=5, column=2, sticky="w")

     # Filter selection based on plot_by
     tk.Label(root, text="Filters:").grid(row=6, column=0, sticky="w")
     tk.Checkbutton(root, text="Low Rank", variable=filter_low_rank).grid(row=6, column=1, sticky="w")
     tk.Checkbutton(root, text="K", variable=filter_K).grid(row=6, column=2, sticky="w")
     tk.Checkbutton(root, text="Prior Variance", variable=filter_prior_var).grid(row=6, column=3, sticky="w")


     # Button to trigger plot update
     tk.Button(root, text="Plot", command=update_plot).grid(row=7, column=0, columnspan=4)

     # Start the GUI loop
     root.mainloop()

if __name__ == "__main__":
    main_gui()