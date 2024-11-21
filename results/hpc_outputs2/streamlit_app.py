import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import streamlit as st

st.set_page_config(layout="wide")

# Load the data
with open(Path(r"results\hpc_outputs2\nested_result_dict.pkl"), "rb") as f:
    nested_result_dict = pickle.load(f)

# Define variables
prior_sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
low_ranks = [0, 1, 2, 5, 10]
K_list = [10, 100, 1000]
color_list = ["lightsteelblue", "royalblue", "darkblue"]
color_list2 = ["pink", "red", "darkred"]

# Define the plotting function
def plot_graph(result_dict, dataset="mnist", methods=["BBVI"], 
               T_types={"BBVI": ["fixed"], "Langevin": ["fixed"]},  
               plot_by="prior_var", filter_by={"low_rank":[0], "K":[10], "prior_var": [0.01]}, 
               metric=["Accuracy"]):

    KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
               "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}
    
    fm_baseline = {"Accuracy": 0.9055, "Entropy": 0.169, "LPD": -0.289, "ECE": 0.031, "MCE": 0.271, "OOD": 0.375}
    m_baseline = {"Accuracy": 0.989, "Entropy": 0.023, "LPD": -0.031, "ECE": 0.004, "MCE": 0.189, "OOD": 1.217}

    method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), "Langevin": plt.get_cmap('Reds', 20), "LLLA": plt.get_cmap('Greens', 20)}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if dataset == "mnist":
        ax.axhline(y=m_baseline[metric[0]], color='black', linestyle='--', label="MNIST Baseline")
    elif dataset == "fashion_mnist":
        ax.axhline(y=fm_baseline[metric[0]], color='black', linestyle='--', label="Fashion MNIST Baseline")
    
    # Main plotting loop
    for method in methods:
        col_idx = 8
        if method == "LLLA":
            for m in metric:
                y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K=0"][f"T=1"][f"low_rank={0}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                ax.plot(prior_sigmas, y, 'o-', label=f"{method}", c=method_color_dict[method](col_idx))
                col_idx += 2
            ax.set_xscale('log')
            ax.set_xlabel("Prior sigma")
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
                                ax.plot(prior_sigmas, y, 'o-', label=f"{method}, {T_type} T, low_rank={low_rank}, K={K}", c=method_color_dict[method](col_idx))
                                col_idx += 2
                            else:
                                y = [result_dict[f"method={method}"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                                ax.plot(prior_sigmas, y, 'o-', label=f"{method}, {KT_dict[dataset][f'K={K}']}, low_rank={low_rank}, K={K}", c=method_color_dict[method](col_idx))
                                col_idx += 2
                ax.set_xscale('log')
                ax.set_xlabel("Prior sigma")

    ax.set_ylabel(metric[0])
    ax.set_title(f"{dataset}\n{metric[0]} vs {plot_by}")
    ax.legend()
    st.pyplot(fig)

# Streamlit app setup
st.title("Interactive Plot for VI Methods")

# Dataset dropdown
col1, col2 = st.columns(2)

with col1:

    col3, col4 = st.columns(2)
    with col3:
        dataset = st.radio("Dataset:", ["mnist", "fashion_mnist"], index=0)
        plot_by =  st.radio("Plot by", ["prior_var", "K", "low_rank"])
        metric = st.radio("Metric:", ["Accuracy", "Entropy", "LPD", "ECE", "MCE", "OOD"])

    with col4:
        methods = st.multiselect("Select Methods:", ["BBVI", "Langevin", "LLLA"], default=["BBVI"])

        # T types checkboxes
        T_types_selected = st.multiselect("Select T types:", ["fixed (BBVI)", "fixed (Langevin)", "vary (BBVI)", "vary (Langevin)"], default=["fixed (BBVI)", "fixed (Langevin)"])
        T_type_dict = {"BBVI": [], "Langevin": []}
        for t in T_types_selected:
            if "BBVI" in t:
                T_type_dict["BBVI"].append(t.split(" ")[0])
            else:
                T_type_dict["Langevin"].append(t.split(" ")[0])

        # Filter checkboxes
        filter_K = st.multiselect("Filter by K:", K_list, default=[10])
        filter_low_rank = st.multiselect("Filter by Low Rank:", low_ranks, default=[0])
        filter_prior_var = st.multiselect("Filter by Prior Sigma:", prior_sigmas, default=[0.01])

with col2:
# Button to plot
    if st.button("Plot"):
        filter_dict = {
            "K": filter_K if "All" not in filter_K else K_list,
            "low_rank": filter_low_rank if "All" not in filter_low_rank else low_ranks,
            "prior_var": filter_prior_var if "All" not in filter_prior_var else prior_sigmas,
        }
        plot_graph(nested_result_dict, dataset=dataset, methods=methods, T_types=T_type_dict, plot_by=plot_by, filter_by=filter_dict, metric=[metric])
