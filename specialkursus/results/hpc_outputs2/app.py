import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import streamlit as st
from plotting_helpers import plot_all

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

# Streamlit app setup
#st.title("Interactive Plot for VI Methods")

# Dataset dropdown
col1, col2 = st.columns([0.3, 0.7])

with col1:

    col3, col4 = st.columns(2)
    with col3:
        dataset = st.radio("**Dataset:**", ["mnist", "fashion_mnist"], index=1)
        st.write("**Select inference Methods:**")
        method1 = st.checkbox("BBVI", value=True)
        method2 = st.checkbox("Langevin", value=False)
        method3 = st.checkbox("LLLA", value=False)
        method4 = st.checkbox("NF", value=False)
        method5 = st.checkbox("subspace+LL", value=False)
        methods = []
        if method1:
            methods.append("BBVI")
        if method2:
            methods.append("Langevin")
        if method3:
            methods.append("LLLA")
        if method4:
            methods.append("NF")
        if method5:
            methods.append("subspace+LL")

        num_transformations = []
        if method4:
            st.markdown("**Number of layers in flow**")
            num_trans_10 = st.checkbox("10 transformations", value=False)
            num_trans_30 = st.checkbox("30 transformations", value=True)
            if num_trans_10:
                num_transformations.append(10)
            if num_trans_30:
                num_transformations.append(30)
        
        plot_by =  st.radio("**Plot by**", ["prior_var", "K", "low_rank"])
        metric = st.radio("**Metric:**", ["Accuracy", "Entropy", "LPD", "ECE", "MCE", "OOD"])

    with col4:
        T_type_dict = {"BBVI": [], "Langevin": [], "NF": [], "subspace+LL": []}
        if method1 or method2 or method4 or method5:
            st.markdown("**Select T-type**)")
        if method1:
            T_types_BBVI_fixed = st.checkbox("T=1 (BBVI)", value=True)
            if T_types_BBVI_fixed:
                T_type_dict["BBVI"].append("T=1")
            T_types_BBVI_vary = st.checkbox("vary (BBVI)", value=False)
            if T_types_BBVI_vary:
                T_type_dict["BBVI"].append("vary")
        if method2:
            T_types_Langevin_fixed = st.checkbox("T=1 (Langevin)", value=True)
            if T_types_Langevin_fixed:
                T_type_dict["Langevin"].append("T=1")
            T_types_Langevin_vary = st.checkbox("vary (Langevin)", value=False)
            if T_types_Langevin_vary:
                T_type_dict["Langevin"].append("vary")
        if method4:
            T_types_NF_fixed = st.checkbox("T=1 (NF)", value=True)
            if T_types_NF_fixed:
                T_type_dict["NF"].append("T=1")
            T_types_NF_vary = st.checkbox("vary (NF)", value=False)
            if T_types_NF_vary:
                T_type_dict["NF"].append("vary")
        if method5:
            T_types_subspace_and_full_LL_fixed = st.checkbox("T=1 (subspace+LL)", value=True)
            if T_types_subspace_and_full_LL_fixed:
                T_type_dict["subspace+LL"].append("T=1")
            #T_types_subspace_and_full_LL_vary = st.checkbox("vary (subspace+LL)", value=False)
            #if T_types_subspace_and_full_LL_vary:
            #    T_type_dict["subspace+LL"].append("vary")
            
        # Filter checkboxes
        if plot_by == "prior_var":
            filter_prior_var = []
            st.markdown("**Select K's**")
            filter_K10 = st.checkbox("K=10", value=False)
            filter_K100 = st.checkbox("K=100", value=False)
            filter_K1000 = st.checkbox("K=1000", value=True)
            filter_K = []
            if filter_K10:
                filter_K.append(10)
            if filter_K100:
                filter_K.append(100)
            if filter_K1000:
                filter_K.append(1000)
            
            st.markdown("**Select low_ranks**")
            filter_low_rank0 = st.checkbox("low_rank=0", value=True)
            filter_low_rank1 = st.checkbox("low_rank=1", value=False)
            filter_low_rank2 = st.checkbox("low_rank=2", value=False)
            filter_low_rank5 = st.checkbox("low_rank=5", value=False)
            filter_low_rank10 = st.checkbox("low_rank=10", value=False)

            filter_low_rank = []
            if filter_low_rank0:
                filter_low_rank.append(0)
            if filter_low_rank1:
                filter_low_rank.append(1)
            if filter_low_rank2:
                filter_low_rank.append(2)
            if filter_low_rank5:
                filter_low_rank.append(5)
            if filter_low_rank10:
                filter_low_rank.append(10)

        elif plot_by == "K":
            filter_K = []
            st.markdown("**Select low_ranks**")
            filter_low_rank0 = st.checkbox("low_rank=0", value=True)
            filter_low_rank1 = st.checkbox("low_rank=1", value=False)
            filter_low_rank2 = st.checkbox("low_rank=2", value=False)
            filter_low_rank5 = st.checkbox("low_rank=5", value=False)
            filter_low_rank10 = st.checkbox("low_rank=10", value=False)

            filter_low_rank = []
            if filter_low_rank0:
                filter_low_rank.append(0)
            if filter_low_rank1:
                filter_low_rank.append(1)
            if filter_low_rank2:
                filter_low_rank.append(2)
            if filter_low_rank5:
                filter_low_rank.append(5)
            if filter_low_rank10:
                filter_low_rank.append(10)

            st.markdown("**Select Prior Sigmas**")
            filter_prior_var1e6 = st.checkbox("sigma=1e-6", value=False)
            filter_prior_var1e5 = st.checkbox("sigma=1e-5", value=False)
            filter_prior_var1e4 = st.checkbox("sigma=1e-4", value=False)
            filter_prior_var1e3 = st.checkbox("sigma=1e-3", value=True)
            filter_prior_var1e2 = st.checkbox("sigma=1e-2", value=False)
            filter_prior_var1e1 = st.checkbox("sigma=1e-1", value=False)
            filter_prior_var1 = st.checkbox("sigma=1", value=False)
            filter_prior_var10 = st.checkbox("sigma=10", value=False)

            filter_prior_var = []
            if filter_prior_var1e6:
                filter_prior_var.append(1e-6)
            if filter_prior_var1e5:
                filter_prior_var.append(1e-5)
            if filter_prior_var1e4:
                filter_prior_var.append(1e-4)
            if filter_prior_var1e3:
                filter_prior_var.append(1e-3)
            if filter_prior_var1e2:
                filter_prior_var.append(1e-2)
            if filter_prior_var1e1:
                filter_prior_var.append(1e-1)
            if filter_prior_var1:
                filter_prior_var.append(1)
            if filter_prior_var10:
                filter_prior_var.append(10)

        elif plot_by == "low_rank":
            filter_low_rank = []
            st.markdown("**Select K's**")
            filter_K10 = st.checkbox("K=10", value=False)
            filter_K100 = st.checkbox("K=100", value=False)
            filter_K1000 = st.checkbox("K=1000", value=True)

            filter_K = []
            if filter_K10:
                filter_K.append(10)
            if filter_K100:
                filter_K.append(100)
            if filter_K1000:
                filter_K.append(1000)
            
            st.markdown("**Select Prior Sigmas**")  
            filter_prior_var1e6 = st.checkbox("sigma=1e-6", value=False)
            filter_prior_var1e5 = st.checkbox("sigma=1e-5", value=False)
            filter_prior_var1e4 = st.checkbox("sigma=1e-4", value=False)
            filter_prior_var1e3 = st.checkbox("sigma=1e-3", value=True)
            filter_prior_var1e2 = st.checkbox("sigma=1e-2", value=False)
            filter_prior_var1e1 = st.checkbox("sigma=1e-1", value=False)
            filter_prior_var1 = st.checkbox("sigma=1", value=False)
            filter_prior_var10 = st.checkbox("sigma=10", value=False)

            filter_prior_var = []
            if filter_prior_var1e6:
                filter_prior_var.append(1e-6)
            if filter_prior_var1e5:
                filter_prior_var.append(1e-5)
            if filter_prior_var1e4:
                filter_prior_var.append(1e-4)
            if filter_prior_var1e3:
                filter_prior_var.append(1e-3)
            if filter_prior_var1e2:
                filter_prior_var.append(1e-2)
            if filter_prior_var1e1:
                filter_prior_var.append(1e-1)
            if filter_prior_var1:
                filter_prior_var.append(1)
            if filter_prior_var10:
                filter_prior_var.append(10)


with col2:
# Button to plot
    #if st.button("Plot"):
    zoom = st.checkbox("Zoom", value=False)
    y_lower_lim = 0
    y_upper_lim = 1
    if zoom:
        y_lower_lim = st.number_input("Y-axis lower limit", value=0.0, step = 0.001)
        y_upper_lim = st.number_input("Y-axis upper limit", value=1.0, step = 0.001)
    filter_dict = {
        "K": filter_K if "All" not in filter_K else K_list,
        "low_rank": filter_low_rank if "All" not in filter_low_rank else low_ranks,
        "prior_var": filter_prior_var if "All" not in filter_prior_var else prior_sigmas,
    }
    fig = plot_all(nested_result_dict, dataset=dataset, methods=methods, 
                    T_types=T_type_dict, num_transformations=num_transformations, 
                    plot_by=plot_by, filter_by=filter_dict, metric=[metric], zoom = zoom, 
                    y_lower_lim = y_lower_lim, y_upper_lim = y_upper_lim)
    
    st.pyplot(fig) 

    
