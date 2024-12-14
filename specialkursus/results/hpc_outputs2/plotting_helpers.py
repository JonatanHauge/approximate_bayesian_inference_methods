import matplotlib.pyplot as plt

# Define variables
prior_sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
low_ranks = [0, 1, 2, 5, 10]
K_list = [10, 100, 1000]
color_list = ["lightsteelblue", "royalblue", "darkblue"]
color_list2 = ["pink", "red", "darkred"]
col_step = 5

def plot_baseline(ax, dataset="mnist", metric="Accuracy"):
    fm_baseline = {"Accuracy": 0.9055, "Entropy": 0.169, "LPD": -0.289, "ECE": 0.031, "MCE": 0.271, "OOD": 0.375}
    m_baseline = {"Accuracy": 0.989, "Entropy": 0.023, "LPD": -0.031, "ECE": 0.004, "MCE": 0.189, "OOD": 1.217}
    if dataset == "mnist":
        ax.axhline(y=m_baseline[metric[0]], color='black', linestyle='--', label="MNIST Baseline")
    elif dataset == "fashion_mnist":
        ax.axhline(y=fm_baseline[metric[0]], color='black', linestyle='--', label="Fashion MNIST Baseline")


def plot_BBVI(ax, result_dict, dataset="mnist",
               T_types=["T=1"],  
               plot_by="prior_var", filter_by={"low_rank":[0], "K":[10], "prior_var": [0.01]}, 
               metric=["Accuracy"]):

    method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), "Langevin": plt.get_cmap('Reds', 20), "LLLA": plt.get_cmap('Purples', 20)}
    col_idx = 8
    KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
               "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}

    for _, T_type in enumerate(T_types):
        if plot_by == "prior_var":
            for low_rank in filter_by["low_rank"]:
                for K in filter_by["K"]:
                    for m in metric:
                        if T_type == "T=1":
                            y = [result_dict[f"method=BBVI"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                            ax.plot(prior_sigmas, y, 'o-', label=f"BBVI, {T_type}, low_rank={low_rank}, K={K}", c=method_color_dict["BBVI"](col_idx))
                            col_idx += col_step
                        else:
                            y = [result_dict[f"method=BBVI"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                            ax.plot(prior_sigmas, y, 'o-', label=f"BBVI, {KT_dict[dataset][f'K={K}']}, low_rank={low_rank}, K={K}", c=method_color_dict["BBVI"](col_idx))
                            col_idx += col_step

        elif plot_by == "low_rank":
            for sigma in filter_by["prior_var"]:
                for K in filter_by["K"]:
                    for m in metric:
                        if T_type == "T=1":
                            y = [result_dict[f"method=BBVI"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for low_rank in low_ranks]
                            ax.plot(low_ranks, y, 'o-', label=f"BBVI, {T_type}, sigma={sigma}, K={K}", c = method_color_dict["BBVI"](col_idx))
                            col_idx += col_step
                        else:
                            y = [result_dict[f"method=BBVI"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for low_rank in low_ranks]
                            ax.plot(low_ranks, y, 'o-', label=f"BBVI, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, K={K}", c = method_color_dict["BBVI"](col_idx))
                            col_idx += col_step

        elif plot_by == "K":
            for sigma in filter_by["prior_var"]:
                for low_rank in filter_by["low_rank"]:
                    for m in metric:
                        if T_type == "T=1":
                            y = [result_dict[f"method=BBVI"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for K in K_list]
                            ax.plot(K_list, y, 'o-', label=f"BBVI, {T_type}, sigma={sigma}, low_rank={low_rank}", c = method_color_dict["BBVI"](col_idx))
                            col_idx += col_step
                        else:
                            y = [result_dict[f"method=BBVI"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank={low_rank}"][f"sigma={sigma}"][m] for K in K_list]
                            ax.plot(K_list, y, 'o-', label=f"BBVI, {T_type} T, sigma={sigma}, low_rank={low_rank}", c = method_color_dict["BBVI"](col_idx))
                            col_idx += col_step




def plot_Langevin(ax, result_dict, dataset="mnist",
               T_types=["T=1"],  
               plot_by="prior_var", filter_by={"low_rank":[0], "K":[10], "prior_var": [0.01]}, 
               metric=["Accuracy"]):
    
    method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), "Langevin": plt.get_cmap('Reds', 20), "LLLA": plt.get_cmap('Purples', 20)}
    col_idx = 8
    KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
               "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}

    for _, T_type in enumerate(T_types):
        if plot_by == "prior_var":
            for K in filter_by["K"]:
                for m in metric:
                    if T_type == "T=1":
                        y = [result_dict[f"method=Langevin"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                        ax.plot(prior_sigmas, y, 'o-', label=f"Langevin, {T_type}, K={K}", c=method_color_dict["Langevin"](col_idx))
                        col_idx += col_step
                    else:
                        y = [result_dict[f"method=Langevin"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank=0"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                        ax.plot(prior_sigmas, y, 'o-', label=f"Langevin, {KT_dict[dataset][f'K={K}']}, K={K}", c=method_color_dict["Langevin"](col_idx))
                        col_idx += col_step

        elif plot_by == "low_rank":
            for sigma in filter_by["prior_var"]:
                for K in filter_by["K"]:
                    for m in metric:
                        if T_type == "T=1":
                            y = result_dict[f"method=Langevin"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"sigma={sigma}"][m]
                            ax.axhline(y, color = method_color_dict["Langevin"](col_idx),  linestyle='--', label=f"Langevin, {T_type}, sigma={sigma}, K={K}")
                            col_idx += col_step
                        else:
                            y = result_dict[f"method=Langevin"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank=0"][f"sigma={sigma}"][m]
                            ax.axhline(y, color = method_color_dict["Langevin"](col_idx),  linestyle='--', label=f"Langevin, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, K={K}")
                            col_idx += col_step

        elif plot_by == "K":
            for sigma in filter_by["prior_var"]:
                for m in metric:
                    if T_type == "T=1":
                        y = [result_dict[f"method=Langevin"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"sigma={sigma}"][m] for K in K_list]
                        ax.plot(K_list, y, 'o-', label=f"Langevin, {T_type}, sigma={sigma}", c = method_color_dict["Langevin"](col_idx))
                        col_idx += col_step
                    else:
                        y = [result_dict[f"method=Langevin"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank=0"][f"sigma={sigma}"][m] for K in K_list]
                        ax.plot(K_list, y, 'o-', label=f"Langevin, {T_type} T, sigma={sigma}", c = method_color_dict["Langevin"](col_idx))
                        col_idx += col_step


def plot_LLLA(ax, result_dict, dataset="mnist",
               plot_by="prior_var", filter_by={"low_rank":[0], "K":[10], "prior_var": [0.01]}, 
               metric=["Accuracy"]):
    
    method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), "Langevin": plt.get_cmap('Reds', 20), "LLLA": plt.get_cmap('Greys', 20)}
    col_idx = 8
    KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
               "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}

    if plot_by == "prior_var":
        for m in metric:
            y = [result_dict[f"method=LLLA"][f"dataset={dataset}"][f"K=0"]["T=1"][f"low_rank=0"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
            ax.plot(prior_sigmas, y, 'o-', label=f"LLLA", c=method_color_dict["LLLA"](col_idx))
            col_idx += col_step

    elif plot_by == "low_rank":
        for sigma in filter_by["prior_var"]:
            for m in metric:
                y = result_dict[f"method=LLLA"][f"dataset={dataset}"][f"K=0"][f"T=1"][f"low_rank=0"][f"sigma={sigma}"][m]
                ax.axhline(y, color = method_color_dict["LLLA"](col_idx), linestyle = '--', label=f"LLLA, prior sigma={sigma}")
                col_idx += col_step

    elif plot_by == "K":
        for sigma in filter_by["prior_var"]:
            for m in metric:
                y = result_dict[f"method=LLLA"][f"dataset={dataset}"][f"K=0"][f"T=1"][f"low_rank=0"][f"sigma={sigma}"][m]
                ax.axhline(y, color = method_color_dict["LLLA"](col_idx), linestyle = '--', label=f"LLLA, prior sigma={sigma}")
                col_idx += col_step


def plot_Flows(ax, result_dict, dataset="mnist", T_types = ["T=1"], num_transformations = [10],
               plot_by="prior_var", filter_by={"low_rank":[0], "K":[10], "prior_var": [0.01]}, 
               metric=["Accuracy"]):
    
    method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), 
                         "Langevin": plt.get_cmap('Reds', 20), 
                         "LLLA": plt.get_cmap('Purples', 20),
                         "Flows": plt.get_cmap('Greens', 20)}
    col_idx = 8
    KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
               "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}
    
    for _, T_type in enumerate(T_types):
        if plot_by == "prior_var":
            for K in filter_by["K"]:
                for num_trans in num_transformations:
                    for m in metric:
                        if T_type == "T=1":
                            y = [result_dict[f"method=Flows"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"num_transformations={num_trans}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                            ax.plot(prior_sigmas, y, 'o-', label=f"Flows, {T_type}, K={K}, num_trans={num_trans}", c=method_color_dict["Flows"](col_idx))
                            col_idx += col_step
                        else:
                            y = [result_dict[f"method=Flows"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank=0"][f"num_transformations={num_trans}"][f"sigma={sigma}"][m] for sigma in prior_sigmas]
                            ax.plot(prior_sigmas, y, 'o-', label=f"Flows, {KT_dict[dataset][f'K={K}']}, K={K}, num_trans={num_trans}", c=method_color_dict["Flows"](col_idx))
                            col_idx += col_step

        elif plot_by == "low_rank":
            for sigma in filter_by["prior_var"]:
                for K in filter_by["K"]:
                    for num_trans in num_transformations:
                        for m in metric:
                            if T_type == "T=1":
                                y = result_dict[f"method=Flows"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"num_transformations={num_trans}"][f"sigma={sigma}"][m]
                                ax.axhline(y, color = method_color_dict["Flows"](col_idx),  linestyle='--', label=f"Flows, {T_type}, sigma={sigma}, K={K}, num_trans={num_trans}")
                                col_idx += col_step
                            else:
                                y = result_dict[f"method=Flows"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank=0"][f"num_transformations={num_trans}"][f"sigma={sigma}"][m]
                                ax.axhline(y, color = method_color_dict["Flows"](col_idx),  linestyle='--', label=f"Flows, {KT_dict[dataset][f'K={K}']}, sigma={sigma}, K={K}, num_trans={num_trans}")
                                col_idx += col_step

        elif plot_by == "K":
            for sigma in filter_by["prior_var"]:
                for num_trans in num_transformations:
                    for m in metric:
                        if T_type == "T=1":
                            y = [result_dict[f"method=Flows"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"num_transformations={num_trans}"][f"sigma={sigma}"][m] for K in K_list]
                            ax.plot(K_list, y, 'o-', label=f"Flows, {T_type}, sigma={sigma}, num_trans={num_trans}", c = method_color_dict["Flows"](col_idx))
                            col_idx += col_step
                        else:
                            y = [result_dict[f"method=Flows"][f"dataset={dataset}"][f"K={K}"][f"{KT_dict[dataset][f'K={K}']}"][f"low_rank=0"][f"num_transformations={num_trans}"][f"sigma={sigma}"][m] for K in K_list]
                            ax.plot(K_list, y, 'o-', label=f"Flows, {T_type} T, sigma={sigma}, num_trans={num_trans}", c = method_color_dict["Flows"](col_idx))
                            col_idx += col_step


def plot_subspace_and_full_LL(ax, result_dict, dataset="mnist",
               T_types=["T=1"],  
               plot_by="prior_var", filter_by={"low_rank":[0], "K":[10], "prior_var": [0.01]}, 
               metric=["Accuracy"]):
    
    method_color_dict = {"BBVI": plt.get_cmap('Blues', 20), 
                         "Langevin": plt.get_cmap('Reds', 20), 
                         "LLLA": plt.get_cmap('Purples', 20),
                         "Flows": plt.get_cmap('Greens', 20),
                         "subspace_and_full_LL": plt.get_cmap('Purples', 20)}
    col_idx = 8
    KT_dict = {"mnist": {"K=10": "T=200", "K=100": "T=20", "K=1000": "T=2"},
               "fashion_mnist": {"K=10": "T=2500", "K=100": "T=250", "K=1000": "T=25"}}

    for _, T_type in enumerate(T_types):
        if plot_by == "prior_var":
            for K in filter_by["K"]:
                for m in metric:
                    if T_type == "T=1":
                        y = [result_dict[f"method=subspace_and_full_LL"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"sigma={sigma}"][m] for sigma in [1e-3, 1e-2, 1e-1, 1, 10, 100]]
                        ax.plot([1e-3, 1e-2, 1e-1, 1, 10, 100], y, 'o-', label=f"subspace_and_full_LL, {T_type}, K={K}", c=method_color_dict["subspace_and_full_LL"](col_idx))
                        col_idx += col_step

        elif plot_by == "low_rank":
            for sigma in filter_by["prior_var"]:
                for K in filter_by["K"]:
                    for m in metric:
                        if T_type == "T=1":
                            y = result_dict[f"method=subspace_and_full_LL"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"sigma={sigma}"][m]
                            ax.axhline(y, color = method_color_dict["subspace_and_full_LL"](col_idx),  linestyle='--', label=f"subspace_and_full_LL, {T_type}, sigma={sigma}, K={K}")
                            col_idx += col_step
                        

        elif plot_by == "K":
            for sigma in filter_by["prior_var"]:
                for m in metric:
                    if T_type == "T=1":
                        y = [result_dict[f"method=subspace_and_full_LL"][f"dataset={dataset}"][f"K={K}"][f"T={1}"][f"low_rank=0"][f"sigma={sigma}"][m] for K in K_list]
                        ax.plot(K_list, y, 'o-', label=f"subspace_and_full_LL, {T_type}, sigma={sigma}", c = method_color_dict["subspace_and_full_LL"](col_idx))
                        col_idx += col_step
                    

            



def plot_all(result_dict, dataset = "mnist", methods = ["BBVI"], 
               T_types = {"BBVI": ["T=1"], "Langevin": ["T=1"]},  num_transformations = [10], 
               plot_by="prior_var", 
               filter_by = {"low_rank":[0], "K":[10], "prior_var": [0.01]}, metric = ["Accuracy"], 
               zoom = False, y_lower_lim = 0.9, y_upper_lim = 1.1):
    
    fig, ax = plt.subplots(figsize=(15, 8))

    plot_baseline(ax, dataset=dataset, metric=metric)
    if "BBVI" in methods:
        plot_BBVI(ax, result_dict, dataset=dataset, T_types=T_types["BBVI"], plot_by=plot_by, filter_by=filter_by, metric=metric)
    if "Langevin" in methods:
        plot_Langevin(ax, result_dict, dataset=dataset, T_types=T_types["Langevin"], plot_by=plot_by, filter_by=filter_by, metric=metric)
    if "LLLA" in methods:
        plot_LLLA(ax, result_dict, dataset=dataset, plot_by=plot_by, filter_by=filter_by, metric=metric)
    if "NF" in methods:
        plot_Flows(ax, result_dict, dataset=dataset, T_types=T_types["NF"], num_transformations=num_transformations, plot_by=plot_by, filter_by=filter_by, metric=metric)
    if "subspace+LL" in methods:
        plot_subspace_and_full_LL(ax, result_dict, dataset=dataset, T_types=T_types["subspace+LL"], plot_by=plot_by, filter_by=filter_by, metric=metric)


    if plot_by == "prior_var":
        ax.set_xscale('log')
        ax.set_xlabel("Prior variance")
    elif plot_by == "low_rank":
        ax.set_xlabel("Low Rank")
    elif plot_by == "K":
        ax.set_xlabel("K")
        ax.set_xscale('log')
    
    if metric[0] in ["Accuracy", "LPD", "OOD"]:
        ax.set_ylabel(metric[0] + "  -->")
    else:
        ax.set_ylabel("<--  " + metric[0])
    ax.set_title(f"{dataset}\n{metric[0]} vs {plot_by}")
    ax.legend()

    box = ax.get_position()
    #Shrink current axis's height by 10% on the bottom
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=True, shadow=True, ncol=3, fontsize=14)
    
    if zoom:
        ax.set_ylim(y_lower_lim, y_upper_lim)
        
        
    return fig