from utils_lowrank import BlackBoxVariationalInference, log_like_NN_classification
import numpy as np
import torch 
import os
import sys
from models.LeNet5 import LeNet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


# settings
sys.stdout.flush()
print('BBVI job is now running')
datasets = ['fashion_mnist', 'mnist']
hyper_param_dict = {
                    'fashion_mnist': {'K_T_list': [(10, 2500), (100, 250), (1000, 25)]},
                    'mnist': {'K_T_list': [(10, 200), (100, 20), (1000, 2)]}
                    }
hyper_param_dict = {
                    'fashion_mnist': {'K_T_list': [(10, 2500)]},
                    'mnist': {'K_T_list': [(10, 200)]}
                    }
for dataset in datasets:
    result_table = pd.DataFrame(index = ['Accuracy', 'Entropy', 'LPD', 'ECE', 'MCE', 'OOD'])
    for K, T in hyper_param_dict[dataset]['K_T_list']:
        for low_rank in [1]:
            print("Runing for: ", dataset, ", K:", K, "T:", T, "low_rank:", low_rank)
            seed = 4242
            torch.manual_seed(seed)
            max_itt = 1000
            start_plots = [0]
            step_size = 0.0005
            batch_size = 5
            prior_sigma = 0.1
            SWA = True
            verbose = True
            save_fig = True

            fashion_MLE_log_like = -7202.52
            mnist_MLE_log_like = -534.40

            # Check if CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            # Load data
            if dataset == 'mnist':
                xtrain, ytrain = torch.load('./datasets/mnist_train.pt', map_location=device)
                train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
                xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)
                test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
                weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94%.pth') # load the weights
            elif dataset == 'fashion_mnist':
                xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
                train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
                xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
                test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
                weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55%.pth')


            assert len(test_loader.dataset) % test_loader.batch_size == 0, 'Test loader batch size must divide test set size'

            net = LeNet().to(device=device) 
            theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights
            theta_map = theta_map.to(device=device)

            num_params = sum(p.numel() for p in net.parameters())
            P = torch.randn(num_params, K, device=device)
            P /= torch.norm(P, dim=0)  # Normalize columns to norm 1

            #net.load_state_dict(weights)
            #fashion_MLE_log_like = -torch.nn.CrossEntropyLoss(reduction='sum')(net(xtrain), ytrain).item()
            #print(fashion_MLE_log_like)

            bbvi = BlackBoxVariationalInference(net, theta_map, P, log_like_NN_classification, 
                                                K, low_rank, step_size, max_itt, batch_size, seed, verbose, 
                                                T, prior_sigma, SWA, device)

            bbvi.fit(train_loader)

            #acc, entropy, lpd, ece, mce = bbvi.compute_all_metrics(test_loader, num_samples=100, num_bins=10)
            acc, entropy, lpd, ece, mce, ood = 0.,0.,0.,0.,0.,0.  

            #OOD error 
            if False:
                if dataset == 'mnist':
                        xood, yood = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
                        ood_loader = DataLoader(TensorDataset(xood, yood), batch_size=batch_size, shuffle=False)
                        ood = bbvi.compute_entropy_posterior_predictive(ood_loader)
                elif dataset == 'fashion_mnist':
                    xood, yood = torch.load('./datasets/mnist_test.pt', map_location=device)
                    ood_loader = DataLoader(TensorDataset(xood, yood), batch_size=batch_size, shuffle=False)
                    ood = bbvi.compute_entropy_posterior_predictive(ood_loader)

            print('Accuracy:', np.round(acc, 3))
            print('Entropy:', np.round(entropy, 3))
            print('LPD:', np.round(lpd, 3))
            print('ECE:', np.round(ece, 3))
            print('MCE:', np.round(mce, 3))
            print('OOD:', np.round(ood, 3))
            result_table[f'K={K}, T={T}, low_rank={low_rank}'] = [acc, entropy, lpd, ece, mce, ood]


            for start_plot in start_plots:
                fig, axes = plt.subplots(3, 3, figsize=(15, 6))

                fig.suptitle(f'{dataset}_Theta_MAP plot from {start_plot} with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, MCE={mce:.3f}, LPD: {lpd:.3f}, OOD: {ood:.3f}, sigma_init=exp(-1), use_SWA={SWA}')

                log_like_series = pd.Series(bbvi.log_like_history)
                running_mean_log_like = log_like_series.rolling(window=500).mean().dropna()

                ELBO_series = pd.Series(bbvi.ELBO_history)
                running_mean_ELBO = ELBO_series.rolling(window=500).mean().dropna()

                log_prior_series = pd.Series(bbvi.log_prior_history)
                running_mean_log_prior = log_prior_series.rolling(window=500).mean().dropna()

                axes[0,0].plot(range(start_plot, start_plot + len(running_mean_log_like[start_plot:])), running_mean_log_like[start_plot:], label=f'log-likelihood/{T}')
                if dataset == 'mnist':
                    axes[0,0].axhline(y = mnist_MLE_log_like/T, color = 'r', label = f'MLE_log_like/{T}', linestyle = '--')
                    axes[0,0].set_ylim(bottom = np.min(running_mean_log_like[start_plot:])*1.1, top = mnist_MLE_log_like/T* 0.9)
                elif dataset == 'fashion_mnist':
                    axes[0,0].axhline(y = fashion_MLE_log_like/T, color = 'r', label = f'MLE_log_like/{T}', linestyle = '--')
                    axes[0,0].set_ylim(np.min(running_mean_log_like[start_plot:])*1.1, fashion_MLE_log_like/T*0.9)

                axes[1,0].plot(range(start_plot, start_plot + len(running_mean_log_like[start_plot:])), running_mean_log_prior[start_plot:], label='log-prior')
                axes[2,0].plot(range(start_plot, max_itt), bbvi.entropy_history[start_plot:], label='entropy')

                axes[0,1].plot(range(start_plot, start_plot + len(running_mean_log_like[start_plot:])), running_mean_ELBO[start_plot:], label='ELBO')
                axes[1,1].plot(range(start_plot, max_itt), bbvi.m_history[start_plot:, 0], '-', linewidth=0.5, label = "prior m")
                axes[2,1].plot(range(start_plot, max_itt), np.log(bbvi.v_history[start_plot:, 0]), '-', linewidth=0.5, label = "log(prior v)")
                axes[1,2].plot(range(start_plot, max_itt), bbvi.L_history[start_plot:, 0], '-', linewidth=0.5, label = "L")
                axes[2,1].axhline(y = np.log(prior_sigma))
                axes[2,1].set_ylim(bottom = -4, top = -1.5)
                axes[2,1].set_ylabel('log(v)')

                number_of_param_to_plot = min(K,50)

                for i in range(2,number_of_param_to_plot):
                    axes[1,1].plot(range(start_plot, max_itt),bbvi.m_history[start_plot:, i], '-', linewidth=0.5)
                    axes[2,1].plot(range(start_plot, max_itt), np.log(bbvi.v_history[start_plot:, i]), '-', linewidth=0.5)
                    axes[1,2].plot(range(start_plot, max_itt), bbvi.L_history[start_plot:, i], '-', linewidth=0.5)

                for i in range(9):
                    axes.flat[i].legend(loc = 'lower left')

                sns.heatmap((torch.diag(torch.exp(bbvi.v)) + bbvi.L @ bbvi.L.T).detach().cpu().numpy(), ax = axes[2,2])
                axes[2,2].set_xlabel('Heatmap of L @ L.T')
                

                if save_fig:
                    if SWA:
                        #plt.savefig('plots/simple_test_plot.png')
                        plt.savefig(f'plots/{dataset}_lowrank_{low_rank}_SWA_plotfrom_{start_plot}_K_{K}_T_{T}_batch_size_{batch_size}_step_size_{step_size}_acc_{acc:.3}_Entropy_{entropy:.3f}_ECE_{ece:.3f}_LPD_{lpd:.3f}_OOD_{ood:.2f}_MCE_{mce:.3f}_sigma_init_exp.png')
                    else:
                        plt.savefig(f'plots/{dataset}_lowrank_{low_rank}_No_SWA_plotfrom={start_plot}_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}_LPD={lpd:.3f}_OOD={ood:.2f}_MCE={mce:.3f}_sigma_init=exp(-1).png')

    if save_fig:
        if SWA:
            result_table.to_csv(f"results/{dataset}_lowrank_{low_rank}_SWA_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_iter_{max_itt}.csv")
        else:
            result_table.to_csv(f"results/{dataset}_lowrank_{low_rank}_No_SWA_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_iter_{max_itt}.csv")

