import os
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as pl
import time
import wandb
import pickle

from core.utils_GF import load_data, w2
import core.gradient_flow as gradient_flow
from db_tsw.utils import generate_trees_frames
import cfg
args = cfg.parse_args()
from tqdm import tqdm
# Configuration
dataset_name = args.dataset_name
nofiterations = args.num_iter
seeds = range(1,args.num_seeds+1)
modes = ['linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear']
titles = ['SW', 'TSW-SL-distance-based', 'TSW-SL-uniform', 'TSW-SL-orthorgonal', 'LCVSW', 'SWGG', 'MaxSW']
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink']

# Arrays to store results
results = {}
for title in titles:
    results[title] = {'raw_w2': np.zeros((nofiterations, len(seeds)))}

Xs = []
for i, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    N = 100  # Number of samples from p_X
    Xs.append(load_data(name=dataset_name, n_samples=N, dim=2))
    Xs[i] -= Xs[i].mean(dim=0)[np.newaxis, :]  # Normalization
lear_rates = [args.lr_sw, args.lr_tsw_sl, args.lr_tsw_sl, args.lr_tsw_sl, args.lr_sw, args.lr_sw, args.lr_sw, args.lr_sw]
n_projs = [args.L, int(args.L / args.n_lines), int(args.L / args.n_lines), int(args.L / args.n_lines), args.L, args.L, args.L, args.L]


for k, title in enumerate(titles):
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        X = Xs[i].detach().clone()
        meanX = 0
        _, d = X.shape

        # Construct folder name based on hyperparameters
        args_dict = vars(args)
        folder_info = '-'.join([f"{key.replace('_', '')}{value}" for key, value in args_dict.items()])
        results_folder = f"./Results_reduced/Gradient_Flow_{folder_info}/seed{seed}"
        os.makedirs(results_folder, exist_ok=True)
        # if not os.path.isdir(results_folder):
        #     os.mkdir(results_folder)
        # if os.path.exists(results_folder):
        #     shutil.rmtree(results_folder)

        foldername = os.path.join(results_folder, 'Gifs', dataset_name + '_Comparison')
        os.makedirs(foldername, exist_ok=True)

        # Use GPU if available, CPU otherwise
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the initial distribution
        temp = np.random.normal(loc=meanX, scale=.25, size=(N, d))

        # Define the variables to store the loss (2-Wasserstein distance)
        dist = 'w2'
        w2_dist = np.nan * np.zeros((nofiterations))

        # Define the optimizers and gradient flow objects
        Y = torch.tensor(temp, dtype=torch.float, device=device, requires_grad=True)
        optimizer = optim.Adam([Y], lr=lear_rates[k])
        gsw_res = gradient_flow.GF(ftype=modes[k], nofprojections=n_projs[k], device=device)

        # s = len(modes)
        # fig = pl.figure(figsize=(4 * s, 8 + 3))

        mean_X = torch.mean(X, dim=0, keepdim=True).to(device)
        std_X = torch.std(X, dim=0, keepdim=True).to(device)

        for t in tqdm(range(nofiterations)):
            theta = torch.ones(len(modes), d)

            loss = 0

            if k == 0:
                start_time = time.time()  # Start timing
                loss += gsw_res.sw(X.to(device), Y, theta=None)
                end_time = time.time()  # End timing
                # print(f"Time taken for SW: {end_time - start_time:.4f} seconds")

            elif k == 1:
                start_time = time.time()  # Start timing
                theta_twd, intercept_twd = generate_trees_frames(
                    ntrees=int(args.L / args.n_lines),
                    nlines=args.n_lines,
                    d=X.shape[1],
                    mean=mean_X,
                    std=args.std,
                    gen_mode='gaussian_raw',
                    device='cuda'
                )  # distance_based
                loss += gradient_flow.TWD(X=X.to(device), Y=Y, theta=theta_twd, intercept=intercept_twd, mass_division='distance_based', p=args.p, delta=args.delta)
                end_time = time.time()  # End timing
                # print(f"Time taken for TWD distance based: {end_time - start_time:.4f} seconds")

            elif k == 2:
                start_time = time.time()  # Start timing
                theta_twd, intercept_twd = generate_trees_frames(
                    ntrees=int(args.L / args.n_lines),
                    nlines=args.n_lines,
                    d=X.shape[1],
                    mean=mean_X,
                    std=args.std,
                    gen_mode='gaussian_raw',
                    device='cuda'
                )  # uniform
                loss += gradient_flow.TWD(X=X.to(device), Y=Y, theta=theta_twd, intercept=intercept_twd, mass_division='uniform', p=args.p)
                end_time = time.time()  # End timing
                # print(f"Time taken for TWD uniform: {end_time - start_time:.4f} seconds")

            elif k == 3:
                start_time = time.time()  # Start timing
                theta_twd, intercept_twd = generate_trees_frames(
                    ntrees=int(args.L / args.n_lines),
                    nlines=args.n_lines,
                    d=X.shape[1],
                    mean=mean_X,
                    std=args.std,
                    gen_mode='gaussian_orthogonal',
                    device='cuda'
                )  # orthogonal
                loss += gradient_flow.TWD(X=X.to(device), Y=Y, theta=theta_twd, intercept=intercept_twd, mass_division='distance_based', p=args.p, delta=args.delta)
                end_time = time.time()  # End timing
                # print(f"Time taken for TWD orthogonal: {end_time - start_time:.4f} seconds")

            elif k == 4:
                start_time = time.time()  # Start timing
                loss += gradient_flow.LCVSW(X.to(device), Y.to(device), L=args.L)
                end_time = time.time()  # End timing
                # print(f"Time taken for LCVSW: {end_time - start_time:.4f} seconds")

            elif k == 5:
                start_time = time.time()  # Start timing
                l, theta = gsw_res.SWGG_CP(X.to(device), Y.to(device), theta=None)
                loss += l
                end_time = time.time()  # End timing
                # print(f"Time taken for SWGG_CP: {end_time - start_time:.4f} seconds")

            elif k == 6:
                start_time = time.time()  # Start timing
                l, theta, loss_max = gsw_res.max_sw(X.to(device), Y, iterations=100, lr=lear_rates[k])
                loss += l
                end_time = time.time()  # End timing
                # print(f"Time taken for max SW: {end_time - start_time:.4f} seconds")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dist == 'w2':
                w2_dist[t] = w2(X.detach().cpu().numpy(), Y.detach().cpu().numpy())
            # if i == nofiterations - 1:
            #     pl.plot(np.log10(w2_dist[:, :]), linewidth=3)
            #     pl.title('2-Wasserstein Distance', fontsize=10)
            #     pl.ylabel(r'$Log_{10}(W_2)$', fontsize=22)
            #     pl.legend(titles, fontsize=10, loc='lower left')

            #     display.clear_output(wait=True)
            #     display.display(pl.gcf())
            #     time.sleep(1e-5)

            #     # Save the figure
            #     fig.savefig(foldername + '/img%03d.png' % (i))
            #     for k in range(s):
            #         pl.cla()

        results[title]['raw_w2'][:, i] = w2_dist

# np.savetxt(f"{results_folder}/{dataset_name}_SW_mean.txt", sw_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_TSW_SL_distance_based_mean.txt", tsw_sl_distance_based_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_TSW_SL_uniform_mean.txt", tsw_sl_uniform_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_TSW_SL_orthogonal_mean.txt", tsw_sl_orthogonal_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_LCV_SW_mean.txt", lcvsw_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_Max_SW_mean.txt", maxsw_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_SWGG_mean.txt", swgg_mean)
# np.savetxt(f"{results_folder}/{dataset_name}_SWGG_optim_mean.txt", swgg_optim_mean)


#Calculate the mean and standard deviation for each iteration
for title in titles:
    results[title]['mean'] = np.mean(results[title]['raw_w2'], axis=1)
    results[title]['std'] =  np.std(results[title]['raw_w2'], axis=1)
    results[title]['log10_w2'] = np.log10(results[title]['raw_w2'])
    results[title]['mean_log10'] =  np.mean(results[title]['log10_w2'], axis=1)
    results[title]['std_log10'] =  np.std(results[title]['log10_w2'], axis=1)

# save results to pickle
with open(f"{results_folder}/{dataset_name}_results.pkl", 'wb') as f:
    pickle.dump(results, f)

wandb.init(project="twd_gradient_flow", name = folder_info)
wandb.config.update(args)

for t in range(nofiterations):
    log_dict = {}
    for title in titles:
        log_dict[f'{title}/mean'] = results[title]['mean'][t]
        log_dict[f'{title}/mean_log10'] = results[title]['mean_log10'][t]
        log_dict[f'{title}/std'] = results[title]['std'][t]
        log_dict[f'{title}/std_log10'] = results[title]['std_log10'][t]
    wandb.log(log_dict)

if args.plot:
    # Plot the results
    pl.figure(figsize=(10, 6))

    # Plot SW with mean and shaded standard deviation (log scale)
    for i, title in enumerate(titles):
        pl.plot(results[title]['mean_log10'], label=title, color=colors[i])
        pl.fill_between(range(nofiterations), 
                        results[title]['mean_log10'] - results[title]['std_log10'], 
                        results[title]['mean_log10'] + results[title]['std_log10'], 
                        color=colors[i], alpha=0.2)

    # Add text box with argument information
    # Prepare the text box content without dataset_name
    args_info = [f'{key.replace("_", " ").capitalize()}: {value}' for key, value in args_dict.items()]

    # Join the list into a single string with newline separation
    textstr = '\n'.join(args_info)

    # Place a text box with argument information
    pl.gca().text(0.05, 0.95, textstr, transform=pl.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Finalize the plot with dataset name in the title
    pl.title(f'Log Wasserstein Distance over 10 Runs ({dataset_name})')
    pl.xlabel('Iterations')
    pl.ylabel(r'$W_2$ Distance (log scale)')
    pl.legend()
    pl.grid(True)

    plot_filename = os.path.join(results_folder, f'{folder_info}_log.png')
    pl.savefig(plot_filename)
    pl.clf()

    wandb.log({'Image': [wandb.Image(plot_filename)]})

wandb.finish()
