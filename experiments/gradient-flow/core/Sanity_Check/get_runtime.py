import torch
import argparse
import time
import ot
import numpy as np
import cupy as cp
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tqdm.auto import trange


import os
import sys
sys.path.append('..')
from sw import sliced_wasserstein, max_sw


from SWGG import get_SWGG_smooth,SWGG_smooth
from SWGG import SWGG_CP

from SRW import SubspaceRobustWasserstein
# from Optimization.projectedascent import ProjectedGradientAscent
from Optimization.frankwolfe import FrankWolfe

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="number of restart")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

mu0 = torch.tensor([1,0,0], dtype=torch.float, device=device)
Sigma0 = torch.eye(2, dtype=torch.float, device=device)

ntry = args.ntry

ds = [3, 20, 100]
samples = [int(1e2),int(1e3),int(1e4),int(1e5/2),int(1e5)] #,int(1e6/2)]
projs = [200, 500]

L_swggmc = np.zeros((len(ds), len(projs), len(samples), ntry+1))
L_sw = np.zeros((len(ds), len(projs), len(samples), ntry+1))
L_sd = np.zeros((len(ds), len(projs), len(samples), ntry+1))

L_swggoptim = np.zeros((len(ds), len(samples), ntry+1))
L_maxsw = np.zeros((len(ds), len(samples), ntry+1))
L_srw = np.zeros((len(ds), len(samples), ntry+1))

L_fc = np.zeros((len(ds), len(samples), ntry+1))
L_w = np.zeros((len(ds), len(samples), ntry+1))
L_s = np.zeros((len(ds), len(samples), ntry+1))





if __name__ == "__main__":    
    for i, d in enumerate(ds):
        gaussian = D.MultivariateNormal(torch.zeros(d, device=device), torch.eye(d, device=device))
        for k, n_samples in enumerate(samples):
            x0 = gaussian.sample((n_samples,))
            x1 = gaussian.sample((n_samples,))
            
            if args.pbar:
                bar = trange(ntry+1)
            else:
                bar = range(ntry+1)

            print(d, n_samples, flush=True)
            print(L_swggmc)
            print(L_swggoptim)
                
            for j in bar:
                for l, n_projs in enumerate(projs):
#                     print(x0.device, x1.device)
#                     sw = ot.sliced.sliced_wasserstein_distance(x0, x1, n_projections=n_projs, p=2)
                    sw = sliced_wasserstein(x0, x1, n_projs, device, p=2)
                    try:
                        t0 = time.time()
#                         sw = ot.sliced.sliced_wasserstein(x0, x1, n_projections=n_projs, p=2)
                        sw = sliced_wasserstein(x0, x1, n_projs, device, p=2)
                        L_sw[i, l, k, j] = time.time()-t0
#                         print(time.time()-t0)
                    except:
                        L_sw[i,l,k,j] = np.inf
                        
                                            
                    try:
                        t0 = time.time()
                        
                        ## A wrapper dans un fonction ?
#                         theta = random_slice(n_projs, d).T
#                         W, _, _ = upperW2(x0, x1, theta.to(x0.device))
#                         sw2 = torch.min(W)
                        swgg = min_swgg(x0, x1, n_projs)
                        L_swggmc[i, l, k, j] = time.time()-t0
                    except:
                        L_swggmc[i,l,k,j] = np.inf        
                        
                        
                try:
                    t0 = time.time()
                    fc = ot.factored.factored_optimal_transport(x0, x1)
                    L_fc[i, k, j] = time.time()-t0
                except:
                    L_fc[i, k, j] = np.inf

                try:
                    t2 = time.time()

                    a = torch.ones((n_samples,), device=device)/n_samples
                    b = torch.ones((n_samples,), device=device)/n_samples
                    M = ot.dist(x0, x1)**2
                    w = ot.sinkhorn2(a, b, M/M.max(), reg=1, numitermax=10000, stopThr=1e-15)

                    L_s[i,k,j] = time.time()-t2
                except:
                    L_s[i,k,j] = np.inf

                try:
                    t1 = time.time()

                    a = torch.ones((n_samples,), device=device)/n_samples
                    b = torch.ones((n_samples,), device=device)/n_samples
                    M = ot.dist(x0, x1)**2
                    w = ot.emd2(a, b, M)

                    L_w[i,k,j] = time.time()-t1
                except:
                    L_w[i,k,j] = np.inf
                    
                    
                try:
                    t1 = time.time()
                    sw, _, _ = max_sw(x0, x1, iterations=100, lr=1)
                    L_maxsw[i,k,j] = time.time()-t1
                except:
                    L_maxsw[i,k,j] = np.inf
                    

                try:
                    t0 = time.time()
                    theta, loss_smooth_l, _ = get_minSW_smooth(x0, x1, lr=1.0, num_iter=100, 
                                                               s=50, std=0.5, bar=False)
                    theta.requires_grad = False
                    theta_optim = theta/torch.norm(theta)

                    sw = upperW2_smooth(x0, x1, theta_optim, s=1, std=0)                    
                    L_swggoptim[i, k, j] = time.time()-t0
                except:
                    L_swggoptim[i, k, j] = np.inf
                    
                    
                try:
                    reg = 0.2 # Entropic regularization strengh
                    max_iter = 1000 # Maximum number of iterations
                    max_iter_sinkhorn = 30 # Maximum number of iterations in Sinkhorn
                    threshold = 0.05 # Stopping threshold
                    threshold_sinkhorn = 1e-3 # Stopping threshold in Sinkhorn
                    
                    X = cp.array(x0.cpu().numpy())
                    Y = cp.array(x1.cpu().numpy())
                    
                    a = cp.ones(n_samples)/n_samples
                    b = cp.ones(n_samples)/n_samples
                    
                    t0 = time.time()      
                    algo = FrankWolfe(reg=reg, step_size_0=None, max_iter=max_iter, 
                                      max_iter_sinkhorn=max_iter_sinkhorn, threshold=threshold, 
                                      threshold_sinkhorn=threshold_sinkhorn, use_gpu=True)

                    SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, 2)
                    SRW.run()
                    L_srw[i, k, j] = time.time()-t0
                except:
                    L_srw[i, k, j] = np.inf
                    
                    
    for i, d in enumerate(ds):
        for l, n_projs in enumerate(projs):
            np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_SW_projs_"+str(n_projs)+"_d"+str(d), L_sw[i, l, :, 1:])
            np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_SWGGMC_projs_"+str(n_projs)+"_d"+str(d), L_swggmc[i, l, :, 1:])

        np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_FC_d"+str(d), L_fc[i,:,1:])
        np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_SW_W_d"+str(d), L_w[i,:,1:])
        np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_SW_Sinkhorn_d"+str(d), L_s[i,:,1:])
        np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_SWGGOptim_d"+str(d), L_swggoptim[i,:,1:])
        np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_maxSW_d"+str(d), L_maxsw[i,:,1:])
        np.savetxt("../../Notebook/Results/Sanity_Check/Comparison_SRW_d"+str(d), L_srw[i,:,1:])

