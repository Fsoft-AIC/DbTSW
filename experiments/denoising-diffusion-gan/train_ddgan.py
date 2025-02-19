# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np

import os

import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from power_spherical import PowerSpherical
import torch.multiprocessing as mp
import torch.distributed as dist
import shutil
import time

import wandb
from db_tsw.db_tsw import TWConcurrentLines
from db_tsw.utils import generate_trees_frames

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x

#%%
            
def train(rank, args):
    def rand_projections(dim, num_projections=1000,device='cpu'):
        projections = torch.randn((num_projections, dim),device=device)
        projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
        return projections


    def one_dimensional_Wasserstein_prod(X,Y,theta,p):
        X_prod = torch.matmul(X, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y, theta.transpose(0, 1))
        X_prod = X_prod.view(X_prod.shape[0], -1)
        Y_prod = Y_prod.view(Y_prod.shape[0], -1)
        wasserstein_distance = torch.abs(
            (
                    torch.sort(X_prod, dim=0)[0]
                    - torch.sort(Y_prod, dim=0)[0]
            )
        )
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
        return wasserstein_distance

    def SW(X, Y, L=10, p=2, device="cuda"):
        dim = X.size(1)
        theta = rand_projections(dim, L,device)
        sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
        return  torch.pow(sw,1./p)
    def RPSW(X, Y, L=10, p=2, device="cpu",kappa=50):
        dim = X.size(1)
        theta = (X.detach()[np.random.choice(X.shape[0], L, replace=True)] - Y.detach()[np.random.choice(Y.shape[0], L, replace=True)])
        theta = theta/ torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).detach()
        ps = PowerSpherical(
            loc=theta,
            scale=torch.full((theta.shape[0],),kappa,device=device),
        )
        theta = ps.rsample()
        sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
        return torch.pow(sw, 1. / p)
    def EBRPSW(X, Y, L=10, p=2, device="cpu",kappa=50):
        dim = X.size(1)
        theta = (X.detach()[np.random.choice(X.shape[0], L, replace=True)] - Y.detach()[np.random.choice(Y.shape[0], L, replace=True)])
        theta = theta/ torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).detach()
        ps = PowerSpherical(
            loc=theta,
            scale=torch.full((theta.shape[0],),kappa,device=device),
        )
        theta = ps.rsample()
        wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
        wasserstein_distances =  wasserstein_distances.view(1,L)
        weights = torch.softmax(wasserstein_distances,dim=1)
        sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
        return torch.pow(sw, 1. / p)
    def DSW(X,Y,L=10,kappa=50,p=2,s_lr=0.1,n_lr=100,device="cpu"):
        dim = X.size(1)
        epsilon = torch.randn((1, dim), device=device, requires_grad=True)
        epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1,keepdim=True))
        optimizer = torch.optim.SGD([epsilon], lr=s_lr)
        X_detach = X.detach()
        Y_detach = Y.detach()
        for _ in range(n_lr-1):
            vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
            theta = vmf.rsample((L,)).view(L, -1)
            negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach,Y_detach,theta,p=p).mean(),1./p)
            optimizer.zero_grad()
            negative_sw.backward()
            optimizer.step()
            epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1,keepdim=True))
        vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
        theta = vmf.rsample((L,)).view(L, -1)
        sw = one_dimensional_Wasserstein_prod(X, Y,theta, p=p).mean()
        return torch.pow(sw,1./p)
    def MaxSW(X,Y,p=2,s_lr=0.1,n_lr=100,device="cpu",adam=False):
        dim = X.size(1)
        theta = torch.randn((1, dim), device=device, requires_grad=True)
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
        if(adam):
            optimizer = torch.optim.Adam([theta], lr=s_lr)
        else:
            optimizer = torch.optim.SGD([theta], lr=s_lr)
        X_detach = X.detach()
        Y_detach = Y.detach()
        for _ in range(n_lr-1):
            negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach,Y_detach,theta,p=p).mean(),1./p)
            optimizer.zero_grad()
            negative_sw.backward()
            optimizer.step()
            theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
        sw = one_dimensional_Wasserstein_prod(X, Y,theta, p=p).mean()
        return torch.pow(sw,1./p)
    
    def ISEBSW(X, Y, L=10, p=2, device="cpu"):
        dim = X.size(1)
        theta = rand_projections(dim, L,device)
        wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
        wasserstein_distances =  wasserstein_distances.view(1,L)
        weights = torch.softmax(wasserstein_distances,dim=1)
        sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
        return  torch.pow(sw,1./p)
    
    def CLTWD(X, Y, theta, intercept, TWD_obj):
        return TWD_obj(X, Y, theta, intercept)
        
    
    from score_sde.models.discriminator import Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    
    if rank == 0:
        run_name = args.wandb_run_name or args.exp
        wandb.init(project=args.wandb_project_name, 
                entity=args.wandb_entity,
                name = run_name,)
        wandb.config.update(args)
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(rank))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    if args.dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        # calculate mean of pixel values of the dataset
    else:
        raise NotImplementedError('Dataset not implemented')

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    netG = NCSNpp(args).to(device)
    

    if args.dataset == 'cifar10':
        netD = Discriminator_small(nc = 2*args.num_channels, ngf = args.ngf,
                               t_emb_dim = args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)
    
    #ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[rank])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[rank])
    
    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir,exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
    
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    if args.dataset == 'cifar10':
        x_t_1 = torch.randn((64,3,32,32),device=device)
    print(args.dataset)
    
    if args.loss == 'cltwd':
        # calculate mean of pixel values of the dataset
        pixel_mean = 0
        num_sample = 0
        d = 0
        for i, (x, _) in enumerate(data_loader):
            pixel_mean += x.mean()
            num_sample += x.shape[0]
            d = x.shape[1] * x.shape[2] * x.shape[3] + 1
            
        pixel_mean = pixel_mean / num_sample
        TWD_obj = torch.compile(TWConcurrentLines(p=2, delta=args.twd_delta, mass_division='distance_based', device=device))
        
    start_time = time.time()
    for epoch in range(init_epoch, args.num_epoch+1):
        if rank == 0:
            print(f"Epoch {epoch - 1}: {time.time() - start_time}s")
            start_time = time.time()
        train_sampler.set_epoch(epoch)
        if args.loss == 'cltwd':
            theta, intercept = generate_trees_frames(ntrees=args.T, nlines=args.L, d=d, 
                                                     mean=pixel_mean, std=args.twd_std, device=device, gen_mode=args.twd_gen_mode)
        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():  
                p.requires_grad = True  
            netD.zero_grad()
            
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            #sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            
    
            # train with real
            D_real,_ = netD(x_t, t, x_tp1.detach())
            
            errD_real = F.softplus(-D_real.view(-1))
            errD_real = errD_real.mean()
            
            errD_real.backward(retain_graph=True)
            
            
            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                    grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            
         
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output,_ = netD(x_pos_sample, t, x_tp1.detach())
                
            
            errD_fake = F.softplus(output.view(-1))
            errD_fake = errD_fake.mean()
            errD_fake.backward()
    
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            D_real,real_feature = netD(x_t, t, x_tp1.detach())
            
            #errD_real = F.softplus(-D_real.view(-1))
            
            latent_z = torch.randn(batch_size, nz,device=device)
           
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output,fake_feature = netD(x_pos_sample, t, x_tp1.detach())

            if args.dataset=='cifar10':
                X = torch.cat([x_t.view(x_t.shape[0],-1),D_real.view(D_real.shape[0],-1)],dim=1)
                Y = torch.cat([x_pos_sample.view(x_pos_sample.shape[0],-1),output.view(output.shape[0],-1)],dim=1)
            index = int(X.shape[0]/2)

            X1,X2,Y1,Y2 = X[:index],X[index:],Y[:index],Y[index:]
            n_min = np.min([X1.shape[0],X2.shape[0]])
            X1,X2,Y1,Y2 = X1[:n_min],X2[:n_min],Y1[:n_min],Y2[:n_min]
            if(args.loss=='sw'):
                errG = 2*SW(X,Y,L=args.L,device=device) - SW(X1,X2,L=args.L,device=device)- SW(Y1,Y2,L=args.L,device=device)
            elif(args.loss=='rpsw'):
                w=((args.num_epoch-epoch-1)/(args.num_epoch-1))**args.beta
                kappa = args.kappa*w + args.kappa2*(1-w)
                errG = 2*RPSW(X,Y,L=args.L,kappa=kappa,device=device) - RPSW(X1,X2,L=args.L,kappa=kappa,device=device)- RPSW(Y1,Y2,L=args.L,kappa=kappa,device=device)
            elif(args.loss=='iwrpsw'):
                w=((args.num_epoch-epoch-1)/(args.num_epoch-1))**args.beta
                kappa = args.kappa*w + args.kappa2*(1-w)
                errG = 2*EBRPSW(X,Y,L=args.L,kappa=kappa,device=device) - EBRPSW(X1,X2,L=args.L,kappa=kappa,device=device)- EBRPSW(Y1,Y2,L=args.L,kappa=kappa,device=device)
            elif(args.loss=='dsw'):
                w=((args.num_epoch-epoch-1)/(args.num_epoch-1))**args.beta
                kappa = args.kappa*w + args.kappa2*(1-w)
                errG = 2*DSW(X,Y,n_lr=args.T,L=args.L,kappa=kappa,device=device) - DSW(X1,X2,n_lr=args.T,L=args.L,kappa=kappa,device=device)- DSW(Y1,Y2,n_lr=args.T,L=args.L,kappa=kappa,device=device)
            elif(args.loss=='maxsw'):
                errG = 2*MaxSW(X,Y,n_lr=args.L,device=device) - MaxSW(X1,X2,n_lr=args.L,device=device)- MaxSW(Y1,Y2,n_lr=args.L,device=device)
            elif(args.loss=='ebsw'):
                errG = 2*ISEBSW(X,Y,L=args.L,device=device) - ISEBSW(X1,X2,L=args.L,device=device)- ISEBSW(Y1,Y2,L=args.L,device=device)
            elif(args.loss=='gan'):
                errG = F.softplus(-output.view(-1))
                errG = errG.mean()
            elif(args.loss=='cltwd'):
                errG = 2*CLTWD(X,Y,theta,intercept,TWD_obj) - CLTWD(X1,X2,theta,intercept,TWD_obj) - CLTWD(Y1,Y2,theta,intercept,TWD_obj)
            errG.backward()
            optimizerG.step()
                
            global_step += 1
            if iteration % 100 == 0:
                log_tensor = torch.tensor([errG.item(), errD.item()], device=device)
                torch.distributed.reduce(log_tensor, dst=0)
                dist.barrier()
                if rank == 0:
                    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
                    g_loss = log_tensor[0].item() / local_world_size
                    d_loss = log_tensor[1].item() / local_world_size
                    log_dict = {
                        'train/epoch': epoch,
                        'train/iteration': iteration,
                        'train/G_loss': g_loss,
                        'train/D_loss': d_loss,
                    }
                    wandb.log(log_dict)
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch, iteration, g_loss, d_loss))
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
        
        # synchronize all processes, ensure all processes have finished the current epoch
        dist.barrier()
        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
            
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            if args.dataset == 'cifar10':
                n_row=8
            elif args.dataset == 'celeba':
                n_row=4
            elif args.dataset == 'celebahq':
                n_row=2
            sample_path = os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch))
            torchvision.utils.save_image(fake_sample, sample_path, normalize=True,nrow = n_row)
            wandb.log({'train/sample': [wandb.Image(sample_path)]}, commit=False)
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                
                    
def init_processes():
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"Rank {os.environ['LOCAL_RANK']} process created")
    
    
def main(args):
    init_processes()
    
    train(int(os.environ["LOCAL_RANK"]), args)
    
    cleanup()

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
   
    ###optimal transport
    parser.add_argument('--loss', type=str, default='sw',
                        help='sw')
    parser.add_argument('--L', type=int, default=100,
                        help='Number of lines of each tree in CLTWD.')
    parser.add_argument('--T', type=int, default=100,
                        help='Number of trees in CLTWD.')
    parser.add_argument('--twd_delta', type=int, default=1,
                        help='Softmax temperature term in CLTWD.')
    parser.add_argument('--kappa', type=float, default=100,
                        help='L')
    parser.add_argument('--kappa2', type=float, default=100,
                        help='L')
    parser.add_argument('--beta', type=float, default=10,
                        help='beta')
    parser.add_argument('--twd_gen_mode', type=str, 
                        choices=["gaussian_raw", "gaussian_orthogonal"], 
                        help='tree generation mode, further information in `generate_tree_frames` method')
    parser.add_argument('--twd_std', type=float, default=0.1, help="std of the tree generation")
    # wandb-related parameters
    parser.add_argument('--wandb_project_name', type=str, default='twd', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='twd', help='wandb entity, username or team name')
    parser.add_argument('--wandb_run_name', type=str, default=None, 
                        help='wandb run name, if not specified, will be set to the current experiment name')

   
    args = parser.parse_args()
    
    main(args)

   
                