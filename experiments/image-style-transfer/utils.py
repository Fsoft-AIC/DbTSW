import numpy as np
import torch
from torch.autograd import Variable
import ot
import random
from scipy.stats import norm
from db_tsw.db_tsw import TWConcurrentLines
from db_tsw.utils import generate_trees_frames
from torch import optim

# comment for wandb rate limit optimization
# from tqdm import tqdm
def tqdm(x):
    return x

class Tree():
    def __init__(self, L, nlines, d, mean=128, std=0.1, device='cuda', fixed_trees = True):
        self.L = L
        self.d = d
        
        self.nlines = nlines
        self.device = device
        self.fixed_trees = fixed_trees
        self.mean = mean
        self.std = std
        self.already_generate = 1
        if self.fixed_trees == True:
            self.already_generate = 0

    def get_tree(self):
        if self.fixed_trees:
            print("Executing here")
            if self.already_generate:
                return self.theta, self.intercept
            else:
                self.already_generate = 1
                self.generate_trees()
                return self.theta, self.intercept
        else:
            self.generate_trees()
            return self.theta, self.intercept
    def generate_trees(self):
        self.theta, self.intercept = generate_trees_frames(self.L, self.nlines, self.d, self.mean, self.std)


def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cuda',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def rand_projections(dim, num_projections=1000,device='cuda'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X.to('cuda'), theta.transpose(0, 1))
    Y_prod = torch.matmul(Y.to('cuda'), theta.transpose(0, 1))
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


def SW(X, Y, theta = None, L=10, p=2,device="cuda"):
    dim = X.size(1)
    if theta is None:
        theta = rand_projections(dim, L, device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)

def TWD(X, Y, theta, intercept, mass_division = 'distance_based', p = 2, delta = 1., device = 'cuda'):
    # print(p)
    # print(delta)
    # exit()
    L = theta.shape[0]
    nlines = theta.shape[1]
    TWD_obj = TWConcurrentLines(p=p, delta=delta, mass_division=mass_division, device=device)
    return TWD_obj(X, Y, theta, intercept)

# def max_tw(X,Y,n_lines,iterations=50,lr=1e-4, device="cuda"):
#     N,dn = X.shape
#     M,dm = Y.shape
#     assert dn==dm and M==N
#     theta=torch.randn((1,n_lines,dn),device=device,requires_grad=True)
#     theta.data/=torch.sqrt(torch.sum((theta.data)**2))
#     theta, intercept, subsequent_sources = TW.generate_trees_frames(L = 1, d = dn, theta = theta, range_root_1=-1.0, range_root_2=1.0, range_source_1=-0.1, range_source_2=0.1, nlines=n_lines, device='cuda', type_lines='sequence_of_lines')

#     optimizer=optim.Adam([theta],lr=lr)
#     loss_l=[]
#     for i in range(iterations):
#         optimizer.zero_grad()
#         loss=- 1000 * TWD(X.to(device),Y.to(device), theta.to(device), intercept.to(device), subsequent_sources.to(device))
#         #print('test4')
#         loss_l.append(loss.data)
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         theta.data/=torch.norm(theta.data)
#         theta, intercept, subsequent_sources = TW.generate_trees_frames(L = 1, d = dn, theta = theta.data, range_root_1=-1.0, range_root_2=1.0, range_source_1=-0.1, range_source_2=0.1, nlines=n_lines, device='cuda', type_lines='sequence_of_lines')

#         #print('test5')

#     res = 1000 * TWD(X.to(device),Y.to(device),theta.to(device), intercept, subsequent_sources)
#     return res

def max_sw(X,Y,iterations=50,lr=1e-4, device="cuda"):
    N,dn = X.shape
    M,dm = Y.shape
    assert dn==dm and M==N
    theta=torch.randn((1,dn),device=device,requires_grad=True)
    theta.data/=torch.sqrt(torch.sum((theta.data)**2))
    optimizer=optim.Adam([theta],lr=lr)
    loss_l=[]
    for i in range(iterations):
        optimizer.zero_grad()
        loss=-SW(X.to(device),Y.to(device), theta.to(device))
        #print('test4')
        loss_l.append(loss.data)
        loss.backward(retain_graph=True)
        optimizer.step()
        theta.data/=torch.norm(theta.data)
        #print('test5')

    res = SW(X.to(device),Y.to(device),theta.to(device))
    return res

def LCVSW(X,Y,L=10,p=2,device="cuda"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    diff_m1_m2 = diff_m1_m2.to('cuda')
    G_mean = torch.mean((diff_m1_m2)**2) #+ (sigma_1-sigma_2)**2
    theta = rand_projections(dim, L, device).to('cuda')
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 #+(sigma_1-sigma_2)**2
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)

def UCVSW(X,Y,L=10,p=2,device="cuda"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    diff_m1_m2 = diff_m1_m2.to('cuda')
    diff_X_m1 = X-m_1
    diff_X_m1 =diff_X_m1.to('cuda')
    diff_Y_m2 = Y-m_2
    diff_Y_m2 = diff_Y_m2.to('cuda')
    G_mean = torch.mean((diff_m1_m2)**2) +  torch.mean((diff_X_m1)**2)+  torch.mean((diff_Y_m2)**2)
    theta = rand_projections(dim, L, device).to('cuda')
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 +torch.mean(torch.matmul(theta,diff_X_m1.transpose(0,1))**2,dim=1)+torch.mean(torch.matmul(theta,diff_Y_m2.transpose(0,1))**2,dim=1)
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)




def transform_SW(src,target,src_label,origin,sw_type='sw',L=10,num_iter=1000, lr_tw = 0.01, num_iter_tw = 2000, n_trees_tw = 25, n_lines_tw = 4, delta = 1., std = 0.1):
    
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    device='cuda'
    s = np.array(src).reshape(-1, 3)
    s = torch.from_numpy(s).float()
    s = torch.nn.parameter.Parameter(s)
    t = np.array(target).reshape(-1, 3)
    t = torch.from_numpy(t).float()
    mean_t = torch.mean(t, dim=0, keepdim=True).to(device)
    opt = torch.optim.SGD([s], lr=1.)
    if (sw_type == 'nqsw' or sw_type == 'rnqsw' or sw_type == 'rrnqsw'  ):
        soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
        theta = soboleng.draw(L)
        theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
        theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
        theta = theta.to(device)
        theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
    elif(sw_type=='qsw' or sw_type=='rqsw' or sw_type=='rrqsw'):
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
        net = soboleng.draw(L)
        alpha = net[:, [0]]
        tau = net[:, [1]]
        theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                           2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
            device)
    elif(sw_type=='sqsw' or sw_type=='rsqsw'):
        Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
        theta1 = torch.arccos(Z)
        theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        theta = torch.cat(
            [torch.sin(theta1) * torch.cos(theta2), torch.sin(theta1) * torch.sin(theta2), torch.cos(theta1)],
            dim=1)
        theta = theta.to(device)
    elif(sw_type=='odqsw' or sw_type=='rodqsw'):
        Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
        theta1 = np.arccos(Z)
        theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)],
                                axis=1)
        theta0 = torch.from_numpy(thetas)
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(100):
            loss = - torch.cdist(thetas, thetas, p=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        theta = thetas.to(device).float()
    elif (sw_type == 'ocqsw' or sw_type=='rocqsw'):
        Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
        theta1 = np.arccos(Z)
        theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)],
                                axis=1)
        theta0 = torch.from_numpy(thetas)
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(100):
            loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        theta = thetas.to(device).float()
    if(sw_type == 'twd'):
        opt_twd_distance = torch.optim.SGD([s], lr=lr_tw)
        for _ in tqdm(range(num_iter_tw)):
            opt_twd_distance.zero_grad()
            theta_twd, intercept_twd = generate_trees_frames(ntrees=n_trees_tw, nlines=n_lines_tw, d=s.shape[1], mean=mean_t, std = std, gen_mode = 'gaussian_raw', device='cuda')
            g_loss_twd_distance = TWD(s, t, theta_twd, intercept_twd, mass_division = 'distance_based', delta = delta)
            g_loss_twd_distance =torch.sqrt(g_loss_twd_distance.mean())
            g_loss_twd_distance = g_loss_twd_distance * s.shape[0]
            opt_twd_distance.zero_grad()
            g_loss_twd_distance.backward()
            opt_twd_distance.step()
            s.data = torch.clamp(s, min=0)
            
    if(sw_type == 'twd_ortho'):
        opt_twd_ortho = torch.optim.SGD([s], lr=lr_tw)
        for _ in tqdm(range(num_iter_tw)):
            opt_twd_ortho.zero_grad()
            theta_twd, intercept_twd = generate_trees_frames(ntrees=n_trees_tw, nlines=n_lines_tw, d=s.shape[1], mean=mean_t, std = std, gen_mode = 'gaussian_orthogonal', device='cuda')
            g_loss_twd_ortho = TWD(s, t, theta_twd, intercept_twd, mass_division = 'distance_based', delta = delta)
            g_loss_twd_ortho =torch.sqrt(g_loss_twd_ortho.mean())
            g_loss_twd_ortho = g_loss_twd_ortho * s.shape[0]
            opt_twd_ortho.zero_grad()
            g_loss_twd_ortho.backward()
            opt_twd_ortho.step()
            s.data = torch.clamp(s, min=0)
                
    if(sw_type == 'twd_uniform'):
        opt_twd_uniform = torch.optim.SGD([s], lr=lr_tw)
        for _ in tqdm(range(num_iter_tw)):
            opt_twd_uniform.zero_grad()
            theta_twd, intercept_twd = generate_trees_frames(ntrees=n_trees_tw, nlines=n_lines_tw, d=s.shape[1], mean=mean_t, std = std, device='cuda')
            g_loss_twd_uniform = TWD(s, t, theta_twd, intercept_twd, mass_division = 'uniform', delta = delta)
            g_loss_twd_uniform =torch.sqrt(g_loss_twd_uniform.mean())
            
            g_loss_twd_uniform = g_loss_twd_uniform*s.shape[0]
            opt_twd_uniform.zero_grad()
            g_loss_twd_uniform.backward()
            opt_twd_uniform.step()
            s.data = torch.clamp(s, min=0)
    if sw_type != 'twd' and sw_type != 'twd_uniform' and sw_type != 'twd_ortho':
        for _ in tqdm(range(num_iter)):
            opt.zero_grad()
            if (sw_type == 'sw'):
                g_loss = SW(s, t, L=L,p=2)
            elif(sw_type == 'lcvsw'):
                g_loss = LCVSW(s, t, L=L,p=2)

            elif(sw_type == 'ucvsw'):
                g_loss = UCVSW(s, t, L=L,p=2)

            elif (sw_type == 'maxsw'):
                g_loss = max_sw(s, t)

            elif(sw_type=='nqsw' or sw_type=='qsw' or sw_type=='sqsw' or sw_type=='ocqsw' or sw_type=='odqsw'):
                g_loss=one_dimensional_Wasserstein_prod(s,t,theta,p=2)
            elif(sw_type=='rnqsw'):
                soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
                theta = soboleng.draw(L)
                theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
                theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
                theta = theta.to('cuda')
                theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
                g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
            elif(sw_type=='rqsw'):
                soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
                net = soboleng.draw(L)
                alpha = net[:, [0]]
                tau = net[:, [1]]
                theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                                2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
                    device)
                g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
            elif (sw_type == 'rrnqsw' or sw_type == 'rrqsw' or sw_type == 'rsqsw' or sw_type == 'rocqsw' or sw_type == 'rodqsw'):
                U = torch.qr(torch.randn(3, 3))[0]
                thetaprime = torch.matmul(theta.to('cuda'), U.to('cuda'))
                g_loss = one_dimensional_Wasserstein_prod(s, t, thetaprime, p=2)
            g_loss =torch.sqrt(g_loss.mean())
            g_loss = g_loss*s.shape[0]
            opt.zero_grad()
            g_loss.backward()
            opt.step()
            s.data = torch.clamp(s, min=0)

    s = torch.clamp( s,min=0).cpu().detach().numpy()
    img_ot_transf = s[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype("uint8")
    return s, img_ot_transf

