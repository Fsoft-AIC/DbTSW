import numpy as np

import torch
from torch import optim
import ot
from db_tsw.db_tsw import TWConcurrentLines
from db_tsw.utils import generate_trees_frames

class Tree():
    def __init__(self, L, nlines, d, mean=128, std=0.1, device='cuda', gen_mode = 'gaussian_raw', fixed_trees = True):
        self.L = L
        self.d = d
        
        self.nlines = nlines
        self.device = device
        self.fixed_trees = fixed_trees
        self.mean = mean
        self.std = std
        self.already_generate = 1
        self.gen_mode = gen_mode
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
        self.theta, self.intercept = generate_trees_frames(self.L, self.nlines, self.d, self.mean, self.std, gen_mode = self.gen_mode)

class GF():
    def __init__(self,ftype='linear',nofprojections=10,device='cpu'):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.device=device
        self.theta=None # This is for max-GSW

    def sw(self,X,Y,theta=None):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        return torch.sum((Xslices_sorted-Yslices_sorted)**2)
    
    def SWGG_CP(self,X,Y,theta):
        n,dn=X.shape
        if theta is None:
            theta=self.random_slice(dn).T

        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)

        X_line_sort, u = torch.sort(X_line, axis=0)
        Y_line_sort, v = torch.sort(Y_line, axis=0)
        
        W=torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0)
        idx=torch.argmin(W)
        return W[idx],theta[:,idx]
        
    def PWD(self,X,Y,theta):
        n,dn=X.shape
        if theta is None:
            theta=self.random_slice(dn).T

        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)

        X_line_sort, u = torch.sort(X_line, axis=0)
        Y_line_sort, v = torch.sort(Y_line, axis=0)
        
        W=torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0)
        return torch.mean(W)


##### GRADIENT DESCENT ######
    def SWGG_smooth(self,X,Y,theta,s=1,std=0):
        n,dim=X.shape
    
        X_line=torch.matmul(X,theta)
        Y_line=torch.matmul(Y,theta)
    
        X_line_sort,u=torch.sort(X_line,axis=0)
        Y_line_sort,v=torch.sort(Y_line,axis=0)
    
        X_sort=X[u]
        Y_sort=Y[v]
    
        Z_line=(X_line_sort+Y_line_sort)/2
        Z=Z_line[:,None]*theta[None,:]
    
        W_XZ=torch.sum((X_sort-Z)**2)/n
        W_YZ=torch.sum((Y_sort-Z)**2)/n
    
        X_line_extend = X_line_sort.repeat_interleave(s,dim=0)
        X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape,device=self.device)
        Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
        Y_line_extend_blur = Y_line_extend + 0.5 * std * torch.randn(Y_line_extend.shape,device=self.device)
    
        X_line_extend_blur_sort,u_b=torch.sort(X_line_extend_blur,axis=0)
        Y_line_extend_blur_sort,v_b=torch.sort(Y_line_extend_blur,axis=0)

        X_extend=X_sort.repeat_interleave(s,dim=0)
        Y_extend=Y_sort.repeat_interleave(s,dim=0)
        X_sort_extend=X_extend[u_b]
        Y_sort_extend=Y_extend[v_b]
    
        bary_extend=(X_sort_extend+Y_sort_extend)/2
        bary_blur=torch.mean(bary_extend.reshape((n,s,dim)),dim=1)
    
        W_baryZ=torch.sum((bary_blur-Z)**2)/n
        return -4*W_baryZ+2*W_XZ+2*W_YZ

    def get_minSWGG_smooth(self,X,Y,lr=1e-2,num_iter=100,s=1,std=0,init=None):
        if init is None :
             theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
        else :
            theta=torch.tensor(init,device=X.device, dtype=X.dtype,requires_grad=True)
        
        #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
        optimizer = torch.optim.SGD([theta], lr=lr)
        loss_l=torch.empty(num_iter)
        #proj_l=torch.empty((num_iter,X.shape[1]))
        for i in range(num_iter):
            theta.data/=torch.norm(theta.data)
            optimizer.zero_grad()
            
            loss = self.SWGG_smooth(X,Y,theta,s=s,std=std)
            loss.backward()
            optimizer.step()
        
            loss_l[i]=loss.data
            #proj_l[i,:]=theta.data
        res=self.SWGG_smooth(X,Y,theta.data.float(),s=1,std=0)
        return res,theta.data, loss_l#,proj_l
    

    def max_sw(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N
#         if self.theta is None:
        if self.ftype=='linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        self.theta=theta
        optimizer=optim.Adam([self.theta],lr=lr)
        loss_l=[]
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.sw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            #print('test4')
            loss_l.append(loss.data)
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.norm(self.theta.data)
            #print('test5')

        res = self.sw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
        return res,self.theta.to(self.device).data,loss_l



    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())
        

def TWD(X, Y, theta, intercept, mass_division = 'distance_based', p = 2, delta = 2., device = 'cuda'):
    # print(p)
    # print(delta)
    # exit()
    L = theta.shape[0]
    nlines = theta.shape[1]
    TWD_obj = TWConcurrentLines(p=p, delta=delta, mass_division=mass_division, device=device)
    return TWD_obj(X, Y, theta, intercept)


import numpy as np
import torch
import ot

def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

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
    wasserstein_distance = torch.mean (torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance



def LCVSW(X,Y,L=10,p=2,device="cuda"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    G_mean = torch.mean((diff_m1_m2)**2) #+ (sigma_1-sigma_2)**2
    theta = rand_projections(dim, L, device)
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
    diff_X_m1 = X-m_1
    diff_Y_m2 = Y-m_2
    G_mean = torch.mean((diff_m1_m2)**2) +  torch.mean((diff_X_m1)**2)+  torch.mean((diff_Y_m2)**2)
    theta = rand_projections(dim, L, device)
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 +torch.mean(torch.matmul(theta,diff_X_m1.transpose(0,1))**2,dim=1)+torch.mean(torch.matmul(theta,diff_Y_m2.transpose(0,1))**2,dim=1)
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)

def maxTWD(X,Y,n_lines,iterations=50,lr=1e-4, device="cuda"):
    N,dn = X.shape
    M,dm = Y.shape
    assert dn==dm and M==N
    theta=torch.randn((1,n_lines,dn),device=device,requires_grad=True)
    theta.data/=torch.sqrt(torch.sum((theta.data)**2))
    theta, intercept, subsequent_sources = generate_trees_frames(L = 1, d = dn, theta = theta, range_root_1=-1.0, range_root_2=1.0, range_source_1=-0.1, range_source_2=0.1, nlines=n_lines, device='cuda', type_lines='sequence_of_lines')

    optimizer=optim.Adam([theta],lr=lr)
    loss_l=[]
    for i in range(iterations):
        optimizer.zero_grad()
        loss=-TWD(X.to(device),Y.to(device), theta.to(device), intercept.to(device), subsequent_sources.to(device))
        #print('test4')
        loss_l.append(loss.data)
        loss.backward(retain_graph=True)
        optimizer.step()
        theta.data/=torch.norm(theta.data)
        theta, intercept, subsequent_sources = generate_trees_frames(L = 1, d = dn, theta = theta, range_root_1=-1.0, range_root_2=1.0, range_source_1=-0.1, range_source_2=0.1, nlines=n_lines, device='cuda', type_lines='sequence_of_lines')

        #print('test5')

    res = TWD(X.to(device),Y.to(device),theta.to(device), intercept, subsequent_sources)
    return res




