import torch
from tqdm import trange
import numpy as np


#======== 1st version ==========
#SWGG_GG = 2W+2W-4W
#return the value and the permutations

def compute_bary_line(X_line_sort,Y_line_sort,theta) :
    Z_line_sort=(X_line_sort+Y_line_sort)/2
    Z = Z_line_sort[:,None]*theta[None,:]
    return Z

def W2_proj(X,X_line,theta): 
    n=X.shape[0]
    X_proj = X_line[:,None,:]*theta[None,:,:]
    return torch.norm(X[:,:,None]-X_proj,dim=(0,1))**2
    
def W2_1d(X_line_sort,Y_line_sort):
    return torch.sum((X_line_sort-Y_line_sort)**2,axis=0)

    
def SWGG_GG(X,Y,theta):
    n=X.shape[0]
    
    X_line=torch.matmul(X,theta)
    Y_line=torch.matmul(Y,theta)
    
    X_line_sort,u=torch.sort(X_line,axis=0)
    Y_line_sort,v=torch.sort(Y_line,axis=0)
    
    X_sort=X[u].transpose(1,2)
    Y_sort=Y[v].transpose(1,2)
    
    Z=compute_bary_line(X_line_sort,Y_line_sort,theta)
    bary=(X_sort+Y_sort)/2

    W_projX = W2_proj(X,X_line,theta)
    W_projY = W2_proj(Y,Y_line,theta)
    W_1d = W2_1d(X_line_sort,Y_line_sort)
    
    W_projbary = torch.norm(Z-bary,dim=(0,1))**2
    
    return (-4*W_projbary+2*W_projX+2*W_projY+W_1d)/n,u,v




#======== 2nd version ==========
#SWGG_CP=\|X[u]-Y[v]\|^2_2
#return the value and the permutations

def SWGG_CP(X, Y, theta):
    n = X.shape[0]

    X_line = torch.matmul(X, theta)
    Y_line = torch.matmul(Y, theta)

    X_line_sort, u = torch.sort(X_line, axis=0)
    Y_line_sort, v = torch.sort(Y_line, axis=0)

    return torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0),u,v
    


#======== 3rd version ==========
#SWGG with optimization scheme
#return the optimal theta, the loss and all the theta

def get_SWGG_smooth(X,Y,lr=1e-2,num_iter=100,s=1,std=0,device='cpu',verbose=True):
    theta=torch.randn((X.shape[1],), device=device, dtype=X.dtype,requires_grad=True)
    
    #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
    optimizer = torch.optim.SGD([theta], lr=lr)
    
    loss_l=torch.empty(num_iter)
    proj_l=torch.empty((num_iter,X.shape[1]))
    if verbose: pbar = trange(num_iter)
    else: pbar = range(num_iter)
    for i in pbar:
        theta.data/=torch.norm(theta.data)
        
        optimizer.zero_grad()
        loss = SWGG_smooth(X,Y,theta,s=s,std=std,device=device)
        loss.backward()
        optimizer.step()
        
        loss_l[i]=loss.data
        proj_l[i,:]=theta.data
        if verbose:
            pbar.set_postfix_str(f"loss = {loss.item():.3f}")
    return theta, loss_l, proj_l



def SWGG_smooth(X,Y,theta,s=1,std=0,device='cpu'):
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
    X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape,device=device)
    Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
    Y_line_extend_blur = Y_line_extend + 0.5 * std * torch.randn(Y_line_extend.shape,device=device)
    
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
    
    
    
#======== 1st Version quantile version
def quantile_SWGG_CP(X,Y,a,b,theta):
    ns,d=X.shape
    nt=Y.shape[0]
    n_proj=theta.shape[1]
    
    X_line=torch.matmul(X,theta)
    Y_line=torch.matmul(Y,theta)
    
    X_line_sort,u=torch.sort(X_line,axis=0)
    Y_line_sort,v=torch.sort(Y_line,axis=0)

    r_X=torch.cumsum(a[u],axis=0)
    r_X=torch.cat((torch.zeros((1,n_proj)),r_X))
    r_Y=torch.cumsum(b[v],axis=0)
    r_Y=torch.cat((torch.zeros((1,n_proj)),r_Y))
    r,_=torch.sort(torch.cat((r_X,r_Y)),axis=0)
    r=r[1:-1]#Is the x-axis of the quantile function
    
    w_a=torch.searchsorted(r_X.T.contiguous(),r.T.contiguous(),side='right').T[:-1]-1 #Transport plan of X after ordering u
    w_b=torch.searchsorted(r_Y.T.contiguous(),r.T.contiguous(),side='right').T[:-1]-1

    u_w=torch.take_along_dim(u,w_a,dim=0)#permutation of X
    v_w=torch.take_along_dim(v,w_b,dim=0)
    
    q_X=X[u_w]#Is the y-axis of the quantile function
    q_Y=Y[v_w]
    
    delta_r = r[1:] - r[:-1]
    return torch.sum(delta_r*torch.sum((q_X-q_Y)**2,axis=-1),axis=0),delta_r,w_a,w_b,u,v
  
  
  
  
#========Utils for Color transfer    
def SWGG_CP_color(X, Y, theta):
    n = X.shape[0]

    X_line = torch.matmul(X, theta)
    Y_line = torch.matmul(Y, theta)

    X_line_sort, u = torch.sort(X_line, axis=0)
    Y_line_sort, v = torch.sort(Y_line, axis=0)
    
    W=torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0)

    idx=torch.argmin(W)
    return W[idx],u[:,idx],v[:,idx]

