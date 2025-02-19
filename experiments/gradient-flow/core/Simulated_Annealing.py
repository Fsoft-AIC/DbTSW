import numpy as np
import torch

from SWGG import SWGG_CP



def temperature(n_iter,T_0=1):
    T=[T_0]
    for i in range(n_iter):
        #T.append(1/(2**i))
        T.append(1-(i+1)/n_iter)
    return T
    
def SWGG_annealing(X,Y,theta_0,n_iter,eps=1e-1,device='cpu'):

    n,d=X.shape
    
    schedule=temperature(n_iter)
    E_0=SWGG_CP(X,Y,theta_0)[0]
    
    theta_l=[theta_0]
    loss_l=[E_0]
    for i,T in enumerate(schedule):
        #print('T',T)
        theta_1=theta_0+torch.normal(torch.zeros(d),eps).to(device)
        theta_1/=torch.norm(theta_1)
        
        E_1=SWGG_CP(X,Y,theta_1)[0]
        E=E_0-E_1
        if E>0:
            theta_0=theta_1
            E_0=E_1
            
        elif (torch.exp(E/T))>torch.rand(1,device=device):

            theta_0=theta_1
            E_0=E_1
        theta_l.append(theta_0)
        loss_l.append(E_0)
    return theta_l,E_0,loss_l
