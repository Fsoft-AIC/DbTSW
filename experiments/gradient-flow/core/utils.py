import torch

def random_slice(n_proj,dim,device='cpu'):
    theta=torch.randn((n_proj,dim))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
    return theta.to(device)
    
def sort_to_plan(u,v):
    n=u.shape[0]
    temp=torch.arange(n)
    P1=torch.zeros((n,n))
    P2=torch.zeros((n,n))

    P1[u,temp]=1
    P2[v,temp]=1
    return (P1@P2.T)/n
    
def quantile_to_plan(r,w_a,w_b,u,v):
    ns=u.shape[0]
    nt=v.shape[0]

    P=torch.zeros((ns,nt))
    P[w_a,w_b]=r
    
    P1=torch.zeros((ns,ns))
    P2=torch.zeros((nt,nt))
    P1[u,torch.arange(ns)]=1
    P2[v,torch.arange(nt)]=1

    return P1@P@P2.T
    
