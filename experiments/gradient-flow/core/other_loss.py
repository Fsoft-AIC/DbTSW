import torch
import numpy as np

def sw(X,Y,theta=None):
    N,dn = X.shape
    M,dm = Y.shape
    assert dn==dm and M==N
    if theta is None:
        theta=self.random_slice(dn)

    Xslices=torch.matmul(X,theta)
    Yslices=torch.matmul(Y,theta)

    Xslices_sorted=torch.sort(Xslices,dim=0)[0]
    Yslices_sorted=torch.sort(Yslices,dim=0)[0]
    return torch.sum((Xslices_sorted-Yslices_sorted)**2,axis=0)/N
    
def max_sw(X,Y,num_iter=500,lr=1e-4):
    N,dn = X.shape
    M,dm = Y.shape
    theta=torch.randn((1,dn),requires_grad=True)
    theta.data/=torch.sqrt(torch.sum((theta.data)**2))
    optimizer=torch.optim.Adam([theta],lr=lr)
    loss_l=[]
    for i in range(num_iter):
        optimizer.zero_grad()
        loss=-sw(X,Y,theta.T)
        loss_l.append(-loss.data)
        loss.backward(retain_graph=True)
        optimizer.step()
        theta.data/=torch.norm(theta.data)
            #print('test5')

    res = sw(X,Y,theta.T)
    return loss_l,theta.data,res.data
    
def PWD(X, Y, theta):
    n = X.shape[0]

    X_line = torch.matmul(X, theta)
    Y_line = torch.matmul(Y, theta)

    X_line_sort, u = torch.sort(X_line, axis=0)
    Y_line_sort, v = torch.sort(Y_line, axis=0)

    return torch.mean(torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0))

