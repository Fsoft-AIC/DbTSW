import numpy as np
import torch
import sklearn
from sklearn import datasets

get_rot= lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
def make_blobs_reg(n_samples, n_blobs, scale=.5):
    per_blob=int(n_samples/n_blobs)
    result = np.random.randn(per_blob,2) * scale + 5
    theta=(2*np.pi)/(n_blobs)
    for r in range(1,n_blobs):
        new_blob=(np.random.randn(per_blob,2) * scale + 5).dot(get_rot(theta*r))
        result = np.vstack((result,new_blob))
    return result

# random MoG
def make_blobs_random(n_samples, n_blobs, scale=.5, offset=3):
    per_blob=int(n_samples/n_blobs)
    result = np.random.randn(per_blob,2) * scale + np.random.randn(1,2)*offset
    for r in range(1,n_blobs):
        new_blob=np.random.randn(per_blob,2) * scale + np.random.randn(1,2)*offset
        result = np.vstack((result,new_blob))
    return result

#%%
def make_spiral(n_samples, noise=.5):
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.array(np.hstack((d1x,d1y)))

def make_cube(n_samples,dim):
    n_samples=int(n_samples)
    return np.random.random((n_samples, dim)) * 2 - 1
    
def make_data(expe,n_samples,dim=2,device='cpu'):
    if expe=='spirals':
        r=2.5
        xs = make_spiral(n_samples=n_samples, noise=1)
        xt = make_spiral(n_samples=n_samples, noise=1).dot(get_rot(r))
    elif expe=='mog_reg':
        r=.5
        xs = make_blobs_reg(n_samples=n_samples, n_blobs=3)
        xt = make_blobs_reg(n_samples=n_samples, n_blobs=3).dot(get_rot(r))
    elif expe=='mog_reg2':
        r=.5
        xs = make_blobs_reg(n_samples=n_samples, n_blobs=4)
        xt = make_blobs_reg(n_samples=n_samples, n_blobs=4).dot(get_rot(r))
    elif expe=='mog_random':
        xs = make_blobs_random(n_samples=n_samples, scale=.3,n_blobs=10)
        xt = make_blobs_random(n_samples=n_samples, scale=.3,n_blobs=10)
    elif expe=='custom':
        xs = make_blobs_random(n_samples=n_samples, scale=.3,n_blobs=1,offset=0)-6
        xt = make_spiral(n_samples=n_samples, noise=1)
    elif expe=='two_moons':
        X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.05)
        xs = X[y==0,:]
        xt = X[y==1,:]
    elif expe=='cube':
        #xs=Transform_cube(make_cube(n_samples,dim))
        xs=make_cube(n_samples,dim)
        xt=make_cube(n_samples,dim)
    elif expe=='gaussians' :
        mu_s=np.ones(dim)
        cov_s=np.ones(dim)
        cov_s = cov_s * np.eye(dim)

        mu_t=-np.ones(dim)
        cov_t=np.ones(dim)
        cov_t = cov_t * np.eye(dim)
        
        xs = np.random.multivariate_normal(mu_s, cov_s, n_samples)
        xt = np.random.multivariate_normal(mu_t, cov_t, n_samples)
    elif expe =='sub-manifold':
        mu_s = np.zeros(dim)
        mu_s[0:2]=1
    
        cov_s = np.zeros(dim)
        cov_s[0]=100
        cov_s[1]=1
        cov_s = np.diag(cov_s)
        cov_s = cov_s * np.eye(dim)

        mu_t = np.zeros(dim)
        mu_t[0:2]=-1
    
        cov_t = np.zeros(dim)
        cov_t[0]=100
        cov_t[1]=1
        cov_t=np.diag(cov_t)
        cov_t = cov_t * np.eye(dim)
        
        xs = np.random.multivariate_normal(mu_s, cov_s, n_samples)
        xt = np.random.multivariate_normal(mu_t, cov_t, n_samples)
        
    X = torch.from_numpy(xs).to(device)
    Y = torch.from_numpy(xt).to(device)
    X=X.float()
    Y=Y.float()
    return X,Y
