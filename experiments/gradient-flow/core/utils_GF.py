import numpy as np
import torch
import ot
from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_blobs,make_spd_matrix
from scipy.stats import random_correlation
from tqdm import trange


def w2(X,Y):
    M=ot.dist(X,Y)
    a=np.ones((X.shape[0],))/X.shape[0]
    b=np.ones((Y.shape[0],))/Y.shape[0]
    return ot.emd2(a,b,M)
   


def load_data(name='swiss_roll', n_samples=1000,dim=2):
    N=n_samples
    if name == 'gaussian' :
        mu_s = np.random.randint(-10,-1,dim)
        cov_s = np.diag(np.random.randint(1,10,dim))
        cov_s = cov_s * np.eye(dim)
        temp = np.random.multivariate_normal(mu_s, cov_s, n_samples)
    elif name == 'gaussian_2d':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1))+4 for i in mu_s]) 
        #cov_s = cov_s * np.eye(2)
        cov_s = np.array([[0.5,-2], [-2, 5]])
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_2d_small_v':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((2, 2))*1
        cov_s = cov_s * np.eye(2)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_2d_big_v':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = random_correlation.rvs((.2, 1.8))*2
        #cov_s = cov_s * np.eye(2)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_200d_small_v':
        mu_s = np.ones(200)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((200, 200))*1
        cov_s = cov_s * np.eye(200)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_10d_small_v':
        mu_s = np.ones(10)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((10, 10))
        cov_s = cov_s * np.eye(10)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_20d_small_v':
        mu_s = np.ones(20)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((20, 20))
        cov_s = cov_s * np.eye(20)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_25d_small_v':
        mu_s = np.ones(25)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((25, 25))
        cov_s = cov_s * np.eye(25)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        
    elif name == 'gaussian_75d_small_v':
        mu_s = np.ones(75)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((75, 75))
        cov_s = cov_s * np.eye(75)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_30d_small_v':
        mu_s = np.ones(30)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((30, 30))
        cov_s = cov_s * np.eye(30)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_40d_small_v':
        mu_s = np.ones(40)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((40, 40))
        cov_s = cov_s * np.eye(40)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_50d_small_v':
        mu_s = np.ones(50)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((50, 50))
        cov_s = cov_s * np.eye(50)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_100d_small_v':
        mu_s = np.ones(100)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((100, 100))*1
        cov_s = cov_s * np.eye(100)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_150d_small_v':
        mu_s = np.ones(150)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((150, 150))*1
        cov_s = cov_s * np.eye(150)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        
    elif name == 'gaussian_250d_small_v':
        mu_s = np.ones(250)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((250, 250))*1
        cov_s = cov_s * np.eye(250)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        
    elif name == 'gaussian_300d_small_v':
        mu_s = np.ones(300)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((300, 300))*1
        cov_s = cov_s * np.eye(300)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        
    elif name == 'gaussian_500d_small_v':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((500, 500))*1
        cov_s = cov_s * np.eye(500)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_500d_big_v':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        eigs = np.random.rand(500,1)*2
        eigs = eigs / np.sum(eigs) * 500
        rr = eigs.reshape(-1)
        cov_s = random_correlation.rvs(rr)*10
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((500, 500))*50
        cov_s = cov_s * np.eye(500)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_500d_spd_cov':
        dim = 500
        mu_s = np.ones(dim)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = make_spd_matrix(dim, random_state=3)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == 'half_moons':
        temp=make_moons(n_samples=N)[0]
        temp/=abs(temp).max()
    elif name == '8gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '25gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=2*N)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    else:
        raise Exception("Dataset not found: name must be 'gaussian_2d', 'gaussian_500d', swiss_roll', 'half_moons', 'circle', '8gaussians' or '25gaussians'.")
    X=torch.from_numpy(temp).float()
    return X
