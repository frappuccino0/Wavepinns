"""
@author: Yufeng Wang
"""

import sys
sys.path.insert(0, '../')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from pyDOE import lhs

from Pinns_class import PhysicsInformedNN

if __name__ == "__main__": 


    # N_train points for physics constrain as well as boundary constrain
    N_u_train = 5000
    N_f_train = 10000
    
    # first layer = 3 for (x y t) and the last layer = 1 for (u)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # Load FD Data
    data = scipy.io.loadmat('../weq_300_150.mat')         
    U_star = data['u_star'] # N x T
    T_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    vp_star = data['vp_star'] # N x 1
    N = X_star.shape[0]
    T = T_star.shape[0]

    XX = np.tile(X_star[:,0:1], (1,T)).T # T x N
    YY = np.tile(X_star[:,1:2], (1,T)).T # T x N
    VP = np.tile(vp_star[:,0:1], (1,T)).T # T x N
    TT = np.tile(T_star[:,0:1], (1,N)) # T x N   
    UU = U_star.T # T x N
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    vp = VP.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    u = UU.flatten()[:,None] # NT x 1
    
    
    ######################################################################
    ############################### Data #################################
    ######################################################################
    X = np.concatenate([x, y, t], 1)       
    lb = X.min(0)
    ub = X.max(0)

    # Training Data randomly selected from N*T dimensional sapce for training   
    np.random.seed(1234)
    tf.set_random_seed(1234)


    #FD simulation data set
    idx_u = np.random.choice(N*T, N_u_train, replace=False)
    x_u_train = x[idx_u,:]
    y_u_train = y[idx_u,:]
    t_u_train = t[idx_u,:]
    u_train = u[idx_u,:]
    

    # physics constrain from all data set
    X_f_train = lb + (ub-lb)*lhs(3, N_f_train)
    x_f_train = X_f_train[:,0:1]
    y_f_train = X_f_train[:,1:2]
    t_f_train = X_f_train[:,2:3]
    vp_train = 2500
    
    
    # #random select physics constrain
    # idx_f = np.random.choice(N*T, N_f_train, replace=False)  
    # x_f_train = x[idx_f,:]
    # y_f_train = y[idx_f,:]
    # t_f_train = t[idx_f,:]
    # vp_train = vp[idx_f,:]

    # Training
    niter_bfgs = 10000

    train_or_not = 1
    history_loss = []

    if train_or_not == 1:
        model = PhysicsInformedNN(x_u_train, y_u_train, t_u_train, x_f_train, y_f_train, t_f_train, u_train, vp_train, layers, niter_bfgs)
        # loss_out = model.train(iter=5000, learning_rate=1e-3, batch_num=1, whichloss = 0)
        # history_loss.extend(loss_out)
        # with open('history_loss_adam.txt', 'ab') as f:
        #     np.savetxt(f, history_loss)
        model.train_bfgs(batch_num=1, whichloss = 1)
        
    else:   
        model = PhysicsInformedNN_separate(x_u_train, y_u_train, t_u_train, x_f_train, y_f_train, t_f_train, u_train, vp_train, layers, niter_bfgs, ExistModel=1, modelDir='NN_1.pickle')

    model.save_NN('NN_1.pickle')

    print('Max time step for u training: %e' % (t_u_train.max()))
    print('Max time step for f training: %e' % (t_f_train.max()))