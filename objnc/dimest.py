import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress, spearmanr, pearsonr
from math import sqrt


def estimate_dim(dists, fraction = 0.9, plot_dim = True):
    mu = np.zeros((dists.shape[0],1))
    for i in range(dists.shape[0]):
        x = np.sort(dists[i,:])
        mu[i] = x[2]/x[1]
    mu = np.sort(mu, axis=None)
    npoints = int(np.floor(dists.shape[0]*fraction))
    y = np.ones((1,dists.shape[0]-1)) - np.arange(dists.shape[0]-1,dtype= np.float64) / dists.shape[0]
    y = np.squeeze(y)
    y = -np.log(y)
    x = np.log(mu[0:dists.shape[0]-1])
    reg = linregress(x[0:npoints], y[0:npoints])
    r, pval = pearsonr(x[0:npoints], y[0:npoints])   
        
    if plot_dim:
        fg = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x[0:npoints], y[0:npoints], '.r')
        ax.plot(x[npoints:], y[npoints:], '.b')
        ax.plot([0, x[-1]], [   reg.intercept, reg.slope * x[-1]], '-r')
        ax.set_title('Dim : ' + str(np.round(reg.slope,3)) +  ', Corr : ' + str(np.round(reg.rvalue,3) ) )
       
    return (x,y,reg,r,pval)
	
def block_analysis(dists,  fraction, blocks = list(range(1, 21)), plot_blocks = True):
    """ Docstring"""

    dim = np.zeros(len(blocks))
    std = np.zeros(len(blocks))
    res = estimate_dim(dists, fraction, plot_dim = False)
    max_dim = res[2].slope
   
    for b in blocks:        
        idx = np.random.permutation(N)
        npoints = N / (b + 1)
        I = np.meshgrid(idx[0:npoints], idx[0:npoints], indexing='ij')
        tdists = dists[I]
        tdim = np.zeros(b)
        for i in range(b):           
            res = estimate_dim(tdists, fraction, plot_dim = False)
            tdim[i] = res[2].slope            
        dim[blocks.index(b)] = np.mean(tdim)
        std[blocks.index(b)] = np.std(tdim)

    if plot_blocks :
        fg = plt.figure()
        ax = plt.subplot(111)
        ax.errorbar(blocks, dim, yerr = std, fmt = '-o')
        ax.plot([blocks[0], blocks[-1]],  [max_dim, max_dim], '--g')
        ax.set_title("Block Analysis")
        ax.invert_xaxis()
        
    return (max_dim, dim, std)

#---------------- plots
	
def plot_cumulatives(R,fraction,models):
    fig = plt.figure(figsize=(10,10))
    for i in range(len(R)):
        x = R[i][0]    
        y = R[i][1]
        npoints = int(np.floor(x.size * fraction))   
        plt.subplot(3,3,i + 1)
        plt.plot(x[0:npoints],y[0:npoints],'r.',markersize=0.3)   
        plt.plot([0, x[npoints]], [   R[i][2].intercept, R[i][2].slope * x[npoints]], '-b', linewidth = 0.2)
        plt.title(models[i] + ' --- ' + str(np.round(R[i][2].slope , 3) ))
        plt.ylim([0., 3])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
def plot_correlations(R,models):
    c = [ x[2].rvalue for x in R]
    fig = plt.figure()
    plt.plot(c,'-bo')      
    plt.plot(np.zeros((len(c),1)) + 0.99, '--r')
    plt.ylim([0.95, 1.01])
    plt.show()
    
def plot_dimensions(R,models):
    layers = range(len(models))
    #c = np.log( [ x[2].slope for x in R] )
    c = [ x[2].slope for x in R] 
    fig = plt.figure()
    plt.plot(layers, c,'-o')   
    plt.xticks(layers, models, rotation='vertical')
    plt.ylim([0, 1.1*max(c)])
    plt.xlabel('Layer')
    plt.ylabel('Estimated Dimension')
    plt.show()
    
