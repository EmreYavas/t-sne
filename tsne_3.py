import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.special import logsumexp

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


def qtable_fill(y,N):
    qtable = np.empty([N,N])
    denom = 0.0
    for k in range(N):
        for l in range(N):
            if(k!=l):
                denom=denom+(1+(np.linalg.norm(y[k]-y[l]))**2)**(-1)

    for i in range (N):
        for j in range(N):
           num=(1+(np.linalg.norm(y[i]-y[j]))**2)**(-1)
           qtable[i,j]=num/denom

    return qtable 


def gradient(p,q,y,i,N):
    grad=0.0
    for j in range (N):
        grad=grad+(p[i,j]-q[i,j])*(y[i]-y[j])/(1+np.linalg.norm(y[i]-y[j])**2)
    
    grad=grad*4
    return grad


def y_update(x,ptable,labels,N):
    dim_y=2
    lrate=100 # learning rate
    alpha=0.9 # momentum
    T=61
    np.random.seed(2)
    #y=np.random.multivariate_normal(np.zeros(dim_y), 0.01*np.identity(dim_y), size=N)
    #
    y=my_pca(x,2)
    filename='y_matrix_t_'+'0'+'.npy'
    np.save(filename,y)

    hist=np.zeros([N,dim_y,2])
    hist[:,:,0]=y
    qtable = qtable_fill(y,N)

    for i in range (N):
        y[i]=y[i]+lrate*gradient(ptable,qtable,y,i,N)
    hist[:,:,1]=y

    for t in range (2,T):
        qtable=qtable_fill(y,N)
        for i in range (N):
            grad=-gradient(ptable,qtable,y,i,N)
            upd=lrate*grad+alpha*(hist[i,:,1]-hist[i,:,0])
            y[i]=y[i]+upd
        hist[:,:,0]=hist[:,:,1]
        hist[:,:,1]=y
        print('y_upd ',t)

        if(t%40==0):
            filename='y_matrix_t_'+str(t)+'.npy'
            np.save(filename,y)
        

    return y



def perplexity(x,i,var_i,N):
    sums=0.0
    pji_denom=pj_i_denom(x,i,var_i,N)
    for j in range(N):
        if(i!=j):
            pj_i = pj_i_num(x,i,j,var_i)-pji_denom
            if(pj_i<-700): #numerical stability
                pj_i=-700
                pj_i = np.exp(pj_i)
                sums=sums+pj_i*np.log2(pj_i)
            else:
                pj_i = np.exp(pj_i)
                sums=sums+pj_i*np.log2(pj_i)            
        elif(i==j):
            pj_i = pj_i_num(x,i,j,var_i)-pji_denom
            pj_i = np.exp(pj_i)
            sums=sums+0 #pj_i*np.log2(pj_i) because pii=0

    perp=2**(-sums)

    return perp


def search_sigma(x,k,N):
    max_itr=100
    tol=1e-2
    sigma_list=np.empty(N)
    for i in range (N):
        lower=0
        upper=500
        perp=0
        for itr in range(max_itr):
            sigma_est=(lower+upper)/2
            perp=perplexity(x,i,sigma_est,N)
            #print(sigma_est)
            #print(perp)
            if np.abs(perp-k)<tol:
                sigma_list[i]=sigma_est
                #print(itr)
                break
            elif(itr==(max_itr-1)):
                sigma_list[i]=sigma_est
                #print(sigma_list[i])
                print('closest perplexity fori=:',i,'is',perp)
            if(k<perp):
                upper=sigma_est
            else:
                lower=sigma_est

    return (sigma_list)



def ptable_fill(x,var_list,N):
    pj_i_table = np.empty([N,N])
    ptable=np.empty([N,N])
    for i in range (N):
        pji_denom=pj_i_denom(x,i,var_list[i],N)
        for j in range(N):
            if(j!=i):
                #since they are log base division becomes subtraction
                pj_i_table[j,i]=pj_i_num(x,i,j,var_list[i])-pji_denom
                #then normal base
                pj_i_table[j,i]=np.exp(pj_i_table[j,i])
            elif (j==i):
                pj_i_table[j,i]=0.0
    

    for i in range (N):
        for j in range(N):
            ptable[i,j]=(pj_i_table[j,i]+pj_i_table[i,j])/(2*N)

    return ptable 
    
def pj_i_denom(x,i,var_i,N):
    xi=x[i]
    denom_list=np.empty(N)
    for k in range(N):
        if i!=k:
            denom_list[k]=(-(np.linalg.norm(xi-x[k]))**2/(2*var_i))
        else:
            denom_list[k]=0.0
    denom_list=np.delete(denom_list,i)
    #I use log-sum-exp to avoid underflow
    denom=(logsumexp(denom_list))
    return denom

def pj_i_num(x,i,j,var_i):
    xi=x[i]
    xj=x[j]
    #I use log base to avoid underflow
    num = (-(np.linalg.norm(xi-xj))**2/(2*var_i))
    return num


def my_pca(data,dim):
    N,d=data.shape
    cov_matr = np.dot(data.T,data)
    cov_matr = cov_matr/(N-1)
    eigs,eig_vecs = np.linalg.eigh(cov_matr) #returns in ascending order
    eig_vecs=np.flip(eig_vecs)
    B_p=eig_vecs[:,:dim]
    pca_data=np.dot(data,B_p)
    return pca_data


def getdata(name,N,some_digits=True):
    
    if name=='MNIST':
        data =  pd.read_csv('mnist_train.csv')
        labels = data['label']
        data.drop('label',axis = 1,inplace = True)
        data=data.to_numpy()
        labels=labels.to_numpy()
        if(some_digits):
            data2=[]
            labels2=[]
            for i in range (labels.size):
                if (labels[i]==1 or labels[i]==0 or labels[i]==8):
                    data2.append(data[i])
                    labels2.append(labels[i])
                if len(labels2)==N:
                    break
            data=np.array(data2)
            labels=np.array(labels2)
        data=data[:N]
        labels=labels[:N]

    elif name=='load_digits':
        data,labels=load_digits(return_X_y=True)
        data=data[:N]
        labels=labels[:N]

    normalizer=StandardScaler()
    data=normalizer.fit_transform(data)
    np.save('labels.npy',labels)
    print('Number of samples:',data.shape[0],'Number of pixels:',data.shape[1])
    return data,labels

def built_in(data,labels):

    data=PCA(n_components=30).fit_transform(data)
    y= TSNE(n_components=2).fit_transform(data)
    plt.scatter(y[:,0],y[:,1],c=labels, cmap='tab10')
    plt.show()


if __name__ == '__main__':
    name='load_digits'
    N=200
    data,labels = getdata(name,N,True)
    
    pca_dim=30
    data=my_pca(data,pca_dim)
    print('pca is computed')
    perplex=80
    var_list = search_sigma(data,perplex,N)
    #np.save('varlist.npy',var_list)
    #var_list=np.load('varlist.npy')
    print('var_list is computed')
    ptable=ptable_fill(data,var_list,N)
    #np.save('ptable.npy',ptable)
    #ptable=np.load('ptable.npy')
    print('ptable is computed')
    y=y_update(data,ptable,labels,N)
    print('2 dimensional data is ready')
    np.save('y_final.npy',y)

    digits=np.unique(labels)
    for i in range(10):
        idx=np.where(labels==i)
        plt.scatter(y[idx,0],y[idx,1])
        plt.legend(labels,loc="lower left", title="digits")
    plt.show()

    '''
    built_in(data,labels)
    '''
