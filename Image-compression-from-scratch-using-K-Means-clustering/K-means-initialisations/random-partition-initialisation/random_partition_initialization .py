
""" K-means implementation with different initialization centroids"""

# Second one is random partition initialization 

""" In this method, we randomly assign each point in the data to a random cluster id
and then we will group the points by their cluster id and take the mean to yield initial points
In this method, the intial points chosen lie very close or near to the global
mean of the data. This is not best method to recommend for k-means initialization."""

# import required libraries
import numpy as np
import numpy.matlib
from skimage import io
import matplotlib.pyplot as plt
import imageio
import random
import warnings
warnings.filterwarnings('ignore')

# Reading the GeorgiaTech image
gt_im = io.imread('GeorgiaTech.bmp')
io.imshow(gt_im)
io.show()

# size of the image
im_size = gt_im.shape
print(im_size)

"""The image consists of 3 channels such as Red, Green and Blue(RGB). Each having values in the range of 0 to 255. 
So, before we used this image, first we need to normalize by dividing by 255 """

# Normalized the image
gt_im = gt_im/255
m = gt_im.shape[0]
n = gt_im.shape[1]

# Reshape the image
X = gt_im.reshape(gt_im.shape[0]*gt_im.shape[1],3)
print(X.shape)

num_clusters = [2,4,8,16]
n_iter = 50 # maximum number of iteractions

try:
    
  for K in num_clusters:


    %%time
    # random partition initialization
    def random_partition(X, K):
      
        idx = np.random.choice(range(0, K), replace = True, size = X.shape[0])
        mean = []
        for count in range(K):
            mean.append(X[idx == count].mean(axis=0))
        
        return np.concatenate([value[ None, :] for value in mean], axis = 0)
    
    # We are using Euclidean distance or squared-l2 metric 
    def compute_closest_cents(X,c):
        K = np.size(c,0)
        idx = np.zeros((np.size(X,0),1))
        arr = np.empty((np.size(X,0),1))
        for i in range(0,K):
            y = c[i]
            temp = np.ones((np.size(X,0),1))*y
            a = np.power(np.subtract(X,temp),2)
            b = np.sum(a,axis = 1)
            b = np.asarray(b)
            b.resize((np.size(X,0),1))
            arr = np.append(arr, b, axis=1)
        arr = np.delete(arr,0,axis=1)
        idx = np.argmin(arr, axis=1)
        return idx

    # computing centroids
    def compute_cents(X,idx,K):
        n = np.size(X,1)
        cents = np.zeros((K,n))
        for i in range(0,K):
            ci = idx==i
            ci = ci.astype(int)
            total_num = sum(ci)
            ci.resize((np.size(X,0),1))
            total_matrix = np.matlib.repmat(ci,1,n)
            ci = np.transpose(ci)
            total = np.multiply(X,total_matrix)
            cents[i] = (1/total_num)*np.sum(total,axis=0)
        return cents

    # Running KMean
    def run_KMean(X,init_cents,n_iter):
        m = np.size(X,0)
        n = np.size(X,1)
        K = np.size(init_cents,0)
        cents = init_cents
        previous_cents = cents
        idx = np.zeros((m,1))
        for i in range(1,n_iter):
           idx = compute_closest_cents(X,cents)
           cents = compute_cents(X,idx,K)
        return cents,idx
    
    # Intializing centroids
    init_cents = random_partition(X, K)
    cents,idx = run_KMean(X, init_cents, n_iter)
    #idx.resize((np.size(X,0),1))
    print(np.shape(cents))
    print(np.shape(idx))

   # Recovering compressed images
    idx = compute_closest_cents(X,cents)
    X_compressed = cents[idx]
    print(np.shape(X_compressed))
    X_compressed = np.reshape(X_compressed, (m, n, 3))
    print(np.shape(X_compressed))

    #save compressed image
    imageio.imsave(f'gt_compressed_{K}.bmp', X_compressed)
    
    # Displaying compressed images
    im = imageio.imread(f'gt_compressed_{K}.bmp')
    io.imshow(im)
    io.show() 
    
except:
   print('An error occured')