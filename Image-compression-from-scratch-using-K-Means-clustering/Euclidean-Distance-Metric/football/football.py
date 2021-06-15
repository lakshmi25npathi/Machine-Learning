# import required libraries
import numpy as np
import numpy.matlib
from skimage import io
import matplotlib.pyplot as plt
import imageio
import random
import warnings
warnins.filterwarnings('ignore')

# Reading the football image
fb_im = io.imread('football.bmp')
io.imshow(fb_im)
io.show()

# size of the image
im_size = fb_im.shape
print(im_size)

"""The image consists of 3 channels such as Red, Green and Blue(RGB). Each having values in the range of 0 to 255. 
So, before we used this image, first we need to normalize by dividing by 255 """

# Normalized the image
fb_im = fb_im/255
m = fb_im.shape[0]
n = fb_im.shape[1]

# Reshape the image
X = fb_im.reshape(fb_im.shape[0]*fb_im.shape[1],3)
print(X.shape)

num_clusters = [2,4,8,16]
n_iter = 50 # maximum number of iteractions

for K in num_clusters:

    %%time
    # Intialize the centroids randomely
    def init_cents(X, K):
        cent = random.sample(list(X), K)
        return cent
    
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
    init_cents = init_cents(X, K)
    cents,idx = run_KMean(X, init_cents, n_iter)
    #idx.resize((np.size(X,0),1))
    print(np.shape(cents))
    print(np.shape(idx))

   # Recovering compressed image
    idx = compute_closest_cents(X,cents)
    X_compressed = cents[idx]
    print(np.shape(X_compressed))
    X_compressed = np.reshape(X_compressed, (m, n, 3))
    print(np.shape(X_compressed))

    #save compressed image
    imageio.imsave(f'fb_compressed_{K}.bmp', X_compressed)
    
    # Displaying compressed images
    im = imageio.imread(f'fb_compressed_{K}.bmp')
    io.imshow(im)
    io.show() 
