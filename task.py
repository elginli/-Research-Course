import numpy as np
#import cupy as cp
import time
import threading

arr1 = np.random.rand(384,1152,1152)
arr2 = np.random.rand(384,1152,1152)

#arr1_gpu = cp.asarray(arr1)
#arr2_gpu = cp.asarray(arr2)

block_size = 2


def time_it(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(func.__name__ + " took " + str((end-start)*1000) + "ms")
    return result

'''

'''
#@time_it
def dis_cpu(arr1, arr2, block_size, threshold):
    result = np.zeros((384//block_size,1152//block_size,1152//block_size))
    #1.calculate L2 norm from arr2 and calculate 

    l2_norm = np.sqrt(np.subtract(arr1, arr2)**2)

    for i in range(384//block_size):
        for j in range(1152//block_size):
            for k in range(1152//block_size):

                block = l2_norm[i * block_size : (i+1) * block_size, 
                                j * block_size : (j+1) * block_size,
                                k * block_size : (k+1) * block_size]

    #2. compare ave L2 norm within a block with threshold and return result
                avg_l2_norm = np.mean(block)

                if(avg_l2_norm > threshold):
                    result[i, j, k] = 0
                else:
                    result[i, j, k] = 1
    return result



#@time_it
#def dis_gpu(arr1, arr2, block_size,threshold):
    #1.  calculate L2 norm from arr1 and arr2


    #2. compare ave L2 norm within a block with threshold and return result




