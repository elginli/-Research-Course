import numpy as np
#import cupy as cp
import time
import threading

arr1 = np.random.rand(400,1200,1200)
arr2 = np.random.rand(400,1200,1200)

#arr1_gpu = cp.asarray(arr1)
#arr2_gpu = cp.asarray(arr2)

block_size = 25
threshold = 0.5


def time_it(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ + " took " + str((end-start)) + "s")
        return result
    return wrapper

'''

'''
def calculate_block_result(result, arr1, arr2, block_size, threshold, i, j, k):
    #1.calculate L2 norm from arr2 and calculate 

    block1 = arr1[i * block_size: (i + 1) * block_size,j * block_size: (j + 1) * block_size,k * block_size: (k + 1) * block_size]
    block2 = arr2[i * block_size: (i + 1) * block_size,j * block_size: (j + 1) * block_size,k * block_size: (k + 1) * block_size]
     

    l2_norm = np.sqrt(np.subtract(block1, block2)**2)

    #2. compare ave L2 norm within a block with threshold and return result
    avg_l2_norm = np.mean(l2_norm)
    if avg_l2_norm > threshold:
        result[i, j, k] = 0
    else:
        result[i, j, k] = 1


@time_it
def dis_cpu(arr1, arr2, block_size, threshold):
    result = np.zeros((400//block_size,1200//block_size,1200//block_size))
    
    threads = []
    for i in range(400 // block_size):
            for j in range(1200 // block_size):
                for k in range(1200 // block_size):
                    t1 = threading.Thread(target=calculate_block_result, args=(result, arr1, arr2, block_size, threshold, i, j, k))
                    t1.start()
                    threads.append(t1)

    for t1 in threads:
        t1.join()

    return result


#@time_it
#def dis_gpu(arr1, arr2, block_size,threshold):
    #1.  calculate L2 norm from arr1 and arr2


    #2. compare ave L2 norm within a block with threshold and return result

if __name__ == "__main__":
    cpu_result = dis_cpu(arr1, arr2, block_size, threshold)

    print(cpu_result)
