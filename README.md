# -Research-Course
Tasks from Research Course

Tasks one to three are CPU and GPU functions that take two 3d matrices and we are going to use these two matrices to compute the L2 norm between them, which also results in a 3d matrix of the same size. Then according to the set threshold, we will find which regions are larger than the threshold and set the corresponding regions to 1.

Task.png is a diagram of what the functions are supposed to achieve.

Task one is a basic CPU version just using numpy.

Task two is a faster CPU version using import threading and numpy.

Task three is the fastest and the GPU version using cupy.

The other tasks starting with elgin- are tasks to follow the faiss tutorial.

Flat is the most exhaustive search through the vector for the most similar items.

IVFFLat is a less exhaustive search but searches the closest centroids to give a faster but still accurate search. nprobe decides how many closest centroids to search for.

IVFPQ is product quantization which the even less of an exhaustive search and is a little less accurate of a search. 
