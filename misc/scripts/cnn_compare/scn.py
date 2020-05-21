import cv2
import torch
import array
import sparseconvnet as scn
import time
import numpy as np

n = 256
num_ch = 16

input_voxel = array.array('f')
# f = open('bunny.bin', 'rb')
f = open('bunny_sparse.bin', 'rb')
input_voxel.fromfile(f, num_ch * n * n * n)

input_voxel = torch.tensor(input_voxel)
input_voxel = input_voxel.reshape(1, num_ch, n, n, n)
for i in range(256):
    img = np.array(input_voxel[0, 0, i])
    cv2.imshow('img', img * 255)
    cv2.waitKey(1)
print(input_voxel.sum())
input_voxel = input_voxel.cuda()

dense_to_sparse = scn.DenseToSparse(3)
sparse_voxel = dense_to_sparse(input_voxel)
print(sparse_voxel)

conv = scn.Convolution(3, num_ch, num_ch, 3, 1, None)
conv.weight.data.zero_()
print(conv.weight.shape)
'''
for co in range(0, 16):
    for ci in range(0, 16):
        inc = 0.1
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        conv.weight.data[(z+1)*9+(y+1)*3+(x+1), ci, co] = inc
                    inc += 0.1
'''
conv.weight.data.fill_(1.0 / 16.0)
conv = conv.cuda()

conv_sparse_voxel = conv(sparse_voxel)
min_time = 1e10
for i in range(1):
    start = time.time()
    conv_sparse_voxel = conv(sparse_voxel)
    end = time.time()
    if end - start < min_time:
        min_time = end - start
    print(min_time)
print(conv_sparse_voxel)
print('bulk time:', min_time)

sparse_to_dense = scn.SparseToDense(3, 16)
output_voxel = sparse_to_dense(conv_sparse_voxel)
output_voxel = output_voxel[0, 0, :, :, :].cpu().reshape(-1)
output_voxel = array.array('f', output_voxel)

f = open('scn_bunny.bin', 'wb')
output_voxel.tofile(f)
