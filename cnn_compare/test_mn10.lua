#!/usr/bin/env th

local common = dofile('common.lua')
require('os')
require('nn')
require('cunn')
require('cudnn')
package.path = '/home/tzumao/octnet/th/?/init.lua;' .. package.path
require('oc')

local n_grids = 4096 -- Parallelization granularity
net = nn.Sequential()
  :add( oc.OctreeConvolutionMM(16, 16, n_grids) )

local s = net:get(1).weight:size()
local inc = 0.0
for co = 1, s[1] do
    for ci = 1, s[2] do
        for i = 1, s[3] do
            for j = 1, s[4] do
                for k = 1, s[5] do
                    net:get(1).weight[co][ci][i][j][k] = inc
                    inc = inc + 0.1
                end
            end
        end
    end
end
net:cuda()

vx_res = 256
channels = 16
local tensor = torch.FloatTensor(1, channels, vx_res, vx_res, vx_res)
oc.read_dense_from_bin(arg[1], tensor)
local input_cpu = oc.FloatOctree()
input_cpu = input_cpu:create_from_dense(tensor)
local input = input_cpu:cuda()

local output = net:forward(input)
local timer = torch.Timer()
for i = 1, 20 do
    output = net:forward(input)
end
print(string.format('[INFO] net fwd took %f[s]', timer:time().real / 20.0))
output = output:cpu():to_cdhw()
local f = assert(io.open('octnet_output.bin', 'rw'))
local t = tensor[{1, 1, {1, 256}, {1, 256}, {1, 256}}]
for i = 1, 256 do
    for j = 1, 256 do
        for k = 1, 256 do
            io.write(t[i, j, k])
        end
    end
end
