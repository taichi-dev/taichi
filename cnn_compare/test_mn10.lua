#!/usr/bin/env th

local common = dofile('common.lua')
require('os')
require('nn')
require('cunn')
require('cudnn')
package.path = '/home/tzumao/octnet/th/?/init.lua;' .. package.path
require('oc')
require('struct')

local channels = 1
local n_grids = 4096 -- Parallelization granularity
net = nn.Sequential()
  :add( oc.OctreeConvolutionMM(channels, channels, n_grids) )

local s = net:get(1).weight:size()
net:get(1).weight:zero()
for co = 1, s[1] do
    for ci = 1, s[2] do
        local inc = 0.1
        for i = 1, s[3] do
            for j = 1, s[4] do
                for k = 1, s[5] do
                    net:get(1).weight[{co, ci, i, j, k}] = 1.0
                    inc = inc + 0.1
                end
            end
        end
    end
end
net:get(1).bias:zero()
net:cuda()

vx_res = 8
local tensor = torch.FloatTensor(1, channels, vx_res, vx_res, vx_res)
oc.read_dense_from_bin(arg[1], tensor)
local input_cpu = oc.FloatOctree()
input_cpu = input_cpu:create_from_dense(tensor)
print(input_cpu:size())
local input = input_cpu:cuda()

local output = net:forward(input)
local timer = torch.Timer()
for i = 1, 20 do
    output = net:forward(input)
end
print(string.format('[INFO] net fwd took %f[s]', timer:time().real / 20.0))

output = output:float():to_cdhw()
local f = assert(io.open('octnet_output.bin', 'wb'))
local t = output[{1, 1, {1, vx_res}, {1, vx_res}, {1, vx_res}}]
for i = 1, vx_res do
    for j = 1, vx_res do
        for k = 1, vx_res do
            f:write(struct.pack('f', t[{i, j, k}]))
        end
    end
end
f:close()
