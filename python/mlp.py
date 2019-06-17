import numpy as np
import random
import taichi_lang as ti
import pickle

input = ti.var(ti.f32)
weight = ti.var(ti.f32)
output = ti.var(ti.f32)
output_exp = ti.var(ti.f32)
output_softmax = ti.var(ti.f32)
softmax_sum = ti.var(ti.f32)
gt = ti.var(ti.f32)
loss = ti.var(ti.f32)

n_input = 28 ** 2
n_output = 10

@ti.layout
def layout():
  ti.root.dense(ti.i, n_input).place(input)
  ti.root.dense(ti.ij, (n_input, n_output)).place(weight)
  ti.root.dense(ti.i, n_output).place(gt)
  ti.root.dense(ti.i, n_output).place(output)
  ti.root.dense(ti.i, n_output).place(output_exp)
  ti.root.dense(ti.i, n_output).place(output_softmax)
  ti.root.place(softmax_sum)
  ti.root.place(loss)


@ti.kernel
def layer1():
  for i in range(layer1):
    for j in range(n_output):
      output[j].atomic_add(input[i] * weight[i, j])

@ti.kernel
def layer2():
  for i in range(n_output):
    output_exp[i] = ti.exp(output[i])

@ti.kernel
def layer3():
  for i in range(n_output):
    softmax_sum[i].atomic_add(output[i])

@ti.kernel
def layer4():
  for i in range(n_output):
    output_softmax[i] = output_exp[i] / softmax_sum

@ti.kernel
def layer5():
  for i in range(n_output):
    loss.atomic_add((output_softmax[i] - gt[i]) ** 2)


with open('mnist.pkl', 'rb') as f:
  mnist = pickle.load(f)

training_images = mnist['training_images']

print(training_images.shape)

for k in range(100):
  img = training_images[random.randrange(0, len(training_images))]
  for i in range(n_input):
    input[i] = img[i]
  layer1()
  layer2()
  layer3()
  layer4()
  layer5()

  layer5.grad()
  layer4.grad()
  layer3.grad()
  layer2.grad()
  layer1.grad()
