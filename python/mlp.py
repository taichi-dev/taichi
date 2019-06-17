import numpy as np
import random
import taichi_lang as ti
import pickle

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True

input = ti.var(ti.f32)
weight = ti.var(ti.f32)
output = ti.var(ti.f32)
output_exp = ti.var(ti.f32)
output_softmax = ti.var(ti.f32)
softmax_sum = ti.var(ti.f32)
gt = ti.var(ti.f32)
loss = ti.var(ti.f32)
learning_rate = ti.var(ti.f32)

n_input = 28 ** 2
n_output = 10


@ti.kernel
def init_weights():
  for i in range(n_input):
    for j in range(n_output):
      weight[i, j] = ti.random() * 0.001


@ti.kernel
def clear_weight_grad():
  for i in range(n_input):
    for j in range(n_output):
      weight.grad[i, j] = 0


@ti.layout
def layout():
  ti.root.dense(ti.i, n_input).place(input)
  ti.root.dense(ti.ij, (n_input, n_output)).place(weight)
  ti.root.dense(ti.i, n_output).place(gt)
  ti.root.dense(ti.i, n_output).place(output)
  ti.root.dense(ti.i, n_output).place(output_exp)
  ti.root.dense(ti.i, n_output).place(output_softmax)
  ti.root.place(softmax_sum)
  ti.root.place(loss, learning_rate)

  ti.root.lazy_grad()


def clear_outputs():
  for i in range(n_output):
    output[i] = 0
    output_exp[i] = 0
    output_softmax[i] = 0
    output.grad[i] = 0
    output_exp.grad[i] = 0
    output_softmax.grad[i] = 0


@ti.kernel
def layer1():
  for i in range(n_input):
    for j in range(n_output):
      output[j].atomic_add(input[i] * weight[i, j])


@ti.kernel
def layer2():
  for i in range(n_output):
    output_exp[i] = ti.exp(output[i])


@ti.kernel
def layer3():
  for i in range(n_output):
    softmax_sum.atomic_add(output_exp[i] + 1e-6)


@ti.kernel
def layer4():
  for i in range(n_output):
    output_softmax[i] = output_exp[i] / softmax_sum


@ti.kernel
def layer5():
  for i in range(n_output):
    loss.atomic_add(-gt[i] * ti.log(output_softmax[i]) + (gt[i] - 1) * ti.log(
      1 - output_softmax[i]))


@ti.kernel
def gd():
  for i in range(n_input):
    for j in range(n_output):
      weight[i, j] -= learning_rate * weight.grad[i, j]


with open('mnist.pkl', 'rb') as f:
  mnist = pickle.load(f)

training_images = mnist['training_images']
training_labels = mnist['training_labels']
test_images = mnist['test_images']
test_labels = mnist['test_labels']

init_weights()

learning_rate = 1e-3

losses = []

for k in range(12000):
  img_id = random.randrange(0, len(training_images))
  img = training_images[img_id]

  for i in range(n_input):
    input[i] = img[i] / 255

  for j in range(n_output):
    gt[j] = int(training_labels[img_id] == j)

  clear_weight_grad()
  clear_outputs()
  softmax_sum[None] = 0
  loss[None] = 0

  layer1()
  layer2()
  layer3()
  layer4()
  layer5()


  l = loss[None]

  losses.append(l)
  if len(losses) > 100:
    losses = losses[-100:]
    print('loss : ', sum(losses) / len(losses))

  loss.grad[None] = 1
  softmax_sum.grad[None] = 0

  layer5.grad()
  layer4.grad()
  layer3.grad()
  layer2.grad()
  layer1.grad()

  gd()

ntest = len(test_images) // 5
acc = 0
for k in range(ntest):
  img = test_images[k]

  for i in range(n_input):
    input[i] = img[i] / 255

  for j in range(n_output):
    gt[j] = int(test_labels[k] == j)

  clear_outputs()
  clear_weight_grad()
  loss[None] = 0

  layer1()

  logits = []
  for j in range(n_output):
    logits.append(output[j])
  pred = logits.index(max(logits))
  acc += int(pred == test_labels[k])

print('test accuracy: {:.2f}%'.format(100 * acc / ntest))
