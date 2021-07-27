import pickle
import random

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True

input = ti.field(ti.f32)

weight1 = ti.field(ti.f32)
output1 = ti.field(ti.f32)
output1_nonlinear = ti.field(ti.f32)

weight2 = ti.field(ti.f32)
output = ti.field(ti.f32)
output_exp = ti.field(ti.f32)
output_softmax = ti.field(ti.f32)
softmax_sum = ti.field(ti.f32)
gt = ti.field(ti.f32)
loss = ti.field(ti.f32)
learning_rate = ti.field(ti.f32)

n_input = 28**2
n_hidden = 500
n_output = 10

ti.root.dense(ti.i, n_input).place(input)
ti.root.dense(ti.ij, (n_input, n_hidden)).place(weight1)
ti.root.dense(ti.i, n_hidden).place(output1)
ti.root.dense(ti.i, n_hidden).place(output1_nonlinear)
ti.root.dense(ti.ij, (n_hidden, n_output)).place(weight2)
ti.root.dense(ti.i, n_output).place(gt)
ti.root.dense(ti.i, n_output).place(output)
ti.root.dense(ti.i, n_output).place(output_exp)
ti.root.dense(ti.i, n_output).place(output_softmax)
ti.root.place(softmax_sum)
ti.root.place(loss, learning_rate)

ti.root.lazy_grad()


@ti.kernel
def init_weights1():
    for i in range(n_input):
        for j in range(n_hidden):
            weight1[i, j] = ti.random() * 0.005


@ti.kernel
def init_weights2():
    for i in range(n_hidden):
        for j in range(n_output):
            weight2[i, j] = ti.random() * 0.005


@ti.kernel
def clear_weight1_grad():
    for i in range(n_input):
        for j in range(n_hidden):
            weight1.grad[i, j] = 0


@ti.kernel
def clear_weight2_grad():
    for i in range(n_hidden):
        for j in range(n_output):
            weight2.grad[i, j] = 0


def clear_output1():
    for i in range(n_hidden):
        output1[i] = 0
        output1_nonlinear[i] = 0
        output1.grad[i] = 0
        output1_nonlinear.grad[i] = 0


def clear_output2():
    for i in range(n_output):
        output[i] = 0
        output_exp[i] = 0
        output_softmax[i] = 0
        output.grad[i] = 0
        output_exp.grad[i] = 0
        output_softmax.grad[i] = 0


def layer(func):
    layer.list.append(func)


layer.list = []


@layer
@ti.kernel
def w1():
    for i in range(n_input):
        for j in range(n_hidden):
            output1[j].atomic_add(input[i] * weight1[i, j])


@layer
@ti.kernel
def nonlinear1():
    for i in range(n_hidden):
        output1_nonlinear[i] = ti.tanh(output1[i])


@layer
@ti.kernel
def w2():
    for i in range(n_hidden):
        for j in range(n_output):
            output[j].atomic_add(output1_nonlinear[i] * weight2[i, j])


@layer
@ti.kernel
def nonlinear2():
    for i in range(n_output):
        output_exp[i] = ti.exp(output[i])


@layer
@ti.kernel
def reduce():
    for i in range(n_output):
        softmax_sum.atomic_add(output_exp[i] + 1e-6)


@layer
@ti.kernel
def softmax():
    for i in range(n_output):
        output_softmax[i] = output_exp[i] / softmax_sum


@layer
@ti.kernel
def xent():
    for i in range(n_output):
        loss.atomic_add(-gt[i] * ti.log(output_softmax[i]) +
                        (gt[i] - 1) * ti.log(1 - output_softmax[i]))


@ti.kernel
def gd_w1():
    for i in range(n_input):
        for j in range(n_hidden):
            weight1[i, j] -= learning_rate * weight1.grad[i, j]


@ti.kernel
def gd_w2():
    for i in range(n_hidden):
        for j in range(n_output):
            weight2[i, j] -= learning_rate * weight2.grad[i, j]


try:
    f = open('mnist.pkl', 'rb')
except FileNotFoundError:
    raise FileNotFoundError(
        'mnist.pkl not found, please run examples/mnist_download_data.py first.'
    )
with f:
    mnist = pickle.load(f)

training_images = mnist['training_images']
training_labels = mnist['training_labels']
test_images = mnist['test_images']
test_labels = mnist['test_labels']

init_weights1()
init_weights2()


def test_accuracy():
    ntest = len(test_images) // 10
    acc = 0
    for k in range(ntest):
        img = test_images[k]

        for i in range(n_input):
            input[i] = img[i] / 255

        for j in range(n_output):
            gt[j] = int(test_labels[k] == j)

        clear_output1()
        clear_output2()
        clear_weight1_grad()
        clear_weight2_grad()

        loss[None] = 0

        for f in layer.list:
            f()

        logits = []
        for j in range(n_output):
            logits.append(output[j])
        pred = logits.index(max(logits))
        acc += int(pred == test_labels[k])

    return acc / ntest


losses = []
accs = []
niter = 10000
for k in range(niter):
    learning_rate = 5e-3 * (0.1**(2 * k // niter))
    img_id = random.randrange(0, len(training_images))
    img = training_images[img_id]

    for i in range(n_input):
        input[i] = img[i] / 255

    for j in range(n_output):
        gt[j] = int(training_labels[img_id] == j)

    clear_output1()
    clear_output2()
    clear_weight1_grad()
    clear_weight2_grad()

    softmax_sum[None] = 0
    loss[None] = 0

    for f in layer.list:
        f()

    l = loss[None]

    losses.append(l)
    losses = losses[-100:]
    if k % 100 == 0:
        print('k =', k, ' loss : ', sum(losses) / len(losses))
    if k % 1000 == 0:
        acc = test_accuracy()
        print('test accuracy: {:.2f}%'.format(100 * acc))
        accs.append(acc)

    loss.grad[None] = 1
    softmax_sum.grad[None] = 0

    for f in reversed(layer.list):
        f.grad()

    gd_w1()
    gd_w2()
