"""
Originally created on Sat Nov 18 12:39:16 2016 by manojacharya
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
from scipy.misc import imread, imresize
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json


means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
  [transforms.ToTensor(),
   normalize])


def imshow(inp, title=None):
  """Imshow for Tensor."""
  plt.figure()
  inp = inp.data[0]
  plt.imshow(inp)
  plt.axis('off')
  if title is not None:
    plt.title(title)


with open('imagenet.json') as f:
  imagenet_labels = json.load(f)

# In[model]:

model = models.vgg16(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

def preprocess_and_forward(img):
  img = (img - torch.tensor(means[None, None, :])) / torch.tensor(std[None, None, :])
  img = img[None, :, :, :]
  img = torch.transpose(torch.transpose(img, 1, 3), 2, 3)
  return model(img)
  
  

def predict(img):
  img = (img - torch.tensor(means[None, None, :])) / torch.tensor(std[None, None, :])
  img = img[None, :, :, :]
  img = torch.transpose(torch.transpose(img, 1, 3), 2, 3)
  
  pred_raw = model(img)
  pred = F.softmax(pred_raw)
  _, indices = torch.topk(pred, k=1)
  for ind in indices.data.numpy().ravel():
    print("%.2f%% , class: %s (%s)" % (
    100 * pred.data[0][ind], str(ind), imagenet_labels[ind]))


# In[image ]:

def main():
  # everything is RGB instead of BGR in this file
  # import cv2
  peppers = imread("squirrel.jpg")
  # cv2.imshow('p', peppers)
  # cv2.waitKey(0)
  img = (imresize(peppers, (224, 224)) / 255.0).astype(np.float32)
  img = torch.tensor(img)
  
  imgvar = Variable(img, requires_grad=False)
  imgvard = Variable(img, requires_grad=True)
  
  loss_fn = nn.CrossEntropyLoss()

  label = torch.LongTensor(1)
  # classify the object as this label
  label[0] = 1
  label = Variable(label)
  eps = 2 / 255.0

  # %%
  Nepochs = 10
  print("Starting ...........", predict(imgvar))
  for epoch in range(Nepochs):
    pred_raw = preprocess_and_forward(imgvard)
    loss = loss_fn(pred_raw, label)
    
    loss.backward()
    imgvar -= imgvard.grad * 1
    
    print('epoch: {}/{}, loss: {}'.format(epoch + 1, Nepochs, loss.item()))
    predict(imgvard)
  print('Finished Training')

  # %%
  imshow(imgvard)

  # %%
  '''
  plt.figure()
  diffimg = diff[0].numpy()
  diffimg = diffimg.transpose((1, 2, 0))
  plt.imshow(diffimg)
  plt.show()
  '''

if __name__ == '__main__':
  main()