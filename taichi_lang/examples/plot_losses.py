import matplotlib.pyplot as plt
import pickle
import sys
from scipy.ndimage.filters import gaussian_filter

losses = pickle.load(open(sys.argv[1], 'rb'))

iterations = len(losses[list(losses.keys())[0]][0])

for key, item in losses.items():
  for i in range(len(item)):
    item[i] = gaussian_filter(item[i], 3)

colors = ['r', 'g']
for id, (key, item) in enumerate(losses.items()):
  l = iterations
  mean_loss = []
  max_loss = []
  min_loss = []
  for i in range(l):
    L = 0
    MAX = -1e10
    MIN = 1e10
    for j in range(len(item)):
      t = item[j][i]
      L += t
      MAX = max(MAX, t)
      MIN = min(MIN, t)
    
    mean_loss.append(L / len(item))
    max_loss.append(MAX)
    min_loss.append(MIN)
  
  plt.fill_between(list(range(iterations)), min_loss, max_loss, color=colors[id], alpha=0.3)
  plt.plot(mean_loss, label='TOI' if key else 'Naive', color=colors[id])
  

fig = plt.gcf()
plt.legend()
fig.set_size_inches(4, 3)
plt.title('Rigid Body Robot 1')
plt.ylabel('Loss')
plt.xlabel('Gradient Descent Iterations')
plt.tight_layout()
plt.legend()
plt.show()
