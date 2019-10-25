import numpy as np
import matplotlib.pyplot as plt
import os

baseline = []
ours = []
models = []

for fn in sorted(os.listdir('results')):
    models.append(fn[:-8])
    with open('results/' + fn) as f:
        rets = list(map(float, f.readlines()))
        print(rets)
        ours.append(1.0 / rets[0])
        baseline.append(1.0 / rets[1])


ind = np.arange(len(baseline))  # the x locations for the groups
width = 0.35  # the width of the bars

for i in range(len(ours)):
    base = baseline[i]
    baseline[i] /= base
    ours[i] /= base

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width/2, baseline, width,
                color='SkyBlue', label='OpenVDB')
rects2 = ax.bar(ind + width/2, ours, width,
                color='IndianRed', label='Ours')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Throughput (Normalized)')
ax.set_title('3D Sparse Grid Mean Filter Benchmark')
ax.set_xticks(ind)
ax.set_xticklabels(models)
ax.legend()


def autolabel(rects, xpos='center'):
    xpos = xpos.lower()  # normalize the case of the parameter
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{:.1f}x'.format(height), ha='center', va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

plt.show()