import math

import yaml

with open('benchmark.yml') as f:
    data = yaml.load(f)

records = {}

for case in data:
    for metrics in ['exec_t', 'launched_tasks']:
        for arch in ['x64', 'cuda']:
            key = metrics, arch

            d = []
            for scheme in ['sync', 'async']:
                d.append(data[case][metrics][arch][scheme])

            if key not in records:
                records[key] = []

            records[key].append(d[0] / d[1])

for k in records:
    rec = records[k]
    # Compute geometric mean
    p = 1
    for r in rec:
        p *= r
    p = p**(1 / len(rec))
    print(f'{k}, {p:.3f}x')
