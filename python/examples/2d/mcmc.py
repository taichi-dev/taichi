import numpy as np
import matplotlib.pyplot as plt
import random
from math import *

n = 64
N_SAMPLES = 1000000

def gaussian(x, mu, sigma):
    return exp(-(x - mu) ** 2 / sigma ** 2)

def eval(x):
    return gaussian(x, n * 0.2, n * 0.1) + gaussian(x, n * 0.7, n * 0.1)

def mh():
    T = N_SAMPLES
    samples = [0]
    for i in range(T):
        x = samples[-1]
        if (i + 1) % (T / 10) == 0:
            print x, '@' + str(100 * i / T) + '%'
            plt.hist(samples, bins=n, range=(0, n))
            plt.draw()
            plt.show()
        y = (n + x + random.randint(0, 1) * 2 - 1) % n
        if eval(x) > 0:
            acceptance = min(eval(y) / eval(x), 1)
            if random.random() < acceptance:
                x = y
        samples.append(x)

def dwmh():
    T = N_SAMPLES
    samples = [0]
    weights = [1]
    for i in range(T):
        x = samples[-1]
        w = weights[-1]
        if i % (T / 10) == 0:
            print 'x', x, 'w', w, '@' + str(100 * i / T) + '%'
            plt.hist(samples, weights=weights, bins=n, range=(0, n))
            plt.draw()
            plt.show()
        y = (n + x + random.randint(0, 1) * 2 - 1) % n
        fx, fy = eval(x), eval(y)
        theta = 0.1
        accepted = False
        if fx > 0:
            r = w * fy / fx
            a = r / (r + theta)
            if random.random() < r:
                accepted = True
        else:
            a = 0
            
        if accepted:
            x, w = y, r / a
        else:
            x, w = x, w / (1 - a)

        samples.append(x)
        weights.append(w)

def dwis_r():
    samples = [0]
    weights = [1]
    n_min = 5
    n_low = 10
    n_up = 20
    n_max = 40
    population = [(random.randint(0, n - 1), 1) for i in range(n_min)]
    def apepcs(population_in):
        weights = map(lambda xw: xw[1], population)
        W_low = sum(weights) / n_up
        W_up = sum(weights) / n_low

        def population_control(population, l):
            w_low = W_low * l
            w_up = W_up * l
            new_population = []
            for i in range(len(population)):
                x, w = population[i]
                # Pruned
                if w < w_low:
                    if random.random() < 1 - w / w_low:
                        pass # Drop
                    else:
                        new_population.append((x, w_low))

                # Enriched
                elif w > w_up:
                    d = int(floor(w / w_up) + 1)
                    w /= d
                    for k in range(d):
                        new_population.append((x, w))
                else:
                    new_population.append((x, w))

            return new_population

        population_in = population_control(population_in, 1)

        # Checking
        if len(population_in) > n_max:
            population_in = population_control(population_in, 2)
        elif len(population_in) < n_min:
            population_in = population_control(population_in, 1.0 / 2)

        return population_in

    def dwmh(xw, theta):
        x, w = xw
        y = (n + x + random.randint(0, 1) * 2 - 1) % n
        fx, fy = eval(x), eval(y)
        accepted = False
        r = 0
        if fx > 0 or theta == 0:
            r = w * fy / fx
            a = r / (r + theta)
            if random.random() < r:
                accepted = True
        else:
            a = 0
        if accepted:
            x, w = y, r / a
        else:
            x, w = x, w / (1 - a)
        return x, w

    T = N_SAMPLES / n_max
    for i in range(T):
        if i % (T / 10) == 0:
            print max(weights), 'population', len(population)
            print '@' + str(100 * i / T) + '%'
            plt.hist(samples, weights=weights, bins=n, range=(0, n))
            plt.draw()
            plt.show()
        w_c = 10
        theta = 1 if sum(map(lambda xw: xw[1], population)) / n_low < w_c else 0.0
        theta = 0.1
        for i in range(len(population)):
            population[i] = dwmh(population[i], theta)
        population = apepcs(population)

        for x, w in population:
            samples.append(x)
            weights.append(w)

if __name__ == '__main__':
    mh()