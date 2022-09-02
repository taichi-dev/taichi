---
sidebar_position: 2
---

# Accelerate Python with Taichi

Taichi is a domain-specific language *embedded* in Python. One of its key features is that Taichi can accelerate computation-intensive Python programs and help these programs [achieve comparable performance to C/C++ or even CUDA](https://docs.taichi-lang.org/blog/is-taichi-lang-comparable-to-or-even-faster-than-cuda). This makes Taichi much better positioned in the area of scientific computation.

In the following sections, we provide two examples to give you a sense as to how much acceleration Taichi can bring to your Python programs.

## Count the primes

Large-scale or nested for loops in Python always leads to poor runtime performance. The following demo counts the primes within a specified range and involves nested for loops (see [here](https://github.com/taichi-dev/faster-python-with-taichi/blob/main/count_primes.py) for the complete version). Simply by importing Taichi or switching to Taichi's GPU backends, you will see a significant boost to the overall performance:

```python
"""Count the prime numbers in the range [1, n]
"""

# Checks if a positive integer is a prime number
def is_prime(n: int):
    result = True
    # Traverses the range between 2 and sqrt(n)
    # - Returns False if n can be divided by one of them;
    # - otherwise, returns True
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
        result = False
        break
    return result

# Traverses the range between 2 and n
# Counts the primes according to the return of is_prime()
def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
        count += 1

    return count

print(count_primes(1000000))
```

1. Save the code as **count_prime.py** and run the following command in your terminal:

```bash
time python count_primes.py
```
   *The count of prime numbers along with the execution time appears on the screen. It takes 2.235s to run this program.*
```bash
78498

real        0m2.235s
user        0m2.235s
sys        0m0.000s
```

2.  Now, let's change the code a bit: import Taichi to your Python code and initialize it using the CPU backend:

```python
import taichi as ti
ti.init(arch=ti.cpu)
```

3. Decorate `is_prime()` with `@ti.func` and `count_primes()` with `@ti.kernel`:

> - Taichi's compiler compiles the Python code decorated with `@ti.kernel` onto different devices, such as CPU and GPU, for high-performance computation.
> - See [Kernels & Functions](../kernels/syntax.md) for a detailed explanation of Taichi's core concepts: kernels and functions.

```python
@ti.func
def is_prime(n: int):
    result = True
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result

@ti.kernel
def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
            count += 1

    return count
```

4. Rerun **count_primes.py**：

```bash
time python count_primes.py
```
   *The calculation speed is six times up (2.235/0.363).*

```bash
78498

real        0m0.363s
user        0m0.546s
sys        0m0.179s
```

5.  Increase `N` tenfold to `10,000,000` and rerun **count_primes.py**:

   *The calculation time with Taichi is 0.8s vs. 55s with Python only. The calculation speed with Taichi is 70x up.*

6. Change Taichi's backend from CPU to GPU and give it a rerun:

```python
ti.init(arch=ti.gpu)
```
   *The calculation time with Taichi is 0.45s vs. 55s with Python only. The calculation speed with Taichi is taken further to 120x up.*

## Dynamic programming: longest common subsequence

The core philosophy behind dynamic programming is that it sacrifices some storage space for less execution time and stores intermediate results to avoid repetitive computation. In the following section, we will walk you through a complete implementation of DP, and demonstrate another area where Taichi can make a real 'acceleration'.

The example below follows the philosophy of DP to work out the length of the longest common subsequence (LCS) of two given sequences. For instance, the LCS of sequences a = [**0**, **1**, 0, 2, **4**, **3**, 1, **2**, 1] and b = [4, **0**, **1**, **4**, 5, **3**, 1, **2**],  is [0, 1, 4, 3, 1, 2], and the LCS' length is six. Let's get started:

1. Import NumPy and Taichi to your Python program:

```python
import taichi as ti
import numpy as np
```

2. Initialize Taichi:

```python
ti.init(arch=ti.cpu)
```

3. Create two 15,000-long NumPy arrays of random integers in the range of [0, 100] to compare:

```python
N = 15000
a_numpy = np.random.randint(0, 100, N, dtype=np.int32)
b_numpy = np.random.randint(0, 100, N, dtype=np.int32)
```

4. Here we define an `N`&times;`N` [Taichi field](../basic/field.md) `f`, using its `[i, j]`-th element to represent the length of the LCS of sequence `a`'s first `i` elements and sequence `b`'s first `j` elements:

```python
f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))
```

5. Now we turn the dynamic programming issue to the traversal of a field `f`, where `a` and `b` are the two sequences to compare:

```python
f[i, j] = max(f[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
              max(f[i - 1, j], f[i, j - 1]))
```

6. Define a kernel function `compute_lcs()`, which takes in two sequences and works out the length of their LCS.

```python
@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
       len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=True) # Disable auto-parallelism in Taichi
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            f[i, j] = max(f[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
                          max(f[i - 1, j], f[i, j - 1]))

    return f[len_a, len_b]
```

> - NumPy arrays are stored as ndarray in Taichi.
> - Ensure that you set `ti.loop_config(serialize=True)` to disable auto-parallelism in Taichi. The iterations *here* should not happen in parallelism because the computation of a loop iteration is dependent on its previous iterations.

7. Print the result of `compute_lcs(a_numpy, b_numpy)`.
   *Now you get the following program:*

```python
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

benchmark = True

N = 15000

f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))

if benchmark:
    a_numpy = np.random.randint(0, 100, N, dtype=np.int32)
    b_numpy = np.random.randint(0, 100, N, dtype=np.int32)
else:
    a_numpy = np.array([0, 1, 0, 2, 4, 3, 1, 2, 1], dtype=np.int32)
    b_numpy = np.array([4, 0, 1, 4, 5, 3, 1, 2], dtype=np.int32)

@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=True) # Disable auto-parallelism in Taichi
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
               f[i, j] = max(f[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
                          max(f[i - 1, j], f[i, j - 1]))

    return f[len_a, len_b]


print(compute_lcs(a_numpy, b_numpy))
```

8. Save the above code as **lcs.py** and run:

```bash
time python lcs.py
```
   *The system prints the length of the LCS, along with the execution time.*

```bash
2721

real        0m1.409s
user        0m1.112s
sys        0m0.549s
```

In [this repo](https://github.com/taichi-dev/faster-python-with-taichi/blob/main/lcs.py), we provide our implementation of this dynamic planning algorithm in Taichi and NumPy:

- With Python only, it takes 476s to work out the length of the LCS of two 15,000-long random sequences.
- With Taichi, it only takes about 0.9s and sees an up to 500x speed up!

:::note
The actual execution time may vary depending on your machine, but we believe that the performance improvements you will see is comparable to ours.
:::
