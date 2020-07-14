from time import time
from colorama import Fore
t = time()
import taichi as ti
print('\nTotal load time:', Fore.LIGHTRED_EX, (time() - t) * 1000, Fore.RESET, 'ms\n')

ti.profiler.print()
