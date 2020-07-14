from time import time
t = time()
import taichi
t = time() - t
from colorama import Fore
print('\nTotal load time:', Fore.LIGHTRED_EX, t * 1000, Fore.RESET, 'ms\n')
