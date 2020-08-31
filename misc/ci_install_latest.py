import subprocess
import sys


with open('docs/version') as f:
    version = f.read().strip()

requirement = 'taichi==' + version
subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
