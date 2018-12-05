import os

os.system("rsync -avz ./ yuanming@kun:/home/yuanming/repos/taichi/projects/taichi_lang")
os.system('ssh yuanming@kun "source ~/.zshrc && export PYTHONIOENCODING=utf-8 &&  ti build && ti test_tlang"')