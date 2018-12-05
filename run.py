import os

os.system("rsync --exclude=.git --exclude=_tlang_cache -avz ./ yuanming@kun:/home/yuanming/repos/taichi/projects/taichi_lang")
os.system('ssh yuanming@kun "source ~/.zshrc && export PYTHONIOENCODING=utf-8 &&  ti build"')
#os.system('ssh yuanming@kun "source ~/.zshrc && export PYTHONIOENCODING=utf-8 &&  ti build && ti test_tlang"')
