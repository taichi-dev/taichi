import os
import pwd

def execute_command(line):
  print(line)
  os.system(line)
  
if __name__ == '__main__':
  usename = pwd.getpwuid(os.getuid())[0]
  execute_command('sudo apt-get install sudo apt-get install git python3 git build-essential cmake make g++ libtbb-dev alien dpkg-dev debhelper ffmpeg python3-tk python3-pip')
  execute_command('cd ~')
  execute_command('mkdir -p repos')
  if os.path.exists('taichi'):
    print('Please remove original taichi installation in ~/repos/taichi')
    exit(-1)
  execute_command('git clone git@github.com:yuanming-hu/taichi.git')
  execute_command('git checkout dev')
  execute_command('git clone https://github.com/yuanming-hu/taichi_runtime runtimes')
  
  # Make sure there is no existing Taichi ENV
  execute_command('echo "export TAICHI_NUM_THREADS=4" > ~/.bashrc')
  execute_command('echo "export TAICHI_ROOT_DIR=/home/yuanming/repos/" > ~/.bashrc')
  execute_command('echo "export PYTHONPATH=$TAICHI_ROOT_DIR/taichi/python/:$PYTHONPATH" > ~/.bashrc')
  
  execute_command('python3 -c "import taichi"')
  
