import os
import pwd
import sys

def execute_command(line):
  print(line)
  return os.system(line)
  
if __name__ == '__main__':
  usename = pwd.getpwuid(os.getuid())[0]
  
  if len(sys.argv) > 1:
    build_type = sys.argv[1]
    print('Build type: ', build_type)
    assert build_type in ['default', 'ci']
    
  try:
    import pip
  except Exception as e:
    print(e)
    execute_command('wget https://bootstrap.pypa.io/get-pip.py')
    execute_command('python3 get-pip.py --user')
    execute_command('rm get-pip.py')
  execute_command('sudo apt-get update')
  execute_command('sudo apt-get install -y python3-dev git build-essential cmake make g++ python3-tk')
  execute_command('cd /home/{}/'.format(usename))
  execute_command('mkdir -p repos')
  os.chdir('repos')
  if os.path.exists('taichi'):
    print('Please remove original taichi installation in ~/repos/')
    exit(-1)
  execute_command('git clone https://github.com/yuanming-hu/taichi.git')
  os.chdir('taichi')
  execute_command('git checkout dev')
  execute_command('git clone https://github.com/yuanming-hu/taichi_runtime external/lib')
  
  # Make sure there is no existing Taichi ENV
  taichi_root_dir = "/home/{}/repos/".format(usename)
  execute_command('echo "export TAICHI_NUM_THREADS=8" >> ~/.bashrc')
  execute_command('echo "export TAICHI_ROOT_DIR={}" >> ~/.bashrc'.format(taichi_root_dir))
  execute_command('echo "export PYTHONPATH=\$TAICHI_ROOT_DIR/taichi/python/:\$PYTHONPATH" >> ~/.bashrc')
  execute_command('echo "export PATH=\$TAICHI_ROOT_DIR/taichi/bin/:\$PATH" >> ~/.bashrc')

  os.environ['TAICHI_NUM_THREADS'] = '8'
  os.environ['TAICHI_ROOT_DIR'] = taichi_root_dir
  os.environ['PYTHONPATH'] = '{}/taichi/python/;'.format(taichi_root_dir) + os.environ['PYTHONPATH']
  
  sys.path.append(os.path.join(taichi_root_dir, 'bin'))
  sys.path.append(os.path.join(taichi_root_dir, 'python'))
  execute_command('python3 -c "import taichi as tc" && echo "Successfully Installed Taichi at ~/repos."')

