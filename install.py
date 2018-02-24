import os
import pwd
import sys
import platform
import argparse


# Utils

def get_username():
  if build_type == 'ci':
    os.environ['TC_CI'] = '1'
    username = 'travis'
  else:
    username = pwd.getpwuid(os.getuid())[0]
  return username

def check_command_existence(cmd):
  return os.system('type {}'.format(cmd)) == 0

def execute_command(line):
  print('Executing command:', line)
  return os.system(line)

def get_os_name():
  name = platform.platform()
  if name.lower().startswith('darwin'):
    return 'osx'
  elif name.lower().startswith('windows'):
    return 'win'
  elif name.lower().startswith('linux'):
    return 'linux'
  assert False, "Unknown platform name %s" % name

def get_default_directory_name():
  os = get_os_name()
  username = get_username()
  if os == 'linux':
    return '/home/{}/'.format(username)
  elif os == 'osx':
    return '/Users/{}/'.format(username)
  else:
    assert False, 'Unsupported OS: {}'.format(os)


# (Stateful) Installer class

class Installer:
  def __init__(self):
    parser = argparse.ArgumentParser()
    parser.parse_args()
    self.build_type = None

  def detect_or_setup_repo(self):
    cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
    print("Current directory:", cwd)

    if os.path.exists(os.path.join(cwd, 'include', 'taichi')):
      print("Taichi source detected.")
      self.root_dir = cwd
    else:
      print("Cloning taichi from github...")
      os.chdir(get_default_directory_name())
      execute_command('mkdir -p repos')
      os.chdir('repos')
      if os.path.exists('taichi'):
        print('Existing taichi installation detected.')
        print('Please remove original taichi installation in ~/repos/')
        exit(-1)
      self.root_dir = os.curdir()
      execute_command('git clone https://github.com/yuanming-hu/taichi.git')
      os.chdir('taichi')
      execute_command('git clone https://github.com/yuanming-hu/taichi_runtime external/lib')

  def run(self):
    assert get_os_name() in ['linux', 'osx'], \
      'Platform {} is not currently supported by this script. Please install manually.'.format(get_os_name())
    global build_type
    if len(sys.argv) > 1:
      self.build_type = sys.argv[1]
      print('Build type: ', build_type)
    else:
      build_type = 'default'

    print('Build type = {}'.format(build_type))

    assert build_type in ['default', 'ci']

    check_command_existence('wget')
    try:
      import pip
    except Exception as e:
      print(e)
      execute_command('wget https://bootstrap.pypa.io/get-pip.py')
      execute_command('python3 get-pip.py --user')
      execute_command('rm get-pip.py')

    if get_os_name() == 'osx':
      # Check command existence
      check_command_existence('git')
      check_command_existence('cmake')
      check_command_existence('python3')
      # TODO: ship ffmpeg
      #check_command_existence('ffmpeg')
    else:
      check_command_existence('sudo')
      execute_command('sudo apt-get update')
      execute_command('sudo apt-get install -y python3-dev git build-essential cmake make g++ python3-tk ffmpeg')

    self.detect_or_setup_repo()


    #TODO: Make sure there is no existing Taichi ENV
    execute_command('echo "export TAICHI_NUM_THREADS=8" >> ~/.bashrc')
    execute_command('echo "export TAICHI_ROOT_DIR={}" >> ~/.bashrc'.format(self.root_dir))
    execute_command('echo "export PYTHONPATH=\$TAICHI_ROOT_DIR/taichi/python/:\$PYTHONPATH" >> ~/.bashrc')
    execute_command('echo "export PATH=\$TAICHI_ROOT_DIR/taichi/bin/:\$PATH" >> ~/.bashrc')

    os.environ['TAICHI_NUM_THREADS'] = '8'
    os.environ['TAICHI_ROOT_DIR'] = self.root_dir
    os.environ['PYTHONPATH'] = '{}/taichi/python/:'.format(self.root_dir) + os.environ.get('PYTHONPATH', '')
    os.environ['PATH'] = os.path.join(self.root_dir, 'taichi/bin') + ':' + os.environ.get('PATH', '')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    print('PYTHONPATH={}'.format(os.environ['PYTHONPATH']))

    if execute_command('echo $PYTHONPATH; python3 -c "import taichi as tc"') == 0:
      if execute_command('ti') != 0:
        print('  Warning: shortcut "ti" does not work.')
      if execute_command('taichi') != 0:
        print('  Warning: shortcut "taichi" does not work.')
      print('  Successfully Installed Taichi at ~/repos/taichi.')
      print('  Please execute')
      print('    source ~/.bashrc')
      print('  or restart your terminal.')
    else:
      print('  Error: installation failed.')
      exit(-1)

if __name__ == '__main__':
  installer = Installer()
  installer.run()
