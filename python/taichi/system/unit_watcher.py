from taichi.core import tc_core
from taichi.misc.settings import get_output_path, get_repo_directory
from taichi.misc.util import get_unique_task_id
import os
import shutil
import time


class UnitWatcher:
    def __init__(self, target_file):
        self.target_file = target_file
        self.src = ''
        self.dir = get_output_path(get_unique_task_id())
        os.mkdir(self.dir)
        self.unit_dll = tc_core.create_unit_dll()
        self.src_path = os.path.join(self.dir, 'unit.cpp')
        self.dll_path = os.path.join(self.dir, 'build', 'libunit.dylib')
        self.cmakelists_path = os.path.join(self.dir, 'CMakeLists.txt')
        cmakelists_src = os.path.join(get_repo_directory(), 'python', 'taichi',
                                      'misc', 'CMakeLists.txt')
        shutil.copy(cmakelists_src, self.cmakelists_path)

    def load_src(self):
        with open(self.target_file, 'r') as f:
            self.src = f.read()

    def need_update(self):
        old_src = self.src
        self.load_src()
        return old_src != self.src

    def update(self):
        while True:
            self.load_src()
            with open(self.src_path, 'w') as f:
                f.write(self.src)
            if not self.build():
                time.sleep(3)
                continue
            self.reload(self.dll_path)
            break

    def build(self):
        # TODO: make it work for Windows
        old_dir = os.getcwd()
        os.chdir(self.dir)
        succ = os.system('cmake . && make') == 0
        os.chdir(old_dir)
        return succ

    def reload(self, dll_path):
        if self.unit_dll.loaded():
            self.unit_dll.close_dll()
        self.unit_dll.open_dll(dll_path)
