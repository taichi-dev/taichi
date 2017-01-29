import taichi as tc

class UnitManager:
    def __init__(self):
        self.units = {}

    def load(self, dll_path):
        self.units[dll_path] = tc.core.create_unit_dll()
        self.units[dll_path].open_dll(dll_path)

    def unload(self, dll_path):
        pass




dll_path = tc.settings.get_output_path('cpp/build/libunit.dylib')

unit_dll = tc.core.create_unit_dll()

unit_dll.open_dll(dll_path)

sdf = tc.core.create_sdf('new_sdf')
print sdf.eval(tc.Vector(2, 2, 2))
sdf = None
#unit_dll.close_dll()
