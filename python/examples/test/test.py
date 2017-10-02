import time
import taichi as tc
from taichi.gui.image_viewer import show_image

tc.core.test_vector_and_matrix()
exit()

tc.core.print_all_units()
tc.core.test()

tex = tc.Texture('const', value=(0.1, 0.1, 0.1, 0.1))
tex = tex.flip(0)

try:
  tc.core.test_raise_error()
except Exception as e:
  print 'Exception:', e

img = tc.util.imread(tc.get_asset_path('textures/vegas.jpg'))
show_image('img', img)
time.sleep(1)

print 'Testing passed.'
