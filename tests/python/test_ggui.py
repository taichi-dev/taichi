import numpy as np
import pytest

import taichi as ti

REGENERATE_IMAGES = True

def verify_image(window, image_name):
    ground_truth_name = f"tests/python/images/{image_name}_truth.png"
    actual_name = f"tests/python/images/{image_name}_actual_{str(ti.cfg.arch)}.png"
    #ground_truth_name = actual_name
    if REGENERATE_IMAGES:
        window.write_image(ground_truth_name)
    else:
        window.write_image(actual_name)

        ground_truth_np = ti.imread(ground_truth_name)
        actual_np = ti.imread(actual_name)
        assert len(ground_truth_np.shape) == len(actual_np.shape)
        for i in range(len(ground_truth_np.shape)):
            assert ground_truth_np.shape[i] == actual_np.shape[i]
        diff = ground_truth_np - actual_np
        mse = np.mean(diff*diff)
        assert mse <= 0.01



@ti.test(arch=[ti.cuda])
def test_canvas():
    window = ti.ui.Window('test',(640,480),show_window = False)
    canvas = window.get_canvas()

    n_circles_0 = 10
    circle_positions_0 = ti.Vector.field(2,ti.f32,shape = n_circles_0)
    for i in range(n_circles_0):
        circle_positions_0[i] = ti.Vector([0.1,i*0.1]) 

    n_circles_1 = 10
    circle_positions_1 = ti.Vector.field(2,ti.f32,shape = n_circles_1)
    circle_colors_1 = ti.Vector.field(3,ti.f32,shape = n_circles_1)
    for i in range(n_circles_0):
        circle_positions_1[i] = ti.Vector([0.2,i*0.1]) 
        circle_colors_1[i] = ti.Vector([i*0.1, 1.0 - i*0.1, 0.5]) 
    
    while window.running:
    
        canvas.circles(circle_positions_0, radius = 0.05, color = (1,0,0))
        canvas.circles(circle_positions_1, radius = 0.05, per_vertex_color = circle_colors_1)
        verify_image(window,'test_canvas_0')
        #window.write_image('test_canvas_0.png')
        #window.show()
    