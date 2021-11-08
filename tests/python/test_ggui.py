import numpy as np
import pytest
import tempfile
import os
import taichi as ti

REGENERATE_GROUNDTRUTH_IMAGES = True
RENDER_REPEAT = 5

def get_temp_png():
    f = tempfile.mkstemp(suffix='.png')
    return f[1]

def write_temp_image(window):
    f = get_temp_png()
    window.write_image(f)
    os.remove(f)

def verify_image(window, image_name):
    ground_truth_name = f"tests/python/images/truth_{image_name}_truth.png"
    if REGENERATE_GROUNDTRUTH_IMAGES:
        window.write_image(ground_truth_name)
    else:
        actual_name = get_temp_png()
        window.write_image(actual_name)
        ground_truth_np = ti.imread(ground_truth_name)
        actual_np = ti.imread(actual_name)
        assert len(ground_truth_np.shape) == len(actual_np.shape)
        for i in range(len(ground_truth_np.shape)):
            assert ground_truth_np.shape[i] == actual_np.shape[i]
        diff = ground_truth_np - actual_np
        mse = np.mean(diff*diff)
        assert mse <= 0.1 # the pixel values are 0~255 
        os.remove(actual_name)

@ti.test(arch=[ti.cuda, ti.vulkan])
def test_geometry_2d():
    window = ti.ui.Window('test',(640,480),show_window = False)
    canvas = window.get_canvas()

    def render():
        # simple circles
        n_circles_0 = 10
        circle_positions_0 = ti.Vector.field(2,ti.f32,shape = n_circles_0)
        for i in range(n_circles_0):
            circle_positions_0[i] = ti.Vector([0.1,i*0.1]) 
        canvas.circles(circle_positions_0, radius = 0.05, color = (1,0,0))

        # circles with per vertex colors
        n_circles_1 = 10
        circle_positions_1 = ti.Vector.field(2,ti.f32,shape = n_circles_1)
        circle_colors_1 = ti.Vector.field(3,ti.f32,shape = n_circles_1)
        for i in range(n_circles_0):
            circle_positions_1[i] = ti.Vector([0.2,i*0.1]) 
            circle_colors_1[i] = ti.Vector([i*0.1, 1.0 - i*0.1, 0.5]) 
        canvas.circles(circle_positions_1, radius = 0.05, per_vertex_color = circle_colors_1)

        # simple triangles
        n_triangles_0 = 10
        triangles_positions_0 = ti.Vector.field(2,ti.f32,shape = 3*n_triangles_0)
        for i in range(n_triangles_0):
            triangles_positions_0[3*i] = ti.Vector([0.3,i*0.1])
            triangles_positions_0[3*i+1] = ti.Vector([0.35,i*0.1])
            triangles_positions_0[3*i+2] = ti.Vector([0.35,i*0.1+0.05]) 
        canvas.triangles(triangles_positions_0, color = (0,0,1))

        # triangles with per vertex colors and indices
        triangles_positions_1 = ti.Vector.field(2,ti.f32,shape = 4)
        triangles_colors_1 = ti.Vector.field(3,ti.f32,shape = 4)
        triangles_positions_1[0] = ti.Vector([0.4,0])
        triangles_positions_1[1] = ti.Vector([0.4,1])
        triangles_positions_1[2] = ti.Vector([0.45,0])
        triangles_positions_1[3] = ti.Vector([0.45,1])
        triangles_colors_1[0] = ti.Vector([0,0,0])
        triangles_colors_1[1] = ti.Vector([1,0,0])
        triangles_colors_1[2] = ti.Vector([0,1,0])
        triangles_colors_1[3] = ti.Vector([1,1,0])
        triangle_indices_1 = ti.Vector.field(3,ti.i32,shape= 2)
        triangle_indices_1[0] = ti.Vector([0,1,3])
        triangle_indices_1[1] = ti.Vector([0,2,3])
        canvas.triangles(triangles_positions_1,  per_vertex_color = triangles_colors_1,indices = triangle_indices_1)

        # simple lines
        n_lines_0 = 10
        lines_positions_0 = ti.Vector.field(2,ti.f32,shape = 2*n_lines_0)
        for i in range(n_lines_0):
            lines_positions_0[2*i] = ti.Vector([0.5,i*0.1])
            lines_positions_0[2*i+1] = ti.Vector([0.5,i*0.1+0.05]) 
        canvas.lines(lines_positions_0,width = 0.01, color = (0,1,0))

        # lines with per vertex colors and indices
        lines_positions_1 = ti.Vector.field(2,ti.f32,shape = 4)
        lines_colors_1 = ti.Vector.field(3,ti.f32,shape = 4)
        lines_positions_1[0] = ti.Vector([0.6,0])
        lines_positions_1[1] = ti.Vector([0.6,1])
        lines_positions_1[2] = ti.Vector([0.65,0])
        lines_positions_1[3] = ti.Vector([0.65,1])
        lines_colors_1[0] = ti.Vector([0,0,0])
        lines_colors_1[1] = ti.Vector([1,0,0])
        lines_colors_1[2] = ti.Vector([0,1,0])
        lines_colors_1[3] = ti.Vector([1,1,0])
        lines_indices_1 = ti.Vector.field(2,ti.i32,shape = 6)
        line_id = 0
        for i in range(4):
            for j in range(i+1,4):
                lines_indices_1[line_id] = ti.Vector([i,j])
                line_id += 1
        canvas.lines(lines_positions_1,width = 0.01, per_vertex_color = lines_colors_1,indices = lines_indices_1)
    
    for _ in range(RENDER_REPEAT):
        render()
        write_temp_image(window)
    render()
    verify_image(window,'test_geometry_2d')
    window.destroy()




@ti.test(arch=[ti.cuda, ti.vulkan])
def test_geometry_3d():

    window = ti.ui.Window('test',(640,480),show_window = False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.0, 0.0, 2)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    def render():
        scene.point_light(pos=(2,2, 2), color=(1, 1, 1))

        # simple particles
        num_per_dim = 32
        num_particles_0 = int(num_per_dim ** 3)
        particles_positions_0 = ti.Vector.field(3, ti.f32, shape = num_particles_0)

        scene.particles(particles_positions_0, radius=0.2, color=(0.5, 0, 0))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        write_temp_image(window)
    render()
    verify_image(window,'test_geometry_3d')
    window.destroy()


