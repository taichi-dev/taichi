# Smoothed-particle hydrodynamics (SPH) is a computational method used for simulating the mechanics of continuum media, such as solid mechanics and fluid flows.
# Here we utilize SPH to simulate a fountain, who tries to hit a target given by the user.
# The SPH simulator here implemented using Taichi is differentiable. 
# Therefore, it can be easily embedding into the training pipeline of a neural network modelled controller.  

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        '--train',
        action='store_true',
        help='whether train model, default false')
parser.add_argument('place_holder', nargs='*')
args = parser.parse_args()


TRAIN = args.train
TRAIN_OUTPUT_IMG = False
TRAIN_VISUAL = False
TRAIN_VISUAL_SHOW = False
INFER_OUTPUT_IMG = False
ti.init(arch=ti.gpu, device_memory_fraction=0.5, random_seed=5)
screen_res = (1000, 1000)

real = ti.f32
scalar = lambda: ti.field(dtype=real)


@ti.data_oriented
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for w in self.params:
            self._step(w)

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            w[I] -= min(max(w.grad[I], -20.0), 20.0) * self.lr

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


@ti.data_oriented
class Linear:
    def __init__(self, n_models, batch_size, n_steps, n_input, n_hidden, n_output, needs_grad=False, activation=False):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.activation = activation

        self.hidden = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root.dense(ti.i, self.n_models)
        self.n_hidden_node = self.batch_node.dense(ti.j, self.n_hidden)
        self.weights1_node = self.n_hidden_node.dense(ti.k, self.n_input)

        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_hidden)).place(self.hidden)
        self.batch_node.dense(ti.axes(1, 2, 3), (self.n_steps, self.batch_size, self.n_output)).place(self.output)

        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.weights1, self.bias1]

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6 / self.n_input) * 0.01
        for model_id, i, j in ti.ndrange(self.n_models, self.n_hidden, self.n_input):
            self.weights1[model_id, i, j] = (ti.random() * 2 - 1) * q1

    @ti.kernel
    def _forward(self, t: ti.i32, nn_input: ti.template()):
        for model_id, k, i, j in ti.ndrange(self.n_models, self.batch_size, self.n_hidden, self.n_input):
            self.hidden[model_id, t, k, i] += self.weights1[model_id, i, j] * nn_input[model_id, t, k, j]
        if ti.static(self.activation):
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = ti.tanh(self.hidden[model_id, t, k, i] + self.bias1[model_id, i])
        else:
            for model_id, k, i in ti.ndrange(self.n_models, self.batch_size, self.n_hidden):
                self.output[model_id, t, k, i] = self.hidden[model_id, t, k, i] + self.bias1[model_id, i]

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.
        for I in ti.grouped(self.output):
            self.output[I] = 0.

    def forward(self, t, nn_input):
        self._forward(t, nn_input)

    def dump_weights(self, name="save.pkl"):
        w_val = []
        for w in self.parameters():
            w = w.to_numpy()
            w_val.append(w[0])
        pkl.dump(w_val, open(name, "wb"))

    def load_weights(self, name="save.pkl", model_id=0):
        w_val = pkl.load(open(name, 'rb'))
        self.load_weights_from_value(w_val, model_id)

    def load_weights_from_value(self, w_val, model_id=0):
        for w, val in zip(self.parameters(), w_val):
            if val.shape[0] == 1:
                val = val[0]
            self.copy_from_numpy(w, val, model_id)

    @ti.kernel
    def copy_from_numpy(self, dst: ti.template(), src: ti.ext_arr(), model_id: ti.i32):
        for I in ti.grouped(src):
            dst[model_id, I] = src[I]


# NN model
model_num = 1
steps = 128
n_input = 3
n_hidden = 32
n_output = 16
n_output_act = 3
learning_rate = 1e-3
loss = ti.field(float, shape=(), needs_grad=True)


if TRAIN:
    batch_size = 16
    input_states = ti.field(float, shape=(model_num, steps, batch_size, n_input), needs_grad=True)
    fc1 = Linear(n_models=model_num, batch_size=batch_size, n_steps=steps, n_input=n_input, n_hidden=n_hidden,
                 n_output=n_output, needs_grad=True, activation=False)
    fc2 = Linear(n_models=model_num, batch_size=batch_size, n_steps=steps, n_input=n_output, n_hidden=n_hidden,
                 n_output=n_output_act, needs_grad=True, activation=True)
    fc1.weights_init()
    fc2.weights_init()
    NNs = [fc1, fc2]
    parameters = []
    for layer in NNs:
        parameters.extend(layer.parameters())
    optimizer = SGD(params=parameters, lr=learning_rate)

    # Training data generation
    sample_num = batch_size * 25
    x_range = (0.05, 0.45)
    y_range = (0.4, 1.0)
    z_range = (0.05, 0.45)

    def targets_generation(num, x_range, y_range, z_range):
        low = np.array([x_range[0], y_range[0], z_range[0]])
        high = np.array([x_range[1], y_range[1], z_range[1]])
        return np.array([np.random.uniform(low=low, high=high) for _ in range(num)])

    np.random.seed(0)
    all_data = targets_generation(sample_num, x_range, y_range, z_range)
    training_sample_num = batch_size * 4
    training_data = all_data[:training_sample_num, :]
    test_data = all_data[training_sample_num:, :]
    print("training data ", training_data.shape, "test data ", test_data.shape)
else:
    batch_size = 1
    input_states = ti.field(float, shape=(model_num, steps, batch_size, n_input), needs_grad=False)
    fc1 = Linear(n_models=model_num, batch_size=batch_size, n_steps=steps, n_input=n_input, n_hidden=n_hidden,
                 n_output=n_output, needs_grad=False, activation=False)
    fc2 = Linear(n_models=model_num, batch_size=batch_size, n_steps=steps, n_input=n_output, n_hidden=n_hidden,
                 n_output=n_output_act, needs_grad=False, activation=True)
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    fc1.load_weights(f"{file_dir_path}/fc1_pretrained.pkl", model_id=0)
    fc2.load_weights(f"{file_dir_path}/fc2_pretrained.pkl", model_id=0)
    print(f"Model at {file_dir_path} loaded. ")

# Simulation configuration
boundary_box_np = np.array([[0, 0, 0], [0.5, 1.5, 0.5]])
spawn_box_np = np.array([[0.0, 0.0, 0.0], [0.5, 0.05, 0.5]])
target_box_np = np.array([[0.15, 0.90, 0.15], [0.2, 0.95, 0.2]])

target_centers = ti.Vector.field(3, float, shape=batch_size, needs_grad=True)
min_dist = ti.field(float, shape=batch_size, needs_grad=True)
max_dist = ti.field(float, shape=batch_size, needs_grad=True)
max_height = ti.field(float, shape=batch_size, needs_grad=True)
max_left = ti.field(float, shape=batch_size, needs_grad=True)
max_right = ti.field(float, shape=batch_size, needs_grad=True)
jet_force_max = ti.Vector([9.81*3, 9.81*10, 9.81*3])

# Simulation parameters
particle_radius = 0.01
particle_diameter = particle_radius * 2
N_np = ((spawn_box_np[1] - spawn_box_np[0]) / particle_diameter + 1).astype(int)
N_target_np = ((target_box_np[1] - target_box_np[0]) / particle_diameter + 1).astype(int)

h = 4.0 * particle_radius
fluid_particle_num = N_np[0] * N_np[1] * N_np[2]
target_particle_num = N_target_np[0] * N_target_np[1] * N_target_np[2]
particle_num = fluid_particle_num + target_particle_num
print(f"Particle num: {particle_num}")

pos = ti.Vector.field(3, float)
vel = ti.Vector.field(3, float)
acc = ti.Vector.field(3, float)
jet_force = ti.Vector.field(3, float, shape=(steps, batch_size), needs_grad=True)


col = ti.Vector.field(3, float)
material = ti.field(int)
den = ti.field(float)
pre = ti.field(float)

pos_vis_buffer = ti.Vector.field(3, float, shape=particle_num)
pos_output_buffer = ti.Vector.field(3, float, shape=(steps, particle_num))
ti.root.dense(ti.ijk, (batch_size, steps, int(particle_num))).place(pos, vel, acc, den, pre)
ti.root.dense(ti.i, int(particle_num)).place(material, col)
ti.root.lazy_grad()

boundary_box = ti.Vector.field(3, float, shape=2)
spawn_box = ti.Vector.field(3, float, shape=2)
target_box = ti.Vector.field(3, float, shape=2)
N_fluid = ti.Vector([N_np[0], N_np[1], N_np[2]])
N_target = ti.Vector([N_target_np[0], N_target_np[1], N_target_np[2]])

gravity = ti.Vector([0.0, -9.8, 0.0])

boundary_box.from_numpy(boundary_box_np)
spawn_box.from_numpy(spawn_box_np)
target_box.from_numpy(target_box_np)

rest_density = 1000.0
mass = rest_density * particle_diameter * particle_diameter * particle_diameter * 0.8
pressure_scale = 10000.0
viscosity_scale = 0.1 * 3
tension_scale = 0.005
gamma = 1.0
substeps = 5
dt = 0.016 / substeps
eps = 1e-6
damping = 0.5
pi = 3.1415926535


@ti.func
def W_poly6(R, h):
    r = R.norm(eps)
    res = 0.0
    if r <= h:
        h2 = h * h
        h4 = h2 * h2
        h9 = h4 * h4 * h
        h2_r2 = h2 - r * r
        res = 315.0 / (64 * pi * h9) * h2_r2 * h2_r2 * h2_r2
    else:
        res = 0.0
    return res


@ti.func
def W_spiky_gradient(R, h):
    r = R.norm(eps)
    res = ti.Vector([0.0, 0.0, 0.0])
    if r == 0.0:
        res = ti.Vector([0.0, 0.0, 0.0])
    elif r <= h:
        h3 = h * h * h
        h6 = h3 * h3
        h_r = h - r
        res = -45.0 / (pi * h6) * h_r * h_r * (R / r)
    else:
        res = ti.Vector([0.0, 0.0, 0.0])
    return res


W = W_poly6
W_gradient = W_spiky_gradient


@ti.kernel
def initialize_fluid_particle(t: ti.int32, pos: ti.template(), N_fluid: ti.template()):
    # Allocate fluid
    for bs, i in ti.ndrange(batch_size, fluid_particle_num):
        pos[bs, t, i] = (
            ti.Vector(
                [int(i % N_fluid[0]), int(i / N_fluid[0]) % N_fluid[1], int(i / N_fluid[0] / N_fluid[1] % N_fluid[2])]
            )
            * particle_diameter
            + spawn_box[0]
        )
        vel[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        material[i] = 0
        col[i] = ti.Vector([0.4, 0.7, 1.0])
        acc.grad[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        pos.grad[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        vel.grad[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def initialize_dists():
    for bs in range(batch_size):
        min_dist[bs] = 1000.0
        max_height[bs] = 0.0
        max_left[bs] = 0.0
        max_right[bs] = 0.0


@ti.kernel
def initialize_target_particle(t: ti.int32, pos: ti.template(), N_target:ti.template(), current_pos: ti.int32):
    # Allocate target cube
    for bs, i in ti.ndrange(batch_size, (fluid_particle_num, fluid_particle_num + target_particle_num)):
        pos[bs, t, i] = (
            ti.Vector(
                [int(i % N_target[0]), int(i / N_target[0]) % N_target[1], int(i / N_target[0] / N_target[1] % N_target[2])]
            )
            * particle_diameter
            + target_centers[current_pos]
        )
        vel[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        material[i] = 1
        col[i] = ti.Vector([1.0, 0.65, 0.0])
        acc.grad[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        pos.grad[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        vel.grad[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def initialize_density(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        den[bs, t, i] = 0.0


@ti.kernel
def update_density(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        for j in range(particle_num):
            R = pos[bs, t, i] - pos[bs, t, j]
            den[bs, t, i] += mass * W(R, h)


@ti.kernel
def update_pressure(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        pre[bs, t, i] = pressure_scale * max(pow(den[bs, t, i] / rest_density, gamma) - 1, 0)


@ti.kernel
def controller_output(t: ti.int32):
    for bs in range(batch_size):
        for j in ti.static(range(3)):
            jet_force[t, bs][j] = fc2.output[0, t, bs, j] * jet_force_max[j]


@ti.kernel
def apply_force(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        if material[i] == 1:
            acc[bs, t, i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            if pos[bs, t, i][0] > 0.2 and pos[bs, t, i][0] < 0.3 and pos[bs, t, i][1] < 0.2 and pos[bs, t, i][2] > 0.2 and pos[bs, t, i][2] < 0.3:
                indicator = (steps - t) // (steps // 2)
                acc[bs, t, i] = jet_force[t, bs] + gravity + indicator * (- gravity) * 0.1
            else:
                acc[bs, t, i] = gravity


@ti.kernel
def update_force(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        for j in range(particle_num):
            R = pos[bs, t, i] - pos[bs, t, j]
            # Pressure forces
            acc[bs, t, i] += -mass * (pre[bs, t, i] / (den[bs, t, i] * den[bs, t, i]) + pre[bs, t, j] / (den[bs, t, j] * den[bs, t, j])) * W_gradient(R, h)

            # Viscosity forces
            acc[bs, t, i] += viscosity_scale * mass \
                         * (vel[bs, t, i] - vel[bs, t, j]).dot(R) / (R.norm(eps) + 0.01 * h * h) / den[bs, t, j] \
                         * W_gradient(R, h)


@ti.kernel
def advance(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        if material[i] == 0:
            vel[bs, t, i] = vel[bs, t - 1, i] + acc[bs, t - 1, i] * dt
        pos[bs, t, i] = pos[bs, t - 1, i] + vel[bs, t, i] * dt


@ti.kernel
def boundary_handle(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        collision_normal = ti.Vector([0.0, 0.0, 0.0])
        for j in ti.static(range(3)):
            if pos[bs, t, i][j] < boundary_box[0][j]:
                pos[bs, t, i][j] = boundary_box[0][j]
                collision_normal[j] += -1.0
        for j in ti.static(range(3)):
            if pos[bs, t, i][j] > boundary_box[1][j]:
                pos[bs, t, i][j] = boundary_box[1][j]
                collision_normal[j] += 1.0
        collision_normal_length = collision_normal.norm()
        if collision_normal_length > eps:
            collision_normal /= collision_normal_length
            vel[bs, t, i] -= (1.0 + damping) * collision_normal.dot(vel[bs, t, i]) * collision_normal


@ti.kernel
def compute_dist(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        if material[i] == 0:
            dist = 0.0
            for j in ti.static(range(3)):
                dist += (pos[bs, t, i][j] - target_centers[bs][j])**2
            dist_sqr = ti.sqrt(dist)
            ti.atomic_min(min_dist[bs], dist_sqr)
            ti.atomic_max(max_height[bs], pos[bs, t, i][1])
            ti.atomic_max(max_left[bs], pos[bs, t, i][0])
            ti.atomic_max(max_right[bs], pos[bs, t, i][2])


@ti.kernel
def compute_loss(t: ti.int32):
    for bs in range(batch_size):
        max_dist[bs] = ti.sqrt((max_left[bs] - target_centers[bs][0])**2 + (max_right[bs] - target_centers[bs][2])**2 + (max_height[bs] - target_centers[bs][1])**2)
        loss[None] += (min_dist[bs] + 0.2 * max_dist[bs]) / batch_size


@ti.kernel
def copy_back(t: ti.int32):
    for bs, i in ti.ndrange(batch_size, particle_num):
        pos[bs, 0, i] = pos[bs, t, i]
        vel[bs, 0, i] = vel[bs, t, i]
        acc[bs, 0, i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def copy_to_vis(t: ti.int32, bs: ti.int32):
    for i in range(particle_num):
        for j in ti.static(range(3)):
            pos_vis_buffer[i][j] = pos[bs, t, i][j]


@ti.kernel
def copy_to_output_buffer(t: ti.int32, bs: ti.int32):
    for i in range(particle_num):
        for j in ti.static(range(3)):
            pos_output_buffer[t, i][j] = pos[bs, t, i][j]


@ti.kernel
def copy_from_output_to_vis(t: ti.int32):
    for i in range(particle_num):
        for j in ti.static(range(3)):
            pos_vis_buffer[i][j] = pos_output_buffer[t, i][j]


@ti.kernel
def fill_target_centers(current_pos: ti.int32, data: ti.any_arr()):
    for i in range(current_pos, current_pos+batch_size):
        for j in ti.static(range(3)):
            target_centers[i][j] = data[i, j]
    print('target_centers ', target_centers[current_pos])


@ti.kernel
def fill_input_states(current_pos: ti.int32):
    for t, bs in ti.ndrange(steps, (current_pos, current_pos+batch_size)):
        for j in ti.static(range(3)):
            input_states[0, t, bs, j] = target_centers[bs][j]


if __name__ == "__main__":
    show_window = True
    if TRAIN:
        show_window = False
    window = ti.ui.Window("Diff SPH", screen_res, show_window=show_window)
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.5, 1.0, 2.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(70)
    scene.set_camera(camera)
    canvas = window.get_canvas()
    movement_speed = 0.02

    if TRAIN:
        losses = []
        losses_epoch_avg = []
        opt_iters = 7
        for opt_iter in range(opt_iters):
            loss_epoch = 0.0
            cnt = 0
            for current_data_offset in range(0, training_sample_num, batch_size):
                fill_target_centers(current_data_offset, training_data)
                fill_input_states(current_data_offset)
                initialize_fluid_particle(0, pos, N_fluid)
                initialize_dists()
                initialize_target_particle(0, pos, N_target, current_data_offset)
                fc1.clear()
                fc2.clear()
                with ti.Tape(loss=loss):
                    for i in range(1, steps):
                        initialize_density(i - 1)
                        update_density(i - 1)
                        update_pressure(i - 1)
                        fc1.forward(i - 1, input_states)
                        fc2.forward(i - 1, fc1.output)
                        controller_output(i - 1)
                        apply_force(i - 1)
                        update_force(i - 1)
                        advance(i)
                        boundary_handle(i)
                        if i % substeps == 0:
                            copy_to_output_buffer(i, 0)
                        compute_dist(i)
                    compute_loss(steps - 1)
                optimizer.step()
                print(f"current opt progress: {current_data_offset + batch_size}/{training_sample_num}, loss: {loss[None]}")
                losses.append(loss[None])
                loss_epoch += loss[None]
                cnt += 1
            print(f'opt iter {opt_iter} done. Average loss: {loss_epoch / cnt}')
            losses_epoch_avg.append(loss_epoch / cnt)

            if TRAIN_VISUAL:
                if opt_iter % 1 == 0:
                    os.makedirs(f"output_img/{opt_iter}", exist_ok=True)
                    for i in range(1, steps):
                        if i % substeps == 0:
                            copy_from_output_to_vis(i)
                        scene.set_camera(camera)
                        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
                        scene.particles(pos_vis_buffer, radius=particle_radius, per_vertex_color=col)
                        canvas.scene(scene)
                        if TRAIN_OUTPUT_IMG:
                            if i % substeps == 0:
                                window.write_image(f'output_img/{opt_iter}/{i:04}.png')
                        if TRAIN_VISUAL_SHOW:
                            window.show()
            if opt_iter % 2 == 0:
                os.makedirs(f"saved_models/{opt_iter}", exist_ok=True)
                fc1.dump_weights(name=f"saved_models/{opt_iter}/fc1_{opt_iter:04}.pkl")
                fc2.dump_weights(name=f"saved_models/{opt_iter}/fc2_{opt_iter:04}.pkl")

        plt.plot([i for i in range(len(losses))], losses, label='loss per iteration')
        plt.plot([i * (training_sample_num // batch_size) for i in range(len(losses_epoch_avg))], losses_epoch_avg, label='loss epoch avg.')
        plt.title("Training Loss")
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        current_data_offset = 0
        initialize_fluid_particle(0, pos, N_fluid)
        target_centers[current_data_offset][0] = 0.25
        target_centers[current_data_offset][1] = 0.50
        target_centers[current_data_offset][2] = 0.25

        print("Start... ")
        cnt = 0
        paused = ti.field(int, shape=())
        while window.running:
            window.GUI.begin("Diff SPH", 0.05, 0.05, 0.2, 0.2)
            window.GUI.text("Space: pause")
            window.GUI.text("Set target positions:")

            target_centers[current_data_offset][0] = window.GUI.slider_float("X",
                                                                             target_centers[current_data_offset][0],
                                                                             0.05, 0.45)
            target_centers[current_data_offset][1] = window.GUI.slider_float("Y",
                                                                             target_centers[current_data_offset][1],
                                                                             0.4, 1.0)
            target_centers[current_data_offset][2] = window.GUI.slider_float("Z",
                                                                             target_centers[current_data_offset][2],
                                                                             0.05, 0.45)
            window.GUI.end()

            if not paused[None]:
                fill_input_states(current_data_offset)
                initialize_target_particle(0, pos, N_target, current_data_offset)
                fc1.clear()
                fc2.clear()
                for i in range(1, substeps):
                    initialize_density(i - 1)
                    update_density(i - 1)
                    update_pressure(i - 1)
                    fc1.forward(i - 1, input_states)
                    fc2.forward(i - 1, fc1.output)
                    controller_output(i - 1)
                    apply_force(i - 1)
                    update_force(i - 1)
                    advance(i)
                    boundary_handle(i)
                copy_to_vis(substeps - 1, 0)
                copy_back(substeps - 1)
                cnt += 1
            # user controlling of camera
            position_change = ti.Vector([0.0, 0.0, 0.0])
            up = ti.Vector([0.0, 1.0, 0.0])
            # move camera up and down
            if window.is_pressed("e"):
                position_change = up * movement_speed
            if window.is_pressed("q"):
                position_change = -up * movement_speed

            for e in window.get_events(ti.ui.PRESS):
                if e.key == ti.ui.SPACE:
                    paused[None] = not paused[None]
            camera.position(*(camera.curr_position + position_change))
            camera.lookat(*(camera.curr_lookat + position_change))
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)

            scene.set_camera(camera)
            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            scene.particles(pos_vis_buffer, radius=particle_radius, per_vertex_color=col)
            canvas.scene(scene)
            if INFER_OUTPUT_IMG:
                if cnt % 2 == 0:
                    os.makedirs(f"demo_output_interactive/", exist_ok=True)
                    window.write_image(f'demo_output_interactive/{cnt:04}.png')
            window.show()

