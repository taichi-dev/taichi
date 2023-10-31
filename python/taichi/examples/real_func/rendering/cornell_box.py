# pylint: disable=W0622,W0621,W0401
import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32)

image_resolution = (512, 512)
image_buffer = ti.Vector.field(4, float, image_resolution)
image_pixels = ti.Vector.field(3, float, image_resolution)

Ray = ti.types.struct(origin=vec3, direction=vec3, color=vec3)
Material = ti.types.struct(albedo=vec3, emission=vec3)
Transform = ti.types.struct(position=vec3, rotation=vec3, scale=vec3, matrix=mat3)
SDFObject = ti.types.struct(distance=float, transform=Transform, material=Material)

objects = SDFObject.field(shape=8)
objects[0] = SDFObject(
    transform=Transform(vec3(0, 0, -1), vec3(0, 0, 0), vec3(1, 1, 0.2)),
    material=Material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
objects[1] = SDFObject(
    transform=Transform(vec3(0, 1, 0), vec3(90, 0, 0), vec3(1, 1, 0.2)),
    material=Material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
objects[2] = SDFObject(
    transform=Transform(vec3(0, -1, 0), vec3(90, 0, 0), vec3(1, 1, 0.2)),
    material=Material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
objects[3] = SDFObject(
    transform=Transform(vec3(-1, 0, 0), vec3(0, 90, 0), vec3(1, 1, 0.2)),
    material=Material(vec3(1, 0, 0) * 0.5, vec3(1)),
)
objects[4] = SDFObject(
    transform=Transform(vec3(1, 0, 0), vec3(0, 90, 0), vec3(1, 1, 0.2)),
    material=Material(vec3(0, 1, 0) * 0.5, vec3(1)),
)
objects[5] = SDFObject(
    transform=Transform(vec3(-0.275, -0.3, -0.2), vec3(0, 112, 0), vec3(0.25, 0.5, 0.25)),
    material=Material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
objects[6] = SDFObject(
    transform=Transform(vec3(0.275, -0.55, 0.2), vec3(0, -197, 0), vec3(0.25, 0.25, 0.25)),
    material=Material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
objects[7] = SDFObject(
    transform=Transform(vec3(0, 0.809, 0), vec3(90, 0, 0), vec3(0.2, 0.2, 0.01)),
    material=Material(vec3(1, 1, 1) * 1, vec3(100)),
)


@ti.real_func
def rotate(a: vec3) -> mat3:
    s, c = sin(a), cos(a)
    return (
        mat3(c.z, s.z, 0, -s.z, c.z, 0, 0, 0, 1)
        @ mat3(c.y, 0, -s.y, 0, 1, 0, s.y, 0, c.y)
        @ mat3(1, 0, 0, 0, c.x, s.x, 0, -s.x, c.x)
    )


@ti.real_func
def signed_distance(obj: SDFObject, pos: vec3) -> float:
    p = obj.transform.matrix @ (pos - obj.transform.position)
    q = abs(p) - obj.transform.scale
    return length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0)


@ti.real_func
def nearest_object(p: vec3) -> (int, float):
    index, min_dis = 0, 1e32
    for i in ti.static(range(8)):
        dis = signed_distance(objects[i], p)
        if dis < min_dis:
            min_dis, index = dis, i
    return index, min_dis


@ti.real_func
def calc_normal(obj: SDFObject, p: vec3) -> vec3:
    e = vec2(1, -1) * 0.5773 * 0.005
    return normalize(
        e.xyy * signed_distance(obj, p + e.xyy)
        + e.yyx * signed_distance(obj, p + e.yyx)
        + e.yxy * signed_distance(obj, p + e.yxy)
        + e.xxx * signed_distance(obj, p + e.xxx)
    )


@ti.real_func
def raycast(ray: Ray) -> (SDFObject, vec3, bool):
    w, s, d, cerr = 1.6, 0.0, 0.0, 1e32
    index, t, position, hit = 0, 0.005, vec3(0), False
    for _ in range(64):
        position = ray.origin + ray.direction * t
        index, distance = nearest_object(position)

        ld, d = d, distance
        if ld + d < s:
            s -= w * s
            t += s
            w *= 0.5
            w += 0.5
            continue
        err = d / t
        if err < cerr:
            cerr = err

        s = w * d
        t += s
        hit = err < 0.001
        if t > 5.0 or hit:
            break
    return objects[index], position, hit


@ti.real_func
def hemispheric_sampling(normal: vec3) -> vec3:
    z = 2.0 * ti.random() - 1.0
    a = ti.random() * 2.0 * pi
    xy = sqrt(1.0 - z * z) * vec2(sin(a), cos(a))
    return normalize(normal + vec3(xy, z))


@ti.real_func
def raytrace(ray: Ray) -> Ray:
    for _ in range(3):
        object, position, hit = raycast(ray)
        if not hit:
            ray.color = vec3(0)
            break

        normal = calc_normal(object, position)
        ray.direction = hemispheric_sampling(normal)
        ray.color *= object.material.albedo
        ray.origin = position

        intensity = dot(ray.color, vec3(0.299, 0.587, 0.114))
        ray.color *= object.material.emission
        visible = dot(ray.color, vec3(0.299, 0.587, 0.114))
        if intensity < visible or visible < 0.000001:
            break
    return ray


@ti.kernel
def build_scene():
    for i in objects:
        rotation = radians(objects[i].transform.rotation)
        objects[i].transform.matrix = rotate(rotation)


@ti.kernel
def render(camera_position: vec3, camera_lookat: vec3, camera_up: vec3):
    for i, j in image_pixels:
        z = normalize(camera_position - camera_lookat)
        x = normalize(cross(camera_up, z))
        y = cross(z, x)

        half_width = half_height = tan(radians(35) * 0.5)
        lower_left_corner = camera_position - half_width * x - half_height * y - z
        horizontal = 2.0 * half_width * x
        vertical = 2.0 * half_height * y

        uv = (vec2(i, j) + vec2(ti.random(), ti.random())) / vec2(image_resolution)
        po = lower_left_corner + uv.x * horizontal + uv.y * vertical
        rd = normalize(po - camera_position)

        ray = raytrace(Ray(camera_position, rd, vec3(1)))
        buffer = image_buffer[i, j]
        buffer += vec4(ray.color, 1.0)
        image_buffer[i, j] = buffer

        color = buffer.rgb / buffer.a
        color = pow(color, vec3(1.0 / 2.2))
        color = (
            mat3(
                0.597190,
                0.35458,
                0.04823,
                0.07600,
                0.90834,
                0.01566,
                0.02840,
                0.13383,
                0.83777,
            )
            @ color
        )
        color = (color * (color + 0.024578) - 0.0000905) / (color * (0.983729 * color + 0.4329510) + 0.238081)
        color = (
            mat3(
                1.60475,
                -0.531,
                -0.0736,
                -0.102,
                1.10813,
                -0.00605,
                -0.00327,
                -0.07276,
                1.07602,
            )
            @ color
        )
        image_pixels[i, j] = clamp(color, 0, 1)


def main():
    window = ti.ui.Window("Cornell Box", image_resolution)
    canvas = window.get_canvas()
    build_scene()
    while window.running:
        render(vec3(0, 0, 3.5), vec3(0, 0, -1), vec3(0, 1, 0))
        canvas.set_image(image_pixels)
        window.show()


if __name__ == "__main__":
    main()
