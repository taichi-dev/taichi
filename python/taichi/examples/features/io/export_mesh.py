import numpy as np

import taichi as ti

# A 2D grid with quad faces
#     y
#     |
# z---/
#    x
#         19---15---11---07---03
#         |    |    |    |    |
#         18---14---10---06---02
#         |    |    |    |    |
#         17---13---19---05---01
#         |    |    |    |    |
#         16---12---08---04---00

writer = ti.tools.PLYWriter(num_vertices=20, num_faces=12, face_type="quad")

# For the vertices, the only required channel is the position,
# which can be added by passing 3 np.array x, y, z into the following function.

x = np.zeros(20)
y = np.array(list(np.arange(0, 4)) * 5)
z = np.repeat(np.arange(5), 4)
writer.add_vertex_pos(x, y, z)

# For faces (if any), the only required channel is the list of vertex indices that each face contains.
indices = np.array([0, 1, 5, 4] * 12) + np.repeat(
    np.array(list(np.arange(0, 3)) * 4) + 4 * np.repeat(np.arange(4), 3), 4
)
writer.add_faces(indices)

# Add custom vertex channel, the input should include a key, a supported datatype and, the data np.array
vdata = np.random.rand(20)
writer.add_vertex_channel("vdata1", "double", vdata)

# Add custom face channel
foo_data = np.zeros(12)
writer.add_face_channel("foo_key", "foo_data_type", foo_data)
# error! because "foo_data_type" is not a supported datatype. Supported ones are
# ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double']

# PLYwriter already defines several useful helper functions for common channels
# Add vertex color, alpha, and rgba
# using float/double r g b alpha to reprent color, the range should be 0 to 1
r = np.random.rand(20)
g = np.random.rand(20)
b = np.random.rand(20)
alpha = np.random.rand(20)
writer.add_vertex_color(r, g, b)
writer.add_vertex_alpha(alpha)
# equivilantly
# add_vertex_rgba(r, g, b, alpha)

# vertex normal
writer.add_vertex_normal(np.ones(20), np.zeros(20), np.zeros(20))

# vertex index, and piece (group id)
writer.add_vertex_id()
writer.add_vertex_piece(np.ones(20))

# Add face index, and piece (group id)
# Indexing the existing faces in the writer and add this channel to face channels
writer.add_face_id()
# Set all the faces is in group 1
writer.add_face_piece(np.ones(12))

series_prefix = "example.ply"
series_prefix_ascii = "example_ascii.ply"
# Export a single file
# use ascii so you can read the content
writer.export_ascii(series_prefix_ascii)

# alternatively, use binary for a bit better performance
writer.export(series_prefix)

# Export a sequence of files, ie in 10 frames
for frame in range(10):
    # write each frame as i.e. "example_000000.ply" in your current running folder
    writer.export_frame_ascii(frame, series_prefix_ascii)
    # alternatively, use binary
    writer.export_frame(frame, series_prefix)

    # update location/color
    x = x + 0.1 * np.random.rand(20)
    y = y + 0.1 * np.random.rand(20)
    z = z + 0.1 * np.random.rand(20)
    r = np.random.rand(20)
    g = np.random.rand(20)
    b = np.random.rand(20)
    alpha = np.random.rand(20)
    # re-fill
    writer = ti.tools.PLYWriter(num_vertices=20, num_faces=12, face_type="quad")
    writer.add_vertex_pos(x, y, z)
    writer.add_faces(indices)
    writer.add_vertex_channel("vdata1", "double", vdata)
    writer.add_vertex_color(r, g, b)
    writer.add_vertex_alpha(alpha)
    writer.add_vertex_normal(np.ones(20), np.zeros(20), np.zeros(20))
    writer.add_vertex_id()
    writer.add_vertex_piece(np.ones(20))
    writer.add_face_id()
    writer.add_face_piece(np.ones(12))
