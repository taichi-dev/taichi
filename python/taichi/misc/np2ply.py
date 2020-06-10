# convert numpy array to ply files
import sys
import numpy as np


class PLYWriter:
    def __init__(self, num_vertices: int, num_faces=0, face_type="tri", comment="created by PLYWriter"):
        assert num_vertices > 0, "num_vertices should be greater than 0"
        assert num_faces >= 0, "num_faces shouldn't be less than 0"
        assert face_type == "tri" or face_type == "quad", "Only tri and quad faces are supported for now"

        self.ply_supported_types = [
            'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double']
        self.corresponding_numpy_types = [
            np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32, np.float64]
        self.type_map = {}
        for i in range(len(self.ply_supported_types)):
            self.type_map[self.ply_supported_types[i]
                          ] = self.corresponding_numpy_types[i]

        self.num_vertices = num_vertices
        self.num_vertex_channels = 0
        self.vertex_channels = []
        self.vertex_data_type = []
        self.vertex_data = []
        self.num_faces = num_faces
        self.num_face_channels = 0
        self.face_channels = []
        self.face_data_type = []
        self.face_data = []
        self.face_type = face_type
        if face_type == "tri":
            self.face_indices = -np.ones((self.num_faces, 3), dtype=int)
        elif face_type == "quad":
            self.face_indices = -np.ones((self.num_faces, 4), dtype=int)
        self.comment = comment

    def add_vertex_channel(self, key: str, type: str, data: np.array):
        if type not in self.ply_supported_types:
            print("Unknown type " + type + " detected, skipping this channel")
            return
        if data.ndim == 1:
            assert data.size == self.num_vertices, "The dimension of the vertex channel is not correct"
            self.num_vertex_channels += 1
            if key in self.vertex_channels:
                print("WARNING: duplicate key " + key + " detected")
            self.vertex_channels.append(key)
            self.vertex_data_type.append(type)
            self.vertex_data.append(self.type_map[type](data))
        else:
            num_col = data.size // self.num_vertices
            assert data.ndim == 2 and data.size == num_col * \
                self.num_vertices, "The dimension of the vertex channel is not correct"
            data.shape = (self.num_vertices, num_col)
            self.num_vertex_channels += num_col
            for i in range(num_col):
                item_key = key + "_" + str(i+1)
                if item_key in self.vertex_channels:
                    print("WARNING: duplicate key " + item_key + " detected")
                self.vertex_channels.append(item_key)
                self.vertex_data_type.append(type)
                self.vertex_data.append(self.type_map[type](data[:, i]))

    def add_vertex_pos(self, x: np.array, y: np.array, z: np.array):
        self.add_vertex_channel("x", "float", x)
        self.add_vertex_channel("y", "float", y)
        self.add_vertex_channel("z", "float", z)

    def add_vertex_normal(self, nx: np.array, ny: np.array, nz: np.array):
        self.add_vertex_channel("nx", "float", nx)
        self.add_vertex_channel("ny", "float", ny)
        self.add_vertex_channel("nz", "float", nz)

    def add_vertex_color(self, r: np.array, g: np.array, b: np.array):
        self.add_vertex_channel("red", "float", r)
        self.add_vertex_channel("green", "float", g)
        self.add_vertex_channel("blue", "float", b)

    def add_vertex_color_uchar(self, r: np.array, g: np.array, b: np.array):
        self.add_vertex_channel("red", "uchar", r)
        self.add_vertex_channel("green", "uchar", g)
        self.add_vertex_channel("blue", "uchar", b)

    def add_faces(self, indices: np.array):
        if self.face_type == "tri":
            vert_per_face = 3
        else:
            vert_per_face = 4
        assert vert_per_face * \
            self.num_faces == indices.size, "The dimension of the face vertices is not correct"
        self.face_indices = np.reshape(
            indices, (self.num_faces, vert_per_face))

    def add_face_channel(self, key: str, type: str, data: np.array):
        if type not in self.ply_supported_types:
            print("Unknown type " + type + " detected, skipping this channel")
            return
        if data.ndim == 1:
            assert data.size == self.num_faces, "The dimension of the face channel is not correct"
            self.num_face_channels += 1
            if key in self.face_channels:
                print("WARNING: duplicate key " + key + " detected")
            self.face_channels.append(key)
            self.face_data_type.append(type)
            self.face_data.append(self.type_map[type](data))
        else:
            num_col = data.size // self.num_faces
            assert data.ndim == 2 and data.size == num_col * \
                self.num_faces, "The dimension of the face channel is not correct"
            data.shape = (self.num_faces, num_col)
            self.num_face_channels += num_col
            for i in range(num_col):
                item_key = key + "_" + str(i+1)
                if item_key in self.face_channels:
                    print("WARNING: duplicate key " + item_key + " detected")
                self.face_channels.append(item_key)
                self.face_data_type.append(type)
                self.face_data.append(self.type_map[type](data[:, i]))

    def sanity_check(self):
        assert "x" in self.vertex_channels, "The vertex pos channel is missing"
        assert "y" in self.vertex_channels, "The vertex pos channel is missing"
        assert "z" in self.vertex_channels, "The vertex pos channel is missing"
        if self.num_faces > 0:
            for idx in self.face_indices.flatten():
                assert idx >= 0 and idx < self.num_vertices, "The face indices are invalid"

    def print_header(self, path: str, format: str):
        with open(path, "w") as f:
            f.writelines(["ply\n", "format " + format + " 1.0\n",
                          "comment " + self.comment + "\n"])
            f.write("element vertex " + str(self.num_vertices) + "\n")
            for i in range(self.num_vertex_channels):
                f.write(
                    "property " + self.vertex_data_type[i] + " " + self.vertex_channels[i] + "\n")
            if(self.num_faces != 0):
                f.write("element face " + str(self.num_faces) + "\n")
                f.write("property list uchar int vertex_indices\n")
                for i in range(self.num_face_channels):
                    f.write(
                        "property " + self.face_data_type[i] + " " + self.face_channels[i] + "\n")
                f.write("end_header\n")

    def export(self, path):
        self.sanity_check()
        self.print_header(path, "binary_" + sys.byteorder + "_endian")
        with open(path, "ab") as f:
            for i in range(self.num_vertices):
                for j in range(self.num_vertex_channels):
                    f.write(self.vertex_data[j][i])
            if self.face_type == "tri":
                vert_per_face = np.uint8(3)
            else:
                vert_per_face = np.uint8(4)
            for i in range(self.num_faces):
                f.write(vert_per_face)
                for j in range(vert_per_face):
                    f.write(self.face_indices[i, j])
                for j in range(self.num_face_channels):
                    f.write(self.face_data[j][i])

    def export_ascii(self, path):
        self.sanity_check()
        self.print_header(path, "ascii")
        with open(path, "a") as f:
            for i in range(self.num_vertices):
                for j in range(self.num_vertex_channels):
                    f.write(str(self.vertex_data[j][i]) + " ")
                f.write("\n")
            if self.face_type == "tri":
                vert_per_face = 3
            else:
                vert_per_face = 4
            for i in range(self.num_faces):
                f.writelines(
                    [str(vert_per_face) + " ", " ".join(map(str, self.face_indices[i, :])), " "])
                for j in range(self.num_face_channels):
                    f.write(str(self.face_data[j][i]) + " ")
                f.write("\n")
