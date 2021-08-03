# convert numpy array to ply files
import sys

import numpy as np

import taichi as ti


class PLYWriter:
    def __init__(self,
                 num_vertices: int,
                 num_faces=0,
                 face_type="tri",
                 comment="created by PLYWriter"):
        assert num_vertices > 0, "num_vertices should be greater than 0"
        assert num_faces >= 0, "num_faces shouldn't be less than 0"
        assert face_type == "tri" or face_type == "quad", "Only tri and quad faces are supported for now"

        self.ply_supported_types = [
            'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float',
            'double'
        ]
        self.corresponding_numpy_types = [
            np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
            np.float32, np.float64
        ]
        self.type_map = {}
        for i in range(len(self.ply_supported_types)):
            self.type_map[self.ply_supported_types[
                i]] = self.corresponding_numpy_types[i]

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
            self.face_indices = -np.ones((self.num_faces, 3), dtype=np.int32)
        elif face_type == "quad":
            self.face_indices = -np.ones((self.num_faces, 4), dtype=np.int32)
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
                item_key = key + "_" + str(i + 1)
                if item_key in self.vertex_channels:
                    print("WARNING: duplicate key " + item_key + " detected")
                self.vertex_channels.append(item_key)
                self.vertex_data_type.append(type)
                self.vertex_data.append(self.type_map[type](data[:, i]))

    def add_vertex_pos(self, x: np.array, y: np.array, z: np.array):
        self.add_vertex_channel("x", "float", x)
        self.add_vertex_channel("y", "float", y)
        self.add_vertex_channel("z", "float", z)

    # TODO active and refactor later if user feedback indicates the necessity for a compact the input list
    # pass ti vector/matrix field directly
    # def add_vertex_pos(self, pos):
    #     assert isinstance(pos, (np.ndarray, ti.Matrix))
    #     if not isinstance(pos, np.ndarray):
    #         pos = pos.to_numpy()
    #     dim = pos.shape[pos.ndim-1]
    #     assert dim == 2 or dim == 3, "Only 2D and 3D positions are supported"
    #     n = pos.size // dim
    #     assert n == self.num_vertices, "Size of the input is not correct"
    #     pos = np.reshape(pos, (n, dim))
    #     self.add_vertex_channel("x", "float", pos[:, 0])
    #     self.add_vertex_channel("y", "float", pos[:, 1])
    #     if(dim == 3):
    #         self.add_vertex_channel("z", "float", pos[:, 2])
    #     if(dim == 2):
    #         self.add_vertex_channel("z", "float", np.zeros(n))

    def add_vertex_normal(self, nx: np.array, ny: np.array, nz: np.array):
        self.add_vertex_channel("nx", "float", nx)
        self.add_vertex_channel("ny", "float", ny)
        self.add_vertex_channel("nz", "float", nz)

    # TODO active and refactor later if user feedback indicates the necessity for a compact the input list
    # pass ti vector/matrix field directly
    # def add_vertex_normal(self, normal):
    #     assert isinstance(normal, (np.ndarray, ti.Matrix))
    #     if not isinstance(normal, np.ndarray):
    #         normal = normal.to_numpy()
    #     dim = normal.shape[normal.ndim-1]
    #     assert dim == 3, "Only 3D normal is supported"
    #     n = normal.size // dim
    #     assert n == self.num_vertices, "Size of the input is not correct"
    #     normal = np.reshape(normal, (n, dim))
    #     self.add_vertex_channel("nx", "float", normal[:, 0])
    #     self.add_vertex_channel("ny", "float", normal[:, 1])
    #     self.add_vertex_channel("nz", "float", normal[:, 2])

    def add_vertex_color(self, r: np.array, g: np.array, b: np.array):
        self.add_vertex_channel("red", "float", r)
        self.add_vertex_channel("green", "float", g)
        self.add_vertex_channel("blue", "float", b)

    def add_vertex_alpha(self, alpha: np.array):
        self.add_vertex_channel("Alpha", "float", alpha)

    def add_vertex_rgba(self, r: np.array, g: np.array, b: np.array,
                        a: np.array):
        self.add_vertex_channel("red", "float", r)
        self.add_vertex_channel("green", "float", g)
        self.add_vertex_channel("blue", "float", b)
        self.add_vertex_channel("Alpha", "float", a)

    # TODO active and refactor later if user feedback indicates the necessity for a compact the input list
    # pass ti vector/matrix field directly
    # def add_vertex_color(self, color):
    #     assert isinstance(color, (np.ndarray, ti.Matrix))
    #     if not isinstance(color, np.ndarray):
    #         color = color.to_numpy()
    #     channels = color.shape[color.ndim-1]
    #     assert channels == 3 or channels == 4, "The dimension for color should be either be 3 (rgb) or 4 (rgba)"
    #     n = color.size // channels
    #     assert n == self.num_vertices, "Size of the input is not correct"
    #     color = np.reshape(color, (n, channels))
    #     self.add_vertex_channel("red", "float", color[:, 0])
    #     self.add_vertex_channel("green", "float", color[:, 1])
    #     self.add_vertex_channel("blue", "float", color[:, 2])
    #     if channels == 4:
    #         self.add_vertex_channel("Alpha", "float", color[:, 3])

    def add_vertex_id(self):
        self.add_vertex_channel("id", "int", np.arange(self.num_vertices))

    def add_vertex_piece(self, piece: np.array):
        self.add_vertex_channel("piece", "int", piece)

    def add_faces(self, indices: np.array):
        if self.face_type == "tri":
            vert_per_face = 3
        else:
            vert_per_face = 4
        assert vert_per_face * \
            self.num_faces == indices.size, "The dimension of the face vertices is not correct"
        self.face_indices = np.reshape(indices,
                                       (self.num_faces, vert_per_face))
        self.face_indices = self.face_indices.astype(np.int32)

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
                item_key = key + "_" + str(i + 1)
                if item_key in self.face_channels:
                    print("WARNING: duplicate key " + item_key + " detected")
                self.face_channels.append(item_key)
                self.face_data_type.append(type)
                self.face_data.append(self.type_map[type](data[:, i]))

    def add_face_id(self):
        self.add_face_channel("id", "int", np.arange(self.num_faces))

    def add_face_piece(self, piece: np.array):
        self.add_face_channel("piece", "int", piece)

    def sanity_check(self):
        assert "x" in self.vertex_channels, "The vertex pos channel is missing"
        assert "y" in self.vertex_channels, "The vertex pos channel is missing"
        assert "z" in self.vertex_channels, "The vertex pos channel is missing"
        if self.num_faces > 0:
            for idx in self.face_indices.flatten():
                assert idx >= 0 and idx < self.num_vertices, "The face indices are invalid"

    def print_header(self, path: str, format: str):
        with open(path, "w") as f:
            f.writelines([
                "ply\n", "format " + format + " 1.0\n",
                "comment " + self.comment + "\n"
            ])
            f.write("element vertex " + str(self.num_vertices) + "\n")
            for i in range(self.num_vertex_channels):
                f.write("property " + self.vertex_data_type[i] + " " +
                        self.vertex_channels[i] + "\n")
            if (self.num_faces != 0):
                f.write("element face " + str(self.num_faces) + "\n")
                f.write("property list uchar int vertex_indices\n")
                for i in range(self.num_face_channels):
                    f.write("property " + self.face_data_type[i] + " " +
                            self.face_channels[i] + "\n")
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
                f.writelines([
                    str(vert_per_face) + " ",
                    " ".join(map(str, self.face_indices[i, :])), " "
                ])
                for j in range(self.num_face_channels):
                    f.write(str(self.face_data[j][i]) + " ")
                f.write("\n")

    def export_frame_ascii(self, series_num: int, path: str):
        # if path has ply ending
        last_4_char = path[-4:]
        if last_4_char == ".ply":
            path = path[:-4]

        real_path = path + "_" + "{0:0=6d}".format(series_num) + ".ply"
        self.export_ascii(real_path)

    def export_frame(self, series_num: int, path: str):
        # if path has ply ending
        last_4_char = path[-4:]
        if last_4_char == ".ply":
            path = path[:-4]

        real_path = path + "_" + "{0:0=6d}".format(series_num) + ".ply"
        self.export(real_path)
