import taichi as ti
from math import ceil
import marching_cubes_data as mc
import numpy as np

@ti.data_oriented
class Mesher:
    def __init__(self,space_min,space_max,sdf_cell_size,mesh_cell_size):
        self.space_min = ti.Vector([*space_min])
        self.space_max = ti.Vector([*space_max])

        self.sdf_cell_size = sdf_cell_size
        self.mesh_cell_size = mesh_cell_size

        space_size = self.space_max - self.space_min
        self.space_size = space_size

        self.sdf_grid_size = ceil(space_size[0]/sdf_cell_size)+3,ceil(space_size[1]/sdf_cell_size)+3,ceil(space_size[2]/sdf_cell_size)+3
        self.mesh_grid_size = ceil(space_size[0]/mesh_cell_size)+2,ceil(space_size[1]/mesh_cell_size)+2,ceil(space_size[2]/mesh_cell_size)+2
        
        self.sdf = ti.field(ti.f32,shape = self.sdf_grid_size)
        self.sdf_weights = ti.field(ti.f32,shape = self.sdf_grid_size)
        self.mean_x = ti.Vector.field(3,ti.f32,shape = self.sdf_grid_size)

        self.mc_table = ti.field(ti.i32,shape = (256,15))
        self.mc_table.from_numpy(np.array(mc.table))

        self.mc_edge_to_vert = ti.field(ti.i32,shape = (12,2))
        self.mc_edge_to_vert.from_numpy(np.array(mc.edge_to_vert))

        self.mc_vert_coords = ti.field(ti.i32,shape = (8,3))
        self.mc_vert_coords.from_numpy(np.array(mc.vert_coords))

        self.num_vertices = 1 << 22

        self.vertices = ti.Vector.field(3,ti.f32,shape = self.num_vertices)
        self.normals = ti.Vector.field(3,ti.f32,shape = self.num_vertices)

    @ti.func
    def round_vec(self,v):
        return int(ti.Vector([ti.floor(v[0]+0.5),ti.floor(v[1]+0.5),ti.floor(v[2]+0.5)]))  

    @ti.func
    def get_sdf_cell_index(self,pos):
        return self.round_vec((pos-self.space_min) / self.sdf_cell_size + 1)

    @ti.func
    def get_sdf_cell_pos(self,cell):
        return (cell - 1) * self.sdf_cell_size + self.space_min

    @ti.func
    def get_mesh_cell_pos(self,cell):
        return (cell - 1) * self.mesh_cell_size + self.space_min

    @ti.func
    def get_sdf_at_pos(self,pos):
        x = (pos-self.space_min) / self.sdf_cell_size
        base = int(x)
        fx = x - base
        w = [1.0-fx,fx]

        sum_sdf = 0.0
        sum_weights = 0.0

        for offset in ti.static(ti.grouped(ti.ndrange(1,1,1))):
            weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
            this_sdf = self.sdf[base + offset]
            sum_sdf += weight * this_sdf
            sum_weights += weight

        mean_sdf = sum_sdf / sum_weights
        return mean_sdf

    @ti.func
    def zhu05_kernel(self,r,support):
        r2 = r.dot(r)
        h2 = support * support
        temp = 1.0 - r2 / h2
        return max(0.0,temp * temp * temp)

    

    @ti.kernel
    def compute_sdf_kernel(self,particles:ti.template(),radius:float):
        for p in particles:
            pos = particles[p]
            base = self.get_sdf_cell_index(pos)
            for offset in ti.static(ti.grouped(ti.ndrange(5,5,5))):
                cell = base + offset - ti.Vector([2,2,2])
                cell_pos = self.get_sdf_cell_pos(cell)
                w = self.zhu05_kernel(pos-cell_pos,self.sdf_cell_size*2)
                self.sdf_weights[cell] += w
                self.mean_x[cell] += w * pos
        for cell in ti.grouped(self.sdf):
            if self.sdf_weights[cell] > 0:
                mean_x = self.mean_x[cell] / self.sdf_weights[cell]
                cell_pos = self.get_sdf_cell_pos(cell)
                sdf = (mean_x-cell_pos).norm() - radius
                self.sdf[cell] = sdf


    @ti.kernel
    def mesh_kernel(self):
        triangle_index = 0
        for cell in ti.grouped(ti.ndrange(*self.mesh_grid_size)):
            cube_type = 0
            this_triangle_index = 0
            for i in range(8):
                this_cell = cell + ti.Vector([self.mc_vert_coords[i,0],self.mc_vert_coords[i,1],self.mc_vert_coords[i,2]])
                pos = self.get_mesh_cell_pos(this_cell)
                sdf = self.get_sdf_at_pos(pos)
                if sdf < 0:
                    cube_type += 1 << i
            vertex_id = 0
            while vertex_id < 15:
                edge = self.mc_table[cube_type,vertex_id]
                if edge == -1:
                    break
                if vertex_id%3 == 0:
                    this_triangle_index = ti.atomic_add(triangle_index,1)
                if this_triangle_index * 3 >= self.num_vertices:
                    break
                
                vert_a = self.mc_edge_to_vert[edge,0]
                vert_b = self.mc_edge_to_vert[edge,1]

                coords_a = float(ti.Vector([self.mc_vert_coords[vert_a,0],self.mc_vert_coords[vert_a,1],self.mc_vert_coords[vert_a,2] ]))
                coords_b = float(ti.Vector([self.mc_vert_coords[vert_b,0],self.mc_vert_coords[vert_b,1],self.mc_vert_coords[vert_b,2] ]))

                coords_a = self.get_mesh_cell_pos(coords_a + cell)
                coords_b = self.get_mesh_cell_pos(coords_b + cell)

                sdf_a = self.get_sdf_at_pos(coords_a)
                sdf_b = self.get_sdf_at_pos(coords_b)

                factor = -sdf_a / (sdf_b - sdf_a)
                

                coords = coords_a * (1.0-factor) + coords_b*factor
                self.vertices[this_triangle_index * 3 + vertex_id % 3] = coords

                vertex_id += 1


    def mesh(self,particles,radius):
        self.sdf.fill(0.0)
        self.sdf_weights.fill(0.0)
        self.mean_x.fill(0.0)
        self.vertices.fill(0.0)
        self.compute_sdf_kernel(particles,radius)
        self.mesh_kernel()
        return self.vertices