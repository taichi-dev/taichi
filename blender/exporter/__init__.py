bl_info = {
    "name": "Taichi Exporter",
    "author": "Yuanming Hu (iterator)",
    "version": (0, 0, 0),
    "blender": (2, 7, 8),
#    "location": "Properties editor > Particles Tabs",
    "description": (""),
    "warning": "",
#    "tracker_url": "http://huyuanming.com",
    "category": "Object"
}

export_path = 'C:/tmp/blender_export'
#export_path = '/tmp/'

import bpy

import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

from xml.dom import minidom
import os

import sys


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def dump_matrix(mat, name, node):
    matrix_node = SubElement(node, name)
    # matrix_node.set('name', name)
    for row in mat:
        row_node = SubElement(matrix_node, 'row')
        row_node.text = ' %+.10f %+.10f %+.10f %+.10f' % (row[0], row[1], row[2], row[3])

def export_scene(scene):
    bpy.ops.object.select_all(action='DESELECT')  
    def dump_mesh(obj, scene_node, filepath):
        mesh = obj.to_mesh(scene, True, 'RENDER')
        transform = obj.matrix_world
        ffaces_mats = {}
        mesh_faces = mesh.tessfaces
        vertices = mesh.vertices
        mesh_node = SubElement(scene_node, 'mesh')

        def get_property(name, def_val):
            try:
                return obj[name]
            except:
                return def_val

        def write_property(name, def_val=0):
            SubElement(mesh_node, name).text = str(get_property(name, def_val))

        SubElement(mesh_node, 'filepath').text = filepath.lower()
        write_property('temperature')
        write_property('const_temp')
        write_property('need_voxelization')

        def str_color(color):
            return str('%f %f %f' % (color.r, color.g, color.b))

        if len(obj.material_slots) > 0:
            material_node = SubElement(mesh_node, 'material')
            material = bpy.data.materials[obj.material_slots[0].name]
            emission = material.emit
            material_type = ''

            if emission > 0:
                # light source
                material_type = 'light_source'
                emission_color = material.diffuse_color * emission
                SubElement(material_node, "emission_color").text = str_color(emission_color)
            else:
                material_type = 'pbr'
                # specular (glossy)
                specular_color = material.specular_color * material.specular_intensity
                glossiness = material.specular_hardness
                if glossiness == 511:
                    # reflection
                    glossiness = -1
                else:
                    # glossy
                    pass

                # diffuse
                diffuse_color = material.diffuse_color * material.diffuse_intensity
                diffuse_shader = material.diffuse_shader
                # refraction
                transparent = material.use_transparency
                if transparent:
                    ior = material.raytrace_transparency.ior
                # SSS?

                # Writing...
                color = material.diffuse_color
                SubElement(material_node, "diffuse_color").text = str_color(diffuse_color)
                SubElement(material_node, "specular_color").text = str_color(specular_color)
                SubElement(material_node, "glossiness").text = str(glossiness)
                SubElement(material_node, "transparent").text = str(int(transparent))
                if transparent:
                    SubElement(material_node, "ior").text = str(ior)
            textures_node = SubElement(material_node, "textures")
            for texture_id in range(len(material.texture_slots)):
                texture_slot = material.texture_slots[texture_id]
                if texture_slot is None:
                    continue
                texture = bpy.data.textures[texture_slot.name]
                if texture.image is None:
                    continue
                image = texture.image
                texture_node = SubElement(textures_node, "texture")
                SubElement(texture_node, "filepath").text = image.filepath.lower()
                if texture_slot.use_map_color_diffuse:
                    SubElement(texture_node, "use_map_color_diffuse").text = str(texture_slot.diffuse_color_factor)
                if texture_slot.use_map_color_spec:
                    SubElement(texture_node, "use_map_color_spec").text = str(texture_slot.specular_color_factor)
                if texture_slot.use_map_normal:
                    SubElement(texture_node, "use_map_normal").text = str(texture_slot.normal_factor)
                if texture_slot.use_map_displacement:
                    SubElement(texture_node, "use_map_displacement").text = str(texture_slot.displacement_factor)

            mesh_node.set('name', obj.name)
            SubElement(material_node, "type").text = material_type

    def dump_camera(obj):
        camera_node = SubElement(scene_node, 'camera')
        camera_name = bpy.data.cameras[0].name
        transform = bpy.data.objects[camera_name].matrix_world
        dump_matrix(transform, 'transform', camera_node)

    def dump_render():
        render_node = SubElement(scene_node, 'render')
        res_rate = int(bpy.data.scenes[0].render.resolution_percentage)
        SubElement(render_node, "resolution_x").text = str(res_rate * bpy.data.scenes[0].render.resolution_x // 100)
        SubElement(render_node, "resolution_y").text = str(res_rate * bpy.data.scenes[0].render.resolution_y // 100)

    scene_node = Element('scene')

    meshes_node = SubElement(scene_node, 'meshes')

    for obj in bpy.data.objects:
        scene.objects.active = obj
        obj.select = True
        if obj.type == 'MESH' and not obj.hide:
            filepath = os.path.join(export_path, obj.name.lower() + '.obj')
            dump_mesh(obj, meshes_node, filepath)
            bpy.ops.export_scene.obj(
                filepath=filepath,
                use_selection=True,
                use_normals=True,
                use_uvs=True,
                use_materials=False,
                use_triangles=True,
                axis_up='Z',
                axis_forward='Y',
                )
        elif obj.type == 'CAMERA':
            dump_camera(obj)
        obj.select = False

    dump_render()

    file_path = export_path + '/' + bpy.data.worlds[0].name + '.xml'
    with open(file_path, 'w') as f:
        f.write(prettify(scene_node))


class ExportToTinge(bpy.types.Operator):
    """Export Current Scene to Tinge for Rendering"""      # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "object.move_x"        # unique identifier for buttons and menu items to reference.
    bl_label = "Export to Tinge"       # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # enable undo for the operator. 
    def execute(self, context):        # execute() is called by blender when running the operator.
        scene = context.scene
        for i in range(10):
            print()
        export_scene(scene)
        return {'FINISHED'}            # this lets blender know the operator finished successfully.

addon_keymaps = []

def register():
    bpy.utils.register_class(ExportToTinge)

    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')

    kmi = km.keymap_items.new(ExportToTinge.bl_idname, 'E', 'PRESS', 
        ctrl=True, shift=False)
    #kmi.properties.total = 4

    addon_keymaps.append(km)


def unregister():
    bpy.utils.unregister_class(ExportToTinge)

    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)

    addon_keymaps.clear()


# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()
