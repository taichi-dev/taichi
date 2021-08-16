#version 450


layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    vec3 color;
    int use_per_vertex_color;
} ubo;


layout(location = 1) in vec3 selectedColor;


void main() {
    outColor = vec4(selectedColor,1);
}