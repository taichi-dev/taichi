#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 selectedColor;

layout(binding = 0) uniform UniformBufferObject {
    vec3 color;
    int use_per_vertex_color;
} ubo;


void main() {
    float x = inPosition.x * 2.0 - 1.0;
    float y = - (inPosition.y * 2.0 - 1.0);

    gl_Position = vec4(x,y,0.0, 1.0);
    fragTexCoord = inTexCoord;


    if(ubo.use_per_vertex_color == 0){
        selectedColor = ubo.color;
    }
    else{
        selectedColor = inColor;
    }
}