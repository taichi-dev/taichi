#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 selectedColor;


struct SceneUBO{
    vec3 camera_pos;
    mat4 view;
    mat4 projection;
    vec3 ambient_light;
    int point_light_count;
};

struct PointLight{
    vec3 pos;
    vec3 color;
};

layout(binding = 0) uniform UBO {
    SceneUBO scene;
    vec3 color;
    int use_per_vertex_color;
} ubo;

void main() {
    gl_Position =  ubo.scene.projection * ubo.scene.view * vec4(inPosition,1.0);
    gl_Position.y *= -1.0;
    fragTexCoord = inTexCoord;
    fragPos = inPosition;
    fragNormal = inNormal;

    if(ubo.use_per_vertex_color == 0){
        selectedColor = ubo.color;
    }
    else{
        selectedColor = inColor;
    }
}