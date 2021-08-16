#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;


struct SceneUBO{
    vec3 camera_pos;
    mat4 view;
    mat4 projection;
    vec3 ambient_light;
    int point_light_count;
};

layout(binding = 0) uniform UBO {
    SceneUBO scene;
    vec3 color;
    int use_per_vertex_color;
    float radius;
    float window_width;
    float window_height;
    float tan_half_fov;
} ubo;

layout(location = 0) out vec4 posToCamera;
layout(location = 1) out vec3 selectedColor;
 
void main()
{
 	vec3 cameraToPoint = inPosition - ubo.scene.camera_pos;
	float distance = length(cameraToPoint);

	float sizeAtDistanceForHalfScreen = ubo.tan_half_fov * distance;
	gl_PointSize = (ubo.window_height / 2.0) * ubo.radius / sizeAtDistanceForHalfScreen;
	
	posToCamera = ubo.scene.view * vec4(inPosition, 1.0);
    gl_Position = ubo.scene.projection * posToCamera;
    gl_Position.y *= -1;

    if(ubo.use_per_vertex_color == 0){
        selectedColor = ubo.color;
    }
    else{
        selectedColor = inColor;
    }

}
