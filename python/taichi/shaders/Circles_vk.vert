#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inColor;

layout(binding = 0) uniform UBO {
    vec3 color;
    int use_per_vertex_color;
    float radius;
} ubo;


layout(location = 1) out vec3 selectedColor;
 
void main()
{
	gl_PointSize =  ubo.radius * 2;
	
    float x = inPosition.x * 2.0 - 1.0;
    float y = - (inPosition.y * 2.0 - 1.0);

    gl_Position = vec4(x,y,0.0, 1.0);

    if(ubo.use_per_vertex_color == 0){
        selectedColor = ubo.color;
    }
    else{
        selectedColor = inColor;
    }
}
