#version 450

layout(binding = 0) uniform UBO {
    vec3 color;
    int use_per_vertex_color;
    float radius;
} ubo;

layout(location = 1) in vec3 selectedColor;

layout(location = 0) out vec4 outColor;

  

void main()
{  
	vec2 coord2D;
	coord2D = gl_PointCoord* 2.0 - vec2(1); 
 
	float distanceToCenter = length(coord2D);
	if(distanceToCenter >= 1.0) {
        discard;
    }
      
    outColor = vec4(selectedColor,1.0); 

}
