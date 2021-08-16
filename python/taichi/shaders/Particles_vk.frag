#version 450

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

struct PointLight{
    vec3 pos;
    vec3 color;
};

layout(binding = 1, std430) buffer SSBO {
    PointLight point_lights[];
} ssbo;


layout(location = 0) out vec4 outColor;

layout(location = 0) in vec4 posToCamera;
layout(location = 1) in vec3 selectedColor;

 
float projectZ(float viewZ) {
	vec3 dummyViewSpacePoint = vec3(0, 0, viewZ);
	vec4 projected = ubo.scene.projection * vec4(dummyViewSpacePoint, 1);
	return projected.z / projected.w;
}

vec3 posToCameraSpace(vec3 pos){
    vec4 temp = ubo.scene.view * vec4(pos,1.0);
    return temp.xyz/temp.w;
}

// operates in camera space !!
vec3 lambertian(vec3 fragPos,vec3 fragNormal){
    
    vec3 ambient = ubo.scene.ambient_light * selectedColor;
    vec3 result = ambient;

    for(int i = 0;i<ubo.scene.point_light_count;++i){
        vec3 lightColor = ssbo.point_lights[i].color;

        vec3 lightDir = normalize(posToCameraSpace(ssbo.point_lights[i].pos) - fragPos);
        vec3 normal = normalize(fragNormal);
        vec3 diffuse = max(dot(lightDir, normal), 0.0) * selectedColor * lightColor;
        
        result += diffuse;
    }

    return result;
}

void main()
{  
	vec2 coord2D;
	coord2D = gl_PointCoord* 2.0 - vec2(1); 
    coord2D.y *= -1;

	float distanceToCenter = length(coord2D);
	if(distanceToCenter >= 1.0) {
        discard;
    }
     

	float zInSphere = sqrt(1-coord2D.x*coord2D.x - coord2D.y * coord2D.y);
	vec3 coordInSphere = vec3(coord2D,zInSphere);

    vec3 fragPos = posToCamera.xyz / posToCamera.w + coordInSphere * ubo.radius;
    vec3 fragNormal = coordInSphere;
    vec3 color = lambertian(fragPos,fragNormal);
    outColor = vec4(color,1.0);


	float depth = (posToCamera.z / posToCamera.w) + zInSphere * ubo.radius;


	gl_FragDepth = projectZ(depth);
	gl_FragDepth = 0.5 * (1.0 + gl_FragDepth);

}
