#version 330 core
out vec4 FragColor;

in vec4 color;
in vec2 TexCoords;
in vec4 lightSpacePos;
uniform sampler2D depthMap;
uniform sampler2D diffuseMap;
void main()
{
	float depthValue = texture(depthMap, TexCoords).r;
	vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
	 projCoords = projCoords * 0.5 + 0.5;
	 float closestDepth = texture(depthMap, projCoords.xy).r; 
	  float currentDepth = projCoords.z;
	  float bias = -0.003;// 不加这个就有条纹出现
	float shadow = currentDepth + bias > closestDepth  ? 0.0 : 1.0;
	FragColor = vec4(vec3(shadow), 1.0); // orthographic
	//FragColor = texture(depthMap, TexCoords);

}