#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 Position;
uniform sampler2D diffuseMap;
uniform vec3 cameraPos;
uniform samplerCube skybox;

void main()
{
	FragColor = texture(diffuseMap,TexCoords);
	vec3 I = normalize(Position - cameraPos);
	vec3 R = reflect(I,normalize(Normal));
	FragColor = vec4(texture(skybox,R).rgb,1.0);
}