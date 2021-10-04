#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 TexCoords;
out vec3 Normal;
out vec3 Position;

void main()
{ 
	gl_Position = projection * view * model * vec4(aPos , 1.0);
	TexCoords = aTexCoords;
	Normal = mat3(transpose(inverse(model)))*aNormal;
	Position = vec3(model * vec4(aPos,1.0));
}