#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoords;
uniform float gScale;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
out vec4 color;
out vec4 fragPos;

void main()
{ 
	gl_Position = projection * view * model * vec4(aPos , 1.0);
	color.xyz = aColor;
	fragPos = vec4(aPos,1.0);
}