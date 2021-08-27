#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
uniform float gScale;
out vec4 color;

void main()
{ 
 
		gl_Position = vec4(aPos * gScale, 1.0);
	    color.xyz = aColor;
}