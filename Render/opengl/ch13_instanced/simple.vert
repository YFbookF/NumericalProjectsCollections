
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
uniform float gScale;
uniform mat4 worldToCameraMatrix;
out vec4 color;
out vec2 TexCoord;

void main()
{ 
		gl_Position = worldToCameraMatrix * vec4(aPos * gScale, 1.0);
	    color.xyz = aColor;
		TexCoord = vec2(aTexCoord.x,aTexCoord.y);
}