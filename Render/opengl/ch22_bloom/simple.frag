#version 330 core
out vec4 FragColor;

in vec4 color;
in vec4 fragPos;
uniform float near_plane;
uniform float far_plane;

void main()
{
	float depth = gl_FragCoord.z ;
	float ndc = depth * 2.0 - 1.0; 
	float linearDepth = 2 *near_plane * far_plane / (far_plane + near_plane -  ndc * (far_plane - near_plane)) / far_plane;
	FragColor = vec4(linearDepth,linearDepth,linearDepth,1.0);
	FragColor = vec4(depth,depth,depth,1.0);
	//FragColor = color;
}