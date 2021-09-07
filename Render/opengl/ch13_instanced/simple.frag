#version 330 core
out vec4 FragColor;

in vec4 color;
in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
void main()
{

	FragColor = (texture(texture1,TexCoord) + texture(texture2,TexCoord))/ 2;
}