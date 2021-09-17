#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D diffuseMap;

void main()
{             
    FragColor = texture(diffuseMap, TexCoords) * 0.5;
}