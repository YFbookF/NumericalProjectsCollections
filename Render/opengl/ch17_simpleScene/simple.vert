#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormals;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = aTexCoords;
    vec3 aaPos = vec3(aPos.x,aPos.y,aPos.z + 1);
    vec4 worldPos = model * vec4(aPos, 1.0);
    gl_Position = projection * view *  worldPos;
}