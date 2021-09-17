#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 worldPos;

uniform sampler2D diffuseMap;
uniform float near_plane;
uniform float far_plane;

// required when using a perspective projection matrix
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));	
}

void main()
{             
    //float depthValue = texture(depthMap, TexCoords).r;
    //FragColor = vec4(vec3(LinearizeDepth(depthValue) / 10000), 1.0); // perspective
    //FragColor = vec4(vec3(depthValue), 1.0); // orthographic
     //FragColor = vec4(vec3(gl_FragCoord.z / gl_FragCoord.w * 0.01), 1.0);
    // gl_FragDepth = gl_FragCoord.z / gl_FragCoord.w * 0.01;
     FragColor = vec4(1.0,0.0,1.0, 1.0);
     //FragColor = texture(diffuseMap, TexCoords);
}