
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;
uniform float gScale;
uniform mat4 worldToCameraMatrix;

out VS_OUT{
	vec3 FragPos;
	vec2 TexCoords;
	vec3 TangentLightPos;
	vec3 TangentViewPos;
	vec3 TangentFragPos;
	vec4 color;
	vec3 tbnn;
}vs_out;

uniform vec3 lightPos;
uniform vec3 viewPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{ 
		gl_Position = model * vec4(aPos, 1.0);
		vs_out.TexCoords = aTexCoord;
		vs_out.color = vec4(aPos, 1.0);
		vs_out.FragPos = vec3(model * vec4(aPos, 1.0));   
		vec3 T = normalize(mat3(model) * aTangent);
		vec3 B = normalize(mat3(model) * aBitangent);
		vec3 N = normalize(mat3(model) * aNormal);
		mat3 TBN = transpose(mat3(T,B,N));
		vs_out.tbnn = B;
		vs_out.TangentLightPos = TBN * lightPos;
		vs_out.TangentViewPos = TBN * viewPos;
		vs_out.TangentFragPos = TBN * vs_out.FragPos;
		
	
}