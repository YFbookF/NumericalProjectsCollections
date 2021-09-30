#version 330 core
out vec4 FragColor;

in vec4 color;
in vec2 TexCoords;
in vec4 lightSpacePos;
in vec3 Normal;
in vec3 FragPos;
uniform sampler2D depthMap;
uniform sampler2D diffuseMap;

uniform vec3 lightPos;
uniform vec3 cameraPos;

void main()
{
		float depthValue = texture(depthMap, TexCoords).r;
		vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
		projCoords = projCoords * 0.5 + 0.5;
		float closestDepth = texture(depthMap, projCoords.xy).r; 
		float currentDepth = projCoords.z;
		
		
		//FragColor = texture(depthMap, TexCoords);
		vec3 lightDir = normalize(lightPos - FragPos);
		vec3 normal = normalize(Normal);
		float diff = max(dot(lightDir, normal), 0.0);
		vec3 lightColor = vec3(0.8);
		vec3 diffuse = diff * lightColor;

		vec3 viewDir = normalize(cameraPos - FragPos);
		vec3 reflectDir = reflect(-lightDir,normal);
		float spec = 0.0;
		vec3 halfwayDir = normalize(lightDir + viewDir);  
		spec = pow(max(dot(normal, halfwayDir), 0.0), 8.0);
		vec3 specular = spec * lightColor;    

		vec3 color = texture(diffuseMap, TexCoords).rgb;
		 vec3 ambient = 0.3 * color;

		 float bias = max(0.001 * (1.0 - dot(normal, lightDir)), 0.003);
		float shadow = 0.0;
		vec2 texelSize = 1.0 / textureSize(depthMap,0);
		for(int x = -1;x <= 1;x++)
		{
			for(int y = -1;y <= 1;y++)
			{
				float pcfDepth = texture(depthMap,projCoords.xy + vec2(x,y) * texelSize).r;
				shadow += currentDepth - bias > pcfDepth ? 0.0:1.0;
			}
		}
		shadow /= 9.0;
		if(projCoords.z > 1.0)
			shadow = 0.0;

		vec3 lighting = (ambient + shadow * (diffuse + specular)) * color;

		FragColor = vec4(vec3(lighting), 1.0); // orthographic

}