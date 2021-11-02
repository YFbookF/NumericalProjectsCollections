#version 330 core
out vec4 FragColor;

in vec4 color;
in vec2 TexCoords;
in vec4 lightSpacePos;
uniform sampler2D depthMap;
uniform sampler2D diffuseMap;

uniform vec3 eyePos;
uniform vec3 eyeDir;
uniform vec3 eyeUp;
uniform vec3 eyeRight;

float nearest;
vec3 goal;
vec3 eyeDirNormalized;
int reflectDepth = 50;

bool hit_spherebool()
{

   vec3 oc = eyePos -  vec3(0,0,0);
   float sphere_radius = 0.5;
   float eqn_a = dot(eyeDirNormalized,eyeDirNormalized);
   float eqn_b = 2 * dot(oc,eyeDirNormalized);
   float eqn_c = dot(oc,oc) - sphere_radius * sphere_radius;
   float discriminant = eqn_b * eqn_b - 4 * eqn_a * eqn_c;
   return discriminant > 0;
}

vec3 hit_sphere_normal(vec3 spherePos)
{

   vec3 oc = eyePos -  spherePos;
   float sphere_radius = 2.0;
   float eqn_a = dot(eyeDirNormalized,eyeDirNormalized);
   float eqn_b = 2 * dot(oc,eyeDirNormalized);
   float eqn_c = dot(oc,oc) - sphere_radius * sphere_radius;
   float discriminant = eqn_b * eqn_b - 4 * eqn_a * eqn_c;
   if(discriminant < 0)
   {
		return vec3(0,0,0);
   }else
   {
        float t1 = (-eqn_b - sqrt(discriminant) ) / (2.0*eqn_a);
		float t2 = (-eqn_b + sqrt(discriminant) ) / (2.0*eqn_a);
		if(nearest < t1)
		{
			return vec3(0,0,0);
		}
		nearest = t1;
		vec3 hitPos = eyePos + t1 * eyeDirNormalized;
		vec3 normal = normalize(hitPos - spherePos);
		return normal * 0.5 + vec3(0.5,0.5,0.5);
   }
}

void hit_sphere_multiple(vec3 spherePos)
{

   vec3 oc = eyePos -  spherePos;
   float sphere_radius = 1.0;
   float eqn_a = dot(eyeDirNormalized,eyeDirNormalized);
   float eqn_b = 2 * dot(oc,eyeDirNormalized);
   float eqn_c = dot(oc,oc) - sphere_radius * sphere_radius;
   float discriminant = eqn_b * eqn_b - 4 * eqn_a * eqn_c;
   if(discriminant > 0)
   {
        float t1 = (-eqn_b - sqrt(discriminant) ) / (2.0*eqn_a);
		if(nearest > t1)
		{
			nearest = t1;
		}
   }
}

void hit_disk_multiple(vec3 diskPos,vec3 diskNormal,float diskRadius)
{

    float possibleT = dot(diskPos - eyePos,diskNormal) / dot(eyeDirNormalized,diskNormal);
	vec3 hitPos = eyePos + possibleT * eyeDirNormalized;
	float dis = distance(hitPos,diskPos);
	if((dis < diskRadius) && (possibleT < nearest))
	{
		nearest = possibleT;
	}
}

void newRay(vec3 rayPos,vec3 rayDir)
{



}

void main()
{

	goal = eyeDir + eyeUp * 0.5 * (TexCoords.y * 2.0 - 1.0) + eyeRight * 0.5 * (TexCoords.x * 2.0 - 1.0);
    eyeDirNormalized = normalize(goal);
    nearest = 100;
	float depthValue = texture(depthMap, TexCoords).r;
	vec3 diffuseValue = texture(diffuseMap, TexCoords).rgb;
	vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
	projCoords = projCoords * 0.5 + 0.5;
	 float closestDepth = texture(depthMap, projCoords.xy).r; 
	  float currentDepth = projCoords.z;
	  float bias = -0.003;// 不加这个就有条纹出现
	float shadow = currentDepth + bias > closestDepth  ? 0.0 : 1.0;
	FragColor = vec4(diffuseValue * shadow, 1.0); // orthographic
	//FragColor = texture(depthMap, TexCoords);
    /*
	if(hit_spherebool())
    FragColor = vec4(1.0,1.0,1.0,1.0); // orthographic
    vec3 sphereColor = hit_sphere_normal(vec3(0,0,0)) + hit_sphere_normal(vec3(1,0,0));
    FragColor = vec4(sphereColor,1.0); // orthographic
	*/
	newRay(eyePos,eyeDir);
		hit_disk_multiple(vec3(0,-1,0),vec3(0,1,0),3.0);
	hit_sphere_multiple(vec3(-3,0,0));
	hit_sphere_multiple(vec3(0.5,0,0));
	vec3 hitPos = eyePos + nearest * eyeDirNormalized;
	vec3 normal = normalize(hitPos - vec3(0,0,0));
	normal = normal * 0.5 + vec3(0.5,0.5,0.5);
	FragColor = vec4(normal,1.0); // orthographic
}