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


vec3 goal;
vec3 eyeDirNormalized;
int reflectDepth = 50;

const float HitableType_Sphere = 0.0;
const float HitableType_Disk = 1.0;
const float HitableType_Box = 2.0;
const int HitableNum = 3;

int hitted_object_idx;
float nearest;

const float scene[33] = float[](
	//type	centerx	centery	centerz	radiusx,radiusy,radiusz,normalx,normaly,normalz,matidx
	HitableType_Disk,0,-1,0,	3,3,3,	0,1,0,	0,
	HitableType_Sphere,	-3,0,0,	1,2,2,	0,0,0,	0,
	HitableType_Sphere,	0.5,0,0,1,2,2,	0,0,0,	0
);

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

float randunit(float f1,float f2,float f3,float f4,float f5,float f6,float f7)
{
	return (sin(3*f1 + 5*f2 + f5*5) + cos(f3*f4 + 10 * f7 + f6*7)) * 0.5;
}

void hit_sphere(int object_idx)
{
	float centerx = scene[object_idx + 1];
	float centery = scene[object_idx + 2];
	float centerz = scene[object_idx + 3];
	float radius = scene[object_idx + 4];
	vec3 rayPos = eyePos;
	vec3 rayDir = normalize(eyeDir + eyeUp * 0.5 * (TexCoords.y * 2.0 - 1.0) + eyeRight * 0.5 * (TexCoords.x * 2.0 - 1.0));
	vec3 oc = rayPos - vec3(centerx,centery,centerz);
	float eqn_a = dot(rayDir,rayDir);
	float eqn_b = 2 * dot(oc,rayDir);
	float eqn_c = dot(oc,oc) - radius * radius;
	float discriminant = eqn_b * eqn_b - 4 * eqn_a * eqn_c;
	if(discriminant > 0)
	{
		float t = (- eqn_b - sqrt(discriminant)) / ( 2 * eqn_a);
		if(nearest > t)
		{
			nearest = t;
			hitted_object_idx = object_idx;
		}
	}
}

void RayTracer()
{
	int object_idx;
	hitted_object_idx = -1;
	nearest = 100;
	for(int i = 0;i < HitableNum;i++)
	{
		object_idx = i * 11;
		if(scene[object_idx] == HitableType_Sphere)
		{
			hit_sphere(object_idx);
		}
	}
	vec3 rayPos = eyePos;
	vec3 rayDir = normalize(eyeDir + eyeUp * 0.5 * (TexCoords.y * 2.0 - 1.0) + eyeRight * 0.5 * (TexCoords.x * 2.0 - 1.0));
	vec3 hitPos = rayPos + nearest * rayDir;
	vec3 hitobject_center = vec3(scene[hitted_object_idx + 1],scene[hitted_object_idx + 2],scene[hitted_object_idx+3]);
	vec3 normal = normalize(hitPos - hitobject_center);
	FragColor = vec4(normal,1.0); // orthographic
}

void main()
{
	RayTracer();
}