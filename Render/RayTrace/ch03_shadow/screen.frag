#version 330 core
out vec4 FragColor;
//第三版，可以处理阴影，但是采样不均匀，引入时间变量后，发现就是采样的问题
//下一版改用蒙特卡洛采样
//再下一版使用透明材质
//再下一版使用bvh
in vec4 color;
in vec2 TexCoords;
in vec4 lightSpacePos;
uniform sampler2D depthMap;
uniform sampler2D diffuseMap;

uniform vec3 eyePos;
uniform vec3 eyeDir;
uniform vec3 eyeUp;
uniform vec3 eyeRight;

uniform float time;

#define INFINITY 10000.0


vec3 goal;
vec3 eyeDirNormalized;
int reflectDepth = 50;

const float HitableType_Sphere = 0.0;
const float HitableType_Disk = 1.0;
const float HitableType_Box = 2.0;
const float HitableType_Rect = 3.0;
const int HitableNum = 5;

int hitted_object_idx;
float nearest;
vec3 lightPos;

const float scene[55] = float[](
	//type	centerx	centery	centerz	radiusx,radiusy,radiusz,normalx,normaly,normalz,matidx
	HitableType_Rect,	-2,-1,-2,	5,0,0,	0,0,5,	0.1,
	HitableType_Sphere,	0,0,0,	1,2,2,		0,0,0,	0.1,
	HitableType_Sphere,	2,0,0,	0.6,2,2,	0,0,0,	0.1,
	HitableType_Sphere,	0,2,0,	0.3,2,2,	0,0,0,	0.1,
	HitableType_Sphere,	0,0,2,	0.1,2,2,	0,0,0,	0.1
);

float randunit(float f1,float f2,float f3,float f4)
{
	return (sin(3*f1 + 5*f3 + f4*5) + cos(f3*5 + 10 * f2 + f3*7)) * 0.5;
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


//https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-1/blob/master/smallptCUDA.cu

float hit_sphere(vec3 rayPos,vec3 rayDir,vec3 center,float radius)
{

	vec3 oc = rayPos - center;
	float eqn_a = dot(rayDir,rayDir);
	float eqn_b = 2 * dot(oc,rayDir);
	float eqn_c = dot(oc,oc) - radius * radius;
	float discriminant = eqn_b * eqn_b - 4 * eqn_a * eqn_c;
	if(discriminant < 0)return  INFINITY;
	float t = (- eqn_b - sqrt(discriminant)) / ( 2 * eqn_a);
	if(t > 0.0001)return t;
	t = (- eqn_b + sqrt(discriminant)) / ( 2 * eqn_a);
	if(t > 0.0001)return t;
	return  INFINITY;
}

float hit_rect(in vec3 rayPos,in vec3 rayDir,in vec3 center,in vec3 u,in vec3 v,in vec3 planeNormal)
{
	float dt = dot(rayDir,planeNormal);
	float t = dot(center - rayPos,planeNormal) / dt;
	float length_u = length(u);
	float length_v = length(v);
	u = normalize(u);
	v = normalize(v);
	if(t > 0.001)
	{
		vec3 p = rayPos + t * rayDir;
		vec3 vi = p - center;
		float a1 = dot(u,vi);
		if(a1 >= 0. && a1 <= length_u)
		{
			float a2 = dot(v,vi);
			if(a2 >= 0. && a2 <= length_v)
			{
				return t;
			}
		}
	}
	return  INFINITY;
}

float hit_AABB(vec3 rayPos,vec3 rayDir,vec3 minCorner, vec3 maxCorner)
{
	vec3 invDir = 1.0 / rayDir;
	vec3 f = (maxCorner - rayPos) * invDir;
	vec3 n = (minCorner - rayPos) * invDir;
	vec3 tmax = max(f,n);
	vec3 tmin = min(f,n);
	float t1 = min(tmax.x,min(tmax.y,tmax.z));
	float t0 = max(tmin.x,max(tmin.y,tmin.z));
	return (t1 >= t0)?(t0 > 0.f ? t0:t1):-1.0;
}

bool hit_any(vec3 rayPos,vec3 rayDir)
{
	int object_idx = 0;
	for(int i = 0;i < HitableNum;i++)
	{
			object_idx = i * 11;
			if(scene[object_idx] == HitableType_Sphere)
			{
				vec3 center = vec3(scene[object_idx+1],scene[object_idx+2],scene[object_idx+3]);
				float radius = scene[object_idx+4];
				float t = hit_sphere(rayPos,rayDir,center,radius);
				if(t < INFINITY)
				{
					return true;
				}
			}else if(scene[object_idx] == HitableType_Rect)
			{
				vec3 center = vec3(scene[object_idx+1],scene[object_idx+2],scene[object_idx+3]);
				vec3 rect_u = vec3(scene[object_idx+4],scene[object_idx+5],scene[object_idx+6]);
				vec3 rect_v = vec3(scene[object_idx+7],scene[object_idx+8],scene[object_idx+9]);
				vec3 normal = normalize(cross(rect_u,rect_v));
				float t = hit_rect(rayPos,rayDir,center,rect_u,rect_v,normal);
				if(t < INFINITY)
				{
					return true;
				}
			}
	}
	return false;
}


vec3 CosineSampleHemisphere(float r1,float r2)
{
	vec3 dir;
    dir.z = r1;
	float tempr = sqrt(1 - r1 * r1);
	float phi = 2 * 3.14159 * r2;
    dir.x = tempr * cos(phi);
    dir.y = tempr * sin(phi);
    return dir;
}
vec3 UniformSampleSphere(float r1,float r2,float r3)
{
	vec3 dir = vec3(r1,r2,r3);
    return normalize(dir);
}

vec3 SampleHemisphere(float r1,float r2,float r3,vec3 normal)
{
	vec3 unit_sphere = UniformSampleSphere(r1,r2,r3);
	if(dot(unit_sphere,normal) > 0)
	{
		return unit_sphere;
	}
	return -unit_sphere;
}

vec3 UniformSampleHemisphere(float r1,float r2)
{
	vec3 dir;
    dir.z = sqrt(r1);
	float tempr = sqrt(1 - r1 * r1);
	float phi = 2 * 3.14159 * r2;
    dir.x = tempr * cos(phi);
    dir.y = tempr * sin(phi);
    return dir;
}

void DisneySample()
{
	float pdf = 0;
}

vec3 DirectionLight(vec3 rayPos,vec3 surfaceNormal)
{
	
	vec3 LightDir =  lightPos - rayPos;
	if(hit_any(rayPos,LightDir))
	{
		return vec3(0,0,0);
	}
	float diffuse_light_intensity = max(0,dot(LightDir,surfaceNormal));
	//float specular_light_intensity = pow(max(0.0f,reflect(rayDir,surfaceNormal) * ))
	vec3 LightColor = vec3(0.5,0.5,0.5);
	return LightColor * diffuse_light_intensity;
}

void getTangent(in vec3 normal,inout vec3 tangent,inout vec3 bitangent)
{
	vec3 UpVector = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    tangent = normalize(cross(UpVector, normal));
    bitangent = cross(normal, tangent);
}

//根据光照计算公式计算diffuse
//    diffuse_light_intensity += max (0.f, light_dir * N) * lights[i].intensity;
// 根据光照计算公式计算specular
//specular_light_intensity += pow(max(0.f, reflect(light_dir, N) * dir), material.specular_exponent) * lights[i].intensity; 
void RayTracer()
{
	lightPos = vec3(3.0,5.0,3.0);
	int object_idx;
	//bounce
	int bounce_times = 32;
	
	vec3 rayPos = eyePos;
	vec3 rayDir = normalize(eyeDir + eyeUp * 0.5 * (TexCoords.y * 2.0 - 1.0) + eyeRight * 0.5 * (TexCoords.x * 2.0 - 1.0));
	
	vec3 radiance = vec3(0.0);
	vec3 throughput = vec3(1.0);
	vec3 bgColor = vec3(0.5,0.5,1.0);
	vec3 radiance_total = vec3(0.0);
	int sample_times = 16;

	for(int si = 0;si < sample_times;si++)
	{
		throughput = vec3(1.0);
		for(int bi = 0;bi < bounce_times;bi++)
		{
			hitted_object_idx = -1;
			nearest = INFINITY;
			for(int i = 0;i < HitableNum;i++)
			{
				object_idx = i * 11;
				if(scene[object_idx] == HitableType_Sphere)
				{
					vec3 center = vec3(scene[object_idx+1],scene[object_idx+2],scene[object_idx+3]);
					float radius = scene[object_idx+4];
					float t = hit_sphere(rayPos,rayDir,center,radius);
					if(nearest > t)
					{
						nearest = t;
						hitted_object_idx = object_idx;
					}
				}else if(scene[object_idx] == HitableType_Rect)
				{
					vec3 center = vec3(scene[object_idx+1],scene[object_idx+2],scene[object_idx+3]);
					vec3 rect_u = vec3(scene[object_idx+4],scene[object_idx+5],scene[object_idx+6]);
					vec3 rect_v = vec3(scene[object_idx+7],scene[object_idx+8],scene[object_idx+9]);
					vec3 normal = normalize(cross(rect_u,rect_v));
					float t = hit_rect(rayPos,rayDir,center,rect_u,rect_v,normal);
					if(nearest > t)
					{
						nearest = t;
						hitted_object_idx = object_idx;
					}
				}
			}
			if(nearest > 999)
			{
				radiance += bgColor * throughput ;
				break;
			}
			vec3 hitPos = rayPos + nearest * rayDir;
			vec3 hitobject_center = vec3(scene[hitted_object_idx + 1],scene[hitted_object_idx + 2],scene[hitted_object_idx+3]);
			vec3 normal = normalize(hitPos - hitobject_center);
			vec3 tangent;
			vec3 bitangent;
			getTangent(normal,tangent,bitangent);
		


			float r1 = randunit(hitPos.x,bi,si,normal.z);
			float r2 = randunit(rayPos.y,bi,rayDir.x,si);
			float r3 = randunit(nearest,si,bi,normal.x);

			//vec3 randDir = UniformSampleSphere(r1,r2,r3);
			//rayPos = hitPos + normal;
			//rayDir = randDir;
		
			rayPos = hitPos + normal * 0.05;
			vec3 randDir = CosineSampleHemisphere(r1,r2);
			rayDir = tangent * randDir.x + bitangent * randDir.y + normal * randDir.z;
			radiance += DirectionLight(rayPos,normal) * throughput.x;
			//radiance += vec3(0.5);
			throughput = throughput * scene[hitted_object_idx + 10];
		}
	}
	radiance = radiance / sample_times;

	FragColor = vec4(radiance,1.0); 

}

void main()
{
	RayTracer();
}