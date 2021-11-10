//pbrt
vec3 sampleRectangleLight(vec3 x, vec3 nl, Rectangle light, out float weight)
{
	vec3 u = normalize(cross( abs(light.normal.y) < 0.9 ? vec3(0, 1, 0) : vec3(0, 0, 1), light.normal));
	vec3 v = cross(light.normal, u);
	vec3 randPointOnLight = light.position;

	randPointOnLight += mix(u * -light.radiusU * 0.9, u * light.radiusU * 0.9, rng());
	randPointOnLight += mix(v * -light.radiusV * 0.9, v * light.radiusV * 0.9, rng());
	
	vec3 dirToLight = randPointOnLight - x;
	float r2 = (light.radiusU * 2.0) * (light.radiusV * 2.0);
	float d2 = dot(dirToLight, dirToLight);
	float cos_a_max = sqrt(1.0 - clamp( r2 / d2, 0.0, 1.0));

	dirToLight = normalize(dirToLight);
	float dotNlRayDir = max(0.0, dot(nl, dirToLight)); 
	weight = 2.0 * (1.0 - cos_a_max) * max(0.0, -dot(dirToLight, light.normal)) * dotNlRayDir;
	weight = clamp(weight, 0.0, 1.0);

	return dirToLight;
}
