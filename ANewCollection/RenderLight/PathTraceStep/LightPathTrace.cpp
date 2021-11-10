//https://github.com/erichlof/THREE.js-PathTracing-Renderer

	vec3 lightHitEmission = quads[0].emission;
	vec3 randPointOnLight;
	randPointOnLight.x = mix(quads[0].v0.x, quads[0].v1.x, rng());
	randPointOnLight.y = mix(quads[0].v0.y, quads[0].v3.y, rng());
	randPointOnLight.z = quads[0].v0.z;
	vec3 lightHitPos = randPointOnLight;
	vec3 lightNormal = normalize(quads[0].normal);

	rayDirection = randomCosWeightedDirectionInHemisphere(lightNormal);
	rayOrigin = randPointOnLight + lightNormal * uEPS_intersect; // move light ray out to prevent self-intersection with light

	t = SceneIntersect(rayOrigin, rayDirection, checkModels);

	if (hitType == DIFF)
	{
		lightHitPos = rayOrigin + rayDirection * t;
		weight = max(0.0, dot(-rayDirection, normalize(hitNormal)));
		lightHitEmission *= hitColor * weight;
	}
		if (hitType == DIFF && sampleLight)
		{
			ableToJoinPaths = abs(t - lightHitDistance) < 0.5;

			pixelSharpness = 0.0;

			if (ableToJoinPaths)
			{
				weight = max(0.0, dot(normalize(hitNormal), -rayDirection));
				accumCol = mask * lightHitEmission * weight;
			}

			break;
		}
