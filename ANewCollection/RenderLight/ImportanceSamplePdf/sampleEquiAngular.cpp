//https://github.com/erichlof/THREE.js-PathTracing-Renderer
/* Credit: Some of the equi-angular sampling code is borrowed from https://www.shadertoy.com/view/Xdf3zB posted by user 'sjb' ,
// who in turn got it from the paper 'Importance Sampling Techniques for Path Tracing in Participating Media' ,
which can be viewed at: https://docs.google.com/viewer?url=https%3A%2F%2Fwww.solidangle.com%2Fresearch%2Fegsr2012_volume.pdf */
void sampleEquiAngular( float u, float maxDistance, Ray r, vec3 lightPos, out float dist, out float pdf )
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - r.origin, r.direction);
	
	// get distance this point is from light
	float D = distance(r.origin + delta*r.direction, lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);

	// take sample
	float t = D*tan(mix(thetaA, thetaB, u));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}
